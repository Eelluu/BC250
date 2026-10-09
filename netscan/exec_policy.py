#!/usr/bin/env python3
"""Ejecución acotada de comandos EXEC -- el "tool permissions" de OpenDots, en Python.

`is_command_safe()` (en queue-runner.py) sigue siendo el PORTERO: allowlist de
binarios, sin metacaracteres de shell, `cat`/`ls` confinados, intérpretes sólo
desde SCRIPT_ROOT y `curl` sin escritura. Este módulo NO lo sustituye; añade las
tres cosas que a ese modelo le faltan cuando el ejecutor es root en el host:

  1. LÍMITES DE RECURSOS por ejecución (RLIMIT_AS/CPU/NPROC/FSIZE). Hoy un script
     aprobado (o alucinado y aprobado por error) puede agotar los 16 GB de UMA y
     llevarse por delante a llama.cpp. Con esto, el proceso muere solo.
  2. SANDBOX opcional en podman para intérpretes/scripts: sin red, sistema de
     ficheros de sólo lectura, usuario sin privilegios, memoria y CPU acotadas.
     Apagado por defecto (haría falta una imagen con sus dependencias).
  3. ALLOWLIST DECLARATIVA (exec-policy.json) para ampliar la lista de binarios
     sin tocar el código, con motivo y fecha por entrada (auditable).

Variables de entorno:
  EXEC_LIMITS_MB=2048    memoria máxima del proceso (RLIMIT_AS)
  EXEC_CPU_S=30          CPU máxima (coincide con SIGNAL_EXEC_TIMEOUT_S)
  EXEC_NPROC=128         procesos/hilos máximos
  EXEC_FSIZE_MB=64       tamaño máximo de fichero que puede escribir
  EXEC_SANDBOX=0         1 = los comandos de `sandbox_bins` van a podman
  EXEC_SANDBOX_IMAGE=    imagen para el sandbox (p.ej. python:3.12-alpine)
  EXEC_SANDBOX_NET=0     1 = dar red al sandbox
  EXEC_AUDIT_LOG=        fichero donde anotar cada EXEC (opcional)
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
POLICY_PATH = Path(os.environ.get("EXEC_POLICY_FILE", _HERE / "exec-policy.json"))

# Intérpretes: los que pueden ejecutar código arbitrario y por tanto los que más
# se benefician del sandbox.
DEFAULT_SANDBOX_BINS = ("python3", "bash", "sh")

_warned: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        print(f"[exec_policy] {msg}", flush=True)


# ─── política declarativa ───────────────────────────────────────────────────

def load_policy() -> dict:
    """Lee exec-policy.json. Ausente o ilegible -> política vacía (deny extra)."""
    try:
        with open(POLICY_PATH, encoding="utf-8") as fh:
            pol = json.load(fh)
        if not isinstance(pol, dict):
            raise ValueError("la raíz debe ser un objeto")
        return pol
    except FileNotFoundError:
        return {}
    except Exception as exc:
        _warn_once(f"no pude leer {POLICY_PATH}: {exc} -- sigo sin reglas extra")
        return {}


def extra_allowed(binary: str, policy: dict | None = None) -> str | None:
    """¿La política permite un binario que no está en ALLOWED_TOOLS?

    Devuelve el motivo si está permitido, None si no. Deny por defecto: una lista
    vacía o ausente no amplía nada.
    """
    pol = policy if policy is not None else load_policy()
    for entry in pol.get("extra_allow", []) or []:
        if isinstance(entry, dict) and entry.get("bin") == binary:
            return entry.get("why", "extra_allow")
    return None


def sandbox_bins(policy: dict | None = None) -> tuple[str, ...]:
    pol = policy if policy is not None else load_policy()
    bins = pol.get("sandbox_bins")
    return tuple(bins) if bins else DEFAULT_SANDBOX_BINS


# ─── límites ────────────────────────────────────────────────────────────────

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def limits_mb() -> int:
    return _int_env("EXEC_LIMITS_MB", 2048)


def limits_prefix() -> list[str]:
    """Prefijo `prlimit` con los topes. Vacío si prlimit no está (con aviso)."""
    if not shutil.which("prlimit"):
        _warn_once("prlimit no está: los comandos EXEC correrán SIN límites de recursos "
                   "(apk add util-linux)")
        return []
    mb = limits_mb()
    return [
        "prlimit",
        f"--as={mb * 1024 * 1024}",
        f"--cpu={_int_env('EXEC_CPU_S', 30)}",
        f"--nproc={_int_env('EXEC_NPROC', 128)}",
        f"--fsize={_int_env('EXEC_FSIZE_MB', 64) * 1024 * 1024}",
        "--",
    ]


# ─── ejecución ──────────────────────────────────────────────────────────────

def _sandbox_enabled() -> bool:
    return os.environ.get("EXEC_SANDBOX", "0") not in ("", "0", "false", "no")


def sandbox_argv(cmd: str, timeout_s: int) -> list[str] | None:
    """argv de podman para `cmd`, o None si no se puede montar el sandbox.

    El sistema de ficheros va de sólo lectura y se monta SCRIPT_ROOT en /work.
    """
    image = os.environ.get("EXEC_SANDBOX_IMAGE", "").strip()
    if not image:
        _warn_once("sandbox pedido pero EXEC_SANDBOX_IMAGE está vacío: sigo en el host")
        return None
    if not shutil.which("podman"):
        _warn_once("sandbox pedido pero no encuentro podman: sigo en el host")
        return None
    root = os.environ.get("EXEC_SCRIPT_ROOT", "/opt/netscan/")
    net = "host" if os.environ.get("EXEC_SANDBOX_NET", "0") not in ("", "0", "false", "no") else "none"
    return [
        "podman", "run", "--rm",
        "--network", net,
        f"--memory={limits_mb()}m",
        f"--cpus={os.environ.get('EXEC_SANDBOX_CPUS', '2')}",
        "--user", "65534:65534",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=64m,mode=1777",
        "-v", f"{root}:/work:ro",
        "-w", "/work",
        image,
        "sh", "-lc", cmd,
    ]


def run(cmd: str, timeout_s: int = 30) -> tuple[int, str]:
    """Ejecuta `cmd` (ya validado por is_command_safe) y devuelve (rc, salida).

    El sandbox es opt-in: si está apagado, se ejecuta en el host pero con los
    límites de recursos puestos.
    """
    policy = load_policy()
    binary = os.path.basename(shlex.split(cmd)[0]) if cmd.strip() else ""

    argv = None
    mode = "host"
    if _sandbox_enabled() and binary in sandbox_bins(policy):
        argv = sandbox_argv(cmd, timeout_s)
        if argv is not None:
            mode = "sandbox"

    try:
        if argv is not None:
            res = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
        else:
            prefix = limits_prefix()
            # shell=True como hoy (el filtro de metacaracteres del portero sigue
            # aplicando) para no romper el globbing de `ls /tmp/*.log`.
            res = subprocess.run(" ".join([*prefix, cmd]) if prefix else cmd,
                                 shell=True, capture_output=True, text=True, timeout=timeout_s)
        rc = res.returncode
        out = (res.stdout + res.stderr).strip() or "[No output]"
    except subprocess.TimeoutExpired:
        rc, out = 124, f"[Timeout: {timeout_s}s]"
    except Exception as exc:  # noqa: BLE001 - nunca debe tumbar al bot
        rc, out = 127, f"[Error: {exc}]"

    audit(cmd, rc, mode, policy)
    return rc, out


def audit(cmd: str, rc: int, mode: str, policy: dict | None = None) -> None:
    """Anota la ejecución en EXEC_AUDIT_LOG si está configurado (JSONL)."""
    path = os.environ.get("EXEC_AUDIT_LOG", "").strip()
    if not path:
        return
    try:
        pol = policy if policy is not None else load_policy()
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "mode": mode,
                "rc": rc,
                "limits_mb": limits_mb(),
                "policy_version": pol.get("version"),
                "cmd": cmd[:400],
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass
