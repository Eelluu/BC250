#!/usr/bin/env bash
# bench-rocm-vs-vulkan.sh - compara llama-bench con backend ROCm/HIP contra Vulkan
# en el BC-250, con el mismo modelo y los mismos parametros.
#
# Uso:
#   ./bench-rocm-vs-vulkan.sh -m /opt/llama-models/modelo.gguf
#   ./bench-rocm-vs-vulkan.sh -m /opt/llama-models/modelo.gguf \
#       --rocm-model /root/bc250/models/modelo.gguf \
#       --vulkan /opt/llama.cpp/build/bin/llama-bench \
#       --rocm "podman exec bc250-rocm /root/bc250/llama.cpp/build-hip/bin/llama-bench"
#
# Opciones:
#   -m FILE          modelo GGUF para el backend del host (obligatorio)
#   --rocm-model F   mismo modelo, con la ruta que ve el contenedor (ver abajo)
#   --vulkan CMD     binario/orden de llama-bench para Vulkan (por defecto: autodetecta)
#   --rocm CMD       binario/orden para ROCm (por defecto: el contenedor bc250-rocm)
#   --reps N         repeticiones que pide llama-bench (def. 3; da media y sigma)
#   --pp N           tokens de prefill a medir (def. 512)
#   --tg N           tokens de generacion a medir (def. 128)
#   --ngl N          capas en GPU (def. 99)
#   --fa N           flash attention: 0/1/auto (def. 1, como los comandos de akandr)
#                    Su patch 0004 vale ~40 % del prefill *con* FA: sin esto, ROCm
#                    mide como su build de control de tres parches.
#   --pin N          fija GfxClk a N MHz con 'arieltune apu gpu pin' durante la medida
#                    (y 'unpin' al final). IMPRESCINDIBLE para comparar: el governor
#                    no sube el reloj con cargas HIP, asi que ROCm en frio ~350 MHz.
#   --extra "..."    flags extra para llama-bench (ej: "--extra \"-b 2048\"")
#   --only vulkan|rocm   ejecuta solo uno
#   --free-gpu       para llama-swap antes y lo rearranca al terminar (root)
#
# IMPORTANTE - la ruta del modelo:
#   El contenedor NO ve el sistema de ficheros del host salvo lo que se le monta.
#   Si el backend es "podman exec ...", hay que pasar --rocm-model con una ruta
#   que exista DENTRO del contenedor. Lo mas comodo: un hardlink al disco
#   persistente del contenedor (instantaneo, sin copiar):
#       mkdir -p /opt/rocm-bc250/models
#       ln /opt/llama-models/modelo.gguf /opt/rocm-bc250/models/   (o cp si no es el mismo fs)
#       --rocm-model /root/bc250/models/modelo.gguf
#   (Los contenedores creados con 'up' ya montan /opt/llama-models; los viejos no.)
#
# AVISO: la comparacion solo es justa si AMBOS binarios salen del MISMO llama.cpp.
# El de ROCm lleva los 13 parches de akandr/bc250-rocm; el script imprime el
# commit de cada uno y avisa si no coinciden.
set -euo pipefail

MODEL=""; ROCM_MODEL=""; VULKAN=""
ROCM="podman exec bc250-rocm /root/bc250/llama.cpp/build-hip/bin/llama-bench"
REPS=3; PP=512; TG=128; NGL=99; EXTRA=""; ONLY=""; FREE=0; FA=1; PIN=""

while [ $# -gt 0 ]; do
	case "$1" in
		-m|--model)     MODEL="${2:?}"; shift 2 ;;
		--rocm-model)   ROCM_MODEL="${2:?}"; shift 2 ;;
		--vulkan)       VULKAN="${2:?}"; shift 2 ;;
		--rocm)         ROCM="${2:?}"; shift 2 ;;
		--reps)         REPS="${2:?}"; shift 2 ;;
		--pp)           PP="${2:?}"; shift 2 ;;
		--tg)           TG="${2:?}"; shift 2 ;;
		--ngl)          NGL="${2:?}"; shift 2 ;;
		--extra)        EXTRA="${2:?}"; shift 2 ;;
		--fa)           FA="${2:?}"; shift 2 ;;
		--pin)          PIN="${2:?}"; shift 2 ;;
		--only)         ONLY="${2:?}"; shift 2 ;;
		--free-gpu)     FREE=1; shift ;;
		-h|--help)      sed -n '2,40p' "$0"; exit 0 ;;
		*) echo "opcion desconocida: $1" >&2; exit 1 ;;
	esac
done

[ -n "$MODEL" ] || { echo "Falta -m <modelo.gguf>" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "No existe el modelo en el host: $MODEL" >&2; exit 1; }
[ -n "$ROCM_MODEL" ] || ROCM_MODEL="$MODEL"

if [ -z "$VULKAN" ]; then
	for c in "$(command -v llama-bench 2>/dev/null || true)" \
	         /opt/llama.cpp/build/bin/llama-bench \
	         /opt/llama.cpp/build-vulkan/bin/llama-bench; do
		if [ -n "$c" ] && [ -x "$c" ]; then VULKAN="$c"; break; fi
	done
fi
if [ "$ONLY" != "rocm" ] && [ -z "$VULKAN" ]; then
	echo "No encuentro llama-bench de Vulkan; pasa --vulkan /ruta" >&2; exit 1
fi

TMP="$(mktemp -d)"

# Ejecuta "$cmd" con los argumentos dados, aunque cmd sea una orden con espacios
# ("podman exec contenedor /ruta/bin").
run() { local cmd="$1"; shift; bash -c "$cmd \"\$@\"" _ "$@"; }

# Si el backend corre dentro de un contenedor, comprueba que ve el modelo ANTES
# de lanzar nada: un "failed to load model" a los 30 s es un mal mensaje de error.
ctr_of() {  # nombre del contenedor si el comando es "podman exec [opts] X ..."
	local cmd="$1"; local -a a
	read -r -a a <<<"$cmd"
	# Si tras 'exec' hay una opcion (-e, --env, -i...) no intentamos adivinar el
	# nombre: devolvemos vacio y el preflight se salta (antes que dar un falso error).
	if [ "${a[0]:-}" = "podman" ] && [ "${a[1]:-}" = "exec" ] && [ "${a[2]:-}" != -* ]; then
		printf '%s' "${a[2]}"
	fi
}

check_visible() {  # etiqueta cmd modelo
	local label="$1" cmd="$2" model="$3" ctr
	ctr="$(ctr_of "$cmd")"
	[ -n "$ctr" ] || return 0
	if ! podman exec "$ctr" test -f "$model" 2>/dev/null; then
		echo
		echo "ERROR: el backend $label no ve el modelo dentro del contenedor '$ctr':"
		echo "         $model"
		echo "  El contenedor solo ve lo que tiene montado. Opciones:"
		echo "    a) ruta alternativa:"
		echo "         mkdir -p /opt/rocm-bc250/models"
		echo "         ln /opt/llama-models/$(basename "$model") /opt/rocm-bc250/models/  # o cp"
		echo "         $0 ... --rocm-model /root/bc250/models/$(basename "$model")"
		echo "    b) recrear el contenedor con el montaje del directorio de modelos:"
		echo "         /root/rocm/rocm-bc250-setup.sh recreate   (cuesta un 'deps')"
		exit 2
	fi
}

stop_gpu_users() {
	echo "== Liberando la GPU =="
	rc-service llama-swap stop 2>/dev/null && echo "  llama-swap parado" || echo "  (llama-swap no estaba corriendo)"
	sleep 2
}
start_gpu_users() {
	rc-service llama-swap start 2>/dev/null && echo "  llama-swap arrancado de nuevo" || true
}

# Reloj: sin pin, el governor decide -- y no lo sube con cargas HIP, asi que ROCm
# en frio mide a ~350 MHz. Con --pin se fija con 'arieltune apu gpu pin' (transitorio,
# pensado justo para benchmarks) y se suelta con 'unpin' al terminar.
pin_clock() {
	echo "== Reloj de GPU =="
	if command -v arieltune >/dev/null 2>&1; then
		arieltune apu gpu status 2>/dev/null | grep -iE 'manual pin|auto mode|current GfxClk' | sed 's/^ */  /'
		if [ -n "$PIN" ]; then
			echo "  fijando GfxClk a $PIN MHz (arieltune apu gpu pin)"
			if arieltune apu gpu pin "$PIN" >/dev/null 2>&1; then echo "  pin aplicado"; else echo "  ⚠️ no pude aplicar el pin"; fi
		else
			echo "  AVISO: sin --pin el governor decide el reloj, y NO lo sube con cargas HIP:"
			echo "         medir ROCm en frio da ~350 MHz (3,5x mas lento). Usa --pin 1500 (o 2230)."
		fi
	else
		echo "  (arieltune no disponible: no puedo leer ni fijar el reloj)"
	fi
}
unpin_clock() { if [ -n "$PIN" ] && command -v arieltune >/dev/null 2>&1; then arieltune apu gpu unpin >/dev/null 2>&1 || true; fi; }

show_results() {  # tsv etiqueta
	local f="$1" label="$2"
	if [ ! -s "$f" ]; then
		echo "  (sin resultados de $label)"
		return
	fi
	echo "  --- $label: velocidad ---"
	while IFS=$'\t' read -r be test avg sd commit; do
		[ -n "$test" ] || continue
		printf '    %-8s %9s t/s  (±%s)\n' "$test" "$avg" "$sd"
	done <"$f"
}

run_one() {  # etiqueta comando modelo
	local label="$1" cmd="$2" model="$3" t0 t1 args=()
	check_visible "$label" "$cmd" "$model"
	echo
	echo "=== $label ==="
	echo "  cmd: $cmd -m $model -ngl $NGL -p $PP -n $TG -r $REPS -fa $FA -o json"
	t0=$(date +%s)
	args=(-m "$model" -ngl "$NGL" -p "$PP" -n "$TG" -r "$REPS" -fa "$FA" -o json)
	if [ -n "$EXTRA" ]; then read -r -a XA <<<"$EXTRA"; args+=("${XA[@]}"); fi
	run "$cmd" "${args[@]}" >"$TMP/$label.json" 2>"$TMP/$label.err" || true
	t1=$(date +%s)
	echo "  tiempo: $((t1 - t0))s"
	if python3 - "$TMP/$label.json" <<'PY' >"$TMP/$label.tsv"
import json, sys

def test_label(r):
    # llama-bench no emite "test" en el JSON: la etiqueta pp512/tg128 se deriva
    # de n_prompt/n_gen, igual que hace su propia tabla markdown.
    if r.get("test"):
        return str(r["test"])
    p = int(r.get("n_prompt", 0) or 0)
    g = int(r.get("n_gen", 0) or 0)
    d = int(r.get("n_depth", 0) or 0)
    if p and not g:
        base = f"pp{p}"
    elif g and not p:
        base = f"tg{g}"
    elif p and g:
        base = f"pp{p}+tg{g}"
    else:
        base = "?"
    return base + (f"@d{d}" if d else "")

def num(r, *keys):
    for k in keys:
        if r.get(k) is not None:
            return float(r[k])
    return 0.0

raw = open(sys.argv[1], encoding="utf-8", errors="replace").read()
i = raw.find("[")
if i < 0:
    sys.exit(3)
try:
    data = json.loads(raw[i:])
except json.JSONDecodeError:
    sys.exit(3)
for r in data:
    print("\t".join([str(r.get("backend", "?")), test_label(r),
                     f'{num(r, "avg_ts", "t_avg"):.2f}',
                     f'{num(r, "stddev_ts", "t_stddev"):.2f}',
                     str(r.get("build_commit", ""))[:9]]))
PY
	then
		show_results "$TMP/$label.tsv" "$label"
	else
		echo "  ⚠️  sin JSON valido: el backend no llego a medir. Ultimas lineas de su salida:"
		tail -n 6 "$TMP/$label.err" | sed 's/^/      /' || true
	fi
}

if [ "$FREE" = 1 ]; then stop_gpu_users; fi
pin_clock
trap 'unpin_clock; [ "$FREE" = 1 ] && start_gpu_users; rm -rf "$TMP"' EXIT

if [ "$ONLY" != "rocm" ];   then run_one vulkan "$VULKAN" "$MODEL"; fi
if [ "$ONLY" != "vulkan" ]; then run_one rocm   "$ROCM"   "$ROCM_MODEL"; fi

echo
echo "=== Comparacion (t/s, media de $REPS reps) ==="
python3 - "$TMP/vulkan.tsv" "$TMP/rocm.tsv" <<'PY'
import sys, os

def load(p):
    d = {}
    if not os.path.exists(p):
        return d, ""
    commit = ""
    for line in open(p, encoding="utf-8", errors="replace"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        be, test, avg, sd, commit = parts
        d[test] = (float(avg), float(sd))
    return d, commit

vk, vk_c = load(sys.argv[1])
ro, ro_c = load(sys.argv[2])

print(f"  Vulkan commit: {vk_c or 'n/d'}")
print(f"  ROCm   commit: {ro_c or 'n/d'}")
if vk_c and ro_c and vk_c != ro_c:
    print("  AVISO: commits distintos: parte de la diferencia es de los parches, no del backend.")
print()
if not vk or not ro:
    print("  (faltan resultados de un backend)")
else:
    key = lambda t: (t[:2] != "pp", t)
    print(f"  {'test':<10} {'Vulkan':>10} {'sd':>6} {'ROCm':>10} {'sd':>6} {'ROCm/Vulkan':>12}")
    for t in sorted((x for x in vk if x in ro), key=key):
        va, vs = vk[t]; ra, rs = ro[t]
        ratio = ra / va if va else 0
        print(f"  {t:<10} {va:>10.2f} {vs:>6.2f} {ra:>10.2f} {rs:>6.2f} {ratio:>12.2f}")
    print()
    print("  >1 = ROCm mas rapido. Referencia de akandr/bc250-rocm: prefill 0,96-1,29x,")
    print("  decode 0,81-1,01x a contexto vacio (con sus 13 parches aplicados).")

print()
print("  Notas:")
print("   - Un solo backend a la vez: la GPU es compartida, no corras ambos en paralelo.")
print("   - Este chip cae a 1000 MHz si pasa de ~93 C (-28%): compara reps y mira la sigma.")
print("   - El reset de GPU no funciona aqui: si un run se cuelga, reinicia antes del siguiente.")
PY
