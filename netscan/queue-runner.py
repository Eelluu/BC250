#!/usr/bin/env python3
"""
Sequential Job Queue Runner — v10 (Secure Discord Bot + Precise Cron)
================================================================
Modified for Discord and llama-swap (buun-llama-cpp) integration.
Includes security hardening against prompt injection and unauthorized EXEC.
"""

import subprocess
import json
import os
import re
import shlex
import shutil
import threading
import time
import sys
import signal
import socket
import glob
import base64
import hashlib
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
import discord
from discord import app_commands
from discord.ext import commands
import asyncio
import queue
import traceback

# ─── Configuration ──────────────────────────────────────────────────────────
JOBS_JSON = Path("/opt/netscan/data/jobs.json")
STATE_FILE = Path("/opt/netscan/data/queue-runner-state.json")
LOG_FILE = Path("/opt/netscan/data/queue-runner.log")
DATA_DIR = Path("/opt/netscan/data")

# LLM Proxy (llama-swap / OpenAI format)
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8080/v1")
LLM_SERVICE = "llama-swap" # Service to stop for VRAM management

# Health / resilience
LLM_HEALTH_TIMEOUT_S = 10
LLM_RECOVER_WAIT_S = 60
NETWORK_CHECK_INTERVAL_S = 300
MAX_CONSECUTIVE_FAILURES = 5
GPU_BUSY_MAX_WAIT_S = 3600
SD_NOTIFY = 'NOTIFY_SOCKET' in os.environ

# ─── Discord Chat ────────────────────────────────────────────────────────────
DISCORD_TOKEN = os.environ.get('DISCORD_TOKEN', '')
# 'or 0': int('') would crash startup if the var is present but EMPTY in .env
DISCORD_OWNER_ID = int(os.environ.get('DISCORD_OWNER_ID') or 0)

# Smart Routing Models (Dynamic via Env)
# Gemma 4 12B: multimodal (vision+text), ~7GB Q4, fits 65K context in 16GB VRAM
# .strip() + 'or' so an EMPTY LLM_MODEL in .env falls back to the default instead of
# sending a blank model name to llama-swap (which 404s).
SIGNAL_CHAT_MODEL = os.environ.get("LLM_MODEL", "").strip() or "gemma4-12b-q3"
SIGNAL_CHAT_MAX_EXEC = 15
SIGNAL_EXEC_TIMEOUT_S = 30
SIGNAL_LLM_TIMEOUT_S = 900
SIGNAL_MAX_REPLY = 2000
# Stream the answer into the Discord message as tokens arrive — a big perceived
# latency win on BC-250, where time-to-first-token dominates. Set CHAT_STREAMING=0
# in .env to fall back to one send per turn.
CHAT_STREAMING = os.environ.get("CHAT_STREAMING", "1").strip().lower() not in ("0", "false", "no", "off")
# Discord allows ~5 edits / 5s per message; stay well under it.
STREAM_EDIT_INTERVAL_S = float(os.environ.get("STREAM_EDIT_INTERVAL_S", "2.0"))
# Optional: sync slash commands to this guild for instant availability (global
# sync can take up to an hour to propagate). 0 = global.
DISCORD_GUILD_ID = int(os.environ.get('DISCORD_GUILD_ID') or 0)

# ─── Mode Switching ────────────────────────────────────────────────────────
# Keywords to switch modes (no AI needed — pure string matching)
MODE_SMART_KEYWORDS = ('inteligente', 'intelligent', 'modo inteligente', 'smart mode', 'deep mode')
MODE_ASSISTANT_KEYWORDS = ('asistente', 'assistant', 'modo asistente', 'normal mode')
MODE_SLEEP_KEYWORDS = ('dormir', 'descansar', 'sleep mode', 'bot off', 'apagar bot', 'pausar bot')
MODE_WAKE_KEYWORDS = ('despertar', 'encender bot', 'wake up', 'bot on', 'activar bot')
# Auto-sleep: force sleep mode after this hour if in smart mode (prevents forgetting)
SMART_MODE_CURFEW_HOUR = int(os.environ.get("SMART_MODE_CURFEW_HOUR", "1"))  # 1 = 1:00 AM
# Current mode: 'assistant' (default), 'smart', or 'sleep'
_current_mode = 'assistant'

# ─── Stable Diffusion ──────────────────────────────────────────────────────
SD_CLI = os.environ.get("SD_CLI", "/opt/stable-diffusion.cpp/build/bin/sd-cli")
SD_MODELS_BASE = os.environ.get("SD_MODELS_DIR", "/opt/stable-diffusion.cpp/models")

# Model directories (relative to SD_MODELS_BASE, overridable via env)
SD_FLUX_DIR = os.environ.get("SD_FLUX_DIR", f"{SD_MODELS_BASE}/flux")
SD_FLUX2_DIR = os.environ.get("SD_FLUX2_DIR", f"{SD_MODELS_BASE}/flux2")
SD_WAN_DIR = os.environ.get("SD_WAN_DIR", f"{SD_MODELS_BASE}/wan")
SD_QWEN_DIR = os.environ.get("SD_QWEN_DIR", f"{SD_MODELS_BASE}/qwen_image")

# Individual model files (set in .env to override auto-detection)
# Accepts .gguf AND .safetensors — both formats supported by stable-diffusion.cpp

# HiDream-O1 (PRIMARY — pixel-space unified transformer, no VAE/CLIP/T5 needed)
SD_HIDREAM_DIR = os.environ.get("SD_HIDREAM_DIR", f"{SD_MODELS_BASE}/hidream")
SD_HIDREAM_MODEL = os.environ.get("SD_HIDREAM_MODEL", "")  # e.g. HiDream-O1-Image-Dev-BF16.safetensors

# VRAM: BF16 HiDream (~16.4GB) exceeds 16GB VRAM, so quantize only the LLM mlp
# gate/up_proj + self_attn to Q8_0 at load (--tensor-type-rules). Keeping
# mlp.down_proj in BF16 avoids the Vulkan noise bug (stable-diffusion.cpp #1496).
# Shared by both the generate-image and image-editing sd-cli invocations.
HIDREAM_TENSOR_RULES = (
    "^model\\.language_model\\.layers\\.[0-9]+\\.mlp\\.(gate|up)_proj\\.weight=q8_0,"
    "^model\\.language_model\\.layers\\.[0-9]+\\.self_attn\\.[kqvo]_proj\\.weight=q8_0"
)

# Qwen-Image 2.1 (SOTA text-to-image + native instruction-based image editing)
SD_QWEN_MODEL = os.environ.get("SD_QWEN_MODEL", "")        # e.g. qwen_image_2.1-Q4_K.gguf
SD_QWEN_VAE = os.environ.get("SD_QWEN_VAE", "")            # e.g. qwen_image_2.1_vae_bf16.safetensors
SD_QWEN_LLM = os.environ.get("SD_QWEN_LLM", "")            # e.g. Qwen3VL-8B-Instruct-Q4_K_M.gguf
SD_QWEN_VISION = os.environ.get("SD_QWEN_VISION", "")      # e.g. mmproj-Qwen3VL-8B-Instruct-F16.gguf

# FLUX.1 (text-to-image, fast — fallback)
SD_FLUX_MODEL = os.environ.get("SD_FLUX_MODEL", "")   # e.g. flux1-schnell-q4_k.gguf or flux1-dev-Q4_K_M.gguf
SD_FLUX_VAE = os.environ.get("SD_FLUX_VAE", "")       # e.g. ae.safetensors
SD_FLUX_CLIP_L = os.environ.get("SD_FLUX_CLIP_L", "")  # e.g. clip_l.safetensors
SD_FLUX_T5XXL = os.environ.get("SD_FLUX_T5XXL", "")    # e.g. t5-v1_1-xxl-encoder-Q4_K_M.gguf
# FLUX.2 (text-to-image, best quality)
SD_FLUX2_MODEL = os.environ.get("SD_FLUX2_MODEL", "")   # e.g. flux-2-klein-9b-Q4_0.gguf
SD_FLUX2_VAE = os.environ.get("SD_FLUX2_VAE", "")      # e.g. flux2-vae.safetensors
SD_FLUX2_LLM = os.environ.get("SD_FLUX2_LLM", "")      # e.g. qwen3-8b-Q4_K_M.gguf
# FLUX Kontext (image editing)
SD_KONTEXT_MODEL = os.environ.get("SD_KONTEXT_MODEL", "")  # e.g. flux1-kontext-dev-Q4_K_M.gguf
# WAN 2.2 (video generation — two-pass: high-noise + low-noise diffusion models)
SD_WAN_MODEL = os.environ.get("SD_WAN_MODEL", "")       # e.g. Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf or .safetensors
SD_WAN_HIGH_NOISE = os.environ.get("SD_WAN_HIGH_NOISE", "")  # e.g. Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf (2.2 two-pass)
SD_WAN_VAE = os.environ.get("SD_WAN_VAE", "")           # e.g. wan_2.1_vae.safetensors
SD_WAN_T5XXL = os.environ.get("SD_WAN_T5XXL", "")       # e.g. umt5-xxl-encoder-Q4_K_M.gguf
# ESRGAN (upscale)
SD_ESRGAN_MODEL = os.environ.get("SD_ESRGAN_MODEL", f"{SD_MODELS_BASE}/esrgan/RealESRGAN_x4plus.pth")

# Sampling knobs for Qwen-Image 2.1. The defaults are exactly what sd.cpp's own
# examples use (--cfg-scale 6.0, 20 steps, 512px). ComfyUI workflows often run a
# much lower CFG for editing, but that does NOT transfer to sd.cpp (its
# multimodal conditioning path differs) — test before changing. Steps above the
# default, and sizes above 512, cost VRAM a 16 GB UMA box may not have.
SD_QWEN_CFG_SCALE = float(os.environ.get("SD_QWEN_CFG_SCALE", "6.0"))
SD_QWEN_STEPS = int(os.environ.get("SD_QWEN_STEPS", "20"))
SD_QWEN_SIZE = int(os.environ.get("SD_QWEN_SIZE", "512"))
# Extra flags appended as `--model-args`. Recommended on this box:
#   SD_QWEN_MODEL_ARGS="qwen_image_2_1_prefix_cache_type=q8_0"
# which stores the reference/text prefix cache in 8-bit (~1 GiB saved per condition).
SD_QWEN_MODEL_ARGS = os.environ.get("SD_QWEN_MODEL_ARGS", "").strip()
# Editing (Qwen-Image 2.1 is the only editor here). Kept separate from the
# generation knobs because the two want different settings, and the output size
# must follow the reference's aspect ratio instead of being squashed square.
SD_QWEN_EDIT_CFG_SCALE = float(os.environ.get("SD_QWEN_EDIT_CFG_SCALE", str(SD_QWEN_CFG_SCALE)))
SD_QWEN_EDIT_STEPS = int(os.environ.get("SD_QWEN_EDIT_STEPS", str(SD_QWEN_STEPS)))
# Long side of the edit output, snapped to a multiple of 32; the short side keeps
# the reference's aspect. SD_QWEN_EDIT_SQUARE=1 restores the old square squash.
SD_QWEN_EDIT_SIZE = int(os.environ.get("SD_QWEN_EDIT_SIZE", str(SD_QWEN_SIZE)))
SD_QWEN_EDIT_SQUARE = os.environ.get("SD_QWEN_EDIT_SQUARE", "0").strip().lower() in ("1", "yes", "true", "on")
# --offload-to-cpu keeps weights in RAM, but on this UMA box the driver's
# free-memory report wraps at 1024px (see the sd.cpp log). SD_QWEN_OFFLOAD=gpu
# drops the flag and lets sd.cpp manage the whole pool.
SD_QWEN_OFFLOAD = os.environ.get("SD_QWEN_OFFLOAD", "cpu").strip().lower() not in ("no", "0", "false", "gpu")
# Negative prompt for image jobs ("" = none, the sd.cpp default).
SD_NEGATIVE_PROMPT = os.environ.get("SD_NEGATIVE_PROMPT", "").strip()

# ── Z-Image Turbo (fast distilled DiT: 8 steps, cfg 1.0) ────────────────────
# Much lighter than Qwen-Image 2.1: a 4B text encoder (Qwen3-4B) instead of the
# 8B Qwen3-VL, and 8 steps without CFG. Its VAE is the same FLUX
# `ae.safetensors`, so on this box nothing else needs downloading.
SD_ZIMAGE_DIR = os.environ.get("SD_ZIMAGE_DIR", f"{SD_MODELS_BASE}/z_image")
SD_ZIMAGE_MODEL = os.environ.get("SD_ZIMAGE_MODEL", "")  # e.g. z_image_turbo-Q4_K.gguf
SD_ZIMAGE_VAE = os.environ.get("SD_ZIMAGE_VAE", "")      # e.g. .../flux/ae.safetensors
SD_ZIMAGE_LLM = os.environ.get("SD_ZIMAGE_LLM", "")      # e.g. Qwen3-4B-Instruct-2507-Q4_K_M.gguf
# Z-Image Turbo is distilled for few steps at cfg 1.0. There is no
# classifier-free guidance at cfg 1.0, so a negative prompt does nothing — do
# NOT reuse the Qwen knobs (cfg 6.0 / 20 steps): that burns the image.
SD_ZIMAGE_CFG_SCALE = float(os.environ.get("SD_ZIMAGE_CFG_SCALE", "1.0"))
SD_ZIMAGE_STEPS = int(os.environ.get("SD_ZIMAGE_STEPS", "8"))
SD_ZIMAGE_SIZE = int(os.environ.get("SD_ZIMAGE_SIZE", "1024"))

# Which image model a request uses. "auto" prefers Z-Image Turbo (fast), then
# Qwen-Image 2.1 (slow, multimodal editing + alpha), then HiDream, then FLUX.
# A prompt can override it with a leading `--model=<x>` or `@<x>`.
IMAGE_MODELS = ("auto", "zimage", "qwen", "hidream", "flux", "flux2")
SD_IMAGE_MODEL = os.environ.get("SD_IMAGE_MODEL", "auto").strip().lower()
if SD_IMAGE_MODEL not in IMAGE_MODELS:
    SD_IMAGE_MODEL = "auto"


def _split_image_model(text, model=None):
    """Detach an optional leading model selector from a prompt.

    Accepts `--model=zimage <prompt>` and `@zimage <prompt>` so every caller
    (slash command, deferred GPU queue, the LLM's `EXEC: generate-image ...`
    line) can pick the model without extra plumbing.
    """
    text = (text or "").strip()
    selected = model if model in IMAGE_MODELS else SD_IMAGE_MODEL
    for pattern in (r'^--model=([a-z0-9]+)\s+(.*)$', r'^@([a-z0-9]+)\s+(.*)$'):
        match = re.match(pattern, text, re.IGNORECASE | re.DOTALL)
        if match and match.group(1).lower() in IMAGE_MODELS:
            return match.group(2).strip(), match.group(1).lower()
    return text, selected


def _sd_extra_args():
    """Optional sd.cpp flags shared by the image invocations."""
    extra = []
    if SD_NEGATIVE_PROMPT:
        extra += ["-n", SD_NEGATIVE_PROMPT]
    if SD_QWEN_MODEL_ARGS:
        extra += ["--model-args", SD_QWEN_MODEL_ARGS]
    return extra


def _qwen_offload_flags():
    """--offload-to-cpu (weights in RAM) unless SD_QWEN_OFFLOAD=gpu, plus --fa."""
    flags = []
    if SD_QWEN_OFFLOAD:
        flags.append("--offload-to-cpu")
    flags.append("--fa")
    return flags


def _qwen_args():
    """Sampling flags for the Qwen-Image 2.1 *generation* run.

    Kept as a function so the invocation can be unit-tested without spawning
    sd-cli. Order is irrelevant to sd.cpp's parser.
    """
    return [
        "--cfg-scale", str(SD_QWEN_CFG_SCALE),
        "--sampling-method", "euler",
        *_qwen_offload_flags(),
        "-W", str(SD_QWEN_SIZE), "-H", str(SD_QWEN_SIZE),
        "--steps", str(SD_QWEN_STEPS),
        *_sd_extra_args(),
    ]


def _qwen_edit_args(width, height):
    """Sampling flags for the Qwen-Image 2.1 *edit* run.

    Own cfg/steps (SD_QWEN_EDIT_*) and the reference's aspect ratio as output
    size, both multiples of 32 as sd.cpp requires.
    """
    return [
        "--cfg-scale", str(SD_QWEN_EDIT_CFG_SCALE),
        "--sampling-method", "euler",
        *_qwen_offload_flags(),
        "-W", str(width), "-H", str(height),
        "--steps", str(SD_QWEN_EDIT_STEPS),
        *_sd_extra_args(),
    ]


def _zimage_args():
    """Sampling flags for Z-Image Turbo (distilled: few steps, cfg 1.0).

    Deliberately separate from _qwen_args(): the two models need very different
    settings, and at cfg 1.0 there is no guidance so no negative prompt is sent.
    """
    return [
        "--cfg-scale", str(SD_ZIMAGE_CFG_SCALE),
        "--sampling-method", "euler",
        "--offload-to-cpu", "--fa",
        "-W", str(SD_ZIMAGE_SIZE), "-H", str(SD_ZIMAGE_SIZE),
        "--steps", str(SD_ZIMAGE_STEPS),
    ]


# Output / timeouts
SD_OUTPUT_PATH = "/tmp/sd-output.png"
SD_VIDEO_OUTPUT = "/tmp/sd-output.avi"
SD_VIDEO_MP4 = "/tmp/sd-output.mp4"
SD_TIMEOUT_S = 300
SD_VIDEO_TIMEOUT_S = 2400
# Editing is the slowest image path: it loads the diffusion model *and* the 8B
# text encoder *and* the vision projector, conditions twice (image + text) and then
# samples. On this box a Qwen-Image 2.1 edit takes several minutes, so it gets its
# own, larger budget instead of the generation 300 s.
SD_EDIT_TIMEOUT_S = int(os.environ.get("SD_EDIT_TIMEOUT_S", "900"))
SD_UPSCALE_TILE_SIZE = 192
SD_UPSCALE_TIMEOUT_S = 120


def _find_model(directory, env_value, *candidates):
    """Find a model file: env var takes priority, then scan the directory tree
    (recursively, so models in subfolders like WAN 2.2 HighNoise/ are found)
    for candidate names. Supports .gguf, .safetensors and .pth extensions."""
    # 1. If env var is set and file exists, use it directly
    if env_value:
        # Could be absolute path or just filename
        if os.path.isabs(env_value) and os.path.exists(env_value):
            return env_value
        # Try relative to directory
        rel = os.path.join(directory, env_value)
        if os.path.exists(rel):
            return rel
    # 2. Auto-detect: walk the tree ONCE, then match candidate names (try multiple
    # extensions). Walks recursively so models in subfolders (e.g. WAN 2.2
    # HighNoise/ + LowNoise/) are found too.
    if os.path.isdir(directory):
        found = []  # (root, basename) pairs, top-down order
        for root, _, files in os.walk(directory):
            for f in files:
                found.append((root, f))
        extensions = ('.gguf', '.safetensors', '.pth')
        for name in candidates:
            for ext in extensions:
                fname = name + ext
                for root, f in found:
                    if f == fname:
                        return os.path.join(root, fname)
            # Also try exact name (user might have renamed)
            for root, f in found:
                if f == name:
                    return os.path.join(root, name)
        # 3. Fuzzy / partial match (case-insensitive substring)
        for name in candidates:
            needle = name.lower().replace("-", "_")
            for root, f in found:
                f_norm = f.lower().replace("-", "_")
                if needle in f_norm and any(f.lower().endswith(ext) for ext in extensions):
                    return os.path.join(root, f)
    return None

# ─── Vision (unified: same model for chat + vision, Gemma 4 12B is natively multimodal) ──
VISION_TIMEOUT_S = 300  # 5 min for vision inference

WHISPER_CLI = "/opt/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/opt/whisper.cpp/models/ggml-large-v3-turbo.bin"
WHISPER_THREADS = 6

# ─── Job scheduling constants ──────────────────────────────────────────────
USEFUL_HOURS_START = int(os.environ.get('USEFUL_HOURS_START', '9'))
USEFUL_HOURS_END = int(os.environ.get('USEFUL_HOURS_END', '21'))
DEFAULT_TIMEOUT_S = 3600
GPU_CLK_IDLE_MHZ = 1200
GPU_SCLK_PATH = "/sys/class/drm/card1/device/pp_dpm_sclk"

# ─── Globals ────────────────────────────────────────────────────────────────
running = True
_deferred_sd_queue = []
# True while a cron job is actually executing. The chat worker runs in its own
# thread, so without this flag a "generate-image" request could stop llama-swap
# out from under a running job. When set, media work is deferred (same path as
# the existing "GPU busy" handling) instead of racing the job.
_job_active = False
_chat_histories = {} # thread_id -> list of {"role": ..., "content": ...}
_thread_last_active = {} # thread_id -> timestamp of last message
CHAT_HISTORY_MAX_MB = 100  # Soft cap: evict LRU threads when exceeded
CHAT_HISTORY_TARGET_MB = 80  # Evict down to this target
CHAT_CLEANUP_INTERVAL_S = 300  # Check every 5 minutes

def signal_handler(sig, frame):
    global running
    log("Received shutdown signal, finishing current job...")
    running = False

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def _vram_free_mb():
    """Free GPU VRAM in MB. Tries rocm-smi first (ROCm stack), then falls back
    to the amdgpu sysfs interface (/sys/class/drm/card*/device/mem_info_vram_*)
    which works on any amdgpu driver without ROCm installed.
    Returns None if the GPU memory can't be read (e.g. no AMD GPU, or the
    tooling is missing) — callers should treat None as "unknown", not 0."""
    # 1) rocm-smi --showmeminfo vram
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout:
            total_kb = used_kb = None
            for line in out.stdout.splitlines():
                m = re.search(r"vram\s+total\s*\(KB\)\s*:\s*(\d+)", line)
                if m:
                    total_kb = int(m.group(1))
                m = re.search(r"vram\s+used\s*\(KB\)\s*:\s*(\d+)", line)
                if m:
                    used_kb = int(m.group(1))
            if total_kb is not None and used_kb is not None:
                return (total_kb - used_kb) / 1024.0
    except Exception:
        pass
    # 2) amdgpu sysfs (bytes)
    try:
        for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
            total_f = os.path.join(card, "device", "mem_info_vram_total")
            used_f = os.path.join(card, "device", "mem_info_vram_used")
            if os.path.isfile(total_f) and os.path.isfile(used_f):
                with open(total_f) as f:
                    total_b = int(f.read().strip())
                with open(used_f) as f:
                    used_b = int(f.read().strip())
                if total_b > 0:
                    return (total_b - used_b) / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


def _sd_estimate_mb(*paths):
    """Rough VRAM estimate (MB) for sd-cli from model file sizes on disk.
    Weights are usually the dominant term; activations/KV add a bit on top."""
    total = 0.0
    for p in paths:
        if p and os.path.isfile(p):
            try:
                total += os.path.getsize(p) / (1024.0 * 1024.0)
            except Exception:
                pass
    return total


def _warn_sd_vram(model_label, needed_mb, headroom_mb=1024):
    """Log a warning (never blocks) when free VRAM is likely too small for the
    requested model before sd-cli is launched. HiDream with --tensor-type-rules
    runs at ~68% of its on-disk size; pass the already-adjusted needed_mb."""
    free_mb = _vram_free_mb()
    if free_mb is None:
        log(f"  VRAM: cannot read GPU memory (rocm-smi/sysfs unavailable) — {model_label} needs ~{needed_mb:.0f} MB")
        return
    if free_mb < needed_mb + headroom_mb:
        log(f"  ⚠️ VRAM: only {free_mb:.0f} MB free but {model_label} needs ~{needed_mb:.0f} MB (+{headroom_mb:.0f} headroom) — generation may fail")
    else:
        log(f"  VRAM OK: {model_label} needs ~{needed_mb:.0f} MB, {free_mb:.0f} MB free")


# ─── Logging & Systemd ──────────────────────────────────────────────────────
def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, 'a') as f: f.write(line + '\n')
    except Exception: pass

def sd_notify(state):
    addr = os.environ.get('NOTIFY_SOCKET')
    if not addr: return
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        if addr.startswith('@'): addr = '\0' + addr[1:]
        sock.sendto(state.encode(), addr)
        sock.close()
    except Exception: pass

def sd_watchdog_ping(): sd_notify('WATCHDOG=1')
def sd_status(msg): sd_notify(f'STATUS={msg}')

# ─── Health Checks ──────────────────────────────────────────────────────────
def is_llm_healthy():
    try:
        req = urllib.request.Request(f"{LLM_URL}/models")
        with urllib.request.urlopen(req, timeout=LLM_HEALTH_TIMEOUT_S) as resp:
            return resp.status == 200
    except Exception: return False

_llm_models_cache = (0.0, [])
def get_llm_models(max_age_s=60):
    """Fetch the model ids advertised by the LLM proxy (/v1/models), cached briefly.
    Returns [] if the proxy is unreachable — callers treat that as "unknown", not empty."""
    global _llm_models_cache
    now = time.time()
    if now - _llm_models_cache[0] < max_age_s:
        return _llm_models_cache[1]
    try:
        req = urllib.request.Request(f"{LLM_URL}/models")
        with urllib.request.urlopen(req, timeout=LLM_HEALTH_TIMEOUT_S) as resp:
            data = json.loads(resp.read())
        ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
        _llm_models_cache = (now, ids)
        return ids
    except Exception:
        return []

def check_llm_model_config():
    """Warn loudly if the configured LLM model name is not advertised by llama-swap.
    Common cause of 'HTTP 404' on chat: the interactive model menu writes config.yaml
    keys using the FILENAME as alias, so an LLM_MODEL typed by hand (or left over from
    an older configure run) doesn't match any key and every request 404s."""
    ids = get_llm_models()
    if not ids:
        log(f"  ⚠️ Could not read model list from {LLM_URL}/models — cannot verify LLM_MODEL")
        return
    if SIGNAL_CHAT_MODEL not in ids:
        log(f"  ⚠️ LLM_MODEL='{SIGNAL_CHAT_MODEL}' is NOT in llama-swap (available: {', '.join(ids)})")
        log(f"    Fix: run './install.sh configure' and assign the role in the model selection menu,"
            f" or set LLM_MODEL in /opt/netscan/.env to one of the exact names above, then"
            f" restart queue-runner.")
    else:
        log(f"  ✓ LLM_MODEL '{SIGNAL_CHAT_MODEL}' found in llama-swap")

def is_useful_time():
    hour = datetime.now().hour
    if USEFUL_HOURS_START <= USEFUL_HOURS_END:
        return USEFUL_HOURS_START <= hour < USEFUL_HOURS_END
    else:
        return hour >= USEFUL_HOURS_START or hour < USEFUL_HOURS_END

def wait_for_llm(max_wait_s=300):
    wait = 5
    total = 0
    while total < max_wait_s and running:
        if is_llm_healthy(): return True
        log(f"  LLM Proxy not ready, waiting {wait}s")
        sd_watchdog_ping()
        time.sleep(wait)
        total += wait
        wait = min(wait * 2, 60)
    return is_llm_healthy()

def is_gpu_idle():
    try:
        with open(GPU_SCLK_PATH) as f:
            for line in f:
                if '*' in line:
                    mhz = int(line.split(':')[1].strip().rstrip('Mhz *'))
                    return mhz < GPU_CLK_IDLE_MHZ
        return True
    except Exception: return True

# ─── Security Hardening ─────────────────────────────────────────────────────
# Read-only file tools may only touch these roots. Deny-by-default: without
# this, `cat /proc/self/environ` would leak DISCORD_TOKEN to the chat.
READABLE_ROOTS = ('/opt/netscan/', '/tmp/')
# Script interpreters must target scripts under this root.
SCRIPT_ROOT = '/opt/netscan/'


def _resolve_arg(arg, base='/opt/netscan'):
    """Resolve a CLI path argument to an absolute real path (for allowlist checks)."""
    if os.path.isabs(arg):
        return os.path.realpath(arg)
    return os.path.realpath(os.path.join(base, arg))


def _path_in_roots(path_arg, roots):
    """True if a path argument resolves inside one of the allowed roots."""
    if not path_arg:
        return True
    resolved = _resolve_arg(path_arg)
    return any(resolved == r.rstrip('/') or resolved.startswith(r) for r in roots)


# ── EXEC acotado (límites de recursos + sandbox opcional + allowlist declarativa).
# Capa opcional: si el módulo no está, el bot se comporta exactamente como antes.
try:
    import exec_policy
except Exception:  # noqa: BLE001
    exec_policy = None


def is_command_safe(cmd):
    """Checks if a command is allowed and safe to execute."""
    # 1. Binary whitelist
    ALLOWED_TOOLS = ['ddgr', 'curl', 'ls', 'cat', 'python3', 'bash', 'generate-image', 'generate-video', 'upscale-image', 'edit-image', 'save-memory', 'add-lesson', 'add-correction', 'forget-memory', 'list-memory', 'add-function', 'remove-function', 'list-functions', 'propose-script', 'tool-help', 'edit-file']
    try:
        parts = shlex.split(cmd)
    except: return False, "Malformed command"
    
    if not parts: return False, "Empty command"
    binary = os.path.basename(parts[0])
    if binary not in ALLOWED_TOOLS:
        return False, f"Tool '{binary}' not allowed"

    # 2. Block shell injection characters
    dangerous = [';', '&', '|', '>', '<', '`', '$', '(', ')']
    if any(c in cmd for c in dangerous):
        return False, "Shell metacharacters forbidden"

    # 3. Block access to sensitive system files. Matched component-wise so a
    #    URL like https://api.github.com/... does not trip the '.git' check.
    forbidden_names = {'.env', '.git', 'chat-history.json', 'id_rsa', 'shadow', 'passwd'}
    for tok in parts:
        comps = tok.replace('\\', '/').rstrip('/').split('/')
        if any(c in forbidden_names for c in comps):
            return False, "Access to sensitive file denied"

    # 4. File-reading tools are confined to READABLE_ROOTS (deny by default).
    #    Flags are skipped; `ls` / `cat` with no path operate on the cwd.
    if binary in ('cat', 'ls'):
        for arg in [a for a in parts[1:] if not a.startswith('-')]:
            if not _path_in_roots(arg, READABLE_ROOTS):
                return False, f"{binary} denied outside {', '.join(READABLE_ROOTS)}"

    # 5. Interpreters must target scripts under SCRIPT_ROOT (both python3 and
    #    bash — previously `bash <any path>` was unconstrained, which combined
    #    with `curl -o` allowed arbitrary code execution).
    if binary in ('python3', 'bash'):
        script = next((a for a in parts[1:] if not a.startswith('-')), None)
        if script is None:
            return False, f"{binary} requires an explicit script path"
        if not _path_in_roots(script, (SCRIPT_ROOT,)):
            return False, f"{binary} scripts must be in {SCRIPT_ROOT}"

    # 6. curl must not write to disk or read options from a file — this closes
    #    the `curl -o payload.sh <url>` + `bash payload.sh` RCE chain.
    if binary == 'curl':
        write_flags = ('--output', '--remote-name', '--upload-file', '--config',
                       '--output-dir', '--create-dirs')
        for a in parts[1:]:
            if a in write_flags or any(a.startswith(f + '=') for f in write_flags):
                return False, "curl file write/upload forbidden"
            if a[:2] in ('-o', '-O', '-T', '-K'):
                return False, "curl file write/upload forbidden"

    return True, "Safe"

# ─── Discord Bot ────────────────────────────────────────────────────────────
# ─── Discord error helpers ──────────────────────────────────────────────────
def _exc_summary(e):
    """Format an exception as 'Type: message | HTTP <status> | code <code>' so
    Discord failures carry enough detail to diagnose (token, intents, limits)."""
    parts = [type(e).__name__, str(e) or "(no message)"]
    status = getattr(e, 'status', None)
    if status:
        parts.append(f"HTTP {status}")
    code = getattr(e, 'code', None)
    if code:
        parts.append(f"code {code}")
    text = getattr(e, 'text', None)
    if text:
        body = str(text).replace('\n', ' ').strip()[:300]
        if body:
            parts.append(f"body={body}")
    return " | ".join(parts)


def _discord_login_hint(e):
    """Map a Discord login/gateway exception to a short fix hint for the log."""
    msg = str(e).lower()
    status = getattr(e, 'status', None)
    code = getattr(e, 'code', None)
    if 'privileged' in msg or 'intents' in msg:
        return " (¿intents privilegiados? activa MESSAGE CONTENT INTENT en Discord Developer Portal)"
    if 'improper token' in msg or 'invalid token' in msg or status == 401 or code == 401:
        return " (token inválido — revisa DISCORD_TOKEN en /opt/netscan/.env)"
    if status == 429 or code == 429 or 'rate limit' in msg:
        return " (rate limit 429 — espera y reintenta)"
    if status == 403 or code == 403:
        return " (403 Forbidden — revisa permisos/token)"
    if code == 4014 or '4014' in msg:
        return " (close code 4014: intents no habilitados en el Developer Portal)"
    if 'name or service not known' in msg or 'getaddrinfo' in msg or 'connection refused' in msg or 'timed out' in msg:
        return " (fallo de red/DNS al conectar a discord.com — revisa conectividad)"
    return ""


class OwnerCommandTree(discord.app_commands.CommandTree):
    """CommandTree that only serves DISCORD_OWNER_ID.

    on_message already filters by owner id, but slash commands never go through
    it: without this check anyone who can see the bot could run `/modo dormir`
    (stopping llama-swap) or `/olvidar`.
    """

    async def interaction_check(self, interaction) -> bool:
        if interaction.user.id == DISCORD_OWNER_ID:
            return True
        name = getattr(getattr(interaction, 'command', None), 'name', '?')
        log(f"  Discord: rejected /{name} from unauthorized user {interaction.user.id}")
        try:
            await interaction.response.send_message("⛔ Solo el owner puede usar este bot.", ephemeral=True)
        except Exception:
            pass
        return False

    async def on_error(self, interaction, error):
        """Never leave a slash command silently unacknowledged."""
        log(f"  Discord: slash command error: {_exc_summary(error)}\n{traceback.format_exc()}")
        try:
            msg = f"⚠️ Error: {_exc_summary(error)}"
            if interaction.response.is_done():
                await interaction.followup.send(msg, ephemeral=True)
            else:
                await interaction.response.send_message(msg, ephemeral=True)
        except Exception:
            pass


class DiscordBot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.msg_queue = queue.Queue()
        self.tree = OwnerCommandTree(self)
        _register_slash_commands(self.tree)

    async def on_ready(self):
        log(f"  Discord: Logged in as {self.user}")
        self._register_persistent_views()
        await self._sync_slash_commands()

    async def _sync_slash_commands(self):
        """Publish slash commands once per process.

        Discord rate-limits command syncs, so this must not run on every
        reconnect; a guild sync (DISCORD_GUILD_ID) is instant, a global sync can
        take up to an hour to propagate.
        """
        global _commands_synced
        if _commands_synced:
            return
        try:
            if DISCORD_GUILD_ID:
                await self.tree.sync(guild=discord.Object(id=DISCORD_GUILD_ID))
                log(f"  Discord: slash commands synced to guild {DISCORD_GUILD_ID}")
            else:
                await self.tree.sync()
                log("  Discord: slash commands synced globally (may take up to 1h to appear)")
            _commands_synced = True
        except Exception as e:
            log(f"  Discord: slash command sync failed: {_exc_summary(e)}")

    def _register_persistent_views(self):
        """(Re)register every persistent approval view with THIS client.

        Must run on EVERY on_ready, not only when the pending dict is empty:
        the reconnect loop (start_discord_bot) builds a fresh DiscordBot after
        each gateway drop, and a new Client starts with an empty view store.
        `_pending_scripts`/`_pending_edits` live in module globals and survive
        the drop, so the old `if name not in _pending_scripts` guard skipped
        add_view() and left every button dead. add_view() is keyed by
        custom_id, so calling it again for a known id is harmless.
        """
        registered = 0
        try:
            if PENDING_SCRIPTS_DIR.exists():
                for f in sorted(PENDING_SCRIPTS_DIR.iterdir()):
                    if f.suffix != '.py':
                        continue
                    name = f.stem
                    if name not in _pending_scripts:
                        _pending_scripts[name] = {
                            "code": f.read_text(encoding='utf-8'),
                            "summary": name,
                            "why": "(rehydrated after restart)",
                            "channel": None,
                            "path": str(f),
                            "timestamp": datetime.now().isoformat(),
                        }
                        log(f"  Rehydrated pending script: {name}")
                    self.add_view(_make_approval_view(name))
                    registered += 1
        except Exception as e:
            log(f"  Warning: Could not rehydrate pending scripts: {_exc_summary(e)}")
        try:
            if PENDING_EDITS_DIR.exists():
                for f in sorted(PENDING_EDITS_DIR.glob('*.json')):
                    try:
                        info = json.loads(f.read_text(encoding='utf-8'))
                    except Exception:
                        continue
                    edit_id = info.get('edit_id') or f.stem
                    if edit_id not in _pending_edits:
                        info['edit_id'] = edit_id
                        _pending_edits[edit_id] = info
                        log(f"  Rehydrated pending edit: {edit_id}")
                    self.add_view(_make_edit_view(edit_id))
                    registered += 1
        except Exception as e:
            log(f"  Warning: Could not rehydrate pending edits: {_exc_summary(e)}")
        if registered:
            log(f"  Discord: {registered} persistent approval view(s) registered")

    async def on_error(self, event_method, *args, **kwargs):
        log(f"  Discord: unhandled error in event '{event_method}':\n{traceback.format_exc()}")

    async def on_disconnect(self):
        log("  Discord: gateway disconnected (reconnect will be attempted)")

    async def on_message(self, message):
        if message.author == self.user: return
        if message.author.id != DISCORD_OWNER_ID: return
        
        # --- Mode switch detection (no AI needed) ---
        global _current_mode
        text_lower = (message.content or '').lower()
        for kw in MODE_SMART_KEYWORDS:
            if kw in text_lower:
                if _current_mode != 'smart':
                    _current_mode = 'smart'
                    log(f"  MODE SWITCH: → SMART (keyword: '{kw}')")
                    await message.channel.send(
                        f"🧠 **Modo Inteligente activado** — Modelo potente cargado.\n"
                        f"Los cron jobs están pausados. Accede al LLM desde tu portátil:\n"
                        f"`http://server-ip:8080/v1`\n"
                        f"Para volver al modo asistente, escribe: **{MODE_ASSISTANT_KEYWORDS[0]}**"
                    )
                # Don't queue this message — it was just a mode switch
                return
        for kw in MODE_ASSISTANT_KEYWORDS:
            if kw in text_lower:
                if _current_mode != 'assistant':
                    _current_mode = 'assistant'
                    log(f"  MODE SWITCH: → ASSISTANT (keyword: '{kw}')")
                    # Start llama-swap if coming from sleep
                    subprocess.run(["rc-service", "llama-swap", "start"], capture_output=True)
                    await message.channel.send(
                        f"🦞 **Modo Asistente reanudado** — Modelo ligero activo, cron jobs reanudados."
                    )
                return
        for kw in MODE_SLEEP_KEYWORDS:
            if kw in text_lower:
                if _current_mode != 'sleep':
                    _current_mode = 'sleep'
                    log(f"  MODE SWITCH: → SLEEP (keyword: '{kw}') — stopping llama-swap")
                    subprocess.run(["rc-service", "llama-swap", "stop"], capture_output=True)
                    await message.channel.send(
                        f"💤 **Modo Dormir activado** — llama-swap parado, VRAM liberada.\n"
                        f"Los cron jobs arrancan el modelo automáticamente antes de ejecutarse.\n"
                        f"Para despertar, escribe: **{MODE_WAKE_KEYWORDS[0]}**"
                    )
                return
        for kw in MODE_WAKE_KEYWORDS:
            if kw in text_lower:
                if _current_mode != 'assistant':
                    _current_mode = 'assistant'
                    log(f"  MODE SWITCH: → WAKE/ASSISTANT (keyword: '{kw}') — starting llama-swap")
                    subprocess.run(["rc-service", "llama-swap", "start"], capture_output=True)
                    await message.channel.send(
                        f"☀️ **Despertar** — llama-swap arrancado, modelo cargado en VRAM."
                    )
                return
        
        # In smart mode, bot still listens but just acks
        if _current_mode == 'smart':
            await message.channel.send("🧠 Modo inteligente activo — el LLM está disponible vía OpenAI API en `:8080`. Discord en pausa.")
            return
        # In sleep mode, bot is silent — just ack once and ignore
        if _current_mode == 'sleep':
            await message.channel.send("💤 Modo dormir — escribe **despertar** para activar el bot.")
            return

        attachments = []
        audio_paths = []
        pdf_paths = []
        text_contents = []
        TEXT_EXTENSIONS = ('.txt', '.md', '.json', '.py', '.js', '.ts', '.sh', '.bash', '.zsh', '.yaml', '.yml', '.toml', '.cfg', '.conf', '.ini', '.env', '.log', '.csv', '.tsv', '.xml', '.html', '.css', '.rs', '.go', '.c', '.cpp', '.h', '.hpp', '.java', '.rb', '.php', '.sql', '.r', '.lua', '.pl', '.swift', '.kt', '.scala', '.ex', '.exs', '.hs', '.ml', '.vue', '.svelte', '.jsx', '.tsx', '.makefile', '.cmake', '.dockerfile', '.gitignore', '.editorconfig', '.prettierrc', '.eslintrc', '.babelrc', '.lock', '.sum')
        for att in message.attachments:
            path = f"/tmp/discord_{att.id}_{att.filename}"
            if att.content_type and att.content_type.startswith('image/'):
                await att.save(path)
                attachments.append(path)
            elif att.content_type and att.content_type.startswith('audio/'):
                await att.save(path)
                audio_paths.append(path)
            elif att.content_type == 'application/pdf' or att.filename.lower().endswith('.pdf'):
                await att.save(path)
                pdf_paths.append(path)
            elif att.filename.lower().endswith(TEXT_EXTENSIONS) or (att.content_type and att.content_type.startswith('text/')):
                await att.save(path)
                try:
                    with open(path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read(8000)
                    text_contents.append(f"[File: {att.filename}]\n{content}")
                except Exception:
                    pass

        is_thread = message.channel.type in (
            discord.ChannelType.public_thread,
            discord.ChannelType.private_thread,
        )

        # Main channel → auto-create thread (like chat rooms)
        thread_channel = message.channel
        if not is_thread:
            try:
                # Sanitize thread name: strip newlines, truncate to 100 chars
                raw_name = re.sub(r'[\n\r]+', ' ', message.content).strip() if message.content else "Chat"
                thread_name = raw_name[:100] or "Chat"
                thread_channel = await message.create_thread(
                    name=thread_name,
                    auto_archive_duration=60,  # Archive after 1h of inactivity
                )
                log(f"  Discord: Created thread '{thread_channel.name}'")
            except Exception as e:
                log(f"  Discord: Failed to create thread: {_exc_summary(e)}")
                # Fallback: reply in main channel
                thread_channel = message.channel

        # --- Text-based fallback: aprobar/rechazar <name> (buttons are primary) ---
        if _handle_approval(text_lower, thread_channel):
            return

        # Prepend text file contents to message
        full_text = message.content or ""
        if text_contents:
            full_text = "\n\n".join(text_contents) + ("\n\n" + full_text if full_text else "")

        self.msg_queue.put({
            'text': full_text,
            'thread_id': thread_channel.id,
            'is_thread': True,  # Always True now (threads or auto-created)
            'thread_channel': thread_channel,
            'attachments': attachments,
            'audio_paths': audio_paths,
            'pdf_paths': pdf_paths,
            'reply_to': message
        })

_discord_bot = None
_commands_synced = False  # slash commands are published once per process

def start_discord_bot():
    global _discord_bot
    if not DISCORD_TOKEN:
        log("  Discord: DISCORD_TOKEN not set, chat disabled")
        return
    intents = discord.Intents.default()
    intents.message_content = True
    def run_bot():
        # Self-healing connection loop: a login/gateway failure used to kill
        # this thread SILENTLY (bot stayed 'started' but never logged in again,
        # so Discord went quiet). Now every attempt is logged and retried with
        # backoff; each retry uses a FRESH DiscordBot + event loop because a
        # Client can't be reused after start() fails or closes.
        global _discord_bot
        retry_delay = 5
        while running:
            # Set the thread's event loop BEFORE constructing the Client so
            # the first attempt never depends on a pre-existing thread loop.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            bot = DiscordBot(intents=intents)
            _discord_bot = bot
            try:
                # start() blocks while connected; returns (or raises) when the
                # gateway drops — either way we loop and reconnect.
                loop.run_until_complete(bot.start(DISCORD_TOKEN))
                log("  Discord: connection closed, reconnecting...")
            except Exception as e:
                log(f"  Discord: connection error — {_exc_summary(e)} — retrying in {retry_delay}s{_discord_login_hint(e)}")
                log(f"  Discord: traceback:\n{traceback.format_exc()}")
            finally:
                try:
                    loop.run_until_complete(bot.close())
                except Exception:
                    pass
                try:
                    loop.close()
                except Exception:
                    pass
            if not running:
                break
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)
        _discord_bot = None
    threading.Thread(target=run_bot, daemon=True).start()

def discord_reply(msg_obj, text):
    if not _discord_bot: return False
    for chunk in _split_message(text[:SIGNAL_MAX_REPLY] if len(text) <= SIGNAL_MAX_REPLY else text, SIGNAL_MAX_REPLY):
        future = asyncio.run_coroutine_threadsafe(msg_obj.reply(chunk), _discord_bot.loop)
        try:
            future.result(timeout=15)
        except Exception as e:
            log(f"  Discord: reply failed: {_exc_summary(e)}")
            return False
    return True

def discord_send_to_channel(channel, text):
    """Send a message to a specific channel/thread."""
    if not _discord_bot: return False
    for chunk in _split_message(text[:SIGNAL_MAX_REPLY] if len(text) <= SIGNAL_MAX_REPLY else text, SIGNAL_MAX_REPLY):
        future = asyncio.run_coroutine_threadsafe(channel.send(chunk), _discord_bot.loop)
        try:
            future.result(timeout=15)
        except Exception as e:
            log(f"  Discord: channel send failed: {_exc_summary(e)}")
            return False
    return True

def discord_send_attachment(msg_obj, text, path):
    if not _discord_bot: return False
    async def send(): await msg_obj.reply(text[:SIGNAL_MAX_REPLY], file=discord.File(path))
    future = asyncio.run_coroutine_threadsafe(send(), _discord_bot.loop)
    try:
        return future.result(timeout=30)
    except Exception as e:
        log(f"  Discord: attachment reply failed: {_exc_summary(e)}")
        return False

def discord_send_file_to_channel(channel, text, path):
    """Send a file to a specific channel/thread."""
    if not _discord_bot: return False
    async def send(): await channel.send(text[:SIGNAL_MAX_REPLY], file=discord.File(path))
    future = asyncio.run_coroutine_threadsafe(send(), _discord_bot.loop)
    try:
        return future.result(timeout=30)
    except Exception as e:
        log(f"  Discord: file send failed: {_exc_summary(e)}")
        return False

class DiscordTyping:
    """Context manager that sends Discord typing status continuously while active."""
    def __init__(self, channel):
        self.channel = channel
        self.stop_event = threading.Event()
        self.future = None

    def __enter__(self):
        if not _discord_bot or not self.channel or not getattr(_discord_bot, 'loop', None):
            return self
        if getattr(_discord_bot.loop, 'is_closed', lambda: False)():
            return self
        if not hasattr(self.channel, 'typing'):
            return self

        async def _run():
            try:
                async with self.channel.typing():
                    while not self.stop_event.is_set():
                        await asyncio.sleep(0.5)
            except Exception:
                pass

        try:
            self.future = asyncio.run_coroutine_threadsafe(_run(), _discord_bot.loop)
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()

def _split_message(text, max_length=2000):
    """Split a message into chunks respecting Discord's limit, breaking at newlines when possible."""
    if len(text) <= max_length:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        # Try to break at a newline near the limit
        breakpoint = text.rfind('\n', 0, max_length)
        if breakpoint == -1:
            # No newline found, break hard at max_length
            breakpoint = max_length
        chunks.append(text[:breakpoint].rstrip('\n'))
        text = text[breakpoint:].lstrip('\n')
    return chunks


def _alert_owner(msg):
    """Send an error alert to the owner via Discord DM (reuses _dm_owner)."""
    _dm_owner(f"⚠️ **Error:** {msg}")


# ─── Streaming delivery ─────────────────────────────────────────────────────
def _bot_loop():
    """The Discord event loop if it is usable from another thread, else None."""
    loop = getattr(_discord_bot, 'loop', None) if _discord_bot else None
    if loop is None or getattr(loop, 'is_closed', lambda: False)():
        return None
    return loop


def _stream_send(channel, text):
    """Create the streaming placeholder message. Returns it, or None on failure."""
    loop = _bot_loop()
    if loop is None or not channel:
        return None
    async def _do():
        return await channel.send(text)
    try:
        return asyncio.run_coroutine_threadsafe(_do(), loop).result(timeout=8)
    except Exception as e:
        log(f"  Discord: stream placeholder failed: {_exc_summary(e)}")
        return None


def _stream_edit(message, text):
    """Replace the placeholder content. Returns True on success."""
    loop = _bot_loop()
    if loop is None:
        return False
    async def _do():
        await message.edit(content=text)
    try:
        asyncio.run_coroutine_threadsafe(_do(), loop).result(timeout=8)
        return True
    except Exception as e:
        log(f"  Discord: stream edit failed: {_exc_summary(e)}")
        return False


class DiscordStreamer:
    """Live-updates ONE Discord message with the answer as it is generated.

    Driven from the chat worker thread. Edits are throttled (Discord allows about
    5 edits per 5s per message) and the final answer is always written. If the
    placeholder can't be created, `delivered` stays False and the caller falls
    back to the normal single send, so streaming can never lose a reply.
    """

    PREVIEW_CHARS = 1900  # headroom under Discord's 2000-char limit

    def __init__(self, send_fn, edit_fn, min_interval=None):
        self._send = send_fn
        self._edit = edit_fn
        self._min_interval = STREAM_EDIT_INTERVAL_S if min_interval is None else min_interval
        self._msg = None
        self._text = ""
        self._status = ""
        self._last = 0.0
        self._broken = False
        self.delivered = False

    # ── internals ──
    def _render(self):
        if self._status:
            return self._status[:self.PREVIEW_CHARS]
        text = self._text.strip()
        if len(text) > self.PREVIEW_CHARS:
            text = text[:self.PREVIEW_CHARS] + " …"
        return text or "🦞 …"

    def _flush(self, force=False):
        if self._broken or self._send is None or self._edit is None:
            return False
        if not force and (time.time() - self._last) < self._min_interval:
            return False
        content = self._render()
        if self._msg is None:
            self._msg = self._send(content)
            if self._msg is None:
                self._broken = True     # page couldn't be created; caller will send
                return False
        elif self._edit(self._msg, content) is False:
            return False
        self._last = time.time()
        return True

    # ── API driven by chat_with_llm ──
    def begin_round(self):
        """Start a new LLM round: the final answer streams into the placeholder."""
        self._text = ""
        self._status = ""

    def set_status(self, text):
        """Short progress line (tool running, thinking). Hides raw EXEC noise."""
        self._status = text
        self._flush()

    def clear_status(self):
        """Drop the status line so streamed content can show again (after thinking)."""
        self._status = ""
        self._flush()

    def feed(self, delta):
        """Accumulate a token; switch to a status line if an EXEC directive shows up."""
        self._text += delta
        if self._status:
            return
        if re.search(r'(?m)^\s*EXEC:', self._text):
            self.set_status("🔧 trabajando…")
            return
        self._flush()

    def fail(self, text):
        """Show an error in the placeholder instead of leaving it hanging."""
        self._status = text
        if self._flush(force=True):
            self.delivered = True

    def finalize(self, text):
        """Write the final answer (extra chunks become follow-up messages)."""
        if self._send is None or self._edit is None:
            return
        text = (text or "").strip() or "🦞 (sin respuesta)"
        chunks = _split_message(text, SIGNAL_MAX_REPLY)
        if self._msg is None:
            self._msg = self._send(chunks[0])
            if self._msg is None:
                self._broken = True
                return
        elif not self._edit(self._msg, chunks[0]):
            return
        self.delivered = True
        for extra in chunks[1:]:
            self._send(extra)


def _new_streamer(channel):
    """A streamer bound to `channel`, or None when streaming is unavailable."""
    if not CHAT_STREAMING or not channel or _bot_loop() is None:
        return None
    return DiscordStreamer(lambda t: _stream_send(channel, t),
                            lambda m, t: _stream_edit(m, t))


# ─── Slash commands ─────────────────────────────────────────────────────────
# These make the common actions deterministic and discoverable instead of relying
# on keyword matching. Heavy work is always handed to the chat worker thread
# (never run inline) so the Discord event loop is never blocked.
MODE_CHOICES = [
    app_commands.Choice(name="asistente", value="asistente"),
    app_commands.Choice(name="inteligente", value="inteligente"),
    app_commands.Choice(name="dormir", value="dormir"),
]
_MODE_TARGET = {'asistente': 'assistant', 'inteligente': 'smart', 'dormir': 'sleep'}

# Image-model choices for /imagen. "auto" follows SD_IMAGE_MODEL (Z-Image Turbo
# when it is installed). The value is what lands in the prompt as
# `--model=<value>`, which generate_and_send_image() parses.
IMAGE_MODEL_CHOICES = [
    app_commands.Choice(name="auto (Z-Image Turbo si está)", value="auto"),
    app_commands.Choice(name="Z-Image Turbo (rápido, 8 pasos)", value="zimage"),
    app_commands.Choice(name="Qwen-Image 2.1 (lento, edita/alfa)", value="qwen"),
    app_commands.Choice(name="HiDream-O1", value="hidream"),
    app_commands.Choice(name="FLUX.2", value="flux2"),
]

HELP_TEXT = (
    "**Comandos de Clawd**\n"
    "`/estado` — modo, modelo, cola y salud de llama-swap\n"
    "`/modo <asistente|inteligente|dormir>` — cambiar de modo\n"
    "`/imagen <prompt>` — generar una imagen (`modelo`: auto/zimage/qwen/hidream/flux2)\n"
    "`/investigar <consulta>` — investigación profunda con búsqueda web\n"
    "`/olvidar <clave>` — borrar una preferencia guardada\n"
    "`/ayuda` — esta lista\n\n"
    "También puedes escribir normal en el canal; los modos siguen funcionando por palabra clave.\n"
    f"Modelo de imagen por defecto: `{SD_IMAGE_MODEL}`. Para forzar otro en lenguaje "
    "natural, empieza el texto con `@zimage`, `@qwen`, `@hidream` o `@flux2`."
)


def _status_text():
    """Read-only status snapshot for /estado. Blocks on HTTP -> call in a thread."""
    qsize = 'n/d'
    if _discord_bot:
        try: qsize = _discord_bot.msg_queue.qsize()
        except Exception: pass
    try:
        models = ", ".join(m.split(':')[0] for m in get_llm_models()[:6]) or 'n/d'
    except Exception:
        models = 'n/d'
    try:
        healthy = 'OK' if is_llm_healthy() else 'no responde'
    except Exception:
        healthy = 'n/d'
    return "\n".join([
        f"**Modo:** {_current_mode}",
        f"**Modelo chat:** `{SIGNAL_CHAT_MODEL}`",
        f"**Modelo imagen:** `{SD_IMAGE_MODEL}`",
        f"**Streaming:** {'on' if CHAT_STREAMING else 'off'}",
        f"**Job activo:** {'sí' if _job_active else 'no'}",
        f"**Cola de chat:** {qsize} mensaje(s)",
        f"**Media diferido:** {len(_deferred_sd_queue)}",
        f"**Propuestas pendientes:** {len(_pending_scripts)} script(s), {len(_pending_edits)} edición(es)",
        f"**llama-swap:** {healthy}",
        f"**Modelos:** {models}",
    ])


def _apply_mode_request(value):
    """Deterministic mode switch for /modo. Returns the reply text."""
    target = _MODE_TARGET.get((value or '').strip().lower())
    if target is None:
        return f"Modo no válido: `{value}`. Usa asistente, inteligente o dormir."
    global _current_mode
    if target == _current_mode:
        return f"Ya estás en modo **{value}**."
    _current_mode = target
    log(f"  SLASH: mode → {target.upper()}")
    if target == 'sleep':
        subprocess.run(["rc-service", "llama-swap", "stop"], capture_output=True)
    else:
        subprocess.run(["rc-service", "llama-swap", "start"], capture_output=True)
    return f"Modo cambiado a **{value}**."


def _forget_pref(key):
    """Delete a saved preference by key. Returns the reply text."""
    memory = _get_memory()
    if key in memory.get("preferences", {}):
        del memory["preferences"][key]
        _save_memory(memory)
        return f"Olvidado: `{key}`"
    return f"No encontré la clave `{key}`."


async def _direct_channel(interaction, title):
    """Host channel for a direct action's output.

    Mirrors on_message: in a plain text channel a thread is created so the result
    (an image, a research dump) doesn't get buried in the main channel. Already in
    a thread, or wherever threads aren't available (DMs, forums, missing
    permission), the channel is used as-is.
    """
    ch = interaction.channel
    if ch is None:
        return None
    create_thread = getattr(ch, 'create_thread', None)
    if create_thread is None:
        return ch                     # threads/DMs have no create_thread
    try:
        name = re.sub(r'\s+', ' ', title or 'acción').strip()[:90] or 'acción'
        return await create_thread(name=name, auto_archive_duration=60)
    except Exception as e:
        log(f"  Discord: could not create thread ({_exc_summary(e)}), using the channel")
        return ch


def _enqueue_direct(interaction, tool, arg, channel=None):
    """Queue a direct action for the chat worker so the Discord loop never blocks."""
    if not _discord_bot:
        return False
    channel = channel or interaction.channel
    if channel is None:
        return False
    _discord_bot.msg_queue.put({
        'text': f"[slash {tool}] {arg}",
        'thread_id': channel.id,
        'is_thread': True,
        'thread_channel': channel,
        'attachments': [], 'audio_paths': [], 'pdf_paths': [],
        'reply_to': interaction,
        'direct': (tool, arg),
    })
    return True


def _register_slash_commands(tree):
    """Declare the bot's slash commands. Synced once from DiscordBot.on_ready."""

    @tree.command(name="estado", description="Estado del bot, modo y salud del modelo")
    async def estado(interaction: discord.Interaction):
        try:
            text = await asyncio.to_thread(_status_text)
        except Exception as e:
            text = f"Error obteniendo estado: {_exc_summary(e)}"
        await interaction.response.send_message(text, ephemeral=True)

    @tree.command(name="modo", description="Cambiar de modo")
    @app_commands.choices(modo=MODE_CHOICES)
    async def modo_cmd(interaction: discord.Interaction, modo: app_commands.Choice[str]):
        text = await asyncio.to_thread(_apply_mode_request, modo.value)
        await interaction.response.send_message(text)

    @tree.command(name="imagen", description="Generar una imagen con el modelo local")
    @app_commands.describe(prompt="Qué dibujar",
                           modelo="Modelo a usar (por defecto: el de SD_IMAGE_MODEL)")
    @app_commands.choices(modelo=IMAGE_MODEL_CHOICES)
    async def imagen(interaction: discord.Interaction, prompt: str,
                     modelo: app_commands.Choice[str] = None):
        await interaction.response.defer(thinking=True)
        thread = await _direct_channel(interaction, f"🎨 {prompt}")
        chosen = modelo.value if modelo else SD_IMAGE_MODEL
        arg = f"--model={chosen} {prompt}"
        if _enqueue_direct(interaction, 'generate-image', arg, thread):
            await interaction.followup.send(
                f"🎨 Encolado en {thread.mention} [{chosen}]: `{prompt[:120]}`")
        else:
            await interaction.followup.send("❌ El worker de chat no está activo.")

    @tree.command(name="investigar", description="Investigación profunda con búsqueda web")
    @app_commands.describe(consulta="Qué investigar")
    async def investigar(interaction: discord.Interaction, consulta: str):
        await interaction.response.defer(thinking=True)
        thread = await _direct_channel(interaction, f"🔬 {consulta}")
        if _enqueue_direct(interaction, 'research', consulta, thread):
            await interaction.followup.send(f"🔬 Investigación encolada en {thread.mention}.")
        else:
            await interaction.followup.send("❌ El worker de chat no está activo.")

    @tree.command(name="olvidar", description="Olvidar una preferencia guardada")
    @app_commands.describe(clave="Clave a borrar")
    async def olvidar(interaction: discord.Interaction, clave: str):
        text = await asyncio.to_thread(_forget_pref, clave)
        await interaction.response.send_message(text, ephemeral=True)

    @tree.command(name="ayuda", description="Lista de comandos disponibles")
    async def ayuda(interaction: discord.Interaction):
        await interaction.response.send_message(HELP_TEXT, ephemeral=True)


def _dm_owner(text):
    """Send a message to the owner via Discord DM, split into <=2000-char chunks.
    Returns True if the DM was delivered."""
    if not _discord_bot:
        return False
    chunks = _split_message(text, SIGNAL_MAX_REPLY)
    async def _send():
        try:
            owner = await _discord_bot.fetch_user(DISCORD_OWNER_ID)
            if not owner:
                log(f"  Discord: could not fetch owner (id={DISCORD_OWNER_ID}) for DM")
                return False
            for c in chunks:
                await owner.send(c)
            return True
        except Exception as e:
            log(f"  Discord: owner DM failed: {_exc_summary(e)}")
            return False
    future = asyncio.run_coroutine_threadsafe(_send(), _discord_bot.loop)
    try:
        return future.result(timeout=30)
    except Exception as e:
        log(f"  Discord: owner DM failed (timeout/loop): {_exc_summary(e)}")
        return False


# ─── LLM Integration ────────────────────────────────────────────────────────
HISTORY_FILE = DATA_DIR / "chat-history.json"

def load_history():
    global _chat_histories
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE) as f:
                data = json.load(f)
                _chat_histories = {int(k): v for k, v in data.items()}
                for tid, msgs in _chat_histories.items():
                    _thread_last_active[tid] = time.time() - (len(msgs) * 120)
            log(f"  History loaded: {len(_chat_histories)} threads from {HISTORY_FILE}")
        else:
            log(f"  History file not found: {HISTORY_FILE} — starting fresh")
    except Exception as e:
        log(f"  History load ERROR: {e}")
        _alert_owner(f"History load failed: {e}")

def save_history():
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(_chat_histories, f, ensure_ascii=False)
    except Exception as e:
        log(f"  History save ERROR: {e}")
        _alert_owner(f"History save failed: {e}")

def _estimate_history_size_mb():
    """Estimate RAM usage of chat histories in MB (JSON-serializable approximation)."""
    try:
        raw = json.dumps(_chat_histories, ensure_ascii=False)
        return len(raw.encode('utf-8')) / (1024 * 1024)
    except Exception:
        return 0

def cleanup_chat_history():
    """Evict least-recently-used thread histories when total exceeds soft cap."""
    if not _chat_histories:
        return
    size_mb = _estimate_history_size_mb()
    if size_mb < CHAT_HISTORY_MAX_MB:
        return
    log(f"  History cleanup: {size_mb:.1f}MB exceeds {CHAT_HISTORY_MAX_MB}MB cap")
    # Sort threads by last activity (oldest first)
    sorted_threads = sorted(
        _chat_histories.keys(),
        key=lambda tid: _thread_last_active.get(tid, 0)
    )
    evicted = 0
    for tid in sorted_threads:
        if size_mb <= CHAT_HISTORY_TARGET_MB:
            break
        thread_size = len(json.dumps(_chat_histories[tid], ensure_ascii=False).encode('utf-8')) / (1024 * 1024)
        del _chat_histories[tid]
        _thread_last_active.pop(tid, None)
        size_mb -= thread_size
        evicted += 1
    if evicted:
        log(f"  History cleanup: evicted {evicted} idle threads, now {size_mb:.1f}MB")
        save_history()

def clean_llm_reply(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'^EXEC:.*$', '', text, flags=re.MULTILINE)
    return text.strip()

def _format_history_window(history, max_recent=8):
    """Keep the last `max_recent` messages verbatim. If there are older messages,
    condense them into a compact context summary to save prompt prefill tokens on BC-250."""
    if not history or len(history) <= max_recent:
        return list(history)

    older = history[:-max_recent]
    recent = list(history[-max_recent:])

    summary_items = []
    for m in older:
        role = "AK" if m.get("role") == "user" else "Clawd"
        text = m.get("content", "").strip()
        if text.startswith("Command outputs:"):
            cmds = [l.strip() for l in text.splitlines() if l.startswith("$ ")]
            if cmds:
                text = f"Salida de: {', '.join(cmds[:3])}"
            else:
                text = "Comando ejecutado."
        elif len(text) > 160:
            text = text[:160] + "..."
        summary_items.append(f"- {role}: {text}")

    condensed_text = "Contexto previo resumido:\n" + "\n".join(summary_items)
    return [{"role": "system", "content": condensed_text}] + recent

def _completion_once(messages, temperature=0.7, max_tokens=8192, timeout=None):
    """Single non-streamed chat completion (fallback path)."""
    payload = json.dumps({
        "model": SIGNAL_CHAT_MODEL, "messages": messages, "stream": False,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(f"{LLM_URL}/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout or SIGNAL_LLM_TIMEOUT_S) as resp:
        data = json.loads(resp.read())
    return data['choices'][0]['message']['content'].strip()


def _stream_completion(messages, temperature=0.7, max_tokens=8192, timeout=None, on_reasoning=None):
    """Yield content deltas from the OpenAI-compatible SSE stream.

    Reasoning models (gemma-4, qwen3.x, deepseek...) send the chain of thought as
    `reasoning_content` deltas *before* any `content`. Those are not part of the
    reply, so they are not yielded, but `on_reasoning` gets them so the UI can show
    progress instead of an empty "typing…" for the whole thinking phase.
    """
    payload = json.dumps({
        "model": SIGNAL_CHAT_MODEL, "messages": messages, "stream": True,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(f"{LLM_URL}/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json",
                                          "Accept": "text/event-stream"})
    with urllib.request.urlopen(req, timeout=timeout or SIGNAL_LLM_TIMEOUT_S) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if piece:
                yield piece
                continue
            think = delta.get("reasoning_content") or delta.get("reasoning")
            if think and on_reasoning is not None:
                on_reasoning(think)


def _complete(messages, temperature=0.7, max_tokens=8192, streamer=None):
    """Complete a turn, streaming into `streamer` when possible.

    Falls back to a single non-streamed request when the proxy ignores
    stream=True (it would answer with one JSON body that the SSE parser skips) or
    when the stream breaks before producing any text.
    """
    if streamer is not None:
        text = ""
        _st = {"reason_chars": 0, "last_status": 0.0, "thinking": False, "content": False}

        def _on_reasoning(chunk):
            # El modelo está razonando: sin esto el placeholder se queda vacío y
            # parece colgado. Refresca como mucho cada 2 s (límite de edits de Discord).
            _st["reason_chars"] += len(chunk)
            _st["thinking"] = True
            now = time.time()
            if now - _st["last_status"] >= 2.0:
                _st["last_status"] = now
                streamer.set_status(f"🧠 pensando… ({_st['reason_chars']} chars)")

        try:
            for piece in _stream_completion(messages, temperature, max_tokens,
                                            on_reasoning=_on_reasoning):
                if _st["thinking"] and not _st["content"]:
                    _st["content"] = True
                    streamer.clear_status()   # a partir de aquí manda el contenido
                text += piece
                streamer.feed(piece)
        except urllib.error.HTTPError:
            if not text:
                raise                        # let chat_with_llm map the error
            log(f"  Stream interrupted after {len(text)} chars, keeping partial reply")
        except Exception as e:
            if not text:
                log(f"  Stream failed ({_exc_summary(e)}), falling back to one-shot")
        if text:
            return text
    return _completion_once(messages, temperature, max_tokens)


_FOREIGN_ACTION_RE = re.compile(r'\\?"action\\?"\s*:\s*\\?"([^"\\]+)\\?"', re.IGNORECASE)
# El prompt suele venir dentro de action_input como string escapado
# (`\"prompt\": \"...\"`), así que las comillas y barras son opcionales.
_FOREIGN_PROMPT_RE = re.compile(
    r'\\?"(?:prompt|instruction|text|image_description)\\?"\s*:\s*\\?"((?:[^"\\]|\\.)*?)\\?"',
    re.IGNORECASE)


def _translate_foreign_tool_call(reply):
    """Traduce a EXEC los "tool calls" ajenos que alucinan algunos modelos.

    Caso real: {"action": "dalle.text2im", "action_input": "{ \\"prompt\\": ... }"}.
    El bot sólo entiende `EXEC: <cmd>`, así que en vez de devolver un error
    incomprensible se mapea a la orden equivalente. Devuelve [] si no hay nada.
    """
    if '"action"' not in reply:
        return []
    m_action = _FOREIGN_ACTION_RE.search(reply)
    action = (m_action.group(1) if m_action else "").strip().lower()
    m_prompt = _FOREIGN_PROMPT_RE.search(reply)
    if not m_prompt:
        return []
    try:
        prompt = json.loads(f'"{m_prompt.group(1)}"')   # des-escapa \n, \" ...
    except Exception:
        prompt = m_prompt.group(1)
    prompt = " ".join(prompt.split()).strip()
    if not prompt:
        return []
    if "edit" in action or "img2img" in action or "inpaint" in action:
        return [f"edit-image {prompt}"]
    if any(k in action for k in ("text2im", "image", "imagen", "generate", "dalle")):
        return [f"generate-image {prompt}"]
    return []


def chat_with_llm(user_text, allow_sd=True, attachment_path=None, thread_id=None, msg_obj=None, is_thread=True, thread_channel=None, streamer=None):
    """Chat with LLM. Supports vision — send images as base64 for multimodal analysis."""
    active_channel = thread_channel or (getattr(msg_obj, 'channel', None) if msg_obj else None)
    with DiscordTyping(active_channel):
        history = _chat_histories.get(thread_id, []) if is_thread else []

        # Pre-flight: si el proxy está caído (lo paró un job de media, una prueba
        # manual o un OOM), arráncalo y espera, en vez de devolver "Connection
        # refused" al usuario. Nunca dentro de un job de media, que es quien lo
        # para a propósito.
        if not _job_active:
            try:
                if not is_llm_healthy():
                    log("  LLM no responde; arrancando llama-swap y esperando…")
                    subprocess.run(["rc-service", "llama-swap", "start"], capture_output=True)
                    if wait_for_llm(max_wait_s=180):
                        log("  llama-swap de nuevo en pie")
                    else:
                        log("  llama-swap no respondió ni tras arrancarlo")
            except Exception as _pf_exc:  # nunca debe impedir el turno
                log(f"  Pre-flight LLM: {_pf_exc}")
        
        system_prompt = (
            "Eres Clawd 🦞, el asistente técnico en el servidor de AK (AMD BC-250, Linux embebido). Te comunicas por Discord.\n"
            "TONO: Compañero senior de sistemas y embedded Linux. Directo, técnico, conciso, relajado y al grano. Sin saludos artificiales ni cortesías robóticas. Responde en español por defecto salvo que te hablen en otro idioma o se requiera código/términos en inglés.\n"
            "\n"
            "CONTEXTO DE AK Y EMPLEO:\n"
            "AK es desarrollador de Embedded Linux (kernel drivers, V4L2, cámaras/SerDes, Vulkan, C/C++, Rust).\n"
            "Para cualquier tema de búsqueda de empleo, ofertas laborales, CV o detalles de su experiencia y proyectos, consulta bajo demanda:\n"
            "  • EXEC: cat /opt/netscan/profile.json (filtros, roles preferidos y preferencias laborales)\n"
            "  • EXEC: cat /opt/netscan/cv_master.md (currículum completo y trayectoria técnica)\n"
            "No inventes datos de su experiencia; léelos de los ficheros con EXEC si los necesitas.\n"
            "\n"
            "CAPACIDADES: Imágenes (generar/editar), audio (transcripción), PDFs (texto+OCR).\n"
            "\n"
            "EXEC COMMANDS (use 'EXEC: <cmd>'):\n"
            "  MEDIA: generate-image <prompt>, edit-image <instruction>, upscale-image, generate-video <prompt> (attach an image for image-to-video)\n"
            "    Fondo TRANSPARENTE (PNG con alfa): usa generate-image con este formato exacto — 'This is an RGBA image with transparency. <descripción>. The image has alpha channel and the background is transparent.'\n"
            "    RUTEO OBLIGATORIO: si el usuario ADJUNTA una imagen y pide cambiarla, retocarla, añadirle texto o cualquier cosa sobre ella, usa SIEMPRE `edit-image <instrucción>`. El pipeline SÍ edita píxeles: no digas que no puedes. Tarda varios minutos, avísalo pero no lo evites por eso.\n"
            "    Si NO adjunta imagen y pide crear una nueva, usa `generate-image <prompt>`. Texto dentro de la imagen (carteles, rótulos): también `edit-image` o `generate-image`, con el texto entre comillas en el prompt.\n"
            "    PROTOCOLO: el ÚNICO formato válido es `EXEC: <cmd>` en su propia línea. NUNCA emitas tool-calls JSON de otros formatos ({\"action\": ..., \"action_input\": ..., \"dalle.*\", \"thought\": ...}): aquí no existen y no se ejecutan.\n"
            "  FILES: ls [path], cat <file> (solo dentro de /opt/netscan y /tmp), python3 /opt/netscan/..., bash /opt/netscan/...\n"
            "  SEARCH: ddgr <query>, curl <url> (sin volcar a disco: -o/-O están bloqueados)\n"
            "  MEMORY: save-memory <k> <v>, forget-memory <k>, list-memory, add-lesson <text>, add-correction <wrong> -> <correct>\n"
            "  SKILLS: add-function <name> \"desc\" \"cmd\", remove-function <name>, list-functions\n"
            "  SCRIPTS: propose-script \"summary\" \"why\" <code> — owner approves via Discord buttons\n"
            "  EDIT: edit-file <path> — propose file edits. Owner approves via Discord buttons.\n"
            "    Two formats:\n"
            "    • SEARCH/REPLACE (surgical):\n"
            "      EXEC: edit-file /path/to/file.py\n"
            "      <<<<<<< SEARCH\n"
            "      exact text currently in the file\n"
            "      =======\n"
            "      new text to replace it with\n"
            "      >>>>>>>\n"
            "    • Full rewrite (entire file):\n"
            "      EXEC: edit-file /path/to/file.py\n"
            "      ```python\n"
            "      full content of the file here\n"
            "      ```\n"
            "    Use SEARCH/REPLACE for small changes, full rewrite for new files or major overhauls.\n"
            "  EXAMPLE:\n"
            "    EXEC: edit-file /etc/some-config.conf\n"
            "    <<<<<<< SEARCH\n"
            "    TIMEOUT=30\n"
            "    =======\n"
            "    TIMEOUT=60\n"
            "    >>>>>>>\n"
            "  IMPORTANT: Put the content block AFTER the EXEC line, not on the same line.\n"
            "  HELP: tool-help <name> — get full syntax+examples for any tool\n"
            "\n"
            "EXEC RULES:\n"
            "  - Tienes hasta 15 comandos EXEC por turno. Úsalos con criterio.\n"
            "  - Envía UN comando a la vez si necesitas ver el resultado antes del siguiente paso.\n"
            "  - Puedes enviar múltiples EXEC juntos si son comprobaciones independientes.\n"
            "  - Tras cada ronda EXEC, analiza la salida antes de tu respuesta final o siguiente paso.\n"
            "  - REGLAS DE MEMORIA: guarda preferencias con save-memory al detectar 'recuerda que...', añade lecciones/correcciones ante errores.\n"
            "  - CRÍTICO: Si el usuario te pide guardar o recordar algo, TIENES que emitir el comando 'EXEC: save-memory ...'. Nunca digas solo 'guardado' sin el comando EXEC.\n"
        )
        # Inject memory context into system prompt
        memory_ctx = _build_memory_context()
        if memory_ctx:
            system_prompt += memory_ctx
        
        user_content = user_text or ""
        # Build multimodal content array if image is attached
        if attachment_path and os.path.exists(attachment_path):
            try:
                with open(attachment_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                user_content = [
                    {"type": "text", "text": user_text or "What do you see in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            except Exception:
                user_content = f"[Photo: {attachment_path}] " + user_content
        
        # Save text-only version of user content for history
        if isinstance(user_content, list):
            text_parts = [p['text'] for p in user_content if isinstance(p, dict) and p.get('type') == 'text']
            history_entry_text = ' '.join(text_parts)
        else:
            history_entry_text = user_content

        # Apply rolling window with condensation for older turns to prevent BC-250 prefill lag
        formatted_history = _format_history_window(history, max_recent=8)
        messages = [{"role": "system", "content": system_prompt}] + formatted_history + [{"role": "user", "content": user_content}]
        
        exec_count = 0
        reply = ""
        tool_turns = []
        for _round in range(SIGNAL_CHAT_MAX_EXEC + 1):
            if streamer:
                streamer.begin_round()
            try:
                reply = _complete(messages, streamer=streamer)
            except urllib.error.HTTPError as e:
                # llama-swap returns 500 ('unspecific error: upstream command exited
                # prematurely') when the spawned llama-server dies — the body has the
                # real reason, so surface it plus a hint on where to find the crash.
                _body = ""
                try:
                    _body = e.read().decode("utf-8", "replace")[:300].strip()
                except Exception:
                    pass
                if e.code == 404:
                    available = ", ".join(get_llm_models()) or "unknown"
                    err = (f"🦞 LLM error: model '{SIGNAL_CHAT_MODEL}' not found on the proxy (HTTP 404).\n"
                           f"Available models: {available}\n"
                           f"Fix: run ./install.sh configure and assign the model role in the menu,"
                           f" or set LLM_MODEL in /opt/netscan/.env to one of the exact names above.")
                elif e.code >= 500:
                    err = (f"🦞 LLM error: HTTP {e.code} {e.reason} — proxy: {_body or '(no body)'}\n"
                           f"The inference server (llama-server) likely crashed while loading the model."
                           f" Check 'grep -i llama /var/log/messages' or run the llama-server command"
                           f" from /opt/llama-swap/config.yaml manually to see the real error.")
                else:
                    err = f"🦞 LLM error: HTTP {e.code} {e.reason}"
                if streamer:
                    streamer.fail(err)
                return err
            except Exception as e:
                err = f"🦞 LLM error: {e}"
                if streamer:
                    streamer.fail(err)
                return err

            exec_matches = re.findall(r'^EXEC:\s*(.+)$', reply, re.MULTILINE)
            if not exec_matches:
                # Red de seguridad: algunos modelos emiten tool-calls de OpenAI/DALL·E
                # en vez de `EXEC:` (ver _translate_foreign_tool_call).
                _translated = _translate_foreign_tool_call(reply)
                if _translated:
                    log(f"  ⚠️ tool-call ajeno traducido a EXEC: {_translated}")
                    exec_matches = _translated
            if exec_matches and exec_count < SIGNAL_CHAT_MAX_EXEC:
                outputs = []
                for raw_cmd in exec_matches:
                    cmd = raw_cmd.strip()
                    if exec_count >= SIGNAL_CHAT_MAX_EXEC:
                        break
                    exec_count += 1

                    if streamer:
                        streamer.set_status(f"🔧 `{cmd[:60]}`")

                    # Intercept image/media generation BEFORE the shell-injection
                    # check. Media prompts legitimately contain '&', '(', ')' (e.g.
                    # "cyberpunk & neon", "(masterpiece)") and are NEVER passed to
                    # the shell, so the metacharacter filter must not apply to them.
                    # The binary-name match is stricter than the old substring
                    # search, so only real media commands are exempted.
                    media_tool = None
                    for _t in ("generate-image", "edit-image", "upscale-image", "generate-video"):
                        if cmd == _t or cmd.startswith(_t + " "):
                            media_tool = _t
                            break
                    if media_tool:
                        media_res = None
                        if media_tool == "generate-image":
                            if not allow_sd:
                                _deferred_sd_queue.append(('generate', (msg_obj, thread_channel, cmd.replace("generate-image", "").strip())))
                                media_res = "🦞 GPU busy — queued image."
                            else:
                                media_res = generate_and_send_image(cmd.replace("generate-image", "").strip(), msg_obj, thread_channel)
                        elif media_tool == "edit-image":
                            if not allow_sd:
                                media_res = "🦞 GPU busy — cannot edit right now."
                            elif not attachment_path:
                                media_res = "🦞 No reference image provided for editing."
                            else:
                                media_res = edit_image(cmd.replace("edit-image", "").strip(), attachment_path, msg_obj, thread_channel)
                        elif media_tool == "upscale-image":
                            if not attachment_path:
                                media_res = "🦞 No image provided for upscaling."
                            elif not allow_sd:
                                media_res = "🦞 GPU busy — cannot upscale right now."
                            else:
                                media_res = upscale_image(attachment_path, msg_obj, thread_channel)
                        elif media_tool == "generate-video":
                            if not allow_sd:
                                media_res = "🦞 GPU busy — cannot generate video right now."
                            else:
                                media_res = generate_video(cmd.replace("generate-video", "").strip(), msg_obj, thread_channel, input_image=attachment_path)

                        if media_res is not None:
                            if is_thread and thread_id is not None:
                                if thread_id not in _chat_histories:
                                    _chat_histories[thread_id] = []
                                _chat_histories[thread_id].append({"role": "user", "content": history_entry_text})
                                _chat_histories[thread_id].append({"role": "assistant", "content": f"EXEC: {cmd}\n{media_res}"})
                                if len(_chat_histories[thread_id]) > 40:
                                    _chat_histories[thread_id] = _chat_histories[thread_id][-40:]
                                _thread_last_active[thread_id] = time.time()
                                save_history()
                            return media_res

                    # Security Check: el portero de siempre, y encima la allowlist
                    # declarativa (exec-policy.json) si está disponible.
                    safe, reason = is_command_safe(cmd)
                    if not safe and exec_policy is not None:
                        try:
                            _bin = os.path.basename(shlex.split(cmd)[0])
                        except Exception:
                            _bin = ""
                        _why = exec_policy.extra_allowed(_bin) if _bin else None
                        if _why:
                            safe, reason = True, _why
                            log(f"ℹ️ EXEC por política ({_why}): {_bin}")
                    if not safe:
                        log(f"⚠️ SECURITY BLOCK: {reason} - CMD: {cmd}")
                        outputs.append(f"$ {cmd}\n[Access Denied: {reason}]")
                        continue

                    # Intercept self-improvement commands (memory, functions, propose)
                    out = None
                    mem_output = _handle_memory_command(cmd)
                    if mem_output is not None:
                        out = mem_output
                    elif cmd.startswith('tool-help ') or cmd.strip() == 'tool-help':
                        out = _handle_tool_help(cmd)
                    elif cmd.startswith('propose-script '):
                        m = re.match(r'^propose-script\s+"(.+?)"\s+"(.+?)"\s+(.+)', cmd, re.DOTALL)
                        if m:
                            summary, why, code_or_path = m.group(1), m.group(2), m.group(3).strip()
                            if os.path.isfile(code_or_path):
                                code = Path(code_or_path).read_text(encoding='utf-8')
                            else:
                                code = code_or_path
                            out = _handle_script_proposal(code, summary, why, thread_channel)
                        else:
                            out = "Usage: propose-script \"summary\" \"why needed\" <file_path_or_inline_python>"
                    elif cmd.startswith('edit-file '):
                        out = _handle_edit_file(reply, cmd, thread_channel)
                    else:
                        func_output = _handle_function_command(cmd)
                        if func_output is not None:
                            out = func_output
                        else:
                            try:
                                if exec_policy is not None:
                                    # Límites de recursos por ejecución (+ sandbox si
                                    # EXEC_SANDBOX=1): un script desbocado ya no se
                                    # lleva por delante a llama.cpp.
                                    _rc, out = exec_policy.run(cmd, SIGNAL_EXEC_TIMEOUT_S)
                                else:
                                    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=SIGNAL_EXEC_TIMEOUT_S)
                                    out = (res.stdout + res.stderr).strip() or "[No output]"
                            except Exception as e:
                                out = f"[Error: {e}]"

                    if out is not None:
                        outputs.append(f"$ {cmd}\n{out}")

                combined_output = "\n\n".join(outputs)[:2000]
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"Command outputs:\n{combined_output}"})

                # Record intermediate tool interactions for history continuity
                hist_out = combined_output[:800] + ("\n... [truncated for history]" if len(combined_output) > 800 else "")
                tool_turns.append({"role": "assistant", "content": reply})
                tool_turns.append({"role": "user", "content": f"Command outputs:\n{hist_out}"})
                continue
            break

        reply = clean_llm_reply(reply)
        # Only save history for threads (main channel = stateless)
        # Don't save multimodal messages (with base64 images) — they're too large for JSON
        if is_thread and thread_id is not None:
            if thread_id not in _chat_histories:
                _chat_histories[thread_id] = []
            _chat_histories[thread_id].append({"role": "user", "content": history_entry_text})
            for turn in tool_turns:
                _chat_histories[thread_id].append(turn)
            _chat_histories[thread_id].append({"role": "assistant", "content": reply})
            if len(_chat_histories[thread_id]) > 40:
                _chat_histories[thread_id] = _chat_histories[thread_id][-40:]
            _thread_last_active[thread_id] = time.time()
            save_history()
        if streamer:
            streamer.finalize(reply)
        return reply

def analyze_image_with_vision(path, prompt=""):
    """Analyze image using the same multimodal model as chat (Gemma 4 12B)."""
    try:
        with open(path, "rb") as f: img_b64 = base64.b64encode(f.read()).decode()
        payload = json.dumps({
            "model": SIGNAL_CHAT_MODEL, "stream": False,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt or "Describe this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]}]
        }).encode()
        req = urllib.request.Request(f"{LLM_URL}/chat/completions", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=VISION_TIMEOUT_S) as resp:
            data = json.loads(resp.read())
        return data['choices'][0]['message']['content'].strip()
    except Exception: return None

def _convert_to_wav16k(audio_path):
    """Convert any audio format to 16kHz mono WAV (required for both native audio and whisper)."""
    wav_path = "/tmp/discord_audio.wav"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    conv = subprocess.run([
        ffmpeg, "-y", "-i", audio_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        wav_path
    ], capture_output=True, text=True, timeout=30)
    if conv.returncode != 0 or not os.path.exists(wav_path):
        log(f"  Audio conversion failed: {conv.stderr[:200]}")
        return None
    return wav_path

def _try_native_audio(wav_path, prompt=""):
    """Try native audio via Gemma 4's input_audio API (llama.cpp PR #24118+).
    Returns response text or None if not supported."""
    try:
        with open(wav_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        user_prompt = prompt or "Transcribe and respond to this audio."
        payload = json.dumps({
            "model": SIGNAL_CHAT_MODEL, "stream": False,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
            ]}],
            "max_tokens": 300
        }).encode()
        req = urllib.request.Request(f"{LLM_URL}/chat/completions", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=VISION_TIMEOUT_S) as resp:
            data = json.loads(resp.read())
        reply = data['choices'][0]['message']['content'].strip()
        if reply:
            log(f"  Native audio response: {reply[:200]}")
            return reply
    except Exception as e:
        log(f"  Native audio failed (mmproj may not support audio): {e}")
    return None

def _try_whisper_transcribe(wav_path, prompt=""):
    """Fallback: transcribe with whisper.cpp, then send text to LLM."""
    if not os.path.exists(WHISPER_CLI) or not os.path.exists(WHISPER_MODEL):
        log(f"  Whisper not available ({WHISPER_CLI})")
        return None
    try:
        log(f"  Transcribing with whisper.cpp...")
        whisper_out = subprocess.run([
            WHISPER_CLI,
            "-m", WHISPER_MODEL,
            "-f", wav_path,
            "-t", str(WHISPER_THREADS),
            "--no-prints"
        ], capture_output=True, text=True, timeout=120)
        transcription = whisper_out.stdout.strip()
        if not transcription:
            log("  Whisper returned empty transcription")
            return None
        log(f"  Whisper transcription: {transcription[:200]}")
        combined = f"[Audio transcription]: {transcription}"
        if prompt:
            combined += f"\n\n[User request]: {prompt}"
        return combined
    except Exception as e:
        log(f"  Whisper error: {e}")
    return None

def transcribe_audio(audio_path, prompt=""):
    """Process Discord voice message: native audio (Gemma 4) → whisper.cpp fallback.
    Discord voice messages are OGG Opus — convert to WAV 16kHz first."""
    wav_path = _convert_to_wav16k(audio_path)
    if not wav_path:
        return None
    try:
        # 1. Try native audio first (Gemma 4 + llama.cpp with mmproj)
        result = _try_native_audio(wav_path, prompt)
        if result:
            return result
        # 2. Fallback to whisper.cpp
        result = _try_whisper_transcribe(wav_path, prompt)
        if result:
            return result
        log("  Both native audio and whisper failed")
        return None
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

PDF_PROCESSING_TIMEOUT_S = 300  # 5 min for OCR on large PDFs
PDF_OCR_BATCH_SIZE = 2  # Pages per batch for Gemma 4 OCR
PDF_MAX_PAGES = 50  # Safety limit — very large PDFs get truncated with warning


def _extract_pdf_text(pdf_path):
    """Extract text from PDF using pdftotext (poppler-utils).
    Returns extracted text or None if extraction fails/yields minimal content."""
    pdftotext = shutil.which('pdftotext')
    if not pdftotext:
        log("  pdftotext not found (install poppler-utils)")
        return None
    try:
        res = subprocess.run(
            [pdftotext, '-layout', pdf_path, '-'],
            capture_output=True, text=True, timeout=30
        )
        text = res.stdout.strip()
        # Heuristic: if output is very short or mostly whitespace, it's probably a scanned PDF
        if len(text) < 50:
            return None
        return text
    except Exception as e:
        log(f"  pdftotext error: {e}")
        return None


def _pdf_get_page_count(pdf_path):
    """Get PDF page count using pdfinfo."""
    pdfinfo = shutil.which('pdfinfo')
    if not pdfinfo:
        return 1  # Fallback: assume 1 page
    try:
        res = subprocess.run(
            [pdfinfo, pdf_path],
            capture_output=True, text=True, timeout=10
        )
        for line in res.stdout.split('\n'):
            if line.startswith('Pages:'):
                return int(line.split(':')[1].strip())
    except Exception:
        pass
    return 1


def _pdf_pages_to_images(pdf_path, pages):
    """Convert specific PDF pages to PNG images using pdftoppm.
    pages: list of 1-based page numbers.
    Returns list of image file paths."""
    pdftoppm = shutil.which('pdftoppm')
    if not pdftoppm:
        log("  pdftoppm not found (install poppler-utils)")
        return []
    images = []
    # pdftoppm processes ranges like '1-3' or individual pages '-f 2 -l 2'
    # For clarity, process each page individually
    for page_num in pages:
        out_prefix = f"/tmp/pdf_page_{page_num}"
        try:
            subprocess.run(
                [pdftoppm, '-png', '-r', '150',  # 150 DPI — good balance of quality vs size
                 '-f', str(page_num), '-l', str(page_num),
                 pdf_path, out_prefix],
                capture_output=True, timeout=30
            )
            # pdftoppm outputs: out_prefix-{page}.png
            expected = f"{out_prefix}-{page_num:02d}.png"
            if not os.path.exists(expected):
                # Try without zero-padding
                expected = f"{out_prefix}-{page_num}.png"
            if os.path.exists(expected):
                images.append(expected)
        except Exception as e:
            log(f"  pdftoppm error on page {page_num}: {e}")
    return images


def _ocr_pdf_batch(image_paths, user_prompt=""):
    """Send a batch of page images to Gemma 4 for OCR.
    Returns transcribed text."""
    try:
        content = [
            {"type": "text", "text": user_prompt or "Extract and transcribe ALL text from these PDF pages. Preserve formatting, tables, and structure. Output the text exactly as it appears."}
        ]
        for img_path in image_paths:
            with open(img_path, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })
        payload = json.dumps({
            "model": SIGNAL_CHAT_MODEL, "stream": False,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 8192
        }).encode()
        req = urllib.request.Request(
            f"{LLM_URL}/chat/completions", data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=VISION_TIMEOUT_S) as resp:
            data = json.loads(resp.read())
        return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        log(f"  OCR batch error: {e}")
        return None
    finally:
        for p in image_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def process_pdf(pdf_path, user_prompt=""):
    """Process a PDF: try text extraction first, fall back to OCR via Gemma 4 vision.
    For OCR, processes pages in batches of 2 to manage VRAM and context.
    Returns the combined extracted text."""
    log(f"  PDF processing: {os.path.basename(pdf_path)}")
    page_count = _pdf_get_page_count(pdf_path)
    log(f"  PDF pages: {page_count}")

    # 1. Try text extraction first
    text = _extract_pdf_text(pdf_path)
    if text:
        log(f"  PDF text extracted: {len(text)} chars")
        # If user asked a question about the PDF, wrap it nicely
        if user_prompt:
            return f"[PDF text content ({page_count} pages)]:\n\n{text[:8000]}\n\n---\n[User request]: {user_prompt}"
        return f"[PDF text content ({page_count} pages)]:\n\n{text[:8000]}"

    # 2. Text extraction failed — OCR via Gemma 4 vision
    log(f"  PDF has no extractable text — using OCR (Gemma 4 vision)")
    actual_pages = min(page_count, PDF_MAX_PAGES)
    if page_count > PDF_MAX_PAGES:
        log(f"  PDF too large ({page_count} pages), processing first {PDF_MAX_PAGES}")

    # Process in batches of 2
    all_ocr_text = []
    for batch_start in range(1, actual_pages + 1, PDF_OCR_BATCH_SIZE):
        batch_pages = list(range(batch_start, min(batch_start + PDF_OCR_BATCH_SIZE, actual_pages + 1)))
        log(f"  OCR batch: pages {batch_pages[0]}-{batch_pages[-1]}")

        # Convert pages to images
        image_paths = _pdf_pages_to_images(pdf_path, batch_pages)
        if not image_paths:
            log(f"  pdftoppm failed for pages {batch_pages}")
            continue

        # Send to model for OCR
        batch_prompt = f"These are pages {batch_pages[0]}-{batch_pages[-1]} of a {page_count}-page PDF document. Extract ALL text."
        if user_prompt:
            batch_prompt += f" The user's question about this PDF: {user_prompt}"
        result = _ocr_pdf_batch(image_paths, batch_prompt)
        if result:
            all_ocr_text.append(f"--- Pages {batch_pages[0]}-{batch_pages[-1]} ---\n{result}")

    if not all_ocr_text:
        return "🦞 Could not process PDF — text extraction and OCR both failed. Is poppler-utils installed?"

    combined = "\n\n".join(all_ocr_text)
    log(f"  OCR complete: {len(combined)} chars from {page_count} pages")
    return f"[PDF OCR result ({page_count} pages)]:\n\n{combined[:12000]}"


# ─── Self-Improvement System ─────────────────────────────────────────────
MEMORY_FILE = DATA_DIR / "clawd-memory.json"
FUNCTIONS_FILE = DATA_DIR / "clawd-functions.json"
PENDING_SCRIPTS_DIR = DATA_DIR / "pending-scripts"
PENDING_EDITS_DIR = DATA_DIR / "pending-edits"

# In-memory state for script approval flow (script_name -> {code, summary, why, channel, thread_id})
_pending_scripts = {}
_pending_edits = {}
PENDING_SCRIPT_EXPIRY_S = 86400  # 24 hours — auto-reject scripts older than this


def _pending_edit_file(edit_id):
    return PENDING_EDITS_DIR / f"{edit_id}.json"


def _persist_pending_edit(edit_id, info):
    """Write a pending edit to disk so its buttons survive a bot restart.

    Without this, edit proposals only exist in RAM: after any restart the
    Discord buttons stay visible but the edit is gone and clicking them does
    nothing (the callback has no state left to act on).
    """
    try:
        PENDING_EDITS_DIR.mkdir(parents=True, exist_ok=True)
        # channel is a live Discord object and current_content is display-only
        data = {k: v for k, v in info.items() if k not in ('channel', 'current_content')}
        data['edit_id'] = edit_id
        _pending_edit_file(edit_id).write_text(
            json.dumps(data, ensure_ascii=False), encoding='utf-8')
    except Exception as e:
        log(f"  Warning: could not persist pending edit {edit_id}: {_exc_summary(e)}")


def _discard_pending_edit(edit_id):
    """Remove the persisted copy of a pending edit once it is applied/rejected."""
    try:
        p = _pending_edit_file(edit_id)
        if p.exists():
            p.unlink()
    except Exception:
        pass


def _make_approval_view(script_name: str) -> discord.ui.View:
    """Factory: creates a View with Approve/Reject/Ask buttons whose custom_ids
    embed the script_name (e.g. 'script_approve_my_tool').
    Using a closure ensures the custom_id is baked into the @discord.ui.button
    decorator — the only reliable way in discord.py 2.x (class-level descriptors
    ignore per-instance overrides of custom_id)."""

    class _View(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=None)  # auto-expire handled by _cleanup_expired_scripts

        async def on_error(self, interaction, error, item):
            # Without this, discord.py logs to stderr (journal) and the button
            # silently shows "interaction failed" — invisible in queue-runner.log.
            log(f"  Discord: button '{getattr(item, 'custom_id', '?')}' failed: "
                f"{_exc_summary(error)}\n{traceback.format_exc()}")
            try:
                msg = "⚠️ Error interno al procesar el botón. Revisa los logs del bot."
                if interaction.response.is_done():
                    await interaction.followup.send(msg, ephemeral=True)
                else:
                    await interaction.response.send_message(msg, ephemeral=True)
            except Exception:
                pass

        @discord.ui.button(label="✅ Aprobar", style=discord.ButtonStyle.success,
                           custom_id=f"script_approve_{script_name}")
        async def approve_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
            if interaction.user.id != DISCORD_OWNER_ID:
                await interaction.response.send_message("Solo el owner puede aprobar.", ephemeral=True)
                return
            info = _pending_scripts.get(script_name)
            if info is None:
                await interaction.response.send_message("Este script ya fue procesado o expiró.", ephemeral=True)
                return
            skills_dir = Path("/opt/netscan/skills")
            try:
                skills_dir.mkdir(parents=True, exist_ok=True)
                approved_path = skills_dir / f"{script_name}.py"
                approved_path.write_text(info["code"], encoding='utf-8')
                registry = _get_functions()
                registry["functions"] = [f for f in registry["functions"] if f["name"] != script_name]
                registry["functions"].append({
                    "name": script_name,
                    "description": info["summary"],
                    "command": f"python3 {approved_path}",
                })
                _save_functions(registry)
            except Exception as e:
                # Keep it pending so the owner can retry after fixing the cause
                await interaction.response.send_message(f"❌ No se pudo aprobar: {e}", ephemeral=True)
                log(f"  Script approve ERROR: {script_name}: {e}")
                return
            _pending_scripts.pop(script_name, None)
            pending_path = Path(info["path"])
            if pending_path.exists():
                pending_path.unlink()
            for child in self.children:
                child.disabled = True
            await interaction.response.edit_message(view=self)
            await interaction.followup.send(
                f"✅ **Script aprobado:** `{script_name}`\n"
                f"Guardado en `{approved_path}` y añadido al registry de funciones.",
                ephemeral=False
            )
            log(f"  Script approved via button: {script_name}")

        @discord.ui.button(label="❌ Rechazar", style=discord.ButtonStyle.danger,
                           custom_id=f"script_reject_{script_name}")
        async def reject_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
            if interaction.user.id != DISCORD_OWNER_ID:
                await interaction.response.send_message("Solo el owner puede rechazar.", ephemeral=True)
                return
            info = _pending_scripts.pop(script_name, None)
            if info is None:
                await interaction.response.send_message("Este script ya fue procesado o expiró.", ephemeral=True)
                return
            pending_path = Path(info["path"])
            if pending_path.exists():
                pending_path.unlink()
            for child in self.children:
                child.disabled = True
            await interaction.response.edit_message(view=self)
            await interaction.followup.send(
                f"❌ **Script rechazado:** `{script_name}` — eliminado.",
                ephemeral=False
            )
            log(f"  Script rejected via button: {script_name}")

        @discord.ui.button(label="❓ Preguntar más", style=discord.ButtonStyle.secondary,
                           custom_id=f"script_ask_{script_name}")
        async def ask_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
            if interaction.user.id != DISCORD_OWNER_ID:
                await interaction.response.send_message("Solo el owner.", ephemeral=True)
                return
            info = _pending_scripts.get(script_name)
            if info is None:
                await interaction.response.send_message("Este script ya fue procesado o expiró.", ephemeral=True)
                return
            await interaction.response.send_message(
                f"💬 Pregunta sobre `{script_name}` enviada a Clawd. Espera su respuesta.",
                ephemeral=True
            )
            if _discord_bot:
                _discord_bot.msg_queue.put({
                    'text': (
                        f"AK quiere saber más sobre el script propuesto '{script_name}'. "
                        f"Resumen: {info['summary']}. "
                        f"Razón: {info['why']}. "
                        f"Explica en detalle qué hace, por qué es útil, y si hay algo que mejorar.\n\n"
                        f"Código:\n```python\n{info['code'][:1500]}{'...' if len(info['code']) > 1500 else ''}\n```"
                    ),
                    'thread_id': interaction.channel.id,
                    'is_thread': True,
                    'thread_channel': interaction.channel,
                    'attachments': [],
                    'audio_paths': [],
                    'pdf_paths': [],
                    'reply_to': interaction.message,
                })
            log(f"  Script info requested via button: {script_name}")

    return _View()

def _cleanup_expired_scripts():
    """Auto-reject pending scripts AND edits older than PENDING_SCRIPT_EXPIRY_S (24h).
    Called periodically from main loop."""
    now = datetime.now()

    expired = []
    for name, info in list(_pending_scripts.items()):
        try:
            ts = datetime.fromisoformat(info["timestamp"])
            if (now - ts).total_seconds() > PENDING_SCRIPT_EXPIRY_S:
                expired.append(name)
        except Exception:
            expired.append(name)
    for name in expired:
        info = _pending_scripts.pop(name, None)
        if info is None:
            continue
        # Delete pending file
        pending_path = Path(info["path"])
        if pending_path.exists():
            pending_path.unlink()
        # Notify owner
        channel = info.get("channel")
        if channel:
            discord_send_to_channel(
                channel,
                f"⏰ **Script expirado:** `{name}` — no se aprobó en 24h, eliminado automáticamente."
            )
        log(f"  Script expired (24h timeout): {name}")

    for edit_id, info in list(_pending_edits.items()):
        try:
            ts = datetime.fromisoformat(info.get("timestamp", ""))
            is_expired = (now - ts).total_seconds() > PENDING_SCRIPT_EXPIRY_S
        except Exception:
            is_expired = True
        if not is_expired:
            continue
        _pending_edits.pop(edit_id, None)
        _discard_pending_edit(edit_id)
        channel = info.get("channel")
        if channel:
            discord_send_to_channel(
                channel,
                f"⏰ **Edición expirada:** `{edit_id}` ({info.get('filepath', '?')}) — no se aprobó en 24h."
            )
        log(f"  Edit expired (24h timeout): {edit_id}")

def _send_approval_embed(channel, embed, view):
    """Send an approval embed with buttons to a channel. Must run on Discord event loop."""
    if not _discord_bot: return False
    async def send():
        await channel.send(embed=embed, view=view)
    future = asyncio.run_coroutine_threadsafe(send(), _discord_bot.loop)
    try: return future.result(timeout=15)
    except Exception as e:
        log(f"  Error sending approval embed: {e}")
        return False

# Cached memory/functions (loaded once, persisted on write)
_memory_cache = None
_functions_cache = None
def _get_memory():
    global _memory_cache
    if _memory_cache is None:
        _memory_cache = _load_memory()
    return _memory_cache
def _get_functions():
    global _functions_cache
    if _functions_cache is None:
        _functions_cache = _load_functions()
    return _functions_cache

def _load_memory():
    """Load persistent memory (preferences, lessons, corrections, profile)."""
    try:
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE) as f:
                mem = json.load(f)
                # Ensure profile field exists (migration from older schema)
                if "profile" not in mem:
                    mem["profile"] = ""
                return mem
    except Exception: pass
    return {"preferences": {}, "lessons": [], "corrections": [], "profile": ""}

def _load_functions():
    """Load function registry."""
    try:
        if FUNCTIONS_FILE.exists():
            with open(FUNCTIONS_FILE) as f:
                return json.load(f)
    except Exception: pass
    return {"functions": []}

def _save_memory(memory):
    global _memory_cache
    _memory_cache = memory
    try:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_FILE, 'w') as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log(f"  Memory save error: {e}")
        _alert_owner(f"Memory save failed: {e}")

def _handle_memory_command(cmd):
    """Handle memory-related EXEC commands. Returns output string or None."""
    memory = _get_memory()

    # save-memory <key> <value>
    m = re.match(r'^save-memory\s+(\S+)\s+(.+)', cmd, re.DOTALL)
    if m:
        key, value = m.group(1), m.group(2).strip()
        memory["preferences"][key] = value
        _save_memory(memory)
        return f"Memory saved: {key} = {value[:100]}"

    # add-lesson <lesson text>
    m = re.match(r'^add-lesson\s+(.+)', cmd, re.DOTALL)
    if m:
        lesson = m.group(1).strip()
        memory["lessons"].append({"text": lesson, "added": datetime.now().isoformat()})
        # Keep only last 50 lessons
        memory["lessons"] = memory["lessons"][-50:]
        _save_memory(memory)
        return f"Lesson added ({len(memory['lessons'])} total): {lesson[:100]}"

    # add-correction <what was wrong> -> <what is correct>
    m = re.match(r'^add-correction\s+(.+?)\s*->\s*(.+)', cmd, re.DOTALL)
    if m:
        wrong, correct = m.group(1).strip(), m.group(2).strip()
        memory["corrections"].append({"wrong": wrong, "correct": correct, "added": datetime.now().isoformat()})
        memory["corrections"] = memory["corrections"][-30:]
        _save_memory(memory)
        return f"Correction saved: \"{wrong[:50]}\" → \"{correct[:50]}\""

    # forget-memory <key>
    m = re.match(r'^forget-memory\s+(\S+)', cmd)
    if m:
        key = m.group(1)
        if key in memory["preferences"]:
            del memory["preferences"][key]
            _save_memory(memory)
            return f"Forgot: {key}"
        return f"Key '{key}' not found in memory."

    # list-memory
    if cmd.strip() == 'list-memory':
        parts = []
        if memory["preferences"]:
            parts.append("**Preferences:**")
            for k, v in memory["preferences"].items():
                parts.append(f"  {k}: {v}")
        if memory["lessons"]:
            parts.append(f"**Lessons:** ({len(memory['lessons'])} stored)")
            for l in memory["lessons"][-5:]:
                parts.append(f"  • {l['text'][:80]}")
        if memory["corrections"]:
            parts.append(f"**Corrections:** ({len(memory['corrections'])} stored)")
            for c in memory["corrections"][-3:]:
                parts.append(f"  • \"{c['wrong'][:40]}\" → \"{c['correct'][:40]}\"")
        if not parts:
            return "Memory is empty. Use save-memory, add-lesson, or add-correction to populate."
        return "\n".join(parts)

    return None

def _save_functions(registry):
    global _functions_cache
    _functions_cache = registry
    try:
        FUNCTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FUNCTIONS_FILE, 'w') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log(f"  Functions save error: {e}")
        _alert_owner(f"Functions save failed: {e}")

def _handle_function_command(cmd):
    """Handle function registry commands. Returns output string or None."""
    registry = _get_functions()

    # add-function <name> <description> <command>
    m = re.match(r'^add-function\s+(\S+)\s+"(.+?)"\s+"(.+?)"', cmd)
    if m:
        name, desc, command = m.group(1), m.group(2), m.group(3)
        # Remove existing with same name
        registry["functions"] = [f for f in registry["functions"] if f["name"] != name]
        registry["functions"].append({"name": name, "description": desc, "command": command})
        _save_functions(registry)
        return f"Function added: {name} — {desc}"

    # remove-function <name>
    m = re.match(r'^remove-function\s+(\S+)', cmd)
    if m:
        name = m.group(1)
        before = len(registry["functions"])
        registry["functions"] = [f for f in registry["functions"] if f["name"] != name]
        if len(registry["functions"]) < before:
            _save_functions(registry)
            return f"Function removed: {name}"
        return f"Function '{name}' not found."

    # list-functions
    if cmd.strip() == 'list-functions':
        if not registry["functions"]:
            return "No functions registered. Use add-function to add one."
        parts = [f"**Registered Functions ({len(registry['functions'])}):**"]
        for f in registry["functions"]:
            parts.append(f"  🔧 **{f['name']}** — {f['description']}")
            parts.append(f"     Command: `{f['command']}`")
        return "\n".join(parts)

    return None

def _handle_script_proposal(code, summary, why, channel):
    """Propose a new script for approval. Saves to pending, sends Discord embed."""
    # Sanitize filename
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', summary[:30].strip()).strip('_').lower()
    if not safe_name:
        safe_name = f"script_{int(time.time())}"

    # Validate Python syntax before saving
    try:
        compile(code, '<propose>', 'exec')
    except SyntaxError as e:
        return f"Script has syntax error: {e}. Fix the code and try again."

    PENDING_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    pending_path = PENDING_SCRIPTS_DIR / f"{safe_name}.py"
    pending_path.write_text(code, encoding='utf-8')

    # Store in memory for approval
    _pending_scripts[safe_name] = {
        "code": code,
        "summary": summary,
        "why": why,
        "channel": channel,
        "path": str(pending_path),
        "timestamp": datetime.now().isoformat(),
    }

    # Send approval request as embed with buttons
    embed = discord.Embed(
        title="📝 Nueva herramienta propuesta",
        color=discord.Color.blue(),
        description=(
            f"**Nombre:** `{safe_name}`\n"
            f"**Qué hace:** {summary}\n"
            f"**Por qué la necesito:** {why}"
        ),
    )
    embed.add_field(
        name="Código",
        value=f"```python\n{code[:1000]}{'...(truncated)' if len(code) > 1000 else ''}\n```",
        inline=False,
    )
    embed.set_footer(text="Usa los botones o escribe aprobar/rechazar <nombre>")
    view = _make_approval_view(safe_name)
    _send_approval_embed(channel, embed, view)
    log(f"  Script proposed for approval: {safe_name}")
    return f"Script '{safe_name}' proposed for approval. Waiting for owner confirmation."

def _handle_edit_file(reply, cmd, channel):
    """Propose a file edit for approval. Parses SEARCH/REPLACE or full content from reply."""
    filepath = cmd[len('edit-file '):].strip()
    if not filepath:
        return "Usage: edit-file <path> — then include the new content or SEARCH/REPLACE block after the EXEC line."

    # Extract content from reply: look for SEARCH/REPLACE block or code block after the EXEC line
    content_block = None

    # Pattern 1: SEARCH/REPLACE block
    sr_match = re.search(r'<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>>', reply, re.DOTALL)
    if sr_match:
        old_content = sr_match.group(1).strip()
        new_content = sr_match.group(2).strip()
        content_block = ("SEARCH/REPLACE", old_content, new_content)
    else:
        # Pattern 2: code block after the edit-file EXEC line
        # Find the text after the EXEC: edit-file line
        after_exec = reply.split('EXEC: edit-file ' + filepath.split('/')[-1], 1)
        if len(after_exec) == 1:
            after_exec = reply.split('EXEC: edit-file ' + filepath, 1)
        if len(after_exec) > 1:
            rest = after_exec[1]
            # Look for a code block
            cb = re.search(r'''`(?:\w*)\n(.*?)`''', rest, re.DOTALL)
            if cb:
                content_block = ("FULL", cb.group(1).strip(), None)
            else:
                # Use the rest as content
                text = rest.strip()
                if text:
                    content_block = ("FULL", text, None)

    if not content_block:
        return "Edit proposed but no content found. Include a code block or SEARCH/REPLACE block after the EXEC line."

    edit_type = content_block[0]

    # Read current file if it exists
    current_content = ""
    if os.path.isfile(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                current_content = f.read()
        except:
            pass

    edit_id = f"edit_{int(time.time())}_{len(_pending_edits)}"

    # Store for approval
    _pending_edits[edit_id] = {
        "filepath": filepath,
        "edit_type": edit_type,
        "old_content": content_block[1] if edit_type == "SEARCH/REPLACE" else None,
        "new_content": content_block[2] if edit_type == "SEARCH/REPLACE" else content_block[1],
        "current_content": current_content,
        "channel": channel,
        "timestamp": datetime.now().isoformat(),
    }
    _persist_pending_edit(edit_id, _pending_edits[edit_id])

    # Build description
    if edit_type == "SEARCH/REPLACE":
        desc_lines = [f"**Archivo:** {filepath}", f"**Tipo:** Search/Replace (quirúrgico)"]
    else:
        desc_lines = [f"**Archivo:** {filepath}", "**Tipo:** Reemplazo completo"]

    # Show diff preview (first 500 chars of old and new)
    old_preview = (content_block[1][:500] + '...') if len(content_block[1]) > 500 else content_block[1]
    diff_preview = content_block[2][:500] + '...' if (content_block[2] and len(content_block[2]) > 500) else (content_block[2] or '')

    embed = discord.Embed(
        title="✏️ Edición de archivo propuesta",
        color=discord.Color.gold(),
        description='\n'.join(desc_lines),
    )
    if edit_type == "SEARCH/REPLACE":
        embed.add_field(name="Texto original a reemplazar", value=f"`\n{old_preview}\n`", inline=False)
        embed.add_field(name="Texto nuevo", value=f"`\n{diff_preview}\n`", inline=False)
    else:
        if current_content:
            embed.add_field(name="Contenido actual (primeros 500 chars)", value=f"`\n{current_content[:500]}\n`", inline=False)
        else:
            embed.add_field(name="Estado", value="🆕 Archivo nuevo (no existe aún)", inline=False)
        embed.add_field(name="Contenido propuesto", value=f"`\n{diff_preview}\n`", inline=False)

    embed.set_footer(text="Usa los botones o escribe aprobar-edicion / rechazar-edicion")

    view = _make_edit_view(edit_id)
    _send_approval_embed(channel, embed, view)
    log(f"  Edit proposed for approval: {filepath} ({edit_id})")
    return f"Edit to '{filepath}' proposed for approval (ID: {edit_id}). Waiting for owner confirmation."


def _make_edit_view(edit_id: str) -> discord.ui.View:
    """Factory: creates a View with Approve/Reject buttons for file edits."""
    class _EditView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=None)

        async def on_error(self, interaction, error, item):
            log(f"  Discord: button '{getattr(item, 'custom_id', '?')}' failed: "
                f"{_exc_summary(error)}\n{traceback.format_exc()}")
            try:
                msg = "⚠️ Error interno al procesar el botón. Revisa los logs del bot."
                if interaction.response.is_done():
                    await interaction.followup.send(msg, ephemeral=True)
                else:
                    await interaction.response.send_message(msg, ephemeral=True)
            except Exception:
                pass

        @discord.ui.button(label="✅ Aprobar edición", style=discord.ButtonStyle.success,
                           custom_id=f"edit_approve_{edit_id}")
        async def approve_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
            if interaction.user.id != DISCORD_OWNER_ID:
                await interaction.response.send_message("Solo el owner puede aprobar.", ephemeral=True)
                return
            info = _pending_edits.get(edit_id)
            if info is None:
                await interaction.response.send_message("Esta edición ya fue procesada o expiró.", ephemeral=True)
                return

            filepath = info["filepath"]
            try:
                if info["edit_type"] == "SEARCH/REPLACE":
                    # Read current file and do search/replace
                    if os.path.isfile(filepath):
                        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                            curr = f.read()
                        old_text = info["old_content"]
                        new_text = info["new_content"]
                        if old_text not in curr:
                            # Keep the pending edit so the owner can retry
                            await interaction.response.send_message(
                                f"⚠️ No se encontró el texto original en {filepath}. "
                                "Puede que el archivo haya cambiado desde que se propuso la edición. "
                                "La edición sigue pendiente.",
                                ephemeral=True
                            )
                            return
                        new_file = curr.replace(old_text, new_text, 1)
                    else:
                        await interaction.response.send_message(f"⚠️ El archivo {filepath} ya no existe.", ephemeral=True)
                        return
                else:
                    # Full file replacement
                    new_file = info["new_content"]

                # Create parent directories if needed
                Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_file)
            except Exception as e:
                # Keep the edit pending so the owner can retry
                await interaction.response.send_message(f"❌ Error al aplicar edición: {e}", ephemeral=True)
                log(f"  Edit ERROR: {filepath} ({edit_id}): {e}")
                return

            # Only drop the pending state once the edit is actually on disk
            _pending_edits.pop(edit_id, None)
            _discard_pending_edit(edit_id)
            for child in self.children:
                child.disabled = True
            await interaction.response.edit_message(view=self)
            await interaction.followup.send(
                f"✅ **Edición aprobada:** {filepath} — archivo actualizado.",
                ephemeral=False
            )
            log(f"  Edit approved: {filepath} ({edit_id})")

        @discord.ui.button(label="❌ Rechazar edición", style=discord.ButtonStyle.danger,
                           custom_id=f"edit_reject_{edit_id}")
        async def reject_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
            if interaction.user.id != DISCORD_OWNER_ID:
                await interaction.response.send_message("Solo el owner puede rechazar.", ephemeral=True)
                return
            info = _pending_edits.pop(edit_id, None)
            if info is None:
                await interaction.response.send_message("Esta edición ya fue procesada o expiró.", ephemeral=True)
                return
            _discard_pending_edit(edit_id)
            for child in self.children:
                child.disabled = True
            await interaction.response.edit_message(view=self)
            await interaction.followup.send(
                f"❌ **Edición rechazada:** {info['filepath']} — no se aplicaron cambios.",
                ephemeral=False
            )
            log(f"  Edit rejected: {info['filepath']} ({edit_id})")

    return _EditView()


def _handle_approval(text_lower, channel):
    """Check if a message is an approve/reject for a pending script. Returns True if handled."""
    if not _pending_scripts:
        return False

    text = text_lower.strip()
    # Only match exact ✅/❌ (not mixed with other text) or explicit aprobar/rechazar
    if text == '✅' or text == 'aprobar':
        return _process_approval(None, approve=True, channel=channel)
    if text == '❌' or text == 'rechazar':
        return _process_approval(None, approve=False, channel=channel)
    # Check for edit approvals
    edit_approve_match = re.match(r'^aprobar-edicion\s+(\S+)', text)
    if edit_approve_match:
        return _process_edit_approval(edit_approve_match.group(1), approve=True, channel=channel)
    edit_reject_match = re.match(r'^rechazar-edicion\s+(\S+)', text)
    if edit_reject_match:
        return _process_edit_approval(edit_reject_match.group(1), approve=False, channel=channel)
    if text == 'aprobar-edicion':
        return _process_edit_approval(None, approve=True, channel=channel)
    if text == 'rechazar-edicion':
        return _process_edit_approval(None, approve=False, channel=channel)

    approve_match = re.match(r'^aprobar\s+(\S+)', text)
    if approve_match:
        return _process_approval(approve_match.group(1), approve=True, channel=channel)
    reject_match = re.match(r'^rechazar\s+(\S+)', text)
    if reject_match:
        return _process_approval(reject_match.group(1), approve=False, channel=channel)

    return False

def _process_approval(script_name, approve, channel):
    """Process script approval or rejection."""
    # If no name specified and only one pending, use it
    if not script_name and len(_pending_scripts) == 1:
        script_name = list(_pending_scripts.keys())[0]
    elif not script_name:
        discord_send_to_channel(channel, f"Multiple scripts pending: {', '.join(_pending_scripts.keys())}. Specify name: `aprobar <name>`")
        return True

    if script_name not in _pending_scripts:
        discord_send_to_channel(channel, f"No pending script named '{script_name}'.")
        return True

    info = _pending_scripts.get(script_name)
    if info is None:
        discord_send_to_channel(channel, f"Script '{script_name}' was already processed or expired.")
        return True

    if approve:
        # Move to approved scripts dir
        try:
            skills_dir = Path("/opt/netscan/skills")
            skills_dir.mkdir(parents=True, exist_ok=True)
            approved_path = skills_dir / f"{script_name}.py"
            approved_path.write_text(info["code"], encoding='utf-8')

            # Add to function registry
            registry = _get_functions()
            registry["functions"] = [f for f in registry["functions"] if f["name"] != script_name]
            registry["functions"].append({
                "name": script_name,
                "description": info["summary"],
                "command": f"python3 {approved_path}",
            })
            _save_functions(registry)
        except Exception as e:
            # Keep it pending so the owner can retry after fixing the cause
            discord_send_to_channel(channel, f"❌ No se pudo aprobar `{script_name}`: {e}. Sigue pendiente.")
            log(f"  Script approve ERROR: {script_name}: {e}")
            return True

        # Clean up pending
        pending_path = Path(info["path"])
        if pending_path.exists():
            pending_path.unlink()
        _pending_scripts.pop(script_name, None)

        discord_send_to_channel(channel, f"✅ **Script aprobado:** `{script_name}`\nGuardado en `{approved_path}` y añadido al registry de funciones.")
        log(f"  Script approved: {script_name}")
    else:
        # Delete pending
        pending_path = Path(info["path"])
        if pending_path.exists():
            pending_path.unlink()
        _pending_scripts.pop(script_name, None)
        discord_send_to_channel(channel, f"❌ **Script rechazado:** `{script_name}` — eliminado.")
        log(f"  Script rejected: {script_name}")

    return True


def _process_edit_approval(edit_id, approve, channel):
    """Process file edit approval or rejection via text commands."""
    if not edit_id and len(_pending_edits) == 1:
        edit_id = list(_pending_edits.keys())[0]
    elif not edit_id:
        parts = []
        for k in _pending_edits:
            parts.append(k)
        discord_send_to_channel(channel, f"Multiple edits pending: {', '.join(parts)}. Specify: `aprobar-edicion <id>`")
        return True

    info = _pending_edits.get(edit_id)
    if info is None:
        discord_send_to_channel(channel, f"No pending edit with ID '{edit_id}'.")
        return True

    filepath = info["filepath"]

    if approve:
        try:
            if info["edit_type"] == "SEARCH/REPLACE":
                if os.path.isfile(filepath):
                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        curr = f.read()
                    old_text = info["old_content"]
                    new_text = info["new_content"]
                    if old_text not in curr:
                        discord_send_to_channel(channel,
                            f"⚠️ No se encontró el texto original en `{filepath}`. Puede que el archivo haya cambiado. La edición sigue pendiente.")
                        return True
                    new_file = curr.replace(old_text, new_text, 1)
                else:
                    discord_send_to_channel(channel, f"⚠️ El archivo `{filepath}` ya no existe.")
                    return True
            else:
                new_file = info["new_content"]

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_file)

            discord_send_to_channel(channel, f"✅ **Edición aprobada:** `{filepath}` — archivo actualizado.")
            log(f"  Edit approved via text: {filepath} ({edit_id})")
        except Exception as e:
            # Keep the edit pending so the owner can retry
            discord_send_to_channel(channel, f"❌ Error al aplicar edición: {e}")
            return True
    else:
        discord_send_to_channel(channel, f"❌ **Edición rechazada:** `{filepath}` — no se aplicaron cambios.")
        log(f"  Edit rejected via text: {filepath} ({edit_id})")

    _pending_edits.pop(edit_id, None)
    _discard_pending_edit(edit_id)
    return True


def _build_memory_context():
    """Build memory context string for system prompt injection."""
    memory = _get_memory()
    registry = _get_functions()
    parts = []

    # Preferences
    if memory.get("preferences"):
        prefs = "; ".join(f"{k}={v}" for k, v in memory["preferences"].items())
        parts.append(f"Remembered preferences: {prefs}")

    # Profile (LLM-summarized compact memory — replaces raw lessons/corrections)
    if memory.get("profile"):
        parts.append(f"Profile: {memory['profile']}")
    else:
        # Fallback: show raw lessons/corrections if no profile yet
        if memory.get("lessons"):
            recent = memory["lessons"][-5:]
            lessons_str = "; ".join(l["text"][:60] for l in recent)
            parts.append(f"Lessons learned: {lessons_str}")
        if memory.get("corrections"):
            recent_corr = memory["corrections"][-3:]
            corr_str = "; ".join(f'"{c["wrong"][:30]}" → "{c["correct"][:30]}"' for c in recent_corr)
            parts.append(f"Corrections: {corr_str}")

    # Functions (abbreviated: names only, use tool-help for details)
    if registry.get("functions"):
        funcs_str = ", ".join(f["name"] for f in registry["functions"])
        parts.append(f"Available skills ({len(registry['functions'])}): {funcs_str}. Use EXEC: tool-help <name> for details.")

    if not parts:
        return ""
    return "\n" + "\n".join(parts) + "\n"


MEMORY_SUMMARIZE_THRESHOLD = 5  # Trigger summarization when lessons+corrections exceed this
MEMORY_SUMMARIZE_INTERVAL_S = 3600  # Don't summarize more than once per hour
_last_memory_summarize = 0


def _maybe_summarize_memory():
    """Periodically compress lessons + corrections into a concise profile using the LLM.
    Only runs when GPU is idle and threshold is exceeded. Silent — no Discord output."""
    global _last_memory_summarize
    now = time.time()
    # Rate limiting handled by main loop timer (last_memory_check)
    memory = _get_memory()
    lessons = memory.get("lessons", [])
    corrections = memory.get("corrections", [])
    if len(lessons) + len(corrections) < MEMORY_SUMMARIZE_THRESHOLD:
        return
    if not is_llm_healthy() or not is_gpu_idle():
        return
    log(f"  Memory summarization: {len(lessons)} lessons + {len(corrections)} corrections → compressing...")
    _last_memory_summarize = now

    # Build summarization prompt
    existing_profile = memory.get("profile", "")
    lessons_text = "\n".join(f"- {l['text']}" for l in lessons[-20:])  # Last 20 max
    corrections_text = "\n".join(f"- WRONG: {c['wrong']} → CORRECT: {c['correct']}" for c in corrections[-15:])
    prefs_text = "; ".join(f"{k}={v}" for k, v in memory.get("preferences", {}).items())

    prompt = (
        "You are a memory compressor for an AI assistant. Rewrite the profile below to incorporate "
        "the new lessons and corrections. Keep it under 200 words, direct, factual. "
        "Preserve ALL preferences, corrections, and lessons — don't lose information. "
        "Output ONLY the updated profile, nothing else.\n\n"
        f"CURRENT PROFILE:\n{existing_profile or '(empty)'}\n\n"
        f"PREFERENCES:\n{prefs_text or '(none)'}\n\n"
        f"NEW LESSONS:\n{lessons_text}\n\n"
        f"NEW CORRECTIONS:\n{corrections_text}"
    )

    try:
        payload = json.dumps({
            "model": SIGNAL_CHAT_MODEL, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "temperature": 0.3, "max_tokens": 400
        }).encode()
        req = urllib.request.Request(
            f"{LLM_URL}/chat/completions", data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        new_profile = data['choices'][0]['message']['content'].strip()
        if new_profile and len(new_profile) > 20:
            memory["profile"] = new_profile
            # Clear lessons/corrections that have been absorbed into the profile
            memory["lessons"] = memory["lessons"][-3:]  # keep last 3 as safety net
            memory["corrections"] = memory["corrections"][-2:]  # keep last 2 as safety net
            _save_memory(memory)
            log(f"  Memory summarized: profile = {len(new_profile)} chars, lessons/corrections cleared")
        else:
            log("  Memory summarization: LLM returned empty/short response, skipping")
    except Exception as e:
        log(f"  Memory summarization error: {e}")


def _handle_tool_help(cmd):
    """Return full usage details for a specific tool. Only loaded on demand (lazy loading)."""
    m = re.match(r'^tool-help\s+(\S+)', cmd)
    if not m:
        return "Usage: tool-help <tool_name> — shows full syntax and examples for a tool."
    name = m.group(1)

    # Built-in tools with their full documentation
    TOOL_DOCS = {
        "save-memory": "EXEC: save-memory <key> <value> — Save a preference, fact, or user instruction. Example: save-memory timezone UTC+1",
        "forget-memory": "EXEC: forget-memory <key> — Remove a saved preference by key. Example: forget-memory timezone",
        "list-memory": "EXEC: list-memory — Show all saved preferences, recent lessons and corrections.",
        "add-lesson": "EXEC: add-lesson <text> — Record something learned from the conversation. Example: add-lesson AK prefers concise responses",
        "add-correction": "EXEC: add-correction <wrong> -> <correct> — Fix a mistake. Example: add-correction temperature is 30C -> temperature is 45C",
        "add-function": 'EXEC: add-function <name> "description" "command" — Register a reusable skill. Example: add-function check-disk "Check disk usage" "df -h"',
        "remove-function": "EXEC: remove-function <name> — Unregister a skill. Example: remove-function check-disk",
        "list-functions": "EXEC: list-functions — Show all registered skills with their commands.",
        "propose-script": 'EXEC: propose-script "summary" "why needed" <python_code> — Propose a new script for approval. The owner approves/rejects via Discord buttons. Use when you notice repeated patterns (3+ times).',
        "tool-help": "EXEC: tool-help <tool_name> — Show full syntax and examples for any tool. Example: tool-help save-memory",
        "generate-image": "EXEC: generate-image <prompt> — Generate an image (Qwen-Image 2.1, HiDream-O1 or FLUX). Example: generate-image a red lobster on a beach. For a TRANSPARENT PNG, use: 'This is an RGBA image with transparency. <desc>. The image has alpha channel and the background is transparent.'",
        "edit-image": "EXEC: edit-image <instruction> — Edit the attached reference image (Qwen-Image 2.1, HiDream-O1 or Kontext). Requires an image attachment. Be explicit about what to preserve: 'keep the face and identity unchanged, change the background to a Tokyo skyline'. Example: edit-image make the sky purple",
        "upscale-image": "EXEC: upscale-image — Upscale the attached image 4x with ESRGAN. Requires an image attachment.",
        "generate-video": "EXEC: generate-video <prompt> — Generate a video with WAN 2.2 (~30-40 min). Attach an image to animate it (image-to-video). Example: generate-video waves crashing on shore",
        "ddgr": "EXEC: ddgr <query> — Search the web using DuckDuckGo. Example: ddgr linux kernel 6.12 release date",
        "curl": "EXEC: curl <url> — Fetch a URL to stdout. Writing/uploading files (-o, -O, -T, --output) is blocked. Example: curl https://api.github.com/zen",
        "ls": "EXEC: ls [path] — List directory contents. Restricted to /opt/netscan and /tmp. Example: ls /opt/netscan/data",
        "cat": "EXEC: cat <file> — Show file contents. Restricted to /opt/netscan and /tmp. Example: cat /opt/netscan/data/clawd-memory.json",
        "python3": "EXEC: python3 /opt/netscan/<name>.py — Run a Python script (must be under /opt/netscan/). Example: python3 /opt/netscan/scripts/check_temp.py",
        "bash": "EXEC: bash /opt/netscan/<name>.sh — Run a bash script (must be under /opt/netscan/). Example: bash /opt/netscan/scripts/status.sh",
    }

    if name in TOOL_DOCS:
        return TOOL_DOCS[name]

    # Check registered functions
    registry = _get_functions()
    for func in registry.get("functions", []):
        if func["name"] == name:
            return f"EXEC: {func['command']} — {func['description']}"

    return f"Unknown tool '{name}'. Use list-functions to see registered skills, or try: save-memory, add-lesson, generate-image, ddgr, curl, ls, cat."


def _remove_tmp_files(*path_lists):
    """Best-effort cleanup of downloaded Discord uploads (images/audio/PDFs)."""
    for paths in path_lists:
        for p in paths or []:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def process_discord_chat(tag=None):
    if not _discord_bot: return 0
    # In smart/sleep mode, skip all assistant processing (saves RAM)
    if _current_mode in ('smart', 'sleep'):
        # Drain queue silently
        while True:
            try: _discord_bot.msg_queue.get_nowait()
            except queue.Empty: break
        return 0
    # Media generation stops llama-swap, which would abort a running cron job.
    # Defer it while a job is active (mirrors the "GPU busy" deferral path).
    sd_ok = (tag is None) and not _job_active
    if sd_ok and _deferred_sd_queue:
        while _deferred_sd_queue:
            action, args = _deferred_sd_queue.pop(0)
            if action == 'generate': generate_and_send_image(args[2], args[0], args[1])
            sd_watchdog_ping()
    count = 0
    while True:
        try: msg = _discord_bot.msg_queue.get_nowait()
        except queue.Empty: break
        count += 1
        # NB: not msg.get('thread_channel', msg['reply_to'].channel) — that default
        # is evaluated eagerly and crashes when reply_to has no .channel.
        thread_channel = msg.get('thread_channel') or msg['reply_to'].channel
        # Slash commands hand heavy work to this worker; run it here so the
        # Discord event loop is never blocked by image generation or research.
        if msg.get('direct'):
            tool, arg = msg['direct']
            with DiscordTyping(thread_channel):
                if tool == 'generate-image':
                    if sd_ok:
                        generate_and_send_image(arg, msg['reply_to'], thread_channel)
                    else:
                        _deferred_sd_queue.append(('generate', (msg['reply_to'], thread_channel, arg)))
                        discord_send_to_channel(thread_channel, "🦞 GPU ocupada — imagen en cola.")
                elif tool == 'research':
                    discord_send_to_channel(thread_channel, "🔬 Investigando…")
                    run_research_from_discord(arg, thread_channel)
            continue
        att_path = msg['attachments'][0] if msg['attachments'] else None
        audio_paths = msg.get('audio_paths', [])
        pdf_paths = msg.get('pdf_paths', [])
        with DiscordTyping(thread_channel):
            # Handle PDFs: text extraction → OCR fallback
            if pdf_paths:
                pdf_result = process_pdf(pdf_paths[0], msg['text'])
                _remove_tmp_files(pdf_paths)  # Cleanup downloaded PDF
                if pdf_result:
                    reply = chat_with_llm(pdf_result, allow_sd=sd_ok, thread_id=msg['thread_id'], msg_obj=msg['reply_to'], is_thread=msg.get('is_thread', True), thread_channel=thread_channel)
                    discord_send_to_channel(thread_channel, f"📄 {reply}")
                    continue
                else:
                    discord_send_to_channel(thread_channel, "📄 Could not process PDF. Is poppler-utils installed?")
                    continue
            # Handle voice messages: transcribe with whisper, then chat with LLM
            if audio_paths:
                audio_text = transcribe_audio(audio_paths[0], msg['text'])
                _remove_tmp_files(audio_paths)  # Cleanup original upload (wav already removed by transcribe_audio)
                if audio_text:
                    reply = chat_with_llm(audio_text, allow_sd=sd_ok, thread_id=msg['thread_id'], msg_obj=msg['reply_to'], is_thread=msg.get('is_thread', True), thread_channel=thread_channel)
                    discord_send_to_channel(thread_channel, reply)
                    continue
                else:
                    discord_send_to_channel(thread_channel, "🎤 Could not transcribe audio. Is whisper.cpp installed?")
                    continue
            if att_path and msg['text'] and not any(kw in msg['text'].lower() for kw in ("edit", "change", "make", "draw", "video")):
                res = analyze_image_with_vision(att_path, msg['text'])
                if res:
                    discord_send_to_channel(thread_channel, f"👁️ {res}")
                    _remove_tmp_files([att_path])
                    continue
            # Research mode: autonomous multi-step research
            if is_research_intent(msg['text']):
                run_research_from_discord(msg['text'], thread_channel)
                _remove_tmp_files([att_path])
                continue
            # Main chat path: stream the answer into a live-updated message.
            streamer = _new_streamer(thread_channel)
            reply = chat_with_llm(msg['text'], allow_sd=sd_ok, attachment_path=att_path, thread_id=msg['thread_id'], msg_obj=msg['reply_to'], is_thread=msg.get('is_thread', True), thread_channel=thread_channel, streamer=streamer)
            # When streaming already delivered the answer into its placeholder,
            # don't send it a second time; otherwise do the usual send.
            if streamer is None or not streamer.delivered:
                discord_send_to_channel(thread_channel, reply)
            _remove_tmp_files([att_path])
    return count


# ─── Chat worker (decoupled from the cron scheduler) ────────────────────────
def chat_worker_loop():
    """Consume Discord messages in a dedicated thread.

    The scheduler used to call process_discord_chat() once per cycle, *after*
    running every job: a 2h career-scan left the bot mute for the whole job, and
    conversely a slow LLM turn stalled the scheduler. Both directions are now
    independent — the bot answers while jobs run, and jobs keep progressing while
    the bot thinks.

    Chat history and pending-approval state are only touched from here (plus the
    Discord event thread for button callbacks), keeping them effectively
    single-writer.
    """
    log("  Chat worker: started")
    last_cleanup = 0.0
    last_expire = 0.0
    while running:
        try:
            process_discord_chat()
        except Exception as e:
            log(f"  Discord chat error: {e}")
            _alert_owner(f"Discord chat loop error: {e}")
        now = time.time()
        # Periodic chat history cleanup (LRU eviction above the MB cap)
        if now - last_cleanup >= CHAT_CLEANUP_INTERVAL_S:
            cleanup_chat_history()
            last_cleanup = now
        # Auto-expire pending scripts/edits older than 24h
        if now - last_expire >= 300:
            _cleanup_expired_scripts()
            last_expire = now
        # Short poll: a new message is picked up within ~1s instead of waiting
        # for the scheduler's 10s cycle (and for whatever job was running).
        time.sleep(1)
    log("  Chat worker: stopped")


def start_chat_worker():
    """Start the chat consumer thread (no-op when Discord is not configured)."""
    if not DISCORD_TOKEN:
        log("  Chat worker: DISCORD_TOKEN not set, chat disabled")
        return
    threading.Thread(target=chat_worker_loop, daemon=True, name="chat-worker").start()


from croniter import croniter

# ─── Research Agent ─────────────────────────────────────────────────────────
RESEARCH_KEYWORDS = (
    'investiga', 'investigar', 'research', 'deep research',
    'busca en internet', 'busca sobre', 'search for', 'search about',
    'analiza esto', 'analiza cómo', 'analyze how',
    'clona y prueba', 'clone and test', 'compila y prueba',
    'prueba en mi sistema', 'test on my system',
)


def is_research_intent(text):
    """Detect if a message is a research/deep-dive request."""
    lower = text.lower()
    return any(kw in lower for kw in RESEARCH_KEYWORDS)


def run_research_from_discord(text, thread_channel):
    """Run the research agent and send results to Discord thread."""
    try:
        from research_agent import run_research
        discord_send_to_channel(thread_channel, "🔬 **Starting deep research...** This may take a few minutes.")
        answer = run_research(
            query=text,
            llm_url=LLM_URL,
            model=SIGNAL_CHAT_MODEL,
            discord_send_fn=discord_send_to_channel,
            thread_channel=thread_channel,
        )
        # Discord has 2000 char limit — split if needed
        if len(answer) <= SIGNAL_MAX_REPLY:
            discord_send_to_channel(thread_channel, f"🔬 **Research complete:**\n\n{answer}")
        else:
            # Send in chunks
            for i in range(0, len(answer), SIGNAL_MAX_REPLY):
                chunk = answer[i:i + SIGNAL_MAX_REPLY]
                discord_send_to_channel(thread_channel, chunk)
    except ImportError:
        discord_send_to_channel(thread_channel, "❌ research_agent.py not found on server.")
    except Exception as e:
        log(f"Research agent error: {e}")
        discord_send_to_channel(thread_channel, f"❌ Research failed: {e}")

# ─── Job Execution Logic ────────────────────────────────────────────────────
def should_run_now(job, last_run_ts):
    schedule = job.get('schedule')
    if not schedule: return True
    now = datetime.now()
    last_run_dt = datetime.fromtimestamp(last_run_ts) if last_run_ts else now - timedelta(days=365)
    try:
        iter = croniter(schedule, last_run_dt)
        return now >= iter.get_next(datetime)
    except: return True

def save_state(state):
    try:
        with open(STATE_FILE, 'w') as f: json.dump(state, f)
    except: pass

# Last report digest per job, for report.dedupe. In memory on purpose: after a
# restart a persisting condition is announced once more, which is the safe side to
# err on.
_REPORT_DIGESTS = {}

# Noise that changes on every single run: log stamps ([00:10:16]) and dated
# filenames (watchdog-20261018-0020.json). Left in, no two reports would ever
# compare equal and dedupe would silently do nothing at all.
_REPORT_NOISE = re.compile(
    r"\[[^\]]{0,60}\]"          # [00:10:16]
    r"|\d{8}[-_]\d{4}"          # watchdog-20261018-0020
    r"|\d{4}-\d{2}-\d{2}"       # 2026-10-18
    r"|\b\d{8}\b"               # 20261018
)


def _report_digest(msg):
    """Digest of a report with per-run noise removed, for report.dedupe."""
    norm = re.sub(r"\s+", " ", _REPORT_NOISE.sub(" ", msg)).strip()
    return hashlib.sha1(norm.encode("utf-8", "replace")).hexdigest()


def _job_output_tail(res, lines=15, limit=1800):
    """Last lines of a job's stdout+stderr, for a report or an error notice."""
    if res is None:
        return ""
    blob = ((res.stdout or "") + (res.stderr or "")).strip()
    if not blob:
        return ""
    return "\n".join(blob.splitlines()[-max(1, lines):])[-limit:]


def _maybe_send_report(name, job, res=None):
    """After a job succeeds, DM its report to the owner IF the job declares one
    via jobs.json:

        "report": {"file": "/path/report.txt", "title": "..."}   # a file it wrote
        "report": {"output": true, "title": "...", "tail": 20}   # its own output

    Opt-in per job, so only 'important' report-producing jobs notify Discord
    (hardware/health scanners simply omit the field).

    The `output` form exists because the bot already captures every job's stdout
    and stderr and then threw it away, which is why a failure notice could only say
    "exit 127, check the log" while the reason sat in the captured output."""
    report = job.get("report") or {}
    title = (report.get("title") or f"📋 {name}").strip()
    parts = []

    if report.get("output") or report.get("tail"):
        out = _job_output_tail(res, int(report.get("tail") or 15))
        if out:
            parts.append(out)

    report_file = (report.get("file") or "").strip()
    if report_file:
        # Support a glob so jobs can point at dated files (latest-*.md etc.)
        paths = sorted(glob.glob(report_file)) if glob.has_magic(report_file) else []
        if not paths and os.path.isfile(report_file):
            paths = [report_file]
        if not paths:
            log(f"  ⚠️ Report file not found for {name}: {report_file}")
        for p in paths:
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    txt = f.read().strip()
                if txt:
                    parts.append(txt)
            except Exception as e:
                log(f"  ⚠️ Could not read report for {name}: {p} ({e})")

    if not parts:
        return
    msg = f"{title}\n\n" + "\n\n".join(parts)

    # Optional gate: send only when the message matches. This is what makes a
    # report safe on a job that runs every 5 or 10 minutes — a health scanner can
    # then report only when it has something to report, instead of staying silent
    # (alerts lost) or flooding (144 messages a day).
    only_if = (report.get("only_if") or "").strip()
    if only_if:
        try:
            if not re.search(only_if, msg, re.IGNORECASE):
                log(f"  📄 Report for {name} did not match only_if — not sent")
                return
        except re.error as e:
            log(f"  ⚠️ Bad only_if regex for {name}: {e}")

    # Optional dedupe: skip a report identical to the last one sent for this job,
    # so a persisting condition (3 criticals that are still 3 criticals) is
    # announced once rather than every cycle. Compared on the de-noised text, so
    # a new timestamp does not count as news.
    digest = None
    if report.get("dedupe"):
        digest = _report_digest(msg)
        if _REPORT_DIGESTS.get(name) == digest:
            log(f"  📄 Report for {name} unchanged since last time — not repeated")
            return

    if _dm_owner(msg):
        if digest is not None:
            _REPORT_DIGESTS[name] = digest
        log(f"  📨 Report sent to owner: {name} ({len(msg)} chars)")
    else:
        log(f"  ⚠️ Could not DM report for {name} (Discord unavailable?)")


def run_job(job, state):
    global _job_active
    name = job['name']
    job_id = job.get('id', name)
    last_runs = state.setdefault('last_runs', {})
    last_run_ts = last_runs.get(job_id, 0)
    if not should_run_now(job, last_run_ts): return True, 0.0
    is_scrape = name.endswith('-scrape') or name.startswith('netscan-')
    # In smart mode, skip LLM-dependent jobs (save VRAM for the smart model)
    if _current_mode == 'smart' and not is_scrape:
        log(f"  SKIPPED: {name} (smart mode active)")
        return True, 0.0
    # In sleep mode, wake llama-swap for LLM-dependent jobs, then sleep again
    _sleep_woken = False
    if _current_mode == 'sleep' and not is_scrape:
        log(f"  SLEEP WAKE: starting llama-swap for {name}")
        subprocess.run(["rc-service", "llama-swap", "start"], capture_output=True)
        if not wait_for_llm(max_wait_s=120):
            log(f"  SLEEP WAKE FAILED: llama-swap not ready for {name}")
            return False, 0.0
        _sleep_woken = True
    if not is_scrape and is_useful_time():
        log(f"  PAUSED: {name} (Useful hours)")
        return True, 0.0
    if not is_scrape and not is_llm_healthy():
        if not wait_for_llm(): return False, 0.0
    log(f"🚀 Running: {name}")
    sd_status(f'Running: {name}')
    start = time.time()
    _job_active = True
    # Usar sys.executable asegura que los scripts hijos usen el mismo VENV que el bot
    cmd = [sys.executable, f"/opt/netscan/{name}.py"] if not name.endswith('.sh') else ["bash", f"/opt/netscan/{name}"]
    # Honor the per-job timeout from jobs.json (payload.timeoutSeconds);
    # fall back to the global default when not set (tolerate a bad value).
    try:
        timeout_s = int(job.get("payload", {}).get("timeoutSeconds") or DEFAULT_TIMEOUT_S)
    except (ValueError, TypeError):
        timeout_s = DEFAULT_TIMEOUT_S
    try:
        res = subprocess.run(cmd, timeout=timeout_s, capture_output=True, text=True)
        elapsed = time.time() - start
        log(f"  Done: {name} ({elapsed:.0f}s)")
        
        last_runs[job_id] = time.time()
        save_state(state)
        if res.returncode == 0:
            # Success — deliver the job's report to the owner if the job declares
            # one (jobs.json 'report' config; opt-in per job so only 'important'
            # report-producing jobs DM, not the hardware scanners).
            _maybe_send_report(name, job, res)
        else:
            # NOTIFICACIÓN DE FALLO POR DISCORD — carrying the job's own last lines.
            # "exit 127, check the log" is not actionable; the reason was in the
            # captured output all along and simply never made it into the message.
            tail = _job_output_tail(res, 20, 1500)
            error_msg = (f"❌ **ERROR EN JOB:** `{name}`\n"
                         f"El script ha terminado con código de error {res.returncode}.")
            if tail:
                error_msg += f"\n```\n{tail}\n```"
            else:
                error_msg += ("\n(sin salida; revisa "
                              "`/opt/netscan/data/queue-runner.log`)")
            _dm_owner(error_msg)
        # If we woke llama-swap for this job in sleep mode, stop it again
        if _sleep_woken:
            log(f"  SLEEP: stopping llama-swap after {name}")
            subprocess.run(["rc-service", "llama-swap", "stop"], capture_output=True)
        return res.returncode == 0, elapsed
    except Exception as e:
        log(f"  Error: {name} - {e}")
        # If we woke llama-swap, stop it again even on error
        if _sleep_woken:
            subprocess.run(["rc-service", "llama-swap", "stop"], capture_output=True)
        # NOTIFICACIÓN DE EXCEPCIÓN POR DISCORD
        _dm_owner(f"⚠️ **EXCEPCIÓN CRÍTICA:** `{name}`\n{str(e)[:500]}")
        return False, time.time() - start
    finally:
        _job_active = False

def generate_and_send_image(prompt, msg_obj=None, thread_channel=None, model=None):
    prompt, want = _split_image_model(prompt, model)
    if thread_channel:
        discord_send_to_channel(thread_channel, "🎨 Generating image...")
    elif msg_obj:
        discord_reply(msg_obj, "🎨 Generating image...")
    # Stop llama-swap to free VRAM
    log("  Stopping llama-swap service...")
    subprocess.run(["rc-service", "llama-swap", "stop"])
    time.sleep(3)

    # ── Z-Image Turbo (fast distilled DiT: 8 steps, cfg 1.0) ──
    z_model = _find_model(SD_ZIMAGE_DIR, SD_ZIMAGE_MODEL,
        "z_image_turbo-Q4_K", "z_image_turbo-Q5_0", "z_image_turbo-Q6_K",
        "z_image_turbo-Q8_0", "z_image_turbo-Q4_0", "z_image_turbo-Q3_K",
        "z_image_turbo")
    # Z-Image reuses the FLUX VAE (ae.safetensors): look in z_image/ first, then flux/.
    z_vae = _find_model(SD_ZIMAGE_DIR, SD_ZIMAGE_VAE, "ae", "z_image_vae") \
        or _find_model(SD_FLUX_DIR, SD_FLUX_VAE, "ae")
    z_llm = _find_model(SD_ZIMAGE_DIR, SD_ZIMAGE_LLM,
        "Qwen3-4B-Instruct-2507-Q4_K_M", "Qwen3-4B-Instruct-2507-Q4_K_S",
        "Qwen3-4B-Instruct-2507-Q5_K_M", "Qwen3-4B-Instruct-2507-Q6_K",
        "Qwen3-4B-Instruct-2507")

    # ── Qwen-Image 2.1 (SOTA DiT, unified text-to-image) ──
    qwen_diff = _find_model(SD_QWEN_DIR, SD_QWEN_MODEL,
        "qwen_image_2.1-Q4_K", "qwen_image_2.1-Q4_0", "qwen_image_2.1-Q5_0",
        "qwen_image_2.1-Q8_0", "qwen_image_2.1_int8_convrot", "qwen_image_2.1")
    qwen_vae = _find_model(SD_QWEN_DIR, SD_QWEN_VAE,
        "qwen_image_2.1_vae_bf16", "qwen_image_2.1_vae", "qwen_image_vae")
    qwen_llm = _find_model(SD_QWEN_DIR, SD_QWEN_LLM,
        "Qwen3VL-8B-Instruct-Q4_K_M", "Qwen3VL-8B-Instruct-Q4_0",
        "Qwen3VL-8B-Instruct", "qwen3vl-8b", "qwen3-vl")

    # ── HiDream-O1 — unified pixel transformer, no external encoders ──
    hidream_model = _find_model(SD_HIDREAM_DIR, SD_HIDREAM_MODEL,
        "HiDream-O1-Image-Dev-BF16", "HiDream-O1-Image-Dev-FP8",
        "HiDream-O1-Image-Full-FP8", "HiDream-O1-Image-Dev",
        "HiDream-O1-Image-Full", "hidream-o1-dev", "hidream-o1")

    z_ok = bool(z_model and z_vae and z_llm)
    qwen_ok = bool(qwen_diff and qwen_vae and qwen_llm)

    # An explicit request with no files on disk is better reported than silently
    # swapped for another model.
    if want == "zimage" and not z_ok:
        subprocess.run(["rc-service", "llama-swap", "start"])
        return "🦞 Z-Image no disponible (faltan modelos en z_image/)."
    if want == "qwen" and not qwen_ok:
        subprocess.run(["rc-service", "llama-swap", "start"])
        return "🦞 Qwen-Image no disponible (faltan modelos en qwen_image/)."
    if want == "hidream" and not hidream_model:
        subprocess.run(["rc-service", "llama-swap", "start"])
        return "🦞 HiDream-O1 no disponible."

    allow_zimage = want in ("auto", "zimage")
    allow_qwen = want in ("auto", "qwen")
    allow_hidream = want in ("auto", "hidream")

    if allow_zimage and z_ok:
        log(f"  Using Z-Image Turbo: {os.path.basename(z_model)} "
            f"(cfg {SD_ZIMAGE_CFG_SCALE}, {SD_ZIMAGE_STEPS} steps, {SD_ZIMAGE_SIZE}px)")
        _warn_sd_vram("Z-Image-Turbo", _sd_estimate_mb(z_model, z_vae, z_llm), headroom_mb=768)
        log("  Generating image...")
        if os.path.exists(SD_OUTPUT_PATH):
            os.remove(SD_OUTPUT_PATH)
        start = time.time()
        proc = subprocess.Popen([
            SD_CLI,
            "--diffusion-model", z_model,
            "--vae", z_vae,
            "--llm", z_llm,
            "-p", prompt, "-o", SD_OUTPUT_PATH,
            *_zimage_args()
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        waited = 0
        while waited < SD_TIMEOUT_S:
            if os.path.exists(SD_OUTPUT_PATH) and os.path.getsize(SD_OUTPUT_PATH) > 1000:
                elapsed = time.time() - start
                log(f"  Image ready after {elapsed:.0f}s, killing sd-cli...")
                time.sleep(2)
                proc.kill()
                proc.wait()
                subprocess.run(["killall", "-9", "sd-cli"], capture_output=True)
                break
            time.sleep(3)
            waited += 3
        else:
            proc.kill()
            proc.wait()
            subprocess.run(["killall", "-9", "sd-cli"], capture_output=True)
            log("  ERROR: Z-Image image generation timed out")
            subprocess.run(["rc-service", "llama-swap", "start"])
            return "🦞 Image generation timed out."
        elapsed = time.time() - start
        log(f"  Image: {SD_OUTPUT_PATH} ({os.path.getsize(SD_OUTPUT_PATH)} bytes) in {elapsed:.0f}s")
        # No ESRGAN pass on purpose: Z-Image already renders at SD_ZIMAGE_SIZE
        # (1024 by default), so a 4x upscale would be 4096px and blow
        # SD_UPSCALE_TIMEOUT_S.
        label = f"🧠 Z-Image Turbo {SD_ZIMAGE_SIZE}px"
        log("  Restarting llama-swap service...")
        subprocess.run(["rc-service", "llama-swap", "start"])
        wait_for_llm()
        if thread_channel:
            discord_send_to_channel(thread_channel, f"🎨 {prompt}")
            discord_send_file_to_channel(thread_channel, label, SD_OUTPUT_PATH)
        elif msg_obj:
            discord_reply(msg_obj, f"🎨 {prompt}")
            discord_send_attachment(msg_obj, label, SD_OUTPUT_PATH)
        try:
            if os.path.exists(SD_OUTPUT_PATH): os.remove(SD_OUTPUT_PATH)
        except Exception: pass
        return "Image sent!"

    use_qwen = bool(qwen_ok and allow_qwen and (SD_QWEN_MODEL or want == "qwen" or not hidream_model))
    if use_qwen:
        log(f"  Using Qwen-Image 2.1: {os.path.basename(qwen_diff)}")
        _warn_sd_vram("Qwen-Image-2.1", _sd_estimate_mb(qwen_diff, qwen_vae, qwen_llm))
        log("  Generating image...")
        if os.path.exists(SD_OUTPUT_PATH):
            os.remove(SD_OUTPUT_PATH)
        start = time.time()
        proc = subprocess.Popen([
            SD_CLI,
            "--diffusion-model", qwen_diff,
            "--vae", qwen_vae,
            "--llm", qwen_llm,
            "-p", prompt, "-o", SD_OUTPUT_PATH,
            *_qwen_args()
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        waited = 0
        while waited < SD_TIMEOUT_S:
            if os.path.exists(SD_OUTPUT_PATH) and os.path.getsize(SD_OUTPUT_PATH) > 1000:
                elapsed = time.time() - start
                log(f"  Image ready after {elapsed:.0f}s, killing sd-cli...")
                time.sleep(2)
                proc.kill()
                proc.wait()
                subprocess.run(["killall", "-9", "sd-cli"], capture_output=True)
                break
            time.sleep(3)
            waited += 3
        else:
            proc.kill()
            proc.wait()
            subprocess.run(["killall", "-9", "sd-cli"], capture_output=True)
            log("  ERROR: Qwen-Image 2.1 image generation timed out")
            subprocess.run(["rc-service", "llama-swap", "start"])
            return "🦞 Image generation timed out."
        elapsed = time.time() - start
        log(f"  Image: {SD_OUTPUT_PATH} ({os.path.getsize(SD_OUTPUT_PATH)} bytes) in {elapsed:.0f}s")
        upscaled_path = "/tmp/sd-output-upscaled.png"
        if os.path.exists(SD_ESRGAN_MODEL):
            log("  Upscaling 4x with ESRGAN...")
            upscale_start = time.time()
            up_proc = subprocess.run([
                SD_CLI, "--upscale-model", SD_ESRGAN_MODEL,
                "--upscale-repeats", "1",
                "-i", SD_OUTPUT_PATH, "-o", upscaled_path
            ], capture_output=True, text=True, timeout=SD_UPSCALE_TIMEOUT_S)
            if up_proc.returncode == 0 and os.path.exists(upscaled_path):
                log(f"  Upscaled in {time.time() - upscale_start:.0f}s")
            else:
                log(f"  Upscale failed: {up_proc.stderr[:200]}")
                upscaled_path = None
        else:
            upscaled_path = None
        log("  Restarting llama-swap service...")
        subprocess.run(["rc-service", "llama-swap", "start"])
        wait_for_llm()
        if thread_channel:
            discord_send_to_channel(thread_channel, f"🎨 {prompt}")
            discord_send_file_to_channel(thread_channel, "🎨 Original 512x512", SD_OUTPUT_PATH)
            if upscaled_path:
                discord_send_file_to_channel(thread_channel, "🔍 Upscaled 4x", upscaled_path)
        elif msg_obj:
            discord_reply(msg_obj, f"🎨 {prompt}")
            discord_send_attachment(msg_obj, "🎨 Original 512x512", SD_OUTPUT_PATH)
            if upscaled_path:
                discord_send_attachment(msg_obj, "🔍 Upscaled 4x", upscaled_path)
        try:
            if os.path.exists(SD_OUTPUT_PATH): os.remove(SD_OUTPUT_PATH)
            if upscaled_path and os.path.exists(upscaled_path): os.remove(upscaled_path)
        except Exception: pass
        return "Image sent!"

    elif allow_hidream and hidream_model:
        log(f"  Using HiDream-O1: {os.path.basename(hidream_model)} (unified, ~20-40s)")
        # tensor-type-rules quantizes part of the model to Q8_0 → ~68% of file size
        _warn_sd_vram("HiDream-O1", _sd_estimate_mb(hidream_model) * 0.68, headroom_mb=768)
        log("  Generating image...")
        if os.path.exists(SD_OUTPUT_PATH):
            os.remove(SD_OUTPUT_PATH)
        start = time.time()
        proc = subprocess.Popen([
            SD_CLI, "-m", hidream_model,
            "-p", prompt, "-o", SD_OUTPUT_PATH,
            "--steps", "12", "-W", "256", "-H", "256", "--cfg-scale", "1.0",
            "--tensor-type-rules", HIDREAM_TENSOR_RULES
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        waited = 0
        while waited < SD_TIMEOUT_S:
            if os.path.exists(SD_OUTPUT_PATH) and os.path.getsize(SD_OUTPUT_PATH) > 1000:
                elapsed = time.time() - start
                log(f"  Image ready after {elapsed:.0f}s, killing sd-cli...")
                time.sleep(2)
                proc.kill()
                proc.wait()
                subprocess.run(["killall", "-9", "sd-cli"], capture_output=True)
                break
            time.sleep(3)
            waited += 3
        else:
            proc.kill()
            proc.wait()
            subprocess.run(["killall", "-9", "sd-cli"], capture_output=True)
            log("  ERROR: HiDream image generation timed out")
            subprocess.run(["rc-service", "llama-swap", "start"])
            return "\U0001f99e Image generation timed out."
        elapsed = time.time() - start
        log(f"  Image: {SD_OUTPUT_PATH} ({os.path.getsize(SD_OUTPUT_PATH)} bytes) in {elapsed:.0f}s")
        upscaled_path = "/tmp/sd-output-upscaled.png"
        if os.path.exists(SD_ESRGAN_MODEL):
            log("  Upscaling 4x with ESRGAN...")
            upscale_start = time.time()
            up_proc = subprocess.run([
                SD_CLI, "--upscale-model", SD_ESRGAN_MODEL,
                "--upscale-repeats", "1",
                "-i", SD_OUTPUT_PATH, "-o", upscaled_path
            ], capture_output=True, text=True, timeout=SD_UPSCALE_TIMEOUT_S)
            if up_proc.returncode == 0 and os.path.exists(upscaled_path):
                log(f"  Upscaled in {time.time() - upscale_start:.0f}s")
            else:
                log(f"  Upscale failed: {up_proc.stderr[:200]}")
                upscaled_path = None
        else:
            upscaled_path = None
        log("  Restarting llama-swap service...")
        subprocess.run(["rc-service", "llama-swap", "start"])
        wait_for_llm()
        if thread_channel:
            discord_send_to_channel(thread_channel, f"\U0001f3a8 {prompt}")
            discord_send_file_to_channel(thread_channel, "\U0001f3a8 Original 512x512", SD_OUTPUT_PATH)
            if upscaled_path:
                discord_send_file_to_channel(thread_channel, "\U0001f50d Upscaled 4x", upscaled_path)
        elif msg_obj:
            discord_reply(msg_obj, f"\U0001f3a8 {prompt}")
            discord_send_attachment(msg_obj, "\U0001f3a8 Original 512x512", SD_OUTPUT_PATH)
            if upscaled_path:
                discord_send_attachment(msg_obj, "\U0001f50d Upscaled 4x", upscaled_path)
        try:
            if os.path.exists(SD_OUTPUT_PATH): os.remove(SD_OUTPUT_PATH)
            if upscaled_path and os.path.exists(upscaled_path): os.remove(upscaled_path)
        except Exception: pass
        return "Image sent!"

    # ── FLUX.2 / FLUX.1 (fallbacks; also selected explicitly with --model=flux*) ──
    flux_model_path = None
    flux_type = None
    flux_cfg = {}

    # FLUX.2 9B (highest quality)
    fd = _find_model(SD_FLUX2_DIR, SD_FLUX2_MODEL, "flux-2-klein-9b", "flux2-klein-9b")
    fv = _find_model(SD_FLUX2_DIR, SD_FLUX2_VAE, "flux2-vae", "ae")
    fl = _find_model(SD_FLUX2_DIR, SD_FLUX2_LLM, "qwen3-8b-Q4_K_M", "qwen3-8b", "qwen3-4b-Q4_K_M", "qwen3-4b", "t5xxl", "t5-v1_1-xxl")
    if fd and fv and fl:
        flux_model_path = fd
        flux_type = "FLUX.2-9B"
        flux_cfg = {"diffusion": fd, "vae": fv, "llm": fl}
    else:
        # FLUX.2 4B fallback
        fd = _find_model(SD_FLUX2_DIR, SD_FLUX2_MODEL, "flux-2-klein-4b", "flux2-klein-4b", "flux2-4b")
        fv = _find_model(SD_FLUX2_DIR, SD_FLUX2_VAE, "flux2-vae", "ae")
        fl = _find_model(SD_FLUX2_DIR, SD_FLUX2_LLM, "qwen3-4b-Q4_K_M", "qwen3-4b", "t5xxl", "t5-v1_1-xxl")
        if fd and fv and fl:
            flux_model_path = fd
            flux_type = "FLUX.2-4B"
            flux_cfg = {"diffusion": fd, "vae": fv, "llm": fl}

    if want == "flux":
        flux_model_path = None  # explicit FLUX.1 request: skip the FLUX.2 path

    if flux_model_path:
        log(f"  Using {flux_type}: {os.path.basename(flux_model_path)}")
        _warn_sd_vram(flux_type, _sd_estimate_mb(flux_cfg["diffusion"], flux_cfg["vae"], flux_cfg["llm"]))
        log("  Generating image...")
        if os.path.exists(SD_OUTPUT_PATH):
            os.remove(SD_OUTPUT_PATH)
        start = time.time()
        proc = subprocess.Popen([
            SD_CLI, "--diffusion-model", flux_cfg["diffusion"],
            "--vae", flux_cfg["vae"],
            "--llm", flux_cfg["llm"],
            "--offload-to-cpu", "--diffusion-fa", "--vae-tiling",
            "-p", prompt, "-o", SD_OUTPUT_PATH,
            "--steps", "4", "-W", "512", "-H", "512", "--cfg-scale", "1.0"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for file (max 5 min)
        waited = 0
        while waited < SD_TIMEOUT_S:
            if os.path.exists(SD_OUTPUT_PATH) and os.path.getsize(SD_OUTPUT_PATH) > 1000:
                elapsed = time.time() - start
                log(f"  Image ready after {elapsed:.0f}s, killing sd-cli...")
                time.sleep(2)
                proc.kill()
                proc.wait()
                subprocess.run(["killall", "-9", "sd-cli"], capture_output=True)
                break
            time.sleep(3)
            waited += 3
        else:
            proc.kill()
            proc.wait()
            subprocess.run(["killall", "-9", "sd-cli"], capture_output=True)
            log(f"  ERROR: {flux_type} image generation timed out")
            subprocess.run(["rc-service", "llama-swap", "start"])
            return "\U0001f99e Image generation timed out."
        elapsed = time.time() - start
        log(f"  Image: {SD_OUTPUT_PATH} ({os.path.getsize(SD_OUTPUT_PATH)} bytes) in {elapsed:.0f}s")
        # ESRGAN upscale
        upscaled_path = "/tmp/sd-output-upscaled.png"
        if os.path.exists(SD_ESRGAN_MODEL):
            log("  Upscaling 4x with ESRGAN...")
            upscale_start = time.time()
            up_proc = subprocess.run([
                SD_CLI, "--upscale-model", SD_ESRGAN_MODEL,
                "--upscale-repeats", "1",
                "-i", SD_OUTPUT_PATH, "-o", upscaled_path
            ], capture_output=True, text=True, timeout=SD_UPSCALE_TIMEOUT_S)
            if up_proc.returncode == 0 and os.path.exists(upscaled_path):
                log(f"  Upscaled in {time.time() - upscale_start:.0f}s")
            else:
                log(f"  Upscale failed: {up_proc.stderr[:200]}")
                upscaled_path = None
        else:
            upscaled_path = None
        # Restart llama-swap BEFORE sending
        log("  Restarting llama-swap service...")
        subprocess.run(["rc-service", "llama-swap", "start"])
        wait_for_llm()
        # Send to Discord
        if thread_channel:
            discord_send_to_channel(thread_channel, f"\U0001f3a8 {prompt}")
            discord_send_file_to_channel(thread_channel, "\U0001f3a8 Original 512x512", SD_OUTPUT_PATH)
            if upscaled_path:
                discord_send_file_to_channel(thread_channel, "\U0001f50d Upscaled 4x", upscaled_path)
        elif msg_obj:
            discord_reply(msg_obj, f"\U0001f3a8 {prompt}")
            discord_send_attachment(msg_obj, "\U0001f3a8 Original 512x512", SD_OUTPUT_PATH)
            if upscaled_path:
                discord_send_attachment(msg_obj, "\U0001f50d Upscaled 4x", upscaled_path)
        try:
            if os.path.exists(SD_OUTPUT_PATH): os.remove(SD_OUTPUT_PATH)
            if upscaled_path and os.path.exists(upscaled_path): os.remove(upscaled_path)
        except Exception: pass
        return "Image sent!"

    # ── FLUX.1 (LAST FALLBACK) — latent diffusion with VAE + text encoders ──
    flux_dir = SD_FLUX_DIR
    t5xxl = ""
    extra_flags = []
    fm = _find_model(flux_dir, SD_FLUX_MODEL, 'flux1-schnell-q4_k', 'flux1-schnell', 'flux1-dev-Q4_K_M', 'flux1-dev')
    fv = _find_model(flux_dir, SD_FLUX_VAE, 'ae')
    fc = _find_model(flux_dir, SD_FLUX_CLIP_L, 'clip_l')
    t5q4 = _find_model(flux_dir, SD_FLUX_T5XXL, 't5-v1_1-xxl-encoder-Q4_K_M')
    t5q8 = _find_model(flux_dir, SD_FLUX_T5XXL, 't5-v1_1-xxl-encoder-Q8_0')
    if fm and fv and fc:
        if t5q4:
            t5xxl = t5q4
        elif t5q8:
            t5xxl = t5q8
            extra_flags = ["--mmap"]
        if t5xxl:
            log(f"  Using FLUX.1: {os.path.basename(fm)} with {os.path.basename(t5xxl)}")
        else:
            log("  ERROR: No T5XXL encoder found")
            subprocess.run(["rc-service", "llama-swap", "start"])
            return "\U0001f99e No FLUX model found (missing T5XXL encoder)."
    else:
        log("  ERROR: No FLUX model found")
        subprocess.run(["rc-service", "llama-swap", "start"])
        return "\U0001f99e No FLUX model found. Drop models into flux/ or flux2/ dirs, or set SD_* vars in .env."
    # Generate
    _warn_sd_vram("FLUX.1", _sd_estimate_mb(fm, fv, t5xxl))
    log("  Generating image...")
    if os.path.exists(SD_OUTPUT_PATH):
        os.remove(SD_OUTPUT_PATH)
    start = time.time()
    cmd = [
        SD_CLI, "--diffusion-model", fm,
        "--vae", fv,
        "--clip_l", fc,
        "--t5xxl", t5xxl,
        "--clip-on-cpu", "--offload-to-cpu", "--fa", "--vae-tiling",
        "-p", prompt, "-o", SD_OUTPUT_PATH,
        "--steps", "4", "-W", "512", "-H", "512", "--cfg-scale", "1.0",
        "--sampling-method", "euler"
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Wait for file (max 5 min)
    waited = 0
    while waited < SD_TIMEOUT_S:
        if os.path.exists(SD_OUTPUT_PATH) and os.path.getsize(SD_OUTPUT_PATH) > 1000:
            elapsed = time.time() - start
            log(f"  Image ready after {elapsed:.0f}s, killing sd-cli...")
            time.sleep(2)
            proc.kill()
            proc.wait()
            subprocess.run(["killall", "-9", "sd-cli"], capture_output=True)
            break
        time.sleep(3)
        waited += 3
    else:
        proc.kill()
        proc.wait()
        subprocess.run(["killall", "-9", "sd-cli"], capture_output=True)
        log("  ERROR: Image generation timed out")
        subprocess.run(["rc-service", "llama-swap", "start"])
        return "🦞 Image generation timed out."
    size = os.path.getsize(SD_OUTPUT_PATH)
    elapsed = time.time() - start
    log(f"  Image: {SD_OUTPUT_PATH} ({size} bytes) in {elapsed:.0f}s")
    # ESRGAN upscale
    upscaled_path = "/tmp/sd-output-upscaled.png"
    if os.path.exists(SD_ESRGAN_MODEL):
        log("  Upscaling 4x with ESRGAN...")
        upscale_start = time.time()
        up_proc = subprocess.run([
            SD_CLI, "--upscale-model", SD_ESRGAN_MODEL,
            "--upscale-repeats", "1",
            "-i", SD_OUTPUT_PATH, "-o", upscaled_path
        ], capture_output=True, text=True, timeout=SD_UPSCALE_TIMEOUT_S)
        if up_proc.returncode == 0 and os.path.exists(upscaled_path):
            up_elapsed = time.time() - upscale_start
            log(f"  Upscaled in {up_elapsed:.0f}s")
        else:
            log(f"  Upscale failed: {up_proc.stderr[:200]}")
            upscaled_path = None
    else:
        upscaled_path = None
    # Restart llama-swap BEFORE sending
    log("  Restarting llama-swap service...")
    subprocess.run(["rc-service", "llama-swap", "start"])
    wait_for_llm()
    # Send to Discord
    if thread_channel:
        discord_send_to_channel(thread_channel, f"🎨 {prompt}")
        discord_send_file_to_channel(thread_channel, "🎨 Original 512x512", SD_OUTPUT_PATH)
        if upscaled_path:
            discord_send_file_to_channel(thread_channel, "🔍 Upscaled 4x", upscaled_path)
    elif msg_obj:
        discord_reply(msg_obj, f"🎨 {prompt}")
        discord_send_attachment(msg_obj, "🎨 Original 512x512", SD_OUTPUT_PATH)
        if upscaled_path:
            discord_send_attachment(msg_obj, "🔍 Upscaled 4x", upscaled_path)
    # Cleanup tmp files
    try:
        if os.path.exists(SD_OUTPUT_PATH): os.remove(SD_OUTPUT_PATH)
        if upscaled_path and os.path.exists(upscaled_path): os.remove(upscaled_path)
    except Exception: pass
    return "Image sent!"


def _edit_target_size(reference_path):
    """(w, h) para editar: lado largo = SD_QWEN_EDIT_SIZE, aspecto de la referencia,
    ambos múltiplos de 32 (lo que exige sd.cpp). Cuadrado si no hay PIL o tamaño.
    """
    base = SD_QWEN_EDIT_SIZE
    if SD_QWEN_EDIT_SQUARE:
        return base, base
    try:
        from PIL import Image
        with Image.open(reference_path) as im:
            w, h = im.size
    except Exception as exc:
        log(f"  No pude leer el tamaño de la referencia ({exc}); edito en {base}x{base}")
        return base, base
    if not w or not h:
        return base, base
    scale = base / float(max(w, h))
    tw = max(32, int(round(w * scale)) // 32 * 32)
    th = max(32, int(round(h * scale)) // 32 * 32)
    return tw, th


def edit_image(instruction, reference_path, msg_obj=None, thread_channel=None):
    if thread_channel:
        discord_send_to_channel(thread_channel, "🖌️ Editing image...")
    elif msg_obj:
        discord_reply(msg_obj, "🖌️ Editing image...")
    log("  Stopping llama-swap service...")
    subprocess.run(["rc-service", "llama-swap", "stop"])
    time.sleep(3)

    # ── Qwen-Image 2.1 editing (SOTA native instruction-based DiT editing) ──
    qwen_diff = _find_model(SD_QWEN_DIR, SD_QWEN_MODEL,
        "qwen_image_2.1-Q4_K", "qwen_image_2.1-Q4_0", "qwen_image_2.1-Q5_0",
        "qwen_image_2.1-Q8_0", "qwen_image_2.1_int8_convrot", "qwen_image_2.1")
    qwen_vae = _find_model(SD_QWEN_DIR, SD_QWEN_VAE,
        "qwen_image_2.1_vae_bf16", "qwen_image_2.1_vae", "qwen_image_vae")
    qwen_llm = _find_model(SD_QWEN_DIR, SD_QWEN_LLM,
        "Qwen3VL-8B-Instruct-Q4_K_M", "Qwen3VL-8B-Instruct-Q4_0",
        "Qwen3VL-8B-Instruct", "qwen3vl-8b", "qwen3-vl")
    qwen_vision = _find_model(SD_QWEN_DIR, SD_QWEN_VISION,
        "mmproj-Qwen3VL-8B-Instruct-F16", "mmproj-Qwen3VL-8B-Instruct",
        "Qwen3VL-8B-Instruct-mmproj", "mmproj-Qwen3VL")

    # ── HiDream-O1 editing (PRIMARY) — native instruction-based editing ──
    hidream_model = _find_model(SD_HIDREAM_DIR, SD_HIDREAM_MODEL,
        'HiDream-O1-Image-Dev-BF16', 'HiDream-O1-Image-Dev-FP8',
        'HiDream-O1-Image-Full-FP8',
        'HiDream-O1-Image-Dev', 'HiDream-O1-Image-Full',
        'hidream-o1-dev', 'hidream-o1')

    use_qwen_edit = bool(qwen_diff and qwen_vae and qwen_llm and qwen_vision and (SD_QWEN_MODEL or not hidream_model))
    if use_qwen_edit:
        log(f"  Editing with Qwen-Image 2.1: {os.path.basename(qwen_diff)}")
        _warn_sd_vram("Qwen-Image-2.1 edit", _sd_estimate_mb(qwen_diff, qwen_vae, qwen_llm, qwen_vision))
        # Referencia con su propio aspecto: el achatado a cuadrado deformaba toda
        # foto no cuadrada antes de editarla. sd.cpp quiere múltiplos de 32.
        edit_w, edit_h = _edit_target_size(reference_path)
        resized_path = "/tmp/sd-edit-reference.png"
        try:
            from PIL import Image
            with Image.open(reference_path) as src:
                src.convert("RGB").resize((edit_w, edit_h), Image.Resampling.LANCZOS).save(resized_path)
            log(f"  Editando la referencia a {edit_w}x{edit_h} (aspecto original)")
        except Exception as exc:
            log(f"  Resize warning: {exc} — using original image (may fail if dimensions not divisible by 32)")
            resized_path, edit_w, edit_h = reference_path, SD_QWEN_SIZE, SD_QWEN_SIZE
        edited_path = "/tmp/sd-edited.png"
        if os.path.exists(edited_path):
            os.remove(edited_path)
        start = time.time()
        proc = subprocess.Popen([
            SD_CLI,
            "--diffusion-model", qwen_diff,
            "--vae", qwen_vae,
            "--llm", qwen_llm,
            "--llm_vision", qwen_vision,
            "-r", resized_path,
            "-p", instruction,
            "-o", edited_path,
            *_qwen_edit_args(edit_w, edit_h)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        waited = 0
        while waited < SD_EDIT_TIMEOUT_S:
            if os.path.exists(edited_path) and os.path.getsize(edited_path) > 1000:
                elapsed = time.time() - start
                log(f"  Edited image ready after {elapsed:.0f}s")
                time.sleep(2)
                proc.kill()
                proc.wait()
                subprocess.run(["killall", "-9", "sd-cli"], capture_output=True)
                break
            time.sleep(3)
            waited += 3
        else:
            proc.kill()
            proc.wait()
            subprocess.run(["killall", "-9", "sd-cli"], capture_output=True)
            log("  ERROR: Qwen-Image 2.1 editing timed out")
            subprocess.run(["rc-service", "llama-swap", "start"])
            return "🦞 Image editing timed out."
        log("  Restarting llama-swap...")
        subprocess.run(["rc-service", "llama-swap", "start"])
        wait_for_llm()
        if thread_channel:
            discord_send_file_to_channel(thread_channel, f"🖌️ {instruction}", edited_path)
        elif msg_obj:
            discord_send_attachment(msg_obj, f"🖌️ {instruction}", edited_path)
        try:
            if os.path.exists(edited_path): os.remove(edited_path)
            if resized_path != reference_path and os.path.exists(resized_path): os.remove(resized_path)
        except Exception: pass
        return "Image edited!"

    elif hidream_model:
        log(f"  Editing with HiDream-O1: {os.path.basename(hidream_model)}")
        _warn_sd_vram("HiDream-O1 edit", _sd_estimate_mb(hidream_model) * 0.68, headroom_mb=768)
        edited_path = "/tmp/sd-edited.png"
        if os.path.exists(edited_path):
            os.remove(edited_path)
        start = time.time()
        proc = subprocess.Popen([
            SD_CLI, "-m", hidream_model,
            "-p", instruction, "-i", reference_path, "-o", edited_path,
            "--steps", "12", "-W", "256", "-H", "256", "--cfg-scale", "1.0",
            "--tensor-type-rules", HIDREAM_TENSOR_RULES
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        waited = 0
        while waited < SD_EDIT_TIMEOUT_S:
            if os.path.exists(edited_path) and os.path.getsize(edited_path) > 1000:
                elapsed = time.time() - start
                log(f"  Edited image ready after {elapsed:.0f}s")
                time.sleep(2)
                proc.kill()
                proc.wait()
                subprocess.run(["killall", "-9", "sd-cli"], capture_output=True)
                break
            time.sleep(3)
            waited += 3
        else:
            proc.kill()
            proc.wait()
            subprocess.run(["killall", "-9", "sd-cli"], capture_output=True)
            log("  ERROR: HiDream image editing timed out")
            subprocess.run(["rc-service", "llama-swap", "start"])
            return "🦞 Image editing timed out."
        log("  Restarting llama-swap...")
        subprocess.run(["rc-service", "llama-swap", "start"])
        wait_for_llm()
        if thread_channel:
            discord_send_file_to_channel(thread_channel, f"🖌️ {instruction}", edited_path)
        elif msg_obj:
            discord_send_attachment(msg_obj, f"🖌️ {instruction}", edited_path)
        try:
            if os.path.exists(edited_path): os.remove(edited_path)
        except Exception: pass
        return "Image edited!"

    # ── FLUX Kontext editing (FALLBACK) — needs VAE + CLIP + T5 ──
    flux_dir = SD_FLUX_DIR
    kontext_model = _find_model(flux_dir, SD_KONTEXT_MODEL, 'flux1-kontext-dev-Q4_K_M', 'flux1-kontext-dev')
    if not kontext_model:
        log("  ERROR: FLUX.1-Kontext-dev model not found")
        subprocess.run(["rc-service", "llama-swap", "start"])
        return "🦞 Kontext model not found. Set SD_KONTEXT_MODEL in .env or drop it in flux/."
    # Resize reference to 512x512
    resized_path = "/tmp/sd-edit-reference.png"
    try:
        from PIL import Image
        img = Image.open(reference_path)
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
        img.save(resized_path)
    except Exception as e:
        log(f"  Resize warning: {e} — using original image (may fail if not 512x512)")
        resized_path = reference_path
    edited_path = "/tmp/sd-edited.png"
    if os.path.exists(edited_path):
        os.remove(edited_path)
    t5xxl = _find_model(flux_dir, SD_FLUX_T5XXL, 't5-v1_1-xxl-encoder-Q4_K_M', 't5-v1_1-xxl-encoder-Q8_0')
    fv = _find_model(flux_dir, SD_FLUX_VAE, 'ae')
    fc = _find_model(flux_dir, SD_FLUX_CLIP_L, 'clip_l')
    if not t5xxl:
        log("  ERROR: No T5XXL found for Kontext")
        subprocess.run(["rc-service", "llama-swap", "start"])
        return "🦞 No T5XXL found for editing. Set SD_FLUX_T5XXL in .env."
    log(f"  Editing with {os.path.basename(kontext_model)}...")
    start = time.time()
    if not fv or not fc:
        log("  ERROR: Missing VAE or CLIP-L for Kontext editing")
        subprocess.run(["rc-service", "llama-swap", "start"])
        return "🦞 Missing VAE or CLIP-L. Set SD_FLUX_VAE / SD_FLUX_CLIP_L in .env."
    proc = subprocess.Popen([
        SD_CLI, "--diffusion-model", kontext_model,
        "--vae", fv,
        "--clip_l", fc,
        "--t5xxl", t5xxl,
        "-p", instruction, "--img2img", resized_path, "-o", edited_path,
        "--steps", "4", "-W", "512", "-H", "512", "--cfg-scale", "1.0"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    waited = 0
    while waited < SD_EDIT_TIMEOUT_S:
        if os.path.exists(edited_path) and os.path.getsize(edited_path) > 1000:
            elapsed = time.time() - start
            log(f"  Edited image ready after {elapsed:.0f}s")
            time.sleep(2)
            proc.kill()
            proc.wait()
            subprocess.run(["killall", "-9", "sd-cli"], capture_output=True)
            break
        time.sleep(3)
        waited += 3
    else:
        proc.kill()
        proc.wait()
        subprocess.run(["killall", "-9", "sd-cli"], capture_output=True)
        log("  ERROR: Image editing timed out")
        subprocess.run(["rc-service", "llama-swap", "start"])
        return "🦞 Image editing timed out."
    # Restart llama-swap
    log("  Restarting llama-swap...")
    subprocess.run(["rc-service", "llama-swap", "start"])
    wait_for_llm()
    if thread_channel:
        discord_send_file_to_channel(thread_channel, f"🖌️ {instruction}", edited_path)
    elif msg_obj:
        discord_send_attachment(msg_obj, f"🖌️ {instruction}", edited_path)
    # Cleanup tmp files
    try:
        if os.path.exists(edited_path): os.remove(edited_path)
        if os.path.exists(resized_path) and resized_path != reference_path: os.remove(resized_path)
    except Exception: pass
    return "Image edited!"


def upscale_image(image_path, msg_obj=None, thread_channel=None):
    if thread_channel:
        discord_send_to_channel(thread_channel, "🔍 Upscaling 4x...")
    elif msg_obj:
        discord_reply(msg_obj, "🔍 Upscaling 4x...")
    log("  Stopping llama-swap service...")
    subprocess.run(["rc-service", "llama-swap", "stop"])
    time.sleep(2)
    if not os.path.exists(SD_ESRGAN_MODEL):
        subprocess.run(["rc-service", "llama-swap", "start"])
        return "🦞 ESRGAN model not found."
    upscaled_path = "/tmp/sd-upscaled.png"
    if os.path.exists(upscaled_path):
        os.remove(upscaled_path)
    log("  Upscaling with ESRGAN...")
    start = time.time()
    up_proc = subprocess.run([
        SD_CLI, "--upscale-model", SD_ESRGAN_MODEL,
        "--upscale-repeats", "1",
        "-i", image_path, "-o", upscaled_path
    ], capture_output=True, text=True, timeout=SD_UPSCALE_TIMEOUT_S)
    if up_proc.returncode == 0 and os.path.exists(upscaled_path):
        elapsed = time.time() - start
        log(f"  Upscaled in {elapsed:.0f}s")
        if thread_channel:
            discord_send_file_to_channel(thread_channel, "🔍 Upscaled 4x", upscaled_path)
        elif msg_obj:
            discord_send_attachment(msg_obj, "🔍 Upscaled 4x", upscaled_path)
    else:
        log(f"  Upscale failed: {up_proc.stderr[:200]}")
    log("  Restarting llama-swap...")
    subprocess.run(["rc-service", "llama-swap", "start"])
    wait_for_llm()
    try:
        if os.path.exists(upscaled_path): os.remove(upscaled_path)
    except Exception: pass
    return "Image upscaled!" if (up_proc.returncode == 0 and os.path.exists(upscaled_path)) else "🦞 Upscale failed."


def generate_video(prompt, msg_obj=None, thread_channel=None, input_image=None):
    """Generate a video with WAN. If input_image is provided, use image-to-video
    (I2V) with the WAN 2.2 I2V models; otherwise text-to-video (T2V)."""
    is_i2v = bool(input_image and os.path.exists(input_image))
    if thread_channel:
        discord_send_to_channel(thread_channel, "🎬 Generating video (this takes ~30–40 min on Vulkan)...")
    elif msg_obj:
        discord_reply(msg_obj, "🎬 Generating video (this takes ~30–40 min on Vulkan)...")
    # Stop llama-swap to free VRAM
    log("  Stopping llama-swap service...")
    subprocess.run(["rc-service", "llama-swap", "stop"])
    time.sleep(3)
    # Auto-detect WAN model (env var > directory scan, supports .gguf + .safetensors)
    # WAN 2.2 is two-pass: LowNoise diffusion + optional HighNoise diffusion model.
    wan_dir = SD_WAN_DIR
    if is_i2v:
        # Image-to-video: use WAN 2.2 I2V A14B LowNoise + HighNoise
        wan_model = _find_model(wan_dir, SD_WAN_MODEL,
            'Wan2.2-I2V-A14B-LowNoise-Q4_K_M', 'Wan2.2-I2V-A14B-LowNoise-Q8_0')
        if not wan_model:
            log("  ERROR: No WAN I2V model found")
            subprocess.run(["rc-service", "llama-swap", "start"])
            return "🦞 No WAN 2.2 I2V model found. Set SD_WAN_MODEL to a Wan2.2-I2V-A14B-LowNoise file or download it via configure."
        log(f"  Using WAN 2.2 I2V: {os.path.basename(wan_model)}")
        wan_high = _find_model(wan_dir, SD_WAN_HIGH_NOISE,
            'Wan2.2-I2V-A14B-HighNoise-Q4_K_M', 'Wan2.2-I2V-A14B-HighNoise-Q8_0')
    else:
        # Text-to-video: WAN 2.2 T2V or legacy WAN 2.1 T2V
        wan_model = _find_model(wan_dir, SD_WAN_MODEL,
            'Wan2.2-T2V-A14B-LowNoise-Q4_K_M', 'Wan2.2-T2V-A14B-LowNoise-Q8_0',
            'wan2.1_t2v_1.3B_fp16', 'wan2.1_t2v_1.3B_q8_0', 'wan2.1_t2v_1.3B_q4_0')
        if not wan_model:
            log("  ERROR: No WAN video model found")
            subprocess.run(["rc-service", "llama-swap", "start"])
            return "🦞 No WAN model found. Set SD_WAN_MODEL in .env or drop it in wan/."
        log(f"  Using WAN: {os.path.basename(wan_model)}")
        wan_high = _find_model(wan_dir, SD_WAN_HIGH_NOISE,
            'Wan2.2-T2V-A14B-HighNoise-Q4_K_M', 'Wan2.2-T2V-A14B-HighNoise-Q8_0')
    wan_vae = _find_model(wan_dir, SD_WAN_VAE, 'wan_2.1_vae')
    if not wan_vae:
        log("  ERROR: WAN VAE not found")
        subprocess.run(["rc-service", "llama-swap", "start"])
        return "🦞 WAN VAE missing. Set SD_WAN_VAE in .env."
    t5xxl = _find_model(wan_dir, SD_WAN_T5XXL,
        'umt5-xxl-encoder-Q5_K_M', 'umt5-xxl-encoder-Q4_K_M', 'umt5-xxl-encoder-Q8_0')
    if not t5xxl:
        log("  ERROR: No umt5-xxl encoder found for WAN")
        subprocess.run(["rc-service", "llama-swap", "start"])
        return "🦞 WAN text encoder (umt5-xxl) missing. Set SD_WAN_T5XXL in .env."
    # Generate video
    log(f"  Generating video with WAN {'I2V' if is_i2v else 'T2V'}...")
    if os.path.exists(SD_VIDEO_OUTPUT):
        os.remove(SD_VIDEO_OUTPUT)
    if os.path.exists(SD_VIDEO_MP4):
        os.remove(SD_VIDEO_MP4)
    start = time.time()
    cmd = [
        SD_CLI, "-M", "vid_gen",
        "--diffusion-model", wan_model,
        "--vae", wan_vae,
        "--t5xxl", t5xxl,
        "--diffusion-fa", "--vae-tiling",
        "-p", prompt,
        "--steps", "20", "--cfg-scale", "5",
        "--sampling-method", "euler", "--scheduler", "simple",
        "-W", "384", "-H", "216", "--fps", "12", "--video-frames", "85",
        "-o", SD_VIDEO_OUTPUT
    ]
    if wan_high:
        # WAN 2.2 two-pass (docs recommend flow-shift 3.0 for all WAN 2.2 modes)
        log(f"  WAN 2.2 two-pass: high-noise model {os.path.basename(wan_high)}")
        cmd += ["--high-noise-diffusion-model", wan_high,
                "--high-noise-steps", "8", "--high-noise-cfg-scale", "5",
                "--high-noise-sampling-method", "euler",
                "--flow-shift", "3.0"]
    if is_i2v:
        # I2V: feed the attached image
        if "--flow-shift" not in cmd:
            cmd += ["--flow-shift", "3.0"]
        cmd += ["-i", input_image]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Wait for output (max 40 min, poll every 10s)
    waited = 0
    while waited < SD_VIDEO_TIMEOUT_S:
        if os.path.exists(SD_VIDEO_OUTPUT) and os.path.getsize(SD_VIDEO_OUTPUT) > 10000:
            elapsed = time.time() - start
            log(f"  Video ready after {elapsed:.0f}s, killing sd-cli...")
            time.sleep(2)
            proc.kill()
            proc.wait()
            subprocess.run(["killall", "-9", "sd-cli"], capture_output=True)
            break
        time.sleep(10)
        waited += 10
    else:
        proc.kill()
        proc.wait()
        subprocess.run(["killall", "-9", "sd-cli"], capture_output=True)
        log("  ERROR: Video generation timed out")
        subprocess.run(["rc-service", "llama-swap", "start"])
        return "🦞 Video generation timed out (40 min limit)."
    size = os.path.getsize(SD_VIDEO_OUTPUT)
    elapsed = time.time() - start
    log(f"  Video: {SD_VIDEO_OUTPUT} ({size} bytes) in {elapsed:.0f}s")
    # Convert AVI -> MP4 with ffmpeg if available (Discord prefers MP4)
    final_path = SD_VIDEO_OUTPUT
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        log("  Converting AVI -> MP4 with ffmpeg...")
        conv = subprocess.run([
            ffmpeg_path, "-y", "-i", SD_VIDEO_OUTPUT,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            SD_VIDEO_MP4
        ], capture_output=True, text=True, timeout=300)
        if conv.returncode == 0 and os.path.exists(SD_VIDEO_MP4):
            final_path = SD_VIDEO_MP4
            log(f"  MP4 ready: {os.path.getsize(final_path)} bytes")
        else:
            log(f"  ffmpeg conversion failed, sending AVI")
    # Restart llama-swap BEFORE sending
    log("  Restarting llama-swap service...")
    subprocess.run(["rc-service", "llama-swap", "start"])
    wait_for_llm()
    # Send to Discord
    if thread_channel:
        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        if size_mb > 8:
            discord_send_to_channel(thread_channel, f"🎬 Video ready ({size_mb:.1f} MB) — exceeds Discord 8MB limit for non-Nitro. File: {final_path}")
        else:
            discord_send_file_to_channel(thread_channel, f"🎬 {prompt}", final_path)
    elif msg_obj:
        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        if size_mb > 8:
            discord_reply(msg_obj, f"🎬 Video ready ({size_mb:.1f} MB) — exceeds Discord 8MB limit for non-Nitro. File: {final_path}")
        else:
            discord_send_attachment(msg_obj, f"🎬 {prompt}", final_path)
    # Cleanup tmp files
    try:
        if os.path.exists(SD_VIDEO_OUTPUT): os.remove(SD_VIDEO_OUTPUT)
        if os.path.exists(SD_VIDEO_MP4): os.remove(SD_VIDEO_MP4)
    except Exception: pass
    return "Video sent!"

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    global _current_mode
    log("Starting Queue Runner v10 (Secure + Cron + Discord)")
    load_history()
    start_discord_bot()
    start_chat_worker()
    # Warn once if LLM_MODEL doesn't match any model advertised by llama-swap
    # (turns the cryptic 'HTTP 404' on chat into an actionable startup warning).
    # Retries briefly in case llama-swap is still booting.
    for _ in range(3):
        if is_llm_healthy():
            check_llm_model_config()
            break
        time.sleep(2)
    sd_notify('READY=1')
    state = {'cycle': 0, 'last_runs': {}}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f: state = json.load(f)
        except: pass
    last_memory_check = 0
    while running:
        state['cycle'] += 1
        all_jobs = []
        if os.path.exists(JOBS_JSON):
            try:
                with open(JOBS_JSON) as f: all_jobs = json.load(f)['jobs']
            except: pass
        for job in all_jobs:
            if not running: break
            if job.get('enabled', True):
                run_job(job, state)
        # NOTE: Discord chat, chat-history cleanup and pending-script expiry are
        # NOT handled here — they run in their own thread (see chat_worker_loop),
        # so a long job can no longer mute the bot, and a slow LLM turn can no
        # longer stall the scheduler.
        now = time.time()
        # Memory summarization (compress lessons/corrections into profile when GPU idle)
        if now - last_memory_check >= 3600:  # Every hour, summarize memory if needed
            _maybe_summarize_memory()
            last_memory_check = now
        # Auto-sleep curfew: if smart mode is active past curfew hour, switch to sleep
        curfew_hour = datetime.now().hour
        if _current_mode == 'smart' and curfew_hour >= SMART_MODE_CURFEW_HOUR and curfew_hour < USEFUL_HOURS_START:
            log(f"  CURFEW: smart mode active past {SMART_MODE_CURFEW_HOUR}:00 — switching to sleep")
            _current_mode = 'sleep'
            subprocess.run(["rc-service", "llama-swap", "stop"], capture_output=True)
            if _discord_bot:
                async def notify_curfew():
                    owner = await _discord_bot.fetch_user(DISCORD_OWNER_ID)
                    if owner: await owner.send(f"💤 **Curfew** — modo smart activo después de las {SMART_MODE_CURFEW_HOUR}:00. Cambiando a modo sleep para ahorrar energía. Escribe **despertar** para reactivar.")
                asyncio.run_coroutine_threadsafe(notify_curfew(), _discord_bot.loop)
        if '--once' in sys.argv: break
        time.sleep(10)
if __name__ == '__main__': main()
