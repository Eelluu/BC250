#!/bin/bash
# BC-250 Comprehensive Image Generation Benchmark
# Tests ALL diffusion models at multiple resolutions and step counts
# Runs on BC-250 directly. Stops Ollama to free GPU memory.
#
# Usage: bash bench-sd-comprehensive.sh

set -uo pipefail  # Don't exit on error (set +e), continue benchmarking

SD_CLI="/opt/stable-diffusion.cpp/build/bin/sd-cli"
M="/opt/stable-diffusion.cpp/models"
OUT="/opt/netscan/tmp/bench-sd-images"
LOG="/opt/netscan/tmp/bench-sd-comprehensive.log"
RESULTS="/opt/netscan/tmp/bench-sd-results.json"

mkdir -p "$OUT"

PROMPT="A photorealistic cyberpunk cityscape at night with neon lights reflecting on wet streets, detailed architecture, volumetric lighting"

# Stop Ollama to free all GPU memory for image gen
echo "Stopping Ollama and queue-runner..." | tee "$LOG"
sudo systemctl stop ollama 2>/dev/null || true
sudo systemctl stop queue-runner 2>/dev/null || true
sleep 3

echo "================================================================" | tee -a "$LOG"
echo "BC-250 Image Gen Benchmark — $(date '+%Y-%m-%d %H:%M')" | tee -a "$LOG"
echo "sd.cpp $(cd /opt/stable-diffusion.cpp && git describe --tags 2>/dev/null || echo 'unknown')" | tee -a "$LOG"
echo "Vulkan · GFX1013 · --offload-to-cpu --fa" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"

# Initialize JSON results
echo "[]" > "$RESULTS"

append_result() {
    local model="$1" res="$2" steps="$3" time="$4" status="$5" fsize="$6" notes="$7"
    python3 -c "
import json, sys
with open('$RESULTS') as f: data = json.load(f)
data.append({
    'model': '$model', 'resolution': '$res', 'steps': $steps,
    'time_s': $time, 'status': '$status', 'file_size': '$fsize', 'notes': '$notes'
})
with open('$RESULTS', 'w') as f: json.dump(data, f, indent=2)
"
}

run_test() {
    local label="$1"; shift
    local model_name="$1"; shift
    local resolution="$1"; shift
    local steps="$1"; shift
    local outfile="$OUT/${label}.png"
    rm -f "$outfile"

    printf "%-50s " "$label" | tee -a "$LOG"

    local t0=$SECONDS
    # Run with timeout, capture stderr for timing info
    timeout 900 "$SD_CLI" "$@" --offload-to-cpu --fa -p "$PROMPT" --seed 42 -o "$outfile" >/dev/null 2>&1 &
    local pid=$!

    # Poll for output file (GFX1013 sometimes hangs after write)
    local i=0
    while [ $i -lt 180 ]; do
        sleep 5
        i=$((i+1))
        if [ -f "$outfile" ]; then
            local sz
            sz=$(stat -c%s "$outfile" 2>/dev/null || echo 0)
            if [ "$sz" -gt 1000 ]; then
                sleep 2
                local elapsed=$((SECONDS - t0))
                local fsize=$(du -h "$outfile" | cut -f1)
                local sstep=""
                if [ "$steps" -gt 0 ]; then
                    sstep=$(python3 -c "print(f'{$elapsed/$steps:.1f}')")
                fi
                kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
                pkill -f sd-cli 2>/dev/null; sleep 2
                echo "OK  ${elapsed}s  (${sstep}s/step)  ${fsize}" | tee -a "$LOG"
                append_result "$model_name" "$resolution" "$steps" "$elapsed" "OK" "$fsize" "${sstep}s/step"
                return 0
            fi
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            if [ -f "$outfile" ] && [ "$(stat -c%s "$outfile" 2>/dev/null || echo 0)" -gt 1000 ]; then
                local elapsed=$((SECONDS - t0))
                local fsize=$(du -h "$outfile" | cut -f1)
                echo "OK  ${elapsed}s  ${fsize}" | tee -a "$LOG"
                append_result "$model_name" "$resolution" "$steps" "$elapsed" "OK" "$fsize" ""
                return 0
            else
                echo "FAIL (exit, no output)" | tee -a "$LOG"
                append_result "$model_name" "$resolution" "$steps" "0" "FAIL" "0" "process exited without output"
                return 1
            fi
        fi
    done
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    pkill -f sd-cli 2>/dev/null
    echo "TIMEOUT (900s)" | tee -a "$LOG"
    append_result "$model_name" "$resolution" "$steps" "900" "TIMEOUT" "0" ""
    return 1
}

# =============================================================================
# FLUX.2-klein-9B Q4_0 — current default, highest quality
# =============================================================================
echo "" | tee -a "$LOG"
echo "=== FLUX.2-klein-9B Q4_0 ===" | tee -a "$LOG"

run_test "klein9b_512x512_s4" "FLUX.2-klein-9B" "512x512" 4 \
  --diffusion-model "$M/flux2/flux-2-klein-9b-Q4_0.gguf" \
  --vae "$M/flux2/flux2-vae.safetensors" \
  --llm "$M/flux2/qwen3-8b-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu \
  -W 512 -H 512 --steps 4

run_test "klein9b_768x768_s4" "FLUX.2-klein-9B" "768x768" 4 \
  --diffusion-model "$M/flux2/flux-2-klein-9b-Q4_0.gguf" \
  --vae "$M/flux2/flux2-vae.safetensors" \
  --llm "$M/flux2/qwen3-8b-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu --vae-tiling \
  -W 768 -H 768 --steps 4

run_test "klein9b_1024x1024_s4" "FLUX.2-klein-9B" "1024x1024" 4 \
  --diffusion-model "$M/flux2/flux-2-klein-9b-Q4_0.gguf" \
  --vae "$M/flux2/flux2-vae.safetensors" \
  --llm "$M/flux2/qwen3-8b-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu --vae-tiling \
  -W 1024 -H 1024 --steps 4

run_test "klein9b_512x512_s8" "FLUX.2-klein-9B" "512x512" 8 \
  --diffusion-model "$M/flux2/flux-2-klein-9b-Q4_0.gguf" \
  --vae "$M/flux2/flux2-vae.safetensors" \
  --llm "$M/flux2/qwen3-8b-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu \
  -W 512 -H 512 --steps 8

# =============================================================================
# FLUX.2-klein-4B Q4_0 — fast alternative
# =============================================================================
echo "" | tee -a "$LOG"
echo "=== FLUX.2-klein-4B Q4_0 ===" | tee -a "$LOG"

run_test "klein4b_512x512_s4" "FLUX.2-klein-4B" "512x512" 4 \
  --diffusion-model "$M/flux2/flux-2-klein-4b-Q4_0.gguf" \
  --vae "$M/flux2/flux2-vae.safetensors" \
  --llm "$M/flux2/qwen3-4b-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu \
  -W 512 -H 512 --steps 4

run_test "klein4b_768x768_s4" "FLUX.2-klein-4B" "768x768" 4 \
  --diffusion-model "$M/flux2/flux-2-klein-4b-Q4_0.gguf" \
  --vae "$M/flux2/flux2-vae.safetensors" \
  --llm "$M/flux2/qwen3-4b-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu --vae-tiling \
  -W 768 -H 768 --steps 4

run_test "klein4b_1024x1024_s4" "FLUX.2-klein-4B" "1024x1024" 4 \
  --diffusion-model "$M/flux2/flux-2-klein-4b-Q4_0.gguf" \
  --vae "$M/flux2/flux2-vae.safetensors" \
  --llm "$M/flux2/qwen3-4b-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu --vae-tiling \
  -W 1024 -H 1024 --steps 4

run_test "klein4b_512x512_s8" "FLUX.2-klein-4B" "512x512" 8 \
  --diffusion-model "$M/flux2/flux-2-klein-4b-Q4_0.gguf" \
  --vae "$M/flux2/flux2-vae.safetensors" \
  --llm "$M/flux2/qwen3-4b-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu \
  -W 512 -H 512 --steps 8

run_test "klein4b_1024x1024_s8" "FLUX.2-klein-4B" "1024x1024" 8 \
  --diffusion-model "$M/flux2/flux-2-klein-4b-Q4_0.gguf" \
  --vae "$M/flux2/flux2-vae.safetensors" \
  --llm "$M/flux2/qwen3-4b-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu --vae-tiling \
  -W 1024 -H 1024 --steps 8

# =============================================================================
# FLUX.1-schnell Q4_K — previous default
# =============================================================================
echo "" | tee -a "$LOG"
echo "=== FLUX.1-schnell Q4_K ===" | tee -a "$LOG"

run_test "schnell_512x512_s4" "FLUX.1-schnell" "512x512" 4 \
  --diffusion-model "$M/flux/flux1-schnell-q4_k.gguf" \
  --vae "$M/flux/ae.safetensors" --clip_l "$M/flux/clip_l.safetensors" \
  --t5xxl "$M/flux/t5-v1_1-xxl-encoder-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu \
  -W 512 -H 512 --steps 4

run_test "schnell_768x768_s4" "FLUX.1-schnell" "768x768" 4 \
  --diffusion-model "$M/flux/flux1-schnell-q4_k.gguf" \
  --vae "$M/flux/ae.safetensors" --clip_l "$M/flux/clip_l.safetensors" \
  --t5xxl "$M/flux/t5-v1_1-xxl-encoder-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu --vae-tiling \
  -W 768 -H 768 --steps 4

run_test "schnell_1024x1024_s4" "FLUX.1-schnell" "1024x1024" 4 \
  --diffusion-model "$M/flux/flux1-schnell-q4_k.gguf" \
  --vae "$M/flux/ae.safetensors" --clip_l "$M/flux/clip_l.safetensors" \
  --t5xxl "$M/flux/t5-v1_1-xxl-encoder-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu --vae-tiling \
  -W 1024 -H 1024 --steps 4

# =============================================================================
# FLUX.1-kontext-dev Q4_0 — image editing / inpainting
# =============================================================================
echo "" | tee -a "$LOG"
echo "=== FLUX.1-kontext-dev Q4_0 ===" | tee -a "$LOG"

run_test "kontext_512x512_s20" "FLUX.1-kontext-dev" "512x512" 20 \
  --diffusion-model "$M/flux/flux1-kontext-dev-Q4_0.gguf" \
  --vae "$M/flux/ae.safetensors" --clip_l "$M/flux/clip_l.safetensors" \
  --t5xxl "$M/flux/t5-v1_1-xxl-encoder-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu \
  -W 512 -H 512 --steps 20

run_test "kontext_768x768_s20" "FLUX.1-kontext-dev" "768x768" 20 \
  --diffusion-model "$M/flux/flux1-kontext-dev-Q4_0.gguf" \
  --vae "$M/flux/ae.safetensors" --clip_l "$M/flux/clip_l.safetensors" \
  --t5xxl "$M/flux/t5-v1_1-xxl-encoder-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu --vae-tiling \
  -W 768 -H 768 --steps 20

# =============================================================================
# Chroma flash Q4_0
# =============================================================================
echo "" | tee -a "$LOG"
echo "=== Chroma flash Q4_0 ===" | tee -a "$LOG"

run_test "chroma_512x512_s4" "Chroma" "512x512" 4 \
  --diffusion-model "$M/flux/chroma-unlocked-v47-flash-Q4_0.gguf" \
  --vae "$M/flux/ae.safetensors" \
  --t5xxl "$M/flux/t5-v1_1-xxl-encoder-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu \
  -W 512 -H 512 --steps 4

run_test "chroma_512x512_s8" "Chroma" "512x512" 8 \
  --diffusion-model "$M/flux/chroma-unlocked-v47-flash-Q4_0.gguf" \
  --vae "$M/flux/ae.safetensors" \
  --t5xxl "$M/flux/t5-v1_1-xxl-encoder-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu \
  -W 512 -H 512 --steps 8

run_test "chroma_768x768_s8" "Chroma" "768x768" 8 \
  --diffusion-model "$M/flux/chroma-unlocked-v47-flash-Q4_0.gguf" \
  --vae "$M/flux/ae.safetensors" \
  --t5xxl "$M/flux/t5-v1_1-xxl-encoder-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu --vae-tiling \
  -W 768 -H 768 --steps 8

# =============================================================================
# FLUX.1-dev Q4_K_S — high quality, slow
# =============================================================================
echo "" | tee -a "$LOG"
echo "=== FLUX.1-dev Q4_K_S ===" | tee -a "$LOG"

run_test "dev_512x512_s20" "FLUX.1-dev" "512x512" 20 \
  --diffusion-model "$M/flux/flux1-dev-Q4_K_S.gguf" \
  --vae "$M/flux/ae.safetensors" --clip_l "$M/flux/clip_l.safetensors" \
  --t5xxl "$M/flux/t5-v1_1-xxl-encoder-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu \
  -W 512 -H 512 --steps 20

run_test "dev_768x768_s20" "FLUX.1-dev" "768x768" 20 \
  --diffusion-model "$M/flux/flux1-dev-Q4_K_S.gguf" \
  --vae "$M/flux/ae.safetensors" --clip_l "$M/flux/clip_l.safetensors" \
  --t5xxl "$M/flux/t5-v1_1-xxl-encoder-Q4_K_M.gguf" \
  --cfg-scale 1.0 --sampling-method euler --clip-on-cpu --vae-tiling \
  -W 768 -H 768 --steps 20

# =============================================================================
# SD3.5-medium Q4_0
# =============================================================================
echo "" | tee -a "$LOG"
echo "=== SD3.5-medium Q4_0 ===" | tee -a "$LOG"

run_test "sd35med_512x512_s28" "SD3.5-medium" "512x512" 28 \
  --diffusion-model "$M/sd3/sd3.5_medium-q4_0.gguf" \
  --vae "$M/sd3/sd3_vae_f16.safetensors" \
  --clip_l "$M/flux/clip_l.safetensors" --clip_g "$M/sd3/clip_g.safetensors" \
  --t5xxl "$M/flux/t5-v1_1-xxl-encoder-Q4_K_M.gguf" \
  --cfg-scale 5.0 --sampling-method euler --clip-on-cpu \
  -W 512 -H 512 --steps 28

run_test "sd35med_768x768_s28" "SD3.5-medium" "768x768" 28 \
  --diffusion-model "$M/sd3/sd3.5_medium-q4_0.gguf" \
  --vae "$M/sd3/sd3_vae_f16.safetensors" \
  --clip_l "$M/flux/clip_l.safetensors" --clip_g "$M/sd3/clip_g.safetensors" \
  --t5xxl "$M/flux/t5-v1_1-xxl-encoder-Q4_K_M.gguf" \
  --cfg-scale 5.0 --sampling-method euler --clip-on-cpu --vae-tiling \
  -W 768 -H 768 --steps 28

run_test "sd35med_1024x1024_s28" "SD3.5-medium" "1024x1024" 28 \
  --diffusion-model "$M/sd3/sd3.5_medium-q4_0.gguf" \
  --vae "$M/sd3/sd3_vae_f16.safetensors" \
  --clip_l "$M/flux/clip_l.safetensors" --clip_g "$M/sd3/clip_g.safetensors" \
  --t5xxl "$M/flux/t5-v1_1-xxl-encoder-Q4_K_M.gguf" \
  --cfg-scale 5.0 --sampling-method euler --clip-on-cpu --vae-tiling \
  -W 1024 -H 1024 --steps 28

# =============================================================================
# SD-Turbo — fast baseline
# =============================================================================
echo "" | tee -a "$LOG"
echo "=== SD-Turbo ===" | tee -a "$LOG"

run_test "sdturbo_512x512_s1" "SD-Turbo" "512x512" 1 \
  -m "$M/sd-turbo.safetensors" --clip-on-cpu \
  -W 512 -H 512 --steps 1

run_test "sdturbo_512x512_s4" "SD-Turbo" "512x512" 4 \
  -m "$M/sd-turbo.safetensors" --clip-on-cpu \
  -W 512 -H 512 --steps 4

run_test "sdturbo_768x768_s4" "SD-Turbo" "768x768" 4 \
  -m "$M/sd-turbo.safetensors" --clip-on-cpu --vae-tiling \
  -W 768 -H 768 --steps 4

run_test "sdturbo_1024x1024_s4" "SD-Turbo" "1024x1024" 4 \
  -m "$M/sd-turbo.safetensors" --clip-on-cpu --vae-tiling \
  -W 1024 -H 1024 --steps 4

# =============================================================================
# WAN 2.1 T2V 1.3B — video generation
# =============================================================================
echo "" | tee -a "$LOG"
echo "=== WAN 2.1 T2V 1.3B ===" | tee -a "$LOG"

run_test "wan21_480x320_17f_s50" "WAN-2.1-T2V-1.3B" "480x320x17f" 50 \
  --diffusion-model "$M/wan/Wan2.1-T2V-1.3B-Q4_0.gguf" \
  --vae "$M/wan/wan_2.1_vae.safetensors" \
  --t5xxl "$M/wan/umt5-xxl-encoder-Q4_K_M.gguf" \
  --clip-on-cpu --offload-to-cpu \
  -W 480 -H 320 --steps 50 --video

# =============================================================================
# WAN 2.2 TI2V 5B — video gen (may OOM)
# =============================================================================
echo "" | tee -a "$LOG"
echo "=== WAN 2.2 TI2V 5B (expecting OOM) ===" | tee -a "$LOG"

run_test "wan22_480x320_17f_s30" "WAN-2.2-TI2V-5B" "480x320x17f" 30 \
  --diffusion-model "$M/wan/Wan2.2-TI2V-5B-Q4_0.gguf" \
  --vae "$M/wan/wan2.2_vae.safetensors" \
  --t5xxl "$M/wan/umt5-xxl-encoder-Q4_K_M.gguf" \
  --clip-on-cpu --offload-to-cpu \
  -W 480 -H 320 --steps 30 --video

# =============================================================================
# ESRGAN 4x upscale
# =============================================================================
echo "" | tee -a "$LOG"
echo "=== ESRGAN 4x Upscale ===" | tee -a "$LOG"

# Use a 512x512 test image from the first successful generation
ESRGAN_INPUT="$OUT/klein4b_512x512_s4.png"
if [ -f "$ESRGAN_INPUT" ]; then
    for TILE in 128 192 256; do
        run_test "esrgan_tile${TILE}" "ESRGAN-4x" "512->2048" 0 \
          --upscale-model "$M/esrgan/RealESRGAN_x4plus.pth" \
          --mode upscale --upscale-repeats 1 \
          --upscale-tile "$TILE" \
          -i "$ESRGAN_INPUT"
    done
else
    echo "No 512x512 test image for ESRGAN — skipping" | tee -a "$LOG"
fi

# =============================================================================
# Done — restart services
# =============================================================================
echo "" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "=== ALL IMAGE GEN BENCHMARKS DONE ===" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"

sudo systemctl start ollama
sudo systemctl start queue-runner
echo "Services restarted." | tee -a "$LOG"

echo ""
echo "Results log: $LOG"
echo "JSON results: $RESULTS"
echo "Images: $OUT/"
ls -lh "$OUT/" 2>/dev/null | tail -20
