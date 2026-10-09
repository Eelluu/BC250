#!/usr/bin/env python3
"""BC-250 Scientific Context Benchmark — March 2026

METHODOLOGY:
  Previous benchmarks set num_ctx=N with a tiny prompt — this only tests
  KV *allocation*, not actual context *utilization*. Ollama may silently
  cap or truncate.

  This benchmark:
  1. FILLS context with actual tokens (repeated text blocks)
  2. Verifies the model processes ALL tokens (checks prompt_eval_count)
  3. Measures generation speed at each fill level (not just allocation)
  4. Monitors system RAM + swap at each step
  5. Pushes until OOM, swap thrashing, or deadlock
  6. Records incremental JSON after each test

  For each (model, context_size):
    a) Build a prompt that is ~80% of context_size tokens
    b) Set num_ctx = context_size
    c) Generate 50 tokens
    d) Verify prompt_eval_count ≈ expected token count
    e) Record: gen tok/s, prefill tok/s, prompt_eval_count, system RAM, swap
    f) If swap > threshold or timeout → mark as ceiling

PREREQUISITES:
  - Run ON the BC-250 directly
  - queue-runner and signal services STOPPED
  - OLLAMA_CONTEXT_LENGTH set high (524288) or removed
  - OLLAMA_KV_CACHE_TYPE=q4_0 in systemd override

Usage:
  python3 bench-context-scientific.py [--models model1,model2] [--skip-stop]
"""

import json, time, subprocess, sys, os, datetime, argparse
import urllib.request

OLLAMA = "http://localhost:11434"
RESULTS_DIR = "/opt/netscan/tmp/bench-results"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")

# ─── Configuration ──────────────────────────────────────────────────────────

# Models to test (order: fastest → slowest for quick feedback)
PRIMARY_MODELS = [
    "qwen3.5-35b-a3b-iq2m:latest",   # MoE primary
    "qwen3.5:9b",                      # Dense primary (vision)
]

DENSE_MODELS = [
    "qwen2.5:3b",
    "phi4-mini:latest",
    "qwen3:8b",
    "qwen3:14b",
    "phi4:14b",
    "mistral-nemo:12b",
]

# Context sizes to test (tokens). We go until failure.
CTX_SIZES_MOE = [
    4096, 8192, 16384, 32768, 65536, 98304, 131072,
    163840, 196608, 229376, 262144, 327680, 393216, 524288,
]
CTX_SIZES_DENSE_SMALL = [  # ≤4B
    4096, 8192, 16384, 32768, 65536, 98304, 131072,
    163840, 196608, 262144,
]
CTX_SIZES_DENSE_MED = [    # 7-9B
    4096, 8192, 16384, 32768, 65536, 98304, 131072,
    163840, 196608,
]
CTX_SIZES_DENSE_LARGE = [  # 12-14B
    4096, 8192, 16384, 32768, 65536, 98304, 131072,
    163840,
]

# Swap threshold: if swap usage exceeds this during a test, mark as degraded
SWAP_WARN_MB = 1500   # warn
SWAP_FAIL_MB = 3000   # mark as ceiling (thrashing)

# Timeout per request
REQUEST_TIMEOUT = 1200  # 20 min — 96K fill takes ~580s, 128K needs more

# How much of the context window to fill with prompt tokens (~80%)
FILL_RATIO = 0.80

# ─── Text Block for Context Filling ────────────────────────────────────────
# ~500 tokens of diverse English text per block. We repeat it to fill context.
FILL_BLOCK = """The evolution of semiconductor manufacturing represents one of the most
remarkable engineering achievements in human history. From the first transistor
at Bell Labs in 1947 to modern 3nm process nodes, the industry has maintained
exponential scaling for over seven decades. Each generation of lithography
brought new challenges: optical diffraction limits led to immersion lithography,
then extreme ultraviolet (EUV) sources. The economics are equally staggering —
a modern fab costs $20 billion or more, yet produces chips at less than a cent
per transistor. Memory technologies evolved in parallel: from magnetic core to
SRAM, DRAM, and now 3D NAND flash with hundreds of layers. The interface between
processor and memory — the "memory wall" — remains the fundamental bottleneck
in computing performance. Bandwidth grows slower than compute, creating an
ever-widening gap that architects address through deeper cache hierarchies,
prefetching, and data-flow optimizations. On the software side, compilers have
become extraordinarily sophisticated, performing loop vectorization, automatic
parallelization, and profile-guided optimization. The interaction between
hardware and software design creates a co-evolution where each enables and
constrains the other. In artificial intelligence, this manifests as the
transformer architecture's quadratic attention mechanism — theoretically elegant
but practically bounded by memory bandwidth on real hardware. Quantization
techniques (INT8, FP16, mixed-precision) represent the latest chapter: trading
mathematical precision for throughput, enabled by hardware that natively supports
reduced-precision arithmetic. The BC-250's GFX1013 GPU, with its scalar Vulkan
compute path and 16GB unified memory, represents an interesting data point in
this landscape — capable inference without dedicated matrix cores, using the
same memory pool for weights, KV cache, and system operations.

"""

# ─── Helpers ────────────────────────────────────────────────────────────────

def api(endpoint, data=None, timeout=REQUEST_TIMEOUT):
    url = f"{OLLAMA}{endpoint}"
    req = urllib.request.Request(url, method="POST" if data else "GET")
    if data:
        req.add_header("Content-Type", "application/json")
        body = json.dumps(data).encode()
    else:
        body = None
    with urllib.request.urlopen(req, body, timeout=timeout) as resp:
        return json.loads(resp.read())

def unload_all():
    """Unload all models from GPU."""
    try:
        ps = api("/api/ps", timeout=10)
        if ps and "models" in ps:
            for m in ps["models"]:
                name = m.get("name", "")
                if name:
                    api("/api/generate", {"model": name, "keep_alive": 0}, timeout=30)
                    time.sleep(1)
    except Exception:
        pass
    time.sleep(3)

def get_system_memory():
    """Get system RAM and swap usage from /proc/meminfo."""
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    val_kb = int(parts[1])
                    info[key] = val_kb
        total_mb = info.get("MemTotal", 0) / 1024
        free_mb = info.get("MemAvailable", 0) / 1024
        used_mb = total_mb - free_mb
        swap_total_mb = info.get("SwapTotal", 0) / 1024
        swap_free_mb = info.get("SwapFree", 0) / 1024
        swap_used_mb = swap_total_mb - swap_free_mb
        return {
            "ram_total_mb": round(total_mb),
            "ram_used_mb": round(used_mb),
            "ram_free_mb": round(free_mb),
            "swap_total_mb": round(swap_total_mb),
            "swap_used_mb": round(swap_used_mb),
        }
    except Exception as e:
        return {"error": str(e)}

def get_vram_info(model):
    """Get VRAM usage from ollama ps."""
    try:
        ps = api("/api/ps", timeout=10)
        if ps and "models" in ps:
            for m in ps["models"]:
                if model in m.get("name", "") or model in m.get("model", ""):
                    sv = m.get("size_vram", 0)
                    st = m.get("size", 0)
                    gpu_pct = round(sv / st * 100) if st > 0 else -1
                    return {
                        "vram_bytes": sv,
                        "vram_gib": round(sv / (1024**3), 2),
                        "total_bytes": st,
                        "gpu_pct": gpu_pct,
                    }
    except Exception:
        pass
    return {}

def get_ollama_logs_kv(n=20):
    """Parse recent Ollama logs for KV cache allocation details."""
    try:
        result = subprocess.run(
            ["sudo", "journalctl", "-u", "ollama", "-n", str(n), "--no-pager"],
            capture_output=True, text=True, timeout=10
        )
        kv_info = {}
        for line in result.stdout.split("\n"):
            if "kv cache" in line.lower():
                # Extract: msg="kv cache" device=Vulkan0 size="2.2 GiB"
                if 'size="' in line:
                    size = line.split('size="')[1].split('"')[0]
                    kv_info["kv_cache_reported"] = size
            if "total memory" in line.lower():
                if 'size="' in line:
                    size = line.split('size="')[1].split('"')[0]
                    kv_info["total_memory_reported"] = size
            if "KvSize:" in line:
                for part in line.split():
                    if part.startswith("KvSize:"):
                        kv_info["kv_size_requested"] = int(part.split(":")[1])
            if "offloaded" in line.lower() and "layers" in line.lower():
                if "offloaded " in line:
                    # "offloaded 41/41 layers to GPU"
                    try:
                        seg = line.split("offloaded ")[1]
                        kv_info["layers_offloaded"] = seg.split(" ")[0]
                    except Exception:
                        pass
        return kv_info
    except Exception:
        return {}

def build_fill_prompt(target_tokens):
    """Build a prompt that is approximately target_tokens long.

    Strategy: repeat FILL_BLOCK (roughly 500 tokens per block).
    We overshoot slightly — Ollama will tokenize and we verify
    prompt_eval_count in the response.
    """
    # Rough estimate: 1 token ≈ 4 chars for English text
    chars_per_token = 3.8  # slightly conservative
    target_chars = int(target_tokens * chars_per_token)
    block_chars = len(FILL_BLOCK)
    repeats = max(1, target_chars // block_chars)

    # Build prompt with instruction at end (so model generates after reading all)
    prompt = FILL_BLOCK * repeats
    # Trim to approximate target
    prompt = prompt[:target_chars]
    # Add generation instruction at the end
    prompt += "\n\nBased on the above text, write a brief 2-sentence summary of the key themes discussed."

    return prompt

def run_single_test(model, ctx_size, fill_tokens):
    """Run one benchmark: load model at ctx_size, fill with fill_tokens, generate.

    Returns detailed result dict.
    """
    result = {
        "model": model,
        "num_ctx": ctx_size,
        "target_fill_tokens": fill_tokens,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    # Record pre-test memory
    result["mem_before"] = get_system_memory()

    # Build prompt
    prompt = build_fill_prompt(fill_tokens)
    result["prompt_chars"] = len(prompt)

    # Send request
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": ctx_size,
            "num_predict": 50,  # Just need a few tokens to measure speed
        },
        "keep_alive": "5m",
    }

    try:
        t0 = time.time()
        resp = api("/api/generate", data, timeout=REQUEST_TIMEOUT)
        wall = time.time() - t0

        if "error" in resp:
            result["status"] = "ERROR"
            result["error"] = resp["error"][:200]
            return result

        eval_count = resp.get("eval_count", 0)
        eval_dur = resp.get("eval_duration", 0) / 1e9
        prompt_count = resp.get("prompt_eval_count", 0)
        prompt_dur = resp.get("prompt_eval_duration", 0) / 1e9
        load_dur = resp.get("load_duration", 0) / 1e9

        gen_toks = eval_count / eval_dur if eval_dur > 0 else 0
        prefill_toks = prompt_count / prompt_dur if prompt_dur > 0 else 0

        result["status"] = "OK"
        result["gen_tok_s"] = round(gen_toks, 2)
        result["prefill_tok_s"] = round(prefill_toks, 2)
        result["eval_count"] = eval_count
        result["prompt_eval_count"] = prompt_count
        result["prompt_eval_duration_s"] = round(prompt_dur, 2)
        result["eval_duration_s"] = round(eval_dur, 2)
        result["load_duration_s"] = round(load_dur, 2)
        result["wall_s"] = round(wall, 1)
        result["ttft_s"] = round(load_dur + prompt_dur, 2)

        # Verification: did the model actually process our tokens?
        fill_ratio_actual = prompt_count / ctx_size if ctx_size > 0 else 0
        result["fill_ratio_actual"] = round(fill_ratio_actual, 3)
        result["fill_verified"] = prompt_count >= fill_tokens * 0.7  # Allow 30% margin for tokenizer differences

        # Context utilization: did Ollama silently cap?
        if fill_tokens > 1000 and prompt_count < fill_tokens * 0.5:
            result["context_truncated"] = True
            result["truncation_note"] = f"Expected ~{fill_tokens} tokens, got {prompt_count} — Ollama likely capped context"
        else:
            result["context_truncated"] = False

    except Exception as e:
        result["status"] = "TIMEOUT" if "timed out" in str(e).lower() else "FAIL"
        result["error"] = str(e)[:200]
        result["wall_s"] = round(time.time() - t0, 1)

    # Record post-test memory
    result["mem_after"] = get_system_memory()

    # Get VRAM info
    if result.get("status") == "OK":
        result["vram"] = get_vram_info(model)
        result["ollama_logs"] = get_ollama_logs_kv(30)

    # Check swap status
    swap_mb = result.get("mem_after", {}).get("swap_used_mb", 0)
    if swap_mb > SWAP_FAIL_MB:
        result["swap_status"] = "THRASHING"
    elif swap_mb > SWAP_WARN_MB:
        result["swap_status"] = "ELEVATED"
    else:
        result["swap_status"] = "OK"

    return result

def get_ctx_sizes_for_model(model):
    m = model.lower()
    if "35b" in m or "30b" in m:
        return CTX_SIZES_MOE
    if "27b" in m:
        return CTX_SIZES_MOE[:8]  # Don't push too hard
    if ":3b" in m or "2.5:3" in m or "phi4-mini" in m or ":4b" in m:
        return CTX_SIZES_DENSE_SMALL
    if ":14b" in m or "14b" in m or ":12b" in m or "12b" in m:
        return CTX_SIZES_DENSE_LARGE
    return CTX_SIZES_DENSE_MED  # 7-9B

def benchmark_model(model, results_file, prior_results=None):
    """Run complete context sweep for one model."""
    print(f"\n{'='*70}")
    print(f"  MODEL: {model}")
    print(f"  Time: {datetime.datetime.now().isoformat()}")
    print(f"{'='*70}")

    ctx_sizes = get_ctx_sizes_for_model(model)
    all_results = []
    ceiling_found = False

    for ctx in ctx_sizes:
        if ceiling_found:
            break

        fill_tokens = int(ctx * FILL_RATIO)
        print(f"\n  ctx={ctx:>7,} ({ctx//1024}K)  fill={fill_tokens:>7,} tokens")
        print(f"  {'─'*55}")

        # Unload and wait for memory to settle
        unload_all()
        time.sleep(2)

        # Phase 1: Allocation test (empty prompt, quick)
        print(f"    Phase 1: Allocation test (tiny prompt)...", end=" ", flush=True)
        alloc_result = run_single_test(model, ctx, 10)  # ~10 tokens
        alloc_result["phase"] = "allocation"

        if alloc_result["status"] not in ("OK",):
            print(f"FAIL: {alloc_result.get('error', '?')[:60]}")
            alloc_result["ceiling_reason"] = f"allocation_failed_at_{ctx}"
            all_results.append(alloc_result)
            ceiling_found = True
            continue

        print(f"OK  gen={alloc_result['gen_tok_s']} tok/s  "
              f"swap={alloc_result['mem_after'].get('swap_used_mb', '?')}MB")
        all_results.append(alloc_result)

        # Phase 2: Filled context test (actual utilization)
        print(f"    Phase 2: Fill test ({fill_tokens:,} tokens)...", end=" ", flush=True)
        fill_result = run_single_test(model, ctx, fill_tokens)
        fill_result["phase"] = "filled"

        if fill_result["status"] not in ("OK",):
            print(f"FAIL: {fill_result.get('error', '?')[:60]}")
            fill_result["ceiling_reason"] = f"fill_failed_at_{ctx}"
            all_results.append(fill_result)
            ceiling_found = True
            continue

        # Print detailed results
        verified = "✓" if fill_result.get("fill_verified") else "✗ TRUNCATED"
        truncated = " (TRUNCATED!)" if fill_result.get("context_truncated") else ""
        print(f"OK  gen={fill_result['gen_tok_s']} tok/s  "
              f"prefill={fill_result['prefill_tok_s']} tok/s  "
              f"tokens={fill_result['prompt_eval_count']}/{fill_tokens} {verified}{truncated}")
        print(f"           TTFT={fill_result['ttft_s']}s  wall={fill_result['wall_s']}s  "
              f"swap={fill_result['mem_after'].get('swap_used_mb','?')}MB  "
              f"ram_free={fill_result['mem_after'].get('ram_free_mb','?')}MB")

        if fill_result.get("context_truncated"):
            fill_result["ceiling_reason"] = f"context_truncated_at_{ctx}"
            print(f"    *** CONTEXT TRUNCATED: model only saw {fill_result['prompt_eval_count']} tokens at ctx={ctx}")

        # Check for swap thrashing
        if fill_result["swap_status"] == "THRASHING":
            fill_result["ceiling_reason"] = f"swap_thrashing_at_{ctx}"
            print(f"    *** SWAP THRASHING: {fill_result['mem_after'].get('swap_used_mb')}MB — stopping")
            all_results.append(fill_result)
            ceiling_found = True
            continue

        all_results.append(fill_result)

        # Save incrementally (include prior model results)
        save_results((prior_results or []) + all_results, results_file)

    # Final: unload model
    unload_all()

    return all_results

def save_results(results, filepath):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

def print_summary(all_results):
    """Print final summary table."""
    print(f"\n{'='*70}")
    print(f"  SUMMARY — Scientific Context Benchmark")
    print(f"{'='*70}\n")

    # Group by model
    models = {}
    for r in all_results:
        m = r["model"]
        if m not in models:
            models[m] = []
        models[m].append(r)

    print(f"{'Model':<35} {'Phase':<10} {'Ctx':>7} {'Fill':>7} {'Actual':>7} "
          f"{'Gen':>7} {'PF':>7} {'Swap':>6} {'Status'}")
    print("─" * 110)

    for model, results in models.items():
        for r in results:
            ctx = r.get("num_ctx", 0)
            ctx_str = f"{ctx//1024}K" if ctx >= 1024 else str(ctx)
            fill = r.get("target_fill_tokens", 0)
            actual = r.get("prompt_eval_count", 0)
            gen = r.get("gen_tok_s", 0)
            pf = r.get("prefill_tok_s", 0)
            swap = r.get("mem_after", {}).get("swap_used_mb", 0)
            status = r.get("status", "?")
            phase = r.get("phase", "?")
            trunc = " TRUNC" if r.get("context_truncated") else ""
            verified = " ✓" if r.get("fill_verified") else ""

            if phase == "allocation":
                fill_str = "alloc"
                actual_str = ""
            else:
                fill_str = f"{fill:,}"
                actual_str = f"{actual:,}"

            print(f"{model:<35} {phase:<10} {ctx_str:>7} {fill_str:>7} {actual_str:>7} "
                  f"{gen:>7.1f} {pf:>7.1f} {swap:>5}M {status}{trunc}{verified}")

def main():
    parser = argparse.ArgumentParser(description="BC-250 Scientific Context Benchmark")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names to test")
    parser.add_argument("--skip-stop", action="store_true",
                        help="Don't stop queue-runner (for testing)")
    parser.add_argument("--primary-only", action="store_true",
                        help="Only test primary models (MoE + 9B)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = f"{RESULTS_DIR}/bench_scientific_{TIMESTAMP}.json"

    print(f"BC-250 Scientific Context Benchmark")
    print(f"Results: {results_file}")
    print(f"Time: {datetime.datetime.now().isoformat()}")
    print(f"Fill ratio: {FILL_RATIO} (prompt fills {FILL_RATIO*100:.0f}% of context)")
    print(f"Swap thresholds: warn={SWAP_WARN_MB}MB fail={SWAP_FAIL_MB}MB")
    print()

    # Determine models
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    elif args.primary_only:
        models = PRIMARY_MODELS
    else:
        models = PRIMARY_MODELS + DENSE_MODELS

    print(f"Models to test: {len(models)}")
    for m in models:
        print(f"  - {m}")

    # Stop services
    if not args.skip_stop:
        print("\nStopping services for clean measurements...")
        subprocess.run(["sudo", "systemctl", "stop", "queue-runner"], capture_output=True, timeout=10)
        subprocess.run(["sudo", "systemctl", "stop", "signal-cli"], capture_output=True, timeout=10)
        time.sleep(3)
        print("Services stopped.")

    # Record initial system state
    print(f"\nInitial system memory: {get_system_memory()}")

    # Check Ollama config
    try:
        result = subprocess.run(
            ["sudo", "journalctl", "-u", "ollama", "-n", "5", "--no-pager"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split("\n"):
            if "OLLAMA_CONTEXT_LENGTH" in line:
                # Extract the value
                for part in line.split():
                    if "OLLAMA_CONTEXT_LENGTH" in part:
                        print(f"Ollama config: {part}")
    except Exception:
        pass

    all_results = []

    for model in models:
        model_results = benchmark_model(model, results_file, prior_results=all_results)
        all_results.extend(model_results)
        save_results(all_results, results_file)

    # Print summary
    print_summary(all_results)

    # Restart services
    if not args.skip_stop:
        print("\nRestarting services...")
        subprocess.run(["sudo", "systemctl", "start", "signal-cli"], capture_output=True, timeout=10)
        subprocess.run(["sudo", "systemctl", "start", "queue-runner"], capture_output=True, timeout=10)
        print("Services restarted.")

    print(f"\nResults saved to: {results_file}")
    print(f"Done at {datetime.datetime.now().isoformat()}")

if __name__ == "__main__":
    main()
