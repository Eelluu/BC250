#!/usr/bin/env python3
"""BC-250 Comprehensive LLM Benchmark — July 2026

Tests ALL models with Q4_0 KV cache at multiple context sizes.
Measures: generation tok/s, prefill tok/s, stable context ceiling, VRAM.

Designed to run directly on BC-250. Stops queue-runner for clean measurements.

Usage: python3 bench-comprehensive.py [--skip-unload] [--models model1,model2]
"""

import json, time, subprocess, sys, os, signal, argparse, datetime

OLLAMA = "http://localhost:11434"
RESULTS_DIR = "/opt/netscan/tmp/bench-results"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")

# ALL models on this BC-250
ALL_MODELS = [
    # Small (≤3B)
    "qwen2.5:3b",
    # Medium (7-9B)
    "qwen2.5:7b",
    "qwen2.5-coder:7b",
    "qwen3:8b",
    "qwen3:8b-nothink",
    "huihui_ai/qwen3-abliterated:8b",
    "qwen3-abl-nothink:latest",
    "llama3.1:8b",
    "mannix/llama3.1-8b-lexi:latest",
    "huihui_ai/seed-coder-abliterate:latest",
    # Large Dense (9-14B)
    "gemma2:9b",
    "qwen3.5:9b",
    "gemma3:12b",
    "mistral-nemo:12b",
    "qwen3:14b",
    "qwen3-14b-16k:latest",
    "qwen3-14b-abl-nothink:latest",
    "huihui_ai/qwen3-abliterated:14b",
    "phi4:14b",
    # MoE / XL
    "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q2_K",
    "qwen3.5-35b-a3b-iq2m:latest",
    "qwen3.5-27b-iq2m:latest",
]

# Context sizes by model category
CTX_TINY   = [4096, 8192, 16384, 32768, 65536, 131072]          # ≤3B
CTX_SMALL  = [4096, 8192, 16384, 32768, 65536, 131072]          # 7-8B
CTX_MED    = [4096, 8192, 16384, 32768, 65536, 131072]          # 9-12B
CTX_LARGE  = [4096, 8192, 16384, 32768, 49152, 65536, 131072]   # 14B
CTX_XL     = [4096, 16384, 32768, 65536, 131072, 262144]        # MoE/30B+

# Prefill prompts of increasing size
PREFILL_PROMPTS = {
    "tiny": "Hello",
    "short": "Explain quantum computing in one paragraph.",
    "medium": "Write a detailed technical analysis of memory management in modern operating systems, covering virtual memory, paging, segmentation, and garbage collection. Include comparisons between Linux, Windows, and macOS approaches. Discuss the trade-offs between different allocation strategies and how they impact performance in server workloads versus desktop applications. Cover topics including buddy allocators, slab allocators, NUMA-aware allocation, and transparent huge pages.",
    "long": """Write an exhaustive technical report on the evolution of CPU architecture from the early CISC designs of the 1970s through modern heterogeneous computing platforms. Cover the following topics in detail:

1. The transition from CISC to RISC architectures, including the key innovations of the Berkeley RISC project and Stanford MIPS project. Explain how instruction set simplification enabled pipelining and higher clock frequencies.

2. The superscalar revolution of the 1990s, including out-of-order execution, register renaming, branch prediction, and speculative execution. Discuss the Pentium Pro, MIPS R10000, and Alpha 21264 as case studies.

3. The memory wall problem and the evolution of cache hierarchies from simple direct-mapped L1 caches to modern multi-level inclusive/exclusive cache designs with prefetching and non-blocking behavior.

4. The shift to multi-core processing, driven by power wall limitations. Cover Dennard scaling breakdown, the end of frequency scaling, Amdahl's law implications, and the challenges of parallel programming.

5. SIMD extensions from MMX through SSE, AVX, AVX-512, and ARM SVE. Explain auto-vectorization challenges and the programmer's role in achieving peak SIMD throughput.

6. GPU computing and the GPGPU revolution, from fixed-function graphics pipelines to general-purpose CUDA and OpenCL. Discuss the architectural differences between GPU and CPU designs.

7. Modern heterogeneous architectures including Apple Silicon (unified memory), AMD APUs (like the BC-250's Cyan Skillfish with shared GDDR6), and Intel's hybrid P-core/E-core designs. Discuss the implications of unified memory architectures for AI inference workloads.

8. The emerging role of dedicated AI accelerators (NPUs, TPUs) and their integration into mainstream processor designs. Include discussion of matrix multiplication hardware, reduced precision formats (INT8, FP16, BF16), and sparsity support.

For each era, provide specific benchmark comparisons, transistor counts, process nodes, and power consumption figures where available.""",
}

import urllib.request

def api(endpoint, data=None, timeout=600):
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
    try:
        ps = api("/api/ps")
        if ps and "models" in ps:
            for m in ps["models"]:
                name = m.get("name", "")
                if name:
                    api("/api/generate", {"model": name, "keep_alive": 0})
                    time.sleep(1)
    except:
        pass
    time.sleep(3)

def get_model_size_category(model):
    m = model.lower()
    if ":3b" in m or "2.5:3" in m:
        return "tiny"
    if "35b" in m or "30b" in m or "27b" in m:
        return "xl"
    if ":14b" in m or "14b" in m:
        return "large"
    if ":12b" in m or "12b" in m or ":9b" in m or "9b" in m:
        return "med"
    return "small"  # 7-8B

def get_ctx_sizes(model):
    cat = get_model_size_category(model)
    return {"tiny": CTX_TINY, "small": CTX_SMALL, "med": CTX_MED,
            "large": CTX_LARGE, "xl": CTX_XL}[cat]

def benchmark_generate(model, prompt, num_ctx, num_predict=100, timeout=600):
    """Single generation benchmark. Returns dict with metrics or error."""
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": num_predict},
        "keep_alive": "5m",
    }
    try:
        t0 = time.time()
        resp = api("/api/generate", data, timeout=timeout)
        wall = time.time() - t0

        if "error" in resp:
            return {"status": "FAIL", "error": resp["error"][:120]}

        eval_count = resp.get("eval_count", 0)
        eval_dur = resp.get("eval_duration", 0) / 1e9
        prompt_count = resp.get("prompt_eval_count", 0)
        prompt_dur = resp.get("prompt_eval_duration", 0) / 1e9
        load_dur = resp.get("load_duration", 0) / 1e9

        gen_toks = eval_count / eval_dur if eval_dur > 0 else 0
        prefill_toks = prompt_count / prompt_dur if prompt_dur > 0 else 0

        return {
            "status": "OK",
            "gen_tok_s": round(gen_toks, 1),
            "prefill_tok_s": round(prefill_toks, 1),
            "eval_count": eval_count,
            "prompt_count": prompt_count,
            "load_s": round(load_dur, 1),
            "wall_s": round(wall, 1),
        }
    except Exception as e:
        return {"status": "FAIL", "error": str(e)[:120]}

def get_vram_info(model):
    """Get VRAM usage from ollama ps."""
    try:
        ps = api("/api/ps")
        if ps and "models" in ps:
            for m in ps["models"]:
                if model in m.get("name", "") or model in m.get("model", ""):
                    sv = m.get("size_vram", 0)
                    st = m.get("size", 0)
                    gpu_pct = f"{sv/st*100:.0f}%" if st > 0 else "?"
                    vram_gib = sv / (1024**3)
                    return {"vram_gib": round(vram_gib, 1), "gpu_pct": gpu_pct}
    except:
        pass
    return {"vram_gib": -1, "gpu_pct": "?"}

def find_context_ceiling(model):
    """Binary-search style context ceiling finder.
    Returns (max_working_ctx, results_list)."""
    sizes = get_ctx_sizes(model)
    results = []
    max_working = 0
    prompt = PREFILL_PROMPTS["short"]

    print(f"    Context ceiling search: {sizes}")
    for ctx in sizes:
        unload_all()
        print(f"    Testing ctx={ctx}...", end=" ", flush=True)
        r = benchmark_generate(model, prompt, ctx, num_predict=20, timeout=300)
        r["num_ctx"] = ctx
        results.append(r)

        if r["status"] == "OK":
            max_working = ctx
            vinfo = get_vram_info(model)
            r.update(vinfo)
            print(f"OK  gen={r['gen_tok_s']} tok/s  vram={r.get('vram_gib', '?')}G")
        else:
            print(f"FAIL: {r.get('error', '?')[:60]}")
            # No point testing higher contexts
            break

    return max_working, results

def run_prefill_benchmark(model, num_ctx):
    """Test prefill speed at various prompt sizes."""
    results = []
    for label, prompt in PREFILL_PROMPTS.items():
        # Model should already be loaded
        r = benchmark_generate(model, prompt, num_ctx, num_predict=50, timeout=300)
        r["prompt_label"] = label
        r["num_ctx"] = num_ctx
        results.append(r)
        if r["status"] == "OK":
            print(f"    Prefill {label:8s}: prefill={r['prefill_tok_s']:>6.1f} tok/s  "
                  f"gen={r['gen_tok_s']:>5.1f} tok/s  prompt_tokens={r['prompt_count']}")
        else:
            print(f"    Prefill {label:8s}: FAIL {r.get('error', '')[:50]}")
    return results

def run_full_model_benchmark(model):
    """Complete benchmark for one model."""
    print(f"\n{'='*70}")
    print(f"MODEL: {model}")
    print(f"{'='*70}")

    result = {
        "model": model,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    # Phase 1: Quick speed test at 4K context (headline numbers)
    print("  Phase 1: Speed test @4K context")
    unload_all()
    r = benchmark_generate(model, PREFILL_PROMPTS["medium"], 4096, num_predict=100, timeout=300)
    if r["status"] != "OK":
        print(f"  *** Model failed at 4K: {r.get('error', '?')[:80]}")
        result["status"] = "FAIL"
        result["error"] = r.get("error", "?")
        return result

    vinfo = get_vram_info(model)
    result["speed_4k"] = r
    result["speed_4k"].update(vinfo)
    print(f"  gen={r['gen_tok_s']} tok/s  prefill={r['prefill_tok_s']} tok/s  "
          f"vram={vinfo['vram_gib']}G  gpu={vinfo['gpu_pct']}")

    # Phase 2: Context ceiling search
    print("  Phase 2: Context ceiling")
    max_ctx, ctx_results = find_context_ceiling(model)
    result["max_ctx"] = max_ctx
    result["ctx_results"] = ctx_results
    print(f"  → Max stable context: {max_ctx}")

    # Phase 3: Prefill benchmark at optimal context
    prefill_ctx = min(max_ctx, 16384)  # Use reasonable context for prefill test
    print(f"  Phase 3: Prefill benchmark @{prefill_ctx} context")
    unload_all()
    # Warm up model first
    benchmark_generate(model, "warmup", prefill_ctx, num_predict=5, timeout=120)
    prefill_results = run_prefill_benchmark(model, prefill_ctx)
    result["prefill"] = prefill_results

    result["status"] = "OK"
    return result

def stop_queue_runner():
    print("Stopping queue-runner for clean measurements...")
    subprocess.run(["sudo", "systemctl", "stop", "queue-runner"], capture_output=True, timeout=10)
    time.sleep(2)

def start_queue_runner():
    print("Restarting queue-runner...")
    subprocess.run(["sudo", "systemctl", "start", "queue-runner"], capture_output=True, timeout=10)

def format_results_table(results):
    """Format results as a README-compatible markdown table."""
    lines = []
    lines.append("")
    lines.append("## Comprehensive LLM Benchmark Results")
    lines.append(f"*Measured {TIMESTAMP} · Ollama 0.18.0 · Vulkan · Q4_0 KV cache · BC-250 GFX1013*")
    lines.append("")

    # Main compatibility table
    lines.append("### Model Compatibility Table (Q4_0 KV Cache)")
    lines.append("")
    lines.append("| Model | tok/s | Prefill | Max Ctx | VRAM @4K | GPU% | Status |")
    lines.append("|-------|:-----:|:-------:|:-------:|:--------:|:----:|--------|")

    for r in results:
        if r.get("status") != "OK":
            lines.append(f"| {r['model']} | — | — | — | — | — | ❌ {r.get('error', 'Failed')[:40]} |")
            continue

        s = r["speed_4k"]
        max_ctx = r["max_ctx"]
        ctx_str = f"{max_ctx // 1024}K" if max_ctx >= 1024 else str(max_ctx)
        lines.append(
            f"| {r['model']} | **{s['gen_tok_s']}** | **{s['prefill_tok_s']}** | "
            f"**{ctx_str}** | {s.get('vram_gib', '?')} GiB | {s.get('gpu_pct', '?')} | ✅ |"
        )

    lines.append("")

    # Context ceiling grid
    lines.append("### Context Ceiling Grid (Q4_0 KV)")
    lines.append("")
    ctx_headers = ["4K", "8K", "16K", "32K", "64K", "128K", "256K"]
    ctx_values = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
    header = "| Model | " + " | ".join(ctx_headers) + " |"
    sep = "|-------|" + "|".join([":---:" for _ in ctx_headers]) + "|"
    lines.append(header)
    lines.append(sep)

    for r in results:
        if r.get("status") != "OK":
            continue
        ctx_results = {cr["num_ctx"]: cr["status"] for cr in r.get("ctx_results", [])}
        cells = []
        for cv in ctx_values:
            status = ctx_results.get(cv, "—")
            if status == "OK":
                cells.append("✅")
            elif status == "FAIL":
                cells.append("❌")
            else:
                cells.append("—")
        lines.append(f"| {r['model']} | " + " | ".join(cells) + " |")

    lines.append("")

    # Prefill details for each model
    lines.append("### Prefill Benchmarks (per model)")
    lines.append("")
    for r in results:
        if r.get("status") != "OK" or not r.get("prefill"):
            continue
        lines.append(f"**{r['model']}:**")
        lines.append("")
        lines.append("| Prompt | Tokens | Prefill tok/s | Gen tok/s |")
        lines.append("|--------|:------:|:-------------:|:---------:|")
        for p in r["prefill"]:
            if p["status"] == "OK":
                lines.append(
                    f"| {p['prompt_label']} | {p['prompt_count']} | "
                    f"{p['prefill_tok_s']} | {p['gen_tok_s']} |"
                )
        lines.append("")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="BC-250 Comprehensive LLM Benchmark")
    parser.add_argument("--models", type=str, help="Comma-separated model list (default: all)")
    parser.add_argument("--skip-stop", action="store_true", help="Don't stop queue-runner")
    args = parser.parse_args()

    models = args.models.split(",") if args.models else ALL_MODELS

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print(f"BC-250 Comprehensive LLM Benchmark — {TIMESTAMP}")
    print(f"Testing {len(models)} models with Q4_0 KV cache")
    print(f"Results: {RESULTS_DIR}/")
    print("=" * 70)

    if not args.skip_stop:
        stop_queue_runner()

    all_results = []
    for i, model in enumerate(models):
        print(f"\n[{i+1}/{len(models)}]", end="")
        try:
            result = run_full_model_benchmark(model)
            all_results.append(result)

            # Save incrementally
            with open(f"{RESULTS_DIR}/bench_{TIMESTAMP}.json", "w") as f:
                json.dump(all_results, f, indent=2)

        except Exception as e:
            print(f"  *** EXCEPTION: {e}")
            all_results.append({"model": model, "status": "ERROR", "error": str(e)[:200]})

    # Generate markdown report
    report = format_results_table(all_results)
    report_path = f"{RESULTS_DIR}/bench_{TIMESTAMP}.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nMarkdown report: {report_path}")

    if not args.skip_stop:
        start_queue_runner()

    # Print summary
    ok = sum(1 for r in all_results if r.get("status") == "OK")
    fail = len(all_results) - ok
    print(f"\n{'='*70}")
    print(f"DONE: {ok} OK, {fail} FAIL out of {len(all_results)} models")
    print(f"JSON: {RESULTS_DIR}/bench_{TIMESTAMP}.json")
    print(f"Report: {report_path}")

if __name__ == "__main__":
    main()
