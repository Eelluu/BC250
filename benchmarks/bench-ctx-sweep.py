#!/usr/bin/env python3
"""BC-250 Long-Context Investigation — Master Benchmark
Captures full system instrumentation for every run.
Run on BC-250 directly.
"""
import json, time, subprocess, sys, os, urllib.request, traceback

OLLAMA = "http://localhost:11434"
RESULTS_DIR = "/tmp/bc250-ctx-bench"
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROMPT_SHORT = "Explain quantum entanglement in detail, covering the EPR paradox, Bell's theorem, and practical applications in quantum computing and cryptography."
PROMPT_MEDIUM = "Write a comprehensive technical analysis covering: 1) Memory management in modern operating systems including virtual memory, paging, TLB behavior, and NUMA effects. 2) How GPU unified memory architectures differ from discrete GPU memory models. 3) The implications for large language model inference on memory-constrained systems. Include specific examples from Linux kernel memory management." * 2

def sysfs_read(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except:
        return -1

def gtt_used_bytes():
    return sysfs_read("/sys/class/drm/card1/device/mem_info_gtt_used")

def gtt_total_bytes():
    return sysfs_read("/sys/class/drm/card1/device/mem_info_gtt_total")

def vram_used_bytes():
    return sysfs_read("/sys/class/drm/card1/device/mem_info_vram_used")

def vram_total_bytes():
    return sysfs_read("/sys/class/drm/card1/device/mem_info_vram_total")

def pages_limit():
    return sysfs_read("/sys/module/ttm/parameters/pages_limit")

def free_mem():
    try:
        out = subprocess.check_output(["free", "-m"], text=True)
        lines = out.strip().split("\n")
        mem = lines[1].split()
        swap = lines[2].split() if len(lines) > 2 else ["Swap:", "0", "0", "0"]
        return {
            "total_mb": int(mem[1]), "used_mb": int(mem[2]),
            "free_mb": int(mem[3]), "available_mb": int(mem[6]) if len(mem) > 6 else -1,
            "swap_total_mb": int(swap[1]), "swap_used_mb": int(swap[2]),
        }
    except:
        return {}

def buddyinfo():
    try:
        with open("/proc/buddyinfo") as f:
            return f.read().strip()
    except:
        return ""

def dmesg_gpu(n=20):
    try:
        out = subprocess.check_output(
            f"sudo dmesg | grep -iE 'ttm|amdgpu|oom|kill|gpu|vulkan|memory' | tail -{n}",
            shell=True, text=True, timeout=5
        )
        return out.strip().split("\n") if out.strip() else []
    except:
        return []

def journal_ollama(since="3 min ago"):
    try:
        return subprocess.check_output(
            ["journalctl", "-u", "ollama", "--since", since, "--no-pager", "-q"],
            text=True, timeout=10
        ).strip()
    except:
        return ""

def ollama_api(endpoint, data=None, timeout=300):
    url = f"{OLLAMA}{endpoint}"
    req = urllib.request.Request(url, method="POST" if data else "GET")
    if data:
        req.add_header("Content-Type", "application/json")
        body = json.dumps(data).encode()
    else:
        body = None
    try:
        with urllib.request.urlopen(req, body, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e), "error_type": type(e).__name__}

def ollama_ps():
    return ollama_api("/api/ps")

def unload_all():
    ps = ollama_ps()
    if ps and "models" in ps:
        for m in ps["models"]:
            name = m.get("name", "")
            if name:
                ollama_api("/api/generate", {"model": name, "keep_alive": 0})
                time.sleep(2)
    time.sleep(5)

def snapshot(label=""):
    return {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gtt_used_bytes": gtt_used_bytes(),
        "gtt_used_gib": round(gtt_used_bytes() / (1024**3), 3),
        "gtt_total_gib": round(gtt_total_bytes() / (1024**3), 3),
        "vram_used_bytes": vram_used_bytes(),
        "vram_used_gib": round(vram_used_bytes() / (1024**3), 3),
        "pages_limit": pages_limit(),
        "pages_limit_gib": round(pages_limit() * 4096 / (1024**3), 2),
        "free": free_mem(),
        "buddyinfo": buddyinfo(),
    }

def bench_one(model, num_ctx, prompt=None, timeout=600, restart_ollama_first=False):
    """Run one benchmark with full instrumentation."""
    if prompt is None:
        prompt = PROMPT_SHORT

    result = {
        "model": model,
        "num_ctx": num_ctx,
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if restart_ollama_first:
        print(f"  Restarting ollama...", end=" ", flush=True)
        subprocess.run(["sudo", "systemctl", "restart", "ollama"], timeout=30)
        time.sleep(12)
        print("done.", flush=True)
    else:
        unload_all()

    # Pre snapshot
    result["pre"] = snapshot("pre-load")

    # Generate
    print(f"  Generating (ctx={num_ctx//1024}K)...", end=" ", flush=True)
    t0 = time.time()
    resp = ollama_api("/api/generate", {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": 100, "temperature": 0.7},
        "think": False,
        "keep_alive": "1m",
    }, timeout=timeout)
    elapsed = time.time() - t0
    result["elapsed_s"] = round(elapsed, 1)

    # Post snapshot
    result["post"] = snapshot("post-generate")

    if "error" in resp:
        result["status"] = "FAIL"
        result["error"] = resp["error"][:500]
        result["error_type"] = resp.get("error_type", "unknown")
        result["journal"] = journal_ollama("5 min ago")[-2000:]
        result["dmesg_gpu"] = dmesg_gpu()
        # Check if ollama process is still alive
        try:
            subprocess.check_output(["pgrep", "-f", "ollama serve"], timeout=3)
            result["ollama_alive"] = True
        except:
            result["ollama_alive"] = False
        # Check if model still loaded
        ps = ollama_ps()
        result["model_still_loaded"] = bool(ps and "models" in ps and len(ps.get("models", [])) > 0)
        print(f"FAIL ({elapsed:.0f}s): {result['error'][:80]}", flush=True)
    else:
        result["status"] = "OK"
        ed = resp.get("eval_duration", 0) / 1e9
        ec = resp.get("eval_count", 0)
        pd = resp.get("prompt_eval_duration", 0) / 1e9
        pc = resp.get("prompt_eval_count", 0)
        ld = resp.get("load_duration", 0) / 1e9
        result["gen_tok_s"] = round(ec / ed, 1) if ed > 0 else 0
        result["prefill_tok_s"] = round(pc / pd, 1) if pd > 0 else 0
        result["load_s"] = round(ld, 1)
        result["eval_count"] = ec
        result["prompt_eval_count"] = pc

        # GPU info from api/ps
        ps = ollama_ps()
        if ps and "models" in ps:
            for m in ps["models"]:
                if model in m.get("name", ""):
                    sv = m.get("size_vram", 0)
                    st = m.get("size", 0)
                    result["size_vram_gib"] = round(sv / (1024**3), 3)
                    result["size_total_gib"] = round(st / (1024**3), 3)
                    result["gpu_pct"] = round(sv / st * 100, 1) if st > 0 else 0
                    result["context_length"] = m.get("context_length", 0)
                    break

        # Phantom overhead
        gtt = gtt_used_bytes() / (1024**3)
        sv = result.get("size_vram_gib", 0)
        if sv > 0:
            result["phantom_overhead_gib"] = round(gtt - sv, 3)

        print(f"OK ({elapsed:.0f}s): {result['gen_tok_s']} tok/s gen, "
              f"{result['prefill_tok_s']} tok/s pre, "
              f"GPU: {result.get('gpu_pct', '?')}%, "
              f"VRAM: {result.get('size_vram_gib', '?')} GiB, "
              f"GTT: {result['post']['gtt_used_gib']:.2f} GiB, "
              f"phantom: {result.get('phantom_overhead_gib', '?')} GiB",
              flush=True)
    return result

def save_results(results, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {path}")
    return path

def print_summary(results):
    print("\n" + "=" * 110)
    print(f"{'Model':<35} {'Ctx':>5} {'Status':>6} {'GenT/s':>7} {'PreT/s':>7} "
          f"{'GPU%':>5} {'VRAM':>6} {'GTT':>6} {'Phantom':>8} {'Time':>6}")
    print("-" * 110)
    for r in results:
        ctx_k = f"{r['num_ctx']//1024}K"
        if r["status"] == "OK":
            print(f"{r['model']:<35} {ctx_k:>5} {'OK':>6} "
                  f"{r.get('gen_tok_s', 0):>7.1f} {r.get('prefill_tok_s', 0):>7.1f} "
                  f"{r.get('gpu_pct', 0):>5.0f} {r.get('size_vram_gib', 0):>6.2f} "
                  f"{r['post']['gtt_used_gib']:>6.2f} "
                  f"{r.get('phantom_overhead_gib', 0):>8.3f} "
                  f"{r['elapsed_s']:>6.1f}")
        else:
            print(f"{r['model']:<35} {ctx_k:>5} {'FAIL':>6} "
                  f"{'—':>7} {'—':>7} {'—':>5} {'—':>6} "
                  f"{r['post']['gtt_used_gib']:>6.2f} {'—':>8} "
                  f"{r['elapsed_s']:>6.1f}")

# ═══════════════════════════════════════════════════════════
# TASK RUNNERS
# ═══════════════════════════════════════════════════════════

def task0_pages_limit_baseline():
    """Baseline MoE context sweep with CURRENT pages_limit (before fix)."""
    print("=" * 70)
    print("TASK 0a — Baseline MoE context sweep (pages_limit = current)")
    print(f"  pages_limit = {pages_limit()} ({pages_limit()*4096/(1024**3):.1f} GiB)")
    print("=" * 70)

    model = "qwen3.5-35b-a3b-iq2m"
    results = []
    for ctx in [4096, 8192, 16384, 20480, 24576, 28672, 32768]:
        r = bench_one(model, ctx, restart_ollama_first=True)
        results.append(r)
        if r["status"] != "OK":
            # Try one more larger size to see if it's a cliff or gradual
            if ctx < 32768:
                print(f"  Failed at {ctx//1024}K, trying next size for cliff detection...")
                continue
            break
    print_summary(results)
    return save_results(results, "task0a-baseline-moe.json")

def task0_pages_limit_fix_and_retest():
    """Fix pages_limit to 4194304 and re-test MoE."""
    print("\n" + "=" * 70)
    print("TASK 0b — Fix pages_limit to 4194304 (16 GiB) and re-test")
    print("=" * 70)

    # Fix pages_limit
    print("Fixing pages_limit...", flush=True)
    subprocess.run("echo 4194304 | sudo tee /sys/module/ttm/parameters/pages_limit",
                    shell=True, check=True)
    subprocess.run("echo 4194304 | sudo tee /sys/module/ttm/parameters/page_pool_size",
                    shell=True, check=True)
    time.sleep(2)
    new_pl = pages_limit()
    print(f"  pages_limit now = {new_pl} ({new_pl*4096/(1024**3):.1f} GiB)")
    print(f"  GTT total = {gtt_total_bytes()/(1024**3):.1f} GiB")

    model = "qwen3.5-35b-a3b-iq2m"
    results = []
    for ctx in [4096, 8192, 16384, 20480, 24576, 28672, 32768, 40960]:
        r = bench_one(model, ctx, restart_ollama_first=True)
        results.append(r)
        if r["status"] != "OK":
            if ctx < 40960:
                continue
            break
    print_summary(results)
    return save_results(results, "task0b-fixed-pages-moe.json")

def task1_failure_mode():
    """Detailed failure mode classification for MoE."""
    print("\n" + "=" * 70)
    print("TASK 1 — Failure mode classification (MoE, fine-grained context sweep)")
    print(f"  pages_limit = {pages_limit()} ({pages_limit()*4096/(1024**3):.1f} GiB)")
    print("=" * 70)

    model = "qwen3.5-35b-a3b-iq2m"
    results = []
    # Fine-grained sweep around the expected ceiling
    for ctx in [16384, 18432, 20480, 22528, 24576, 26624, 28672]:
        r = bench_one(model, ctx, restart_ollama_first=True, timeout=600)
        results.append(r)
    print_summary(results)
    return save_results(results, "task1-failmode-moe.json")

def task2_memory_budget():
    """Per-component memory budget at each context size."""
    print("\n" + "=" * 70)
    print("TASK 2 — Memory budget per-component (MoE)")
    print("=" * 70)

    model = "qwen3.5-35b-a3b-iq2m"
    results = []
    for ctx in [4096, 8192, 12288, 16384, 20480, 24576]:
        r = bench_one(model, ctx, restart_ollama_first=True)
        results.append(r)
        if r["status"] != "OK":
            continue
    print_summary(results)

    # Print memory breakdown
    print("\n--- MEMORY BREAKDOWN ---")
    print(f"{'Ctx':>6} {'OllamaVRAM':>11} {'OllamaTotal':>12} {'GTT':>8} {'Phantom':>9} "
          f"{'SysFree':>8} {'SwapUsed':>9} {'Available':>10}")
    print("-" * 90)
    for r in results:
        if r["status"] == "OK":
            ctx_k = f"{r['num_ctx']//1024}K"
            print(f"{ctx_k:>6} {r.get('size_vram_gib',0):>10.3f}G "
                  f"{r.get('size_total_gib',0):>11.3f}G "
                  f"{r['post']['gtt_used_gib']:>7.3f}G "
                  f"{r.get('phantom_overhead_gib',0):>8.3f}G "
                  f"{r['post']['free']['free_mb']:>7}M "
                  f"{r['post']['free']['swap_used_mb']:>8}M "
                  f"{r['post']['free']['available_mb']:>9}M")
    return save_results(results, "task2-memory-budget.json")

def task3_kv_type_comparison():
    """Compare KV cache types: q4_0 (current), q8_0, fp16."""
    print("\n" + "=" * 70)
    print("TASK 3 — KV cache type comparison (MoE)")
    print("NOTE: This requires restarting ollama with different env vars!")
    print("=" * 70)

    model = "qwen3.5-35b-a3b-iq2m"
    ctx_sizes = [16384, 20480, 24576, 32768]
    all_results = {}

    for kv_type in ["q4_0", "q8_0", ""]:  # "" = fp16 default
        kv_label = kv_type if kv_type else "fp16"
        print(f"\n--- Testing KV type: {kv_label} ---")

        # Update ollama env
        if kv_type:
            env_line = (f"Environment=PATH=/home/user/.local/bin:/home/user/bin:"
                       f"/usr/local/bin:/usr/bin "
                       f"OLLAMA_KEEP_ALIVE=30m OLLAMA_MAX_LOADED_MODELS=1 "
                       f"OLLAMA_VULKAN=1 OLLAMA_FLASH_ATTENTION=1 "
                       f"OLLAMA_GPU_OVERHEAD=0 OLLAMA_CONTEXT_LENGTH=16384 "
                       f"OLLAMA_MAX_QUEUE=4 OLLAMA_KV_CACHE_TYPE={kv_type}")
        else:
            env_line = (f"Environment=PATH=/home/user/.local/bin:/home/user/bin:"
                       f"/usr/local/bin:/usr/bin "
                       f"OLLAMA_KEEP_ALIVE=30m OLLAMA_MAX_LOADED_MODELS=1 "
                       f"OLLAMA_VULKAN=1 OLLAMA_FLASH_ATTENTION=1 "
                       f"OLLAMA_GPU_OVERHEAD=0 OLLAMA_CONTEXT_LENGTH=16384 "
                       f"OLLAMA_MAX_QUEUE=4")

        # Write temp override
        override = f"""[Service]
{env_line}
OOMScoreAdjust=-1000
"""
        with open("/tmp/_ollama_override.conf", "w") as f:
            f.write(override)
        subprocess.run("sudo cp /tmp/_ollama_override.conf /etc/systemd/system/ollama.service.d/override.conf",
                       shell=True, check=True)
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
        subprocess.run(["sudo", "systemctl", "restart", "ollama"], check=True)
        time.sleep(12)

        # Verify env
        env_check = subprocess.check_output(
            "systemctl show ollama --property=Environment --no-pager", shell=True, text=True
        ).strip()
        print(f"  Ollama env: ...{env_check[-80:]}")

        results = []
        for ctx in ctx_sizes:
            r = bench_one(model, ctx, restart_ollama_first=True)
            r["kv_type"] = kv_label
            results.append(r)
        all_results[kv_label] = results
        print_summary(results)

    # Save combined
    flat = []
    for kv, rs in all_results.items():
        flat.extend(rs)
    return save_results(flat, "task3-kv-comparison.json")

def task7_candidate_models():
    """Benchmark alternative models."""
    print("\n" + "=" * 70)
    print("TASK 7 — Alternative model benchmarks")
    print("=" * 70)

    candidates = [
        ("gemma3:12b", [4096, 8192, 16384, 24576, 32768, 49152, 65536]),
        ("qwen3.5:9b", [4096, 16384, 32768, 49152, 65536]),
        ("qwen3.5-35b-a3b-iq2m", [4096, 16384, 24576, 32768]),
        ("qwen3.5-27b-iq2m", [4096, 8192]),
        ("phi4:14b", [4096, 16384, 24576, 32768, 40960]),
    ]

    all_results = []
    for model, ctx_sizes in candidates:
        print(f"\n--- Model: {model} ---")
        for ctx in ctx_sizes:
            r = bench_one(model, ctx, restart_ollama_first=True)
            all_results.append(r)
            if r["status"] != "OK":
                print(f"  Failed at {ctx//1024}K, skipping larger for {model}")
                break
    print_summary(all_results)
    return save_results(all_results, "task7-candidates.json")

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BC-250 Context Investigation")
    parser.add_argument("task", choices=["0a", "0b", "1", "2", "3", "7", "all"],
                        help="Task to run: 0a=baseline, 0b=fix pages_limit, 1=failmode, "
                             "2=memory, 3=kv-types, 7=candidates, all=0a+0b+1+2")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"BC-250 Long-Context Investigation — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"pages_limit = {pages_limit()} ({pages_limit()*4096/(1024**3):.1f} GiB)")
    print(f"GTT total = {gtt_total_bytes()/(1024**3):.1f} GiB")
    print(f"{'='*70}\n")

    if args.task == "0a":
        task0_pages_limit_baseline()
    elif args.task == "0b":
        task0_pages_limit_fix_and_retest()
    elif args.task == "1":
        task1_failure_mode()
    elif args.task == "2":
        task2_memory_budget()
    elif args.task == "3":
        task3_kv_type_comparison()
    elif args.task == "7":
        task7_candidate_models()
    elif args.task == "all":
        task0_pages_limit_baseline()
        task0_pages_limit_fix_and_retest()
        task1_failure_mode()
        task2_memory_budget()
