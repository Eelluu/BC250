#!/usr/bin/env python3
"""Task 7: Benchmark candidate models at escalating context sizes.
Tests each model until failure, recording speed + memory.
"""
import json, time, subprocess, sys, os, urllib.request

OLLAMA = "http://localhost:11434"
RESULTS_DIR = "/tmp/bc250-ctx-bench"
os.makedirs(RESULTS_DIR, exist_ok=True)
sys.stdout.reconfigure(line_buffering=True)

PROMPT = "Explain quantum entanglement in detail, covering the EPR paradox, Bell's theorem, and practical applications in quantum computing and cryptography."

# Models and their context size sweeps (ascending — stop on 2 consecutive fails)
CANDIDATES = [
    ("gemma3:12b",              [4096, 8192, 16384, 24576, 32768, 49152, 65536]),
    ("qwen3.5:9b",              [4096, 16384, 32768, 49152, 65536, 81920, 98304]),
    ("phi4:14b",                [4096, 16384, 24576, 32768, 40960, 49152]),
    ("qwen3:14b",               [4096, 16384, 24576, 32768, 40960]),
    ("qwen3.5-27b-iq2m:latest", [4096, 8192, 16384, 24576, 32768]),
    ("mistral-nemo:12b",        [4096, 16384, 32768, 49152, 65536, 81920]),
]

def sysfs_read(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except:
        return -1

def gtt_used_gib():
    return round(sysfs_read("/sys/class/drm/card1/device/mem_info_gtt_used") / (1024**3), 3)

def free_mem_mb():
    try:
        out = subprocess.check_output(["free", "-m"], text=True)
        parts = out.strip().split("\n")[1].split()
        return int(parts[6]) if len(parts) > 6 else -1
    except:
        return -1

def ollama_api(endpoint, data=None, timeout=600):
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

def bench_one(model, num_ctx, timeout=600):
    """Restart ollama, load model at num_ctx, generate 100 tokens."""
    print(f"  {model} ctx={num_ctx//1024}K ...", end=" ", flush=True)

    subprocess.run(["sudo", "systemctl", "restart", "ollama"], timeout=30,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(12)

    pre_gtt = gtt_used_gib()
    t0 = time.time()
    resp = ollama_api("/api/generate", {
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": 100, "temperature": 0.7},
        "think": False,
        "keep_alive": "1m",
    }, timeout=timeout)
    elapsed = time.time() - t0
    post_gtt = gtt_used_gib()

    r = {
        "model": model,
        "num_ctx": num_ctx,
        "elapsed_s": round(elapsed, 1),
        "pre_gtt_gib": pre_gtt,
        "post_gtt_gib": post_gtt,
        "free_available_mb": free_mem_mb(),
    }

    if "error" in resp:
        r["status"] = "FAIL"
        r["error"] = resp["error"][:300]
        r["error_type"] = resp.get("error_type", "unknown")
        try:
            subprocess.check_output(["pgrep", "-f", "ollama serve"], timeout=3)
            r["ollama_alive"] = True
        except:
            r["ollama_alive"] = False
        print(f"FAIL ({elapsed:.0f}s): {r['error'][:80]}", flush=True)
    else:
        r["status"] = "OK"
        ed = resp.get("eval_duration", 0) / 1e9
        ec = resp.get("eval_count", 0)
        pd = resp.get("prompt_eval_duration", 0) / 1e9
        pc = resp.get("prompt_eval_count", 0)
        r["gen_tok_s"] = round(ec / ed, 1) if ed > 0 else 0
        r["prefill_tok_s"] = round(pc / pd, 1) if pd > 0 else 0
        r["load_s"] = round(resp.get("load_duration", 0) / 1e9, 1)
        r["eval_count"] = ec
        r["prompt_eval_count"] = pc

        ps = ollama_ps()
        if ps and "models" in ps:
            for m in ps["models"]:
                if model.split(":")[0] in m.get("name", ""):
                    sv = m.get("size_vram", 0)
                    st = m.get("size", 0)
                    r["size_vram_gib"] = round(sv / (1024**3), 3)
                    r["size_total_gib"] = round(st / (1024**3), 3)
                    r["gpu_pct"] = round(sv / st * 100, 1) if st > 0 else 0
                    break
        print(f"OK ({elapsed:.0f}s): gen={r['gen_tok_s']} tok/s, "
              f"pre={r['prefill_tok_s']} tok/s, "
              f"vram={r.get('size_vram_gib','?')}G, gtt={post_gtt}G", flush=True)
    return r


def main():
    print(f"{'='*70}")
    print(f"TASK 7 — Candidate Model Benchmarks")
    print(f"KV type: q4_0 (production config)")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    all_results = []

    for model, ctx_sizes in CANDIDATES:
        print(f"\n--- Model: {model} ---")
        # Check model exists
        check = ollama_api("/api/show", {"name": model})
        if "error" in check:
            print(f"  SKIP: model not available ({check['error'][:60]})")
            all_results.append({
                "model": model, "status": "SKIP",
                "error": f"Model not available: {check['error'][:100]}"
            })
            continue

        consecutive_fails = 0
        for ctx in ctx_sizes:
            r = bench_one(model, ctx)
            all_results.append(r)
            if r["status"] != "OK":
                consecutive_fails += 1
                if consecutive_fails >= 2:
                    print(f"  2 consecutive fails, stopping {model}")
                    break
            else:
                consecutive_fails = 0

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'model':<28} {'ctx':>6} {'status':<6} {'gen_t/s':>8} {'pre_t/s':>8} {'vram_G':>7} {'gtt_G':>7}")
    print("-" * 78)
    current_model = ""
    for r in all_results:
        m = r.get("model", "?")
        if m != current_model:
            if current_model:
                print()
            current_model = m
        ctx_k = f"{r.get('num_ctx', 0)//1024}K" if r.get('num_ctx') else "-"
        st = r.get("status", "?")
        gen = r.get("gen_tok_s", "-")
        pre = r.get("prefill_tok_s", "-")
        vram = r.get("size_vram_gib", "-")
        gtt = r.get("post_gtt_gib", "-")
        print(f"{m:<28} {ctx_k:>6} {st:<6} {gen:>8} {pre:>8} {vram:>7} {gtt:>7}")

    # Save
    path = os.path.join(RESULTS_DIR, "task7-candidates.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
