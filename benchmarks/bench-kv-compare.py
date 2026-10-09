#!/usr/bin/env python3
"""Task 3: KV cache type comparison — q4_0 vs q8_0 vs fp16.
Tests the MoE at multiple context sizes under each KV type.
Properly modifies override.conf preserving its format.
"""
import json, time, subprocess, sys, os, urllib.request

OLLAMA = "http://localhost:11434"
RESULTS_DIR = "/tmp/bc250-ctx-bench"
os.makedirs(RESULTS_DIR, exist_ok=True)
sys.stdout.reconfigure(line_buffering=True)

MODEL = "qwen3.5-35b-a3b-iq2m"
PROMPT = "Explain quantum entanglement in detail, covering the EPR paradox, Bell's theorem, and practical applications in quantum computing and cryptography."

# Context sizes: escalating to find ceiling for each KV type
CTX_SIZES = [4096, 8192, 16384, 24576, 32768, 49152, 65536, 81920, 98304, 131072]

OVERRIDE_TEMPLATE = """[Service]
Environment="OLLAMA_KEEP_ALIVE=30m"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_VULKAN=1"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_GPU_OVERHEAD=0"
Environment="OLLAMA_CONTEXT_LENGTH=16384"
Environment="OLLAMA_MAX_QUEUE=4"
OOMScoreAdjust=-1000
{kv_line}
"""

def sysfs_read(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except:
        return -1

def gtt_used_gib():
    return round(sysfs_read("/sys/class/drm/card1/device/mem_info_gtt_used") / (1024**3), 3)

def vram_used_gib():
    return round(sysfs_read("/sys/class/drm/card1/device/mem_info_vram_used") / (1024**3), 3)

def free_mem_mb():
    try:
        out = subprocess.check_output(["free", "-m"], text=True)
        parts = out.strip().split("\n")[1].split()
        return {"available_mb": int(parts[6]) if len(parts) > 6 else -1}
    except:
        return {}

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

def set_kv_type(kv_type):
    """Update override.conf and restart ollama with given KV type."""
    if kv_type == "fp16":
        kv_line = '# No KV cache type override (fp16 default)'
    else:
        kv_line = f'Environment="OLLAMA_KV_CACHE_TYPE={kv_type}"'

    override = OVERRIDE_TEMPLATE.format(kv_line=kv_line)
    with open("/tmp/_ollama_override.conf", "w") as f:
        f.write(override)

    subprocess.run("sudo cp /tmp/_ollama_override.conf /etc/systemd/system/ollama.service.d/override.conf",
                   shell=True, check=True)
    subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
    subprocess.run(["sudo", "systemctl", "restart", "ollama"], check=True)
    time.sleep(12)

    # Verify
    env = subprocess.check_output(
        "systemctl show ollama --property=Environment --no-pager", shell=True, text=True
    ).strip()
    print(f"  Ollama env: {env}", flush=True)

def bench_one(num_ctx, timeout=600):
    """Generate with instrumentation, return dict."""
    print(f"    ctx={num_ctx//1024}K ...", end=" ", flush=True)

    # Restart to clear state
    subprocess.run(["sudo", "systemctl", "restart", "ollama"], timeout=30)
    time.sleep(12)

    pre_gtt = gtt_used_gib()
    t0 = time.time()
    resp = ollama_api("/api/generate", {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": 100, "temperature": 0.7},
        "think": False,
        "keep_alive": "1m",
    }, timeout=timeout)
    elapsed = time.time() - t0

    post_gtt = gtt_used_gib()

    r = {
        "num_ctx": num_ctx,
        "elapsed_s": round(elapsed, 1),
        "pre_gtt_gib": pre_gtt,
        "post_gtt_gib": post_gtt,
        "free_mem": free_mem_mb(),
    }

    if "error" in resp:
        r["status"] = "FAIL"
        r["error"] = resp["error"][:300]
        r["error_type"] = resp.get("error_type", "unknown")
        # Check ollama alive
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

        ps = ollama_ps()
        if ps and "models" in ps:
            for m in ps["models"]:
                if MODEL in m.get("name", ""):
                    sv = m.get("size_vram", 0)
                    st = m.get("size", 0)
                    r["size_vram_gib"] = round(sv / (1024**3), 3)
                    r["gpu_pct"] = round(sv / st * 100, 1) if st > 0 else 0
                    break
        print(f"OK ({elapsed:.0f}s): gen={r['gen_tok_s']} tok/s, "
              f"pre={r['prefill_tok_s']} tok/s, "
              f"vram={r.get('size_vram_gib','?')}G, gtt={post_gtt}G", flush=True)
    return r


def main():
    print(f"{'='*70}")
    print(f"TASK 3 — KV Cache Type Comparison")
    print(f"Model: {MODEL}")
    print(f"Context sizes: {[c//1024 for c in CTX_SIZES]}K")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    all_results = {}

    for kv_type in ["q4_0", "q8_0", "fp16"]:
        print(f"\n{'='*50}")
        print(f"  KV TYPE: {kv_type}")
        print(f"{'='*50}")
        set_kv_type(kv_type)

        results = []
        consecutive_fails = 0
        for ctx in CTX_SIZES:
            r = bench_one(ctx)
            r["kv_type"] = kv_type
            results.append(r)
            if r["status"] != "OK":
                consecutive_fails += 1
                if consecutive_fails >= 2:
                    print(f"  2 consecutive fails at {ctx//1024}K, stopping {kv_type}")
                    break
            else:
                consecutive_fails = 0
        all_results[kv_type] = results

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'kv_type':<8} {'ctx':>6} {'status':<6} {'gen_tok/s':>10} {'pre_tok/s':>10} {'vram_gib':>9} {'gtt_gib':>8}")
    print("-" * 70)
    for kv_type in ["q4_0", "q8_0", "fp16"]:
        for r in all_results.get(kv_type, []):
            st = r["status"]
            gen = r.get("gen_tok_s", "-")
            pre_val = r.get("prefill_tok_s", "-")
            vram = r.get("size_vram_gib", "-")
            gtt = r.get("post_gtt_gib", "-")
            print(f"{kv_type:<8} {r['num_ctx']//1024:>5}K {st:<6} {gen:>10} {pre_val:>10} {vram:>9} {gtt:>8}")
        print()

    # Save
    flat = []
    for kv, rs in all_results.items():
        flat.extend(rs)
    path = os.path.join(RESULTS_DIR, "task3-kv-comparison.json")
    with open(path, "w") as f:
        json.dump(flat, f, indent=2)
    print(f"\nSaved to {path}")

    # IMPORTANT: Restore q4_0 (production config)
    print("\nRestoring production config (q4_0)...")
    set_kv_type("q4_0")
    print("Done.")


if __name__ == "__main__":
    main()
