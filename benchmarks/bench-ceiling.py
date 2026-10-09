#!/usr/bin/env python3
"""Push beyond 128K — find the actual ceiling for the MoE with Q4_0 KV.
Tests: 128K, 160K, 192K, 224K, 256K, 320K, 384K, 448K, 512K
Also tests at the model's native rope limit boundaries.

IMPORTANT: Stop queue-runner before running to avoid interference:
  sudo systemctl stop queue-runner
"""
import json, time, subprocess, sys, os, urllib.request

OLLAMA = "http://localhost:11434"
RESULTS_DIR = "/tmp/bc250-ctx-bench"
os.makedirs(RESULTS_DIR, exist_ok=True)
sys.stdout.reconfigure(line_buffering=True)

MODEL = "qwen3.5-35b-a3b-iq2m"
PROMPT = "Explain quantum entanglement in detail, covering the EPR paradox, Bell's theorem, and practical applications in quantum computing and cryptography."

# Escalating context sizes beyond 128K
CTX_SIZES = [
    131072,   # 128K — known good, baseline
    163840,   # 160K
    196608,   # 192K
    229376,   # 224K
    262144,   # 256K
    327680,   # 320K
    393216,   # 384K
    458752,   # 448K
    524288,   # 512K
]

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

def free_mem():
    try:
        out = subprocess.check_output(["free", "-m"], text=True)
        parts = out.strip().split("\n")[1].split()
        swap = out.strip().split("\n")[2].split()
        return {
            "available_mb": int(parts[6]) if len(parts) > 6 else -1,
            "swap_used_mb": int(swap[2]) if len(swap) > 2 else -1,
        }
    except:
        return {}

def dmesg_recent(n=5):
    try:
        out = subprocess.check_output(
            f"sudo dmesg | tail -{n}", shell=True, text=True, timeout=5
        )
        return out.strip()
    except:
        return ""

def ollama_api(endpoint, data=None, timeout=900):
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

def bench_one(num_ctx, timeout=900):
    print(f"  ctx={num_ctx//1024}K ({num_ctx}) ...", end=" ", flush=True)

    # Full restart — clean slate
    subprocess.run(["sudo", "systemctl", "restart", "ollama"], timeout=30,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(15)

    pre_gtt = gtt_used_gib()
    pre_mem = free_mem()
    pre_dmesg = dmesg_recent(3)

    t0 = time.time()
    resp = ollama_api("/api/generate", {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": 50, "temperature": 0.7},
        "think": False,
        "keep_alive": "1m",
    }, timeout=timeout)
    elapsed = time.time() - t0

    post_gtt = gtt_used_gib()
    post_mem = free_mem()
    post_dmesg = dmesg_recent(3)

    r = {
        "num_ctx": num_ctx,
        "num_ctx_k": f"{num_ctx//1024}K",
        "elapsed_s": round(elapsed, 1),
        "pre_gtt_gib": pre_gtt,
        "post_gtt_gib": post_gtt,
        "pre_mem": pre_mem,
        "post_mem": post_mem,
        "new_dmesg": post_dmesg != pre_dmesg,
        "dmesg_tail": post_dmesg if post_dmesg != pre_dmesg else "",
    }

    if "error" in resp:
        r["status"] = "FAIL"
        r["error"] = resp["error"][:500]
        r["error_type"] = resp.get("error_type", "unknown")
        try:
            subprocess.check_output(["pgrep", "-f", "ollama serve"], timeout=3)
            r["ollama_alive"] = True
        except:
            r["ollama_alive"] = False
        print(f"FAIL ({elapsed:.0f}s): {r['error'][:80]}", flush=True)
        if r.get("new_dmesg"):
            print(f"    NEW DMESG: {post_dmesg[-200:]}", flush=True)
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
                    r["reported_ctx"] = m.get("context_length", 0)
                    break
        print(f"OK ({elapsed:.0f}s): gen={r['gen_tok_s']} tok/s, "
              f"vram={r.get('size_vram_gib','?')}G, gtt={post_gtt}G, "
              f"reported_ctx={r.get('reported_ctx', '?')}, "
              f"avail={post_mem.get('available_mb','?')}MB", flush=True)
    return r


def main():
    # Check queue-runner is stopped
    qr = subprocess.run(["systemctl", "is-active", "queue-runner"],
                        capture_output=True, text=True)
    if "active" in qr.stdout.strip():
        print("WARNING: queue-runner is active! Stopping it for clean benchmarks...")
        subprocess.run(["sudo", "systemctl", "stop", "queue-runner"], check=True)
        time.sleep(2)

    print(f"{'='*70}")
    print(f"PUSH BEYOND 128K — Find Real Ceiling")
    print(f"Model: {MODEL}")
    print(f"Context sizes: {[c//1024 for c in CTX_SIZES]}K")
    print(f"Queue-runner: stopped for clean measurements")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    results = []
    consecutive_fails = 0

    for ctx in CTX_SIZES:
        r = bench_one(ctx)
        results.append(r)

        if r["status"] != "OK":
            consecutive_fails += 1
            if consecutive_fails >= 2:
                print(f"\n  2 consecutive fails — ceiling found.")
                break
        else:
            consecutive_fails = 0

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'ctx':>7} {'status':<6} {'gen_t/s':>8} {'vram_G':>8} {'gtt_G':>8} {'avail_MB':>9} {'rpt_ctx':>8}")
    print("-" * 65)
    for r in results:
        ctx_k = f"{r['num_ctx']//1024}K"
        st = r["status"]
        gen = r.get("gen_tok_s", "-")
        vram = r.get("size_vram_gib", "-")
        gtt = r.get("post_gtt_gib", "-")
        avail = r.get("post_mem", {}).get("available_mb", "-")
        rpt = r.get("reported_ctx", "-")
        print(f"{ctx_k:>7} {st:<6} {gen:>8} {vram:>8} {gtt:>8} {avail:>9} {rpt:>8}")

    # Save
    path = os.path.join(RESULTS_DIR, "task-ceiling.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {path}")

    # Restart queue-runner
    print("\nRestarting queue-runner...")
    subprocess.run(["sudo", "systemctl", "start", "queue-runner"])
    print("Done.")


if __name__ == "__main__":
    main()
