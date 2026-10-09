#!/usr/bin/env python3
"""128K reproducibility test — run 5 trials at 128K context.
Captures detailed timing, memory, and failure data per trial.
"""
import json, time, subprocess, sys, os, urllib.request

OLLAMA = "http://localhost:11434"
sys.stdout.reconfigure(line_buffering=True)

MODEL = "qwen3.5-35b-a3b-iq2m"
CTX = 131072  # 128K
TRIALS = 5
PROMPT = "Explain quantum entanglement in detail, covering EPR paradox, Bell's theorem, and applications."

def sysfs_read(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except:
        return -1

def gtt_used_gib():
    return round(sysfs_read("/sys/class/drm/card1/device/mem_info_gtt_used") / (1024**3), 3)

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

def main():
    print(f"{'='*70}")
    print(f"128K Reproducibility Test — {TRIALS} trials")
    print(f"Model: {MODEL}, Context: {CTX}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    results = []
    for trial in range(1, TRIALS + 1):
        print(f"\n--- Trial {trial}/{TRIALS} ---")

        # Full restart
        subprocess.run(["sudo", "systemctl", "restart", "ollama"], timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(15)  # generous wait

        pre_gtt = gtt_used_gib()
        t0 = time.time()
        resp = ollama_api("/api/generate", {
            "model": MODEL,
            "prompt": PROMPT,
            "stream": False,
            "options": {"num_ctx": CTX, "num_predict": 50, "temperature": 0.7},
            "think": False,
            "keep_alive": "1m",
        }, timeout=600)
        elapsed = time.time() - t0
        post_gtt = gtt_used_gib()

        r = {"trial": trial, "elapsed_s": round(elapsed, 1), "pre_gtt": pre_gtt, "post_gtt": post_gtt}

        if "error" in resp:
            r["status"] = "FAIL"
            r["error"] = resp["error"][:200]
            r["error_type"] = resp.get("error_type", "unknown")
            print(f"  FAIL ({elapsed:.0f}s): {r['error'][:80]}")
        else:
            r["status"] = "OK"
            ed = resp.get("eval_duration", 0) / 1e9
            ec = resp.get("eval_count", 0)
            pd = resp.get("prompt_eval_duration", 0) / 1e9
            pc = resp.get("prompt_eval_count", 0)
            ld = resp.get("load_duration", 0) / 1e9
            r["gen_tok_s"] = round(ec / ed, 1) if ed > 0 else 0
            r["prefill_tok_s"] = round(pc / pd, 1) if pd > 0 else 0
            r["load_s"] = round(ld, 1)
            r["eval_count"] = ec

            ps = ollama_ps()
            if ps and "models" in ps:
                for m in ps["models"]:
                    if MODEL in m.get("name", ""):
                        r["size_vram_gib"] = round(m.get("size_vram", 0) / (1024**3), 3)
                        break
            print(f"  OK ({elapsed:.0f}s): gen={r['gen_tok_s']} tok/s, "
                  f"load={r['load_s']}s, vram={r.get('size_vram_gib','?')}G, gtt={post_gtt}G")
        results.append(r)

    # Summary
    ok = [r for r in results if r["status"] == "OK"]
    fail = [r for r in results if r["status"] != "OK"]
    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(ok)}/{TRIALS} passed, {len(fail)}/{TRIALS} failed")
    if ok:
        gen_speeds = [r["gen_tok_s"] for r in ok]
        load_times = [r["load_s"] for r in ok]
        print(f"Gen tok/s: min={min(gen_speeds)}, max={max(gen_speeds)}, avg={sum(gen_speeds)/len(gen_speeds):.1f}")
        print(f"Load time: min={min(load_times)}, max={max(load_times)}, avg={sum(load_times)/len(load_times):.1f}")
    if fail:
        for r in fail:
            print(f"  Fail trial {r['trial']}: {r.get('error_type', '?')} — {r.get('error', '?')[:80]}")

    path = "/tmp/bc250-ctx-bench/task6-128k-repro.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {path}")

if __name__ == "__main__":
    main()
