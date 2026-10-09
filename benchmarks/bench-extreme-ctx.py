#!/usr/bin/env python3
"""Push MoE context to extreme: 40K-128K"""
import json, time, urllib.request, subprocess, sys
sys.stdout.reconfigure(line_buffering=True)

OLLAMA = "http://localhost:11434"

def api(ep, data=None, timeout=600):
    url = f"{OLLAMA}{ep}"
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

def gtt():
    with open("/sys/class/drm/card1/device/mem_info_gtt_used") as f:
        return int(f.read().strip()) / (1024**3)

def pages_limit():
    with open("/sys/module/ttm/parameters/pages_limit") as f:
        return int(f.read().strip())

model = "qwen3.5-35b-a3b-iq2m"
prompt = "Explain quantum entanglement in detail, covering the EPR paradox, Bell inequalities, and quantum teleportation."

print(f"pages_limit = {pages_limit()} ({pages_limit()*4096/(1024**3):.1f} GiB)")
print(f"Model: {model}")
print(f"Testing extreme context sizes...\n")

results = []
for ctx in [40960, 49152, 57344, 65536, 81920, 98304, 131072]:
    ctx_k = f"{ctx//1024}K"
    print(f"Restarting ollama for ctx={ctx_k}...", end=" ", flush=True)
    subprocess.run(["sudo", "systemctl", "restart", "ollama"], timeout=30,
                   capture_output=True)
    time.sleep(12)
    print("generating...", end=" ", flush=True)
    
    t0 = time.time()
    resp = api("/api/generate", {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_ctx": ctx, "num_predict": 50, "temperature": 0.7},
        "think": False, "keep_alive": "1m",
    }, timeout=600)
    elapsed = time.time() - t0
    
    if "error" in resp:
        err = resp["error"][:120]
        print(f"FAIL ({elapsed:.0f}s): {err}", flush=True)
        results.append({"ctx": ctx, "status": "FAIL", "error": err, "elapsed": elapsed})
    else:
        ed = resp.get("eval_duration", 0) / 1e9
        ec = resp.get("eval_count", 0)
        gen = round(ec / ed, 1) if ed > 0 else 0
        pd = resp.get("prompt_eval_duration", 0) / 1e9
        pc = resp.get("prompt_eval_count", 0)
        pre = round(pc / pd, 1) if pd > 0 else 0
        
        ps = api("/api/ps")
        sv, st, actual_ctx = 0, 0, 0
        if ps and "models" in ps:
            for m in ps["models"]:
                sv = m.get("size_vram", 0) / (1024**3)
                st = m.get("size", 0) / (1024**3)
                actual_ctx = m.get("context_length", 0)
        
        g = gtt()
        print(f"OK {gen} tok/s (pre={pre}), VRAM={sv:.3f}G, GTT={g:.2f}G, "
              f"actual_ctx={actual_ctx}, elapsed={elapsed:.0f}s", flush=True)
        results.append({
            "ctx": ctx, "status": "OK", "gen_tok_s": gen, "prefill_tok_s": pre,
            "vram_gib": sv, "gtt_gib": g, "actual_ctx": actual_ctx, "elapsed": elapsed,
        })

print("\n=== SUMMARY ===")
for r in results:
    ctx_k = f"{r['ctx']//1024}K"
    if r["status"] == "OK":
        print(f"  {ctx_k:>5}: OK  {r['gen_tok_s']:>5.1f} tok/s  VRAM={r['vram_gib']:.3f}G  "
              f"GTT={r['gtt_gib']:.2f}G  actual_ctx={r['actual_ctx']}")
    else:
        print(f"  {ctx_k:>5}: FAIL  {r.get('error','')[:80]}")

# Save
with open("/tmp/bc250-ctx-bench/task0c-extreme.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: /tmp/bc250-ctx-bench/task0c-extreme.json")
