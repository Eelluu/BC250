#!/usr/bin/env python3
"""
Benchmark BC-250 response times against Readme.md claims.
Requires SSH tunnel: ssh -f -N -L 11434:localhost:11434 127.0.0.1
"""

import urllib.request
import json
import time
import sys
import base64

OLLAMA = "http://localhost:11434"
MOE_MODEL = "qwen3.5-35b-a3b-iq2m"
VISION_MODEL = "qwen3.5:9b"

CLAIMS = {
    "cold_start": (30, 60),
    "text_warm":  (10, 30),
    "vision":     (40, 80),
}


def ollama_post(endpoint, payload, timeout=300):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OLLAMA}{endpoint}", data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp.read().decode()


def ollama_get(endpoint, timeout=10):
    return urllib.request.urlopen(f"{OLLAMA}{endpoint}", timeout=timeout).read().decode()


def parse_streaming(raw):
    content = []
    stats = {}
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            c = obj.get("message", {}).get("content", "")
            if c:
                content.append(c)
            if obj.get("done"):
                stats = obj
        except json.JSONDecodeError:
            pass
    return "".join(content), stats


def unload_model(model):
    try:
        ollama_post("/api/chat", {
            "model": model, "messages": [{"role": "user", "content": "x"}],
            "stream": False, "keep_alive": 0,
        }, timeout=30)
    except Exception:
        pass


def unload_all():
    try:
        ps = json.loads(ollama_get("/api/ps"))
        for m in ps.get("models", []):
            unload_model(m["name"])
    except Exception:
        pass
    time.sleep(2)


def test_cold_start():
    print("\n─── Test 1: Cold Start (model reload) ───")
    unload_all()
    print(f"  All models unloaded. Loading {MOE_MODEL} cold...")

    t0 = time.time()
    raw = ollama_post("/api/chat", {
        "model": MOE_MODEL,
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "stream": True,
        "options": {"num_ctx": 16384, "num_predict": 50},
    }, timeout=180)
    wall = time.time() - t0

    text, stats = parse_streaming(raw)
    load_s = stats.get("load_duration", 0) / 1e9
    prompt_s = stats.get("prompt_eval_duration", 0) / 1e9
    eval_count = stats.get("eval_count", 0)
    eval_dur = stats.get("eval_duration", 0) / 1e9
    tps = eval_count / eval_dur if eval_dur > 0 else 0

    print(f"  Wall: {wall:.1f}s | Load: {load_s:.1f}s | Prompt eval: {prompt_s:.1f}s")
    print(f"  Tokens: {eval_count} @ {tps:.1f} tok/s")
    print(f"  Response: {text[:120]}")

    lo, hi = CLAIMS["cold_start"]
    measured = load_s if load_s > 0 else wall
    verdict = "✅ PASS" if lo <= measured <= hi * 1.1 else ("⚠️ FASTER" if measured < lo else "❌ SLOWER")
    print(f"  Claim: {lo}–{hi}s | Load duration: {measured:.1f}s | {verdict}")
    return {"test": "cold_start", "wall_s": wall, "load_s": load_s, "measured": measured}


def test_text_warm():
    print(f"\n─── Test 2: Warm Text Reply ({MOE_MODEL}) ───")
    print("  Warm-up query...")
    ollama_post("/api/chat", {
        "model": MOE_MODEL,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "options": {"num_ctx": 16384, "num_predict": 5},
    }, timeout=180)
    time.sleep(1)

    prompts = [
        "What is the capital of France? Answer in one sentence.",
        "Explain what a floppy disk is in 2-3 sentences.",
        "What are the main differences between ARM and x86? Be brief, 3-4 sentences.",
    ]

    results = []
    for i, prompt in enumerate(prompts):
        print(f"  Query {i+1}: {prompt[:60]}...")
        t0 = time.time()
        raw = ollama_post("/api/chat", {
            "model": MOE_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Be concise."},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "options": {"num_ctx": 16384, "num_predict": 200},
        }, timeout=120)
        wall = time.time() - t0

        text, stats = parse_streaming(raw)
        eval_count = stats.get("eval_count", 0)
        eval_dur = stats.get("eval_duration", 0) / 1e9
        prompt_dur = stats.get("prompt_eval_duration", 0) / 1e9
        tps = eval_count / eval_dur if eval_dur > 0 else 0

        print(f"    Wall: {wall:.1f}s | TTFT: {prompt_dur:.1f}s | {eval_count} tok @ {tps:.1f} tok/s")
        print(f"    → {text[:100]}")
        results.append({"wall": wall, "tokens": eval_count, "tps": tps})

    avg_wall = sum(r["wall"] for r in results) / len(results)
    avg_tps = sum(r["tps"] for r in results) / len(results)
    lo, hi = CLAIMS["text_warm"]
    verdict = "✅ PASS" if lo <= avg_wall <= hi else ("⚠️ FASTER" if avg_wall < lo else "❌ SLOWER")
    print(f"  Avg wall: {avg_wall:.1f}s | Avg speed: {avg_tps:.1f} tok/s")
    print(f"  Claim: {lo}–{hi}s | Measured avg: {avg_wall:.1f}s | {verdict}")
    return {"test": "text_warm", "results": results, "avg_wall": avg_wall, "measured": avg_wall}


def test_vision():
    print(f"\n─── Test 3: Vision Analysis ({VISION_MODEL}) ───")

    unload_model(MOE_MODEL)
    time.sleep(2)

    img_path = "images/shadow-marshall-floppy.jpg"
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    print(f"  Image: {len(img_b64)//1024} KB base64")

    print(f"  Sending to {VISION_MODEL} (cold load + vision inference)...")
    t0 = time.time()
    raw = ollama_post("/api/chat", {
        "model": VISION_MODEL,
        "messages": [
            {"role": "user", "content": "Describe this image in detail.", "images": [img_b64]},
        ],
        "stream": True,
        "options": {"num_ctx": 4096, "num_predict": 500, "temperature": 0.3},
    }, timeout=300)
    wall = time.time() - t0

    text, stats = parse_streaming(raw)
    load_s = stats.get("load_duration", 0) / 1e9
    prompt_s = stats.get("prompt_eval_duration", 0) / 1e9
    eval_count = stats.get("eval_count", 0)
    eval_dur = stats.get("eval_duration", 0) / 1e9
    tps = eval_count / eval_dur if eval_dur > 0 else 0

    print(f"  Wall: {wall:.1f}s | Load: {load_s:.1f}s | Prompt eval: {prompt_s:.1f}s")
    print(f"  Tokens: {eval_count} @ {tps:.1f} tok/s")
    print(f"  Response: {text[:200]}")

    lo, hi = CLAIMS["vision"]
    verdict = "✅ PASS" if lo <= wall <= hi else ("⚠️ FASTER" if wall < lo else "❌ SLOWER")
    print(f"  Claim: {lo}–{hi}s | Measured: {wall:.1f}s | {verdict}")

    # Restore MoE
    print(f"  Restoring {MOE_MODEL}...")
    unload_model(VISION_MODEL)

    return {"test": "vision", "wall_s": wall, "load_s": load_s, "tokens": eval_count, "measured": wall}


def main():
    print("=" * 60)
    print("BC-250 Response Time Benchmark vs Readme.md Claims")
    print("=" * 60)

    try:
        ps = json.loads(ollama_get("/api/ps"))
        loaded = [m["name"] for m in ps.get("models", [])]
        print(f"Ollama reachable. Loaded: {loaded or 'none'}")
    except Exception as e:
        print(f"❌ Cannot reach Ollama at {OLLAMA}: {e}")
        print("  Run: ssh -f -N -L 11434:localhost:11434 127.0.0.1")
        sys.exit(1)

    results = []
    results.append(test_cold_start())
    results.append(test_text_warm())
    results.append(test_vision())

    print("\n" + "=" * 60)
    print("SUMMARY — BC-250 Response Times vs Readme.md")
    print("=" * 60)
    print(f"  {'Test':<25} {'Readme Claim':<15} {'Measured':<15} {'Result'}")
    print("  " + "-" * 65)
    for r in results:
        test = r["test"]
        lo, hi = CLAIMS[test]
        m = r["measured"]
        if lo <= m <= hi:
            v = "✅ PASS"
        elif m < lo:
            v = "⚠️ FASTER"
        else:
            v = "❌ SLOWER"
        print(f"  {test:<25} {lo}–{hi}s{'':<9} {m:.1f}s{'':<10} {v}")
    print()


if __name__ == "__main__":
    main()
