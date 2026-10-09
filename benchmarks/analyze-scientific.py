#!/usr/bin/env python3
"""Analyze scientific benchmark results."""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/opt/netscan/tmp/bench-results/bench_scientific_20260320_1007.json"
d = json.load(open(path))

# Group by model
models = {}
for r in d:
    m = r["model"]
    if m not in models:
        models[m] = []
    models[m].append(r)

for model, results in models.items():
    print(f"\n{'='*80}")
    print(f"  {model}")
    print(f"{'='*80}")
    print(f"{'Ctx':>6} {'Phase':<11} {'Status':<8} {'Gen':>7} {'Prefill':>8} {'Tokens':>18} {'Swap':>6} {'Wall':>8} {'Trunc':>6} {'Note'}")
    print("-"*100)
    for r in results:
        ctx = r.get("num_ctx", 0)
        ctx_str = f"{ctx//1024}K" if ctx >= 1024 else str(ctx)
        phase = r.get("phase", "?")
        status = r["status"]
        gen = r.get("gen_tok_s", 0)
        pf = r.get("prefill_tok_s", 0)
        prompt = r.get("prompt_eval_count", 0)
        fill = r.get("target_fill_tokens", 0)
        swap = r.get("mem_after", {}).get("swap_used_mb", 0)
        wall = r.get("wall_s", 0)
        trunc = "YES" if r.get("context_truncated") else ""
        ceiling = r.get("ceiling_reason", "")
        verified = "✓" if r.get("fill_verified") else ""
        
        tok_str = f"{prompt:>7}/{fill:<7}" if phase == "filled" else "alloc"
        print(f"{ctx_str:>6} {phase:<11} {status:<8} {gen:>7.1f} {pf:>8.1f} {tok_str:>18} {swap:>5}M {wall:>7.1f}s {trunc:>6} {verified} {ceiling}")

print(f"\nTotal tests: {len(d)}")
