#!/usr/bin/env python3
"""Analyze Phase 3 context scaling results."""
import json

d = json.load(open("benchmarks/results-phase3.json"))

# Group by model and context size, use run 1 for prefill (run 2 is cached)
results = {}
for r in d:
    if r.get("status") != "OK":
        continue
    model = r["model"].split("/")[-1][:35]
    ctx_k = r["num_ctx"] // 1024
    run = r.get("run", 1)
    key = (model, ctx_k)
    if key not in results:
        results[key] = {"gen_runs": [], "prefill_run1": None, "prompt_eval_count": None, "truncated": False}
    results[key]["gen_runs"].append(r["gen_tok_s"])
    if run == 1:
        results[key]["prefill_run1"] = r["prefill_tok_s"]
        results[key]["prompt_eval_count"] = r.get("prompt_eval_count", 0)
        results[key]["ttft"] = r.get("ttft_s", 0)
        results[key]["swap_mb"] = r.get("mem_after", {}).get("swap_used_mb", 0)
    results[key]["truncated"] = r.get("context_truncated", False)

# Print table
print("Context Scaling — Filled to 80% (July 2026 re-benchmark)")
print("=" * 110)
print("%-35s %5s %8s %8s %8s %7s %6s %5s" % (
    "Model", "Ctx", "Gen R1", "Gen R2", "Prefill", "TTFT", "Swap", "Trunc"))
print("-" * 110)

models_order = {}
for (model, ctx_k), data in sorted(results.items(), key=lambda x: (x[0][0], x[0][1])):
    gen_runs = data["gen_runs"]
    pre = data["prefill_run1"] or 0
    ttft = data.get("ttft", 0)
    swap = data.get("swap_mb", 0)
    trunc = "YES" if data["truncated"] else ""
    gen_r1 = gen_runs[0] if len(gen_runs) >= 1 else 0
    gen_r2 = gen_runs[1] if len(gen_runs) >= 2 else 0
    print("%-35s %4dK %7.1f %7.1f %7.1f %6.1fs %5dMB %s" % (
        model, ctx_k, gen_r1, gen_r2, pre, ttft, swap, trunc))
