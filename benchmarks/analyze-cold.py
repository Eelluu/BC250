#!/usr/bin/env python3
"""Analyze cold-start results."""
import json

d = json.load(open("benchmarks/results-phase5.json"))
print("Cold Start TTFT (3 runs)")
print("=" * 70)
for r in d:
    if r.get("status") == "OK":
        m = r["model"].split("/")[-1][:35]
        print("%-35s run=%d  cold=%5.1fs  load=%5.1fs  TTFT=%5.1fs  gen=%5.1f" % (
            m, r["run"], r["cold_total_s"], r["load_s"], r["ttft_s"], r["gen_tok_s"]
        ))
    else:
        m = r.get("model", "?").split("/")[-1][:35]
        print("%-35s run=%d  FAIL: %s" % (m, r.get("run", 0), r.get("error", "?")[:60]))
