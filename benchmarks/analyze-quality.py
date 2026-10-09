#!/usr/bin/env python3
"""Analyze quality benchmark results."""
import json

d = json.load(open("benchmarks/results-phase4.json"))

models = {}
for r in d:
    model = r.get("model", "?").split("/")[-1][:35]
    task = r.get("task", "?")
    scores = r.get("quality_scores", {})
    key = (model, task)
    if key not in models:
        models[key] = []
    models[key].append(scores)

print("Quality Assessment Results")
print("=" * 90)
for (model, task), runs in sorted(models.items()):
    pass_count = 0
    detail = ""
    for s in runs:
        if task == "summarize":
            p = s.get("keywords_pass", False) and s.get("sentences_pass", False)
            detail = "kw=%d/%d sent=%d" % (s.get("keywords_found", 0), s.get("keywords_total", 0), s.get("sentence_count", 0))
        elif task == "json_extract":
            p = s.get("valid_json", False) and s.get("keys_pass", False)
            detail = "json=%s keys=%d/%d" % (s.get("valid_json", False), s.get("keys_found", 0), s.get("keys_total", 0))
        elif task == "fact_recall":
            p = s.get("keywords_pass", False)
            detail = "kw=%d/%d" % (s.get("keywords_found", 0), s.get("keywords_total", 0))
        elif task == "instruction_follow":
            p = s.get("items_pass", False)
            detail = "items=%d/%d" % (s.get("numbered_items", 0), s.get("items_expected", 0))
        elif task == "arithmetic":
            p = s.get("contains_target", False)
            detail = "has_391=%s nums=%s" % (s.get("contains_target", False), s.get("numbers_found", []))
        else:
            p = False
        if p:
            pass_count += 1
    print("%35s  %-20s  %d/%d pass  (%s)" % (model, task, pass_count, len(runs), detail))
