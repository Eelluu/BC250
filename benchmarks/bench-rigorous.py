#!/usr/bin/env python3
"""BC-250 Rigorous Benchmark Suite — July 2026

Publication-quality benchmarks with:
- Repeated measurements (3 runs) → median, min, max
- Standardized prompts with known token counts
- Quality benchmarks (summarization, JSON extraction, fact recall)
- Complete metrics: gen, prefill, TTFT, prompt_eval_count, swap, RAM, GPU%
- Cold-start TTFT measurements

Phases (run independently):
  --phase perf      All models, 1 run at 4K, standardized prompt, full metrics
  --phase stats     8 key models, 3 runs at 4K, statistical analysis
  --phase context   6 key models, 2 runs at 4K/16K/32K filled, 64K for production
  --phase quality   2 production models, 5 quality tasks × 3 runs
  --phase cold      2 production models, cold-start TTFT × 3
  --phase all       All phases sequentially

Usage:
  python3 bench-rigorous.py --phase perf
  python3 bench-rigorous.py --phase stats
  python3 bench-rigorous.py --phase quality
  python3 bench-rigorous.py --phase all
"""

import json, time, subprocess, sys, os, datetime, argparse, re, statistics
import urllib.request

OLLAMA = "http://localhost:11434"
RESULTS_DIR = "/opt/netscan/tmp/bench-rigorous"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")

# ─── Model Lists ────────────────────────────────────────────────────────────

ALL_MODELS = [
    # Small (≤4B)
    "llama3.2:3b", "qwen2.5:3b", "phi4-mini:latest", "gemma3:4b", "qwen3:4b",
    # Medium (7-8B)
    "qwen2.5:7b", "qwen2.5-coder:7b", "llama3.1:8b",
    "mannix/llama3.1-8b-lexi:latest", "huihui_ai/seed-coder-abliterate:latest",
    "granite3.3:8b", "qwen3-abl-nothink:latest", "huihui_ai/qwen3-abliterated:8b",
    "glm4:9b", "qwen3:8b", "qwen3:8b-nothink", "deepseek-r1:8b",
    # Large Dense (9-14B)
    "gemma2:9b", "qwen3.5:9b", "mistral-nemo:12b", "gemma3:12b",
    "deepseek-r1:14b", "phi4:14b", "qwen3-14b-16k:latest",
    "huihui_ai/qwen3-abliterated:14b", "qwen3-14b-abl-nothink:latest", "qwen3:14b",
    # MoE / XL
    "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q2_K",
    "qwen3.5-35b-a3b-iq2m:latest",
    "qwen3.5-27b-iq2m:latest",
    # Q8_0 variant
    "qwen3:8b-q8_0",
]

KEY_MODELS = [
    "qwen3.5-35b-a3b-iq2m:latest",   # MoE primary
    "qwen3.5:9b",                      # Dense primary (vision)
    "phi4-mini:latest",                # Best quality/speed
    "llama3.2:3b",                     # Fastest
    "qwen3:8b",                        # Mid-range reference
    "qwen3:14b",                       # Dense 14B reference
    "mistral-nemo:12b",                # Non-Qwen reference
    "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q2_K",  # Second MoE
]

CONTEXT_MODELS = [
    "qwen3.5-35b-a3b-iq2m:latest",
    "qwen3.5:9b",
    "phi4-mini:latest",
    "qwen3:8b",
    "qwen3:14b",
    "mistral-nemo:12b",
]

PRODUCTION_MODELS = [
    "qwen3.5-35b-a3b-iq2m:latest",
    "qwen3.5:9b",
]

# ─── Standardized Prompt ────────────────────────────────────────────────────
# ~400 tokens. Used for ALL performance measurements to ensure fair comparison.
STANDARD_PROMPT = """Explain the key differences between RISC and CISC processor architectures. Cover instruction set design, pipelining implications, memory access patterns, and power efficiency trade-offs. Include specific examples of processors from each category (e.g., ARM Cortex-A for RISC, x86 for CISC) and discuss how modern designs have blurred the traditional boundaries between these two approaches. Conclude with a brief assessment of which approach better suits embedded systems with strict power constraints versus high-performance server workloads."""

# ─── Fill Block for Context Tests ───────────────────────────────────────────
FILL_BLOCK = """The evolution of semiconductor manufacturing represents one of the most
remarkable engineering achievements in human history. From the first transistor
at Bell Labs in 1947 to modern 3nm process nodes, the industry has maintained
exponential scaling for over seven decades. Each generation of lithography
brought new challenges: optical diffraction limits led to immersion lithography,
then extreme ultraviolet (EUV) sources. The economics are equally staggering —
a modern fab costs $20 billion or more, yet produces chips at less than a cent
per transistor. Memory technologies evolved in parallel: from magnetic core to
SRAM, DRAM, and now 3D NAND flash with hundreds of layers. The interface between
processor and memory — the memory wall — remains the fundamental bottleneck
in computing performance. Bandwidth grows slower than compute, creating an
ever-widening gap that architects address through deeper cache hierarchies,
prefetching, and data-flow optimizations. On the software side, compilers have
become extraordinarily sophisticated, performing loop vectorization, automatic
parallelization, and profile-guided optimization. The interaction between
hardware and software design creates a co-evolution where each enables and
constrains the other. In artificial intelligence, this manifests as the
transformer architecture's quadratic attention mechanism — theoretically elegant
but practically bounded by memory bandwidth on real hardware. Quantization
techniques reduce mathematical precision for throughput, enabled by hardware
that natively supports reduced-precision arithmetic.\n\n"""

# ─── Quality Task Definitions ──────────────────────────────────────────────

QUALITY_TASKS = {
    "summarize": {
        "prompt": """Read the following passage carefully, then write exactly 3 sentences summarizing the key points.

PASSAGE:
The AMD BC-250 is a crypto-mining board built around AMD's Cyan Skillfish APU, featuring a Zen 2 CPU with 6 cores and 12 threads alongside a GFX1013 GPU with 24 compute units and 1536 stream processors. The board has 16 GB of GDDR6 unified memory shared between CPU and GPU (UMA architecture), with a 256-bit memory bus. Originally deployed in multi-board mining racks by ASRock Rack, these boards became available on the secondary market after decommissioning. The GPU does not support ROCm's userspace libraries, making Vulkan the only viable compute path for AI inference. Despite lacking dedicated matrix multiplication hardware (tensor cores), the board can run a 35-billion parameter Mixture-of-Experts language model at 38 tokens per second using quantized weights and a Vulkan compute backend.

YOUR 3-SENTENCE SUMMARY:""",
        "checks": {
            "keywords": ["BC-250", "Vulkan", "16 GB", "35"],
            "min_sentences": 2,
            "max_sentences": 5,
        },
    },
    "json_extract": {
        "prompt": """Extract the following fields from the text below and return ONLY valid JSON. No explanation, no markdown, just the JSON object.

TEXT: "Dr. Maria Chen, age 42, works as a Senior Research Scientist at NVIDIA's Santa Clara campus. She joined in 2018 and specializes in GPU compiler optimization. Her team has 12 members and she holds 7 patents related to shader compilation."

Required fields: name, age, title, company, year_joined, team_size, patent_count

JSON:""",
        "checks": {
            "valid_json": True,
            "required_keys": ["name", "age", "title", "company"],
            "expected_values": {"name": "Maria Chen", "age": 42},
        },
    },
    "fact_recall": {
        "prompt": """I will give you some facts. Memorize them, then answer my question.

FACTS:
- The administrative capital of Myanmar is Naypyidaw (moved from Yangon in 2006).
- The deepest point in the ocean is the Challenger Deep at 10,935 meters.
- The chemical symbol for tungsten is W (from German: Wolfram).
- The speed of light in vacuum is exactly 299,792,458 meters per second.
- Ada Lovelace is widely regarded as the first computer programmer.

QUESTION: What is the chemical symbol for tungsten, and what word is it derived from?

ANSWER:""",
        "checks": {
            "keywords": ["W", "Wolfram"],
        },
    },
    "instruction_follow": {
        "prompt": """List exactly 5 advantages of using solar energy for electricity generation. Format each as a numbered item (1. through 5.). Be concise — one sentence per item. Do not include any introduction or conclusion, just the 5 numbered items.""",
        "checks": {
            "has_numbered_items": 5,
        },
    },
    "arithmetic": {
        "prompt": """Solve this arithmetic problem. Give ONLY the final numerical answer, nothing else.

What is 17 × 23?

Answer:""",
        "checks": {
            "contains_number": 391,
        },
    },
}

# ─── Helpers ────────────────────────────────────────────────────────────────

REQUEST_TIMEOUT = 1200  # 20 min

def api(endpoint, data=None, timeout=REQUEST_TIMEOUT):
    url = f"{OLLAMA}{endpoint}"
    req = urllib.request.Request(url, method="POST" if data else "GET")
    if data:
        req.add_header("Content-Type", "application/json")
        body = json.dumps(data).encode()
    else:
        body = None
    with urllib.request.urlopen(req, body, timeout=timeout) as resp:
        return json.loads(resp.read())


def unload_all():
    try:
        ps = api("/api/ps", timeout=10)
        if ps and "models" in ps:
            for m in ps["models"]:
                name = m.get("name", "")
                if name:
                    api("/api/generate", {"model": name, "keep_alive": 0}, timeout=30)
                    time.sleep(1)
    except Exception:
        pass
    time.sleep(3)


def get_system_memory():
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
        total = info.get("MemTotal", 0) / 1024
        free = info.get("MemAvailable", 0) / 1024
        swap_total = info.get("SwapTotal", 0) / 1024
        swap_free = info.get("SwapFree", 0) / 1024
        return {
            "ram_total_mb": round(total),
            "ram_used_mb": round(total - free),
            "ram_free_mb": round(free),
            "swap_used_mb": round(swap_total - swap_free),
        }
    except Exception as e:
        return {"error": str(e)}


def get_vram_info(model):
    try:
        ps = api("/api/ps", timeout=10)
        if ps and "models" in ps:
            for m in ps["models"]:
                if model in m.get("name", "") or model in m.get("model", ""):
                    sv = m.get("size_vram", 0)
                    st = m.get("size", 0)
                    details = m.get("details", {})
                    return {
                        "vram_gib": round(sv / (1024**3), 2),
                        "total_gib": round(st / (1024**3), 2),
                        "gpu_pct": round(sv / st * 100) if st > 0 else -1,
                        "family": details.get("family", ""),
                        "parameter_size": details.get("parameter_size", ""),
                        "quantization": details.get("quantization_level", ""),
                    }
    except Exception:
        pass
    return {}


def get_offload_layers():
    """Parse Ollama journal for layers offloaded."""
    try:
        result = subprocess.run(
            ["sudo", "journalctl", "-u", "ollama", "-n", "30", "--no-pager", "-q"],
            capture_output=True, text=True, timeout=10
        )
        for line in reversed(result.stdout.split("\n")):
            if "offloaded" in line and "layers" in line:
                m = re.search(r"offloaded (\d+/\d+) layers", line)
                if m:
                    return m.group(1)
    except Exception:
        pass
    return ""


def run_generate(model, prompt, num_ctx=4096, num_predict=100, timeout=600):
    """Run a single generation and return complete metrics."""
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": num_predict},
        "keep_alive": "5m",
        "think": False,
    }
    mem_before = get_system_memory()
    t0 = time.time()
    try:
        resp = api("/api/generate", data, timeout=timeout)
        wall = time.time() - t0

        if "error" in resp:
            return {"status": "FAIL", "error": resp["error"][:200], "wall_s": round(wall, 1)}

        eval_count = resp.get("eval_count", 0)
        eval_dur = resp.get("eval_duration", 0) / 1e9
        prompt_count = resp.get("prompt_eval_count", 0)
        prompt_dur = resp.get("prompt_eval_duration", 0) / 1e9
        load_dur = resp.get("load_duration", 0) / 1e9

        gen_toks = eval_count / eval_dur if eval_dur > 0 else 0
        prefill_toks = prompt_count / prompt_dur if prompt_dur > 0 else 0

        mem_after = get_system_memory()

        return {
            "status": "OK",
            "gen_tok_s": round(gen_toks, 2),
            "prefill_tok_s": round(prefill_toks, 2),
            "eval_count": eval_count,
            "prompt_eval_count": prompt_count,
            "ttft_s": round(load_dur + prompt_dur, 2),
            "load_s": round(load_dur, 1),
            "prompt_eval_s": round(prompt_dur, 2),
            "eval_s": round(eval_dur, 2),
            "wall_s": round(wall, 1),
            "response": resp.get("response", "")[:500],
            "mem_before": mem_before,
            "mem_after": mem_after,
            "swap_delta_mb": mem_after.get("swap_used_mb", 0) - mem_before.get("swap_used_mb", 0),
        }
    except Exception as e:
        return {
            "status": "TIMEOUT" if "timed out" in str(e).lower() else "FAIL",
            "error": str(e)[:200],
            "wall_s": round(time.time() - t0, 1),
        }


def build_fill_prompt(target_tokens):
    chars_per_token = 3.8
    target_chars = int(target_tokens * chars_per_token)
    block_chars = len(FILL_BLOCK)
    repeats = max(1, target_chars // block_chars)
    prompt = (FILL_BLOCK * repeats)[:target_chars]
    prompt += "\n\nBased on the above text, write a brief 2-sentence summary of the key themes discussed."
    return prompt


def save_results(data, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  → Saved: {path}")
    return path


def short_name(model):
    """Short display name for a model."""
    m = model.replace("huihui_ai/", "").replace("mannix/", "")
    m = m.replace("hf.co/unsloth/Qwen3-30B-A3B-GGUF:", "Qwen3-30B-A3B-")
    m = m.replace(":latest", "")
    return m


# ─── Phase 1: Performance Baseline ─────────────────────────────────────────

def phase_perf(models=None):
    """All models, 1 run at 4K, standardized prompt, full metrics."""
    models = models or ALL_MODELS
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Performance Baseline — {len(models)} models")
    print(f"  Prompt: ~400 tokens, num_predict=100, num_ctx=4096")
    print(f"{'='*70}")

    results = []
    for i, model in enumerate(models, 1):
        print(f"\n  [{i}/{len(models)}] {short_name(model)}")
        unload_all()

        r = run_generate(model, STANDARD_PROMPT, num_ctx=4096, num_predict=100, timeout=300)
        r["model"] = model
        r["phase"] = "perf"
        r["timestamp"] = datetime.datetime.now().isoformat()

        if r["status"] == "OK":
            vinfo = get_vram_info(model)
            layers = get_offload_layers()
            r["vram"] = vinfo
            r["layers_offloaded"] = layers
            print(f"    gen={r['gen_tok_s']:>6.1f} tok/s  prefill={r['prefill_tok_s']:>6.1f} tok/s  "
                  f"TTFT={r['ttft_s']:.1f}s  vram={vinfo.get('vram_gib', '?')}G  "
                  f"gpu={vinfo.get('gpu_pct', '?')}%  layers={layers}  "
                  f"swap={r['mem_after'].get('swap_used_mb', '?')}MB")
        else:
            print(f"    FAIL: {r.get('error', '?')[:80]}")

        results.append(r)
        save_results(results, f"phase1-perf-{TIMESTAMP}.json")

    return results


# ─── Phase 2: Statistical Rigor ────────────────────────────────────────────

def phase_stats(models=None, runs=3):
    """Key models, N runs at 4K, statistical analysis."""
    models = models or KEY_MODELS
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Statistical Rigor — {len(models)} models × {runs} runs")
    print(f"{'='*70}")

    results = []
    for i, model in enumerate(models, 1):
        print(f"\n  [{i}/{len(models)}] {short_name(model)}")
        unload_all()

        model_runs = []
        for run_idx in range(runs):
            if run_idx == 0:
                # First run: cold load (model not in memory)
                r = run_generate(model, STANDARD_PROMPT, num_ctx=4096, num_predict=100, timeout=300)
            else:
                # Subsequent runs: warm (model already loaded)
                r = run_generate(model, STANDARD_PROMPT, num_ctx=4096, num_predict=100, timeout=300)

            r["model"] = model
            r["run"] = run_idx + 1
            r["phase"] = "stats"
            r["timestamp"] = datetime.datetime.now().isoformat()

            if r["status"] == "OK":
                if run_idx == 0:
                    vinfo = get_vram_info(model)
                    layers = get_offload_layers()
                    r["vram"] = vinfo
                    r["layers_offloaded"] = layers
                print(f"    Run {run_idx+1}: gen={r['gen_tok_s']:>6.1f}  "
                      f"prefill={r['prefill_tok_s']:>6.1f}  TTFT={r['ttft_s']:.1f}s")
            else:
                print(f"    Run {run_idx+1}: FAIL")

            model_runs.append(r)
            time.sleep(1)  # Brief pause between runs

        # Compute statistics
        ok_runs = [r for r in model_runs if r["status"] == "OK"]
        if len(ok_runs) >= 2:
            gen_vals = [r["gen_tok_s"] for r in ok_runs]
            pre_vals = [r["prefill_tok_s"] for r in ok_runs]
            stats_summary = {
                "model": model,
                "n_runs": len(ok_runs),
                "gen_median": round(statistics.median(gen_vals), 1),
                "gen_min": round(min(gen_vals), 1),
                "gen_max": round(max(gen_vals), 1),
                "gen_stdev": round(statistics.stdev(gen_vals), 2) if len(gen_vals) >= 2 else 0,
                "gen_cv_pct": round(statistics.stdev(gen_vals) / statistics.mean(gen_vals) * 100, 1) if len(gen_vals) >= 2 and statistics.mean(gen_vals) > 0 else 0,
                "prefill_median": round(statistics.median(pre_vals), 1),
                "prefill_min": round(min(pre_vals), 1),
                "prefill_max": round(max(pre_vals), 1),
            }
            print(f"    Stats: gen={stats_summary['gen_median']} [{stats_summary['gen_min']}-{stats_summary['gen_max']}] "
                  f"CV={stats_summary['gen_cv_pct']}%  "
                  f"prefill={stats_summary['prefill_median']} [{stats_summary['prefill_min']}-{stats_summary['prefill_max']}]")
            model_runs.append({"summary": stats_summary})

        results.extend(model_runs)
        save_results(results, f"phase2-stats-{TIMESTAMP}.json")

    return results


# ─── Phase 3: Context Scaling ──────────────────────────────────────────────

def phase_context(models=None, runs=2):
    """Key models, N runs at 4K/16K/32K filled, 64K for production."""
    models = models or CONTEXT_MODELS
    print(f"\n{'='*70}")
    print(f"  PHASE 3: Context Scaling — {len(models)} models × {runs} runs")
    print(f"{'='*70}")

    results = []
    for i, model in enumerate(models, 1):
        name = short_name(model)
        print(f"\n  [{i}/{len(models)}] {name}")

        # Context sizes depend on model
        is_production = model in PRODUCTION_MODELS
        ctx_sizes = [4096, 16384, 32768]
        if is_production:
            ctx_sizes.append(65536)

        for ctx in ctx_sizes:
            fill_tokens = int(ctx * 0.80)
            print(f"\n    ctx={ctx//1024}K  fill={fill_tokens:,} tokens")
            unload_all()

            for run_idx in range(runs):
                prompt = build_fill_prompt(fill_tokens)
                r = run_generate(model, prompt, num_ctx=ctx, num_predict=50,
                                timeout=1200 if ctx >= 65536 else 600)
                r["model"] = model
                r["num_ctx"] = ctx
                r["target_fill_tokens"] = fill_tokens
                r["run"] = run_idx + 1
                r["phase"] = "context"
                r["timestamp"] = datetime.datetime.now().isoformat()

                if r["status"] == "OK":
                    # Check for silent truncation
                    r["context_truncated"] = (
                        fill_tokens > 1000 and
                        r["prompt_eval_count"] < fill_tokens * 0.5
                    )
                    if run_idx == 0:
                        vinfo = get_vram_info(model)
                        layers = get_offload_layers()
                        r["vram"] = vinfo
                        r["layers_offloaded"] = layers

                    status = "TRUNC" if r["context_truncated"] else "OK"
                    print(f"      Run {run_idx+1}: {status}  gen={r['gen_tok_s']:>6.1f}  "
                          f"prefill={r['prefill_tok_s']:>6.1f}  "
                          f"TTFT={r['ttft_s']:.1f}s  "
                          f"prompt_eval={r['prompt_eval_count']:,}  "
                          f"swap={r['mem_after'].get('swap_used_mb', '?')}MB")
                else:
                    print(f"      Run {run_idx+1}: {r['status']} ({r.get('error', '?')[:60]})")
                    # Don't continue runs if first one fails
                    results.append(r)
                    if run_idx == 0:
                        break
                    continue

                results.append(r)
                time.sleep(2)

        save_results(results, f"phase3-context-{TIMESTAMP}.json")

    return results


# ─── Phase 4: Quality Assessment ───────────────────────────────────────────

def check_quality(task_name, response, checks):
    """Score a response against objective checks. Returns dict of scores."""
    scores = {}
    text = response.strip()

    if "keywords" in checks:
        found = sum(1 for kw in checks["keywords"] if kw.lower() in text.lower())
        scores["keywords_found"] = found
        scores["keywords_total"] = len(checks["keywords"])
        scores["keywords_pass"] = found >= len(checks["keywords"]) * 0.5

    if "valid_json" in checks and checks["valid_json"]:
        # Try to extract JSON from response (model may wrap in markdown)
        json_text = text
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0].strip()
        try:
            parsed = json.loads(json_text)
            scores["valid_json"] = True
            if "required_keys" in checks:
                present = sum(1 for k in checks["required_keys"] if k in parsed)
                scores["keys_found"] = present
                scores["keys_total"] = len(checks["required_keys"])
                scores["keys_pass"] = present == len(checks["required_keys"])
            if "expected_values" in checks:
                matched = 0
                for k, v in checks["expected_values"].items():
                    if k in parsed:
                        pv = parsed[k]
                        if isinstance(v, str) and isinstance(pv, str):
                            if v.lower() in pv.lower():
                                matched += 1
                        elif pv == v:
                            matched += 1
                scores["values_matched"] = matched
                scores["values_total"] = len(checks["expected_values"])
        except (json.JSONDecodeError, IndexError):
            scores["valid_json"] = False

    if "has_numbered_items" in checks:
        expected = checks["has_numbered_items"]
        # Count numbered items (e.g., "1.", "2.", etc.)
        items = re.findall(r'^\s*\d+[\.\)]\s', text, re.MULTILINE)
        scores["numbered_items"] = len(items)
        scores["items_expected"] = expected
        scores["items_pass"] = len(items) == expected

    if "contains_number" in checks:
        target = str(checks["contains_number"])
        scores["contains_target"] = target in text
        # Also check for the number anywhere
        numbers = re.findall(r'\d+', text)
        scores["numbers_found"] = numbers[:10]

    if "min_sentences" in checks:
        # Rough sentence count
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        scores["sentence_count"] = len(sentences)
        scores["sentences_pass"] = (
            checks.get("min_sentences", 0) <= len(sentences) <= checks.get("max_sentences", 999)
        )

    return scores


def phase_quality(models=None, runs=3):
    """Production models, 5 quality tasks × N runs."""
    models = models or PRODUCTION_MODELS
    print(f"\n{'='*70}")
    print(f"  PHASE 4: Quality Assessment — {len(models)} models × {len(QUALITY_TASKS)} tasks × {runs} runs")
    print(f"{'='*70}")

    results = []
    for model in models:
        name = short_name(model)
        print(f"\n  Model: {name}")
        unload_all()

        # Warm up model
        print("    Warming up...", end=" ", flush=True)
        warmup = run_generate(model, "Hello", num_ctx=4096, num_predict=10, timeout=120)
        if warmup["status"] != "OK":
            print(f"FAIL: {warmup.get('error', '?')[:60]}")
            continue
        print("OK")

        for task_name, task in QUALITY_TASKS.items():
            print(f"\n    Task: {task_name}")
            for run_idx in range(runs):
                # Use /think:false for qwen3.5 models to avoid empty responses
                r = run_generate(model, task["prompt"], num_ctx=4096, num_predict=300, timeout=120)
                r["model"] = model
                r["task"] = task_name
                r["run"] = run_idx + 1
                r["phase"] = "quality"
                r["timestamp"] = datetime.datetime.now().isoformat()

                if r["status"] == "OK":
                    r["quality_scores"] = check_quality(task_name, r["response"], task["checks"])
                    scores_str = json.dumps(r["quality_scores"], separators=(",", ":"))
                    print(f"      Run {run_idx+1}: gen={r['gen_tok_s']:.1f} tok/s  scores={scores_str}")
                else:
                    print(f"      Run {run_idx+1}: FAIL")

                results.append(r)
                time.sleep(0.5)

        save_results(results, f"phase4-quality-{TIMESTAMP}.json")

    return results


# ─── Phase 5: Cold Start ───────────────────────────────────────────────────

def phase_cold(models=None, runs=3):
    """Production models, cold-start TTFT."""
    models = models or PRODUCTION_MODELS
    print(f"\n{'='*70}")
    print(f"  PHASE 5: Cold Start TTFT — {len(models)} models × {runs} runs")
    print(f"{'='*70}")

    results = []
    for model in models:
        name = short_name(model)
        print(f"\n  Model: {name}")

        for run_idx in range(runs):
            # Fully unload — ensure model is NOT in memory
            unload_all()
            time.sleep(5)  # Extra wait for full eviction

            t0 = time.time()
            r = run_generate(model, STANDARD_PROMPT, num_ctx=4096, num_predict=20, timeout=180)
            cold_total = time.time() - t0

            r["model"] = model
            r["run"] = run_idx + 1
            r["phase"] = "cold"
            r["cold_total_s"] = round(cold_total, 2)
            r["timestamp"] = datetime.datetime.now().isoformat()

            if r["status"] == "OK":
                print(f"    Run {run_idx+1}: cold_total={cold_total:.1f}s  "
                      f"load={r['load_s']}s  TTFT={r['ttft_s']:.1f}s  "
                      f"gen={r['gen_tok_s']:.1f} tok/s")
            else:
                print(f"    Run {run_idx+1}: FAIL ({r.get('error', '?')[:60]})")

            results.append(r)
            # Must unload before next run
            unload_all()
            time.sleep(3)

        save_results(results, f"phase5-cold-{TIMESTAMP}.json")

    return results


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BC-250 Rigorous Benchmark Suite")
    parser.add_argument("--phase", required=True,
                       choices=["perf", "stats", "context", "quality", "cold", "all"],
                       help="Which benchmark phase to run")
    parser.add_argument("--models", type=str, default=None,
                       help="Comma-separated model names (overrides default list)")
    parser.add_argument("--runs", type=int, default=None,
                       help="Override number of runs (default: 3 for stats/quality, 2 for context)")
    args = parser.parse_args()

    models = args.models.split(",") if args.models else None

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"BC-250 Rigorous Benchmark Suite")
    print(f"Phase: {args.phase}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Started: {datetime.datetime.now().isoformat()}")

    all_results = {}

    if args.phase in ("perf", "all"):
        all_results["perf"] = phase_perf(models)

    if args.phase in ("stats", "all"):
        runs = args.runs or 3
        all_results["stats"] = phase_stats(models, runs=runs)

    if args.phase in ("context", "all"):
        runs = args.runs or 2
        all_results["context"] = phase_context(models, runs=runs)

    if args.phase in ("quality", "all"):
        runs = args.runs or 3
        all_results["quality"] = phase_quality(models, runs=runs)

    if args.phase in ("cold", "all"):
        runs = args.runs or 3
        all_results["cold"] = phase_cold(models, runs=runs)

    print(f"\n{'='*70}")
    print(f"  COMPLETE — {datetime.datetime.now().isoformat()}")
    print(f"  Results in: {RESULTS_DIR}")
    print(f"{'='*70}")

    # Save combined results
    save_results(all_results, f"combined-{args.phase}-{TIMESTAMP}.json")


if __name__ == "__main__":
    main()
