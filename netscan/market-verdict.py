#!/usr/bin/env python3
"""market-verdict.py - LLM reading of prediction-market resolution rules.

The analyst stage. For each candidate market it asks the local model ONE
question that a language model is actually good at: "read these resolution
rules and say what makes this resolve YES, how ambiguous they are, and what
probabilities you would put on it". The market price is passed in as the
crowd's baseline, and the model is required to justify any gap from it.

This is deliberately NOT a trader: it emits a probability and its reasoning,
with the price that was on screen when it decided, and nothing else happens.
market-score.py is what later checks those probabilities against reality.

Output: /opt/netscan/data/markets/
  - verdicts-YYYYMMDD.jsonl   append-only: one line per verdict
  - latest-verdicts.json      merged view, read by market-score.py

Usage:
  market-verdict.py --selftest                       # offline, stubbed model
  market-verdict.py --dry-run --max-markets 3        # see the prompts
  market-verdict.py --max-markets 60 --kinds fuzzy   # the night run

Schedule: nightly, after market-collect.py. Budget ~60 markets (~1.5 h at
~29 tok/s on this board, with the GPU shared with chat and media jobs).

Location on bc250: /opt/netscan/market-verdict.py
"""

import argparse
import hashlib
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_sanitize import LLM_MODEL, LLM_URL, strip_to_json  # noqa: E402

DEFAULT_DATA = Path("/opt/netscan/data/markets")
REQUIRED_KEYS = ("market_id", "p_yes", "resolution_reading", "resolves_yes_if",
                 "ambiguity", "confidence", "key_evidence", "counter_case")
RULE_CHARS_MAX = 4000
KIND_ORDER = {"fuzzy": 0, "event": 1, "feed": 2}

SYSTEM = """You analyse prediction markets. You do not chat, you do not explain \
yourself outside the JSON, and you never invent facts.

Your job, in order:
1. Read the RESOLUTION RULES literally and state what makes the market resolve YES.
2. Rate how ambiguous the rules are: 0.0 = a single unambiguous source decides it, \
1.0 = the wording admits several readings.
3. Give your probability that it resolves YES (p_yes), and rate your own confidence.
4. If p_yes differs from CURRENT PRICE by more than 0.05, say exactly why in \
price_gap_reason. The price is the crowd's estimate: disagreeing with it is allowed, \
unjustified disagreement is not.
5. State the strongest case for NO in counter_case. If you cannot articulate one, \
your p_yes is too high.

Rules for evidence: every claim must be something you can point at — the rules text \
itself, the resolution source, or a fact you are confident about. If you would be \
guessing, put it in counter_case instead, or set confidence low. Never claim to have \
read a URL.

Answer with ONE JSON object, no markdown fences, no commentary:
{
  "market_id": "<echo the id you were given>",
  "p_yes": <float 0.0-1.0>,
  "resolution_reading": "<what the rules say, 1-3 sentences>",
  "resolves_yes_if": "<one verifiable condition>",
  "ambiguity": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "key_evidence": [{"claim": "<fact>", "source": "<where it comes from>"}],
  "counter_case": "<strongest reason this resolves NO>",
  "price_gap_reason": "<why you differ from the price, or 'none' if you do not>"
}"""


def log(msg, quiet=False):
    if not quiet:
        print(f"  {msg}", flush=True)


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rules_hash(market):
    blob = f"{market.get('question')}|{market.get('description')}|{market.get('resolution_source')}"
    return hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()[:12]


# ── The model call (thin wrapper, so the selftest can stub it) ─────────────


# The raw response of the last call, kept for the rejection dumps: a model that
# answers in reasoning_content looks identical to one that failed, unless you
# keep both halves.
_LAST_RESPONSE = {}


def _raw_llm(system, user, max_tokens, think=True):
    """Call llama-swap directly, rather than through llm_sanitize.call_llm.

    Two reasons, both learned the hard way on this box:

    1. gemma-4-12B is a reasoning model whose chat template honours
       `chat_template_kwargs: {"enable_thinking": false}`. Measured on the box:
       the same trivial question costs 6 completion tokens instead of 102, and
       the answer lands in `content` instead of being buried in
       `reasoning_content` that the model never finishes. The repo's usual
       "/nothink" prefix does NOT work for this model (it is a Qwen convention);
       without a working switch, a real market spent the whole 2500-token budget
       thinking and then emitted a bare "{".
    2. call_llm() returns a single string, so a rejection cannot show whether the
       text arrived in `content` or `reasoning_content`.
    """
    import urllib.error
    import urllib.request

    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    _LAST_RESPONSE.clear()
    _LAST_RESPONSE["think"] = bool(think)
    req = urllib.request.Request(f"{LLM_URL}/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            payload = json.loads(resp.read())
    except Exception as ex:  # llama-swap down, model failed to load, HTTP 500...
        _LAST_RESPONSE.update({"http_error": f"{type(ex).__name__}: {ex}"})
        return ""

    message = (payload.get("choices") or [{}])[0].get("message", {}) or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    _LAST_RESPONSE.update({
        "content": content,
        "reasoning": reasoning,
        "usage": payload.get("usage") or {},
        "timings": payload.get("timings") or {},
        "content_empty": not content.strip(),
    })
    # Same courtesy fallback as llm_sanitize: an empty `content` with reasoning
    # present is not an empty answer, and validation will judge it.
    return content if content.strip() else reasoning


def llm_health(timeout=10):
    """Is llama-swap answering? Same pre-flight idea as the bot's chat path."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{LLM_URL}/models", timeout=timeout) as resp:
            payload = json.loads(resp.read())
        return True, [m.get("id") for m in payload.get("data", [])]
    except Exception as ex:
        return False, f"{type(ex).__name__}: {ex}"


# ── Prompt ─────────────────────────────────────────────────────────────────


def build_prompt(market):
    rules = (market.get("description") or "(no description provided)")[:RULE_CHARS_MAX]
    price = market.get("price_yes")
    lines = [
        f"MARKET ID: {market.get('id')}",
        f"QUESTION: {market.get('question')}",
        f"RESOLUTION SOURCE (as listed): {market.get('resolution_source') or 'not listed'}",
        f"CLOSES (UTC): {market.get('end_date')}  ({_fmt(market.get('hours_to_end'), 1)}h from now)",
        f"OUTCOMES: {market.get('outcomes')}",
        f"CURRENT PRICE (crowd, 0-1): {price}",
        f"BEST BID/ASK: {market.get('best_bid')} / {market.get('best_ask')}"
        f"   (spread {market.get('spread')} — this is what a taker pays to get in)",
        f"24h VOLUME: {market.get('volume24h')}   LIQUIDITY: {market.get('liquidity')}",
        f"MARKET CLASS: {market.get('kind')} (feed = resolved by a price print, "
        f"event = a measurable real-world event, fuzzy = judgement)",
        "",
        "RESOLUTION RULES (verbatim):",
        rules,
    ]
    return "\n".join(lines)


# ── Validation ─────────────────────────────────────────────────────────────


def validate_verdict(obj, market):
    """Return a list of problems; empty means acceptable. Strict on purpose:
    a malformed probability is worse than no probability at all."""
    problems = []
    if not isinstance(obj, dict):
        return ["not a JSON object"]
    for key in REQUIRED_KEYS:
        if key not in obj:
            problems.append(f"missing key '{key}'")
    if str(obj.get("market_id")) != str(market.get("id")):
        problems.append(f"market_id {obj.get('market_id')!r} != {market.get('id')!r}")
    for key in ("p_yes", "ambiguity", "confidence"):
        value = obj.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{key} is not a number ({value!r})")
        elif not 0.0 <= float(value) <= 1.0:
            problems.append(f"{key}={value} outside 0..1")
    for key in ("resolution_reading", "resolves_yes_if", "counter_case"):
        if not isinstance(obj.get(key), str) or not obj.get(key, "").strip():
            problems.append(f"{key} is empty")
    evidence = obj.get("key_evidence")
    if not isinstance(evidence, list) or not evidence:
        problems.append("key_evidence is not a non-empty list")
    elif not all(isinstance(e, dict) and e.get("claim") for e in evidence):
        problems.append("key_evidence entries need a 'claim' field")
    return problems


def extract_json(text, prefer_keys=("market_id", "p_yes")):
    """Parse the JSON object that is actually the verdict, from a model answer.

    A greedy regex (`\\{[\\s\\S]*\\}`, which is what strip_to_json does) is not
    enough: a reasoning model cheerfully writes an object and then comments on
    it, or writes a scratch object first and the real answer second, and a
    greedy match swallows the lot and dies with "Extra data" — which is exactly
    how the first live run failed.

    So: collect every complete top-level object with json.JSONDecoder.raw_decode
    (which stops at the end of the first complete value), prefer one carrying
    the verdict keys, look one level into wrappers like {"verdict": {...}}, and
    otherwise hand back the first object so validation can explain what is wrong.
    """
    cleaned = strip_to_json(text)
    decoder = json.JSONDecoder()
    found = []
    for start, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            found.append(obj)

    if not found:
        raise json.JSONDecodeError("no complete JSON object found", cleaned or "", 0)

    def looks_like_verdict(value):
        return isinstance(value, dict) and all(k in value for k in prefer_keys)

    for obj in found:
        if looks_like_verdict(obj):
            return obj
    for obj in found:  # unwrap {"verdict": {...}} / {"analysis": ..., "output": {...}}
        for value in obj.values():
            if looks_like_verdict(value):
                return value
    return found[0]


def ask_model(market, max_tokens=2500, quiet=False, llm=_raw_llm, think=True):
    """One market -> (verdict_dict, meta) or (None, meta).

    One repair attempt: models routinely emit near-JSON, and a second, pointed
    question ('your output failed validation for these reasons') fixes it far
    more often than a blind retry with the same prompt.
    """
    user = build_prompt(market)
    meta = {"asked_at": now_iso(), "repaired": False, "latency_s": None,
            "error": None, "raw_chars": 0, "raw": ""}

    for attempt in (1, 2):
        t0 = time.time()
        raw = llm(SYSTEM, user if attempt == 1 else user + "\n\n" + meta["repair_hint"],
                  max_tokens, think)
        meta["latency_s"] = round(time.time() - t0, 1)
        meta["raw_chars"] = len(raw or "")
        meta["raw"] = (raw or "")[:6000]
        meta["response"] = dict(_LAST_RESPONSE)
        if not raw:
            meta["error"] = ("empty model output — a reasoning model that runs out "
                             "of tokens while thinking returns nothing; raise "
                             "--max-tokens (or use --no-think)")
            return None, meta

        try:
            obj = extract_json(raw)
        except json.JSONDecodeError as ex:
            meta["repair_hint"] = (f"Your previous answer was not valid JSON ({ex}). "
                                   "Return only the JSON object, nothing else.")
            meta["repaired"] = True
            meta["error"] = f"invalid JSON: {ex}"
            continue

        problems = validate_verdict(obj, market)
        if not problems:
            meta["error"] = None
            return obj, meta
        meta["error"] = "; ".join(problems)
        meta["repair_hint"] = ("Your previous answer failed validation: "
                               f"{'; '.join(problems)}. Return the corrected JSON object.")
        meta["repaired"] = True
        log(f"verdict rejected ({meta['error'][:90]}) — one repair attempt", quiet)

    return None, meta


# ── Selection ──────────────────────────────────────────────────────────────


def load_markets(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return doc.get("markets", []), doc


def load_seen(data_dir, refresh_hours=24.0):
    """Market ids already verdicted recently (verdicts are per market, not per
    run: re-asking the same market daily is a waste of the GPU budget)."""
    seen = {}
    cutoff = time.time() - refresh_hours * 3600
    for path in sorted(Path(data_dir).glob("verdicts-*.jsonl")):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                ts = rec.get("asked_at", "")
                try:
                    epoch = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
                        tzinfo=timezone.utc).timestamp()
                except ValueError:
                    epoch = 0
                if epoch >= cutoff:
                    seen[rec.get("market_id")] = ts
        except (json.JSONDecodeError, OSError):
            continue
    return seen


def pick(markets, seen, kinds=None, min_hours=2.0, max_hours=0.0, max_markets=60):
    """Priority: judgement markets first, then the most liquid, then the ones
    with room left before they close. Logs why the rest were skipped."""
    out, dropped = [], {"seen": 0, "kind": 0, "horizon": 0, "no_price": 0}
    for m in markets:
        if str(m.get("id")) in seen:
            dropped["seen"] += 1
            continue
        if kinds and m.get("kind") not in kinds:
            dropped["kind"] += 1
            continue
        hours = m.get("hours_to_end")
        if hours is None or hours < min_hours:
            dropped["horizon"] += 1
            continue
        if max_hours and hours > max_hours:
            dropped["horizon"] += 1
            continue
        if m.get("price_yes") is None:
            dropped["no_price"] += 1
            continue
        out.append(m)
    out.sort(key=lambda m: (KIND_ORDER.get(m.get("kind"), 9),
                            -(m.get("liquidity") or 0),
                            -(m.get("hours_to_end") or 0)))
    return out[:max_markets], dropped


# ── Output ─────────────────────────────────────────────────────────────────


def write_verdicts(data_dir, records):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    jsonl = data_dir / f"verdicts-{stamp}.jsonl"
    with jsonl.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    latest = data_dir / "latest-verdicts.json"
    existing = {}
    if latest.exists():
        try:
            for rec in json.loads(latest.read_text(encoding="utf-8")).get("verdicts", []):
                existing[rec.get("market_id")] = rec
        except (json.JSONDecodeError, OSError):
            existing = {}
    for rec in records:
        existing[rec.get("market_id")] = rec

    doc = {"generated_at": now_iso(), "model": LLM_MODEL,
           "verdicts": sorted(existing.values(), key=lambda r: r.get("asked_at") or "")}
    latest.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    return jsonl, latest


def _fmt(value, nd=3):
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def doubtful(record, ambiguity=0.35, confidence=0.60):
    """Is this verdict shaky enough to be worth a second, thinking pass?

    Doubt is measurable: the model reports how ambiguous it found the rules and
    how confident it is, and a verdict that needed a repair or cites fewer than
    two pieces of evidence is weak by its own standard. Cheap pass first,
    thinking only where it might change something — and because both answers are
    kept in the JSONL, market-score.py can later check which mode was the better
    calibrated one, which is a question nobody can answer from a single mode.
    """
    verdict = record.get("verdict") or {}
    return (float(verdict.get("ambiguity") or 0.0) >= ambiguity
            or float(verdict.get("confidence") or 1.0) <= confidence
            or bool(record.get("repaired"))
            or len(verdict.get("key_evidence") or []) < 2)


def record_for(market, verdict, meta, mode):
    """One verdict record: the answer, the reasoning, and the world it saw."""
    return {
        "market_id": market["id"],
        "question": market.get("question"),
        "kind": market.get("kind"),
        "url": market.get("url"),
        "event_title": market.get("event_title"),
        "end_date": market.get("end_date"),
        "asked_at": meta["asked_at"],
        "latency_s": meta["latency_s"],
        "repaired": meta["repaired"],
        "mode": mode,
        "model": LLM_MODEL,
        "rules_hash": rules_hash(market),
        "price_yes_at_decision": market.get("price_yes"),
        "spread_at_decision": market.get("spread"),
        "fee_bps_at_decision": market.get("fee_bps"),
        "edge_required": market.get("edge_required"),
        "verdict": verdict,
    }


def print_report(records, dropped, elapsed, quiet=False):
    log(f"verdicts written: {len(records)}  in {elapsed / 60:.1f} min", quiet)
    if any(dropped.values()):
        log("skipped: " + ", ".join(f"{k}={v}" for k, v in dropped.items() if v), quiet)
    if quiet:
        return
    print(f"\n  {'market_id':<9} {'m':>1} {'kind':<6} {'p_yes':>6} {'price':>6} {'gap':>7} "
          f"{'amb':>5} {'conf':>5} {'s':>5}  question")
    print("  " + "-" * 106)
    for rec in records:
        v = rec["verdict"]
        price = rec.get("price_yes_at_decision")
        gap = (v["p_yes"] - price) if price is not None else None
        print(f"  {rec['market_id']:<9} {'T' if rec.get('mode') == 'think' else '':>1} "
              f"{rec.get('kind', '-'):<6} "
              f"{_fmt(v['p_yes'], 6)} {_fmt(price, 6)} "
              f"{(('%+' + _fmt(gap, 3)) if gap is not None else '-'):>7} "
              f"{_fmt(v['ambiguity'], 5)} {_fmt(v['confidence'], 5)} "
              f"{rec.get('latency_s') or 0:>5.0f}  {(rec.get('question') or '')[:44]}")


# ── Selftest (offline, stubbed model) ──────────────────────────────────────


STUB_GOOD = json.dumps({
    "market_id": "900002", "p_yes": 0.30,
    "resolution_reading": "Yes requires the Sejm to pass the bill and the President "
                           "to sign it before 30 Nov.",
    "resolves_yes_if": "Bill passed by the Sejm and signed by the President before 2026-11-30",
    "ambiguity": 0.25, "confidence": 0.6,
    "key_evidence": [{"claim": "The rules name a hard deadline", "source": "rules text"}],
    "counter_case": "The bill can be withdrawn at any point before the vote",
    "price_gap_reason": "Timing risk of the presidential signature",
})


def selftest():
    """Offline checks of the prompt/parse/validate/repair path, no model needed."""
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="market-verdict-selftest-"))
    market = {"id": "900002", "question": "Will the Polish Sejm pass the media bill?",
              "description": "Resolves Yes if ... before November 30, 2026.",
              "resolution_source": "Official Sejm voting records",
              "end_date": "2026-11-30T23:59:00Z", "hours_to_end": 1000.0,
              "outcomes": ["Yes", "No"], "price_yes": 0.34, "best_bid": 0.31,
              "best_ask": 0.37, "spread": 0.06, "volume24h": 400.0,
              "liquidity": 3200.0, "kind": "fuzzy", "url": "https://x/1"}

    prompt = build_prompt(market)
    ok &= "RESOLUTION RULES" in prompt and "0.34" in prompt
    print(f"  build_prompt -> {len(prompt)} chars, rules + price present")

    # 1. A clean answer is accepted.
    verdict, meta = ask_model(market, quiet=True, llm=lambda *a: STUB_GOOD)
    ok &= verdict is not None and not meta["repaired"] and verdict["p_yes"] == 0.30
    print(f"  valid JSON      -> accepted, repaired={meta['repaired']}, "
          f"p_yes={verdict and verdict['p_yes']}")

    # 2. Markdown fences / preamble are stripped by strip_to_json.
    fenced = "Sure, here you go:\n```json\n" + STUB_GOOD + "\n```\n"
    verdict2, meta2 = ask_model(market, quiet=True, llm=lambda *a: fenced)
    ok &= verdict2 is not None and not meta2["repaired"]
    print(f"  fenced JSON     -> accepted, repaired={meta2['repaired']}")

    # 3. Broken JSON triggers exactly one repair attempt, then succeeds.
    calls = []

    def flaky(system, user, max_tokens, think=True):
        calls.append(user)
        return "{not json at all" if len(calls) == 1 else STUB_GOOD

    verdict3, meta3 = ask_model(market, quiet=True, llm=flaky)
    ok &= verdict3 is not None and meta3["repaired"] and len(calls) == 2
    ok &= "not valid JSON" in calls[1]
    print(f"  broken JSON     -> repaired after {len(calls)} calls, "
          f"hint sent back to the model")

    # 4. Out-of-range probability triggers repair; a stubborn model is rejected.
    bad = json.loads(STUB_GOOD)
    bad["p_yes"] = 1.7
    verdict4, meta4 = ask_model(market, quiet=True,
                                llm=lambda *a: json.dumps(bad))
    ok &= verdict4 is None and "p_yes" in (meta4["error"] or "")
    print(f"  p_yes=1.7 always -> rejected (error: {meta4['error']})")

    # 5. Missing key is caught by validation.
    problems = validate_verdict({"market_id": "900002", "p_yes": 0.5}, market)
    ok &= any("missing key" in p for p in problems)
    print(f"  sparse object   -> {len(problems)} validation problems")

    # 6. JSON extraction: a reasoning model writes the object and then keeps
    # talking, or writes a scratch object first — both killed the first live run.
    trailing = "Here is my answer:\n" + STUB_GOOD + "\n\nI hope that helps. Note that " \
               "the deadline may slip, so treat 0.30 as approximate."
    two_objects = '{"thought": "let me think"}\n' + STUB_GOOD
    nested = '{"analysis": {"note": "thinking"}, "verdict": ' + STUB_GOOD + '}'
    checks = {"trailing prose": extract_json(trailing),
              "scratch object first": extract_json(two_objects),
              "wrapped in a dict": extract_json(nested)}
    for label, obj in checks.items():
        good = obj.get("p_yes") == 0.30 and obj.get("market_id") == "900002"
        ok &= good
        print(f"  json extraction -> {label:<21} {'OK' if good else 'FAIL'} "
              f"(p_yes={obj.get('p_yes')})")

    # 7. Escalation: doubt is measurable, so only doubtful verdicts get the
    # expensive thinking pass.
    clean = record_for(market, {"ambiguity": 0.1, "confidence": 0.9,
                                "key_evidence": [{"claim": "a"}, {"claim": "b"}]},
                       {"asked_at": now_iso(), "latency_s": 12.0, "repaired": False},
                       mode="nothink")
    unclear = record_for(market, {"ambiguity": 0.6, "confidence": 0.9,
                                  "key_evidence": [{"claim": "a"}, {"claim": "b"}]},
                         {"asked_at": now_iso(), "latency_s": 12.0, "repaired": False},
                         mode="nothink")
    thin = record_for(market, {"ambiguity": 0.1, "confidence": 0.9,
                               "key_evidence": [{"claim": "a"}]},
                      {"asked_at": now_iso(), "latency_s": 12.0, "repaired": False},
                      mode="nothink")
    ok &= not doubtful(clean) and doubtful(unclear) and doubtful(thin)
    ok &= clean["mode"] == "nothink" and clean["market_id"] == "900002"
    print(f"  escalation      -> doubtful: clean={doubtful(clean)} "
          f"ambiguous={doubtful(unclear)} thin_evidence={doubtful(thin)} "
          f"(want False/True/True)")

    # 8. Selection: priority + skips.
    markets = [dict(market, id="1", kind="feed", liquidity=90000),
               dict(market, id="2", kind="fuzzy", liquidity=1000),
               dict(market, id="3", kind="event", liquidity=5000),
               dict(market, id="4", kind="fuzzy", liquidity=100, hours_to_end=0.5)]
    picked, dropped = pick(markets, seen={"3": "x"}, max_markets=10)
    ok &= [m["id"] for m in picked] == ["2", "1"]
    ok &= dropped["seen"] == 1 and dropped["horizon"] == 1
    print(f"  pick            -> {[m['id'] for m in picked]} (fuzzy first, "
          f"seen/horizon dropped: {dropped['seen']}/{dropped['horizon']})")

    # 7. Writers: append-only JSONL + merged latest view.
    rec = {"market_id": "900002", "verdict": verdict, "asked_at": now_iso(),
           "price_yes_at_decision": 0.34, "kind": "fuzzy", "question": market["question"]}
    jsonl, latest = write_verdicts(tmp, [rec])
    ok &= jsonl.exists() and latest.exists()
    doc = json.loads(latest.read_text(encoding="utf-8"))
    ok &= len(doc["verdicts"]) == 1
    write_verdicts(tmp, [dict(rec, market_id="900003")])
    doc = json.loads(latest.read_text(encoding="utf-8"))
    ok &= len(doc["verdicts"]) == 2
    print(f"  write_verdicts  -> {jsonl.name} + {latest.name} "
          f"({len(doc['verdicts'])} merged, no dupes)")

    up = llm_health(timeout=3)
    print(f"  llm_health (live, may fail offline): ok={up[0]} {str(up[1])[:60]}")

    print(f"  temp dir: {tmp}")
    print(f"\n  selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ── Main ───────────────────────────────────────────────────────────────────


def run(args):
    data_dir = Path(args.data_dir)
    markets, doc = load_markets(args.markets_file or data_dir / "latest-markets.json")
    log(f"loaded {len(markets)} markets from "
        f"{args.markets_file or data_dir / 'latest-markets.json'}", args.quiet)
    gb = (doc.get("geoblock") or {})
    if gb.get("blocked") is not None:
        log(f"geoblock at collection time: blocked={gb.get('blocked')} "
            f"country={gb.get('country')}", args.quiet)

    seen = load_seen(data_dir, refresh_hours=args.refresh_hours)
    kinds = tuple(k.strip() for k in args.kinds.split(",")) if args.kinds else None
    picked, dropped = pick(markets, seen, kinds=kinds, min_hours=args.min_hours,
                           max_hours=args.max_hours, max_markets=args.max_markets)
    log(f"to analyse: {len(picked)} (budget {args.max_markets}, "
        f"runtime cap {args.max_runtime / 60:.0f} min)", args.quiet)

    if args.dry_run:
        for m in picked:
            print(f"\n  --- {m['id']} {m['kind']} p={m['price_yes']} "
                  f"{str(m.get('question'))[:70]}")
            print("  " + build_prompt(m).replace("\n", "\n  ")[:1200])
        print_report([], dropped, 0, args.quiet)
        return 0

    if not picked:
        print_report([], dropped, 0, args.quiet)
        return 0

    healthy, info = llm_health()
    if not healthy:
        log(f"llama-swap not answering at {LLM_URL} ({info}) — nothing to do", args.quiet)
        print_report([], dropped, 0, args.quiet)
        return 2
    log(f"llama-swap up, model={LLM_MODEL}", args.quiet)

    t_start = time.time()
    records, failures = [], 0
    by_id = {str(m["id"]): m for m in picked}
    for i, market in enumerate(picked, 1):
        elapsed = time.time() - t_start
        if elapsed > args.max_runtime:
            log(f"runtime cap reached after {i - 1} markets "
                f"({elapsed / 60:.1f} min) — stopping cleanly", args.quiet)
            break
        verdict, meta = ask_model(market, max_tokens=args.max_tokens,
                                  quiet=args.quiet, think=args.think)
        if verdict is None:
            failures += 1
            log(f"[{i}/{len(picked)}] {market['id']} REJECTED: {meta['error']}", args.quiet)
            if meta.get("raw"):
                # Keep what the model actually said, split into the half that
                # answered and the half that thought. Rejections are the only
                # way to see this model's real failure modes, and guessing at
                # them from an error string wastes whole evenings.
                dump_dir = Path(args.data_dir) / "rejected"
                dump_dir.mkdir(parents=True, exist_ok=True)
                resp = meta.get("response") or {}
                stamp = meta["asked_at"].replace(":", "")
                (dump_dir / f"{market['id']}-{stamp}.txt").write_text(
                    f"# error: {meta['error']}\n"
                    f"# market: {market.get('question')}\n"
                    f"# asked_at: {meta['asked_at']}  model: {LLM_MODEL}  "
                    f"think: {resp.get('think')}  repaired: {meta['repaired']}  "
                    f"latency_s: {meta['latency_s']}  usage: {json.dumps(resp.get('usage') or {})}\n"
                    f"# http_error: {resp.get('http_error')}\n"
                    f"# content ({len(resp.get('content') or '')} chars):\n"
                    f"{(resp.get('content') or meta.get('raw', ''))[:6000]}\n"
                    f"# reasoning ({len(resp.get('reasoning') or '')} chars):\n"
                    f"{(resp.get('reasoning') or '')[:3000]}\n", encoding="utf-8")
            continue
        records.append(record_for(market, verdict, meta, mode="nothink"))
        log(f"[{i}/{len(picked)}] {market['id']} p_yes={_fmt(verdict['p_yes'])} "
            f"price={_fmt(market.get('price_yes'))} "
            f"amb={_fmt(verdict['ambiguity'], 2)} "
            f"conf={_fmt(verdict['confidence'], 2)} "
            f"({meta['latency_s']}s{' repaired' if meta['repaired'] else ''})", args.quiet)

    # Second pass, only where it can change the answer. Measured on this box:
    # ~12 s per market without thinking versus ~130 s with it, so the thinking
    # budget goes to the verdicts that are doubtful by the model's own account.
    escalated = 0
    if args.escalate and records:
        shaky = [r for r in records
                 if doubtful(r, args.escalate_ambiguity, args.escalate_confidence)]
        if shaky:
            log(f"escalating {len(shaky)} doubtful verdict(s) to thinking mode "
                f"(max_tokens {args.think_max_tokens})", args.quiet)
        for rec in shaky:
            if time.time() - t_start > args.max_runtime:
                log("runtime cap reached during escalation — stopping cleanly", args.quiet)
                break
            market = by_id.get(str(rec["market_id"]))
            if not market:
                continue
            verdict, meta = ask_model(market, max_tokens=args.think_max_tokens,
                                      quiet=args.quiet, think=True)
            if verdict is None:
                log(f"  escalation rejected for {rec['market_id']}: {meta['error']}",
                    args.quiet)
                continue
            records.append(record_for(market, verdict, meta, mode="think"))
            escalated += 1
            log(f"  escalated {rec['market_id']}: p_yes {_fmt(rec['verdict']['p_yes'])} "
                f"-> {_fmt(verdict['p_yes'])}  amb {_fmt(rec['verdict']['ambiguity'], 2)} "
                f"-> {_fmt(verdict['ambiguity'], 2)}  ({meta['latency_s']}s)", args.quiet)
        if escalated:
            log(f"escalated {escalated} verdict(s); both modes stay in the JSONL so "
                f"market-score can compare which one calibrated better", args.quiet)

    if records:
        jsonl, latest = write_verdicts(data_dir, records)
        log(f"wrote {jsonl} and {latest}", args.quiet)
    if failures:
        log(f"{failures} market(s) rejected by validation and not written", args.quiet)
    print_report(records, dropped, time.time() - t_start, args.quiet)
    return 0


def probe(max_tokens=64, think=False):
    """Ask the model something trivial and show the RAW split.

    Worth having on this box specifically: the chat model is a *reasoning*
    model, so an answer can arrive in `reasoning_content` while `content` stays
    empty — and when the token budget runs out mid-thought, `content` can hold a
    single "{". That is invisible through call_llm()'s single return value, and
    it is exactly how the first live runs failed.
    """
    t0 = time.time()
    text = _raw_llm("You are terse.", "Reply with the single word OK", max_tokens, think)
    elapsed = time.time() - t0
    resp = dict(_LAST_RESPONSE)
    usage = resp.get("usage") or {}
    timings = resp.get("timings") or {}
    print(f"  model      : {LLM_MODEL}  (think={think})")
    print(f"  latency    : {elapsed:.1f}s   completion_tokens={usage.get('completion_tokens')}")
    if timings.get("predicted_per_second"):
        print(f"  speed      : {timings['predicted_per_second']:.1f} tok/s   "
              f"draft accepted: {timings.get('draft_n_accepted')}/{timings.get('draft_n')}")
    if resp.get("http_error"):
        print(f"  http error : {resp['http_error']}")
        return 1
    print(f"  content    : {(resp.get('content') or '')[:200]!r}")
    reasoning = resp.get("reasoning") or ""
    print(f"  reasoning  : {reasoning[:120]!r}{'...' if len(reasoning) > 120 else ''}")
    print(f"  returned   : {text[:80]!r}")
    if resp.get("content_empty") and reasoning:
        print("  >> text arrived only in reasoning_content: pass --max-tokens 6000+ "
              "to let it finish, or keep thinking off (the default)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="LLM analyst for prediction markets")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--markets-file", help="override latest-markets.json")
    ap.add_argument("--max-markets", type=int, default=60)
    ap.add_argument("--max-runtime", type=float, default=5400.0,
                    help="hard stop in seconds (default 1.5 h, the night budget)")
    ap.add_argument("--max-tokens", type=int, default=2500,
                    help="answer budget; a reasoning model needs room to think "
                         "before it writes the JSON")
    ap.add_argument("--think", action="store_true",
                    help="think on every market. Measured on this box: with "
                         "--max-tokens 2500 the model spent the entire budget "
                         "thinking and emitted no JSON at all, so this is off by "
                         "default and wants 6000+ tokens when enabled")
    ap.add_argument("--escalate", action="store_true",
                    help="after the cheap pass, re-ask with thinking the verdicts "
                         "that are doubtful (see --escalate-*)")
    ap.add_argument("--escalate-ambiguity", type=float, default=0.35,
                    help="escalate when the model reports at least this ambiguity")
    ap.add_argument("--escalate-confidence", type=float, default=0.60,
                    help="escalate when the model reports at most this confidence")
    ap.add_argument("--think-max-tokens", type=int, default=6000,
                    help="token budget for escalated verdicts")
    ap.add_argument("--probe", action="store_true",
                    help="one trivial call, show the raw response and exit")
    ap.add_argument("--kinds", help="feed,event,fuzzy (default: all, fuzzy first)")
    ap.add_argument("--min-hours", type=float, default=2.0,
                    help="skip markets closing sooner than this")
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--refresh-hours", type=float, default=24.0)
    ap.add_argument("--dry-run", action="store_true", help="print prompts, call nothing")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="offline stubbed-model checks")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.probe:
        return probe()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
