#!/usr/bin/env python3
"""market-score.py - does the model actually beat the market?

The verdicts in data/markets/verdicts-*.jsonl record a probability and the
price that was on screen at that moment. Once the market resolves, that pair
becomes a scored prediction, and the score is the only thing that answers the
question this whole pipeline exists to ask.

The benchmark is the market price, and that is a brutal benchmark: it is the
consensus of people with money at stake. A model that beats it on Brier score
over a few hundred markets has something; a model that merely agrees with it has
nothing, and agreeing with it is the normal, expected outcome.

Output: /opt/netscan/data/markets/
  - scoreboard.json      full numbers, machine readable
  - scoreboard.md        short summary, for the daily digest
  - scores-YYYYMMDD.jsonl  one line per scored verdict (an audit trail)

Usage:
  market-score.py --selftest                     # offline, fixture, no network
  market-score.py                                # score what has resolved
  market-score.py --resolved-file gamma.json     # offline: saved Gamma response
  market-score.py --min-observations 30

Schedule: daily, after market-collect.py. Needs resolved markets: expect
"nothing resolved yet" for the first days.

Location on bc250: /opt/netscan/market-score.py
"""

import argparse
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from polymarket import get_markets_by_id  # noqa: E402

DEFAULT_DATA = Path("/opt/netscan/data/markets")
EPS = 1e-6


def log(msg, quiet=False):
    if not quiet:
        print(f"  {msg}", flush=True)


# ── Scoring maths ──────────────────────────────────────────────────────────


def brier(p, y):
    """Brier score of one forecast: lower is better, 0 is perfect."""
    return (float(p) - float(y)) ** 2


def logloss(p, y):
    """Log loss of one forecast (clipped so a confident miss is not infinite)."""
    p = min(max(float(p), EPS), 1.0 - EPS)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p):
    p = min(max(float(p), EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def platt_fit(pairs, iterations=800, lr=0.5):
    """Fit sigmoid(a * logit(p) + b) by gradient descent, in plain Python.

    No numpy on this box, and no need: two parameters and a few hundred
    observations. Returns (a, b, calibrated_logloss). In-sample by construction
    — with a small n it flatters itself, so report the sample size next to it.
    """
    if len(pairs) < 2:
        return None, None, None
    xs = [logit(p) for p, _ in pairs]
    ys = [float(y) for _, y in pairs]
    a, b = 1.0, 0.0
    n = len(pairs)
    for _ in range(iterations):
        ga = gb = 0.0
        for x, y in zip(xs, ys):
            err = sigmoid(a * x + b) - y
            ga += err * x
            gb += err
        a -= lr * ga / n
        b -= lr * gb / n
    calibrated = mean([logloss(sigmoid(a * logit(p) + b), y) for p, y in pairs])
    return round(a, 4), round(b, 4), round(calibrated, 5) if calibrated is not None else None


def calibration_table(pairs, bins=10):
    """Predicted vs observed frequency: the honest question 'when you said 0.7,
    how often was it true?'"""
    table = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        bucket = [(p, y) for p, y in pairs if lo <= p < hi or (i == bins - 1 and p == 1.0)]
        if bucket:
            table.append({
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": len(bucket),
                "predicted": round(mean([p for p, _ in bucket]), 3),
                "observed": round(mean([y for _, y in bucket]), 3),
            })
    return table


# ── Resolutions ────────────────────────────────────────────────────────────


def yes_won(raw):
    """Did YES win? Resolved binary markets publish outcomePrices 1/0.

    Returns True/False for a resolved binary market, None otherwise (still open,
    not yet resolved, or not a two-outcome market we can score honestly).
    """
    outcomes = raw.get("outcomes")
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except json.JSONDecodeError:
            outcomes = []
    prices = raw.get("outcomePrices")
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except json.JSONDecodeError:
            prices = []
    if not isinstance(outcomes, list) or len(outcomes) != 2:
        return None
    if not isinstance(prices, list) or len(prices) != 2:
        return None
    try:
        p0, p1 = float(prices[0]), float(prices[1])
    except (TypeError, ValueError):
        return None
    if not raw.get("closed"):
        return None
    if p0 >= 0.99 and p1 <= 0.01:
        return True
    if p0 <= 0.01 and p1 >= 0.99:
        return False
    return None


def fetch_resolutions(market_ids, resolved_file=None):
    """{market_id: {"yes_won": bool, "rules_hash_source": raw}} for resolved ones."""
    if resolved_file:
        raw_markets = json.loads(Path(resolved_file).read_text(encoding="utf-8"))
        if isinstance(raw_markets, dict):
            raw_markets = raw_markets.get("markets", [])
    else:
        raw_markets = get_markets_by_id(market_ids)
    out = {}
    for raw in raw_markets or []:
        won = yes_won(raw)
        if won is None:
            continue
        out[str(raw.get("id"))] = {"yes_won": won, "raw": raw}
    return out


# ── Verdicts ───────────────────────────────────────────────────────────────


def load_verdicts(data_dir):
    records = []
    for path in sorted(Path(data_dir).glob("verdicts-*.jsonl")):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        except (json.JSONDecodeError, OSError) as ex:
            log(f"skipping {path.name}: {ex}")
    return records


def paper_trade(record, yes):
    """The trade this verdict would have made, and what it paid.

    Per share, in USDC: cost = the price actually paid (ask side) plus half the
    spread plus the taker fee; payoff = 1 if that side wins, 0 if not. Returns
    None when the model's gap does not clear the recorded edge_required, which
    is the whole point of that number.
    """
    verdict = record.get("verdict") or {}
    price = record.get("price_yes_at_decision")
    if price is None or verdict.get("p_yes") is None:
        return None
    gap = float(verdict["p_yes"]) - float(price)
    if abs(gap) <= float(record.get("edge_required") or 0.0):
        return None

    half_spread = float(record.get("spread_at_decision") or 0.0) / 2.0
    fee = float(record.get("fee_bps_at_decision") or 0.0) / 10000.0
    side_yes = gap > 0
    cost = (float(price) if side_yes else 1.0 - float(price)) + half_spread + fee
    won = yes if side_yes else (not yes)
    pnl = (1.0 if won else 0.0) - cost
    return {
        "side": "YES" if side_yes else "NO",
        "cost": round(cost, 5),
        "won": won,
        "pnl": round(pnl, 5),
        "roi": round(pnl / cost, 4) if cost else None,
        "gap": round(gap, 4),
    }


def score(records, resolutions):
    scored = []
    for rec in records:
        res = resolutions.get(str(rec.get("market_id")))
        if not res:
            continue
        y = 1 if res["yes_won"] else 0
        p = rec.get("verdict", {}).get("p_yes")
        price = rec.get("price_yes_at_decision")
        if p is None or price is None:
            continue
        scored.append({
            "market_id": str(rec.get("market_id")),
            "question": rec.get("question"),
            "kind": rec.get("kind"),
            "mode": rec.get("mode") or "nothink",
            "asked_at": rec.get("asked_at"),
            "model": rec.get("model"),
            "p_yes": p,
            "price_yes": price,
            "yes_won": bool(res["yes_won"]),
            "brier_model": round(brier(p, y), 6),
            "brier_market": round(brier(price, y), 6),
            "logloss_model": round(logloss(p, y), 6),
            "logloss_market": round(logloss(price, y), 6),
            "trade": paper_trade(rec, res["yes_won"]),
            "rules_changed": (rec.get("rules_hash") is not None and
                              _rules_hash_now(res["raw"]) not in (None, rec.get("rules_hash"))),
        })
    return scored


def _rules_hash_now(raw):
    import hashlib
    blob = f"{raw.get('question')}|{raw.get('description')}|{raw.get('resolutionSource')}"
    return hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()[:12]


def summarise(scored):
    """Model vs market, overall and split by answering mode."""
    def block(rows):
        if not rows:
            return None
        bm = mean([r["brier_model"] for r in rows])
        bk = mean([r["brier_market"] for r in rows])
        pairs = [(r["p_yes"], 1 if r["yes_won"] else 0) for r in rows]
        trades = [r["trade"] for r in rows if r["trade"]]
        wins = [t for t in trades if t["won"]]
        return {
            "n": len(rows),
            "brier_model": round(bm, 6),
            "brier_market": round(bk, 6),
            "brier_delta": round(bm - bk, 6),
            "logloss_model": round(mean([r["logloss_model"] for r in rows]), 6),
            "logloss_market": round(mean([r["logloss_market"] for r in rows]), 6),
            "beat_market_on": sum(1 for r in rows if r["brier_model"] < r["brier_market"]),
            "trades": len(trades),
            "trades_won": len(wins),
            "pnl_units": round(sum(t["pnl"] for t in trades), 4) if trades else 0.0,
            "roi": round(sum(t["pnl"] for t in trades) /
                         sum(max(t["cost"], EPS) for t in trades), 4) if trades else None,
            "calibration": calibration_table(pairs),
            "platt": platt_fit(pairs) if len(pairs) >= 10 else (None, None, None),
            "rules_changed": sum(1 for r in rows if r["rules_changed"]),
        }

    modes = sorted({r["mode"] for r in scored})
    return {
        "scored": len(scored),
        "overall": block(scored),
        "by_mode": {m: block([r for r in scored if r["mode"] == m]) for m in modes},
    }


# ── Report ─────────────────────────────────────────────────────────────────


def print_report(summary, min_obs, quiet=False):
    overall = summary.get("overall")
    if not overall:
        log(f"nothing to score yet: 0 of {summary['scored']} verdicts have a "
            f"resolved market behind them", quiet)
        return

    print(f"\n  scored verdicts : {summary['scored']}   "
          f"(minimum for a meaningful read: {min_obs})")
    rows = [("model", overall["brier_model"], overall["logloss_model"]),
            ("market", overall["brier_market"], overall["logloss_market"])]
    print(f"\n  {'':<8} {'Brier':>9} {'log loss':>10}   lower is better")
    print("  " + "-" * 40)
    for label, b, ll in rows:
        print(f"  {label:<8} {b:>9.5f} {ll:>10.5f}")
    verdict = ("the MODEL is ahead" if overall["brier_delta"] < 0
               else "the MARKET is ahead")
    print(f"\n  {verdict} on Brier by {abs(overall['brier_delta']):.5f} "
          f"(model beat the market on {overall['beat_market_on']}/{overall['n']} markets)")

    for mode, block in (summary.get("by_mode") or {}).items():
        if block:
            print(f"    mode {mode:<8} n={block['n']:<4} model {block['brier_model']:.5f} "
                  f"market {block['brier_market']:.5f}  "
                  f"(trades {block['trades']}, pnl {block['pnl_units']:+.2f} units)")

    if overall["trades"]:
        roi = f", roi {overall['roi']:+.1%}" if overall["roi"] is not None else ""
        print(f"\n  paper ledger    : {overall['trades']} trades, "
              f"{overall['trades_won']} won, pnl {overall['pnl_units']:+.2f} units{roi}")
    else:
        print("\n  paper ledger    : no verdict cleared its edge_required, so no "
              "trades — which is the system working, not failing")

    if overall["calibration"]:
        print("\n  calibration (model):")
        print(f"    {'bin':<9} {'n':>4} {'predicted':>10} {'observed':>9}")
        for row in overall["calibration"]:
            print(f"    {row['bin']:<9} {row['n']:>4} {row['predicted']:>10.3f} "
                  f"{row['observed']:>9.3f}")
    a, b, calibrated = overall["platt"]
    if a is not None:
        print(f"\n  Platt (in-sample) : a={a} b={b}  log loss "
              f"{overall['logloss_model']:.5f} -> {calibrated:.5f}")
    else:
        print(f"\n  Platt             : needs >= 10 scored verdicts "
              f"(have {overall['n']})")
    if overall["rules_changed"]:
        print(f"\n  warning         : {overall['rules_changed']} verdict(s) were made on "
              f"resolution rules that have since been edited")


def write_outputs(data_dir, summary, scored, quiet=False):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    doc = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           **summary}
    (data_dir / "scoreboard.json").write_text(
        json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    with (data_dir / f"scores-{stamp}.jsonl").open("a", encoding="utf-8") as fh:
        for row in scored:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    overall = summary.get("overall")
    if overall:
        md = [
            f"# Market scoreboard — {doc['generated_at']}",
            "",
            f"Scored verdicts: **{summary['scored']}**",
            "",
            "| | Brier | log loss |",
            "|---|---|---|",
            f"| model | {overall['brier_model']:.5f} | {overall['logloss_model']:.5f} |",
            f"| market | {overall['brier_market']:.5f} | {overall['logloss_market']:.5f} |",
            "",
            (f"**The model is ahead** by {abs(overall['brier_delta']):.5f} on Brier "
             f"(beat the market on {overall['beat_market_on']}/{overall['n']} markets)."
             if overall["brier_delta"] < 0 else
             f"**The market is ahead** by {abs(overall['brier_delta']):.5f} on Brier. "
             "This is the expected result and not a failure: the price is a hard "
             "benchmark."),
            "",
            f"Paper ledger: {overall['trades']} trades, {overall['trades_won']} won, "
            f"pnl {overall['pnl_units']:+.2f} units.",
        ]
        (data_dir / "scoreboard.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"wrote scoreboard.json + scoreboard.md ({len(scored)} scored)", quiet)


# ── Selftest (offline, exact numbers) ──────────────────────────────────────


def selftest():
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="market-score-selftest-"))

    # 1. The maths, against hand-computed values.
    ok &= abs(brier(0.8, 1) - 0.04) < 1e-9
    ok &= abs(brier(0.5, 1) - 0.25) < 1e-9
    ok &= abs(logloss(0.8, 1) - 0.22314) < 1e-4
    ok &= abs(logloss(0.5, 1) - 0.69315) < 1e-4
    ok &= logloss(0.0, 1) < 20  # clipped, not infinite
    print(f"  brier/logloss   -> (0.8,1)={brier(0.8, 1)} (0.5,1)={brier(0.5, 1)} "
          f"ll(0.8,1)={logloss(0.8, 1):.5f}")

    # 2. Resolution reading: 1/0 = decided, anything else = not scoreable.
    ok &= yes_won({"closed": True, "outcomes": '["Yes","No"]',
                   "outcomePrices": '["1","0"]'}) is True
    ok &= yes_won({"closed": True, "outcomes": '["Yes","No"]',
                   "outcomePrices": '["0","1"]'}) is False
    ok &= yes_won({"closed": False, "outcomes": '["Yes","No"]',
                   "outcomePrices": '["1","0"]'}) is None
    ok &= yes_won({"closed": True, "outcomes": '["A","B","C"]',
                   "outcomePrices": '["1","0","0"]'}) is None
    ok &= yes_won({"closed": True, "outcomes": '["Yes","No"]',
                   "outcomePrices": '["0.6","0.4"]'}) is None
    print("  yes_won         -> decided/undecided/not-binary handled")

    # 3. The paper ledger: cost, payoff, and the edge_required gate.
    base = {"market_id": "M1", "verdict": {"p_yes": 0.8}, "price_yes_at_decision": 0.5,
            "spread_at_decision": 0.02, "fee_bps_at_decision": 0.0, "edge_required": 0.05}
    trade = paper_trade(base, yes=True)
    ok &= trade["side"] == "YES" and abs(trade["cost"] - 0.51) < 1e-9
    ok &= abs(trade["pnl"] - 0.49) < 1e-9 and trade["won"] is True
    no_trade = paper_trade(dict(base, verdict={"p_yes": 0.52}), yes=True)
    ok &= no_trade is None
    losing = paper_trade(dict(base, verdict={"p_yes": 0.2}), yes=True)
    ok &= losing["side"] == "NO" and losing["won"] is False and losing["pnl"] < 0
    print(f"  paper_trade     -> YES cost {trade['cost']} pnl {trade['pnl']}; "
          f"no-edge -> {no_trade}; NO side pnl {losing['pnl']}")

    # 4. Score a tiny set end to end: 4 verdicts, 2 of them right, and verify
    # the model's Brier against the market's by hand.
    verdicts = [
        {"market_id": "1", "verdict": {"p_yes": 0.9}, "price_yes_at_decision": 0.5,
         "spread_at_decision": 0.01, "fee_bps_at_decision": 0, "edge_required": 0.05},
        {"market_id": "2", "verdict": {"p_yes": 0.1}, "price_yes_at_decision": 0.5,
         "spread_at_decision": 0.01, "fee_bps_at_decision": 0, "edge_required": 0.05},
    ]
    resolutions = {"1": {"yes_won": True, "raw": {}}, "2": {"yes_won": False, "raw": {}}}
    scored = score(verdicts, resolutions)
    ok &= len(scored) == 2
    ok &= abs(scored[0]["brier_model"] - 0.01) < 1e-9
    ok &= abs(scored[0]["brier_market"] - 0.25) < 1e-9
    ok &= abs(scored[1]["brier_model"] - 0.01) < 1e-9
    ok &= scored[0]["trade"]["won"] is True and scored[1]["trade"]["won"] is True
    summary = summarise(scored)
    ok &= summary["overall"]["beat_market_on"] == 2
    ok &= summary["overall"]["trades"] == 2 and summary["overall"]["trades_won"] == 2
    print(f"  score+summarise -> model brier {summary['overall']['brier_model']} vs "
          f"market {summary['overall']['brier_market']}, "
          f"2/2 beat the market, pnl {summary['overall']['pnl_units']:+.2f}")

    # 5. Calibration table and Platt on a deliberately miscalibrated set: the
    # model is under-confident, so the fit should improve the log loss.
    pairs = []
    for i in range(200):
        p = 0.30 + 0.40 * ((i % 10) / 9.0)           # predictions in [0.30, 0.70]
        true_p = min(0.98, max(0.02, 0.5 + 2.0 * (p - 0.5)))  # reality is sharper
        pairs.append((round(p, 3), 1 if ((i * 7919) % 1000) / 1000.0 < true_p else 0))
    raw = mean([logloss(p, y) for p, y in pairs])
    a, b, calibrated = platt_fit(pairs)
    ok &= a is not None and calibrated < raw
    table = calibration_table(pairs)
    ok &= len(table) >= 3 and all("observed" in row for row in table)
    print(f"  calibration     -> {len(table)} bins; Platt a={a} b={b}: "
          f"log loss {raw:.5f} -> {calibrated:.5f} (better, as it should be)")

    # 6. Writers.
    write_outputs(tmp, summary, scored, quiet=True)
    ok &= (tmp / "scoreboard.json").exists() and (tmp / "scoreboard.md").exists()
    doc = json.loads((tmp / "scoreboard.json").read_text(encoding="utf-8"))
    ok &= doc["scored"] == 2 and "overall" in doc
    print(f"  write_outputs   -> scoreboard.json + scoreboard.md + "
          f"scores-*.jsonl ({doc['scored']} scored)")

    print(f"  temp dir: {tmp}")
    print(f"\n  selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ── Main ───────────────────────────────────────────────────────────────────


def run(args):
    data_dir = Path(args.data_dir)
    records = load_verdicts(data_dir)
    ids = sorted({str(r.get("market_id")) for r in records if r.get("market_id")})
    log(f"{len(records)} verdict(s) over {len(ids)} market(s)", args.quiet)
    if not records:
        log("no verdicts yet — run market-verdict.py first", args.quiet)
        return 0

    resolutions = fetch_resolutions(ids[:args.limit], args.resolved_file)
    log(f"{len(resolutions)} of {len(ids)} markets are resolved", args.quiet)

    scored = score(records, resolutions)
    summary = summarise(scored)
    if not scored:
        log("nothing scoreable yet: no verdict has a resolved market behind it. "
            "This is expected on day one — come back when the markets close.", args.quiet)
        print(f"\n  scored 0 of {summary['scored']} verdicts; "
              f"{len(ids)} markets tracked, 0 resolved")
        return 0

    print_report(summary, args.min_observations, args.quiet)
    if not args.dry_run:
        write_outputs(data_dir, summary, scored, args.quiet)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score verdicts against resolved markets")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--resolved-file", help="saved Gamma JSON (offline testing)")
    ap.add_argument("--limit", type=int, default=200, help="markets to resolve per run")
    ap.add_argument("--min-observations", type=int, default=30,
                    help="sample size below which a read is not meaningful")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="offline checks, exact maths")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
