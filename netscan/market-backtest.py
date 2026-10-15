#!/usr/bin/env python3
"""market-backtest.py - would the model have made money on markets already decided?

The live pipeline needs resolved markets, which means waiting days or weeks for
the first honest answer. This gets one now: take markets that closed recently,
reconstruct the price that was on the board a chosen number of hours before they
closed (the public CLOB keeps price history), ask the model the same question the
live pipeline asks, then score it against what actually happened.

Three things it is honest about, because a backtest that hides its biases is
worse than no backtest:

  * Look-ahead. The model may have seen these events in training, and some
    resolution descriptions even narrate what happened ("Trump announced the
    reinstatement on July 13"). So the run reports how far the model beat the
    market: beating a market of people with money at stake by a wide margin is
    evidence of leakage, not of skill. Recent resolutions with a long lead time
    are the least contaminated, which is why --lead-hours and --days-back exist.
  * Costs. The historical order book is gone, so the spread is assumed
    (--assumed-spread) rather than observed, and --cost-multiplier lets the
    result be checked against worse assumptions.
  * Sample size. Forty markets is not evidence of profitability, and the report
    says so with an interval instead of a headline.

Verdicts are written to a throwaway data dir, so a backtest never pollutes the
live verdict history that the real scoreboard reads.

Output: /opt/netscan/data/markets/
  - backtest-YYYYMMDD.json   selection, verdicts, scores, simulation

Usage:
  market-backtest.py --selftest                       # offline, no network/model
  market-backtest.py --dry-run                        # show what it would ask
  market-backtest.py --max-markets 40 --kinds fuzzy --lead-hours 48
  market-backtest.py --cost-multiplier 2.0 --sizing flat

Location on bc250: /opt/netscan/market-backtest.py
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import market_scoring as ms  # noqa: E402
import polymarket as pm  # noqa: E402

DEFAULT_DATA = Path("/opt/netscan/data/markets")
VERDICT_SCRIPT = Path(__file__).with_name("market-verdict.py")


def log(msg, quiet=False):
    if not quiet:
        print(f"  {msg}", flush=True)


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Market selection ───────────────────────────────────────────────────────


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fetch_closed_markets(days_back=120, min_volume=5000.0, min_liquidity=1000.0,
                         pages=4, quiet=False):
    """Markets that closed between now-days_back and yesterday, biggest first.

    Gamma answers HTTP 500 for the full filter combination — each filter works on
    its own, the set of them does not (verified against production, repeatably).
    So the query stays minimal, with a fallback if even that is rejected, and the
    volume/liquidity floors are applied here.
    """
    now = datetime.now(timezone.utc)
    window = {
        "closed": True,
        "end_date_min": (now - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_date_max": (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    raw = []
    for extra in ({"order": "volumeNum", "ascending": False}, {}):
        try:
            raw = list(pm.iter_markets(max_pages=pages, page_size=100, sleep=0.4,
                                       **window, **extra))
            break
        except pm.MarketApiError as ex:
            log(f"gamma rejected {sorted(list(window) + list(extra))}: {ex}", quiet)
    log(f"gamma: {len(raw)} closed markets in the window", quiet)

    keep = []
    for m in raw:
        volume = _num(m.get("volumeNum")) or _num(m.get("volume"))
        if volume < min_volume:
            continue
        # A closed market reports no liquidity at all (the book is gone), so this
        # floor only applies when the field is actually there. Applying it
        # unconditionally discards every closed market, which is exactly what the
        # first run did.
        liquidity = m.get("liquidityNum", m.get("liquidity"))
        if min_liquidity and liquidity is not None and _num(liquidity) < min_liquidity:
            continue
        keep.append(m)
    keep.sort(key=lambda m: -(_num(m.get("volumeNum")) or _num(m.get("volume"))))
    log(f"{len(keep)} of them clear volume {min_volume:.0f} "
        f"(liquidity {min_liquidity:.0f} where reported)", quiet)
    return keep


def build_case(raw, lead_hours, assumed_spread, fidelity, quiet=False):
    """One backtest case: the market as it looked `lead_hours` before closing.

    Returns None when the case cannot be built honestly: not binary, already
    decided at that moment, or no usable price history.
    """
    market = pm.parse_market(raw)
    if len(market.get("outcomes") or []) != 2 or not market.get("token_ids"):
        return None
    end_iso = market.get("end_date")
    if not end_iso:
        return None
    try:
        end_dt = datetime.fromisoformat(str(end_iso).replace("Z", "+00:00"))
    except ValueError:
        return None
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    target = end_dt.timestamp() - lead_hours * 3600.0

    history = pm.prices_history(market["token_ids"][0], interval="max",
                                fidelity=fidelity)
    pick = price_at(history, target)
    if not pick:
        return None
    delta, price, epoch = pick
    # If the closest observation is far from the moment we wanted, the tape does
    # not cover this market's life and the "price at decision time" is a fiction.
    if abs(delta) > max(6 * 3600.0, lead_hours * 3600.0 * 0.25):
        return None
    if price <= 0.01 or price >= 0.99:
        # Already decided at that moment: nothing to predict, and asking would
        # only measure the model's ability to read a resolved market.
        return None

    case = dict(market)
    case["price_yes"] = price
    case["price_yes_mid"] = price
    case["price_at_epoch"] = int(epoch)
    case["price_delta_s"] = int(delta)
    case["spread"] = assumed_spread
    case["hours_to_end"] = lead_hours
    case["lead_hours"] = lead_hours
    case["assumed_spread"] = assumed_spread
    case["fee_bps"] = market.get("fee_bps") or 0.0
    case["edge_required"] = pm.edge_required(case)
    return case


def price_at(history, target_epoch):
    """Closest observation to a moment in the past.

    prices-history gives [{"t": epoch, "p": "0.62"}, ...] — no order guarantee,
    so this scans rather than trusting the ends. Returns (delta_seconds, price,
    epoch) or None.
    """
    best = None
    for point in history or []:
        if not isinstance(point, dict):
            continue
        t, p = point.get("t"), point.get("p")
        try:
            epoch, price = float(t), float(p)
        except (TypeError, ValueError):
            continue
        delta = abs(epoch - float(target_epoch))
        if best is None or delta < best[0]:
            best = (delta, price, epoch)
    return best


def verdict_input(cases):
    """The shape market-verdict.py reads: a latest-markets.json document."""
    return {
        "generated_at": now_iso(),
        "geoblock": {"blocked": None, "country": None,
                     "note": "backtest: reconstructed prices, no live call"},
        "source": {"gamma": "closed markets", "clob": "prices-history",
                   "mode": "backtest"},
        "counts": {},
        "markets": cases,
    }


# ── Running the analyst on the cases ───────────────────────────────────────


def run_analyst(cases, workdir, llm_model=None, kinds=None, quiet=False, env=None,
                escalate=False):
    """Call the real analyst script on a throwaway data dir.

    Reusing the production CLI (rather than importing its internals) means the
    backtest exercises exactly the code the live pipeline runs, including its
    prompt, validation and repair path.
    """
    workdir = Path(workdir)
    markets_file = workdir / "backtest-markets.json"
    markets_file.write_text(json.dumps(verdict_input(cases), ensure_ascii=False),
                            encoding="utf-8")
    data_dir = workdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(VERDICT_SCRIPT),
           "--markets-file", str(markets_file), "--data-dir", str(data_dir),
           "--max-markets", str(len(cases)), "--min-hours", "0", "--max-hours", "0"]
    if escalate:
        cmd.append("--escalate")
    if kinds:
        cmd += ["--kinds", kinds]
    if quiet:
        cmd += ["--quiet"]
    child_env = dict(os.environ if env is None else env)
    if llm_model:
        child_env["LLM_MODEL"] = llm_model
    log(f"analyst: {' '.join(cmd[1:])}", quiet)
    proc = subprocess.run(cmd, env=child_env)
    return data_dir, proc.returncode


# ── Report ─────────────────────────────────────────────────────────────────


def print_backtest_report(cases, scored, sim, lead_hours, days_back, assumed_spread,
                          quiet=False):
    print(f"\n  BACKTEST  lead {lead_hours:.0f}h before close, resolved in the last "
          f"{days_back}d, assumed spread {assumed_spread:.3f}")
    print("  " + "-" * 74)
    print(f"  cases built      : {len(cases)}")
    print(f"  verdicts scored  : {len(scored)}")
    if not scored:
        print("  nothing scored — the analyst produced no usable verdicts")
        return
    by_kind = {}
    for row in scored:
        by_kind.setdefault(row.get("kind") or "?", []).append(row)
    print(f"\n  {'kind':<7} {'n':>4} {'brier model':>12} {'brier market':>13} "
          f"{'trades':>7} {'won':>5} {'pnl units':>10}")
    for kind, rows in sorted(by_kind.items()):
        trades = [r["trade"] for r in rows if r["trade"]]
        won = sum(1 for t in trades if t["won"])
        kind_model = ms.mean([r["brier_model"] for r in rows])
        kind_market = ms.mean([r["brier_market"] for r in rows])
        print(f"  {kind:<7} {len(rows):>4} {kind_model:>12.5f} {kind_market:>13.5f} "
              f"{len(trades):>7} {won:>5} {sum(t['pnl'] for t in trades):>10.2f}")
        if len(rows) >= 2 and kind_model - kind_market < -0.05:
            print(f"    >> {kind}: the model is {abs(kind_model - kind_market):.3f} "
                  f"Brier better than the market on {len(rows)} markets — on a small "
                  f"sample that smells of look-ahead rather than skill")
    overall_model = ms.mean([r["brier_model"] for r in scored])
    overall_market = ms.mean([r["brier_market"] for r in scored])
    gap = overall_model - overall_market
    print(f"\n  model Brier {overall_model:.5f} vs market {overall_market:.5f} "
          f"({'model ahead' if gap < 0 else 'market ahead'} by {abs(gap):.5f})")
    if gap < -0.05:
        print("  >> SUSPICIOUS: beating a real money market by this much usually "
              "means the model has seen the outcome, not that it is clever")
    print(f"\n  paper P&L ({sim['sizing']}, bankroll {sim['bankroll_start']:.0f}): "
          f"{sim['trades']} trades, {sim['bankroll_end']:.2f} "
          f"({sim['total_return']:+.2%}), max dd {sim['max_drawdown']:.2%}")
    if sim["return_ci"][0] is not None:
        print(f"  per trade: {sim['mean_return_per_trade']:+.4f} "
              f"(90% CI {sim['return_ci'][0]:+.4f} .. {sim['return_ci'][2]:+.4f})")
    print(f"\n  {sim['verdict']}")
    print(f"\n  CAVEATS: assumed spread (the real book is gone), possible "
          f"look-ahead, {len(scored)} markets is not a sample")


# ── Selftest ───────────────────────────────────────────────────────────────


def selftest():
    """Offline: price reconstruction, case building, scoring and simulation."""
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="market-backtest-selftest-"))

    history = [{"t": 100000, "p": "0.40"}, {"t": 200000, "p": "0.55"},
               {"t": 300000, "p": "0.70"}]
    ok &= price_at(history, 205000)[1] == 0.55      # nearest wins
    ok &= price_at(history, 100000)[1] == 0.40
    ok &= price_at([], 1) is None
    ok &= price_at([{"t": "x", "p": "0.5"}], 1) is None
    print(f"  price_at        -> nearest {price_at(history, 205000)[1]}, "
          f"empty/unparseable -> None")

    from polymarket import FIXTURE
    cases = [dict(pm.parse_market(FIXTURE[1]))]
    cases[0]["price_yes"] = 0.34
    cases[0]["spread"] = 0.02
    cases[0]["edge_required"] = pm.edge_required(cases[0])
    doc = verdict_input(cases)
    ok &= doc["markets"][0]["id"] == "900002" and doc["source"]["mode"] == "backtest"
    print(f"  verdict_input   -> {len(doc['markets'])} case(s), "
          f"edge_required {cases[0]['edge_required']}")

    # Score a fabricated verdict against a fabricated outcome, then simulate.
    verdicts = [
        {"market_id": "A", "question": "QA", "kind": "fuzzy", "mode": "nothink",
         "asked_at": "2026-01-01T00:00:00Z", "verdict": {"p_yes": 0.75},
         "price_yes_at_decision": 0.45, "spread_at_decision": 0.02,
         "fee_bps_at_decision": 0.0, "edge_required": 0.05, "rules_hash": "x"},
        {"market_id": "B", "question": "QB", "kind": "fuzzy", "mode": "nothink",
         "asked_at": "2026-01-01T00:00:00Z", "verdict": {"p_yes": 0.75},
         "price_yes_at_decision": 0.45, "spread_at_decision": 0.02,
         "fee_bps_at_decision": 0.0, "edge_required": 0.05, "rules_hash": "y"},
    ]
    resolutions = {"A": {"yes_won": True, "raw": {"closed": True, "id": "A"}},
                   "B": {"yes_won": False, "raw": {"closed": True, "id": "B"}}}
    scored = ms.score(verdicts, resolutions, cost_multiplier=1.0)
    ok &= len(scored) == 2 and scored[0]["trade"]["won"] and not scored[1]["trade"]["won"]
    stressed = ms.score(verdicts, resolutions, cost_multiplier=3.0)
    ok &= stressed[0]["trade"]["cost"] > scored[0]["trade"]["cost"]
    sim = ms.simulate(scored, bankroll=100.0, sizing="flat", max_stake_frac=0.1)
    # Hand-checkable: stake 10 at cost 0.46 -> 21.74 shares -> +11.74 on the win,
    # then -10 on the loss: 100 -> 101.74.
    ok &= sim["trades"] == 2 and abs(sim["bankroll_end"] - 101.74) < 0.05
    print(f"  score+simulate  -> 1 win / 1 loss, bankroll 100 -> {sim['bankroll_end']} "
          f"(want 101.74), 3x costs raises cost {scored[0]['trade']['cost']} -> "
          f"{stressed[0]['trade']['cost']}")
    print(f"  temp dir: {tmp}")
    print(f"\n  selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ── Main ───────────────────────────────────────────────────────────────────


def run(args):
    data_dir = Path(args.data_dir)
    raw = fetch_closed_markets(args.days_back, args.min_volume, args.min_liquidity,
                               args.pages, args.quiet)
    if args.resolved_file:
        raw = json.loads(Path(args.resolved_file).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("markets", [])
        log(f"loaded {len(raw)} closed markets from file", args.quiet)

    cases = []
    for item in raw:
        if len(cases) >= args.max_markets:
            break
        case = build_case(item, args.lead_hours, args.assumed_spread, args.fidelity,
                          args.quiet)
        if case:
            cases.append(case)
    log(f"{len(cases)} usable case(s) of {len(raw)} closed markets", args.quiet)
    if not cases:
        log("no usable cases: no closed market had both a price history and an "
            "undecided board at the lead time", args.quiet)
        return 0
    if args.kinds:
        wanted = {k.strip() for k in args.kinds.split(",")}
        cases = [c for c in cases if c.get("kind") in wanted]
        log(f"{len(cases)} case(s) left after --kinds {args.kinds}", args.quiet)

    if args.dry_run:
        print(f"\n  {'id':<9} {'kind':<6} {'price':>7} {'edge>':>6} {'Δh':>5}  question")
        print("  " + "-" * 96)
        for c in cases[:20]:
            print(f"  {c['id']:<9} {c.get('kind', '-'):<6} {c['price_yes']:>7.3f} "
                  f"{c['edge_required']:>6.3f} {c['price_delta_s'] / 3600:>5.1f}  "
                  f"{(c.get('question') or '')[:54]}")
        print(f"\n  {len(cases)} case(s) ready; nothing asked, nothing written")
        return 0

    workdir = Path(tempfile.mkdtemp(prefix="market-backtest-"))
    resolutions = {}
    for raw_item in raw:
        won = ms.yes_won(raw_item)
        if won is not None:
            resolutions[str(pm.parse_market(raw_item)["id"])] = {
                "yes_won": won, "raw": raw_item}
    log(f"{len(resolutions)} resolved outcome(s) available to score against", args.quiet)

    if args.verdicts_from:
        data_dir_for_verdicts = Path(args.verdicts_from)
        log(f"using existing verdicts from {data_dir_for_verdicts}", args.quiet)
    else:
        data_dir_for_verdicts, code = run_analyst(
            cases, workdir, args.llm_model, args.kinds, args.quiet,
            escalate=args.escalate)
        if code not in (0, 2):
            log(f"analyst exited {code}", args.quiet)

    records = ms.load_verdicts(data_dir_for_verdicts)
    log(f"{len(records)} verdict(s) to score", args.quiet)
    scored = ms.score(records, resolutions, cost_multiplier=args.cost_multiplier)
    # The holding period is defined by construction here: the decision is taken
    # `lead_hours` before close, so that is how long the capital was locked. Using
    # the real asked_at would give a negative number, since the analyst runs now
    # and the market closed in the past.
    for row in scored:
        row["days_held"] = round(args.lead_hours / 24.0, 3)
    sim = ms.simulate(scored, bankroll=args.bankroll, sizing=args.sizing,
                      kelly_scale=args.kelly, max_stake_frac=args.max_stake,
                      max_drawdown_stop=args.dd_stop)
    print_backtest_report(cases, scored, sim, args.lead_hours, args.days_back,
                          args.assumed_spread, args.quiet)

    if not args.dry_run:
        data_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        out = Path(args.out) if args.out else data_dir / f"backtest-{stamp}.json"
        out.write_text(json.dumps({
            "generated_at": now_iso(),
            "params": {"lead_hours": args.lead_hours, "days_back": args.days_back,
                       "assumed_spread": args.assumed_spread,
                       "cost_multiplier": args.cost_multiplier,
                       "kinds": args.kinds, "max_markets": args.max_markets,
                       "llm_model": args.llm_model or os.environ.get("LLM_MODEL")},
            "cases": [{k: v for k, v in c.items() if k != "verdict"} for c in cases],
            "scored": scored, "simulation": sim,
        }, indent=1, ensure_ascii=False), encoding="utf-8")
        log(f"wrote {out}", args.quiet)
    log(f"work dir kept for inspection: {workdir}", args.quiet)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Backtest the analyst on resolved markets")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--max-markets", type=int, default=40)
    ap.add_argument("--pages", type=int, default=4)
    ap.add_argument("--lead-hours", type=float, default=48.0,
                    help="decide this many hours before the market closed")
    ap.add_argument("--days-back", type=int, default=120,
                    help="only markets that closed within N days (recent ones are "
                         "the least likely to be in the model's training data)")
    ap.add_argument("--min-volume", type=float, default=5000.0)
    ap.add_argument("--min-liquidity", type=float, default=1000.0)
    ap.add_argument("--assumed-spread", type=float, default=0.02,
                    help="the historical book is gone; assume this spread")
    ap.add_argument("--cost-multiplier", type=float, default=1.0)
    ap.add_argument("--fidelity", type=int, default=60, help="price history minutes")
    ap.add_argument("--kinds", help="feed,event,fuzzy")
    ap.add_argument("--llm-model", help="override LLM_MODEL for the analyst")
    ap.add_argument("--verdicts-from", help="score existing verdicts instead of asking")
    ap.add_argument("--escalate", action="store_true",
                    help="let the analyst escalate doubtful verdicts to thinking")
    ap.add_argument("--resolved-file", help="saved Gamma JSON of closed markets")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--sizing", choices=("kelly", "flat"), default="flat")
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--max-stake", type=float, default=0.05)
    ap.add_argument("--dd-stop", type=float, default=0.5)
    ap.add_argument("--out", help="output file")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
