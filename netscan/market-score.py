#!/usr/bin/env python3
"""market-score.py - is the model any good, and would it have made money?

Two questions, one report:

  * Quality: Brier score and log loss of the model's probabilities against the
    market price, plus a calibration table. The market price is the benchmark,
    and it is a brutal one.
  * Profitability: a paper ledger priced with the real spread and fee, run
    through a bankroll with position sizing, drawdown and a bootstrap interval.
    A model can be well calibrated and still lose to the spread, or make money
    on six lucky trades, so the interval is reported next to the P&L.

Both need resolved markets. On day one this prints "nothing scoreable yet",
which is the correct answer rather than a bug.

Requirements: verdicts from market-verdict.py, and markets that have closed.
Scoring maths lives in market_scoring.py; for a backtest over already-resolved
markets see market-backtest.py.

Output: /opt/netscan/data/markets/
  - scoreboard.json      full numbers, machine readable
  - scoreboard.md        short summary for the daily digest
  - scores-YYYYMMDD.jsonl  one line per scored verdict (audit trail)

Usage:
  market-score.py --selftest
  market-score.py
  market-score.py --bankroll 500 --sizing kelly --kelly 0.25 --cost-multiplier 1.5

Schedule: daily, after market-collect.py.

Location on bc250: /opt/netscan/market-score.py
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import market_scoring as ms  # noqa: E402

DEFAULT_DATA = Path("/opt/netscan/data/markets")


def log(msg, quiet=False):
    if not quiet:
        print(f"  {msg}", flush=True)


# ── Report ─────────────────────────────────────────────────────────────────


def print_report(summary, sim, min_obs, quiet=False):
    overall = summary.get("overall")
    if not overall:
        log(f"nothing to score yet: 0 of {summary['scored']} verdicts have a "
            f"resolved market behind them", quiet)
        return

    print(f"\n  scored verdicts : {summary['scored']}   "
          f"(minimum for a meaningful read: {min_obs})")
    print(f"\n  {'':<8} {'Brier':>9} {'log loss':>10}   lower is better")
    print("  " + "-" * 40)
    for label, b, ll in (("model", overall["brier_model"], overall["logloss_model"]),
                         ("market", overall["brier_market"], overall["logloss_market"])):
        print(f"  {label:<8} {b:>9.5f} {ll:>10.5f}")
    verdict = ("the MODEL is ahead" if overall["brier_delta"] < 0
               else "the MARKET is ahead")
    print(f"\n  {verdict} on Brier by {abs(overall['brier_delta']):.5f} "
          f"(model beat the market on {overall['beat_market_on']}/{overall['n']} markets)")

    for mode, block in (summary.get("by_mode") or {}).items():
        if block:
            print(f"    mode {mode:<9} n={block['n']:<4} model {block['brier_model']:.5f} "
                  f"market {block['brier_market']:.5f}  (trades {block['trades']}, "
                  f"pnl {block['pnl_units']:+.2f} units)")

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
        print(f"\n  Platt             : needs >= 10 scored verdicts (have {overall['n']})")
    if overall["rules_changed"]:
        print(f"\n  warning         : {overall['rules_changed']} verdict(s) were made on "
              f"resolution rules that have since been edited")

    print_profitability(sim)


def print_profitability(sim):
    print(f"\n  PROFITABILITY (paper, bankroll {sim['bankroll_start']:.0f}, "
          f"sizing {sim['sizing']})")
    print("  " + "-" * 40)
    if not sim["trades"]:
        print(f"  {sim['verdict']}")
        return
    print(f"  trades          : {sim['trades']}  (won {sim['hit_rate']:.0%})")
    print(f"  bankroll        : {sim['bankroll_start']:.2f} -> {sim['bankroll_end']:.2f}  "
          f"({sim['total_return']:+.2%})")
    print(f"  max drawdown    : {sim['max_drawdown']:.2%}")
    if sim.get("avg_days_held"):
        print(f"  avg days held   : {sim['avg_days_held']}  ->  annualised "
              f"{sim['annualised_return']:+.1%}" if sim.get("annualised_return")
              else f"  avg days held   : {sim['avg_days_held']}")
    if sim.get("avg_win") is not None:
        print(f"  avg win / loss  : {sim['avg_win']:+.2f} / {sim['avg_loss']:+.2f}"
              + (f"   profit factor {sim['profit_factor']}" if sim.get("profit_factor") else ""))
    lo, mid, hi = sim["return_ci"]
    print(f"  per trade       : {sim['mean_return_per_trade']:+.4f}  "
          f"(90% CI {lo:+.4f} .. {hi:+.4f})")
    if sim.get("stopped_at"):
        print(f"  KILL SWITCH     : stopped at trade {sim['stopped_at']['n']} on a "
              f"{sim['stopped_at']['drawdown']:.1%} drawdown")
    print(f"\n  {sim['verdict']}")


def write_outputs(data_dir, summary, scored, sim, quiet=False):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc = {"generated_at": now, "profitability": sim, **summary}
    (data_dir / "scoreboard.json").write_text(
        json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    with (data_dir / f"scores-{stamp}.jsonl").open("a", encoding="utf-8") as fh:
        for row in scored:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    overall = summary.get("overall")
    if overall:
        won = round(sim["hit_rate"] * sim["trades"]) if sim.get("hit_rate") else 0
        md = [
            f"# Market scoreboard — {now}",
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
             "That is the expected result, not a failure: the price is the consensus "
             "of people with money at stake."),
            "",
            "## Profitability (paper)",
            "",
            f"- bankroll {sim['bankroll_start']:.0f} -> {sim['bankroll_end']:.2f} "
            f"({sim['total_return']:+.2%}), max drawdown {sim['max_drawdown']:.2%}",
            f"- {sim['trades']} trades, {won} won, mean "
            f"{sim['mean_return_per_trade']:+.4f} per trade "
            f"(90% CI {sim['return_ci'][0]:+.4f} .. {sim['return_ci'][2]:+.4f})",
            f"- {sim['verdict']}",
        ]
        (data_dir / "scoreboard.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"wrote scoreboard.json + scoreboard.md ({len(scored)} scored)", quiet)


# ── Selftest ───────────────────────────────────────────────────────────────


def selftest():
    """The scoring maths lives in market_scoring and is tested there; here the
    CLI plumbing: the file round trip and the resolution path."""
    ok = True
    import tempfile

    ok &= ms.selftest() == 0
    tmp = Path(tempfile.mkdtemp(prefix="market-score-selftest-"))

    verdicts = [
        {"market_id": "1", "question": "Q1", "kind": "fuzzy", "mode": "nothink",
         "asked_at": "2026-01-01T00:00:00Z", "verdict": {"p_yes": 0.9},
         "price_yes_at_decision": 0.5, "spread_at_decision": 0.01,
         "fee_bps_at_decision": 0, "edge_required": 0.05, "rules_hash": "abc"},
        {"market_id": "2", "question": "Q2", "kind": "fuzzy", "mode": "think",
         "asked_at": "2026-01-02T00:00:00Z", "verdict": {"p_yes": 0.1},
         "price_yes_at_decision": 0.5, "spread_at_decision": 0.01,
         "fee_bps_at_decision": 0, "edge_required": 0.05, "rules_hash": "def"},
    ]
    resolutions = {
        "1": {"yes_won": True, "raw": {"closed": True, "id": "1", "closedTime":
                                       "2026-01-03T00:00:00Z"}},
        "2": {"yes_won": False, "raw": {"closed": True, "id": "2", "closedTime":
                                        "2026-01-06T00:00:00Z"}},
    }
    scored = ms.score(verdicts, resolutions)
    ok &= len(scored) == 2 and scored[0]["days_held"] == 2.0
    summary = ms.summarise(scored)
    sim = ms.simulate(scored, bankroll=1000.0, sizing="kelly")
    ok &= sim["trades"] == 2 and sim["total_return"] > 0
    avg_days = sim["avg_days_held"]
    ok &= avg_days == 3.0          # 2 days on one market, 4 on the other
    if avg_days != 3.0:
        print(f"    avg_days_held={avg_days} (want 3.0)")
    write_outputs(tmp, summary, scored, sim, quiet=True)
    ok &= (tmp / "scoreboard.json").exists() and (tmp / "scoreboard.md").exists()
    doc = json.loads((tmp / "scoreboard.json").read_text(encoding="utf-8"))
    ok &= doc["scored"] == 2 and "profitability" in doc and doc["profitability"]["trades"] == 2
    print(f"\n  scoreboard.json -> {doc['scored']} scored, profitability present, "
          f"md written")
    print(f"  temp dir: {tmp}")
    print(f"\n  selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ── Main ───────────────────────────────────────────────────────────────────


def run(args):
    data_dir = Path(args.data_dir)
    records = ms.load_verdicts(data_dir)
    ids = sorted({str(r.get("market_id")) for r in records if r.get("market_id")})
    log(f"{len(records)} verdict(s) over {len(ids)} market(s)", args.quiet)
    if not records:
        log("no verdicts yet — run market-verdict.py first", args.quiet)
        return 0

    resolutions = ms.fetch_resolutions(ids[:args.limit], args.resolved_file)
    log(f"{len(resolutions)} of {len(ids)} markets are resolved", args.quiet)

    scored = ms.score(records, resolutions, cost_multiplier=args.cost_multiplier)
    summary = ms.summarise(scored)
    sim = ms.simulate(scored, bankroll=args.bankroll, sizing=args.sizing,
                      kelly_scale=args.kelly, max_stake_frac=args.max_stake,
                      max_drawdown_stop=args.dd_stop)
    if not scored:
        log(f"nothing scoreable yet: no verdict has a resolved market behind it. "
            f"This is expected on day one — {len(ids)} markets tracked, 0 resolved.",
            args.quiet)
        print(f"\n  scored 0 of 0 verdicts; {len(ids)} markets tracked, 0 resolved")
        return 0

    print_report(summary, sim, args.min_observations, args.quiet)
    if not args.dry_run:
        write_outputs(data_dir, summary, scored, sim, args.quiet)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score verdicts, and price the P&L")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--resolved-file", help="saved Gamma JSON (offline testing)")
    ap.add_argument("--limit", type=int, default=200, help="markets to resolve per run")
    ap.add_argument("--min-observations", type=int, default=30,
                    help="sample size below which a quality read is not meaningful")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--sizing", choices=("kelly", "flat"), default="kelly")
    ap.add_argument("--kelly", type=float, default=0.25,
                    help="fraction of full Kelly (full Kelly is too aggressive on "
                         "estimates that are themselves uncertain)")
    ap.add_argument("--max-stake", type=float, default=0.05,
                    help="hard cap per trade, as a fraction of the bankroll")
    ap.add_argument("--dd-stop", type=float, default=0.5,
                    help="stop simulating after this drawdown")
    ap.add_argument("--cost-multiplier", type=float, default=1.0,
                    help="stress the costs: 2.0 doubles spread and fee")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="offline, exact maths")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
