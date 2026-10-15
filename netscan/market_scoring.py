#!/usr/bin/env python3
"""market_scoring.py - shared maths for scoring predictions and P&L.

Imported by the market-* CLI jobs. Named without a dash, like llm_sanitize.py
and exec_policy.py, so `import market_scoring` actually works.

Two different questions live in here, and they must not be confused:

  * Is the model any good?  -> Brier score, log loss and calibration, against
    the market price, which is the only benchmark that matters.
  * Would it have made money? -> a paper ledger priced with the real spread and
    fee, a bankroll, position sizing, drawdown, and a bootstrap interval,
    because a P&L over a handful of trades is noise and saying so is part of the
    job. A model can be well calibrated and still lose to the spread; it can
    also make money on six lucky trades, which is what the interval is for.

Usage:
    from market_scoring import brier, logloss, yes_won, fetch_resolutions, \
        load_verdicts, score, summarise, simulate

Location on bc250: /opt/netscan/market_scoring.py
"""

import hashlib
import json
import math
import random
from pathlib import Path

from polymarket import get_markets_by_id

EPS = 1e-6
# |p - 0.5| within this is treated as an abstention, not a prediction. See
# is_abstention() for why that distinction is worth code.
ABSTAIN_TOL = 0.02


# ── Forecast maths ─────────────────────────────────────────────────────────


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

    Two parameters and a few hundred observations do not justify a numpy
    dependency on this box. In-sample by construction, so report the sample size
    next to the result: with small n it flatters itself.
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
    return (round(a, 4), round(b, 4),
            round(calibrated, 5) if calibrated is not None else None)


def calibration_table(pairs, bins=10):
    """When the model said 0.7, how often was it right?"""
    table = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        bucket = [(p, y) for p, y in pairs
                  if lo <= p < hi or (i == bins - 1 and p == 1.0)]
        if bucket:
            table.append({
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": len(bucket),
                "predicted": round(mean([p for p, _ in bucket]), 3),
                "observed": round(mean([y for _, y in bucket]), 3),
            })
    return table


def bootstrap_ci(values, iterations=1000, seed=7, lo=5, hi=95):
    """Percentile interval for the mean of `values`.

    The reason this exists: 'I made 12 units on 9 trades' is not evidence of
    anything. The interval says whether the result is distinguishable from zero,
    and it usually is not until the sample is big.
    """
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None, None, None
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(iterations):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (round(means[int(lo / 100 * iterations)], 6),
            round(means[iterations // 2], 6),
            round(means[min(iterations - 1, int(hi / 100 * iterations))], 6))


# ── Resolutions ────────────────────────────────────────────────────────────


def _json_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            got = json.loads(value)
            return got if isinstance(got, list) else []
        except json.JSONDecodeError:
            return []
    return []


def yes_won(raw):
    """Did YES win? A resolved binary market publishes outcomePrices 1/0.

    None means "not scoreable": still open, unresolved, or not two-outcome.
    """
    outcomes = _json_list(raw.get("outcomes"))
    prices = _json_list(raw.get("outcomePrices"))
    if len(outcomes) != 2 or len(prices) != 2 or not raw.get("closed"):
        return None
    try:
        p0, p1 = float(prices[0]), float(prices[1])
    except (TypeError, ValueError):
        return None
    if p0 >= 0.99 and p1 <= 0.01:
        return True
    if p0 <= 0.01 and p1 >= 0.99:
        return False
    return None


def resolved_at(raw):
    """When the market closed, as an epoch or None."""
    from datetime import datetime, timezone
    for key in ("closedTime", "umaEndDate", "endDate"):
        value = raw.get(key)
        if not value:
            continue
        txt = str(value).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(txt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except (ValueError, TypeError):
            continue
    return None


def fetch_resolutions(market_ids, resolved_file=None):
    """{market_id: {"yes_won": bool, "raw": raw}} for markets that resolved."""
    if resolved_file:
        raw_markets = json.loads(Path(resolved_file).read_text(encoding="utf-8"))
        if isinstance(raw_markets, dict):
            raw_markets = raw_markets.get("markets", [])
    else:
        raw_markets = get_markets_by_id(market_ids)
    out = {}
    for raw in raw_markets or []:
        won = yes_won(raw)
        if won is not None:
            out[str(raw.get("id"))] = {"yes_won": won, "raw": raw}
    return out


def load_verdicts(data_dir):
    """Every verdict ever written, oldest file first."""
    records = []
    for path in sorted(Path(data_dir).glob("verdicts-*.jsonl")):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            continue
    return records


def rules_hash_now(raw):
    blob = f"{raw.get('question')}|{raw.get('description')}|{raw.get('resolutionSource')}"
    return hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()[:12]


# ── Scoring a verdict ──────────────────────────────────────────────────────


def is_abstention(record, tol=ABSTAIN_TOL):
    """Is this verdict a coin flip wearing a prediction's clothes?

    Measured, not theorised: in a backtest over 17 already-resolved markets the
    model answered exactly 0.500 on nine of them — all sports, where it has
    nothing to go on — while reporting 0.90-0.95 confidence. A 0.500 carries no
    information, and trading it means betting against the market with a coin:
    with an edge_required of 0.035, a 0.500 against a 0.545 price looks like a
    "gap" and is not one.

    So abstentions never trade, and they stay out of the calibration sample —
    they are not forecasts to be scored, they are the absence of one.
    """
    p = (record.get("verdict") or {}).get("p_yes")
    return p is not None and abs(float(p) - 0.5) <= tol


def paper_trade(record, yes, cost_multiplier=1.0, abstain_tol=ABSTAIN_TOL):
    """Would this verdict have traded, at what price, and what did it pay?

    Per share, in USDC: cost = the price actually paid (ask side) plus half the
    spread plus the taker fee; payoff = 1 if that side wins, 0 if not. Returns
    None when the model's gap does not clear the recorded edge_required — which
    is the entire purpose of that number.

    `cost_multiplier` exists for backtests: the historical spread is gone, so a
    backtest has to assume one, and being able to make it worse on purpose is
    how you find out whether the result survives real costs.
    """
    verdict = record.get("verdict") or {}
    price = record.get("price_yes_at_decision")
    if price is None or verdict.get("p_yes") is None:
        return None
    if is_abstention(record, abstain_tol):
        return None
    gap = float(verdict["p_yes"]) - float(price)
    half_spread = float(record.get("spread_at_decision") or 0.0) / 2.0
    fee = float(record.get("fee_bps_at_decision") or 0.0) / 10000.0
    cost = half_spread + fee
    if cost_multiplier != 1.0:
        cost *= cost_multiplier
    if abs(gap) <= float(record.get("edge_required") or 0.0):
        return None

    side_yes = gap > 0
    pay_price = float(price) if side_yes else 1.0 - float(price)
    total_cost = pay_price + cost
    won = yes if side_yes else (not yes)
    pnl = (1.0 if won else 0.0) - total_cost
    return {
        "side": "YES" if side_yes else "NO",
        "cost": round(total_cost, 5),
        "won": bool(won),
        "shares_per_unit": round(1.0 / total_cost, 4) if total_cost > 0 else None,
        "pnl": round(pnl, 5),
        "roi": round(pnl / total_cost, 4) if total_cost else None,
        "gap": round(gap, 4),
        "pay_price": round(pay_price, 5),
    }


def score(records, resolutions, cost_multiplier=1.0, liquidity=None,
          abstain_tol=ABSTAIN_TOL):
    """Line up each verdict that has a resolution against the market price."""
    scored = []
    for rec in records:
        res = resolutions.get(str(rec.get("market_id")))
        if not res:
            continue
        y = 1 if res["yes_won"] else 0
        p = (rec.get("verdict") or {}).get("p_yes")
        price = rec.get("price_yes_at_decision")
        if p is None or price is None:
            continue
        closed = resolved_at(res["raw"])
        asked = rec.get("asked_at")
        days_held = None
        if closed and asked:
            from datetime import datetime, timezone
            try:
                start = datetime.strptime(asked, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc).timestamp()
                days_held = round((closed - start) / 86400.0, 3)
            except ValueError:
                days_held = None
        scored.append({
            "market_id": str(rec.get("market_id")),
            "question": rec.get("question"),
            "kind": rec.get("kind"),
            "mode": rec.get("mode") or "unknown",
            "asked_at": asked,
            "model": rec.get("model"),
            "p_yes": p,
            "price_yes": price,
            "yes_won": bool(res["yes_won"]),
            "days_held": days_held,
            "resolved_at": closed,
            "liquidity": (float(liquidity[str(rec.get("market_id"))])
                          if liquidity and str(rec.get("market_id")) in liquidity
                          else rec.get("liquidity_at_decision")),
            "brier_model": round(brier(p, y), 6),
            "brier_market": round(brier(price, y), 6),
            "logloss_model": round(logloss(p, y), 6),
            "logloss_market": round(logloss(price, y), 6),
            "trade": paper_trade(rec, res["yes_won"], cost_multiplier, abstain_tol),
            "abstain": is_abstention(rec, abstain_tol),
            "rules_changed": (rec.get("rules_hash") is not None and
                              rules_hash_now(res["raw"]) not in (None,
                                                                 rec.get("rules_hash"))),
        })
    return scored


# ── Bankroll simulation ────────────────────────────────────────────────────


def kelly_fraction(p_model, price, side_yes):
    """Optimal bet fraction for a binary contract, Kelly.

    Buying YES at price p for a contract that pays 1: net odds b = (1-p)/p, so
    f* = (q*b - (1-q)) / b = (q - p) / (1 - p). Buying NO is the mirror. Only
    interesting when the model's gap is positive, which the caller checks.
    """
    if side_yes:
        if price >= 1.0:
            return 0.0
        return max(0.0, (p_model - price) / (1.0 - price))
    if price <= 0.0:
        return 0.0
    return max(0.0, ((1.0 - p_model) - (1.0 - price)) / (1.0 - (1.0 - price)))


def simulate(scored, bankroll=1000.0, sizing="kelly", kelly_scale=0.25,
             max_stake_frac=0.05, min_stake=1.0, max_drawdown_stop=0.5,
             liquidity_frac=0.01):
    """Walk the scored verdicts chronologically and run a bankroll.

    Sizing:
      flat  — the same stake every time (the naive benchmark: this is what the
              simple ledger in market-score.py already reports)
      kelly — a fraction of the *current* bankroll from the model's own edge,
              scaled down (full Kelly is famously too aggressive on estimates
              that are themselves uncertain) and capped.

    The cap matters more than the formula: without it, one confident-but-wrong
    verdict ends the experiment. A stake is also capped at `liquidity_frac` of
    the market's liquidity, because you cannot get filled for size at the
    quoted price, and the run stops entirely after `max_drawdown_stop`.
    """
    rows = sorted([r for r in scored if r["trade"]], key=lambda r: r["asked_at"] or "")
    equity = bankroll
    peak = bankroll
    max_dd = 0.0
    stopped_at = None
    curve = [{"n": 0, "equity": round(equity, 2), "market_id": None}]
    records = []

    for i, row in enumerate(rows, 1):
        drawdown = (peak - equity) / peak if peak else 0.0
        if drawdown >= max_drawdown_stop:
            stopped_at = {"n": i, "drawdown": round(drawdown, 4)}
            break

        trade = row["trade"]
        gap = abs(trade["gap"])
        # One source of truth: the trade carries the side, the price paid and the
        # gap, so Kelly is computed from those rather than from a row field that
        # could disagree with them.
        if sizing == "kelly":
            q_side = row["p_yes"] if trade["side"] == "YES" else 1.0 - row["p_yes"]
            frac = kelly_fraction(q_side, trade["pay_price"], side_yes=True)
            stake = equity * min(frac * kelly_scale, max_stake_frac)
        else:
            stake = min(bankroll * max_stake_frac, equity * max_stake_frac)
        if row.get("liquidity"):
            stake = min(stake, float(row["liquidity"]) * liquidity_frac)
        if stake < min_stake:
            continue

        shares = stake / trade["cost"]
        payout = shares * (1.0 if trade["won"] else 0.0)
        pnl = payout - stake
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak else 0.0)
        curve.append({"n": i, "equity": round(equity, 2), "market_id": row["market_id"]})
        records.append({
            "market_id": row["market_id"], "side": trade["side"], "gap": gap,
            "stake": round(stake, 2), "cost": trade["cost"], "won": trade["won"],
            "pnl": round(pnl, 2),
            "return_on_stake": round(pnl / stake, 4) if stake else None,
            "days_held": row.get("days_held"), "equity_after": round(equity, 2),
            "mode": row.get("mode"),
        })

    returns = [r["return_on_stake"] for r in records]
    winners = [r for r in records if r["won"]]
    losers = [r for r in records if not r["won"]]
    held = [r["days_held"] for r in records if r.get("days_held")]
    total_return = (equity - bankroll) / bankroll if bankroll else None
    # A time-weighted view, because prediction-market capital is locked up until
    # resolution: the same 3% is a fine trade held two days and a bad one held
    # three months.
    avg_days = mean(held)
    annualised = None
    if total_return is not None and avg_days and avg_days > 0 and equity > 0:
        annualised = (1.0 + total_return) ** (365.0 / (avg_days * max(len(records), 1))) - 1.0

    return {
        "sizing": sizing, "bankroll_start": bankroll, "bankroll_end": round(equity, 2),
        "trades": len(records), "skipped_below_min_stake": len(rows) - len(records),
        "skipped_abstentions": sum(1 for r in scored if r.get("abstain")),
        "total_return": round(total_return, 4) if total_return is not None else None,
        "annualised_return": round(annualised, 4) if annualised is not None else None,
        "avg_days_held": round(avg_days, 2) if avg_days else None,
        "hit_rate": round(len(winners) / len(records), 4) if records else None,
        "avg_win": round(mean([r["pnl"] for r in winners]), 2) if winners else None,
        "avg_loss": round(mean([r["pnl"] for r in losers]), 2) if losers else None,
        "profit_factor": (round(sum(r["pnl"] for r in winners) /
                               abs(sum(r["pnl"] for r in losers)), 3)
                          if losers and sum(r["pnl"] for r in losers) != 0 else None),
        "max_drawdown": round(max_dd, 4),
        "stopped_at": stopped_at,
        "mean_return_per_trade": round(mean(returns), 4) if returns else None,
        "return_ci": bootstrap_ci(returns),
        "equity_curve": curve,
        "trades_log": records,
        "verdict": _profitability_verdict(records, returns),
    }


def _profitability_verdict(records, returns):
    """Say out loud whether the P&L means anything yet."""
    if not records:
        return "no trades: nothing cleared its cost threshold"
    lo, _, hi = bootstrap_ci(returns)
    if lo is None:
        return f"{len(records)} trade(s): far too few to say anything"
    if lo > 0:
        return (f"the mean return per trade is positive with a 90% interval of "
                f"{lo:+.3f} to {hi:+.3f} — an edge the sample supports, if the "
                f"verdicts were not looking at the future")
    if hi < 0:
        return (f"losing with a 90% interval of {lo:+.3f} to {hi:+.3f}: this "
                f"strategy as configured would have lost money")
    return (f"indistinguishable from zero (90% interval {lo:+.3f} to {hi:+.3f} on "
            f"{len(records)} trades) — which is what a small sample looks like, "
            f"and it stays like that until there are a few hundred")


# ── Summary for the scoreboard report ──────────────────────────────────────


def summarise(scored):
    """Model vs market overall, and split by answering mode."""
    def block(rows):
        if not rows:
            return None
        bm = mean([r["brier_model"] for r in rows])
        bk = mean([r["brier_market"] for r in rows])
        # Abstentions are not forecasts: scoring them would flatter both sides
        # and pollute the calibration table.
        pairs = [(r["p_yes"], 1 if r["yes_won"] else 0)
                 for r in rows if not r.get("abstain")]
        trades = [r["trade"] for r in rows if r["trade"]]
        won = [t for t in trades if t["won"]]
        return {
            "n": len(rows),
            "abstentions": sum(1 for r in rows if r.get("abstain")),
            "scored_excl_abstentions": len(pairs),
            "brier_model": round(bm, 6),
            "brier_market": round(bk, 6),
            "brier_delta": round(bm - bk, 6),
            "logloss_model": round(mean([r["logloss_model"] for r in rows]), 6),
            "logloss_market": round(mean([r["logloss_market"] for r in rows]), 6),
            "beat_market_on": sum(1 for r in rows if r["brier_model"] < r["brier_market"]),
            "trades": len(trades),
            "trades_won": len(won),
            "pnl_units": round(sum(t["pnl"] for t in trades), 4) if trades else 0.0,
            "roi": (round(sum(t["pnl"] for t in trades) /
                          sum(max(t["cost"], EPS) for t in trades), 4)
                    if trades else None),
            "calibration": calibration_table(pairs),
            "platt": list(platt_fit(pairs)) if len(pairs) >= 10 else [None, None, None],
            "rules_changed": sum(1 for r in rows if r["rules_changed"]),
        }

    modes = sorted({r["mode"] for r in scored})
    return {
        "scored": len(scored),
        "overall": block(scored),
        "by_mode": {m: block([r for r in scored if r["mode"] == m]) for m in modes},
    }


# ── Selftest ───────────────────────────────────────────────────────────────


def selftest():
    """Exact numbers, hand-checkable, no network."""
    ok = True

    ok &= abs(brier(0.8, 1) - 0.04) < 1e-9
    ok &= abs(brier(0.5, 1) - 0.25) < 1e-9
    ok &= abs(logloss(0.8, 1) - 0.22314) < 1e-4
    ok &= logloss(0.0, 1) < 20
    print(f"  brier/logloss   -> (0.8,1)={brier(0.8, 1):.2f} (0.5,1)={brier(0.5, 1):.2f} "
          f"ll(0.8,1)={logloss(0.8, 1):.5f}")

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
    print("  yes_won         -> decided / open / not-binary handled")

    base = {"market_id": "M1", "verdict": {"p_yes": 0.8}, "price_yes_at_decision": 0.5,
            "spread_at_decision": 0.02, "fee_bps_at_decision": 0.0, "edge_required": 0.05}
    trade = paper_trade(base, yes=True)
    ok &= trade["side"] == "YES" and abs(trade["cost"] - 0.51) < 1e-9
    ok &= abs(trade["pnl"] - 0.49) < 1e-9 and trade["won"] is True
    ok &= paper_trade(dict(base, verdict={"p_yes": 0.52}), yes=True) is None
    losing = paper_trade(dict(base, verdict={"p_yes": 0.2}), yes=True)
    ok &= losing["side"] == "NO" and losing["won"] is False
    doubled = paper_trade(base, yes=True, cost_multiplier=2.0)
    ok &= doubled["cost"] > trade["cost"]

    # An answer of exactly 0.500 is an abstention: no trade, and it must not
    # count as a forecast in the calibration sample either.
    abstain_rec = {"market_id": "M2", "verdict": {"p_yes": 0.5},
                   "price_yes_at_decision": 0.6, "spread_at_decision": 0.01,
                   "fee_bps_at_decision": 0.0, "edge_required": 0.035}
    ok &= is_abstention(abstain_rec) and paper_trade(abstain_rec, yes=True) is None
    ok &= not is_abstention(base)
    print(f"  abstention      -> p=0.500 against a 0.600 price (a 0.10 'gap') "
          f"trades nothing: {paper_trade(abstain_rec, yes=True) is None}")
    print(f"  paper_trade     -> YES cost {trade['cost']} pnl {trade['pnl']}; "
          f"no-edge -> None; 2x costs -> cost {doubled['cost']}")

    # Kelly: q=0.8, price=0.5 -> f* = 0.3/0.5 = 0.6; mirror for NO at q=0.2.
    ok &= abs(kelly_fraction(0.8, 0.5, True) - 0.6) < 1e-9
    ok &= abs(kelly_fraction(0.2, 0.5, False) - 0.6) < 1e-9
    ok &= kelly_fraction(0.4, 0.5, True) == 0.0
    print(f"  kelly           -> f*(0.8 vs 0.5)={kelly_fraction(0.8, 0.5, True):.3f} "
          f"(want 0.600), no edge -> 0.0")

    # Simulation arithmetic, hand-checkable: bankroll 100, flat 5% stakes.
    # Trade 1 wins: stake 5, cost 0.51 -> 9.80 shares -> payout 9.80 -> pnl +4.80.
    # Trade 2 loses: stake min(5, ...) -> pnl -5.00.  Final 99.80.
    rows = [
        {"market_id": "1", "asked_at": "2026-01-01T00:00:00Z", "p_yes": 0.8,
         "price_yes": 0.5, "yes_won": True, "days_held": 2.0, "mode": "nothink",
         "trade": paper_trade(base, yes=True)},
        {"market_id": "2", "asked_at": "2026-01-02T00:00:00Z", "p_yes": 0.8,
         "price_yes": 0.5, "yes_won": False, "days_held": 2.0, "mode": "nothink",
         "trade": paper_trade(base, yes=False)},
    ]
    sim = simulate(rows, bankroll=100.0, sizing="flat", max_stake_frac=0.05)
    ok &= sim["trades"] == 2 and abs(sim["bankroll_end"] - 99.80) < 0.02
    ok &= abs(sim["trades_log"][0]["pnl"] - 4.80) < 0.02
    ok &= sim["trades_log"][1]["pnl"] == -5.0
    ok &= sim["hit_rate"] == 0.5 and sim["max_drawdown"] > 0.04
    print(f"  simulate(flat)  -> 2 trades, end {sim['bankroll_end']} (want 99.80), "
          f"dd {sim['max_drawdown']:.3f}, ci {sim['return_ci']}")

    # The kill switch must stop the run, not just warn.
    kill = simulate([dict(r, trade=paper_trade(dict(base, verdict={"p_yes": 0.2}),
                                               yes=True)) for r in rows[:1]] * 4,
                    bankroll=100.0, sizing="flat", max_stake_frac=0.25,
                    max_drawdown_stop=0.05)
    ok &= kill["stopped_at"] is not None and kill["trades"] < 4
    print(f"  kill switch     -> stopped at trade {kill['stopped_at']} "
          f"(dd {kill['stopped_at']['drawdown']:.2f}) after {kill['trades']} trade(s)")

    # Kelly sizing must not stake more than the cap, and must stake less with a
    # smaller edge. The row has to be self-consistent: p_yes, price and trade.
    thin_edge = {"market_id": "3", "asked_at": "2026-01-03T00:00:00Z", "p_yes": 0.55,
                 "price_yes": 0.5, "yes_won": True, "days_held": 2.0, "mode": "nothink",
                 "trade": paper_trade(dict(base, verdict={"p_yes": 0.55},
                                           edge_required=0.01), yes=True)}
    small = simulate([thin_edge], bankroll=100.0, sizing="kelly", kelly_scale=0.25,
                     max_stake_frac=0.05)
    big = simulate(rows[:1], bankroll=100.0, sizing="kelly", kelly_scale=0.25,
                   max_stake_frac=0.05)
    ok &= big["trades_log"][0]["stake"] >= small["trades_log"][0]["stake"]
    ok &= big["trades_log"][0]["stake"] <= 100.0 * 0.05 + 1e-9
    print(f"  simulate(kelly) -> stake {big['trades_log'][0]['stake']} at gap 0.30, "
          f"{small['trades_log'][0]['stake']} at gap 0.05, cap 5.00")

    # The bootstrap must be honest about a small sample: same sign, wide range.
    lo, mid, hi = bootstrap_ci([0.9, 0.9, -1.0, -1.0, 0.9])
    ok &= lo < 0 < hi
    print(f"  bootstrap_ci    -> {lo} / {mid} / {hi} straddles zero on 5 rows")
    ok &= "indistinguishable" in _profitability_verdict(rows, [0.9, -1.0])
    ok &= "no trades" in _profitability_verdict([], [])

    pairs = []
    for i in range(200):
        p = 0.30 + 0.40 * ((i % 10) / 9.0)
        true_p = min(0.98, max(0.02, 0.5 + 2.0 * (p - 0.5)))
        pairs.append((round(p, 3), 1 if ((i * 7919) % 1000) / 1000.0 < true_p else 0))
    raw = mean([logloss(p, y) for p, y in pairs])
    a, b, calibrated = platt_fit(pairs)
    ok &= a is not None and calibrated < raw
    print(f"  platt           -> a={a} b={b}, log loss {raw:.5f} -> {calibrated:.5f}")

    print(f"\n  selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
