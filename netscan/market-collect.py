#!/usr/bin/env python3
"""market-collect.py - Polymarket discovery + price snapshots (read-only).

Pulls candidate markets from the public Gamma API, overlays live prices and
spreads in bulk from the public CLOB API, and writes one snapshot per run.

This job only READS. There is no account, no API key, no wallet and no order
endpoint anywhere in it. That is deliberate on two counts: it keeps the
measurement pipeline runnable from any jurisdiction, and it means a bug can
never move money. The trading side is a separate, deliberate decision — see the
"Prediction markets" section of the Readme for the restrictions that apply.

Output: /opt/netscan/data/markets/
  - raw-YYYYMMDD.jsonl       append-only price history (one line per market/run)
  - snapshot-YYYYMMDD.json   merged state for the day (later runs win per market)
  - latest-markets.json      symlink to the newest snapshot
  - geoblock.json            jurisdiction record: what Polymarket says about us

Usage:
  market-collect.py                                  # defaults: liquid, <72h
  market-collect.py --kinds fuzzy --hours-max 720    # long-tail judgement markets
  market-collect.py --from-file gamma-page.json      # offline, no network
  market-collect.py --selftest                       # offline fixture run

Schedule: hourly via queue-runner cron (cheap: a few hundred KB, ~4 requests).
The GPU stage is market-verdict.py, which reads latest-markets.json in the
night window.

Location on bc250: /opt/netscan/market-collect.py
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from polymarket import (  # noqa: E402
    FIXTURE,
    clob_last_trades,
    clob_midpoints,
    clob_spreads,
    edge_required,
    geoblock,
    iter_markets,
    parse_market,
)

DEFAULT_OUT = Path(os.environ.get("MARKET_DATA") or "/opt/netscan/data/markets")
ALL_KINDS = ("feed", "event", "fuzzy")


def log(msg, quiet=False):
    if not quiet:
        print(f"  {msg}", flush=True)


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Selection ──────────────────────────────────────────────────────────────


def select_markets(raw_markets, min_volume=5000.0, min_liquidity=1000.0,
                   hours_max=72.0, kinds=None, exclude_tags=(), max_markets=300):
    """Filter and cap candidates. Server-side filters are applied by the caller;
    this is the belt-and-braces pass, and it logs what it dropped and why."""
    chosen = []
    exclude = {str(t).lower() for t in exclude_tags}
    skipped = {"closed": 0, "low_volume": 0, "low_liquidity": 0,
               "horizon": 0, "kind": 0, "tag": 0, "no_prices": 0}
    for raw in raw_markets:
        m = parse_market(raw)
        if m["closed"] or not m["active"] or not m["accepting_orders"]:
            skipped["closed"] += 1
            continue
        if min_volume and (m["volume"] or 0) < min_volume:
            skipped["low_volume"] += 1
            continue
        if min_liquidity and (m["liquidity"] or 0) < min_liquidity:
            skipped["low_liquidity"] += 1
            continue
        hours = m["hours_to_end"]
        if hours_max and (hours is None or hours < 0 or hours > hours_max):
            skipped["horizon"] += 1
            continue
        if kinds and m["kind"] not in kinds:
            skipped["kind"] += 1
            continue
        if exclude & {str(t).lower() for t in m["tags"]}:
            skipped["tag"] += 1
            continue
        if m["price_yes"] is None and not m["token_ids"]:
            skipped["no_prices"] += 1
            continue
        chosen.append(m)
        if len(chosen) >= max_markets:
            break
    return chosen, skipped


# ── Pricing (bulk, public CLOB) ────────────────────────────────────────────


def overlay_live_prices(markets, batch=100, use_clob=True, quiet=False):
    """Overlay live CLOB midpoints/spreads/last-trades onto Gamma metadata.

    Gamma's `outcomePrices` is a snapshot that can be minutes stale; the CLOB
    mid is what a taker would actually face. Batched: three calls per 100
    markets instead of three per market.
    """
    if not use_clob or not markets:
        return 0
    token_ids = []
    for m in markets:
        token_ids.extend(m["token_ids"][:2])

    mids, spreads, trades = {}, {}, {}
    for i in range(0, len(token_ids), batch):
        chunk = token_ids[i:i + batch]
        mids.update(clob_midpoints(chunk))
        spreads.update(clob_spreads(chunk))
        trades.update(clob_last_trades(chunk))
    log(f"CLOB bulk: {len(mids)} midpoints, {len(spreads)} spreads, "
        f"{len(trades)} last trades for {len(markets)} markets", quiet)

    priced = 0
    for m in markets:
        ids = m["token_ids"]
        if ids:
            yes_mid = mids.get(ids[0])
            if yes_mid is not None:
                m["price_yes_mid"] = yes_mid
                m["price_yes"] = yes_mid
                m["price_no_mid"] = mids.get(ids[1]) if len(ids) > 1 else None
                priced += 1
            if (spread := spreads.get(ids[0])) is not None:
                m["spread_clob"] = spread
                m["spread"] = spread
            if (last := trades.get(ids[0])):
                m["last_trade"] = last.get("price")
                m["last_trade_side"] = last.get("side")
        m["edge_required"] = edge_required(m)
    return priced


# ── Output ─────────────────────────────────────────────────────────────────


def write_outputs(out_dir, markets, meta):
    """Write the append-only history and the merged day snapshot.

    Returns (snapshot_path, latest_path). Re-running on the same day merges by
    market id (later wins), so an hourly job accumulates history in the JSONL
    while `latest-markets.json` stays a single coherent view for the LLM stage.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    snap = out_dir / f"snapshot-{stamp}.json"

    existing = {}
    if snap.exists():
        try:
            for m in json.loads(snap.read_text(encoding="utf-8")).get("markets", []):
                existing[m["id"]] = m
        except (json.JSONDecodeError, OSError, KeyError):
            existing = {}
    for m in markets:
        existing[m["id"]] = m

    doc = {
        "generated_at": meta["generated_at"],
        "geoblock": meta["geoblock"],
        "source": meta["source"],
        "counts": meta["counts"],
        "markets": sorted(existing.values(),
                          key=lambda x: -(x.get("volume24h") or 0)),
    }
    blob = json.dumps(doc, indent=1, ensure_ascii=False)
    snap.write_text(blob, encoding="utf-8")

    latest = out_dir / "latest-markets.json"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(snap.name)
    except OSError:  # filesystems without symlinks (or Windows without privilege)
        latest.write_text(blob, encoding="utf-8")

    with (out_dir / f"raw-{stamp}.jsonl").open("a", encoding="utf-8") as fh:
        for m in markets:
            fh.write(json.dumps({"collected_at": doc["generated_at"], **m},
                                ensure_ascii=False) + "\n")

    (out_dir / "geoblock.json").write_text(
        json.dumps({"checked_at": doc["generated_at"], **meta["geoblock"]},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    return snap, latest


# ── Report ─────────────────────────────────────────────────────────────────


def print_report(markets, meta, quiet=False):
    counts = meta["counts"]
    log(f"selected {len(markets)} markets  "
        f"(feed={counts.get('feed', 0)} event={counts.get('event', 0)} "
        f"fuzzy={counts.get('fuzzy', 0)})", quiet)
    skipped = meta.get("skipped") or {}
    if any(skipped.values()):
        log("dropped: " + ", ".join(f"{k}={v}" for k, v in skipped.items() if v), quiet)
    gb = meta["geoblock"]
    if gb.get("blocked") is True:
        log(f"GEOBLOCK: Polymarket reports this IP as restricted "
            f"(country={gb.get('country')}) — read-only, no trading", quiet)
    elif gb.get("blocked") is False:
        log(f"geoblock: not restricted (country={gb.get('country')})", quiet)
    else:
        log(f"geoblock: unknown ({gb.get('error')})", quiet)

    if quiet or not markets:
        return
    print(f"\n  {'id':<9} {'kind':<6} {'p_yes':>6} {'spread':>7} "
          f"{'edge>':>6} {'vol24h':>9} {'h':>7}  question")
    print("  " + "-" * 104)
    for m in markets[:15]:
        print(f"  {m['id']:<9} {m['kind']:<6} "
              f"{_fmt(m['price_yes'], 6)} {_fmt(m.get('spread'), 7)} "
              f"{_fmt(m['edge_required'], 6)} {_fmt(m['volume24h'], 9, 0)} "
              f"{_fmt(m['hours_to_end'], 7, 1)}  {(m['question'] or '')[:52]}")
    if len(markets) > 15:
        print(f"  ... and {len(markets) - 15} more")


def _fmt(value, width, nd=3):
    if value is None:
        return " " * (width - 1) + "-"
    try:
        return f"{float(value):>{width}.{nd}f}"
    except (TypeError, ValueError):
        return f"{str(value):>{width}}"


# ── Selftest (offline) ─────────────────────────────────────────────────────


def selftest():
    """Offline end-to-end: fixture -> select -> overlay -> write -> merge.

    Runs without network access, which is the point: the pipeline can be
    verified on a machine that cannot reach Polymarket.
    """
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="market-collect-selftest-"))
    meta = {
        "generated_at": now_iso(),
        "geoblock": {"blocked": None, "country": None, "region": None, "ip": None},
        "source": {"gamma": "fixture", "clob": "skipped"},
        "counts": {},
    }

    chosen, skipped = select_markets(FIXTURE, min_volume=0, min_liquidity=0,
                                     hours_max=0)
    counts = {}
    for m in chosen:
        counts[m["kind"]] = counts.get(m["kind"], 0) + 1
    meta["counts"] = counts
    meta["skipped"] = skipped
    ok &= len(chosen) == len(FIXTURE)
    print(f"  select_markets -> {len(chosen)}/{len(FIXTURE)} chosen, counts={counts}")

    # CLOB overlay must be a no-op when disabled, and must not lose metadata.
    before = json.dumps(chosen, sort_keys=True)
    n = overlay_live_prices(chosen, use_clob=False, quiet=True)
    ok &= n == 0 and json.dumps(chosen, sort_keys=True) == before
    print(f"  overlay (--no-clob) -> priced={n} (want 0), metadata untouched")

    # Filters actually filter.
    only_fuzzy, _ = select_markets(FIXTURE, min_volume=0, min_liquidity=0,
                                   hours_max=0, kinds=("fuzzy",))
    ok &= len(only_fuzzy) == 1 and only_fuzzy[0]["kind"] == "fuzzy"
    print(f"  --kinds fuzzy -> {len(only_fuzzy)} (want 1)")

    low_vol, _ = select_markets(FIXTURE, min_volume=50000, min_liquidity=0, hours_max=0)
    ok &= len(low_vol) == 1
    print(f"  --min-volume 50000 -> {len(low_vol)} (want 1: only the 120k-volume one; "
          f"the filter uses total volume, not volume24h)")

    snap, latest = write_outputs(tmp, chosen, meta)
    ok &= snap.exists() and latest.exists()
    lines = (tmp / f"raw-{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl")
    n_lines = len(lines.read_text(encoding="utf-8").strip().splitlines())
    ok &= n_lines == len(chosen)
    print(f"  write_outputs -> {snap.name}, {latest.name}, jsonl lines={n_lines} "
          f"(want {len(chosen)})")

    # Re-run must merge, not duplicate.
    write_outputs(tmp, chosen[:1], meta)
    merged = json.loads(snap.read_text(encoding="utf-8"))["markets"]
    ok &= len(merged) == len(chosen)
    print(f"  re-run merge -> {len(merged)} markets (want {len(chosen)}, no dupes)")

    gb = json.loads((tmp / "geoblock.json").read_text(encoding="utf-8"))
    ok &= "blocked" in gb
    print(f"  geoblock.json -> keys={sorted(gb)[:4]}")

    print(f"  temp dir: {tmp}")
    print(f"\n  selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ── Main ───────────────────────────────────────────────────────────────────


def collect(args):
    meta = {
        "generated_at": now_iso(),
        "geoblock": geoblock() if not args.from_file else {"blocked": None,
                                                           "country": None,
                                                           "region": None,
                                                           "ip": None,
                                                           "error": "skipped (offline)"},
        "source": {},
        "counts": {},
    }

    end_max = None
    now = datetime.now(timezone.utc)
    if args.hours_max:
        end_max = (now + timedelta(hours=args.hours_max)).strftime("%Y-%m-%dT%H:%M:%SZ")

    if args.from_file:
        raw_markets = json.loads(Path(args.from_file).read_text(encoding="utf-8"))
        if isinstance(raw_markets, dict):  # a saved events payload
            raw_markets = [m for ev in raw_markets.get("events", [])
                           for m in ev.get("markets", [])]
        meta["source"] = {"gamma": f"file:{args.from_file}", "clob": "disabled"}
        log(f"loaded {len(raw_markets)} raw markets from {args.from_file}", args.quiet)
    else:
        raw_markets = list(iter_markets(
            max_pages=args.pages, page_size=100, order="volume24hr", ascending=False,
            closed=False, volume_num_min=args.min_volume or None,
            liquidity_num_min=args.min_liquidity or None,
            end_date_min=now.strftime("%Y-%m-%dT%H:%M:%SZ"), end_date_max=end_max,
            sleep=0.4))
        meta["source"] = {"gamma": "https://gamma-api.polymarket.com/markets",
                          "clob": "https://clob.polymarket.com",
                          "order": "volume24hr desc",
                          "params": {"volume_num_min": args.min_volume,
                                     "liquidity_num_min": args.min_liquidity,
                                     "end_date_max": end_max}}
        log(f"gamma: {len(raw_markets)} raw markets", args.quiet)

    kinds = tuple(k.strip() for k in args.kinds.split(",")) if args.kinds else None
    chosen, skipped = select_markets(
        raw_markets, min_volume=args.min_volume, min_liquidity=args.min_liquidity,
        hours_max=args.hours_max, kinds=kinds,
        exclude_tags={t.strip().lower() for t in args.exclude_tags.split(",") if t.strip()},
        max_markets=args.max_markets)

    counts = {}
    for m in chosen:
        counts[m["kind"]] = counts.get(m["kind"], 0) + 1
    meta["counts"] = counts
    meta["skipped"] = skipped

    priced = overlay_live_prices(chosen, batch=args.price_batch,
                                 use_clob=not args.no_clob and not args.from_file,
                                 quiet=args.quiet)
    if priced:
        meta["source"]["clob_priced"] = priced

    if not args.dry_run:
        snap, _ = write_outputs(args.out_dir, chosen, meta)
        log(f"wrote {snap}", args.quiet)

    print_report(chosen, meta, args.quiet)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Polymarket discovery + snapshots (read-only)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--max-markets", type=int, default=300)
    ap.add_argument("--pages", type=int, default=6, help="Gamma pages to walk (100/page)")
    ap.add_argument("--min-volume", type=float, default=5000.0)
    ap.add_argument("--min-liquidity", type=float, default=1000.0)
    ap.add_argument("--hours-max", type=float, default=72.0,
                    help="only markets ending within N hours (0 = no limit)")
    ap.add_argument("--kinds", help="feed,event,fuzzy (default: all)")
    ap.add_argument("--exclude-tags", default="", help="comma list of tag names to skip")
    ap.add_argument("--price-batch", type=int, default=100)
    ap.add_argument("--no-clob", action="store_true", help="skip live CLOB pricing")
    ap.add_argument("--from-file", help="read raw Gamma JSON from a file (offline)")
    ap.add_argument("--dry-run", action="store_true", help="do not write anything")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="offline fixture run")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    return collect(args)


if __name__ == "__main__":
    sys.exit(main())
