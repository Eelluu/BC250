#!/usr/bin/env python3
"""polymarket.py — Read-only client for Polymarket's public APIs.

Shared by the market-* jobs (market-collect / market-verdict / market-score).

Every endpoint used here is PUBLIC READ: no account, no API key, no KYC, no
wallet. That is deliberate — it keeps the measurement pipeline usable from any
jurisdiction. Trading is not implemented here, and for some jurisdictions it is
not permitted at all: Polymarket itself reports Poland as a restricted
(close-only) jurisdiction, and Kalshi requires US-eligible, KYC-verified
accounts. `geoblock()` is called and recorded so the logs state this rather
than hide it.

Sources:
  Gamma API     https://gamma-api.polymarket.com/markets   market metadata
  CLOB API      https://clob.polymarket.com                price / order book
  Geoblock      https://polymarket.com/api/geoblock        jurisdiction check

Usage:
    from polymarket import geoblock, iter_markets, parse_market, classify

    print(geoblock())
    raw = list(list_markets(limit=5))[0]
    m = parse_market(raw)
    print(m["kind"], m["price_yes"], m["question"])

CLI:
    polymarket.py --geoblock
    polymarket.py --markets --limit 5 [--kinds fuzzy] [--hours 72]
    polymarket.py --selftest            # offline: parsing + cost model

Location on bc250: /opt/netscan/polymarket.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
GEOBLOCK = "https://polymarket.com/api/geoblock"

# Polymarket's edge sits behind a WAF; an honest UA is required, and anonymous
# urllib UAs get 403 far more often.
UA = "bc250-market/1.0 (+read-only; https://github.com/Eelluu/BC250)"

DEFAULT_TIMEOUT = 25
DEFAULT_RETRIES = 3

# ── Carrier workaround: pin a host to a working IP (opt-in) ────────────────
#
#   MARKET_RESOLVE="gamma-api.polymarket.com=104.18.0.1,clob.polymarket.com=104.18.0.1"
#
# Why this exists, in one line: some networks reset the TLS connection to the
# *specific* Cloudflare anycast IPs that public DNS returns for these hosts,
# while other Cloudflare IPs serve the very same host fine — verified with the
# same SNI against six Cloudflare IPs, four of which answered HTTP 200. So this
# is a network workaround, not an API trick; it is announced on stderr, recorded
# in every snapshot, and every call falls back to DNS automatically if the
# pinned IP stops answering. TLS/SNI and the Host header still use the real
# hostname: only the address the socket dials changes.
_RESOLVE = {}
_REAL_GETADDRINFO = socket.getaddrinfo


def _pinned_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _REAL_GETADDRINFO(_RESOLVE.get(host, host), port, family, type, proto, flags)


def resolve_overrides():
    """The active host->IP pins (empty dict when running through normal DNS)."""
    return dict(_RESOLVE)


def install_resolve(spec):
    """Parse "host=ip,host=ip" and pin those names. Returns the active mapping."""
    for item in (spec or "").split(","):
        if "=" not in item:
            continue
        host, ip = (part.strip() for part in item.split("=", 1))
        if host and ip:
            _RESOLVE[host] = ip
    if _RESOLVE:
        socket.getaddrinfo = _pinned_getaddrinfo
        print("  [polymarket] DNS pinned: "
              + ", ".join(f"{h} -> {ip}" for h, ip in _RESOLVE.items()), file=sys.stderr)
    return dict(_RESOLVE)


def _uninstall_resolve():
    _RESOLVE.clear()
    socket.getaddrinfo = _REAL_GETADDRINFO


install_resolve(os.environ.get("MARKET_RESOLVE", ""))

# ── HTTP ───────────────────────────────────────────────────────────────────


def _headers():
    return {"User-Agent": UA, "Accept": "application/json"}


def _do(req, timeout, retries, allow_fallback=True):
    """Perform a request with bounded retries on 429/5xx and transport errors.

    Never raises for an HTTP status: returns (status, payload_or_error_dict) so
    callers decide. A collector must not die because one page blipped.
    """
    last = None
    for attempt in range(max(1, retries)):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                try:
                    return resp.status, json.loads(body)
                except json.JSONDecodeError as ex:
                    return resp.status, {"_error": f"bad json: {ex}",
                                         "_body": body[:400].decode("utf-8", "replace")}
        except urllib.error.HTTPError as ex:
            last = {"_error": f"HTTP {ex.code}", "_url": req.full_url}
            if ex.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return ex.code, last
        except Exception as ex:
            last = {"_error": f"{type(ex).__name__}: {ex}", "_url": req.full_url}
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
    # A pinned IP can go stale (anycast moves, filters lift). Try once more the
    # ordinary way instead of failing the run: _uninstall_resolve makes the
    # recursion below a plain DNS attempt.
    if allow_fallback and _RESOLVE:
        print(f"  [polymarket] pinned IP failed for {req.full_url} — retrying via DNS",
              file=sys.stderr)
        _uninstall_resolve()
        return _do(req, timeout, retries, allow_fallback=False)
    return 0, last or {"_error": "unreachable", "_url": req.full_url}


def http_json(url, params=None, timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES):
    """GET a JSON endpoint. Returns (status, payload_or_error_dict)."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
    return _do(urllib.request.Request(url, headers=_headers()), timeout, retries)


def http_json_post(url, body, timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES):
    """POST a JSON body. Same never-raise contract as http_json."""
    data = json.dumps(body).encode()
    headers = dict(_headers(), **{"Content-Type": "application/json"})
    return _do(urllib.request.Request(url, data=data, headers=headers), timeout, retries)


def geoblock(timeout=DEFAULT_TIMEOUT):
    """Ask Polymarket how it classifies this IP. Returns a dict, never raises.

    Shape: {"blocked": bool, "ip": str, "country": str, "region": str}
    Recorded in every snapshot: the pipeline documents where it runs.
    """
    status, payload = http_json(GEOBLOCK, timeout=timeout, retries=2)
    if status != 200 or not isinstance(payload, dict):
        return {"blocked": None, "country": None, "region": None, "ip": None,
                "error": (payload or {}).get("_error", f"HTTP {status}")}
    return {
        "blocked": payload.get("blocked"),
        "country": payload.get("country"),
        "region": payload.get("region"),
        "ip": payload.get("ip"),
    }


# ── Gamma: markets ─────────────────────────────────────────────────────────

# Documented query parameters we expose. Anything else goes through `extra`.
_MARKET_PARAMS = (
    "limit", "offset", "order", "ascending", "closed",
    "liquidity_num_min", "volume_num_min", "volume_num_max",
    "end_date_min", "end_date_max", "start_date_min", "start_date_max",
    "tag_id", "related_tags", "uma_resolution_status", "include_tag",
)


def list_markets(**filters):
    """One page of markets (raw Gamma dicts). Filters -> query params."""
    params = {}
    for key, value in filters.items():
        if key not in _MARKET_PARAMS or value is None:
            continue
        # Gamma wants lowercase booleans; Python's urlencode would send "False".
        params[key] = ("true" if value is True else "false" if value is False else value)
    params.setdefault("limit", 100)
    params.setdefault("closed", "false")
    params.update(filters.get("extra") or {})
    status, payload = http_json(f"{GAMMA}/markets", params)
    if status != 200 or not isinstance(payload, list):
        err = (payload or {}).get("_error", f"HTTP {status}") if isinstance(payload, dict) else f"HTTP {status}"
        raise MarketApiError(f"list_markets failed: {err}")
    return payload


class MarketApiError(RuntimeError):
    pass


def iter_markets(max_pages=10, page_size=100, sleep=0.4, **filters):
    """Yield raw markets across pages, stopping when a page is short/empty.

    `max_pages` is a hard stop: this box shares one GPU and one uplink with the
    rest of the stack, so no job gets to page forever.
    """
    offset = 0
    for _ in range(max(1, max_pages)):
        page = list_markets(limit=page_size, offset=offset, **filters)
        if not page:
            return
        for raw in page:
            yield raw
        if len(page) < page_size:
            return
        offset += page_size
        if sleep:
            time.sleep(sleep)


def get_markets_by_id(ids):
    """Resolve market ids in bulk (Gamma accepts up to 100 per call)."""
    out = []
    ids = [str(i) for i in ids]
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        status, payload = http_json(f"{GAMMA}/markets",
                                    {"id": chunk, "closed": "any", "limit": len(chunk)})
        if status == 200 and isinstance(payload, list):
            out.extend(payload)
    return out


# ── CLOB: prices and books ─────────────────────────────────────────────────


def _parse_json_field(value):
    """Gamma returns arrays-of-things as JSON *strings*. Decode defensively."""
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        got = json.loads(value)
        return got if isinstance(got, (list, dict)) else []
    except json.JSONDecodeError:
        return []


def clob_price(token_id, side="buy", timeout=DEFAULT_TIMEOUT):
    """Best price for one outcome token. side: buy -> ask, sell -> bid."""
    status, payload = http_json(f"{CLOB}/price", {"token_id": token_id, "side": side},
                                timeout=timeout)
    if status == 200 and isinstance(payload, dict) and "price" in payload:
        try:
            return float(payload["price"])
        except (TypeError, ValueError):
            return None
    return None


def clob_midpoint(token_id, timeout=DEFAULT_TIMEOUT):
    """Midpoint of one token. Documented shape: {"mid_price": "0.45"}."""
    status, payload = http_json(f"{CLOB}/midpoint", {"token_id": token_id},
                                timeout=timeout)
    if status == 200 and isinstance(payload, dict):
        for key in ("mid_price", "mid"):  # mid_price per the published spec
            if _isnum(payload.get(key)):
                return float(payload[key])
    return None


def clob_midpoints(token_ids, timeout=DEFAULT_TIMEOUT):
    """Bulk midpoints: one request for the whole page -> {token_id: float}.

    POST with a body, not the documented GET: the live API answers
    {"error":"Invalid payload"} to GET /midpoints?token_ids=... while POST with
    [{"token_id": ...}] returns the map correctly (verified against production).
    A GET per token would be 100x the requests for the same answer.
    """
    ids = [str(t) for t in token_ids if t]
    if not ids:
        return {}
    status, payload = http_json_post(f"{CLOB}/midpoints",
                                     [{"token_id": t} for t in ids], timeout=timeout)
    if status != 200 or not isinstance(payload, dict):
        return {}
    return {str(k): float(v) for k, v in payload.items() if _isnum(v)}


def clob_spreads(token_ids, timeout=DEFAULT_TIMEOUT):
    """Bulk quoted spreads (POST body) -> {token_id: spread}."""
    ids = [str(t) for t in token_ids if t]
    if not ids:
        return {}
    status, payload = http_json_post(f"{CLOB}/spreads",
                                     [{"token_id": t} for t in ids], timeout=timeout)
    if status != 200 or not isinstance(payload, dict):
        return {}
    return {str(k): float(v) for k, v in payload.items() if _isnum(v)}


def clob_last_trades(token_ids, timeout=DEFAULT_TIMEOUT):
    """Bulk last trade prices -> {token_id: {"price": float, "side": str}}.

    POST for the same reason as clob_midpoints: GET answers "Invalid payload".
    """
    ids = [str(t) for t in token_ids if t]
    if not ids:
        return {}
    status, payload = http_json_post(f"{CLOB}/last-trades-prices",
                                     [{"token_id": t} for t in ids], timeout=timeout)
    out = {}
    if status == 200 and isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict) and row.get("token_id"):
                out[str(row["token_id"])] = {
                    "price": _f(row.get("price")),
                    "side": row.get("side") or "",
                }
    return out


def clob_fee_rate(token_id, timeout=DEFAULT_TIMEOUT):
    """Taker base fee in basis points. Documented example: {"base_fee": 30}.

    Most Polymarket markets are fee-free and some sports markets are not, so
    the score job asks per market instead of guessing from Gamma's feeSchedule.
    """
    status, payload = http_json(f"{CLOB}/fee-rate", {"token_id": token_id},
                                timeout=timeout)
    if status == 200 and isinstance(payload, dict) and _isnum(payload.get("base_fee")):
        return float(payload["base_fee"])
    return None


def server_time(timeout=DEFAULT_TIMEOUT):
    """CLOB server time (epoch seconds) — for clock-skew checks."""
    status, payload = http_json(f"{CLOB}/time", timeout=timeout)
    return payload if (status == 200 and _isnum(payload)) else None


def clob_book(token_id, timeout=DEFAULT_TIMEOUT):
    """Raw order book: {"bids": [...], "asks": [...]} (best last/first per doc)."""
    status, payload = http_json(f"{CLOB}/book", {"token_id": token_id}, timeout=timeout)
    if status == 200 and isinstance(payload, dict):
        return payload
    return None


def clob_spread(token_id, timeout=DEFAULT_TIMEOUT):
    """(best_bid, best_ask, spread) from the live book, or (None, None, None)."""
    book = clob_book(token_id, timeout=timeout)
    if not book:
        return None, None, None
    bids = [(float(b["price"]), float(b.get("size", 0))) for b in book.get("bids") or []
            if _isnum(b.get("price"))]
    asks = [(float(a["price"]), float(a.get("size", 0))) for a in book.get("asks") or []
            if _isnum(a.get("price"))]
    best_bid = max((p for p, _ in bids), default=None)
    best_ask = min((p for p, _ in asks), default=None)
    spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None
    return best_bid, best_ask, spread


def prices_history(token_id, interval="1h", fidelity=10, timeout=DEFAULT_TIMEOUT):
    """OHLC-ish history for one token: [{"t": epoch, "p": price}, ...]."""
    status, payload = http_json(f"{CLOB}/prices-history",
                                {"market": token_id, "interval": interval,
                                 "fidelity": fidelity}, timeout=timeout)
    if status == 200 and isinstance(payload, dict) and isinstance(payload.get("history"), list):
        return payload["history"]
    return []


def _isnum(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ── Normalisation ──────────────────────────────────────────────────────────


def _hours_to(end_iso, now=None):
    if not end_iso:
        return None
    txt = str(end_iso).strip().replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = (datetime.fromisoformat(txt) if fmt is None
                  else datetime.strptime(txt, fmt))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return (dt - (now or datetime.now(timezone.utc))).total_seconds() / 3600.0
        except (ValueError, TypeError):
            continue
    return None


def parse_market(raw):
    """Flatten a raw Gamma market into the fields the pipeline actually uses.

    Notes on the source format (verified against the published OpenAPI spec):
    `outcomes`, `outcomePrices` and `clobTokenIds` are JSON-encoded STRINGS,
    not arrays, and prices are decimal strings ('0.62') or already floats
    depending on endpoint/version — both are handled.
    """
    outcomes = [str(o) for o in _parse_json_field(raw.get("outcomes"))]
    prices = [_f(p) for p in _parse_json_field(raw.get("outcomePrices"))]
    token_ids = [str(t) for t in _parse_json_field(raw.get("clobTokenIds"))]

    events = raw.get("events") or []
    event = events[0] if isinstance(events, list) and events else {}
    tags = [t.get("label") or t.get("slug")
            for t in (raw.get("tags") or []) if isinstance(t, dict)]
    event_tags = [t.get("label") or t.get("slug")
                  for t in (event.get("tags") or []) if isinstance(t, dict)]
    if not tags:  # event-level tags carry the topical category when market tags don't
        tags = event_tags

    price_yes = prices[0] if prices else None
    price_no = prices[1] if len(prices) > 1 else (1.0 - price_yes if price_yes is not None else None)

    slug = raw.get("slug")
    event_slug = event.get("slug")
    url = (f"https://polymarket.com/event/{event_slug}" if event_slug
           else f"https://polymarket.com/market/{slug}" if slug else None)

    market = {
        "id": str(raw.get("id")),
        "question": raw.get("question"),
        "slug": slug,
        "condition_id": raw.get("conditionId"),
        "description": raw.get("description"),
        "resolution_source": raw.get("resolutionSource"),
        "end_date": raw.get("endDate") or raw.get("endDateIso"),
        "start_date": raw.get("startDate"),
        "created_at": raw.get("createdAt"),
        "hours_to_end": _hours_to(raw.get("endDate") or raw.get("endDateIso")),
        "outcomes": outcomes,
        "prices": prices,
        "token_ids": token_ids,
        "price_yes": price_yes,
        "price_no": price_no,
        "best_bid": _f(raw.get("bestBid")),
        "best_ask": _f(raw.get("bestAsk")),
        "last_trade": _f(raw.get("lastTradePrice")),
        "spread": _f(raw.get("spread")),
        "volume": _f(raw.get("volumeNum")) or _f(raw.get("volume")),
        "volume24h": _f(raw.get("volume24hr")),
        "liquidity": _f(raw.get("liquidityNum")) or _f(raw.get("liquidity")),
        "neg_risk": bool(raw.get("negRisk") or event.get("negRisk")),
        "active": bool(raw.get("active")),
        "closed": bool(raw.get("closed")),
        "accepting_orders": bool(raw.get("acceptingOrders")),
        "order_book": bool(raw.get("enableOrderBook")),
        "uma_status": raw.get("umaResolutionStatus"),
        "tick_size": _f(raw.get("orderPriceMinTickSize")),
        "min_size": _f(raw.get("orderMinSize")),
        "fee_bps": market_fee_bps(raw, price_yes),
        "tags": tags,
        "event_title": event.get("title"),
        "event_slug": event_slug,
        "url": url,
        "market_type": raw.get("marketType"),
    }
    market["kind"] = classify(market)
    market["edge_required"] = edge_required(market)
    return market


def market_fee_bps(raw, price=None):
    """Taker fee in basis points for this market, per the documented formula.

        fee (USDC per share) = feeRate x p x (1 - p)

    That is the formula published at docs.polymarket.com/trading/fees: the fee
    peaks at 50% probability, is symmetric either side of it, and only takers
    pay. It matters a lot here — reading `rate` as a flat percentage would
    overstate the cost of entering at p=0.675 by roughly 4.5x, and the cost
    model is the whole point of this pipeline. Geopolitical and world-events
    markets are fee-free and simply carry no feeSchedule.

    One share pays 1 USDC, so a fee expressed in USDC per share is numerically
    the same as a fee in probability units — exactly what edge_required() adds.
    """
    sched = raw.get("feeSchedule")
    if not (isinstance(sched, dict) and _isnum(sched.get("rate"))):
        return 0.0
    rate = _f(sched.get("rate"), 0.0)
    exponent = _f(sched.get("exponent"), 1.0) or 1.0

    p = _f(price)
    if p is None:
        prices = [_f(x) for x in _parse_json_field(raw.get("outcomePrices"))]
        p = prices[0] if prices and prices[0] is not None else 0.5
    p = min(max(p, 0.0), 1.0)

    return round(rate * (p * (1.0 - p)) ** exponent * 10000.0, 4)


# Markers for a market whose truth is a NUMBER that a feed publishes. Matched
# as whole words (see _marker_regex): the abbreviation "ath" must not fire on
# "weather", and "et on" only ever means a US-Eastern timestamp.
_FEED_MARKERS = (
    "up or down", "up/down", "higher or lower", "above $", "below $",
    "close above", "close below", "price of", "price above", "price below",
    "market cap", "gas price", "hashrate", "all-time high", "ath",
    "twap", "candle", "at 12:00", "at 3:00", "at 4:00", "et on",
    "hit $", "reach $", "dip to", "touch $",
)
# Words that mark a market whose truth is a measurable real-world EVENT.
_EVENT_MARKERS = (
    "weather", "temperature", "hurricane", "snow", "rain", "storm",
    "earthquake", "launch", "landing", "will win", "wins the", "beat the",
    "score", "points", "goals", "match", "game", "tournament", "election day",
    "turnout", "unemployment", "cpi", "inflation", "gdp", "interest rate",
    "fed", "bps", "rate cut", "rate hike", "earnings",
)


def _marker_regex(markers):
    """Whole-word alternation for the keyword heuristics.

    Substring matching is how you end up classifying a snow market as a price
    feed because "weather" contains "ath".
    """
    parts = []
    for m in markers:
        esc = re.escape(m)
        left = r"\b" if m[0].isalnum() else ""
        right = r"\b" if m[-1].isalnum() else ""
        parts.append(f"{left}{esc}{right}")
    return re.compile("|".join(parts))


_FEED_RE = _marker_regex(_FEED_MARKERS)
_EVENT_RE = _marker_regex(_EVENT_MARKERS)


def classify(market):
    """Tag a market by what its truth is made of: feed | event | fuzzy.

      feed  — resolved by a price/index print (crypto up/down, "above $X at
              3PM"). A numeric feed beats any language model here: the model
              cannot add information the feed does not already have.
      event — resolved by a measurable real-world event (weather, sports,
              macro prints). Reasonable but heavily modelled by the crowd.
      fuzzy — resolved by judgement on news/politics/awards/etc., where the
              rules text is the hard part and that is exactly the LLM's job.

    Heuristic on purpose: it only decides *where the GPU budget goes*, and
    market-verdict can override it with --kinds.
    """
    return classify_reason(market)[0]


def classify_reason(market):
    """classify(), plus why — so the heuristic can be tuned against real data
    and a misroute is visible in one line instead of being a silent mystery."""
    question = (market.get("question") or "").lower()
    desc = (market.get("description") or "").lower()
    tags = " ".join(str(t).lower() for t in (market.get("tags") or []))
    blob = f"{question} {desc} {tags}"

    hit = _FEED_RE.search(blob)
    if hit:
        return "feed", f'feed marker "{hit.group(0)}"'
    hit = _EVENT_RE.search(blob)
    if hit:
        return "event", f'event marker "{hit.group(0)}"'

    hours = market.get("hours_to_end")
    if hours is not None and 0 <= hours <= 24 and any(
            t in tags for t in ("crypto", "crypto prices", "prices",
                                "bitcoin", "ethereum")):
        return "feed", f"crypto/prices tag with {hours:.1f}h to close"
    return "fuzzy", "no marker (judgement market)"


DEFAULT_SLIPPAGE_BPS = 50.0
DEFAULT_EDGE_BPS = 200.0


def edge_required(market, slippage_bps=DEFAULT_SLIPPAGE_BPS, min_edge_bps=DEFAULT_EDGE_BPS):
    """Edge (in probability units) a bet must clear to be worth paper-trading.

    Half the quoted spread + the taker fee + an explicit slippage allowance +
    a floor. The fee comes from market_fee_bps(), which follows the published
    formula (a function of price, not a flat rate). This is the honest cost of
    trading a thin book, and it is what turns "the model says 0.70 and the
    market says 0.62" into "that is not an edge, that is noise plus spread and
    fee".
    """
    half_spread = 0.0
    spread = market.get("spread")
    if _isnum(spread):
        half_spread = max(0.0, float(spread)) / 2.0
    elif market.get("best_bid") is not None and market.get("best_ask") is not None:
        half_spread = max(0.0, (market["best_ask"] - market["best_bid"])) / 2.0

    fee = (market.get("fee_bps") or 0.0) / 10000.0
    return round(half_spread + fee + slippage_bps / 10000.0 + min_edge_bps / 10000.0, 4)


# ── CLI ────────────────────────────────────────────────────────────────────

FIXTURE = [
    {   # feed market: numeric feed decides, the LLM has nothing to add
        "id": "900001", "question": "Bitcoin Up or Down — October 15, 3:00 PM ET",
        "slug": "bitcoin-up-or-down-oct-15-3pm", "conditionId": "0xfeed1",
        "description": "This market resolves to 'Up' if the Binance BTC/USDT "
                       "close price at 3:00 PM ET is higher than at 2:00 PM ET.",
        "resolutionSource": "https://www.binance.com/en/trade/BTC_USDT",
        "endDate": "2026-10-15T19:00:00Z", "outcomes": '["Up", "Down"]',
        "outcomePrices": '["0.52", "0.48"]', "clobTokenIds": '["11", "22"]',
        "volumeNum": 120000.5, "liquidityNum": 45000.0, "volume24hr": 90000.0,
        "spread": 0.02, "bestBid": 0.51, "bestAsk": 0.53, "active": True,
        "closed": False, "acceptingOrders": True, "enableOrderBook": True,
        "negRisk": False, "orderPriceMinTickSize": 0.01, "orderMinSize": 5,
        "feeSchedule": {"rate": 0.07, "exponent": 1, "takerOnly": True},
        "tags": [{"label": "Crypto"}, {"label": "Bitcoin"}],
        "events": [{"slug": "bitcoin-up-or-down-oct-15", "title": "Bitcoin Up or Down"}],
    },
    {   # fuzzy market: judgement on news, rules text is the hard part
        "id": "900002",
        "question": "Will the Polish Sejm pass the media ownership bill by 30 Nov?",
        "slug": "polish-sejm-media-bill", "conditionId": "0xfuzzy1",
        "description": "Resolves Yes if the Sejm votes in favour of the media "
                       "ownership bill and the President signs it before "
                       "November 30, 2026. Resolves No if the bill is rejected, "
                       "withdrawn, or not signed by that date.",
        "resolutionSource": "Official Sejm voting records",
        "endDate": "2026-11-30T23:59:00Z", "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.34", "0.66"]', "clobTokenIds": '["33", "44"]',
        "volumeNum": 8500.0, "liquidityNum": 3200.0, "volume24hr": 400.0,
        "spread": 0.06, "bestBid": 0.31, "bestAsk": 0.37, "active": True,
        "closed": False, "acceptingOrders": True, "enableOrderBook": True,
        "negRisk": False, "tags": [{"label": "Politics"}],
        "events": [{"slug": "polish-politics-2026", "title": "Polish Politics"}],
    },
    {   # event market: measurable real-world event
        "id": "900003", "question": "Will Warsaw see snow before 1 December?",
        "slug": "warsaw-snow-november", "conditionId": "0xevent1",
        "description": "Resolves Yes on the first measurable snowfall reported "
                       "at Warsaw Chopin Airport before December 1.",
        "resolutionSource": "IMGW",
        "endDate": "2026-12-01T00:00:00Z", "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.41", "0.59"]', "clobTokenIds": '["55", "66"]',
        "volumeNum": 2100.0, "liquidityNum": 900.0, "spread": 0.08,
        "active": True, "closed": False, "acceptingOrders": True,
        "enableOrderBook": True, "tags": [{"label": "Weather"}],
        "events": [{"slug": "warsaw-weather", "title": "Warsaw Weather"}],
    },
]


def selftest():
    """Offline check: parsing, classification and the cost model."""
    ok = True
    expect_kinds = {"900001": "feed", "900002": "fuzzy", "900003": "event"}
    print("  fixture -> parse_market + classify + edge_required")
    for raw in FIXTURE:
        m = parse_market(raw)
        kind, reason = classify_reason(m)
        m["kind"] = kind
        kind_ok = kind == expect_kinds[m["id"]]
        ok &= kind_ok
        print(f"    {m['id']}  kind={m['kind']:<6} (want {expect_kinds[m['id']]:<6} "
              f"{'OK' if kind_ok else 'FAIL'})  p_yes={m['price_yes']}  "
              f"spread={m['spread']}  edge_required={m['edge_required']}  "
              f"fee={m['fee_bps']}bps  url={m['url']}")
        print(f"            why: {reason}")
        assert isinstance(m["outcomes"], list) and m["outcomes"], "outcomes not decoded"
        assert m["token_ids"] and isinstance(m["token_ids"][0], str), "token ids not decoded"

    # Cost model: an edge must clear half-spread + slippage + floor. Check both
    # sides of the threshold, and that a tighter book lowers the bar.
    fuzzy = parse_market(FIXTURE[1])
    feed = parse_market(FIXTURE[0])
    need = fuzzy["edge_required"]
    big, small = 0.08, 0.02
    ok &= big > need and small <= need
    ok &= feed["edge_required"] < need  # tighter spread -> lower bar
    print(f"    cost model: need |edge| > {need}  ->  +{big:.2f} "
          f"{'clears' if big > need else 'does NOT clear'} (want: clears), "
          f"+{small:.2f} {'clears' if small > need else 'does NOT clear'} "
          f"(want: does not clear)")
    print(f"    cost model: feed market need {feed['edge_required']} < fuzzy "
          f"{need} (tighter book, lower bar)")

    # Fee model: the published formula is fee = feeRate x p x (1 - p), NOT a
    # flat rate. Crypto is 0.07, so p=0.52 gives 174.7 bps — reading 0.07 as a
    # flat percentage would have said 700 bps and doubled the required edge.
    want = 0.07 * 0.52 * 0.48 * 10000.0
    ok &= abs(feed["fee_bps"] - want) < 1.0
    ok &= fuzzy["fee_bps"] == 0.0
    print(f"    fee model: crypto p=0.52 -> {feed['fee_bps']} bps "
          f"(formula {want:.1f}); no feeSchedule -> {fuzzy['fee_bps']} bps")

    # Resolve override: pin a host, confirm the socket path uses it, unpin.
    install_resolve("gamma-api.polymarket.com=104.18.0.1")
    try:
        pinned = socket.getaddrinfo("gamma-api.polymarket.com", 443,
                                    type=socket.SOCK_STREAM)[0][4][0]
    except OSError as ex:
        pinned = f"error: {ex}"
    ok &= pinned == "104.18.0.1"
    print(f"    resolve override: gamma-api -> {pinned} (want 104.18.0.1)")
    _uninstall_resolve()
    ok &= resolve_overrides() == {}
    print(f"    resolve override cleared: {resolve_overrides()} (want {{}})")

    # Geoblock parsing with a stubbed payload (network is optional here).
    print("  geoblock shape: {\"blocked\": bool, \"country\": str, ...} — "
          "queried live by market-collect, not by the selftest")
    print(f"\n  selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only Polymarket client")
    ap.add_argument("--geoblock", action="store_true", help="ask the geoblock endpoint")
    ap.add_argument("--markets", action="store_true", help="list markets (live)")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--order", default="volume24hr")
    ap.add_argument("--hours", type=float, help="only markets ending within N hours")
    ap.add_argument("--kinds", help="comma list: feed,event,fuzzy")
    ap.add_argument("--min-volume", type=float, default=0)
    ap.add_argument("--selftest", action="store_true", help="offline checks, no network")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    if args.geoblock:
        print(json.dumps(geoblock(), indent=2))
        return 0

    if args.markets:
        want = {k.strip() for k in args.kinds.split(",")} if args.kinds else None
        shown = 0
        for raw in iter_markets(max_pages=3, page_size=100, order=args.order,
                                ascending=False, volume_num_min=args.min_volume or None):
            m = parse_market(raw)
            if want and m["kind"] not in want:
                continue
            if args.hours is not None and (m["hours_to_end"] is None
                                           or m["hours_to_end"] > args.hours):
                continue
            print(f"  {m['id']:<8} {m['kind']:<6} p={m['price_yes']} "
                  f"h={m['hours_to_end'] and round(m['hours_to_end'], 1)} "
                  f"vol={m['volume'] and int(m['volume'])} "
                  f"edge>={m['edge_required']}  {(m['question'] or '')[:70]}")
            shown += 1
            if shown >= args.limit:
                break
        if not shown:
            print("  (no market matched)")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
