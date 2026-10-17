#!/usr/bin/env python3
"""
music-discover.py — Weekly music discovery for YouTube House Mixes.
Focuse on Creative Commons / Royalty Free tracks for monetization.

Two things this job cannot control, and one it can:

  * ddgr 2.2 caps --num at 25 and exits 2 *with nothing on stdout* if you ask for
    more. That read as an empty JSON document and killed the job with a
    four-word traceback, so the cap is a constant here and the exit code and
    stderr are actually looked at.
  * DuckDuckGo throttles: it answers HTTP 202 and ddgr reports the empty result
    set with an [ERROR] line on stderr. That is "no results this run", not a
    crash, and it is reported as such.
  * What it can control is the model call: the local gemma-4 models are
    reasoning models, so thinking is off for a structured-extraction task (see
    llm_sanitize.call_llm) and the reply is parsed with extract_json, which
    survives a model that writes the array and then keeps talking.

Usage:
  music-discover.py                 # the weekly run
  music-discover.py --selftest      # offline: fixtures, no network, no model

Output: /opt/netscan/data/music/
  - house-mix-YYYY-Www.json   the curated tracks
  - latest-report.md          Discord report (queue-runner DMs it to the owner)
  - rejected/*.txt           raw model replies that failed to parse
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_sanitize import call_llm, extract_json  # noqa: E402

# ─── Config ───
DATA_DIR = Path(os.environ.get("MUSIC_DATA") or "/opt/netscan/data/music")
GENRE = "House Music"
PLATFORMS = ["SoundCloud", "Bandcamp"]
# ddgr 2.2: "--num: 50 is not a non-negative integer <= 25" — and it fails by
# writing nothing to stdout, which is why this is not a tunable free of context.
DDGR_MAX_RESULTS = 25
DDGR_TIMEOUT_S = 120

# SearXNG is the primary source: one local service queries several engines and
# returns one clean JSON list, so an engine answering with a captcha is no longer
# fatal. Set up by searxng/searxng-setup.sh; binds 127.0.0.1 only.
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://127.0.0.1:8890")
SEARXNG_TIMEOUT_S = 40

# Jamendo is the source that carries the licence as DATA rather than as something
# the model has to infer from a search snippet: its Creative Commons catalogue
# returns the licence per track, so "is this safe to monetise" stops being an
# opinion. Needs a free client_id (devportal.jamendo.com) in JAMENDO_CLIENT_ID.
JAMENDO_URL = "https://api.jamendo.com/v3.0/tracks/"
JAMENDO_CLIENT_ID = os.environ.get("JAMENDO_CLIENT_ID", "")
JAMENDO_TIMEOUT_S = 30


def log(msg):
    print(f"  {msg}", flush=True)


def search_terms(genre=GENRE):
    """The query itself, without a date.

    SearXNG takes the time window as a parameter (time_range=week) and ddgr has
    no such thing, so only the ddgr flavour appends DuckDuckGo's after: operator.
    """
    return (f'site:soundcloud.com "{genre}" ("no copyright" OR "commercial use" '
            f'OR "royalty free" OR "CC-BY")')


def build_query(genre=GENRE, days=7):
    """The ddgr flavour: date in the query, because ddgr has no time_range."""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    return f"{search_terms(genre)} after:{since}"


def run_search(query, runner=subprocess.run, timeout=DDGR_TIMEOUT_S):
    """Run ddgr and return the parsed results, or None with a reason printed.

    `runner` is injectable so every failure mode below can be tested without
    DuckDuckGo being up, online, or in the mood.
    """
    cmd = ["ddgr", "--num", str(DDGR_MAX_RESULTS), "--json", query]
    try:
        result = runner(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as ex:  # ddgr missing, or it hung past the timeout
        log(f"ddgr could not run: {type(ex).__name__}: {ex}")
        return None

    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        log(f"ddgr failed (exit {result.returncode}): {stderr[:200] or 'no stderr'}")
        return None
    if not (result.stdout or "").strip():
        log(f"ddgr wrote nothing to stdout{f': {stderr[:200]}' if stderr else ''}")
        return None
    try:
        results = json.loads(result.stdout)
    except json.JSONDecodeError as ex:
        log(f"ddgr output was not JSON ({ex}): {result.stdout[:120]!r}")
        return None
    if not isinstance(results, list):
        log(f"ddgr returned {type(results).__name__}, expected a list")
        return None
    if not results:
        # DuckDuckGo throttles with HTTP 202, which ddgr reports as an empty set
        # plus an [ERROR] line. Nothing to curate, and nothing broken.
        note = f" ({stderr[:120]})" if stderr else ""
        log(f"no search results this run{note}")
        return None
    return results


def search_searxng(query, base=SEARXNG_URL, time_range=None, limit=DDGR_MAX_RESULTS,
                   pages=1, opener=None, timeout=SEARXNG_TIMEOUT_S):
    """Ask the local SearXNG for JSON. Returns [{title,url,type,abstract}, ...] or None.

    The reason this exists: ddgr gets throttled (HTTP 202) and there is nothing to
    do about it from the client side. SearXNG aggregates several engines, so one
    of them sulking is survivable — measured on this box the job's query returns
    20 results from google cse while brave is "Suspended: too many requests" and
    duckduckgo answers with a CAPTCHA.

    Each result is tagged with `type` (track / profile / set), because that is the
    measured bottleneck: of those 20 results only 8 were individual tracks — 7 were
    artist pages and 5 were playlists, since a query that demands licence wording
    attracts exactly the pages that mention licence wording. The model gets the
    tag and is told not to pick a playlist.

    `time_range` is off by default, and that is a measurement rather than a
    preference: with site:soundcloud.com, week returns 0 results and month returns
    2, while year or no filter return the same 20 (all older than a month).

    `opener` is injectable so the parsing, the pagination and the failure modes are
    testable without a running service.
    """
    opener = opener or urllib.request.urlopen
    seen, out, page = set(), [], 1
    while page <= max(1, pages):
        params = {"q": query, "format": "json", "pageno": page}
        if time_range:
            params["time_range"] = time_range
        url = f"{base.rstrip('/')}/search?{urllib.parse.urlencode(params)}"
        try:
            with opener(url, timeout=timeout) as resp:
                payload = json.loads(resp.read())
        except Exception as ex:
            log(f"searxng page {page} failed: {type(ex).__name__}: {ex}")
            break
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            log(f"searxng page {page} answered without a results list")
            break
        fresh = 0
        for r in results:
            if not isinstance(r, dict) or not r.get("url") or r["url"] in seen:
                continue
            seen.add(r["url"])
            out.append({"title": r.get("title"), "url": r["url"],
                        "type": download_kind(r["url"]),
                        "abstract": (r.get("content") or "")[:400]})
            fresh += 1
        log(f"searxng page {page}: {len(results)} results, {fresh} new")
        if not fresh:  # engines have run out, or are repeating themselves
            break
        if limit and len(out) >= limit:
            break
        page += 1

    if not out:
        return None
    tracks = sum(1 for r in out if r["type"] == "track")
    log(f"searxng: {len(out)} unique result(s), {tracks} of them individual tracks")
    return out[:limit] if limit else out


def cc_usable(ccurl):
    """Can a monetised YouTube mix legally use a track under this CC licence?

    The rule, which is the whole reason Jamendo is worth the API key: a mix is a
    derivative work on a channel that carries ads, so
      * NonCommercial (-nc) is out — monetised is commercial, full stop;
      * NoDerivatives (-nd) is out — cutting a track into a mix is a derivative;
      * Attribution (by), ShareAlike (by-sa), CC0/Public Domain and the Public
        Domain Mark are fine, with the attribution the report already prints.

    Returns (ok, label) so the report can say which licence it relied on.
    """
    text = (ccurl or "").lower()
    if not text:
        return False, ""
    if "publicdomain" in text or "/zero/" in text or "/mark/" in text:
        return True, "CC0 / Public Domain"
    if "creativecommons.org" not in text:
        return False, ""            # "all rights reserved" and friends
    if "-nc" in text or "-nd" in text or "/nc" in text:
        return False, ""
    for slug, label in (("/by-sa/", "CC-BY-SA"), ("/by/", "CC-BY")):
        if slug in text:
            return True, label
    return False, ""


def search_jamendo(tags="house", client_id=None, limit=50, opener=None,
                   timeout=JAMENDO_TIMEOUT_S):
    """Search Jamendo's Creative Commons catalogue. Returns track dicts or None.

    Unlike the search-engine sources, every result here arrives with its licence
    attached, and the unusable ones are dropped before the model ever sees them:
    no amount of good taste compensates for a NonCommercial or NoDerivatives
    track in an ad-carrying mix.
    """
    client_id = client_id or JAMENDO_CLIENT_ID
    if not client_id:
        log("jamendo: no JAMENDO_CLIENT_ID set (free at devportal.jamendo.com)")
        return None
    params = {
        "client_id": client_id,
        "format": "json",
        "limit": str(max(1, min(limit, 200))),   # Jamendo caps limit at 200
        "fuzzytags": tags,                        # OR between tags; `tags` would AND
        "include": "licenses musicinfo",
        "audioformat": "mp32",
        "order": "releasedate_desc",
        # Ask the catalogue for usable licences before filtering locally: ccnc and
        # ccnd are the Creative Commons NonCommercial and NoDerivatives switches,
        # i.e. exactly the two a monetised, derivative mix cannot use. The
        # client-side cc_usable() check stays as the authority, because reading
        # those booleans backwards is an easy mistake to make and the cost of the
        # mistake is a copyright claim.
        #
        # Deliberately NOT set: prolicensing / probackground. Those return the paid
        # Jamendo *Licensing* catalogue, which is a different product with a
        # different licence model — the free Creative Commons catalogue is the
        # whole point here.
        "ccnc": "0",
        "ccnd": "0",
    }
    # The response carries content_id_free per track ("will not trigger YouTube's
    # Content ID"), which is reported rather than filtered on: filtering would drop
    # usable tracks, and the owner should see the flag before deciding.
    url = f"{JAMENDO_URL}?{urllib.parse.urlencode(params)}"
    opener = opener or urllib.request.urlopen
    try:
        with opener(url, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except Exception as ex:
        log(f"jamendo failed: {type(ex).__name__}: {ex}")
        return None

    results = payload.get("results") if isinstance(payload, dict) else None
    # Jamendo reports a bad key inside the payload (headers.status = failed,
    # error_message = "invalid client_id"). Without this the failure looks like
    # "0 useful results", which sends you looking in entirely the wrong place.
    headers = payload.get("headers") if isinstance(payload, dict) else None
    if isinstance(headers, dict) and headers.get("status") not in (None, "success"):
        log(f"jamendo rejected the request: "
            f"{headers.get('error_message') or headers.get('status')} "
            f"(check JAMENDO_CLIENT_ID in .env)")
        return None
    if not isinstance(results, list):
        log("jamendo answered without a results list")
        return None
    if not results:
        # Seen live: the identical query returned 40 tracks and, a minute later, an
        # empty set — a hiccup on their side. Say so, rather than "0 usable
        # results", which reads like the catalogue had nothing to offer.
        log("jamendo returned an empty result set (transient hiccup, or the tags are "
            "too narrow) — retry, or try other --jamendo-tags")
        return None

    out, rejected, no_download = [], 0, 0
    for r in results:
        if not isinstance(r, dict) or not r.get("shareurl"):
            continue
        ok, label = cc_usable(r.get("license_ccurl"))
        if not ok:
            rejected += 1
            continue
        # A Creative Commons licence does not imply a downloadable file: the artist
        # chooses, and Jamendo blanks `audiodownload` when they say no.
        audio = r.get("audiodownload") or ""
        if not audio or r.get("audiodownload_allowed") is False:
            no_download += 1
        out.append({
            "artist": r.get("artist_name") or "?",
            "title": r.get("name") or "?",
            "url": r["shareurl"],
            "audio_url": audio,
            "type": "track",
            "license": label,
            "content_id_free": r.get("content_id_free"),
            "download_allowed": bool(audio) and r.get("audiodownload_allowed") is not False,
            "abstract": f"{label} — verified in Jamendo's CC catalogue"
                        + (" — no direct download" if not audio else "")
                        + (f" (tags: {r.get('musicinfo', {}).get('tags', {})})"
                           if isinstance(r.get("musicinfo"), dict) else ""),
        })
    log(f"jamendo: {len(out)} usable result(s), {rejected} dropped for NC/ND/other, "
        f"{no_download} without a direct download")
    return out or None


SUBGENRE_RULE = """SUBGENRE COHERENCE (IMPORTANT):
- Pick ONE subgenre of {genre} for this week's report (e.g. Deep House,
  Tech House, Melodic House, Afro House, Progressive House).
- EVERY track in the output array must belong to that SAME subgenre.
- Set the 'genre' field to that exact subgenre on every track.
- If no single subgenre has at least ~4 tracks, pick the one with the most
  tracks and only include those."""

WIDE_RULE = """SUBGENRES:
- This report is a varied mix on purpose: include tracks from any house
  subgenre (Deep House, Tech House, Melodic House, Afro House, Progressive
  House, Tropical House...).
- Set the 'genre' field on each track to that track's own subgenre."""


def build_prompt(genre=GENRE, mix="coherent", has_verified=False, has_inferred=True):
    """The curation prompt, in one of two shapes.

    `coherent` (the default, and the behaviour this job shipped with) asks for a
    single subgenre and keeps only it, which caps the report at whatever the
    biggest subgenre can muster. Measured on a pool of 26 individual tracks with
    22 mentioning licence wording, that cap is what turns ~15 usable tracks into
    3: the model picks the largest subgenre and drops the rest by instruction, not
    by judgement. `wide` drops the cap and keeps the same strict licence filter.

    `verified=True` is for sources that ship the licence as data (Jamendo). Then
    the licence question is already settled and the model's job is taste, not
    copyright: asking it to re-derive a licence it has been handed only invites it
    to second-guess good data.
    """
    rule = WIDE_RULE if mix == "wide" else SUBGENRE_RULE.format(genre=genre)
    blocks = []
    if has_inferred:
        blocks.append("""LICENCES YOU MUST JUDGE (results with no "license" field):
These come from a web search, so the licence is only what the page claims. Keep a
track only if it is 'Royalty Free', 'No Copyright' or 'CC-BY' AND allows commercial
use. Discard anything 'personal use only', 'No Derivatives', or whose monetization
safety is not clearly High. An empty licence field is not a licence.""")
    if has_verified:
        blocks.append("""LICENCES ALREADY VERIFIED (results WITH a "license" field):
Those come from a Creative Commons catalogue and were checked there; NonCommercial
and NoDerivatives are already filtered out. Do not second-guess that field, and do
not require the title or description to mention a licence: most Creative Commons
tracks have no such wording, and the field is the field. For these, judge only
whether the track will work musically in the mix.""")
    criteria = "\n\n".join(blocks) or """Keep only tracks that are unambiguously safe
to monetise, and discard anything you cannot vouch for."""
    return f"""You are a professional music curator for a YouTube house-mix channel.
GOAL: choose tracks that are safe to monetise and that will sound good together.

{criteria}

{rule}

Output ONLY a valid JSON array of objects. ONLY include tracks with HIGH monetization safety.

RESULT TYPES: every search result carries a "type" field.
  track   = an individual track page — the only thing you may pick
  profile = an artist or label page, not a track
  set     = a playlist, not a track
A playlist whose title says "no copyright" is not a licence for the tracks inside
it, and a profile page has no licence at all. Skip both.

Each object must be:
{{
  "artist": "...",
  "title": "...",
  "url": "...",
  "license_type": "CC-BY / Royalty Free",
  "monetization_safety": "High",
  "reason": "Explain why this is safe for ads.",
  "genre": "This track's subgenre"
}}
If no 100% safe tracks are found, return []."""


SYSTEM_PROMPT = build_prompt()


def track_list(value):
    """Accept a bare array, or the same array wrapped in an object.

    Models like to help by wrapping: {"tracks": [...]}. Accepting that costs
    three lines and saves a rejected run.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("tracks", "results", "tracks_safe", "data", "output"):
            if isinstance(value.get(key), list):
                return value[key]
        for item in value.values():
            if isinstance(item, list):
                return item
    return None


def _enrich(tracks, candidates):
    """Reattach the metadata the model cannot be trusted to echo.

    The model answers with its own JSON: artist, title, url, and its own paraphrase
    of the licence. Everything we actually *verified* — that the licence is CC-BY
    and not NC/ND, whether a direct download exists, whether it can trigger Content
    ID, and which catalogue it came from — lives in the candidate list we sent it.
    So it is joined back here by URL instead of being expected in the reply.

    Measured, not theorised: the first real Jamendo run produced 23 tracks with no
    provenance label at all, because the `source` field never survived the round
    trip through the model.
    """
    by_url = {}
    for c in candidates or []:
        url = (c.get("url") or "").strip()
        if url:
            by_url[url] = c
            by_url[url.rstrip("/")] = c
    for t in tracks:
        url = (t.get("url") or "").strip()
        c = by_url.get(url) or by_url.get(url.rstrip("/"))
        if not c:
            continue
        for key in ("source", "license", "audio_url", "content_id_free",
                    "download_allowed", "type"):
            if c.get(key) is not None:
                t[key] = c[key]
        if c.get("license"):            # the verified label beats a paraphrase
            t["license_type"] = c["license"]
    return tracks


TASTE_PROMPT = """You are curating one continuous DJ mix from a pool of house tracks.

Every track in the pool is already cleared for monetisation: the licences are
verified Creative Commons (CC-BY, CC-BY-SA or CC0), there are no NonCommercial or
NoDerivatives tracks, and every entry carries its verified licence in a "license"
field. Do not comment on licences and do not second-guess that field — that question
is settled, which is precisely why this task exists: the only thing left to judge is
the music.

Your job:
- Pick 10-15 tracks that could sit together in one continuous mix: consistent tempo
  and energy, no jarring mood changes, one coherent subgenre.
- Order them the way you would play them.
- Give each a one-line note about its vibe, enough for someone to decide.

Answer with ONE JSON array and nothing else:
[{"artist": "...", "title": "...", "url": "...",
   "license_type": "<copy the license field>", "monetization_safety": "verified",
   "reason": "<one line about the vibe>", "genre": "<the subgenre>"}]"""


def curate(results, llm=call_llm, data_dir=DATA_DIR, mix="coherent",
           has_verified=False, has_inferred=True, taste=False):
    """Ask the model to filter the candidates. Returns (tracks, genre) or (None, None).

    An empty list means "nothing safe this week", which is a valid answer; None
    means the model's reply could not be used at all. `mix` picks the prompt: the
    coherence filter is only applied for "coherent", because in "wide" mode every
    subgenre is wanted.
    """
    if has_verified and not has_inferred:
        # Nothing to judge about licences here: every candidate was checked against
        # the catalogue, so asking the model about them only adds variance. Measured
        # reason: given 40 verified and 40 SoundCloud candidates in one list, the
        # model returned eight SoundCloud ones and skipped every verified track — it
        # was reading the titles that literally say "no copyright".
        #
        # --curate asks it about the music instead, which is the part it is good at.
        tracks = None
        if taste:
            log(f"{len(results)} candidates are already verified — asking the model "
                f"about the music only, not the licences")
            raw_reply = llm(TASTE_PROMPT,
                            f"Pool of {len(results)} tracks:\n{json.dumps(results, indent=2)}",
                            temperature=0.4, max_tokens=3000, think=False)
            try:
                tracks = track_list(extract_json(raw_reply))
            except json.JSONDecodeError as ex:
                log(f"the model's curation had no usable JSON ({ex})")
                tracks = None
            if tracks:
                tracks = _enrich([t for t in tracks if isinstance(t, dict)], results)
                log(f"curated to {len(tracks)} track(s); verified metadata reattached")
            else:
                log("curation unusable — falling back to the whole verified pool")
        if tracks is None:
            log(f"using all {len(results)} verified candidates as they came")
            tracks = []
            for r in results:
                t = dict(r)
                t["title"] = t.get("title") or "?"
                t["artist"] = t.get("artist") or "?"
                t["license_type"] = t.get("license") or ""
                tracks.append(t)
        if mix == "wide":
            return tracks, f"{GENRE} (CC verificado)"
        genre_label, tracks = pick_dominant_genre(tracks)
        return tracks, genre_label

    user_prompt = f"Search Results:\n{json.dumps(results, indent=2)}"
    raw_reply = llm(build_prompt(GENRE, mix, has_verified, has_inferred), user_prompt,
                    temperature=0.2, max_tokens=4000, think=False)
    if not (raw_reply or "").strip():
        log("the model returned nothing (empty content)")
        return None, None
    try:
        parsed = extract_json(raw_reply)
    except json.JSONDecodeError as ex:
        path = dump_raw(raw_reply, data_dir)
        log(f"the model's reply had no usable JSON ({ex}) — saved to {path}")
        return None, None

    tracks = track_list(parsed)
    if tracks is None:
        path = dump_raw(raw_reply, data_dir)
        log(f"the model returned JSON that is not a track list — saved to {path}")
        return None, None
    tracks = [t for t in tracks if isinstance(t, dict)]
    tracks = _enrich(tracks, results)
    if mix == "wide":
        return tracks, f"{GENRE} (mix)"
    genre_label, tracks = pick_dominant_genre(tracks)
    return tracks, genre_label


def dump_raw(raw_reply, data_dir=DATA_DIR):
    """Keep what the model actually said, for the same reason the market jobs do."""
    out_dir = Path(data_dir) / "rejected"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    path.write_text(raw_reply, encoding="utf-8")
    return path


def pick_dominant_genre(tracks):
    """Return (genre_label, filtered_tracks).

    Keeps only the most common subgenre so a single report is coherent
    ("más o menos" the same genre). Filters only when a clear majority
    exists (>=50% of tracks, and at least 2 tracks); otherwise returns
    everything unchanged so a thin mix never becomes an empty report.
    Missing 'genre' fields fall back to the base GENRE.
    """
    if not tracks:
        return GENRE, tracks

    def norm(t):
        return ((t.get("genre") or "").strip() or GENRE)

    counts = {}
    for t in tracks:
        g = norm(t)
        counts[g] = counts.get(g, 0) + 1
    # Deterministic tie-break: most common, then lexicographically first
    top_genre = max(counts, key=lambda g: (counts[g], g))
    top_n = counts[top_genre]
    if top_n >= 2 and top_n * 2 >= len(tracks):
        kept = [t for t in tracks if norm(t) == top_genre]
        if len(kept) != len(tracks):
            print(f"  🎧 Coherente: {len(tracks)} → {len(kept)} pistas de "
                  f"«{top_genre}» (se descartaron los otros subgéneros)")
        return top_genre, kept
    return top_genre, tracks


def download_kind(url):
    """Is this a single track, or a set/profile/album yt-dlp would swallow whole?

    Measured, and it matters for a monetised channel: of the first four tracks the
    model picked, one was an artist profile and two were /sets/ playlists. The
    download command would have pulled entire playlists, and "no copyright" in a
    playlist title is not a per-track licence. Leads are still useful; they just
    do not belong in an automatic download.
    """
    try:
        parts = urllib.parse.urlparse(url)
    except ValueError:
        return "unknown"
    host = (parts.hostname or "").lower()
    seg = [p for p in parts.path.split("/") if p]
    if not seg:
        return "unknown"
    if "soundcloud.com" in host:
        if len(seg) == 1:
            return "profile"
        if seg[1] in ("sets", "playlists", "albums"):
            return "set"
        return "track"
    if "bandcamp.com" in host:
        if seg[0] == "track":
            return "track"
        if seg[0] in ("album", "sets"):
            return "set"
        return "unknown"
    return "track"  # an unknown host is presumably a single page


def write_report(tracks, genre_label=None, data_dir=DATA_DIR):
    """Write a Discord-friendly report: name + link per track, plus a ready
    yt-dlp command to download them ALL at once (owner checks before running).
    queue-runner DMs this file to the owner after a successful job run."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    report_path = data_dir / "latest-report.md"
    week = datetime.now().strftime("%Y-W%V")
    lines = [
        f"🎵 **Música semanal — {genre_label or GENRE} (CC / royalty-free)**",
        f"Semana {week} · {len(tracks)} pista(s) encontrada(s)",
        "",
    ]
    if not tracks:
        lines.append("No se encontraron pistas 100% seguras para monetización "
                     "esta semana.")
    else:
        track_urls, other_urls = [], []
        for i, t in enumerate(tracks, 1):
            artist = (t.get("artist") or "?").strip()
            title = (t.get("title") or "?").strip()
            url = (t.get("url") or "").strip().replace('"', "")
            lic = (t.get("license_type") or t.get("license") or "").strip()
            kind = download_kind(url) if url else "unknown"
            warning = {"set": "  ⚠️ playlist — no entra en la descarga automática",
                       "profile": "  ⚠️ perfil, no es una pista",
                       "unknown": "  ⚠️ enlace no reconocido"}.get(kind, "")
            # Jamendo flags worth seeing before downloading or uploading anything.
            if t.get("download_allowed") is False:
                warning += "  ⚠️ sin descarga directa (el artista no la permitió)"
            if t.get("content_id_free") is False:
                warning += "  ⚠️ puede activar Content ID en YouTube"
            origin = {"jamendo": "· licencia verificada",
                      "searxng": "· licencia inferida",
                      "ddgr": "· licencia inferida"}.get(t.get("source") or "", "")
            lines.append(f"{i}. **{artist}** — *{title}*"
                         + (f"  ({lic})" if lic else "")
                         + (f"  {origin}" if origin else "") + warning)
            if url:
                # Jamendo hands over the audio file directly; prefer it for the
                # download command and keep the page for attribution.
                audio = (t.get("audio_url") or "").strip()
                lines.append(f"   🔗 {url}" + (f"   ⬇️ {audio}" if audio else ""))
                link = audio or url
                (track_urls if download_kind(link) == "track" else other_urls).append(link)
        lines += ["", "▶️ **Descarga todas de golpe** (revísalas antes de descargar):"]
        # Keep the command under ~1800 chars so the DM code block is never split
        # mid-line by the 2000-char Discord limit (many tracks would break it).
        prefix = "yt-dlp --extract-audio --audio-format mp3 --audio-quality 0"
        cmd_parts, used = [], 0
        for u in track_urls:
            piece = f'"{u}"'
            if used + len(piece) + 1 > 1800:
                break
            cmd_parts.append(piece)
            used += len(piece) + 1
        cmd = prefix + " " + " ".join(cmd_parts)
        if len(cmd_parts) < len(track_urls):
            cmd += (f"\n# …y {len(track_urls) - len(cmd_parts)} pista(s) más — ver "
                    f"informe completo en el servidor")
        lines += ["```bash", cmd, "```"]
        if other_urls:
            lines.append(f"_{len(other_urls)} enlace(s) son playlists o perfiles: "
                         f"revísalos a mano antes de descargar nada._")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def search_tracks(data_dir=DATA_DIR, runner=subprocess.run, llm=call_llm,
                  source="auto", searxng_url=SEARXNG_URL, opener=None,
                  time_range="", pages=2, max_candidates=40, mix="coherent",
                  jamendo_tags=None, jamendo_client_id=None, curate_taste=False):
    """One weekly run. Returns the list of tracks (possibly empty) or None.

    `source`: auto goes Jamendo (verified CC licences) -> local SearXNG -> ddgr,
    each one skipped if it is unavailable, which is also how you find out which
    half is unhappy.
    """
    genre = GENRE
    print(f"🔎 Starting STRICT discovery for monetizable {genre}...")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    results = []
    jam_tags = jamendo_tags or "house"
    cid = jamendo_client_id if jamendo_client_id is not None else JAMENDO_CLIENT_ID

    if source in ("auto", "both", "jamendo"):
        print(f"  Searching Jamendo (CC catalogue, tags={jam_tags})")
        jam = search_jamendo(tags=jam_tags, client_id=cid,
                             limit=max_candidates, opener=opener)
        for r in jam or []:
            r["source"] = "jamendo"
        results += jam or []

    # 'both' merges instead of choosing, because Jamendo is a different catalogue,
    # not the same songs with better paperwork. SoundCloud's CC community uploads
    # under "no copyright music" and often clears Content ID itself, which makes it
    # a genuinely useful pool — its licences are just inferred from a snippet
    # rather than verified. Merging keeps both and labels which is which.
    if source == "both" or (not results and source in ("auto", "searxng")):
        print(f"  Searching SearXNG ({searxng_url}) for: {search_terms(genre)}"
              f"{f' [time_range={time_range}]' if time_range else ''}")
        sx = search_searxng(search_terms(genre), base=searxng_url,
                            opener=opener, time_range=time_range,
                            pages=pages, limit=max_candidates)
        for r in sx or []:
            r["source"] = "searxng"
        results += sx or []

    if not results and source in ("auto", "ddgr"):
        query = build_query(genre)
        print(f"  Searching ddgr for: {query}")
        dd = run_search(query, runner=runner)
        for r in dd or []:
            r["source"] = "ddgr"
        results += dd or []

    if not results:
        return None
    has_verified = any(r.get("license") for r in results)
    has_inferred = any(not r.get("license") for r in results)
    print(f"  candidates: {len(results)} "
          f"({sum(1 for r in results if r.get('license'))} with a verified licence)")

    print(f"  Analyzing {len(results)} candidates for MONETIZATION...")
    tracks, genre_label = curate(results, llm=llm, data_dir=data_dir, mix=mix,
                                 has_verified=has_verified,
                                 has_inferred=has_inferred, taste=curate_taste)
    if tracks is None:
        print("❌ The model's curation could not be used (raw reply kept).")
        return None

    out_file = data_dir / f"house-mix-{datetime.now().strftime('%Y-W%V')}.json"
    out_file.write_text(json.dumps(tracks, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"✅ Discovery complete. {len(tracks)} tracks found.")
    print(f"  Report saved to: {out_file}")
    print(f"  Discord report saved to: {write_report(tracks, genre_label, data_dir)}")
    return tracks


# ─── Selftest (offline: fixtures, no network, no model) ────


class _FakeResp:
    """Minimal stand-in for an urlopen() response."""

    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body.encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def selftest():
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="music-discover-selftest-"))
    rows = [{"title": "Deep One", "url": "https://soundcloud.com/a/1"},
            {"title": "Deep Two", "url": "https://soundcloud.com/a/2"}]
    good_reply = json.dumps([
        {"artist": "A", "title": "Deep One", "url": "https://soundcloud.com/a/1",
         "license_type": "CC-BY", "monetization_safety": "High", "genre": "Deep House"},
        {"artist": "A", "title": "Deep Two", "url": "https://soundcloud.com/a/2",
         "license_type": "CC-BY", "monetization_safety": "High", "genre": "Deep House"},
        {"artist": "B", "title": "Afro One", "url": "https://soundcloud.com/b/1",
         "license_type": "CC-BY", "monetization_safety": "High", "genre": "Afro House"},
    ])

    # 1. ddgr's failure modes, all of which happened in production.
    cases = [
        ("--num too large (ddgr 2.2 caps at 25)",
         _FakeProc("", "ddgr: error: argument -n/--num: 50 is not a non-negative "
                       "integer <= 25", 2), None),
        ("stdout empty", _FakeProc("", "", 0), None),
        ("not JSON", _FakeProc("<html>captcha</html>", "", 0), None),
        ("throttled: [] with a 202 on stderr",
         _FakeProc("[]\n", "[ERROR] HTTP Error 202: Accepted", 0), None),
        ("json object instead of a list", _FakeProc('{"a": 1}', "", 0), None),
        ("normal results", _FakeProc(json.dumps(rows), "", 0), rows),
    ]
    print("  ddgr failure modes:")
    for label, proc, want in cases:
        got = run_search("q", runner=lambda *a, **k: proc)
        ok &= got == want
        print(f"    {label:<38} -> {'parsed' if got else 'None'} "
              f"{'OK' if got == want else 'FAIL'}")

    # 1b. SearXNG: the JSON the local instance returns, and its failure modes.
    sx = json.dumps({"results": [
        {"title": "Deep One", "url": "https://soundcloud.com/a/1",
         "content": "royalty free deep house"},
        {"title": "no url here", "content": "dropped"},
        {"title": "Deep Two", "url": "https://soundcloud.com/a/2", "content": ""},
    ]})
    got = search_searxng("q", opener=lambda url, timeout=0: _FakeResp(sx))
    ok &= got is not None and len(got) == 2 and got[0]["url"].endswith("/1")
    no_results = search_searxng("q", opener=lambda url, timeout=0: _FakeResp("{}"))
    boom = search_searxng("q", opener=lambda url, timeout=0: (_ for _ in ()).throw(
        OSError("connection refused")))
    ok &= no_results is None and boom is None
    print(f"  searxng         -> {len(got) if got else 0} of 3 kept (url-less dropped); "
          f"no 'results' key -> {no_results}; down -> {boom}")

    # 2. The model reply parsing: bare array, array then chatter, wrapped, broken.
    # Two tracks come out, not three: the coherence filter keeps the Deep House
    # majority and drops the single Afro House one, which is its job.
    wrapped = '{"tracks": ' + good_reply + '}'
    cases2 = [("bare array", good_reply),
              ("array + trailing prose",
               good_reply + "\nLet me know if you need more!"),
              ("wrapped in an object", wrapped),
              ("bare array with marks", "Sure:\n```json\n" + good_reply + "\n```\n")]
    print("  model reply parsing (3 tracks in, coherence filter -> 2 Deep House):")
    for label, reply in cases2:
        tracks, genre = curate(rows, llm=lambda *a, **k: reply, data_dir=tmp)
        n = len(tracks) if tracks is not None else 0
        good = n == 2 and genre == "Deep House"
        ok &= good
        print(f"    {label:<26} -> {n} track(s), genre '{genre}' "
              f"{'OK' if good else 'FAIL'}")

    broken = curate(rows, llm=lambda *a, **k: "I could not find any safe tracks.",
                    data_dir=tmp)
    ok &= broken == (None, None)
    kept = list((tmp / "rejected").glob("*.txt"))
    ok &= len(kept) == 1
    print(f"    {'no JSON at all':<26} -> rejected, raw kept: "
          f"{kept[0].name if kept else 'MISSING'}")

    # 2b. Playlists and profiles must never reach the download command.
    kinds = {"https://soundcloud.com/artist/track-name": "track",
             "https://soundcloud.com/artist/sets/a-set": "set",
             "https://soundcloud.com/artistname": "profile",
             "https://artist.bandcamp.com/track/x": "track",
             "https://artist.bandcamp.com/album/y": "set"}
    bad = {u: (download_kind(u), want) for u, want in kinds.items()
           if download_kind(u) != want}
    ok &= not bad
    mixed = [{"artist": "A", "title": "T", "url": "https://soundcloud.com/a/t"},
             {"artist": "B", "title": "S", "url": "https://soundcloud.com/b/sets/s"}]
    text_mixed = write_report(mixed, "Deep House", tmp).read_text(encoding="utf-8")
    ok &= '"https://soundcloud.com/a/t"' in text_mixed
    ok &= '"https://soundcloud.com/b/sets/s"' not in text_mixed
    ok &= "revísalos a mano" in text_mixed
    print(f"  download safety -> {len(kinds)} url shapes classified"
          f"{'' if not bad else ' FAIL ' + str(bad)}; playlist kept out of the "
          f"yt-dlp command, still listed as a lead")

    # 2c. The mix knob: coherent caps the report to one subgenre, wide does not.
    # This is the measured reason a pool of 26 tracks became a report of 3.
    ok &= "SAME subgenre" in build_prompt(GENRE, "coherent")
    ok &= "SAME subgenre" not in build_prompt(GENRE, "wide")
    wide_tracks, wide_genre = curate(rows, llm=lambda *a, **k: good_reply,
                                     data_dir=tmp, mix="wide")
    ok &= len(wide_tracks) == 3 and "mix" in wide_genre
    print(f"  mix knob        -> coherent keeps the majority of 3 (2), wide keeps "
          f"all {len(wide_tracks)} as '{wide_genre}'")

    # 2d. Jamendo: the licence filter is the point, not the search. NonCommercial
    # and NoDerivatives must never reach the report, whatever the model thinks.
    ok &= cc_usable("http://creativecommons.org/licenses/by/3.0/")[1] == "CC-BY"
    ok &= cc_usable("http://creativecommons.org/licenses/by-sa/4.0/")[1] == "CC-BY-SA"
    ok &= cc_usable("http://creativecommons.org/publicdomain/zero/1.0/")[0]
    ok &= not cc_usable("http://creativecommons.org/licenses/by-nc/3.0/")[0]
    ok &= not cc_usable("http://creativecommons.org/licenses/by-nd/3.0/")[0]
    ok &= not cc_usable("http://creativecommons.org/licenses/by-nc-nd/2.5/")[0]
    ok &= not cc_usable("")[0] and not cc_usable("https://x.example/all-rights")[0]
    jam = json.dumps({"results": [
        {"name": "Deep One", "artist_name": "A",
         "shareurl": "https://www.jamendo.com/track/1",
         "audiodownload": "https://mp3l.jamendo.com/?trackid=1",
         "audiodownload_allowed": True, "content_id_free": True,
         "license_ccurl": "http://creativecommons.org/licenses/by/3.0/"},
        {"name": "NC one", "artist_name": "B",
         "shareurl": "https://www.jamendo.com/track/2",
         "license_ccurl": "http://creativecommons.org/licenses/by-nc/3.0/"},
        {"name": "No download", "artist_name": "C",
         "shareurl": "https://www.jamendo.com/track/3", "audiodownload": "",
         "audiodownload_allowed": False, "content_id_free": False,
         "license_ccurl": "http://creativecommons.org/licenses/by-sa/4.0/"},
    ]})
    seen_urls = []

    def _capture(url, timeout=0):
        seen_urls.append(url)
        return _FakeResp(jam)

    jtracks = search_jamendo("house", client_id="test", opener=_capture)
    # 2 usable of 3 (the NC one is dropped): the CC-BY and the no-download CC-BY-SA.
    ok &= jtracks is not None and len(jtracks) == 2
    ok &= jtracks[0]["license"] == "CC-BY" and jtracks[0]["audio_url"].startswith("https://mp3l")
    ok &= jtracks[1]["download_allowed"] is False and jtracks[1]["content_id_free"] is False
    # The request itself must ask for usable licences and not for the paid catalogue.
    ok &= "ccnc=0" in seen_urls[0] and "ccnd=0" in seen_urls[0]
    ok &= "prolicensing" not in seen_urls[0] and "probackground" not in seen_urls[0]
    # We read content_id_free from the response instead of filtering on it.
    ok &= "content_id_free=" not in seen_urls[0]
    ok &= search_jamendo("house", client_id="", opener=lambda *a, **k: None) is None
    # A bad key must say so, not look like an empty catalogue.
    jam_fail = json.dumps({"headers": {"status": "failed",
                                       "error_message": "invalid client_id"},
                           "results": []})
    jfail = search_jamendo("house", client_id="bad",
                           opener=lambda url, timeout=0: _FakeResp(jam_fail))
    ok &= jfail is None
    print(f"  jamendo         -> by/by-sa/zero accepted, nc/nd rejected; "
          f"{len(jtracks) if jtracks else 0} of 3 kept (NC dropped), direct audio "
          f"link present, no-download Content-ID flag surfaced; bad key reported "
          f"({jfail is None})")

    # 2e. Merge mode ('both'): two catalogues in one candidate list, each labelled,
    # because Jamendo is not the same songs with better paperwork — it is other
    # songs. The prompt must then carry both licence rules.
    def _by_url(url, timeout=0):
        if "jamendo.com" in url:
            return _FakeResp(json.dumps({"results": [{
                "name": "J track", "artist_name": "J",
                "shareurl": "https://www.jamendo.com/track/9",
                "audiodownload": "https://mp3l.jamendo.com/?trackid=9",
                "audiodownload_allowed": True, "content_id_free": True,
                "license_ccurl": "http://creativecommons.org/licenses/by/4.0/"}]}))
        return _FakeResp(json.dumps({"results": [
            {"title": "S track", "url": "https://soundcloud.com/x/y",
             "content": "royalty free house"}]}))

    prompts = []

    def _capture_llm(system, user, **kw):
        prompts.append((system, user))
        return good_reply

    merged = search_tracks(data_dir=tmp, source="both", opener=_by_url,
                           llm=_capture_llm, jamendo_client_id="test",
                           max_candidates=5)
    both_rules = (prompts and 'no "license" field' in prompts[0][0]
                  and 'WITH a "license" field' in prompts[0][0])
    both_catalogues = prompts and "jamendo.com" in prompts[0][1] and "soundcloud.com" in prompts[0][1]
    ok &= merged is not None and both_rules and both_catalogues
    # And the enrichment: what the model returns is only artist/title/url, so the
    # verified metadata has to be joined back or the report loses its labels.
    enriched = _enrich([{"artist": "A", "title": "T",
                         "url": "https://www.jamendo.com/track/9",
                         "license_type": "the model's guess"}],
                       [{"url": "https://www.jamendo.com/track/9", "source": "jamendo",
                         "license": "CC-BY", "audio_url": "https://mp3l/x.mp3",
                         "content_id_free": False, "download_allowed": True}])
    ok &= enriched[0]["source"] == "jamendo" and enriched[0]["license_type"] == "CC-BY"
    ok &= enriched[0]["content_id_free"] is False and enriched[0]["audio_url"].endswith(".mp3")
    print(f"  source=both     -> {len(merged) if merged else 0} curated from 2 "
          f"catalogues; prompt carried both licence rules ({both_rules}) and both "
          f"results ({both_catalogues})")
    print(f"  enrichment      -> source={enriched[0]['source']}, "
          f"license_type={enriched[0]['license_type']!r} (verified beats the model's "
          f"guess), Content-ID flag and audio link recovered")

    # 2f. Verified-only candidates: the model is not called at all, because there
    # is nothing for it to judge and its answer only adds variance.
    called = []
    verified_cands = [{"url": "https://www.jamendo.com/track/1", "artist": "A",
                       "title": "T1", "license": "CC-BY", "source": "jamendo",
                       "audio_url": "https://mp3l/1.mp3", "type": "track"}]
    vtracks, vgenre = curate(verified_cands, llm=lambda *a, **k: called.append(1),
                             data_dir=tmp, mix="wide",
                             has_verified=True, has_inferred=False)
    ok &= vtracks is not None and len(vtracks) == 1 and not called
    ok &= "verificado" in (vgenre or "") and vtracks[0]["license_type"] == "CC-BY"
    print(f"  verified only   -> {len(vtracks) if vtracks else 0} track(s) with "
          f"{len(called)} model call(s) (want 0), genre '{vgenre}'")

    # 2g. --curate on a verified pool: the model is asked about the music only, and
    # if its answer is unusable the whole pool is used instead of nothing.
    taste_prompts = []

    def _taste_llm(system, user, **kw):
        taste_prompts.append(system)
        return good_reply

    ttracks, _tgenre = curate(verified_cands, llm=_taste_llm, data_dir=tmp, mix="wide",
                              has_verified=True, has_inferred=False, taste=True)
    asked_about_music = bool(taste_prompts) and "Do not comment on licences" in taste_prompts[0]
    ok &= ttracks is not None and asked_about_music
    fb, _fgenre = curate(verified_cands, llm=lambda *a, **k: "no json at all",
                         data_dir=tmp, mix="wide", has_verified=True,
                         has_inferred=False, taste=True)
    ok &= fb is not None and len(fb) == len(verified_cands)
    print(f"  --curate        -> asked about the music only ({asked_about_music}); an "
          f"unusable reply falls back to the whole verified pool ({len(fb)} track(s))")

    # 3. Coherence: a clear majority filters, a thin field does not.
    many = ([{"genre": "Deep House"}] * 3) + [{"genre": "Tech House"}]
    label, kept = pick_dominant_genre(many)
    ok &= label == "Deep House" and len(kept) == 3
    thin = [{"genre": "Deep House"}, {"genre": "Tech House"}, {"genre": ""}]
    label2, kept2 = pick_dominant_genre(thin)
    ok &= len(kept2) == 3
    ok &= pick_dominant_genre([]) == (GENRE, [])
    print(f"  coherence       -> 3+1 -> {len(kept)} '{label}'; "
          f"thin field -> {len(kept2)}; empty -> {GENRE}")

    # 4. The report file, including the yt-dlp block and the empty case.
    path = write_report([{"artist": "A", "title": "T", "url": "https://x/1",
                          "license_type": "CC-BY"}], "Deep House", tmp)
    text = path.read_text(encoding="utf-8")
    ok &= "yt-dlp" in text and "https://x/1" in text and "Deep House" in text
    empty = write_report([], GENRE, tmp).read_text(encoding="utf-8")
    ok &= "No se encontraron" in empty
    print(f"  report          -> {path.name} ({len(text)} chars, yt-dlp block); "
          f"empty case says so")

    # 5. Too many urls: the command is truncated, not split.
    long_tracks = [{"artist": "x", "title": "y", "url": f"https://soundcloud.com/{'z' * 120}/{i}"}
                   for i in range(30)]
    text2 = write_report(long_tracks, "Deep House", tmp).read_text(encoding="utf-8")
    longest = max(len(line) for line in text2.splitlines())
    ok &= "pista(s) más" in text2 and longest < 2000
    print(f"  long report     -> truncated with a pointer home, longest line "
          f"{longest} chars")

    print(f"  temp dir: {tmp}")
    print(f"\n  selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Weekly CC/royalty-free house music discovery")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--source", choices=("auto", "both", "jamendo", "searxng", "ddgr"),
                    default="auto",
                    help="auto = Jamendo if JAMENDO_CLIENT_ID is set (verified CC "
                         "licences), else the local SearXNG, else ddgr. 'both' merges "
                         "Jamendo with SearXNG: they are different catalogues, not "
                         "the same tracks with different paperwork")
    ap.add_argument("--jamendo-tags", default="house",
                    help="Jamendo fuzzytags to search (house, deephouse, techhouse...)")
    ap.add_argument("--searxng-url", default=SEARXNG_URL)
    ap.add_argument("--time-range", default="",
                    choices=("", "day", "week", "month", "year"),
                    help="recency filter. Empty by default because with "
                         "site:soundcloud.com a week returns 0 results and a "
                         "month returns 2, while the full set (20) is all older "
                         "than a month: a web index does not have this week's "
                         "uploads for a query this narrow")
    ap.add_argument("--pages", type=int, default=2,
                    help="search result pages to pull and merge")
    ap.add_argument("--max-candidates", type=int, default=40)
    ap.add_argument("--mix", choices=("coherent", "wide"), default="coherent",
                    help="coherent = one subgenre, as this job has always done "
                         "(and the reason a pool of 26 tracks yields 3); wide = "
                         "keep any house subgenre, same strict licence filter")
    ap.add_argument("--curate", dest="curate_taste", action="store_true",
                    help="when the licences are already verified, ask the model to "
                         "pick a musically coherent subset (it judges the music, "
                         "not the licensing). Without it, verified candidates are "
                         "reported as they come, which is instant and deterministic")
    ap.add_argument("--selftest", action="store_true",
                    help="offline checks: ddgr and SearXNG fixtures, stubbed model")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    tracks = search_tracks(data_dir=args.data_dir, source=args.source,
                           searxng_url=args.searxng_url,
                           time_range=args.time_range, pages=args.pages,
                           max_candidates=args.max_candidates, mix=args.mix,
                           jamendo_tags=args.jamendo_tags,
                           curate_taste=args.curate_taste)
    # 0 on success (including "nothing safe this week"), 1 when the run could not
    # be completed at all — the scheduler and the DM step care about the difference.
    return 0 if tracks is not None else 1


if __name__ == "__main__":
    sys.exit(main())
