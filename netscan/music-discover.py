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
                   opener=None, timeout=SEARXNG_TIMEOUT_S):
    """Ask the local SearXNG for JSON. Returns [{title,url,abstract}, ...] or None.

    The reason this exists: ddgr gets throttled (HTTP 202) and there is nothing to
    do about it from the client side. SearXNG aggregates several engines, so one
    of them sulking is survivable — measured on this box the job's query returns
    20 results from google cse while brave is "Suspended: too many requests" and
    duckduckgo answers with a CAPTCHA.

    `time_range` is off by default, and that is a measurement rather than a
    preference: with site:soundcloud.com, week returns 0 results and month
    returns 2, while year or no filter return the same 20 (all older than a
    month). A web index simply does not have this week's uploads for a query that
    narrow, so claiming "this week" would be a lie the report tells.

    `opener` is injectable so the parsing and the failure modes are testable
    without a running service.
    """
    params = {"q": query, "format": "json"}
    if time_range:
        params["time_range"] = time_range
    url = f"{base.rstrip('/')}/search?{urllib.parse.urlencode(params)}"
    opener = opener or urllib.request.urlopen
    try:
        with opener(url, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except Exception as ex:
        log(f"searxng failed: {type(ex).__name__}: {ex}")
        return None
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        log("searxng answered without a results list")
        return None
    out = [{"title": r.get("title"),
            "url": r.get("url"),
            "abstract": (r.get("content") or "")[:400]}
           for r in results if isinstance(r, dict) and r.get("url")]
    log(f"searxng: {len(out)} result(s) from {len(results)} raw")
    return out[:limit] or None


SYSTEM_PROMPT = f"""You are a professional music copyright attorney and YouTube curator.
Review the following search results for {GENRE} tracks uploaded this week.
GOAL: Find ONLY tracks that are 100% safe for YouTube Monetization (Ads).

STRICT CRITERIA:
1. Must be 'Royalty Free', 'No Copyright', or 'CC-BY' (Creative Commons Attribution).
2. Must allow 'Commercial Use'.
3. Discard ANY track that is 'Personal Use Only' or has 'No Derivatives'.
4. Discard ANY track where monetization safety is not 'High'.

SUBGENRE COHERENCE (IMPORTANT):
- Pick ONE subgenre of {GENRE} for this week's report (e.g. Deep House,
  Tech House, Melodic House, Afro House, Progressive House).
- EVERY track in the output array must belong to that SAME subgenre.
- Set the 'genre' field to that exact subgenre on every track.
- If no single subgenre has at least ~4 tracks, pick the one with the most
  tracks and only include those.

Output ONLY a valid JSON array of objects. ONLY include tracks with HIGH monetization safety.
{{
  "artist": "...",
  "title": "...",
  "url": "...",
  "license_type": "CC-BY / Royalty Free",
  "monetization_safety": "High",
  "reason": "Explain why this is safe for ads.",
  "genre": "Exact subgenre, shared by ALL tracks in the array"
}}
If no 100% safe tracks are found, return []."""


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


def curate(results, llm=call_llm, data_dir=DATA_DIR):
    """Ask the model to filter the candidates. Returns (tracks, genre) or (None, None).

    () with an empty list means "nothing safe this week", which is a valid
    answer; None means the model's reply could not be used at all.
    """
    user_prompt = f"Search Results:\n{json.dumps(results, indent=2)}"
    raw_reply = llm(SYSTEM_PROMPT, user_prompt, temperature=0.2, max_tokens=4000,
                    think=False)
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
            lic = (t.get("license_type") or "").strip()
            kind = download_kind(url) if url else "unknown"
            warning = {"set": "  ⚠️ playlist — no entra en la descarga automática",
                       "profile": "  ⚠️ perfil, no es una pista",
                       "unknown": "  ⚠️ enlace no reconocido"}.get(kind, "")
            lines.append(f"{i}. **{artist}** — *{title}*"
                         + (f"  ({lic})" if lic else "") + warning)
            if url:
                lines.append(f"   🔗 {url}")
                (track_urls if kind == "track" else other_urls).append(url)
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
                  time_range=""):
    """One weekly run. Returns the list of tracks (possibly empty) or None.

    `source`: auto tries SearXNG first and falls back to ddgr; "searxng" and
    "ddgr" force one, which is how you find out which half is unhappy.
    """
    genre = GENRE
    print(f"🔎 Starting STRICT discovery for monetizable {genre}...")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    results = None
    if source in ("auto", "searxng"):
        print(f"  Searching SearXNG ({searxng_url}) for: {search_terms(genre)}"
              f"{f' [time_range={time_range}]' if time_range else ''}")
        results = search_searxng(search_terms(genre), base=searxng_url,
                                 opener=opener, time_range=time_range)
    if results is None and source in ("auto", "ddgr"):
        query = build_query(genre)
        print(f"  Searching ddgr for: {query}")
        results = run_search(query, runner=runner)
    if results is None:
        return None

    print(f"  Analyzing {len(results)} candidates for MONETIZATION...")
    tracks, genre_label = curate(results, llm=llm, data_dir=data_dir)
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
    ap.add_argument("--source", choices=("auto", "searxng", "ddgr"), default="auto",
                    help="auto = local SearXNG first, ddgr as fallback")
    ap.add_argument("--searxng-url", default=SEARXNG_URL)
    ap.add_argument("--time-range", default="",
                    choices=("", "day", "week", "month", "year"),
                    help="recency filter. Empty by default because with "
                         "site:soundcloud.com a week returns 0 results and a "
                         "month returns 2, while the full set (20) is all older "
                         "than a month: a web index does not have this week's "
                         "uploads for a query this narrow")
    ap.add_argument("--selftest", action="store_true",
                    help="offline checks: ddgr and SearXNG fixtures, stubbed model")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    tracks = search_tracks(data_dir=args.data_dir, source=args.source,
                           searxng_url=args.searxng_url,
                           time_range=args.time_range)
    # 0 on success (including "nothing safe this week"), 1 when the run could not
    # be completed at all — the scheduler and the DM step care about the difference.
    return 0 if tracks is not None else 1


if __name__ == "__main__":
    sys.exit(main())
