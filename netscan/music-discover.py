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


def log(msg):
    print(f"  {msg}", flush=True)


def build_query(genre=GENRE, days=7):
    """Strictly monetizable terms, this week, on SoundCloud."""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    return (f'site:soundcloud.com "{genre}" ("no copyright" OR "commercial use" '
            f'OR "royalty free" OR "CC-BY") after:{since}')


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
        for i, t in enumerate(tracks, 1):
            artist = (t.get("artist") or "?").strip()
            title = (t.get("title") or "?").strip()
            url = (t.get("url") or "").strip().replace('"', "")
            lic = (t.get("license_type") or "").strip()
            lines.append(f"{i}. **{artist}** — *{title}*" + (f"  ({lic})" if lic else ""))
            if url:
                lines.append(f"   🔗 {url}")
        lines += ["", "▶️ **Descarga todas de golpe** (revísalas antes de descargar):"]
        urls = [t.get("url", "").strip().replace('"', "")
                for t in tracks if t.get("url", "").strip()]
        # Keep the command under ~1800 chars so the DM code block is never split
        # mid-line by the 2000-char Discord limit (many tracks would break it).
        prefix = "yt-dlp --extract-audio --audio-format mp3 --audio-quality 0"
        cmd_parts, used = [], 0
        for u in urls:
            piece = f'"{u}"'
            if used + len(piece) + 1 > 1800:
                break
            cmd_parts.append(piece)
            used += len(piece) + 1
        cmd = prefix + " " + " ".join(cmd_parts)
        if len(cmd_parts) < len(urls):
            cmd += (f"\n# …y {len(urls) - len(cmd_parts)} pista(s) más — ver "
                    f"informe completo en el servidor")
        lines += ["```bash", cmd, "```"]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def search_tracks(data_dir=DATA_DIR, runner=subprocess.run, llm=call_llm):
    """One weekly run. Returns the list of tracks (possibly empty) or None."""
    genre = GENRE
    print(f"🔎 Starting STRICT discovery for monetizable {genre}...")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    query = build_query(genre)
    print(f"  Searching: {query}")
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
    ap.add_argument("--selftest", action="store_true",
                    help="offline checks: ddgr fixtures and a stubbed model")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    tracks = search_tracks(data_dir=args.data_dir)
    # 0 on success (including "nothing safe this week"), 1 when the run could not
    # be completed at all — the scheduler and the DM step care about the difference.
    return 0 if tracks is not None else 1


if __name__ == "__main__":
    sys.exit(main())
