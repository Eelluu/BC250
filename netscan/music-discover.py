#!/usr/bin/env python3
"""
music-discover.py — Weekly music discovery for YouTube House Mixes.
Focuse on Creative Commons / Royalty Free tracks for monetization.
"""

import json
import os
import subprocess
import time
import urllib.request
from datetime import datetime, timedelta
from llm_sanitize import call_llm, sanitize_llm_output

# ─── Config ───
DATA_DIR = "/opt/netscan/data/music"
OUTPUT_FILE = os.path.join(DATA_DIR, f"house-mix-{datetime.now().strftime('%Y-W%V')}.json")
GENRE = "House Music"
PLATFORMS = ["SoundCloud", "Bandcamp"]

def search_tracks():
    print(f"🔎 Starting STRICT discovery for monetizable {GENRE}...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Calculate date for "this week"
    last_week = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    # Strictly target monetizable / royalty-free terms
    search_query = f'site:soundcloud.com "{GENRE}" ("no copyright" OR "commercial use" OR "royalty free" OR "CC-BY") after:{last_week}'
    
    print(f"  Searching: {search_query}")
    cmd = ["ddgr", "--num", "50", "--json", search_query]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        search_results = json.loads(result.stdout)
    except Exception as e:
        print(f"  Error running search: {e}")
        return

    # Phase 2: LLM Filtering & Curation (Ultra-Strict)
    print(f"  Analyzing {len(search_results)} candidates for MONETIZATION...")
    
    system_prompt = f"""You are a professional music copyright attorney and YouTube curator.
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

    user_prompt = f"Search Results:\n{json.dumps(search_results, indent=2)}"

    raw_reply = call_llm(system_prompt, user_prompt, temperature=0.2, max_tokens=4000)
    curated_json = sanitize_llm_output(raw_reply)

    if curated_json and curated_json.startswith("["):
        tracks = json.loads(curated_json)
        # Coherence: keep the dominant subgenre so the whole report sounds
        # "más o menos" the same genre (user request).
        genre_label, tracks = pick_dominant_genre(tracks)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(tracks, f, indent=2, ensure_ascii=False)
        print(f"✅ Discovery complete. {len(tracks)} tracks found.")
        print(f"  Report saved to: {OUTPUT_FILE}")
        write_discord_report(tracks, genre_label)
    else:
        print("❌ AI failed to produce a valid track list.")


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
    norm = lambda t: ((t.get("genre") or "").strip() or GENRE)
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
            print(f"  🎧 Coherente: {len(tracks)} → {len(kept)} pistas de «{top_genre}» (se descartaron los otros subgéneros)")
        return top_genre, kept
    return top_genre, tracks


def write_discord_report(tracks, genre_label=None):
    """Write a Discord-friendly report: name + link per track, plus a ready
    yt-dlp command to download them ALL at once (owner checks before running).
    queue-runner DMs this file to the owner after a successful job run."""
    os.makedirs(DATA_DIR, exist_ok=True)
    report_path = os.path.join(DATA_DIR, "latest-report.md")
    week = datetime.now().strftime("%Y-W%V")
    lines = [
        f"🎵 **Música semanal — {genre_label or GENRE} (CC / royalty-free)**",
        f"Semana {week} · {len(tracks)} pista(s) encontrada(s)",
        "",
    ]
    if not tracks:
        lines.append("No se encontraron pistas 100% seguras para monetización esta semana.")
    else:
        for i, t in enumerate(tracks, 1):
            artist = (t.get("artist") or "?").strip()
            title = (t.get("title") or "?").strip()
            url = (t.get("url") or "").strip().replace('"', "")
            lic = (t.get("license_type") or "").strip()
            lines.append(f"{i}. **{artist}** — *{title}*" + (f"  ({lic})" if lic else ""))
            if url:
                lines.append(f"   🔗 {url}")
        lines.append("")
        lines.append("▶️ **Descarga todas de golpe** (revísalas antes de descargar):")
        urls = [t.get("url", "").strip().replace('"', "") for t in tracks if t.get("url", "").strip()]
        # Keep the command under ~1800 chars so the DM code block is never split
        # mid-line by the 2000-char Discord limit (many tracks would break it).
        prefix = "yt-dlp --extract-audio --audio-format mp3 --audio-quality 0"
        cmd_parts = []
        used = 0
        for u in urls:
            piece = f'"{u}"'
            if used + len(piece) + 1 > 1800:
                break
            cmd_parts.append(piece)
            used += len(piece) + 1
        cmd = prefix + " " + " ".join(cmd_parts)
        if len(cmd_parts) < len(urls):
            cmd += f"\n# …y {len(urls) - len(cmd_parts)} pista(s) más — ver informe completo en el servidor"
        lines.append("```bash")
        lines.append(cmd)
        lines.append("```")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Discord report saved to: {report_path}")


if __name__ == "__main__":
    search_tracks()
