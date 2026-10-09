#!/usr/bin/env python3
"""
history-curator.py — Weekly discovery of curious history topics and script generation.
Ensures no repeats and performs basic AI fact-checking.
"""

import os
import json
import subprocess
from datetime import datetime
from llm_sanitize import call_llm, sanitize_llm_output

# ─── Config ───
DATA_DIR = "/opt/netscan/data/history"
LOG_FILE = os.path.join(DATA_DIR, "topics_log.md")
SCRIPTS_DIR = os.path.join(DATA_DIR, "scripts")

def get_past_topics():
    """Read the log of previously used topics to avoid duplicates."""
    if not os.path.exists(LOG_FILE):
        return ""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except: return ""

def save_topic(title):
    """Log the new topic title."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"- [{datetime.now().strftime('%Y-%m-%d')}] {title}\n")

def save_script(title, content):
    """Save the full video script to a file."""
    os.makedirs(SCRIPTS_DIR, exist_ok=True)
    # Clean title for filename
    safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "_")).rstrip()
    filename = f"{datetime.now().strftime('%Y-%m-%d')}_{safe_title.replace(' ', '_')}.md"
    path = os.path.join(SCRIPTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def run_curator():
    print("📜 Starting Weekly History Curator (Grounded in Web Search)...")
    past_topics = get_past_topics()
    
    # Phase 1: Search for a unique topic inspiration
    print("🔎 Searching for inspiration...")
    
    system_prompt_find = """You are a specialized historian. Your goal is to find ONE extremely curious, 
lesser-known historical event or figure (Spanish or World history) suitable for a viral video.

RULES:
- It must NOT be in the following list of past topics:
{past_topics}
- Focus on "Hidden History", "Bizarre facts" or "Historical anomalies".
- Output ONLY the Title.
"""
    
    topic_title = call_llm(system_prompt_find.format(past_topics=past_topics), "Find a new amazing history topic.")
    topic_title = topic_title.strip().strip('"').split("\n")[0]
    
    if not topic_title:
        print("❌ Failed to find a topic.")
        return

    print(f"✨ New Topic Found: {topic_title}")

    # Phase 2: SEARCH INTERNET FOR FACT CHECKING
    print(f"📡 Searching the web for: {topic_title}...")
    search_query = f"{topic_title} historia real datos curiosos fuentes"
    cmd = ["ddgr", "--num", "10", "--json", search_query]
    
    search_context = ""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            search_data = json.loads(result.stdout)
            search_context = json.dumps(search_data, indent=2)
            print(f"  Found {len(search_data)} search results for grounding.")
    except Exception as e:
        print(f"  Warning: Search failed ({e}), relying on internal knowledge.")

    # Phase 3: Generate the Script Grounded in Search Data
    print(f"✍️ Generating detailed script for: {topic_title}...")
    
    system_prompt_script = """You are a professional YouTube scriptwriter and historical researcher.
Create a complete video script for the topic provided. 

CRITICAL: Use the provided SEARCH DATA to ensure factual accuracy. 
If the search data contradicts common myths, highlight it in the Fact-Check section.

STRUCTURE:
1. THE HOOK: A 10-second opening to grab attention.
2. THE STORY: Detailed chronological or thematic explanation.
3. CURIOUS DETAILS: 3-4 facts that people don't usually know.
4. FACT-CHECK BOX: List verified facts and debunk potential myths using the search context.
5. SOURCES: List the actual URLs or sources found in the search data.

STYLE: Engaging, mysterious, and clear.
LANGUAGE: Spanish (Español).
"""

    user_prompt = f"TOPIC: {topic_title}\n\nSEARCH DATA FOR GROUNDING:\n{search_context}"

    script_content = call_llm(system_prompt_script, user_prompt, temperature=0.3, max_tokens=5000)
    script_content = sanitize_llm_output(script_content)

    if script_content:
        path = save_script(topic_title, script_content)
        save_topic(topic_title)
        print(f"✅ Script saved to: {path}")
    else:
        print("❌ Failed to generate script.")

if __name__ == "__main__":
    run_curator()
