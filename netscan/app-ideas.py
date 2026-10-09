#!/usr/bin/env python3
"""
app-ideas.py — Weekly generation of 5 simple app ideas with full business/tech plans.
Includes duplication prevention and detailed architectural blueprints.
"""

import os
import json
import subprocess
from datetime import datetime
from llm_sanitize import call_llm, sanitize_llm_output

# ─── Config ───
DATA_DIR = "/opt/netscan/data/apps"
LOG_FILE = os.path.join(DATA_DIR, "apps_log.md")
PLANS_DIR = os.path.join(DATA_DIR, "plans")

def get_past_apps():
    """Read the log of previously generated apps to avoid duplicates."""
    if not os.path.exists(LOG_FILE):
        return ""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except: return ""

def save_to_log(apps_list):
    """Log the new app names and descriptions."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for app in apps_list:
            f.write(f"- [{datetime.now().strftime('%Y-%m-%d')}] {app['name']}: {app['description']}\n")

def save_plan(app_name, content):
    """Save the full architectural plan to a file."""
    os.makedirs(PLANS_DIR, exist_ok=True)
    safe_name = "".join(c for c in app_name if c.isalnum() or c in (" ", "_")).rstrip()
    filename = f"{datetime.now().strftime('%Y-%m-%d')}_{safe_name.replace(' ', '_')}.md"
    path = os.path.join(PLANS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def run_app_architect():
    print("🚀 Starting Weekly App Architect (Kotlin & Tech Research)...")
    past_apps = get_past_apps()
    
    # Phase 1: Generate 5 Unique Mobile App Ideas
    print("🧠 Thinking of 5 new mobile app ideas (Kotlin focus)...")
    
    system_prompt_ideas = """You are a creative mobile product manager.
Generate 5 unique, simple, and useful MOBILE app ideas.

RULES:
- Language: Kotlin / Android.
- They must NOT be similar to these past ideas:
{past_apps}
- Focus on "Kotlin Multiplatform (KMP)", "Jetpack Compose", or "Material 3".
- Simple enough for a solo dev but high value.

Output ONLY a valid JSON array of objects:
[
  {{"name": "App Name", "description": "One-line catchy description"}},
  ...
]
"""
    
    ideas_raw = call_llm(system_prompt_ideas.format(past_apps=past_apps), "Generate 5 mobile ideas.")
    ideas_json = sanitize_llm_output(ideas_raw)
    
    try:
        new_apps = json.loads(ideas_json)
    except:
        print("❌ AI failed to generate ideas JSON.")
        return

    # Phase 2: Tech Research & Plan Generation for each idea
    for app in new_apps:
        print(f"🏗️  Researching & Architecting: {app['name']}...")
        
        # SEARCH for latest tech/libraries for this specific idea
        search_query = f"latest android libraries kotlin permissions for {app['name']} {app['description']}"
        search_context = ""
        try:
            cmd = ["ddgr", "--num", "5", "--json", search_query]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                search_context = result.stdout
        except: pass

        system_prompt_plan = f"""You are a Senior Android Architect (Kotlin Specialist).
Create a MASSIVELY DETAILED plan for the following app idea.

RESEARCH DATA (Latest tech/libraries):
{search_context}

STRUCTURE:
1. OVERVIEW: Target audience and problem solved.
2. FEATURE ROADMAP: (EASY / MEDIUM / HARD)
3. DESIGN PLAN: Jetpack Compose components, Material 3 theme, UI flow.
4. CONSTRUCTION PLAN: 
   - Tech Stack: Kotlin, latest versions of libraries (Room, Retrofit, Hilt, etc.)
   - Permissions: List specific Android Manifest permissions needed.
   - Build Guide: Step-by-step implementation.
5. MONETIZATION: 3 specific ways (Ads, IAP, Subscriptions).

STYLE: Technical, precise, using latest Android standards.
LANGUAGE: Spanish (Español).
"""

        full_plan = call_llm(system_prompt_plan, f"APP IDEA: {app['name']} - {app['description']}", temperature=0.3, max_tokens=5000)
        full_plan = sanitize_llm_output(full_plan)

        if full_plan:
            path = save_plan(app['name'], full_plan)
            print(f"  ✅ Spec saved: {os.path.basename(path)}")
        
    save_to_log(new_apps)
    print("✨ All 5 Kotlin app specs generated successfully!")

if __name__ == "__main__":
    run_app_architect()
