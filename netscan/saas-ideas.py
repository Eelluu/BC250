#!/usr/bin/env python3
"""
saas-ideas.py — Weekly generation of profitable SaaS ideas in underexploited niches.
Includes deep market analysis and construction blueprints.
"""

import os
import json
import subprocess
from datetime import datetime
from llm_sanitize import call_llm, sanitize_llm_output

# ─── Config ───
DATA_DIR = "/opt/netscan/data/saas"
LOG_FILE = os.path.join(DATA_DIR, "saas_log.md")
PLANS_DIR = os.path.join(DATA_DIR, "plans")

def get_past_saas():
    """Read the log of previously generated SaaS ideas to avoid duplicates."""
    if not os.path.exists(LOG_FILE):
        return ""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except: return ""

def save_to_log(saas_list):
    """Log the new SaaS names and descriptions."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for item in saas_list:
            f.write(f"- [{datetime.now().strftime('%Y-%m-%d')}] {item['name']}: {item['description']}\n")

_DIGEST = []


def save_plan(name, content):
    """Save the full business/tech plan to a file."""
    os.makedirs(PLANS_DIR, exist_ok=True)
    safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_")).rstrip()
    filename = f"{datetime.now().strftime('%Y-%m-%d')}_{safe_name.replace(' ', '_')}.md"
    path = os.path.join(PLANS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    _DIGEST.append("- **%s** -> %s" % (name, os.path.basename(path)))
    with open(os.path.join(DATA_DIR, "latest-digest.md"), "w", encoding="utf-8") as f:
        f.write("# SaaS ideas - %s\n\n%s\n" % (datetime.now().strftime("%Y-%m-%d"),
                                                 "\n".join(_DIGEST)))
    print("  Digest: " + DATA_DIR + "/latest-digest.md")
    return path

def run_saas_visionary():
    print("💰 Starting Weekly SaaS Visionary...")
    past_ideas = get_past_saas()
    
    # Phase 1: Generate 3-5 Highly Profitable SaaS Ideas
    print("🧠 Researching underexploited SaaS niches...")
    
    system_prompt_ideas = """You are a venture capitalist and expert SaaS founder.
Generate 3-5 SaaS ideas (Software as a Service) that target UNEXPLOITED or NEW niches.

CRITERIA:
- High profitability potential.
- Low competition or inefficient existing alternatives.
- Focus on "Vertical SaaS" (specific industries) or "AI-First Micro-SaaS".
- Must NOT be similar to these past ideas:
{past_ideas}

Output ONLY a valid JSON array of objects:
[
  {{"name": "SaaS Name", "description": "Compelling business value proposition"}},
  ...
]
"""
    
    ideas_raw = call_llm(system_prompt_ideas.format(past_ideas=past_ideas), "Generate SaaS ideas.")
    ideas_json = sanitize_llm_output(ideas_raw)
    
    try:
        new_saas = json.loads(ideas_json)
    except:
        print("❌ AI failed to generate SaaS JSON.")
        return

    # Phase 2: Market Research & Blueprint Generation
    for item in new_saas:
        print(f"📈 Analyzing Market & Architecting: {item['name']}...")
        
        # Search for competition and potential gaps
        search_query = f"SaaS alternatives and market gap for {item['name']} {item['description']}"
        search_context = ""
        try:
            cmd = ["ddgr", "--num", "5", "--json", search_query]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                search_context = result.stdout
        except: pass

        system_prompt_blueprint = f"""You are a Senior SaaS Architect and Growth Hacker.
Create a MASSIVELY DETAILED business and technical blueprint for this SaaS.

MARKET CONTEXT (Search data):
{search_context}

STRUCTURE:
1. THE OPPORTUNITY: Why now? What is the specific "pain point" and why alternatives suck.
2. FEATURE ROADMAP:
   - Phase 1: Core Automation (MVP)
   - Phase 2: Multi-user & Integration
   - Phase 3: AI/Enterprise features
3. TECHNICAL STACK: Recommend a modern, scalable stack (ej: Next.js, Go, Supabase, etc.).
4. GO-TO-MARKET: How to get the first 10 paying customers without spending on ads.
5. PRICING & MONETIZATION: Detailed pricing tiers (Freemium, Pro, Enterprise).

STYLE: Business-focused, strategic, and technical.
LANGUAGE: Spanish (Español).
"""

        blueprint = call_llm(system_prompt_blueprint, f"SaaS IDEA: {item['name']} - {item['description']}", temperature=0.3, max_tokens=5000)
        blueprint = sanitize_llm_output(blueprint)

        if blueprint:
            path = save_plan(item['name'], blueprint)
            print(f"  ✅ Blueprint saved: {os.path.basename(path)}")
        
    save_to_log(new_saas)
    print("✨ SaaS portfolio updated with new blueprints!")

if __name__ == "__main__":
    run_saas_visionary()
