#!/usr/bin/env python3
"""
system-think.py — Weekly self-audit and meta-optimization script.
Analyzes logs, researches latest tech, and proposes system improvements.
"""

import os
import json
import subprocess
from datetime import datetime
from llm_sanitize import call_llm, sanitize_llm_output

# ─── Config ───
DATA_DIR = "/opt/netscan/data/system"
LOG_FILE = "/opt/netscan/data/queue-runner.log"
PROPOSAL_FILE = os.path.join(DATA_DIR, f"upgrade-proposal-{datetime.now().strftime('%Y-%V')}.md")
JOBS_FILE = "/opt/netscan/data/jobs.json"

def get_recent_logs(lines=200):
    """Read the tail of the log file to analyze system health."""
    if not os.path.exists(LOG_FILE): return "No logs found."
    try:
        result = subprocess.run(["tail", "-n", str(lines), LOG_FILE], capture_output=True, text=True)
        return result.stdout
    except: return "Error reading logs."

def get_current_jobs():
    """Read the current jobs configuration."""
    if not os.path.exists(JOBS_FILE): return "No jobs.json found."
    try:
        with open(JOBS_FILE, "r") as f: return f.read()
    except: return "Error reading jobs."

def run_self_audit():
    print("🧠 Starting Weekly System Self-Audit...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    logs = get_recent_logs()
    jobs = get_current_jobs()
    
    # Phase 1: Research latest optimizations
    print("📡 Researching latest AI & Linux optimizations...")
    search_query = "latest llama.cpp performance optimizations AMD APU Alpine Linux 2026"
    search_context = ""
    try:
        cmd = ["ddgr", "--num", "5", "--json", search_query]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        search_context = result.stdout
    except: pass

    # Phase 2: Analyze and Propose
    print("🤖 Analyzing meta-data and generating upgrade proposal...")
    
    system_prompt = """You are a Principal Systems Architect and AI Infrastructure Specialist.
Your goal is to improve the 'Clawd BC-250' system.

INPUT DATA:
1. RECENT LOGS (Check for errors, timeouts, OOM):
{logs}

2. CURRENT JOBS:
{jobs}

3. LATEST TECH RESEARCH:
{search_context}

TASKS:
1. Identify bottlenecks or recurring errors.
2. Propose 3 specific code or config optimizations.
3. Recommend 1 new feature or Job that would add massive value.
4. Check if current LLM models are still the best choice.

Output a high-level Markdown report in Spanish. Be technical and precise.
"""

    proposal = call_llm(system_prompt.format(logs=logs, jobs=jobs, search_context=search_context), "Analyze my system.")
    proposal = sanitize_llm_output(proposal)

    if proposal:
        with open(PROPOSAL_FILE, "w", encoding="utf-8") as f:
            f.write(proposal)
        print(f"✅ Upgrade proposal saved to: {PROPOSAL_FILE}")
    else:
        print("❌ AI failed to generate proposal.")

if __name__ == "__main__":
    run_self_audit()
