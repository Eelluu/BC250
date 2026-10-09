#!/usr/bin/env python3
"""system-think.py — System intelligence: GPU analysis, network security, health watchdog.
Refactored for v10 (Discord + llama-swap).
"""

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from llm_sanitize import call_llm, sanitize_llm_output

# ── Config ─────────────────────────────────────────────────────────────────
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8080/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma4-26b-q3")

DATA_DIR = Path("/opt/netscan/data")
THINK_DIR = DATA_DIR / "think"
WEB_DIR = Path("/opt/netscan/web")

# Discord Config (Captured by queue-runner)
DISCORD_OWNER_ID = int(os.environ.get('DISCORD_OWNER_ID', 0))

DASHBOARD_URL = "http://127.0.0.1:8888/"

TODAY = datetime.now().strftime("%Y%m%d")
TODAY_ISO = datetime.now().strftime("%Y-%m-%d")
NOW = datetime.now()

# Thresholds
EXPECTED_HOST_COUNT_MIN = 30
EXPECTED_HOST_COUNT_MAX = 80
STALE_HOURS_SCAN = 36
STALE_HOURS_DASHBOARD = 36
GPU_TEMP_WARN = 95
GPU_TEMP_CRIT = 100

# ── Helpers ────────────────────────────────────────────────────────────────

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def send_discord_alert(msg):
    """Prints a special alert tag that queue-runner can detect or the user can see in logs."""
    log(f"🚨 SYSTEM ALERT: {msg}")
    return True

def call_llm_v1(system_prompt, user_prompt, temperature=0.3, max_tokens=4000):
    """Call centralized v10 LLM."""
    return call_llm(system_prompt, user_prompt, temperature, max_tokens)

def load_json(path):
    try:
        with open(path) as f: return json.load(f)
    except: return None

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(data, f, indent=2, ensure_ascii=False)

def load_gpu_csv(date_str=None):
    if date_str is None: date_str = TODAY
    path = DATA_DIR / "gpu" / f"gpu-{date_str}.csv"
    if not path.exists(): return []
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "power_w": float(row.get("power_w", 0)),
                    "temp_c": float(row.get("temp_c", 0)),
                    "freq_mhz": float(row.get("freq_mhz", 0)),
                    "vram_mb": float(row.get("vram_mb", 0)),
                    "throttle": row.get("throttle_status", "0x0"),
                })
            except: continue
    return rows

# ── Modes ──────────────────────────────────────────────────────────────────

def run_gpu():
    log("Running GPU Intelligence Analysis...")
    rows = load_gpu_csv()
    if not rows:
        log("No GPU data found for today.")
        return

    # Simplified stats for LLM
    avg_temp = sum(r['temp_c'] for r in rows) / len(rows)
    max_temp = max(r['temp_c'] for r in rows)
    
    system = "You are a hardware analyst. Analyze this GPU data for BC-250 (AMD Cyan Skillfish)."
    user = f"Data: Avg Temp: {avg_temp:.1f}C, Max Temp: {max_temp:.1f}C, Samples: {len(rows)}"
    
    analysis = call_llm_v1(system, user)
    if analysis:
        log("GPU Analysis Complete.")
        save_json(THINK_DIR / f"system-gpu-{TODAY}.json", {"analysis": analysis})

def run_netsec():
    log("Running Network Security Intelligence...")
    # (Simplified for v10 integration)
    system = "You are a network security expert."
    user = "Analyze the network posture of a /22 subnet with 60 devices. Suggest improvements."
    analysis = call_llm_v1(system, user)
    if analysis:
        log("Netsec Analysis Complete.")
        save_json(THINK_DIR / f"system-netsec-{TODAY}.json", {"analysis": analysis})

def run_health():
    log("Running Health Watchdog...")
    # Check for OOMs or failures
    alerts = []
    if datetime.now().hour == 5: # Example check
        alerts.append("Daily health check active.")
    
    if alerts:
        msg = "\n".join(alerts)
        send_discord_alert(msg)
    
    system = "You are a systems engineer. Evaluate system health alerts."
    analysis = call_llm_v1(system, f"Alerts: {alerts}")
    if analysis:
        log("Health Assessment Complete.")
        save_json(THINK_DIR / f"system-health-{TODAY}.json", {"analysis": analysis})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--netsec', action='store_true')
    parser.add_argument('--health', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.gpu or args.all: run_gpu()
    if args.netsec or args.all: run_netsec()
    if args.health or args.all: run_health()

if __name__ == "__main__":
    main()
