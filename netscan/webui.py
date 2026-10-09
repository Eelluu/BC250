#!/usr/bin/env python3
"""
BC-250 Configuration Web UI
============================
Temporary web interface for configuring the AI system.
Launches on demand, shuts down when done.

Usage: python3 /opt/netscan/webui.py [--port 8888]
"""

import http.server
import json
import os
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

NETSCAN_DIR = Path("/opt/netscan")
DATA_DIR = NETSCAN_DIR / "data"
ENV_FILE = NETSCAN_DIR / ".env"
JOBS_FILE = DATA_DIR / "jobs.json"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--port" else 8888

# ─── HTML Frontend (embedded) ───────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🦞 BC-250 AI Config</title>
<style>
:root {
  --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
  --fg: #c9d1d9; --fg2: #8b949e; --accent: #58a6ff;
  --green: #3fb950; --red: #f85149; --orange: #d29922;
  --border: #30363d; --radius: 8px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--fg); line-height: 1.5; }
.container { max-width: 900px; margin: 0 auto; padding: 24px; }
h1 { font-size: 28px; margin-bottom: 8px; }
h2 { font-size: 18px; color: var(--accent); margin-bottom: 12px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
.subtitle { color: var(--fg2); margin-bottom: 24px; font-size: 14px; }
.section { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 20px; margin-bottom: 16px; }
label { display: block; font-size: 13px; color: var(--fg2); margin-bottom: 4px; margin-top: 12px; }
label:first-child { margin-top: 0; }
input[type="text"], input[type="number"], input[type="password"], textarea, select {
  width: 100%; padding: 8px 12px; background: var(--bg); border: 1px solid var(--border);
  border-radius: 6px; color: var(--fg); font-size: 14px; font-family: inherit; }
input:focus, textarea:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(88,166,255,0.15); }
textarea { resize: vertical; min-height: 60px; }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
.btn { display: inline-flex; align-items: center; gap: 6px; padding: 10px 20px;
  border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; border: none; }
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover { background: #79c0ff; }
.btn-danger { background: var(--red); color: #fff; }
.btn-danger:hover { background: #ff7b72; }
.btn-secondary { background: var(--bg3); color: var(--fg); border: 1px solid var(--border); }
.btn-secondary:hover { background: var(--border); }
.btn-row { display: flex; gap: 12px; margin-top: 20px; }
.status { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px;
  border-radius: 12px; font-size: 12px; font-weight: 600; }
.status-ok { background: rgba(63,185,80,0.15); color: var(--green); }
.status-err { background: rgba(248,81,73,0.15); color: var(--red); }
.status-warn { background: rgba(210,153,34,0.15); color: var(--orange); }
.jobs-list { list-style: none; }
.jobs-list li { display: flex; align-items: center; gap: 12px; padding: 10px 0;
  border-bottom: 1px solid var(--border); }
.jobs-list li:last-child { border-bottom: none; }
.jobs-list .job-name { font-weight: 600; flex: 1; }
.jobs-list .job-schedule { color: var(--fg2); font-size: 13px; font-family: monospace; }
.jobs-list input[type="checkbox"] { width: 18px; height: 18px; accent-color: var(--accent); }
.toast { position: fixed; bottom: 24px; right: 24px; padding: 12px 20px; border-radius: 8px;
  font-size: 14px; font-weight: 600; z-index: 1000; opacity: 0; transition: opacity 0.3s;
  pointer-events: none; }
.toast.show { opacity: 1; }
.toast-ok { background: var(--green); color: #fff; }
.toast-err { background: var(--red); color: #fff; }
.save-bar { position: sticky; bottom: 0; background: var(--bg); border-top: 1px solid var(--border);
  padding: 16px 0; display: flex; justify-content: space-between; align-items: center; }
.tab-bar { display: flex; gap: 4px; margin-bottom: 20px; flex-wrap: wrap; }
.tab { padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px;
  font-weight: 600; background: transparent; color: var(--fg2); border: none; }
.tab.active { background: var(--accent); color: #fff; }
.tab:hover:not(.active) { background: var(--bg3); }
.tab-content { display: none; }
.tab-content.active { display: block; }
.help { font-size: 12px; color: var(--fg2); margin-top: 4px; }
hr { border: none; border-top: 1px solid var(--border); margin: 16px 0; }
</style>
</head>
<body>
<div class="container">
<h1>🦞 BC-250 AI Configuration</h1>
<p class="subtitle">Configure your AI system — changes are saved to .env and jobs.json</p>

<div class="tab-bar">
  <button class="tab active" onclick="switchTab('general')">⚙️ General</button>
  <button class="tab" onclick="switchTab('models')">🧠 Models</button>
  <button class="tab" onclick="switchTab('sd')">🎨 Image Gen</button>
  <button class="tab" onclick="switchTab('jobs')">📋 Jobs</button>
  <button class="tab" onclick="switchTab('status')">📊 Status</button>
</div>

<!-- GENERAL TAB -->
<div id="tab-general" class="tab-content active">
<div class="section">
<h2>🦞 Discord Bot</h2>
<label>Discord Token</label>
<input type="password" id="DISCORD_TOKEN" placeholder="your-bot-token">
<label>Owner ID</label>
<input type="text" id="DISCORD_OWNER_ID" placeholder="123456789">
</div>
<div class="section">
<h2>💤 Mode Settings</h2>
<div class="row">
  <div>
    <label>Smart Mode Curfew (hour, 0-23)</label>
    <input type="number" id="SMART_MODE_CURFEW_HOUR" min="0" max="23" value="1">
    <p class="help">Auto-sleep after this hour if smart mode is active (1 = 1:00 AM)</p>
  </div>
  <div>
    <label>Useful Hours Start</label>
    <input type="number" id="USEFUL_HOURS_START" min="0" max="23" value="9">
    <p class="help">Cron jobs pause during these hours</p>
  </div>
</div>
<div class="row">
  <div>
    <label>Useful Hours End</label>
    <input type="number" id="USEFUL_HOURS_END" min="0" max="23" value="21">
  </div>
  <div></div>
</div>
</div>
<div class="section">
<h2>⬇️ Hugging Face</h2>
<div class="row">
  <div>
    <label>HF_XET_HIGH_PERFORMANCE</label>
    <select id="HF_XET_HIGH_PERFORMANCE">
      <option value="">Disabled</option>
      <option value="1">Enabled (parallel chunks)</option>
    </select>
  </div>
  <div>
    <label>HF_HUB_DOWNLOAD_TIMEOUT (seconds)</label>
    <input type="number" id="HF_HUB_DOWNLOAD_TIMEOUT" value="60">
  </div>
</div>
</div>
</div>

<!-- MODELS TAB -->
<div id="tab-models" class="tab-content">
<div class="section">
<h2>🧠 LLM Models (llama-swap)</h2>
<label>LLM URL</label>
<input type="text" id="LLM_URL" value="http://localhost:8080/v1">
<label>Primary Chat Model</label>
<input type="text" id="LLM_MODEL" placeholder="gemma4-26b-q3">
<label>Fallback Model</label>
<input type="text" id="LLM_FALLBACK_MODEL" placeholder="qwen3.5-35b-a3b-iq2m">
<label>Smart Mode Model</label>
<input type="text" id="LLM_SMART_MODEL" placeholder="qwen3.5-35b-a3b-iq2m">
</div>
</div>

<!-- SD TAB -->
<div id="tab-sd" class="tab-content">
<div class="section">
<h2>🎨 Stable Diffusion Models</h2>
<label>Models Base Directory</label>
<input type="text" id="SD_MODELS_DIR" value="/opt/stable-diffusion.cpp/models">
<hr>
<h3 style="color:var(--accent);margin:12px 0 8px">HiDream-O1 (PRIMARY)</h3>
<div class="row">
  <div><label>Model file</label><input type="text" id="SD_HIDREAM_MODEL" placeholder="HiDream-O1-Image-Full-FP8.safetensors"></div>
  <div><label>Directory</label><input type="text" id="SD_HIDREAM_DIR" placeholder=".../models/hidream"></div>
</div>
<hr>
<h3 style="color:var(--accent);margin:12px 0 8px">Qwen-Image 2.1 (SOTA Generation &amp; Editing)</h3>
<div class="row">
  <div><label>Diffusion Model</label><input type="text" id="SD_QWEN_MODEL" placeholder="qwen_image_2.1-Q4_K.gguf"></div>
  <div><label>VAE</label><input type="text" id="SD_QWEN_VAE" placeholder="qwen_image_2.1_vae_bf16.safetensors"></div>
</div>
<div class="row">
  <div><label>LLM Text Encoder</label><input type="text" id="SD_QWEN_LLM" placeholder="Qwen3VL-8B-Instruct-Q4_K_M.gguf"></div>
  <div><label>Vision Projector (editing)</label><input type="text" id="SD_QWEN_VISION" placeholder="mmproj-Qwen3VL-8B-Instruct-F16.gguf"></div>
</div>
<div class="row">
  <div><label>Directory</label><input type="text" id="SD_QWEN_DIR" placeholder=".../models/qwen_image"></div>
</div>
<hr>
<h3 style="color:var(--fg2);margin:12px 0 8px">FLUX (Fallback)</h3>
<div class="row">
  <div><label>FLUX.1 Model</label><input type="text" id="SD_FLUX_MODEL" placeholder="flux1-schnell-q4_k.gguf"></div>
  <div><label>FLUX.1 VAE</label><input type="text" id="SD_FLUX_VAE" placeholder="ae.safetensors"></div>
</div>
<div class="row">
  <div><label>FLUX.2 Model</label><input type="text" id="SD_FLUX2_MODEL" placeholder="flux-2-klein-9b-Q4_0.gguf"></div>
  <div><label>FLUX.2 VAE</label><input type="text" id="SD_FLUX2_VAE" placeholder="flux2-vae.safetensors"></div>
</div>
<div class="row">
  <div><label>FLUX.2 LLM Encoder</label><input type="text" id="SD_FLUX2_LLM" placeholder="qwen3-8b-Q4_K_M.gguf"></div>
  <div><label>Kontext (editing)</label><input type="text" id="SD_KONTEXT_MODEL" placeholder="flux1-kontext-dev-Q4_K_M.gguf"></div>
</div>
<div class="row">
  <div><label>CLIP-L</label><input type="text" id="SD_FLUX_CLIP_L" placeholder="clip_l.safetensors"></div>
  <div><label>T5XXL</label><input type="text" id="SD_FLUX_T5XXL" placeholder="t5-v1_1-xxl-encoder-Q4_K_M.gguf"></div>
</div>
<hr>
<h3 style="color:var(--fg2);margin:12px 0 8px">WAN 2.1 (Video)</h3>
<div class="row3">
  <div><label>WAN Model</label><input type="text" id="SD_WAN_MODEL" placeholder="wan2.1_t2v_1.3B_q4_0.gguf"></div>
  <div><label>WAN VAE</label><input type="text" id="SD_WAN_VAE" placeholder="wan_2.1_vae.safetensors"></div>
  <div><label>WAN T5XXL</label><input type="text" id="SD_WAN_T5XXL" placeholder="umt5-xxl-encoder-Q4_K_M.gguf"></div>
</div>
<hr>
<h3 style="color:var(--fg2);margin:12px 0 8px">Upscale</h3>
<label>ESRGAN Model</label>
<input type="text" id="SD_ESRGAN_MODEL" placeholder=".../esrgan/RealESRGAN_x4plus.pth">
</div>
</div>

<!-- JOBS TAB -->
<div id="tab-jobs" class="tab-content">
<div class="section">
<h2>📋 Cron Jobs</h2>
<ul class="jobs-list" id="jobs-list">
  <li><span class="job-name" style="color:var(--fg2)">Loading...</span></li>
</ul>
</div>
</div>

<!-- STATUS TAB -->
<div id="tab-status" class="tab-content">
<div class="section">
<h2>📊 Service Status</h2>
<div id="status-content">Loading...</div>
</div>
<div class="section">
<h2>💾 Disk Usage</h2>
<div id="disk-content">Loading...</div>
</div>
</div>

<div class="save-bar">
  <span id="save-status" style="font-size:13px;color:var(--fg2)">Changes are not saved yet</span>
  <div class="btn-row" style="margin:0">
    <button class="btn btn-secondary" onclick="location.reload()">↻ Reload</button>
    <button class="btn btn-primary" onclick="saveConfig()">💾 Save Config</button>
    <button class="btn btn-danger" onclick="saveAndClose()">💾 Save & Close</button>
  </div>
</div>
</div>

<div class="toast" id="toast"></div>

<script>
let config = {};
let jobs = {};

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('tab-' + tab).classList.add('active');
  if (tab === 'jobs') loadJobs();
  if (tab === 'status') loadStatus();
}

function toast(msg, type='ok') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast toast-' + type + ' show';
  setTimeout(() => el.className = 'toast', 3000);
}

async function loadConfig() {
  const resp = await fetch('/api/config');
  config = await resp.json();
  for (const [key, val] of Object.entries(config)) {
    const el = document.getElementById(key);
    if (el) el.value = val;
  }
  document.getElementById('save-status').textContent = 'Loaded current config';
}

async function loadJobs() {
  const resp = await fetch('/api/jobs');
  jobs = await resp.json();
  const list = document.getElementById('jobs-list');
  list.innerHTML = '';
  if (!jobs.jobs || jobs.jobs.length === 0) {
    list.innerHTML = '<li><span class="job-name" style="color:var(--fg2)">No jobs configured</span></li>';
    return;
  }
  jobs.jobs.forEach((job, i) => {
    const li = document.createElement('li');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = job.enabled !== false;
    cb.onchange = () => { jobs.jobs[i].enabled = cb.checked; };
    const name = document.createElement('span');
    name.className = 'job-name';
    name.textContent = job.name;
    const sched = document.createElement('span');
    sched.className = 'job-schedule';
    sched.textContent = job.schedule || 'manual';
    li.appendChild(cb);
    li.appendChild(name);
    li.appendChild(sched);
    list.appendChild(li);
  });
}

async function loadStatus() {
  const resp = await fetch('/api/status');
  const st = await resp.json();
  let html = '';
  for (const [name, info] of Object.entries(st.services)) {
    const cls = info.running ? 'status-ok' : 'status-err';
    const txt = info.running ? 'Running' : 'Stopped';
    html += '<p><span class="status ' + cls + '">' + txt + '</span> <strong>' + name + '</strong>';
    if (info.pid) html += ' <span style="color:var(--fg2)">PID ' + info.pid + '</span>';
    html += '</p>';
  }
  document.getElementById('status-content').innerHTML = html || '<p style="color:var(--fg2)">No services detected</p>';
  let disk = '';
  if (st.disk) {
    disk = '<p><strong>Models:</strong> ' + st.disk.models + '</p>';
    disk += '<p><strong>Data:</strong> ' + st.disk.data + '</p>';
    disk += '<p><strong>Research:</strong> ' + st.disk.research + '</p>';
  }
  document.getElementById('disk-content').innerHTML = disk || '<p style="color:var(--fg2)">N/A</p>';
}

function gatherConfig() {
  const fields = document.querySelectorAll('input[id], select[id], textarea[id]');
  const cfg = {};
  fields.forEach(el => { if (el.id && (el.id.startsWith('SD_') || el.id.startsWith('LLM_') || el.id.startsWith('DISCORD_') || el.id.startsWith('HF_') || el.id.startsWith('SMART_') || el.id.startsWith('USEFUL_'))) cfg[el.id] = el.value; });
  return cfg;
}

async function saveConfig() {
  const cfg = gatherConfig();
  const resp = await fetch('/api/config', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(cfg) });
  if (resp.ok) { toast('Config saved!', 'ok'); document.getElementById('save-status').textContent = 'Saved ✓'; }
  else { toast('Save failed', 'err'); }
  // Save jobs too
  if (jobs.jobs) {
    await fetch('/api/jobs', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(jobs) });
  }
}

async function saveAndClose() {
  await saveConfig();
  toast('Shutting down...', 'ok');
  await fetch('/api/shutdown', { method: 'POST' });
  setTimeout(() => {
    document.body.innerHTML = '<div class="container" style="text-align:center;padding-top:100px"><h1>✅ Config saved</h1><p class="subtitle">Web UI has been shut down. You can close this tab.</p></div>';
  }, 500);
}

loadConfig();
</script>
</body>
</html>"""


# ─── Config Helpers ─────────────────────────────────────────────────────────
def read_env():
    """Parse .env file into a dict."""
    cfg = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, _, val = line.partition('=')
                cfg[key.strip()] = val.strip().strip('"').strip("'")
    return cfg


def write_env(updates):
    """Write updated config back to .env, preserving comments and order."""
    lines = []
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text().splitlines()

    # Build a map of existing keys to their line indices
    existing = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and '=' in stripped:
            key = stripped.split('=', 1)[0].strip()
            existing[key] = i

    # Update existing keys
    for key, val in updates.items():
        if key in existing:
            lines[existing[key]] = f'{key}="{val}"'
        else:
            lines.append(f'{key}="{val}"')

    ENV_FILE.write_text('\n'.join(lines) + '\n')


def read_jobs():
    if JOBS_FILE.exists():
        return json.loads(JOBS_FILE.read_text())
    return {"jobs": []}


def write_jobs(data):
    JOBS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def get_status():
    services = {}
    for svc in ['llama-swap', 'queue-runner']:
        try:
            res = subprocess.run(['rc-service', svc, 'status'], capture_output=True, text=True, timeout=5)
            running = res.returncode == 0
            pid = ''
            if running:
                for line in (res.stdout + res.stderr).splitlines():
                    if 'pid' in line.lower():
                        for word in line.split():
                            if word.isdigit():
                                pid = word
                                break
            services[svc] = {"running": running, "pid": pid}
        except Exception:
            services[svc] = {"running": False, "pid": ""}

    def dir_size(path):
        try:
            total = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
            if total > 1024**3:
                return f"{total / 1024**3:.1f} GB"
            return f"{total / 1024**2:.0f} MB"
        except Exception:
            return "N/A"

    return {
        "services": services,
        "disk": {
            "models": dir_size("/opt/stable-diffusion.cpp/models"),
            "data": dir_size(str(DATA_DIR)),
            "research": dir_size(str(DATA_DIR / "research")),
        }
    }


# ─── HTTP Handler ───────────────────────────────────────────────────────────
class ConfigHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default logging

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        elif self.path == '/api/config':
            self._json_response(read_env())
        elif self.path == '/api/jobs':
            self._json_response(read_jobs())
        elif self.path == '/api/status':
            self._json_response(get_status())
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length else b''

        if self.path == '/api/config':
            updates = json.loads(body)
            write_env(updates)
            self._json_response({"ok": True})
        elif self.path == '/api/jobs':
            data = json.loads(body)
            write_jobs(data)
            self._json_response({"ok": True})
        elif self.path == '/api/shutdown':
            self._json_response({"ok": True})
            # Shutdown after response is sent
            threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0)), daemon=True).start()
        else:
            self.send_error(404)

    def _json_response(self, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    bind_addr = os.environ.get('WEBUI_BIND', '0.0.0.0')
    server = ThreadedServer((bind_addr, PORT), ConfigHandler)

    # Auto-shutdown after 30 minutes of inactivity
    def auto_shutdown():
        time.sleep(1800)
        print(f"\n  Web UI auto-shutdown after 30 min timeout.")
        os._exit(0)
    threading.Thread(target=auto_shutdown, daemon=True).start()

    # Handle SIGTERM/SIGINT gracefully
    def handle_sig(sig, frame):
        print("\n  Shutting down web UI...")
        server.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGTERM, handle_sig)
    signal.signal(signal.SIGINT, handle_sig)

    # Try to detect the server's IP for the URL
    hostname = 'localhost'
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        hostname = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🦞 BC-250 Configuration Web UI                             ║
║                                                              ║
║  Open in browser:                                            ║
║    http://{hostname}:{PORT}{' ' * (42 - len(hostname) - len(str(PORT)))}║
║    http://localhost:{PORT}{' ' * (40 - len(str(PORT)))}║
║                                                              ║
║  Press Ctrl+C to shut down                                   ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Try to auto-open browser
    try:
        import webbrowser
        webbrowser.open(f'http://localhost:{PORT}')
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        print("  Web UI shut down.")


if __name__ == '__main__':
    main()
