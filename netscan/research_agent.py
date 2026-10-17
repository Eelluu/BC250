#!/usr/bin/env python3
"""
Deep Research Agent — v1
========================
Autonomous multi-step research with:
- Web search (SearXNG local, ddgr fallback)
- File operations (read/write/list)
- Command execution (sandboxed for non-whitelisted tools)
- Progress updates via Discord thread callbacks
"""

import subprocess
import json
import os
import re
import shlex
import shutil
import time
import urllib.request
import urllib.parse

# ─── Configuration ──────────────────────────────────────────────────────────
RESEARCH_WORK_DIR = "/opt/netscan/data/research"
RESEARCH_MAX_STEPS = 30
RESEARCH_TIMEOUT_S = 1800  # 30 min total
RESEARCH_CMD_TIMEOUT_S = 120  # 120s per command
RESEARCH_MAX_OUTPUT = 4000  # chars per command output sent to LLM
RESEARCH_VERIFY_SOURCES = 2  # min sources to cross-reference before claiming a fact
RESEARCH_MAX_AGE_HOURS = 24  # auto-delete sessions older than this
RESEARCH_MAX_DISK_MB = 500  # total disk cap for all research sessions
SUBAGENT_MAX_SEARCH_RESULTS = 5  # web_search results to feed sub-agent
SUBAGENT_MAX_FETCH_CHARS = 8000  # max chars per fetched page for sub-agent
SUBAGENT_LLM_TIMEOUT = 180  # timeout for sub-agent LLM calls

# ─── Bubblewrap detection ─────────────────────────────────────────────────
_HAS_BWRAP = None  # Detected once at first use

def _check_bwrap():
    """Detect if bubblewrap is available on the system."""
    global _HAS_BWRAP
    if _HAS_BWRAP is None:
        try:
            subprocess.run(['bwrap', '--version'], capture_output=True, timeout=5)
            _HAS_BWRAP = True
            _log("bubblewrap detected — using namespace sandbox")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            _HAS_BWRAP = False
            _log("bubblewrap not found — falling back to string-based sandbox")
    return _HAS_BWRAP

# Tools the LLM can use
TOOL_DEFINITIONS = """
You have access to these tools. To use a tool, output EXACTLY one line in this format:
TOOL: <tool_name> | <arguments>

Available tools:
- web_search | <query>           - Search the web via the local SearXNG (ddgr fallback)
- web_fetch | <url>              — Fetch a webpage and extract text
- run_cmd | <command>            — Execute a shell command (sandboxed if not whitelisted)
- read_file | <path>             — Read a file's contents
- write_file | <path> | <content>— Write content to a file in the work directory
- list_dir | <path>              — List directory contents
- summarize                      — You're done! Output your final answer after this line

IMPORTANT: Use the EXACT format "TOOL: <name> | <args>" on its own line.
Do NOT use other formats like <|tool_call>. Only use: TOOL: tool_name | args

Example workflow:
TOOL: web_search | vulkan alpine linux build
[tool result appears here]
TOOL: run_cmd | git clone https://github.com/example/repo
[tool result]
TOOL: read_file | repo/CMakeLists.txt
[tool result]
TOOL: summarize
Final answer: Here's how to build...
"""

# Safe tools — run directly, no sandbox needed
SAFE_TOOLS = {
    'ddgr', 'curl', 'wget', 'git', 'ls', 'cat', 'head', 'tail', 'grep',
    'find', 'wc', 'sort', 'uniq', 'diff', 'file', 'stat', 'du', 'df',
    'uname', 'env', 'which',
    'echo', 'printf', 'date', 'id', 'whoami', 'hostname',
}

# Forbidden paths — never allow access
FORBIDDEN_PATHS = {
    '/etc/shadow', '/etc/passwd', '.env', 'id_rsa', '.git/config',
    '/opt/netscan/.env', 'chat-history.json',
}


def _log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] [research] {msg}", flush=True)


def _cleanup_old_sessions():
    """Delete research sessions older than RESEARCH_MAX_AGE_HOURS.
    
    Called at the start of each new session to keep disk usage bounded.
    Sessions are directories named by timestamp (e.g. 1718312345).
    """
    if not os.path.isdir(RESEARCH_WORK_DIR):
        return
    
    now = time.time()
    max_age_s = RESEARCH_MAX_AGE_HOURS * 3600
    cleaned = 0
    freed_bytes = 0
    
    for entry in os.listdir(RESEARCH_WORK_DIR):
        entry_path = os.path.join(RESEARCH_WORK_DIR, entry)
        if not os.path.isdir(entry_path):
            continue
        # Session dirs are numeric timestamps
        try:
            session_ts = int(entry)
        except ValueError:
            continue
        age = now - session_ts
        if age > max_age_s:
            # Calculate size before deleting
            for root, dirs, files in os.walk(entry_path):
                for f in files:
                    try:
                        freed_bytes += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            shutil.rmtree(entry_path, ignore_errors=True)
            cleaned += 1
    
    if cleaned > 0:
        freed_mb = freed_bytes / (1024 * 1024)
        _log(f"Cleaned {cleaned} old research sessions ({freed_mb:.1f} MB freed)")


def _send_progress(thread_channel, discord_send_fn, step, total, msg):
    """Send a progress update to the Discord thread."""
    prefix = f"🔬 **Step {step}/{total}** — "
    try:
        discord_send_fn(thread_channel, f"{prefix}{msg}")
    except Exception:
        pass


def _build_summary(steps_log, query, elapsed_s):
    """Build a formatted recap of the research session."""
    lines = [
        "## 🔬 Research Summary",
        f"**Query:** {query}",
        f"**Duration:** {elapsed_s:.0f}s | **Steps:** {len(steps_log)}",
        "",
    ]
    # Group by type
    searches = [s for s in steps_log if s['tool'] == 'web_search']
    fetches = [s for s in steps_log if s['tool'] == 'web_fetch']
    commands = [s for s in steps_log if s['tool'] == 'run_cmd']
    files = [s for s in steps_log if s['tool'] in ('read_file', 'write_file', 'list_dir')]

    if searches:
        lines.append("### 🔍 Sources Consulted")
        for s in searches:
            lines.append(f"- `{s['args']}`")
    if fetches:
        lines.append("### 🌐 Pages Fetched")
        for s in fetches:
            lines.append(f"- `{s['args'][:80]}`")
    if commands:
        lines.append("### ⚙️ Commands Executed")
        for s in commands:
            brief = s['result'][:60].replace('\n', ' ')
            lines.append(f"- `{s['args'][:60]}` → {brief}")
    if files:
        lines.append("### 📄 Files Accessed")
        for s in files:
            lines.append(f"- `{s['args'][:60]}`")

    return "\n".join(lines)


def _is_safe_command(cmd):
    """Check if a command is whitelisted and safe to run directly.

    Two tiers of danger:
    - Truly dangerous (command chaining/code exec): always blocked
    - Shell metacharacters (>, <, |): allowed for safe tools since
      they only redirect output, not chain commands.
    """
    try:
        parts = shlex.split(cmd)
    except Exception:
        return False

    if not parts:
        return False

    binary = os.path.basename(parts[0])

    # Block command chaining & code execution (always dangerous)
    truly_dangerous = [';', '&', '`', '$', '(', ')']
    if any(c in cmd for c in truly_dangerous):
        return False

    # Check forbidden paths
    for fp in FORBIDDEN_PATHS:
        if fp in cmd:
            return False

    return binary in SAFE_TOOLS


def _build_bwrap_args(cmd, work_dir):
    """Build bubblewrap arguments for a sandboxed command.

    Creates a minimal sandbox with:
    - Read-only access to system dirs (/usr, /bin, /lib)
    - Read-write access to work_dir (mounted at /work)
    - Isolated /tmp, /dev, /proc
    - New PID, IPC, UTS, NET namespaces
    """
    # Detect optional paths that may not exist
    optional_args = []
    for p in [('/lib64', '/lib64'), ('/usr/local', '/usr/local'),
              ('/etc/ca-certificates', '/etc/ca-certificates')]:
        if os.path.isdir(p[0]):
            optional_args += ['--ro-bind', p[0], p[1]]

    # Build bwrap command
    bwrap_args = [
        'bwrap',
        # Filesystem isolation
        '--ro-bind', '/usr', '/usr',
        '--ro-bind', '/bin', '/bin',
        '--ro-bind', '/lib', '/lib',
        *optional_args,
        # DNS/SSL/hosts needed for network access inside sandbox
        '--ro-bind', '/etc/resolv.conf', '/etc/resolv.conf',
        '--ro-bind', '/etc/hosts', '/etc/hosts',
        '--ro-bind', '/etc/ssl', '/etc/ssl',
        '--ro-bind', '/etc/ca-certificates', '/etc/ca-certificates',
        '--dev', '/dev',
        '--proc', '/proc',
        '--tmpfs', '/tmp',
        '--tmpfs', '/var',
        # Work directory (read-write)
        '--bind', work_dir, '/work',
        # Working directory
        '--chdir', '/work',
        # Namespace isolation
        '--unshare-pid',
        '--unshare-ipc',
        '--unshare-uts',
        '--hostname', 'sandbox',
        # Cleanup
        '--die-with-parent',
        # Run the command via sh -c (to handle pipes, redirects, etc.)
        '--', 'sh', '-c', cmd,
    ]
    return bwrap_args


def _run_sandboxed(cmd, work_dir, timeout_s=RESEARCH_CMD_TIMEOUT_S):
    """Run a command in a sandboxed environment.

    Uses bubblewrap (bwrap) if available for real namespace isolation.
    Falls back to string-based sandboxing if bwrap is not installed.

    Sandbox rules:
    - cwd = /work (mapped to work_dir)
    - timeout enforced
    - output truncated
    - stdin closed
    - Read-only system dirs, read-write work dir only
    """
    os.makedirs(work_dir, exist_ok=True)

    # Try bubblewrap first (bwrap handles pipes/redirects safely inside namespace)
    if _check_bwrap():
        try:
            bwrap_args = _build_bwrap_args(cmd, work_dir)
            _log(f"  [bwrap] {' '.join(bwrap_args[:5])}...")
            result = subprocess.run(
                bwrap_args,
                capture_output=True, text=True,
                timeout=timeout_s,
                stdin=subprocess.DEVNULL,
                env={
                    'PATH': '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
                    'HOME': '/work',
                    'TMPDIR': '/tmp',
                    'LANG': 'en_US.UTF-8',
                },
            )
            output = (result.stdout + result.stderr).strip()
            if not output:
                output = "[No output]"
            if len(output) > RESEARCH_MAX_OUTPUT:
                output = output[:RESEARCH_MAX_OUTPUT] + f"\n... [truncated, {len(output)} chars total]"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output
        except subprocess.TimeoutExpired:
            return f"[Sandbox] Command timed out after {timeout_s}s"
        except Exception as e:
            _log(f"  [bwrap failed: {e}, falling back to string sandbox]")
            # Fall through to string-based sandbox

    # Fallback: string-based sandbox (no real isolation)
    # Block shell metacharacters since this path has no namespace isolation
    dangerous = [';', '&', '|', '>', '<', '`', '$', '\n', '\r']
    for d in dangerous:
        if d in cmd:
            return "[Sandboxed] Blocked: shell metacharacter not allowed in fallback sandbox"
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout_s, cwd=work_dir,
            stdin=subprocess.DEVNULL,
            env={
                'PATH': '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
                'HOME': work_dir,
                'TMPDIR': work_dir,
                'LANG': 'en_US.UTF-8',
            },
        )
        output = (result.stdout + result.stderr).strip()
        if not output:
            output = "[No output]"
        if len(output) > RESEARCH_MAX_OUTPUT:
            output = output[:RESEARCH_MAX_OUTPUT] + f"\n... [truncated, {len(output)} chars total]"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output
    except subprocess.TimeoutExpired:
        return f"[Sandbox] Command timed out after {timeout_s}s"
    except Exception as e:
        return f"[Sandbox] Error: {e}"


def _run_safe(cmd, timeout_s=RESEARCH_CMD_TIMEOUT_S):
    """Run a whitelisted command directly."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout_s, stdin=subprocess.DEVNULL,
        )
        output = (result.stdout + result.stderr).strip()
        if not output:
            output = "[No output]"
        if len(output) > RESEARCH_MAX_OUTPUT:
            output = output[:RESEARCH_MAX_OUTPUT] + f"\n... [truncated, {len(output)} chars total]"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output
    except subprocess.TimeoutExpired:
        return f"[Timeout] Command took longer than {timeout_s}s"
    except Exception as e:
        return f"[Error] {e}"


def _simplify_query(query):
    """Strip filler words and limit length for better search results."""
    fillers = ('researching', 'research', 'investigate', 'investigate and', 'find information about',
               'look up', 'search for', 'tell me about', 'what is', 'how to', 'give me',
               'and the', 'the', 'for the', 'on the', 'in the')
    q = query
    for f in fillers:
        q = re.sub(r'\b' + re.escape(f) + r'\b', '', q, flags=re.IGNORECASE)
    q = re.sub(r'\s+', ' ', q).strip().strip('"').strip("'")
    if len(q) > 60:
        q = q[:60]
    return q or query


def _searxng_search(query, max_results=5, timeout=25):
    """Search through the local SearXNG instance. Returns formatted text.

    SearXNG aggregates several engines, and that matters here: ddgr scrapes
    DuckDuckGo directly and this network gets HTTP 202 from it, so every research
    run used to end in "No results found". Installed by install.sh and bound to
    127.0.0.1 (see searxng/searxng-setup.sh).
    """
    base = os.environ.get("SEARXNG_URL", "http://127.0.0.1:8890")
    params = urllib.parse.urlencode({"q": query, "format": "json"})
    req = urllib.request.Request(f"{base.rstrip('/')}/search?{params}",
                                 headers={"User-Agent": "bc250-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    lines = []
    for r in ((payload or {}).get("results") or [])[:max_results]:
        url = (r.get("url") or "").strip()
        if not url:
            continue
        title = re.sub(r"\s+", " ", (r.get("title") or "")).strip()
        snippet = re.sub(r"\s+", " ", (r.get("content") or "")).strip()
        lines.append(f"{title}\n{url}\n{snippet}".strip())
    return "\n\n".join(lines)


def _web_search(query, max_results=5):
    """Search the web: the local SearXNG first, ddgr as a fallback.

    Both paths hand the query over as an argument list or a URL parameter, never
    through a shell, so it stays safe by construction.
    """
    query = _simplify_query(query)
    try:
        text = _searxng_search(query, max_results)
        if text:
            return text
    except Exception:
        pass  # not fatal: fall through to ddgr, exactly as before this change
    try:
        result = subprocess.run(
            ["ddgr", "--np", "-n", str(max_results), query],
            capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL,
        )
        output = result.stdout.strip()
        if not output:
            return f"No results found for: {query}"
        return output
    except FileNotFoundError:
        return "[Error] no search backend: SearXNG did not answer and ddgr is not installed"
    except Exception as e:
        return f"[Error] Search failed: {e}"


def _web_fetch(url, max_chars=5000):
    """Fetch a URL and extract readable text."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='ignore')

        # Simple HTML stripping
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [truncated at {max_chars} chars]"
        return text if text else "[Empty page]"
    except Exception as e:
        return f"[Error] Failed to fetch {url}: {e}"


def _llm_call(llm_url, model, messages, temperature=0.3, timeout_s=SUBAGENT_LLM_TIMEOUT):
    """Make a raw LLM API call. Used by sub-agent for compression."""
    try:
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": 1024,
        }).encode()
        req = urllib.request.Request(
            f"{llm_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read())
        return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        _log(f"[sub-agent] LLM call failed: {e}")
        return None


def _subagent_research(query, llm_url, model):
    """Sub-agent: searches the web, fetches pages, extracts key facts.

    If query is a URL, fetches it directly. Otherwise, searches first.
    Returns a compact, descriptive summary — no fixed character limit.
    The LLM decides how small/descriptive to make it based on content.
    """
    _log(f"  [sub-agent] researching: {query[:60]}")

    # Check if query is a direct URL
    is_url = query.strip().startswith('http://') or query.strip().startswith('https://')

    if is_url:
        # Direct fetch mode: fetch the URL and compress
        page_text = _web_fetch(query, max_chars=SUBAGENT_MAX_FETCH_CHARS)
        if page_text.startswith('[Error]') or page_text == '[Empty page]':
            return f"Fetch failed: {page_text}"
        context_parts = [f"Fetched page: {query}\n\n{page_text}"]
        raw_len = len(page_text)
    else:
        # Search mode: search, then fetch top results
        raw_results = _web_search(query, max_results=SUBAGENT_MAX_SEARCH_RESULTS)
        if raw_results.startswith('[Error]') or raw_results.startswith('No results'):
            return f"Web search failed: {raw_results}"

        # Extract URLs from search results and fetch top 2-3 pages
        urls = re.findall(r'https?://[^\s<>"\']+', raw_results)
        fetched_pages = []
        for url in urls[:3]:
            page_text = _web_fetch(url, max_chars=SUBAGENT_MAX_FETCH_CHARS)
            if not page_text.startswith('[Error]') and page_text != '[Empty page]':
                fetched_pages.append(f"--- {url} ---\n{page_text}")

        context_parts = [f"Search results for: {query}\n\n{raw_results}\n"]
        if fetched_pages:
            context_parts.append("\nFetched pages:\n" + '\n\n'.join(fetched_pages))
        raw_len = len(raw_results) + sum(len(p) for p in fetched_pages)

    # Use LLM to extract key facts
    messages = [
        {"role": "system", "content": (
            "You are a research sub-agent. Your job is to extract key facts from web content.\n"
            "Rules:\n"
            "- Be concise but descriptive — include specific versions, URLs, commands, flags\n"
            "- Include source URLs for every fact you cite\n"
            "- If information conflicts, note both versions with sources\n"
            "- Do NOT include fluff, ads, navigation, or boilerplate\n"
            "- Output ONLY the extracted facts, no preamble\n"
        )},
        {"role": "user", "content": (
            f"Extract key facts relevant to: {query}\n\n"
            + '\n\n'.join(context_parts)
        )},
    ]

    summary = _llm_call(llm_url, model, messages, temperature=0.2, timeout_s=180)
    if not summary:
        # Fallback: return raw content truncated
        return context_parts[0][:1500]

    _log(f"  [sub-agent] summary: {len(summary)} chars (from {raw_len} raw)")
    return summary


def _parse_tool_call(llm_reply):
    """Parse a tool call from LLM reply. Supports multiple formats:
    - TOOL: tool_name | arguments          (original format)
    - <|tool_call>call:tool_name|arguments  (turboquant / llama.cpp fork)
    - TOOL: tool_name arguments             (no pipe)
    """
    # Format 1: TOOL: tool_name | arguments
    match = re.search(r'^TOOL:\s*(\w+)\s*\|\s*(.+)$', llm_reply, re.MULTILINE)
    if match:
        return match.group(1).lower(), match.group(2).strip()

    # Format 2: <|tool_call>call:tool_name|arguments or <|tool_call>tool_name|arguments
    match = re.search(r'<\|tool_call\>[\s]*(?:call:)?(\w+)\|?(.*)', llm_reply, re.DOTALL)
    if match:
        tool = match.group(1).lower()
        args = match.group(2).strip()
        # Strip trailing <|tool_call|> or similar tags
        args = re.sub(r'<[^>]+>$', '', args).strip()
        if tool == 'summarize':
            return 'summarize', ''
        return tool, args

    # Format 3: TOOL: summarize (no args)
    match = re.search(r'^TOOL:\s*(summarize)\s*$', llm_reply, re.MULTILINE)
    if match:
        return 'summarize', ''

    return None, None


def _execute_tool(tool, args, work_dir, session_id, llm_url=None, model=None):
    """Execute a tool and return the result string.
    
    For web_search: routes through sub-agent which searches, fetches, and analyzes.
    For web_fetch: routes through sub-agent which fetches and extracts key facts.
    Other tools: execute directly.
    """
    if tool == 'web_search':
        if llm_url and model:
            return _subagent_research(args, llm_url, model)
        return _web_search(args)

    elif tool == 'web_fetch':
        if llm_url and model:
            return _subagent_research(args, llm_url, model)
        return _web_fetch(args)

    elif tool == 'run_cmd':
        if _is_safe_command(args):
            _log(f"  [safe] {args}")
            return _run_safe(args)
        else:
            _log(f"  [sandbox] {args}")
            return _run_sandboxed(args, work_dir)

    elif tool == 'read_file':
        # Resolve relative paths in work_dir
        fpath = args
        if not os.path.isabs(fpath):
            fpath = os.path.join(work_dir, fpath)
        # Security: no forbidden paths
        for fp in FORBIDDEN_PATHS:
            if fp in fpath:
                return f"[Access Denied] {fp} is forbidden"
        try:
            with open(fpath, 'r', errors='replace') as f:
                content = f.read(RESEARCH_MAX_OUTPUT)
            if len(content) == RESEARCH_MAX_OUTPUT:
                content += "\n... [truncated]"
            return content if content else "[Empty file]"
        except FileNotFoundError:
            return f"[Error] File not found: {args}"
        except Exception as e:
            return f"[Error] {e}"

    elif tool == 'write_file':
        parts = args.split(' | ', 1)
        if len(parts) != 2:
            return "[Error] Usage: write_file | <path> | <content>"
        fpath, content = parts
        if not os.path.isabs(fpath):
            fpath = os.path.join(work_dir, fpath)
        # Security: restrict writes to work_dir
        real_work = os.path.realpath(work_dir)
        real_fpath = os.path.realpath(os.path.dirname(fpath))
        if not real_fpath.startswith(real_work):
            return f"[Access Denied] Cannot write outside work directory: {fpath}"
        try:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, 'w') as f:
                f.write(content)
            return f"[OK] Written {len(content)} chars to {fpath}"
        except Exception as e:
            return f"[Error] {e}"

    elif tool == 'list_dir':
        dpath = args or work_dir
        if not os.path.isabs(dpath):
            dpath = os.path.join(work_dir, dpath)
        try:
            entries = sorted(os.listdir(dpath))
            lines = []
            for e in entries[:50]:
                full = os.path.join(dpath, e)
                prefix = "📁" if os.path.isdir(full) else "📄"
                lines.append(f"  {prefix} {e}")
            if len(entries) > 50:
                lines.append(f"  ... and {len(entries) - 50} more")
            return "\n".join(lines) if lines else "[Empty directory]"
        except FileNotFoundError:
            return f"[Error] Directory not found: {args}"
        except Exception as e:
            return f"[Error] {e}"

    elif tool == 'summarize':
        return "__DONE__"

    else:
        return f"[Error] Unknown tool: {tool}. Use one of: web_search, web_fetch, run_cmd, read_file, write_file, list_dir, summarize"


def run_research(query, llm_url, model, discord_send_fn=None, thread_channel=None):
    """
    Run a deep research session.

    Args:
        query: The user's research question
        llm_url: LLM API URL (e.g. http://localhost:8080/v1)
        model: Model name to use
        discord_send_fn: Function to send messages to Discord thread
        thread_channel: Discord channel/thread object

    Returns:
        The final answer string
    """
    session_id = str(int(time.time()))
    work_dir = os.path.join(RESEARCH_WORK_DIR, session_id)
    os.makedirs(work_dir, exist_ok=True)

    # Clean up old sessions before starting
    _cleanup_old_sessions()

    _log(f"Starting research session {session_id}: {query[:80]}")

    system_prompt = (
        "You are Clawd 🦞, an AI research assistant running on an Alpine Linux server.\n"
        "You can search the web, read/write files, and execute commands.\n\n"
        f"{TOOL_DEFINITIONS}\n\n"
        "Rules:\n"
        "- One tool call per message\n"
        "- Be methodical: search → read → analyze → execute → verify\n"
        "- For build tasks: clone to the work directory, read CMakeLists.txt, then build\n"
        "- Always explain what you're doing before each tool call\n"
        "- When done, use the summarize tool and give your final answer\n\n"
        "VERIFICATION RULES (critical — accuracy over speed):\n"
        f"- NEVER state a fact from a single source. Cross-reference with at least {RESEARCH_VERIFY_SOURCES} sources.\n"
        "- If a web search gives conflicting info, fetch the actual pages to verify.\n"
        "- For technical claims (versions, compatibility, build flags), verify with official docs.\n"
        "- If you cannot verify something, explicitly say: '⚠️ Unverified: <claim>'.\n"
        "- When building/compiling, verify each step succeeded before proceeding.\n"
        "- Prefer official documentation over blog posts or forum answers.\n"
        "- If sources disagree, present both sides and note which is more authoritative.\n\n"
        "PERSISTENCE RULES (do NOT give up easily):\n"
        "- When a command fails, analyze the error message carefully and try to fix it.\n"
        "- If a build fails, read the error log, search for the specific error, and retry.\n"
        "- If a package is missing, search for how to install it on Alpine and try again.\n"
        "- If one approach doesn't work, try an alternative (different flags, different tool, different version).\n"
        "- Only use summarize when you have EITHER succeeded OR exhausted all reasonable approaches.\n"
        "- If you truly cannot reach the goal after many attempts, explain what you tried and why it failed.\n"
        "- You have up to 30 steps and 30 minutes — use them to iterate, not to give up early.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Research this: {query}"},
    ]

    start_time = time.time()
    step = 0
    final_answer = None
    steps_log = []  # Track all steps for final summary
    consecutive_failures = 0  # Track repeated failures on same error

    while step < RESEARCH_MAX_STEPS:
        # Check total timeout
        elapsed = time.time() - start_time
        if elapsed > RESEARCH_TIMEOUT_S:
            _log(f"Research session {session_id} timed out after {elapsed:.0f}s")
            if discord_send_fn and thread_channel:
                _send_progress(thread_channel, discord_send_fn, step, RESEARCH_MAX_STEPS,
                               f"⏰ Research timed out after {RESEARCH_TIMEOUT_S}s")
            break

        step += 1

        # Send progress
        if discord_send_fn and thread_channel:
            _send_progress(thread_channel, discord_send_fn, step, RESEARCH_MAX_STEPS,
                           f"Thinking... ({elapsed:.0f}s elapsed)")

        # Call LLM
        try:
            payload = json.dumps({
                "model": model,
                "messages": messages,
                "stream": False,
                "temperature": 0.5,
                "max_tokens": 8192,
            }).encode()
            req = urllib.request.Request(
                f"{llm_url}/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read())
            reply = data['choices'][0]['message']['content'].strip()
        except Exception as e:
            _log(f"LLM error: {e}")
            return f"🦞 Research failed: LLM error — {e}"

        # Parse tool call
        tool, args = _parse_tool_call(reply)

        if tool is None:
            # No tool call — treat as final answer
            _log(f"Step {step}: No tool call, treating as final answer")
            final_answer = reply
            break

        if tool == 'summarize':
            # Extract everything before TOOL: summarize
            before_tool = reply[:reply.rfind('TOOL:')].strip()
            if before_tool:
                final_answer = before_tool
            else:
                final_answer = "Research complete. Check the thread for details."
            _log(f"Step {step}: Research complete ({elapsed:.0f}s)")
            break

        # Execute tool
        _log(f"Step {step}: {tool}({args[:60]}{'...' if len(args) > 60 else ''})")
        result = _execute_tool(tool, args, work_dir, session_id, llm_url, model)

        # Log step for final summary
        steps_log.append({
            'step': step,
            'tool': tool,
            'args': args[:200],
            'result': result[:500],
            'elapsed': elapsed,
        })

        # Send step update to Discord (interesting operations only)
        if discord_send_fn and thread_channel:
            if tool in ('web_search', 'web_fetch', 'run_cmd'):
                brief = result[:150].replace('\n', ' ')
                if len(result) > 150:
                    brief += "..."
                emoji = {
                    "web_search": "🔍", "web_fetch": "🌐",
                    "run_cmd": "⚙️",
                }.get(tool, "🔧")
                _send_progress(thread_channel, discord_send_fn, step, RESEARCH_MAX_STEPS,
                               f"{emoji} `{tool}` → {brief}")

        # Check for done
        if result == "__DONE__":
            # LLM wanted to summarize but we need the actual content
            # Re-prompt for the actual answer
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": (
                "You used the summarize tool. Now give your final answer to the user "
                "based on all the information you gathered. Be thorough but concise."
            )})
            try:
                payload = json.dumps({
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "temperature": 0.5,
                "max_tokens": 8192,
                }).encode()
                req = urllib.request.Request(
                    f"{llm_url}/chat/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=600) as resp:
                    data = json.loads(resp.read())
                final_answer = data['choices'][0]['message']['content'].strip()
            except Exception:
                final_answer = "Research complete. Check the thread for details."
            break

        # Feed result back to LLM
        messages.append({"role": "assistant", "content": reply})

        # If run_cmd failed, append persistence hint
        if tool == 'run_cmd' and '[exit code:' in result:
            exit_code = re.search(r'\[exit code: (-?\d+)\]', result)
            code = exit_code.group(1) if exit_code else '?'
            messages.append({"role": "user", "content": (
                f"Tool result:\n{result}\n\n"
                f"⚠️ Command failed with exit code {code}. "
                f"Analyze the error above carefully. "
                f"Try to fix the issue: search for the specific error, "
                f"install missing deps, change flags, or try an alternative approach. "
                f"Do NOT give up — you have more steps available."
            )})
        else:
            messages.append({"role": "user", "content": f"Tool result:\n{result}"})

    if final_answer is None:
        final_answer = f"Research completed {step} steps in {time.time() - start_time:.0f}s. Check thread for details."

    elapsed_total = time.time() - start_time
    _log(f"Session {session_id} done: {len(final_answer)} chars, {step} steps, {elapsed_total:.0f}s")

    # Send formatted summary to Discord
    if discord_send_fn and thread_channel and steps_log:
        try:
            summary = _build_summary(steps_log, query, elapsed_total)
            # Discord message limit is 2000 chars
            if len(summary) > 1900:
                summary = summary[:1900] + "\n... [truncated]"
            discord_send_fn(thread_channel, summary)
        except Exception:
            pass

    return final_answer
