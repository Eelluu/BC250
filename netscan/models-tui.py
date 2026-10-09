#!/usr/bin/env python3
"""
BC-250 Model Manager — Interactive TUI for managing llama-swap models.
Usage: python3 models-tui.py [config.yaml path]
"""

import os
import re
import shlex
import sys
import termios
import tty

# ─── ANSI helpers ────────────────────────────────────────────────────────────

CLEAR = "\033[2J\033[H"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
WHITE = "\033[97m"
BG_BLUE = "\033[44m"

# ─── YAML helpers (minimal — no PyYAML dependency) ──────────────────────────

def parse_config(path):
    """Parse llama-swap config.yaml into a dict of {alias: {cmd: ..., ttl: ...}}.
    Returns (models_dict, raw_header_comment)."""
    models = {}
    header_lines = []
    if not os.path.exists(path):
        return models, ""

    with open(path) as f:
        lines = f.readlines()

    current_alias = None
    current_cmd = None
    current_ttl = 600
    in_models = False

    for line in lines:
        stripped = line.strip()

        # Top-level "models:" entry
        if re.match(r'^models:\s*$', line.rstrip()):
            in_models = True
            continue

        if not in_models:
            header_lines.append(line)
            continue

        # Model alias line: "  alias-name:"
        m = re.match(r'^  ([a-zA-Z0-9._-]+):\s*$', line.rstrip())
        if m:
            # Save previous model
            if current_alias:
                models[current_alias] = {"cmd": current_cmd or "", "ttl": current_ttl}
            current_alias = m.group(1)
            current_cmd = None
            current_ttl = 600
            continue

        # cmd line
        m = re.match(r'^\s+cmd:\s*(.+)$', line.rstrip())
        if m and current_alias:
            current_cmd = m.group(1).strip()
            continue

        # ttl line
        m = re.match(r'^\s+ttl:\s*(\d+)', line.rstrip())
        if m and current_alias:
            current_ttl = int(m.group(1))
            continue

        # comment lines within models block — skip
        if stripped.startswith('#') or not stripped:
            continue

    # Save last model
    if current_alias:
        models[current_alias] = {"cmd": current_cmd or "", "ttl": current_ttl}

    return models, "".join(header_lines)


def write_config(path, models, header=""):
    """Write models dict back to config.yaml."""
    with open(path, "w") as f:
        if header and header.strip():
            f.write(header.rstrip() + "\n")
        f.write("models:\n")
        for alias, cfg in models.items():
            f.write(f"  {alias}:\n")
            f.write(f"    cmd: {cfg['cmd']}\n")
            f.write(f"    ttl: {cfg.get('ttl', 600)}\n")
            f.write("\n")


# ─── Command parsing helpers ────────────────────────────────────────────────

def parse_cmd(cmd):
    """Extract key flags from a llama-server command string."""
    result = {
        "model": "",
        "ngl": "99",
        "ctx": "65536",
        "kv_type": "q4_0",
        "flash_attn": False,
        "mmproj": "",
        "mtp_enabled": False,
        "mtp_model": "",
        "mtp_n": "2",
        "port": "${PORT}",
        "host": "127.0.0.1",
        "extra": [],
    }

    if not cmd:
        return result

    try:
        parts = shlex.split(cmd)
    except ValueError:
        parts = cmd.split()

    i = 0

    while i < len(parts):
        p = parts[i]

        if p == "--model" or p == "-m":
            if i + 1 < len(parts):
                result["model"] = parts[i + 1]
                i += 2
                continue
        elif p == "-ngl":
            if i + 1 < len(parts):
                result["ngl"] = parts[i + 1]
                i += 2
                continue
        elif p == "-c":
            if i + 1 < len(parts):
                result["ctx"] = parts[i + 1]
                i += 2
                continue
        elif p == "-ctk":
            if i + 1 < len(parts):
                result["kv_type"] = parts[i + 1]
                i += 2
                continue
        elif p == "-ctv":
            # same as ctk, skip
            if i + 1 < len(parts):
                i += 2
                continue
        elif p == "--flash-attn" or p.startswith("--flash-attn="):
            # Modern llama.cpp requires a value: --flash-attn on|off|auto.
            # Bare --flash-attn (older builds) is still accepted as enabled.
            if p.startswith("--flash-attn="):
                result["flash_attn"] = p.split("=", 1)[1].strip() != "off"
                i += 1
                continue
            if i + 1 < len(parts) and parts[i + 1] in ("on", "off", "auto"):
                result["flash_attn"] = parts[i + 1] != "off"
                i += 2
            else:
                result["flash_attn"] = True
                i += 1
            continue
        elif p == "--mmproj":
            if i + 1 < len(parts):
                result["mmproj"] = parts[i + 1]
                i += 2
                continue
        elif p == "--spec-type":
            if i + 1 < len(parts) and parts[i + 1] == "draft-mtp":
                result["mtp_enabled"] = True
            i += 2
            continue
        elif p == "--spec-draft-model":
            if i + 1 < len(parts):
                result["mtp_model"] = parts[i + 1]
            i += 2
            continue
        elif p == "--spec-draft-n-max":
            if i + 1 < len(parts):
                result["mtp_n"] = parts[i + 1]
            i += 2
            continue
        elif p == "--host":
            if i + 1 < len(parts):
                result["host"] = parts[i + 1]
            i += 2
            continue
        elif p == "--port":
            if i + 1 < len(parts):
                result["port"] = parts[i + 1]
            i += 2
            continue
        elif p.startswith("-"):
            # Unknown flag — might have a value
            if i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                result["extra"].append(f"{p} {parts[i + 1]}")
                i += 2
            else:
                result["extra"].append(p)
                i += 1
            continue
        else:
            # Positional arg (e.g. the server binary path)
            i += 1
            continue

    return result


def build_cmd(parsed, server_bin="/opt/llama.cpp/build/bin/llama-server"):
    """Rebuild a llama-server command string from parsed fields."""
    parts = [
        server_bin,
        "--port", parsed["port"],
        "--model", parsed["model"],
        "-ngl", parsed["ngl"],
        "-c", parsed["ctx"],
        "-ctk", parsed["kv_type"],
        "-ctv", parsed["kv_type"],
    ]

    if parsed["flash_attn"]:
        # Value required by modern llama.cpp (bare --flash-attn makes the parser
        # swallow the next flag as its value and llama-server exits).
        parts.extend(["--flash-attn", "on"])

    if parsed["mmproj"]:
        parts.extend(["--mmproj", parsed["mmproj"]])

    if parsed["mtp_enabled"]:
        parts.extend(["--spec-type", "draft-mtp"])
        if parsed["mtp_model"]:
            parts.extend(["--spec-draft-model", parsed["mtp_model"]])
        parts.extend(["--spec-draft-n-max", parsed["mtp_n"]])

    parts.extend(["--host", parsed["host"]])

    for extra in parsed.get("extra", []):
        parts.append(extra)

    return " ".join(parts)


# ─── Terminal helpers ────────────────────────────────────────────────────────

def get_key():
    """Read a single keypress from the terminal."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                if ch3 == "A":
                    return "up"
                if ch3 == "B":
                    return "down"
                if ch3 == "C":
                    return "right"
                if ch3 == "D":
                    return "left"
            return "escape"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == " ":
            return "space"
        if ch == "\x03":
            return "ctrl-c"
        return ch.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def input_line(prompt, default=""):
    """Prompt for a line of text input.
    Returns the string, default on empty input, or None on Esc/cancel."""
    sys.stdout.write(f"  {BOLD}{prompt}{RESET} [{DIM}{default}{RESET}]: ")
    sys.stdout.flush()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        new = termios.tcgetattr(fd)
        new[3] &= ~termios.ECHO
        new[6][termios.VMIN] = 1
        new[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, new)
        buf = ""
        while True:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                sys.stdout.write("\n")
                return None
            if ch in ("\r", "\n"):
                sys.stdout.write("\n")
                line = buf.strip()
                return line if line else default
            if ch in ("\x7f", "\x08"):
                if buf:
                    buf = buf[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            buf += ch
            sys.stdout.write(ch)
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def input_yesno(prompt, default=True):
    """Prompt for y/n. Returns True/False or None on Esc."""
    hint = "Y/n" if default else "y/N"
    result = input_line(f"{prompt} [{hint}]", "")
    if result is None:
        return None
    if not result:
        return default
    return result.lower() in ("y", "yes")


# ─── Model Manager TUI ──────────────────────────────────────────────────────

SERVER_BIN = "/opt/llama.cpp/build/bin/llama-server"


class ModelsManager:
    def __init__(self, config_file):
        self.config_file = config_file
        self.models = {}
        self.header = ""
        self.selected = 0
        self.modified = False
        self.page = "main"  # main | edit | new
        self.load()

    def load(self):
        self.models, self.header = parse_config(self.config_file)

    def save(self):
        try:
            write_config(self.config_file, self.models, self.header)
            self.modified = False
            return True
        except Exception as e:
            print(f"  {RED}Error saving: {e}{RESET}")
            return False

    def aliases(self):
        return list(self.models.keys())

    def draw_header(self, title):
        print(CLEAR)
        mod = f"  {YELLOW}● Unsaved{RESET}" if self.modified else ""
        print(f"  {BOLD}{BLUE}{title}{RESET}{mod}")
        print(f"  {DIM}{'─' * 68}{RESET}")

    # ── Main screen ──────────────────────────────────────────────────────

    def draw_main(self):
        self.draw_header("🦙 LLaMA-Swap Model Manager")
        print()

        aliases = self.aliases()

        if not aliases:
            print(f"  {DIM}No models configured in {self.config_file}{RESET}")
            print(f"  {DIM}Press [N] to add a model.{RESET}")
        else:
            # Table header
            print(f"  {BOLD}  {'Alias':<18} {'Model':<22} {'ngl':<5} {'ctx':<7} {'FA':<4} {'mmproj':<5} {'MTP':<5}{RESET}")
            print(f"  {DIM}  {'─'*18} {'─'*22} {'─'*5} {'─'*7} {'─'*4} {'─'*5} {'─'*5}{RESET}")

            for i, alias in enumerate(aliases):
                cfg = self.models[alias]
                parsed = parse_cmd(cfg.get("cmd", ""))

                model_name = os.path.basename(parsed["model"]) if parsed["model"] else "—"
                if len(model_name) > 20:
                    model_name = "…" + model_name[-19:]

                ngl = parsed["ngl"]
                ctx = parsed["ctx"]
                fa = f"{GREEN}✓{RESET}" if parsed["flash_attn"] else f"{DIM}·{RESET}"
                mmproj = f"{GREEN}✓{RESET}" if parsed["mmproj"] else f"{DIM}·{RESET}"
                mtp = f"{GREEN}✓{RESET}" if parsed["mtp_enabled"] else f"{DIM}·{RESET}"

                if i == self.selected:
                    line = f"  {alias:<18} {model_name:<22} {ngl:<5} {ctx:<7}  {fa}   {mmproj}   {mtp}"
                    print(f"{BG_BLUE}{WHITE}{line}{RESET}")
                else:
                    print(f"  {GREEN}{alias:<18}{RESET} {DIM}{model_name:<22} {ngl:<5} {ctx:<7}{RESET}  {fa}   {mmproj}   {mtp}")

        print()
        print(f"  {DIM}{'─' * 68}{RESET}")
        if aliases:
            print(f"  {BOLD}[↑↓]{RESET} Move  {BOLD}[Enter/E]{RESET} Edit  {BOLD}[N]{RESET} New"
                  f"  {BOLD}[D]{RESET} Delete  {BOLD}[S]{RESET} Save  {BOLD}[Q]{RESET} Quit")
        else:
            print(f"  {BOLD}[N]{RESET} New model  {BOLD}[Q]{RESET} Quit")
        print()

    # ── Edit model screen ────────────────────────────────────────────────

    def edit_model(self):
        aliases = self.aliases()
        if not aliases or self.selected >= len(aliases):
            return

        alias = aliases[self.selected]
        cfg = self.models[alias]

        while True:
            parsed = parse_cmd(cfg.get("cmd", ""))

            self.draw_header(f"⚙  Edit Model: {BOLD}{alias}{RESET}")
            print()
            print(f"    {BOLD}[1]{RESET} Alias:            {CYAN}{alias}{RESET}")
            print(f"    {BOLD}[2]{RESET} Model path (.gguf): {CYAN}{parsed['model'] or '(not set)'}{RESET}")
            print(f"    {BOLD}[3]{RESET} GPU layers (-ngl):  {CYAN}{parsed['ngl']}{RESET}")
            print(f"    {BOLD}[4]{RESET} Context size (-c):  {CYAN}{parsed['ctx']}{RESET}")
            print(f"    {BOLD}[5]{RESET} KV cache type:      {CYAN}{parsed['kv_type']}{RESET}")
            print(f"    {BOLD}[6]{RESET} Flash Attention:    {CYAN}{'yes' if parsed['flash_attn'] else 'no'}{RESET}")
            print(f"    {BOLD}[7]{RESET} mmproj (vision):    {CYAN}{parsed['mmproj'] or '(none)'}{RESET}")
            print()
            mtp_str = "enabled"
            if parsed["mtp_enabled"]:
                mtp_str = f"enabled (draft={parsed['mtp_n']}"
                if parsed["mtp_model"]:
                    mtp_str += f", model={os.path.basename(parsed['mtp_model'])}"
                mtp_str += ")"
            print(f"    {BOLD}[8]{RESET} MTP speculative:    {CYAN}{mtp_str}{RESET}")
            print(f"    {BOLD}[9]{RESET} TTL (seconds):      {CYAN}{cfg.get('ttl', 600)}{RESET}")
            print(f"    {BOLD}[R]{RESET} Raw command string")
            print()
            print(f"    {BOLD}[0]{RESET} Back")
            print()

            k = get_key()

            if k in ("0", "escape", "q"):
                break

            elif k == "1":
                new_alias = input_line("New alias", alias)
                if new_alias and new_alias != alias:
                    self.models[new_alias] = self.models.pop(alias)
                    alias = new_alias
                    self.modified = True

            elif k == "2":
                val = input_line("Model path", parsed["model"])
                if val is not None:
                    parsed["model"] = val
                    cfg["cmd"] = build_cmd(parsed)
                    self.modified = True

            elif k == "3":
                val = input_line("GPU layers", parsed["ngl"])
                if val is not None:
                    parsed["ngl"] = val
                    cfg["cmd"] = build_cmd(parsed)
                    self.modified = True

            elif k == "4":
                val = input_line("Context size", parsed["ctx"])
                if val is not None:
                    parsed["ctx"] = val
                    cfg["cmd"] = build_cmd(parsed)
                    self.modified = True

            elif k == "5":
                print(f"\n  {BOLD}KV cache type:{RESET}")
                print(f"    {BOLD}[1]{RESET} q4_0 (default, low VRAM)")
                print(f"    {BOLD}[2]{RESET} q8_0 (balanced)")
                print(f"    {BOLD}[3]{RESET} f16  (full precision, high VRAM)")
                kv_k = get_key()
                kv_map = {"1": "q4_0", "2": "q8_0", "3": "f16"}
                if kv_k in kv_map:
                    parsed["kv_type"] = kv_map[kv_k]
                    cfg["cmd"] = build_cmd(parsed)
                    self.modified = True

            elif k == "6":
                parsed["flash_attn"] = not parsed["flash_attn"]
                cfg["cmd"] = build_cmd(parsed)
                self.modified = True

            elif k == "7":
                val = input_line("mmproj path (empty=none)", parsed["mmproj"])
                if val is not None:
                    parsed["mmproj"] = val
                    cfg["cmd"] = build_cmd(parsed)
                    self.modified = True

            elif k == "8":
                # MTP submenu
                self.draw_header(f"⚡ MTP for: {alias}")
                print()
                print(f"    Current: {'enabled' if parsed['mtp_enabled'] else 'disabled'}")
                print()
                print(f"    {BOLD}Careful: MTP keeps a draft head and its own cache in memory. On a 16 GB")
                print(f"    box a model that otherwise fits (e.g. 27B at -c 65536) will die loading")
                print(f"    with 'upstream command exited prematurely'. Enable it only with room to spare.{RESET}")
                print()
                print(f"    {BOLD}[1]{RESET} Enable MTP (bundled head)")
                print(f"    {BOLD}[2]{RESET} Enable MTP (separate file)")
                print(f"    {BOLD}[3]{RESET} Disable MTP")
                print()
                mtp_k = get_key()

                if mtp_k == "1":
                    parsed["mtp_enabled"] = True
                    parsed["mtp_model"] = ""
                    val = input_line("Draft tokens (1-4)", parsed["mtp_n"])
                    if val and val in "1234":
                        parsed["mtp_n"] = val
                    cfg["cmd"] = build_cmd(parsed)
                    self.modified = True

                elif mtp_k == "2":
                    parsed["mtp_enabled"] = True
                    val = input_line("MTP model file path", parsed["mtp_model"])
                    if val is not None:
                        parsed["mtp_model"] = val
                    val = input_line("Draft tokens (1-4)", parsed["mtp_n"])
                    if val and val in "1234":
                        parsed["mtp_n"] = val
                    cfg["cmd"] = build_cmd(parsed)
                    self.modified = True

                elif mtp_k == "3":
                    parsed["mtp_enabled"] = False
                    parsed["mtp_model"] = ""
                    cfg["cmd"] = build_cmd(parsed)
                    self.modified = True

            elif k == "9":
                val = input_line("TTL (seconds)", str(cfg.get("ttl", 600)))
                if val is not None:
                    try:
                        cfg["ttl"] = int(val)
                        self.modified = True
                    except ValueError:
                        pass

            elif k == "r":
                val = input_line("Raw command", cfg.get("cmd", ""))
                if val is not None:
                    cfg["cmd"] = val
                    self.modified = True

    # ── New model wizard ─────────────────────────────────────────────────

    def create_model(self):
        self.draw_header("➕ Add New Model")
        print()
        print(f"  {DIM}Fill in the fields below. Press Enter for defaults.{RESET}")
        print(f"  {DIM}Press Esc at any prompt to cancel.{RESET}")
        print()

        # Alias
        alias = input_line("Model alias (e.g. gemma4-12b)")
        if not alias:
            return

        # Model path
        model_path = input_line("Full path to .gguf file")
        if not model_path:
            print(f"  {RED}Model path is required{RESET}")
            get_key()
            return

        # GPU layers
        ngl = input_line("GPU layers", "99")

        # Context
        ctx = input_line("Context length", "65536")

        # KV cache
        print(f"\n  {BOLD}KV cache type:{RESET}")
        print(f"    {BOLD}[1]{RESET} q4_0 (default)")
        print(f"    {BOLD}[2]{RESET} q8_0")
        print(f"    {BOLD}[3]{RESET} f16")
        kv_k = get_key()
        kv_map = {"1": "q4_0", "2": "q8_0", "3": "f16"}
        kv_type = kv_map.get(kv_k, "q4_0")

        # Flash Attention
        fa = input_yesno("Enable Flash Attention?", default=True)

        # mmproj
        mmproj = ""
        if input_yesno("Does this model support vision/audio? (mmproj)", default=False):
            mmproj = input_line("mmproj file path") or ""

        # MTP
        mtp_enabled = False
        mtp_model = ""
        mtp_n = "2"
        if input_yesno("Enable MTP speculative decoding?", default=False):
            mtp_enabled = True
            if input_yesno("Do you have a separate MTP .gguf file?", default=False):
                mtp_model = input_line("MTP model file path") or ""
            mtp_n = input_line("Draft tokens (1-4)", "2") or "2"

        # TTL
        ttl_str = input_line("TTL (seconds)", "600")
        try:
            ttl = int(ttl_str)
        except (ValueError, TypeError):
            ttl = 600

        # Build parsed and cmd
        parsed = {
            "model": model_path,
            "ngl": ngl,
            "ctx": ctx,
            "kv_type": kv_type,
            "flash_attn": fa if fa is not None else True,
            "mmproj": mmproj,
            "mtp_enabled": mtp_enabled,
            "mtp_model": mtp_model,
            "mtp_n": mtp_n,
            "port": "${PORT}",
            "host": "127.0.0.1",
            "extra": [],
        }

        cmd = build_cmd(parsed)

        # Confirm
        print()
        print(f"  {BOLD}New model:{RESET}")
        print(f"    Alias:   {GREEN}{alias}{RESET}")
        print(f"    Model:   {os.path.basename(model_path)}")
        print(f"    GPU:     {ngl} layers")
        print(f"    Context: {ctx}")
        print(f"    KV:      {kv_type}")
        print(f"    FA:      {'yes' if parsed['flash_attn'] else 'no'}")
        if mmproj:
            print(f"    mmproj:  {mmproj}")
        if mtp_enabled:
            print(f"    MTP:     enabled (draft={mtp_n})")
        print(f"    TTL:     {ttl}s")
        print()

        if input_yesno("Add this model?", default=True):
            self.models[alias] = {"cmd": cmd, "ttl": ttl}
            self.selected = len(self.aliases()) - 1
            self.modified = True

    # ── Delete model ─────────────────────────────────────────────────────

    def delete_model(self):
        aliases = self.aliases()
        if not aliases or self.selected >= len(aliases):
            return

        alias = aliases[self.selected]
        self.draw_header(f"🗑  Delete: {alias}")
        print()
        print(f"  {RED}Delete model '{alias}'?{RESET}")

        parsed = parse_cmd(self.models[alias].get("cmd", ""))
        if parsed["model"]:
            print(f"  {DIM}Model: {os.path.basename(parsed['model'])}{RESET}")
        print()

        if input_yesno("Delete", default=False):
            del self.models[alias]
            self.selected = max(0, min(self.selected, len(self.aliases()) - 1))
            self.modified = True

    # ── Main loop ────────────────────────────────────────────────────────

    def run(self):
        try:
            self._loop()
        except KeyboardInterrupt:
            pass
        finally:
            print(CLEAR)
            print(f"  {DIM}Model manager exited.{RESET}\n")
            if self.modified:
                print(f"  {YELLOW}⚠ You have unsaved changes!{RESET}\n")

    def _loop(self):
        while True:
            if self.page == "main":
                self.draw_main()
                key = get_key()
                aliases = self.aliases()

                if key == "up":
                    self.selected = max(0, self.selected - 1)
                elif key == "down":
                    self.selected = min(max(0, len(aliases) - 1), self.selected + 1)
                elif key in ("enter", "e") and aliases:
                    self.edit_model()
                elif key == "n":
                    self.create_model()
                elif key == "d" and aliases:
                    self.delete_model()
                elif key == "s":
                    if self.save():
                        print(f"  {GREEN}✓ Saved to {self.config_file}{RESET}")
                        get_key()
                elif key == "q":
                    if self.modified:
                        self.draw_header("⚠  Unsaved Changes")
                        print()
                        print(f"  {YELLOW}You have unsaved changes!{RESET}")
                        print(f"  {BOLD}[S]{RESET} Save & quit  {BOLD}[Q]{RESET} Quit without saving  {BOLD}[Esc]{RESET} Cancel")
                        k2 = get_key()
                        if k2 == "s":
                            self.save()
                            return
                        elif k2 == "q":
                            return
                        # else: cancel, go back to main
                    else:
                        return
                elif key == "ctrl-c":
                    return


def main():
    config_file = sys.argv[1] if len(sys.argv) > 1 else "/opt/llama-swap/config.yaml"
    if not os.path.exists(config_file):
        print(f"{RED}Config file not found: {config_file}{RESET}")
        print(f"Run 'install.sh configure' first to create it.")
        sys.exit(1)
    mgr = ModelsManager(config_file)
    mgr.run()


if __name__ == "__main__":
    main()
