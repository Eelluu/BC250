#!/usr/bin/env python3
"""
BC-250 Job Manager — Interactive TUI for managing job schedules.
Usage: python3 jobs-tui.py [jobs.json path]
"""

import json
import os
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
WHITE = "\033[97m"
BG_BLUE = "\033[44m"

# ─── Cron helpers ────────────────────────────────────────────────────────────

DAYS = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}


def cron_to_desc(cron):
    """Convert cron expression to human-readable description."""
    parts = cron.strip().split()
    if len(parts) != 5:
        return cron
    minute, hour, dom, month, dow = parts

    if minute.startswith("*/") and hour == "*" and dow == "*":
        return f"every {minute[2:]} min"
    if not minute.startswith("*/") and hour.startswith("*/") and dom == "*":
        return f"every {hour[2:]}h at :{minute.zfill(2)}"
    if hour == "*" and dom == "*" and dow == "*" and not minute.startswith("*/"):
        return f"every hour at :{minute.zfill(2)}"
    if dow != "*" and hour != "*" and minute != "*" and dom == "*":
        day_names = [DAYS.get(int(d), d) for d in dow.split(",")]
        return f"{','.join(day_names)} at {hour.zfill(2)}:{minute.zfill(2)}"
    if dom != "*" and hour != "*" and minute != "*" and month == "*":
        return f"monthly (day {dom}) at {hour.zfill(2)}:{minute.zfill(2)}"
    if dom == "*" and hour != "*" and minute != "*" and dow == "*":
        return f"daily at {hour.zfill(2)}:{minute.zfill(2)}"
    return cron


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
            return "escape"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == " ":
            return "space"
        return ch.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ─── TUI ─────────────────────────────────────────────────────────────────────

PRESETS = [
    ("1", "Every 5 minutes", "*/5 * * * *"),
    ("2", "Every 15 minutes", "*/15 * * * *"),
    ("3", "Every 30 minutes", "*/30 * * * *"),
    ("4", "Every hour", "0 * * * *"),
    ("5", "Every day at 02:00", "0 2 * * *"),
    ("6", "Every day at 06:00", "0 6 * * *"),
    ("7", "Weekly (Mon 02:00)", "0 2 * * 1"),
    ("8", "Monthly (1st 02:00)", "0 2 1 * *"),
]


class JobManager:
    def __init__(self, jobs_file):
        self.jobs_file = jobs_file
        self.data = {"jobs": []}
        self.selected = 0
        self.modified = False
        self.page = "main"  # main | schedule | custom | create
        self.load()

    def load(self):
        try:
            with open(self.jobs_file) as f:
                self.data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"{RED}Error loading {self.jobs_file}: {e}{RESET}")
            sys.exit(1)

    def save(self):
        try:
            with open(self.jobs_file, "w") as f:
                json.dump(self.data, f, indent=2)
            self.modified = False
            return True
        except Exception as e:
            print(f"  {RED}Error saving: {e}{RESET}")
            return False

    def jobs(self):
        return self.data.get("jobs", [])

    def draw_header(self, title):
        print(CLEAR)
        mod = f"  {YELLOW}● Unsaved{RESET}" if self.modified else ""
        print(f"  {BOLD}{BLUE}{title}{RESET}{mod}")
        print(f"  {DIM}{'─' * 60}{RESET}")

    # ── Main screen ──────────────────────────────────────────────────────

    def draw_main(self):
        jobs = self.jobs()
        self.draw_header("🤖 BC-250 Job Manager")
        print()

        if not jobs:
            print(f"  {DIM}No jobs found in {self.jobs_file}{RESET}")
        else:
            for i, job in enumerate(jobs):
                enabled = job.get("enabled", False)
                icon = f"{GREEN}●{RESET}" if enabled else f"{RED}○{RESET}"
                name = job.get("name", "unknown")
                sched = job.get("schedule", "manual")
                desc = cron_to_desc(sched)

                if i == self.selected:
                    line = f" {icon} {name:<22} {DIM}{sched:<16}{RESET} {CYAN}{desc}{RESET}"
                    print(f"{BG_BLUE}{WHITE}{line}{RESET}")
                else:
                    clr = GREEN if enabled else DIM
                    print(f"  {clr}{icon} {name:<22}{RESET} {DIM}{sched:<16}{desc}{RESET}")

        print()
        print(f"  {DIM}──────────────────────────────────────────────────────────{RESET}")
        if jobs:
            print(f"  {BOLD}[↑↓]{RESET} Move  {BOLD}[Space]{RESET} Toggle  {BOLD}[E]{RESET} Edit schedule"
                  f"  {BOLD}[N]{RESET} New  {BOLD}[D]{RESET} Delete  {BOLD}[S]{RESET} Save  {BOLD}[Q]{RESET} Quit")
        else:
            print(f"  {BOLD}[N]{RESET} New job  {BOLD}[Q]{RESET} Quit")
        print()

    # ── Schedule editor ──────────────────────────────────────────────────

    def draw_schedule(self):
        jobs = self.jobs()
        if self.selected >= len(jobs):
            self.page = "main"
            return
        job = jobs[self.selected]
        sched = job.get("schedule", "manual")

        self.draw_header(f"📅 Schedule: {job['name']}")
        print()
        print(f"  Current: {BOLD}{sched}{RESET}  {DIM}({cron_to_desc(sched)}){RESET}")
        print()

        for key, label, cron in PRESETS:
            marker = f"  {GREEN}◄{RESET}" if sched == cron else ""
            print(f"    {BOLD}[{key}]{RESET} {label}{marker}")

        print(f"    {BOLD}[9]{RESET} Custom cron expression")
        print()
        print(f"    {BOLD}[0]{RESET} Back")
        print()
        print(f"  {DIM}──────────────────────────────────────────────────────────{RESET}")
        print()

    # ── Custom cron input ────────────────────────────────────────────────

    def input_cron(self):
        self.draw_header("⌨  Custom Cron Expression")
        print()
        print(f"  {DIM}Format: min hour dom month dow{RESET}")
        print(f"  {DIM}Examples:{RESET}")
        print(f"    {DIM}*/30 * * * *       every 30 minutes{RESET}")
        print(f"    {DIM}0 9 * * 1-5        weekdays at 09:00{RESET}")
        print(f"    {DIM}0 2 1 * *          1st of month at 02:00{RESET}")
        print(f"    {DIM}0 3 * * 0,6        weekends at 03:00{RESET}")
        print()
        sys.stdout.write(f"  {BOLD}>{RESET} ")
        sys.stdout.flush()

        # Restore cooked mode for line input
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            new = termios.tcgetattr(fd)
            new[3] &= ~termios.ECHO
            new[6][termios.VMIN] = 1
            new[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, new)
            cron = sys.stdin.readline().strip()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

        parts = cron.split()
        if len(parts) == 5:
            return cron
        if cron:
            print(f"\n  {RED}Need exactly 5 fields, got {len(parts)}{RESET}")
            sys.stdout.write("  Press any key...")
            sys.stdout.flush()
            get_key()
        return None

    # ── Text input helper ──────────────────────────────────────────────

    def input_line(self, prompt, default="", required=False):
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
                    # Esc pressed — cancel
                    sys.stdout.write("\n")
                    return None
                if ch in ("\r", "\n"):
                    sys.stdout.write("\n")
                    line = buf.strip()
                    if line:
                        return line
                    return default if default else (None if required else "")
                if ch == "\x7f" or ch == "\x08":
                    # Backspace
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

    def input_yesno(self, prompt, default=True):
        """Prompt for y/n. Returns True/False or None on Esc."""
        hint = "Y/n" if default else "y/N"
        result = self.input_line(f"{prompt} [{hint}]", "")
        if result is None:
            return None
        if not result:
            return default
        return result.lower() in ("y", "yes")

    # ── Create job screen ───────────────────────────────────────────────

    def draw_create(self):
        self.draw_header("➕ New Job")
        print()
        print(f"  {DIM}Fill in the fields below. Press Enter to accept defaults.{RESET}")
        print(f"  {DIM}Press Esc at any prompt to cancel.{RESET}")
        print()
        print(f"  {BOLD}Fields:{RESET}")
        print(f"    Name        — display name (e.g. my-scanner)")
        print(f"    Command     — what to run (e.g. python3 /opt/netscan/my-script.py)")
        print(f"    Schedule    — cron expression or preset number (1-8)")
        print(f"    Timeout     — max seconds (default 600)")
        print()
        print(f"  {DIM}──────────────────────────────────────────────────────────{RESET}")
        print()

    def create_job(self):
        self.draw_create()

        # Name
        name = self.input_line("Name")
        if name is None:
            return False
        if not name:
            print(f"  {RED}Name is required{RESET}")
            get_key()
            return False

        # ID (auto-generate from name)
        job_id = name.lower().replace(" ", "-").replace("_", "-")
        job_id = ".".join(c for c in job_id if c.isalnum() or c in "-.")

        # Command
        cmd = self.input_line("Command", f"python3 /opt/netscan/{name}.py")
        if cmd is None:
            return False

        # Schedule - show presets
        print()
        print(f"  {BOLD}Schedule presets:{RESET}")
        for key, label, cron in PRESETS:
            print(f"    {BOLD}[{key}]{RESET} {label}")
        print(f"    {BOLD}[9]{RESET} Custom cron expression")
        print()

        sched_input = self.input_line("Schedule [1-9] or cron", "5")
        if sched_input is None:
            return False

        preset_map = {k: cron for k, _, cron in PRESETS}
        if sched_input in preset_map:
            schedule = preset_map[sched_input]
        elif sched_input == "9":
            schedule = self.input_cron()
            if not schedule:
                return False
        else:
            # Assume raw cron
            parts = sched_input.split()
            if len(parts) == 5:
                schedule = sched_input
            else:
                print(f"  {RED}Invalid cron (need 5 fields){RESET}")
                get_key()
                return False

        # Timeout
        timeout_str = self.input_line("Timeout (seconds)", "600")
        if timeout_str is None:
            return False
        try:
            timeout = int(timeout_str)
        except ValueError:
            timeout = 600

        # Build job
        new_job = {
            "id": job_id,
            "name": name,
            "enabled": True,
            "schedule": schedule,
            "payload": {
                "message": f"Run {cmd}",
                "timeoutSeconds": timeout,
            },
        }

        # Confirm
        print()
        print(f"  {BOLD}New job:{RESET}")
        print(f"    Name:     {GREEN}{name}{RESET}")
        print(f"    ID:       {job_id}")
        print(f"    Command:  {cmd}")
        print(f"    Schedule: {schedule} ({cron_to_desc(schedule)})")
        print(f"    Timeout:  {timeout}s")
        print()

        if self.input_yesno("Create this job?", default=True):
            self.data["jobs"].append(new_job)
            self.selected = len(self.jobs()) - 1
            self.modified = True
            return True
        return False

    # ── Delete job ───────────────────────────────────────────────────────

    def delete_job(self):
        jobs = self.jobs()
        if not jobs or self.selected >= len(jobs):
            return
        job = jobs[self.selected]
        name = job.get("name", "unknown")

        self.draw_header(f"🗑  Delete: {name}")
        print()
        print(f"  {RED}Are you sure you want to delete '{name}'?{RESET}")
        print()

        if self.input_yesno("Delete", default=False):
            jobs.pop(self.selected)
            self.selected = min(self.selected, len(jobs) - 1)
            self.modified = True

    # ── Main loop ────────────────────────────────────────────────────────

    def run(self):
        try:
            self._loop()
        except KeyboardInterrupt:
            pass
        finally:
            print(CLEAR)
            print(f"  {DIM}Job manager exited.{RESET}\n")
            if self.modified:
                print(f"  {YELLOW}⚠ You have unsaved changes!{RESET}\n")

    def _loop(self):
        while True:
            if self.page == "main":
                self.draw_main()
                key = get_key()
                jobs = self.jobs()

                if key == "up":
                    self.selected = max(0, self.selected - 1)
                elif key == "down":
                    self.selected = min(len(jobs) - 1, self.selected + 1)
                elif key == "space" and jobs:
                    j = jobs[self.selected]
                    j["enabled"] = not j.get("enabled", True)
                    self.modified = True
                elif key == "e" and jobs:
                    self.page = "schedule"
                elif key == "n":
                    self.create_job()
                    self.page = "main"
                elif key == "d" and jobs:
                    self.delete_job()
                elif key == "s":
                    if self.save():
                        print(f"  {GREEN}✓ Saved to {self.jobs_file}{RESET}")
                        get_key()
                elif key == "q":
                    if self.modified:
                        print(f"\n  {YELLOW}Unsaved changes!{RESET} Press {BOLD}S{RESET} to save, {BOLD}Q{RESET} again to quit.")
                        get_key()
                        k2 = get_key()
                        if k2 == "s":
                            self.save()
                    return

            elif self.page == "schedule":
                self.draw_schedule()
                key = get_key()
                jobs = self.jobs()
                if self.selected >= len(jobs):
                    self.page = "main"
                    continue
                job = jobs[self.selected]

                preset_map = {k: cron for k, _, cron in PRESETS}

                if key in preset_map:
                    job["schedule"] = preset_map[key]
                    self.modified = True
                    self.page = "main"
                elif key == "9":
                    cron = self.input_cron()
                    if cron:
                        job["schedule"] = cron
                        self.modified = True
                    self.page = "main"
                elif key in ("0", "escape", "q", "enter"):
                    self.page = "main"


def main():
    jobs_file = sys.argv[1] if len(sys.argv) > 1 else "/opt/netscan/data/jobs.json"
    mgr = JobManager(jobs_file)
    mgr.run()


if __name__ == "__main__":
    main()
