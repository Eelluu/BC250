#!/usr/bin/env python3
"""bc250-remote.py — the operator's side of the box, over SSH.

A thin, explicit wrapper around the system `ssh`/`scp` binaries. It exists so
that deploying a job, tailing a service and reading the box's state use exactly
one code path — the same one a human would type by hand, just shorter and less
error-prone.

No secrets and no addresses are stored in this file: host, user and key come
from the environment (BC250_HOST, BC250_USER, BC250_KEY) or from explicit
flags, so the file is identical for everybody and safe to keep in the repo.

Usage:
  bc250-remote.py status                          # uname, uptime, RAM, GPU, services
  bc250-remote.py run "free -h"                   # any command, output passthrough
  bc250-remote.py shell                           # interactive session
  bc250-remote.py put netscan/market-collect.py /opt/netscan/
  bc250-remote.py get /opt/netscan/data/markets/latest-markets.json .
  bc250-remote.py log llama-swap --lines 40
  bc250-remote.py restart queue-runner
  bc250-remote.py selftest                        # checks the plumbing

Env:
  BC250_HOST   host or IP of the box           (required)
  BC250_USER   ssh user                        (default: root)
  BC250_KEY    private key path                (default: ~/.ssh/bc250_agent)

Location: workstation-side tool; the box itself needs none of this.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new")

# One round trip for the whole picture: the box is a small machine and this is
# the command an operator would otherwise paste from memory, badly.
STATUS_CMDS = r"""
echo "--- host ---";      uname -srm; uptime
echo "--- memory ---";    free -h | head -3
echo "--- disk ---";      df -h /opt /root 2>/dev/null | tail -3
echo "--- llama-swap ---"; rc-service llama-swap status 2>&1 | head -3
echo "--- queue-runner ---"; rc-service queue-runner status 2>&1 | head -3
echo "--- models ---";    curl -s -m 5 localhost:8080/v1/models | head -c 400
echo
echo "--- market data ---"; ls -la /opt/netscan/data/markets/ 2>/dev/null | tail -6
echo "--- python ---";    python3 -V
"""


class Remote:
    def __init__(self, host, user="root", key=None, port=22):
        self.host, self.user, self.port = host, user, port
        self.key = str(Path(key).expanduser()) if key else None

    def _base(self, batch=True):
        cmd = ["ssh", "-p", str(self.port)]
        cmd += SSH_OPTS if batch else ("-o", "ConnectTimeout=8")
        if self.key:
            cmd += ["-i", self.key]
        return cmd

    def target(self):
        return f"{self.user}@{self.host}"

    def run(self, remote_cmd, batch=True, check=False):
        argv = self._base(batch) + [self.target(), remote_cmd]
        proc = subprocess.run(argv, text=True)
        if check and proc.returncode:
            sys.exit(proc.returncode)
        return proc.returncode

    def scp(self, direction, src, dst):
        if not shutil.which("scp"):
            print("scp not found on PATH", file=sys.stderr)
            return 127
        argv = ["scp", "-P", str(self.port)] + list(SSH_OPTS[2:])
        if self.key:
            argv += ["-i", self.key]
        argv += (["-r", src, f"{self.target()}:{dst}"] if direction == "put"
                 else [f"{self.target()}:{src}", dst])
        return subprocess.run(argv, text=True).returncode

    def check(self):
        """Connection + key sanity, without touching anything on the box."""
        if not shutil.which("ssh"):
            print("ssh not found on PATH")
        if self.key and not Path(self.key).exists():
            print(f"key not found: {self.key}")
            return False
        code = self.run("echo bc250-reachable; whoami; hostname", batch=True)
        return code == 0


def selftest(remote):
    """Local checks first (they explain most failures), then one round trip."""
    print("  plumbing:")
    print(f"    ssh      : {shutil.which('ssh') or 'MISSING'}")
    print(f"    scp      : {shutil.which('scp') or 'MISSING'}")
    print(f"    key      : {remote.key or '(none given)'}")
    if remote.key:
        path = Path(remote.key)
        print(f"    key file : {'exists' if path.exists() else 'MISSING'} {path}")
        if path.exists() and path.with_suffix(path.suffix + ".pub").exists():
            print(f"    pubkey   : {path}.pub present (authorise this one on the box)")
    print(f"    target   : {remote.target()}:{remote.port}")
    ok = remote.check()
    print(f"\n  selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _common_flags():
    """Flags that work both before and after the subcommand.

    argparse otherwise forces `--host X status` and rejects `status --host X`,
    which is a trap for the hand-typed case. SUPPRESS keeps the subparser from
    clobbering a value already parsed by the main parser.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--host", default=argparse.SUPPRESS)
    p.add_argument("--user", default=argparse.SUPPRESS)
    p.add_argument("--key", default=argparse.SUPPRESS)
    p.add_argument("--port", type=int, default=argparse.SUPPRESS)
    return p


def main(argv=None):
    common = _common_flags()
    ap = argparse.ArgumentParser(description="operate the bc250 over SSH",
                                 parents=[common])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", parents=[common],
                   help="host, memory, disk, services, models, market data")
    p_run = sub.add_parser("run", parents=[common],
                           help="run an arbitrary command on the box")
    p_run.add_argument("command")
    sub.add_parser("shell", parents=[common], help="interactive ssh session")
    p_put = sub.add_parser("put", parents=[common], help="copy local -> box")
    p_put.add_argument("src")
    p_put.add_argument("dst")
    p_get = sub.add_parser("get", parents=[common], help="copy box -> local")
    p_get.add_argument("src")
    p_get.add_argument("dst")
    p_log = sub.add_parser("log", parents=[common], help="tail an OpenRC service")
    p_log.add_argument("service")
    p_log.add_argument("--lines", type=int, default=40)
    p_res = sub.add_parser("restart", parents=[common],
                           help="restart an OpenRC service")
    p_res.add_argument("service")
    sub.add_parser("selftest", parents=[common], help="check ssh, key and connectivity")

    args = ap.parse_args(argv)

    # Defaults resolve here, env first: nothing about the box is committed.
    host = getattr(args, "host", None) or os.environ.get("BC250_HOST")
    user = getattr(args, "user", None) or os.environ.get("BC250_USER", "root")
    key = getattr(args, "key", None) or os.environ.get("BC250_KEY", "~/.ssh/bc250_agent")
    port = getattr(args, "port", None) or int(os.environ.get("BC250_PORT", 22))

    if args.cmd == "selftest":
        return selftest(Remote(host or "unset", user, key, port))

    if not host:
        print("BC250_HOST is not set (or pass --host). Nothing is hardcoded on "
              "purpose: this file carries no addresses.", file=sys.stderr)
        return 2

    remote = Remote(host, user, key, port)

    if args.cmd == "status":
        return remote.run(STATUS_CMDS.strip())
    if args.cmd == "run":
        return remote.run(args.command)
    if args.cmd == "shell":
        return remote.run("exec bash -l", batch=False)
    if args.cmd == "put":
        return remote.scp("put", args.src, args.dst)
    if args.cmd == "get":
        return remote.scp("get", args.src, args.dst)
    if args.cmd == "log":
        return remote.run(f"tail -n {int(args.lines)} /var/log/{args.service}.log 2>/dev/null "
                          f"|| rc-service {args.service} status 2>&1 | tail -n 20")
    if args.cmd == "restart":
        return remote.run(f"rc-service {args.service} restart && rc-service {args.service} status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
