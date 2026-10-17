#!/bin/sh
# searxng-setup.sh - a local SearXNG for the netscan jobs, in a podman container.
#
# Why this exists: ddgr scrapes DuckDuckGo directly, and DuckDuckGo throttles this
# box's IP (HTTP 202), which is how the music job managed to fail silently for two
# months. SearXNG is a metasearch front-end: one local service queries several
# engines and returns clean JSON, so a throttled engine is no longer fatal.
#
# What it does NOT do is make upstream captchas disappear — it spreads the load
# across engines instead of betting everything on one. And it does not change the
# nature of the thing: a metasearch scraper is still a scraper, so a source that
# ships license metadata (Jamendo, ccMixter) remains the better foundation for
# anything that has to be legally right.
#
# Cost, measured on this box: see `status` for the real RSS. It is a Python web
# app with no GPU code path, so llama.cpp's UMA memory is untouched.
#
# Usage:
#   searxng-setup.sh check      # prerequisites, ports, image, current state
#   searxng-setup.sh up         # create/recreate the container and self-test it
#   searxng-setup.sh down       # stop and remove it
#   searxng-setup.sh status     # state + memory + the settings that matter
#   searxng-setup.sh test [q]   # one JSON query, print how many results came back
#   searxng-setup.sh logs       # last 40 lines
#   searxng-setup.sh boot       # make podman start at boot (OpenRC runlevel)
#
# Layout:  /opt/searxng/settings.yml (generated, holds the secret, never in git)
#          http://127.0.0.1:8890  (localhost only: nothing is exposed to the LAN)
set -e

NAME="searxng"
IMAGE="docker.io/searxng/searxng:latest"
HOST_DIR="/opt/searxng"
BIND_IP="127.0.0.1"
HOST_PORT="8890"
BASE_URL="http://${BIND_IP}:${HOST_PORT}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

say() { printf '  %s\n' "$*"; }
die() { printf '  ERROR: %s\n' "$*" >&2; exit 1; }

have_podman() { command -v podman >/dev/null 2>&1; }

running() { podman container exists "$NAME" 2>/dev/null; }

gen_secret() { head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'; }

write_settings() {
    [ -f "$HOST_DIR/settings.yml" ] && { say "settings.yml already there (left alone)"; return 0; }
    tmpl="$SCRIPT_DIR/settings.yml.template"
    [ -f "$tmpl" ] || die "template not found: $tmpl"
    mkdir -p "$HOST_DIR"
    secret="$(gen_secret)"
    # Portable in-place substitution without sed -i (busybox has no -i without a
    # suffix) and without leaving the secret in the shell history or an argument.
    awk -v s="$secret" '{gsub(/__SECRET__/, s); print}' "$tmpl" > "$HOST_DIR/settings.yml"
    # 644, not 600: SearXNG runs as a non-root user inside the container and has
    # to read this mount. It holds a session secret for a service that only
    # listens on 127.0.0.1 on a single-user box; the file itself stays out of git.
    chmod 644 "$HOST_DIR/settings.yml"
    say "wrote $HOST_DIR/settings.yml with a fresh secret (mode 644, readable by the container user)"
}

port_busy() {
    netstat -ltn 2>/dev/null | grep -q ":${HOST_PORT} " && return 0
    return 1
}

# ── subcommands ────────────────────────────────────────────────────────────

cmd_check() {
    say "podman:  $(podman --version 2>/dev/null || echo MISSING)"
    if have_podman; then :; else die "podman is required (apk add podman)"; fi

    if podman image exists "$IMAGE" 2>/dev/null; then
        say "image:   present ($(podman images --format '{{.Size}}' "$IMAGE" | head -1))"
    else
        say "image:   not pulled yet (up will fetch it, ~271 MB)"
    fi

    if running; then
        say "container: exists ($(podman inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null))"
    else
        say "container: not created"
    fi

    if port_busy; then
        say "port $HOST_PORT: BUSY — something is already listening"
    else
        say "port $HOST_PORT: free"
    fi

    say "settings: $([ -f "$HOST_DIR/settings.yml" ] && echo present || echo 'will be generated')"
    say "boot:     $(rc-update show 2>/dev/null | grep -q '^ *podman' && echo 'podman in a runlevel' || echo 'podman NOT in a runlevel — run: searxng-setup.sh boot')"
    avail=$(free -m 2>/dev/null | awk '/Mem:/ {print $7}')
    [ -n "$avail" ] && say "memory:   ${avail} MB available right now"
}

cmd_up() {
    have_podman || die "podman not found"
    write_settings

    if ! podman image exists "$IMAGE" 2>/dev/null; then
        say "pulling $IMAGE ..."
        podman pull "$IMAGE" >/dev/null || die "pull failed"
    fi

    if running; then
        say "recreating the container so the config takes effect"
        podman rm -f "$NAME" >/dev/null
    fi

    port_busy && die "port $HOST_PORT is already in use"

    # Host networking on purpose. The default bridge needs iptables to program
    # port forwarding, and this box has none: podman fails with
    # "netavark: iptables: No such file or directory". With --network=host there
    # is no NAT to program, no dependency to install, and SearXNG itself binds
    # 127.0.0.1:8890 (see settings.yml.template), so nothing is exposed to the
    # LAN. If you would rather have the isolation, `apk add iptables` and put the
    # -p mapping back.
    podman run -d --name "$NAME" \
        --restart=unless-stopped \
        --network=host \
        -v "${HOST_DIR}:/etc/searxng:rw" \
        --cap-drop ALL --cap-add CHOWN --cap-add SETGID --cap-add SETUID \
        -e "SEARXNG_BASE_URL=${BASE_URL}/" \
        -e "SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml" \
        -e "SEARXNG_BIND_ADDRESS=${BIND_IP}" \
        -e "SEARXNG_PORT=${HOST_PORT}" \
        -e "GRANIAN_HOST=${BIND_IP}" \
        "$IMAGE" >/dev/null || die "podman run failed"
    # SEARXNG_BIND_ADDRESS / SEARXNG_PORT are not redundant with settings.yml: in
    # the official image the environment WINS over the settings file. Its default
    # is 0.0.0.0:8080, which with host networking is llama-swap, so without these
    # two the container restarts forever on "Address already in use" (84 restarts
    # before anyone read the log).
    #
    # GRANIAN_HOST is the one that actually keeps it off the LAN: the app is served
    # by granian (auto_envvar_prefix='GRANIAN'), and on host networking it happily
    # binds [::]:8890 — every interface — unless told otherwise. Verified with
    # netstat in the check below rather than assumed.

    say "container started, waiting for the web app..."
    i=0
    while [ "$i" -lt 30 ]; do
        if curl -fsS -m 4 "${BASE_URL}/" >/dev/null 2>&1; then
            say "up at ${BASE_URL}"
            listener="$(netstat -ltn 2>/dev/null | grep ":${HOST_PORT} " | awk '{print $4}' | head -1)"
            case "$listener" in
                127.0.0.1:*) say "listening on ${listener} (localhost only)" ;;
                "") say "warning: nothing seen listening on ${HOST_PORT}" ;;
                *) say "WARNING: listening on ${listener} — that is not localhost only. Set GRANIAN_HOST or use bridged networking." ;;
            esac
            cmd_test ""
            return 0
        fi
        i=$((i + 1)); sleep 2
    done
    say "it did not answer in 60s — check the logs: $0 logs"
    return 1
}

cmd_down() {
    running || { say "no container named $NAME"; return 0; }
    podman rm -f "$NAME" >/dev/null && say "removed"
}

cmd_status() {
    if running; then
        say "state:  $(podman inspect -f '{{.State.Status}}' "$NAME")"
        say "uptime: $(podman inspect -f '{{.State.StartedAt}}' "$NAME")"
        rss=$(podman stats --no-stream --format "{{.MemUsage}}" "$NAME" 2>/dev/null | head -1)
        say "memory: ${rss:-unavailable}"
        cpu=$(podman stats --no-stream --format "{{.CPUPerc}}" "$NAME" 2>/dev/null | head -1)
        say "cpu:    ${cpu:-unavailable}"
    else
        say "container not created"
    fi
    if [ -f "$HOST_DIR/settings.yml" ]; then
        say "settings: formats=$(grep -A3 '^search:' "$HOST_DIR/settings.yml" | grep -o 'json\|html' | tr '\n' ',' | sed 's/,$//')  limiter=$(grep -o 'limiter:.*' "$HOST_DIR/settings.yml" | head -1 | cut -d: -f2 | tr -d ' ')"
    fi
    host_free=$(free -m 2>/dev/null | awk '/Mem:/ {print $7}')
    [ -n "$host_free" ] && say "host free memory: ${host_free} MB"
}

cmd_test() {
    q="${1:-house music royalty free}"
    out=$(curl -fsS -m 25 --get --data-urlencode "q=$q" --data "format=json" \
              --data "time_range=week" "${BASE_URL}/search" 2>/dev/null) || {
        say "no answer from ${BASE_URL} (is it running? see: $0 status)"; return 1; }
    n=$(printf '%s' "$out" | grep -o '"url"' | wc -l)
    say "query: $q"
    say "results: $n"
    if [ "$n" -gt 0 ]; then
        printf '%s' "$out" | tr ',' '\n' | grep '"url"' | head -3 | sed 's/^/    /'
    fi
}

cmd_logs() { podman logs --tail 40 "$NAME" 2>&1 || say "no logs"; }

cmd_boot() {
    if [ -f /etc/init.d/podman ]; then
        rc-update add podman default >/dev/null 2>&1 \
            && say "podman added to the default runlevel" \
            || say "could not add podman to a runlevel (check rc-update)"
    else
        say "no /etc/init.d/podman on this system — the container will not come back by itself"
    fi
    say "the container itself carries --restart=unless-stopped"
}

case "${1:-help}" in
    check)  cmd_check ;;
    up)     cmd_up ;;
    down)   cmd_down ;;
    status) cmd_status ;;
    test)   shift; cmd_test "$@" ;;
    logs)   cmd_logs ;;
    boot)   cmd_boot ;;
    *)      sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
