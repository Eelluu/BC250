#!/bin/bash
# =============================================================================
# BC-250 v10 Installer — Alpine / CachyOS / Arch / Fedora
# llama-swap + llama.cpp (TurboQuant / mainline / bee.cpp, Vulkan) + Discord Bot
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NETSCAN_DIR="/opt/netscan"
DATA_DIR="${NETSCAN_DIR}/data"
MODELS_DIR="/opt/llama-models"
VENV_DIR="${NETSCAN_DIR}/venv"

# llama.cpp source (default: TurboQuant fork). Overridden interactively by
# select_llama_fork() during install.
LLAMA_FORK_NAME="TurboQuant"
LLAMA_REPO_URL="https://github.com/TheTom/llama-cpp-turboquant"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_err()  { echo -e "${RED}[ERR]${NC} $*" >&2; }

# ─── Distro detection ─────────────────────────────────────────────────────────
if [[ -f /etc/os-release ]]; then
    source /etc/os-release
    OS_ID="${ID:-unknown}"
    OS_ID_LIKE="${ID_LIKE:-}"
else
    OS_ID="unknown"
    OS_ID_LIKE=""
fi

is_alpine() { [[ "$OS_ID" == "alpine" ]]; }
is_arch()   { [[ "$OS_ID" == "arch" || "$OS_ID" == "cachyos" || "$OS_ID_LIKE" == *"arch"* ]]; }
is_fedora() { [[ "$OS_ID" == "fedora" || "$OS_ID_LIKE" == *"fedora"* || "$OS_ID_LIKE" == *"rhel"* ]]; }
is_systemd() { is_arch || is_fedora; }

if is_alpine; then
    DISTRO_NAME="Alpine Linux"
elif is_arch; then
    DISTRO_NAME="Arch-based (${OS_ID})"
elif is_fedora; then
    DISTRO_NAME="Fedora/RHEL-based (${OS_ID})"
else
    DISTRO_NAME="Unknown (${OS_ID})"
fi

# ─── Service helpers (OpenRC / systemd) ────────────────────────────────────────
service_cmd() {
    local action="$1"
    local svc="$2"
    if is_systemd; then
        systemctl "${action}" "${svc}.service"
    else
        rc-service "${svc}" "${action}"
    fi
}

enable_service() {
    if is_systemd; then
        systemctl enable "$1.service"
    else
        rc-update add "$1" default
    fi
}

start_service()   { service_cmd start "$1"; }
stop_service()    { service_cmd stop "$1"; }
restart_service() { service_cmd restart "$1"; }
status_service()  { service_cmd status "$1"; }

# ─── Helpers ──────────────────────────────────────────────────────────────────
# Prompt for a single line of input.
# The prompt is written to stderr (unbuffered, always visible immediately) and a
# newline is always printed after the reply, so prompts never pile up on one line.
prompt_input() {
    local prompt="$1"
    local var="$2"
    printf '%s' "${prompt}" >&2
    if IFS= read -r "${var}"; then
        printf '\n' >&2
        return 0
    fi
    printf '\n' >&2
    return 1
}

# Prompt for a secret (hidden input). Uses stty explicitly instead of read -s so
# terminal echo is reliably restored — otherwise subsequent prompts pile up on the
# same line and typed input stays invisible.
read_secret() {
    local prompt="$1"
    local var="$2"
    local _rc=1
    printf '%s' "${prompt}" >&2
    stty -echo 2>/dev/null || true
    # Restore echo even if the user interrupts (Ctrl+C) or the function exits abnormally
    trap 'stty echo 2>/dev/null || true' RETURN INT TERM EXIT
    if IFS= read -r "${var}"; then
        _rc=0
    fi
    stty echo 2>/dev/null || true
    trap - RETURN INT TERM EXIT
    printf '\n' >&2
    return "$_rc"
}

prompt_yn() {
    local prompt="$1"
    local default="${2:-y}"
    while true; do
        printf '%s' "${prompt} [y/n] (${default}): " >&2
        if ! IFS= read -r ans; then
            printf '\n' >&2
            return 1
        fi
        ans="${ans:-$default}"
        case "$ans" in
            [Yy]* ) printf '\n' >&2; return 0;;
            [Nn]* ) printf '\n' >&2; return 1;;
            * ) printf '%s\n' "Please answer y or n." >&2;;
        esac
    done
}

require_root() {
    if [[ $EUID -ne 0 ]]; then
        log_err "This script must be run as root (use sudo)."
        exit 1
    fi
}

# ── Optional component: local SearXNG ────────────────────────────────────────
# The netscan jobs used to search through ddgr, which scrapes DuckDuckGo and gets
# throttled (HTTP 202) from residential IPs — that is how music-discover managed to
# fail silently for two months. SearXNG is a metasearch front-end: one local
# service queries several engines and returns clean JSON, so one engine sulking is
# no longer fatal. It costs about 160 MB of RSS and has no GPU code path, so
# llama.cpp's memory is untouched.
#
# Non-fatal on purpose: a failed search service must never abort an install that
# has already built llama.cpp. Everything here logs a warning and moves on.
ensure_podman() {
    if command -v podman >/dev/null 2>&1; then
        return 0
    fi
    log_info "Installing podman (needed for the optional SearXNG container)..."
    if is_alpine; then
        apk add --no-cache podman >/dev/null 2>&1
    elif is_arch; then
        pacman -S --noconfirm --needed podman >/dev/null 2>&1
    elif is_fedora; then
        dnf install -y podman >/dev/null 2>&1
    fi
    # Last statement is the function's status; the caller tests it, so a missing
    # podman is a clean 'no' rather than a failure under set -e.
    command -v podman >/dev/null 2>&1
}

install_searxng() {
    log_info "Setting up the local SearXNG search service (optional)..."
    if [[ ! -d "${SCRIPT_DIR}/searxng" ]]; then
        log_warn "No searxng/ directory in ${SCRIPT_DIR}; skipping."
        return 0
    fi
    if ! ensure_podman; then
        log_warn "podman is not available on this system; skipping SearXNG."
        log_warn "The jobs fall back to ddgr, which may get rate limited."
        return 0
    fi

    mkdir -p /opt/searxng
    if ! cp -r "${SCRIPT_DIR}/searxng"/* /opt/searxng/; then
        log_warn "Could not copy searxng/ to /opt/searxng; skipping."
        return 0
    fi
    chmod +x /opt/searxng/searxng-setup.sh || true

    if sh /opt/searxng/searxng-setup.sh up; then
        sh /opt/searxng/searxng-setup.sh boot || true
        log_ok "SearXNG is up on http://127.0.0.1:8890 (localhost only)"
    else
        log_warn "SearXNG did not come up; see: sh /opt/searxng/searxng-setup.sh logs"
        log_warn "The music job will fall back to ddgr until it does."
    fi
    return 0
}

# ─── llama.cpp fork selection ────────────────────────────────────────────────
# Ask which llama.cpp source to build: TurboQuant fork (default), mainline
# ggml-org/llama.cpp, or bee llama.cpp. Sets the LLAMA_FORK_NAME /
# LLAMA_REPO_URL globals used by cmd_install. Non-interactive input (no tty)
# defaults to TurboQuant so automation keeps working.
select_llama_fork() {
    local choice=""
    while true; do
        printf '%s\n' "" >&2
        printf '%s\n' "── 🦞 llama.cpp source ──" >&2
        printf '%s\n' "  [1] TurboQuant fork (TheTom/llama-cpp-turboquant) — MTP speculative decoding, tuned for fast generation" >&2
        printf '%s\n' "  [2] Mainline llama.cpp (ggml-org/llama.cpp) — the original upstream" >&2
        printf '%s\n' "  [3] bee llama.cpp (Anbeeld/beellama.cpp) — DFlash + flash-attention / speculative improvements" >&2
        printf '%s' "Choose [1-3] (default 1): " >&2
        if ! IFS= read -r choice; then
            printf '\n' >&2
            choice="1"
        fi
        choice="${choice:-1}"
        case "$choice" in
            1) LLAMA_FORK_NAME="TurboQuant";            LLAMA_REPO_URL="https://github.com/TheTom/llama-cpp-turboquant"; return 0;;
            2) LLAMA_FORK_NAME="mainline";              LLAMA_REPO_URL="https://github.com/ggml-org/llama.cpp";           return 0;;
            3) LLAMA_FORK_NAME="bee llama.cpp";         LLAMA_REPO_URL="https://github.com/Anbeeld/beellama.cpp";         return 0;;
            *) printf '%s\n' "Please choose 1, 2 or 3." >&2;;
        esac
    done
}

# Locate the standalone hf CLI (installed by 'curl -LsSf https://hf.co/cli/install.sh | bash'
# into ~/.local/bin) or the pip-installed huggingface-cli. Prints the first one found.
find_hf_cli() {
    local candidate
    for candidate in \
        "$(command -v hf 2>/dev/null)" \
        "$HOME/.local/bin/hf" \
        "/root/.local/bin/hf" \
        "${VENV_DIR}/bin/hf" \
        "${VENV_DIR}/bin/huggingface-cli"; do
        if [[ -n "$candidate" ]] && [[ -x "$candidate" ]]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

# Escape & \ and the | delimiter for use inside a sed s||| replacement string.
sed_repl_escape() {
    printf '%s' "$1" | sed 's/[&\\|]/\\&/g'
}

# Download a single file from HuggingFace with the hf CLI, skipping if already
# present. Usage: hf_download <repo> <file> <local-dir>
hf_download() {
    local repo="$1" file="$2" dest="$3"
    local _hf_cli
    if [[ -f "${dest}/${file}" ]]; then
        log_ok "  Already present: ${file} — skipping"
        return 0
    fi
    mkdir -p "${dest}"
    if _hf_cli=$(find_hf_cli); then
        log_info "Downloading ${file} from ${repo}..."
        "$_hf_cli" download "${repo}" "${file}" --local-dir "${dest}" || log_warn "Download failed: ${file}"
    else
        log_warn "hf CLI not found. Run: pip install -U huggingface_hub"
    fi
}

# Find the first file matching <pattern> (case-insensitive) under <dir>,
# recursively. Prints the absolute path (or nothing). Used by the SD model
# auto-detect to locate downloaded .gguf / .safetensors files.
find_model_file() {
    local dir="$1" pattern="$2"
    [[ -d "$dir" ]] || return 1
    find "$dir" -type f -iname "$pattern" 2>/dev/null | head -n 1
}

# Write KEY="value" into .env, but ONLY if the key is currently empty (never
# overwrite a path the user typed manually during configure).
sd_env_set() {
    local key="$1" value="$2"
    [[ -n "$value" ]] || return 0  # nothing to write
    if grep -qE "^${key}=\"[^\"]+\"" "${NETSCAN_DIR}/.env"; then
        return 0  # already has a non-empty value (user typed it manually)
    fi
    sed -i "s|^${key}=.*|${key}=\"$(sed_repl_escape "$value")\"|" "${NETSCAN_DIR}/.env"
    log_info "  ${key} = ${value}"
}

# Verify the distro Pillow is visible inside the venv and satisfies discord.py (>=8.0).
# Uses the venv Python explicitly and handles version suffixes like "10.0.0.dev0".
# Returns:
#   0 on success and prints the Pillow version on stdout.
#   1 if the venv Python is missing.
#   2 if Pillow is not installed.
#   3 if the Pillow version is < 8.0.
check_venv_pillow() {
    if [[ ! -x "${VENV_DIR}/bin/python3" ]]; then
        return 1
    fi
    "${VENV_DIR}/bin/python3" <<'PY'
import sys, re
try:
    import PIL
except Exception:
    sys.exit(2)
parts = [p for p in re.split(r'\D+', PIL.__version__) if p][:3]
if len(parts) < 2:
    sys.exit(3)
parts += ['0'] * (3 - len(parts))
if tuple(map(int, parts)) < (8, 0, 0):
    sys.exit(3)
print(PIL.__version__)
PY
}

install_system_packages() {
    log_info "Installing system packages for ${DISTRO_NAME}..."
    if is_alpine; then
        apk update
        log_info "Adding Alpine edge/testing repo for ddgr..."
        grep -q "edge/testing" /etc/apk/repositories 2>/dev/null || \
            echo "https://dl-cdn.alpinelinux.org/alpine/edge/testing" >> /etc/apk/repositories
        apk update
        apk add --no-cache \
            python3 py3-pip py3-virtualenv py3-pillow curl bash git git-lfs \
            build-base cmake vulkan-headers vulkan-loader-dev mesa-vulkan-ati \
            shaderc libwebp-dev spirv-headers \
            go nodejs npm ffmpeg ca-certificates ddgr py3-yaml poppler-utils \
            pciutils usbutils util-linux psmisc bubblewrap
    elif is_arch; then
        # CachyOS/Arch use pacman and the AUR helper yay/paru if available for ddgr
        log_warn "Syncing package databases and installing essential packages (system will NOT be upgraded)."
        pacman -Sy --needed --noconfirm \
            python python-pip python-virtualenv python-pillow curl bash git git-lfs \
            base-devel cmake vulkan-headers vulkan-icd-loader vulkan-radeon \
            shaderc libwebp spirv-headers \
            go nodejs npm ffmpeg ca-certificates python-yaml \
            pciutils usbutils util-linux psmisc bubblewrap

        # Optional packages that may conflict on rolling distros; don't abort the install if they fail
        log_info "Installing optional packages..."
        pacman -S --needed --noconfirm poppler || log_warn "poppler install failed — PDF text extraction (pdftotext) may be unavailable"

        # ddgr is in the AUR; try to install it via available AUR helper
        if ! command -v ddgr >/dev/null 2>&1; then
            if command -v yay >/dev/null 2>&1; then
                log_info "Installing ddgr from AUR via yay..."
                yay -S --needed --noconfirm ddgr || log_warn "ddgr AUR install failed"
            elif command -v paru >/dev/null 2>&1; then
                log_info "Installing ddgr from AUR via paru..."
                paru -S --needed --noconfirm ddgr || log_warn "ddgr AUR install failed"
            else
                log_warn "ddgr is in the AUR. Install it manually (e.g. yay -S ddgr) if you need web search."
            fi
        fi
    elif is_fedora; then
        log_info "Installing system packages via DNF..."
        # --skip-unavailable prevents DNF from aborting the whole transaction if a package is not found
        dnf install -y --skip-unavailable \
            python3 python3-pip python3-pillow curl bash git git-lfs \
            make cmake gcc gcc-c++ vulkan-headers vulkan-loader-devel mesa-vulkan-drivers \
            libshaderc-devel glslc libwebp-devel spirv-headers-devel \
            golang nodejs npm ffmpeg ca-certificates python3-pyyaml poppler-utils \
            pciutils usbutils util-linux psmisc bubblewrap

        if ! command -v ddgr >/dev/null 2>&1; then
            log_warn "ddgr is not available in Fedora repos. Install it manually if you need web search."
        fi
    else
        log_err "Unsupported distribution: ${OS_ID}"
        log_err "This installer supports Alpine Linux, Arch-based distributions (CachyOS, Arch, Manjaro, etc.), and Fedora/RHEL."
        exit 1
    fi
    log_ok "System packages installed"
}

# ─── Subcommand: install ────────────────────────────────────────────────────
cmd_install() {
    require_root
    log_info "=== BC-250 v10 Installer (${DISTRO_NAME}) ==="
    log_info "Target: ${NETSCAN_DIR}"

    # ── 1. System dependencies ──
    install_system_packages

    # ── 1.5 Kernel GPU memory tuning (TTM) + CPU governor ──
    if prompt_yn "Apply BC-250 kernel tuning (TTM pages_limit + CPU governor)?" y; then
        log_info "Applying kernel TTM tuning..."
        if [[ -f /sys/module/ttm/parameters/pages_limit ]]; then
            echo 4194304 > /sys/module/ttm/parameters/pages_limit
            echo 4194304 > /sys/module/ttm/parameters/page_pool_size
            log_ok "TTM pages_limit set to 4194304 (16 GiB)"
        else
            log_warn "TTM module not loaded yet. Persistence will apply on next boot."
        fi
        echo "options ttm pages_limit=4194304 page_pool_size=4194304" > /etc/modprobe.d/ttm-gpu-memory.conf

        # Common tuning script used by both init systems
        mkdir -p /opt/netscan/bin
        cat > /opt/netscan/bin/bc250-kernel-tune.sh <<'EOF'
#!/bin/bash
# BC-250 kernel tuning: TTM pages + CPU governor
if [ -f /sys/module/ttm/parameters/pages_limit ]; then
    echo 4194304 > /sys/module/ttm/parameters/pages_limit
    echo 4194304 > /sys/module/ttm/parameters/page_pool_size
fi
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -f "$gov" ] && echo performance > "$gov" 2>/dev/null || true
done
EOF
        chmod +x /opt/netscan/bin/bc250-kernel-tune.sh

        if is_alpine; then
            cat > /etc/local.d/ttm.start <<'EOF'
#!/bin/sh
/opt/netscan/bin/bc250-kernel-tune.sh
EOF
            chmod +x /etc/local.d/ttm.start
            rc-update add local default 2>/dev/null || true
        elif is_systemd; then
            cat > /etc/systemd/system/bc250-kernel-tune.service <<'EOF'
[Unit]
Description=BC-250 kernel tuning (TTM + CPU governor)
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/opt/netscan/bin/bc250-kernel-tune.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
            systemctl daemon-reload
            systemctl enable --now bc250-kernel-tune.service || true
        fi

        log_info "Setting CPU governor to performance..."
        for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            if [ -f "$gov" ]; then
                echo performance > "$gov" || true
            fi
        done
        log_ok "Kernel tuning applied"
    fi

    # ── 2. Prepare directories ──
    log_info "Creating directories..."
    mkdir -p "${NETSCAN_DIR}" "${DATA_DIR}" "${MODELS_DIR}" \
        "${DATA_DIR}/career" "${DATA_DIR}/music" "${DATA_DIR}/history" \
        "${DATA_DIR}/apps" "${DATA_DIR}/saas" "${DATA_DIR}/system" \
        "${DATA_DIR}/leaks" "${DATA_DIR}/market" "${DATA_DIR}/think" \
        "${DATA_DIR}/web" "${DATA_DIR}/research"

    # ── 3. Python venv ──
    if [[ ! -d "${VENV_DIR}" ]]; then
        log_info "Creating Python virtual environment..."
        # --system-site-packages lets the venv reuse distro packages (e.g. pillow on Python 3.14)
        python3 -m venv --system-site-packages "${VENV_DIR}"
    elif [[ -f "${VENV_DIR}/pyvenv.cfg" ]] && ! grep -q '^include-system-site-packages = true' "${VENV_DIR}/pyvenv.cfg"; then
        # Existing venv was created without --system-site-packages. It will fail on Python 3.14
        # because Pillow has no official wheel and must come from the distro package.
        local venv_backup="${VENV_DIR}.bak.$(date +%s)"
        log_warn "Existing venv at ${VENV_DIR} was created without --system-site-packages."
        log_warn "Moving it to ${venv_backup} and recreating with system-site-packages enabled."
        mv "${VENV_DIR}" "${venv_backup}"
        python3 -m venv --system-site-packages "${VENV_DIR}"
    else
        log_info "Reusing existing virtual environment at ${VENV_DIR}."
    fi
    source "${VENV_DIR}/bin/activate"
    log_info "Installing Python packages..."

    # Verify the distro Pillow is visible inside the venv (critical on Python 3.14).
    # discord.py requires Pillow >= 8.0; ensure the distro package satisfies it.
    local pillow_ver pillow_status
    pillow_ver=$(check_venv_pillow 2>/dev/null)
    pillow_status=$?
    if [[ $pillow_status -eq 0 ]]; then
        log_ok "Distro Pillow ${pillow_ver} is visible inside the virtual environment"
    elif [[ $pillow_status -eq 1 ]]; then
        log_err "Virtual environment Python not found at ${VENV_DIR}/bin/python3"
        log_err "Delete ${VENV_DIR} and re-run 'sudo ${0} install'"
        exit 1
    elif [[ $pillow_status -eq 2 ]]; then
        log_err "Pillow is not installed in ${VENV_DIR}."
        log_err "Make sure the distro pillow package is installed (py3-pillow / python-pillow) and the venv was created with --system-site-packages."
        exit 1
    else
        log_err "Pillow >= 8.0 is not visible inside ${VENV_DIR} (found version ${pillow_ver:-unknown})."
        log_err "Update the distro pillow package (py3-pillow / python-pillow) or install a newer version."
        exit 1
    fi

    pip install --upgrade pip
    # Pillow is installed from the distro repos (py3-pillow / python-pillow) to avoid
    # source-compilation / wheel issues on bleeding-edge Python versions.
    # Purge pip cache to avoid reinstalling a corrupted wheel from a previous failed run.
    pip cache purge >/dev/null 2>&1 || true
    # With --system-site-packages, the distro Pillow should satisfy discord.py; if pip
    # still tries to reinstall it, the distro Pillow version is too old — update it.
    pip install discord.py aiohttp croniter huggingface_hub
    log_ok "Python venv ready"

    # ── 3.5 Install standalone hf CLI (faster than pip version) ──
    if ! command -v hf >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/hf" ]] && [[ ! -x "${VENV_DIR}/bin/hf" ]]; then
        log_info "Installing standalone hf CLI (Hugging Face)..."
        curl -LsSf https://hf.co/cli/install.sh | bash || log_warn "hf CLI install failed — will fall back to huggingface-cli from pip"
    fi
    # The standalone installer puts hf in ~/.local/bin; symlink it into the venv so
    # 'configure' and 'webui' can always find it (they look in ${VENV_DIR}/bin).
    if [[ ! -e "${VENV_DIR}/bin/hf" ]]; then
        if [[ -x "$HOME/.local/bin/hf" ]]; then
            ln -s "$HOME/.local/bin/hf" "${VENV_DIR}/bin/hf" 2>/dev/null || true
        elif command -v hf >/dev/null 2>&1; then
            ln -s "$(command -v hf)" "${VENV_DIR}/bin/hf" 2>/dev/null || true
        fi
    fi
    if command -v hf >/dev/null 2>&1 || [[ -x "${VENV_DIR}/bin/hf" ]] || [[ -x "${VENV_DIR}/bin/huggingface-cli" ]]; then
        log_ok "hf CLI ready (use: hf download <repo> <file> --local-dir ${MODELS_DIR})"
    else
        log_warn "hf CLI not available — run: pip install -U huggingface_hub"
    fi

    # ── 4. Build llama.cpp (Vulkan) — selectable fork ──
    select_llama_fork
    log_info "Cloning llama.cpp (${LLAMA_FORK_NAME})..."
    if [[ -d /opt/llama.cpp ]]; then
        # If the installed checkout is from a different fork, warn loudly and
        # default the re-clone prompt to yes (the user just picked another source).
        local _cur_remote
        _cur_remote=$(git -C /opt/llama.cpp remote get-url origin 2>/dev/null || true)
        if [[ -n "$_cur_remote" ]] && [[ "$_cur_remote" != "$LLAMA_REPO_URL" ]]; then
            log_warn "llama.cpp exists but is from a different source: ${_cur_remote}"
            if prompt_yn "Remove it and re-clone from ${LLAMA_FORK_NAME}?" y; then
                rm -rf /opt/llama.cpp
            else
                log_warn "Keeping the existing checkout — it will be rebuilt as-is (${_cur_remote})."
                # Reflect what is actually being built so later logs aren't misleading
                LLAMA_FORK_NAME="existing (${_cur_remote})"
                LLAMA_REPO_URL="$_cur_remote"
            fi
        else
            log_warn "llama.cpp already exists at /opt/llama.cpp (${LLAMA_FORK_NAME})"
            if prompt_yn "Remove and re-clone?" n; then
                rm -rf /opt/llama.cpp
            fi
        fi
    fi

    if [[ ! -d /opt/llama.cpp ]]; then
        git clone --depth 1 "${LLAMA_REPO_URL}" /opt/llama.cpp
    fi

    cd /opt/llama.cpp
    log_info "Building llama.cpp (${LLAMA_FORK_NAME}) with Vulkan support (this may take a while)..."
    # Always build from a clean tree: a stale build/ dir (objects from a
    # previous fork/commit) produces a broken binary that segfaults at load.
    rm -rf build
    cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j"$(nproc)"
    log_ok "llama.cpp built: /opt/llama.cpp/build/bin/llama-server"

    # ── 5. Build mainline llama-swap ──
    log_info "Cloning mainline llama-swap (mostlygeek)..."
    if [[ -d /opt/llama-swap ]]; then
        log_warn "llama-swap already exists at /opt/llama-swap"
        if prompt_yn "Remove and re-clone?" n; then
            rm -rf /opt/llama-swap
        fi
    fi

    if [[ ! -d /opt/llama-swap ]]; then
        git clone --depth 1 https://github.com/mostlygeek/llama-swap /opt/llama-swap
    fi

    cd /opt/llama-swap
    log_info "Building llama-swap (Go + Node UI)..."
    GOTOOLCHAIN=auto make clean all
    # Upstream Makefile now names binaries with arch suffix (e.g. llama-swap-linux-amd64)
    # Create a stable symlink so the OpenRC service and config can reference /opt/llama-swap/build/llama-swap
    local _arch
    _arch=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')
    if [[ -f "/opt/llama-swap/build/llama-swap-linux-${_arch}" ]]; then
        ln -sf "/opt/llama-swap/build/llama-swap-linux-${_arch}" /opt/llama-swap/build/llama-swap
    fi
    log_ok "llama-swap built: /opt/llama-swap/build/llama-swap"

    # ── 5.5 Build stable-diffusion.cpp (Vulkan) ──
    log_info "Cloning stable-diffusion.cpp (leejet) with Vulkan..."
    if [[ -d /opt/stable-diffusion.cpp ]]; then
        log_warn "stable-diffusion.cpp already exists at /opt/stable-diffusion.cpp"
        if prompt_yn "Remove and re-clone?" n; then
            rm -rf /opt/stable-diffusion.cpp
        fi
    fi

    if [[ ! -d /opt/stable-diffusion.cpp ]]; then
        git clone --recursive --depth 1 https://github.com/leejet/stable-diffusion.cpp /opt/stable-diffusion.cpp
    fi

    cd /opt/stable-diffusion.cpp
    log_info "Building stable-diffusion.cpp with Vulkan support..."
    # -DSD_SERVER_BUILD_FRONTEND=OFF skips the pnpm/Corepack step that installs the
    # sd-server web UI — it fails on a stock box ("Cannot verify the identity of the
    # @pnpm/exe.linux-x64 native binary") and we only need sd-cli.
    cmake -B build -DSD_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DSD_SERVER_BUILD_FRONTEND=OFF
    cmake --build build --config Release --target sd-cli -j"$(nproc)"
    log_ok "stable-diffusion.cpp built: /opt/stable-diffusion.cpp/build/bin/sd-cli"

    # ── 5.6 Build whisper.cpp (Vulkan) ──
    log_info "Cloning whisper.cpp (ggml-org) with Vulkan..."
    if [[ -d /opt/whisper.cpp ]]; then
        log_warn "whisper.cpp already exists at /opt/whisper.cpp"
        if prompt_yn "Remove and re-clone?" n; then
            rm -rf /opt/whisper.cpp
        fi
    fi

    if [[ ! -d /opt/whisper.cpp ]]; then
        git clone --depth 1 https://github.com/ggml-org/whisper.cpp /opt/whisper.cpp
    fi

    cd /opt/whisper.cpp
    log_info "Building whisper.cpp with Vulkan support..."
    cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j"$(nproc)"
    log_ok "whisper.cpp built: /opt/whisper.cpp/build/bin/whisper-cli"

    # ── 6. Copy netscan scripts ──
    log_info "Installing netscan scripts..."
    if [[ -d "${SCRIPT_DIR}/netscan" ]]; then
        cp -r "${SCRIPT_DIR}/netscan"/* "${NETSCAN_DIR}/"
        chmod +x "${NETSCAN_DIR}"/*.py "${NETSCAN_DIR}"/*.sh 2>/dev/null || true
    else
        log_warn "No local netscan/ directory found. Assuming scripts are already in ${NETSCAN_DIR}"
    fi

    # ── 7. Install jobs.json template ──
    if [[ ! -f "${DATA_DIR}/jobs.json" ]]; then
        if [[ -f "${SCRIPT_DIR}/jobs.json" ]]; then
            cp "${SCRIPT_DIR}/jobs.json" "${DATA_DIR}/jobs.json"
        elif [[ -f "${SCRIPT_DIR}/jobs.json.template" ]]; then
            cp "${SCRIPT_DIR}/jobs.json.template" "${DATA_DIR}/jobs.json"
        fi
    fi

    # ── 8. Interactive job selection ──
    log_info "=== Interactive Job Selection ==="
    log_info "Enable/disable each cron job. Press Enter for default (yes)."

    select_jobs

    # Record sync state AFTER job selection so the hash matches the final installed file
    if [[ -f "${DATA_DIR}/jobs.json" ]]; then
        sha256sum "${DATA_DIR}/jobs.json" | awk '{print $1}' > "${DATA_DIR}/.jobs.json.sha256" || true
    fi

    # ── 9. NVMe swap ──
    if prompt_yn "Configure NVMe swap (16GB, low priority)?" y; then
        log_info "Setting up NVMe swap..."
        # btrfs does not support fallocate for swap files (creates sparse files that swapon rejects)
        # On btrfs, use dd + chattr +C to disable copy-on-write
        local _swap_fs
        _swap_fs=$(stat -f -c %T "${DATA_DIR}" 2>/dev/null || echo "unknown")
        # Remove any leftover swapfile from previous failed attempts
        if [[ -f "${DATA_DIR}/swapfile" ]]; then
            swapoff "${DATA_DIR}/swapfile" 2>/dev/null || true
            rm -f "${DATA_DIR}/swapfile"
            # Also clean up fstab entry so boot doesn't try to swapon a deleted file
            sed -i "\|${DATA_DIR}/swapfile|d" /etc/fstab 2>/dev/null || true
            log_info "Removed existing swapfile from previous attempt"
        fi
        if [[ "$_swap_fs" == "btrfs" ]]; then
            log_info "btrfs detected — creating swap file with btrfs-native method"
            # btrfs requires NOCOW set BEFORE writing data; the modern way is btrfs filesystem mkswapfile (kernel 6.1+)
            if command -v btrfs >/dev/null 2>&1 && btrfs filesystem mkswapfile --help >/dev/null 2>&1; then
                btrfs filesystem mkswapfile --size 16G "${DATA_DIR}/swapfile"
            else
                # Fallback: create empty file, set NOCOW, then fill with dd
                truncate -s 16G "${DATA_DIR}/swapfile"
                chattr +C "${DATA_DIR}/swapfile"
                dd if=/dev/zero of="${DATA_DIR}/swapfile" bs=1M count=16384 status=progress
            fi
        else
            fallocate -l 16G "${DATA_DIR}/swapfile"
        fi
        chmod 600 "${DATA_DIR}/swapfile"
        mkswap "${DATA_DIR}/swapfile"
        swapon -p -2 "${DATA_DIR}/swapfile"
        if ! grep -q "${DATA_DIR}/swapfile" /etc/fstab; then
            echo "${DATA_DIR}/swapfile none swap sw,pri=-2 0 0" >> /etc/fstab
        fi
        log_ok "Swap configured"
    fi

    # ── 10. Create services (OpenRC or systemd) ──
    if is_alpine; then
        # OpenRC init scripts
        log_info "Creating OpenRC services..."
        cat > /etc/init.d/queue-runner <<'SERVEOF'
#!/sbin/openrc-run

name="Clawd Bot"
command="/opt/netscan/venv/bin/python3"
command_args="/opt/netscan/queue-runner.py"
pidfile="/run/queue-runner.pid"
command_background="yes"

start_pre() {
    # Same stale-pidfile trap as llama-swap: after a hard death 'start' refuses
    # ("already started") and the bot never comes back up.
    if [ -f "$pidfile" ] && ! kill -0 "$(cat "$pidfile" 2>/dev/null)" 2>/dev/null; then
        ewarn "stale pidfile $pidfile (PID is gone): removing it"
        rm -f "$pidfile"
    fi
}

start() {
    ebegin "Starting Clawd Bot"
    # Load environment variables from .env
    if [ -f /opt/netscan/.env ]; then
        while IFS= read -r line; do
            case "$line" in
                \#*|"") continue ;;
                *=*) eval "export \"$line\"" ;;
            esac
        done < /opt/netscan/.env
    fi
    start-stop-daemon --start --background \
        --make-pidfile --pidfile "$pidfile" \
        --exec "$command" -- $command_args
    eend $?
}

stop() {
    ebegin "Stopping Clawd Bot"
    start-stop-daemon --stop --pidfile "$pidfile"
    eend $?
}

depend() {
    need net
    after llama-swap
}
SERVEOF
        chmod +x /etc/init.d/queue-runner
        enable_service queue-runner

        cat > /etc/init.d/llama-swap <<'SERVEOF'
#!/sbin/openrc-run

name="llama-swap"
command="/opt/llama-swap/build/llama-swap"
command_args="--config /opt/llama-swap/config.yaml --listen 0.0.0.0:8080"
pidfile="/run/llama-swap.pid"
command_background="yes"

start_pre() {
    if [ ! -x "$command" ]; then
        eerror "llama-swap binary not found: $command"
        return 1
    fi
    if [ ! -f /opt/llama-swap/config.yaml ]; then
        eerror "config.yaml not found: /opt/llama-swap/config.yaml"
        eerror "Run 'install.sh configure' to create it."
        return 1
    fi
    # A stale pidfile (the process died hard: OOM, kill, crash) leaves the service
    # wedged: 'start' answers "already started" and never actually starts, while
    # 'status' says crashed. Clear it here so a restart always works.
    if [ -f "$pidfile" ] && ! kill -0 "$(cat "$pidfile" 2>/dev/null)" 2>/dev/null; then
        ewarn "stale pidfile $pidfile (PID is gone): removing it"
        rm -f "$pidfile"
    fi
}

depend() {
    need net
}
SERVEOF
        chmod +x /etc/init.d/llama-swap
        enable_service llama-swap
        log_ok "OpenRC services created and enabled"

    elif is_systemd; then
        # systemd unit files
        log_info "Creating systemd services..."
        cat > /etc/systemd/system/llama-swap.service <<'SERVEOF'
[Unit]
Description=llama-swap LLM proxy
After=network.target

[Service]
Type=simple
Restart=always
RestartSec=5
ExecStart=/opt/llama-swap/build/llama-swap --config /opt/llama-swap/config.yaml --listen 0.0.0.0:8080

[Install]
WantedBy=multi-user.target
SERVEOF

        # Wrapper to source .env with quoted values (systemd EnvironmentFile does not strip quotes)
        mkdir -p /opt/netscan/bin
        cat > /opt/netscan/bin/queue-runner-wrapper.sh <<'SERVEOF'
#!/bin/bash
# Load .env and preserve quoted values
if [ -f /opt/netscan/.env ]; then
    set -a
    source /opt/netscan/.env
    set +a
fi
exec /opt/netscan/venv/bin/python3 /opt/netscan/queue-runner.py
SERVEOF
        chmod +x /opt/netscan/bin/queue-runner-wrapper.sh

        cat > /etc/systemd/system/queue-runner.service <<'SERVEOF'
[Unit]
Description=Clawd Bot (BC-250 queue runner)
After=network.target llama-swap.service
Wants=llama-swap.service

[Service]
Type=simple
ExecStart=/opt/netscan/bin/queue-runner-wrapper.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVEOF
        systemctl daemon-reload
        systemctl enable llama-swap.service queue-runner.service
        log_ok "systemd services created and enabled"
    fi

    # ── 11. Create default config.yaml if missing ──
    if [[ ! -f /opt/llama-swap/config.yaml ]]; then
        cat > /opt/llama-swap/config.yaml <<'YAMLEOF'
models:
  # Run 'install.sh configure' to add models interactively
YAMLEOF
        log_ok "Created default /opt/llama-swap/config.yaml"
    fi

    # ── 12. Optional: local SearXNG ──
    # Asked rather than assumed, and skipped entirely with --no-searxng or when
    # stdin is not a terminal (unattended installs must not hang on a prompt).
    local _want_searxng=1
    for _arg in "$@"; do
        if [[ "$_arg" == "--no-searxng" ]]; then _want_searxng=0; fi
    done
    if [[ "$_want_searxng" == "1" ]]; then
        if [[ -t 0 ]]; then
            prompt_yn "Install the local SearXNG search service (~160MB RAM, podman)?" y \
                && install_searxng
        else
            install_searxng
        fi
    else
        log_info "Skipping SearXNG (--no-searxng). Jobs will use ddgr."
    fi

    log_info ""
    log_ok "=== INSTALL COMPLETE ==="
    log_info ""
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  📋 BEFORE running '${0} configure', gather the following:"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo ""
    echo "  1. 🦞 DISCORD BOT + RESEARCH AGENT"
    echo "     • Create a bot at https://discord.com/developers/applications"
    echo "     • Copy the Bot Token (keep it secret — it will be hidden on input)"
    echo "     • Copy your Discord User ID (right-click your name in Discord)"
    echo "     • Research agent: ddgr (web search) + git + sandboxed cmd execution"
    echo "     • Deep research: say \"investiga...\" or \"busca...\" in Discord"
    echo ""
    echo "  2. 🧠 LLM MODELS (place .gguf files in ${MODELS_DIR})"
    echo "     Primary chat+vision: gemma4-12b-q3.gguf  (Gemma 4 12B, Q4_K_M — native multimodal)"
    echo "     Fallback:            qwen3.5-35b-a3b-iq2m.gguf  (Qwen MoE, IQ2_M)"
    echo "     "
    echo "     Download Gemma 4 12B (multimodal — reads images + text):"
    echo "       hf download bartowski/gemma-4-12B-it-GGUF gemma-4-12B-it-Q4_K_M.gguf --local-dir ${MODELS_DIR}"
    echo "       hf download bartowski/gemma-4-12B-it-GGUF mmproj-gemma-4-12B-it-f16.gguf --local-dir ${MODELS_DIR}"
    echo "     "
    echo "     The mmproj file enables vision — llama-swap loads it with --mmproj flag."
    echo "     MTP (Multi-Token Prediction): --spec-type draft-mtp speeds up generation but keeps"
    echo "     a draft head and its own cache in memory. On a 16 GB box a 27B that otherwise fits"
    echo "     at -c 90000 dies loading with 'upstream command exited prematurely'. Enable it only"
    echo "     with room to spare (or on a smaller model)."
    echo "     DFlash (block diffusion): bee.cpp fork + *-DFlash*.gguf draft → --spec-type draft-dflash"
    echo ""
    echo "  3. 🎨 IMAGE GENERATION (stable-diffusion.cpp + Vulkan)"
    echo "     ┌─ Z-Image Turbo: FAST default (8 steps, cfg 1.0, ~6GB total)"
    echo "     │  Model:     ${sd_base_dir:-/opt/stable-diffusion.cpp/models}/z_image/z_image_turbo-Q4_K.gguf (~3.7GB)"
    echo "     │  LLM:       ${sd_base_dir:-/opt/stable-diffusion.cpp/models}/z_image/Qwen3-4B-Instruct-2507-Q4_K_M.gguf (~2.4GB)"
    echo "     │  VAE:       reuses ${sd_base_dir:-/opt/stable-diffusion.cpp/models}/flux/ae.safetensors"
    echo "     │"
    echo "     ├─ Qwen-Image 2.1: SOTA unified generation + instruction editing (7B DiT)"
    echo "     │  Model:     ${sd_base_dir:-/opt/stable-diffusion.cpp/models}/qwen_image/qwen_image_2.1-Q4_K.gguf (~3.9GB)"
    echo "     │             (uncensored drop-in: qwen-image-2.1-UC-Q4_K_M.gguf, same VAE/encoder)"
    echo "     │  VAE:       ${sd_base_dir:-/opt/stable-diffusion.cpp/models}/qwen_image/vae/qwen_image_2.1_vae_bf16.safetensors (~680MB)"
    echo "     │  LLM:       ${sd_base_dir:-/opt/stable-diffusion.cpp/models}/qwen_image/Qwen3VL-8B-Instruct-Q4_K_M.gguf (~4.7GB)"
    echo "     │  Vision:    ${sd_base_dir:-/opt/stable-diffusion.cpp/models}/qwen_image/mmproj-Qwen3VL-8B-Instruct-F16.gguf (~1.1GB)"
    echo "     │"
    echo "     ├─ HiDream-O1 (unified pixel transformer, no VAE/CLIP/T5)"
    echo "     │  Download:  hf download Comfy-Org/HiDream-O1-Image checkpoints/hidream_o1_image_dev_bf16.safetensors --local-dir /opt/stable-diffusion.cpp/models/hidream/"
    echo "     │  Model:     /opt/stable-diffusion.cpp/models/hidream/HiDream-O1-Image-Dev-BF16.safetensors (~16.4GB)"
    echo "     │  VRAM:      fits 16GB via --tensor-type-rules (gate/up_proj+attn → Q8_0, see #1496)"
    echo "     │"
    echo "     └─ FALLBACK: FLUX (latent diffusion, needs VAE + text encoders)"
    echo "        FLUX.2-9B: /opt/stable-diffusion.cpp/models/flux2/flux-2-klein-9b-Q4_0.gguf"
    echo "        FLUX.1:    /opt/stable-diffusion.cpp/models/flux/flux1-schnell-q4_k.gguf + ae.safetensors"
    echo "     ESRGAN:           /opt/stable-diffusion.cpp/models/esrgan/RealESRGAN_x4plus.pth"
    echo ""
    echo "  4. 🎬 VIDEO GENERATION (WAN 2.2 — two-pass high-noise + low-noise)"
    echo "     T2V HighNoise: /opt/stable-diffusion.cpp/models/wan/HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf"
    echo "     T2V LowNoise:  /opt/stable-diffusion.cpp/models/wan/LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf"
    echo "     I2V: attach an image in Discord + generate-video <prompt> to animate it"
    echo "     I2V models:    /opt/stable-diffusion.cpp/models/wan/{HighNoise,LowNoise}/Wan2.2-I2V-A14B-*-Q4_K_M.gguf"
    echo "     WAN VAE:          /opt/stable-diffusion.cpp/models/wan/wan_2.1_vae.safetensors"
    echo "     umt5-xxl encoder: /opt/stable-diffusion.cpp/models/wan/umt5-xxl-encoder-Q4_K_M.gguf"
    echo ""
    echo "  5. 📝 OPTIONAL FILES"
    echo "     cv_master.md  — your skills for job-tailoring (placed in ${NETSCAN_DIR})"
    echo "     profile.json  — your identity/preferences (auto-created if missing)"
    echo ""
    echo ""
    echo "  6. 🚀 START THE BOT"
    if is_systemd; then
        echo "     sudo systemctl start llama-swap"
        echo "     sudo systemctl start queue-runner"
    else
        echo "     sudo rc-service llama-swap start"
        echo "     sudo rc-service queue-runner start"
    fi
    echo ""
    echo "  7. 💤 MODE SWITCHING (from Discord)"
    echo "     'dormir' / 'bot off' / 'apagar bot'  → Free VRAM, stop llama-swap"
    echo "     'despertar' / 'bot on' / 'activar bot'  → Reload LLM, resume bot"
    echo "     'inteligente' / 'smart mode'         → Smart mode (potent model, pause cron)"
    echo "     'asistente'                          → Normal mode (resume everything)"
    echo ""
    echo "  7b. / SLASH COMMANDS (type '/' in Discord — deterministic, no LLM needed)"
    echo "     /estado              → Mode, model, chat queue, pending approvals, llama-swap health"
    echo "     /modo <modo>         → Same mode switches as above, without keyword guessing"
    echo "     /imagen <prompt>     → Generate an image (delegated to the worker thread)"
    echo "     /investigar <tema>   → Deep-research session"
    echo "     /olvidar <clave>     → Delete a saved preference"
    echo "     /ayuda               → Command list"
    echo "     (Set DISCORD_GUILD_ID in .env for instant sync; global sync takes up to 1h)"
    echo ""
    echo "  7c. ⚡ STREAMING + NON-BLOCKING CHAT"
    echo "     The answer streams into the Discord message as tokens arrive (~2s edits)."
    echo "     Chat runs in its own thread: a long cron job no longer mutes the bot."
    echo "     Media generation defers while a job holds the GPU instead of killing it."
    echo "     Knobs: CHAT_STREAMING=0 (off), STREAM_EDIT_INTERVAL_S (default 2.0)"
    echo ""
    echo "  8. 📄 PDF SUPPORT (via Discord — send any .pdf file)"
    echo "     Text PDFs: extracted with pdftotext (poppler-utils)"
    echo "     Scanned PDFs: OCR via Gemma 4 vision (pages sent as images, 2 at a time)"
    echo "     Max 50 pages — very large PDFs are truncated with a warning"
    echo ""
    echo "  9. 🎤 AUDIO SUPPORT (send voice messages in Discord)"
    echo "     Native: Gemma 4 reads audio directly via input_audio API"
    echo "     Fallback: whisper.cpp transcription (ggml-large-v3-turbo)"
    echo "     Format: Discord OGG → ffmpeg → WAV 16kHz → model"
    echo ""
    echo "  10. 🧠 SELF-IMPROVEMENT & TOKEN EFFICIENCY (the bot learns and optimizes)"
    echo "     Memory: save-memory, add-lesson, add-correction (persistent JSON)"
    echo "     Auto-summarize: lessons & corrections compressed into profile by LLM (saves tokens)"
    echo "     tool-help: lazy tool loading — full syntax only when requested (saves tokens)"
    echo "     Skills: add-function, list-functions (reusable tool registry)"
    echo "     Auto-scripts: propose-script → Discord approval → /opt/netscan/skills/"
    echo ""
    echo "  11. 🌐 WEB UI (temporary config interface)"
    echo "     sudo ${0} webui [port]  → Opens browser-based config (auto-shutdown 30 min)"
    echo ""
    echo "  Next step:  sudo ${0} configure"
    echo "═══════════════════════════════════════════════════════════════════════"
    log_info ""
}

# ─── Job selection helper ────────────────────────────────────────────────────
select_jobs() {
    local jobs_file="${DATA_DIR}/jobs.json"

    if [[ ! -f "$jobs_file" ]]; then
        log_warn "No jobs.json found. Skipping job selection."
        return
    fi

    # Skip interactive selection if stdin is not a terminal
    if [[ ! -t 0 ]]; then
        log_warn "Non-interactive mode detected. Skipping job selection (keeping defaults)."
        return
    fi

    # Use TUI job manager if available
    if [[ -f "${NETSCAN_DIR}/jobs-tui.py" ]]; then
        log_info "Opening job manager TUI..."
        python3 "${NETSCAN_DIR}/jobs-tui.py" "$jobs_file"
    else
        log_warn "jobs-tui.py not found. Skipping job selection."
    fi
}

# ─── configure_models ────────────────────────────────────────────────────────
# Set up llama-swap models ONLY: scan ${MODELS_DIR}, run the interactive model
# selection menu, rewrite /opt/llama-swap/config.yaml, and sync the LLM role
# variables (primary/fallback/smart) to .env. Shared by 'configure' (which
# passes its $do_write_env flag) and the standalone 'configure-models'
# subcommand (which always writes the roles).
# $1: 0/1 — write the LLM_* roles back to .env (default 1).
configure_models() {
    local write_env="${1:-1}"
    local llm_model="" llm_fallback="" llm_smart="" first_alias=""

    log_info "--- llama-swap Configuration ---"

    if [[ ! -x /opt/llama-swap/build/llama-swap ]]; then
        log_warn "llama-swap binary not found at /opt/llama-swap/build/llama-swap"
        log_warn "Run '${0} install' first before configuring models."
    fi

    mkdir -p /opt/llama-swap

    # Scan model dir and classify: main models / mmproj (vision) / MTP drafts / DFlash drafts
    local models=() mmprojs=() mtps=() dflashs=()
    if [[ -d "${MODELS_DIR}" ]]; then
        local _f _base
        while IFS= read -r -d '' _f; do
            _base=$(basename "$_f")
            if [[ "$_base" == *mmproj* ]]; then
                mmprojs+=("$_f")
            elif [[ "${_base,,}" == *dflash* ]]; then
                # DFlash block-diffusion draft (bee.cpp / llama.cpp PR #22105).
                # Case-insensitive: real files are named *-DFlash-*.gguf.
                dflashs+=("$_f")
            elif [[ "$_base" == mtp-* ]] || [[ "$_base" == *-mtp* ]] || [[ "$_base" == *draft* ]]; then
                # Separate MTP draft files (mtp-*.gguf, *-mtp*.gguf, *draft*)
                mtps+=("$_f")
            else
                models+=("$_f")
            fi
        done < <(find "${MODELS_DIR}" -maxdepth 2 -name "*.gguf" -print0 2>/dev/null || true)
    fi

    log_info "Setting up llama-swap config.yaml..."
    log_info "Found: ${#models[@]} model(s), ${#mmprojs[@]} mmproj(s), ${#mtps[@]} MTP draft(s), ${#dflashs[@]} DFlash draft(s) in ${MODELS_DIR}"
    log_info "Pick a model by number; enter 0 (or just press Enter) when finished."

    # Build config.yaml
    cat > /opt/llama-swap/config.yaml <<'YAMLEOF'
models:
YAMLEOF

    while true; do
        echo ""
        echo -e "  ${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
        echo -e "  ${YELLOW}🧠 MODEL SELECTION${NC} — choose a model to add"
        echo -e "  ${BLUE}═══════════════════════════════════════════════════════════════════${NC}"

        if [[ ${#models[@]} -gt 0 ]]; then
            local _i=1
            for _m in "${models[@]}"; do
                printf "    ${GREEN}[%s]${NC} %s\n" "$_i" "$(basename "$_m")"
                _i=$((_i+1))
            done
            echo -e "    ${YELLOW}[c]${NC} enter a custom .gguf path"
            echo -e "    ${YELLOW}[0]${NC} finish"
            prompt_input "  Select model: " choice
        else
            log_warn "No .gguf models detected in ${MODELS_DIR} — you can still enter a path manually."
            log_info "Download models with: hf download <repo> <file> --local-dir ${MODELS_DIR}"
            prompt_input "  Enter .gguf path (0 = finish): " choice
        fi

        # Finish
        if [[ -z "$choice" || "$choice" == "0" ]]; then
            break
        fi

        if [[ ${#models[@]} -eq 0 ]]; then
            # No detected models: treat the input directly as a path
            gguf_path="$choice"
            if [[ ! -f "$gguf_path" ]]; then
                log_warn "File not found: $gguf_path"
                if ! prompt_yn "  Use anyway?" n; then
                    continue
                fi
            fi
        elif [[ "$choice" == "c" || "$choice" == "C" ]]; then
            # Custom path
            prompt_input "  Full path to .gguf file: " gguf_path
            if [[ -z "$gguf_path" ]]; then
                continue
            fi
            if [[ ! -f "$gguf_path" ]]; then
                log_warn "File not found: $gguf_path"
                if ! prompt_yn "  Use anyway?" n; then
                    continue
                fi
            fi
        else
            # Numbered choice
            if ! [[ "$choice" =~ ^[0-9]+$ ]] || [[ "$choice" -lt 1 ]] || [[ "$choice" -gt ${#models[@]} ]]; then
                log_warn "Invalid choice: '$choice'"
                continue
            fi
            gguf_path="${models[$((choice-1))]}"
        fi

        # Suggest alias from filename
        local _hint
        _hint=$(basename "$gguf_path" .gguf)
        prompt_input "  Model alias [${_hint}]: " alias
        alias="${alias:-$_hint}"

        echo ""
        echo -e "  ${BLUE}── ⚙️  Options for '${alias}' (press Enter to accept defaults) ──${NC}"
        prompt_input "    • Context size  [65536] (e.g. 8192 / 32768 / 131072): " ctx
        ctx="${ctx:-65536}"
        prompt_input "    • KV cache type [q4_0] (q4_0 / q8_0 / f16): " kv
        kv="${kv:-q4_0}"
        prompt_input "    • GPU layers    [99] (0 = CPU only): " ngl
        ngl="${ngl:-99}"
        prompt_input "    • Temperature   [0.7] (0.0 – 2.0): " temp
        temp="${temp:-0.7}"
        prompt_input "    • Flash attention [y/n] (y): " fa
        fa="${fa:-y}"
        local fa_flag=""
        # NOTE: modern llama.cpp (2026) requires a value: --flash-attn on|off|auto.
        # A bare --flash-attn makes the parser swallow the NEXT flag as its value
        # ("unknown value for --flash-attn: '--mmproj'") and llama-server exits →
        # llama-swap 500 "upstream command exited prematurely".
        [[ "$fa" == "y" ]] && fa_flag=" --flash-attn on"

        # Vision projector (mmproj) — auto-offer detected files
        local mmproj_flag=""
        if [[ ${#mmprojs[@]} -gt 0 ]]; then
            echo ""
            echo -e "  ${BLUE}── 🖼️  VISION — mmproj detected, attach for image/audio support ──${NC}"
            local _j=1
            for _mp in "${mmprojs[@]}"; do
                printf "    ${GREEN}[%s]${NC} %s\n" "$_j" "$(basename "$_mp")"
                _j=$((_j+1))
            done
            echo -e "    ${YELLOW}[0]${NC} no vision"
            prompt_input "  Attach mmproj (number): " mchoice
            if [[ "$mchoice" =~ ^[0-9]+$ ]] && [[ "$mchoice" -ge 1 ]] && [[ "$mchoice" -le ${#mmprojs[@]} ]]; then
                mmproj_flag=" --mmproj ${mmprojs[$((mchoice-1))]}"
            fi
        elif prompt_yn "  Does this model support vision/audio (needs mmproj)?" n; then
            prompt_input "    mmproj file path: " mmproj_path_cfg
            if [[ -n "$mmproj_path_cfg" ]]; then
                if [[ ! -f "$mmproj_path_cfg" ]]; then
                    log_warn "  File not found: $mmproj_path_cfg"
                    if ! prompt_yn "  Use anyway?" n; then
                        mmproj_path_cfg=""
                    fi
                fi
                [[ -n "$mmproj_path_cfg" ]] && mmproj_flag=" --mmproj ${mmproj_path_cfg}"
            fi
        fi

        # Speculative decoding: MTP (multi-token prediction) or DFlash (block
        # diffusion). MTP needs either a built-in MTP head (Gemma 4, Qwen3.5+)
        # or a separate mtp-*.gguf draft. DFlash needs a dedicated *-dflash*.gguf
        # block-diffusion draft AND a DFlash-capable build (bee.cpp / llama.cpp
        # with PR #22105) — plain TurboQuant/mainline may not support draft-dflash.
        local spec_flag=""
        echo ""
        echo -e "  ${BLUE}── ⚡ SPECULATIVE DECODING (MTP / DFlash, faster generation) ──${NC}"
        if [[ ${#mtps[@]} -gt 0 ]]; then
            local _k=1
            for _mt in "${mtps[@]}"; do
                printf "    ${GREEN}[%s]${NC} MTP draft: %s\n" "$_k" "$(basename "$_mt")"
                _k=$((_k+1))
            done
        fi
        if [[ ${#dflashs[@]} -gt 0 ]]; then
            local _dk=1
            for _df in "${dflashs[@]}"; do
                printf "    ${GREEN}[f%s]${NC} DFlash draft: %s\n" "$_dk" "$(basename "$_df")"
                _dk=$((_dk+1))
            done
        fi
        echo -e "    ${YELLOW}[b]${NC} MTP bundled INSIDE the main model"
        echo -e "    ${YELLOW}[0]${NC} disable speculative decoding"
        prompt_input "  Spec choice (Enter = disable): " specchoice
        local mtp_n=2
        if [[ -z "$specchoice" || "$specchoice" == "0" ]]; then
            log_info "  Speculative decoding disabled"
        elif [[ "$specchoice" == "b" ]]; then
            log_warn "  Requires a model with a built-in MTP head (e.g. Gemma 4, Qwen3.5+)."
            prompt_input "    MTP draft tokens (1-4, default 2): " mtp_n
            mtp_n="${mtp_n:-2}"
            [[ "$mtp_n" =~ ^[1-4]$ ]] || mtp_n=2
            spec_flag=" --spec-type draft-mtp --spec-draft-n-max ${mtp_n}"
            log_ok "  MTP bundled in main model enabled (${mtp_n} draft tokens)"
        elif [[ "$specchoice" =~ ^[0-9]+$ ]] && [[ "$specchoice" -ge 1 ]] && [[ "$specchoice" -le ${#mtps[@]} ]]; then
            prompt_input "    MTP draft tokens (1-4, default 2): " mtp_n
            mtp_n="${mtp_n:-2}"
            [[ "$mtp_n" =~ ^[1-4]$ ]] || mtp_n=2
            spec_flag=" --spec-type draft-mtp --spec-draft-model ${mtps[$((specchoice-1))]} --spec-draft-n-max ${mtp_n}"
            log_ok "  MTP enabled with separate draft: $(basename "${mtps[$((specchoice-1))]}")"
        elif [[ "$specchoice" =~ ^f[0-9]+$ ]]; then
            # DFlash: dedicated block-diffusion draft. Draft depth (n-max) is read
            # from the model metadata (block_size - 1), so no manual depth needed.
            local _di=$(( ${specchoice#f} - 1 ))
            if [[ "$_di" -ge 0 ]] && [[ "$_di" -lt ${#dflashs[@]} ]]; then
                spec_flag=" --spec-type draft-dflash --spec-draft-model ${dflashs[$_di]} --spec-draft-ngl all --spec-dm-controller profit"
                log_ok "  DFlash enabled with draft: $(basename "${dflashs[$_di]}")"
            else
                log_warn "  Invalid DFlash choice '$specchoice' — speculative decoding disabled"
            fi
        else
            log_warn "  Invalid choice '$specchoice' — speculative decoding disabled"
        fi

        # Extra llama.cpp arguments (one per line, empty to finish)
        log_info "Extra llama.cpp args (one per line, e.g. --log-disable). Enter empty line when done:"
        local extra_args=""
        while true; do
            prompt_input "    extra arg: " extra
            [[ -z "$extra" ]] && break
            extra_args="${extra_args} ${extra}"
        done

        cat >> /opt/llama-swap/config.yaml <<EOF
  ${alias}:
    cmd: /opt/llama.cpp/build/bin/llama-server --port \${PORT} --model ${gguf_path} -ngl ${ngl} -c ${ctx} -ctk ${kv} -ctv ${kv}${fa_flag}${mmproj_flag}${spec_flag} --temp ${temp} --host 127.0.0.1${extra_args}
    ttl: 600

EOF
        log_ok "Added model: ${alias}  (${gguf_path})"

        # Remember the first added model so we can default the primary role to it
        [[ -z "${first_alias:-}" ]] && first_alias="${alias}"

        # Assign a role (primary / fallback / smart) so .env stays in sync
        echo ""
        echo -e "  Model role: ${GREEN}[p]${NC} primary  ${GREEN}[f]${NC} fallback  ${GREEN}[s]${NC} smart  ${YELLOW}[0]${NC} none"
        prompt_input "  Role for this model (Enter = none): " role
        case "$role" in
            p|P) llm_model="${alias}"; log_ok "  → LLM_MODEL (primary) = ${alias}";;
            f|F) llm_fallback="${alias}"; log_ok "  → LLM_FALLBACK_MODEL = ${alias}";;
            s|S) llm_smart="${alias}"; log_ok "  → LLM_SMART_MODEL = ${alias}";;
            0|"") :;;
            *)   log_warn "  Invalid role '$role' — no role assigned";;
        esac
    done

    # Apply model roles to .env (assigned during model selection)
    if [[ $write_env -eq 1 ]]; then
        # If no primary role was chosen but models were added, default to the first one
        if [[ -z "${llm_model:-}" ]] && [[ -n "${first_alias:-}" ]]; then
            llm_model="${first_alias}"
            log_info "  No primary chosen — defaulting LLM_MODEL to first model: ${first_alias}"
        fi
        if [[ -f "${NETSCAN_DIR}/.env" ]]; then
            if [[ -n "${llm_model:-}" ]]; then
                sed -i "s|^LLM_MODEL=.*|LLM_MODEL=\"$(sed_repl_escape "$llm_model")\"|" "${NETSCAN_DIR}/.env"
            fi
            if [[ -n "${llm_fallback:-}" ]]; then
                sed -i "s|^LLM_FALLBACK_MODEL=.*|LLM_FALLBACK_MODEL=\"$(sed_repl_escape "$llm_fallback")\"|" "${NETSCAN_DIR}/.env"
            fi
            if [[ -n "${llm_smart:-}" ]]; then
                sed -i "s|^LLM_SMART_MODEL=.*|LLM_SMART_MODEL=\"$(sed_repl_escape "$llm_smart")\"|" "${NETSCAN_DIR}/.env"
            fi
        else
            log_warn "${NETSCAN_DIR}/.env not found — creating a minimal one with the LLM roles."
            cat > "${NETSCAN_DIR}/.env" <<EOF
LLM_MODEL="${llm_model:-}"
LLM_FALLBACK_MODEL="${llm_fallback:-}"
LLM_SMART_MODEL="${llm_smart:-}"
LLM_URL="http://localhost:8080/v1"
EOF
            chmod 600 "${NETSCAN_DIR}/.env"
        fi
        log_info "LLM roles in .env: primary=${llm_model:-<default>} fallback=${llm_fallback:-<default>} smart=${llm_smart:-<none>}"
    elif [[ -n "${llm_model:-}" || -n "${llm_fallback:-}" || -n "${llm_smart:-}" ]]; then
        log_warn "Roles assigned in the model menu were NOT written: you chose to keep the existing .env."
        log_warn "Edit ${NETSCAN_DIR}/.env manually (LLM_MODEL / LLM_FALLBACK_MODEL / LLM_SMART_MODEL)."
    fi

    # Add example fallback if nothing was added
    if [[ ! -s /opt/llama-swap/config.yaml ]] || [[ $(grep -c "cmd:" /opt/llama-swap/config.yaml || true) -eq 0 ]]; then
        log_warn "No models configured. Adding a template example."
        cat >> /opt/llama-swap/config.yaml <<'EOF'
  # Example — replace with your actual model path:
  #  gemma4-12b-q3:
  #   cmd: /opt/llama.cpp/build/bin/llama-server --port ${PORT} --model /opt/llama-models/gemma4-12b-q3.gguf -ngl 99 -c 65536 -ctk q4_0 -ctv q4_0 --flash-attn on --mmproj /opt/llama-models/mmproj-gemma-4-12B-it-f16.gguf --host 127.0.0.1
  #   ttl: 600
EOF
    fi

    log_ok "llama-swap config written to /opt/llama-swap/config.yaml"

    # Validate YAML syntax AND duplicate keys. PyYAML's safe_load silently keeps
    # the last duplicate, while llama-swap's Go parser refuses the file outright:
    # the service then crashes on start and nothing listens on 8080. So a
    # repeated model name has to fail here, not at runtime.
    log_info "Validating llama-swap YAML..."
    if ! python3 -c "import yaml" 2>/dev/null; then
        log_warn "py3-yaml not installed. Skipping YAML validation."
    else
        local _yaml_err
        _yaml_err="$(python3 - "/opt/llama-swap/config.yaml" <<'PYEOF' 2>&1 || true
import sys, yaml

class NoDup(yaml.SafeLoader):
    pass

def _no_dup(loader, node, deep=False):
    seen = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, "duplicate key: %r" % (key,), key_node.start_mark)
        seen[key] = loader.construct_object(value_node, deep=deep)
    return seen

NoDup.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dup)
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        yaml.load(fh, Loader=NoDup)
except Exception as exc:
    print(exc)
PYEOF
)"
        if [ -z "$_yaml_err" ]; then
            log_ok "YAML syntax is valid (no duplicate keys)"
        else
            log_warn "YAML validation failed — llama-swap rejects duplicate keys and then crashes:"
            printf '%s\n' "$_yaml_err"
            log_warn "Fix: give each model under 'models:' a unique name, then rc-service llama-swap restart."
        fi
    fi
}

# ─── Subcommand: configure ──────────────────────────────────────────────────
cmd_configure() {
    require_root
    log_info "=== BC-250 v10 Configuration (${DISTRO_NAME}) ==="

    # ── 1. Ensure directories exist ──
    mkdir -p "${NETSCAN_DIR}" "${DATA_DIR}" "${MODELS_DIR}"

    # ── 2. Discord .env ──
    log_info "--- Discord Bot Configuration ---"
    log_info "Create your bot at: https://discord.com/developers/applications"
    log_info "Get your User ID: right-click your name in Discord -> Copy User ID"

    local discord_token=""
    local discord_owner=""
    local llm_model=""
    local llm_fallback=""
    local llm_smart=""
    local hf_perf=""
    local do_write_env=1

    if [[ -f "${NETSCAN_DIR}/.env" ]]; then
        log_warn "Existing .env found."
        if ! prompt_yn "Overwrite existing .env?" n; then
            log_info "Keeping existing .env. Edit ${NETSCAN_DIR}/.env manually if needed."
            do_write_env=0
        fi
    fi

    if [[ $do_write_env -eq 1 ]]; then
        read_secret "Discord Bot Token (hidden): " discord_token
        while true; do
            prompt_input "Your Discord User ID (numbers only): " discord_owner
            if [[ "$discord_owner" =~ ^[0-9]+$ ]]; then
                break
            fi
            log_warn "Invalid ID. Must be numeric only."
        done
        log_info "Discord owner ID: ${discord_owner}"
        log_info "Model roles (primary/fallback/smart) are assigned later, in the model selection menu."

        log_info ""
        log_info "--- Stable Diffusion Model Configuration ---"
        log_info "Leave blank to auto-detect from directories. Supports .gguf and .safetensors."
        prompt_input "SD models base dir [/opt/stable-diffusion.cpp/models]: " sd_base
        prompt_input "Qwen-Image 2.1 model file (e.g. qwen_image_2.1-Q4_K.gguf, Enter=auto): " sd_qwen_model
        prompt_input "Qwen-Image 2.1 VAE (e.g. qwen_image_2.1_vae_bf16.safetensors, Enter=auto): " sd_qwen_vae
        prompt_input "Qwen-Image 2.1 LLM encoder (e.g. Qwen3VL-8B-Instruct-Q4_K_M.gguf, Enter=auto): " sd_qwen_llm
        prompt_input "Qwen-Image 2.1 vision projector (for editing, e.g. mmproj-Qwen3VL-8B-Instruct-F16.gguf, Enter=auto): " sd_qwen_vision
        prompt_input "FLUX.1 model file (e.g. flux1-schnell-q4_k.gguf, Enter=auto): " sd_flux_model
        prompt_input "FLUX.2 diffusion model (e.g. flux-2-klein-9b-Q4_0.gguf, Enter=auto): " sd_flux2_model
        prompt_input "FLUX.2 VAE (e.g. flux2-vae.safetensors, Enter=auto): " sd_flux2_vae
        prompt_input "FLUX.2 LLM encoder (e.g. qwen3-8b-Q4_K_M.gguf, Enter=auto): " sd_flux2_llm
        prompt_input "FLUX Kontext (editing, e.g. flux1-kontext-dev-Q4_K_M.gguf, Enter=auto): " sd_kontext
        prompt_input "WAN video model (e.g. Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf, Enter=auto): " sd_wan_model
        prompt_input "WAN high-noise model (2.2 two-pass, e.g. Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf, Enter=auto): " sd_wan_high_noise
        prompt_input "HiDream-O1 model (PRIMARY, e.g. HiDream-O1-Image-Dev-BF16.safetensors, Enter=auto): " sd_hidream
        prompt_input "ESRGAN upscale model (Enter=auto): " sd_esrgan

        log_info ""
        log_info "--- Hugging Face Configuration ---"
        if prompt_yn "Configure Hugging Face (for model downloads)?" y; then
            log_info "Get your free token at: https://huggingface.co/settings/tokens"
            local _hf_cli
            if ! _hf_cli=$(find_hf_cli); then
                # Auto-install the standalone hf CLI (https://hf.co/cli)
                log_info "Installing standalone hf CLI..."
                curl -LsSf https://hf.co/cli/install.sh | bash >/dev/null 2>&1 || log_warn "hf CLI install failed — will fall back to huggingface-cli from pip"
                _hf_cli=$(find_hf_cli) || true
            fi
            if [[ -n "$_hf_cli" ]]; then
                # Already logged in? (hf auth whoami returns 0 when authenticated)
                if [[ "$(basename "$_hf_cli")" == "hf" ]] && "$_hf_cli" auth whoami >/dev/null 2>&1; then
                    log_ok "Already logged in to Hugging Face — skipping login"
                else
                    log_info "Logging in to Hugging Face (token stored locally)..."
                    if [[ "$(basename "$_hf_cli")" == "hf" ]]; then
                        "$_hf_cli" auth login || log_warn "HF login failed or skipped."
                    else
                        "$_hf_cli" login || log_warn "HF login failed or skipped."
                    fi
                fi
            else
                log_warn "Neither 'hf' nor 'huggingface-cli' found. Run: pip install -U huggingface_hub"
            fi
            if prompt_yn "Enable high-performance downloads (parallel chunks via hf-xet)?" y; then
                hf_perf="1"
            fi
        fi

        cat > "${NETSCAN_DIR}/.env" <<EOF
# Discord Bot
DISCORD_TOKEN="${discord_token}"
DISCORD_OWNER_ID="${discord_owner}"
# Optional: sync slash commands to this guild for instant availability
# (blank/0 = global sync, which can take up to 1h to propagate).
DISCORD_GUILD_ID=""
# Chat streaming: edit the Discord message as tokens arrive (1) or send once (0).
CHAT_STREAMING="1"

# LLM Models (must match names in llama-swap config.yaml — roles are assigned in
# the model selection menu and written back to .env afterwards; if no role is
# chosen, LLM_MODEL defaults to the first model added. An empty value falls back
# to queue-runner's built-in default, and queue-runner warns at startup when the
# name doesn't match any model advertised by llama-swap).
LLM_MODEL="${llm_model:-}"
LLM_FALLBACK_MODEL="${llm_fallback:-}"
LLM_SMART_MODEL="${llm_smart:-}"
LLM_URL="http://localhost:8080/v1"

# Stable Diffusion Models (leave empty for auto-detection)
# Base directory for all SD models
SD_MODELS_DIR="${sd_base:-/opt/stable-diffusion.cpp/models}"
# Default image model for /imagen: auto | zimage | qwen | hidream | flux | flux2
# "auto" prefers Z-Image Turbo when its files are present.
SD_IMAGE_MODEL="auto"
# Z-Image Turbo (distilled DiT: 8 steps, cfg 1.0, Qwen3-4B encoder). Reuses the
# FLUX ae.safetensors VAE, so SD_ZIMAGE_VAE empty falls back to flux/.
SD_ZIMAGE_DIR="${sd_base:-/opt/stable-diffusion.cpp/models}/z_image"
SD_ZIMAGE_MODEL="${sd_zimage_model:-}"
SD_ZIMAGE_VAE="${sd_zimage_vae:-}"
SD_ZIMAGE_LLM="${sd_zimage_llm:-}"
# Turbo is distilled: few steps at cfg 1.0 (no guidance -> no negative prompt).
# Do NOT reuse the Qwen knobs (6.0 / 20 steps) here.
SD_ZIMAGE_CFG_SCALE="1.0"
SD_ZIMAGE_STEPS="8"
SD_ZIMAGE_SIZE="1024"
# Qwen-Image 2.1 (SOTA text-to-image + instruction-based image editing)
SD_QWEN_DIR="${sd_base:-/opt/stable-diffusion.cpp/models}/qwen_image"
SD_QWEN_MODEL="${sd_qwen_model:-}"
SD_QWEN_VAE="${sd_qwen_vae:-}"
SD_QWEN_LLM="${sd_qwen_llm:-}"
SD_QWEN_VISION="${sd_qwen_vision:-}"
# Sampling. Defaults are sd.cpp's own examples (--cfg-scale 6.0, 20 steps, 512px).
# A ComfyUI graph may run editing at CFG 1.5 — that does NOT transfer to sd.cpp.
# SD_QWEN_SIZE must be a multiple of 32.
SD_QWEN_CFG_SCALE="6.0"
SD_QWEN_STEPS="20"
SD_QWEN_SIZE="512"
# Extra sd.cpp flags. Recommended here:
#   SD_QWEN_MODEL_ARGS="qwen_image_2_1_prefix_cache_type=q8_0"   (~1 GiB less cache)
SD_QWEN_MODEL_ARGS=""
# Negative prompt for image jobs (empty = none, the sd.cpp default)
SD_NEGATIVE_PROMPT=""
# FLUX.1 (text-to-image, fast) — supports .gguf and .safetensors
SD_FLUX_MODEL="${sd_flux_model:-}"
SD_FLUX_VAE=""
SD_FLUX_CLIP_L=""
SD_FLUX_T5XXL=""
# FLUX.2 (text-to-image, best quality)
SD_FLUX2_MODEL="${sd_flux2_model:-}"
SD_FLUX2_VAE="${sd_flux2_vae:-}"
SD_FLUX2_LLM="${sd_flux2_llm:-}"
# FLUX Kontext (image editing)
SD_KONTEXT_MODEL="${sd_kontext:-}"
# WAN 2.2 (video generation — two-pass: high-noise + low-noise diffusion models)
SD_WAN_MODEL="${sd_wan_model:-}"
SD_WAN_HIGH_NOISE="${sd_wan_high_noise:-}"
SD_WAN_VAE=""
SD_WAN_T5XXL=""
# ESRGAN (upscale)
SD_ESRGAN_MODEL="${sd_esrgan:-}"
# HiDream-O1 (PRIMARY model — unified pixel transformer, no VAE/CLIP/T5 needed)
SD_HIDREAM_DIR="${sd_base:-/opt/stable-diffusion.cpp/models}/hidream"
SD_HIDREAM_MODEL="${sd_hidream:-}"

# Hugging Face Download Optimization (hf-xet parallel chunks)
HF_XET_HIGH_PERFORMANCE="${hf_perf:-}"
HF_HUB_DOWNLOAD_TIMEOUT="60"

# Auto-sleep curfew: force sleep mode after this hour if in smart mode (default 1 = 1:00 AM)
SMART_MODE_CURFEW_HOUR="1"
EOF
        chmod 600 "${NETSCAN_DIR}/.env"
        log_ok ".env written to ${NETSCAN_DIR}/.env (permissions 600)"
        log_info ""
        log_info "--- .env summary (token masked) ---"
        log_info "  DISCORD_OWNER_ID        = ${discord_owner}"
        log_info "  LLM roles               = assigned in model selection menu (mmproj per model)"
        log_info "  SD_MODELS_DIR           = ${sd_base:-/opt/stable-diffusion.cpp/models}"
        log_info "  HF_XET_HIGH_PERFORMANCE = ${hf_perf:-<empty>}"
    fi

    # ── 3. Profile.json ──
    if [[ ! -f "${NETSCAN_DIR}/profile.json" ]]; then
        cat > "${NETSCAN_DIR}/profile.json" <<'EOF'
{
  "name": "User",
  "interest_keywords": {
    "high": [],
    "medium": []
  },
  "location": "",
  "career": {
    "title": "",
    "skills": []
  }
}
EOF
        log_ok "Created default profile.json"
    fi

    # ── 4. cv_master.md ──
    # Source of truth for career-scan pitches. Prefer the rich template from the
    # repo (cv_master.md.template, next to install.sh); fall back to a minimal
    # stub if the template is missing. Never overwrites an existing CV.
    if [[ ! -f "${NETSCAN_DIR}/cv_master.md" ]]; then
        if [[ -f "$(dirname "$0")/cv_master.md.template" ]]; then
            cp "$(dirname "$0")/cv_master.md.template" "${NETSCAN_DIR}/cv_master.md"
            log_info "Copied cv_master.md.template -> ${NETSCAN_DIR}/cv_master.md (edit it to personalize)"
        else
            cat > "${NETSCAN_DIR}/cv_master.md" <<'EOF'
# Master CV / Skills

List all your skills, projects, courses, and experience here.
Clawd will use this to tailor job recommendations.
EOF
        fi
        log_ok "Created default cv_master.md"
    fi

    # ── 5. Model Download ──
    log_info "--- Download AI Models ---"
    if prompt_yn "Download models from HuggingFace now?" y; then
        local sd_base_dir="${sd_base:-/opt/stable-diffusion.cpp/models}"

        log_info ""
        log_info "📦 LLM (chat/vision) — saved to ${MODELS_DIR}"
        while true; do
            prompt_input "HuggingFace repo (Enter to skip): " hf_repo
            [[ -z "$hf_repo" ]] && break
            prompt_input "File to download (e.g. gemma-4-12b-it-Q4_K_M.gguf): " hf_file
            [[ -n "$hf_file" ]] && hf_download "${hf_repo}" "${hf_file}" "${MODELS_DIR}"
            if ! prompt_yn "Download another file from this repo?" n; then
                break
            fi
        done

        # Also offer mmproj download if not present
        if [[ ! -f "${MODELS_DIR}/mmproj-F16.gguf" ]] && [[ ! -f "${MODELS_DIR}/mmproj-gemma-4-12B-it-f16.gguf" ]]; then
            if prompt_yn "Download vision projector (mmproj) for Gemma 4?" y; then
                hf_download "unsloth/gemma-4-12b-it-GGUF" "mmproj-F16.gguf" "${MODELS_DIR}"
            fi
        fi

        log_info ""
        log_info "🖼️  IMAGE models — saved to ${sd_base_dir}"
        if prompt_yn "Download FLUX.1 schnell (fast text-to-image)?" y; then
            hf_download "city96/FLUX.1-schnell-gguf" "flux1-schnell-Q4_K_S.gguf" "${sd_base_dir}/flux"
            hf_download "black-forest-labs/FLUX.1-schnell" "ae.safetensors" "${sd_base_dir}/flux"
            hf_download "comfyanonymous/flux_text_encoders" "clip_l.safetensors" "${sd_base_dir}/flux"
            hf_download "city96/t5-v1_1-xxl-encoder-gguf" "t5-v1_1-xxl-encoder-Q4_K_M.gguf" "${sd_base_dir}/flux"
        fi
        if prompt_yn "Download FLUX.2 klein-9B (best quality, needs VAE + LLM encoder)?" n; then
            hf_download "leejet/FLUX.2-klein-9B-GGUF" "flux-2-klein-9b-Q4_0.gguf" "${sd_base_dir}/flux2"
            hf_download "Aisha-AI-Official/flux2-vae" "ae.safetensors" "${sd_base_dir}/flux2"
            hf_download "Qwen/Qwen3-8B-GGUF" "Qwen3-8B-Q4_K_M.gguf" "${sd_base_dir}/flux2"
        fi
        if prompt_yn "Download Z-Image Turbo (fast image default: 8 steps, ~6 GB)?" y; then
            log_info "  Z-Image Turbo: distilled DiT + Qwen3-4B encoder. Reuses the FLUX ae.safetensors VAE."
            log_info "  VRAM: ~6.4 GB total with --offload-to-cpu --fa."
            mkdir -p "${sd_base_dir}/z_image"
            hf_download "leejet/Z-Image-Turbo-GGUF" "z_image_turbo-Q4_K.gguf" "${sd_base_dir}/z_image"
            hf_download "unsloth/Qwen3-4B-Instruct-2507-GGUF" "Qwen3-4B-Instruct-2507-Q4_K_M.gguf" "${sd_base_dir}/z_image"
        fi
        if prompt_yn "Download Qwen-Image 2.1 (SOTA text-to-image + native editing, 7B DiT)?" y; then
            log_info "  Qwen-Image 2.1: 7B DiT model + Qwen3-VL-8B encoder + dedicated VAE."
            log_info "  VRAM: ~10 GB total with --offload-to-cpu --fa (fits BC-250 16GB memory)."
            mkdir -p "${sd_base_dir}/qwen_image"
            hf_download "leejet/Qwen-Image-2.1-GGUF" "qwen_image_2.1-Q4_K.gguf" "${sd_base_dir}/qwen_image"
            # NB: the repo path is vae/… — there is NO split_files/ prefix here.
            hf_download "Comfy-Org/Qwen-Image-2.1" "vae/qwen_image_2.1_vae_bf16.safetensors" "${sd_base_dir}/qwen_image"
            hf_download "Qwen/Qwen3-VL-8B-Instruct-GGUF" "Qwen3VL-8B-Instruct-Q4_K_M.gguf" "${sd_base_dir}/qwen_image"
            hf_download "Qwen/Qwen3-VL-8B-Instruct-GGUF" "mmproj-Qwen3VL-8B-Instruct-F16.gguf" "${sd_base_dir}/qwen_image"
        fi
        # Optional drop-in replacement for the diffusion model only (same VAE + encoder).
        if prompt_yn "Also download the UNCENSORED Qwen-Image 2.1 weights?" n; then
            log_info "  Uncensored: unfiltered diffusion weights (~4.3GB, Q4_K_M)."
            log_info "  The official weights stay the .env default; select this one with SD_QWEN_MODEL."
            hf_download "abenzerps/Qwen-Image-2.1-Uncensored-GGUF" "qwen-image-2.1-UC-Q4_K_M.gguf" "${sd_base_dir}/qwen_image"
        fi
        if prompt_yn "Download HiDream-O1 (PRIMARY image model, ~16.4GB BF16)?" y; then
            log_info "  Note: HiDream-O1-Image-Dev-BF16 from Comfy-Org (public repo, no HF login needed)."
            log_info "  VRAM: sd-cli quantizes only gate/up_proj + self_attn to Q8_0 at load"
            log_info "        (--tensor-type-rules) so it fits 16GB VRAM and avoids the"
            log_info "        Vulkan noise bug from issue #1496."
            local _hd_target="${sd_base_dir}/hidream/HiDream-O1-Image-Dev-BF16.safetensors"
            if [[ -f "${_hd_target}" ]]; then
                log_ok "  Already present: HiDream-O1-Image-Dev-BF16.safetensors — skipping"
            else
                hf_download "Comfy-Org/HiDream-O1-Image" "checkpoints/hidream_o1_image_dev_bf16.safetensors" "${sd_base_dir}/hidream"
                # hf_download saves under dest/checkpoints/... — move to hidream root with a
                # recognizable name so the auto-detect and the bot's _find_model find it.
                if [[ -f "${sd_base_dir}/hidream/checkpoints/hidream_o1_image_dev_bf16.safetensors" ]]; then
                    mv "${sd_base_dir}/hidream/checkpoints/hidream_o1_image_dev_bf16.safetensors" "${_hd_target}"
                    rmdir "${sd_base_dir}/hidream/checkpoints" 2>/dev/null || true
                    log_ok "  Saved as HiDream-O1-Image-Dev-BF16.safetensors"
                fi
            fi
        fi

        log_info ""
        log_info "🎬 VIDEO models (WAN 2.2 two-pass) — saved to ${sd_base_dir}/wan"
        if prompt_yn "Download WAN 2.2 T2V A14B (text-to-video, HighNoise + LowNoise Q4_K_M)?" n; then
            hf_download "QuantStack/Wan2.2-T2V-A14B-GGUF" "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf" "${sd_base_dir}/wan"
            hf_download "QuantStack/Wan2.2-T2V-A14B-GGUF" "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf" "${sd_base_dir}/wan"
        fi
        if prompt_yn "Download WAN 2.2 I2V A14B (image-to-video, HighNoise + LowNoise Q4_K_M)?" n; then
            log_info "  Note: attach an image in Discord with generate-video to animate it."
            hf_download "QuantStack/Wan2.2-I2V-A14B-GGUF" "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf" "${sd_base_dir}/wan"
            hf_download "QuantStack/Wan2.2-I2V-A14B-GGUF" "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf" "${sd_base_dir}/wan"
        fi
        # VAE + text encoder are shared by both T2V and I2V
        if [[ ! -f "${sd_base_dir}/wan/wan_2.1_vae.safetensors" ]] && \
           [[ ! -f "${sd_base_dir}/wan/split_files/vae/wan_2.1_vae.safetensors" ]]; then
            hf_download "Comfy-Org/Wan_2.1_ComfyUI_repackaged" "split_files/vae/wan_2.1_vae.safetensors" "${sd_base_dir}/wan"
        fi
        if [[ ! -f "${sd_base_dir}/wan/umt5-xxl-encoder-Q4_K_M.gguf" ]]; then
            hf_download "city96/umt5-xxl-encoder-gguf" "umt5-xxl-encoder-Q4_K_M.gguf" "${sd_base_dir}/wan"
        fi

        log_info ""
        log_info "🔍 ESRGAN upscaler — saved to ${sd_base_dir}/esrgan"
        if prompt_yn "Download ESRGAN upscaler?" n; then
            log_info "Downloading RealESRGAN_x4plus.pth from schwgHao/RealESRGAN_x4plus..."
            # ai-forever/Real-ESRGAN only ships RealESRGAN_x{2,4,8}.pth (no _x4plus);
            # schwgHao/RealESRGAN_x4plus has the exact file the bot expects.
            hf_download "schwgHao/RealESRGAN_x4plus" "RealESRGAN_x4plus.pth" "${sd_base_dir}/esrgan"
        fi
    fi

    # ── 5.8 Auto-detect downloaded SD models → write to .env ──
    # Fill empty SD_* keys with the exact files present on disk (never
    # overwrite a path the user typed manually). Only runs when .env is being
    # (re)written. Absolute paths are used — queue-runner.py's _find_model
    # accepts absolute paths directly and falls back to name scans otherwise.
    if [[ $do_write_env -eq 1 ]]; then
        local sd_base_dir="${sd_base:-/opt/stable-diffusion.cpp/models}"
        log_info ""
        log_info "🛰️  Auto-detecting SD models in ${sd_base_dir} → .env"
        local _f
        # FLUX.1 (fast text-to-image)
        _f=$(find_model_file "${sd_base_dir}/flux" 'flux1-schnell-*.gguf') && sd_env_set SD_FLUX_MODEL "$_f"
        _f=$(find_model_file "${sd_base_dir}/flux" 'ae.safetensors') && sd_env_set SD_FLUX_VAE "$_f"
        _f=$(find_model_file "${sd_base_dir}/flux" 'clip_l.safetensors') && sd_env_set SD_FLUX_CLIP_L "$_f"
        _f=$(find_model_file "${sd_base_dir}/flux" 't5-v1_1-xxl-encoder-*.gguf') && sd_env_set SD_FLUX_T5XXL "$_f"
        _f=$(find_model_file "${sd_base_dir}/flux" 'flux1-kontext-*.gguf') && sd_env_set SD_KONTEXT_MODEL "$_f"
        # FLUX.2 (best quality)
        _f=$(find_model_file "${sd_base_dir}/flux2" 'flux-2-klein-*.gguf') && sd_env_set SD_FLUX2_MODEL "$_f"
        _f=$(find_model_file "${sd_base_dir}/flux2" 'ae.safetensors') && sd_env_set SD_FLUX2_VAE "$_f"
        _f=$(find_model_file "${sd_base_dir}/flux2" 'qwen3-*.gguf') && sd_env_set SD_FLUX2_LLM "$_f"
        # Z-Image Turbo (fast image default) — diffusion + Qwen3-4B encoder;
        # its VAE is the FLUX ae.safetensors, so SD_ZIMAGE_VAE stays empty
        # unless a copy is dropped in z_image/ as well.
        _f=$(find_model_file "${sd_base_dir}/z_image" 'z_image_turbo-*.gguf') && sd_env_set SD_ZIMAGE_MODEL "$_f"
        _f=$(find_model_file "${sd_base_dir}/z_image" 'ae.safetensors') && sd_env_set SD_ZIMAGE_VAE "$_f"
        _f=$(find_model_file "${sd_base_dir}/z_image" 'Qwen3-4B-Instruct-2507-*.gguf') && sd_env_set SD_ZIMAGE_LLM "$_f"
        # Qwen-Image 2.1 (SOTA text-to-image + instruction-based image editing)
        _f=$(find_model_file "${sd_base_dir}/qwen_image" 'qwen_image_2.1-*.gguf' 'qwen_image_2.1*.safetensors') && sd_env_set SD_QWEN_MODEL "$_f"
        _f=$(find_model_file "${sd_base_dir}/qwen_image" 'qwen_image_2.1_vae*.safetensors' 'qwen_image_vae*.safetensors') && sd_env_set SD_QWEN_VAE "$_f"
        _f=$(find_model_file "${sd_base_dir}/qwen_image" 'Qwen3VL-8B-*.gguf' 'qwen3-vl-*.gguf') && sd_env_set SD_QWEN_LLM "$_f"
        _f=$(find_model_file "${sd_base_dir}/qwen_image" 'mmproj-Qwen3VL-*.gguf' 'Qwen3VL-*-mmproj*.gguf') && sd_env_set SD_QWEN_VISION "$_f"
        # WAN: diffusion models stay empty (the bot picks T2V vs I2V per mode);
        # only the shared VAE + text encoder are pinned explicitly.
        _f=$(find_model_file "${sd_base_dir}/wan" 'wan_2.1_vae.safetensors') && sd_env_set SD_WAN_VAE "$_f"
        _f=$(find_model_file "${sd_base_dir}/wan" 'umt5-xxl-encoder-*.gguf') && sd_env_set SD_WAN_T5XXL "$_f"
        # HiDream (primary image) + ESRGAN upscaler — prefer BF16 if an old
        # FP8 file also lingers in hidream/ (both would match the wildcard).
        _f=$(find_model_file "${sd_base_dir}/hidream" 'HiDream-O1-Image-Dev-BF16.safetensors') || true
        if [[ -z "$_f" ]]; then
            _f=$(find_model_file "${sd_base_dir}/hidream" 'HiDream-O1-*.safetensors') || true
        fi
        sd_env_set SD_HIDREAM_MODEL "$_f"
        _f=$(find_model_file "${sd_base_dir}/esrgan" 'RealESRGAN_x4plus.pth') && sd_env_set SD_ESRGAN_MODEL "$_f"
    fi

    # ── 6. llama-swap config.yaml ── (model menu lives in configure_models)
    configure_models "$do_write_env"

    # ── 6. Verify / test ──
    log_info ""
    log_info "=== Configuration Summary ==="
    log_info "Discord Bot:     ${NETSCAN_DIR}/.env"
    log_info "                 (every key is documented in ${SCRIPT_DIR}/.env.template)"
    log_info "LLM Proxy:       /opt/llama-swap/config.yaml"
    if is_systemd; then
        log_info "Queue Runner:    /etc/systemd/system/queue-runner.service"
        log_info "llama-swap:      /etc/systemd/system/llama-swap.service"
    else
        log_info "Queue Runner:    /etc/init.d/queue-runner"
        log_info "llama-swap:      /etc/init.d/llama-swap"
    fi
    log_info "Models dir:      ${MODELS_DIR}"
    log_info "Jobs:            ${DATA_DIR}/jobs.json"

    log_info ""
    if [[ ! -x /opt/llama-swap/build/llama-swap ]]; then
        log_warn "llama-swap binary not found at /opt/llama-swap/build/llama-swap"
        log_warn "Run '${0} install' first to build llama-swap, then start it."
    elif prompt_yn "Start llama-swap now to test?" y; then
        if ! start_service llama-swap; then
            log_warn "Could not start llama-swap. Check service status."
        fi
        sleep 2
        if curl -s http://localhost:8080/v1/models >/dev/null 2>&1; then
            log_ok "llama-swap is responding on http://localhost:8080"
        else
            log_warn "llama-swap not responding yet. Check service status."
        fi
    fi

    log_info ""
    log_ok "=== CONFIGURE COMPLETE ==="
    if is_systemd; then
        log_info "Start the Discord bot: systemctl start queue-runner"
        log_info "View logs:             journalctl -u queue-runner -f"
    else
        log_info "Start the Discord bot: rc-service queue-runner start"
        log_info "View logs:             tail -f ${DATA_DIR}/queue-runner.log"
    fi
    log_info "Download models:       hf download <repo> <file> --local-dir ${MODELS_DIR}"
}

# ─── Subcommand: configure-models ─────────────────────────────────────────────
# Re-run ONLY the llama-swap model menu from 'configure': scans ${MODELS_DIR},
# adds/edits models, assigns primary/fallback/smart roles, and rewrites
# /opt/llama-swap/config.yaml + the LLM_* keys in .env. Does NOT touch Discord,
# SD models, Hugging Face, downloads, or any other .env settings.
cmd_configure_models() {
    require_root
    log_info "=== BC-250 v10 Models Configuration (${DISTRO_NAME}) ==="
    configure_models 1
    log_ok "=== MODELS CONFIGURATION COMPLETE ==="
    if is_systemd; then
        log_info "Restart llama-swap to apply: systemctl restart llama-swap"
    else
        log_info "Restart llama-swap to apply: rc-service llama-swap restart"
    fi
}

# ─── Subcommand: switch-llama ────────────────────────────────────────────────
# Lightweight fork switcher: only re-clones/rebuilds llama.cpp from a chosen
# fork (TurboQuant / mainline / bee.cpp). Does NOT touch system packages, the
# venv, or the other builds — use 'install' for the full setup.
cmd_switch_llama() {
    require_root
    log_info "=== BC-250 v10 llama.cpp fork switch (${DISTRO_NAME}) ==="

    log_info "Stopping services before rebuild..."
    stop_service llama-swap 2>/dev/null || true
    stop_service queue-runner 2>/dev/null || true

    select_llama_fork
    log_info "Target: ${LLAMA_FORK_NAME} (${LLAMA_REPO_URL})"

    if [[ -d /opt/llama.cpp ]]; then
        # Same logic as cmd_install section 4: warn if a different fork is
        # installed (default re-clone to yes), else ask with default no.
        local _cur_remote
        _cur_remote=$(git -C /opt/llama.cpp remote get-url origin 2>/dev/null || true)
        if [[ -n "$_cur_remote" ]] && [[ "$_cur_remote" != "$LLAMA_REPO_URL" ]]; then
            log_warn "llama.cpp exists but is from a different source: ${_cur_remote}"
            if prompt_yn "Remove it and re-clone from ${LLAMA_FORK_NAME}?" y; then
                rm -rf /opt/llama.cpp
            else
                log_warn "Keeping the existing checkout — it will be rebuilt as-is (${_cur_remote})."
                LLAMA_FORK_NAME="existing (${_cur_remote})"
                LLAMA_REPO_URL="$_cur_remote"
            fi
        else
            log_warn "llama.cpp already exists at /opt/llama.cpp (${LLAMA_FORK_NAME})"
            if prompt_yn "Remove and re-clone?" n; then
                rm -rf /opt/llama.cpp
            fi
        fi
    fi

    if [[ ! -d /opt/llama.cpp ]]; then
        log_info "Cloning llama.cpp (${LLAMA_FORK_NAME})..."
        git clone --depth 1 "${LLAMA_REPO_URL}" /opt/llama.cpp
    fi

    cd /opt/llama.cpp
    log_info "Building llama.cpp (${LLAMA_FORK_NAME}) with Vulkan support (this may take a while)..."
    # Always build from a clean tree: a stale build/ dir (objects from a
    # previous fork/commit) produces a broken binary that segfaults at load.
    rm -rf build
    cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j"$(nproc)"
    log_ok "llama.cpp built: /opt/llama.cpp/build/bin/llama-server"
    if is_systemd; then
        log_info "Restart services: systemctl start llama-swap queue-runner"
    else
        log_info "Restart services: rc-service llama-swap start && rc-service queue-runner start"
    fi
}

# ─── Subcommand: update ───────────────────────────────────────────────────────
cmd_update() {
    require_root
    log_info "=== BC-250 v10 Update (${DISTRO_NAME}) ==="
    log_info "Updating llama.cpp, llama-swap, stable-diffusion.cpp, and whisper.cpp from upstream..."

    log_info "Stopping services before rebuild..."
    stop_service llama-swap 2>/dev/null || true
    stop_service queue-runner 2>/dev/null || true

    if [[ -d /opt/llama.cpp ]]; then
        cd /opt/llama.cpp
        # Update from the fork that is actually installed (TurboQuant / mainline /
        # bee.cpp). 'update' never switches forks — use 'install' to change.
        local _llama_upstream
        _llama_upstream=$(git remote get-url origin 2>/dev/null | sed 's|.*/||; s|\.git$||' || echo unknown)
        log_info "Updating llama.cpp (${_llama_upstream})..."
        git fetch --depth 1 origin
        # In a shallow clone origin/HEAD can be stale or absent after upstream
        # rebases/force-pushes; FETCH_HEAD always points at what was just fetched.
        git reset --hard FETCH_HEAD
        log_info "Rebuilding llama.cpp with Vulkan..."
        # Clean build — stale objects from the previous build break the binary.
        rm -rf build
        cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
        cmake --build build --config Release -j"$(nproc)"
        log_ok "llama.cpp updated"
    else
        log_warn "/opt/llama.cpp not found. Run '${0} install' first."
    fi

    if [[ -d /opt/llama-swap ]]; then
        cd /opt/llama-swap
        log_info "Updating llama-swap..."
        git fetch --depth 1 origin
        git reset --hard FETCH_HEAD
        log_info "Rebuilding llama-swap..."
        GOTOOLCHAIN=auto make clean all
        # Create stable symlink (see cmd_install)
        local _arch
        _arch=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')
        if [[ -f "/opt/llama-swap/build/llama-swap-linux-${_arch}" ]]; then
            ln -sf "/opt/llama-swap/build/llama-swap-linux-${_arch}" /opt/llama-swap/build/llama-swap
        fi
        log_ok "llama-swap updated"
    else
        log_warn "/opt/llama-swap not found. Run '${0} install' first."
    fi

    if [[ -d /opt/stable-diffusion.cpp ]]; then
        cd /opt/stable-diffusion.cpp
        log_info "Updating stable-diffusion.cpp..."
        git fetch --depth 1 origin
        # FETCH_HEAD, not origin/HEAD: in a shallow clone the latter can be stale
        # or absent after upstream rebases/force-pushes (same reasoning as the
        # llama.cpp path above).
        git reset --hard FETCH_HEAD
        git submodule update --init --recursive --depth 1
        log_info "Rebuilding stable-diffusion.cpp with Vulkan..."
        # Same flags as cmd_install: skip the sd-server pnpm frontend (it fails on
        # a stock box) and build only the binary we actually use.
        cmake -B build -DSD_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DSD_SERVER_BUILD_FRONTEND=OFF
        cmake --build build --config Release --target sd-cli -j"$(nproc)"
        log_ok "stable-diffusion.cpp updated"
    else
        log_warn "/opt/stable-diffusion.cpp not found. Run '${0} install' first."
    fi

    if [[ -d /opt/whisper.cpp ]]; then
        cd /opt/whisper.cpp
        log_info "Updating whisper.cpp..."
        git fetch --depth 1 origin
        git reset --hard FETCH_HEAD
        log_info "Rebuilding whisper.cpp with Vulkan..."
        cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
        cmake --build build --config Release -j"$(nproc)"
        log_ok "whisper.cpp updated"
    else
        log_warn "/opt/whisper.cpp not found. Run '${0} install' first."
    fi

    log_info ""
    log_ok "=== UPDATE COMPLETE ==="
    if is_systemd; then
        log_info "Restart services: systemctl start llama-swap && systemctl start queue-runner"
    else
        log_info "Restart services: rc-service llama-swap start && rc-service queue-runner start"
    fi
    log_info ""
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  💡 New in this update:"
    echo ""
    echo "  📄 PDF support — send .pdf files to Discord bot"
    echo "     Text extraction + OCR via Gemma 4 vision (2 pages at a time)"
    echo "     Requires: apk add poppler-utils (Alpine), pacman -S poppler (Arch/CachyOS), or dnf install poppler-utils (Fedora/RHEL)"
    echo ""
    echo "  🎤 Native audio — Gemma 4 reads voice messages directly"
    echo "     No separate model needed — same model for text, vision, and audio"
    echo ""
    echo "  🧠 Self-improvement — memory auto-summarization, lazy tool loading (tool-help), script approval"
    echo "     Persistent memory with auto-summarization, tool-help (lazy loading), function registry, script proposals with approval"
    echo ""
    echo "  🌐 Web UI — browser-based configuration"
    echo "     Run 'sudo ${0} webui' to configure everything from your browser"
    echo ""
    echo "  🎨 Qwen-Image 2.1 support (SOTA unified text-to-image + native instruction editing)"
    echo "     Download: hf download leejet/Qwen-Image-2.1-GGUF qwen_image_2.1-Q4_K.gguf --local-dir /opt/stable-diffusion.cpp/models/qwen_image/"
    echo "     Uses Qwen3-VL-8B encoder + dedicated VAE. Fits ~11GB VRAM with --offload-to-cpu --fa"
    echo ""
    echo "  🎨 HiDream-O1 support (PRIMARY image model — unified, no VAE/CLIP/T5)"
    echo "     Download: hf download Comfy-Org/HiDream-O1-Image checkpoints/hidream_o1_image_dev_bf16.safetensors --local-dir /opt/stable-diffusion.cpp/models/hidream/"
    echo "     (rename to HiDream-O1-Image-Dev-BF16.safetensors; runs with --tensor-type-rules to fit 16GB VRAM)"
    echo ""
    echo "  ⚡ MTP / DFlash speculative decoding — enable during configure for faster generation (Gemma 4, Qwen3)"
    echo ""
    echo "  🖌️ SD models configurable via .env (SD_* variables)"
    echo "     Run 'sudo ${0} configure' to set up model paths interactively"
    echo "═══════════════════════════════════════════════════════════════════════"
}

# ─── Subcommand: sync ────────────────────────────────────────────────────────
cmd_sync() {
    require_root
    log_info "=== BC-250 v10 Sync (${DISTRO_NAME}) ==="
    log_info "Syncing bot scripts from ${SCRIPT_DIR}/netscan to ${NETSCAN_DIR}..."

    if [[ ! -d "${SCRIPT_DIR}/netscan" ]]; then
        log_err "No local netscan/ directory found at ${SCRIPT_DIR}/netscan"
        log_err "Run sync from the repo checkout (e.g. ~/bc250-config)."
        return 1
    fi
    if [[ ! -d "${NETSCAN_DIR}" ]]; then
        log_err "${NETSCAN_DIR} does not exist — run '${0} install' first."
        return 1
    fi

    # ── Version check: warn if local repo is behind origin (git pull needed) ──
    if [[ -d "${SCRIPT_DIR}/.git" ]] && command -v git >/dev/null 2>&1; then
        local _upstream _behind
        _upstream="$(git -C "${SCRIPT_DIR}" rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || true)"
        if [[ -n "${_upstream}" ]]; then
            # Refresh remote refs so 'behind origin' is accurate — the local
            # remote-tracking ref is only updated by fetch/pull, so without this
            # the check would miss a missing 'git pull'. Fetch the actual remote
            # that owns the upstream ref (not hardcoded 'origin'); timeout guards
            # offline runs, GIT_TERMINAL_PROMPT=0 makes private-repo auth fail fast.
            if timeout 10 env GIT_TERMINAL_PROMPT=0 git -C "${SCRIPT_DIR}" fetch "${_upstream%/*}" --quiet 2>/dev/null; then
                _behind="$(git -C "${SCRIPT_DIR}" rev-list --count HEAD.."${_upstream}" 2>/dev/null || echo "")"
                if [[ -n "${_behind}" ]] && [[ "${_behind}" =~ ^[0-9]+$ ]]; then
                    if [[ "${_behind}" -gt 0 ]]; then
                        log_warn "Local repo is ${_behind} commit(s) behind ${_upstream} — run 'git pull' in ${SCRIPT_DIR} first!"
                        if ! prompt_yn "Sync anyway with the local (possibly stale) copy?" n; then
                            log_err "Aborted — run 'git pull' then re-run '${0} sync'."
                            return 1
                        fi
                    else
                        log_ok "Repo up to date with ${_upstream}"
                    fi
                else
                    log_warn "Could not count commits behind ${_upstream} (shallow clone?) — continuing"
                fi
            else
                log_warn "Could not reach origin (offline?) — cannot verify if a 'git pull' is pending; continuing"
            fi
        fi
    fi

    log_info "Copying Python/shell scripts (code only — user data like profile.json is preserved)..."
    cp "${SCRIPT_DIR}"/netscan/*.py "${NETSCAN_DIR}/" 2>/dev/null || true
    cp "${SCRIPT_DIR}"/netscan/*.sh "${NETSCAN_DIR}/" 2>/dev/null || true
    chmod +x "${NETSCAN_DIR}"/*.py "${NETSCAN_DIR}"/*.sh 2>/dev/null || true
    log_ok "Scripts copied to ${NETSCAN_DIR}"

    # ── jobs.json sync (only if no local differences; backup first) ──
    local _repo_jobs="${SCRIPT_DIR}/jobs.json"
    local _inst_jobs="${DATA_DIR}/jobs.json"
    local _jobs_state="${DATA_DIR}/.jobs.json.sha256"
    if [[ -f "${_repo_jobs}" ]]; then
        if [[ ! -f "${_inst_jobs}" ]]; then
            cp "${_repo_jobs}" "${_inst_jobs}"
            sha256sum "${_inst_jobs}" | awk '{print $1}' > "${_jobs_state}" || true
            log_ok "jobs.json installed (first time)"
        else
            if cmp -s "${_repo_jobs}" "${_inst_jobs}"; then
                log_info "jobs.json already in sync"
                sha256sum "${_inst_jobs}" | awk '{print $1}' > "${_jobs_state}" || true
            else
                # Backup the running copy before any change
                local _bak="${_inst_jobs}.bak.$(date +%Y%m%d-%H%M%S)"
                cp "${_inst_jobs}" "${_bak}"
                log_info "Backup of current jobs.json: ${_bak}"
                if [[ -f "${_jobs_state}" ]] && [[ "$(sha256sum "${_inst_jobs}" | awk '{print $1}')" == "$(cat "${_jobs_state}")" ]]; then
                    # Installed copy == what we last synced → no local edits → safe to update
                    cp "${_repo_jobs}" "${_inst_jobs}"
                    sha256sum "${_inst_jobs}" | awk '{print $1}' > "${_jobs_state}" || true
                    log_ok "jobs.json updated from repo (no local changes detected)"
                else
                    # Unknown history or local edits → ask
                    if prompt_yn "jobs.json differs from repo and may have local edits — overwrite with repo version?" n; then
                        cp "${_repo_jobs}" "${_inst_jobs}"
                        sha256sum "${_inst_jobs}" | awk '{print $1}' > "${_jobs_state}" || true
                        log_ok "jobs.json updated from repo (confirmed)"
                    else
                        log_warn "Keeping local jobs.json (backup saved to ${_bak})"
                    fi
                fi
            fi
        fi
    else
        log_warn "No jobs.json in repo — skipping jobs sync"
    fi

    log_info "Restarting queue-runner..."
    restart_service queue-runner 2>/dev/null || true
    log_ok "queue-runner restarted"

    log_info ""
    log_ok "=== SYNC COMPLETE ==="
    if is_systemd; then
        log_info "Verify: systemctl status queue-runner && tail -f ${DATA_DIR}/queue-runner.log"
    else
        log_info "Verify: rc-service queue-runner status && tail -f ${DATA_DIR}/queue-runner.log"
    fi
}

# ─── Subcommand: help ────────────────────────────────────────────────────────
cmd_help() {
    cat <<EOF
Usage: ${0} <command>

Commands:
  install      Install dependencies, build llama.cpp (choose fork: TurboQuant / mainline / bee.cpp)
               + llama-swap + stable-diffusion.cpp + whisper.cpp, select jobs interactively,
               create OpenRC/systemd services.
  configure    Set up Discord token, llama-swap models, .env, profile.json.
  configure-models  Re-run ONLY the llama-swap model menu (scan models, add/edit, assign
               primary/fallback/smart roles) without touching the rest of .env.
  update       Pull latest mainline sources and rebuild llama.cpp + llama-swap + stable-diffusion.cpp + whisper.cpp.
  switch-llama Re-clone and rebuild ONLY llama.cpp from a chosen fork (TurboQuant / mainline / bee.cpp).
               Lightweight alternative to 'install' — does not touch packages, venv, or other builds.
  sync         Copy updated bot scripts (.py/.sh) and jobs.json (only if no local edits) from this repo
               to ${NETSCAN_DIR} and restart queue-runner. A backup of jobs.json is kept before overwriting.
               NOTE: fetches the upstream remote and warns if the local repo is behind (missing 'git pull'), aborting unless confirmed.
  jobs         Open interactive TUI to manage job schedules (enable/disable, edit cron).
  models       Open interactive TUI to manage llama-swap models (add/edit/delete).
  webui [port] Launch temporary web configuration UI (default port 8888, auto-shutdown after 30 min).
  searxng      Manage the optional local SearXNG search service (podman container,
               localhost only): check|up|down|status|test|logs|boot.
  help         Show this help.

Examples:
  sudo ${0} install
  sudo ${0} configure
  sudo ${0} configure-models
  sudo ${0} webui 9000
  sudo ${0} jobs
  sudo ${0} update
  sudo ${0} sync

Features:
  🧠 LLM:        Gemma 4 12B (multimodal: text + vision + audio + PDF) + MTP / DFlash speculative decoding
  🎨 Images:      Qwen-Image 2.1 + HiDream-O1 + FLUX (via stable-diffusion.cpp)
  🎬 Video:       WAN 2.1 T2V (via stable-diffusion.cpp)
  📄 PDF:         Text extraction (poppler-utils / poppler) + OCR via Gemma 4 vision
  🎤 Audio:       Native Gemma 4 audio + whisper.cpp fallback
  🔬 Research:    Deep research agent (ddgr + sandboxed execution)
  🧠 Self-learn:  Persistent memory with auto-summarization, tool-help (lazy loading), function registry, script proposals
  🌐 Web UI:      Temporary configuration interface (install.sh webui)

Files created:
  /opt/llama.cpp/                     — llama.cpp (TurboQuant / mainline / bee.cpp, Vulkan build)
  /opt/llama-swap/                    — mainline llama-swap
  /opt/llama-swap/config.yaml         — model definitions
  /opt/netscan/.env                   — Discord token & model names
  /opt/netscan/data/jobs.json         — enabled cron jobs
  /opt/netscan/data/clawd-memory.json — persistent bot memory
  /opt/netscan/data/clawd-functions.json — function registry
  /opt/netscan/skills/                — approved auto-generated scripts
  /opt/netscan/data/pending-scripts/  — scripts awaiting approval
  /etc/init.d/queue-runner            — OpenRC service for Discord bot (Alpine)
  /etc/init.d/llama-swap              — OpenRC service for LLM proxy (Alpine)
  /etc/systemd/system/queue-runner.service — systemd service for Discord bot (Arch/CachyOS/Fedora)
  /etc/systemd/system/llama-swap.service   — systemd service for LLM proxy (Arch/CachyOS/Fedora)

EOF
}

# ─── Subcommand: jobs ───────────────────────────────────────────────────────
cmd_jobs() {
    require_root
    local jobs_file="${DATA_DIR}/jobs.json"
    if [[ ! -f "$jobs_file" ]]; then
        log_err "No jobs.json found at ${jobs_file}"
        log_info "Run '${0} install' first."
        exit 1
    fi
    python3 "${NETSCAN_DIR}/jobs-tui.py" "$jobs_file"
}

# ─── Subcommand: models ────────────────────────────────────────────────────
cmd_models() {
    require_root
    if [[ ! -f "${NETSCAN_DIR}/models-tui.py" ]]; then
        log_err "models-tui.py not found at ${NETSCAN_DIR}"
        log_info "Run '${0} install' first."
        exit 1
    fi
    python3 "${NETSCAN_DIR}/models-tui.py" "/opt/llama-swap/config.yaml"
}

# ─── Subcommand: webui ──────────────────────────────────────────────────────
cmd_webui() {
    require_root
    if [[ ! -f "${NETSCAN_DIR}/webui.py" ]]; then
        log_err "webui.py not found at ${NETSCAN_DIR}/webui"
        log_info "Run '${0} install' first."
        exit 1
    fi
    local port="${2:-8888}"
    log_info "Starting BC-250 Configuration Web UI on port ${port}..."
    log_info "Open http://$(hostname -I 2>/dev/null | awk '{print $1}' || echo localhost):${port} in your browser"
    "${VENV_DIR}/bin/python3" "${NETSCAN_DIR}/webui.py" --port "${port}"
}

# ─── Subcommand: searxng ─────────────────────────────────────────────────────
# Passthrough to the component's own script, so there is one implementation of
# "how SearXNG is run here" rather than two that drift apart.
cmd_searxng() {
    local setup=/opt/searxng/searxng-setup.sh
    if [[ ! -f "$setup" ]]; then
        log_err "SearXNG is not installed (no ${setup})."
        log_info "Install it with '${0} install', or copy searxng/ to /opt/searxng and run: sh ${setup} up"
        exit 1
    fi
    shift  # drop the subcommand name; default to 'status'
    exec sh "$setup" "${@:-status}"
}

main() {
    case "${1:-help}" in
        install)   cmd_install "$@";;
        configure) cmd_configure "$@";;
        configure-models) cmd_configure_models "$@";;
        update)    cmd_update "$@";;
        switch-llama) cmd_switch_llama "$@";;
        sync)      cmd_sync "$@";;
        jobs)      cmd_jobs "$@";;
        models)    cmd_models "$@";;
        webui)     cmd_webui "$@";;
        searxng)   cmd_searxng "$@";;
        help|--help|-h) cmd_help;;
        *)
            log_err "Unknown command: ${1}"
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
