#!/usr/bin/env bash
#
# PhysiClaw installer (macOS + Linux). Windows users: use install.ps1.
#
# Usage:
#   curl -fsSL https://physiclaw.ai/install.sh | bash
#
# Options (pass after `bash -s --`):
#   --version <v>   pin a physiclaw version (or set PHYSICLAW_VERSION)
#   --dry-run       print what would happen; make no changes
#   --help, -h      show help
#
# Environment:
#   PHYSICLAW_VERSION=0.0.3   pin a version
#   PHYSICLAW_DRY_RUN=1       same as --dry-run (handy with `curl | bash`)
#   NO_COLOR=1                plain output
#
# What it does (hardware setup is still a separate step at the end):
#   1. Checks you're on macOS or Linux.
#   2. Installs `uv` if missing (hardened fetch of https://astral.sh/uv/install.sh).
#   3. Ensures Python 3.12 is available (no-op if already cached).
#   4. Installs `physiclaw` (small — no heavy ML deps in the package).
#   5. Runs `physiclaw setup local-vision-model` to install the icon-detector
#      model. By default it fetches a prebuilt ONNX (fast, no extra deps); if
#      that's unreachable it falls back to downloading the upstream PyTorch
#      weights and converting them in an ephemeral `uv run --with` env.
#      Idempotent; no-ops if the ONNX is already cached.

set -euo pipefail

# --- Colors: respect NO_COLOR and whether stdout is a TTY. ------------------
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; D=$'\033[2m'; N=$'\033[0m'
else
  B=''; G=''; Y=''; R=''; D=''; N=''
fi
info() { printf '%s%s→%s %s\n'   "$B" "$G" "$N" "$*"; }
warn() { printf '%s%s!%s %s\n'   "$B" "$Y" "$N" "$*" >&2; }
err()  { printf '%s%s✗%s %s\n'   "$B" "$R" "$N" "$*" >&2; }
die()  { err "$@"; exit 1; }

# Scratch files (the downloaded uv installer) are tracked and removed on exit,
# success or failure.
TMPFILES=()
cleanup() { local f; for f in "${TMPFILES[@]:-}"; do rm -f "$f" 2>/dev/null || true; done; }
trap cleanup EXIT

usage() {
  cat <<EOF
PhysiClaw installer (macOS + Linux)

Usage:
  curl -fsSL https://physiclaw.ai/install.sh | bash
  curl -fsSL ... | bash -s -- [options]

Options:
  --version <v>   pin a physiclaw version (or set PHYSICLAW_VERSION)
  --dry-run       print what would happen; make no changes
  --help, -h      show this help

Environment:
  PHYSICLAW_VERSION=<v>   pin a version
  PHYSICLAW_DRY_RUN=1     same as --dry-run
  NO_COLOR=1              plain output
EOF
}

# --- Args. ------------------------------------------------------------------
VERSION="${PHYSICLAW_VERSION:-}"
DRY_RUN=0
[[ "${PHYSICLAW_DRY_RUN:-}" == "1" ]] && DRY_RUN=1
# A bad invocation exits 2 (usage error) and prints help — distinct from die's
# exit 1, which signals a runtime failure mid-install.
usage_err() { err "$*"; usage >&2; exit 2; }
while (( $# > 0 )); do
  case "$1" in
    --version)   VERSION="${2:-}"; shift 2 ;;
    --version=*) VERSION="${1#*=}"; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) usage_err "unknown option: $1" ;;
  esac
done
# In --dry-run, announce a mutating action instead of running it.
would() { printf '%s%s[dry-run]%s would %s\n' "$B" "$D" "$N" "$*"; }

case "$(uname -s)" in
  Darwin) PLATFORM=mac ;;
  Linux)  PLATFORM=linux ;;
  *) die "install.sh is for macOS and Linux. On Windows, use install.ps1 (iwr -useb <url>/install.ps1 | iex)." ;;
esac

# Detect the distro's package manager and echo an install command for the
# given per-manager package lists, so the hint shows the ONE command that
# works here rather than a wall of apt/dnf/pacman lines. Args: <apt> <dnf>
# <pacman> (zypper maps to dnf names, apk to apt names — close enough for a
# hint). Falls back to apt syntax when nothing is detected.
pkg_install_hint() {
  local apt="$1" dnf="$2" pacman="$3"
  if   command -v apt-get >/dev/null 2>&1; then echo "sudo apt install $apt"
  elif command -v dnf     >/dev/null 2>&1; then echo "sudo dnf install $dnf"
  elif command -v yum     >/dev/null 2>&1; then echo "sudo yum install $dnf"
  elif command -v pacman  >/dev/null 2>&1; then echo "sudo pacman -S $pacman"
  elif command -v zypper  >/dev/null 2>&1; then echo "sudo zypper install $dnf"
  elif command -v apk     >/dev/null 2>&1; then echo "sudo apk add $apt"
  else echo "sudo apt install $apt"
  fi
}

# Hardened download: pin HTTPS + TLS 1.2, retry transient failures, prefer
# curl and fall back to wget so minimal Linux images (which often ship only
# one) still work. Args: <url> <output-path>.
fetch() {
  if   command -v curl >/dev/null 2>&1; then
    curl -fsSL --proto '=https' --tlsv1.2 --retry 3 --retry-delay 1 --retry-connrefused -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then
    wget -q --https-only --secure-protocol=TLSv1_2 --tries=3 --timeout=20 -O "$2" "$1"
  else
    die "need curl or wget to download the uv installer"
  fi
}

# uv drops its tool shims under ~/.local/bin.
export PATH="$HOME/.local/bin:$PATH"

# git is only needed by `physiclaw skills install` (clones a repo with
# skills/<name>/SKILL.md layout). Warns-and-continues on miss so users
# without git can still finish the physiclaw install.
if ! command -v git >/dev/null 2>&1; then
  if [[ "$PLATFORM" == mac ]]; then
    # Escalation: native CLT installer (ships git) → brew if already present
    # → docs warn.
    if ! xcode-select -p >/dev/null 2>&1; then
      if [[ $DRY_RUN == 1 ]]; then
        would "trigger the Xcode Command Line Tools installer (ships git)"
      else
        info "Triggering Xcode Command Line Tools installer (ships git)…"
        xcode-select --install >/dev/null 2>&1 || true
      fi
      warn "Click Install in the popup and wait ~5 min, then re-run this script."
      warn "Continuing — \`physiclaw skills install\` stays unavailable until CLT finishes."
    elif command -v brew >/dev/null 2>&1; then
      info "CLT present but git missing — installing via Homebrew…"
      if [[ $DRY_RUN == 1 ]]; then would "brew install git"; else brew install git >/dev/null; fi
    else
      warn "git not found. Run one of:"
      warn "    xcode-select --install    # macOS Command Line Tools (native)"
      warn "    brew install git          # if you install Homebrew first"
    fi
  else
    # Linux — instruct (no auto-sudo; the user opts into mutating the system).
    warn "git not found. Install it, then re-run:"
    warn "    $(pkg_install_hint git git git)"
    warn "Continuing — \`physiclaw skills install\` stays unavailable until git is on PATH."
  fi
fi

FRESH_UV=0
if ! command -v uv >/dev/null 2>&1; then
  info "Installing uv (Python + tool manager)…"
  if [[ $DRY_RUN == 1 ]]; then
    would "fetch https://astral.sh/uv/install.sh and run it"
  else
    tmp_uv="$(mktemp)"; TMPFILES+=("$tmp_uv")
    fetch https://astral.sh/uv/install.sh "$tmp_uv"
    sh "$tmp_uv"
  fi
  FRESH_UV=1
fi

# Only hit uv's python-build-standalone manifest when 3.12 isn't already cached.
if [[ $DRY_RUN == 1 ]]; then
  would "uv python install 3.12 (if not cached)"
elif ! uv python find 3.12 >/dev/null 2>&1; then
  info "Installing Python 3.12…"
  uv python install 3.12 --quiet
fi

SPEC_PLAIN="physiclaw${VERSION:+==$VERSION}"

# Step 4: install physiclaw. No [vision] extra — conversion deps live in
# a uv-managed ephemeral env, not in the physiclaw install. `--refresh`
# invalidates uv's cached PyPI metadata so re-runs always resolve to the
# actual latest version of physiclaw, not whatever was in the cache from
# the last `uv tool install` an hour ago.
info "Installing ${SPEC_PLAIN}…"
if [[ $DRY_RUN == 1 ]]; then
  would "uv tool install $SPEC_PLAIN --python 3.12 --force --refresh"
else
  uv tool install "$SPEC_PLAIN" --python 3.12 --force --refresh >/dev/null

  command -v physiclaw >/dev/null 2>&1 || {
    rc=$([[ "$PLATFORM" == mac ]] && echo "~/.zshrc" || echo "~/.bashrc")
    warn "physiclaw CLI not on PATH. Add this to $rc and open a new terminal:"
    warn "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    die "Exiting."
  }
  info "Installed: $(physiclaw --version)"
fi

# OpenCV's manylinux wheel links libGL + glib and imports them eagerly, so
# `import cv2` fails on a minimal Linux without them (PhysiClaw itself uses no
# cv2 GUI — these are pure load-time libs). Desktop installs almost always
# have them; warn-and-instruct (no auto-sudo) if not. `physiclaw doctor` runs
# the same check with a real import.
if [[ "$PLATFORM" == linux ]] && ! ldconfig -p 2>/dev/null | grep -q 'libGL\.so\.1'; then
  warn "OpenCV needs system graphics libs that aren't installed. If you later"
  warn "see a 'libGL.so.1' import error, install them:"
  warn "    $(pkg_install_hint 'libgl1 libglib2.0-0' 'mesa-libGL glib2' 'libglvnd glib2')"
fi

# Step 5: install the icon-detector model — fetch the prebuilt ONNX, or
# fall back to download + convert. No-op if already cached.
info "Installing the vision model (one-time)…"
if [[ $DRY_RUN == 1 ]]; then
  would "physiclaw setup local-vision-model"
  printf '\n%s%s✓ Dry run complete.%s No changes made.\n' "$B" "$G" "$N"
  exit 0
fi
# Non-fatal: physiclaw itself is installed by now. The model download needs
# huggingface.co, which a locked-down network may block — warn and point to
# the re-run rather than aborting an otherwise-complete install.
if ! physiclaw setup local-vision-model; then
  warn "Vision model not set up — the download or convert step failed (see above)."
  warn "physiclaw itself is installed. Re-run this once your machine can reach"
  warn "huggingface.co:"
  warn "    physiclaw setup local-vision-model"
fi

printf '\n%s%s✓ Done.%s Next steps:\n' "$B" "$G" "$N"
printf '    %sphysiclaw models key <provider>%s        add your LLM API key\n' "$B" "$N"
printf '    %sphysiclaw models use <provider/model>%s  pick the model the agent runs on\n' "$B" "$N"
printf '    %sphysiclaw doctor%s                       check your environment\n' "$B" "$N"
printf '    %sphysiclaw%s                              start the server — opens the hardware-setup wizard\n' "$B" "$N"
if [[ "$FRESH_UV" == "1" ]]; then
  printf '\n%s%s!%s Open a new terminal so uv is on PATH in your interactive shell.\n' \
    "$B" "$Y" "$N"
fi
