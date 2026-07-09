"""Startup update check for ``physiclaw server`` — two phases so no network or
heavy download ever sits on the startup critical path.

physiclaw installs as a ``uv tool`` (install.sh / install.ps1 run
``uv tool install physiclaw --python 3.12 --force --refresh``, an UNPINNED
spec). We deliberately do NOT self-install: a process can't safely reinstall
its own venv while running — on Windows the running interpreter's ``python`` +
native-extension DLLs are LOCKED, so a reinstall dies mid-swap and corrupts the
env. So updates run through uv, not physiclaw: ``physiclaw update`` just PRINTS
the ``uv tool upgrade physiclaw`` command (:func:`update`), and the server only
CHECKS + NOTIFIES — the user runs the upgrade themselves once the server is
stopped.

``uv tool upgrade`` is the right command: it syncs the latest release INTO the
existing environment without recreating it (uv docs: "the upgrade operates on
this existing environment rather than requiring you to recreate it"), so it
never deletes the running interpreter — like ``npm install -g``: replace the
package, keep the interpreter. (A ``uv tool install --force`` recreates the env
and fails on Windows removing the locked ``Scripts`` dir — ``os error 5``.)

The two phases:

  - Phase B (:func:`maybe_stage_update`) — a background daemon thread after
    serving starts: probe PyPI and, if newer, WARM uv's cache for that version
    (so the user's later ``uv tool upgrade`` links from cache in ms) and drop a
    ``cache/update.json`` marker.
  - Phase A (:func:`notify_staged_update`) — synchronous at startup: if the
    marker names a newer version, print a one-line "run ``uv tool upgrade
    physiclaw``" notice and keep serving. No install.

A notice thus appears one boot after a release is published (stage on boot N,
notify on boot N+1). Every step is fail-soft — an unreachable PyPI or a failed
warm never blocks the server, and nothing the server does can corrupt the
install, because the server never touches it.

Kill switches: ``physiclaw config set update.check false``, or
``PHYSICLAW_DISABLE_UPDATE_CHECK=1`` (also silences the doctor/status banner),
and CI environments are always skipped.
"""

import json
import logging
import os
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

import typer

from physiclaw import __version__ as _pkg_version
from physiclaw import paths
from physiclaw.cli import _format as fmt
from physiclaw.cli._update_check import (
    _disabled_via_env,
    _fetch_pypi_version,
    _is_newer,
    _write_cache,
)
from physiclaw.config import CONFIG
from physiclaw.text import read_text, write_text

log = logging.getLogger(__name__)

# uv resolve+download on a cold cache is normally seconds; the ceiling
# only exists so a wedged network can't hang the server start forever.
_UV_INSTALL_TIMEOUT = 600
_UV_QUERY_TIMEOUT = 30


def _uv() -> str | None:
    return shutil.which("uv")


def _update_check_disabled() -> bool:
    """The shared kill-switch gate for both startup phases: off via
    ``[update] check = false`` / ``PHYSICLAW_DISABLE_UPDATE_CHECK=1``, or running
    under CI (an unattended mid-job version jump is never what a pipeline
    wants)."""
    return (
        _disabled_via_env()
        or os.environ.get("CI", "").strip().lower() in ("1", "true", "yes")
        or not CONFIG.update.check
    )


def _run(cmd: list[str], *, timeout: int = _UV_INSTALL_TIMEOUT) -> subprocess.CompletedProcess | None:
    """Run a uv subcommand, capturing UTF-8 output; None if it couldn't run at
    all (missing exe, timeout). Explicit UTF-8 because uv emits UTF-8 regardless
    of the console codepage, and non-UTF-8 Windows locales (cp936, cp1252) would
    make locale-default decoding throw (same case config.py handles for
    config.toml)."""
    try:
        return subprocess.run(
            cmd, capture_output=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _tool_version(uv: str) -> str | None:
    """Version of the ``uv tool``-managed physiclaw, or None if there
    isn't one (or uv itself fails). Parses ``uv tool list`` lines shaped
    ``physiclaw v0.1.13`` — entry-point lines start with ``- `` and are
    skipped by the first-token match."""
    out = _run([uv, "tool", "list"], timeout=_UV_QUERY_TIMEOUT)
    if out is None or out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "physiclaw" and parts[1].startswith("v"):
            return parts[1][1:]
    return None


def update() -> None:
    """Show how to update physiclaw (it's a uv tool — update it with uv)."""
    # We don't self-install — see the module docstring (a reinstall from inside
    # a running physiclaw locks its own files on Windows). So `physiclaw update`
    # is just a signpost to the real command, since it's the intuitive thing to
    # type. `uv tool upgrade` upgrades in place, but the server maps the native
    # DLLs, so it must be stopped first.
    typer.echo(
        "To update, stop the server and run:\n\n"
        "    uv tool upgrade physiclaw"
    )


# ── Two-phase update check ─────────────────────────────────────────────
#
# The server never installs (the locked-file invariant), so its job is only to
# find out an update exists and tell the user. Split in two so no network sits
# on the startup critical path:
#
#   Phase B (`maybe_stage_update`) — background daemon thread, after serving
#       starts: probe PyPI and, if newer, WARM uv's cache for that version (so
#       the user's later `uv tool upgrade` links from cache in ms) and drop a
#       `cache/update.json` marker. No install; touches only the shared uv cache
#       + the marker, so it's safe mid-serve.
#   Phase A (`notify_staged_update`) — synchronous, at startup: if the marker
#       names a newer version, print a "run `uv tool upgrade physiclaw`" notice
#       and keep serving. No install.
#
# Net: a notice appears one boot after a release is published (stage on boot N,
# notify on boot N+1), with no network or heavy download on any startup path.


def _stage_file() -> Path:
    return paths.cache_dir() / "update.json"


def _read_staged() -> str | None:
    """The newer version a prior background stage flagged as ready, or None
    (missing / unreadable / malformed marker)."""
    p = _stage_file()
    if not p.exists():
        return None
    try:
        data = json.loads(read_text(p))
    except (OSError, ValueError):
        return None
    v = data.get("version") if isinstance(data, dict) else None
    return v if isinstance(v, str) and v else None


def _write_staged(version: str) -> None:
    try:
        p = _stage_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        write_text(p, json.dumps({
            "version": version,
            "staged_at": datetime.now(timezone.utc).isoformat(),
        }))
    except OSError:
        pass  # non-fatal — Phase B just re-stages next cycle


def _clear_staged() -> None:
    try:
        _stage_file().unlink(missing_ok=True)
    except OSError:
        pass


def _warm_cmd(uv: str, version: str) -> list[str]:
    """Warm uv's global cache for ``physiclaw==version`` WITHOUT touching the
    installed tool or running physiclaw's entrypoint: ``uv tool run`` builds a
    throwaway env with that version — downloading + preparing its wheels into
    the shared cache — and runs a no-op ``python -c pass``. ``--refresh`` so a
    just-cut release resolves."""
    return [uv, "tool", "run", "--refresh", "--from", f"physiclaw=={version}",
            "python", "-c", "pass"]


def notify_staged_update() -> None:
    """Phase A — at ``physiclaw server`` startup, NOTIFY that a newer version is
    staged and ready, without touching the install.

    The server never self-installs (reinstalling the venv under a live physiclaw
    corrupts it on Windows — see the module docstring), so we print a one-line
    notice and keep serving. The user applies it with ``uv tool upgrade
    physiclaw`` once the server is stopped (the server has the native-extension
    DLLs mapped, so the upgrade must run with it stopped).

    Silent no-op when: disabled (``[update] check = false`` / env), under CI,
    nothing staged (or the stage isn't actually newer), or not a ``uv tool``
    install (a dev checkout / pip install has nothing we'd tell them to update)."""
    if _update_check_disabled():
        return
    staged = _read_staged()
    if staged is None:
        return  # nothing staged — the common case; return before any cost
    if not _is_newer(_pkg_version, staged):
        _clear_staged()  # already at/past it — drop the stale marker
        return
    uv = _uv()
    if uv is None or _tool_version(uv) is None:
        return  # dev checkout / pip install — nothing to point them at

    typer.echo(fmt.info(
        f"physiclaw {staged} available (you're on {_pkg_version}) — "
        "stop the server and run `uv tool upgrade physiclaw`."
    ))


def maybe_stage_update() -> None:
    """Phase B — background staging for the next startup's notice.

    Probes PyPI and, if a newer release exists, WARMS uv's cache for it (a heavy
    download — physiclaw's opencv/onnxruntime deps — hence background) and drops
    a stage marker (:func:`_write_staged`). No install: it never touches the
    installed tool, only the shared uv cache and the marker file, so it is safe
    to run mid-serve and alongside other servers. Fully fail-soft; meant to run
    in a daemon thread.

    Skips the same kill switches as Phase A. When already at the latest, clears
    any stale marker; when the target is already staged, does nothing."""
    if _update_check_disabled():
        return
    uv = _uv()
    if uv is None:
        return
    if _tool_version(uv) is None:
        return  # dev checkout / pip install — never stage
    latest = _fetch_pypi_version()
    if latest is None:
        return
    _write_cache(latest)
    if not _is_newer(_pkg_version, latest):
        _clear_staged()  # we're current — nothing to stage
        return
    if _read_staged() == latest:
        return  # already staged this version — don't re-download
    proc = _run(_warm_cmd(uv, latest))
    if proc is None or proc.returncode != 0:
        log.info("stage-update: warming cache for %s failed — will retry", latest)
        return  # warm failed — don't mark; retry next cycle
    _write_staged(latest)
    log.info("stage-update: physiclaw %s staged for next startup", latest)


def start_stage_update_thread() -> None:
    """Run Phase B (:func:`maybe_stage_update`) in a background daemon thread so
    the heavy cache warm never blocks startup. Spawned from here (not the caller)
    so ``server``'s own thread wiring stays untouched."""
    threading.Thread(
        target=maybe_stage_update, name="physiclaw-stage-update", daemon=True
    ).start()
