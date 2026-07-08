"""``physiclaw update`` — self-update the uv-managed install, plus the
two-phase update check ``physiclaw server`` runs at startup (Phase B stages the
next release in the background; Phase A notifies at the next start that it's
ready — see the "Two-phase update check" block further down).

The installers (install.sh / install.ps1) put physiclaw on the machine
with ``uv tool install physiclaw --python 3.12 --force --refresh`` — an
UNPINNED spec.

A plain ``physiclaw update`` then runs ``uv tool upgrade physiclaw``: it
syncs the latest release INTO the existing environment without recreating
it, so it never touches the running ``python.exe``. That's what makes
updating safe on Windows — a from-inside ``uv tool install --force``
recreates the env and dies removing a locked ``Scripts`` dir (``os error
5``). Same shape as ``npm install -g``: replace the package, keep the
interpreter. ``--version X`` is the exception — changing the pin (or
downgrading) needs a full ``uv tool install physiclaw==X --force``, which
recreates the env and so can fail on Windows while physiclaw runs; run it
from a fresh shell if so.

Success is judged from ``uv tool list`` — the actual on-disk state —
not uv's exit code, which can lie on Windows (Defender briefly locking
the ``physiclaw.exe`` shim executable; see install.ps1's identical logic).

LOCKED-FILE INVARIANT: installing swaps this venv's code on disk while
Python imports lazily, so a physiclaw process that outlives an install
risks loading NEW modules into an OLD process — and on Windows the
running venv's ``python`` + native-extension DLLs are LOCKED, so
recreating the env from inside a live physiclaw dies mid-swap and
corrupts it. So we NEVER self-apply under a running server:

  - ``physiclaw update`` refuses to run while a server is live, and
    itself performs no physiclaw imports after the install — only
    already-loaded code. This is the one path that actually installs, and
    its default ``uv tool upgrade`` leaves the running interpreter alone.
  - ``physiclaw server`` never installs. It only checks + notifies (see
    the two phases below); the user applies with ``physiclaw update``
    once the server is stopped.

The startup update check (``[update] auto = true``, the default) is
two-phase, so no network or heavy download ever sits on the startup
critical path:

  - Phase B (:func:`maybe_stage_update`) — a background daemon thread
    after serving starts: probe PyPI and, if newer, WARM uv's cache for
    that version (so a later ``physiclaw update`` is fast) and drop a
    ``cache/update.json`` marker.
  - Phase A (:func:`notify_staged_update`) — synchronous at startup: if
    the marker names a newer version, print a one-line "ready — run
    ``physiclaw update`` to apply" notice and keep serving. No install.

A notice thus appears one boot after a release is published (stage on
boot N, notify on boot N+1). Every step is fail-soft — an unreachable
PyPI or a failed warm never blocks the server, and nothing the server
does can corrupt the install, because the server never touches it.

Kill switches: ``physiclaw config set update.auto false``, or
``PHYSICLAW_DISABLE_UPDATE_CHECK=1`` (also silences the doctor/status
banner), and CI environments are always skipped. The explicit
``physiclaw update`` command ignores all of these — typed intent wins.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

import typer

from physiclaw import __version__ as _pkg_version
from physiclaw import paths, runtime_state
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

# Env override that suppresses BOTH startup phases (no notice, no stage) for a
# single run — e.g. a supervisor relaunching physiclaw right after applying an
# update, so it doesn't immediately re-notify. Nothing in-process sets it; it's
# an external escape hatch alongside the config/env kill switches.
_AUTO_UPDATE_MARKER = "_PHYSICLAW_AUTO_UPDATED"

# uv resolve+download on a cold cache is normally seconds; the ceiling
# only exists so a wedged network can't hang the server start forever.
_UV_INSTALL_TIMEOUT = 600
_UV_QUERY_TIMEOUT = 30

_NOT_TOOL_MANAGED_MSG = (
    "this physiclaw isn't managed by `uv tool` (dev checkout, pip install, "
    "or uv can't see it).\n"
    "  Update manually — dev checkout: git pull;  pip: pip install -U physiclaw;\n"
    "  or reinstall the supported way: curl -fsSL https://physiclaw.ai/install.sh | bash"
)


def _uv() -> str | None:
    return shutil.which("uv")


def _auto_update_disabled() -> bool:
    """The shared kill-switch gate for both startup phases: off via
    ``[update] auto = false`` / ``PHYSICLAW_DISABLE_UPDATE_CHECK=1``, the
    per-run ``_AUTO_UPDATE_MARKER`` override, or running under CI (an unattended
    mid-job version jump is never what a pipeline wants)."""
    return (
        _disabled_via_env()
        or bool(os.environ.get(_AUTO_UPDATE_MARKER))
        or os.environ.get("CI", "").strip().lower() in ("1", "true", "yes")
        or not CONFIG.update.auto
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


def _upgrade_cmd(uv: str) -> list[str]:
    # The DEFAULT update path. `uv tool upgrade` syncs the newer version INTO
    # the existing environment — it never recreates/removes it (uv docs:
    # "the upgrade operates on this existing environment rather than requiring
    # you to recreate it"). So it never deletes the running `python.exe`, unlike
    # `install --force`, which recreates the env and dies on Windows removing
    # the locked `Scripts` dir (os error 5). Same idea as
    # `npm install -g`: replace the package, keep the interpreter. physiclaw is
    # installed UNPINNED, so upgrade resolves to the latest PyPI release; it also
    # respects and retains the install-time `--python`, so no `--python` here
    # (passing one would rebuild the env — the very thing we're avoiding).
    return [uv, "tool", "upgrade", "physiclaw"]


def _install_cmd(uv: str, spec: str) -> list[str]:
    # Full reinstall — used ONLY to change the pinned version (`--version`,
    # incl. downgrades), where `upgrade` can't help (it respects the install-time
    # constraint). This RECREATES the environment, so on Windows it can fail
    # while physiclaw is running (its venv `python.exe` is locked); if so, run it
    # from a fresh shell. `--refresh` so a just-cut release still resolves.
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    return [uv, "tool", "install", spec, "--python", py, "--force", "--refresh"]


def _run_cmd(cmd: list[str], *, capture: bool) -> subprocess.CompletedProcess | None:
    """Run an install/upgrade command; None means it couldn't run at all
    (missing exe, timeout). ``capture=False`` streams uv's own progress to the
    user — the best diagnostic when something goes wrong."""
    try:
        return subprocess.run(
            cmd, capture_output=capture, encoding="utf-8", errors="replace",
            timeout=_UV_INSTALL_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def update(
    check: Annotated[
        bool,
        typer.Option(
            "--check",
            help="Only check PyPI; install nothing. Exit 0 = up to date, "
                 "1 = update available, 2 = check failed.",
        ),
    ] = False,
    version: Annotated[
        Optional[str],
        typer.Option(
            "--version",
            help="Install this exact version (allows downgrade; pins the "
                 "uv requirement until the next plain `physiclaw update`).",
        ),
    ] = None,
) -> None:
    """Update physiclaw to the latest PyPI release (via uv)."""
    uv = _uv()
    if uv is None:
        typer.echo(fmt.warn(
            "`uv` not found on PATH — physiclaw is installed and updated "
            "through uv.\n"
            "  Install it: https://docs.astral.sh/uv/getting-started/installation/\n"
            "  Or re-run the installer: curl -fsSL https://physiclaw.ai/install.sh | bash"
        ))
        raise typer.Exit(1)

    latest = _fetch_pypi_version()

    if check:
        if latest is None:
            typer.echo(fmt.warn("could not reach PyPI to check for updates."))
            raise typer.Exit(2)
        _write_cache(latest)
        if _is_newer(_pkg_version, latest):
            typer.echo(f"Update available: physiclaw {_pkg_version} → {latest}")
            typer.echo(fmt.next_hint("physiclaw update"))
            raise typer.Exit(1)
        typer.echo(fmt.ok(f"physiclaw {_pkg_version} is up to date."))
        return

    if _tool_version(uv) is None:
        typer.echo(fmt.warn(_NOT_TOOL_MANAGED_MSG))
        raise typer.Exit(1)

    if version is not None:
        if version == _pkg_version:
            typer.echo(fmt.ok(f"physiclaw is already at {version}."))
            return
        # Changing the pinned version → full reinstall (upgrade can't).
        target, cmd = version, _install_cmd(uv, f"physiclaw=={version}")
    else:
        if latest is None:
            typer.echo(fmt.warn(
                "could not reach PyPI to resolve the latest version — "
                "check your network and retry."
            ))
            raise typer.Exit(1)
        if not _is_newer(_pkg_version, latest):
            _write_cache(latest)
            typer.echo(fmt.ok(f"physiclaw {_pkg_version} is up to date."))
            return
        # Default path → in-place upgrade (keeps the running interpreter).
        target, cmd = latest, _upgrade_cmd(uv)

    live = runtime_state.read_live()
    if live:
        # Refuse, don't just warn: the server imports lazily for its whole
        # life, so swapping the venv underneath it would mix old and new
        # code inside a process that may be mid-gesture on real hardware.
        typer.echo(fmt.warn(
            f"a physiclaw server is running (pid {live['pid']}) — updating "
            "underneath it would mix old and new code in that process.\n"
            "  Stop it (Ctrl-C in its terminal), then re-run: physiclaw update"
        ))
        raise typer.Exit(1)

    typer.echo(f"Updating physiclaw {_pkg_version} → {target} …")
    proc = _run_cmd(cmd, capture=False)
    # From here on: no physiclaw imports — the on-disk code just changed
    # (everything below runs on modules loaded before the install).

    # Trust the on-disk state, not the exit code: on Windows uv can
    # report failure while the tool env updated fine (Defender briefly
    # locking the shim executable — install.ps1 handles the same case).
    new = _tool_version(uv)
    installed = new is not None and (
        new == target or (version is None and _is_newer(_pkg_version, new))
    )
    if not installed:
        rc = "did not run" if proc is None else f"exit {proc.returncode}"
        typer.echo(fmt.warn(
            f"update failed (uv {rc}; installed version: {new or 'unknown'}).\n"
            f"  Retry manually with verbose output:\n"
            f"      {' '.join(cmd)} --verbose"
        ))
        raise typer.Exit(1)

    if proc is None or proc.returncode != 0:
        typer.echo(fmt.warn(
            f"uv reported a failure, but physiclaw {new} is installed — "
            "likely a transient file lock; continuing."
        ))

    # Verify the freshly-installed version actually RUNS before declaring
    # success. `uv tool list` reports metadata, not env health: a mid-install
    # failure (network drop, disk full, Ctrl-C) can leave site-packages
    # half-written while the version still shows as installed, so the next
    # `physiclaw` invocation would crash on import. A subprocess `--version`
    # imports the new code in a throwaway process — invariant-safe (no mixing
    # into this old one) and cross-platform (running the shim never needs to
    # *replace* it, so no Windows file lock). Inconclusive (couldn't spawn) is
    # treated as OK — only a definitive non-zero exit fails.
    shim = shutil.which("physiclaw")
    if shim is not None:
        health = _run([shim, "--version"], timeout=_UV_QUERY_TIMEOUT)
        if health is not None and health.returncode != 0:
            typer.echo(fmt.warn(
                f"physiclaw {new} installed but won't run — the environment may "
                "be half-written. Reinstall to repair:\n"
                f"      {' '.join(_install_cmd(uv, 'physiclaw'))}\n"
                "  or: curl -fsSL https://physiclaw.ai/install.sh | bash"
            ))
            raise typer.Exit(1)

    _write_cache(new)
    typer.echo(fmt.ok(f"Updated physiclaw {_pkg_version} → {new}."))
    if version is not None:
        typer.echo(fmt.info(
            f"Requirement pinned at {version}; the next plain "
            "`physiclaw update` unpins it."
        ))


# ── Two-phase update check ─────────────────────────────────────────────
#
# The server never installs (the locked-file invariant), so its job is only to
# find out an update exists and tell the user. Split in two so no network sits
# on the startup critical path:
#
#   Phase B (`maybe_stage_update`) — background daemon thread, after serving
#       starts: probe PyPI and, if newer, WARM uv's cache for that version (so a
#       later `physiclaw update` links from cache in ms) and drop a
#       `cache/update.json` marker. No install; touches only the shared uv cache
#       + the marker, so it's safe mid-serve.
#   Phase A (`notify_staged_update`) — synchronous, at startup: if the marker
#       names a newer version, print a "ready — run `physiclaw update`" notice
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

    Applying a self-update means reinstalling the tool venv, and that can't be
    done safely under a live physiclaw: on Windows the running venv's ``python``
    + native-extension DLLs are locked, so a ``--force`` reinstall dies mid-swap
    and corrupts the env. Rather than dance around that with a detached
    relaunch, we don't auto-apply in the foreground at all — we print a one-line
    notice and keep serving. The user applies it with ``physiclaw update`` (which
    refuses to run while a server is live, so they stop this one first).

    Silent no-op when: disabled (``[update] auto = false`` / env), under CI,
    nothing staged (or the stage isn't actually newer), or not a ``uv tool``
    install (a dev checkout / pip install has nothing we'd tell them to update)."""
    if _auto_update_disabled():
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
        f"physiclaw {staged} available, run `physiclaw update` to apply."
    ))


def maybe_stage_update() -> None:
    """Phase B — background staging for the next startup's apply.

    Probes PyPI and, if a newer release exists, WARMS uv's cache for it (a heavy
    download — physiclaw's opencv/onnxruntime deps — hence background) and drops
    a stage marker (:func:`_write_staged`). No install: it never touches the
    installed tool, only the shared uv cache and the marker file, so it is safe
    to run mid-serve and alongside other servers. Fully fail-soft; meant to run
    in a daemon thread.

    Skips the same kill switches as Phase A. When already at the latest, clears
    any stale marker; when the target is already staged, does nothing."""
    if _auto_update_disabled():
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
