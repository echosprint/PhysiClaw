"""``physiclaw update`` — self-update the uv-managed install, plus the
two-phase auto-update ``physiclaw server`` runs at startup (Phase A applies a
previously-staged release offline; Phase B stages the next one in the
background — see the "Two-phase auto-update" block further down).

The installers (install.sh / install.ps1) put physiclaw on the machine
with ``uv tool install physiclaw --python 3.12 --force --refresh``.
Updating re-runs that exact command — one proven code path for install
and update — with two twists:

  - ``--python`` is the *running* interpreter's version, so a tool env
    someone moved to a newer Python stays there.
  - The requirement spec stays UNPINNED (plain ``physiclaw``), so a
    manual ``uv tool upgrade physiclaw`` keeps working afterwards.
    ``--version X`` installs ``physiclaw==X`` and says so — the next
    plain ``physiclaw update`` moves off the pin again.

Success is judged from ``uv tool list`` — the actual on-disk state —
not uv's exit code, which can lie on Windows (Defender briefly locking
the fresh ``physiclaw.exe`` shim; see install.ps1's identical logic).

VERSION-MIXING INVARIANT: installing swaps this venv's code on disk
while Python imports lazily, so any physiclaw process that outlives an
install risks loading NEW modules into an OLD process. Three rules keep
versions unmixed:

  - ``physiclaw update`` refuses to run while a server is live (the
    server would keep lazily importing for weeks), and itself performs
    no physiclaw imports after the install — only already-loaded code.
  - A successful apply never returns: it hands the process over to the
    new version (:func:`_handoff`) instead of serving with a swapped
    venv underneath.
  - Phase A (applying a staged update) is skipped entirely when a
    hand-off wouldn't be possible (no shim on PATH) or another live
    server would be swapped underneath.

Auto-update (``[update] auto = true``, the default) is two-phase, so no
network or heavy download ever sits on the startup critical path:

  - Phase B (:func:`maybe_stage_update`) — a background daemon thread
    after serving starts: probe PyPI and, if newer, WARM uv's cache for
    that version and drop a ``cache/update.json`` marker.
  - Phase A (:func:`apply_staged_update`) — synchronous at startup,
    OFFLINE: if the marker names a newer version, ``uv tool install
    …==<ver> --offline`` links the warmed wheels in milliseconds, then
    re-execs into it (a waited child on Windows, whose exes are locked).

An update thus lands one boot after it's published (stage on boot N,
apply on boot N+1). Every step is fail-soft — an unreachable PyPI, a
failed warm, or a failed offline apply never blocks the server.

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

# Set before the post-apply hand-off so the re-exec'd process skips BOTH
# phases (no re-apply, no re-stage) — guards against an update loop if the
# install "succeeds" but the resolved version never advances (e.g. a
# shadowing dev shim).
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
    """The shared kill-switch gate for both auto-update phases: off via
    ``[update] auto = false`` / ``PHYSICLAW_DISABLE_UPDATE_CHECK=1``, already
    handed off this start (marker set), or running under CI (an unattended
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


def _install_cmd(uv: str, spec: str) -> list[str]:
    # `--refresh` invalidates uv's cached PyPI metadata so a release cut
    # minutes ago still resolves (same reason install.sh passes it).
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    return [uv, "tool", "install", spec, "--python", py, "--force", "--refresh"]


def _run_install(uv: str, spec: str, *, capture: bool) -> subprocess.CompletedProcess | None:
    """Run the install; None means it couldn't run at all (missing exe,
    timeout). ``capture=False`` streams uv's own progress to the user —
    it is the best diagnostic when something goes wrong."""
    try:
        return subprocess.run(
            _install_cmd(uv, spec),
            capture_output=capture, encoding="utf-8", errors="replace",
            timeout=_UV_INSTALL_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _handoff(exe: str) -> None:
    """Continue as the freshly installed version, same argv. Never returns.

    Required, not cosmetic: the install swapped this venv's code on
    disk, and Python imports lazily — a process that kept running would
    load NEW modules into an OLD process on its next import. POSIX
    truly re-execs. Windows can't replace a running exe, so the old
    process stays only as a thin waiter on a child running the new
    version — it performs no further imports.
    """
    os.environ[_AUTO_UPDATE_MARKER] = "1"
    argv = [exe, *sys.argv[1:]]
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform != "win32":
        try:
            os.execv(exe, argv)  # does not return
        except OSError:
            pass  # fall through to the waiter
    child = subprocess.Popen(argv)
    while True:
        try:
            code = child.wait()
            break
        except KeyboardInterrupt:
            continue  # Ctrl-C reached the child too; wait for its exit
    raise typer.Exit(code)


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
        target, spec = version, f"physiclaw=={version}"
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
        target, spec = latest, "physiclaw"

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
    proc = _run_install(uv, spec, capture=False)
    # From here on: no physiclaw imports — the on-disk code just changed
    # (everything below runs on modules loaded before the install).

    # Trust the on-disk state, not the exit code: on Windows uv can
    # report failure while the tool env updated fine (Defender briefly
    # locking the new shim — install.ps1 handles the same case).
    new = _tool_version(uv)
    installed = new is not None and (
        new == target or (version is None and _is_newer(_pkg_version, new))
    )
    if not installed:
        rc = "did not run" if proc is None else f"exit {proc.returncode}"
        typer.echo(fmt.warn(
            f"update failed (uv {rc}; installed version: {new or 'unknown'}).\n"
            f"  Retry manually with verbose output:\n"
            f"      {' '.join(_install_cmd(uv, spec))} --verbose"
        ))
        raise typer.Exit(1)

    if proc is None or proc.returncode != 0:
        typer.echo(fmt.warn(
            f"uv reported a failure, but physiclaw {new} is installed — "
            "likely a transient file lock; continuing."
        ))
    _write_cache(new)
    typer.echo(fmt.ok(f"Updated physiclaw {_pkg_version} → {new}."))
    if version is not None:
        typer.echo(fmt.info(
            f"Requirement pinned at {version}; the next plain "
            "`physiclaw update` unpins it."
        ))


# ── Two-phase auto-update ──────────────────────────────────────────────
#
# Self-update can't hot-swap the venv under a serving process (the version-
# mixing invariant), so applying an update means installing + re-execing at
# STARTUP, before serving. To keep the network off the startup critical path we
# split the work:
#
#   Phase B (`maybe_stage_update`) — background daemon thread, after serving
#       starts: probe PyPI and, if newer, WARM uv's cache for that version and
#       drop a `cache/update.json` marker. No install, no re-exec; touches only
#       the shared uv cache + the marker, so it's safe mid-serve.
#   Phase A (`apply_staged_update`) — synchronous, at startup, offline: if the
#       marker names a newer version, `uv tool install …==<ver> --offline` links
#       the warmed wheels in milliseconds, then hands off. Zero network.
#
# Net: an update lands one boot after it's published (stage on boot N, apply on
# boot N+1), with no network or heavy download on any startup critical path.


def _stage_file() -> Path:
    return paths.cache_dir() / "update.json"


def _read_staged() -> str | None:
    """The version a prior background stage prepared for offline apply, or None
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


def _offline_install_cmd(uv: str, version: str) -> list[str]:
    """Apply a staged version from uv's ALREADY-WARMED cache. ``--offline``
    forbids network — it links prepared wheels in milliseconds, or fails fast if
    the cache was evicted. No ``--refresh`` (that would hit the network)."""
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    return [uv, "tool", "install", f"physiclaw=={version}",
            "--python", py, "--force", "--offline"]


def _warm_cmd(uv: str, version: str) -> list[str]:
    """Warm uv's global cache for ``physiclaw==version`` WITHOUT touching the
    installed tool or running physiclaw's entrypoint: ``uv tool run`` builds a
    throwaway env with that version — downloading + preparing its wheels into
    the shared cache — and runs a no-op ``python -c pass``. ``--refresh`` so a
    just-cut release resolves."""
    return [uv, "tool", "run", "--refresh", "--from", f"physiclaw=={version}",
            "python", "-c", "pass"]


def apply_staged_update() -> None:
    """Phase A — at ``physiclaw server`` startup, install a previously STAGED
    newer version OFFLINE (uv links the warmed cache — milliseconds) and hand
    the process over to it. Zero network on the critical path: the wheels were
    fetched by a prior background stage (Phase B, :func:`maybe_stage_update`).

    Silent no-op when: disabled (``[update] auto = false`` / env), under CI,
    already handed off this start, nothing staged (or the stage isn't actually
    newer), another live server exists, not a ``uv tool`` install, or no shim to
    hand off to.

    Never hands off to a broken install. After the offline install it checks
    BOTH that the version advanced AND that the new shim actually runs
    (``--version``). Either failure — a cache eviction (which fails at
    resolve/download, before uv touches the env) or a rare corrupt install —
    clears the marker and keeps serving the current (in-memory) version; Phase B
    re-stages next cycle.

    Same version-mixing invariant as before: a SUCCESSFUL, VERIFIED install
    never returns — it re-execs into the new version via :func:`_handoff`,
    performing no physiclaw imports after the on-disk swap."""
    if _auto_update_disabled():
        return
    staged = _read_staged()
    if staged is None:
        return  # nothing staged — the common case; return before any cost
    if not _is_newer(_pkg_version, staged):
        _clear_staged()  # already at/past it — drop the stale marker
        return
    if runtime_state.read_live():
        return  # don't swap the venv under another live server
    uv = _uv()
    if uv is None:
        return
    exe = shutil.which("physiclaw")
    if exe is None:
        return  # nowhere to hand off to after the swap → don't swap
    if _tool_version(uv) is None:
        return  # dev checkout / pip install — never auto-touch

    typer.echo(
        f"applying staged update: physiclaw {_pkg_version} → {staged} (offline; "
        "disable with `physiclaw config set update.auto false`)"
    )
    proc = _run(_offline_install_cmd(uv, staged))
    # No physiclaw imports past here — the on-disk code just changed. (Everything
    # below is a subprocess or already-loaded code, never a fresh import.)
    new = _tool_version(uv)
    if new is None or not _is_newer(_pkg_version, new):
        # Offline apply didn't advance the version — the common case is a cache
        # eviction, which fails at RESOLVE/DOWNLOAD before uv touches the env,
        # so the current install is untouched. Serve current; Phase B re-stages.
        _clear_staged()
        detail = "did not run" if proc is None else f"exit {proc.returncode}"
        typer.echo(fmt.warn(
            f"staged update to {staged} didn't apply (uv {detail}) — continuing "
            f"on {_pkg_version}; will re-stage."
        ))
        return
    # uv reports the new version installed — but `uv tool list` reports metadata,
    # not env health. Verify the freshly-installed version actually RUNS before
    # we hand off, so a rare corrupt install (uv #8812/#17378) can't crash the
    # re-exec'd process and take the server down. `<shim> --version` is a
    # subprocess (invariant-safe) that imports the new code in a throwaway
    # process; on failure we keep serving the running (old) version.
    healthy = _run([exe, "--version"])
    if healthy is None or healthy.returncode != 0:
        _clear_staged()
        typer.echo(fmt.warn(
            f"physiclaw {new} installed but does not run — staying on "
            f"{_pkg_version} and re-staging. Retry manually: physiclaw update"
        ))
        return
    _write_cache(new)
    _clear_staged()
    typer.echo(fmt.ok(f"physiclaw {new} installed (offline) — restarting."))
    _handoff(exe)


def maybe_stage_update() -> None:
    """Phase B — background staging for the next startup's offline apply.

    Probes PyPI and, if a newer release exists, WARMS uv's cache for it (a heavy
    download — physiclaw's opencv/onnxruntime deps — hence background) and drops
    a stage marker (:func:`_write_staged`). No install and no re-exec: it never
    touches the installed tool, only the shared uv cache and the marker file, so
    it is safe to run mid-serve and alongside other servers. Fully fail-soft;
    meant to run in a daemon thread.

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
