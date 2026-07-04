"""``physiclaw update`` — self-update the uv-managed install, plus the
auto-update hook that ``physiclaw server`` runs at startup.

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
  - A successful auto-update never returns: it hands the process over
    to the new version (:func:`_handoff`) instead of serving with a
    swapped venv underneath.
  - Auto-update is skipped entirely when a hand-off wouldn't be
    possible (no shim on PATH) or another live server would be swapped
    underneath.

Auto-update: ``[update] auto = true`` (the default) makes
``physiclaw server`` check PyPI at startup and install a newer release
before serving, then restart into it — a true re-exec on macOS/Linux;
on Windows the running executables are locked, so the old process
stays only as a thin waiter on a child running the new version. Every
step is fail-soft — an unreachable PyPI or a failed install never
blocks the server.

Kill switches: ``physiclaw config set update.auto false``, or
``PHYSICLAW_DISABLE_UPDATE_CHECK=1`` (also silences the doctor/status
banner), and CI environments are always skipped. The explicit
``physiclaw update`` command ignores all of these — typed intent wins.
"""

import os
import shutil
import subprocess
import sys
from typing import Annotated, Optional

import typer

from physiclaw import __version__ as _pkg_version
from physiclaw import runtime_state
from physiclaw.cli import _format as fmt
from physiclaw.cli._update_check import (
    _disabled_via_env,
    _fetch_pypi_version,
    _is_newer,
    _write_cache,
)
from physiclaw.config import CONFIG

# Set before the post-update hand-off so the new process doesn't check
# again — guards against an update loop if the install "succeeds" but
# the resolved version never advances (e.g. a shadowing dev shim).
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


def _tool_version(uv: str) -> str | None:
    """Version of the ``uv tool``-managed physiclaw, or None if there
    isn't one (or uv itself fails). Parses ``uv tool list`` lines shaped
    ``physiclaw v0.1.13`` — entry-point lines start with ``- `` and are
    skipped by the first-token match."""
    try:
        # Explicit UTF-8: uv emits UTF-8 regardless of the console
        # codepage, and non-UTF-8 Windows locales (cp936, cp1252) would
        # make locale-default decoding throw (same case config.py
        # handles for config.toml).
        out = subprocess.run(
            [uv, "tool", "list"],
            capture_output=True, encoding="utf-8", errors="replace",
            timeout=_UV_QUERY_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
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


def maybe_auto_update() -> None:
    """Best-effort self-update at ``physiclaw server`` startup.

    Silent no-op when: disabled (``[update] auto = false`` or
    ``PHYSICLAW_DISABLE_UPDATE_CHECK=1``), running under CI, already
    handed off once this start, another live server exists (installing
    would swap code underneath it), no shim on PATH to hand off to,
    not a ``uv tool`` install (dev checkouts are never touched), PyPI
    is unreachable, or we're already at the latest version.

    A failed install warns and serves the current version — the server
    must come up regardless. A SUCCESSFUL install never returns: the
    process hands itself over to the new version (see :func:`_handoff`)
    rather than serving with a swapped venv underneath.
    """
    if _disabled_via_env() or os.environ.get(_AUTO_UPDATE_MARKER):
        return
    # CI runs must stay on the version the pipeline installed — an
    # unattended mid-job version jump is never what a pipeline wants.
    if os.environ.get("CI", "").strip().lower() in ("1", "true", "yes"):
        return
    if not CONFIG.update.auto:
        return
    if runtime_state.read_live():
        return  # another instance is serving from this venv
    uv = _uv()
    if uv is None:
        return
    exe = shutil.which("physiclaw")
    if exe is None:
        return  # nowhere to hand off to after the swap → don't swap
    if _tool_version(uv) is None:
        return  # dev checkout / pip install — never auto-touch
    latest = _fetch_pypi_version()
    if latest is None:
        return
    _write_cache(latest)
    if not _is_newer(_pkg_version, latest):
        return

    typer.echo(
        f"auto-update: physiclaw {_pkg_version} → {latest} "
        "(disable with `physiclaw config set update.auto false`)"
    )
    proc = _run_install(uv, "physiclaw", capture=True)
    new = _tool_version(uv)
    if new is None or not _is_newer(_pkg_version, new):
        detail = "did not run" if proc is None else f"exit {proc.returncode}"
        tail = (proc.stderr or "").strip().splitlines()[-3:] if proc else []
        for line in tail:
            typer.echo(fmt.info(line))
        typer.echo(fmt.warn(
            f"auto-update failed (uv {detail}) — continuing on "
            f"{_pkg_version}. Run `physiclaw update` to see the full error."
        ))
        return
    _write_cache(new)
    typer.echo(fmt.ok(f"physiclaw {new} installed — restarting."))
    _handoff(exe)
