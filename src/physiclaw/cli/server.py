"""``physiclaw server`` — run the MCP server (and the agent runtime subprocess)."""

import _thread
import atexit
import logging
import os
import subprocess
import sys
import threading
from typing import Annotated, Optional

import typer

from physiclaw.config import CONFIG


def _spawn_runtime(port: int, verbose: bool, label: str) -> subprocess.Popen:
    """Run the hook loop out-of-process so long-running hooks don't block
    the MCP event loop. Terminated via atexit when the server exits.

    `label` is the pre-resolved engine string (e.g. "engine=claude-code")
    passed by the caller — the caller already did provider resolution to
    record into runtime_state, so reuse that instead of resolving again.
    """
    log = logging.getLogger(__name__)
    cmd = [
        sys.executable,
        "-m",
        "physiclaw.agent.runtime",
        "--server",
        f"http://127.0.0.1:{port}",
    ]
    if verbose:
        cmd.append("--verbose")
    proc = subprocess.Popen(cmd)
    log.info(f"Runtime loop started as subprocess (pid={proc.pid}, {label})")
    return proc


def server(
    port: Annotated[
        int, typer.Option("--port", help="MCP server port.")
    ] = CONFIG.server.port,
    host: Annotated[
        str, typer.Option("--host", help="Bind address.")
    ] = CONFIG.server.host,
    verbose: Annotated[
        bool,
        typer.Option("-v", "--verbose", help="Show detailed debug output."),
    ] = False,
    no_runtime: Annotated[
        bool,
        typer.Option(
            "--no-runtime", help="Don't spawn the agent runtime loop subprocess."
        ),
    ] = False,
    warm_start: Annotated[
        bool,
        typer.Option(
            "--warm-start",
            help="Auto-connect hardware from the saved calibration bundle and "
            "mark ready, skipping `setup hardware`. Falls through if the "
            "bundle is incomplete or hardware connect fails.",
        ),
    ] = False,
    cam_index: Annotated[
        Optional[int],
        typer.Option(
            "--cam-index",
            help="Camera index override for --warm-start (default: value "
            "stored in the bundle, falling back to 0).",
        ),
    ] = None,
    no_setup_hardware: Annotated[
        bool,
        typer.Option(
            "--no-setup-hardware",
            help="Don't auto-open the browser hardware-setup wizard on start.",
        ),
    ] = False,
    auto_calibrate: Annotated[
        bool,
        typer.Option(
            "--auto-calibrate",
            hidden=True,
            help="Internal (used by `physiclaw auto`): skip the desktop wizard "
            "and calibrate unattended when the phone bridge opens.",
        ),
    ] = False,
    save_tool_calls: Annotated[
        bool,
        typer.Option(
            "--save-tool-calls",
            help="Write every peek/screenshot output under the user data dir.",
        ),
    ] = CONFIG.server.save_tool_calls,
    save_snapshots: Annotated[
        bool,
        typer.Option(
            "--save-snapshots",
            help="Write each snapshot frame (rotated, with the bbox overlay) "
                 "under the user data dir.",
        ),
    ] = CONFIG.server.save_snapshots,
    save_screenshots: Annotated[
        bool,
        typer.Option(
            "--save-screenshots",
            help="Write every raw phone-own screenshot under the user data dir.",
        ),
    ] = CONFIG.server.save_screenshots,
    save_raw_camera: Annotated[
        bool,
        typer.Option(
            "--save-raw-camera",
            help="Write every raw camera frame as it's captured under the user "
                 "data dir (calibration, peek, snapshot). Clear with "
                 "`physiclaw clear`.",
        ),
    ] = CONFIG.server.save_raw_camera,
) -> None:
    """Run the PhysiClaw MCP server."""
    # Phase A self-update BEFORE any state is written or hardware touched:
    # if a newer version was staged by a prior run, install it OFFLINE (uv
    # links the warmed cache in ms) and hand this process over to it (re-exec
    # on macOS/Linux, waited child on Windows) — it must not continue here, or
    # lazy imports would load new code into this old process. Zero network on
    # the critical path; fail-soft — never blocks the server (see cli/update.py).
    from physiclaw.cli.update import apply_staged_update

    apply_staged_update()

    # Background daemon threads — both run off the startup critical path so
    # neither blocks serving. After any Phase A hand-off, so the current version
    # does the work:
    #   - skills: sync the official pack; a session picks it up at its next wake.
    #   - update: Phase B — stage the NEXT release (probe + warm uv's cache +
    #     marker) so the next startup can apply it offline.
    from physiclaw.cli.sync_official_skills import maybe_auto_sync
    from physiclaw.cli.update import start_stage_update_thread

    maybe_auto_sync()
    start_stage_update_thread()

    from physiclaw.core.logger import setup_logging

    for enabled, env in (
        (save_tool_calls, "PHYSICLAW_SAVE_TOOL_CALLS"),
        (save_snapshots, "PHYSICLAW_SAVE_SNAPSHOTS"),
        (save_screenshots, "PHYSICLAW_SAVE_SCREENSHOTS"),
        (save_raw_camera, "PHYSICLAW_SAVE_RAW_CAMERA"),
    ):
        if enabled:
            os.environ[env] = "1"

    setup_logging("physiclaw", logging.DEBUG if verbose else logging.INFO)
    logging.getLogger("mcp").setLevel(logging.WARNING)
    from physiclaw.core.server import mcp, shutdown

    atexit.register(shutdown)

    mcp.settings.host = host
    mcp.settings.port = port
    mcp.settings.log_level = "WARNING"

    # Record the live (host, port) so `physiclaw doctor` probes the actual
    # server, not the config-default port. atexit covers normal exits and
    # KeyboardInterrupt; SIGTERM-without-cleanup is handled by doctor's
    # pid-liveness check on read.
    from physiclaw import runtime_state
    from physiclaw.agent.runtime.launcher import engine_label, resolve as _resolve_model

    # Resolve once here (in the same env the user invoked `physiclaw server`
    # from) so `doctor` in another shell can read the live choice instead of
    # re-resolving against an env that may be missing PHYSICLAW_MODEL.
    # The runtime subprocess gets this same resolution via the pre-built
    # label. A bad ref at this point is non-fatal for the HTTP server —
    # record nothing and let the runtime subprocess report the real error.
    try:
        _model_ref, _model_source = _resolve_model()
        _runtime_label = engine_label(_model_ref)
    except RuntimeError:
        _model_ref, _model_source = None, None
        _runtime_label = "engine=(unset)"
    runtime_state.write(
        host, port, model_ref=_model_ref, model_source=_model_source,
    )
    atexit.register(runtime_state.clear)

    from physiclaw.core.bridge import bridge_base_urls

    log = logging.getLogger(__name__)
    primary, fallback = bridge_base_urls(port)
    display_host = "localhost" if host == "0.0.0.0" else host
    log.info(f"PhysiClaw MCP server on http://{display_host}:{port}/mcp")
    log.info(f"QR code (scan with phone): http://localhost:{port}/api/bridge/qr")
    if primary != fallback:
        log.info(
            f"Phone page: {primary}/bridge  (recommended — survives IP changes)"
        )
        log.info(f"Fallback:   {fallback}/bridge  (if mDNS blocked)")
    else:
        log.info(f"Phone page: {fallback}/bridge")
        log.info(
            "Tip: set a stable LocalHostName for <name>.local URLs — "
            "see `physiclaw setup phone` (coming soon)."
        )
    if warm_start:
        # Run warm-start in a background thread so mcp.run() below can start
        # serving HTTP first — the phone needs the server listening to load
        # /bridge and POST screen_dimension / touches. On failure, raise
        # KeyboardInterrupt in the main thread (via _thread.interrupt_main —
        # cross-platform; os.kill(SIGINT) on Windows would TerminateProcess
        # and skip atexit) so mcp.run exits and atexit handlers
        # (shutdown, arm park-off-screen) still fire cleanly.
        from physiclaw.core.server import warm_start as ws

        def _warm_start_thread() -> None:
            if not ws.wait_for_port(host, port):
                log.error(
                    "warm-start: server never started accepting connections; "
                    "exiting."
                )
                _thread.interrupt_main()
                return
            if not ws.try_resume(cam_index):
                log.error(
                    "Exiting. Re-run without --warm-start, then "
                    "`physiclaw setup hardware` to recalibrate."
                )
                _thread.interrupt_main()

        threading.Thread(target=_warm_start_thread, daemon=True).start()
    elif auto_calibrate:
        # `physiclaw auto`: no desktop wizard — a background thread waits for
        # the phone bridge, then runs calibration unattended. The worker lives
        # in the setup module (with the wizard it drives); import it lazily so
        # `physiclaw server` doesn't pull the CLI setup module in at startup.
        log.info("Auto mode: calibration will start when the phone opens /bridge.")

        def _auto_calibrate_thread() -> None:
            import importlib

            hw = importlib.import_module("physiclaw.cli.setup.hardware")
            hw.await_bridge_and_calibrate(host, port)

        threading.Thread(target=_auto_calibrate_thread, daemon=True).start()
    elif no_setup_hardware:
        log.info(
            "Run `physiclaw setup hardware` in another shell to connect "
            "hardware and calibrate — server is waiting."
        )
    else:
        # Open the browser hardware-setup wizard once the server is actually
        # accepting connections (the page immediately calls /api/status).
        # Runs in a daemon thread so mcp.run() below can start serving first.
        from physiclaw.core.server.warm_start import wait_for_port

        setup_url = f"http://localhost:{port}/setup-hardware"
        log.info(f"Hardware-setup wizard: {setup_url}  (disable with --no-setup-hardware)")

        def _open_setup() -> None:
            import webbrowser

            if wait_for_port(host, port):
                try:
                    webbrowser.open(setup_url)
                except Exception:  # noqa: BLE001 — never let a headless box crash startup
                    log.debug("could not open browser for setup wizard", exc_info=True)

        threading.Thread(target=_open_setup, daemon=True).start()

    if no_runtime:
        log.info("Runtime loop disabled by --no-runtime.")
    elif _model_ref is None:
        # First-run case: server is useful for hardware setup + manual MCP
        # tool calls, but the agent can't wake without a model. Skip spawn
        # rather than letting the subprocess crash with a stack trace.
        # Reuse `_NO_MODEL_MSG` so this hint stays in sync with the
        # RuntimeError raised elsewhere — single source of truth.
        from physiclaw.config import _NO_MODEL_MSG
        log.warning(
            "Runtime loop NOT started — %s\n"
            "  The MCP server is running and you can use it for hardware setup,\n"
            "  but the agent won't wake. After setting a model, restart "
            "`physiclaw server`.",
            _NO_MODEL_MSG,
        )
    else:
        runtime_proc = _spawn_runtime(port, verbose, _runtime_label)

        def _stop_runtime() -> None:
            if runtime_proc.poll() is None:
                runtime_proc.terminate()
                try:
                    runtime_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    runtime_proc.kill()

        atexit.register(_stop_runtime)

    try:
        mcp.run(transport="streamable-http")
    except KeyboardInterrupt:
        pass
