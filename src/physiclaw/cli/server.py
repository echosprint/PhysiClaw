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

from physiclaw.common.config import CONFIG


def server(
    port: Annotated[
        int,
        typer.Option(
            "--port",
            help="Control-plane port (MCP + setup). The "
            "LAN phone bridge always listens on port+1.",
        ),
    ] = CONFIG.server.port,
    host: Annotated[
        str,
        typer.Option(
            "--host",
            help="Control-plane bind address. Loopback "
            "by default — the phone bridge is the only plane that binds 0.0.0.0.",
        ),
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
    """Run the PhysiClaw MCP server.

    Startup is a straight sequence of phases — a few inline, the rest in the
    `_*` helpers below: banner + maintenance → save-flag env → logging → MCP
    init → model resolution → endpoint log → hardware bring-up → runtime
    subprocess → serve. The two background flows (hardware bring-up, runtime
    loop) start daemon threads / a subprocess so `_serve` can start first.
    """
    from physiclaw import __version__
    from physiclaw.common.logger import setup_logging

    # Logging first: everything from here on — including the version line and
    # the background auto-sync's outcome — speaks in one voice, the
    # timestamped `[physiclaw]` stream.
    setup_logging("physiclaw", logging.DEBUG if verbose else logging.INFO)
    logging.getLogger("mcp").setLevel(logging.WARNING)
    # Phones opening the bridge URL probe https:// first (Safari's HTTPS
    # upgrade, VPN apps); the TLS bytes hitting our plaintext port make
    # uvicorn warn "Invalid HTTP request received." once per attempt —
    # noise with no action for the user. Drop that one message; every
    # other uvicorn warning still surfaces.
    logging.getLogger("uvicorn.error").addFilter(
        lambda record: "Invalid HTTP request received" not in record.getMessage()
    )
    log = logging.getLogger(__name__)
    log.info("PhysiClaw %s", __version__)
    _run_startup_maintenance()
    _apply_save_flags(
        save_tool_calls, save_snapshots, save_screenshots, save_raw_camera
    )

    from physiclaw.core.server import build_apps, shutdown

    atexit.register(shutdown)

    model_ref, runtime_label = _resolve_and_record_model(host, port)
    _log_endpoints(host, port)
    _start_hardware_bringup(
        host,
        port,
        warm_start=warm_start,
        cam_index=cam_index,
        auto_calibrate=auto_calibrate,
        no_setup_hardware=no_setup_hardware,
    )
    _start_runtime_loop(
        port,
        verbose,
        no_runtime=no_runtime,
        model_ref=model_ref,
        runtime_label=runtime_label,
    )

    try:
        _serve(host, port, *build_apps(host))
    except KeyboardInterrupt:
        pass


def _serve(host: str, port: int, control_app, bridge_app) -> None:
    """Serve the two planes from one process, two listeners.

    Control (MCP + setup + calibrate) binds `host` (loopback by default) on
    `port`; the phone bridge binds 0.0.0.0 on `port`+1 so ONLY those seven
    routes are reachable from the LAN — the arm-driving surface can't be.
    Two listeners because a wildcard and a specific bind on one port don't
    coexist on Linux.

    The bridge runs in a daemon thread (uvicorn skips signal handling off
    the main thread); the control server keeps the main thread, so Ctrl-C
    behaves exactly as it did under `mcp.run()` — uvicorn catches SIGINT,
    shuts down gracefully, then re-raises so atexit handlers still fire.
    """
    import uvicorn

    from physiclaw.core.bridge.lan import bridge_port

    bridge_server = uvicorn.Server(
        uvicorn.Config(
            bridge_app, host="0.0.0.0", port=bridge_port(port), log_level="warning"
        )
    )
    threading.Thread(target=bridge_server.run, name="bridge-http", daemon=True).start()
    uvicorn.Server(
        uvicorn.Config(control_app, host=host, port=port, log_level="warning")
    ).run()


def _run_startup_maintenance() -> None:
    """Startup housekeeping that must NOT block serving.

    Phase A update check: if a prior run staged a newer version, print a
    "run `uv tool upgrade physiclaw`" notice and keep going. The server NEVER
    self-installs (reinstalling the venv under a live physiclaw corrupts it on
    Windows — see cli/update.py); the user upgrades manually via uv.

    Then two off-critical-path background flows:
      - skills: sync the official pack; a session picks it up at its next wake.
      - update: Phase B — stage the NEXT release (probe + warm uv's cache +
        marker) so the next start can notify it's ready and the user's
        `uv tool upgrade` links it from cache fast.
    """
    from physiclaw.cli.sync_official_skills import maybe_auto_sync
    from physiclaw.cli.update import notify_staged_update, start_stage_update_thread

    notify_staged_update()
    maybe_auto_sync()
    start_stage_update_thread()


def _apply_save_flags(
    save_tool_calls: bool,
    save_snapshots: bool,
    save_screenshots: bool,
    save_raw_camera: bool,
) -> None:
    """Translate the --save-* flags into the env vars the core reads."""
    for enabled, env in (
        (save_tool_calls, "PHYSICLAW_SAVE_TOOL_CALLS"),
        (save_snapshots, "PHYSICLAW_SAVE_SNAPSHOTS"),
        (save_screenshots, "PHYSICLAW_SAVE_SCREENSHOTS"),
        (save_raw_camera, "PHYSICLAW_SAVE_RAW_CAMERA"),
    ):
        if enabled:
            os.environ[env] = "1"


def _resolve_and_record_model(host: str, port: int) -> tuple[Optional[str], str]:
    """Resolve the engine model ref and record the live (host, port, model).

    Resolve ONCE here, in the same env the user invoked `physiclaw server`
    from, so `doctor` in another shell reads the live choice instead of
    re-resolving against an env that may be missing PHYSICLAW_MODEL. The
    runtime subprocess reuses this via the pre-built label. A bad ref is
    non-fatal for the HTTP server — record nothing (model_ref=None) and let
    the runtime subprocess report the real error.

    `runtime_state.write` also lets `doctor` probe the actual server, not the
    config-default port. atexit covers normal exits and KeyboardInterrupt;
    SIGTERM-without-cleanup is handled by doctor's pid-liveness check on read.

    Returns (model_ref, runtime_label); model_ref is None when no model is set.
    """
    from physiclaw.common import runtime_state
    from physiclaw.agent.runtime.launcher import engine_label, resolve as _resolve_model

    try:
        model_ref, model_source = _resolve_model()
        runtime_label = engine_label(model_ref)
    except RuntimeError:
        model_ref, model_source = None, None
        runtime_label = "engine=(unset)"
    runtime_state.write(host, port, model_ref=model_ref, model_source=model_source)
    atexit.register(runtime_state.clear)
    return model_ref, runtime_label


def _log_endpoints(host: str, port: int) -> None:
    """Log the MCP URL, the QR link, and the phone /bridge URLs — primary
    (mDNS, survives IP changes) plus IP fallback, or a single URL and a
    LocalHostName tip when the two coincide."""
    log = logging.getLogger(__name__)
    from physiclaw.core.bridge import bridge_base_urls

    primary, fallback = bridge_base_urls(port)
    display_host = "localhost" if host == "0.0.0.0" else host
    log.info(f"PhysiClaw MCP server on http://{display_host}:{port}/mcp")
    log.info(f"QR code (scan with phone): http://localhost:{port}/api/bridge/qr")
    if primary != fallback:
        log.info(f"Phone page: {primary}/bridge  (recommended — survives IP changes)")
        log.info(f"Fallback:   {fallback}/bridge  (if mDNS blocked)")
    else:
        log.info(f"Phone page: {fallback}/bridge")
        log.info(
            "Tip: set a stable LocalHostName for <name>.local URLs — "
            "see `physiclaw setup phone` (coming soon)."
        )


def _start_hardware_bringup(
    host: str,
    port: int,
    *,
    warm_start: bool,
    cam_index: Optional[int],
    auto_calibrate: bool,
    no_setup_hardware: bool,
) -> None:
    """Dispatch hardware bring-up to one of four mutually exclusive modes. Each
    (except --no-setup-hardware, which just waits) runs in a daemon thread so
    `mcp.run()` can serve first — the phone needs the server already listening
    to load /bridge."""
    if warm_start:
        _warm_start_hardware(host, port, cam_index)
    elif auto_calibrate:
        _auto_calibrate_hardware(host, port)
    elif no_setup_hardware:
        logging.getLogger(__name__).info(
            "Run `physiclaw setup hardware` in another shell to connect "
            "hardware and calibrate — server is waiting."
        )
    else:
        _open_hardware_wizard(host, port)


def _warm_start_hardware(host: str, port: int, cam_index: Optional[int]) -> None:
    """--warm-start: resume from the saved calibration bundle in a background
    thread. On failure, raise KeyboardInterrupt in the main thread (via
    _thread.interrupt_main — cross-platform; os.kill(SIGINT) on Windows would
    TerminateProcess and skip atexit) so `mcp.run()` exits and the atexit
    handlers (shutdown, arm park-off-screen) still fire cleanly."""
    log = logging.getLogger(__name__)
    from physiclaw.core.server import warm_start as ws

    def _warm_start_thread() -> None:
        if not ws.wait_for_port(host, port):
            log.error(
                "warm-start: server never started accepting connections; exiting."
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


def _auto_calibrate_hardware(host: str, port: int) -> None:
    """--auto-calibrate (`physiclaw auto`): no desktop wizard — a background
    thread waits for the phone bridge, then calibrates unattended. The worker
    lives in the setup module (with the wizard it drives); import it lazily so
    `physiclaw server` doesn't pull the CLI setup module in at startup."""
    log = logging.getLogger(__name__)
    log.info("Auto mode: calibration will start when the phone opens /bridge.")

    def _auto_calibrate_thread() -> None:
        import importlib

        hw = importlib.import_module("physiclaw.cli.setup.hardware")
        hw.await_bridge_and_calibrate(host, port)

    threading.Thread(target=_auto_calibrate_thread, daemon=True).start()


def _open_hardware_wizard(host: str, port: int) -> None:
    """Default mode: open the browser hardware-setup wizard once the server is
    actually accepting connections (the page immediately calls /api/status).
    Runs in a daemon thread so `mcp.run()` can start serving first."""
    log = logging.getLogger(__name__)
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


def _start_runtime_loop(
    port: int,
    verbose: bool,
    *,
    no_runtime: bool,
    model_ref: Optional[str],
    runtime_label: str,
) -> None:
    """Spawn the agent runtime subprocess and register its atexit terminator,
    unless disabled (--no-runtime) or no model is configured."""
    log = logging.getLogger(__name__)
    if no_runtime:
        log.info("Runtime loop disabled by --no-runtime.")
        return
    if model_ref is None:
        # First-run case: server is useful for hardware setup + manual MCP
        # tool calls, but the agent can't wake without a model. Skip spawn
        # rather than letting the subprocess crash with a stack trace.
        # Reuse `_NO_MODEL_MSG` so this hint stays in sync with the
        # RuntimeError raised elsewhere — single source of truth.
        from physiclaw.common.config import _NO_MODEL_MSG

        log.warning(
            "Runtime loop NOT started — %s\n"
            "  The MCP server is running and you can use it for hardware setup,\n"
            "  but the agent won't wake. After setting a model, restart "
            "`physiclaw server`.",
            _NO_MODEL_MSG,
        )
        return

    runtime_proc = _spawn_runtime(port, verbose, runtime_label)

    def _stop_runtime() -> None:
        if runtime_proc.poll() is None:
            runtime_proc.terminate()
            try:
                runtime_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                runtime_proc.kill()

    atexit.register(_stop_runtime)


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
