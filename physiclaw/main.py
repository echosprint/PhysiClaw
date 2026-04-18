"""PhysiClaw CLI entry point.

Usage:
    uv run physiclaw [--port 8048] [--host 127.0.0.1] [--verbose]
    uv run physiclaw --warm-start [--cam-index 0]

``--warm-start`` resumes from the last saved calibration: if
``data/calibration/bundle.json`` is complete, connect arm + camera and
mark the server ready without running setup.py. Falls back to normal
boot if the bundle is missing or hardware reconnect fails.
"""

import argparse
import atexit
import logging
import subprocess
import sys


def _try_warm_start(cam_index_override: int | None) -> bool:
    """Resume from the saved Calibration bundle.

    The bundle is already loaded into ``physiclaw.calibration`` at
    ``physiclaw.server.app`` import time; this just connects hardware
    and flips the ready flag. Camera index comes from ``--cam-index`` if
    provided, else from ``bundle.cam_index`` (remembered from the last
    successful setup), else 0. Returns True on success, False with a
    warning on any failure — the caller falls through to normal setup.
    """
    from physiclaw.server.app import physiclaw

    log = logging.getLogger(__name__)
    cal = physiclaw.calibration
    if not cal.complete:
        log.warning(
            "--warm-start: no complete calibration on disk; run setup.py first"
        )
        return False
    cam_index = cam_index_override if cam_index_override is not None else (
        cal.cam_index if cal.cam_index is not None else 0
    )
    try:
        physiclaw.connect_arm()
        physiclaw.connect_camera(cam_index)
    except Exception as e:
        log.warning(
            f"--warm-start: hardware reconnect failed ({e}); run setup.py"
        )
        return False
    physiclaw.mark_ready()
    log.info(
        f"--warm-start: resumed from bundle "
        f"(z_tap={cal.z_tap}mm, cam={cam_index}) — MCP tools ready"
    )
    return True


def _spawn_runtime(port: int, verbose: bool) -> subprocess.Popen:
    """Launch the hook loop as a child process.

    Runs out-of-process so long-running hooks (e.g. shelling out to `claude`)
    don't block the MCP event loop. Terminated via atexit when the server
    exits.
    """
    log = logging.getLogger(__name__)
    cmd = [
        sys.executable,
        "-m",
        "agent.runtime",
        "--server",
        f"http://127.0.0.1:{port}",
    ]
    if verbose:
        cmd.append("--verbose")
    proc = subprocess.Popen(cmd)
    log.info(f"Runtime loop started as subprocess (pid={proc.pid})")
    return proc


def main():
    parser = argparse.ArgumentParser(description="PhysiClaw MCP Server")
    parser.add_argument("--port", type=int, default=8048)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed debug output"
    )
    parser.add_argument(
        "--no-runtime",
        action="store_true",
        help="Don't spawn the runtime loop subprocess",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Auto-connect hardware from the saved calibration bundle and "
        "mark ready, skipping setup.py. Falls through if the bundle is "
        "incomplete or hardware connect fails.",
    )
    parser.add_argument(
        "--cam-index",
        type=int,
        default=None,
        help="Camera index override for --warm-start (default: value "
        "stored in the bundle, falling back to 0)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    from physiclaw.server import mcp, shutdown

    atexit.register(shutdown)

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.settings.log_level = "WARNING"

    from physiclaw.bridge import bridge_base_urls

    log = logging.getLogger(__name__)
    primary, fallback = bridge_base_urls(args.port)
    display_host = "localhost" if args.host == "0.0.0.0" else args.host
    log.info(f"PhysiClaw MCP server on http://{display_host}:{args.port}/mcp")
    log.info(f"QR code (scan with phone): http://localhost:{args.port}/api/bridge/qr")
    if primary != fallback:
        log.info(f"Phone page: {primary}/bridge  (recommended — survives IP changes)")
        log.info(f"Fallback:   {fallback}/bridge  (if mDNS blocked)")
    else:
        log.info(f"Phone page: {fallback}/bridge")
        log.info(
            "Tip: set a stable LocalHostName for <name>.local URLs — "
            "see /phone-setup"
        )
    if not (args.warm_start and _try_warm_start(args.cam_index)):
        if args.warm_start:
            log.warning("--warm-start failed — falling back to normal boot")
        log.info(
            "Run /setup in Claude Code (or: uv run python scripts/setup.py) "
            "to connect hardware and calibrate — server is waiting."
        )

    if not args.no_runtime:
        runtime_proc = _spawn_runtime(args.port, args.verbose)

        def _stop_runtime():
            if runtime_proc.poll() is None:
                runtime_proc.terminate()
                try:
                    runtime_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    runtime_proc.kill()

        atexit.register(_stop_runtime)

    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
