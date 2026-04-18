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
import os
import signal
import subprocess
import sys
import threading


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
    if args.warm_start:
        # Run warm-start in a background thread so mcp.run() below can start
        # serving HTTP first — the phone needs the server listening to load
        # /bridge and POST screen_dimension / touches. On failure, send
        # SIGINT to the main thread so mcp.run exits and atexit handlers
        # (shutdown, arm return-to-origin) still fire cleanly.
        from physiclaw.server import warm_start

        def _warm_start_thread():
            if not warm_start.try_resume(args.cam_index):
                log.error(
                    "Exiting. Re-run without --warm-start and then setup.py to "
                    "recalibrate: `uv run physiclaw` then "
                    "`uv run python scripts/setup.py`."
                )
                os.kill(os.getpid(), signal.SIGINT)

        threading.Thread(target=_warm_start_thread, daemon=True).start()
    else:
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

    try:
        mcp.run(transport="streamable-http")
    except KeyboardInterrupt:
        # Normal Ctrl-C OR our warm-start-fail SIGINT. atexit handlers
        # (shutdown → arm return-to-origin, camera close) still fire on
        # the way out; just swallow the traceback.
        pass


if __name__ == "__main__":
    main()
