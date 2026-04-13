"""PhysiClaw CLI entry point.

Usage:
    uv run physiclaw [--port 8048] [--host 127.0.0.1] [--verbose]
"""

import argparse
import atexit
import logging
import subprocess
import sys


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

    from physiclaw.bridge import get_lan_ip

    log = logging.getLogger(__name__)
    lan_ip = get_lan_ip()
    log.info(f"PhysiClaw MCP server on http://{args.host}:{args.port}/mcp")
    log.info(f"QR code (scan with phone): http://localhost:{args.port}/api/bridge/qr")
    log.info(f"Phone page: http://{lan_ip}:{args.port}/bridge")
    log.info("Run /setup in Claude Code to connect hardware and calibrate")

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
