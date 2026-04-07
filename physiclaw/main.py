"""PhysiClaw CLI entry point.

Usage:
    uv run physiclaw [--port 8048] [--host 127.0.0.1] [--verbose]
"""

import argparse
import atexit
import logging


def main():
    parser = argparse.ArgumentParser(description="PhysiClaw MCP Server")
    parser.add_argument("--port", type=int, default=8048)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed debug output"
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
    log.info(f"Annotation UI at http://localhost:{args.port}/annotate")
    log.info(f"QR code (scan with phone): http://localhost:{args.port}/api/bridge/qr")
    log.info(f"Phone page: http://{lan_ip}:{args.port}/bridge")
    log.info("Run /setup in Claude Code to connect hardware and calibrate")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
