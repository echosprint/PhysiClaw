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
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed debug output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    from physiclaw.server import mcp, shutdown

    atexit.register(shutdown)

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    log = logging.getLogger(__name__)
    log.info(f"PhysiClaw MCP server on http://{args.host}:{args.port}/mcp")
    log.info(f"Annotation UI at http://{args.host}:{args.port}/annotate")
    log.info("Run /setup in Claude Code to connect hardware and calibrate")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
