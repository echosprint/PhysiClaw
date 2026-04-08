"""
PhysiClaw MCP Server.

The FastMCP instance lives in `physiclaw.server.mcp`; assembly (singletons,
tool/route registration) lives in `physiclaw.server.app`. This package
re-exports the public surface — `mcp`, `physiclaw`, `shutdown` — so callers
can keep doing `from physiclaw.server import mcp, shutdown`.

Started by physiclaw.main. The server starts instantly without hardware.
Run /setup to connect and calibrate.
"""

from physiclaw.server.app import physiclaw, shutdown
from physiclaw.server.mcp import mcp

__all__ = ["mcp", "physiclaw", "shutdown"]
