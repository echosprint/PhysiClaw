"""
PhysiClaw MCP Server.

The MCPServer instance lives in `physiclaw.core.server.mcp`; assembly (singletons,
tool/route registration) lives in `physiclaw.core.server.app`. This package
re-exports the public surface — `mcp`, `physiclaw`, `shutdown` — so callers
can keep doing `from physiclaw.core.server import mcp, shutdown`.

Started by physiclaw.main. The server starts instantly without hardware.
Run /setup to connect and calibrate.
"""

from physiclaw.core.server.app import build_apps, physiclaw, shutdown
from physiclaw.core.server.mcp import mcp

__all__ = ["build_apps", "mcp", "physiclaw", "shutdown"]
