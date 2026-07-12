"""FastMCP instance construction.

Isolated from `physiclaw.core.server.__init__` so the agent-facing
instructions prompt has a single, focused home. The instance is imported
and wired up (tools, routes, singletons) by `physiclaw.core.server.__init__`.

The `instructions` field is delivered to the client at the MCP initialization
handshake — used by external clients (Claude Desktop, OpenClaw, etc.). The
in-tree agent loads the same file directly as a doctrine slot in
`physiclaw/agent/context/PHYSICLAW.md`, so this file is the single source of truth.
Keep it focused on cross-tool reasoning: mental model, tool-choice
trade-offs, operating loop, coordinate conventions, global safety, and
setup gating. Per-tool mechanics live in `@mcp.tool()` docstrings.
"""

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from physiclaw.common.text import read_text

_PKG_ROOT = Path(__file__).resolve().parents[2]
_INSTRUCTIONS = read_text(_PKG_ROOT / "agent" / "context" / "PHYSICLAW.md")

mcp = FastMCP("physiclaw", instructions=_INSTRUCTIONS)
