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

Doctrine `{{token}}`s render here too (`common.doctrine.fill_tokens`,
the same pass the engine's prompt composer runs) — before this, a
token in PHYSICLAW.md reached external clients as literal braces.
Config is process-constant, so rendering once at import is stable.
"""

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from physiclaw.common.doctrine import fill_tokens
from physiclaw.common.text import read_text

_PKG_ROOT = Path(__file__).resolve().parents[2]
_INSTRUCTIONS = fill_tokens(
    "PHYSICLAW.md", read_text(_PKG_ROOT / "agent" / "context" / "PHYSICLAW.md")
)


def build_mcp() -> FastMCP:
    """A fresh, unwired FastMCP instance — `app.build_server()` uses this
    so each assembly gets its own instance (tools/routes registered twice
    on one instance would collide)."""
    return FastMCP("physiclaw", instructions=_INSTRUCTIONS)


mcp = build_mcp()
