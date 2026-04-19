"""FastMCP instance construction.

Isolated from `physiclaw.server.__init__` so the agent-facing instructions
prompt has a single, focused home. The instance is imported and wired up
(tools, routes, singletons) by `physiclaw.server.__init__`.

The `instructions` field is delivered to the client once at the MCP
initialization handshake. It's authored in ``instructions.md`` next to
this file — edit there for syntax highlighting/preview. Keep it focused
on cross-tool reasoning: mental model, tool-choice trade-offs, operating
loop, coordinate conventions, global safety, and setup gating. Per-tool
mechanics live in `@mcp.tool()` docstrings and are auto-delivered as
tool schemas — do not duplicate them here.
"""

from pathlib import Path

from mcp.server.fastmcp import FastMCP

_INSTRUCTIONS = (Path(__file__).parent / "instructions.md").read_text(encoding="utf-8")

mcp = FastMCP("physiclaw", instructions=_INSTRUCTIONS)
