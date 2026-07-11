"""Static MCP tool inventory — AST-parses `src/physiclaw/core/server/tools.py` to
enumerate tool names + first-line descriptions without importing the
module or requiring the MCP server to be running.

Used by `prompt._render_tooling` so the inline `## Tooling` card is
complete even when the dump is generated offline (no MCP) and as a
guaranteed-present baseline at runtime (live MCP tool_schemas can
augment with richer descriptions but won't drop tools).
"""

import ast
import logging
from pathlib import Path

from physiclaw.text import read_text

log = logging.getLogger(__name__)

_TOOLS_PY = Path(__file__).resolve().parents[2] / "core" / "server" / "tools.py"


def discover_mcp_tools() -> list[dict]:
    """Return `[{"name", "description"}]` for every `@mcp.tool()`-decorated
    function inside `src/physiclaw/core/server/tools.py`. No import, no MCP
    connection. Returns `[]` if the file is missing or unparseable —
    rendering still succeeds, just with engine-local tools only."""
    if not _TOOLS_PY.exists():
        log.debug("mcp_inventory: %s missing; skipping static MCP tools", _TOOLS_PY)
        return []
    try:
        tree = ast.parse(read_text(_TOOLS_PY))
    except SyntaxError:
        log.exception("mcp_inventory: failed to parse %s", _TOOLS_PY)
        return []

    out: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        if not _has_mcp_tool_decorator(node):
            continue
        out.append(
            {
                "name": node.name,
                "description": ast.get_docstring(node) or "",
            }
        )
    return out


def _has_mcp_tool_decorator(node: ast.AsyncFunctionDef | ast.FunctionDef) -> bool:
    """True if any decorator is `@mcp.tool()` or `@mcp.tool`."""
    for d in node.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "tool"
            and isinstance(target.value, ast.Name)
            and target.value.id == "mcp"
        ):
            return True
    return False
