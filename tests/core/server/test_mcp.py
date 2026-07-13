"""Tests for `physiclaw.core.server.mcp` — FastMCP instance.

The module reads `PHYSICLAW.md` from disk at import time and constructs
a `FastMCP("physiclaw", instructions=...)`. The instructions string is
sent to MCP clients at the initialization handshake — verifying it
loads correctly and isn't empty matters for cross-tool reasoning.

Note on imports: `physiclaw.core.server.mcp` (the module) gets shadowed
by `physiclaw.core.server.mcp` (the FastMCP instance) once
`physiclaw.core.server.__init__` runs. We use `importlib.import_module`
to get the module object reliably.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp_mod = importlib.import_module("physiclaw.core.server.mcp")


def test_mcp_is_fastmcp_instance() -> None:
    assert isinstance(mcp_mod.mcp, FastMCP)


def test_mcp_instance_name() -> None:
    """The name appears in the client-side tool catalog as the prefix."""
    assert mcp_mod.mcp.name == "physiclaw"


def test_instructions_loaded_from_disk() -> None:
    """`PHYSICLAW.md` lives in `physiclaw/agent/context/`; module reads
    it at import time. Verify the cached _INSTRUCTIONS matches the
    on-disk content so a typo in the path is caught."""
    expected = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "physiclaw"
        / "agent"
        / "context"
        / "PHYSICLAW.md"
    )
    assert mcp_mod._INSTRUCTIONS == expected.read_text(encoding="utf-8")


def test_instructions_non_empty() -> None:
    """A blank instructions string would silently degrade client UX —
    no cross-tool reasoning to anchor the agent on."""
    assert mcp_mod._INSTRUCTIONS.strip() != ""
    assert len(mcp_mod._INSTRUCTIONS) > 100  # sanity: real content


def test_pkg_root_resolves_to_physiclaw_package() -> None:
    """`_PKG_ROOT` must land on `src/physiclaw/`. Path-arithmetic bugs
    here silently load the wrong file at runtime."""
    assert mcp_mod._PKG_ROOT.name == "physiclaw"
    assert (mcp_mod._PKG_ROOT / "agent" / "context" / "PHYSICLAW.md").exists()
