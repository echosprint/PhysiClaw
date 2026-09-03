"""Tests for `physiclaw.core.server.mcp` — MCPServer instance.

The module reads `PHYSICLAW.md` from disk at import time and constructs
a `MCPServer("physiclaw", instructions=...)`. The instructions string is
sent to MCP clients at the initialization handshake — verifying it
loads correctly and isn't empty matters for cross-tool reasoning.

Note on imports: `physiclaw.core.server.mcp` (the module) gets shadowed
by `physiclaw.core.server.mcp` (the MCPServer instance) once
`physiclaw.core.server.__init__` runs. We use `importlib.import_module`
to get the module object reliably.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from mcp.server.mcpserver import MCPServer

mcp_mod = importlib.import_module("physiclaw.core.server.mcp")


def test_mcp_is_mcpserver_instance() -> None:
    assert isinstance(mcp_mod.mcp, MCPServer)


def test_mcp_instance_name() -> None:
    """The name appears in the client-side tool catalog as the prefix."""
    assert mcp_mod.mcp.name == "physiclaw"


def test_instructions_are_rendered_from_disk() -> None:
    """`PHYSICLAW.md` lives in `physiclaw/agent/context/`; module reads
    it at import time and renders doctrine `{{token}}`s through the same
    `common.doctrine.fill_tokens` pass the engine uses. Verify the
    cached _INSTRUCTIONS matches the rendered on-disk content so both a
    path typo and a skipped render are caught."""
    from physiclaw.common.doctrine import fill_tokens

    expected = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "physiclaw"
        / "agent"
        / "context"
        / "PHYSICLAW.md"
    )
    assert mcp_mod._INSTRUCTIONS == fill_tokens(
        "PHYSICLAW.md", expected.read_text(encoding="utf-8")
    )


def test_instructions_render_engine_thresholds() -> None:
    """The doctrine quotes the same-target hard-block threshold via a
    `{{token}}`; external clients must receive the rendered number."""
    from physiclaw.common.config import CONFIG

    assert f"press #{CONFIG.engine.same_target_block}" in mcp_mod._INSTRUCTIONS


def test_instructions_non_empty() -> None:
    """A blank instructions string would silently degrade client UX —
    no cross-tool reasoning to anchor the agent on."""
    assert mcp_mod._INSTRUCTIONS.strip() != ""
    assert len(mcp_mod._INSTRUCTIONS) > 100  # sanity: real content


def test_instructions_contain_no_unrendered_tokens() -> None:
    """The MCP path renders `{{token}}`s at import via
    `common.doctrine.fill_tokens`, which fails open on unknown tokens
    (warns, leaves the braces). This pin turns that soft failure hard:
    a typo'd or unregistered token in PHYSICLAW.md must fail CI, not
    ship literal braces to external clients."""
    assert "{{" not in mcp_mod._INSTRUCTIONS


def test_pkg_root_resolves_to_physiclaw_package() -> None:
    """`_PKG_ROOT` must land on `src/physiclaw/`. Path-arithmetic bugs
    here silently load the wrong file at runtime."""
    assert mcp_mod._PKG_ROOT.name == "physiclaw"
    assert (mcp_mod._PKG_ROOT / "agent" / "context" / "PHYSICLAW.md").exists()


def test_server_reports_the_package_version() -> None:
    """Clients show `serverInfo.version`; without an explicit value the
    SDK reports an empty string (1.x reported its own SDK version)."""
    from physiclaw import __version__

    assert mcp_mod.mcp.version == __version__
