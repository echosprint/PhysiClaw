"""Tests for `physiclaw.common.doctrine` — the shared `{{token}}` renderer.

Moved out of `agent.engine.prompt` so the MCP instructions path
(`core.server.mcp`) can render the same tokens — core can't import
agent. Unit tests live here; each delivery path keeps its own
integration guard (`test_prompt.py` for the engine render,
`test_mcp.py` for the shipped instructions).
"""

from __future__ import annotations

import logging

import pytest

from physiclaw.common.config import CONFIG
from physiclaw.common.doctrine import DOCTRINE_TOKENS, fill_tokens

# ---------- token map ----------


def test_token_map_mirrors_engine_config() -> None:
    """Every token renders the CONFIG value the engine actually
    enforces — the whole point of the mechanism."""
    assert DOCTRINE_TOKENS == {
        "{{same_target_warn}}": str(CONFIG.engine.same_target_warn),
        "{{same_target_block}}": str(CONFIG.engine.same_target_block),
        "{{step_stuck_warn}}": str(CONFIG.engine.step_stuck_warn),
        "{{step_stuck_urgent}}": str(CONFIG.engine.step_stuck_urgent),
        "{{plan_required_after}}": str(CONFIG.engine.plan_required_after),
    }


# ---------- fill_tokens ----------


def test_fill_tokens_substitutes_all_known_tokens() -> None:
    body = "warn #{{same_target_warn}}, block #{{same_target_block}}"

    out = fill_tokens("TEST.md", body)

    assert out == (
        f"warn #{CONFIG.engine.same_target_warn}, "
        f"block #{CONFIG.engine.same_target_block}"
    )
    assert "{{" not in out


def test_fill_tokens_passthrough_without_tokens() -> None:
    assert fill_tokens("TEST.md", "plain prose") == "plain prose"


def test_fill_tokens_warns_on_unknown_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="physiclaw.common.doctrine"):
        out = fill_tokens("TEST.md", "cap: {{no_such_token}}")

    # Leftover survives (fail open — better literal braces than a crash
    # on the serving path) but the warning names file and token.
    assert "{{no_such_token}}" in out
    assert "TEST.md" in caplog.text
    assert "no_such_token" in caplog.text


def test_fill_tokens_warns_on_bare_open_brace(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="physiclaw.common.doctrine"):
        fill_tokens("TEST.md", "broken {{ token")

    assert "unresolved placeholder" in caplog.text
