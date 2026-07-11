"""Tests for `physiclaw.agent.engine.prompt` — system prompt assembly.

Module-level `CONTEXT_DIR` points at `agent/context/` in the repo;
tests redirect it to a per-test tmp dir so doctrine slot rendering
is deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from physiclaw.agent.engine import memory, prompt


@pytest.fixture(autouse=True)
def _isolate_context_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ctx = tmp_path / "context"
    ctx.mkdir()
    monkeypatch.setattr(prompt, "CONTEXT_DIR", ctx)
    return ctx


@pytest.fixture(autouse=True)
def _stub_memory_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: empty USER.md unless a test overrides."""
    monkeypatch.setattr(memory, "load_user", lambda: "")


@pytest.fixture(autouse=True)
def _stub_mcp_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: no MCP tools unless a test overrides."""
    from physiclaw.agent.engine import mcp_inventory

    monkeypatch.setattr(mcp_inventory, "discover_mcp_tools", list)


@pytest.fixture(autouse=True)
def _stub_screen_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: setup incomplete → screen-layout section absent, so unrelated
    prompt tests stay hermetic. Tests exercising the section override this."""
    from physiclaw.agent.engine import screen_layout

    monkeypatch.setattr(screen_layout, "is_learned", lambda: False)
    monkeypatch.setattr(screen_layout, "load_layout_md", lambda: "")


# ---------- DOCTRINE_FILE_ORDER ----------


def test_doctrine_file_order_pinned() -> None:
    assert prompt.DOCTRINE_FILE_ORDER == (
        "IDENTITY.md",
        "USER.md",
        "SOUL.md",
        "AGENT.md",
        "PHYSICLAW.md",
        "TOOLS.md",
        "PERSISTENCE.md",
        "JOBS.md",
        "CONVENTION.md",
    )


# ---------- _render_doctrine ----------


def test_render_doctrine_empty_when_no_files_exist(_isolate_context_dir) -> None:
    assert prompt._render_doctrine() == []


def test_render_doctrine_emits_section_per_existing_file(
    _isolate_context_dir: Path,
) -> None:
    (_isolate_context_dir / "IDENTITY.md").write_text("I am PhysiClaw.\n")
    (_isolate_context_dir / "AGENT.md").write_text("Be helpful.\n\n")

    out = prompt._render_doctrine()

    text = "\n".join(out)
    assert text.startswith("# Doctrine")
    assert "## IDENTITY.md" in text
    assert "I am PhysiClaw." in text
    assert "## AGENT.md" in text
    assert "Be helpful." in text


def test_render_doctrine_emits_files_in_pinned_order(
    _isolate_context_dir: Path,
) -> None:
    (_isolate_context_dir / "AGENT.md").write_text("agent body")
    (_isolate_context_dir / "IDENTITY.md").write_text("identity body")

    out = "\n".join(prompt._render_doctrine())

    # IDENTITY appears before AGENT in DOCTRINE_FILE_ORDER.
    assert out.index("## IDENTITY.md") < out.index("## AGENT.md")


def test_render_doctrine_user_md_loads_via_memory(
    _isolate_context_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory, "load_user", lambda: "User profile here")

    out = "\n".join(prompt._render_doctrine())

    assert "## USER.md" in out
    assert "User profile here" in out


def test_render_doctrine_substitutes_config_tokens(
    _isolate_context_dir: Path,
) -> None:
    # Doctrine quotes engine-enforced thresholds via {{token}} placeholders
    # so the prompt can never drift from the config the engine enforces.
    from physiclaw.config import CONFIG

    (_isolate_context_dir / "CONVENTION.md").write_text(
        "Blocked after {{plan_required_after}} turns; "
        "warn at press #{{same_target_warn}}."
    )

    out = "\n".join(prompt._render_doctrine())

    assert f"after {CONFIG.engine.plan_required_after} turns" in out
    assert f"press #{CONFIG.engine.same_target_warn}" in out
    assert "{{" not in out


def test_shipped_doctrine_has_no_unresolved_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Integration guard against the REAL context/ files (undoing the
    # autouse isolation): a typo'd {{token}} in shipped doctrine must
    # fail CI, not just log at runtime. Also pins that the substituted
    # thresholds actually appear where doctrine quotes them.
    from physiclaw.config import CONFIG

    real_ctx = Path(prompt.__file__).resolve().parent.parent / "context"
    monkeypatch.setattr(prompt, "CONTEXT_DIR", real_ctx)

    out = "\n".join(prompt._render_doctrine())

    assert "{{" not in out
    assert f"press #{CONFIG.engine.same_target_block}" in out
    assert f"after {CONFIG.engine.plan_required_after} turns" in out


def test_render_doctrine_warns_on_unresolved_token(
    _isolate_context_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    (_isolate_context_dir / "AGENT.md").write_text("Cap: {{no_such_token}}.")

    with caplog.at_level(logging.WARNING, logger="physiclaw.agent.engine.prompt"):
        prompt._render_doctrine()

    assert any("unresolved" in r.getMessage() for r in caplog.records)


def test_render_doctrine_skips_empty_files(_isolate_context_dir: Path) -> None:
    (_isolate_context_dir / "IDENTITY.md").write_text("")
    (_isolate_context_dir / "AGENT.md").write_text("body")

    out = "\n".join(prompt._render_doctrine())

    assert "## IDENTITY.md" not in out
    assert "## AGENT.md" in out


# ---------- _render_tooling ----------


def test_render_tooling_empty_when_no_tools(_isolate_context_dir) -> None:
    assert prompt._render_tooling([]) == []


def test_render_tooling_lists_local_tool_with_first_sentence(
    _isolate_context_dir,
) -> None:
    schemas = [{"name": "tap", "description": "Tap a coordinate. Then peek."}]

    out = prompt._render_tooling(schemas)

    text = "\n".join(out)
    assert "## Tooling" in text
    assert "- **tap** — Tap a coordinate." in text


def test_render_tooling_skips_tools_without_a_name(_isolate_context_dir) -> None:
    out = prompt._render_tooling([{"description": "anonymous"}])

    text = "\n".join(out)
    # Header still emitted because list is non-empty, but the tool has no name.
    assert "anonymous" not in text


def test_render_tooling_handles_tool_without_description() -> None:
    out = prompt._render_tooling([{"name": "noop"}])

    text = "\n".join(out)
    assert "- **noop**" in text
    assert "—" not in text.split("- **noop**")[1].split("\n")[0]


def test_render_tooling_includes_mcp_inventory_tools(
    _isolate_context_dir, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physiclaw.agent.engine import mcp_inventory

    monkeypatch.setattr(
        mcp_inventory,
        "discover_mcp_tools",
        lambda: [{"name": "peek", "description": "Annotated camera frame."}],
    )

    out = "\n".join(prompt._render_tooling([{"name": "tap", "description": "Tap."}]))

    assert "- **peek** — Annotated camera frame." in out
    assert "- **tap** — Tap." in out


# ---------- _render_section ----------


def test_render_section_empty_when_no_context() -> None:
    assert prompt._render_section("") == []


def test_render_section_passes_prerendered_block_through() -> None:
    # The skill module already emits the `##` header + framing; the prompt
    # just drops it in with a trailing blank line.
    assert prompt._render_section("## Skills\n\nbody") == ["## Skills\n\nbody", ""]


# ---------- _render_examples ----------


# ---------- _render_screen_layout ----------
# The first-run nudge is a tail message (screen_layout.inject_tail), not this
# section — here we only cover loading the layout once all pages are captured.


def test_render_screen_layout_loads_md_when_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physiclaw.agent.engine import screen_layout

    monkeypatch.setattr(screen_layout, "is_learned", lambda: True)
    monkeypatch.setattr(screen_layout, "load_layout_md", lambda: "LAYOUT TABLE")
    out = "\n".join(prompt._render_screen_layout())

    assert "## Screen layout" in out
    assert "LAYOUT TABLE" in out


def test_render_screen_layout_empty_while_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physiclaw.agent.engine import screen_layout

    # Even with a partial layout on disk, nothing shows until complete.
    monkeypatch.setattr(screen_layout, "is_learned", lambda: False)
    monkeypatch.setattr(screen_layout, "load_layout_md", lambda: "PARTIAL")
    assert prompt._render_screen_layout() == []


def test_render_screen_layout_says_not_set_when_learned_but_md_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physiclaw.agent.engine import screen_layout

    # Edge case: complete per json but the md file is gone — still show the
    # section, with a placeholder rather than hiding it.
    monkeypatch.setattr(screen_layout, "is_learned", lambda: True)
    monkeypatch.setattr(screen_layout, "load_layout_md", lambda: "")
    out = "\n".join(prompt._render_screen_layout())

    assert "## Screen layout" in out
    assert "not set yet" in out


def test_render_examples_returns_non_empty_block() -> None:
    out = prompt._render_examples()

    text = "\n".join(out)
    assert text.startswith("## Examples")
    assert "❌" in text
    assert "✅" in text


# ---------- _render_reasoning_format ----------


def test_render_reasoning_format_empty_when_provider_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physiclaw.agent import provider as provider_pkg

    monkeypatch.setattr(provider_pkg, "provider_class", lambda pid: None)

    assert prompt._render_reasoning_format("mystery") == []


def test_render_reasoning_format_empty_when_fragment_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeProvider:
        @classmethod
        def system_prompt_fragment(cls) -> str:
            return ""

    from physiclaw.agent.provider import registry

    monkeypatch.setattr(registry, "provider_class", lambda pid: _FakeProvider)

    assert prompt._render_reasoning_format("any") == []


def test_render_reasoning_format_via_real_anthropic_provider() -> None:
    # The real AnthropicProvider has system_prompt_fragment="" (default)
    # — so the section is NOT emitted.
    out = prompt._render_reasoning_format("anthropic")

    assert out == []


# ---------- _render_memory ----------


def test_render_memory_empty_when_no_context() -> None:
    assert prompt._render_memory("") == []


def test_render_memory_wraps_context_with_header() -> None:
    out = "\n".join(prompt._render_memory("user prefers metric"))

    assert out.startswith("## memory.md")
    assert "user prefers metric" in out


# ---------- _first_sentence ----------


def test_first_sentence_takes_first_line() -> None:
    assert prompt._first_sentence("first\nsecond") == "first"


def test_first_sentence_cuts_at_period_space() -> None:
    assert prompt._first_sentence("Tap a coord. Then peek.") == "Tap a coord."


def test_first_sentence_cuts_at_semicolon_space() -> None:
    assert prompt._first_sentence("Foo; bar") == "Foo;"


def test_first_sentence_returns_empty_for_blank_input() -> None:
    assert prompt._first_sentence("") == ""
    assert prompt._first_sentence("   ") == ""


def test_first_sentence_truncates_at_200_chars_when_no_separator() -> None:
    long = "x" * 250
    assert len(prompt._first_sentence(long)) == 200


# ---------- prefix_hash ----------


def test_prefix_hash_returns_64_char_hex_string() -> None:
    h = prompt.prefix_hash("hello")

    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_prefix_hash_deterministic_for_same_input() -> None:
    assert prompt.prefix_hash("hello") == prompt.prefix_hash("hello")


def test_prefix_hash_different_for_different_inputs() -> None:
    assert prompt.prefix_hash("a") != prompt.prefix_hash("b")


def test_prefix_hash_raises_on_non_string_input() -> None:
    with pytest.raises(
        ValueError, match=r"^prefix_hash: system_prompt must be str, got int$"
    ):
        prompt.prefix_hash(42)  # type: ignore[arg-type]


# ---------- render_system_prompts / dump (integration) ----------


def test_render_system_assembles_all_present_sections(
    _isolate_context_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (_isolate_context_dir / "IDENTITY.md").write_text("ID body")
    monkeypatch.setattr(memory, "load_user", lambda: "user profile")

    out = prompt.render_system_prompts(
        local_tool_schemas=[{"name": "tap", "description": "Tap."}],
        memory_ctx="recent fact",
        builtin_skills_ctx="## Built-in Skills\n\n### im\n\nim body",
        user_skills_ctx="## Available skills\n\n- **taobao** — shop",
        provider_id="",
    )

    assert "# Doctrine" in out
    assert "## IDENTITY.md" in out
    assert "## USER.md" in out
    assert "user profile" in out
    assert "## Tooling" in out
    assert "## Built-in Skills" in out
    assert "im body" in out  # built-in body inlined
    assert "## Available skills" in out
    assert "**taobao** — shop" in out  # user index entry
    assert "## Examples" in out
    assert "## memory.md" in out
    assert "recent fact" in out


def test_render_system_returns_empty_when_all_sections_disabled(
    _isolate_context_dir,
) -> None:
    out = prompt.render_system_prompts()

    # Examples is the only section that always emits.
    assert "## Examples" in out
    # No doctrine, no tooling, no skills, no memory.
    assert "# Doctrine" not in out
    assert "## Tooling" not in out
    assert "## Built-in Skills" not in out
    assert "## Available skills" not in out
    assert "## memory.md" not in out


def test_dump_delegates_to_render_system_with_same_args(
    _isolate_context_dir: Path,
) -> None:
    a = prompt.dump(memory_ctx="x")
    b = prompt.render_system_prompts(memory_ctx="x")

    assert a == b
