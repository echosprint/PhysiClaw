"""Tests for `physiclaw.agent.claude.spawn`.

`spawn_claude` (the full retry-loop subprocess driver) is integration-
leaning; covered by exercising helper functions in detail (the session
log's own tests live in test_session_log.py). Full integration deferred
— async subprocess + streaming json + retry logic is brittle to mock
cleanly.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from physiclaw.agent.claude import session_log, spawn
from physiclaw.agent.claude.session_log import _SessionLog
from physiclaw.agent.claude.spawn import (
    _ALLOWED_STATIC,
    _DISALLOWED,
    _ENV_STRIP_PREFIXES,
    _build_cmd,
    _build_trigger_prompt,
    _child_env,
    _mcp_config,
    _mcp_tools,
    _normalize_claude_model_id,
    _render_system_prompt,
    _stream,
    _tooling_card,
    _warn_stray_context,
)
from physiclaw.agent.runtime.hook import Trigger

# ---------- _mcp_tools ----------


def test_mcp_tools_prefixes_names_and_takes_first_line(mocker) -> None:
    mocker.patch.object(
        spawn,
        "discover_mcp_tools",
        return_value=[
            {"name": "peek", "description": "Take a peek\n  multi-line"},
            {"name": "tap", "description": "Tap target"},
            {"name": "noop", "description": None},
        ],
    )

    out = _mcp_tools()

    assert out == [
        {"name": "mcp__physiclaw__peek", "description": "Take a peek"},
        {"name": "mcp__physiclaw__tap", "description": "Tap target"},
        {"name": "mcp__physiclaw__noop", "description": ""},
    ]


def test_mcp_tools_handles_empty_inventory(mocker) -> None:
    mocker.patch.object(spawn, "discover_mcp_tools", return_value=[])

    assert _mcp_tools() == []


# ---------- _mcp_config ----------


def test_mcp_config_uses_env_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHYSICLAW_SERVER", "http://example.com:9000")

    cfg = json.loads(_mcp_config())

    assert cfg == {
        "mcpServers": {
            "physiclaw": {
                "type": "http",
                "url": "http://example.com:9000/mcp",
            }
        }
    }


def test_mcp_config_default_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHYSICLAW_SERVER", raising=False)

    cfg = json.loads(_mcp_config())

    assert cfg["mcpServers"]["physiclaw"]["url"].startswith("http://127.0.0.1:8048")


# ---------- _tooling_card ----------


def test_tooling_card_empty_list_returns_empty_string() -> None:
    assert _tooling_card([]) == ""


def test_tooling_card_lists_tools_as_markdown() -> None:
    out = _tooling_card(
        [
            {"name": "mcp__physiclaw__peek", "description": "see screen"},
            {"name": "mcp__physiclaw__tap", "description": "tap target"},
        ]
    )

    assert "## Tooling" in out
    assert "mcp__physiclaw__" in out
    assert "- **mcp__physiclaw__peek** — see screen" in out
    assert "- **mcp__physiclaw__tap** — tap target" in out


# ---------- _render_system_prompt ----------


def test_render_system_prompt_combines_parts(mocker, tmp_path: Path) -> None:
    fake_md = tmp_path / "CLAUDE.md"
    fake_md.write_text("# Doctrine\nbody\n\n")
    mocker.patch.object(spawn, "CLAUDE_MD", fake_md)
    mocker.patch.object(
        spawn.skill, "render_section", return_value="## Available skills\nfoo"
    )
    tools = [{"name": "mcp__physiclaw__peek", "description": "see"}]

    out = _render_system_prompt(tools, {})

    assert out.startswith("# Doctrine\nbody")
    assert "## Tooling" in out
    assert "## Available skills" in out


def test_render_system_prompt_skips_empty_card_and_section(
    mocker,
    tmp_path: Path,
) -> None:
    fake_md = tmp_path / "CLAUDE.md"
    fake_md.write_text("body")
    mocker.patch.object(spawn, "CLAUDE_MD", fake_md)
    mocker.patch.object(spawn.skill, "render_section", return_value="")
    # Learned + empty layout md → no `## Screen layout` block and no first-run
    # notice, so only the card/section skipping is under test here.
    mocker.patch.object(spawn.screen_layout, "is_learned", return_value=True)
    mocker.patch.object(spawn.screen_layout, "load_layout_md", return_value="")

    out = _render_system_prompt([], {})

    assert out == "body"


def test_render_system_prompt_injects_screen_layout_when_learned(
    mocker,
    tmp_path: Path,
) -> None:
    fake_md = tmp_path / "CLAUDE.md"
    fake_md.write_text("body")
    mocker.patch.object(spawn, "CLAUDE_MD", fake_md)
    mocker.patch.object(spawn.skill, "render_section", return_value="")
    mocker.patch.object(spawn.screen_layout, "is_learned", return_value=True)
    mocker.patch.object(
        spawn.screen_layout,
        "load_layout_md",
        return_value="- backspace [0.1,0.2,0.3,0.4]",
    )

    out = _render_system_prompt([], {})

    assert "## Screen layout" in out
    assert "- backspace [0.1,0.2,0.3,0.4]" in out


def test_render_system_prompt_injects_first_run_notice_when_unlearned(
    mocker,
    tmp_path: Path,
) -> None:
    fake_md = tmp_path / "CLAUDE.md"
    fake_md.write_text("body")
    mocker.patch.object(spawn, "CLAUDE_MD", fake_md)
    mocker.patch.object(spawn.skill, "render_section", return_value="")
    mocker.patch.object(spawn.screen_layout, "is_learned", return_value=False)
    mocker.patch.object(
        spawn.screen_layout, "tail_reminder", return_value="[First-run setup needed]"
    )

    out = _render_system_prompt([], {})

    assert "[First-run setup needed]" in out


# ---------- _build_trigger_prompt ----------


def test_build_trigger_prompt_includes_source_tag_and_think() -> None:
    triggers = [
        Trigger(source="cron:job-a", description="job a fired"),
        Trigger(source="phone", description="screen changed"),
    ]

    out = _build_trigger_prompt(triggers)

    assert "[cron:job-a] job a fired" in out
    assert "[phone] screen changed" in out
    assert out.endswith("think")
    assert "Loop in CLAUDE.md" in out


def test_build_trigger_prompt_sourceless_trigger_has_no_tag() -> None:
    out = _build_trigger_prompt([Trigger(description="d", source="")])

    assert "- d" in out  # no `[]` tag
    assert "[" not in out.split("\n", 1)[0]  # first line is heading


# ---------- _child_env ----------


def test_child_env_strips_anthropic_claude_otel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/elsewhere")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel")
    monkeypatch.setenv("HOME", "/home/test")
    monkeypatch.setenv("PHYSICLAW_HOME", "/home/test/.physiclaw")

    env = _child_env()

    for k in ("ANTHROPIC_API_KEY", "CLAUDE_CONFIG_DIR", "OTEL_EXPORTER_OTLP_ENDPOINT"):
        assert k not in env
    assert env.get("HOME") == "/home/test"
    assert env.get("PHYSICLAW_HOME") == "/home/test/.physiclaw"


def test_child_env_pins_pwd(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _child_env()

    assert env["PWD"] == str(spawn.PROJECT_ROOT)


def test_child_env_sets_virtual_env_to_this_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The child's `uv run` (jobs / screen-layout CLIs) must resolve physiclaw;
    # it does so via VIRTUAL_ENV, overridden to this process's venv even if the
    # parent shell had a different one active.
    monkeypatch.setenv("VIRTUAL_ENV", "/some/other/venv")

    env = _child_env()

    assert env["VIRTUAL_ENV"] == sys.prefix


def test_env_strip_prefixes_constants() -> None:
    # Defensive guard: removing a prefix would silently relax sandbox.
    assert "ANTHROPIC_" in _ENV_STRIP_PREFIXES
    assert "CLAUDE_" in _ENV_STRIP_PREFIXES
    assert "OTEL_" in _ENV_STRIP_PREFIXES


# ---------- _warn_stray_context ----------


def test_warn_stray_context_logs_when_claude_md_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(spawn, "PROJECT_ROOT", tmp_path)
    (tmp_path / "CLAUDE.md").write_text("stray")

    with caplog.at_level(logging.WARNING, logger="physiclaw.agent.claude.spawn"):
        _warn_stray_context()

    assert any(
        "CLAUDE.md" in r.getMessage() and "stray" in r.getMessage()
        for r in caplog.records
    )


def test_warn_stray_context_logs_when_dot_claude_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(spawn, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".claude").mkdir()

    with caplog.at_level(logging.WARNING, logger="physiclaw.agent.claude.spawn"):
        _warn_stray_context()

    assert any(".claude" in r.getMessage() for r in caplog.records)


def test_warn_stray_context_silent_when_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(spawn, "PROJECT_ROOT", tmp_path)

    with caplog.at_level(logging.WARNING, logger="physiclaw.agent.claude.spawn"):
        _warn_stray_context()

    assert not [r for r in caplog.records if "stray" in r.getMessage()]


# ---------- _normalize_claude_model_id ----------


@pytest.mark.parametrize("alias", ["opus", "sonnet", "haiku"])
def test_normalize_claude_model_aliases_pass_through(alias: str) -> None:
    assert _normalize_claude_model_id(alias) == alias


def test_normalize_claude_model_already_prefixed_passes_through() -> None:
    assert _normalize_claude_model_id("claude-opus-4-7") == "claude-opus-4-7"


def test_normalize_claude_model_adds_prefix_to_bare_id() -> None:
    assert _normalize_claude_model_id("opus-4-7") == "claude-opus-4-7"
    assert (
        _normalize_claude_model_id("haiku-4-5-20251001") == "claude-haiku-4-5-20251001"
    )


# ---------- _build_cmd ----------


def test_build_cmd_includes_required_flags(mocker, tmp_path: Path) -> None:
    fake_md = tmp_path / "CLAUDE.md"
    fake_md.write_text("doctrine")
    mocker.patch.object(spawn, "CLAUDE_MD", fake_md)
    mocker.patch.object(spawn, "_mcp_config", return_value="{}")

    triggers = [Trigger(description="d")]
    cmd = _build_cmd(
        triggers,
        plugin_dir=tmp_path,
        system_prompt="prompt",
        mcp_tools=[{"name": "mcp__physiclaw__peek", "description": "x"}],
        model_id="opus-4-7",
    )

    assert cmd[0] == "claude"
    assert "--model" in cmd
    assert "claude-opus-4-7" in cmd
    assert "--append-system-prompt" in cmd
    assert "prompt" in cmd
    assert "--plugin-dir" in cmd
    assert str(tmp_path) in cmd
    assert "--strict-mcp-config" in cmd
    assert "--no-session-persistence" in cmd
    # Allowed = MCP names + static.
    allowed_idx = cmd.index("--allowedTools") + 1
    allowed = cmd[allowed_idx].split(",")
    assert "mcp__physiclaw__peek" in allowed
    for s in _ALLOWED_STATIC:
        assert s in allowed
    disallowed_idx = cmd.index("--disallowedTools") + 1
    for d in _DISALLOWED:
        assert d in cmd[disallowed_idx].split(",")


def test_build_cmd_raises_when_claude_md_missing(
    mocker,
    tmp_path: Path,
) -> None:
    mocker.patch.object(spawn, "CLAUDE_MD", tmp_path / "missing.md")

    with pytest.raises(FileNotFoundError, match="CLAUDE.md not found"):
        _build_cmd(
            [Trigger(description="d")],
            plugin_dir=tmp_path,
            system_prompt="p",
            mcp_tools=[],
            model_id="opus",
        )


# ---------- _stream ----------
#
# `_stream` drives a real `_SessionLog` (now in session_log.py), so these
# tests keep a local copy of its isolated-log-dir fixture and builder.


@pytest.fixture
def _isolated_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(session_log, "LOG_DIR", log_dir)
    return log_dir


def _slog(sources: list[str]) -> _SessionLog:
    """Build a `_SessionLog` from bare source strings — see
    test_session_log.py for the canonical copy."""
    triggers = [Trigger(source=s, description=s) for s in sources]
    return _SessionLog(
        "20260101_120000_test00",
        triggers,
        model_ref="claude-code/test-model",
        prompt_hash="0" * 64,
    )


class _FakeStdout:
    """Async stdout that yields fixed lines then EOF."""

    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)

    async def readline(self) -> bytes:
        if not self._lines:
            return b""
        return self._lines.pop(0)


@pytest.mark.asyncio
async def test_stream_collects_result_event(_isolated_log_dir: Path) -> None:
    proc = SimpleNamespace(
        stdout=_FakeStdout(
            [
                json.dumps({"type": "assistant", "message": {"content": []}}).encode()
                + b"\n",
                json.dumps({"type": "result", "num_turns": 1, "result": "ok"}).encode()
                + b"\n",
                b"",  # EOF
            ]
        )
    )
    slog = _slog([])

    out = await _stream(proc, slog)
    slog.close()

    assert out == {"type": "result", "num_turns": 1, "result": "ok"}


@pytest.mark.asyncio
async def test_stream_skips_blank_lines(_isolated_log_dir: Path) -> None:
    proc = SimpleNamespace(
        stdout=_FakeStdout(
            [
                b"\n",
                b"   \n",
                json.dumps({"type": "result", "result": "x"}).encode() + b"\n",
                b"",
            ]
        )
    )
    slog = _slog([])

    out = await _stream(proc, slog)
    slog.close()

    assert out == {"type": "result", "result": "x"}


@pytest.mark.asyncio
async def test_stream_logs_raw_on_json_decode_error(
    _isolated_log_dir: Path,
) -> None:
    proc = SimpleNamespace(
        stdout=_FakeStdout(
            [
                b"not valid json\n",
                b"",
            ]
        )
    )
    slog = _slog([])

    out = await _stream(proc, slog)
    slog.close()

    assert out is None
    text = list(_isolated_log_dir.glob("claude-*.log"))[0].read_text()
    assert "raw: not valid json" in text


@pytest.mark.asyncio
async def test_stream_returns_none_when_no_result_event(
    _isolated_log_dir: Path,
) -> None:
    proc = SimpleNamespace(
        stdout=_FakeStdout(
            [
                json.dumps({"type": "assistant", "message": {"content": []}}).encode()
                + b"\n",
                b"",
            ]
        )
    )
    slog = _slog([])

    out = await _stream(proc, slog)
    slog.close()

    assert out is None
