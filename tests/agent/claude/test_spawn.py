"""Tests for `physiclaw.agent.claude.spawn`.

`spawn_claude` (the full retry-loop subprocess driver) is integration-
leaning; covered by exercising helper functions and `_SessionLog` in
detail. Full integration deferred — async subprocess + streaming json
+ retry logic is brittle to mock cleanly.
"""

from __future__ import annotations

import sys
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from physiclaw.agent.claude import spawn
from physiclaw.agent.claude.spawn import (
    _ALLOWED_STATIC,
    _DISALLOWED,
    _SessionLog,
    _build_cmd,
    _build_trigger_prompt,
    _child_env,
    _ENV_STRIP_PREFIXES,
    _mcp_config,
    _mcp_tools,
    _normalize_claude_model_id,
    _redact_images,
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
        spawn.screen_layout, "load_layout_md", return_value="- backspace [0.1,0.2,0.3,0.4]"
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


# ---------- _redact_images ----------


def test_redact_images_passes_through_non_list() -> None:
    assert _redact_images("just text") == "just text"
    assert _redact_images(None) is None


def test_redact_images_replaces_image_data_with_placeholder() -> None:
    content = [
        {"type": "text", "text": "hi"},
        {"type": "image", "source": {"data": "AAAA", "media_type": "png"}},
    ]

    out = _redact_images(content)

    assert out[0] == {"type": "text", "text": "hi"}
    assert out[1]["source"]["data"] == "<4b elided>"
    assert out[1]["source"]["media_type"] == "png"


def test_redact_images_image_with_no_source() -> None:
    content = [{"type": "image"}]

    out = _redact_images(content)

    assert out[0]["source"] == {"data": "<0b elided>"}


# ---------- _SessionLog ----------


@pytest.fixture
def _isolated_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(spawn, "LOG_DIR", log_dir)
    return log_dir


def _slog(sources: list[str]) -> _SessionLog:
    """Build a `_SessionLog` from bare source strings. Fixed sid/model/prompt
    so the artifact dir is deterministic; the session dir lands under the
    autouse `physiclaw_home` tmp via `paths.claude_sessions_dir()`."""
    triggers = [Trigger(source=s, description=s) for s in sources]
    return _SessionLog(
        "20260101_120000_test00",
        triggers,
        model_ref="claude-code/test-model",
        prompt_hash="0" * 64,
    )


def test_session_log_init_writes_wake_header(_isolated_log_dir: Path) -> None:
    slog = _slog(["cron:a", "phone"])
    slog.close()

    files = list(_isolated_log_dir.glob("claude-*.log"))
    assert len(files) == 1
    text = files[0].read_text()
    assert "WAKE triggers=['cron:a', 'phone']" in text
    assert "=" * 60 in text


def test_session_log_writes_summary_json(_isolated_log_dir: Path) -> None:
    import json

    from physiclaw.common import paths

    slog = _slog(["phone"])
    # Two assistant events sharing ONE message id — the content-block streaming
    # of a single API response. Must count as ONE provider call, not two, and
    # their (partial, repeated) usage must NOT be summed.
    slog.event(
        {
            "type": "assistant",
            "message": {
                "id": "msg_1",
                "content": [{"type": "tool_use", "name": "peek", "input": {}}],
                "usage": {"input_tokens": 100, "output_tokens": 4},
            },
        }
    )
    slog.event(
        {
            "type": "assistant",
            "message": {
                "id": "msg_1",
                "content": [{"type": "text", "text": ">> DONE - bought it"}],
                "usage": {"input_tokens": 100, "output_tokens": 4},
            },
        }
    )
    # The result event is the authoritative cumulative token/cost/time record.
    slog.event(
        {
            "type": "result",
            "num_turns": 1,
            "total_cost_usd": 0.0123,
            "duration_api_ms": 4200,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 30,
                "cache_read_input_tokens": 400,
                "cache_creation_input_tokens": 0,
            },
        }
    )
    slog.done(0)
    slog.close()

    s = json.loads(
        (paths.claude_sessions_dir() / "20260101_120000_test00" / "summary.json")
        .read_text()
    )
    assert s["schema"] == 1
    assert s["provider"] == "claude-code"
    assert s["model_ref"] == "claude-code/test-model"
    assert s["outcome"] == {"sentinel": "DONE", "recap": "bought it", "crashed": False}
    assert s["turns"] == 1  # one distinct message, despite two stream events
    assert s["provider_calls"] == 1
    assert s["tool_calls"] == {"peek": 1}
    # Tokens from result.usage (authoritative), not summed per-assistant.
    assert s["usage"]["input_tokens"] == 100 + 400  # input + cache_read
    assert s["usage"]["output_tokens"] == 30
    assert s["usage"]["cache_read_tokens"] == 400
    assert s["cost_usd"] == 0.0123
    assert s["provider_time_ms"] == 4200


def test_session_log_extracts_screenshot(_isolated_log_dir: Path) -> None:
    import base64

    from physiclaw.common import paths

    slog = _slog(["phone"])
    slog.event(
        {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "name": "peek", "input": {}}]},
        }
    )
    b64 = base64.b64encode(b"\xff\xd8fake-jpeg").decode()
    slog.event(
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": "elements..."},
                        ],
                    }
                ]
            },
        }
    )
    slog.close()

    imgs = list(
        (paths.claude_sessions_dir() / "20260101_120000_test00" / "images").glob("*")
    )
    assert len(imgs) == 1
    assert imgs[0].name == "00001_t1.jpg"  # turn 1 (one assistant seen), first image
    assert imgs[0].read_bytes() == b"\xff\xd8fake-jpeg"


def test_session_log_event_assistant_text_returns_none(
    _isolated_log_dir: Path,
) -> None:
    slog = _slog([])
    out = slog.event(
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "hello"}]},
        }
    )
    slog.close()

    assert out is None


def test_session_log_event_result_returns_data(_isolated_log_dir: Path) -> None:
    slog = _slog([])
    data = {"type": "result", "num_turns": 5, "result": "all done"}
    out = slog.event(data)
    slog.close()

    assert out == data


def test_session_log_summarizes_tool_use(_isolated_log_dir: Path) -> None:
    slog = _slog([])
    slog.event(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Read", "input": {"path": "x"}},
                ]
            },
        }
    )
    slog.close()

    text = list(_isolated_log_dir.glob("claude-*.log"))[0].read_text()
    assert "tool_use: Read" in text


def test_session_log_summarizes_thinking(_isolated_log_dir: Path) -> None:
    slog = _slog([])
    slog.event(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "let me think"},
                ]
            },
        }
    )
    slog.close()

    text = list(_isolated_log_dir.glob("claude-*.log"))[0].read_text()
    assert "thinking: let me think" in text


def test_session_log_user_event_without_tool_result_no_summary(
    _isolated_log_dir: Path,
) -> None:
    slog = _slog([])
    slog.event(
        {
            "type": "user",
            "message": {"content": [{"type": "text", "text": "ack"}]},
        }
    )
    slog.close()

    # No crash; no tool_result line written.
    text = list(_isolated_log_dir.glob("claude-*.log"))[0].read_text()
    assert "tool_result:" not in text


def test_session_log_summarizes_user_tool_result(_isolated_log_dir: Path) -> None:
    slog = _slog([])
    slog.event(
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "content": "result value"},
                ]
            },
        }
    )
    slog.close()

    text = list(_isolated_log_dir.glob("claude-*.log"))[0].read_text()
    assert "tool_result:" in text


def test_session_log_summarizes_result(_isolated_log_dir: Path) -> None:
    slog = _slog([])
    slog.event({"type": "result", "num_turns": 3, "result": "ok"})
    slog.close()

    text = list(_isolated_log_dir.glob("claude-*.log"))[0].read_text()
    assert "result: turns=3 ok" in text


def test_session_log_unknown_event_type_no_summary(_isolated_log_dir: Path) -> None:
    slog = _slog([])
    slog.event({"type": "system_init"})  # not assistant/user/result
    slog.close()

    # No crash.


def test_session_log_assistant_no_text_no_summary(_isolated_log_dir: Path) -> None:
    slog = _slog([])
    # Empty text block — falsy strip.
    slog.event(
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "  "}]},
        }
    )
    slog.close()


def test_session_log_forward_to_runtime_only_logs_first_line(
    _isolated_log_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="physiclaw.agent.claude.spawn"):
        slog = _slog([])
        slog.event(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "first line\nsecond line"},
                    ]
                },
            }
        )
        slog.close()

    runtime_lines = [r for r in caplog.records if "claude:" in r.getMessage()]
    assert any("first line" in r.getMessage() for r in runtime_lines)
    # Second line must not appear in the live runtime log.
    assert not any("second line" in r.getMessage() for r in runtime_lines)


def test_session_log_raw_writes_truncated_text(_isolated_log_dir: Path) -> None:
    slog = _slog([])
    long = "z" * 1000
    slog.raw(long)
    slog.close()

    text = list(_isolated_log_dir.glob("claude-*.log"))[0].read_text()
    # Truncated at 500 chars.
    assert "raw: " in text
    assert "z" * 500 in text
    assert "z" * 501 not in text


def test_session_log_done_parses_sentinel_on_clean_exit(
    _isolated_log_dir: Path,
) -> None:
    slog = _slog([])
    slog.event(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "wrapping up\n>> DONE - all good"},
                ]
            },
        }
    )

    status = slog.done(0)
    slog.close()

    assert status == "DONE"
    text = list(_isolated_log_dir.glob("claude-*.log"))[0].read_text()
    assert "OUTCOME: DONE - all good" in text
    assert "EXIT code=0" in text


def test_session_log_done_undone_on_nonzero_exit(_isolated_log_dir: Path) -> None:
    slog = _slog([])
    slog.event(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": ">> DONE - claimed but crashed"},
                ]
            },
        }
    )

    status = slog.done(1)
    slog.close()

    assert status == "UNDONE"


def test_session_log_done_undone_when_no_text(_isolated_log_dir: Path) -> None:
    slog = _slog([])
    status = slog.done(0)
    slog.close()

    assert status == "UNDONE"
    text = list(_isolated_log_dir.glob("claude-*.log"))[0].read_text()
    assert "(no text)" in text


def test_session_log_done_truncates_recap_to_200_chars(_isolated_log_dir: Path) -> None:
    slog = _slog([])
    slog.event(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "x" * 500},
                ]
            },
        }
    )

    slog.done(0)
    slog.close()

    text = list(_isolated_log_dir.glob("claude-*.log"))[0].read_text()
    # OUTCOME line specifically — text summary line above contains 500 x's.
    outcome_line = next(line for line in text.splitlines() if "OUTCOME" in line)
    assert "OUTCOME: UNDONE" in outcome_line
    # Recap is exactly 200 x's, no more.
    assert "x" * 200 in outcome_line
    assert "x" * 201 not in outcome_line


def test_session_log_rollover_at_midnight(_isolated_log_dir: Path) -> None:
    from freezegun import freeze_time

    with freeze_time("2026-04-28 23:59:59") as frozen:
        slog = _slog([])
        # Cross midnight before the next write.
        frozen.move_to("2026-04-29 00:00:01")
        slog.event({"type": "result", "num_turns": 1, "result": "ok"})
        slog.close()

    files = sorted(_isolated_log_dir.glob("claude-*.log"))
    assert len(files) == 2
    today_text = (_isolated_log_dir / "claude-2026-04-29.log").read_text()
    assert "ROLLOVER" in today_text


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
