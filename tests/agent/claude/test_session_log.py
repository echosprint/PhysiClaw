"""Tests for `physiclaw.agent.claude.session_log` — the claude wake's
daily narrative + per-session artifact dir. Extracted 1:1 from
test_spawn.py alongside the source split."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from physiclaw.agent.claude import session_log
from physiclaw.agent.claude.session_log import _redact_images, _SessionLog
from physiclaw.agent.runtime.hook import Trigger

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
    monkeypatch.setattr(session_log, "LOG_DIR", log_dir)
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
        (
            paths.claude_sessions_dir() / "20260101_120000_test00" / "summary.json"
        ).read_text()
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


def test_session_log_image_turn_tag_dedups_streamed_message(
    _isolated_log_dir: Path,
) -> None:
    import base64

    from physiclaw.common import paths

    slog = _slog(["phone"])
    # One assistant MESSAGE streamed as two events sharing message.id —
    # the turn must advance ONCE, so the screenshot tags as t1, not t2.
    for block in (
        {"type": "text", "text": "looking"},
        {"type": "tool_use", "name": "peek", "input": {}},
    ):
        slog.event(
            {"type": "assistant", "message": {"id": "msg_1", "content": [block]}}
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
                            }
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
    assert imgs[0].name == "00001_t1.jpg"  # one message despite two events


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
    with caplog.at_level(logging.INFO, logger="physiclaw.agent.claude.session_log"):
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
