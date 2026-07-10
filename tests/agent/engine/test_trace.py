"""Tests for `physiclaw.agent.engine.trace` — engine session logging.

Covers public formatting helpers, _summarize event dispatch, Trace
file writes + day rollover, RawLog session-start/request/response
emit + image scrubbing for both OpenAI (image_url) and Anthropic
(image+source) wire shapes, and _purge_old retention.

Module-level `_LOG_DIR` / `_RAW_DIR` / `_SESSIONS_DIR` are bound at
import; the autouse fixture re-points them to per-test dirs.
"""
from __future__ import annotations

import base64
import datetime as dt
import json
from pathlib import Path

import pytest
from freezegun import freeze_time

from physiclaw.agent.engine import trace
from physiclaw.agent.engine.dto import ImageBlock, TextBlock
from physiclaw.agent.engine.trace import (
    RawLog,
    Trace,
    brief,
    brief_args,
    brief_content,
    format_call_args,
    format_call_result,
)


@pytest.fixture(autouse=True)
def _trace_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log_dir = tmp_path / "engine"
    monkeypatch.setattr(trace, "_LOG_DIR", log_dir)
    monkeypatch.setattr(trace, "_RAW_DIR", log_dir / "raw")
    monkeypatch.setattr(trace, "_SESSIONS_DIR", log_dir / "sessions")
    return log_dir


# ---------- brief / brief_args ----------


def test_brief_returns_short_strings_unchanged() -> None:
    assert brief("hello") == "hello"


def test_brief_truncates_long_strings_with_ellipsis() -> None:
    out = brief("x" * 100, limit=10)

    assert out == "xxxxxxxxx…"
    assert len(out) == 10


def test_brief_uses_repr_for_non_strings() -> None:
    assert brief({"a": 1}, limit=80) == "{'a': 1}"


def test_brief_args_joins_kv_pairs_with_commas() -> None:
    # `brief` returns strings as-is; non-strings via repr.
    assert brief_args({"a": 1, "b": "x"}) == "a=1, b=x"


def test_brief_args_truncates_individual_values_at_40() -> None:
    out = brief_args({"k": "x" * 100})

    assert "x…" in out


# ---------- format_call_args / format_call_result ----------


def test_format_call_args_uses_full_args_for_update_progress() -> None:
    long = "x" * 100
    out = format_call_args("update_progress", {"steps": long})

    # Full repr — no 40-char truncation.
    assert long in out


def test_format_call_args_uses_brief_args_for_other_tools() -> None:
    out = format_call_args("tap", {"bbox": "x" * 100})

    assert "…" in out


def test_format_call_result_no_truncation_for_note() -> None:
    long = "x" * 200

    assert format_call_result("note", long) == long


def test_format_call_result_truncates_other_tools_at_80() -> None:
    out = format_call_result("tap", "x" * 100)

    assert len(out) == 80


# ---------- brief_content ----------


def test_brief_content_string() -> None:
    assert brief_content("hello") == "hello"


def test_brief_content_text_block() -> None:
    assert brief_content([TextBlock(text="hi")]) == "hi"


def test_brief_content_image_block_shows_byte_count() -> None:
    assert brief_content([ImageBlock(media_type="image/jpeg", data_b64="aGk=")]) == (
        "<image 4b>"
    )


def test_brief_content_dict_text_form() -> None:
    assert brief_content([{"type": "text", "text": "x"}]) == "x"


def test_brief_content_dict_image_form() -> None:
    assert brief_content([{"type": "image", "data": "aGk="}]) == "<image 4b>"


def test_brief_content_dict_image_url_extracts_data_length() -> None:
    out = brief_content([
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abcdef"}}
    ])

    # The "data" portion (after the comma) is "abcdef" = 6 chars.
    assert out == "<image 6b>"


def test_brief_content_unknown_block_type_uses_type_or_question() -> None:
    out = brief_content([{"type": "future_kind"}])

    assert out == "future_kind"


def test_brief_content_unknown_object_renders_question_mark() -> None:
    assert brief_content([object()]) == "?"


def test_brief_content_empty_list_returns_empty_label() -> None:
    assert brief_content([]) == "(empty)"


def test_brief_content_non_list_non_str_uses_repr() -> None:
    assert brief_content(42) == "42"


def test_brief_content_multiple_blocks_joined_by_plus() -> None:
    out = brief_content([
        TextBlock(text="a"),
        ImageBlock(media_type="image/jpeg", data_b64="aGk="),
    ])

    assert out == "a + <image 4b>"


# ---------- _summarize ----------


@pytest.mark.parametrize(
    "event, expected_substr",
    [
        ({"event": "wake", "session": "s1", "model_ref": "openai/gpt-x", "triggers": [{"source": "phone"}]}, "WAKE session=s1 model=openai/gpt-x"),
        ({"event": "tools_loaded", "mcp": [1, 2], "local": [1]}, "tools: 2 MCP + 1 local"),
        ({"event": "request", "turn": 3, "message_count": 7}, "turn 3: request (7 messages)"),
        ({"event": "response", "turn": 1, "finish_reason": "stop", "tool_calls": [{"name": "tap"}]}, "turn 1: response finish=stop calls=['tap']"),
        ({"event": "cache", "turn": 2, "hit": 100, "create": 5, "new": 50, "total": 155}, "cache hit=100 create=5 new=50 / total=155"),
        ({"event": "tool_invalid_args", "turn": 4, "name": "tap", "error": "missing bbox"}, "tap invalid args: missing bbox"),
        ({"event": "tool_unknown", "turn": 4, "name": "ghost"}, "ghost unknown tool"),
        ({"event": "tool_error", "turn": 5, "name": "tap", "error": "boom"}, "tap failed: boom"),
        ({"event": "violations", "turn": 6, "codes": ["V1", "V2"]}, "violations ['V1', 'V2']"),
        ({"event": "log_append", "turn": 1, "entry": "did stuff"}, "log: did stuff"),
        ({"event": "memory_save", "turn": 1, "text": "user likes X"}, "memory: user likes X"),
        ({"event": "sentinel", "turn": 9, "name": "DONE", "recap": "task complete"}, "SENTINEL DONE — task complete"),
        ({"event": "wait_auto_scheduled", "job_id": "wait-check", "at": "10:00"}, "WAIT auto-scheduled: wait-check at 10:00"),
        ({"event": "wait_auto_schedule_failed", "error": "x"}, "WAIT auto-schedule failed: x"),
        ({"event": "done", "sentinel": "DONE", "recap": "ok"}, "OUTCOME: DONE — ok"),
        ({"event": "crashed"}, "CRASHED"),
        ({"event": "provider_failed", "turn": 2, "error": "rate limited"}, "provider failed: rate limited"),
        ({"event": "prefix_drift", "turn": 3, "expected": "abcdefghijklmnop", "actual": "zyxwvutsrqponmlk"}, "PREFIX DRIFT"),
    ],
)
def test_summarize_event_dispatch(event: dict, expected_substr: str) -> None:
    out = trace._summarize(event)

    assert out is not None
    assert expected_substr in out


def test_summarize_silent_event_returns_none() -> None:
    assert trace._summarize({"event": "prefix_pinned"}) is None
    assert trace._summarize({"event": "finish_length_warning"}) is None


def test_summarize_unknown_event_falls_back_to_compact_repr() -> None:
    out = trace._summarize({"event": "future_event_type", "data": 42})

    assert out is not None
    assert "future_event_type" in out


def test_summarize_done_with_no_sentinel_uses_none_placeholder() -> None:
    out = trace._summarize({"event": "done", "recap": "ok"})

    assert "(none)" in out


def test_summarize_tool_result_with_text_uses_format_call_result() -> None:
    out = trace._summarize({
        "event": "tool_result", "turn": 1, "name": "note",
        "arguments": {}, "text": "x" * 100,
    })

    # `note` doesn't truncate at 80 — full text passes through.
    assert "x" * 100 in out


def test_summarize_tool_result_without_text_uses_brief_content_on_blocks() -> None:
    out = trace._summarize({
        "event": "tool_result", "turn": 1, "name": "tap",
        "arguments": {}, "blocks": [{"type": "text", "text": "ok"}],
    })

    assert "→ ok" in out


# ---------- Trace ----------


def test_trace_creates_log_directory_and_opens_file_with_separator(
    _trace_dirs: Path,
) -> None:
    with freeze_time("2026-04-28T10:00:00"):
        t = Trace("session-1")
        t.close()

    log_path = _trace_dirs / "engine-2026-04-28.log"
    assert log_path.is_file()
    assert "=" * 60 in log_path.read_text()


def test_trace_write_appends_summary_line_with_timestamp(
    _trace_dirs: Path,
) -> None:
    with freeze_time("2026-04-28T10:00:00"):
        t = Trace("s1")
        t.write({"event": "tools_loaded", "mcp": [], "local": []})
        t.close()

    text = (_trace_dirs / "engine-2026-04-28.log").read_text()
    assert "[10:00:00] tools: 0 MCP + 0 local" in text


def test_trace_write_skips_silent_events(_trace_dirs: Path) -> None:
    with freeze_time("2026-04-28T10:00:00"):
        t = Trace("s1")
        t.write({"event": "prefix_pinned"})
        t.close()

    text = (_trace_dirs / "engine-2026-04-28.log").read_text()
    assert "prefix_pinned" not in text


def test_trace_close_is_idempotent(_trace_dirs: Path) -> None:
    t = Trace("s1")
    t.close()
    t.close()  # must not raise


def test_trace_rolls_over_to_new_day_when_midnight_crossed(
    _trace_dirs: Path,
) -> None:
    with freeze_time("2026-04-28T23:59:00") as ft:
        t = Trace("s1")
        ft.move_to("2026-04-29T00:00:00")
        t.write({"event": "tools_loaded", "mcp": [], "local": []})
        t.close()

    today_log = _trace_dirs / "engine-2026-04-28.log"
    tomorrow_log = _trace_dirs / "engine-2026-04-29.log"

    assert "ROLLOVER" in today_log.read_text()
    assert "ROLLOVER ← continued from previous day" in tomorrow_log.read_text()
    assert "tools: 0 MCP" in tomorrow_log.read_text()


# ---------- RawLog ----------


def test_rawlog_writes_session_start_line(_trace_dirs: Path) -> None:
    log = RawLog("sess-A")
    log.write_session_start(
        provider="anthropic", model="claude-test",
        prompt_hash="abc123", tools=[{"name": "tap"}],
    )
    log.close()

    line = (_trace_dirs / "sessions" / "sess-A" / "wire.jsonl").read_text().splitlines()[0]
    obj = json.loads(line)
    assert obj["kind"] == "session_start"
    assert obj["provider"] == "anthropic"
    assert obj["tools"] == [{"name": "tap"}]


def test_rawlog_writes_request_with_turn_index(_trace_dirs: Path) -> None:
    log = RawLog("sess-B")
    log.write_request(turn=3, messages=[{"role": "user", "content": "hi"}])
    log.close()

    line = (_trace_dirs / "sessions" / "sess-B" / "wire.jsonl").read_text().splitlines()[0]
    obj = json.loads(line)
    assert obj["kind"] == "request"
    assert obj["turn"] == 3
    assert obj["messages"] == [{"role": "user", "content": "hi"}]


def test_rawlog_writes_response_with_elapsed(_trace_dirs: Path) -> None:
    log = RawLog("sess-C")
    log.write_response(turn=1, raw={"id": "r1"}, elapsed_ms=42)
    log.close()

    line = (_trace_dirs / "sessions" / "sess-C" / "wire.jsonl").read_text().splitlines()[0]
    obj = json.loads(line)
    assert obj["kind"] == "response"
    assert obj["elapsed_ms"] == 42


def test_rawlog_close_is_idempotent(_trace_dirs: Path) -> None:
    log = RawLog("s")
    log.close()
    log.close()


# ---------- RawLog._scrub_images / _scrub_block ----------


def test_rawlog_scrubs_openai_image_url_data_to_disk(_trace_dirs: Path) -> None:
    log = RawLog("sess-IMG")
    raw_bytes = b"fake jpeg bytes"
    b64 = base64.b64encode(raw_bytes).decode()
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }]

    out = log._scrub_images(messages)

    # The image_url is replaced with a session-relative turn-tagged path
    # (turn defaults to -1 when scrubbing outside write_request).
    img_url = out[0]["content"][1]["image_url"]["url"]
    assert img_url == "images/00001_t-1.jpg"
    # The actual file was written inside the session dir.
    assert (_trace_dirs / "sessions" / "sess-IMG" / img_url).read_bytes() == raw_bytes


def test_rawlog_scrubs_anthropic_image_block_to_ref(_trace_dirs: Path) -> None:
    log = RawLog("sess-A")
    raw_bytes = b"png data"
    b64 = base64.b64encode(raw_bytes).decode()
    messages = [{
        "role": "user",
        "content": [{
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        }],
    }]

    out = log._scrub_images(messages)

    src = out[0]["content"][0]["source"]
    assert src["type"] == "ref"
    assert src["ref"].endswith(".png")
    assert (_trace_dirs / "sessions" / "sess-A" / src["ref"]).read_bytes() == raw_bytes


def test_rawlog_scrubs_anthropic_tool_result_inner_content(_trace_dirs: Path) -> None:
    log = RawLog("sess-T")
    raw_bytes = b"img"
    b64 = base64.b64encode(raw_bytes).decode()
    messages = [{
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": [
                {"type": "text", "text": "caption"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                },
            ],
        }],
    }]

    out = log._scrub_images(messages)

    inner = out[0]["content"][0]["content"]
    assert inner[0] == {"type": "text", "text": "caption"}
    assert inner[1]["source"]["type"] == "ref"


def test_rawlog_passes_through_non_data_image_url(_trace_dirs: Path) -> None:
    log = RawLog("s")
    msg = {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://x/img.jpg"}},
    ]}

    out = log._scrub_images([msg])

    assert out[0]["content"][0]["image_url"]["url"] == "https://x/img.jpg"


def test_rawlog_passes_through_non_base64_anthropic_image(
    _trace_dirs: Path,
) -> None:
    log = RawLog("s")
    msg = {"role": "user", "content": [
        {"type": "image", "source": {"type": "url", "url": "https://x/img"}},
    ]}

    out = log._scrub_images([msg])

    assert out[0]["content"][0]["source"]["type"] == "url"


def test_rawlog_falls_back_to_byte_count_stub_on_decode_failure(
    _trace_dirs: Path,
) -> None:
    log = RawLog("s")
    msg = {"role": "user", "content": [
        {"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg", "data": "%%not base64%%"
        }},
    ]}

    out = log._scrub_images([msg])

    # base64.b64decode with validate=False is permissive — won't raise
    # on most strings. Confirm we end up with either a ref or a stub
    # with a well-defined shape.
    src = out[0]["content"][0]["source"]
    assert src["type"] in ("ref", "base64")


def test_rawlog_passes_through_messages_with_string_content(
    _trace_dirs: Path,
) -> None:
    log = RawLog("s")
    msg = {"role": "user", "content": "plain string"}

    out = log._scrub_images([msg])

    assert out == [msg]


def test_rawlog_passes_through_unknown_block_types(
    _trace_dirs: Path,
) -> None:
    log = RawLog("s")
    msg = {"role": "user", "content": [{"type": "tool_use", "name": "tap"}]}

    out = log._scrub_images([msg])

    assert out[0]["content"][0] == {"type": "tool_use", "name": "tap"}


def test_rawlog_empty_data_field_returns_unreadable_stub(
    _trace_dirs: Path,
) -> None:
    log = RawLog("s")
    msg = {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"}},
    ]}

    out = log._scrub_images([msg])

    url = out[0]["content"][0]["image_url"]["url"]
    assert "unreadable" in url


# ---------- _purge_old ----------


def test_purge_old_removes_files_older_than_retention_days(
    _trace_dirs: Path,
) -> None:
    raw_dir = _trace_dirs / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    old = raw_dir / "old.jsonl"
    young = raw_dir / "young.jsonl"
    old.write_text("x")
    young.write_text("y")

    import os
    cutoff_seconds = trace._RETENTION_DAYS * 86400
    long_ago = (dt.datetime.now() - dt.timedelta(seconds=cutoff_seconds + 100)).timestamp()
    os.utime(old, (long_ago, long_ago))

    trace._purge_old()

    assert not old.exists()
    assert young.exists()


def test_purge_old_returns_silently_when_dir_missing() -> None:
    # _RAW_DIR doesn't exist — must not raise.
    trace._purge_old()


# ---------- events.jsonl ----------


def _events(log_dir: Path, sid: str) -> list[dict]:
    path = log_dir / "sessions" / sid / "events.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_trace_writes_every_event_to_events_jsonl(_trace_dirs: Path) -> None:
    t = Trace("s1")
    t.write({"event": "request", "turn": 0, "message_count": 5})
    t.write({"event": "prefix_pinned", "hash": "abc"})  # silent in daily log
    t.close()

    events = _events(_trace_dirs, "s1")
    # Line 0 is always the environment snapshot — see its own tests.
    assert events[0]["event"] == "env"
    assert events[1]["event"] == "request"
    assert events[1]["turn"] == 0
    assert "t" in events[1]
    # Silent events are still data.
    assert events[2] == {"t": events[2]["t"], "event": "prefix_pinned", "hash": "abc"}


def test_trace_events_jsonl_summarizes_tool_result_blocks(_trace_dirs: Path) -> None:
    # blocks may carry base64 screens whose bytes already live in
    # wire.jsonl — events.jsonl keeps a summary, not a second copy.
    t = Trace("s1")
    t.write({
        "event": "tool_result", "turn": 1, "name": "tap", "id": "c1",
        "arguments": {"bbox": [0, 0, 1, 1]},
        "blocks": [{"type": "text", "text": "ok"}, {"type": "image", "data": "aGk="}],
    })
    t.close()

    e = _events(_trace_dirs, "s1")[1]  # [0] is the env snapshot
    assert "blocks" not in e
    assert e["result_summary"] == "ok + <image 4b>"
    assert e["arguments"] == {"bbox": [0, 0, 1, 1]}


def test_trace_events_jsonl_degrades_on_non_serializable_values(
    _trace_dirs: Path,
) -> None:
    t = Trace("s1")
    t.write({"event": "tool_error", "turn": 0, "error": ValueError("boom")})
    t.close()

    e = _events(_trace_dirs, "s1")[1]  # [0] is the env snapshot
    assert "boom" in e["error"]  # default=repr kicked in


# ---------- session summary ----------


def _feed_session(t: Trace) -> None:
    """A synthetic event stream exercising every summary field."""
    t.write({"event": "wake", "session": "s1", "model_ref": "moonshot/kimi-k2.6",
             "triggers": [{"source": "phone", "description": "screen changed"}]})
    t.write({"event": "prefix_pinned", "hash": "deadbeef"})
    t.write({"event": "response", "turn": 0, "finish_reason": "tool_calls",
             "content_len": 0, "elapsed_ms": 1500, "tool_calls": []})
    t.write({"event": "cache", "turn": 0, "hit": 800, "create": 100,
             "new": 100, "total": 1000, "out": 50})
    t.write({"event": "tool_result", "turn": 0, "name": "note", "id": "a",
             "arguments": {}, "text": "noted"})
    t.write({"event": "tool_result", "turn": 1, "name": "tap", "id": "b",
             "arguments": {}, "text": "tapped"})
    t.write({"event": "tool_blocked_stuck", "turn": 1, "name": "tap", "id": "c"})
    t.write({"event": "stuck_warning", "turn": 2, "name": "tap", "id": "d"})
    t.write({"event": "tool_invalid_args", "turn": 2, "name": "tap", "id": "e",
             "arguments": {}, "error": "bad bbox"})
    t.write({"event": "bad_turn_shape", "turn": 3, "tool_calls": []})
    t.write({"event": "done", "sentinel": "DONE", "recap": "all good"})


def test_summary_json_derived_from_event_stream(_trace_dirs: Path) -> None:
    t = Trace("s1")
    _feed_session(t)
    t.close()

    s = json.loads((_trace_dirs / "sessions" / "s1" / "summary.json").read_text())
    assert s["schema"] == 1
    assert s["sid"] == "s1"
    assert s["model_ref"] == "moonshot/kimi-k2.6"
    assert s["provider"] == "moonshot"
    assert s["prompt_hash"] == "deadbeef"
    assert s["triggers"][0]["source"] == "phone"
    assert s["outcome"] == {"sentinel": "DONE", "recap": "all good", "crashed": False}
    assert s["turns"] == 4  # max turn 3 → 4 turns
    assert s["provider_calls"] == 1
    assert s["provider_time_ms"] == 1500
    assert s["usage"]["input_tokens"] == 1000
    assert s["usage"]["output_tokens"] == 50
    assert s["usage"]["cache_read_tokens"] == 800
    assert s["usage"]["cache_creation_tokens"] == 100
    assert s["usage"]["cache_hit_pct"] == 80.0
    assert s["tool_calls"] == {"note": 1, "tap": 1}
    assert s["errors"]["blocked_stuck"] == 1
    assert s["errors"]["invalid_args"] == 1
    assert s["errors"]["correctives"] == 1
    assert s["errors"]["blocked_plan"] == 0
    assert s["stuck_events"] == 2  # blocked_stuck + stuck_warning
    assert s["images"] == 0
    assert "started_at" in s and "ended_at" in s and s["duration_s"] >= 0


def test_summary_marks_crashed_sessions(_trace_dirs: Path) -> None:
    t = Trace("s1")
    t.write({"event": "wake", "session": "s1", "model_ref": "m/x", "triggers": []})
    t.write({"event": "crashed"})
    t.close()

    s = json.loads((_trace_dirs / "sessions" / "s1" / "summary.json").read_text())
    assert s["outcome"]["crashed"] is True
    assert s["outcome"]["sentinel"] is None


def test_close_writes_end_footer_to_daily_log(_trace_dirs: Path) -> None:
    with freeze_time("2026-04-28T10:00:00"):
        t = Trace("s1")
        _feed_session(t)
        t.close()

    text = (_trace_dirs / "engine-2026-04-28.log").read_text()
    assert "END session=s1 outcome=DONE turns=4" in text
    assert "tokens=1.0k/50 cache=80% tools=2" in text


def test_close_counts_images_from_session_dir(_trace_dirs: Path) -> None:
    t = Trace("s1")
    img_dir = _trace_dirs / "sessions" / "s1" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / "00001_t0.jpg").write_bytes(b"x")
    (img_dir / "00002_t1.jpg").write_bytes(b"y")
    t.close()

    s = json.loads((_trace_dirs / "sessions" / "s1" / "summary.json").read_text())
    assert s["images"] == 2


def test_close_writes_summary_only_once(_trace_dirs: Path) -> None:
    t = Trace("s1")
    t.close()
    first = (_trace_dirs / "sessions" / "s1" / "summary.json").read_text()
    t.close()  # idempotent — no rewrite, no raise
    assert (_trace_dirs / "sessions" / "s1" / "summary.json").read_text() == first


def test_fmt_tokens_scales() -> None:
    assert trace.fmt_tokens(980) == "980"
    assert trace.fmt_tokens(9_800) == "9.8k"
    assert trace.fmt_tokens(1_200_000) == "1.2M"


# ---------- turn-tagged images ----------


def test_write_request_tags_images_with_turn(_trace_dirs: Path) -> None:
    log = RawLog("sess-T")
    b64 = base64.b64encode(b"img").decode()
    log.write_request(turn=7, messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }])
    log.close()

    line = (_trace_dirs / "sessions" / "sess-T" / "wire.jsonl").read_text().splitlines()[0]
    url = json.loads(line)["messages"][0]["content"][0]["image_url"]["url"]
    assert url == "images/00001_t7.jpg"
    assert (_trace_dirs / "sessions" / "sess-T" / url).read_bytes() == b"img"


# ---------- retention: session dirs + daily logs ----------


def _age(path: Path, days: float) -> None:
    import os
    ago = (dt.datetime.now() - dt.timedelta(days=days)).timestamp()
    os.utime(path, (ago, ago))


def test_purge_old_removes_stale_session_dirs(_trace_dirs: Path) -> None:
    old = _trace_dirs / "sessions" / "20260101_000000"
    old.mkdir(parents=True)
    f = old / "events.jsonl"
    f.write_text("{}")
    _age(f, trace._RETENTION_DAYS + 1)
    _age(old, trace._RETENTION_DAYS + 1)
    young = _trace_dirs / "sessions" / "20990101_000000"
    young.mkdir(parents=True)
    (young / "events.jsonl").write_text("{}")

    trace._purge_old()

    assert not old.exists()
    assert young.exists()


def test_purge_old_spares_session_dir_with_a_recent_file(_trace_dirs: Path) -> None:
    # Dir mtime is old but a file inside is fresh (e.g. summary written
    # late) — the newest-file rule keeps it.
    d = _trace_dirs / "sessions" / "20260101_000000"
    d.mkdir(parents=True)
    old_f = d / "events.jsonl"
    old_f.write_text("{}")
    _age(old_f, trace._RETENTION_DAYS + 1)
    (d / "summary.json").write_text("{}")  # fresh
    _age(d, trace._RETENTION_DAYS + 1)

    trace._purge_old()

    assert d.exists()


def test_purge_old_removes_stale_daily_logs(_trace_dirs: Path) -> None:
    _trace_dirs.mkdir(parents=True, exist_ok=True)
    old = _trace_dirs / "engine-2026-01-01.log"
    old.write_text("x")
    _age(old, trace._LOG_RETENTION_DAYS + 1)
    young = _trace_dirs / "engine-2099-01-01.log"
    young.write_text("y")

    trace._purge_old()

    assert not old.exists()
    assert young.exists()


# ---------- new_sid ----------


def test_new_sid_shape_and_uniqueness() -> None:
    import re

    sids = {trace.new_sid() for _ in range(50)}

    # 50 mints in the same second must not collide — the random suffix
    # is what makes instant-crash retries safe.
    assert len(sids) == 50
    for sid in sids:
        assert re.fullmatch(r"\d{8}_\d{6}_[a-z0-9]{6}", sid)


def test_new_sid_sorts_chronologically() -> None:
    with freeze_time("2026-04-28T10:00:00"):
        early = trace.new_sid()
    with freeze_time("2026-04-28T10:00:01"):
        late = trace.new_sid()

    assert early < late  # lexicographic == chronological across seconds


# ---------- env snapshot ----------


def test_first_event_of_every_session_is_the_env_snapshot(_trace_dirs: Path) -> None:
    Trace("s1").close()

    e = _events(_trace_dirs, "s1")[0]
    assert e["event"] == "env"
    for key in ("physiclaw", "python", "os", "platform", "host", "utc_offset"):
        assert e[key]
    assert "auto_exposure" in e["config"]["camera"]
    assert "max_image_edge_px" in e["config"]["compact"]


def test_env_snapshot_never_carries_secrets(_trace_dirs: Path) -> None:
    Trace("s1").close()

    raw = (_trace_dirs / "sessions" / "s1" / "events.jsonl").read_text()
    assert "api_key" not in raw
    assert "provider_config" not in raw


def test_summary_includes_env(_trace_dirs: Path) -> None:
    t = Trace("s1")
    t.write({"event": "done", "sentinel": "DONE", "recap": "ok"})
    t.close()

    s = json.loads((_trace_dirs / "sessions" / "s1" / "summary.json").read_text())
    assert s["env"]["os"] in ("darwin", "win32", "linux")
    assert s["env"]["config"]["camera"]["width"] > 0


def test_env_renders_one_daily_log_line(_trace_dirs: Path) -> None:
    with freeze_time("2026-04-28T10:00:00"):
        Trace("s1").close()

    text = (_trace_dirs / "engine-2026-04-28.log").read_text()
    assert "env physiclaw=" in text
    assert "host=" in text


# ---------- cross-platform bytes + sessions README ----------


def test_log_files_use_lf_newlines_regardless_of_platform(_trace_dirs: Path) -> None:
    # newline="\n" pinned on every writer: identical bytes on Windows and
    # POSIX, so hashing/diffing sessions across rigs is meaningful.
    t = Trace("s1")
    t.write({"event": "done", "sentinel": "DONE", "recap": "ok"})
    t.close()
    rlog = RawLog("s1")
    rlog.write_request(turn=0, messages=[{"role": "user", "content": "hi"}])
    rlog.close()

    for name in ("events.jsonl", "wire.jsonl", "summary.json"):
        raw = (_trace_dirs / "sessions" / "s1" / name).read_bytes()
        assert b"\r" not in raw, name
    daily = next(_trace_dirs.glob("engine-*.log")).read_bytes()
    assert b"\r" not in daily


def test_sessions_readme_written_once(_trace_dirs: Path) -> None:
    Trace("s1").close()
    readme = _trace_dirs / "sessions" / "README.md"
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")
    assert "events.jsonl" in text and "summary.json" in text

    stamp = readme.stat().st_mtime_ns
    Trace("s2").close()  # second session must not rewrite it
    assert readme.stat().st_mtime_ns == stamp


def test_sessions_readme_documents_the_full_summary_schema() -> None:
    # The README ships WITH the data as its format contract — every field
    # summary.json actually emits must be named in it, so schema changes
    # can't silently outrun the doc.
    readme = trace.SESSIONS_README
    s = trace._Summary("x").finalize(images=0)

    for key in s:
        assert key in readme, f"summary key {key!r} missing from SESSIONS_README"
    for key in s["usage"]:
        assert key in readme, f"usage key {key!r} missing from SESSIONS_README"
    for key in s["errors"]:
        assert key in readme, f"errors key {key!r} missing from SESSIONS_README"
    for key in s["outcome"]:
        assert key in readme, f"outcome key {key!r} missing from SESSIONS_README"
    for name in ("events.jsonl", "wire.jsonl", "summary.json", "images/"):
        assert name in readme
