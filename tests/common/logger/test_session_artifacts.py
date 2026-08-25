"""Shared session-artifact surface: DailyLogWriter banner/rollover/footer,
atomic JSON, image naming, and — the reason the module exists —
summary.json schema parity between the two engines' finalize paths."""

from __future__ import annotations

import functools
import json
from pathlib import Path

from freezegun import freeze_time

from physiclaw.common.logger.session_artifacts import (
    DailyLogWriter,
    append_stats,
    build_summary,
    image_filename,
    write_json_atomic,
)

# ---------- append_stats ----------


def test_append_stats_creates_then_appends_one_line_per_session(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "log" / "engine"  # not pre-created — helper mkdirs

    append_stats(log_dir, {"sid": "a", "turns": 3})
    append_stats(log_dir, {"sid": "b", "turns": 5})

    lines = (log_dir / "stats.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln)["sid"] for ln in lines] == ["a", "b"]


# ---------- DailyLogWriter ----------


def test_daily_log_writer_banner_then_timestamped_lines(tmp_path: Path) -> None:
    with freeze_time("2026-04-28T10:15:30"):
        w = DailyLogWriter(tmp_path / "log", "engine")
        w.line("hello")
        w.close()

    text = (tmp_path / "log" / "engine-2026-04-28.log").read_text()
    assert text == f"\n{'=' * 60}\n[10:15:30] hello\n"


def test_daily_log_writer_footer_closes_the_session_block(tmp_path: Path) -> None:
    w = DailyLogWriter(tmp_path, "claude")
    w.footer()
    w.close()

    text = next(tmp_path.glob("claude-*.log")).read_text()
    assert text.endswith(f"{'=' * 60}\n\n")


def test_daily_log_writer_rolls_over_at_midnight(tmp_path: Path) -> None:
    with freeze_time("2026-04-28T23:59:59") as ft:
        w = DailyLogWriter(tmp_path, "claude")
        w.line("before")
        ft.move_to("2026-04-29T00:00:01")
        w.line("after")
        w.close()

    old = (tmp_path / "claude-2026-04-28.log").read_text()
    new = (tmp_path / "claude-2026-04-29.log").read_text()
    assert "ROLLOVER → claude-2026-04-29.log" in old
    assert "ROLLOVER ← continued from previous day" in new
    assert "after" in new


def test_daily_log_writer_close_is_idempotent(tmp_path: Path) -> None:
    w = DailyLogWriter(tmp_path, "engine")
    w.close()
    w.close()


# ---------- helpers ----------


def test_write_json_atomic_leaves_no_tmp_file(tmp_path: Path) -> None:
    d = tmp_path / "session"
    d.mkdir()
    path = d / "summary.json"
    write_json_atomic(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert list(d.iterdir()) == [path]


def test_image_filename_stamps_time_and_turn() -> None:
    with freeze_time("2026-04-28T10:45:42.123456"):
        assert image_filename(20, "image/jpeg") == "104542_123_t20.jpg"
        assert image_filename(3, "application/x-unknown") == "104542_123_t3.bin"


# ---------- summary schema v1 parity ----------


def _key_tree(d: dict) -> dict:
    return {k: _key_tree(v) if isinstance(v, dict) else None for k, v in d.items()}


@functools.cache
def _engine_summary() -> dict:
    from physiclaw.agent.trace.trace import _Summary

    s = _Summary("eng-sid")
    for event in (
        {"event": "env", "physiclaw": "0.3.x", "os": "darwin"},
        {"event": "wake", "model_ref": "qwen/qwen3.6-plus", "triggers": []},
        {"event": "prefix_pinned", "hash": "abc"},
        {"event": "response", "turn": 0, "elapsed_ms": 100},
        {"event": "cache", "turn": 0, "total": 1000, "hit": 800, "create": 0, "out": 5},
        {"event": "tool_result", "turn": 0, "name": "peek"},
        {"event": "done", "sentinel": "DONE", "recap": "ok"},
    ):
        s.observe(event)
    return s.finalize(images=1)


@functools.cache
def _claude_summary() -> dict:
    from physiclaw.agent.claude.session_log import _ClaudeSummary
    from physiclaw.agent.runtime.hook import Trigger

    s = _ClaudeSummary(
        "claude-sid",
        [Trigger(description="wake", source="cron")],
        model_ref="claude-code/claude-sonnet-4-6",
        prompt_hash="abc",
    )
    for event in (
        {
            "type": "assistant",
            "message": {
                "id": "m1",
                "content": [{"type": "tool_use", "name": "peek", "input": {}}],
            },
        },
        {
            "type": "user",
            "message": {"content": [{"type": "tool_result", "is_error": True}]},
        },
        {
            "type": "result",
            "duration_api_ms": 100,
            "total_cost_usd": 0.01,
            "usage": {
                "input_tokens": 200,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 0,
                "output_tokens": 5,
            },
        },
    ):
        s.observe(event)
    return s.finalize(images=1)


def test_both_engines_emit_the_same_summary_key_tree() -> None:
    """The parity contract: `physiclaw logs` / `jq` read both engines'
    summary.json with one schema, modulo the declared per-engine keys:
    cost_usd is claude-only (the CLI reports it); tool_time_ms, verdicts,
    and conductor_turns are engine-only (the claude stream carries no
    per-tool timing, camera verdicts, or synthesized playbook turns)."""
    engine_full, claude_full = _engine_summary(), _claude_summary()
    engine, claude = _key_tree(engine_full), _key_tree(claude_full)

    assert claude.pop("cost_usd", "absent") is None  # present, scalar
    assert engine.pop("tool_time_ms", "absent") is None  # present, scalar
    assert engine.pop("conductor_turns", "absent") is None  # present, scalar
    assert isinstance(engine.pop("verdicts"), dict)
    # `env` is engine-populated data (the engine relays its env event
    # verbatim), not part of the constructed schema — opaque here.
    assert isinstance(engine.pop("env"), dict) and isinstance(claude.pop("env"), dict)
    assert claude == engine
    # Same top-level ordering too — the files diff cleanly across engines.
    _PER_ENGINE = {"cost_usd", "tool_time_ms", "verdicts", "conductor_turns"}
    assert [k for k in claude_full if k not in _PER_ENGINE] == [
        k for k in engine_full if k not in _PER_ENGINE
    ]


def test_claude_usage_folds_cache_tokens_into_input_tokens() -> None:
    # 200 fresh + 800 cache-read + 0 cache-create → the engine convention
    # where input_tokens is the total and cache_hit_pct derives from it.
    usage = _claude_summary()["usage"]
    assert usage["input_tokens"] == 1000
    assert usage["cache_hit_pct"] == 80.0


def test_claude_errors_render_full_engine_key_set() -> None:
    assert _claude_summary()["errors"] == {
        "blocked_plan": 0,
        "blocked_layout": 0,
        "blocked_stuck": 0,
        "invalid_args": 0,
        "unknown_tool": 0,
        "tool_errors": 1,
        "correctives": 0,
        "provider_failures": 0,
    }


def test_build_summary_zero_input_tokens_yields_zero_pct() -> None:
    s = build_summary(
        sid="s",
        started_at="2026-04-28T10:00:00.000",
        duration_s=1.0,
        model_ref="p/m",
        prompt_hash="",
        triggers=[],
        sentinel=None,
        recap="",
        crashed=False,
        turns=0,
        provider_calls=0,
        provider_time_ms=0,
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        tool_calls={},
        errors={},
        stuck_events=0,
        images=0,
        env={},
    )
    assert s["usage"]["cache_hit_pct"] == 0.0
    assert "cost_usd" not in s
