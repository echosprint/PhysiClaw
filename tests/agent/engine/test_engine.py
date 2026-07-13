"""Tests for `physiclaw.agent.engine.engine` — focused on testable
helpers. The full `run()` loop is integration-tested separately
(`test_engine_loop.py`); here we cover the pure units:
`_chat_with_retry`, `_log_usage`, `_corrective_for_bad_shape`,
`assemble.format_triggers`, `_auto_schedule_wait_check`.
"""

from __future__ import annotations

import datetime as dt

import pytest
from freezegun import freeze_time

from physiclaw.agent.engine import engine as engine_mod
from physiclaw.agent.engine.assemble import format_triggers
from physiclaw.agent.engine.dto import (
    AssistantMessage,
    FinishReason,
    Usage,
)
from physiclaw.agent.engine.engine import (
    _auto_schedule_wait_check,
    _chat_with_retry,
    _corrective_for_bad_shape,
    _log_usage,
)
from physiclaw.agent.engine.session import Session
from physiclaw.agent.engine.trace import Trace
from physiclaw.agent.provider.provider_base import ProviderTransientError
from physiclaw.agent.runtime.hook import Trigger

# ---------- _chat_with_retry ----------


@pytest.mark.asyncio
async def test_chat_with_retry_returns_immediately_on_success(mocker) -> None:
    provider = mocker.MagicMock()
    asst = AssistantMessage(
        content="ok", tool_calls=[], finish_reason=FinishReason.STOP
    )
    provider.chat = mocker.AsyncMock(return_value=asst)

    out = await _chat_with_retry(provider, [], [], attempts=3, backoff=0.0)

    assert out is asst
    assert provider.chat.await_count == 1


@pytest.mark.asyncio
async def test_chat_with_retry_retries_then_succeeds(mocker) -> None:
    mocker.patch("asyncio.sleep")
    provider = mocker.MagicMock()
    asst = AssistantMessage(
        content="ok", tool_calls=[], finish_reason=FinishReason.STOP
    )
    provider.chat = mocker.AsyncMock(
        side_effect=[ProviderTransientError("transient"), asst]
    )

    out = await _chat_with_retry(provider, [], [], attempts=3, backoff=0.0)

    assert out is asst
    assert provider.chat.await_count == 2


@pytest.mark.asyncio
async def test_chat_with_retry_raises_runtime_after_max_attempts(mocker) -> None:
    mocker.patch("asyncio.sleep")
    provider = mocker.MagicMock()
    provider.chat = mocker.AsyncMock(side_effect=ProviderTransientError("nope"))

    with pytest.raises(RuntimeError, match=r"^provider failed after 2 attempts:"):
        await _chat_with_retry(provider, [], [], attempts=2, backoff=0.0)


@pytest.mark.asyncio
async def test_chat_with_retry_does_not_catch_permanent_errors(mocker) -> None:
    provider = mocker.MagicMock()
    provider.chat = mocker.AsyncMock(side_effect=RuntimeError("permanent"))

    with pytest.raises(RuntimeError, match=r"^permanent$"):
        await _chat_with_retry(provider, [], [], attempts=3, backoff=0.0)


# ---------- _log_usage ----------


@pytest.fixture
def trace_stub(mocker):
    t = mocker.MagicMock(spec=Trace)
    return t


def test_log_usage_returns_empty_string_when_no_usage_data(
    trace_stub,
) -> None:
    asst = AssistantMessage(
        content="x",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        usage=Usage(),
    )

    out = _log_usage(turn=1, asst=asst, tr=trace_stub)

    assert out == ""
    trace_stub.write.assert_called_once()


def test_log_usage_emits_cache_event_with_derived_new_count(
    trace_stub,
) -> None:
    asst = AssistantMessage(
        content="x",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        usage=Usage(prompt_tokens=200, cached_tokens=120, cache_creation_tokens=30),
    )

    _log_usage(turn=5, asst=asst, tr=trace_stub)

    trace_stub.write.assert_called_once()
    payload = trace_stub.write.call_args.args[0]
    assert payload["event"] == "cache"
    assert payload["turn"] == 5
    assert payload["hit"] == 120
    assert payload["create"] == 30
    # new = total - cached - created = 200 - 120 - 30 = 50
    assert payload["new"] == 50
    assert payload["total"] == 200


def test_log_usage_returns_token_summary_with_cache_pct(trace_stub) -> None:
    asst = AssistantMessage(
        content="",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        usage=Usage(prompt_tokens=10000, cached_tokens=5000, cache_creation_tokens=0),
    )

    out = _log_usage(turn=1, asst=asst, tr=trace_stub)

    assert out == "token: 10.0k, cache: 50%"


def test_log_usage_clamps_new_at_zero_when_cached_exceeds_total(
    trace_stub,
) -> None:
    # Defensive: if a provider reports cached > total (shouldn't happen
    # but...), `new` floors at 0.
    asst = AssistantMessage(
        content="",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        usage=Usage(prompt_tokens=100, cached_tokens=200, cache_creation_tokens=0),
    )

    _log_usage(turn=1, asst=asst, tr=trace_stub)

    payload = trace_stub.write.call_args.args[0]
    assert payload["new"] == 0


# ---------- _corrective_for_bad_shape ----------


def test_corrective_for_action_without_note() -> None:
    out = _corrective_for_bad_shape(["peek"])

    assert "called `peek` without `note`" in out
    assert "[note(summary=...), peek(...)]" in out


def test_corrective_for_action_without_note_with_extras() -> None:
    out = _corrective_for_bad_shape(["peek", "tap"])

    assert "without `note`" in out
    assert "too many action tools" in out
    assert "['tap']" in out


def test_corrective_for_note_alone() -> None:
    out = _corrective_for_bad_shape(["note"])

    assert "`note` alone with no action tool" in out
    assert "peek()" in out  # default suggestion


def test_corrective_for_note_with_too_many_actions() -> None:
    out = _corrective_for_bad_shape(["note", "peek", "tap"])

    assert "`note` plus 2 action tools" in out


def test_corrective_for_multiple_notes() -> None:
    out = _corrective_for_bad_shape(["note", "note"])

    assert "called `note` 2 times" in out


def test_corrective_for_three_notes() -> None:
    out = _corrective_for_bad_shape(["note", "note", "note"])

    assert "called `note` 3 times" in out


# ---------- assemble.format_triggers ----------


def test_format_triggers_includes_now_and_each_trigger() -> None:
    triggers = [
        Trigger(description="phone IM arrived", source="phone"),
        Trigger(description="cron fired", source="cron:user-greet"),
    ]

    with freeze_time("2026-04-28T14:30:00"):
        out = format_triggers(triggers)

    assert out.startswith("Now: 2026-04-28")
    assert "[Current wake — act on this]" in out
    assert "phone: phone IM arrived" in out
    assert "cron:user-greet: cron fired" in out


def test_format_triggers_uses_manual_for_empty_source() -> None:
    triggers = [Trigger(description="user typed", source="")]

    with freeze_time("2026-04-28T14:30:00"):
        out = format_triggers(triggers)

    assert "manual: user typed" in out


def test_format_triggers_appends_cron_context_when_provided() -> None:
    triggers = [Trigger(description="x", source="phone")]

    with freeze_time("2026-04-28T14:30:00"):
        out = format_triggers(
            triggers, cron_ctx="## Scheduled jobs firing now\n\n### foo"
        )

    assert out.endswith("### foo")


def test_format_triggers_omits_cron_section_when_blank() -> None:
    out = format_triggers([Trigger(description="x", source="phone")])

    assert "Scheduled jobs" not in out


# ---------- _auto_schedule_wait_check ----------


def test_auto_schedule_wait_check_calls_jobs_upsert(
    trace_stub,
    mocker,
) -> None:
    upsert = mocker.patch("physiclaw.agent.engine.jobs.upsert_auto_wait_check")

    with freeze_time("2026-04-28T14:00:00"):
        _auto_schedule_wait_check(trace_stub, minutes=15)

    upsert.assert_called_once()
    target_arg = upsert.call_args.args[0]
    assert isinstance(target_arg, dt.datetime)
    # Target time = now + 15 minutes.
    assert target_arg.hour == 14
    assert target_arg.minute == 15


def test_auto_schedule_wait_check_emits_scheduled_event_on_success(
    trace_stub, mocker
) -> None:
    mocker.patch("physiclaw.agent.engine.jobs.upsert_auto_wait_check")

    _auto_schedule_wait_check(trace_stub, minutes=15)

    payload = trace_stub.write.call_args.args[0]
    assert payload["event"] == "wait_auto_scheduled"
    assert "job_id" in payload
    assert "at" in payload


def test_auto_schedule_wait_check_emits_failure_event_on_exception(
    trace_stub, mocker, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    mocker.patch(
        "physiclaw.agent.engine.jobs.upsert_auto_wait_check",
        side_effect=RuntimeError("disk full"),
    )

    with caplog.at_level(logging.ERROR, logger="physiclaw.agent.engine.engine"):
        _auto_schedule_wait_check(trace_stub, minutes=15)

    payload = trace_stub.write.call_args.args[0]
    assert payload["event"] == "wait_auto_schedule_failed"
    assert "disk full" in payload["error"]


def test_log_usage_emits_output_tokens_for_the_session_summary(
    trace_stub,
) -> None:
    # The session summary sums `cache.out` into usage.output_tokens.
    asst = AssistantMessage(
        content="",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        usage=Usage(
            prompt_tokens=100,
            cached_tokens=0,
            cache_creation_tokens=0,
            completion_tokens=77,
        ),
    )

    engine_mod._log_usage(turn=1, asst=asst, tr=trace_stub)

    payload = trace_stub.write.call_args.args[0]
    assert payload["out"] == 77


def _settings(**over) -> engine_mod.Settings:
    base = dict(
        max_turns=300,
        max_session_attempts=3,
        provider_retry_attempts=3,
        retry_backoff_seconds=0.0,
        wait_default_minutes=15,
    )
    base.update(over)
    return engine_mod.Settings(**base)


@pytest.mark.asyncio
async def test_call_provider_response_event_carries_elapsed_ms(
    mocker,
    trace_stub,
) -> None:
    # The session summary sums `response.elapsed_ms` into provider_time_ms.
    from physiclaw.agent.engine.policy import default_policies

    asst = AssistantMessage(
        content="",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        usage=Usage(),
    )
    provider = mocker.MagicMock()
    provider.chat = mocker.AsyncMock(return_value=asst)
    rlog = mocker.MagicMock()
    run = engine_mod.EngineRun(
        provider=provider,
        mcp=mocker.MagicMock(),
        tool_schemas=[],
        schema_by_name={},
        local_registry={},
        tr=trace_stub,
        rlog=rlog,
        settings=_settings(),
        policies=default_policies(layout_incomplete=False),
    )

    out = await engine_mod._call_provider(run, Session(), [], turn=0)

    assert out is asst
    response_event = next(
        c.args[0]
        for c in trace_stub.write.call_args_list
        if c.args[0].get("event") == "response"
    )
    assert isinstance(response_event["elapsed_ms"], int)
    # RawLog got the same number.
    assert (
        rlog.write_response.call_args.kwargs["elapsed_ms"]
        == response_event["elapsed_ms"]
    )
