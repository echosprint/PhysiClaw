"""Tests for `physiclaw.agent.runtime.runtime`.

Runtime.start tests are integration-leaning (intricate event-loop +
hooks side effects); covered with one happy-path-and-stop scenario,
the rest deferred.
"""

from __future__ import annotations

import asyncio

import pytest
import respx

from physiclaw.agent.runtime import hook, runtime
from physiclaw.agent.runtime.hook import Trigger
from physiclaw.agent.runtime.runtime import Runtime, _check_ready, _maybe_await


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
    hook.clear()
    monkeypatch.setattr(runtime, "_client", None)
    monkeypatch.setenv("PHYSICLAW_SERVER", "http://test.host:8048")


# ---------- _maybe_await ----------


@pytest.mark.asyncio
async def test_maybe_await_returns_sync_value_unchanged() -> None:
    assert await _maybe_await("plain") == "plain"
    assert await _maybe_await(None) is None


@pytest.mark.asyncio
async def test_maybe_await_awaits_coroutines() -> None:
    async def inner():
        return "from-coro"

    assert await _maybe_await(inner()) == "from-coro"


# ---------- _check_ready ----------


@pytest.mark.asyncio
async def test_check_ready_delegates_through_the_module_client(
    respx_mock: respx.MockRouter,
) -> None:
    # One end-to-end probe proving the delegate wiring: the module's
    # long-lived client (base_url from the server env) reaches the shared
    # `common.ready.check_ready`. Payload/error semantics (false, 4xx,
    # missing field) are pinned centrally in tests/common/test_ready.py.
    respx_mock.get("http://test.host:8048/api/status").respond(json={"ready": True})

    assert await _check_ready() is True


# ---------- Runtime construction + stop ----------


def test_runtime_init_with_defaults() -> None:
    r = Runtime(react=lambda triggers: None)

    assert r.interval == 1.0
    assert r.label == ""
    assert r._running is False


def test_runtime_init_with_custom_interval_and_label() -> None:
    r = Runtime(react=lambda t: None, interval=0.5, label="qwen-engine")

    assert r.interval == 0.5
    assert r.label == "qwen-engine"


def test_runtime_stop_flips_flag() -> None:
    r = Runtime(react=lambda t: None)
    r._running = True

    r.stop()

    assert r._running is False


# ---------- Runtime.start integration ----------


def _stop_after_n(rt, n: int):
    """Build an async sleep stub that calls rt.stop() on the Nth call."""
    counter = {"n": 0}

    async def _sleep(_seconds):
        counter["n"] += 1
        if counter["n"] >= n:
            rt.stop()

    return _sleep, counter


@pytest.mark.asyncio
async def test_start_ready_calls_react_when_triggers_fire(mocker) -> None:
    react_calls: list = []

    def _react(triggers):
        react_calls.append(triggers)

    rt = runtime.Runtime(react=_react, interval=0.01, label="qwen")
    sleep_stub, _ = _stop_after_n(rt, 3)
    mocker.patch.object(runtime.asyncio, "sleep", side_effect=sleep_stub)
    mocker.patch.object(runtime, "_check_ready", side_effect=_async_returning(True))
    mocker.patch.object(runtime, "load_hooks")

    triggers = [Trigger(description="phone", source="phone")]
    mocker.patch.object(
        runtime,
        "check_hooks",
        side_effect=[triggers] + [[]] * 10,
    )

    await rt.start()

    assert len(react_calls) >= 1
    assert react_calls[0][0].source == "phone"


@pytest.mark.asyncio
async def test_start_skips_react_when_not_ready(mocker) -> None:
    react_spy = mocker.MagicMock()
    rt = runtime.Runtime(react=react_spy, interval=0.01)
    sleep_stub, _ = _stop_after_n(rt, 2)
    mocker.patch.object(runtime.asyncio, "sleep", side_effect=sleep_stub)
    mocker.patch.object(runtime, "_check_ready", side_effect=_async_returning(False))
    mocker.patch.object(runtime, "load_hooks")
    check_spy = mocker.patch.object(runtime, "check_hooks")

    await rt.start()

    react_spy.assert_not_called()
    check_spy.assert_not_called()


@pytest.mark.asyncio
async def test_start_warns_only_once_per_blip(
    mocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    rt = runtime.Runtime(react=lambda t: None, interval=0.01)
    sleep_stub, _ = _stop_after_n(rt, 4)
    mocker.patch.object(runtime.asyncio, "sleep", side_effect=sleep_stub)

    async def _bad_check():
        raise RuntimeError("server down")

    mocker.patch.object(runtime, "_check_ready", side_effect=_bad_check)
    mocker.patch.object(runtime, "load_hooks")
    mocker.patch.object(runtime, "check_hooks", return_value=[])

    with caplog.at_level(logging.WARNING, logger="physiclaw.agent.runtime.runtime"):
        await rt.start()

    warnings = [r for r in caplog.records if "status poll failed" in r.getMessage()]
    # Exactly one warning despite multiple failed polls.
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_start_logs_ready_transition(
    mocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    rt = runtime.Runtime(react=lambda t: None, interval=0.01, label="X")
    sleep_stub, _ = _stop_after_n(rt, 2)
    mocker.patch.object(runtime.asyncio, "sleep", side_effect=sleep_stub)
    mocker.patch.object(runtime, "_check_ready", side_effect=_async_returning(True))
    mocker.patch.object(runtime, "load_hooks")
    mocker.patch.object(runtime, "check_hooks", return_value=[])

    with caplog.at_level(logging.INFO, logger="physiclaw.agent.runtime.runtime"):
        await rt.start()

    assert any("physiclaw ready=True" in r.getMessage() for r in caplog.records)
    assert any("[X]" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_start_exception_in_tick_logs_and_continues(
    mocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    rt = runtime.Runtime(react=lambda t: None, interval=0.01)
    sleep_stub, _ = _stop_after_n(rt, 3)
    mocker.patch.object(runtime.asyncio, "sleep", side_effect=sleep_stub)
    mocker.patch.object(runtime, "_check_ready", side_effect=_async_returning(True))
    mocker.patch.object(runtime, "load_hooks")
    # Raise on first check, succeed afterwards so loop continues.
    call = {"n": 0}

    async def _check_hooks():
        call["n"] += 1
        if call["n"] == 1:
            raise RuntimeError("hook crashed")
        return []

    mocker.patch.object(runtime, "check_hooks", side_effect=_check_hooks)

    with caplog.at_level(logging.ERROR, logger="physiclaw.agent.runtime.runtime"):
        await rt.start()

    assert any("runtime tick failed" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_start_cancellation_propagates(mocker) -> None:
    rt = runtime.Runtime(react=lambda t: None, interval=0.01)

    async def _cancelled_check():
        raise asyncio.CancelledError

    mocker.patch.object(runtime, "_check_ready", side_effect=_cancelled_check)
    mocker.patch.object(runtime, "load_hooks")

    with pytest.raises(asyncio.CancelledError):
        await rt.start()


def _async_returning(value):
    async def _coro(*a, **kw):
        return value

    return _coro


@pytest.mark.asyncio
async def test_unproductive_sessions_back_off_the_wake_cadence(mocker) -> None:
    # STUCK/no-close streaks double the post-react cooldown (capped);
    # a DONE close resets it. A dead phone costs backoff windows, not a
    # full session per watchdog blip.
    from physiclaw.agent.runtime.contract import SessionOutcome

    outcomes = [
        SessionOutcome(status="STUCK"),
        SessionOutcome(status="STUCK"),
        SessionOutcome(status="DONE"),
    ]

    def _react(_triggers):
        return outcomes.pop(0)

    rt = runtime.Runtime(react=_react, interval=0.01)
    sleeps: list[float] = []

    async def _sleep(seconds):
        sleeps.append(seconds)
        if not outcomes and len(sleeps) >= 8:
            rt.stop()

    mocker.patch.object(runtime.asyncio, "sleep", side_effect=_sleep)
    mocker.patch.object(runtime, "_check_ready", side_effect=_async_returning(True))
    mocker.patch.object(runtime, "load_hooks")
    trig = [Trigger(description="x", source="cron")]
    mocker.patch.object(
        runtime, "check_hooks", side_effect=[trig, trig, trig] + [[]] * 100
    )

    await rt.start()

    base = runtime.CONFIG.engine.react_cooldown_seconds
    cooldowns = [s for s in sleeps if s >= base]
    # streak 1 → base*2, streak 2 → base*4, DONE → reset to base.
    assert cooldowns == [base * 2, base * 4, base]
    assert rt._streak == 0


def _backoff_rt(mocker, react, *, stop_after: int):
    """A Runtime wired for backoff tests: ready, one cron trigger, and a
    sleep collector that stops the loop after `stop_after` sleeps."""
    rt = runtime.Runtime(react=react, interval=0.01)
    sleeps: list[float] = []

    async def _sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= stop_after:
            rt.stop()

    mocker.patch.object(runtime.asyncio, "sleep", side_effect=_sleep)
    mocker.patch.object(runtime, "_check_ready", side_effect=_async_returning(True))
    mocker.patch.object(runtime, "load_hooks")
    trig = [Trigger(description="x", source="cron")]
    mocker.patch.object(runtime, "check_hooks", side_effect=[trig] + [[]] * 100)
    return rt, sleeps


@pytest.mark.asyncio
async def test_backoff_caps_at_cap_seconds(mocker) -> None:
    from physiclaw.agent.runtime.contract import SessionOutcome

    outcome = SessionOutcome(status="STUCK")
    rt, sleeps = _backoff_rt(mocker, lambda _t: outcome, stop_after=3)
    rt._streak = 50  # streak math must clamp, not overflow into hours

    await rt.start()

    assert max(sleeps) == runtime.BACKOFF_CAP_SECONDS


@pytest.mark.asyncio
async def test_legacy_none_react_stays_flat(mocker) -> None:
    # A fire-and-forget react (returns None) never grows a streak.
    rt, sleeps = _backoff_rt(mocker, lambda _t: None, stop_after=3)

    await rt.start()

    assert rt._streak == 0
    assert max(sleeps) == runtime.CONFIG.engine.react_cooldown_seconds
