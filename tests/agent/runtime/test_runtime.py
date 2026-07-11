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
async def test_check_ready_true(respx_mock: respx.MockRouter) -> None:
    respx_mock.get("http://test.host:8048/api/status").respond(json={"ready": True})

    assert await _check_ready() is True


@pytest.mark.asyncio
async def test_check_ready_false(respx_mock: respx.MockRouter) -> None:
    respx_mock.get("http://test.host:8048/api/status").respond(json={"ready": False})

    assert await _check_ready() is False


@pytest.mark.asyncio
async def test_check_ready_raises_on_4xx(respx_mock: respx.MockRouter) -> None:
    import httpx

    respx_mock.get("http://test.host:8048/api/status").respond(404)

    with pytest.raises(httpx.HTTPStatusError):
        await _check_ready()


@pytest.mark.asyncio
async def test_check_ready_returns_false_when_ready_field_missing(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.get("http://test.host:8048/api/status").respond(json={})

    assert await _check_ready() is False


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


pytestmark_phase5 = [pytest.mark.integration]


def _stop_after_n(rt, n: int):
    """Build an async sleep stub that calls rt.stop() on the Nth call."""
    counter = {"n": 0}

    async def _sleep(_seconds):
        counter["n"] += 1
        if counter["n"] >= n:
            rt.stop()

    return _sleep, counter


@pytest.mark.asyncio
@pytest.mark.integration
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
@pytest.mark.integration
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
@pytest.mark.integration
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
@pytest.mark.integration
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
@pytest.mark.integration
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
@pytest.mark.integration
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
