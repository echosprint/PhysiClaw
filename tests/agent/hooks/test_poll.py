"""Tests for `physiclaw.agent.hooks.poll` — phone watchdog hook.

`httpx.AsyncClient` is mocked so tests don't hit the real server.
"""

from __future__ import annotations

import pytest
import respx

from physiclaw.agent.hooks import poll
from physiclaw.agent.hooks.poll import phone_watch
from physiclaw.agent.runtime.hook import Trigger


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Module-level _client + _in_blip state must reset per test."""
    monkeypatch.setattr(poll, "_client", None)
    monkeypatch.setattr(poll, "_in_blip", False)


@pytest.fixture(autouse=True)
def _stub_server_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHYSICLAW_SERVER", "http://test.host:8048")


@pytest.mark.asyncio
async def test_phone_watch_returns_none_when_no_wake(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.get("http://test.host:8048/api/phone/watch").respond(
        json={"wake": False}
    )

    assert await phone_watch() is None


@pytest.mark.asyncio
async def test_phone_watch_returns_trigger_when_wake_true(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.get("http://test.host:8048/api/phone/watch").respond(
        json={"wake": True, "reason": "screen changed"},
    )

    out = await phone_watch()

    assert isinstance(out, Trigger)
    assert out.description == "screen changed"
    assert out.source == "phone"


@pytest.mark.asyncio
async def test_phone_watch_uses_default_reason_when_missing(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.get("http://test.host:8048/api/phone/watch").respond(json={"wake": True})

    out = await phone_watch()

    assert out.description == "phone screen changed"


@pytest.mark.asyncio
async def test_phone_watch_returns_none_on_transport_error(
    respx_mock: respx.MockRouter,
) -> None:
    import httpx

    respx_mock.get("http://test.host:8048/api/phone/watch").mock(
        side_effect=httpx.ConnectError("boom")
    )

    assert await phone_watch() is None


@pytest.mark.asyncio
async def test_phone_watch_logs_warning_only_on_first_failure(
    respx_mock: respx.MockRouter, caplog: pytest.LogCaptureFixture
) -> None:
    import httpx
    import logging

    respx_mock.get("http://test.host:8048/api/phone/watch").mock(
        side_effect=httpx.ConnectError("boom")
    )

    with caplog.at_level(logging.WARNING, logger="physiclaw.agent.hooks.poll"):
        await phone_watch()
        await phone_watch()
        await phone_watch()

    # Only the first failure logs; subsequent suppressed by _in_blip.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_phone_watch_resets_blip_state_on_recovery(
    respx_mock: respx.MockRouter, caplog: pytest.LogCaptureFixture
) -> None:
    import httpx
    import logging

    respx_mock.get("http://test.host:8048/api/phone/watch").mock(
        side_effect=[
            httpx.ConnectError("first"),
            httpx.Response(200, json={"wake": False}),
            httpx.ConnectError("second"),
        ]
    )

    with caplog.at_level(logging.WARNING, logger="physiclaw.agent.hooks.poll"):
        await phone_watch()
        await phone_watch()
        await phone_watch()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    # Two warnings — one per dropout (recovery resets _in_blip).
    assert len(warnings) == 2


@pytest.mark.asyncio
async def test_phone_watch_returns_none_on_4xx(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.get("http://test.host:8048/api/phone/watch").respond(404)

    assert await phone_watch() is None
