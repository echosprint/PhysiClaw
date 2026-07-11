"""Tests for `physiclaw.core.server.watch` — phone watchdog routes.

`fake_mcp` and `async_request` fixtures live in `conftest.py`.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from physiclaw.core.server import watch as watch_reg


pytestmark = [pytest.mark.integration]


def test_watch_register_wires_all_routes(fake_mcp) -> None:
    watch_reg.register(fake_mcp, physiclaw=MagicMock())

    by_path = {(p, ms[0]) for p, ms, _ in fake_mcp.routes}
    assert by_path == {
        ("/api/phone/watch", "GET"),
        ("/api/phone/home", "POST"),
        ("/api/ready", "POST"),
    }


# ---------- /api/phone/watch ----------


@pytest.mark.asyncio
async def test_watch_route_returns_watchdog_result(fake_mcp, async_request) -> None:
    pl = MagicMock()
    pl.watch.return_value = {"wake": True, "reason": "x"}
    watch_reg.register(fake_mcp, pl)

    resp = await fake_mcp.get("/api/phone/watch")(async_request())

    assert json.loads(bytes(resp.body).decode()) == {"wake": True, "reason": "x"}


@pytest.mark.asyncio
async def test_watch_route_runtime_error_returns_no_wake(
    fake_mcp,
    async_request,
) -> None:
    pl = MagicMock()
    pl.watch.side_effect = RuntimeError("not calibrated")
    watch_reg.register(fake_mcp, pl)

    resp = await fake_mcp.get("/api/phone/watch")(async_request())

    assert json.loads(bytes(resp.body).decode()) == {"wake": False, "reason": ""}


@pytest.mark.asyncio
async def test_watch_route_unexpected_exception_returns_503(
    fake_mcp,
    async_request,
) -> None:
    pl = MagicMock()
    pl.watch.side_effect = Exception("kaboom")
    watch_reg.register(fake_mcp, pl)

    resp = await fake_mcp.get("/api/phone/watch")(async_request())

    assert resp.status_code == 503


# ---------- /api/phone/home ----------


@pytest.mark.asyncio
async def test_home_screen_route_dispatches(fake_mcp, async_request) -> None:
    pl = MagicMock()
    watch_reg.register(fake_mcp, pl)

    resp = await fake_mcp.get("/api/phone/home", "POST")(async_request())

    pl.home_screen.assert_called_once()
    assert json.loads(bytes(resp.body).decode()) == {"ok": True}


@pytest.mark.asyncio
async def test_home_screen_route_returns_503_on_failure(
    fake_mcp,
    async_request,
) -> None:
    pl = MagicMock()
    pl.home_screen.side_effect = RuntimeError("arm jammed")
    watch_reg.register(fake_mcp, pl)

    resp = await fake_mcp.get("/api/phone/home", "POST")(async_request())

    assert resp.status_code == 503


# ---------- /api/ready ----------


@pytest.mark.asyncio
async def test_ready_route_marks_and_returns_ready_state(
    fake_mcp,
    async_request,
) -> None:
    pl = MagicMock()
    pl.ready = True
    watch_reg.register(fake_mcp, pl)

    resp = await fake_mcp.get("/api/ready", "POST")(async_request())

    pl.mark_ready.assert_called_once()
    assert json.loads(bytes(resp.body).decode()) == {"ok": True, "ready": True}


@pytest.mark.asyncio
async def test_ready_route_fires_exposure_tune(fake_mcp, async_request) -> None:
    # Fire-and-forget on the default executor: the response must not wait
    # for the tune, but the tune must run.
    import asyncio

    pl = MagicMock()
    pl.ready = True
    watch_reg.register(fake_mcp, pl)

    await fake_mcp.get("/api/ready", "POST")(async_request())

    for _ in range(100):
        if pl.tune_exposure.called:
            break
        await asyncio.sleep(0.01)
    pl.tune_exposure.assert_called_once()
