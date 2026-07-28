"""Tests for `physiclaw.common.ready` — the shared /api/status ready check."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from physiclaw.common import ready

BASE = "http://test.host:8048"


def test_ready_from_status_reads_the_flag() -> None:
    assert ready.ready_from_status({"ready": True}) is True
    assert ready.ready_from_status({"ready": False}) is False
    assert ready.ready_from_status({}) is False


@respx.mock
def test_check_ready_once_reads_the_flag() -> None:
    respx.get(f"{BASE}/api/status").respond(json={"ready": True})

    assert ready.check_ready_once(BASE) is True


@respx.mock
def test_check_ready_once_false() -> None:
    respx.get(f"{BASE}/api/status").respond(json={"ready": False})

    assert ready.check_ready_once(BASE) is False


@respx.mock
def test_check_ready_once_raises_on_4xx() -> None:
    # Blip policy belongs to the caller — the probe must raise, not
    # swallow, so both consumers keep their own hold-and-retry shapes.
    respx.get(f"{BASE}/api/status").respond(404)

    with pytest.raises(httpx.HTTPStatusError):
        ready.check_ready_once(BASE)


@respx.mock
def test_check_ready_async_reads_the_flag() -> None:
    respx.get(f"{BASE}/api/status").respond(json={"ready": True})

    async def _probe() -> bool:
        async with httpx.AsyncClient(base_url=BASE) as client:
            return await ready.check_ready(client)

    assert asyncio.run(_probe()) is True
