"""Shared fixtures for the four `core/server/*.py` register-fn tests.

`FakeMcp` records every `@mcp.custom_route(...)` decoration so each
test can verify both the wire layout (path + methods) and the runtime
dispatch (handler invocation with state args).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


class FakeMcp:
    """Records every (path, methods, handler) registered via custom_route."""

    def __init__(self) -> None:
        self.routes: list[tuple[str, tuple[str, ...], Any]] = []

    def custom_route(self, path: str, methods: list[str]):
        def deco(fn):
            self.routes.append((path, tuple(methods), fn))
            return fn

        return deco

    def get(self, path: str, method: str = "GET"):
        for p, ms, fn in self.routes:
            if p == path and method in ms:
                return fn
        raise KeyError(f"no route {method} {path}")


@pytest.fixture
def fake_mcp() -> FakeMcp:
    """Fresh FakeMcp per test — registrations don't bleed across tests."""
    return FakeMcp()


@pytest.fixture
def async_request():
    """Factory: build a Starlette-shaped fake request with optional JSON body."""

    def _make(json_obj: dict | None = None) -> SimpleNamespace:
        async def _json():
            return json_obj or {}

        return SimpleNamespace(
            json=_json,
            path_params={},
            query_params={},
        )

    return _make
