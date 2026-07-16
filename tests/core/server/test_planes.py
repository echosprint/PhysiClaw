"""Tests for `physiclaw.core.server.planes` — the two-plane split.

The gate is a plain ASGI wrapper, so tests drive it with hand-built
scopes around a dummy inner app — no sockets, no Starlette routing.
"""

from __future__ import annotations

import pytest

from physiclaw.core.server.planes import ControlGate, PhoneApp, build_bridge_app


async def _inner_ok(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


async def _call(
    app,
    path: str = "/api/x",
    *,
    headers: dict[str, str] | None = None,
    client: tuple[str, int] | None = ("192.168.1.5", 40000),
) -> int:
    sent: dict = {}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        if msg["type"] == "http.response.start":
            sent["status"] = msg["status"]

    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [
            (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
        ],
        "client": client,
        "server": ("127.0.0.1", 8048),
    }
    await app(scope, receive, send)
    return sent["status"]


# ---------- ControlGate (loopback plane, Host allowlist) ----------


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["localhost:8048", "127.0.0.1:8048", "[::1]:8048"])
async def test_control_gate_allows_local_hosts(host: str) -> None:
    gate = ControlGate(_inner_ok)

    assert await _call(gate, headers={"host": host}) == 200


@pytest.mark.asyncio
async def test_control_gate_rejects_foreign_host() -> None:
    # DNS rebinding: the attacker's page reaches 127.0.0.1 but its Host
    # header still says evil.com — the gate must refuse before routing.
    gate = ControlGate(_inner_ok)

    assert await _call(gate, headers={"host": "evil.example:8048"}) == 421


@pytest.mark.asyncio
async def test_control_gate_rejects_missing_host() -> None:
    gate = ControlGate(_inner_ok)

    assert await _call(gate) == 421


@pytest.mark.asyncio
async def test_control_gate_admits_configured_bind_host() -> None:
    gate = ControlGate(_inner_ok, host="10.0.0.9")

    assert await _call(gate, headers={"host": "10.0.0.9:8048"}) == 200


# ---------- PhoneApp ----------


def test_phone_app_refuses_non_phone_route() -> None:
    """A control route (arm-driving, setup, calibrate) registered on the
    LAN plane must fail at assembly time, not silently expose."""
    app = PhoneApp()

    with pytest.raises(RuntimeError, match="refusing to expose"):
        app.custom_route("/api/calibrate/arm", methods=["POST"])


def test_phone_app_collects_routes_and_builds_starlette() -> None:
    app = PhoneApp()

    @app.custom_route("/bridge", methods=["GET"])
    async def _page(request):  # noqa: ARG001
        raise NotImplementedError

    built = app.build()
    assert [r.path for r in built.routes] == ["/bridge"]


def test_build_bridge_app_is_the_plain_phone_app() -> None:
    # No gate on the LAN plane: nothing secret crosses the bridge, and
    # the upload guards live in the handler (see planes.py module doc).
    from starlette.applications import Starlette

    assert isinstance(build_bridge_app(PhoneApp()), Starlette)
