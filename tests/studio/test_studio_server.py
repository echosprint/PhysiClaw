"""Tests for `physiclaw.studio.server` — the HTTP marshalling layer,
handlers called directly (the `hardware_setup` test convention)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from physiclaw.studio import server


class FakeSession:
    """A StudioSession double: canned act() results or exceptions."""

    def __init__(self, result=None, error=None, busy=False):
        self.busy = busy
        self._result = result or {"text": "ok", "image": None, "rows": []}
        self._error = error
        self.acted: list[tuple[str, dict]] = []

    def state(self):
        return {"connected": True, "mcp_url": "u", "tools": []}

    async def act(self, tool, args):
        if self._error is not None:
            raise self._error
        self.acted.append((tool, args))
        return self._result


def _request(body: dict | None = None) -> SimpleNamespace:
    async def read_body():
        return json.dumps(body).encode() if body is not None else b"not json"

    return SimpleNamespace(body=read_body)


def _payload(resp) -> dict:
    return json.loads(bytes(resp.body))


@pytest.mark.asyncio
async def test_page_serves_the_studio_app_no_store() -> None:
    resp = await server.handle_page(SimpleNamespace())

    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-store"
    body = bytes(resp.body)
    assert b"PhysiClaw Studio" in body
    assert b"/api/act" in body  # drives the same API throughout


@pytest.mark.asyncio
async def test_state_reports_without_dialing() -> None:
    resp = await server.handle_state(SimpleNamespace(), FakeSession())

    payload = _payload(resp)
    assert payload["status"] == "ok"
    assert payload["connected"] is True and payload["mcp_url"] == "u"
    # The stroke ladder the page maps a drag onto — served, never re-spelled.
    assert list(payload["swipe"]["sizes"]) == ["s", "m", "l", "xl", "xxl"]
    assert payload["swipe"]["speeds"] == ["slow", "medium", "fast"]


@pytest.mark.asyncio
async def test_act_calls_the_tool_with_verbatim_args() -> None:
    session = FakeSession()

    resp = await server.handle_act(
        _request({"tool": "tap", "args": {"bbox": [0.1, 0.2, 0.3, 0.4]}}), session
    )

    assert resp.status_code == 200
    assert _payload(resp)["status"] == "ok"
    assert session.acted == [("tap", {"bbox": [0.1, 0.2, 0.3, 0.4]})]


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (None, "must be JSON"),
        ({"args": {}}, "`tool` is a string"),  # tool missing
        ({"tool": 3}, "`tool` is a string"),
        ({"tool": "tap", "args": [1]}, "`args` a mapping"),
    ],
)
@pytest.mark.asyncio
async def test_act_rejects_malformed_bodies(body, match) -> None:
    resp = await server.handle_act(_request(body), FakeSession())

    assert resp.status_code == 400
    assert match in _payload(resp)["message"]


@pytest.mark.parametrize(
    ("error", "code", "match"),
    [
        (
            ValueError("tool 'calibrate' is not on the published MCP surface"),
            400,
            "published",
        ),
        (ConnectionError("cannot reach the MCP server"), 502, "physiclaw mcp -H"),
        (RuntimeError("tool 'tap' failed: not calibrated"), 500, "not calibrated"),
    ],
)
@pytest.mark.asyncio
async def test_act_maps_session_errors_to_banner_codes(error, code, match) -> None:
    resp = await server.handle_act(_request({"tool": "tap"}), FakeSession(error=error))

    assert resp.status_code == code
    assert match in _payload(resp)["message"]


@pytest.mark.asyncio
async def test_busy_session_answers_409_one_rig() -> None:
    session = FakeSession(busy=True)

    resp = await server.handle_act(_request({"tool": "tap"}), session)

    assert resp.status_code == 409
    assert "one rig" in _payload(resp)["message"]
    assert session.acted == []


@pytest.mark.asyncio
async def test_unrecognized_exceptions_are_bugs_not_refusals() -> None:
    with pytest.raises(TypeError):
        await server.handle_act(
            _request({"tool": "tap"}), FakeSession(error=TypeError("boom"))
        )


@pytest.mark.asyncio
async def test_build_app_routes_and_closes_the_session_on_shutdown() -> None:
    class Closeable(FakeSession):
        def __init__(self):
            super().__init__()
            self.closed = False

        async def close(self):
            self.closed = True

    session = Closeable()
    app = server.build_app(session)

    assert {r.path for r in app.routes} == {"/", "/api/state", "/api/act"}
    async with app.router.lifespan_context(app):
        assert session.closed is False
    assert session.closed is True
