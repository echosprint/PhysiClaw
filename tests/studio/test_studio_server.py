"""Tests for `physiclaw.studio.server` — the HTTP marshalling layer,
handlers called directly (the `hardware_setup` test convention)."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest

from physiclaw.studio import server


class FakeSession:
    """A StudioSession double: canned act() results or exceptions."""

    def __init__(self, result=None, error=None, busy=False):
        self.busy = busy
        self.app = "taobao"
        self._result = result or {"text": "ok", "image": None, "rows": []}
        self._error = error
        self.acted: list[tuple[str, dict]] = []

    def state(self):
        return {"connected": True, "mcp_url": "u", "app": self.app, "tools": []}

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
async def test_state_reports_without_dialing_and_carries_limits() -> None:
    resp = await server.handle_state(SimpleNamespace(), FakeSession())

    payload = _payload(resp)
    assert payload["connected"] is True and payload["app"] == "taobao"
    # The UI renders these — served, never re-spelled in JS.
    assert payload["limits"]["max_steps"] == 50
    assert 0 < payload["limits"]["min_freq"] < 1


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
        ({"args": {}}, "missing field"),  # tool missing
        ({"tool": 3}, "string"),
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
    resp = await server.handle_act(_request({"tool": "tap"}), FakeSession(busy=True))

    assert resp.status_code == 409
    assert "one rig" in _payload(resp)["message"]


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

    assert {r.path for r in app.routes} == {
        "/",
        "/api/state",
        "/api/act",
        "/api/draft",
        "/api/draft/edit",
        "/api/draft/shot",
        "/api/draft/matrix",
        "/api/draft/commit",
        "/api/macro/save",
        "/api/macro/rehearse",
        "/api/playbook/commit",
        "/api/playbook/rehearse",
        "/api/shot/{shot}",
        "/api/shot/{shot}/view",
    }
    async with app.router.lifespan_context(app):
        assert session.closed is False
    assert session.closed is True


# ---------- draft endpoints (page curation) ----------

LISTING = '1 [text] "首页" [0.100,0.900,0.200,0.950] 0.95'


def _seeded_draft(app: str = "taobao") -> None:
    from physiclaw.studio import draft as ds

    d = ds.load_draft(app)
    ds.add_page(d, "home")
    ds.save_draft(app, d)


@pytest.mark.asyncio
async def test_draft_endpoints_refuse_without_an_app() -> None:
    session = FakeSession()
    session.app = None

    resp = await server.handle_draft(SimpleNamespace(), session)

    assert resp.status_code == 400
    assert "--app" in _payload(resp)["message"]


@pytest.mark.asyncio
async def test_draft_edit_mutates_and_returns_the_fresh_draft() -> None:
    _seeded_draft()

    resp = await server.handle_draft_edit(
        _request({"op": "page_update", "page": "home", "anchors": ["首页"]}),
        FakeSession(),
    )

    payload = _payload(resp)
    assert payload["status"] == "ok"
    assert payload["draft"]["pages"]["home"]["anchors"] == ["首页"]
    assert payload["evidence"]["home"]["首页"]["shots"] == 0


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ({"op": "teleport"}, "unknown draft op"),
        ({"op": "page_add"}, "missing field"),
        ({"op": "page_update", "page": "nope", "anchors": ["x"]}, "no drafted page"),
    ],
)
@pytest.mark.asyncio
async def test_draft_edit_maps_refusals_to_400(body, match) -> None:
    _seeded_draft()

    resp = await server.handle_draft_edit(_request(body), FakeSession())

    assert resp.status_code == 400
    assert match in _payload(resp)["message"]


@pytest.mark.asyncio
async def test_draft_shot_peeks_and_attaches_the_view() -> None:
    from physiclaw.studio import draft as ds

    _seeded_draft()
    session = FakeSession(
        result={
            "text": "",
            "image": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(b"\xff\xd8jj").decode(),
            },
            "rows": [],
            "listing": LISTING,
        }
    )

    resp = await server.handle_draft_shot(_request({"page": "home"}), session)

    payload = _payload(resp)
    assert payload["status"] == "ok" and payload["shot"] == "s1"
    assert session.acted == [("peek", {})]
    stored = ds.load_draft("taobao")
    assert stored["shots"]["s1"] == {"page": "home", "listing": LISTING}


@pytest.mark.asyncio
async def test_draft_shot_refuses_an_unknown_page_before_moving_the_arm() -> None:
    _seeded_draft()
    session = FakeSession()

    resp = await server.handle_draft_shot(_request({"page": "nope"}), session)

    assert resp.status_code == 400
    assert session.acted == []


@pytest.mark.asyncio
async def test_shot_view_404_for_an_unknown_shot() -> None:
    _seeded_draft()

    resp = await server.handle_shot_view(
        SimpleNamespace(path_params={"shot": "s9"}), FakeSession()
    )

    assert resp.status_code == 404


# ---------- recording through /api/act ----------


def _armed_macro(app: str = "taobao") -> None:
    from physiclaw.studio import draft as ds
    from physiclaw.studio import record

    d = ds.load_draft(app)
    record.add_macro(d, "search")
    record.start_recording(d, "search")
    ds.save_draft(app, d)


def _view_result() -> dict:
    return {
        "text": "tap ok",
        "image": {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(b"\xff\xd8jj").decode(),
        },
        "rows": [],
        "listing": LISTING,
    }


@pytest.mark.asyncio
async def test_act_records_a_labeled_tap_while_armed() -> None:
    from physiclaw.studio import draft as ds

    _armed_macro()
    session = FakeSession(result=_view_result())

    resp = await server.handle_act(
        _request(
            {"tool": "tap", "args": {"label": "首页", "bbox": [0.1, 0.9, 0.2, 0.95]}}
        ),
        session,
    )

    payload = _payload(resp)
    assert payload["recorded"]["macro"] == "search"
    assert payload["recorded"]["index"] == 0 and payload["recorded"]["armed"] is True
    # The reply carries the fresh draft — no second round-trip needed.
    assert payload["draft"]["macros"]["search"]["steps"][0]["snap"]
    stored = ds.load_draft("taobao")
    step = stored["macros"]["search"]["steps"][0]
    assert step["with"]["label"] == "首页"
    # One snapshot registry: the macro snap's listing lives there.
    assert stored["shots"][step["snap"]] == {"page": None, "listing": LISTING}
    assert ds.shot_jpeg("taobao", step["snap"]).exists()


@pytest.mark.asyncio
async def test_act_refuses_an_unlabeled_press_before_the_arm_moves() -> None:
    _armed_macro()
    session = FakeSession(result=_view_result())

    resp = await server.handle_act(
        _request({"tool": "tap", "args": {"bbox": [0.1, 0.9, 0.2, 0.95]}}), session
    )

    assert resp.status_code == 400
    assert "label" in _payload(resp)["message"]
    assert session.acted == []  # the gesture never fired


@pytest.mark.asyncio
async def test_act_does_not_record_views_or_when_disarmed() -> None:
    from physiclaw.studio import draft as ds

    _armed_macro()
    session = FakeSession(result=_view_result())

    await server.handle_act(_request({"tool": "peek", "args": {}}), session)
    d = ds.load_draft("taobao")
    d["recording"] = None
    ds.save_draft("taobao", d)
    await server.handle_act(
        _request(
            {"tool": "tap", "args": {"label": "x", "bbox": [0.1, 0.9, 0.2, 0.95]}}
        ),
        session,
    )

    assert ds.load_draft("taobao")["macros"]["search"]["steps"] == []


@pytest.mark.asyncio
async def test_macro_rehearse_returns_the_run_verdict() -> None:
    from types import SimpleNamespace as NS

    _armed_macro()
    from physiclaw.studio import draft as ds
    from physiclaw.studio import record

    d = ds.load_draft("taobao")
    snap = ds.save_snap(d, "aGk=", LISTING)
    record.record_step(d, "tap", {"label": "首页", "bbox": [0.1, 0.9, 0.2, 0.95]}, snap)
    ds.save_draft("taobao", d)

    session = FakeSession()
    session.run_macro = None  # replaced below

    async def run_macro(spec, values, start_at=""):
        assert spec.name == "search" and start_at == ""
        return NS(
            blocks=[{"type": "text", "text": "1/1 ok"}],
            ok=True,
            run_id="macro-run-abc123",
            aborted_step=None,
            reason=None,
        )

    session.run_macro = run_macro

    resp = await server.handle_macro_rehearse(_request({"macro": "search"}), session)

    payload = _payload(resp)
    assert payload["ok"] is True and payload["run_id"] == "macro-run-abc123"
    assert "1/1 ok" in payload["text"]


@pytest.mark.asyncio
async def test_macro_save_writes_and_echoes_the_yaml() -> None:
    from physiclaw.studio import draft as ds
    from physiclaw.studio import record

    _armed_macro()
    d = ds.load_draft("taobao")
    snap = ds.save_snap(d, "aGk=", LISTING)
    record.record_step(d, "tap", {"label": "首页", "bbox": [0.1, 0.9, 0.2, 0.95]}, snap)
    ds.save_draft("taobao", d)

    resp = await server.handle_macro_save(_request({"macro": "search"}), FakeSession())

    payload = _payload(resp)
    assert payload["status"] == "ok"
    assert "name: search" in payload["yaml"]
    assert payload["path"].endswith("taobao/macros/search/MACRO.yml")


# ---------- route assembly endpoints ----------


@pytest.mark.asyncio
async def test_draft_edit_playbook_ops_and_validation_in_reply() -> None:
    resp = await server.handle_draft_edit(
        _request({"op": "playbook_add", "name": "walk"}), FakeSession()
    )
    payload = _payload(resp)
    assert "walk" in payload["draft"]["playbooks"]

    resp = await server.handle_draft_edit(
        _request({"op": "route_insert", "playbook": "walk", "entry": {"page": "home"}}),
        FakeSession(),
    )

    payload = _payload(resp)
    assert payload["draft"]["playbooks"]["walk"]["route"] == [{"page": "home"}]
    # No committed pack on this tmp home — validation says so verbatim.
    assert "commit pages first" in payload["validation"]["walk"]
    assert payload["pack"]["error"] is not None


@pytest.mark.asyncio
async def test_playbook_rehearse_arms_from_disk_and_reports_arm_failures() -> None:
    # No committed pack: arm must refuse with the compiler's message,
    # and the session must never be dialed.
    session = FakeSession()

    resp = await server.handle_playbook_rehearse(
        _request({"playbook": "walk"}), session
    )

    assert resp.status_code == 400
    assert session.acted == []


@pytest.mark.asyncio
async def test_playbook_rehearse_happy_path_streams_lines(monkeypatch) -> None:
    from physiclaw.conductor import rehearsal

    def fake_arm(app, name, values, emit_warn):
        emit_warn("shopdemo/walk is disabled — rehearsing it anyway")
        return "PROGRAM", "REGISTRY"

    monkeypatch.setattr(rehearsal, "arm", fake_arm)
    session = FakeSession()

    async def run_walk(program, registry, emit):
        assert (program, registry) == ("PROGRAM", "REGISTRY")
        emit("  step one")
        return "walk finished or handed over — see the notes above"

    session.run_walk = run_walk

    resp = await server.handle_playbook_rehearse(
        _request({"playbook": "walk"}), session
    )

    payload = _payload(resp)
    assert payload["outcome"].startswith("walk finished")
    assert payload["lines"] == [
        "shopdemo/walk is disabled — rehearsing it anyway",
        "  step one",
    ]
