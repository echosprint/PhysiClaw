"""Tests for `physiclaw.core.bridge.handler` — Starlette route handlers.

Each handler takes a Starlette `request` plus state objects. Tests
construct minimal fake requests (with `body()` and `json()` async
methods) and use real `BridgeState` / `CalibrationState` / `PageState`
instances.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from physiclaw.core.bridge import handler
from physiclaw.core.bridge.calib import CalibrationState
from physiclaw.core.bridge.handler import (
    handle_calib_touch,
    handle_clipboard_copied,
    handle_clipboard_fetch,
    handle_mode_switch,
    handle_phone_state,
    handle_screen_dimension,
    handle_screenshot_upload,
    serve_bridge_page,
    serve_qr_page,
)
from physiclaw.core.bridge.page import PageState
from physiclaw.core.bridge.state import BridgeState
from physiclaw.core.calibration.transforms import ViewportShift


@pytest.fixture(autouse=True)
def _no_real_save_screenshot(mocker) -> None:
    mocker.patch("physiclaw.core.bridge.state.save_screenshot")


def _async_returning(value: Any):
    async def _coro():
        return value

    return _coro


def _fake_request(
    body_bytes: bytes | None = None, json_obj: Any = None, url_port: int | None = 8048
):
    req = SimpleNamespace()
    req.body = _async_returning(body_bytes if body_bytes is not None else b"")
    req.json = _async_returning(json_obj)
    req.url = SimpleNamespace(port=url_port)
    return req


# ---------- serve_bridge_page ----------


@pytest.mark.asyncio
async def test_serve_bridge_page_reads_html_and_sets_no_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "bridge.html").write_text("<html>bridge</html>")
    monkeypatch.setattr(handler, "STATIC_DIR", static)

    resp = await serve_bridge_page(_fake_request())

    assert resp.body == b"<html>bridge</html>"
    assert resp.headers["cache-control"] == "no-store"


# ---------- handle_phone_state ----------


@pytest.mark.asyncio
async def test_handle_phone_state_calls_poll_and_returns_state(mocker) -> None:
    bridge = BridgeState()
    cal = CalibrationState()
    phone = PageState(bridge, cal)

    resp = await handle_phone_state(_fake_request(), phone)

    import json

    body = json.loads(resp.body)
    assert body["mode"] == "bridge"
    # `bridge.poll` updates last_seen — was 0, now > 0.
    assert bridge.last_seen > 0


# ---------- handle_clipboard_copied ----------


@pytest.mark.asyncio
async def test_handle_clipboard_copied_marks_event() -> None:
    bridge = BridgeState()
    bridge.send_text("hi")

    resp = await handle_clipboard_copied(_fake_request(), bridge)

    import json

    assert json.loads(resp.body) == {"ok": True}
    assert bridge._clipboard_copied.is_set()


# ---------- handle_screen_dimension ----------


@pytest.mark.asyncio
async def test_handle_screen_dimension_records_into_calib() -> None:
    cal = CalibrationState()

    resp = await handle_screen_dimension(
        _fake_request(
            json_obj={
                "screen_width": 1170,
                "screen_height": 2532,
                "viewport_width": 390,
                "viewport_height": 844,
            }
        ),
        cal,
    )

    import json

    assert json.loads(resp.body) == {"ok": True}
    assert cal.screen_dimension == {
        "width": 1170,
        "height": 2532,
        "viewport_width": 390,
        "viewport_height": 844,
    }


@pytest.mark.asyncio
async def test_handle_screen_dimension_defaults_missing_fields_to_zero() -> None:
    cal = CalibrationState()

    await handle_screen_dimension(_fake_request(json_obj={}), cal)

    assert cal.screen_dimension == {
        "width": 0,
        "height": 0,
        "viewport_width": 0,
        "viewport_height": 0,
    }


# ---------- handle_screenshot_upload ----------


@pytest.mark.asyncio
async def test_handle_screenshot_upload_stores_data_and_returns_size() -> None:
    bridge = BridgeState()
    payload = b"fake image bytes"

    resp = await handle_screenshot_upload(_fake_request(body_bytes=payload), bridge)

    import json

    body = json.loads(resp.body)
    assert body == {"ok": True, "size": len(payload)}
    assert bridge._screenshot_data == payload


@pytest.mark.asyncio
async def test_handle_screenshot_upload_400_on_empty_body() -> None:
    bridge = BridgeState()

    resp = await handle_screenshot_upload(_fake_request(body_bytes=b""), bridge)

    assert resp.status_code == 400


# ---------- handle_recent_screenshots ----------


def _fake_get(query: dict[str, str] | None = None):
    req = SimpleNamespace()
    req.query_params = query or {}
    return req


@pytest.mark.asyncio
async def test_handle_recent_screenshots_returns_base64_oldest_to_newest() -> None:
    import base64
    import json

    from physiclaw.core.bridge.handler import handle_recent_screenshots

    bridge = BridgeState()
    for p in (b"a", b"b"):
        bridge.receive_screenshot(p)

    resp = await handle_recent_screenshots(_fake_get(), bridge)

    shots = json.loads(resp.body)["screenshots"]
    assert [base64.b64decode(s) for s in shots] == [b"a", b"b"]


@pytest.mark.asyncio
async def test_handle_recent_screenshots_honours_n_param() -> None:
    import json

    from physiclaw.core.bridge.handler import handle_recent_screenshots

    bridge = BridgeState()
    for p in (b"a", b"b", b"c"):
        bridge.receive_screenshot(p)

    resp = await handle_recent_screenshots(_fake_get({"n": "1"}), bridge)

    assert len(json.loads(resp.body)["screenshots"]) == 1


@pytest.mark.asyncio
async def test_handle_recent_screenshots_defaults_on_bad_n() -> None:
    import json

    from physiclaw.core.bridge.handler import handle_recent_screenshots

    bridge = BridgeState()
    bridge.receive_screenshot(b"a")

    resp = await handle_recent_screenshots(_fake_get({"n": "notint"}), bridge)

    assert len(json.loads(resp.body)["screenshots"]) == 1


# ---------- handle_clipboard_fetch ----------


@pytest.mark.asyncio
async def test_handle_clipboard_fetch_returns_text_when_queued() -> None:
    bridge = BridgeState()
    bridge.send_text("payload")

    resp = await handle_clipboard_fetch(_fake_request(), bridge)

    assert resp.body == b"payload"
    assert bridge._clipboard_copied.is_set()


@pytest.mark.asyncio
async def test_handle_clipboard_fetch_returns_204_when_no_text() -> None:
    bridge = BridgeState()

    resp = await handle_clipboard_fetch(_fake_request(), bridge)

    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_handle_clipboard_fetch_coerces_non_string_value(mocker) -> None:
    bridge = BridgeState()
    bridge._text = 42  # type: ignore[assignment]
    # Pretend it's queued so fetch returns it.

    resp = await handle_clipboard_fetch(_fake_request(), bridge)

    assert resp.body == b"42"


# ---------- handle_mode_switch ----------


@pytest.mark.asyncio
async def test_mode_switch_bridge_returns_ok() -> None:
    bridge = BridgeState()
    cal = CalibrationState()
    phone = PageState(bridge, cal)

    resp = await handle_mode_switch(_fake_request(json_obj={"mode": "bridge"}), phone)

    import json

    body = json.loads(resp.body)
    assert body == {"ok": True, "mode": "bridge"}


@pytest.mark.asyncio
async def test_mode_switch_calibrate_with_phase_passes_kwargs() -> None:
    bridge = BridgeState()
    cal = CalibrationState()
    phone = PageState(bridge, cal)

    resp = await handle_mode_switch(
        _fake_request(
            json_obj={
                "mode": "calibrate",
                "phase": "dot",
                "dot_x": 0.3,
                "dot_y": 0.7,
            }
        ),
        phone,
    )

    import json

    body = json.loads(resp.body)
    assert body == {"ok": True, "mode": "calibrate", "phase": "dot"}
    assert cal.phase == "dot"
    assert cal.dot_position == (0.3, 0.7)


@pytest.mark.asyncio
async def test_mode_switch_400_on_unknown_mode() -> None:
    phone = PageState(BridgeState(), CalibrationState())

    resp = await handle_mode_switch(_fake_request(json_obj={"mode": "weird"}), phone)

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_mode_switch_400_on_calibrate_without_phase() -> None:
    phone = PageState(BridgeState(), CalibrationState())

    resp = await handle_mode_switch(
        _fake_request(json_obj={"mode": "calibrate"}), phone
    )

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_mode_switch_400_on_unknown_phase() -> None:
    phone = PageState(BridgeState(), CalibrationState())

    resp = await handle_mode_switch(
        _fake_request(
            json_obj={
                "mode": "calibrate",
                "phase": "non-existent",
            }
        ),
        phone,
    )

    assert resp.status_code == 400


# ---------- serve_qr_page ----------


@pytest.mark.asyncio
async def test_serve_qr_page_substitutes_phone_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "qr.html").write_text(
        "<html>__PHONE_URL__ / __PHONE_URL_FALLBACK__</html>"
    )
    monkeypatch.setattr(handler, "STATIC_DIR", static)
    mocker.patch.object(
        handler,
        "bridge_base_urls",
        return_value=("http://mac.local:8048", "http://192.168.1.1:8048"),
    )

    resp = await serve_qr_page(_fake_request(url_port=8048))

    body = resp.body.decode()
    assert "http://mac.local:8048/bridge" in body
    assert "http://192.168.1.1:8048/bridge" in body


@pytest.mark.asyncio
async def test_serve_qr_page_uses_default_port_when_url_port_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "qr.html").write_text("<html></html>")
    monkeypatch.setattr(handler, "STATIC_DIR", static)
    base_urls = mocker.patch.object(
        handler,
        "bridge_base_urls",
        return_value=("p", "f"),
    )

    await serve_qr_page(_fake_request(url_port=None))

    base_urls.assert_called_once_with(8048)


# ---------- handle_calib_touch ----------


@pytest.mark.asyncio
async def test_handle_calib_touch_records_touch_with_screenshot_pcts() -> None:
    cal = CalibrationState()
    cal.viewport_shift = ViewportShift(
        offset_x=0,
        offset_y=0,
        dpr=1.0,
        screenshot_width=200,
        screenshot_height=400,
    )

    resp = await handle_calib_touch(
        _fake_request(json_obj={"clientX": 100, "clientY": 200}),
        cal,
    )

    import json

    assert json.loads(resp.body) == {"ok": True}
    assert len(cal.touches) == 1
    touch = cal.touches[0]
    assert touch["x"] == 0.5  # 100/200
    assert touch["y"] == 0.5  # 200/400
