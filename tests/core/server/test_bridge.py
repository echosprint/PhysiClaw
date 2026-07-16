"""Tests for `physiclaw.core.server.bridge` — bridge HTTP route registration.

Routes are split across the two planes (see core/server/planes.py):
`register` wires the control-plane routes onto the MCP app (loopback);
`register_phone` wires the phone-facing routes onto the LAN bridge app.
`fake_mcp` and `async_request` fixtures live in `conftest.py` — the same
`FakeMcp` recorder works for both since the registrar interface
(`custom_route`) is shared.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from physiclaw.core.server import bridge as bridge_reg


def test_bridge_register_wires_control_routes_only(fake_mcp) -> None:
    bridge_reg.register(
        fake_mcp,
        bridge=MagicMock(),
        phone=MagicMock(),
    )

    expected = {
        ("/api/bridge/qr", "GET"),
        ("/api/bridge/recent-screenshots", "GET"),
        ("/api/bridge/switch", "POST"),
    }
    actual = {(p, ms[0]) for p, ms, _ in fake_mcp.routes}
    assert expected == actual


def test_bridge_register_phone_wires_lan_routes_only(fake_mcp) -> None:
    bridge_reg.register_phone(
        fake_mcp,
        bridge=MagicMock(),
        calib=MagicMock(),
        phone=MagicMock(),
    )

    expected = {
        ("/bridge", "GET"),
        ("/api/bridge/state", "GET"),
        ("/api/bridge/tapped", "POST"),
        ("/api/bridge/screen-dimension", "POST"),
        ("/api/bridge/screenshot", "POST"),
        ("/api/bridge/clipboard", "GET"),
        ("/api/bridge/touch", "POST"),
    }
    actual = {(p, ms[0]) for p, ms, _ in fake_mcp.routes}
    assert expected == actual


@pytest.mark.asyncio
async def test_bridge_routes_forward_to_handlers(
    fake_mcp,
    async_request,
    mocker,
) -> None:
    """Each route, when called, must delegate to the matching handler
    with the right positional args."""
    spies = {
        "serve_bridge_page": mocker.patch.object(bridge_reg, "serve_bridge_page"),
        "serve_qr_page": mocker.patch.object(bridge_reg, "serve_qr_page"),
        "handle_phone_state": mocker.patch.object(bridge_reg, "handle_phone_state"),
        "handle_clipboard_copied": mocker.patch.object(
            bridge_reg,
            "handle_clipboard_copied",
        ),
        "handle_clipboard_fetch": mocker.patch.object(
            bridge_reg,
            "handle_clipboard_fetch",
        ),
        "handle_mode_switch": mocker.patch.object(bridge_reg, "handle_mode_switch"),
        "handle_screen_dimension": mocker.patch.object(
            bridge_reg,
            "handle_screen_dimension",
        ),
        "handle_screenshot_upload": mocker.patch.object(
            bridge_reg,
            "handle_screenshot_upload",
        ),
        "handle_recent_screenshots": mocker.patch.object(
            bridge_reg,
            "handle_recent_screenshots",
        ),
        "handle_calib_touch": mocker.patch.object(bridge_reg, "handle_calib_touch"),
    }

    async def _ok(*a, **kw):
        return "ok"

    for s in spies.values():
        s.side_effect = _ok

    br, cb, ph = MagicMock(), MagicMock(), MagicMock()
    bridge_reg.register(fake_mcp, br, ph)
    bridge_reg.register_phone(fake_mcp, br, cb, ph)

    req = async_request()

    await fake_mcp.get("/bridge")(req)
    spies["serve_bridge_page"].assert_called_once_with(req)

    await fake_mcp.get("/api/bridge/state")(req)
    spies["handle_phone_state"].assert_called_once_with(req, ph)

    await fake_mcp.get("/api/bridge/qr")(req)
    spies["serve_qr_page"].assert_called_once_with(req)

    await fake_mcp.get("/api/bridge/tapped", "POST")(req)
    spies["handle_clipboard_copied"].assert_called_once_with(req, br)

    await fake_mcp.get("/api/bridge/screen-dimension", "POST")(req)
    spies["handle_screen_dimension"].assert_called_once_with(req, cb)

    await fake_mcp.get("/api/bridge/screenshot", "POST")(req)
    spies["handle_screenshot_upload"].assert_called_once_with(req, br)

    await fake_mcp.get("/api/bridge/recent-screenshots")(req)
    spies["handle_recent_screenshots"].assert_called_once_with(req, br)

    await fake_mcp.get("/api/bridge/clipboard")(req)
    spies["handle_clipboard_fetch"].assert_called_once_with(req, br)

    await fake_mcp.get("/api/bridge/switch", "POST")(req)
    spies["handle_mode_switch"].assert_called_once_with(req, ph)

    await fake_mcp.get("/api/bridge/touch", "POST")(req)
    spies["handle_calib_touch"].assert_called_once_with(req, cb)
