"""Tests for `physiclaw.core.server.hardware` — hardware setup route registration.

Named with the `server_` prefix to avoid a basename collision with
`tests/cli/setup/test_hardware.py`.

`fake_mcp` and `async_request` fixtures live in `conftest.py`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from physiclaw.core.server import hardware as hw_reg


def test_hardware_register_wires_all_routes(fake_mcp) -> None:
    hw_reg.register(fake_mcp, rig=MagicMock(), phone=MagicMock())

    by_path = {(p, ms[0]) for p, ms, _ in fake_mcp.routes}
    assert by_path == {
        ("/setup-hardware", "GET"),
        ("/api/status", "GET"),
        ("/api/connect-arm", "POST"),
        ("/api/connect-camera", "POST"),
        ("/api/disconnect-camera", "POST"),
        ("/api/camera-preview/{index}", "GET"),
    }


@pytest.mark.asyncio
async def test_hardware_routes_forward_to_handlers(
    fake_mcp,
    async_request,
    mocker,
) -> None:
    spies = {
        "handle_setup_page": mocker.patch.object(hw_reg, "handle_setup_page"),
        "handle_status": mocker.patch.object(hw_reg, "handle_status"),
        "handle_connect_arm": mocker.patch.object(hw_reg, "handle_connect_arm"),
        "handle_connect_camera": mocker.patch.object(hw_reg, "handle_connect_camera"),
        "handle_camera_preview": mocker.patch.object(hw_reg, "handle_camera_preview"),
    }

    async def _ok(*a, **kw):
        return "ok"

    for s in spies.values():
        s.side_effect = _ok

    pl, ph = MagicMock(), MagicMock()
    hw_reg.register(fake_mcp, pl, ph)

    req = async_request()
    await fake_mcp.get("/setup-hardware")(req)
    spies["handle_setup_page"].assert_called_once_with(req)

    await fake_mcp.get("/api/status")(req)
    spies["handle_status"].assert_called_once_with(req, pl)

    await fake_mcp.get("/api/connect-arm", "POST")(req)
    spies["handle_connect_arm"].assert_called_once_with(req, pl)

    await fake_mcp.get("/api/connect-camera", "POST")(req)
    spies["handle_connect_camera"].assert_called_once_with(req, pl, ph)

    await fake_mcp.get("/api/camera-preview/{index}")(req)
    spies["handle_camera_preview"].assert_called_once_with(req)
