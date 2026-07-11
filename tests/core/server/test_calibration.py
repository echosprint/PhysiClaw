"""Tests for `physiclaw.core.server.calibration` — calibration route registration.

`fake_mcp` and `async_request` fixtures live in `conftest.py`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from physiclaw.core.server import calibration as calib_reg


pytestmark = [pytest.mark.integration]


def test_calibration_register_wires_all_routes(fake_mcp) -> None:
    calib_reg.register(
        fake_mcp,
        physiclaw=MagicMock(),
        bridge=MagicMock(),
        calib=MagicMock(),
        phone=MagicMock(),
    )

    expected = {
        "/api/calibrate/viewport-shift",
        "/api/calibrate/arm",
        "/api/calibrate/camera",
        "/api/calibrate/camera-mapping",
        "/api/calibrate/validate",
        "/api/calibrate/trace-edge",
        "/api/calibrate/assistive-touch/show",
        "/api/calibrate/assistive-touch/verify",
    }
    paths = {p for p, _, _ in fake_mcp.routes}
    assert expected == paths
    assert all(ms == ("POST",) for _, ms, _ in fake_mcp.routes)


@pytest.mark.asyncio
async def test_calibration_routes_forward_to_handlers(
    fake_mcp,
    async_request,
    mocker,
) -> None:
    handlers = [
        "handle_measure_viewport_shift",
        "handle_calibrate_arm",
        "handle_calibrate_camera_frame",
        "handle_compute_camera_mapping",
        "handle_validate_calibration",
        "handle_trace_edge",
        "handle_show_assistive_touch",
        "handle_verify_assistive_touch",
    ]
    spies = {h: mocker.patch.object(calib_reg, h) for h in handlers}

    async def _ok(*a, **kw):
        return "ok"

    for s in spies.values():
        s.side_effect = _ok

    pl, br, cb, ph = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    calib_reg.register(fake_mcp, pl, br, cb, ph)

    req = async_request()
    routes_to_handlers = {
        "/api/calibrate/viewport-shift": "handle_measure_viewport_shift",
        "/api/calibrate/arm": "handle_calibrate_arm",
        "/api/calibrate/camera": "handle_calibrate_camera_frame",
        "/api/calibrate/camera-mapping": "handle_compute_camera_mapping",
        "/api/calibrate/validate": "handle_validate_calibration",
        "/api/calibrate/trace-edge": "handle_trace_edge",
        "/api/calibrate/assistive-touch/show": "handle_show_assistive_touch",
        "/api/calibrate/assistive-touch/verify": "handle_verify_assistive_touch",
    }
    for path, hname in routes_to_handlers.items():
        await fake_mcp.get(path, "POST")(req)
        assert spies[hname].called, f"{hname} not invoked from {path}"
