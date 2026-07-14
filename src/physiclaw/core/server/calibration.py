"""Calibration HTTP routes — register thin wrappers around calibration/handler.py."""

import logging
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import JSONResponse

from physiclaw.core.bridge import BridgeState, CalibrationState, PageState
from physiclaw.core.calibration.handler import (
    handle_calibrate_arm,
    handle_calibrate_camera_frame,
    handle_compute_camera_mapping,
    handle_measure_viewport_shift,
    handle_show_assistive_touch,
    handle_trace_edge,
    handle_validate_calibration,
    handle_verify_assistive_touch,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from physiclaw.core.orchestration import HardwareRig

log = logging.getLogger(__name__)


def register(
    mcp: "FastMCP",
    rig: "HardwareRig",
    bridge: BridgeState,
    calib: CalibrationState,
    phone: PageState,
) -> None:
    """Register the calibration routes."""

    @mcp.custom_route("/api/calibrate/viewport-shift", methods=["POST"])
    async def _viewport_shift(request: Request) -> JSONResponse:
        return await handle_measure_viewport_shift(request, rig, calib, bridge, phone)

    @mcp.custom_route("/api/calibrate/arm", methods=["POST"])
    async def _arm(request: Request) -> JSONResponse:
        return await handle_calibrate_arm(request, rig, calib, phone)

    @mcp.custom_route("/api/calibrate/camera", methods=["POST"])
    async def _camera(request: Request) -> JSONResponse:
        return await handle_calibrate_camera_frame(request, rig, calib)

    @mcp.custom_route("/api/calibrate/camera-mapping", methods=["POST"])
    async def _camera_mapping(request: Request) -> JSONResponse:
        return await handle_compute_camera_mapping(request, rig, calib)

    @mcp.custom_route("/api/calibrate/validate", methods=["POST"])
    async def _validate(request: Request) -> JSONResponse:
        return await handle_validate_calibration(request, rig, calib, phone)

    @mcp.custom_route("/api/calibrate/trace-edge", methods=["POST"])
    async def _trace_edge(request: Request) -> JSONResponse:
        return await handle_trace_edge(request, rig, phone)

    @mcp.custom_route("/api/calibrate/assistive-touch/show", methods=["POST"])
    async def _at_show(request: Request) -> JSONResponse:
        return await handle_show_assistive_touch(request, rig, calib, phone)

    @mcp.custom_route("/api/calibrate/assistive-touch/verify", methods=["POST"])
    async def _at_verify(request: Request) -> JSONResponse:
        return await handle_verify_assistive_touch(request, rig, calib, bridge, phone)
