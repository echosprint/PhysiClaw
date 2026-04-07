"""Calibration HTTP routes — register thin wrappers around calibration/handler.py."""

import logging

from physiclaw.bridge import BridgeState, CalibrationState, PhoneState
from physiclaw.calibration.handler import (
    handle_screenshot_transform,
    handle_find_pen_depth,
    handle_check_arm_tilt,
    handle_detect_camera_rotation,
    handle_pick_frame_rotation,
    handle_compute_grbl_mapping,
    handle_compute_camera_mapping,
    handle_validate_calibration,
    handle_trace_edge,
    handle_show_assistive_touch,
    handle_verify_assistive_touch,
)

log = logging.getLogger(__name__)


def register(mcp, physiclaw,
             bridge: BridgeState,
             calib: CalibrationState,
             phone: PhoneState):
    """Register the calibration routes."""

    @mcp.custom_route("/api/calibrate/screenshot-transform", methods=["POST"])
    async def _screenshot_transform(request):
        return await handle_screenshot_transform(request, physiclaw, calib, bridge, phone)

    @mcp.custom_route("/api/calibrate/pen-depth", methods=["POST"])
    async def _pen_depth(request):
        return await handle_find_pen_depth(request, physiclaw, calib, phone)

    @mcp.custom_route("/api/calibrate/arm-tilt", methods=["POST"])
    async def _arm_tilt(request):
        return await handle_check_arm_tilt(request, physiclaw, calib)

    @mcp.custom_route("/api/calibrate/camera-rotation", methods=["POST"])
    async def _camera_rotation(request):
        return await handle_detect_camera_rotation(request, physiclaw)

    @mcp.custom_route("/api/calibrate/frame-rotation", methods=["POST"])
    async def _frame_rotation(request):
        return await handle_pick_frame_rotation(request, physiclaw, calib)

    @mcp.custom_route("/api/calibrate/grbl-mapping", methods=["POST"])
    async def _grbl_mapping(request):
        return await handle_compute_grbl_mapping(request, physiclaw, calib)

    @mcp.custom_route("/api/calibrate/camera-mapping", methods=["POST"])
    async def _camera_mapping(request):
        return await handle_compute_camera_mapping(request, physiclaw, calib)

    @mcp.custom_route("/api/calibrate/validate", methods=["POST"])
    async def _validate(request):
        return await handle_validate_calibration(request, physiclaw, calib, phone)

    @mcp.custom_route("/api/calibrate/trace-edge", methods=["POST"])
    async def _trace_edge(request):
        return await handle_trace_edge(request, physiclaw, phone)

    @mcp.custom_route("/api/calibrate/assistive-touch/show", methods=["POST"])
    async def _at_show(request):
        return await handle_show_assistive_touch(request, physiclaw, calib, phone)

    @mcp.custom_route("/api/calibrate/assistive-touch/verify", methods=["POST"])
    async def _at_verify(request):
        return await handle_verify_assistive_touch(request, physiclaw, calib, bridge)
