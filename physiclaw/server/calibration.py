"""Calibration HTTP routes — register thin wrappers around calibration/handler.py."""

import logging

from physiclaw.bridge import BridgeState, CalibrationState, PageState
from physiclaw.calibration.handler import (
    handle_measure_viewport_shift,
    handle_calibrate_arm,
    handle_calibrate_camera_frame,
    handle_compute_camera_mapping,
    handle_validate_calibration,
    handle_trace_edge,
    handle_show_assistive_touch,
    handle_verify_assistive_touch,
)

log = logging.getLogger(__name__)


def register(
    mcp, physiclaw, bridge: BridgeState, calib: CalibrationState, phone: PageState
):
    """Register the calibration routes."""

    @mcp.custom_route("/api/calibrate/viewport-shift", methods=["POST"])
    async def _viewport_shift(request):
        return await handle_measure_viewport_shift(
            request, physiclaw, calib, bridge, phone
        )

    @mcp.custom_route("/api/calibrate/arm", methods=["POST"])
    async def _arm(request):
        return await handle_calibrate_arm(request, physiclaw, calib, phone)

    @mcp.custom_route("/api/calibrate/camera", methods=["POST"])
    async def _camera(request):
        return await handle_calibrate_camera_frame(request, physiclaw, calib)

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
