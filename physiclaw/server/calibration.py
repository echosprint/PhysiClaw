"""7-step calibration HTTP routes — register thin wrappers around calibration/routes.py."""

import logging

from physiclaw.bridge import BridgeState, CalibrationState, PhoneState
from physiclaw.calibration.routes import (
    handle_step_screenshot_cal,
    handle_step0,
    handle_step1,
    handle_step2,
    handle_step3,
    handle_step4,
    handle_step5,
    handle_step6,
    handle_verify_edge,
    handle_step7_show,
    handle_step7_tap,
)

log = logging.getLogger(__name__)


def register(mcp, physiclaw,
             bridge: BridgeState,
             calib: CalibrationState,
             phone: PhoneState):
    """Register the 7-step calibration routes."""

    @mcp.custom_route("/api/calibrate/step-screenshot-cal", methods=["POST"])
    async def _step_screenshot_cal(request):
        return await handle_step_screenshot_cal(request, physiclaw, calib, bridge, phone)

    @mcp.custom_route("/api/calibrate/step0-z-depth", methods=["POST"])
    async def _step0(request):
        return await handle_step0(request, physiclaw, calib, phone)

    @mcp.custom_route("/api/calibrate/step1-alignment", methods=["POST"])
    async def _step1(request):
        return await handle_step1(request, physiclaw, calib)

    @mcp.custom_route("/api/calibrate/step2-camera-rotation", methods=["POST"])
    async def _step2(request):
        return await handle_step2(request, physiclaw)

    @mcp.custom_route("/api/calibrate/step3-sw-rotation", methods=["POST"])
    async def _step3(request):
        return await handle_step3(request, physiclaw, calib)

    @mcp.custom_route("/api/calibrate/step4-mapping-a", methods=["POST"])
    async def _step4(request):
        return await handle_step4(request, physiclaw, calib)

    @mcp.custom_route("/api/calibrate/step5-mapping-b", methods=["POST"])
    async def _step5(request):
        return await handle_step5(request, physiclaw, calib)

    @mcp.custom_route("/api/calibrate/step6-validate", methods=["POST"])
    async def _step6(request):
        return await handle_step6(request, physiclaw, calib, phone)

    @mcp.custom_route("/api/calibrate/verify-edge", methods=["POST"])
    async def _verify_edge(request):
        return await handle_verify_edge(request, physiclaw, phone)

    @mcp.custom_route("/api/calibrate/step7-show", methods=["POST"])
    async def _step7_show(request):
        return await handle_step7_show(request, physiclaw, calib, phone)

    @mcp.custom_route("/api/calibrate/step7-tap", methods=["POST"])
    async def _step7_tap(request):
        return await handle_step7_tap(request, physiclaw, calib, bridge)
