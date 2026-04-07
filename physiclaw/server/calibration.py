"""Hardware setup + 7-step calibration HTTP routes."""

import asyncio
import base64
import logging

import cv2

from physiclaw.bridge import BridgeState, CalibrationState, PhoneState
from physiclaw.calibration.plan_calibrate import (
    step_screenshot_cal,
    step0_z_depth, step1_alignment, step2_camera_rotation,
    step3_software_rotation, step4_grbl_screen, step5_camera_screen,
    step6_validate,
    load_pen_depth, save_pen_depth,
)
from physiclaw.core import PhysiClaw

log = logging.getLogger(__name__)


def register(mcp, physiclaw, bridge: BridgeState, calib: CalibrationState, phone: PhoneState):
    """Register hardware setup and calibration routes."""

    # ─── Hardware setup ─────────────────────────────────────

    @mcp.custom_route("/api/status", methods=["GET"])
    async def _status(request):
        from starlette.responses import JSONResponse
        return JSONResponse(physiclaw.status())

    @mcp.custom_route("/api/connect-arm", methods=["POST"])
    async def _connect_arm(request):
        from starlette.responses import JSONResponse

        def _do():
            physiclaw.acquire()
            try:
                physiclaw.connect_arm()
            finally:
                physiclaw.release()

        try:
            await asyncio.get_event_loop().run_in_executor(None, _do)
            return JSONResponse({"status": "ok", "message": "Arm connected"})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)},
                                status_code=500)

    @mcp.custom_route("/api/connect-camera", methods=["POST"])
    async def _connect_camera(request):
        from starlette.responses import JSONResponse
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        index = body.get("index")  # None = auto-detect

        def _do():
            physiclaw.acquire()
            try:
                physiclaw.connect_camera(index)
            finally:
                physiclaw.release()

        try:
            await asyncio.get_event_loop().run_in_executor(None, _do)
            return JSONResponse({"status": "ok",
                                 "message": f"Camera {physiclaw._cam.index} connected",
                                 "index": physiclaw._cam.index})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)},
                                status_code=500)

    @mcp.custom_route("/api/camera-preview/{index}", methods=["GET"])
    async def _camera_preview(request):
        from starlette.responses import JSONResponse
        index = int(request.path_params["index"])
        watermark = request.query_params.get("watermark", "0") == "1"
        try:
            jpeg = await asyncio.get_event_loop().run_in_executor(
                None, PhysiClaw.camera_preview, index, watermark)
            return JSONResponse({"status": "ok", "index": index,
                                 "image": base64.b64encode(jpeg).decode()})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)},
                                status_code=404)

    # ─── Calibration steps ──────────────────────────────────

    @mcp.custom_route("/api/calibrate/step-screenshot-cal", methods=["POST"])
    async def _step_screenshot_cal(request):
        """Pre-calibration: compute viewport→screenshot coordinate transform."""
        from starlette.responses import JSONResponse
        def _do():
            phone.set_mode("calibrate", phase="screenshot_cal")
            result = step_screenshot_cal(calib, bridge)
            physiclaw._cal['screenshot_transform'] = result
            return result
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, _do)
            return JSONResponse({"status": "ok", **result})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    @mcp.custom_route("/api/calibrate/step0-z-depth", methods=["POST"])
    async def _step0(request):
        from starlette.responses import JSONResponse
        def _do():
            if physiclaw._arm is None:
                raise RuntimeError("Arm not connected")
            cached = load_pen_depth()
            if cached is not None:
                z_tap = cached
                log.info(f"Step 0: using cached pen depth {z_tap}mm")
            else:
                phone.set_mode("calibrate")
                physiclaw.acquire()
                try:
                    z_tap = step0_z_depth(physiclaw._arm, calib)
                    save_pen_depth(z_tap)
                finally:
                    physiclaw.release()
            physiclaw._arm.Z_DOWN = z_tap
            physiclaw._cal['z_tap'] = z_tap
            return {"z_tap": z_tap, "cached": cached is not None}
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, _do)
            return JSONResponse({"status": "ok", **result})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    @mcp.custom_route("/api/calibrate/step1-alignment", methods=["POST"])
    async def _step1(request):
        from starlette.responses import JSONResponse
        def _do():
            if physiclaw._arm is None:
                raise RuntimeError("Arm not connected")
            z_tap = physiclaw._cal.get('z_tap')
            if z_tap is None:
                raise RuntimeError("Run step0 first")
            physiclaw.acquire()
            try:
                tilt = step1_alignment(physiclaw._arm, calib, z_tap)
                return {"tilt_ratio": round(tilt, 4), "aligned": tilt < 0.02}
            finally:
                physiclaw.release()
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, _do)
            return JSONResponse({"status": "ok", **result})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    @mcp.custom_route("/api/calibrate/step2-camera-rotation", methods=["POST"])
    async def _step2(request):
        from starlette.responses import JSONResponse
        def _do():
            if physiclaw._cam is None:
                raise RuntimeError("Camera not connected")
            return step2_camera_rotation(physiclaw._cam)
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, _do)
            return JSONResponse({"status": "ok", **result})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    @mcp.custom_route("/api/calibrate/step3-sw-rotation", methods=["POST"])
    async def _step3(request):
        from starlette.responses import JSONResponse
        def _do():
            if physiclaw._cam is None:
                raise RuntimeError("Camera not connected")
            rotation = step3_software_rotation(physiclaw._cam, calib)
            physiclaw._cal['rotation'] = rotation
            name = {-1: "none", 0: "90° CW", 1: "180°", 2: "90° CCW"}.get(rotation, str(rotation))
            return {"rotation": rotation, "rotation_name": name}
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, _do)
            return JSONResponse({"status": "ok", **result})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    @mcp.custom_route("/api/calibrate/step4-mapping-a", methods=["POST"])
    async def _step4(request):
        from starlette.responses import JSONResponse
        def _do():
            if physiclaw._arm is None:
                raise RuntimeError("Arm not connected")
            z_tap = physiclaw._cal.get('z_tap')
            if z_tap is None:
                raise RuntimeError("Run step0 first")
            physiclaw.acquire()
            try:
                pct_to_grbl, touches = step4_grbl_screen(physiclaw._arm, calib, z_tap)
                physiclaw._cal['screen_to_grbl'] = pct_to_grbl
                right_vec = (float(pct_to_grbl[0, 0]), float(pct_to_grbl[1, 0]))
                down_vec = (float(pct_to_grbl[0, 1]), float(pct_to_grbl[1, 1]))
                physiclaw._arm.set_direction_mapping(right_vec, down_vec)
                return {"ok": True, "pairs": len(touches)}
            finally:
                physiclaw.release()
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, _do)
            return JSONResponse({"status": "ok", **result})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    @mcp.custom_route("/api/calibrate/step5-mapping-b", methods=["POST"])
    async def _step5(request):
        from starlette.responses import JSONResponse
        def _do():
            if physiclaw._cam is None:
                raise RuntimeError("Camera not connected")
            rotation = physiclaw._cal.get('rotation', cv2.ROTATE_90_COUNTERCLOCKWISE)
            physiclaw.acquire()
            try:
                if physiclaw._arm and physiclaw._arm.MOVE_DIRECTIONS:
                    ux, uy = physiclaw._arm.MOVE_DIRECTIONS['top']
                    mag = (ux**2 + uy**2)**0.5 or 1
                    physiclaw._arm._fast_move(ux / mag * 80, uy / mag * 80)
                    physiclaw._arm.wait_idle()
                pct_to_cam, cam_size = step5_camera_screen(physiclaw._cam, calib, rotation)
                physiclaw._cal['pct_to_cam'] = pct_to_cam
                physiclaw._cal['cam_size'] = cam_size
                return {"ok": True, "dots": 15, "cam_size": list(cam_size)}
            finally:
                physiclaw.release()
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, _do)
            return JSONResponse({"status": "ok", **result})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    @mcp.custom_route("/api/calibrate/step6-validate", methods=["POST"])
    async def _step6(request):
        from starlette.responses import JSONResponse
        def _do():
            if physiclaw._arm is None:
                raise RuntimeError("Arm not connected")
            z_tap = physiclaw._cal.get('z_tap')
            rotation = physiclaw._cal.get('rotation', cv2.ROTATE_90_COUNTERCLOCKWISE)
            pct_to_grbl = physiclaw._cal.get('screen_to_grbl')
            pct_to_cam = physiclaw._cal.get('pct_to_cam')
            cam_size = physiclaw._cal.get('cam_size', (1920, 1080))
            if not all([z_tap, pct_to_grbl is not None, pct_to_cam is not None]):
                raise RuntimeError("Run steps 0-5 first")
            physiclaw.acquire()
            try:
                results = step6_validate(physiclaw._arm, physiclaw._cam, calib,
                                         z_tap, rotation, pct_to_grbl, pct_to_cam,
                                         cam_size=cam_size)
                passed = sum(1 for r in results if r['passed'])
                if passed >= 2:
                    from physiclaw.calibration import GridCalibration
                    physiclaw._grid_cal = GridCalibration(
                        pct_to_grbl=pct_to_grbl, pct_to_cam=pct_to_cam,
                        cam_size=cam_size)
                    phone.set_mode("bridge")
                return {"results": results, "passed": passed, "total": len(results),
                        "calibrated": passed >= 2}
            finally:
                physiclaw.release()
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, _do)
            return JSONResponse({"status": "ok", **result})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    @mcp.custom_route("/api/calibrate/verify-edge", methods=["POST"])
    async def _verify_edge(request):
        from starlette.responses import JSONResponse

        def _do():
            physiclaw.acquire()
            try:
                result = physiclaw.verify_edge_trace()
                phone.set_mode("bridge")
                return result
            finally:
                physiclaw.release()

        try:
            result = await asyncio.get_event_loop().run_in_executor(None, _do)
            return JSONResponse({"status": "ok", **result})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)},
                                status_code=500)

    @mcp.custom_route("/api/calibrate/step7-show", methods=["POST"])
    async def _step7_show(request):
        """Show AT positioning circle + color nonce barcode on phone."""
        from starlette.responses import JSONResponse
        if calib.screenshot_transform is None:
            return JSONResponse({"status": "error",
                                 "message": "Run pre-cal (step-screenshot-cal) first"},
                                status_code=400)
        nonce = physiclaw._screenshot.generate_nonce()
        physiclaw._screenshot.compute_at_screen_pos(calib.screenshot_transform)
        phone.set_mode("calibrate", phase="assistive_touch", nonce_colors=nonce)
        return JSONResponse({"status": "ok",
                             "at_screen": list(physiclaw._screenshot.at_screen),
                             "nonce_count": len(nonce)})

    @mcp.custom_route("/api/calibrate/step7-tap", methods=["POST"])
    async def _step7_tap(request):
        """Tap + double-tap AT, verify screenshot upload via color nonce."""
        from starlette.responses import JSONResponse
        def _do():
            if physiclaw._arm is None:
                raise RuntimeError("Arm not connected")
            pct_to_grbl = physiclaw._cal.get('screen_to_grbl')
            if pct_to_grbl is None:
                raise RuntimeError("Run steps 0-4 first")
            if not physiclaw._screenshot.at_screen:
                raise RuntimeError("Run step7-show first")
            physiclaw.acquire()
            try:
                return physiclaw._screenshot.setup(
                    physiclaw._arm, bridge, calib,
                    pct_to_grbl)
            finally:
                physiclaw.release()
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, _do)
            return JSONResponse({"status": "ok", **result})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)},
                                status_code=500)
