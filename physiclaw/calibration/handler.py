"""HTTP route handlers for the 7-step calibration plan.

Each handler runs the corresponding `calibrate` step in a thread
executor, mutates the orchestrator's intermediate calibration state, and
returns a JSON response. The Starlette event loop stays responsive
because the blocking step functions run off-thread.

The handlers reach into `physiclaw._cal[...]` and friends because that
is where the 7-step plan accumulates intermediate results between calls.
"""

import asyncio
import logging

import cv2
from starlette.responses import JSONResponse

from physiclaw.bridge import BridgeState, CalibrationState, PageState
from physiclaw.bridge.nonce import generate_nonce
from physiclaw.calibration.calibrate import (
    screenshot_transform,
    find_pen_depth,
    check_arm_tilt,
    detect_camera_rotation,
    pick_frame_rotation,
    compute_grbl_mapping,
    compute_camera_mapping,
    validate_calibration,
    verify_assistive_touch,
    load_pen_depth,
    save_pen_depth,
)

log = logging.getLogger(__name__)


# ─── Helpers ────────────────────────────────────────────────


async def _run_blocking(do_func):
    """Run a sync callable in the default executor."""
    return await asyncio.get_event_loop().run_in_executor(None, do_func)


def _ok(payload):
    return JSONResponse({"status": "ok", **payload})


def _err(message, status_code=500):
    return JSONResponse(
        {"status": "error", "message": message}, status_code=status_code
    )


# ─── Pre-cal: viewport → screenshot transform ───────────────


async def handle_screenshot_transform(
    request, physiclaw, calib: CalibrationState, bridge: BridgeState, phone: PageState
):
    """POST /api/calibrate/screenshot-transform — compute viewport→screenshot pixel mapping."""

    def _do():
        phone.set_mode("calibrate", phase="screenshot_cal")
        result = screenshot_transform(calib, bridge)
        physiclaw._cal["screenshot_transform"] = result
        return result

    try:
        result = await _run_blocking(_do)
        return _ok(result)
    except Exception as e:
        return _err(str(e))


# ─── Step 0: pen depth ──────────────────────────────────────


async def handle_find_pen_depth(
    request, physiclaw, calib: CalibrationState, phone: PageState
):
    """POST /api/calibrate/pen-depth — discover Z depth that just touches the screen."""

    def _do():
        if physiclaw._arm is None:
            raise RuntimeError("Arm not connected")
        cached = load_pen_depth()
        if cached is not None:
            z_tap = cached
            log.info(f"Pen depth: using cached value {z_tap}mm")
        else:
            phone.set_mode("calibrate")
            physiclaw.acquire()
            try:
                z_tap = find_pen_depth(physiclaw._arm, calib)
                save_pen_depth(z_tap)
            finally:
                physiclaw.release()
        physiclaw._arm.Z_DOWN = z_tap
        physiclaw._cal["z_tap"] = z_tap
        return {"z_tap": z_tap, "cached": cached is not None}

    try:
        result = await _run_blocking(_do)
        return _ok(result)
    except Exception as e:
        return _err(str(e))


# ─── Step 1: arm tilt check ─────────────────────────────────


async def handle_check_arm_tilt(request, physiclaw, calib: CalibrationState):
    """POST /api/calibrate/arm-tilt — measure arm tilt vs screen plane."""

    def _do():
        if physiclaw._arm is None:
            raise RuntimeError("Arm not connected")
        z_tap = physiclaw._cal.get("z_tap")
        if z_tap is None:
            raise RuntimeError("Run pen-depth first")
        physiclaw.acquire()
        try:
            tilt = check_arm_tilt(physiclaw._arm, calib, z_tap)
            return {"tilt_ratio": round(tilt, 4), "aligned": tilt < 0.02}
        finally:
            physiclaw.release()

    try:
        result = await _run_blocking(_do)
        return _ok(result)
    except Exception as e:
        return _err(str(e))


# ─── Step 2: camera rotation detection ──────────────────────


async def handle_detect_camera_rotation(request, physiclaw):
    """POST /api/calibrate/camera-rotation — detect physical camera rotation from a frame."""

    def _do():
        if physiclaw._cam is None:
            raise RuntimeError("Camera not connected")
        return detect_camera_rotation(physiclaw._cam)

    try:
        result = await _run_blocking(_do)
        return _ok(result)
    except Exception as e:
        return _err(str(e))


# ─── Step 3: software rotation correction ───────────────────


async def handle_pick_frame_rotation(request, physiclaw, calib: CalibrationState):
    """POST /api/calibrate/frame-rotation — choose cv2 rotation to apply to camera frames."""

    def _do():
        if physiclaw._cam is None:
            raise RuntimeError("Camera not connected")
        rotation = pick_frame_rotation(physiclaw._cam, calib)
        physiclaw._cal["rotation"] = rotation
        name = {-1: "none", 0: "90° CW", 1: "180°", 2: "90° CCW"}.get(
            rotation, str(rotation)
        )
        return {"rotation": rotation, "rotation_name": name}

    try:
        result = await _run_blocking(_do)
        return _ok(result)
    except Exception as e:
        return _err(str(e))


# ─── Step 4: screen → GRBL affine (Mapping A) ───────────────


async def handle_compute_grbl_mapping(request, physiclaw, calib: CalibrationState):
    """POST /api/calibrate/grbl-mapping — compute screen 0-1 → GRBL mm affine."""

    def _do():
        if physiclaw._arm is None:
            raise RuntimeError("Arm not connected")
        z_tap = physiclaw._cal.get("z_tap")
        if z_tap is None:
            raise RuntimeError("Run pen-depth first")
        physiclaw.acquire()
        try:
            pct_to_grbl, touches = compute_grbl_mapping(physiclaw._arm, calib, z_tap)
            physiclaw._cal["screen_to_grbl"] = pct_to_grbl
            right_vec = (float(pct_to_grbl[0, 0]), float(pct_to_grbl[1, 0]))
            down_vec = (float(pct_to_grbl[0, 1]), float(pct_to_grbl[1, 1]))
            physiclaw._arm.set_direction_mapping(right_vec, down_vec)
            return {"ok": True, "pairs": len(touches)}
        finally:
            physiclaw.release()

    try:
        result = await _run_blocking(_do)
        return _ok(result)
    except Exception as e:
        return _err(str(e))


# ─── Step 5: screen → camera affine (Mapping B) ─────────────


async def handle_compute_camera_mapping(request, physiclaw, calib: CalibrationState):
    """POST /api/calibrate/camera-mapping — compute screen 0-1 → camera 0-1 affine."""

    def _do():
        if physiclaw._cam is None:
            raise RuntimeError("Camera not connected")
        rotation = physiclaw._cal.get("rotation", cv2.ROTATE_90_COUNTERCLOCKWISE)
        physiclaw.acquire()
        try:
            # Park the arm 80mm off the top edge so it doesn't occlude the dots
            if physiclaw._arm and physiclaw._arm.MOVE_DIRECTIONS:
                ux, uy = physiclaw._arm.MOVE_DIRECTIONS["top"]
                mag = (ux**2 + uy**2) ** 0.5 or 1
                physiclaw._arm._fast_move(ux / mag * 80, uy / mag * 80)
                physiclaw._arm.wait_idle()
            pct_to_cam, cam_size = compute_camera_mapping(
                physiclaw._cam, calib, rotation
            )
            physiclaw._cal["pct_to_cam"] = pct_to_cam
            physiclaw._cal["cam_size"] = cam_size
            return {"ok": True, "dots": 15, "cam_size": list(cam_size)}
        finally:
            physiclaw.release()

    try:
        result = await _run_blocking(_do)
        return _ok(result)
    except Exception as e:
        return _err(str(e))


# ─── Step 6: full-chain validation ──────────────────────────


async def handle_validate_calibration(
    request, physiclaw, calib: CalibrationState, phone: PageState
):
    """POST /api/calibrate/validate — round-trip validate the calibration chain."""

    def _do():
        if physiclaw._arm is None:
            raise RuntimeError("Arm not connected")
        z_tap = physiclaw._cal.get("z_tap")
        rotation = physiclaw._cal.get("rotation", cv2.ROTATE_90_COUNTERCLOCKWISE)
        pct_to_grbl = physiclaw._cal.get("screen_to_grbl")
        pct_to_cam = physiclaw._cal.get("pct_to_cam")
        cam_size = physiclaw._cal.get("cam_size", (1920, 1080))
        if not all([z_tap, pct_to_grbl is not None, pct_to_cam is not None]):
            raise RuntimeError("Run pen-depth, grbl-mapping, and camera-mapping first")
        physiclaw.acquire()
        try:
            results = validate_calibration(
                physiclaw._arm,
                physiclaw._cam,
                calib,
                z_tap,
                rotation,
                pct_to_grbl,
                pct_to_cam,
                cam_size=cam_size,
            )
            passed = sum(1 for r in results if r["passed"])
            if passed >= 2:
                from physiclaw.calibration import ScreenTransforms

                physiclaw._transforms = ScreenTransforms(
                    pct_to_grbl=pct_to_grbl, pct_to_cam=pct_to_cam, cam_size=cam_size
                )
                phone.set_mode("bridge")
            return {
                "results": results,
                "passed": passed,
                "total": len(results),
                "calibrated": passed >= 2,
            }
        finally:
            physiclaw.release()

    try:
        result = await _run_blocking(_do)
        return _ok(result)
    except Exception as e:
        return _err(str(e))


# ─── Edge-trace verification ────────────────────────────────


async def handle_trace_edge(request, physiclaw, phone: PageState):
    """POST /api/calibrate/trace-edge — arm traces phone screen border for visual check."""
    from physiclaw.calibration.calibrate import trace_screen_edge

    def _do():
        if physiclaw.transforms is None:
            raise RuntimeError("Not calibrated — run /setup first")
        physiclaw.acquire()
        try:
            trace_screen_edge(physiclaw.arm, physiclaw.transforms)
            phone.set_mode("bridge")
            return {"ok": True}
        finally:
            physiclaw.release()

    try:
        result = await _run_blocking(_do)
        return _ok(result)
    except Exception as e:
        return _err(str(e))


# ─── Step 7: AssistiveTouch screenshot verification ─────────


async def handle_show_assistive_touch(
    request, physiclaw, calib: CalibrationState, phone: PageState
):
    """POST /api/calibrate/assistive-touch/show — display AT positioning circle + color nonce."""

    if calib.screenshot_transform is None:
        return _err("Run screenshot-transform first", status_code=400)
    nonce = generate_nonce()
    physiclaw.assistive_touch.compute_at_screen_pos(calib.screenshot_transform)
    phone.set_mode("calibrate", phase="assistive_touch", nonce_bits=nonce)
    return JSONResponse(
        {
            "status": "ok",
            "at_screen": list(physiclaw.assistive_touch.at_screen),
            "nonce_count": len(nonce),
        }
    )


async def handle_verify_assistive_touch(
    request, physiclaw, calib: CalibrationState, bridge: BridgeState
):
    """POST /api/calibrate/assistive-touch/verify — tap AT, verify screenshot upload via color nonce."""

    def _do():
        if physiclaw._arm is None:
            raise RuntimeError("Arm not connected")
        pct_to_grbl = physiclaw._cal.get("screen_to_grbl")
        if pct_to_grbl is None:
            raise RuntimeError("Run grbl-mapping first")
        if not physiclaw.assistive_touch.at_screen:
            raise RuntimeError("Run assistive-touch/show first")
        physiclaw.acquire()
        try:
            return verify_assistive_touch(
                physiclaw._arm,
                physiclaw.assistive_touch,
                bridge,
                calib,
                pct_to_grbl,
            )
        finally:
            physiclaw.release()

    try:
        result = await _run_blocking(_do)
        return _ok(result)
    except Exception as e:
        return _err(str(e))
