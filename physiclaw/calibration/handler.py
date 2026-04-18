"""HTTP route handlers for the 7-step calibration plan.

Each handler runs the corresponding `calibrate` step in a thread
executor, writes the result into ``physiclaw.calibration`` (a typed
:class:`Calibration` dataclass — the single source of truth), and
returns a JSON response. The Starlette event loop stays responsive
because the blocking step functions run off-thread.
"""

import asyncio
import logging

import dataclasses

from starlette.responses import JSONResponse

from physiclaw.bridge import BridgeState, CalibrationState, PageState
from physiclaw.calibration.state import ROTATION_NAMES
from physiclaw.bridge.nonce import generate_nonce
from physiclaw.calibration.calibrate import (
    measure_viewport_shift,
    calibrate_arm,
    detect_camera_rotation,
    pick_frame_rotation,
    compute_camera_mapping,
    validate_calibration,
    verify_assistive_touch,
    load_pen_depth,
    save_pen_depth,
    TILT_ALIGNED_THRESHOLD,
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


# ─── Pre-cal: measure viewport shift ────────────────────────


async def handle_measure_viewport_shift(
    request, physiclaw, calib: CalibrationState, bridge: BridgeState, phone: PageState
):
    """POST /api/calibrate/viewport-shift — measure viewport→screenshot offset and DPR."""

    def _do():
        phone.set_mode("calibrate", phase="screenshot_cal")
        result = measure_viewport_shift(calib, bridge)
        physiclaw.calibration.viewport_shift = result
        return result

    try:
        result = await _run_blocking(_do)
        return _ok(dataclasses.asdict(result))
    except Exception as e:
        return _err(str(e))


# ─── Arm-side unified calibration ───────────────────────────


async def handle_calibrate_arm(
    request, physiclaw, calib: CalibrationState, phone: PageState
):
    """POST /api/calibrate/arm — unified Z depth + screen↔arm mapping.

    Loads the cached ``z_tap`` if present (skipping the Phase A descent),
    runs the probe triangle + 15-point grid taps with z-bump-on-miss, and
    fits the screen↔arm affine. Writes ``z_tap``, ``pct_to_grbl``, and
    the arm direction mapping into the Calibration bundle.
    """

    def _do():
        if physiclaw._arm is None:
            raise RuntimeError("Arm not connected")
        phone.set_mode("calibrate", phase="center")
        cached = load_pen_depth()
        physiclaw.acquire()
        try:
            z_tap, pct_to_grbl, tilt, touches = calibrate_arm(
                physiclaw._arm, calib, z_tap_hint=cached
            )
            if cached is None:
                save_pen_depth(z_tap)
            physiclaw._arm.Z_DOWN = z_tap
            physiclaw.calibration.z_tap = z_tap
            physiclaw.calibration.pct_to_grbl = pct_to_grbl
            right_vec = (float(pct_to_grbl[0, 0]), float(pct_to_grbl[1, 0]))
            down_vec = (float(pct_to_grbl[0, 1]), float(pct_to_grbl[1, 1]))
            physiclaw._arm.set_direction_mapping(right_vec, down_vec)
            return {
                "z_tap": z_tap,
                "pairs": len(touches) + 3,
                "tilt_ratio": round(tilt, 4),
                "aligned": tilt < TILT_ALIGNED_THRESHOLD,
                "z_cached": cached is not None,
            }
        finally:
            physiclaw.release()

    try:
        result = await _run_blocking(_do)
        return _ok(result)
    except Exception as e:
        return _err(str(e))


# ─── Camera rotation diagnostic ─────────────────────────────


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
        physiclaw.calibration.cam_rotation = rotation
        physiclaw._cam.rotation = rotation  # camera consumer reads from here
        return {
            "rotation": rotation,
            "rotation_name": ROTATION_NAMES.get(rotation, str(rotation)),
        }

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
        rotation = physiclaw.calibration.effective_rotation()
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
            physiclaw.calibration.pct_to_cam = pct_to_cam
            physiclaw.calibration.cam_size = cam_size
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
        cal_state = physiclaw.calibration
        if not cal_state.transforms_ready or cal_state.z_tap is None:
            raise RuntimeError("Run arm calibration and camera-mapping first")
        z_tap = cal_state.z_tap
        pct_to_grbl = cal_state.pct_to_grbl
        pct_to_cam = cal_state.pct_to_cam
        cam_size = cal_state.cam_size
        rotation = cal_state.effective_rotation()
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

    if calib.viewport_shift is None:
        return _err("Run viewport-shift first", status_code=400)
    nonce = generate_nonce()
    physiclaw.assistive_touch.compute_at_screen_pos(calib.viewport_shift)
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
        pct_to_grbl = physiclaw.calibration.pct_to_grbl
        if pct_to_grbl is None:
            raise RuntimeError("Run arm calibration first")
        if not physiclaw.assistive_touch.at_screen:
            raise RuntimeError("Run assistive-touch/show first")
        physiclaw.acquire()
        try:
            result = verify_assistive_touch(
                physiclaw._arm,
                physiclaw.assistive_touch,
                bridge,
                calib,
                pct_to_grbl,
            )
            physiclaw._bridge = bridge
            return result
        finally:
            physiclaw.release()

    try:
        result = await _run_blocking(_do)
        return _ok(result)
    except Exception as e:
        return _err(str(e))
