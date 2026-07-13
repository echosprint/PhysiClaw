"""HTTP route handlers for the 7-step calibration plan.

Each handler runs the corresponding `calibrate` step in a thread
executor, writes the result into ``physiclaw.calibration`` (a typed
:class:`Calibration` dataclass — the single source of truth), and
returns a JSON response. The Starlette event loop stays responsive
because the blocking step functions run off-thread.
"""

import asyncio
import dataclasses
import logging

from starlette.responses import JSONResponse

from physiclaw.core.bridge import BridgeState, CalibrationState, PageState
from physiclaw.core.bridge.nonce import generate_nonce
from physiclaw.core.calibration.calibrate import (
    TILT_ALIGNED_THRESHOLD,
    calibrate_arm,
    calibrate_camera_frame,
    compute_camera_mapping,
    measure_viewport_shift,
    validate_calibration,
    verify_assistive_touch,
)
from physiclaw.core.calibration.state import Calibration

log = logging.getLogger(__name__)


# ─── Helpers ────────────────────────────────────────────────


async def _run_blocking(do_func):
    """Run a sync callable in the default executor."""
    return await asyncio.get_event_loop().run_in_executor(None, do_func)


def _ok(payload):
    return JSONResponse({"status": "ok", **payload})


async def _read_body(request) -> dict:
    """Parse JSON body, returning {} for empty/malformed input. Used by
    handlers whose body fields are all optional (e.g. `{"fresh": bool}`)."""
    try:
        return await request.json()
    except Exception:
        return {}


def _err(message, status_code=500):
    return JSONResponse(
        {"status": "error", "message": message}, status_code=status_code
    )


async def _run_locked_step(physiclaw, do, *, precheck=None):
    """Run one calibration step off-thread under the hardware lock.

    ``precheck`` runs BEFORE the non-blocking acquire — a raising
    precondition (arm/camera missing, earlier step not run) must beat
    the busy error, so the wizard reports the actionable problem, and
    pre-lock page staging (``phone.set_mode``) keeps today's ordering.
    ``do`` runs with the lock held and the lock is always released,
    even on failure. Any exception becomes the standard ``_err`` JSON.
    """

    def _step():
        if precheck is not None:
            precheck()
        physiclaw.acquire()
        try:
            return do()
        finally:
            physiclaw.release()

    try:
        result = await _run_blocking(_step)
        return _ok(result)
    except Exception as e:
        return _err(str(e))


# ─── Pre-cal: measure viewport shift ────────────────────────


async def handle_measure_viewport_shift(
    request, physiclaw, calib: CalibrationState, bridge: BridgeState, phone: PageState
):
    """POST /api/calibrate/viewport-shift — measure viewport→screenshot offset and DPR.

    Body flag ``{"fresh": true}`` bypasses the disk-cached screenshot
    and waits for a fresh upload; interactive `physiclaw setup hardware`
    sends this so the operator gets a real measurement, not a stale one.
    """
    fresh = bool((await _read_body(request)).get("fresh"))

    def _do():
        phone.set_mode("calibrate", phase="screenshot_cal")
        result = measure_viewport_shift(calib, bridge, fresh=fresh)
        physiclaw.calibration.viewport_shift = result
        return result

    try:
        result = await _run_blocking(_do)
        return _ok(dataclasses.asdict(result))
    except Exception as e:
        return _err(str(e))


# ─── Arm-side unified calibration ───────────────────────────


def _center_parked_stylus(physiclaw):
    """Drive the parked stylus onto the screen center using the saved
    calibration, so auto mode needs no hand-positioning. Assumes the tip
    rests at ``PARK_PCT`` (the off-screen spot every run parks at). Requires
    a prior bundle; raises a clear error if none exists. Caller holds the lock."""
    cal = physiclaw.calibration
    if cal.pct_to_grbl is None:  # plain server boot doesn't load the bundle
        prior = Calibration.load()
        if prior is None or prior.pct_to_grbl is None:
            raise RuntimeError(
                "Auto needs a previous calibration to place the stylus — "
                "position the stylus over the circle and calibrate manually."
            )
        cal.pct_to_grbl = prior.pct_to_grbl
    physiclaw.restore_park_origin()  # re-pin the frame: current pos = PARK_PCT
    gx, gy = cal.pct_to_grbl_mm(0.5, 0.5)
    physiclaw.arm._fast_move(gx, gy)
    physiclaw.arm.wait_idle()
    physiclaw.arm.set_origin()  # screen center is now arm (0, 0) for the probe


async def handle_calibrate_arm(
    request, physiclaw, calib: CalibrationState, phone: PageState
):
    """POST /api/calibrate/arm — screen↔arm mapping.

    Runs the probe triangle + 15-point grid taps (each fires the solenoid;
    re-fires on a miss) and fits the screen↔arm affine. Writes
    ``pct_to_grbl`` and the arm direction mapping into the in-memory bundle.
    Bundle is only persisted to disk on full setup success (validate).

    Body ``{"from_park": true}`` (sent by the wizard's auto mode) means the
    stylus is resting at the off-screen park spot rather than hand-positioned
    over the screen. We use the saved calibration to drive it onto the screen
    center first, so the probe triangle lands on-screen — needs a prior bundle.
    """
    from_park = bool((await _read_body(request)).get("from_park"))

    def _precheck():
        if physiclaw.arm is None:
            raise RuntimeError("Arm not connected")
        phone.set_mode("calibrate", phase="center")

    def _do():
        if from_park:
            _center_parked_stylus(physiclaw)
        pct_to_grbl, tilt, touches = calibrate_arm(physiclaw.arm, calib)
        physiclaw.calibration.pct_to_grbl = pct_to_grbl
        right_vec = (float(pct_to_grbl[0, 0]), float(pct_to_grbl[1, 0]))
        down_vec = (float(pct_to_grbl[0, 1]), float(pct_to_grbl[1, 1]))
        physiclaw.arm.set_direction_mapping(right_vec, down_vec)
        # Park off-phone before returning so step 8's Photo Booth
        # adjustment shows an unobstructed phone (the stylus would
        # otherwise sit at the last grid-tap position over the
        # screen). `physiclaw.park()` is defensive — it works as
        # soon as `pct_to_grbl` is set, which happens above.
        physiclaw.park()
        return {
            "pairs": len(touches) + 3,
            "tilt_ratio": round(tilt, 4),
            "aligned": tilt < TILT_ALIGNED_THRESHOLD,
        }

    return await _run_locked_step(physiclaw, _do, precheck=_precheck)


# ─── Camera frame calibration — setup check + rotation ──────


async def handle_calibrate_camera_frame(request, physiclaw, calib: CalibrationState):
    """POST /api/calibrate/camera — one-frame camera setup + rotation.

    Parks the stylus off the phone's top-left corner (screen pct
    (-0.1, -0.05)) before reading the frame so the arm doesn't occlude
    the screen during shape/coverage/rotation analysis. Then runs the
    physical-setup diagnostic and picks the cv2 rotation code from
    UP/RIGHT markers. Writes ``cam_rotation`` into the calibration
    bundle; diagnostic ``issues`` and measurements are returned for
    the caller to surface to the user.
    """

    def _precheck():
        if physiclaw.cam is None:
            raise RuntimeError("Camera not connected")

    def _do():
        physiclaw.park()
        result = calibrate_camera_frame(physiclaw.cam, calib)
        physiclaw.calibration.cam_rotation = result["rotation"]
        physiclaw.cam.rotation = result["rotation"]
        return result

    return await _run_locked_step(physiclaw, _do, precheck=_precheck)


# ─── Camera mapping: screen → camera affine (Mapping B) ─────


async def handle_compute_camera_mapping(request, physiclaw, calib: CalibrationState):
    """POST /api/calibrate/camera-mapping — compute screen 0-1 → camera 0-1 affine."""

    def _precheck():
        if physiclaw.cam is None:
            raise RuntimeError("Camera not connected")

    def _do():
        rotation = physiclaw.calibration.effective_rotation()
        physiclaw.park()
        pct_to_cam, cam_size = compute_camera_mapping(physiclaw.cam, calib, rotation)
        physiclaw.calibration.pct_to_cam = pct_to_cam
        physiclaw.calibration.cam_size = cam_size
        return {"ok": True, "dots": 15, "cam_size": list(cam_size)}

    return await _run_locked_step(physiclaw, _do, precheck=_precheck)


# ─── Full-chain validation ──────────────────────────────────


async def handle_validate_calibration(
    request, physiclaw, calib: CalibrationState, phone: PageState
):
    """POST /api/calibrate/validate — round-trip validate the calibration chain."""

    def _precheck():
        if physiclaw.arm is None:
            raise RuntimeError("Arm not connected")
        if not physiclaw.calibration.transforms_ready:
            raise RuntimeError("Run arm calibration and camera-mapping first")

    def _do():
        cal_state = physiclaw.calibration
        results = validate_calibration(
            physiclaw.arm,
            physiclaw.cam,
            calib,
            cal_state.effective_rotation(),
            cal_state.pct_to_grbl,
            cal_state.pct_to_cam,
            cam_size=cal_state.cam_size,
        )
        passed = sum(1 for r in results if r["passed"])
        if passed >= 2:
            phone.set_mode("bridge")
            # Capture the phone's current screen_dimension so warm-start
            # doesn't have to wait for a fresh /bridge page load.
            physiclaw.calibration.screen_dimension = calib.screen_dimension
            physiclaw.calibration.save()
        # Park off-phone after the validation taps so step 11's
        # AssistiveTouch interaction shows an unobstructed phone.
        physiclaw.park()
        return {
            "results": results,
            "passed": passed,
            "total": len(results),
            "calibrated": passed >= 2,
        }

    return await _run_locked_step(physiclaw, _do, precheck=_precheck)


# ─── Edge-trace verification ────────────────────────────────


async def handle_trace_edge(request, physiclaw, phone: PageState):
    """POST /api/calibrate/trace-edge — arm traces phone screen border for visual check."""
    from physiclaw.core.calibration.calibrate import trace_screen_edge

    def _precheck():
        if physiclaw.transforms is None:
            raise RuntimeError("Not calibrated — run /setup first")

    def _do():
        trace_screen_edge(physiclaw.arm, physiclaw.transforms)
        phone.set_mode("bridge")
        # Park off-phone after tracing so the rig ends in a clean
        # state — same convention as steps 7, 8, 9, 10.
        physiclaw.park()
        return {"ok": True}

    return await _run_locked_step(physiclaw, _do, precheck=_precheck)


# ─── AssistiveTouch screenshot verification ─────────────────


async def handle_show_assistive_touch(
    request, physiclaw, calib: CalibrationState, phone: PageState
):
    """POST /api/calibrate/assistive-touch/show — display AT positioning circle + grey nonce grid."""

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
    request, physiclaw, calib: CalibrationState, bridge: BridgeState, phone: PageState
):
    """POST /api/calibrate/assistive-touch/verify — tap AT, verify screenshot upload via the grey nonce grid."""

    def _precheck():
        if physiclaw.arm is None:
            raise RuntimeError("Arm not connected")
        if physiclaw.calibration.pct_to_grbl is None:
            raise RuntimeError("Run arm calibration first")
        if not physiclaw.assistive_touch.at_screen:
            raise RuntimeError("Run assistive-touch/show first")

    def _do():
        return verify_assistive_touch(
            physiclaw.arm,
            physiclaw.assistive_touch,
            bridge,
            calib,
            physiclaw.calibration.pct_to_grbl,
            phone,
        )

    return await _run_locked_step(physiclaw, _do, precheck=_precheck)
