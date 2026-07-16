"""HTTP route handlers for the calibration plan.

Each handler runs the corresponding `calibrate` step in a thread
executor, writes the result into ``rig.calibration`` (a typed
:class:`Calibration` dataclass — the single source of truth), and
returns a JSON response. The Starlette event loop stays responsive
because the blocking step functions run off-thread.
"""

import asyncio
import dataclasses
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from physiclaw.core.bridge import BridgeState, CalibrationState, PageState
from physiclaw.core.bridge.handler import json_or_none
from physiclaw.core.bridge.nonce import generate_nonce
from physiclaw.core.calibration._common import (
    PreconditionError,
    require_viewport_shift,
)
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
from physiclaw.core.calibration.transforms import ViewportShift

if TYPE_CHECKING:
    from physiclaw.core.orchestration import HardwareRig

log = logging.getLogger(__name__)


# ─── Helpers ────────────────────────────────────────────────


async def _run_blocking(do_func: Callable[[], Any]) -> Any:
    """Run a sync callable off-thread (the repo's awaited-offload idiom)."""
    return await asyncio.to_thread(do_func)


def _ok(payload: dict) -> JSONResponse:
    return JSONResponse({"status": "ok", **payload})


async def _read_body(request: Request) -> dict:
    """Parse JSON body, returning {} for empty/malformed input. Used by
    handlers whose body fields are ALL optional (e.g. `{"fresh": bool}`)
    — an empty POST is a legitimate "all defaults" request here, unlike
    the bridge routes where a body is required."""
    return (await json_or_none(request)) or {}


def _err(message: str, status_code: int = 500) -> JSONResponse:
    return JSONResponse(
        {"status": "error", "message": message}, status_code=status_code
    )


async def _run_locked_step(
    rig: "HardwareRig",
    do: Callable[[], dict],
    *,
    precheck: Callable[[], None] | None = None,
) -> JSONResponse:
    """Run one calibration step off-thread under the hardware lock.

    ``precheck`` runs BEFORE the non-blocking acquire — a raising
    precondition (arm/camera missing, earlier step not run) must beat
    the busy error, so the wizard reports the actionable problem, and
    pre-lock page staging (``phone.set_mode``) keeps today's ordering.
    ``do`` runs with the lock held and the lock is always released,
    even on failure; a failing step also parks (best-effort) so the
    tip never rests mid-screen where a retry would re-pin the frame.
    A ``PreconditionError`` becomes a 409 (client state — the wizard
    can act on the message); anything else becomes the standard 500
    ``_err`` JSON (a genuine fault).
    """

    def _step() -> dict:
        if precheck is not None:
            precheck()
        rig.acquire()
        try:
            return do()
        except BaseException:
            # Every success path parks inside `do` (the resting-position
            # invariant auto/from_park retries rely on); a mid-step failure
            # must not leave the tip at the failure position — the retry's
            # restore_park_origin would declare that spot the park frame.
            # rig.park() is defensive: it no-ops when the arm or transform
            # isn't there yet, so this is safe at any calibration stage.
            try:
                rig.park()
            except Exception:
                log.exception("calibration step failed — auto-park also failed")
            raise
        finally:
            rig.release()

    try:
        result = await _run_blocking(_step)
        return _ok(result)
    except PreconditionError as e:
        return _err(str(e), status_code=409)
    except Exception as e:
        return _err(str(e))


# ─── Pre-cal: measure viewport shift ────────────────────────


async def handle_measure_viewport_shift(
    request: Request,
    rig: "HardwareRig",
    calib: CalibrationState,
    bridge: BridgeState,
    phone: PageState,
) -> JSONResponse:
    """POST /api/calibrate/viewport-shift — measure viewport→screenshot offset and DPR.

    Body flag ``{"fresh": true}`` bypasses the disk-cached screenshot
    and waits for a fresh upload; interactive `physiclaw setup hardware`
    sends this so the operator gets a real measurement, not a stale one.
    """
    fresh = bool((await _read_body(request)).get("fresh"))

    def _do() -> ViewportShift:
        phone.set_mode("calibrate", phase="screenshot_cal")
        result = measure_viewport_shift(calib, bridge, fresh=fresh)
        rig.calibration.viewport_shift = result
        return result

    try:
        result = await _run_blocking(_do)
        return _ok(dataclasses.asdict(result))
    except Exception as e:
        return _err(str(e))


# ─── Arm-side unified calibration ───────────────────────────


def _center_parked_stylus(rig: "HardwareRig") -> None:
    """Drive the parked stylus onto the screen center using the saved
    calibration, so auto mode needs no hand-positioning. Assumes the tip
    rests at ``PARK_PCT`` (the off-screen spot every run parks at). Requires
    a prior bundle; raises a clear error if none exists. Caller holds the lock."""
    cal = rig.calibration
    if cal.pct_to_grbl is None:  # plain server boot doesn't load the bundle
        prior = Calibration.load()
        if prior is None or prior.pct_to_grbl is None:
            raise RuntimeError(
                "Auto needs a previous calibration to place the stylus — "
                "position the stylus over the circle and calibrate manually."
            )
        cal.pct_to_grbl = prior.pct_to_grbl
    rig.restore_park_origin()  # re-pin the frame: current pos = PARK_PCT
    gx, gy = cal.pct_to_grbl_mm(0.5, 0.5)
    rig.arm.rapid_to(gx, gy)
    rig.arm.wait_idle()
    rig.arm.set_origin()  # screen center is now arm (0, 0) for the probe


async def handle_calibrate_arm(
    request: Request,
    rig: "HardwareRig",
    calib: CalibrationState,
    phone: PageState,
) -> JSONResponse:
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

    def _precheck() -> None:
        if rig.arm is None:
            raise PreconditionError("Arm not connected")
        phone.set_mode("calibrate", phase="center")

    def _do() -> dict:
        if from_park:
            _center_parked_stylus(rig)
        pct_to_grbl, tilt, touches = calibrate_arm(rig.arm, calib)
        rig.calibration.pct_to_grbl = pct_to_grbl
        right_vec = (float(pct_to_grbl[0, 0]), float(pct_to_grbl[1, 0]))
        down_vec = (float(pct_to_grbl[0, 1]), float(pct_to_grbl[1, 1]))
        rig.arm.set_direction_mapping(right_vec, down_vec)
        # Park off-phone before returning so the camera-aim step's
        # preview shows an unobstructed phone (the stylus would
        # otherwise sit at the last grid-tap position over the
        # screen). `rig.park()` is defensive — it works as
        # soon as `pct_to_grbl` is set, which happens above.
        rig.park()
        return {
            "pairs": len(touches) + 3,
            "tilt_ratio": round(tilt, 4),
            "aligned": tilt < TILT_ALIGNED_THRESHOLD,
        }

    return await _run_locked_step(rig, _do, precheck=_precheck)


# ─── Camera frame calibration — setup check + rotation ──────


async def handle_calibrate_camera_frame(
    request: Request, rig: "HardwareRig", calib: CalibrationState
) -> JSONResponse:
    """POST /api/calibrate/camera — one-frame camera setup + rotation.

    Parks the stylus off the phone's top-left corner (screen pct
    (-0.1, -0.05)) before reading the frame so the arm doesn't occlude
    the screen during shape/coverage/rotation analysis. Then runs the
    physical-setup diagnostic and picks the cv2 rotation code from
    UP/RIGHT markers. Writes ``cam_rotation`` into the calibration
    bundle; diagnostic ``issues`` and measurements are returned for
    the caller to surface to the user.
    """

    def _precheck() -> None:
        if rig.cam is None:
            raise PreconditionError("Camera not connected")

    def _do() -> dict:
        rig.park()
        result = calibrate_camera_frame(rig.cam, calib)
        rig.calibration.cam_rotation = result["rotation"]
        rig.cam.rotation = result["rotation"]
        return result

    return await _run_locked_step(rig, _do, precheck=_precheck)


# ─── Camera mapping: screen → camera affine (Mapping B) ─────


async def handle_compute_camera_mapping(
    request: Request, rig: "HardwareRig", calib: CalibrationState
) -> JSONResponse:
    """POST /api/calibrate/camera-mapping — compute screen 0-1 → camera 0-1 affine."""

    def _precheck() -> None:
        if rig.cam is None:
            raise PreconditionError("Camera not connected")

    def _do() -> dict:
        rotation = rig.calibration.effective_rotation()
        rig.park()
        pct_to_cam, cam_size = compute_camera_mapping(rig.cam, calib, rotation)
        rig.calibration.pct_to_cam = pct_to_cam
        rig.calibration.cam_size = cam_size
        return {"ok": True, "dots": 15, "cam_size": list(cam_size)}

    return await _run_locked_step(rig, _do, precheck=_precheck)


# ─── Full-chain validation ──────────────────────────────────


async def handle_validate_calibration(
    request: Request,
    rig: "HardwareRig",
    calib: CalibrationState,
    phone: PageState,
) -> JSONResponse:
    """POST /api/calibrate/validate — round-trip validate the calibration chain."""

    def _precheck() -> None:
        if rig.arm is None:
            raise PreconditionError("Arm not connected")
        if rig.cam is None:
            raise PreconditionError("Camera not connected")
        if not rig.calibration.transforms_ready:
            raise PreconditionError("Run arm calibration and camera-mapping first")

    def _do() -> dict:
        cal_state = rig.calibration
        results = validate_calibration(
            rig.arm,
            rig.cam,
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
            rig.calibration.screen_dimension = calib.screen_dimension
            rig.calibration.save()
        # Park off-phone after the validation taps so the AssistiveTouch
        # verification step shows an unobstructed phone.
        rig.park()
        return {
            "results": results,
            "passed": passed,
            "total": len(results),
            "calibrated": passed >= 2,
        }

    return await _run_locked_step(rig, _do, precheck=_precheck)


# ─── Edge-trace verification ────────────────────────────────


async def handle_trace_edge(
    request: Request, rig: "HardwareRig", phone: PageState
) -> JSONResponse:
    """POST /api/calibrate/trace-edge — arm traces phone screen border for visual check."""
    from physiclaw.core.calibration.calibrate import trace_screen_edge

    def _precheck() -> None:
        if rig.transforms is None:
            raise PreconditionError("Not calibrated — run /setup first")

    def _do() -> dict:
        trace_screen_edge(rig.arm, rig.transforms)
        phone.set_mode("bridge")
        # Park off-phone after tracing so the rig ends in a clean
        # state — same convention as the other arm-driving steps.
        rig.park()
        return {"ok": True}

    return await _run_locked_step(rig, _do, precheck=_precheck)


# ─── AssistiveTouch screenshot verification ─────────────────


async def handle_show_assistive_touch(
    request: Request,
    rig: "HardwareRig",
    calib: CalibrationState,
    phone: PageState,
) -> JSONResponse:
    """POST /api/calibrate/assistive-touch/show — display AT positioning circle + grey nonce grid."""

    try:
        require_viewport_shift(calib)
    except PreconditionError as e:
        return _err(str(e), status_code=409)
    nonce = generate_nonce()
    rig.assistive_touch.compute_at_screen_pos(calib.viewport_shift)
    phone.set_mode("calibrate", phase="assistive_touch", nonce_bits=nonce)
    return JSONResponse(
        {
            "status": "ok",
            "at_screen": list(rig.assistive_touch.at_screen),
            "nonce_count": len(nonce),
        }
    )


async def handle_verify_assistive_touch(
    request: Request,
    rig: "HardwareRig",
    calib: CalibrationState,
    bridge: BridgeState,
    phone: PageState,
) -> JSONResponse:
    """POST /api/calibrate/assistive-touch/verify — tap AT, verify screenshot upload via the grey nonce grid."""

    def _precheck() -> None:
        if rig.arm is None:
            raise PreconditionError("Arm not connected")
        if rig.calibration.pct_to_grbl is None:
            raise PreconditionError("Run arm calibration first")
        if not rig.assistive_touch.at_screen:
            raise PreconditionError("Run assistive-touch/show first")

    def _do() -> dict:
        return verify_assistive_touch(
            rig.arm,
            rig.assistive_touch,
            bridge,
            calib,
            rig.calibration.pct_to_grbl,
            phone,
        )

    return await _run_locked_step(rig, _do, precheck=_precheck)
