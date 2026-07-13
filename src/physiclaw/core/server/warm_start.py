"""Warm-start: resume from the saved Calibration bundle.

``try_resume`` (called only when ``physiclaw server --warm-start`` is
passed) loads the bundle from disk into ``physiclaw.calibration``,
reconnects hardware, runs an end-to-end sanity tap, and flips the ready
flag only if every test passes. A plain ``physiclaw server`` boot
ignores the bundle entirely — see ``core/server/app.py``.

The resting-position invariant is what makes warm-start work at all. The
arm parks at the off-screen spot (``PARK_PCT``) between every operation
and on clean shutdown, so on reconnect the tip is sitting there — not at
the calibrated origin. ``arm.setup()``'s ``G92 X0 Y0`` would wrongly
declare the park spot to be ``(0, 0)`` and shift the whole frame, so
``restore_park_origin`` re-pins the origin from the park coordinate to
keep ``pct_to_grbl`` valid. The sanity tap is the only mechanism that
catches violations of this invariant (crash mid-move, power cut, arm
bumped — anything that leaves the tip somewhere other than the park spot).
"""

import logging
import sys

from physiclaw.common.config import CONFIG

# Re-exported from core.server.net (its true home) so existing callers —
# `warm_start.wait_for_port`, `from ...warm_start import wait_for_port` —
# keep working. The redundant alias marks it an intentional re-export.
from physiclaw.core.server.net import wait_for_port as wait_for_port

log = logging.getLogger(__name__)

# How long to wait for the phone to (re)load /bridge, in seconds.
BRIDGE_WAIT_TIMEOUT = CONFIG.warm_start.bridge_wait_timeout_seconds
# After a /bridge load is detected, let the page finish rendering before
# we start tapping dots at it.
BRIDGE_SETTLE_SECONDS = CONFIG.warm_start.bridge_settle_seconds


def _sanity(physiclaw, calib, phone) -> bool:
    """Run a compact end-to-end tap verification. Returns True iff every
    tap landed within tolerance. No-touches counts as failure — warm-start
    only succeeds when we've proven the calibration still holds.

    On failure, logs the specific diagnosis (no touches = /bridge not open;
    touches but off = arm/phone/camera moved) so the caller can stay terse.
    """
    from physiclaw.core.calibration.calibrate import validate_calibration

    cal = physiclaw.calibration
    phone.set_mode("calibrate")
    try:
        results = validate_calibration(
            physiclaw.arm,
            physiclaw.cam,
            calib,
            cal.effective_rotation(),
            cal.pct_to_grbl,
            cal.pct_to_cam,
            cam_size=cal.cam_size,
            num_tests=2,
        )
    finally:
        phone.set_mode("bridge")

    total = len(results)
    received = sum(1 for r in results if r.get("error", 999) < 999)
    passed = sum(1 for r in results if r["passed"])
    if passed == total:
        log.info(f"--warm-start: sanity passed ({passed}/{total} taps)")
        return True
    if received == 0:
        log.error(
            "--warm-start: sanity — no taps registered. "
            "Is the phone's /bridge page open and foregrounded?"
        )
    else:
        log.error(
            f"--warm-start: sanity — {passed}/{total} taps within tolerance "
            f"({received}/{total} touches received). Calibration looks stale "
            f"(arm, phone, or camera likely moved since last setup)."
        )
    return False


def try_resume(cam_index_override: int | None) -> bool:
    """Connect hardware, run sanity, flip ready if everything holds.

    Camera index comes from ``--cam-index`` if provided, else from
    ``bundle.cam_index``, else 0.

    Returns True on success; False (with a logged reason) if the bundle
    is incomplete, hardware reconnect fails, or sanity doesn't pass.
    The caller exits non-zero so the user can fall back to plain
    ``uv run physiclaw`` + ``setup.py``.
    """
    from physiclaw.core.calibration.state import Calibration
    from physiclaw.core.server.app import _calib, _phone, physiclaw

    loaded = Calibration.load()
    if loaded is None:
        log.error("--warm-start: no calibration bundle on disk")
        return False
    physiclaw.calibration = loaded
    if loaded.viewport_shift is not None:
        # Mirror into the bridge-side state so calibration handlers that
        # read `calib.viewport_shift` (e.g. show_assistive_touch) see it.
        _calib.viewport_shift = loaded.viewport_shift
        physiclaw.assistive_touch.compute_at_screen_pos(loaded.viewport_shift)
    if loaded.screen_dimension is not None:
        # Restore the CSS-pt dimensions so warm-start's validate can run
        # without waiting for the phone's /bridge page to POST them again.
        _calib.screen_dimension = loaded.screen_dimension
    log.info(
        f"--warm-start: loaded bundle (rotation={loaded.cam_rotation}, "
        f"complete={loaded.complete})"
    )

    cal = physiclaw.calibration
    if not cal.complete:
        log.error("--warm-start: bundle on disk is incomplete")
        return False
    cam_index = (
        cam_index_override
        if cam_index_override is not None
        else (cal.cam_index if cal.cam_index is not None else 0)
    )
    try:
        physiclaw.connect_arm()
        physiclaw.connect_camera(cam_index)
    except Exception as e:
        log.error(f"--warm-start: hardware reconnect failed: {e}")
        return False

    # The camera may have negotiated a different resolution than the
    # bundle was calibrated at (e.g. the [camera] config changed, or the
    # default moved). Same aspect → the 0-1 mapping holds, adopt the live
    # pixel size; different aspect → the mapping is broken, recalibrate.
    live = physiclaw.cam.peek()
    if live is None:
        log.error("--warm-start: camera produced no frame")
        return False
    live_h, live_w = live.shape[:2]
    if not cal.reconcile_cam_size((live_w, live_h)):
        log.error(
            "--warm-start: capture aspect ratio changed since calibration "
            "— run `physiclaw` setup again"
        )
        return False

    # The tip rests at the park spot, not the calibrated origin that
    # arm.setup() just assumed — re-pin the frame from it (see
    # restore_park_origin). The sanity tap below catches the cases where the
    # tip didn't hold that spot (killed mid-move, power yank, arm bumped).
    physiclaw.restore_park_origin()

    # Confirm `locked()` works and the tip is settled off-phone before sanity
    # (already parked post-restore, so the auto-park on exit is a no-op move).
    with physiclaw.locked():
        pass
    if sys.stdin.isatty():
        print()
        print("━" * 60)
        print("Warm-start")
        print("  Open or refresh /bridge on the phone (foreground, not locked).")
        print(f"  Server waits up to {BRIDGE_WAIT_TIMEOUT}s for steady polling.")
        print("━" * 60)
        if not physiclaw._bridge.wait_for_connection(
            BRIDGE_WAIT_TIMEOUT, BRIDGE_SETTLE_SECONDS
        ):
            log.error(
                f"--warm-start: /bridge page not polling within "
                f"{BRIDGE_WAIT_TIMEOUT}s — open or refresh /bridge on the phone."
            )
            return False
    else:
        log.info("--warm-start: non-interactive; running sanity immediately")

    if not _sanity(physiclaw, _calib, _phone):
        # _sanity logged the specific diagnosis.
        return False

    # Match setup.py's final step: send the phone home (swipe from bottom),
    # then flip ready. home_screen's locked() context auto-parks the arm
    # off-screen on exit, so nothing is hovering over the glass afterward.
    physiclaw.home_screen()
    # Home screen showing = the dark scene that exposes AE failure — the
    # right moment to verify/converge exposure. Synchronous is fine:
    # try_resume already runs on a background thread, and tune_exposure
    # is fail-open + bounded by frame-wait timeouts.
    physiclaw.tune_exposure()
    physiclaw.mark_ready()
    log.info(f"--warm-start: resumed from bundle (cam={cam_index}) — MCP tools ready")
    return True
