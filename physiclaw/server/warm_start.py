"""Warm-start: resume from the saved Calibration bundle.

On ``uv run physiclaw --warm-start``, the bundle is already loaded into
``physiclaw.calibration`` at ``physiclaw.server.app`` import time. This
module handles what remains: reconnect hardware, run an end-to-end
sanity tap, and flip the ready flag only if every test passes.

The clean-shutdown invariant is what makes warm-start work at all.
``PhysiClaw.shutdown()`` fast-moves the stylus to ``(0, 0)`` (= screen
center per the bundle's affine) before closing the serial port. On the
next ``connect_arm`` the ``G92 X0 Y0`` in ``arm.setup()`` re-pins the
origin at the same physical spot, keeping ``pct_to_grbl`` valid. The
sanity tap is the only mechanism that catches violations of this
invariant (crash, power cut, arm bumped).
"""

import logging
import sys

log = logging.getLogger(__name__)


def _sanity(physiclaw, calib, phone) -> bool:
    """Run a compact end-to-end tap verification. Returns True iff every
    tap landed within tolerance. No-touches counts as failure — warm-start
    only succeeds when we've proven the calibration still holds.

    On failure, logs the specific diagnosis (no touches = /bridge not open;
    touches but off = arm/phone/camera moved) so the caller can stay terse.
    """
    from physiclaw.calibration.calibrate import validate_calibration

    cal = physiclaw.calibration
    phone.set_mode("calibrate")
    try:
        results = validate_calibration(
            physiclaw.arm,
            physiclaw.cam,
            calib,
            cal.z_tap,
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
    from physiclaw.server.app import physiclaw, _calib, _phone

    cal = physiclaw.calibration
    if not cal.complete:
        log.error("--warm-start: no complete calibration on disk")
        return False
    cam_index = cam_index_override if cam_index_override is not None else (
        cal.cam_index if cal.cam_index is not None else 0
    )
    try:
        physiclaw.connect_arm()
        physiclaw.connect_camera(cam_index)
    except Exception as e:
        log.error(f"--warm-start: hardware reconnect failed: {e}")
        return False

    # Clean shutdown parks the stylus at (0, 0) = screen center, so the
    # fresh setup() on reconnect re-origins there. Warm-start assumes
    # that invariant held; the sanity tap catches cases where it didn't
    # (killed without shutdown, power yank, arm bumped).
    if sys.stdin.isatty():
        print()
        print("━" * 60)
        print("Warm-start")
        print("  Open the phone's /bridge page (keep it foregrounded).")
        print("  The arm will tap 2 random dots to verify the calibration.")
        print("━" * 60)
        input("Press Enter when ready... ")
    else:
        log.info("--warm-start: non-interactive; running sanity immediately")

    if not _sanity(physiclaw, _calib, _phone):
        # _sanity logged the specific diagnosis.
        return False

    # Match setup.py's final step: send the phone home (swipe from bottom),
    # then flip ready. home_screen's locked() context auto-parks the arm
    # off-screen on exit, so nothing is hovering over the glass afterward.
    physiclaw.home_screen()
    physiclaw.mark_ready()
    log.info(
        f"--warm-start: resumed from bundle "
        f"(z_tap={cal.z_tap}mm, cam={cam_index}) — MCP tools ready"
    )
    return True
