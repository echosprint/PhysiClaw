"""AssistiveTouch verification — tap, double-tap, long-press end-to-end.

The runtime depends on three iOS Shortcuts wired to AssistiveTouch
gestures; this step proves each one actually fires from an arm-driven
press. Single-tap takes an iOS screenshot, double-tap uploads it — the
uploaded image must contain the grey nonce grid the server generated,
so a stale photo from the camera roll can't pass. Long-press makes the
phone fetch the server's queued clipboard text; the fetch event (plus
the operator paste-verifying the random text) proves the clipboard
pipeline. The step re-establishes its own nonce grid on entry and
leaves the phone on the plain bridge page, so it is self-sufficient
and idempotent across re-runs.
"""

import logging
import random
import time

import cv2
import numpy as np

from physiclaw.core.bridge import BridgeState, CalibrationState, PageState
from physiclaw.core.bridge.nonce import NONCE_COUNT, verify_nonce
from physiclaw.core.hardware.arm import StylusArm
from physiclaw.core.hardware.iphone import AssistiveTouch

log = logging.getLogger(__name__)


def _failed() -> dict:
    """The screenshot-sub-step failure record: nothing matched, no clipboard."""
    return {
        "passed": False,
        "screenshot": {"passed": False, "matched": 0, "total": NONCE_COUNT},
        "clipboard": {"fetched": False, "text": None},
    }


def verify_assistive_touch(
    arm: StylusArm,
    at: AssistiveTouch,
    bridge: BridgeState,
    cal: CalibrationState,
    pct_to_grbl: np.ndarray,
    phone: PageState,
) -> dict:
    """Verify all three AT gestures end-to-end.

    1. Single-tap → "PhysiClaw Tap" Shortcut takes a screenshot (saved to Photos).
    2. Wait 5s for the screenshot animation to finish.
    3. Double-tap → "PhysiClaw Screenshot" Shortcut uploads the latest photo.
       Verify the uploaded image contains the grey nonce grid.
    4. Long-press → "PhysiClaw Clipboard" Shortcut GETs /api/bridge/clipboard.
       Verify the server's clipboard-copied event fires within the timeout.
       Return the queued text so the user can paste-verify downstream.

    Requires:
    - assistive-touch/show has run (generated the nonce bits) — this call
      re-displays the grid itself, so the phone may be in any mode on entry
    - User has positioned AT at the orange circle
    - cal.viewport_shift is set (from pre-cal)

    Leaves the phone on the plain bridge page after the clipboard confirmation.

    Returns:
        {
          "passed": bool,  # True iff both sub-checks passed
          "screenshot": {"passed": bool, "matched": int, "total": int},
          "clipboard":  {"fetched": bool, "text": str | None},
        }
    """
    if at.at_screen is None:
        raise RuntimeError("AT position not set — call compute_at_screen_pos first")

    log.info("═══ AssistiveTouch screenshot verification ═══")
    log.info(
        f"  AT position: screen 0-1 ({at.at_screen[0]:.3f}, {at.at_screen[1]:.3f})"
    )

    nonce = cal._screenshot_nonce
    if nonce is None:
        raise RuntimeError("No nonce set — call assistive-touch/show first")

    bridge.clear_screenshot()

    # Show the nonce grid for the screenshot sub-step, whatever mode the
    # page was left in — a prior verify ends in bridge mode after its
    # clipboard confirmation. Re-establishing it here makes this call
    # self-sufficient instead of relying on assistive-touch/show still
    # being on-screen.
    phone.set_mode("calibrate", "assistive_touch", nonce_bits=nonce)
    time.sleep(0.5)  # let the phone poll and render the grid before the shot

    log.info("  Single-tap AT (iOS screenshot)...")
    at.tap(arm, pct_to_grbl)

    log.info("  Waiting 5s for screenshot animation...")
    time.sleep(5.0)

    log.info("  Double-tap AT (screenshot + upload)...")
    at.double_tap(arm, pct_to_grbl)

    log.info("  Waiting for screenshot upload...")
    data = bridge.wait_screenshot(timeout=10.0)
    if data is None:
        log.warning("  Screenshot upload timed out")
        return _failed()

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        log.warning("  Failed to decode screenshot")
        return _failed()

    log.info(f"  Screenshot received: {img.shape[1]}×{img.shape[0]}px")

    t = cal.viewport_shift
    if t is None:
        raise RuntimeError("viewport_shift not set — run measure-viewport-shift first")

    shot_passed, matched = verify_nonce(img, t, nonce, css_x=cal.nonce_x())

    if shot_passed:
        log.info(
            f"  ✓ Screenshot pipeline verified ({matched}/{NONCE_COUNT} bits matched)"
        )
    else:
        log.warning(
            f"  ✗ Screenshot verification failed: {matched}/{NONCE_COUNT} bits matched"
        )

    # ─── Long-press: clipboard fetch verification ──────────────
    # Give iOS a moment to finish the previous Shortcut run before we
    # queue new text and trigger another one.
    time.sleep(5.0)
    clip_text = f"PhysiClaw-{random.randbytes(3).hex().upper()}"
    log.info(f"  Queuing clipboard text: {clip_text!r}")
    # Flip the phone to bridge mode so it shows the text, then the
    # "✓ written to clipboard — paste to verify" confirmation once the
    # Shortcut fetches it. The AT circle / nonce grid were only needed
    # for the screenshot sub-steps; the long-press lands on the iOS AT
    # button regardless of what the web page is showing underneath.
    phone.set_mode("bridge")
    bridge.send_text(clip_text)

    log.info("  Long-press AT (iOS Shortcut → fetch bridge text)...")
    at.long_press(arm, pct_to_grbl)

    log.info("  Waiting for iOS Shortcut to fetch clipboard text...")
    clip_fetched = bridge.wait_clipboard(timeout=10.0)
    if clip_fetched:
        log.info(f"  ✓ Clipboard fetched from server — text: {clip_text!r}")
        log.info("    Phone now shows the copied text — paste anywhere to verify.")
        # Hold the confirmation on-screen so the operator can read it
        # before we clear the text out.
        time.sleep(3.0)
    else:
        log.warning("  ✗ Clipboard fetch timed out — server was not hit")

    # Clear the queued text — the phone drops back to the plain bridge page
    # ("PhysiClaw"). We deliberately stay in bridge mode rather than
    # restoring the calibration grid: a re-run re-establishes its own grid at
    # the screenshot sub-step above, so ending here keeps the success screen
    # clean instead of flashing the grid box back up.
    bridge.clear_text()

    passed = shot_passed and clip_fetched
    if passed:
        log.info("  ✓ AssistiveTouch done: tap + double-tap + long-press all verified")
    else:
        log.warning("  ✗ AssistiveTouch verification failed")

    return {
        "passed": passed,
        "screenshot": {
            "passed": shot_passed,
            "matched": matched,
            "total": NONCE_COUNT,
        },
        "clipboard": {
            "fetched": clip_fetched,
            "text": clip_text,
        },
    }
