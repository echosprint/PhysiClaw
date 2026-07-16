"""Shared tap plumbing for the calibration steps.

Every hardware calibration step drives the same primitive: move the
arm, fire the solenoid, read back the touch the phone reported. The
grid geometry is equally shared — arm calibration taps the 15 points,
camera mapping detects red dots at the same 15 points, and both must
iterate them in the identical canonical order or the point pairs
mis-correspond. Keeping the tap cycle and the grid generator in one
leaf module means the steps can't drift apart.

No green flash. Touch events for contact detection; each tap fires the
solenoid — there is no Z depth to find or bump.
"""

import logging
import time
from collections.abc import Iterator

from physiclaw.core.bridge import CalibrationState
from physiclaw.core.hardware.arm import StylusArm

log = logging.getLogger(__name__)


class PreconditionError(RuntimeError):
    """A calibration step can't run yet — hardware missing or an earlier
    step not done. A client-state problem (the HTTP layer maps it to
    409), not a server fault (500): the wizard shows the message as the
    actionable fix."""


def require_viewport_shift(cal) -> None:
    """Hard precondition shared by the affine fits and validation:
    without the shift, a fit would silently land in raw viewport 0-1
    instead of the canonical screenshot 0-1 space — numerically valid
    and wrong."""
    if cal.viewport_shift is None:
        raise PreconditionError(
            "Viewport shift not measured — run the pre-cal step first"
        )


def require_screen_dimension(cal) -> None:
    """Hard precondition for the steps that convert grid positions via
    ``viewport_pct_to_screenshot_pct`` (the fits and validation) — its
    mid-step RuntimeError('Screen dimension not set') would surface as a
    generic 500; checking up front turns it into the actionable 409 the
    sibling preconditions get. Steps that only need the shift (e.g. the
    AT positioning) deliberately don't require this."""
    if cal.screen_dimension is None:
        raise PreconditionError(
            "Screen dimension not received — open /bridge on the phone first"
        )


# Slightly longer than a normal tap so flaky first contacts still register
# during calibration probing.
CAL_STRIKE_DURATION = 0.15


def grid_positions(cal: "CalibrationState") -> Iterator[tuple[float, float]]:
    """Yield (col_pct, row_pct) for each of the 15 grid positions in
    canonical outer-rows / inner-cols order. Used by arm calibration,
    camera mapping, and any downstream code that rebuilds the same grid."""
    for row in cal.GRID_ROWS_PCT:
        for col in cal.GRID_COLS_PCT:
            yield col, row


def _tap_once(arm: StylusArm) -> None:
    """Single calibration tap: fire the solenoid for CAL_STRIKE_DURATION."""
    arm.solenoid.tap(CAL_STRIKE_DURATION)
    arm.wait_idle()


def _tap_and_read(
    arm: StylusArm,
    cal: CalibrationState,
    gx: float,
    gy: float,
    max_retries: int = 3,
) -> dict | None:
    """Move to (gx, gy), tap, return the touch dict (or None on failure).

    On a miss (no touch registered) the solenoid is simply re-fired — the
    stroke is fixed, so there's no depth to deepen; misses are just flaky
    contact or a brief unresponsive screen.
    """
    arm.rapid_to(gx, gy)
    arm.wait_idle()
    for attempt in range(max_retries + 1):
        cal.flush_touches()
        _tap_once(arm)
        time.sleep(0.3)
        got = cal.flush_touches()
        if got:
            return got[-1]
        if attempt < max_retries:
            log.warning(
                f"    tap at arm ({gx:.1f}, {gy:.1f})mm: missed, "
                f"retry {attempt + 1}/{max_retries}"
            )
    log.warning(
        f"    tap at arm ({gx:.1f}, {gy:.1f})mm: FAILED after {max_retries} retries"
    )
    return None
