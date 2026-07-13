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
    arm._fast_move(gx, gy)
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
