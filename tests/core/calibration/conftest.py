"""Shared fixtures for the calibration step tests.

`make_cal` fakes the bridge-side CalibrationState the steps drive. The
fits and validation hard-require a measured viewport shift, so the
default fake carries an identity shift + identity viewport→screenshot
conversion — the same numerics the old no-shift fallback produced.
Pass ``viewport_shift=None`` to exercise the precondition itself.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from physiclaw.core.bridge.calib import CalibrationState
from physiclaw.core.geometry import ViewportShift


def make_cal(*, viewport_shift="identity") -> MagicMock:
    """A CalibrationState mock with the grid constants surfaced."""
    cal = MagicMock()
    cal.GRID_COLS_PCT = CalibrationState.GRID_COLS_PCT
    cal.GRID_ROWS_PCT = CalibrationState.GRID_ROWS_PCT
    if viewport_shift == "identity":
        cal.viewport_shift = ViewportShift(
            offset_x=0, offset_y=0, dpr=1.0, screenshot_width=1, screenshot_height=1
        )
        cal.viewport_pct_to_screenshot_pct = lambda x, y: (x, y)
    else:
        cal.viewport_shift = viewport_shift
    return cal
