"""Tests for `physiclaw.core.calibration._common` — the shared tap cycle
(`_tap_once`, `_tap_and_read`) and the canonical `grid_positions`
generator every calibration step rebuilds the grid from."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from physiclaw.core.bridge.calib import CalibrationState
from physiclaw.core.calibration import _common as common_mod
from physiclaw.core.calibration._common import (
    _tap_and_read,
    _tap_once,
    grid_positions,
)

# ---------- grid_positions ----------


def test_grid_positions_yields_15_in_outer_rows_inner_cols_order() -> None:
    cal = CalibrationState()

    out = list(grid_positions(cal))

    assert len(out) == 15
    # Outer iteration is rows; inner is cols.
    assert out[0] == (cal.GRID_COLS_PCT[0], cal.GRID_ROWS_PCT[0])
    assert out[1] == (cal.GRID_COLS_PCT[1], cal.GRID_ROWS_PCT[0])
    assert out[2] == (cal.GRID_COLS_PCT[2], cal.GRID_ROWS_PCT[0])
    assert out[3] == (cal.GRID_COLS_PCT[0], cal.GRID_ROWS_PCT[1])


# ---------- _tap_once ----------


def test_tap_once_strikes_solenoid() -> None:
    arm = MagicMock()

    _tap_once(arm)

    arm.solenoid.tap.assert_called_once_with(common_mod.CAL_STRIKE_DURATION)
    arm.wait_idle.assert_called_once()


# ---------- _tap_and_read ----------


def test_tap_and_read_succeeds_first_attempt(mocker) -> None:
    mocker.patch.object(common_mod.time, "sleep")
    arm = MagicMock()
    cal = MagicMock()
    cal.flush_touches.side_effect = [[], [{"x": 0.5, "y": 0.5}]]

    touch = _tap_and_read(arm, cal, gx=10, gy=10)

    assert touch == {"x": 0.5, "y": 0.5}
    arm.solenoid.tap.assert_called_once()  # fired once, no retry


def test_tap_and_read_refires_on_miss(mocker) -> None:
    mocker.patch.object(common_mod.time, "sleep")
    arm = MagicMock()
    cal = MagicMock()
    cal.flush_touches.side_effect = [
        [],
        [],  # attempt 0: clear + miss
        [],
        [{"x": 1}],  # attempt 1: clear + hit on re-fire
    ]

    touch = _tap_and_read(arm, cal, gx=0, gy=0, max_retries=3)

    assert touch == {"x": 1}
    assert arm.solenoid.tap.call_count == 2  # re-fired once


def test_tap_and_read_returns_none_after_max_retries(mocker) -> None:
    mocker.patch.object(common_mod.time, "sleep")
    arm = MagicMock()
    cal = MagicMock()
    cal.flush_touches.return_value = []  # always miss

    touch = _tap_and_read(arm, cal, gx=0, gy=0, max_retries=2)

    assert touch is None
    assert arm.solenoid.tap.call_count == 3  # initial + 2 retries


def test_grid_positions_yields_col_then_row_with_inner_col_outer_row() -> None:
    cal = SimpleNamespace(
        GRID_ROWS_PCT=[0.1, 0.5, 0.9],
        GRID_COLS_PCT=[0.2, 0.5, 0.8],
    )

    out = list(grid_positions(cal))

    # Outer row=0.1: cols sweep first
    assert out[0] == (0.2, 0.1)
    assert out[1] == (0.5, 0.1)
    assert out[2] == (0.8, 0.1)
    # Then row=0.5
    assert out[3] == (0.2, 0.5)


def test_grid_positions_yields_full_cartesian_product() -> None:
    cal = SimpleNamespace(
        GRID_ROWS_PCT=[0.1, 0.5, 0.9],
        GRID_COLS_PCT=[0.2, 0.5, 0.8],
    )

    assert len(list(grid_positions(cal))) == 9


def test_grid_positions_empty_when_either_axis_is_empty() -> None:
    cal = SimpleNamespace(GRID_ROWS_PCT=[], GRID_COLS_PCT=[0.5])

    assert list(grid_positions(cal)) == []
