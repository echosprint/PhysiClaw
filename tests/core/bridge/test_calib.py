"""Tests for `physiclaw.core.bridge.calib` — calibration page state.

The phase machine drives what the bridge.html page displays. Tests
cover phase transitions, touch event accumulation, viewport-to-
screenshot coordinate conversion (gated on `viewport_shift`), and the
nested `get_state` payload shape per phase.

`AssistiveTouch.AT_*` constants and the `NONCE_*` constants flow
into the assistive_touch payload — both are imported from their
respective modules and asserted via the public state dict.

Accepted equivalent mutmut survivors:

  - 8 instance-attribute annotation / line-format variations in
    `__init__` (`Type | None` ↔ `Type & None`, multi-line ↔ one-line)
    — annotations are not evaluated at runtime, formatting alone
    has no behavioral effect.
  - 3 `log.debug` format-string mutations in `report_touch` — debug
    log content isn't part of the public API contract; asserting on
    log output of a debug-level emit would couple tests to a
    non-contract surface.
"""

from __future__ import annotations

import pytest

from physiclaw.core.bridge.calib import CalibrationState
from physiclaw.core.bridge.nonce import (
    NONCE_CSS_X,
    NONCE_CSS_Y,
    NONCE_DARK,
    NONCE_GRID_COLS,
    NONCE_LIGHT,
    NONCE_SQUARE_SIZE,
)
from physiclaw.core.calibration.transforms import ViewportShift
from physiclaw.core.hardware.iphone import AssistiveTouch


# ---------- class-level constants ----------


def test_grid_cols_pct_pinned() -> None:
    assert CalibrationState.GRID_COLS_PCT == [0.25, 0.50, 0.75]


def test_grid_rows_pct_pinned() -> None:
    assert CalibrationState.GRID_ROWS_PCT == [0.20, 0.40, 0.50, 0.60, 0.80]


def test_phases_set_pinned_to_known_eight_members() -> None:
    assert CalibrationState.PHASES == {
        "idle",
        "screenshot_cal",
        "center",
        "markers",
        "corners",
        "grid",
        "dot",
        "assistive_touch",
    }


# ---------- init ----------


def test_init_starts_in_idle_phase_with_empty_state() -> None:
    cs = CalibrationState()

    assert cs.phase == "idle"
    assert cs.dot_position is None
    assert cs.touches == []
    assert cs._touch_event.is_set() is False
    assert cs.screen_dimension is None
    assert cs.viewport_shift is None
    assert cs._screenshot_nonce is None


# ---------- set_phase ----------


def test_set_phase_valid_phase_updates_state() -> None:
    cs = CalibrationState()

    cs.set_phase("center")

    assert cs.phase == "center"


def test_set_phase_invalid_raises_value_error() -> None:
    cs = CalibrationState()

    with pytest.raises(ValueError, match=r"^Unknown phase: bogus\."):
        cs.set_phase("bogus")


def test_set_phase_stamps_entry_time_only_on_change() -> None:
    # Re-setting the current phase must keep the original entry time —
    # measure_viewport_shift re-sets screenshot_cal after the wizard already
    # switched to it, and the pending-upload cutoff relies on the earlier
    # timestamp.
    cs = CalibrationState()

    cs.set_phase("screenshot_cal")
    since = cs.phase_since
    assert since > 0.0

    cs.set_phase("screenshot_cal")
    assert cs.phase_since == since

    cs.set_phase("center")
    assert cs.phase_since > since


def test_set_phase_clears_dot_touches_and_event() -> None:
    cs = CalibrationState()
    cs.dot_position = (0.1, 0.2)
    cs.touches = [{"x": 0.5, "y": 0.5}]
    cs._touch_event.set()

    cs.set_phase("center")

    assert cs.dot_position is None
    assert cs.touches == []
    assert cs._touch_event.is_set() is False


def test_set_phase_dot_records_position_from_kwargs() -> None:
    cs = CalibrationState()

    cs.set_phase("dot", dot_x=0.3, dot_y=0.7)

    assert cs.dot_position == (0.3, 0.7)


def test_set_phase_dot_uses_default_center_when_kwargs_omitted() -> None:
    cs = CalibrationState()

    cs.set_phase("dot")

    assert cs.dot_position == (0.5, 0.5)


def test_set_phase_assistive_touch_records_nonce_bits() -> None:
    cs = CalibrationState()
    bits = [0, 1, 0, 1] * 5

    cs.set_phase("assistive_touch", nonce_bits=bits)

    assert cs._screenshot_nonce == bits


def test_set_phase_non_dot_non_assistive_does_not_set_dot_or_nonce() -> None:
    cs = CalibrationState()

    cs.set_phase("center", dot_x=0.5, nonce_bits=[1, 0])

    assert cs.dot_position is None
    assert cs._screenshot_nonce is None


# ---------- report_touch / wait_touch / flush_touches ----------


def test_report_touch_appends_and_signals_event() -> None:
    cs = CalibrationState()

    cs.report_touch({"x": 0.5, "y": 0.4})

    assert cs.touches == [{"x": 0.5, "y": 0.4}]
    assert cs._touch_event.is_set() is True


def test_wait_touch_returns_last_event_when_already_signaled() -> None:
    cs = CalibrationState()
    cs.report_touch({"x": 0.5, "y": 0.4})

    out = cs.wait_touch(timeout=0.01)

    assert out == {"x": 0.5, "y": 0.4}


def test_wait_touch_returns_none_on_timeout() -> None:
    cs = CalibrationState()

    assert cs.wait_touch(timeout=0.01) is None


def test_wait_touch_returns_most_recent_when_multiple_events_pending() -> None:
    cs = CalibrationState()
    cs.report_touch({"x": 0.1, "y": 0.1})
    cs.report_touch({"x": 0.2, "y": 0.2})

    out = cs.wait_touch(timeout=0.01)

    assert out == {"x": 0.2, "y": 0.2}


def test_wait_touch_default_timeout_is_10_seconds() -> None:
    import inspect

    sig = inspect.signature(CalibrationState.wait_touch)
    assert sig.parameters["timeout"].default == 10.0


def test_wait_touch_returns_none_when_event_set_but_touches_empty() -> None:
    # Race-condition path: event signaled but flushed before wait runs.
    cs = CalibrationState()
    cs._touch_event.set()  # signal without populating touches

    assert cs.wait_touch(timeout=0.01) is None


def test_flush_touches_returns_drain_and_clears_queue() -> None:
    cs = CalibrationState()
    cs.report_touch({"x": 0.1, "y": 0.1})
    cs.report_touch({"x": 0.2, "y": 0.2})

    out = cs.flush_touches()

    assert out == [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.2}]
    assert cs.touches == []
    assert cs._touch_event.is_set() is False


def test_flush_touches_returns_empty_list_when_no_events() -> None:
    cs = CalibrationState()

    assert cs.flush_touches() == []


# ---------- viewport_to_screenshot_pct ----------


def test_viewport_to_screenshot_pct_raises_when_shift_unset() -> None:
    cs = CalibrationState()

    with pytest.raises(RuntimeError, match=r"^Viewport shift not measured"):
        cs.viewport_to_screenshot_pct(50, 100)


def test_viewport_to_screenshot_pct_delegates_to_viewport_shift() -> None:
    cs = CalibrationState()
    cs.viewport_shift = ViewportShift(
        offset_x=10,
        offset_y=20,
        dpr=2.0,
        screenshot_width=200,
        screenshot_height=400,
    )

    sx, sy = cs.viewport_to_screenshot_pct(50, 100)

    # css_to_pct: (50*2 + 10) / 200 = 0.55, (100*2 + 20) / 400 = 0.55
    assert (sx, sy) == pytest.approx((0.55, 0.55))


# ---------- viewport_pct_to_screenshot_pct ----------


def test_viewport_pct_to_screenshot_pct_raises_when_screen_dimension_unset() -> None:
    cs = CalibrationState()
    cs.viewport_shift = ViewportShift(
        offset_x=0,
        offset_y=0,
        dpr=1.0,
        screenshot_width=100,
        screenshot_height=100,
    )

    with pytest.raises(RuntimeError, match=r"^Screen dimension not set$"):
        cs.viewport_pct_to_screenshot_pct(0.5, 0.5)


def test_viewport_pct_to_screenshot_pct_chains_through_viewport_shift() -> None:
    cs = CalibrationState()
    cs.screen_dimension = {"viewport_width": 200, "viewport_height": 400}
    cs.viewport_shift = ViewportShift(
        offset_x=0,
        offset_y=0,
        dpr=1.0,
        screenshot_width=200,
        screenshot_height=400,
    )

    sx, sy = cs.viewport_pct_to_screenshot_pct(0.5, 0.25)

    # vx*viewport_width = 100, vy*viewport_height = 100. Then css_to_pct
    # at dpr=1, no offsets: (100/200, 100/400) = (0.5, 0.25)
    assert (sx, sy) == pytest.approx((0.5, 0.25))


# ---------- get_state ----------


def test_get_state_idle_returns_phase_screen_dim_and_grid_only() -> None:
    cs = CalibrationState()

    state = cs.get_state()

    assert state == {
        "phase": "idle",
        "screen_dimension": None,
        "grid": {
            "cols": [0.25, 0.50, 0.75],
            "rows": [0.20, 0.40, 0.50, 0.60, 0.80],
        },
    }


def test_get_state_includes_screen_dimension_when_set() -> None:
    cs = CalibrationState()
    cs.screen_dimension = {"width": 1170, "height": 2532}

    state = cs.get_state()

    assert state["screen_dimension"] == {"width": 1170, "height": 2532}


def test_get_state_dot_phase_includes_x_and_y() -> None:
    cs = CalibrationState()
    cs.set_phase("dot", dot_x=0.3, dot_y=0.7)

    state = cs.get_state()

    assert state["phase"] == "dot"
    assert state["dot"] == {"x": 0.3, "y": 0.7}


def test_get_state_assistive_touch_includes_at_and_nonce_blocks() -> None:
    cs = CalibrationState()
    cs.set_phase("assistive_touch", nonce_bits=[1, 0, 1])

    state = cs.get_state()

    assert state["phase"] == "assistive_touch"
    assert state["at"] == {
        "x": AssistiveTouch.AT_CSS_X,
        "y": AssistiveTouch.AT_CSS_Y,
        "r": AssistiveTouch.AT_RADIUS,
    }
    assert state["nonce"] == {
        "colors": [
            [NONCE_LIGHT] * 3,  # bit 1 → light
            [NONCE_DARK] * 3,  # bit 0 → dark
            [NONCE_LIGHT] * 3,  # bit 1 → light
        ],
        "x": NONCE_CSS_X,  # no screen_dimension yet → fallback
        "y": NONCE_CSS_Y,
        "size": NONCE_SQUARE_SIZE,
        "cols": NONCE_GRID_COLS,
    }


def test_nonce_x_falls_back_without_screen_dimension() -> None:
    cs = CalibrationState()

    assert cs.nonce_x() == NONCE_CSS_X


def test_nonce_x_centers_grid_when_viewport_width_known() -> None:
    cs = CalibrationState()
    cs.screen_dimension = {"viewport_width": 390, "viewport_height": 844}

    # 390/2 - (8*15)/2 = 135
    assert cs.nonce_x() == 135


def test_get_state_serves_the_centered_nonce_x_to_the_page() -> None:
    # The page renders whatever x get_state serves, and verify_nonce is
    # handed the same nonce_x() — this pins the served value to it.
    cs = CalibrationState()
    cs.screen_dimension = {"viewport_width": 390, "viewport_height": 844}
    cs.set_phase("assistive_touch", nonce_bits=[1, 0])

    state = cs.get_state()

    assert state["nonce"]["x"] == cs.nonce_x() == 135


def test_get_state_assistive_touch_without_nonce_omits_at_and_nonce_keys() -> None:
    # `set_phase("assistive_touch")` without nonce_bits leaves
    # `_screenshot_nonce = None` — get_state should NOT include the
    # at/nonce blocks (the `is not None` check gates them).
    cs = CalibrationState()
    cs.set_phase("assistive_touch")

    state = cs.get_state()

    assert "at" not in state
    assert "nonce" not in state


def test_get_state_non_assistive_phase_with_nonce_set_still_omits_blocks() -> None:
    # If the nonce field were populated but the phase later changed,
    # the at/nonce blocks must NOT leak into a non-assistive_touch state.
    cs = CalibrationState()
    cs.set_phase("assistive_touch", nonce_bits=[1, 0])
    cs.set_phase("center")  # phase changes; _screenshot_nonce persists

    state = cs.get_state()

    assert "at" not in state
    assert "nonce" not in state
