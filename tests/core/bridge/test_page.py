"""Tests for `physiclaw.core.bridge.page` — phone-page mode coordinator.

PageState wraps a BridgeState + CalibrationState pair behind a lock.
Tests use mock objects for both deps so we can assert delegation
without exercising the (still untested) state classes.
"""

from __future__ import annotations

import logging
import threading

import pytest

from physiclaw.core.bridge.page import PageState


def _bridge_stub(text: str = "default text", copied: bool = False) -> object:
    obj = type("BridgeStub", (), {})()
    obj.current_text = lambda: text
    obj.is_copied = lambda: copied
    obj.arm_upload_calls: list[float] = []
    obj.arm_upload = lambda seconds: obj.arm_upload_calls.append(seconds)
    return obj


def _cal_stub(
    *,
    screen_dimension: object | None = None,
    state: dict | None = None,
) -> object:
    obj = type("CalStub", (), {})()
    obj.screen_dimension = screen_dimension
    obj.get_state = lambda: state or {}
    obj.set_phase_calls: list[tuple[str, dict]] = []

    def set_phase(phase: str, **kwargs: object) -> None:
        obj.set_phase_calls.append((phase, kwargs))

    obj.set_phase = set_phase
    return obj


# ---------- init ----------


def test_pagestate_initializes_to_bridge_mode_with_a_lock() -> None:
    p = PageState(_bridge_stub(), _cal_stub())

    assert p.mode == "bridge"
    assert isinstance(p.lock, type(threading.Lock()))


def test_pagestate_holds_supplied_bridge_and_cal_references() -> None:
    bridge = _bridge_stub()
    cal = _cal_stub()

    p = PageState(bridge, cal)

    assert p.bridge is bridge
    assert p.cal is cal


# ---------- set_mode ----------


def test_set_mode_transitions_to_calibrate_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    p = PageState(_bridge_stub(), _cal_stub())

    with caplog.at_level(logging.INFO, logger="physiclaw.core.bridge.page"):
        p.set_mode("calibrate")

    assert p.mode == "calibrate"
    assert any(r.getMessage() == "Phone mode → calibrate" for r in caplog.records)


def test_set_mode_no_op_when_already_in_target_mode_does_not_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    p = PageState(_bridge_stub(), _cal_stub())  # starts in "bridge"

    with caplog.at_level(logging.INFO, logger="physiclaw.core.bridge.page"):
        p.set_mode("bridge")

    assert all("Phone mode →" not in r.getMessage() for r in caplog.records)


def test_set_mode_calibrate_with_phase_forwards_to_cal_set_phase() -> None:
    cal = _cal_stub()
    p = PageState(_bridge_stub(), cal)

    p.set_mode("calibrate", phase="grid", index=5)

    assert cal.set_phase_calls == [("grid", {"index": 5})]


def test_set_mode_calibrate_without_phase_does_not_call_set_phase() -> None:
    cal = _cal_stub()
    p = PageState(_bridge_stub(), cal)

    p.set_mode("calibrate")

    assert cal.set_phase_calls == []


def test_set_mode_calibrate_arms_upload_window() -> None:
    # The user may double-tap before pressing the step's button — entering
    # calibrate mode must open the upload window right away.
    from physiclaw.core.bridge.state import CAL_UPLOAD_WINDOW_SECONDS

    bridge = _bridge_stub()
    p = PageState(bridge, _cal_stub())

    p.set_mode("calibrate", phase="screenshot_cal")

    assert bridge.arm_upload_calls == [CAL_UPLOAD_WINDOW_SECONDS]


def test_set_mode_bridge_does_not_arm_upload_window() -> None:
    bridge = _bridge_stub()
    p = PageState(bridge, _cal_stub())

    p.set_mode("bridge")

    assert bridge.arm_upload_calls == []


def test_set_mode_bridge_with_phase_does_not_call_set_phase() -> None:
    # Phase forwarding is gated on `mode == "calibrate"` — passing phase
    # during a switch back to bridge must NOT touch cal.set_phase.
    cal = _cal_stub()
    p = PageState(_bridge_stub(), cal)
    p.set_mode("calibrate")  # populate set_phase_calls baseline
    cal.set_phase_calls.clear()

    p.set_mode("bridge", phase="ignored")

    assert cal.set_phase_calls == []


# ---------- get_state ----------


def test_get_state_in_bridge_mode_includes_text_from_bridge() -> None:
    # The text rides the poll deliberately — the phone must DISPLAY it so
    # the operator sees what's about to be pasted (see PageState.get_state).
    bridge = _bridge_stub("hello user")
    p = PageState(bridge, _cal_stub())

    state = p.get_state()

    assert state == {
        "mode": "bridge",
        "has_device_info": False,
        "text": "hello user",
        "copied": False,
    }


def test_get_state_in_bridge_mode_reflects_copied_flag() -> None:
    bridge = _bridge_stub("done", copied=True)
    p = PageState(bridge, _cal_stub())

    state = p.get_state()

    assert state["copied"] is True


def test_get_state_includes_has_device_info_true_when_screen_dimension_set() -> None:
    cal = _cal_stub(screen_dimension={"width": 1170, "height": 2532})
    p = PageState(_bridge_stub(), cal)

    state = p.get_state()

    assert state["has_device_info"] is True


def test_get_state_in_calibrate_mode_merges_cal_state() -> None:
    cal = _cal_stub(state={"phase": "grid", "step": 3})
    p = PageState(_bridge_stub(), cal)
    p.set_mode("calibrate")

    state = p.get_state()

    # Mode + has_device_info from PageState; phase + step merged in.
    assert state["mode"] == "calibrate"
    assert state["phase"] == "grid"
    assert state["step"] == 3
    # `text` from bridge is NOT included in calibrate mode.
    assert "text" not in state
