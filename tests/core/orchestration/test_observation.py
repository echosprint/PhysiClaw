"""Tests for `physiclaw.core.orchestration.observation` — GestureObserver.

The observer takes plain `park` / `grab` / `detect` callables, so the
grab-retry, verdict, and warning-attachment logic is tested here without
an arm, camera, or the orchestrator's wiring (that wiring is covered in
test_orchestrator.py)."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from physiclaw.core.orchestration import observation
from physiclaw.core.orchestration.observation import GestureObserver, GestureResult


def _sharp() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, size=(200, 100, 3), dtype=np.uint8)


def _flat(value: int = 128) -> np.ndarray:
    return np.full((200, 100, 3), value, dtype=np.uint8)


def _observer(frames: list, *, detect=None) -> GestureObserver:
    """Observer over a scripted frame sequence; quality is neutralized
    via the injectable monitor (re-armed per test through `_quality`),
    settle sleeps are zeroed, and the gesture blur-retry is disabled —
    flat synthetic frames (Laplacian variance 0) would otherwise read
    blurry and consume two frames per grab. Retry tests re-raise the
    threshold themselves."""
    it = iter(frames)
    monitor = MagicMock()
    monitor.observe.return_value = None
    obs = GestureObserver(
        park=MagicMock(),
        grab=lambda: next(it),
        detect=detect or MagicMock(return_value=("LISTING", _flat())),
        monitor=monitor,
    )
    obs.GESTURE_SETTLE_SECONDS = 0
    obs.GRAB_BLUR_THRESHOLD = 0
    return obs


# ---------- grab_screen ----------


def test_grab_screen_parks_before_grabbing() -> None:
    order = []
    obs = GestureObserver(
        park=lambda: order.append("park"),
        grab=lambda: order.append("grab") or _sharp(),
        detect=MagicMock(),
    )
    obs.GRAB_BLUR_THRESHOLD = 50.0

    obs.grab_screen()

    assert order == ["park", "grab"]


def test_grab_screen_returns_sharp_frame_without_retry(mocker) -> None:
    sleep = mocker.patch.object(observation.time, "sleep")
    obs = _observer([_sharp()])
    obs.GRAB_BLUR_THRESHOLD = 50.0

    frame, sharp_flag = obs.grab_screen()

    assert sharp_flag is True
    sleep.assert_not_called()


def test_grab_screen_settles_before_capture(mocker) -> None:
    sleep = mocker.patch.object(observation.time, "sleep")
    obs = _observer([_sharp()])
    obs.GRAB_BLUR_THRESHOLD = 50.0

    obs.grab_screen(settle=1.5)

    sleep.assert_called_once_with(1.5)


def test_grab_screen_retries_once_when_blurry(mocker) -> None:
    # A frame captured mid-autofocus-hunt (low Laplacian variance) is
    # re-grabbed once after a settle wait.
    sleep = mocker.patch.object(observation.time, "sleep")
    sharp = _sharp()
    obs = _observer([_flat(), sharp])
    obs.GRAB_BLUR_THRESHOLD = 50.0

    frame, sharp_flag = obs.grab_screen()

    assert frame is sharp
    assert sharp_flag is True
    sleep.assert_called_once_with(obs.GRAB_BLUR_RETRY_SECONDS)


def test_grab_screen_flags_frame_still_blurry_after_retry(mocker) -> None:
    # The retry frame is kept for the fused view but flagged unsharp so
    # the caller withholds the verdict (blur diffs as changed everywhere).
    mocker.patch.object(observation.time, "sleep")
    blurry = _flat()
    obs = _observer([blurry, blurry])
    obs.GRAB_BLUR_THRESHOLD = 50.0

    frame, sharp_flag = obs.grab_screen()

    assert frame is blurry
    assert sharp_flag is False


def test_grab_screen_fails_open_when_grab_raises() -> None:
    # The view is best-effort — a camera failure must never fail a gesture.
    def grab():
        raise RuntimeError("camera down")

    obs = GestureObserver(park=MagicMock(), grab=grab, detect=MagicMock())

    assert obs.grab_screen() == (None, False)


def test_grab_screen_fails_open_when_park_raises() -> None:
    obs = GestureObserver(
        park=MagicMock(side_effect=RuntimeError("arm jammed")),
        grab=MagicMock(),
        detect=MagicMock(),
    )

    assert obs.grab_screen() == (None, False)


# ---------- peek_frame ----------


def test_peek_frame_returns_sharp_frame_without_retry(mocker) -> None:
    sleep = mocker.patch.object(observation.time, "sleep")
    frame = _sharp()
    obs = _observer([frame])

    assert obs.peek_frame() is frame
    sleep.assert_not_called()


def test_peek_frame_retries_once_and_keeps_retry_frame(mocker) -> None:
    # Unlike grab_screen there's no verdict to protect: the retry frame
    # is used as-is, with no second sharpness check.
    sleep = mocker.patch.object(observation.time, "sleep")
    retry = _flat()
    obs = _observer([_flat(), retry])

    assert obs.peek_frame() is retry
    sleep.assert_called_once_with(obs.PEEK_BLUR_RETRY_SECONDS)


def test_peek_frame_propagates_grab_failure() -> None:
    # Unlike grab_screen, a peek with no frame is a tool error — the
    # failure must reach the caller, not fail open.
    def grab():
        raise RuntimeError("camera down")

    obs = GestureObserver(park=MagicMock(), grab=grab, detect=MagicMock())

    with pytest.raises(RuntimeError, match="camera down"):
        obs.peek_frame()


# ---------- with_view: verdict ----------


def test_with_view_appends_changed_verdict(mocker) -> None:
    mocker.patch.object(observation, "encode_jpeg", return_value=b"VIEW_JPG")
    obs = _observer([_flat(128), _flat(30)])

    out = obs.with_view(lambda: "Acted")

    assert out.text == "Acted | screen: changed"


def test_with_view_appends_unchanged_verdict(mocker) -> None:
    mocker.patch.object(observation, "encode_jpeg", return_value=b"VIEW_JPG")
    frame = _flat(128)
    obs = _observer([frame, frame.copy()])

    out = obs.with_view(lambda: "Acted")

    assert out.text == "Acted | screen: no visible change"


def test_with_view_unmarked_when_grabs_fail() -> None:
    # Fail open: no frames → no marker, gesture result intact.
    def grab():
        raise RuntimeError("camera down")

    obs = GestureObserver(park=MagicMock(), grab=grab, detect=MagicMock())

    out = obs.with_view(lambda: "Acted")

    assert out == GestureResult(text="Acted", jpeg=None, listing=None)


def test_with_view_withholds_verdict_when_side_blurry(mocker) -> None:
    # A blurry side diffs as "changed everywhere" (false `changed`, the
    # harmful direction) — verdict skipped, view still attached.
    mocker.patch.object(observation.time, "sleep")
    mocker.patch.object(observation, "encode_jpeg", return_value=b"VIEW_JPG")
    # Every grab reads blurry → each grab_screen consumes two frames.
    obs = _observer([_flat(128), _flat(128), _flat(30), _flat(30)])
    obs.GRAB_BLUR_THRESHOLD = float("inf")

    out = obs.with_view(lambda: "Acted")

    assert out.text == "Acted"  # unmarked
    assert out.jpeg == b"VIEW_JPG"
    assert out.listing == "LISTING"


def test_with_view_verdict_diff_failure_fails_open(mocker) -> None:
    mocker.patch.object(observation, "encode_jpeg", return_value=b"VIEW_JPG")
    mocker.patch.object(
        observation, "frames_changed", side_effect=RuntimeError("diff crashed"),
    )
    obs = _observer([_flat(128), _flat(30)])

    out = obs.with_view(lambda: "Acted")

    assert out.text == "Acted"  # unmarked, view intact
    assert out.listing == "LISTING"


# ---------- with_view: fused view ----------


def test_with_view_returns_fused_view(mocker) -> None:
    # The after-frame that feeds the verdict also feeds detection.
    mocker.patch.object(observation, "encode_jpeg", return_value=b"VIEW_JPG")
    after = _flat(30)
    detect = MagicMock(return_value=("LISTING", _flat()))
    obs = _observer([_flat(128), after], detect=detect)

    out = obs.with_view(lambda: "Acted")

    assert out.jpeg == b"VIEW_JPG"
    assert out.listing == "LISTING"
    detect.assert_called_once_with(after)


def test_with_view_fails_open_when_detection_raises(mocker) -> None:
    detect = MagicMock(side_effect=RuntimeError("model missing"))
    obs = _observer([_flat(128), _flat(30)], detect=detect)

    out = obs.with_view(lambda: "Acted")

    # Verdict still attaches; only the view is absent.
    assert out.text == "Acted | screen: changed"
    assert out.jpeg is None and out.listing is None


# ---------- with_view: quality warning ----------


def test_with_view_warning_joins_listing(mocker) -> None:
    mocker.patch.object(observation, "encode_jpeg", return_value=b"VIEW_JPG")
    obs = _observer([_flat(128), _flat(30)])
    obs._quality.observe.return_value = "⚠ camera: bad"

    out = obs.with_view(lambda: "Acted")

    assert out.listing == "LISTING\n⚠ camera: bad"
    assert out.text == "Acted | screen: changed"  # verdict untouched


def test_with_view_warning_rides_action_text_when_detect_fails(mocker) -> None:
    # Detection crashing must not silence the observation: a blurry/blown
    # frame is a plausible CAUSE of the failed view.
    detect = MagicMock(side_effect=RuntimeError("model gone"))
    obs = _observer([_flat(128), _flat(30)], detect=detect)
    obs._quality.observe.return_value = "⚠ camera: bad"

    out = obs.with_view(lambda: "Acted")

    assert out.listing is None
    assert out.text.endswith("\n⚠ camera: bad")


def test_with_view_no_quality_check_without_after_frame() -> None:
    def grab():
        raise RuntimeError("camera down")

    obs = GestureObserver(park=MagicMock(), grab=grab, detect=MagicMock())
    obs._quality = MagicMock()

    obs.with_view(lambda: "Acted")

    obs._quality.observe.assert_not_called()


# ---------- observe_quality ----------


def test_observe_quality_returns_warning_line() -> None:
    obs = _observer([])
    obs._quality.observe.return_value = "⚠ camera: bad"

    assert obs.observe_quality("peek", _flat()) == "⚠ camera: bad"


def test_observe_quality_none_on_good_view() -> None:
    obs = _observer([])

    assert obs.observe_quality("peek", _flat()) is None


def test_observe_quality_fails_open_on_crash() -> None:
    # A crash in the check never costs the view.
    obs = _observer([])
    obs._quality.observe.side_effect = RuntimeError("boom")

    assert obs.observe_quality("peek", _flat()) is None


def test_observe_quality_uses_real_assessment() -> None:
    # Un-mocked path: a flat frame reads blurry to the real monitor.
    obs = GestureObserver(park=MagicMock(), grab=MagicMock(), detect=MagicMock())

    line = obs.observe_quality("peek", _flat())

    assert line is not None and "⚠ camera" in line


# ---------- GestureResult ----------


def test_gesture_result_defaults_to_text_only() -> None:
    res = GestureResult(text="Acted")

    assert res.jpeg is None and res.listing is None
    with pytest.raises(AttributeError):
        res.text = "frozen"  # type: ignore[misc]
