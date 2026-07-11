"""Tests for `physiclaw.core.orchestration.gestures` — typed gesture
values and GestureValidator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from physiclaw.core.orchestration import gestures
from physiclaw.core.orchestration.gestures import (
    DoubleTap,
    GestureValidator,
    LongPress,
    SendToClipboard,
    Swipe,
    Tap,
)

BBOX = [0.1, 0.1, 0.2, 0.2]


@pytest.fixture
def at() -> MagicMock:
    m = MagicMock()
    m.overlaps_at.return_value = False
    m.swipe_crosses_at.return_value = False
    return m


@pytest.fixture
def validator(at: MagicMock) -> GestureValidator:
    transforms = MagicMock()
    transforms.bbox_center_pct.side_effect = lambda bbox: (
        (bbox[0] + bbox[2]) / 2,
        (bbox[1] + bbox[3]) / 2,
    )
    return GestureValidator(
        assistive_touch=lambda: at,
        transforms=lambda: transforms,
    )


# ---------- parse_step: typed values ----------


def test_parse_step_tap(validator: GestureValidator) -> None:
    assert validator.parse_step("tap", BBOX) == Tap(BBOX)


def test_parse_step_double_tap(validator: GestureValidator) -> None:
    assert validator.parse_step("double_tap", BBOX) == DoubleTap(BBOX)


def test_parse_step_long_press(validator: GestureValidator) -> None:
    assert validator.parse_step("long_press", BBOX) == LongPress(BBOX)


def test_parse_step_swipe_applies_defaults(validator: GestureValidator) -> None:
    out = validator.parse_step("swipe", {"bbox": BBOX, "direction": "up"})

    assert out == Swipe(BBOX, "up", "m", "medium")


def test_parse_step_swipe_full_args(validator: GestureValidator) -> None:
    out = validator.parse_step(
        "swipe",
        {"bbox": BBOX, "direction": "left", "size": "xl", "speed": "fast"},
    )

    assert out == Swipe(BBOX, "left", "xl", "fast")


def test_parse_step_send_to_clipboard(validator: GestureValidator) -> None:
    assert validator.parse_step("send_to_clipboard", "hi") == SendToClipboard("hi")


# ---------- parse_step: agent-facing errors (pinned text) ----------


def test_parse_step_rejects_bad_bbox(validator: GestureValidator) -> None:
    with pytest.raises(ValueError, match="bbox"):
        validator.parse_step("tap", [0.5, 0.5])


def test_parse_step_swipe_requires_dict_with_keys(validator: GestureValidator) -> None:
    # No repr of the raw arg: the message joins verdict-scanned action text.
    with pytest.raises(
        ValueError, match="swipe arg needs a dict with bbox \\+ direction, got str"
    ):
        validator.parse_step("swipe", "not-a-dict")
    with pytest.raises(ValueError, match="swipe arg needs a dict"):
        validator.parse_step("swipe", {"bbox": [0, 0, 1, 1]})


def test_parse_step_swipe_range_checks(validator: GestureValidator) -> None:
    with pytest.raises(ValueError, match="direction must be"):
        validator.parse_step("swipe", {"bbox": BBOX, "direction": "diagonal"})


def test_parse_step_clipboard_requires_string(validator: GestureValidator) -> None:
    with pytest.raises(
        ValueError, match="send_to_clipboard arg must be a string, got int"
    ):
        validator.parse_step("send_to_clipboard", 42)


def test_parse_step_unknown_tool(validator: GestureValidator) -> None:
    with pytest.raises(ValueError, match="tool 'delete_app' not allowed in sequence"):
        validator.parse_step("delete_app", "anything")


# ---------- validate_swipe ----------


def test_validate_swipe_accepts_valid_args(validator: GestureValidator) -> None:
    validator.validate_swipe(BBOX, "up", "m", "medium")  # no raise


def test_validate_swipe_rejects_bad_direction(validator: GestureValidator) -> None:
    with pytest.raises(ValueError, match="direction must be"):
        validator.validate_swipe(BBOX, "diagonal", "m", "medium")


def test_validate_swipe_rejects_bad_size(validator: GestureValidator) -> None:
    with pytest.raises(ValueError, match="size must be"):
        validator.validate_swipe(BBOX, "up", "huge", "medium")


def test_validate_swipe_rejects_bad_speed(validator: GestureValidator) -> None:
    with pytest.raises(ValueError, match="speed must be"):
        validator.validate_swipe(BBOX, "up", "m", "warp")


# ---------- AssistiveTouch guards ----------


def test_require_no_at_overlap_passes(validator: GestureValidator) -> None:
    validator.require_no_at_overlap(BBOX, "tap")  # no raise


def test_require_no_at_overlap_raises(
    validator: GestureValidator, at: MagicMock
) -> None:
    at.overlaps_at.return_value = True

    with pytest.raises(ValueError, match="overlaps AssistiveTouch button — aim aside"):
        validator.require_no_at_overlap(BBOX, "tap")


def test_require_no_at_crossing_raises(
    validator: GestureValidator, at: MagicMock
) -> None:
    at.swipe_crosses_at.return_value = True

    with pytest.raises(ValueError, match="crosses AssistiveTouch button — aim aside"):
        validator.require_no_at_crossing(BBOX, "up")


def test_guards_check_the_bbox_center(
    validator: GestureValidator, at: MagicMock
) -> None:
    validator.require_no_at_overlap([0.1, 0.1, 0.3, 0.3], "tap")

    at.overlaps_at.assert_called_once()
    cx, cy = at.overlaps_at.call_args.args
    assert cx == pytest.approx(0.2)
    assert cy == pytest.approx(0.2)


def test_guards_read_current_assistive_touch() -> None:
    # The validator holds accessors, not objects — swapping the AT after
    # construction (calibration, test fixtures) must take effect.
    holder = {"at": MagicMock()}
    holder["at"].overlaps_at.return_value = False
    transforms = MagicMock()
    transforms.bbox_center_pct.return_value = (0.5, 0.5)
    v = GestureValidator(
        assistive_touch=lambda: holder["at"],
        transforms=lambda: transforms,
    )
    v.require_no_at_overlap(BBOX, "tap")

    replacement = MagicMock()
    replacement.overlaps_at.return_value = True
    holder["at"] = replacement

    with pytest.raises(ValueError, match="overlaps AssistiveTouch"):
        v.require_no_at_overlap(BBOX, "tap")


# ---------- constants ----------


def test_swipe_distances_cover_all_sizes() -> None:
    assert list(gestures.SWIPE_DISTANCES) == ["s", "m", "l", "xl", "xxl"]
