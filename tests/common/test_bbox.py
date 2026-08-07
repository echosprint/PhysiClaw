"""Tests for `physiclaw.common.bbox` — the shared core/agent bbox contract."""

from __future__ import annotations

from typing import Any

import pytest

from physiclaw.common.bbox import center_of, inside, near, validate_bbox


@pytest.mark.parametrize(
    "bbox",
    [None, "0,0,1,1", [0.1, 0.2, 0.9], [0.1, 0.2, 0.9, 1.0, 0.5], {}, 0.5],
)
def test_wrong_shape_raises(bbox: Any) -> None:
    with pytest.raises(
        ValueError, match=r"^bbox: must be \[left, top, right, bottom\];"
    ):
        validate_bbox(bbox)


@pytest.mark.parametrize(
    "bbox",
    [[0.1, "0.2", 0.9, 1.0], [0.1, None, 0.9, 1.0], [0.1, [0.2], 0.9, 1.0]],
)
def test_non_number_coord_raises(bbox: list) -> None:
    with pytest.raises(ValueError, match=r"^bbox: each coord must be a number;"):
        validate_bbox(bbox)


@pytest.mark.parametrize(
    "bbox",
    [[-0.1, 0.2, 0.9, 1.0], [0.1, 0.2, 1.1, 1.0], [0.1, -2, 0.9, 1.0]],
)
def test_out_of_unit_range_raises(bbox: list[float]) -> None:
    with pytest.raises(ValueError, match=r"^bbox: each coord must be in \[0, 1\];"):
        validate_bbox(bbox)


@pytest.mark.parametrize(
    "bbox",
    [[0.9, 0.2, 0.1, 1.0], [0.1, 1.0, 0.9, 0.2], [0.5, 0.2, 0.5, 1.0]],
)
def test_inverted_or_degenerate_raises(bbox: list[float]) -> None:
    with pytest.raises(ValueError, match=r"^bbox: left < right, top < bottom;"):
        validate_bbox(bbox)


def test_valid_bbox_returned_unchanged() -> None:
    bbox = [0.1, 0.2, 0.9, 1.0]
    assert validate_bbox(bbox) is bbox


def test_tuple_accepted() -> None:
    bbox = (0.0, 0.0, 1.0, 1.0)
    assert validate_bbox(bbox) is bbox


# ---------- cross-layer contract ----------
#
# Both layers must expose the SAME checks so the agent sees an identical
# diagnostic regardless of which side catches the violation. These pins
# turn a drifted re-export or a re-inlined copy into a red test.


def test_core_reexport_is_the_same_object() -> None:
    from physiclaw.core.vision import util

    assert util.validate_bbox is validate_bbox


def test_engine_validator_wraps_the_same_message() -> None:
    from physiclaw.agent.engine.validator import ValidationError, validate_arguments

    bad = [0.9, 0.2, 0.1, 1.0]
    with pytest.raises(ValueError) as common_err:
        validate_bbox(bad)
    with pytest.raises(ValidationError) as engine_err:
        validate_arguments({"bbox": bad}, {"type": "object", "properties": {}})
    assert str(engine_err.value) == str(common_err.value)


# ---------- geometry primitives ----------


@pytest.mark.parametrize(
    "bbox", [None, [], [0.1, 0.2], [0.1, "x", 0.9, 1.0], "garbage", {}]
)
def test_center_of_garbage_is_none(bbox) -> None:
    assert center_of(bbox) is None


def test_center_of_valid_bbox() -> None:
    assert center_of([0.2, 0.4, 0.6, 0.8]) == (0.4, 0.6000000000000001)


def test_center_of_accepts_string_floats() -> None:
    # bboxes re-transcribed by the model sometimes arrive as strings.
    assert center_of(["0.2", "0.4", "0.6", "0.8"]) is not None


def test_near_within_tolerance() -> None:
    a = (0.5, 0.5)
    assert near(a, (0.5 + 0.02 * 0.9, 0.5 - 0.02 * 0.9), tolerance=0.02)
    assert not near(a, (0.5 + 0.02 * 1.5, 0.5), tolerance=0.02)


def test_inside_zero_margin_is_exact_containment() -> None:
    box = [0.2, 0.2, 0.4, 0.4]
    assert inside((0.4, 0.3), box, margin=0.0)
    assert not inside((0.401, 0.3), box, margin=0.0)


def test_inside_expands_by_margin() -> None:
    box = [0.2, 0.2, 0.4, 0.4]
    assert inside((0.4 + 0.02, 0.3), box, margin=0.02)
    assert not inside((0.4 + 0.02 * 1.5, 0.3), box, margin=0.02)
