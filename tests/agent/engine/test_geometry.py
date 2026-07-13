"""Tests for `physiclaw.agent.engine.geometry` — shared press-geometry leaf."""

from __future__ import annotations

import pytest

from physiclaw.agent.engine.geometry import (
    LINT_MARGIN,
    MATCH_TOLERANCE,
    center_of,
    inside,
    near,
)


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
    assert near(a, (0.5 + MATCH_TOLERANCE * 0.9, 0.5 - MATCH_TOLERANCE * 0.9))
    assert not near(a, (0.5 + MATCH_TOLERANCE * 1.5, 0.5))


def test_inside_expands_by_lint_margin() -> None:
    box = [0.2, 0.2, 0.4, 0.4]
    assert inside((0.4 + LINT_MARGIN, 0.3), box)
    assert not inside((0.4 + LINT_MARGIN * 1.5, 0.3), box)


def test_tolerances_are_independent_names() -> None:
    # Deliberately separate constants: same-target matching vs layout
    # lint slack may drift apart. This pin documents the intent, not a
    # value coupling.
    assert MATCH_TOLERANCE > 0 and LINT_MARGIN > 0
