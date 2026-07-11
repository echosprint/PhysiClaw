"""Tests for `physiclaw.core.calibration.transforms`.

Coordinate math used everywhere downstream (gestures, calibration,
screen-match). Two dataclasses:

  - `ViewportShift` — frozen; `css_to_pct` only.
  - `ScreenTransforms` — affine matrix container with helpers.

Hypothesis is used for the `pct → cam_pixel → pct` round-trip; tolerance
is 1/cam_width because `pct_to_cam_pixel` truncates with `int()`.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from physiclaw.core.calibration.transforms import (
    ScreenTransforms,
    ViewportShift,
)

IDENTITY_AFFINE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


# ---------- ViewportShift — class-level contract ----------


def test_viewport_shift_is_frozen() -> None:
    vs = ViewportShift(
        offset_x=0, offset_y=0, dpr=1.0, screenshot_width=100, screenshot_height=200
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        vs.dpr = 2.0  # type: ignore[misc]


def test_screen_transforms_is_dataclass() -> None:
    st_obj = ScreenTransforms(IDENTITY_AFFINE.copy(), IDENTITY_AFFINE.copy())

    assert dataclasses.is_dataclass(st_obj)


# ---------- ViewportShift.css_to_pct ----------


def test_viewport_shift_css_to_pct_origin_at_no_offset_is_zero() -> None:
    vs = ViewportShift(
        offset_x=0, offset_y=0, dpr=1.0, screenshot_width=100, screenshot_height=200
    )

    assert vs.css_to_pct(0, 0) == (0.0, 0.0)


def test_viewport_shift_css_to_pct_full_extent_at_no_offset_is_one() -> None:
    vs = ViewportShift(
        offset_x=0, offset_y=0, dpr=1.0, screenshot_width=100, screenshot_height=200
    )

    assert vs.css_to_pct(100, 200) == (1.0, 1.0)


def test_viewport_shift_css_to_pct_applies_dpr_then_offset() -> None:
    vs = ViewportShift(
        offset_x=10,
        offset_y=20,
        dpr=2.0,
        screenshot_width=200,
        screenshot_height=400,
    )

    # css (50, 100) * dpr 2 + offset (10, 20) = (110, 220)
    # → (110/200, 220/400) = (0.55, 0.55)
    sx, sy = vs.css_to_pct(50, 100)

    assert (sx, sy) == pytest.approx((0.55, 0.55))


@given(
    css_x=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    css_y=st.floats(min_value=0, max_value=200, allow_nan=False, allow_infinity=False),
)
def test_viewport_shift_css_to_pct_keeps_in_unit_square_when_css_in_range(
    css_x: float, css_y: float
) -> None:
    vs = ViewportShift(
        offset_x=0, offset_y=0, dpr=1.0, screenshot_width=100, screenshot_height=200
    )

    sx, sy = vs.css_to_pct(css_x, css_y)

    assert 0.0 <= sx <= 1.0
    assert 0.0 <= sy <= 1.0


# ---------- ScreenTransforms — construction ----------


def test_screen_transforms_default_cam_size_is_1920_1080() -> None:
    st_obj = ScreenTransforms(IDENTITY_AFFINE.copy(), IDENTITY_AFFINE.copy())

    assert st_obj.cam_size == (1920, 1080)


def test_screen_transforms_custom_cam_size_stored() -> None:
    st_obj = ScreenTransforms(
        IDENTITY_AFFINE.copy(), IDENTITY_AFFINE.copy(), cam_size=(800, 600)
    )

    assert st_obj.cam_size == (800, 600)


# ---------- bbox_center_pct ----------


@pytest.mark.parametrize(
    "bbox, expected",
    [
        ([0.0, 0.0, 1.0, 1.0], (0.5, 0.5)),
        ([0.2, 0.4, 0.6, 0.8], (0.4, 0.6)),
        ([0.5, 0.5, 0.5, 0.5], (0.5, 0.5)),  # degenerate point
    ],
)
def test_bbox_center_pct_returns_midpoint(
    bbox: list[float], expected: tuple[float, float]
) -> None:
    st_obj = ScreenTransforms(IDENTITY_AFFINE.copy(), IDENTITY_AFFINE.copy())

    assert st_obj.bbox_center_pct(bbox) == pytest.approx(expected)


# ---------- swipe_end_pct ----------


@pytest.mark.parametrize(
    "direction, dist, expected",
    [
        ("up", 0.1, (0.5, 0.4)),
        ("down", 0.1, (0.5, 0.6)),
        ("left", 0.1, (0.4, 0.5)),
        ("right", 0.1, (0.6, 0.5)),
    ],
)
def test_swipe_end_pct_moves_in_direction(
    direction: str, dist: float, expected: tuple[float, float]
) -> None:
    st_obj = ScreenTransforms(IDENTITY_AFFINE.copy(), IDENTITY_AFFINE.copy())

    end = st_obj.swipe_end_pct([0.4, 0.4, 0.6, 0.6], direction, dist)

    assert end == pytest.approx(expected)


@pytest.mark.parametrize(
    "direction, expected",
    [
        ("up", (0.5, 0.0)),
        ("down", (0.5, 1.0)),
        ("left", (0.0, 0.5)),
        ("right", (1.0, 0.5)),
    ],
)
def test_swipe_end_pct_clamps_to_unit_square(
    direction: str, expected: tuple[float, float]
) -> None:
    st_obj = ScreenTransforms(IDENTITY_AFFINE.copy(), IDENTITY_AFFINE.copy())

    end = st_obj.swipe_end_pct([0.4, 0.4, 0.6, 0.6], direction, 5.0)

    assert end == expected


def test_swipe_end_pct_invalid_direction_raises_value_error() -> None:
    st_obj = ScreenTransforms(IDENTITY_AFFINE.copy(), IDENTITY_AFFINE.copy())

    # Anchored ^ — mutmut wrapping the message in `XX…XX` would otherwise
    # still match a substring search.
    with pytest.raises(
        ValueError, match=r"^direction must be up/down/left/right, got 'diagonal'$"
    ):
        st_obj.swipe_end_pct([0.0, 0.0, 1.0, 1.0], "diagonal", 0.1)


# ---------- pct_to_grbl_mm ----------


def test_pct_to_grbl_mm_identity_returns_input() -> None:
    st_obj = ScreenTransforms(IDENTITY_AFFINE.copy(), IDENTITY_AFFINE.copy())

    assert st_obj.pct_to_grbl_mm(0.3, 0.7) == pytest.approx((0.3, 0.7))


def test_pct_to_grbl_mm_applies_scale_and_translation() -> None:
    # screen 0-1 → mm: x scaled ×100 + offset 5; y scaled ×200 + offset 10
    pct_to_grbl = np.array([[100.0, 0.0, 5.0], [0.0, 200.0, 10.0]])
    st_obj = ScreenTransforms(pct_to_grbl, IDENTITY_AFFINE.copy())

    x_mm, y_mm = st_obj.pct_to_grbl_mm(0.5, 0.25)

    assert (x_mm, y_mm) == pytest.approx((55.0, 60.0))


# ---------- pct_to_cam_pixel ----------


def test_pct_to_cam_pixel_uses_default_cam_size() -> None:
    st_obj = ScreenTransforms(IDENTITY_AFFINE.copy(), IDENTITY_AFFINE.copy())

    assert st_obj.pct_to_cam_pixel(0.5, 0.5) == (960, 540)


def test_pct_to_cam_pixel_uses_custom_cam_size() -> None:
    st_obj = ScreenTransforms(
        IDENTITY_AFFINE.copy(), IDENTITY_AFFINE.copy(), cam_size=(1000, 500)
    )

    assert st_obj.pct_to_cam_pixel(0.4, 0.8) == (400, 400)


def test_pct_to_cam_pixel_truncates_to_int() -> None:
    # cam_size 1000×1000, pct 0.1234 → cam_01 0.1234 → px int(123.4) = 123
    st_obj = ScreenTransforms(
        IDENTITY_AFFINE.copy(), IDENTITY_AFFINE.copy(), cam_size=(1000, 1000)
    )

    px, py = st_obj.pct_to_cam_pixel(0.1234, 0.5678)

    assert (px, py) == (123, 567)


def test_pct_to_cam_pixel_applies_translation() -> None:
    # Non-zero translation — proves the homogeneous `1.0` coord is used as
    # the third element (mutating it to `2.0` would double the offset).
    pct_to_cam = np.array([[1.0, 0.0, 0.1], [0.0, 1.0, 0.2]])
    st_obj = ScreenTransforms(IDENTITY_AFFINE.copy(), pct_to_cam, cam_size=(1000, 1000))

    # cam_01 = (x + 0.1, y + 0.2) ; px = (int((x+0.1)*1000), int((y+0.2)*1000))
    assert st_obj.pct_to_cam_pixel(0.0, 0.0) == (100, 200)
    assert st_obj.pct_to_cam_pixel(0.4, 0.3) == (500, 500)


# ---------- pixel_to_pct ----------


def test_pixel_to_pct_identity_round_trip_at_grid_points() -> None:
    st_obj = ScreenTransforms(
        IDENTITY_AFFINE.copy(), IDENTITY_AFFINE.copy(), cam_size=(1000, 1000)
    )

    # exact pixel → pct: integer pixels round-trip cleanly
    assert st_obj.pixel_to_pct(500, 250) == pytest.approx((0.5, 0.25))


def test_pixel_to_pct_subtracts_translation() -> None:
    # With translation b in pct_to_cam, the inverse must subtract it.
    # Mutating `cam_01 - b` to `cam_01 + b` would double-displace the result.
    pct_to_cam = np.array([[1.0, 0.0, 0.1], [0.0, 1.0, 0.2]])
    st_obj = ScreenTransforms(IDENTITY_AFFINE.copy(), pct_to_cam, cam_size=(1000, 1000))

    # pct_to_cam_pixel(0.4, 0.3) = (500, 500); inverse should return (0.4, 0.3).
    assert st_obj.pixel_to_pct(500, 500) == pytest.approx((0.4, 0.3))


@given(
    px=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
    py=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
)
def test_pct_to_cam_pixel_then_pixel_to_pct_loses_at_most_one_pixel(
    px: float, py: float
) -> None:
    cam_w, cam_h = 1000, 1000
    st_obj = ScreenTransforms(
        IDENTITY_AFFINE.copy(),
        IDENTITY_AFFINE.copy(),
        cam_size=(cam_w, cam_h),
    )

    cam_px = st_obj.pct_to_cam_pixel(px, py)
    rx, ry = st_obj.pixel_to_pct(*cam_px)

    # int() truncation costs at most 1/cam_width per axis
    assert math.isclose(rx, px, abs_tol=1.0 / cam_w)
    assert math.isclose(ry, py, abs_tol=1.0 / cam_h)


# ---------- bbox_to_pixel_rect ----------


def test_bbox_to_pixel_rect_returns_top_left_and_bottom_right() -> None:
    st_obj = ScreenTransforms(
        IDENTITY_AFFINE.copy(), IDENTITY_AFFINE.copy(), cam_size=(1000, 500)
    )

    tl, br = st_obj.bbox_to_pixel_rect([0.1, 0.2, 0.9, 0.8])

    assert tl == (100, 100)
    assert br == (900, 400)
