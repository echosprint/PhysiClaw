"""Tests for `physiclaw.core.vision.grid_detect`."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from physiclaw.core.vision.grid_detect import (
    DOT_MAX_AREA,
    DOT_MIN_AREA,
    _dot_area_bounds,
    compute_affine_transforms,
    detect_bridge_corners,
    detect_orange_dot,
    detect_red_dots,
    detect_screen_corners,
    point_in_polygon,
    screen_polygon,
    sort_dots_to_grid,
)

# ---------- detect_red_dots ----------


def _frame(h: int = 200, w: int = 200) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _draw_dot(frame: np.ndarray, cx: int, cy: int, color, radius: int = 8) -> None:
    cv2.circle(frame, (cx, cy), radius, color, -1)


def test_detect_red_dots_finds_circles() -> None:
    frame = _frame()
    _draw_dot(frame, 50, 50, (0, 0, 255))
    _draw_dot(frame, 150, 50, (0, 0, 255))

    dots = detect_red_dots(frame)

    assert len(dots) == 2
    xs = sorted(d[0] for d in dots)
    assert 40 < xs[0] < 60
    assert 140 < xs[1] < 160


def test_detect_red_dots_returns_empty_on_blank_frame() -> None:
    assert detect_red_dots(_frame()) == []


def test_detect_red_dots_ignores_non_red() -> None:
    frame = _frame()
    _draw_dot(frame, 50, 50, (0, 255, 0))  # green
    _draw_dot(frame, 100, 50, (255, 0, 0))  # blue

    assert detect_red_dots(frame) == []


def test_detect_red_dots_filters_too_small() -> None:
    frame = _frame()
    # 2-pixel dot — area < 50 threshold.
    _draw_dot(frame, 50, 50, (0, 0, 255), radius=1)

    assert detect_red_dots(frame) == []


def test_detect_red_dots_filters_too_large() -> None:
    frame = _frame(h=400, w=400)
    # Huge filled red region — area > 10000 threshold.
    cv2.rectangle(frame, (50, 50), (350, 350), (0, 0, 255), -1)

    assert detect_red_dots(frame) == []


def test_detect_red_dots_filters_non_circular() -> None:
    frame = _frame()
    # Highly elongated rectangle has low circularity.
    cv2.rectangle(frame, (40, 50), (160, 53), (0, 0, 255), -1)

    assert detect_red_dots(frame) == []


def test_detect_red_dots_handles_high_hue_red() -> None:
    """Red wraps around at H=180, second mask covers 160-180."""
    frame = _frame()
    # Pure red is at H=0; its alternate side wrap is what the second
    # range catches. Using BGR (0, 0, 255) hits the first range, but
    # any near-red mid-saturation sample lands in either. This just
    # confirms a normal red is detected.
    _draw_dot(frame, 100, 100, (0, 0, 255))

    assert len(detect_red_dots(frame)) == 1


# ---------- sort_dots_to_grid ----------


def test_sort_dots_to_grid_orders_row_major() -> None:
    # 3x2 grid scrambled.
    dots = [(20, 50), (10, 10), (10, 50), (20, 10), (20, 30), (10, 30)]

    grid = sort_dots_to_grid(dots, rows=3, cols=2)

    # Row-major order: (10,10), (20,10), (10,30), (20,30), (10,50), (20,50)
    expected = np.array(
        [
            [10, 10],
            [20, 10],
            [10, 30],
            [20, 30],
            [10, 50],
            [20, 50],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(grid, expected)
    assert grid.shape == (6, 2)


def test_sort_dots_to_grid_raises_on_count_mismatch() -> None:
    with pytest.raises(RuntimeError, match="Expected 6 red dots but detected 3"):
        sort_dots_to_grid([(0, 0), (1, 1), (2, 2)], rows=3, cols=2)


def test_sort_dots_to_grid_raises_on_too_many() -> None:
    with pytest.raises(RuntimeError, match="Expected 4 red dots but detected 5"):
        sort_dots_to_grid([(0, 0)] * 5, rows=2, cols=2)


def test_sort_dots_to_grid_returns_float64_dtype() -> None:
    grid = sort_dots_to_grid([(0, 0), (1, 0), (0, 1), (1, 1)], rows=2, cols=2)

    assert grid.dtype == np.float64


# ---------- compute_affine_transforms ----------


def test_compute_affine_transforms_returns_2x3_pair() -> None:
    pcts = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    grbl = pcts * 100  # scale to mm
    pixels = pcts * np.array([800, 1000], dtype=np.float32)

    pct_to_grbl, pct_to_pixel = compute_affine_transforms(pcts, grbl, pixels)

    assert pct_to_grbl.shape == (2, 3)
    assert pct_to_pixel.shape == (2, 3)


def test_compute_affine_transforms_round_trips() -> None:
    pcts = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    grbl = pcts * 100
    pixels = pcts * np.array([800, 1000], dtype=np.float32)

    pct_to_grbl, _ = compute_affine_transforms(pcts, grbl, pixels)

    # Apply to (0.5, 0.5) → expect (50, 50).
    out = pct_to_grbl @ np.array([0.5, 0.5, 1.0])
    np.testing.assert_allclose(out, (50.0, 50.0), atol=1e-3)


def test_compute_affine_transforms_raises_on_too_few_points() -> None:
    pcts = np.array([[0.0, 0.0]], dtype=np.float32)

    with pytest.raises(RuntimeError, match="Failed to compute affine"):
        compute_affine_transforms(pcts, pcts, pcts)


# ---------- detect_orange_dot ----------


def test_detect_orange_dot_returns_center(mocker) -> None:
    spy = mocker.patch(
        "physiclaw.core.vision.grid_detect.find_largest_hsv_blob",
        return_value=(123.4, 56.7),
    )
    frame = _frame()

    out = detect_orange_dot(frame)

    assert out == (123.4, 56.7)
    spy.assert_called_once_with(
        frame,
        [5, 100, 100],
        [25, 255, 255],
        min_area=50,
    )


def test_detect_orange_dot_returns_none_when_no_blob(mocker) -> None:
    mocker.patch(
        "physiclaw.core.vision.grid_detect.find_largest_hsv_blob",
        return_value=None,
    )

    assert detect_orange_dot(_frame()) is None


_ORANGE_BGR = (22, 115, 249)  # #f97316


def test_detect_orange_dot_with_expected_picks_nearest_not_largest() -> None:
    # A big orange reflection far away and the small real dot near the prediction.
    frame = _frame(400, 400)
    _draw_dot(frame, 300, 300, _ORANGE_BGR, radius=22)  # largest
    _draw_dot(frame, 60, 60, _ORANGE_BGR, radius=8)  # nearest to expected

    # No hint → largest blob (the reflection).
    cx, _ = detect_orange_dot(frame)
    assert cx > 200

    # With the predicted location → the small real dot, not the big reflection.
    cx, cy = detect_orange_dot(frame, near=(60, 60))
    assert 50 < cx < 70 and 50 < cy < 70


def test_detect_orange_dot_max_dist_rejects_far_match() -> None:
    frame = _frame(400, 400)
    _draw_dot(frame, 300, 300, _ORANGE_BGR, radius=12)

    # Nearest blob is far from the prediction → rejected so the caller can
    # fall back to the known position instead of chasing a reflection.
    assert detect_orange_dot(frame, near=(60, 60), max_dist=50) is None
    # Same blob, generous cap → accepted.
    assert detect_orange_dot(frame, near=(60, 60), max_dist=500) is not None


# ---------- detect_screen_corners / screen_polygon / point_in_polygon ----------


def _draw_corner(frame: np.ndarray, cx: int, cy: int, d: int = 20) -> None:
    """Draw a 2×2 RGBM corner cluster centered at (cx, cy)."""
    _draw_dot(frame, cx - d, cy - d, (0, 0, 255))  # R
    _draw_dot(frame, cx + d, cy - d, (0, 255, 0))  # G
    _draw_dot(frame, cx + d, cy + d, (255, 0, 0))  # B
    _draw_dot(frame, cx - d, cy + d, (255, 0, 255))  # M (magenta)


def test_detect_screen_corners_finds_four() -> None:
    frame = _frame(400, 600)
    for cx, cy in [(80, 80), (520, 80), (520, 320), (80, 320)]:
        _draw_corner(frame, cx, cy)

    corners = detect_screen_corners(frame)

    assert len(corners) == 4


def test_detect_screen_corners_two_diagonal() -> None:
    frame = _frame(400, 600)
    _draw_corner(frame, 80, 80)
    _draw_corner(frame, 520, 320)

    corners = detect_screen_corners(frame)

    assert len(corners) == 2


def test_detect_screen_corners_ignores_single_colour_cluster() -> None:
    # Red dots alone (e.g. the grid) are not corner clusters — need ≥2 colours.
    frame = _frame(400, 600)
    _draw_dot(frame, 80, 80, (0, 0, 255))
    _draw_dot(frame, 520, 320, (0, 0, 255))

    assert detect_screen_corners(frame) == []


def test_screen_polygon_four_corners_is_quad() -> None:
    corners = [(80, 80), (520, 80), (520, 320), (80, 320)]
    poly = screen_polygon(corners)
    assert poly is not None and len(poly) == 4


def test_screen_polygon_two_corners_is_bbox() -> None:
    poly = screen_polygon([(80, 80), (520, 320)])
    assert poly is not None and len(poly) == 4
    assert poly[:, 0].min() == 80 and poly[:, 0].max() == 520


def test_screen_polygon_needs_two_corners() -> None:
    assert screen_polygon([]) is None
    assert screen_polygon([(80, 80)]) is None


def test_screen_polygon_margin_grows_outward() -> None:
    corners = [(100, 100), (500, 100), (500, 300), (100, 300)]
    tight = screen_polygon(corners, margin=0)
    grown = screen_polygon(corners, margin=30)
    assert grown[:, 0].max() > tight[:, 0].max()
    assert grown[:, 0].min() < tight[:, 0].min()


def test_point_in_polygon_inside_outside_and_none() -> None:
    poly = screen_polygon([(100, 100), (500, 300)])
    assert point_in_polygon(poly, 300, 200) is True
    assert point_in_polygon(poly, 5, 5) is False
    assert point_in_polygon(None, 0, 0) is True  # no gate


# ---------- detect_bridge_corners ----------


def _swatch(img: np.ndarray, cx: int, cy: int, color_bgr: tuple, half: int = 6) -> None:
    img[cy - half : cy + half, cx - half : cx + half] = color_bgr


def _draw_rgbm_cluster(
    img: np.ndarray, cx: int, cy: int, spacing: int = 10, swatch: int = 12
) -> None:
    """Draw an R-G-M-B 2×2 cluster around (cx, cy) in clockwise order:
    R at NW, G at NE, M (magenta) at SE, B at SW. M sits diagonal to R.
    Cluster span ≈ 2*spacing+swatch.
    """
    s = swatch // 2
    _swatch(img, cx - spacing, cy - spacing, (0, 0, 255), s)  # NW: R
    _swatch(img, cx + spacing, cy - spacing, (0, 255, 0), s)  # NE: G
    _swatch(img, cx + spacing, cy + spacing, (255, 0, 255), s)  # SE: M
    _swatch(img, cx - spacing, cy + spacing, (255, 0, 0), s)  # SW: B


def test_detect_bridge_corners_returns_none_when_a_color_is_absent() -> None:
    # Only R, G, B — M missing entirely.
    img = _frame(400, 400)
    _swatch(img, 65, 65, (0, 0, 255), 15)
    _swatch(img, 115, 65, (0, 255, 0), 15)
    _swatch(img, 115, 115, (255, 0, 0), 15)

    assert detect_bridge_corners(img) is None


def test_detect_bridge_corners_returns_dict_with_four_centroids() -> None:
    img = _frame(400, 400)
    _draw_rgbm_cluster(img, 200, 200, spacing=10, swatch=12)

    result = detect_bridge_corners(img)

    assert result is not None
    assert set(result.keys()) == {"R", "G", "B", "M"}
    # Centroids land near the drawn positions.
    rx, ry = result["R"]
    assert 180 < rx < 200 and 180 < ry < 200


def test_detect_bridge_corners_uses_explicit_max_cluster_span_when_supplied() -> None:
    # Caller can override the default-derived span. With a generous
    # explicit span the spread-out cluster IS accepted.
    img = _frame(400, 400)
    _draw_rgbm_cluster(img, 200, 200, spacing=10, swatch=12)

    result = detect_bridge_corners(img, max_cluster_span=80)

    assert result is not None


def test_detect_bridge_corners_rejects_clusters_exceeding_max_span() -> None:
    # Spread the four colors across the whole frame — span >> 25%.
    img = _frame(400, 400)
    _swatch(img, 15, 15, (0, 0, 255), 15)  # R top-left
    _swatch(img, 385, 15, (0, 255, 0), 15)  # G top-right
    _swatch(img, 385, 385, (255, 0, 0), 15)  # B bottom-right
    _swatch(img, 15, 385, (255, 0, 255), 15)  # M bottom-left

    # Default max_span = 25% of min(side) = 100 — these are 400 apart.
    assert detect_bridge_corners(img) is None


# ---------- _dot_area_bounds ----------


def test_dot_area_bounds_unscaled_at_reference_resolution() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    assert _dot_area_bounds(frame) == (DOT_MIN_AREA, DOT_MAX_AREA)


def test_dot_area_bounds_scale_with_4k_frame_area() -> None:
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    assert _dot_area_bounds(frame) == (DOT_MIN_AREA * 4, DOT_MAX_AREA * 4)


def test_dot_area_bounds_never_shrink_below_tuned_values() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    assert _dot_area_bounds(frame) == (DOT_MIN_AREA, DOT_MAX_AREA)


def test_detect_red_dots_accepts_4k_scale_dot_over_1080p_max_area() -> None:
    # A calibration dot on a 4K frame covers ~4× the 1080p pixel area —
    # radius 62 ≈ 12k px², over the unscaled DOT_MAX_AREA but a real dot
    # at this resolution.
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    _draw_dot(frame, 400, 400, (0, 0, 255), radius=62)

    dots = detect_red_dots(frame)

    assert len(dots) == 1
    assert abs(dots[0][0] - 400) < 3 and abs(dots[0][1] - 400) < 3
