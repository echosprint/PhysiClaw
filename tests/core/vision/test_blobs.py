"""Tests for `physiclaw.core.vision.blobs` — HSV blob centroids.

All tests use synthetic BGR `np.ndarray` images: saturated colored
squares drawn on black exercise the mask → morphology → contour →
centroid pipeline. The bridge-corner detector built on this pipeline is
tested in test_grid_detect.py.
"""

from __future__ import annotations

import numpy as np

from physiclaw.core.vision.blobs import (
    contour_centroid,
    find_all_hsv_blobs,
    find_largest_hsv_blob,
)

# ---------- contour_centroid ----------


def test_contour_centroid_of_square() -> None:
    cnt = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)

    cx, cy = contour_centroid(cnt)

    assert cx == 5.0 and cy == 5.0


def test_contour_centroid_none_for_degenerate_contour() -> None:
    # A single-point contour has zero area (m00 == 0).
    cnt = np.array([[[5, 5]]], dtype=np.int32)

    assert contour_centroid(cnt) is None


# ---------- helpers ----------


def _draw_rect(
    img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color_bgr: tuple
) -> None:
    img[y1:y2, x1:x2] = color_bgr


def _red_lower_upper() -> tuple[list[int], list[int]]:
    # Saturated red, hue 0–10.
    return ([0, 100, 100], [10, 255, 255])


# ---------- find_largest_hsv_blob ----------


def test_find_largest_hsv_blob_returns_none_when_no_match() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    lower, upper = _red_lower_upper()

    assert find_largest_hsv_blob(img, lower, upper) is None


def test_find_largest_hsv_blob_returns_centroid_of_solid_square() -> None:
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    _draw_rect(img, 50, 60, 150, 140, (0, 0, 255))  # red square
    lower, upper = _red_lower_upper()

    cx, cy = find_largest_hsv_blob(img, lower, upper)

    # Center should be ~(100, 100) within a few pixels of OpenCV moments.
    assert 95 < cx < 105
    assert 95 < cy < 105


def test_find_largest_hsv_blob_picks_the_larger_of_two_blobs() -> None:
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    _draw_rect(img, 10, 10, 30, 30, (0, 0, 255))  # small (20×20)
    _draw_rect(img, 200, 100, 280, 180, (0, 0, 255))  # big (80×80)
    lower, upper = _red_lower_upper()

    cx, cy = find_largest_hsv_blob(img, lower, upper)

    # Centroid lands inside the big blob, not the small one.
    assert 200 < cx < 280
    assert 100 < cy < 180


def test_find_largest_hsv_blob_filters_below_min_area() -> None:
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    _draw_rect(img, 100, 100, 105, 105, (0, 0, 255))  # tiny 5×5 blob
    lower, upper = _red_lower_upper()

    # Default min_area=50 — single 25-pixel blob is below threshold.
    assert find_largest_hsv_blob(img, lower, upper, min_area=50) is None
    # Below-area override — same blob is now found.
    assert find_largest_hsv_blob(img, lower, upper, min_area=10) is not None


# ---------- find_all_hsv_blobs ----------


def test_find_all_hsv_blobs_empty_when_no_match() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    lower, upper = _red_lower_upper()

    assert find_all_hsv_blobs(img, lower, upper) == []


def test_find_all_hsv_blobs_returns_centroid_per_qualifying_contour() -> None:
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    _draw_rect(img, 50, 50, 100, 100, (0, 0, 255))
    _draw_rect(img, 200, 50, 250, 100, (0, 0, 255))
    _draw_rect(img, 50, 150, 100, 195, (0, 0, 255))
    lower, upper = _red_lower_upper()

    centroids = find_all_hsv_blobs(img, lower, upper)

    assert len(centroids) == 3


def test_find_all_hsv_blobs_filters_below_min_area() -> None:
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    _draw_rect(img, 50, 50, 100, 100, (0, 0, 255))  # 50×50 — kept
    _draw_rect(img, 200, 50, 210, 60, (0, 0, 255))  # 10×10 — dropped
    lower, upper = _red_lower_upper()

    centroids = find_all_hsv_blobs(img, lower, upper, min_area=500)

    assert len(centroids) == 1
