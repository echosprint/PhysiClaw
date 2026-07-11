"""Tests for `physiclaw.core.vision.colors` — HSV masks, redness, and
the shared color vocabularies.

These primitives were previously covered only indirectly through the
blob detectors; the direct tests here pin the seam-wrapping OR, the
redness map's clipping, and the calibrated range tables.
"""

from __future__ import annotations

import cv2
import numpy as np

from physiclaw.core.vision.colors import (
    CORNER_HSV_RANGES,
    hsv_mask,
    red_ranges,
    redness,
)


def _hsv_of(color_bgr: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :] = color_bgr
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# ---------- hsv_mask ----------


def test_hsv_mask_single_range_matches_in_band_pixels() -> None:
    hsv = _hsv_of((0, 255, 0))  # pure green, H≈60

    mask = hsv_mask(hsv, [([40, 100, 100], [80, 255, 255])])

    assert mask.min() == 255  # every pixel matched


def test_hsv_mask_single_range_rejects_out_of_band_pixels() -> None:
    hsv = _hsv_of((255, 0, 0))  # pure blue, H≈120

    mask = hsv_mask(hsv, [([40, 100, 100], [80, 255, 255])])

    assert mask.max() == 0


def test_hsv_mask_ors_ranges_across_the_hue_seam() -> None:
    # Pure red sits at H=0; a slightly blue-shifted red lands near H=177.
    # One hsv_mask call with red_ranges() must catch both.
    low_side = _hsv_of((0, 0, 255))
    high_side = _hsv_of((40, 0, 255))

    ranges = red_ranges()

    assert hsv_mask(low_side, ranges).min() == 255
    assert hsv_mask(high_side, ranges).min() == 255


# ---------- redness ----------


def test_redness_high_for_red_zero_for_gray() -> None:
    img = np.zeros((2, 4, 3), dtype=np.uint8)
    img[:, :2] = (0, 0, 200)  # red: R - max(G,B) = 200
    img[:, 2:] = (128, 128, 128)  # gray: R - max(G,B) = 0

    r = redness(img)

    assert r[0, 0] == 200
    assert r[0, 2] == 0


def test_redness_clips_negative_to_zero() -> None:
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[:, :] = (255, 0, 0)  # pure blue: R - max(G,B) = -255

    assert redness(img).max() == 0


def test_redness_catches_desaturated_pink() -> None:
    # A washed-out pink (the camera's rendering of a small red dot on a
    # bright screen) still has a clear red lead — the reason this exists.
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[:, :] = (180, 180, 230)

    assert redness(img).min() == 50


# ---------- red_ranges ----------


def test_red_ranges_covers_both_sides_of_the_seam() -> None:
    ranges = red_ranges()

    assert ranges == [
        ([0, 100, 100], [10, 255, 255]),
        ([170, 100, 100], [180, 255, 255]),
    ]


def test_red_ranges_applies_custom_sv_floor() -> None:
    ranges = red_ranges(80, 90)

    assert ranges[0][0] == [0, 80, 90]
    assert ranges[1][0] == [170, 80, 90]


# ---------- CORNER_HSV_RANGES ----------


def test_corner_ranges_cover_all_four_colors() -> None:
    assert set(CORNER_HSV_RANGES) == {"R", "G", "B", "M"}


def test_corner_ranges_match_their_pure_colors() -> None:
    pure = {
        "R": (0, 0, 255),
        "G": (0, 255, 0),
        "B": (255, 0, 0),
        "M": (255, 0, 255),
    }
    for name, bgr in pure.items():
        assert hsv_mask(_hsv_of(bgr), CORNER_HSV_RANGES[name]).min() == 255, name
