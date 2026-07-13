"""Tests for `physiclaw.core.vision.preprocess` — shared frame stages."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from physiclaw.core.vision.preprocess import (
    CONFIG,
    crop_to_phone_screen,
    gaussian_blur,
    grayscale,
    phone_screen_crop_box,
    resize_to_max_edge,
    resize_to_width,
    to_hsv,
)


# ---------- grayscale / to_hsv ----------


def test_grayscale_converts_bgr() -> None:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 255)

    gray = grayscale(img)

    assert gray.ndim == 2
    # Red's ITU-R 601 luma weight is 0.299 → 76.
    assert int(gray[0, 0]) == 76


def test_grayscale_passes_through_already_gray() -> None:
    gray = np.full((4, 4), 128, dtype=np.uint8)

    assert grayscale(gray) is gray


def test_to_hsv_matches_cv2() -> None:
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[:, :] = (255, 0, 0)  # pure blue

    hsv = to_hsv(img)

    assert tuple(hsv[0, 0]) == (120, 255, 255)


# ---------- gaussian_blur ----------


def test_gaussian_blur_matches_cv2_square_kernel() -> None:
    rng = np.random.default_rng(3)
    gray = rng.integers(0, 255, size=(32, 32), dtype=np.uint8)

    out = gaussian_blur(gray, 5)

    assert np.array_equal(out, cv2.GaussianBlur(gray, (5, 5), 0))


# ---------- resize_to_max_edge ----------


def test_resize_to_max_edge_downscales_keeping_aspect() -> None:
    frame = np.zeros((2000, 1000, 3), dtype=np.uint8)

    out = resize_to_max_edge(frame, 1024)

    assert out.shape == (1024, 512, 3)


def test_resize_to_max_edge_passes_through_small_frames() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    assert resize_to_max_edge(frame, 1024) is frame


def test_resize_to_max_edge_truncates_short_edge() -> None:
    # Short edge scales by int() truncation — pinned so a rounding change
    # (which would shift every downstream pixel measurement) is caught.
    frame = np.zeros((1001, 3000, 3), dtype=np.uint8)

    out = resize_to_max_edge(frame, 1024)

    assert out.shape == (int(1001 * 1024 / 3000), 1024, 3)


# ---------- resize_to_width ----------


def test_resize_to_width_downscales_wide_frames() -> None:
    frame = np.zeros((100, 960), dtype=np.uint8)

    out = resize_to_width(frame, 480)

    assert out.shape == (50, 480)


def test_resize_to_width_never_upscales() -> None:
    # Upscaling would interpolation-smooth sharp pixels into false blur.
    frame = np.zeros((100, 200), dtype=np.uint8)

    assert resize_to_width(frame, 480) is frame


def test_resize_to_width_rounds_height() -> None:
    # Height scales by round() (not truncation) — pinned like the
    # max-edge test above; sharpness thresholds depend on it.
    frame = np.zeros((101, 960), dtype=np.uint8)

    out = resize_to_width(frame, 480)

    assert out.shape == (round(101 * 480 / 960), 480)


# ---------- phone_screen_crop_box ----------


def test_phone_screen_crop_box_returns_none_when_transforms_missing() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    assert phone_screen_crop_box(frame, None) is None


def test_phone_screen_crop_box_returns_clamped_box_for_in_bounds_rect() -> None:
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    transforms = SimpleNamespace(bbox_to_pixel_rect=lambda b: ((100, 80), (700, 520)))

    out = phone_screen_crop_box(frame, transforms)

    assert out == (100, 80, 700, 520)


def test_phone_screen_crop_box_clamps_box_to_frame_bounds() -> None:
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    # tl beyond top-left, br beyond bottom-right.
    transforms = SimpleNamespace(bbox_to_pixel_rect=lambda b: ((-50, -30), (900, 700)))

    out = phone_screen_crop_box(frame, transforms)

    assert out == (0, 0, 800, 600)


def test_phone_screen_crop_box_returns_none_for_degenerate_rect() -> None:
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    # Both corners at the same point — zero area after clamping.
    transforms = SimpleNamespace(bbox_to_pixel_rect=lambda b: ((100, 100), (100, 100)))

    assert phone_screen_crop_box(frame, transforms) is None


def test_phone_screen_crop_box_handles_inverted_corners() -> None:
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    # Coords swapped: br listed first conceptually.
    transforms = SimpleNamespace(bbox_to_pixel_rect=lambda b: ((700, 520), (100, 80)))

    out = phone_screen_crop_box(frame, transforms)

    assert out == (100, 80, 700, 520)


# ---------- crop_to_phone_screen ----------


def test_crop_to_phone_screen_returns_frame_unchanged_when_transforms_none() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    out = crop_to_phone_screen(frame, transforms=None)

    assert out is frame


def test_crop_to_phone_screen_returns_cropped_region_when_box_fits() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    transforms = SimpleNamespace(bbox_to_pixel_rect=lambda b: ((100, 80), (300, 200)))

    out = crop_to_phone_screen(frame, transforms, max_long_edge=1024)

    assert out.shape == (120, 200, 3)  # (h, w, c) — cropped to box size


def test_crop_to_phone_screen_downscales_when_long_edge_exceeds_cap() -> None:
    frame = np.zeros((900, 700, 3), dtype=np.uint8)
    transforms = SimpleNamespace(bbox_to_pixel_rect=lambda b: ((0, 0), (700, 900)))

    out = crop_to_phone_screen(frame, transforms, max_long_edge=400)

    # Long edge 900 → 400; short edge 700 → int(700 * 400/900) = 311.
    assert max(out.shape[:2]) == 400
    assert out.shape == (400, 311, 3)


def test_crop_to_phone_screen_default_cap_reads_compact_config(monkeypatch) -> None:
    # No explicit max_long_edge → the single [compact] max_image_edge_px
    # knob decides how large a view the LLM sees.
    monkeypatch.setattr(CONFIG.compact, "max_image_edge_px", 400)
    frame = np.zeros((900, 700, 3), dtype=np.uint8)
    transforms = SimpleNamespace(bbox_to_pixel_rect=lambda b: ((0, 0), (700, 900)))

    out = crop_to_phone_screen(frame, transforms)

    assert max(out.shape[:2]) == 400
