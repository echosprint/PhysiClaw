"""Tests for `physiclaw.core.vision.screen_match`."""

from __future__ import annotations

import numpy as np

from physiclaw.core.vision.screen_match import (
    MatchResult,
    detect_dark_overlay,
    frames_differ,
    match_best,
    match_screen,
)


def _textured_frame(seed: int = 0, h: int = 200, w: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


# ---------- match_screen ----------


def test_match_screen_identical_images_match() -> None:
    frame = _textured_frame(seed=42)

    result = match_screen(frame, frame)

    assert result.matched is True
    assert result.confidence > 0
    assert result.homography is not None
    assert result.inliers > 0


def test_match_screen_returns_matchresult() -> None:
    frame = _textured_frame(seed=1)

    result = match_screen(frame, frame)

    assert isinstance(result, MatchResult)


def test_match_screen_blank_frames_no_features() -> None:
    blank = np.zeros((200, 200, 3), dtype=np.uint8)

    result = match_screen(blank, blank)

    # ORB finds no descriptors → early return with matched=False.
    assert result.matched is False
    assert result.good_matches == 0


def test_match_screen_different_textures_below_threshold() -> None:
    a = _textured_frame(seed=1)
    b = _textured_frame(seed=99)

    result = match_screen(a, b)

    # Different random textures shouldn't share enough features.
    assert result.matched is False


def test_match_screen_respects_min_matches_param() -> None:
    frame = _textured_frame(seed=7)

    # Force a very high threshold — even an identical-frame match fails.
    result = match_screen(frame, frame, min_matches=100000)

    assert result.matched is False


# ---------- match_best ----------


def test_match_best_picks_highest_confidence() -> None:
    target = _textured_frame(seed=5)
    distractor = _textured_frame(seed=999)
    references = {"distractor": distractor, "target": target}

    name, result = match_best(target, references)

    assert name == "target"
    assert result.matched is True


def test_match_best_returns_none_when_no_match() -> None:
    cam = _textured_frame(seed=1)
    references = {
        "alt1": _textured_frame(seed=99),
        "alt2": _textured_frame(seed=42),
    }

    name, result = match_best(cam, references)

    assert name is None
    assert result.matched is False


def test_match_best_handles_empty_references() -> None:
    cam = _textured_frame(seed=1)

    name, result = match_best(cam, {})

    assert name is None
    assert result.confidence == 0.0


# ---------- frames_differ ----------


def test_frames_differ_identical_returns_false() -> None:
    a = _textured_frame(seed=3)

    assert not frames_differ(a, a)


def test_frames_differ_different_brightness_returns_true() -> None:
    dark = np.full((200, 200, 3), 20, dtype=np.uint8)
    bright = np.full((200, 200, 3), 220, dtype=np.uint8)

    assert frames_differ(dark, bright)


def test_frames_differ_threshold_lets_small_changes_pass() -> None:
    a = np.full((200, 200, 3), 100, dtype=np.uint8)
    b = a.copy()
    b[0, 0] = (200, 200, 200)  # tiny change

    assert not frames_differ(a, b, threshold=0.5)


# ---------- detect_dark_overlay ----------


def test_detect_dark_overlay_bright_frame_false() -> None:
    bright = np.full((100, 100, 3), 220, dtype=np.uint8)

    assert not detect_dark_overlay(bright)


def test_detect_dark_overlay_mostly_dark_frame_true() -> None:
    dark = np.full((100, 100, 3), 10, dtype=np.uint8)

    assert detect_dark_overlay(dark)


def test_detect_dark_overlay_partial_dark_below_ratio() -> None:
    frame = np.full((100, 100, 3), 220, dtype=np.uint8)
    # Only 10% of pixels dark — below default 0.3 threshold.
    frame[:10, :] = 10

    assert not detect_dark_overlay(frame)


def test_detect_dark_overlay_threshold_can_be_relaxed() -> None:
    frame = np.full((100, 100, 3), 220, dtype=np.uint8)
    # 20% dark.
    frame[:20, :] = 10

    assert detect_dark_overlay(frame, dark_ratio_threshold=0.1)


def test_detect_dark_overlay_brightness_threshold_can_be_tightened() -> None:
    frame = np.full((100, 100, 3), 100, dtype=np.uint8)

    # With a high brightness cutoff, the whole frame counts as "dark".
    assert detect_dark_overlay(frame, brightness_threshold=200)
    # With a very low cutoff, nothing counts.
    assert not detect_dark_overlay(frame, brightness_threshold=10)
