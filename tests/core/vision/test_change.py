"""Tests for `physiclaw.core.vision.change` — gesture frame diffing."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from physiclaw.core.vision import change


def _frame(h: int = 200, w: int = 100, value: int = 128) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


# ---------- change_ratio ----------


def test_identical_frames_ratio_zero() -> None:
    a = _frame()
    assert change.change_ratio(a, a.copy()) == 0.0


def test_sensor_noise_below_threshold_ignored() -> None:
    a = _frame()
    rng = np.random.default_rng(42)
    noise = rng.integers(-10, 11, size=a.shape, dtype=np.int16)
    b = np.clip(a.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # ±10 gray levels everywhere is under NOISE_THRESHOLD after blur.
    assert change.change_ratio(a, b) == 0.0


def test_large_region_change_detected() -> None:
    a = _frame()
    b = a.copy()
    b[80:160, 20:80] = 250  # a dialog-sized bright region
    assert change.change_ratio(a, b) > change.RATIO_THRESHOLD


def test_status_bar_change_cropped_out() -> None:
    a = _frame()
    b = a.copy()
    b[:8, :] = 250  # clock ticking over, top 4% of a 200px frame
    assert change.change_ratio(a, b) == 0.0


def test_mismatched_shapes_resized_not_raised() -> None:
    a = _frame(200, 100)
    b = _frame(202, 101)  # re-crop rounding drift
    assert change.change_ratio(a, b) < change.RATIO_THRESHOLD


# ---------- frames_changed ----------


def test_frames_changed_true_on_page_transition() -> None:
    a = _frame(value=128)
    b = _frame(value=30)  # whole screen different
    assert change.frames_changed(a, b) is True


def test_frames_changed_false_on_static_screen() -> None:
    a = _frame()
    assert change.frames_changed(a, a.copy()) is False


def test_small_localized_change_detected() -> None:
    # A cart badge / one quantity digit covers a fraction of a percent of
    # the frame — below the DISTRIBUTED ratio, but a coherent blob the
    # LOCALIZED trigger must catch (this is the sensitivity requirement).
    a = _frame(1000, 500, value=128)
    b = a.copy()
    b[100:124, 100:124] = 240  # 24x24 badge, ~0.12% of the frame
    assert change.change_ratio(a, b) < change.RATIO_THRESHOLD  # ratio alone misses it
    assert change.frames_changed(a, b) is True  # blob catches it


def test_thin_cursor_change_ignored() -> None:
    # A blinking text cursor (~2px wide) is erased by the morphological
    # opening — it must not read as a change.
    a = _frame(1000, 500, value=128)
    b = a.copy()
    b[100:118, 100:102] = 240  # 2px-wide, 18px-tall cursor
    assert change.frames_changed(a, b) is False


@pytest.mark.parametrize("frac", [0.0, change.STATUS_BAR_FRAC])
def test_full_frame_change_detected_regardless_of_crop(frac: float) -> None:
    a = _frame()
    b = np.full_like(a, 240)
    assert change.frames_changed(a, b) is True


# ---------- robustness: vibration / autofocus / exposure ----------


def _textured(h: int = 200, w: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    return cv2.GaussianBlur(frame, (5, 5), 0)


def test_small_vibration_shift_not_a_change() -> None:
    # The rig shaking a few px between frames must not read as changed.
    a = _textured()
    b = np.roll(a, 3, axis=1)  # 3px horizontal shift, same content
    assert change.frames_changed(a, b) is False


def test_large_rig_shift_is_unreliable_not_changed() -> None:
    a = _textured()
    b = np.roll(a, 20, axis=1)  # beyond MAX_ALIGN_SHIFT
    assert change.frames_changed(a, b) is None


def test_ae_shift_within_clamp_ignored() -> None:
    # Global exposure re-metering (+35 gray) on identical content: the
    # clamp corrects 25, leaving a 10-level residue — under threshold.
    a = _textured()
    b = np.clip(a.astype(np.int16) + 35, 0, 255).astype(np.uint8)
    assert change.frames_changed(a, b) is False


def test_page_transition_survives_brightness_clamp() -> None:
    # A dark-to-light page (mean shift ~112) is CONTENT, not exposure —
    # the clamp must not normalize it away.
    a = _frame(value=128)
    b = _frame(value=240)
    assert change.frames_changed(a, b) is True


def test_shift_plus_real_change_still_detected() -> None:
    # Vibration AND a dialog: alignment removes the shift, the dialog
    # survives the diff.
    a = _textured()
    b = np.roll(a, 2, axis=0).copy()
    b[80:160, 20:80] = 250
    assert change.frames_changed(a, b) is True


# ---------- robustness: alignment on realistic content ----------


def test_textured_page_transition_detected() -> None:
    # Two DIFFERENT textured screens = a real page transition. Phase
    # correlation returns a garbage peak on unrelated content — that
    # must read as low-confidence (diff unaligned → change detected),
    # never as "rig moved" (None). This was a real bug: flat-frame
    # transition tests skip alignment and masked it.
    a = _textured(400, 200, seed=1)
    b = _textured(400, 200, seed=2)
    assert change.frames_changed(a, b) is True


def test_periodic_shift_beyond_window_is_unreliable_not_changed() -> None:
    # A rig move just beyond MAX_ALIGN_SHIFT on periodic content: the
    # phase peak spreads (harmonics) and the template peak pins to the
    # search-window border — must read unreliable (None), never a false
    # `changed` that would reset the stuck guard.
    row = np.tile(np.repeat(np.array([60, 200], dtype=np.uint8), 20), 10)[:400]
    patt = np.dstack([np.tile(row[:, None], (1, 200))] * 3)
    for shift in (10, 12):
        assert change.frames_changed(patt, np.roll(patt, shift, axis=0)) is None


def test_periodic_content_with_vibration_not_changed() -> None:
    # Keyboard-row-like stripes + a 3px shake: the phase peak spreads
    # across harmonics (low response); the anchored template match must
    # find the true small shift instead of reading the shake as change.
    row = np.tile(np.repeat(np.array([60, 200], dtype=np.uint8), 20), 10)[:400]
    patt = np.dstack([np.tile(row[:, None], (1, 200))] * 3)
    shaken = np.roll(patt, 3, axis=0)
    assert change.frames_changed(patt, shaken) is False


# ---------- capture-scale normalization ----------


def test_prepare_normalizes_oversized_frames_to_work_scale() -> None:
    # Crops now arrive at the configurable view cap (> 1024); every pixel
    # constant in the module was tuned at the 1024 long edge, so _prepare
    # must bring larger inputs back to that working scale.
    before = _frame(h=2048, w=1000)
    after = _frame(h=2048, w=1000)

    pair = change._prepare(before, after)

    assert pair is not None
    a, b = pair
    assert max(a.shape) <= change._WORK_LONG_EDGE
    assert max(b.shape) <= change._WORK_LONG_EDGE


def test_localized_change_verdict_stable_across_capture_scale() -> None:
    # The same physical badge, seen through a 1080p and a 4K capture,
    # must produce the same verdict — normalization makes the diff
    # resolution-independent.
    before = _frame(h=1000, w=500)
    after = _frame(h=1000, w=500)
    after[500:540, 200:240] = 255  # coherent badge-sized region

    assert change.frames_changed(before, after) is True

    before_2x = cv2.resize(before, (1000, 2000), interpolation=cv2.INTER_NEAREST)
    after_2x = cv2.resize(after, (1000, 2000), interpolation=cv2.INTER_NEAREST)

    assert change.frames_changed(before_2x, after_2x) is True
