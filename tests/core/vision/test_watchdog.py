"""Tests for `physiclaw.core.vision.watchdog`."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from freezegun import freeze_time

from physiclaw.core.vision import watchdog
from physiclaw.core.vision.watchdog import (
    BADGE_HUE_RANGES,
    BADGE_MIN_AREA,
    BADGE_S_MIN,
    BADGE_V_MIN,
    EMA_FAST,
    EMA_SLOW,
    EMA_STALE,
    IDLE_INTERVAL,
    MEAN_DROP_GUARD,
    MEAN_INCREASE,
    STD_INCREASE,
    Watchdog,
    WORK_HOURS,
    ZONES,
    _check_badge,
    _check_content,
    _crop_zones,
    _ema_update,
    _gray,
)


# ---------- _gray ----------


def test_gray_converts_bgr_to_grayscale() -> None:
    bgr = np.full((10, 10, 3), [255, 0, 0], dtype=np.uint8)  # blue

    out = _gray(bgr)

    assert out.ndim == 2
    assert out.shape == (10, 10)


# ---------- _check_content ----------


def test_check_content_no_change() -> None:
    a = np.full((20, 20, 3), 100, dtype=np.uint8)

    out = _check_content(a, a)

    assert out["wake"] is False
    assert out["std_delta"] == 0.0
    assert out["mean_delta"] == 0.0


def test_check_content_wake_on_mean_increase() -> None:
    slow = np.full((20, 20, 3), 100, dtype=np.uint8)
    fast = np.full((20, 20, 3), 100 + int(MEAN_INCREASE) + 5, dtype=np.uint8)

    out = _check_content(slow, fast)

    assert out["wake"] is True


def test_check_content_wake_on_std_increase() -> None:
    slow = np.full((20, 20, 3), 100, dtype=np.uint8)
    # Mix of values for high std.
    fast = slow.copy()
    fast[::2] = 0
    fast[1::2] = 200

    out = _check_content(slow, fast)

    assert out["wake"] is True
    assert out["std_delta"] > STD_INCREASE


def test_check_content_wake_on_dim_scene_below_absolute_threshold() -> None:
    # Dim room (baseline mean ~5): a screen lighting up adds only ~3 gray
    # levels — under the absolute MEAN_INCREASE, but a big *relative* rise, so
    # the adaptive threshold still wakes. This is the dark-room miss we fix.
    slow = np.full((20, 20, 3), 5, dtype=np.uint8)
    fast = np.full((20, 20, 3), 8, dtype=np.uint8)

    out = _check_content(slow, fast)

    assert out["mean_delta"] == 3.0
    assert out["mean_delta"] < MEAN_INCREASE  # would miss the fixed threshold
    assert out["mean_thr"] < MEAN_INCREASE    # adaptive threshold dropped
    assert out["wake"] is True


def test_check_content_no_wake_on_small_shift_in_bright_scene() -> None:
    # Bright baseline (mean 150): a tiny +3 shift stays under both the absolute
    # floor and the proportional threshold (which equals the floor when bright).
    slow = np.full((20, 20, 3), 150, dtype=np.uint8)
    fast = np.full((20, 20, 3), 153, dtype=np.uint8)

    out = _check_content(slow, fast)

    assert out["mean_thr"] == MEAN_INCREASE  # proportional ≥ floor → floor wins
    assert out["wake"] is False


def test_check_content_no_wake_on_brightness_collapse_despite_std_jump() -> None:
    # Screen dimming / app closing: the lower half goes much darker (big mean
    # DROP) and the transition also spikes the std past its threshold. Without
    # the guard this false-wakes (the real-log case: std +4.1, mean -49.5). The
    # brightness-collapse guard suppresses the std wake.
    slow = np.full((20, 20, 3), 100, dtype=np.uint8)  # bright, uniform
    fast = slow.copy()
    fast[::2] = 0
    fast[1::2] = 60  # darker overall (mean ~30) but more structure (std up)

    out = _check_content(slow, fast)

    assert out["std_delta"] > STD_INCREASE            # std alone would fire
    assert out["mean_delta"] < -MEAN_DROP_GUARD       # but the scene collapsed
    assert out["wake"] is False                       # so no wake


# ---------- _check_badge ----------


def test_check_badge_wake_on_red_pixel_increase() -> None:
    slow = np.zeros((20, 20, 3), dtype=np.uint8)
    fast = slow.copy()
    # OpenCV uses BGR — pure red is (0, 0, 255).
    fast[:8, :8] = (0, 0, 255)

    out = _check_badge(slow, fast)

    assert out["wake"] is True
    assert out["warm_delta"] > BADGE_MIN_AREA


def test_check_badge_wake_on_warm_cast_orange() -> None:
    # Warm room light shifts the camera's red toward orange (HSV hue ~15) —
    # outside the old strict red band (0–10), caught by the widened warm band.
    slow = np.zeros((20, 20, 3), dtype=np.uint8)
    fast = slow.copy()
    fast[:8, :8] = (0, 128, 255)  # BGR orange → hue ~15

    out = _check_badge(slow, fast)

    assert out["wake"] is True
    assert out["warm_delta"] > BADGE_MIN_AREA


def test_check_badge_wake_on_dim_desaturated_red() -> None:
    # A dark room desaturates and dims the badge (S≈94, V≈95) — under the old
    # S/V≥100 floors, caught by the relaxed BADGE_S_MIN / BADGE_V_MIN.
    slow = np.zeros((20, 20, 3), dtype=np.uint8)
    fast = slow.copy()
    fast[:8, :8] = (60, 60, 95)  # dim, low-saturation red

    out = _check_badge(slow, fast)

    assert out["wake"] is True


def test_check_badge_no_change() -> None:
    a = np.zeros((10, 10, 3), dtype=np.uint8)

    out = _check_badge(a, a)

    assert out["wake"] is False
    assert out["warm_delta"] == 0


# ---------- _ema_update ----------


def test_ema_update_blends_frames() -> None:
    ema = np.zeros((4, 4), dtype=np.float32)
    frame = np.full((4, 4), 100, dtype=np.uint8)

    out = _ema_update(ema, frame, alpha=0.5)

    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.full((4, 4), 50.0))


# ---------- _crop_zones ----------


def _fake_transforms(*, w: int = 100, h: int = 200):
    """Transforms whose pct_to_cam_pixel maps (px, py) → (px*w, py*h)."""
    t = MagicMock()
    t.pct_to_cam_pixel.side_effect = lambda px, py: (int(px * w), int(py * h))
    return t


def test_crop_zones_returns_three_crops() -> None:
    frame = np.zeros((200, 100, 3), dtype=np.uint8)

    crops = _crop_zones(frame, _fake_transforms(w=100, h=200))

    assert crops is not None
    assert len(crops) == len(ZONES)
    # Each crop has positive area.
    for c in crops:
        assert c.size > 0


def test_crop_zones_returns_none_on_empty_crop() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    t = MagicMock()
    # Both corners map outside frame → empty crop.
    t.pct_to_cam_pixel.side_effect = lambda px, py: (1000, 1000)

    assert _crop_zones(frame, t) is None


# ---------- Watchdog ----------


def _frame() -> np.ndarray:
    return np.zeros((200, 100, 3), dtype=np.uint8)


def test_watchdog_init_state() -> None:
    w = Watchdog()

    assert w._ema is None
    assert w._poll_time == 0.0


def test_watchdog_first_poll_initializes_ema() -> None:
    w = Watchdog()

    out = w.poll(_frame(), _fake_transforms())

    assert out["wake"] is False
    assert out["reason"] == "ema initialized"
    assert w._ema is not None


def test_watchdog_returns_no_wake_when_zones_unavailable() -> None:
    w = Watchdog()
    t = MagicMock()
    t.pct_to_cam_pixel.side_effect = lambda px, py: (1000, 1000)

    out = w.poll(np.zeros((10, 10, 3), dtype=np.uint8), t)

    assert out == {"wake": False, "reason": ""}


def test_watchdog_reinitializes_after_stale_gap(mocker) -> None:
    w = Watchdog()
    w.poll(_frame(), _fake_transforms())
    # Jump time forward past EMA_STALE.
    mocker.patch.object(
        watchdog.time, "monotonic", return_value=w._poll_time + EMA_STALE + 1,
    )

    out = w.poll(_frame(), _fake_transforms())

    assert out["reason"] == "ema initialized"


def test_watchdog_steady_frames_no_wake() -> None:
    w = Watchdog()
    t = _fake_transforms()
    f = _frame()
    w.poll(f, t)

    out = w.poll(f, t)

    assert out["wake"] is False


def test_watchdog_banner_change_wakes() -> None:
    w = Watchdog()
    t = _fake_transforms()
    base = np.full((200, 100, 3), 100, dtype=np.uint8)
    w.poll(base, t)
    # Modify only the banner zone (y 0-0.1 → rows 0-19 of 200).
    bright = base.copy()
    bright[:20] = 200

    out = w.poll(bright, t)

    assert out["wake"] is True
    assert "banner" in out["reason"]


def test_watchdog_bottom_change_wakes() -> None:
    w = Watchdog()
    t = _fake_transforms()
    base = np.full((200, 100, 3), 100, dtype=np.uint8)
    w.poll(base, t)
    bright = base.copy()
    # Bottom zone is y 0.5-1.0 → rows 100-199.
    bright[100:] = 200

    out = w.poll(bright, t)

    assert out["wake"] is True
    assert "lower half" in out["reason"]


def test_watchdog_dock_red_badge_wakes_on_first_poll() -> None:
    # Badge is checked on the RAW dock crop vs the slow baseline, so a newly
    # appeared badge wakes on the very first poll — no waiting for the fast EMA
    # to saturate (that was the miss: a badge arriving with a banner never got
    # the follow-up polls). See the raw-vs-slow note in watchdog.poll.
    w = Watchdog()
    t = _fake_transforms()
    base = np.full((200, 100, 3), 100, dtype=np.uint8)
    w.poll(base, t)  # init EMA, no badge
    badged = base.copy()
    badged[170:200, :] = (0, 0, 255)  # inside dock zone y 0.80-1.0 (rows 160-199)

    out = w.poll(badged, t)

    assert out["wake"] is True
    assert "red badge" in out["reason"]
    assert out["dock"]["warm_delta"] > BADGE_MIN_AREA


@pytest.mark.parametrize("x0", [5, 30, 55, 80])  # 4 dock-slot x positions
def test_watchdog_badge_wakes_in_any_dock_slot(x0: int) -> None:
    # The dock zone is full-width, so a badge in any of the 4 dock slots counts
    # — the IM app can be pinned to any slot, on any iPhone model.
    w = Watchdog()
    t = _fake_transforms()
    base = np.full((200, 100, 3), 100, dtype=np.uint8)
    w.poll(base, t)  # init, no badge
    badged = base.copy()
    badged[165:185, x0:x0 + 15] = (0, 0, 255)  # small badge in one slot (dock rows 160-199)

    out = w.poll(badged, t)

    assert out["wake"] is True
    assert "red badge" in out["reason"]


def test_watchdog_idle_fallback_during_work_hours() -> None:
    w = Watchdog()

    with freeze_time("2026-04-28 10:00:00"):
        # First poll initializes EMA.
        w.poll(_frame(), _fake_transforms())
        # Force last_wake to long ago.
        w._last_wake = time.monotonic() - IDLE_INTERVAL - 100

        out = w.poll(_frame(), _fake_transforms())

    assert out["wake"] is True
    assert "idle check-in" in out["reason"]


def test_watchdog_idle_fallback_outside_work_hours() -> None:
    w = Watchdog()
    w.poll(_frame(), _fake_transforms())
    w._last_wake = time.monotonic() - IDLE_INTERVAL - 100

    with freeze_time("2026-04-28 22:00:00"):
        out = w.poll(_frame(), _fake_transforms())

    # Outside work hours — no idle wake.
    assert out["wake"] is False


def test_watchdog_resets_last_wake_on_real_wake() -> None:
    w = Watchdog()
    t = _fake_transforms()
    base = np.full((200, 100, 3), 100, dtype=np.uint8)
    w.poll(base, t)
    bright = base.copy()
    bright[:20] = 200

    w.poll(bright, t)

    # last_wake updated to most recent poll time.
    assert w._last_wake == w._poll_time


def test_watchdog_constants_unchanged() -> None:
    # Defensive guard: changing thresholds is a behavior change.
    assert STD_INCREASE == 4.0
    assert MEAN_INCREASE == 4.0
    assert MEAN_DROP_GUARD == 15.0
    assert BADGE_MIN_AREA == 30
    assert BADGE_HUE_RANGES == [(0, 25), (155, 180)]
    assert BADGE_S_MIN == 70
    assert BADGE_V_MIN == 70
    assert IDLE_INTERVAL == 1800.0
    assert WORK_HOURS == [(9, 12), (14, 17)]
    assert ZONES == [(0.0, 0.1), (0.5, 1.0), (0.80, 1.0)]
    assert 0 < EMA_FAST < 1
    assert 0 < EMA_SLOW < EMA_FAST
