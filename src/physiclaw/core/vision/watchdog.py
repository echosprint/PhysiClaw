"""Watchdog — detect new notifications on the phone screen.

Watches three zones (skipping AOD clock at y 0.1–0.5):
  - Banner (y 0.0–0.1):  notification banners from top
  - Bottom (y 0.5–1.0):  lock-screen content, app grid
  - Dock   (y 0.80–1.0): red badge on any dock app — FULL WIDTH, so a badge in
                         any of the 4 dock slots counts; the top starts at 0.80
                         (not the ~0.88 a badge sits at) for margin across
                         iPhone models, where the dock's screen fraction varies.

Uses fast (5s) and slow (20s) EMAs of raw pixels. Fires when the fast
EMA diverges from the slow: std or mean increase for content zones,
red pixel increase for dock. Idle fallback wakes every 30 min during
work hours.
"""

import datetime as dt
import logging
import math
import threading
import time

import numpy as np

from physiclaw.common.config import CONFIG
from physiclaw.core.vision.colors import hsv_mask
from physiclaw.core.vision.preprocess import grayscale, resize_to_width, to_hsv

log = logging.getLogger(__name__)

# --- Detection thresholds ---
# Seeded from `[vision]` config (deliberate CONFIG read at import, same
# tradeoff as preprocess.py) so a rig with a different camera/lighting
# can re-tune the wake sensitivity without a source edit.
STD_INCREASE = CONFIG.vision.wake_std_increase
MEAN_INCREASE = CONFIG.vision.wake_mean_increase
# Min new warm-pixels for a badge wake. A well-lit badge at a normal camera
# distance is ~1000+ px, but a farther camera shrinks it (area ~ 1/distance²)
# and dim light + MJPEG chroma-subsampling desaturate the edges, dropping the
# count. Kept low (biased to sensitivity): a spurious wake is cheap, a missed
# message is not. Detection is delta-based (raw vs slow baseline), so static
# warm content — red icons, pink wallpaper — cancels and doesn't eat the budget.
BADGE_MIN_AREA = CONFIG.vision.badge_min_area
# The dock badge is red on the phone, but a camera under warm/yellow room light
# renders it orange/pink, and in the dark it desaturates and dims. So match a
# widened WARM hue band (covers the white-balance cast both ways across the
# 0/180 seam) with relaxed saturation/value floors, instead of a strict,
# well-lit red window that misses it off-white-balance. Keying on the increase
# vs the baseline keeps it specific — static dock colours cancel, only a newly
# appeared vivid mark counts.
BADGE_HUE_RANGES = [
    (0, 25),
    (155, 180),
]  # red→orange (warm cast) and red→pink (cool cast)
BADGE_S_MIN = 70
BADGE_V_MIN = 70
# Illumination-adaptive mean threshold: in a dim scene the same event (a screen
# lighting up) is a smaller absolute delta, so also fire when the brightness
# rises by a fraction of the scene's own level. The effective threshold is the
# LOWER of the absolute floor and this proportional one — bright rooms keep the
# fixed floor, dim rooms get a smaller, more sensitive one. Averaging the zone's
# mean cancels the high sensor noise of low light, so this doesn't false-wake on
# noise (a real screen-on shifts the mean coherently; noise doesn't).
MEAN_RATIO = 0.20
REL_FLOOR = 6.0  # keeps the proportional threshold off ~0 on a near-black frame
# A new notification adds light/structure; it does NOT collapse the scene's
# brightness. A big mean DROP is the opposite event — the screen dimming, an app
# closing, auto-lock — and its transition also blips the std just over the
# threshold, false-waking the agent to nothing. So gate the std-wake path: don't
# fire when the mean fell by more than this. The mean-rise path is unaffected
# (it already needs a positive delta), so a screen lighting up still wakes.
MEAN_DROP_GUARD = 15.0

# --- EMA parameters ---
EMA_FAST = 1 - math.exp(-1 / 5)  # ~0.18, 5s memory
EMA_SLOW = 1 - math.exp(-1 / 20)  # ~0.05, 20s memory
EMA_STALE = 5.0  # re-init if no poll for this long (covers react cooldown)

# --- Idle fallback ---
IDLE_INTERVAL = 1800.0  # 30 min
WORK_HOURS = [(9, 12), (14, 17)]

# --- Screen zones (y0, y1); cropped full-width (x 0→1) ---
# Dock is 0.80–1.0: a real badge sits ~0.88, but the dock's screen fraction
# shifts across iPhone models (home-indicator vs home-button), so start higher
# for margin. Full width covers a badge in any of the 4 dock slots. The extra
# top strip adds no cost — detection is delta-based, so static content cancels.
ZONES = [(0.0, 0.1), (0.5, 1.0), (0.80, 1.0)]


# --- Helpers ---


def _check_content(slow: np.ndarray, fast: np.ndarray) -> dict:
    """Detect new visual content via std/mean divergence. The mean threshold
    adapts to the scene brightness so a screen lighting up in a dim room still
    trips it (see MEAN_RATIO); the std threshold is absolute but gated against a
    brightness collapse (see MEAN_DROP_GUARD) so the screen dimming/off — which
    also blips the std — doesn't false-wake."""
    sg, fg = grayscale(slow), grayscale(fast)
    s_mean = float(np.mean(sg))
    std_delta = round(float(np.std(fg)) - float(np.std(sg)), 1)
    mean_delta = round(float(np.mean(fg)) - s_mean, 1)
    mean_thr = min(MEAN_INCREASE, MEAN_RATIO * (s_mean + REL_FLOOR))
    std_wake = std_delta > STD_INCREASE and mean_delta > -MEAN_DROP_GUARD
    return {
        "std_delta": std_delta,
        "mean_delta": mean_delta,
        "mean_thr": round(mean_thr, 1),
        "wake": std_wake or mean_delta > mean_thr,
    }


def _check_badge(slow: np.ndarray, fast: np.ndarray) -> dict:
    """Detect a new dock badge as an increase in warm-vivid pixels. Uses a
    widened warm hue band + relaxed S/V so the badge is still caught when warm
    or dim room light shifts the camera's red toward orange/pink or desaturates
    it (see BADGE_HUE_RANGES / BADGE_S_MIN / BADGE_V_MIN)."""
    ranges = [
        ([lo, BADGE_S_MIN, BADGE_V_MIN], [hi, 255, 255]) for lo, hi in BADGE_HUE_RANGES
    ]

    def warm(f):
        return int(np.count_nonzero(hsv_mask(to_hsv(f), ranges)))

    delta = warm(fast) - warm(slow)
    return {"warm_delta": delta, "wake": delta > BADGE_MIN_AREA}


def _ema_update(ema: np.ndarray, frame: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * frame.astype(np.float32) + (1 - alpha) * ema


# Zone crops wider than this are downscaled before the EMAs. The badge
# check counts pixels against BADGE_MIN_AREA, so its sensitivity is tied
# to the crop's pixel scale: thresholds were tuned on 1080p captures,
# where the phone screen spans roughly 500–900 camera pixels — those
# rigs pass through untouched (resize_to_width never upscales), while
# 2K/4K captures land back on the tuned scale. Also caps the 1 Hz EMA
# cost at 1080p-era levels regardless of capture resolution. The
# std/mean content checks are distribution-shaped and don't care.
_ZONE_WORK_WIDTH = 900


def _crop_zones(frame, transforms) -> list[np.ndarray] | None:
    """Crop ZONES from camera frame using calibration transforms,
    normalized to at most ``_ZONE_WORK_WIDTH`` wide."""
    h, w = frame.shape[:2]
    crops = []
    for y0, y1 in ZONES:
        tl = transforms.pct_to_cam_pixel(0.0, y0)
        br = transforms.pct_to_cam_pixel(1.0, y1)
        crop = frame[
            max(0, min(tl[1], h)) : max(0, min(br[1], h)),
            max(0, min(tl[0], w)) : max(0, min(br[0], w)),
        ]
        if not crop.size:
            return None
        crops.append(resize_to_width(crop, _ZONE_WORK_WIDTH))
    return crops


# --- Watchdog ---


class Watchdog:
    """EMA-based wake detector. Thread-safe, 1 Hz polling."""

    def __init__(self):
        self._ema = None  # ((fast, slow), ...) per zone, float32
        self._poll_time = 0.0
        self._last_wake = time.monotonic()
        self._lock = threading.Lock()

    def poll(self, frame: np.ndarray, transforms) -> dict:
        """Feed a camera frame. Returns {wake, reason, banner, bottom, dock}."""
        NO = {"wake": False, "reason": ""}

        crops = _crop_zones(frame, transforms)
        if crops is None:
            return NO
        now = time.monotonic()

        with self._lock:
            if self._ema is None or (now - self._poll_time) > EMA_STALE:
                self._ema = tuple(
                    (c.astype(np.float32), c.astype(np.float32)) for c in crops
                )
                self._poll_time = now
                self._last_wake = now
                return {**NO, "reason": "ema initialized"}

            self._ema = tuple(
                (_ema_update(f, c, EMA_FAST), _ema_update(s, c, EMA_SLOW))
                for (f, s), c in zip(self._ema, crops)
            )
            self._poll_time = now
            ema = self._ema

        (bf, bs), (tf, ts), (_, ds) = ema
        banner_d = _check_content(bs.astype(np.uint8), bf.astype(np.uint8))
        bottom_d = _check_content(ts.astype(np.uint8), tf.astype(np.uint8))
        # Badge: RAW dock crop vs the slow-EMA baseline (not fast-vs-slow like
        # the content zones). A badge appears discretely; in the fast EMA it's
        # only ~EMA_FAST (~18%) blended on the poll it appears -> desaturated
        # below BADGE_S_MIN -> missed, and one arriving with a banner never gets
        # the follow-up polls to saturate (banner wakes first, then the EMA
        # re-inits). The raw crop is full-saturation, caught at once; keyed
        # against the slow baseline it stays specific to a newly appeared mark.
        dock_d = _check_badge(ds.astype(np.uint8), crops[2])

        result = {
            "wake": False,
            "reason": "",
            "banner": banner_d,
            "bottom": bottom_d,
            "dock": dock_d,
        }

        if banner_d["wake"]:
            result.update(
                wake=True, reason="notification banner appeared at top of screen"
            )
        elif bottom_d["wake"]:
            result.update(wake=True, reason="screen content changed in lower half")
        elif dock_d["wake"]:
            result.update(wake=True, reason="new red badge appeared on dock app")

        with self._lock:
            if result["wake"]:
                self._last_wake = now
            elif self._is_idle(now):
                result.update(wake=True, reason="idle check-in (no wake for 30+ min)")
                self._last_wake = now

        if result["wake"]:
            # Log the detection values on every wake so thresholds can be tuned
            # against a live environment (e.g. dark rooms). Goes through the
            # tagged logger so it carries the same `HH:MM [physiclaw]` prefix as
            # the rest of the server stream.
            log.info(
                "watchdog WAKE — %s | banner=%s bottom=%s dock=%s",
                result["reason"],
                banner_d,
                bottom_d,
                dock_d,
            )

        return result

    def _is_idle(self, now: float) -> bool:
        """Idle fallback. Caller must hold lock."""
        hour = dt.datetime.now().hour
        if not any(s <= hour < e for s, e in WORK_HOURS):
            return False
        return now - self._last_wake >= IDLE_INTERVAL
