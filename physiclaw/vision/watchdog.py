"""Watchdog — detect new notifications on the phone screen.

Watches three zones (skipping AOD clock at y 0.1–0.5):
  - Banner (y 0.0–0.1):  notification banners from top
  - Bottom (y 0.5–1.0):  lock-screen content, app grid
  - Dock   (y 0.85–1.0): red badge on dock apps

Uses fast (5s) and slow (20s) EMAs of raw pixels. Fires when the fast
EMA diverges from the slow: std or mean increase for content zones,
red pixel increase for dock. Idle fallback wakes every 30 min during
work hours.
"""

import datetime as dt
import math
import threading
import time

import cv2
import numpy as np

# --- Detection thresholds ---
STD_INCREASE = 5.0
MEAN_INCREASE = 5.0
BADGE_MIN_AREA = 50

# --- EMA parameters ---
EMA_FAST = 1 - math.exp(-1 / 5)   # ~0.18, 5s memory
EMA_SLOW = 1 - math.exp(-1 / 20)  # ~0.05, 20s memory
EMA_STALE = 5.0  # re-init if no poll for this long (covers react cooldown)

# --- Idle fallback ---
IDLE_INTERVAL = 1800.0  # 30 min
WORK_HOURS = [(9, 12), (14, 17)]

# --- Screen zones (y0, y1) ---
ZONES = [(0.0, 0.1), (0.5, 1.0), (0.85, 1.0)]


# --- Helpers ---

def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _check_content(slow: np.ndarray, fast: np.ndarray) -> dict:
    """Detect new visual content via std/mean divergence."""
    sg, fg = _gray(slow), _gray(fast)
    std_delta = round(float(np.std(fg)) - float(np.std(sg)), 1)
    mean_delta = round(float(np.mean(fg)) - float(np.mean(sg)), 1)
    return {
        "std_delta": std_delta,
        "mean_delta": mean_delta,
        "wake": std_delta > STD_INCREASE or mean_delta > MEAN_INCREASE,
    }


def _check_badge(slow: np.ndarray, fast: np.ndarray) -> dict:
    """Detect new red badge via HSV red pixel increase."""
    def red(f):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        m2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        return int(np.count_nonzero(m1 | m2))
    delta = red(fast) - red(slow)
    return {"red_delta": delta, "wake": delta > BADGE_MIN_AREA}


def _ema_update(ema: np.ndarray, frame: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * frame.astype(np.float32) + (1 - alpha) * ema


def _crop_zones(frame, transforms) -> list[np.ndarray] | None:
    """Crop ZONES from camera frame using calibration transforms."""
    h, w = frame.shape[:2]
    crops = []
    for y0, y1 in ZONES:
        tl = transforms.pct_to_cam_pixel(0.0, y0)
        br = transforms.pct_to_cam_pixel(1.0, y1)
        crop = frame[
            max(0, min(tl[1], h)):max(0, min(br[1], h)),
            max(0, min(tl[0], w)):max(0, min(br[0], w)),
        ]
        if not crop.size:
            return None
        crops.append(crop)
    return crops


# --- Watchdog ---

class Watchdog:
    """EMA-based wake detector. Thread-safe, 1 Hz polling."""

    def __init__(self):
        self._ema = None          # ((fast, slow), ...) per zone, float32
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

        (bf, bs), (tf, ts), (df, ds) = ema
        banner_d = _check_content(bs.astype(np.uint8), bf.astype(np.uint8))
        bottom_d = _check_content(ts.astype(np.uint8), tf.astype(np.uint8))
        dock_d = _check_badge(ds.astype(np.uint8), df.astype(np.uint8))

        result = {"wake": False, "reason": "",
                  "banner": banner_d, "bottom": bottom_d, "dock": dock_d}

        if banner_d["wake"]:
            result.update(wake=True, reason="notification banner appeared at top of screen")
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

        return result

    def _is_idle(self, now: float) -> bool:
        """Idle fallback. Caller must hold lock."""
        hour = dt.datetime.now().hour
        if not any(s <= hour < e for s, e in WORK_HOURS):
            return False
        return now - self._last_wake >= IDLE_INTERVAL
