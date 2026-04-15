"""Watchdog — detect new notifications on the phone screen.

Watches three zones, skipping the clock area (y 0.1–0.5):
  - Banner (y < 0.1): notification banners sliding from top
  - Bottom (y > 0.5): lock-screen content, app grid changes
  - Dock   (y > 0.85): red badge detection (HSV)

Fires on phash content change (banner + bottom) OR new red badge (dock).
"""

import datetime as dt
import threading
import time

import cv2
import numpy as np

BASELINE_TTL = 60.0  # seconds before baseline expires
IDLE_WAKE_INTERVAL = 1800.0  # fallback: wake every 30 min during work hours
WORK_HOURS = [(9, 11), (14, 17)]  # 9–11 AM, 2–5 PM
PHASH_BITS_LOW = 10  # phash threshold: below = no change
PHASH_BITS_HIGH = 20  # phash threshold: above = definite change (skip std check)
STD_INCREASE = 5.0  # std increase required for mid-range phash (10–20 bits)
BADGE_MIN_AREA = 50  # minimum red pixel increase for new badge


def _phash(frame: np.ndarray, size: int = 16) -> np.ndarray:
    """16×16 difference hash — bool array of per-row greater-than comparisons."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return small[:, 1:] > small[:, :-1]


def _red_pixels(frame: np.ndarray) -> int:
    """Count red pixels in a BGR frame (badge-coloured)."""
    if frame.size == 0:
        return 0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    return int(np.count_nonzero(mask1 | mask2))


def _content_changed(prev: np.ndarray, curr: np.ndarray) -> bool:
    """True if screen content changed meaningfully.

    - phash > HIGH (20 bits): big visual change, fire immediately.
    - phash MID (10-20 bits): ambiguous — require std increase to
      filter out auto-lock, animations, and AOD clock ticks.
    - phash <= LOW (10 bits): no meaningful change.
    """
    diff = int(np.count_nonzero(_phash(prev) ^ _phash(curr)))
    if diff > PHASH_BITS_HIGH:
        return True
    if diff <= PHASH_BITS_LOW:
        return False
    prev_std = float(np.std(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)))
    curr_std = float(np.std(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)))
    return curr_std - prev_std > STD_INCREASE


def _new_badge(prev: np.ndarray, curr: np.ndarray) -> bool:
    """True if red pixels increased (new badge appeared)."""
    return _red_pixels(curr) - _red_pixels(prev) > BADGE_MIN_AREA


class Watchdog:
    """Stateful wake detector — tracks frame-to-frame changes.

    Feed raw camera frames via ``poll(frame, transforms)``. The watchdog
    crops three zones from the phone screen, skipping the clock area.
    Thread-safe.

    Note: WeChat must stay in the background — never force-close it.
    WeChat's broken APNs means iOS push notifications stop arriving
    once the process is killed, and this watchdog depends on those
    notifications to detect new messages.
    """

    def __init__(self):
        self._prev: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._prev_time: float = 0.0
        self._last_wake: float = time.monotonic()
        self._lock = threading.Lock()

    @staticmethod
    def _crop(frame: np.ndarray, transforms,
              zones: list[tuple[float, float]]) -> list[np.ndarray] | None:
        """Crop horizontal strips of the phone screen from a camera frame.

        Returns a list of crops (one per zone), or None if any crop fails.
        """
        h, w = frame.shape[:2]
        crops = []
        for y0, y1 in zones:
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

    # (banner, bottom, dock) — skip clock area y 0.1–0.5
    ZONES = [(0.0, 0.1), (0.5, 1.0), (0.85, 1.0)]

    def poll(self, frame: np.ndarray, transforms) -> dict:
        """Check for wake events. Returns ``{"wake": bool, "reason": str}``.

        Baseline expires after 60s — if stale, the current frame becomes
        the new baseline.
        """
        NO_WAKE = {"wake": False, "reason": ""}

        crops = self._crop(frame, transforms, self.ZONES)
        if crops is None:
            return NO_WAKE
        banner, bottom, dock = crops

        now = time.monotonic()
        with self._lock:
            prev = self._prev
            age = now - self._prev_time
            self._prev = (banner, bottom, dock)
            self._prev_time = now

        if prev is None or age > BASELINE_TTL:
            # Baseline stale (first poll, or agent was working).
            # Reset idle timer so fallback counts from now.
            self._last_wake = now
            return NO_WAKE

        prev_banner, prev_bottom, prev_dock = prev

        if _content_changed(prev_banner, banner):
            self._last_wake = now
            return {"wake": True, "reason": "notification banner appeared at top of screen"}
        if _content_changed(prev_bottom, bottom):
            self._last_wake = now
            return {"wake": True, "reason": "screen content changed in lower half"}
        if _new_badge(prev_dock, dock):
            self._last_wake = now
            return {"wake": True, "reason": "new red badge appeared on dock app"}
        return self._idle_wake(now)

    def _idle_wake(self, now: float) -> dict:
        """Fallback: fire every 30 min during work hours (9–11 AM, 2–5 PM)."""
        hour = dt.datetime.now().hour
        if not any(start <= hour < end for start, end in WORK_HOURS):
            return {"wake": False, "reason": ""}
        if now - self._last_wake < IDLE_WAKE_INTERVAL:
            return {"wake": False, "reason": ""}
        self._last_wake = now
        return {"wake": True, "reason": "idle check-in (no wake detected for 30+ min)"}
