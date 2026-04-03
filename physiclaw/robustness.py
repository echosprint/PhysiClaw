"""
Robustness helpers — popup dismissal, stability checks, timing jitter.

Utilities for reliable unattended operation. Used by the run-skill executor
to handle unexpected popups, verify screen transitions, and add human-like
timing variation.

Architecture plan Phase 7: "Robustness"
"""

import logging
import random
import time

import cv2
import numpy as np

from physiclaw.screen_match import detect_dark_overlay, frames_differ

log = logging.getLogger(__name__)


# ─── Multi-frame stability ───────────────────────────────────

def wait_stable(capture_fn, max_wait: float = 3.0,
                check_interval: float = 0.3,
                threshold: float = 0.03) -> np.ndarray | None:
    """Wait until the screen stops changing (content finished loading).

    Takes screenshots at intervals and compares consecutive frames.
    Returns the stable frame, or the last frame if timeout.

    Args:
        capture_fn: callable that returns a BGR frame (e.g., cam.snapshot)
        max_wait: maximum seconds to wait
        check_interval: seconds between captures
        threshold: frame difference threshold (lower = stricter)
    """
    prev = capture_fn()
    if prev is None:
        return None

    deadline = time.time() + max_wait
    while time.time() < deadline:
        time.sleep(check_interval)
        curr = capture_fn()
        if curr is None:
            continue
        if not frames_differ(prev, curr, threshold=threshold):
            log.debug("Screen is stable")
            return curr
        prev = curr

    log.debug("Stability timeout — returning last frame")
    return prev


# ─── Popup / overlay handling ────────────────────────────────

# Common close button positions (0-1 decimals) for Chinese app popups
# Ordered by likelihood: center-bottom (确定/知道了), top-right (X), center
_CLOSE_POSITIONS = [
    ([0.30, 0.60, 0.70, 0.70], "center-bottom button (确定/关闭)"),
    ([0.85, 0.05, 0.98, 0.12], "top-right X"),
    ([0.02, 0.05, 0.15, 0.12], "top-left X"),
    ([0.35, 0.55, 0.65, 0.65], "center button"),
    ([0.30, 0.75, 0.70, 0.85], "bottom button"),
]


def find_close_button_candidates(frame: np.ndarray) -> list[dict]:
    """Suggest candidate positions for popup close/dismiss buttons.

    When a dark overlay is detected, this function returns likely
    close button positions based on common Chinese app popup layouts.
    The agent should try each position with bbox_target + label test.

    Returns list of {"bbox": [l,t,r,b], "description": str}.
    """
    if not detect_dark_overlay(frame):
        return []

    h, w = frame.shape[:2]
    candidates = []

    # Also try to find bright buttons on dark overlay using V-channel
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    # Look for bright rectangular regions (buttons) in the center area
    center_strip = v[int(0.3 * h):int(0.8 * h), int(0.1 * w):int(0.9 * w)]
    _, bright_mask = cv2.threshold(center_strip, 200, 255, cv2.THRESH_BINARY)

    # Find contours that could be buttons
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # too small for a button
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        if 1.5 < aspect < 8:  # button-like aspect ratio
            # Convert back to full image coordinates (0-1)
            x_abs = (x + int(0.1 * w)) / w
            y_abs = (y + int(0.3 * h)) / h
            candidates.append({
                "bbox": [round(x_abs, 3), round(y_abs, 3),
                         round(x_abs + bw / w, 3), round(y_abs + bh / h, 3)],
                "description": "bright button on dark overlay",
            })

    # Add common positions as fallbacks
    for bbox, desc in _CLOSE_POSITIONS:
        candidates.append({"bbox": bbox, "description": desc})

    return candidates


# ─── Human-like timing ───────────────────────────────────────

def jitter_delay(base: float = 0.5, variance: float = 0.3):
    """Sleep for a randomized duration to appear more human-like.

    Args:
        base: base delay in seconds
        variance: maximum random addition in seconds
    """
    delay = base + random.random() * variance
    time.sleep(delay)


def jitter_coordinate(x: float, y: float,
                      spread: float = 0.005) -> tuple[float, float]:
    """Add slight random offset to a coordinate (0-1 decimals).

    Humans don't tap exactly the same spot every time. This adds
    a small random offset (default ±0.5% of screen) to make taps
    look more natural.

    Args:
        spread: maximum random offset in 0-1 decimals
    """
    dx = (random.random() - 0.5) * 2 * spread
    dy = (random.random() - 0.5) * 2 * spread
    return (
        max(0.0, min(1.0, x + dx)),
        max(0.0, min(1.0, y + dy)),
    )
