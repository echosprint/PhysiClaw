"""Detect phone "wake" events from camera frames.

Two orthogonal signals; either one fires a wake:
  1. Luminance jump — the phone screen turned on (or off), or the overall
     brightness changed significantly.
  2. Perceptual-hash content change — the lit screen gained new content
     (notification banner, app switch, etc.).

Pure functions, no state, no hardware. The stateful watcher that compares
consecutive frames lives in `physiclaw.server.watchdog`.
"""

import cv2
import numpy as np


def _phash(frame: np.ndarray, size: int = 16) -> np.ndarray:
    """16×16 difference hash — bool array of per-row greater-than comparisons."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return small[:, 1:] > small[:, :-1]


def _mean_lum(frame: np.ndarray) -> float:
    """Mean grayscale luminance in the 0–255 range."""
    return float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))


def detect_wake(
    prev: np.ndarray,
    curr: np.ndarray,
    *,
    lum_jump: float = 25.0,
    phash_bits: int = 10,
) -> bool:
    """Return True if the phone screen woke between prev and curr.

    Args:
        prev, curr: BGR frames of the same scene.
        lum_jump:   absolute mean-luminance delta (0–255) that counts as wake.
                    Catches dark → lit transitions cheaply.
        phash_bits: Hamming distance on the 16×16 perceptual hash that counts
                    as a meaningful content change. Catches notification
                    banners and other in-screen content updates.
    """
    if abs(_mean_lum(curr) - _mean_lum(prev)) > lum_jump:
        return True
    diff = int(np.count_nonzero(_phash(prev) ^ _phash(curr)))
    return diff > phash_bits
