"""HSV blob detection — centroids of color-matched regions.

The pipeline every color-target detector shares: mask (via
`physiclaw.core.vision.colors`) → one morphology pass → contours →
area-filtered centroids. Callers pick the color spec and thresholds;
the calibrated color tables live in `colors`.
"""

import cv2
import numpy as np

from physiclaw.core.vision.colors import hsv_mask
from physiclaw.core.vision.preprocess import to_hsv


def _as_ranges(lower, upper):
    """Normalise a colour spec to a list of ``(lower, upper)`` pairs:
    ``(lower, upper)`` → one range; ``upper=None`` → ``lower`` is already a
    list of ranges (e.g. ``red_ranges()``)."""
    return [(lower, upper)] if upper is not None else list(lower)


def contour_centroid(cnt) -> tuple[float, float] | None:
    """Area-weighted centroid ``(cx, cy)`` of a contour, or ``None`` if it's
    degenerate (zero area)."""
    m = cv2.moments(cnt)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def _hsv_blobs(
    hsv: np.ndarray,
    ranges,
    *,
    min_area: int,
    morph_op: int,
    morph_kernel: tuple[int, int],
) -> list[tuple[float, tuple[float, float]]]:
    """Core of the HSV-blob pipeline — every ``(area, centroid)`` pair
    with area ≥ ``min_area``. The one body every public shape below
    filters from, so a pipeline tweak (kernel, retrieval mode) lands
    exactly once."""
    mask = hsv_mask(hsv, ranges)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
    mask = cv2.morphologyEx(mask, morph_op, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[float, tuple[float, float]]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        c = contour_centroid(cnt)
        if c is not None:
            out.append((area, c))
    return out


def hsv_blob_centroids(
    hsv: np.ndarray,
    ranges,
    *,
    min_area: int = 50,
    morph_op: int = cv2.MORPH_OPEN,
    morph_kernel: tuple[int, int] = (5, 5),
) -> list[tuple[float, float]]:
    """Centroids of every matched blob, for callers that already hold an
    HSV frame — one conversion shared across several color tables (the
    corner-cluster detectors in `grid_detect`)."""
    return [
        c
        for _, c in _hsv_blobs(
            hsv, ranges, min_area=min_area, morph_op=morph_op,
            morph_kernel=morph_kernel,
        )
    ]


def find_all_hsv_blobs(
    frame: np.ndarray,
    lower,
    upper=None,
    *,
    min_area: int = 50,
    morph_op: int = cv2.MORPH_OPEN,
    morph_kernel: tuple[int, int] = (5, 5),
) -> list[tuple[float, float]]:
    """Return centroids of every HSV-matched blob above ``min_area``.

    One range as ``lower``/``upper``, or a list of ranges as ``lower``
    (``upper`` omitted) for a wrapping hue — see :func:`colors.red_ranges`.
    Same pipeline as :func:`find_largest_hsv_blob` but keeps every qualifying
    contour; order is undefined, so callers cluster by position.
    """
    return hsv_blob_centroids(
        to_hsv(frame),
        _as_ranges(lower, upper),
        min_area=min_area,
        morph_op=morph_op,
        morph_kernel=morph_kernel,
    )


def find_largest_hsv_blob(
    frame: np.ndarray,
    lower,
    upper=None,
    *,
    min_area: int = 50,
    morph_op: int = cv2.MORPH_OPEN,
    morph_kernel: tuple[int, int] = (5, 5),
) -> tuple[float, float] | None:
    """Centroid (cx, cy) of the largest HSV-matched blob, or None.

    One range as ``lower``/``upper``, or a list of ranges as ``lower``
    (``upper`` omitted) to cover a hue that wraps the 0/180 seam — see
    :func:`colors.red_ranges`. Applies one morphology pass (``open`` kills
    salt-and-pepper, ``close`` seals gaps) and returns the biggest contour's
    centroid, or ``None`` when none reaches ``min_area``.
    """
    blobs = _hsv_blobs(
        to_hsv(frame),
        _as_ranges(lower, upper),
        min_area=min_area,
        morph_op=morph_op,
        morph_kernel=morph_kernel,
    )
    if not blobs:
        return None
    return max(blobs, key=lambda b: b[0])[1]
