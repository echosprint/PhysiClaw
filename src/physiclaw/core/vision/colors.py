"""Color primitives — HSV masks, redness, and the shared color vocabularies.

The leaf layer every color detector builds on: `hsv_mask` is the one
place that knows a hue can wrap the 0/180 seam, `red_ranges` /
`CORNER_HSV_RANGES` are the calibrated range tables, and `redness` is
the saturation-free fallback for targets a camera washes out. Blob
extraction on top of these lives in `physiclaw.core.vision.blobs`.
"""

import cv2
import numpy as np


def hsv_mask(hsv: np.ndarray, ranges) -> np.ndarray:
    """OR of ``cv2.inRange`` over one or more ``(lower, upper)`` HSV ranges.

    Lets one call cover a hue that wraps the 0/180 seam — red is passed both
    ``[0..10]`` and ``[170..180]``. The leaf primitive every colour detector
    builds on (blob centroids in `blobs`, the dock badge's pixel count in
    the watchdog), so "red needs two ranges" lives in exactly one place.
    """
    mask: np.ndarray | None = None
    for lo, hi in ranges:
        m: np.ndarray = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = m if mask is None else (mask | m)
    if mask is None:
        raise ValueError("hsv_mask: at least one HSV range is required")
    return mask


def redness(frame: np.ndarray) -> np.ndarray:
    """Per-pixel "how red" map: ``R - max(G, B)``, clipped to 0–255 (uint8).

    Robust where an HSV saturation floor isn't: small red marks on a bright
    screen desaturate to pink under a camera (low S, dim V), but their red
    channel still sits clearly above green/blue. Redness isolates them when
    ``red_ranges`` + an S/V threshold would wipe them out. Use this for faint
    targets (calibration dots); use :func:`red_ranges` for bold solid red
    (orientation markers, dock badges, corner squares).
    """
    bgr = frame.astype(np.int16)
    r = bgr[:, :, 2] - np.maximum(bgr[:, :, 0], bgr[:, :, 1])
    return np.clip(r, 0, 255).astype(np.uint8)


def red_ranges(s_min: int = 100, v_min: int = 100):
    """The two HSV ranges covering red across the 0/180 hue seam.

    Callers pick the S/V floor that suits them: the orientation marker and the
    camera-pick corner blocks pass 80 (both wash out under a camera on a bright
    screen), while the dock badge keeps the default 100. Hue bounds are fixed.
    Faint targets like the calibration dots use :func:`redness` instead.
    """
    return [
        ([0, s_min, v_min], [10, 255, 255]),
        ([170, s_min, v_min], [180, 255, 255]),
    ]


# Orange validation dot #f97316 ≈ HSV H=20°, S=90%, V=97% → OpenCV H≈10.
# One range, shared by grid_detect's dot paths AND the calibration
# viewport.py screenshot-square detection so the two can't drift.
ORANGE_HSV_RANGE = ([5, 100, 100], [25, 255, 255])

# S/V floor for the corner blocks. On a dim rig the captured blocks sit at
# ~S/V 100-120, so the old floor of 100 was right at the edge — a slightly
# dimmer setup would miss them. 80 adds margin. Don't drop below ~60: there
# the search starts matching colourful home-screen app icons and the cluster
# check mis-reads them as a corner cluster.
_CORNER_SV_MIN = 80
# Magenta, not yellow, is the 4th corner colour: a camera renders the screen's
# yellow as a yellow-green (H≈44) that drifts out of the Yellow band and into
# Green, so the yellow blocks vanish and the all-four-colours check fails.
# Magenta (#ff00ff = red+blue, no green) sits alone in the otherwise-empty
# 130–170 hue channel — far from R/G/B and from anything in a typical workshop.
CORNER_HSV_RANGES = {
    "R": red_ranges(_CORNER_SV_MIN, _CORNER_SV_MIN),
    "G": [([40, _CORNER_SV_MIN, _CORNER_SV_MIN], [80, 255, 255])],
    "B": [([100, _CORNER_SV_MIN, _CORNER_SV_MIN], [130, 255, 255])],
    "M": [([134, _CORNER_SV_MIN, _CORNER_SV_MIN], [169, 255, 255])],
}
