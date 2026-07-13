"""Shared frame preprocessing — grayscale/HSV/blur/resize/crop stages.

Every vision module used to re-implement these conversions inline; this
is the one home, so a camera- or codec-level change is a one-place
edit. Parameters deliberately stay per-caller — detection thresholds
elsewhere (change-diff noise floors, blur gates) are tuned against the
exact kernel and size their caller passes, so this module deduplicates
the code, never the tuning. The one exception is `crop_to_phone_screen`'s
view cap, which defaults to ``CONFIG.compact.max_image_edge_px``: that
number is a global LLM-payload budget, not per-caller algorithm tuning.

Two resize flavors exist because their callers normalize different
dimensions: `resize_to_max_edge` caps the long edge (vision-token cap on
cropped views), `resize_to_width` pins the width (sharpness scores are
only comparable at one working width).
"""

import cv2
import numpy as np

from physiclaw.common.config import CONFIG


def grayscale(frame: np.ndarray) -> np.ndarray:
    """BGR → single-channel gray; frames already gray pass through."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def to_hsv(frame: np.ndarray) -> np.ndarray:
    """BGR → HSV."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


def gaussian_blur(frame: np.ndarray, ksize: int) -> np.ndarray:
    """Square Gaussian blur with auto sigma — kills sensor noise and
    sub-pixel wobble before differencing. Callers pass their tuned
    kernel size; there is no shared default on purpose."""
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)


def resize_to_max_edge(frame: np.ndarray, max_edge: int) -> np.ndarray:
    """Downscale so the long edge is at most ``max_edge`` (aspect kept,
    INTER_AREA); smaller frames pass through untouched."""
    long_edge = max(frame.shape[:2])
    if long_edge <= max_edge:
        return frame
    scale = max_edge / long_edge
    new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    """Downscale so the width is at most ``width`` (aspect kept,
    INTER_AREA); narrower frames pass through untouched — upscaling
    would interpolation-smooth genuinely sharp pixels into false blur."""
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(
        frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA
    )


def phone_screen_crop_box(
    frame: np.ndarray, transforms
) -> tuple[int, int, int, int] | None:
    """Camera pixel rectangle (left, top, right, bottom) enclosing the phone screen.

    Returns None if calibration is missing or the rectangle degenerates
    after clamping to frame bounds.
    """
    if transforms is None:
        return None
    (x0, y0), (x1, y1) = transforms.bbox_to_pixel_rect([0.0, 0.0, 1.0, 1.0])
    h, w = frame.shape[:2]
    left, right = max(0, min(x0, x1)), min(w, max(x0, x1))
    top, bottom = max(0, min(y0, y1)), min(h, max(y0, y1))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def crop_to_phone_screen(
    frame: np.ndarray, transforms, max_long_edge: int | None = None
) -> np.ndarray:
    """Crop to the phone-screen region and downscale to cap vision tokens.

    ``max_long_edge`` defaults to ``CONFIG.compact.max_image_edge_px`` —
    the single knob for how large an image the LLM sees. Higher-resolution
    captures (2K/4K cameras) still land on the same edge; the extra source
    pixels survive as sharpness through the INTER_AREA downscale, not as
    payload.

    Returns the frame untouched if calibration is missing.
    """
    if max_long_edge is None:
        max_long_edge = CONFIG.compact.max_image_edge_px
    box = phone_screen_crop_box(frame, transforms)
    if box is None:
        return frame
    left, top, right, bottom = box
    return resize_to_max_edge(frame[top:bottom, left:right], max_long_edge)
