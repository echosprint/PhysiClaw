"""
Color block segmentation — find colored UI elements via HSV saturation.

Functional apps use high-saturation colors for actionable elements (buttons,
icons, tags, prices) against white/gray backgrounds. This module exploits
that pattern to find colored UI elements without any ML model.

Algorithm:
  1. Convert to HSV, extract S-channel
  2. Otsu threshold on S-channel → binary mask of "saturated" pixels
  3. Morphology (close + open) to clean noise
  4. Connected components with stats → bounding boxes
  5. Classify each blob by H_std (hue standard deviation):
     - H_std > 25 → content image (food photo, product image)
     - H_std < 12 → solid UI element (button, icon, tag)
     - Between → mixed/ambiguous

Works best on clean screenshots (from AssistiveTouch). Also works on camera
frames but with lower accuracy due to lighting variation.

Architecture plan: "Tool 1: Color Block Segmentation (HSV saturation pipeline)"
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ─── HSV color name mapping ──────────────────────────────────

# OpenCV H range is 0-179 (half degrees)
_COLOR_RANGES = [
    (0,   10,  "red"),
    (10,  22,  "orange"),
    (22,  35,  "yellow"),
    (35,  78,  "green"),
    (78,  100, "cyan"),
    (100, 130, "blue"),
    (130, 155, "purple"),
    (155, 180, "red"),      # red wraps around
]


def hue_to_name(h_mean: float) -> str:
    """Convert mean hue (0-179) to a color name."""
    for lo, hi, name in _COLOR_RANGES:
        if lo <= h_mean < hi:
            return name
    return "red"


# ─── Data structures ─────────────────────────────────────────

@dataclass
class ColorBlob:
    """A detected colored region in the image."""
    bbox: tuple[int, int, int, int]     # (x1, y1, x2, y2) in pixels
    area: int                           # pixel area
    color_name: str                     # human-readable color
    h_mean: float                       # mean hue (0-179)
    h_std: float                        # hue std dev — low=solid, high=image
    s_mean: float                       # mean saturation
    is_solid: bool                      # True if H_std < 12 (solid UI element)
    is_image: bool                      # True if H_std > 25 (content image)
    aspect_ratio: float                 # width / height


# ─── Core pipeline ───────────────────────────────────────────

def segment_saturation(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """HSV S-channel Otsu threshold. Returns (binary_mask, hsv_image)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    _, mask = cv2.threshold(s_channel, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask, hsv


def clean_mask(mask: np.ndarray, close_size: int = 7,
               open_size: int = 5) -> np.ndarray:
    """Morphological close (fill gaps) then open (remove noise)."""
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT,
                                             (close_size, close_size))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT,
                                            (open_size, open_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    return mask


def extract_blobs(mask: np.ndarray, hsv: np.ndarray,
                  min_area: int = 200,
                  max_area_ratio: float = 0.4) -> list[ColorBlob]:
    """Extract and classify colored blobs from the mask.

    Args:
        min_area: minimum blob area in pixels (filters dust/noise)
        max_area_ratio: maximum blob area as fraction of image (filters background)
    """
    h, w = mask.shape[:2]
    total_area = h * w
    max_area = int(total_area * max_area_ratio)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    blobs = []
    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        x1 = stats[i, cv2.CC_STAT_LEFT]
        y1 = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        x2 = x1 + bw
        y2 = y1 + bh

        # Extract hue values within this blob's mask
        blob_mask = (labels == i)
        hues = hsv[:, :, 0][blob_mask]
        sats = hsv[:, :, 1][blob_mask]

        # Handle hue wraparound for red (0 and 179 are both red)
        # Shift hues > 150 by -180 to make red continuous around 0
        hues_shifted = hues.astype(np.float32)
        hues_shifted[hues_shifted > 150] -= 180
        h_mean_shifted = float(np.mean(hues_shifted))
        h_std = float(np.std(hues_shifted))
        # Convert back to 0-179 range for color naming
        h_mean = h_mean_shifted if h_mean_shifted >= 0 else h_mean_shifted + 180

        s_mean = float(np.mean(sats))
        aspect = bw / bh if bh > 0 else 1.0

        blobs.append(ColorBlob(
            bbox=(x1, y1, x2, y2),
            area=area,
            color_name=hue_to_name(h_mean),
            h_mean=round(h_mean, 1),
            h_std=round(h_std, 1),
            s_mean=round(s_mean, 1),
            is_solid=(h_std < 12),
            is_image=(h_std > 25),
            aspect_ratio=round(aspect, 2),
        ))

    # Sort by area descending (largest first)
    blobs.sort(key=lambda b: b.area, reverse=True)
    log.debug(f"Extracted {len(blobs)} color blobs "
              f"({sum(1 for b in blobs if b.is_solid)} solid, "
              f"{sum(1 for b in blobs if b.is_image)} image)")
    return blobs


# ─── Main entry point ────────────────────────────────────────

def detect_color_blocks(image_bgr: np.ndarray,
                        min_area: int = 200) -> list[ColorBlob]:
    """Detect colored UI elements and content images in a screenshot.

    Returns list of ColorBlob sorted by area (largest first).
    """
    mask, hsv = segment_saturation(image_bgr)
    mask = clean_mask(mask)
    return extract_blobs(mask, hsv, min_area=min_area)


# ─── 1D card y-scan (list item detection) ────────────────────

def find_card_y_positions(image_bgr: np.ndarray,
                          x_col_pct: float,
                          y_min_pct: float = 0.0,
                          y_max_pct: float = 1.0,
                          col_width_pct: float = 0.15,
                          min_gap_px: int = 10,
                          ) -> list[tuple[float, float]]:
    """Find vertical positions of list cards by scanning for content images.

    Many app lists (food delivery, shopping, chat) have content images
    (photos, avatars) at a fixed x-column. This function scans that column
    vertically and finds rows with high saturation — indicating a content
    image = a list card.

    Args:
        image_bgr: the screenshot/camera frame
        x_col_pct: x-position of the image column as 0-1 decimal
            (from skill layout constants, e.g., 0.02 for left-aligned photos)
        y_min_pct: top of the scrollable region as 0-1 decimal
        y_max_pct: bottom of the scrollable region as 0-1 decimal
        col_width_pct: width of the column to scan as 0-1 decimal
        min_gap_px: minimum gap between cards in pixels

    Returns:
        list of (y_top, y_bottom) as 0-1 decimals for each detected card.
        Sorted top to bottom.
    """
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]

    # Convert pct to pixels
    x_start = max(0, int((x_col_pct - col_width_pct / 2) * w))
    x_end = min(w, int((x_col_pct + col_width_pct / 2) * w))
    y_start = int(y_min_pct * h)
    y_end = int(y_max_pct * h)

    # Extract the column strip and compute mean saturation per row
    strip = s_channel[y_start:y_end, x_start:x_end]
    if strip.size == 0:
        return []
    row_means = np.mean(strip, axis=1)

    # Threshold: rows with saturation above Otsu threshold are "card content"
    _, thresh = cv2.threshold(
        row_means.astype(np.uint8), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contiguous high-saturation runs → card boundaries
    cards = []
    in_card = False
    card_start = 0

    for y_idx in range(len(thresh)):
        if thresh[y_idx] > 0:
            if not in_card:
                card_start = y_idx
                in_card = True
        else:
            if in_card:
                card_end = y_idx
                if card_end - card_start >= min_gap_px:
                    cards.append((card_start, card_end))
                in_card = False

    # Close last card
    if in_card and len(thresh) - card_start >= min_gap_px:
        cards.append((card_start, len(thresh)))

    # Convert back to 0-1 decimals
    result = []
    for cs, ce in cards:
        y_top = (y_start + cs) / h
        y_bot = (y_start + ce) / h
        result.append((round(y_top, 3), round(y_bot, 3)))

    log.debug(f"Card y-scan: {len(result)} cards in column x={x_col_pct:.2f}")
    return result


# ─── Annotation drawing ──────────────────────────────────────

# BGR colors for drawing
_DRAW_COLORS = {
    "red":    (0, 0, 255),
    "orange": (0, 128, 255),
    "yellow": (0, 255, 255),
    "green":  (0, 200, 0),
    "cyan":   (255, 200, 0),
    "blue":   (255, 100, 0),
    "purple": (200, 0, 200),
}


def annotate(image_bgr: np.ndarray, blobs: list[ColorBlob]) -> np.ndarray:
    """Draw numbered bounding boxes on the image.

    Solid UI elements get solid rectangles. Content images get dashed.
    """
    out = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, blob in enumerate(blobs):
        x1, y1, x2, y2 = blob.bbox
        color = _DRAW_COLORS.get(blob.color_name, (128, 128, 128))

        if blob.is_image:
            # Dashed rectangle for content images
            for start in range(x1, x2, 12):
                end = min(start + 6, x2)
                cv2.line(out, (start, y1), (end, y1), color, 2)
                cv2.line(out, (start, y2), (end, y2), color, 2)
            for start in range(y1, y2, 12):
                end = min(start + 6, y2)
                cv2.line(out, (x1, start), (x1, end), color, 2)
                cv2.line(out, (x2, start), (x2, end), color, 2)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label: index + classification
        tag = "img" if blob.is_image else "ui" if blob.is_solid else "?"
        label = f"{i + 1}:{tag}"
        cv2.putText(out, label, (x1 + 2, y1 + 16), font, 0.5, color, 1)

    return out
