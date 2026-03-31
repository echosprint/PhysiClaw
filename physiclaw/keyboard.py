"""
Keyboard key detector — find key bounding boxes from a phone screenshot.

Algorithm:
1. Find space bar bottom (scan from bottom, find wide consecutive equal run)
2. Find 4 row boundaries (scan up from space bar, all-equal rows = separators)
3. Find key boundaries per row (columns where every pixel = background value)

The detected boxes are drawn on a debug image. An AI (Claude) then
labels each numbered box to produce a UI preset file at
.claude/ui-presets/system-keyboard.md — which the agent uses at runtime.

No hardcoded layouts. Works with any keyboard.
"""

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ─── Space bar detection ──────────────────────────────────────

def detect_space_bottom(gray: np.ndarray) -> int | None:
    """Find the bottom edge of the space bar (y pixel).

    Scans from the bottom up. The space bar is the widest key — its rows
    have a long consecutive run of equal pixels in the middle of the screen.
    """
    h, w = gray.shape
    for y in range(h - 1, int(0.5 * h), -1):
        row = gray[y]
        max_run = 1
        max_start = 0
        cur_run = 1
        cur_start = 0
        for i in range(1, w):
            if row[i] == row[i - 1]:
                cur_run += 1
                if cur_run > max_run:
                    max_run = cur_run
                    max_start = cur_start
            else:
                cur_run = 1
                cur_start = i
        left_edge = max_start / w
        right_edge = (max_start + max_run) / w
        if 0.4 > left_edge > 0.25 and 0.8 > right_edge > 0.65:
            return y + 1
    return None


# ─── Row boundary detection ───────────────────────────────────

def detect_row_boundaries(gray: np.ndarray, space_bottom_y: int,
                          num_rows: int = 4):
    """Find key row boundaries by scanning up from the space bar.

    A separator line = all pixels in the row have the same value.

    Returns:
        rows: list of (top_y, bottom_y) from bottom (row 4) to top (row 1)
        bg_value: the keyboard background pixel value
    """
    rows = []
    bg_value = None
    y = space_bottom_y - 1
    row_bottom = space_bottom_y

    while y >= 0 and len(rows) < num_rows:
        # Scan up through key content (non-uniform rows)
        while y >= 0 and not np.all(gray[y] == gray[y, 0]):
            y -= 1
        if y < 0:
            break
        # Now at a separator line — grab the background value
        if bg_value is None:
            bg_value = int(gray[y, 0])
        row_top = y + 1
        if row_bottom > row_top:
            rows.append((row_top, row_bottom))

        # Skip through the separator
        while y >= 0 and np.all(gray[y] == gray[y, 0]):
            y -= 1
        row_bottom = y + 1

    return rows, bg_value


# ─── Key detection within a row ───────────────────────────────

def detect_keys_in_row(gray: np.ndarray, top: int, bottom: int,
                       bg_value: int) -> list[tuple[int, int]]:
    """Find key boundaries within a row.

    For each column, if ALL pixels from top to bottom equal the background
    value, that column is a gap between keys.

    Returns list of (left_x, right_x) for each key.
    """
    w = gray.shape[1]
    strip = gray[top:bottom]

    # For each column: True if every pixel equals background
    is_bg = np.all(strip == bg_value, axis=0)

    # Find key spans (consecutive non-bg columns)
    keys = []
    key_start = None
    for x in range(w):
        if not is_bg[x]:
            if key_start is None:
                key_start = x
        else:
            if key_start is not None:
                keys.append((key_start, x))
                key_start = None
    if key_start is not None:
        keys.append((key_start, w))

    return keys


# ─── Main detection entry point ───────────────────────────────

def detect_key_boxes(frame: np.ndarray,
                     num_rows: int = 4,
                     ) -> list[tuple[float, float, float, float]]:
    """Detect all key bounding boxes from a phone screenshot.

    Returns:
        List of (left, top, right, bottom) as 0-1 decimals, sorted
        top-to-bottom then left-to-right.
        Empty list if detection fails.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sb = detect_space_bottom(gray)
    if sb is None:
        log.warning("Space bar not found")
        return []

    rows, bg = detect_row_boundaries(gray, sb, num_rows)
    if not rows:
        log.warning("No key rows found")
        return []

    log.info(f"Found {len(rows)} rows, bg={bg}")

    boxes = []
    # Reverse so row 1 (top) comes first
    for top, bot in reversed(rows):
        keys = detect_keys_in_row(gray, top, bot, bg)
        for kl, kr in keys:
            boxes.append((
                round(kl / w, 3), round(top / h, 3),
                round(kr / w, 3), round(bot / h, 3),
            ))

    log.info(f"Detected {len(boxes)} key boxes")
    return boxes


# ─── Debug visualization ──────────────────────────────────────

def draw_detected_keys(frame: np.ndarray,
                       boxes: list[tuple],
                       ) -> np.ndarray:
    """Draw numbered bounding boxes on the screenshot."""
    out = frame.copy()
    h, w = out.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (left, top, right, bottom) in enumerate(boxes):
        x1, y1 = int(left * w), int(top * h)
        x2, y2 = int(right * w), int(bottom * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, str(i + 1), (x1 + 4, y1 + 20),
                    font, 0.5, (0, 255, 0), 1)

    return out


def boxes_to_text(boxes: list[tuple]) -> str:
    """Format detected boxes as a numbered text listing."""
    lines = [f"Detected {len(boxes)} key boxes:\n"]
    for i, (left, top, right, bottom) in enumerate(boxes):
        lines.append(f"  {i+1:3d}. [{left:.3f}, {top:.3f}, {right:.3f}, {bottom:.3f}]")
    return "\n".join(lines)
