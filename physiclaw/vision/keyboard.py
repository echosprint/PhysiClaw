"""
Keyboard key detector — find key bounding boxes from a phone screenshot.

Algorithm:
1. Find space bar bottom (scan from bottom, find wide consecutive equal run)
2. Find 4 row boundaries (scan up from space bar, all-equal rows = separators)
3. Find key boundaries per row (columns where every pixel = background value)

The detected boxes are drawn on a bounding box image. An AI (Claude) then
labels each numbered box to produce a UI preset file at
.claude/ui-presets/system-keyboard.md — which the agent uses at runtime.

No hardcoded layouts. Works with any keyboard.
"""

import logging
from pathlib import Path

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


def detect_row_boundaries(gray: np.ndarray, space_bottom_y: int, num_rows: int = 4):
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


def detect_keys_in_row(
    gray: np.ndarray, top: int, bottom: int, bg_value: int
) -> list[tuple[int, int]]:
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


def detect_key_boxes(
    frame: np.ndarray,
    num_rows: int = 4,
) -> tuple[list[list[float]], int | None]:
    """Detect all key bounding boxes from a phone screenshot.

    Returns:
        (boxes, bg_value) where:
        - boxes: list of [left, top, right, bottom] as 0-1 decimals,
          sorted top-to-bottom then left-to-right. Empty if detection fails.
        - bg_value: keyboard background pixel value (for contrast-aware drawing)
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sb = detect_space_bottom(gray)
    if sb is None:
        log.warning("Space bar not found")
        return [], None

    rows, bg = detect_row_boundaries(gray, sb, num_rows)
    if not rows:
        log.warning("No key rows found")
        return [], None

    log.info(f"Found {len(rows)} rows, bg={bg}")

    boxes = []
    # Reverse so row 1 (top) comes first
    for top, bot in reversed(rows):
        keys = detect_keys_in_row(gray, top, bot, bg)
        for kl, kr in keys:
            boxes.append(
                [
                    round(kl / w, 3),
                    round(top / h, 3),
                    round(kr / w, 3),
                    round(bot / h, 3),
                ]
            )

    log.info(f"Detected {len(boxes)} key boxes")
    return boxes, bg


# ─── Debug visualization ──────────────────────────────────────


def draw_detected_keys(
    frame: np.ndarray,
    boxes: list[list[float]],
    bg_value: int | None = None,
) -> np.ndarray:
    """Draw numbered bounding boxes on the screenshot.

    Color auto-adjusts for contrast: green on dark keyboards, red on light.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Pick color that contrasts with keyboard background
    if bg_value is not None:
        is_dark = bg_value < 128
    else:
        # Estimate from the keyboard region (bottom 30% of image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_dark = np.mean(gray[int(0.7 * h) :]) < 128
    color = (0, 255, 0) if is_dark else (0, 0, 255)  # green or red

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, bbox in enumerate(boxes):
        x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
        x2, y2 = int(bbox[2] * w), int(bbox[3] * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, str(i + 1), (x1 + 4, y1 + 28), font, 0.9, color, 2)

    return out


def boxes_to_text(boxes: list[list[float]]) -> str:
    """Format detected boxes as a numbered text listing."""
    lines = [f"Detected {len(boxes)} key boxes:\n"]
    for i, bbox in enumerate(boxes):
        lines.append(
            f"  {i + 1:3d}. [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]"
        )
    return "\n".join(lines)


# ─── Auto-labeling ────────────────────────────────────────────

QWERTY_ROW1 = list("qwertyuiop")
QWERTY_ROW2 = list("asdfghjkl")
QWERTY_ROW3_LETTERS = list("zxcvbnm")
DIGIT_ROW = list("1234567890")


def _label_row(
    keys: list[tuple[int, int]], row_type: str, is_numeric: bool = False
) -> list[dict]:
    """Label keys in a single row based on key count and widths.

    row_type: "letter" or "bottom"
    is_numeric: True if this is the numeric keyboard (changes bottom row labels)

    Returns list of {left, right, element, action} dicts.
    """
    n = len(keys)
    widths = [kr - kl for kl, kr in keys]
    avg_w = sum(widths) / n if n else 0

    labeled = []

    # ── Letter rows: identify by key count ──
    if row_type == "letter":
        if n == 10:
            # Could be QWERTYUIOP or 1234567890
            # Check if row 2 also has 10 → numeric keyboard
            # Caller handles this; default to QWERTY row 1
            for i, (kl, kr) in enumerate(keys):
                labeled.append(
                    {
                        "left": kl,
                        "right": kr,
                        "element": QWERTY_ROW1[i],
                        "action": f"Types '{QWERTY_ROW1[i]}'",
                    }
                )
        elif n == 9:
            # Check if first/last key is wider (shift/delete row)
            if widths[0] > avg_w * 1.2 and widths[-1] > avg_w * 1.2:
                # Row 3: shift + letters + delete
                labeled.append(
                    {
                        "left": keys[0][0],
                        "right": keys[0][1],
                        "element": "⇧ Shift",
                        "action": "Toggle uppercase",
                    }
                )
                for i, (kl, kr) in enumerate(keys[1:-1]):
                    labeled.append(
                        {
                            "left": kl,
                            "right": kr,
                            "element": QWERTY_ROW3_LETTERS[i],
                            "action": f"Types '{QWERTY_ROW3_LETTERS[i]}'",
                        }
                    )
                labeled.append(
                    {
                        "left": keys[-1][0],
                        "right": keys[-1][1],
                        "element": "⌫ Delete",
                        "action": "Delete character",
                    }
                )
            else:
                # Row 2: ASDFGHJKL
                for i, (kl, kr) in enumerate(keys):
                    labeled.append(
                        {
                            "left": kl,
                            "right": kr,
                            "element": QWERTY_ROW2[i],
                            "action": f"Types '{QWERTY_ROW2[i]}'",
                        }
                    )
        else:
            # Unknown row — leave for AI
            for kl, kr in keys:
                labeled.append(
                    {"left": kl, "right": kr, "element": "???", "action": "???"}
                )

    # ── Bottom row: leave all for AI to label from bbox image ──
    elif row_type == "bottom":
        for kl, kr in keys:
            labeled.append({"left": kl, "right": kr, "element": "???", "action": "???"})

    return labeled


def label_keyboard(
    frame: np.ndarray,
    num_rows: int = 4,
) -> list[list[dict]] | None:
    """Detect and auto-label keyboard keys.

    Returns a list of rows, each row is a list of key dicts:
        {element, action, position: [l, t, r, b]}

    QWERTY letters (rows 1-3), shift, and delete are auto-labeled.
    Bottom row and numeric symbol keys get element="???" for the AI to fill in.
    Returns None if detection fails.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sb = detect_space_bottom(gray)
    if sb is None:
        return None

    rows_px, bg = detect_row_boundaries(gray, sb, num_rows)
    if not rows_px:
        return None

    # Reverse to top-first order
    rows_px = list(reversed(rows_px))

    # Detect keys per row
    all_rows = []
    for top, bot in rows_px:
        keys = detect_keys_in_row(gray, top, bot, bg)
        all_rows.append((top, bot, keys))

    # Determine if this is alpha or numeric keyboard
    # Alpha: row 1 has 10 keys, row 2 has 9 keys
    # Numeric: row 1 has 10 keys, row 2 has 10 keys
    row_counts = [len(keys) for _, _, keys in all_rows]
    is_numeric = len(row_counts) >= 2 and row_counts[0] == 10 and row_counts[1] == 10

    result = []
    for i, (top, bot, keys) in enumerate(all_rows):
        is_last = i == len(all_rows) - 1

        if is_last:
            labeled = _label_row(keys, "bottom", is_numeric=is_numeric)
        elif is_numeric and i == 0:
            # Numeric row 1: digits
            labeled = []
            for j, (kl, kr) in enumerate(keys):
                if j < len(DIGIT_ROW):
                    labeled.append(
                        {
                            "left": kl,
                            "right": kr,
                            "element": DIGIT_ROW[j],
                            "action": f"Types '{DIGIT_ROW[j]}'",
                        }
                    )
                else:
                    labeled.append(
                        {"left": kl, "right": kr, "element": "???", "action": "???"}
                    )
        elif is_numeric:
            # Numeric rows 2-3: all symbols — leave for AI
            labeled = []
            for kl, kr in keys:
                labeled.append(
                    {"left": kl, "right": kr, "element": "???", "action": "???"}
                )
        else:
            labeled = _label_row(keys, "letter")

        # Add position as 0-1 decimals
        for item in labeled:
            item["position"] = [
                round(item.pop("left") / w, 3),
                round(top / h, 3),
                round(item.pop("right") / w, 3),
                round(bot / h, 3),
            ]

        result.append(labeled)

    return result


# ─── Preset template generation ───────────────────────────────

TEMPLATE_PATH = (
    Path(__file__).parent.parent.parent
    / ".claude"
    / "skills"
    / "calibrate-keyboard"
    / "template.md"
)


def _render_pages(
    pages: dict[str, list[list[dict]]], bbox_images: dict[str, str] | None = None
) -> str:
    """Render the per-page key tables as markdown."""
    lines = []
    for page_name, rows in pages.items():
        lines.append(f"## {page_name}")
        lines.append("")
        lines.append(f"Fingerprint: On-screen keyboard, {page_name.lower()}")
        if page_name != "Alpha Keyboard":
            lines.append("Entry: Alpha Keyboard → ???")
        lines.append("")

        if bbox_images and page_name in bbox_images:
            lines.append(f"Bounding box image: {bbox_images[page_name]}")
            lines.append("")

        lines.append("| # | Element | Position | Action |")
        lines.append("| --- | --------- | ---------- | -------- |")

        box_idx = 1
        for row in rows:
            for key in row:
                pos = key["position"]
                pos_str = f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}, {pos[3]:.3f}]"
                lines.append(
                    f"| {box_idx} | {key['element']} | {pos_str} | {key['action']} |"
                )
                box_idx += 1

        lines.append("")

    return "\n".join(lines)


def generate_preset(
    pages: dict[str, list[list[dict]]], bbox_images: dict[str, str] | None = None
) -> str:
    """Generate .claude/ui-presets/system-keyboard.md content.

    Reads the template from .claude/skills/calibrate-keyboard/template.md
    and fills in the {{pages}} placeholder with detected key tables.
    """
    template = TEMPLATE_PATH.read_text()
    pages_md = _render_pages(pages, bbox_images)
    return template.replace("{{pages}}", pages_md).rstrip() + "\n"
