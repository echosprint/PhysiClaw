"""Tests for `physiclaw.core.vision.keyboard`."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from physiclaw.core.vision import keyboard
from physiclaw.core.vision.keyboard import (
    DIGIT_ROW,
    QWERTY_ROW1,
    QWERTY_ROW2,
    QWERTY_ROW3_LETTERS,
    _label_row,
    _render_pages,
    boxes_to_text,
    detect_keys_in_row,
    detect_key_boxes,
    detect_row_boundaries,
    detect_space_bottom,
    draw_detected_keys,
    generate_preset,
    label_keyboard,
)


# ---------- detect_space_bottom ----------


def _gray(arr: np.ndarray) -> np.ndarray:
    """Helper — keyboard module reads grayscale arrays."""
    return arr.astype(np.uint8)


def test_detect_space_bottom_returns_y_when_wide_run_in_lower_half() -> None:
    h, w = 200, 100
    img = np.zeros((h, w), dtype=np.uint8)
    img[:] = 50  # background
    # Insert a row of equal pixels spanning x ~ 30%-70% (matches space bar
    # detection criteria) at y=180 in lower half.
    # Make rows ABOVE y=180 non-uniform first so the scan finds 180.
    img[120:180, :] = np.tile(np.arange(w, dtype=np.uint8), (60, 1))
    img[180, :] = 50
    # Pad row 180 to be a clean wide run.
    img[180, 30:70] = 200
    # Surround with bg again so run is exactly span 30-70.
    img[180, :30] = np.arange(30, dtype=np.uint8)
    img[180, 70:] = np.arange(30, dtype=np.uint8)

    out = detect_space_bottom(img)

    # Some y in lower half should be detected.
    assert out is not None


def test_detect_space_bottom_returns_none_when_no_match() -> None:
    # All-zeros frame has no varying texture → entire row is "max_run=w",
    # which is wider than 0.8 → fails right_edge < 0.8.
    img = np.zeros((100, 100), dtype=np.uint8)

    out = detect_space_bottom(img)

    assert out is None


# ---------- detect_row_boundaries ----------


def test_detect_row_boundaries_finds_separators() -> None:
    h, w = 80, 100
    img = np.zeros((h, w), dtype=np.uint8)
    # Set the entire keyboard region to non-uniform (key content), with
    # uniform separator rows between them.
    img[:] = np.tile(np.arange(w, dtype=np.uint8), (h, 1))
    # Separator rows at y=10, 30, 50, 70 with uniform value 50.
    for y in [10, 30, 50, 70]:
        img[y] = 50

    rows, bg = detect_row_boundaries(img, space_bottom_y=70, num_rows=3)

    assert len(rows) == 3
    # Rows are bottom-up: (top, bottom).
    assert all(top < bot for top, bot in rows)
    assert bg == 50


def test_detect_row_boundaries_returns_empty_when_no_separator() -> None:
    h, w = 80, 100
    # Non-uniform all the way up — no separator found.
    img = np.tile(np.arange(w, dtype=np.uint8), (h, 1))

    rows, bg = detect_row_boundaries(img, space_bottom_y=h, num_rows=4)

    assert rows == []
    assert bg is None


# ---------- detect_keys_in_row ----------


def test_detect_keys_in_row_finds_two_keys() -> None:
    img = np.full((10, 50), 50, dtype=np.uint8)  # bg = 50
    # Two keys: x=10..20 and x=30..40 (non-bg pixels somewhere in column).
    img[5, 10:20] = 200
    img[5, 30:40] = 200

    keys = detect_keys_in_row(img, 0, 10, bg_value=50)

    assert keys == [(10, 20), (30, 40)]


def test_detect_keys_in_row_handles_key_at_edge() -> None:
    img = np.full((5, 20), 50, dtype=np.uint8)
    # Key extending all the way to right edge.
    img[2, 15:20] = 200

    keys = detect_keys_in_row(img, 0, 5, bg_value=50)

    assert keys == [(15, 20)]


def test_detect_keys_in_row_empty_when_all_bg() -> None:
    img = np.full((5, 20), 50, dtype=np.uint8)

    assert detect_keys_in_row(img, 0, 5, bg_value=50) == []


# ---------- detect_key_boxes ----------


def test_detect_key_boxes_returns_empty_when_space_not_found(
    mocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mocker.patch.object(keyboard, "detect_space_bottom", return_value=None)

    with caplog.at_level(logging.WARNING, logger="physiclaw.core.vision.keyboard"):
        boxes, bg = detect_key_boxes(np.zeros((100, 100, 3), dtype=np.uint8))

    assert boxes == []
    assert bg is None
    assert any("Space bar not found" in r.getMessage() for r in caplog.records)


def test_detect_key_boxes_returns_empty_when_no_rows(
    mocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mocker.patch.object(keyboard, "detect_space_bottom", return_value=80)
    mocker.patch.object(keyboard, "detect_row_boundaries", return_value=([], None))

    with caplog.at_level(logging.WARNING, logger="physiclaw.core.vision.keyboard"):
        boxes, bg = detect_key_boxes(np.zeros((100, 100, 3), dtype=np.uint8))

    assert boxes == []
    assert any("No key rows" in r.getMessage() for r in caplog.records)


def test_detect_key_boxes_returns_normalized_bboxes(mocker) -> None:
    h, w = 100, 200
    mocker.patch.object(keyboard, "detect_space_bottom", return_value=90)
    mocker.patch.object(
        keyboard,
        "detect_row_boundaries",
        return_value=([(60, 80)], 50),  # one row top=60, bot=80
    )
    mocker.patch.object(
        keyboard,
        "detect_keys_in_row",
        return_value=[(20, 60), (100, 140)],
    )

    boxes, bg = detect_key_boxes(np.zeros((h, w, 3), dtype=np.uint8))

    assert bg == 50
    assert boxes == [
        [round(20 / w, 3), round(60 / h, 3), round(60 / w, 3), round(80 / h, 3)],
        [round(100 / w, 3), round(60 / h, 3), round(140 / w, 3), round(80 / h, 3)],
    ]


# ---------- draw_detected_keys ----------


def test_draw_detected_keys_uses_green_on_dark_keyboard() -> None:
    frame = np.full((100, 100, 3), 30, dtype=np.uint8)
    boxes = [[0.1, 0.1, 0.2, 0.2]]

    out = draw_detected_keys(frame, boxes, bg_value=30)

    # Some pixel should be green (0, 255, 0) along the rectangle.
    assert (out[:, :, 1] == 255).any()
    assert not (out[:, :, 2] == 255).any()


def test_draw_detected_keys_uses_red_on_light_keyboard() -> None:
    frame = np.full((100, 100, 3), 220, dtype=np.uint8)
    boxes = [[0.1, 0.1, 0.2, 0.2]]

    out = draw_detected_keys(frame, boxes, bg_value=220)

    # Red somewhere.
    assert (out[:, :, 2] == 255).any()


def test_draw_detected_keys_estimates_color_when_bg_none() -> None:
    # Bottom 30% bright — should pick red.
    frame = np.full((100, 100, 3), 30, dtype=np.uint8)
    frame[70:] = 220
    boxes = [[0.1, 0.1, 0.2, 0.2]]

    out = draw_detected_keys(frame, boxes, bg_value=None)

    assert (out[:, :, 2] == 255).any()


def test_draw_detected_keys_handles_empty_boxes() -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    out = draw_detected_keys(frame, [], bg_value=50)

    np.testing.assert_array_equal(out, frame)
    assert out is not frame  # copy


# ---------- boxes_to_text ----------


def test_boxes_to_text_renders_numbered_listing() -> None:
    boxes = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ]

    out = boxes_to_text(boxes)

    assert "Detected 2 key boxes" in out
    assert "1." in out and "2." in out
    assert "[0.100, 0.200, 0.300, 0.400]" in out


def test_boxes_to_text_handles_empty() -> None:
    assert "Detected 0 key boxes" in boxes_to_text([])


# ---------- _label_row ----------


def test_label_row_letter_10_keys_qwerty_row1() -> None:
    keys = [(i * 10, i * 10 + 8) for i in range(10)]

    out = _label_row(keys, "letter")

    assert [d["element"] for d in out] == QWERTY_ROW1


def test_label_row_letter_9_keys_with_wide_endpoints_is_row3() -> None:
    # Wide first and last keys — shift + 7 letters + delete.
    keys = [(0, 30)] + [(50 + i * 10, 50 + i * 10 + 8) for i in range(7)] + [(150, 180)]

    out = _label_row(keys, "letter")

    assert out[0]["element"] == "⇧ Shift"
    assert out[-1]["element"] == "⌫ Delete"
    middle_letters = [d["element"] for d in out[1:-1]]
    assert middle_letters == QWERTY_ROW3_LETTERS


def test_label_row_letter_9_keys_uniform_widths_is_row2() -> None:
    keys = [(i * 10, i * 10 + 8) for i in range(9)]

    out = _label_row(keys, "letter")

    assert [d["element"] for d in out] == QWERTY_ROW2


def test_label_row_letter_unknown_count_marks_unknown() -> None:
    keys = [(0, 10), (20, 30), (40, 50)]

    out = _label_row(keys, "letter")

    assert all(d["element"] == "???" for d in out)


def test_label_row_bottom_marks_unknown() -> None:
    keys = [(0, 10), (20, 50)]

    out = _label_row(keys, "bottom")

    assert all(d["element"] == "???" for d in out)


# ---------- label_keyboard ----------


def test_label_keyboard_returns_none_on_no_space(mocker) -> None:
    mocker.patch.object(keyboard, "detect_space_bottom", return_value=None)

    assert label_keyboard(np.zeros((100, 100, 3), dtype=np.uint8)) is None


def test_label_keyboard_returns_none_on_no_rows(mocker) -> None:
    mocker.patch.object(keyboard, "detect_space_bottom", return_value=80)
    mocker.patch.object(keyboard, "detect_row_boundaries", return_value=([], None))

    assert label_keyboard(np.zeros((100, 100, 3), dtype=np.uint8)) is None


def test_label_keyboard_alpha_keyboard_assigns_qwerty(mocker) -> None:
    mocker.patch.object(keyboard, "detect_space_bottom", return_value=90)
    mocker.patch.object(
        keyboard,
        "detect_row_boundaries",
        # bottom-up rows; will be reversed → top-first
        return_value=(
            [(80, 90), (60, 75), (40, 55), (20, 35)],
            50,
        ),
    )

    def fake_keys(_g, top, _bot, _bg):
        # row 1 (top=20): 10 keys; row 2 (top=40): 9 uniform; row 3 (top=60): 9 wide;
        # bottom (top=80): 4 keys.
        if top == 20:
            return [(i * 10, i * 10 + 8) for i in range(10)]
        if top == 40:
            return [(i * 10, i * 10 + 8) for i in range(9)]
        if top == 60:
            return (
                [(0, 30)]
                + [(50 + i * 10, 50 + i * 10 + 8) for i in range(7)]
                + [(150, 180)]
            )
        return [(0, 50), (60, 100), (110, 150), (160, 200)]

    mocker.patch.object(keyboard, "detect_keys_in_row", side_effect=fake_keys)

    rows = label_keyboard(np.zeros((100, 200, 3), dtype=np.uint8))

    assert rows is not None
    assert len(rows) == 4
    # Top row: QWERTY.
    assert [k["element"] for k in rows[0]] == QWERTY_ROW1
    # Bottom row: all "???"
    assert all(k["element"] == "???" for k in rows[3])
    # Each entry has a position field.
    for r in rows:
        for k in r:
            assert "position" in k
            assert len(k["position"]) == 4


def test_label_keyboard_numeric_keyboard_assigns_digits(mocker) -> None:
    mocker.patch.object(keyboard, "detect_space_bottom", return_value=90)
    mocker.patch.object(
        keyboard,
        "detect_row_boundaries",
        return_value=([(80, 90), (60, 75), (40, 55), (20, 35)], 50),
    )

    def fake_keys(_g, top, _bot, _bg):
        if top == 20:
            return [(i * 10, i * 10 + 8) for i in range(10)]
        if top == 40:
            return [(i * 10, i * 10 + 8) for i in range(10)]  # 10 → numeric
        if top == 60:
            return [(i * 10, i * 10 + 8) for i in range(8)]  # symbol row
        return [(0, 50), (60, 200)]

    mocker.patch.object(keyboard, "detect_keys_in_row", side_effect=fake_keys)

    rows = label_keyboard(np.zeros((100, 200, 3), dtype=np.uint8))

    # First row: digits.
    assert [k["element"] for k in rows[0]] == DIGIT_ROW
    # Second row (10 keys, numeric): all "???"
    assert all(k["element"] == "???" for k in rows[1])
    # Third (8 keys, numeric symbols): all "???"
    assert all(k["element"] == "???" for k in rows[2])


def test_label_keyboard_numeric_first_row_extra_keys_marked_unknown(mocker) -> None:
    mocker.patch.object(keyboard, "detect_space_bottom", return_value=90)
    mocker.patch.object(
        keyboard,
        "detect_row_boundaries",
        return_value=([(80, 90), (60, 75), (40, 55), (20, 35)], 50),
    )

    def fake_keys(_g, top, _bot, _bg):
        if top == 20:
            return [
                (i * 10, i * 10 + 8) for i in range(11)
            ]  # 11 keys, exceeds DIGIT_ROW
        if top == 40:
            return [(i * 10, i * 10 + 8) for i in range(10)]
        if top == 60:
            return [(i * 10, i * 10 + 8) for i in range(8)]
        return [(0, 50)]

    mocker.patch.object(keyboard, "detect_keys_in_row", side_effect=fake_keys)

    rows = label_keyboard(np.zeros((100, 200, 3), dtype=np.uint8))

    # 11th key beyond DIGIT_ROW → "???"
    assert rows[0][-1]["element"] == "???"


# ---------- _render_pages / generate_preset ----------


def test_render_pages_emits_table_per_page() -> None:
    pages = {
        "Alpha Keyboard": [
            [
                {
                    "position": [0.0, 0.0, 0.1, 0.1],
                    "element": "q",
                    "action": "Types 'q'",
                },
            ],
        ],
    }

    out = _render_pages(pages)

    assert "## Alpha Keyboard" in out
    assert "Fingerprint:" in out
    assert "| 1 | q |" in out
    # Alpha Keyboard is the entry page → no Entry: line.
    assert "Entry: Alpha Keyboard → ???" not in out


def test_render_pages_includes_entry_for_secondary_pages() -> None:
    pages = {
        "Numeric Keyboard": [
            [{"position": [0, 0, 0.1, 0.1], "element": "1", "action": "Types '1'"}],
        ],
    }

    out = _render_pages(pages)

    assert "Entry: Alpha Keyboard → ???" in out


def test_render_pages_includes_bbox_image_when_provided() -> None:
    pages = {
        "Alpha Keyboard": [
            [{"position": [0, 0, 0.1, 0.1], "element": "q", "action": "x"}],
        ],
    }

    out = _render_pages(pages, bbox_images={"Alpha Keyboard": "alpha.jpg"})

    assert "Bounding box image: alpha.jpg" in out


def test_generate_preset_substitutes_pages(
    tmp_path: Path,
    mocker,
) -> None:
    template = tmp_path / "template.md"
    template.write_text("# Header\n\n{{pages}}\n\nfooter\n")
    mocker.patch.object(keyboard, "TEMPLATE_PATH", template)
    pages = {
        "Alpha Keyboard": [
            [{"position": [0.0, 0.0, 0.1, 0.1], "element": "q", "action": "x"}],
        ],
    }

    out = generate_preset(pages)

    assert "{{pages}}" not in out
    assert "## Alpha Keyboard" in out
    assert out.endswith("\n")


# ---------- constants ----------


def test_qwerty_constants() -> None:
    assert "".join(QWERTY_ROW1) == "qwertyuiop"
    assert "".join(QWERTY_ROW2) == "asdfghjkl"
    assert "".join(QWERTY_ROW3_LETTERS) == "zxcvbnm"
    assert "".join(DIGIT_ROW) == "1234567890"
