"""Tests for `physiclaw.core.vision.util` — codecs, bbox validation,
numpad inference, element formatting, and the phone-in-frame diagnostic.

All tests use synthetic BGR `np.ndarray` images — no real photos
required. `check_phone_in_frame` writes a debug JPEG under
`tempfile.gettempdir()` as a side effect; tests mock `cv2.imwrite` to
keep the host clean.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest

from physiclaw.core.vision.util import (
    bbox_on_screen,
    check_phone_in_frame,
    compact_json,
    decode_image,
    encode_jpeg,
    find_numpad_digit,
    format_elements,
    validate_bbox,
)

# ---------- helpers ----------


def _solid_square(color_bgr: tuple[int, int, int], size: int = 200) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = color_bgr
    return img


def _draw_rect(
    img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color_bgr: tuple
) -> None:
    img[y1:y2, x1:x2] = color_bgr


# ---------- validate_bbox / bbox_on_screen ----------


@pytest.mark.parametrize(
    "bbox",
    ["not list", [0, 0, 1], [0, 0, 1, 1, 0], (), {"l": 0}],
)
def test_validate_bbox_wrong_shape_raises(bbox: Any) -> None:
    with pytest.raises(
        ValueError, match=r"^bbox: must be \[left, top, right, bottom\];"
    ):
        validate_bbox(bbox)


@pytest.mark.parametrize(
    "bbox",
    [["a", 0.5, 0.6, 0.7], [0.1, None, 0.6, 0.7], [0.1, 0.2, 0.3, [0.4]]],
)
def test_validate_bbox_non_number_coord_raises(bbox: list) -> None:
    with pytest.raises(ValueError, match=r"^bbox: each coord must be a number;"):
        validate_bbox(bbox)


@pytest.mark.parametrize(
    "bbox",
    [[-0.1, 0, 0.5, 0.5], [0, 0, 1.1, 0.5]],
)
def test_validate_bbox_out_of_unit_range_raises(bbox: list[float]) -> None:
    with pytest.raises(ValueError, match=r"^bbox: each coord must be in \[0, 1\];"):
        validate_bbox(bbox)


@pytest.mark.parametrize(
    "bbox",
    [[0.5, 0, 0.4, 1], [0, 0.5, 1, 0.4], [0.5, 0, 0.5, 1], [0, 0.5, 1, 0.5]],
)
def test_validate_bbox_inverted_or_degenerate_raises(bbox: list[float]) -> None:
    with pytest.raises(ValueError, match=r"^bbox: left < right, top < bottom;"):
        validate_bbox(bbox)


def test_validate_bbox_valid_returns_input_unchanged() -> None:
    bbox = [0.0, 0.0, 1.0, 1.0]

    assert validate_bbox(bbox) is bbox


def test_validate_bbox_accepts_tuple_form() -> None:
    validate_bbox((0.1, 0.2, 0.3, 0.4))


def test_bbox_on_screen_true_for_valid_bbox() -> None:
    assert bbox_on_screen([0.0, 0.0, 1.0, 1.0]) is True


@pytest.mark.parametrize(
    "bad_bbox",
    [
        [0, 0, 1],  # wrong shape
        ["x", 0, 1, 1],  # non-number
        [-0.1, 0, 1, 1],  # out of range
        [0.5, 0, 0.4, 1],  # inverted
    ],
)
def test_bbox_on_screen_false_for_invalid_bbox(bad_bbox: Any) -> None:
    assert bbox_on_screen(bad_bbox) is False


# ---------- compact_json ----------


def test_compact_json_empty_list() -> None:
    assert compact_json([]) == "[\n\n]\n"


def test_compact_json_single_item_indented_with_two_spaces() -> None:
    out = compact_json([{"a": 1}])

    assert out == '[\n  {"a": 1}\n]\n'


def test_compact_json_multiple_items_separated_by_comma_and_newline() -> None:
    out = compact_json([{"a": 1}, {"b": 2}])

    assert out == '[\n  {"a": 1},\n  {"b": 2}\n]\n'


def test_compact_json_uses_ensure_ascii_false_for_non_ascii() -> None:
    out = compact_json([{"text": "你好"}])

    assert "你好" in out
    assert "\\u4f60" not in out  # not escape-encoded


# ---------- format_elements ----------


def test_format_elements_header_always_present() -> None:
    assert format_elements([]).startswith(
        'id [kind] "label" [left,top,right,bottom] conf'
    )


def test_format_elements_renders_one_line_per_item_with_3_decimal_bbox() -> None:
    items = [
        {
            "id": 1,
            "kind": "icon",
            "label": "settings",
            "bbox": [0.1, 0.2, 0.3, 0.4],
            "conf": 0.875,
        }
    ]

    out = format_elements(items)

    assert out.endswith('1 [icon] "settings" [0.100,0.200,0.300,0.400] 0.88')


def test_format_elements_handles_missing_or_none_label_as_empty_string() -> None:
    items = [
        {"id": 2, "kind": "icon", "label": None, "bbox": [0, 0, 1, 1], "conf": 0.9},
    ]

    out = format_elements(items)

    assert '"" [' in out


# ---------- encode_jpeg / decode_image ----------


def test_encode_jpeg_then_decode_image_round_trips_to_close_pixel_values() -> None:
    img = _solid_square((100, 150, 200), size=64)

    blob = encode_jpeg(img)
    recovered = decode_image(blob)

    assert recovered.shape == img.shape
    # JPEG is lossy — allow ±5 per channel.
    assert np.allclose(recovered, img, atol=5)


def test_decode_image_raises_on_invalid_bytes() -> None:
    with pytest.raises(RuntimeError, match=r"^Failed to decode image bytes$"):
        decode_image(b"not an image")


# ---------- check_phone_in_frame ----------


def _phone_like_frame(
    img_w: int,
    img_h: int,
    phone_w: int,
    phone_h: int,
    cx: int | None = None,
    cy: int | None = None,
) -> np.ndarray:
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    if cx is None:
        cx = img_w // 2
    if cy is None:
        cy = img_h // 2
    _draw_rect(
        img,
        cx - phone_w // 2,
        cy - phone_h // 2,
        cx + phone_w // 2,
        cy + phone_h // 2,
        (255, 255, 255),
    )
    return img


def test_check_phone_in_frame_raises_when_no_bright_region(mocker) -> None:
    mocker.patch.object(cv2, "imwrite")
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match=r"^No bright region in camera frame"):
        check_phone_in_frame(img)


def test_check_phone_in_frame_reports_ok_for_well_aligned_phone(mocker) -> None:
    mocker.patch.object(cv2, "imwrite")
    # Aspect ratio ~2 (long axis 400, short 200), coverage = 80000 / 480000 ≈ 17%.
    # Coverage too low — pad to bigger phone.
    img = _phone_like_frame(800, 600, phone_w=600, phone_h=300)

    result = check_phone_in_frame(img)

    assert result["ok"] is True
    assert result["aspect_ratio"] == 2.0
    assert result["coverage"] >= 0.30


def test_check_phone_in_frame_flags_low_coverage(mocker) -> None:
    mocker.patch.object(cv2, "imwrite")
    img = _phone_like_frame(800, 600, phone_w=200, phone_h=100)  # tiny phone

    result = check_phone_in_frame(img)

    assert result["ok"] is False
    assert any("Move camera closer" in s for s in result["issues"])


def test_check_phone_in_frame_accepts_coverage_just_above_25pct(mocker) -> None:
    # 510×255 / (800×600) ≈ 27% — above the 25% floor but below the old 30%.
    # Pins the threshold: a bump back to 0.30 would flag this as too small.
    mocker.patch.object(cv2, "imwrite")
    img = _phone_like_frame(800, 600, phone_w=510, phone_h=255)

    result = check_phone_in_frame(img)

    assert result["coverage"] == 0.27
    assert not any("Move camera closer" in s for s in result["issues"])


def test_check_phone_in_frame_flags_misaligned_long_axis(mocker) -> None:
    mocker.patch.object(cv2, "imwrite")
    # Image landscape but phone portrait.
    img = _phone_like_frame(800, 600, phone_w=200, phone_h=600)

    result = check_phone_in_frame(img)

    assert any("long axes not aligned" in s for s in result["issues"])


def test_check_phone_in_frame_flags_rotated_phone(mocker) -> None:
    mocker.patch.object(cv2, "imwrite")
    # Build a phone tilted ~10° — drawn via a rotation transform.
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    src_corners = np.array(
        [[200, 150], [600, 150], [600, 450], [200, 450]], dtype=np.float32
    )
    cx, cy = 400, 300
    M = cv2.getRotationMatrix2D((cx, cy), 10, 1.0)
    rotated = cv2.transform(src_corners.reshape(-1, 1, 2), M).reshape(-1, 2)
    cv2.fillPoly(img, [rotated.astype(np.int32)], (255, 255, 255))

    result = check_phone_in_frame(img)

    assert any("Straighten camera" in s for s in result["issues"])


def test_check_phone_in_frame_flags_aspect_ratio_outside_2to1_tolerance(
    mocker,
) -> None:
    mocker.patch.object(cv2, "imwrite")
    # A square (1:1) phone — aspect 1.0, far from 2.0.
    img = _phone_like_frame(800, 600, phone_w=400, phone_h=400)

    result = check_phone_in_frame(img)

    assert any("Camera may be tilted" in s for s in result["issues"])


# ---------- find_numpad_digit ----------


def _pad_element(digit: str, x: float, y: float) -> dict:
    half = 0.05
    return {
        "id": int(digit),
        "kind": "text",
        "label": digit,
        "bbox": [x - half, y - half, x + half, y + half],
        "conf": 0.9,
    }


def test_find_numpad_digit_direct_match_returns_bbox() -> None:
    elements = [_pad_element("5", 0.5, 0.5)]

    bbox = find_numpad_digit(elements, "5")

    assert bbox == [0.45, 0.45, 0.55, 0.55]


def test_find_numpad_digit_returns_none_when_no_digits_visible() -> None:
    assert find_numpad_digit([], "5") is None


def test_find_numpad_digit_infers_layout_from_two_diagonal_keys() -> None:
    # "1" at top-left and "9" at center-right — different rows + columns.
    # Expected: "5" (center, row 1 col 1) lands at midpoint.
    elements = [
        _pad_element("1", 0.30, 0.30),
        _pad_element("9", 0.70, 0.50),
    ]

    bbox = find_numpad_digit(elements, "5")

    assert bbox is not None
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    assert cx == pytest.approx(0.50)
    assert cy == pytest.approx(0.40)


def test_find_numpad_digit_skips_elements_outside_keypad_y_band() -> None:
    # A "5" at y=0.1 is above the [0.2, 0.8] keypad band → ignored.
    elements = [_pad_element("5", 0.5, 0.1)]

    assert find_numpad_digit(elements, "5") is None


def test_find_numpad_digit_skips_two_keys_on_same_row_or_column() -> None:
    # "1" and "3" — same row → not usable for inference.
    elements = [
        _pad_element("1", 0.30, 0.30),
        _pad_element("3", 0.70, 0.30),
    ]

    assert find_numpad_digit(elements, "5") is None


def test_encode_view_jpeg_caps_long_edge_at_compact_config(monkeypatch) -> None:
    from physiclaw.core.vision.util import CONFIG, decode_image, encode_view_jpeg

    monkeypatch.setattr(CONFIG.compact, "max_image_edge_px", 100)
    img = np.full((300, 600, 3), 128, dtype=np.uint8)

    out = decode_image(encode_view_jpeg(img))

    assert max(out.shape[:2]) == 100


def test_encode_view_jpeg_leaves_small_frames_at_native_size() -> None:
    from physiclaw.core.vision.util import decode_image, encode_view_jpeg

    img = np.full((300, 200, 3), 128, dtype=np.uint8)

    out = decode_image(encode_view_jpeg(img))

    assert out.shape == (300, 200, 3)
