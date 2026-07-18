"""Tests for `physiclaw.core.vision.ocr`."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from physiclaw.core.vision.ocr import (
    OCRReader,
    TextResult,
    annotate,
    results_to_elements,
)

# ---------- TextResult dataclass ----------


def test_text_result_holds_fields() -> None:
    t = TextResult(text="hello", bbox=(0, 0, 10, 10), confidence=0.9)

    assert t.text == "hello"
    assert t.bbox == (0, 0, 10, 10)
    assert t.confidence == 0.9


# ---------- OCRReader ----------


def _ocr_result(boxes, txts, scores):
    return SimpleNamespace(boxes=boxes, txts=txts, scores=scores)


def _new_reader(mocker, ocr_callable):
    """Construct OCRReader with rapidocr stubbed out."""
    fake_module = MagicMock()
    fake_module.RapidOCR = MagicMock(return_value=ocr_callable)
    mocker.patch.dict("sys.modules", {"rapidocr": fake_module})
    return OCRReader()


def test_ocr_reader_raises_when_rapidocr_missing(mocker) -> None:
    # Ensure import raises ImportError.
    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def fake_import(name, *a, **kw):
        if name == "rapidocr":
            raise ImportError("nope")
        return real_import(name, *a, **kw)

    mocker.patch("builtins.__import__", side_effect=fake_import)

    with pytest.raises(ImportError, match="rapidocr is required"):
        OCRReader()


def test_read_returns_empty_when_boxes_none(mocker) -> None:
    fake_ocr = MagicMock(return_value=_ocr_result(None, None, None))
    reader = _new_reader(mocker, fake_ocr)

    out = reader.read(np.zeros((10, 10, 3), dtype=np.uint8))

    assert out == []


def test_read_returns_text_results_sorted(mocker) -> None:
    boxes = [
        # Top-bottom, then left-right after sorting.
        [(50, 100), (60, 100), (60, 110), (50, 110)],  # bbox (50,100,60,110)
        [(10, 10), (20, 10), (20, 20), (10, 20)],  # bbox (10,10,20,20) — top-most
        [(15, 100), (25, 100), (25, 110), (15, 110)],  # same row as first, lefter
    ]
    fake_ocr = MagicMock(
        return_value=_ocr_result(
            boxes,
            ["A", "B", "C"],
            [0.9, 0.8, 0.7],
        )
    )
    reader = _new_reader(mocker, fake_ocr)

    out = reader.read(np.zeros((200, 200, 3), dtype=np.uint8))

    # Sorted by (y, x).
    assert [t.text for t in out] == ["B", "C", "A"]
    assert out[0].bbox == (10, 10, 20, 20)
    assert out[0].confidence == 0.8


def test_read_with_crop_box_offsets_bbox(mocker) -> None:
    boxes = [[(5, 5), (15, 5), (15, 15), (5, 15)]]
    fake_ocr = MagicMock(return_value=_ocr_result(boxes, ["X"], [1.0]))
    reader = _new_reader(mocker, fake_ocr)

    out = reader.read(
        np.zeros((100, 100, 3), dtype=np.uint8),
        crop_box=(20, 30, 80, 90),
    )

    # bbox is offset by (left, top) = (20, 30).
    assert out[0].bbox == (25, 35, 35, 45)


def test_read_max_edge_downscales_and_rescales_bbox(mocker) -> None:
    # A 200px-long-edge image capped at max_edge=100 → scale 0.5: OCR sees
    # the 100x100 downscale, and the returned bbox is scaled back up 2x to
    # the original frame's coordinates.
    boxes = [[(10, 10), (20, 10), (20, 20), (10, 20)]]  # in downscaled space
    fake_ocr = MagicMock(return_value=_ocr_result(boxes, ["X"], [1.0]))
    reader = _new_reader(mocker, fake_ocr)

    out = reader.read(np.zeros((200, 200, 3), dtype=np.uint8), max_edge=100)

    assert fake_ocr.call_args.args[0].shape[:2] == (100, 100)  # OCR'd the downscale
    assert out[0].bbox == (20, 20, 40, 40)  # bbox scaled back 2x


def test_read_max_edge_noop_when_already_small(mocker) -> None:
    boxes = [[(5, 5), (15, 5), (15, 15), (5, 15)]]
    fake_ocr = MagicMock(return_value=_ocr_result(boxes, ["X"], [1.0]))
    reader = _new_reader(mocker, fake_ocr)

    out = reader.read(np.zeros((80, 80, 3), dtype=np.uint8), max_edge=100)

    assert fake_ocr.call_args.args[0].shape[:2] == (80, 80)  # untouched
    assert out[0].bbox == (5, 5, 15, 15)


def test_read_crop_clamps_to_frame(mocker) -> None:
    fake_ocr = MagicMock(
        return_value=_ocr_result(
            [[(0, 0), (5, 0), (5, 5), (0, 5)]],
            ["X"],
            [0.9],
        )
    )
    reader = _new_reader(mocker, fake_ocr)

    out = reader.read_crop(
        np.zeros((50, 50, 3), dtype=np.uint8),
        x1=-10,
        y1=-10,
        x2=100,
        y2=100,
    )

    assert out == "X"


def test_read_crop_returns_empty_string_for_zero_area(mocker) -> None:
    fake_ocr = MagicMock(return_value=_ocr_result([], [], []))
    reader = _new_reader(mocker, fake_ocr)

    out = reader.read_crop(
        np.zeros((50, 50, 3), dtype=np.uint8),
        x1=10,
        y1=10,
        x2=10,
        y2=10,
    )

    assert out == ""


def test_read_crop_concatenates_multiple_regions(mocker) -> None:
    boxes = [
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        [(0, 20), (10, 20), (10, 30), (0, 30)],
    ]
    fake_ocr = MagicMock(
        return_value=_ocr_result(
            boxes,
            ["foo", "bar"],
            [1.0, 1.0],
        )
    )
    reader = _new_reader(mocker, fake_ocr)

    out = reader.read_crop(
        np.zeros((100, 100, 3), dtype=np.uint8),
        x1=0,
        y1=0,
        x2=50,
        y2=50,
    )

    assert out == "foo bar"


# ---------- results_to_elements ----------


def _fake_transforms_pixel_to_pct():
    t = MagicMock()
    t.pixel_to_pct.side_effect = lambda x, y: (x / 1000, y / 1000)
    return t


def test_results_to_elements_maps_pixels_to_pct() -> None:
    results = [
        TextResult(text="hi", bbox=(100, 200, 300, 400), confidence=0.92),
    ]

    elements = results_to_elements(results, _fake_transforms_pixel_to_pct())

    assert elements == [
        {
            "id": 0,
            "kind": "text",
            "label": "hi",
            "bbox": [0.1, 0.2, 0.3, 0.4],
            "conf": 0.92,
        }
    ]


def test_results_to_elements_assigns_sequential_ids() -> None:
    results = [
        TextResult(text=f"r{i}", bbox=(i, i, i + 1, i + 1), confidence=0.5)
        for i in range(3)
    ]

    elements = results_to_elements(results, _fake_transforms_pixel_to_pct())

    assert [e["id"] for e in elements] == [0, 1, 2]


def test_results_to_elements_rounds_bbox_and_conf() -> None:
    t = MagicMock()
    t.pixel_to_pct.side_effect = lambda x, y: (x / 3, y / 7)
    results = [TextResult(text="x", bbox=(1, 1, 2, 2), confidence=0.123456)]

    elements = results_to_elements(results, t)

    # 1/3 ≈ 0.333, 1/7 ≈ 0.143, 2/3 ≈ 0.667, 2/7 ≈ 0.286.
    assert elements[0]["bbox"] == [0.333, 0.143, 0.667, 0.286]
    assert elements[0]["conf"] == 0.12


# ---------- annotate ----------


def test_annotate_returns_copy() -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    result = TextResult(text="hi", bbox=(5, 5, 25, 25), confidence=0.9)

    out = annotate(frame, [result])

    assert out is not frame
    assert out.shape == frame.shape
    # Some pixels along the bbox should now be non-zero (red rect drawn).
    assert (out != 0).any()


def test_annotate_handles_empty_results() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    out = annotate(frame, [])

    np.testing.assert_array_equal(out, frame)
    assert out is not frame  # still a copy
