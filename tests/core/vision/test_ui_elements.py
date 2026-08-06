"""Tests for `physiclaw.core.vision.ui_elements`."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from physiclaw.core.vision import ui_elements
from physiclaw.core.vision.ui_elements import (
    UIElement,
    _clean,
    _dedupe,
    _detect_icons,
    _detect_texts,
    _iou,
    detect_ui_elements,
    elements_to_json,
)

# ---------- UIElement ----------


def test_ui_element_to_dict_rounds_values() -> None:
    e = UIElement(
        id=3,
        kind="text",
        label="hi",
        bbox=[0.12345, 0.6789, 0.999, 0.111],
        conf=0.876,
    )

    out = e.to_dict()

    assert out == {
        "id": 3,
        "kind": "text",
        "label": "hi",
        "bbox": [0.123, 0.679, 0.999, 0.111],
        "conf": 0.88,
    }


# ---------- elements_to_json ----------


def test_elements_to_json_dispatches_to_to_dict() -> None:
    items = [
        UIElement(id=0, kind="icon", label="", bbox=[0, 0, 0.1, 0.1], conf=0.9),
        UIElement(id=1, kind="text", label="x", bbox=[0.1, 0.1, 0.2, 0.2], conf=0.8),
    ]

    out = elements_to_json(items)

    assert out[0]["kind"] == "icon"
    assert out[1]["label"] == "x"
    assert len(out) == 2


# ---------- _iou ----------


def test_iou_identical_boxes_is_one() -> None:
    box = [0.0, 0.0, 1.0, 1.0]

    assert _iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero() -> None:
    a = [0.0, 0.0, 0.1, 0.1]
    b = [0.5, 0.5, 0.6, 0.6]

    assert _iou(a, b) == 0.0


def test_iou_partial_overlap() -> None:
    a = [0.0, 0.0, 1.0, 1.0]
    b = [0.5, 0.5, 1.5, 1.5]

    iou = _iou(a, b)

    # intersection 0.25, union 1+1-0.25=1.75 → 1/7 ≈ 0.143.
    assert iou == pytest.approx(0.25 / 1.75)


def test_iou_touching_edges_is_zero() -> None:
    a = [0.0, 0.0, 0.5, 0.5]
    b = [0.5, 0.5, 1.0, 1.0]

    assert _iou(a, b) == 0.0


# ---------- _clean ----------


def _ui(kind, conf, bbox, label="") -> UIElement:
    return UIElement(id=0, kind=kind, label=label, bbox=bbox, conf=conf)


def test_clean_drops_zero_area_boxes() -> None:
    items = [
        _ui("icon", 0.9, [0.1, 0.1, 0.1, 0.2]),  # zero width
        _ui("icon", 0.9, [0.1, 0.1, 0.2, 0.1]),  # zero height
    ]

    assert _clean(items) == []


def test_clean_drops_tiny_boxes() -> None:
    # Area 1e-5 < 1e-4 threshold.
    items = [_ui("icon", 0.9, [0.0, 0.0, 0.005, 0.002])]

    assert _clean(items) == []


def test_clean_drops_low_conf_icons() -> None:
    items = [
        _ui("icon", 0.2, [0.1, 0.1, 0.2, 0.2]),  # below 0.3
        _ui("icon", 0.5, [0.3, 0.3, 0.4, 0.4]),
    ]

    out = _clean(items)

    assert len(out) == 1
    assert out[0].conf == 0.5


def test_clean_drops_low_conf_text() -> None:
    items = [
        _ui("text", 0.5, [0.1, 0.1, 0.2, 0.2], label="x"),  # below 0.7
        _ui("text", 0.9, [0.3, 0.3, 0.4, 0.4], label="y"),
    ]

    out = _clean(items)

    assert len(out) == 1
    assert out[0].label == "y"


def test_clean_drops_full_screen_icon() -> None:
    items = [_ui("icon", 0.9, [0.0, 0.0, 1.0, 1.0])]

    assert _clean(items) == []


def test_clean_keeps_full_screen_text() -> None:
    items = [_ui("text", 0.9, [0.0, 0.0, 1.0, 1.0], label="banner")]

    out = _clean(items)

    assert len(out) == 1


def test_clean_dedupes_overlapping_boxes() -> None:
    items = [
        _ui("icon", 0.9, [0.1, 0.1, 0.2, 0.2]),
        _ui("icon", 0.7, [0.1, 0.1, 0.2, 0.2]),  # duplicate, lower conf
    ]

    out = _clean(items)

    # Higher conf survives.
    assert len(out) == 1
    assert out[0].conf == 0.9


# ---------- _dedupe ----------


def test_dedupe_drops_overlapping_lower_conf() -> None:
    items = [
        _ui("icon", 0.9, [0.0, 0.0, 0.5, 0.5]),
        _ui("icon", 0.5, [0.0, 0.0, 0.45, 0.5]),  # IoU > 0.7
    ]

    out = _dedupe(items)

    assert len(out) == 1
    assert out[0].conf == 0.9


def test_dedupe_keeps_disjoint_boxes() -> None:
    items = [
        _ui("icon", 0.9, [0.0, 0.0, 0.1, 0.1]),
        _ui("icon", 0.8, [0.5, 0.5, 0.6, 0.6]),
    ]

    out = _dedupe(items)

    assert len(out) == 2


# ---------- _detect_icons ----------


def test_detect_icons_uses_supplied_detector(mocker) -> None:
    fake_det = MagicMock()
    fake_det.detect.return_value = [
        MagicMock(bbox=(100, 200, 300, 400), confidence=0.7),
    ]

    out = _detect_icons(
        np.zeros((1000, 500, 3), dtype=np.uint8),
        500,
        1000,
        fake_det,
        0.2,
    )

    assert len(out) == 1
    assert out[0].kind == "icon"
    # Bbox normalized to 0-1.
    assert out[0].bbox == [100 / 500, 200 / 1000, 300 / 500, 400 / 1000]


def test_detect_icons_handles_missing_model(
    mocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_det = MagicMock()
    fake_det.detect.side_effect = FileNotFoundError("no model")

    with caplog.at_level(logging.WARNING, logger="physiclaw.core.vision.ui_elements"):
        out = _detect_icons(
            np.zeros((10, 10, 3), dtype=np.uint8),
            10,
            10,
            fake_det,
            0.2,
        )

    assert out == []
    assert any("icon detection unavailable" in r.getMessage() for r in caplog.records)


# ---------- _detect_texts ----------


def test_detect_texts_uses_supplied_reader(mocker) -> None:
    fake_reader = MagicMock()
    fake_reader.read.return_value = [
        MagicMock(text="hi", bbox=(50, 100, 150, 200), confidence=0.9),
    ]

    out = _detect_texts(
        np.zeros((1000, 500, 3), dtype=np.uint8),
        500,
        1000,
        fake_reader,
    )

    assert len(out) == 1
    assert out[0].kind == "text"
    assert out[0].label == "hi"
    assert out[0].bbox == [50 / 500, 100 / 1000, 150 / 500, 200 / 1000]


def test_detect_texts_collapses_stray_line_breaks_in_ocr_output(mocker) -> None:
    # `Element` refuses labels that are not `splitlines`-clean, so a stray
    # control character out of OCR must be collapsed here — one artifact
    # must not crash the whole detection pass at `to_element()` time.
    fake_reader = MagicMock()
    fake_reader.read.return_value = [
        MagicMock(text="line1\nline2\x0cline3", bbox=(0, 0, 10, 10), confidence=0.9),
    ]

    out = _detect_texts(
        np.zeros((1000, 500, 3), dtype=np.uint8),
        500,
        1000,
        fake_reader,
    )

    assert out[0].label == "line1 line2 line3"
    assert out[0].to_element().label == "line1 line2 line3"  # composes clean


def test_detect_texts_handles_import_error(
    mocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_reader = MagicMock()
    fake_reader.read.side_effect = ImportError("rapidocr")

    with caplog.at_level(logging.WARNING, logger="physiclaw.core.vision.ui_elements"):
        out = ui_elements._detect_texts(
            np.zeros((10, 10, 3), dtype=np.uint8),
            10,
            10,
            fake_reader,
        )

    assert out == []
    assert any("OCR unavailable" in r.getMessage() for r in caplog.records)


def test_detect_texts_degrades_on_runtime_failure(
    mocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A broken RapidOCR install fails at read() time with a runtime
    error, not ImportError — it must degrade to "no text" the same way
    icons degrade, instead of crashing the whole detection pass."""
    fake_reader = MagicMock()
    fake_reader.read.side_effect = RuntimeError("missing recognition model")

    with caplog.at_level(logging.WARNING, logger="physiclaw.core.vision.ui_elements"):
        out = ui_elements._detect_texts(
            np.zeros((10, 10, 3), dtype=np.uint8),
            10,
            10,
            fake_reader,
        )

    assert out == []
    assert any("OCR unavailable" in r.getMessage() for r in caplog.records)


# ---------- detect_ui_elements ----------


def test_detect_ui_elements_combines_and_sorts(mocker) -> None:
    icon_det = MagicMock()
    icon_det.detect.return_value = [
        # bottom (y=400 in 1000-tall frame → 0.4 normalized).
        MagicMock(bbox=(100, 400, 200, 500), confidence=0.5),
    ]
    ocr = MagicMock()
    ocr.read.return_value = [
        MagicMock(text="top", bbox=(50, 50, 150, 100), confidence=0.95),
    ]

    elements, annotated = detect_ui_elements(
        np.zeros((1000, 500, 3), dtype=np.uint8),
        icon_detector=icon_det,
        ocr_reader=ocr,
    )

    # Top text first (y=0.05), then icon (y=0.4).
    assert elements[0].kind == "text"
    assert elements[1].kind == "icon"
    # IDs reassigned in sort order.
    assert [e.id for e in elements] == [0, 1]
    # Annotated frame returned as numpy array.
    assert isinstance(annotated, np.ndarray)


def test_detect_ui_elements_filters_via_clean(mocker) -> None:
    icon_det = MagicMock()
    # Below default 0.3 threshold → dropped.
    icon_det.detect.return_value = [
        MagicMock(bbox=(0, 0, 50, 50), confidence=0.1),
    ]
    ocr = MagicMock()
    ocr.read.return_value = []

    elements, _ = detect_ui_elements(
        np.zeros((100, 100, 3), dtype=np.uint8),
        icon_detector=icon_det,
        ocr_reader=ocr,
    )

    assert elements == []
