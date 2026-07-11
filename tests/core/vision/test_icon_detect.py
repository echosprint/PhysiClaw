"""Tests for `physiclaw.core.vision.icon_detect`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from physiclaw.core.vision import icon_detect
from physiclaw.core.vision.icon_detect import (
    Element,
    IconDetector,
    annotate,
)


def _make_detector(mocker, output: np.ndarray, model_path: Path):
    """Build an IconDetector with the ONNX backend stubbed out."""
    model_path.touch()
    fake_net = MagicMock()
    fake_net.forward.return_value = output[None]  # add batch dim
    mocker.patch.object(
        icon_detect.cv2.dnn,
        "readNetFromONNX",
        return_value=fake_net,
    )
    return IconDetector(model_path=model_path), fake_net


def _yolo_output(
    detections: list[tuple[float, float, float, float, float]], *, total: int = 100
) -> np.ndarray:
    """Build a (5, N) YOLO-format output: cx, cy, w, h, conf."""
    arr = np.zeros((5, total), dtype=np.float32)
    for i, (cx, cy, w, h, c) in enumerate(detections):
        arr[:, i] = [cx, cy, w, h, c]
    return arr


# ---------- Element dataclass ----------


def test_element_dataclass() -> None:
    e = Element(bbox=(0, 0, 10, 10), confidence=0.9)

    assert e.bbox == (0, 0, 10, 10)
    assert e.confidence == 0.9


# ---------- IconDetector init ----------


def test_init_raises_when_model_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Model not found"):
        IconDetector(model_path=tmp_path / "missing.onnx")


def test_init_loads_model(tmp_path: Path, mocker) -> None:
    model = tmp_path / "model.onnx"
    model.touch()
    fake_net = MagicMock()
    spy = mocker.patch.object(
        icon_detect.cv2.dnn,
        "readNetFromONNX",
        return_value=fake_net,
    )

    detector = IconDetector(model_path=model)

    assert detector.net is fake_net
    spy.assert_called_once_with(str(model))


# ---------- detect ----------


def test_detect_returns_empty_when_no_predictions_above_threshold(
    tmp_path: Path,
    mocker,
) -> None:
    output = _yolo_output(
        [
            (640, 640, 100, 100, 0.05),
            (320, 320, 50, 50, 0.10),
        ]
    )
    detector, _ = _make_detector(mocker, output, tmp_path / "m.onnx")

    out = detector.detect(np.zeros((1280, 1280, 3), dtype=np.uint8))

    assert out == []


def test_detect_returns_elements_above_threshold(
    tmp_path: Path,
    mocker,
) -> None:
    output = _yolo_output(
        [
            (640, 640, 100, 100, 0.9),
        ]
    )
    detector, _ = _make_detector(mocker, output, tmp_path / "m.onnx")

    out = detector.detect(np.zeros((1280, 1280, 3), dtype=np.uint8))

    assert len(out) == 1
    bbox = out[0].bbox
    # cx=640, cy=640, w=h=100 → x1=590, y1=590, x2=690, y2=690 (no scaling).
    assert bbox == (590, 590, 690, 690)
    assert out[0].confidence == pytest.approx(0.9)


def test_detect_scales_bbox_back_to_original(tmp_path: Path, mocker) -> None:
    # Frame is 640x640 → letterbox scales 2x → INPUT_SIZE 1280.
    output = _yolo_output(
        [
            (640, 640, 100, 100, 0.9),  # in 1280x1280 model space
        ]
    )
    detector, _ = _make_detector(mocker, output, tmp_path / "m.onnx")

    out = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    bbox = out[0].bbox
    # scale = 1280 / 640 = 2 → divide by 2 to map back.
    assert bbox == (295, 295, 345, 345)


def test_detect_clamps_bbox_to_frame(tmp_path: Path, mocker) -> None:
    # Box predicted off-frame; should clamp to (0, 0, w, h).
    output = _yolo_output(
        [
            (10, 10, 100, 100, 0.9),  # extends past x=0 / y=0
        ]
    )
    detector, _ = _make_detector(mocker, output, tmp_path / "m.onnx")

    out = detector.detect(np.zeros((1280, 1280, 3), dtype=np.uint8))

    x1, y1, x2, y2 = out[0].bbox
    assert x1 == 0
    assert y1 == 0


def test_detect_sorts_top_to_bottom_then_left_to_right(
    tmp_path: Path,
    mocker,
) -> None:
    output = _yolo_output(
        [
            (800, 800, 50, 50, 0.9),  # bottom
            (200, 200, 50, 50, 0.85),  # top-left
            (600, 200, 50, 50, 0.85),  # top-right (same y as top-left)
        ]
    )
    detector, _ = _make_detector(mocker, output, tmp_path / "m.onnx")

    out = detector.detect(np.zeros((1280, 1280, 3), dtype=np.uint8))

    ys = [e.bbox[1] for e in out]
    assert ys == sorted(ys)
    # First two share y; ensure left-then-right.
    assert out[0].bbox[0] < out[1].bbox[0]


def test_detect_respects_custom_confidence_threshold(
    tmp_path: Path,
    mocker,
) -> None:
    output = _yolo_output(
        [
            (640, 640, 50, 50, 0.5),
        ]
    )
    detector, _ = _make_detector(mocker, output, tmp_path / "m.onnx")

    # Default threshold 0.3 → returns one.
    assert len(detector.detect(np.zeros((1280, 1280, 3), dtype=np.uint8))) == 1
    # Tighter threshold → empty.
    assert (
        detector.detect(
            np.zeros((1280, 1280, 3), dtype=np.uint8),
            confidence=0.6,
        )
        == []
    )


# ---------- annotate ----------


def test_annotate_returns_copy_with_boxes_drawn() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    elements = [Element(bbox=(10, 10, 30, 30), confidence=0.9)]

    out = annotate(frame, elements)

    assert out is not frame
    # Some pixel along the rectangle changed.
    assert (out != 0).any()


def test_annotate_handles_empty_elements() -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    out = annotate(frame, [])

    np.testing.assert_array_equal(out, frame)
    assert out is not frame


def test_annotate_labels_each_element_with_index() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    elements = [
        Element(bbox=(10, 10, 30, 30), confidence=0.9),
        Element(bbox=(50, 50, 70, 70), confidence=0.8),
    ]

    out = annotate(frame, elements)

    # Drawing happened — non-zero pixels in both regions.
    assert (out[8:35, 8:35] != 0).any()
    assert (out[48:75, 48:75] != 0).any()
