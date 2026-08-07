"""Tests for `physiclaw.common.image` — the image ingress cap."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from physiclaw.common import config, image


def _encode_jpg(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", arr)
    assert ok
    return buf.tobytes()


def test_scale_image_bytes_scales_when_over_max_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.CONFIG.compact, "max_image_edge_px", 100)
    img = np.full((300, 600, 3), 128, dtype=np.uint8)
    raw = _encode_jpg(img)

    out_bytes, mime = image.scale_image_bytes(raw)

    assert mime == "image/jpeg"
    decoded = cv2.imdecode(np.frombuffer(out_bytes, np.uint8), cv2.IMREAD_COLOR)
    # Long edge 600 → 100; aspect 2 preserved.
    assert max(decoded.shape[:2]) == 100


def test_scale_image_bytes_returns_input_on_decode_failure() -> None:
    raw = b"definitely not an image"

    out_bytes, mime = image.scale_image_bytes(raw)

    assert out_bytes == raw
    assert mime == "application/octet-stream"


def test_scale_image_bytes_jpeg_within_cap_passes_through_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The server already sized this view to the shared knob — a re-encode
    # would only stack a second generation of JPEG loss onto screen text.
    monkeypatch.setattr(config.CONFIG.compact, "max_image_edge_px", 1000)
    raw = _encode_jpg(np.full((300, 200, 3), 128, dtype=np.uint8))

    out_bytes, mime = image.scale_image_bytes(raw)

    assert out_bytes is raw
    assert mime == "image/jpeg"


def test_scale_image_bytes_png_within_cap_reencoded_to_jpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.CONFIG.compact, "max_image_edge_px", 1000)
    ok, buf = cv2.imencode(".png", np.full((300, 200, 3), 128, dtype=np.uint8))
    assert ok
    raw = buf.tobytes()

    out_bytes, mime = image.scale_image_bytes(raw)

    assert mime == "image/jpeg"
    assert out_bytes.startswith(b"\xff\xd8\xff")
