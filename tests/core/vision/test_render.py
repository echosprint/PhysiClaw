"""Tests for `physiclaw.core.vision.render` — overlay drawing.

Two functions; tests focus on input/output shape and structural
properties (pixels changed in expected regions, original frame
untouched). Exact pixel layouts depend on cv2's font metrics, so we
assert behaviorally rather than against a golden image.

Accepted mutmut survivors: this module is pure rendering. Mutations
that shift cx/cy by `w//3` instead of `w//2`, change `scale = h/150`
to `h/151`, alter thickness `2` to `3`, or rgb `255` to `256` (clamps
back to 255) all produce visually-different but structurally-similar
output. Killing them would require golden-image diffs that are
brittle across cv2 versions and font shipments. The visual contract
is vetted by eye, not by unit tests.
"""

from __future__ import annotations

import numpy as np

from physiclaw.core.vision.render import annotate_elements, watermark_index

# ---------- watermark_index ----------


def test_watermark_index_returns_a_new_array_not_the_original() -> None:
    frame = np.full((300, 400, 3), 50, dtype=np.uint8)

    out = watermark_index(frame, 1)

    assert out is not frame
    assert out.shape == frame.shape


def test_watermark_index_does_not_mutate_the_input_frame() -> None:
    frame = np.full((300, 400, 3), 50, dtype=np.uint8)
    snapshot = frame.copy()

    watermark_index(frame, 7)

    np.testing.assert_array_equal(frame, snapshot)


def test_watermark_index_draws_visible_pixels_in_the_center_region() -> None:
    # Uniform grey background — any non-grey pixel is from the overlay.
    frame = np.full((300, 400, 3), 50, dtype=np.uint8)

    out = watermark_index(frame, 3)

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    # Sample a 60×60 box around the center — overlay must change it.
    center_patch = out[cy - 30 : cy + 30, cx - 30 : cx + 30]
    assert (center_patch != 50).any()


def test_watermark_index_changes_pixel_count_grows_with_index_digit_count() -> None:
    # A 2-digit index draws a wider label than a 1-digit one — more
    # pixels diverge from the uniform background.
    frame = np.full((300, 400, 3), 50, dtype=np.uint8)

    one = watermark_index(frame, 1)
    twelve = watermark_index(frame, 12)

    diff_one = int((one != 50).any(axis=2).sum())
    diff_twelve = int((twelve != 50).any(axis=2).sum())
    assert diff_twelve > diff_one


# ---------- annotate_elements ----------


def _icon(elem_id: int, bbox: list[float]) -> dict:
    return {"id": elem_id, "kind": "icon", "label": "", "bbox": bbox, "conf": 0.9}


def _text(elem_id: int, bbox: list[float], label: str = "hello") -> dict:
    return {"id": elem_id, "kind": "text", "label": label, "bbox": bbox, "conf": 0.9}


def test_annotate_elements_returns_copy_not_mutating_original() -> None:
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    snapshot = frame.copy()

    out = annotate_elements(frame, [_icon(1, [0.1, 0.1, 0.3, 0.3])], 800, 600)

    assert out is not frame
    np.testing.assert_array_equal(frame, snapshot)


def test_annotate_elements_skips_text_kind_when_include_text_is_false() -> None:
    # Default include_text=False — only icons get drawn. With only a
    # text element, the output is identical to the input.
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    out = annotate_elements(frame, [_text(1, [0.1, 0.1, 0.3, 0.3])], 800, 600)

    np.testing.assert_array_equal(out, frame)


def test_annotate_elements_draws_text_kind_when_include_text_is_true() -> None:
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    out = annotate_elements(
        frame, [_text(1, [0.1, 0.1, 0.3, 0.3])], 800, 600, include_text=True
    )

    assert (out != 0).any()


def test_annotate_elements_uses_green_for_icon_and_red_for_text() -> None:
    # OpenCV is BGR. Green = (0, 255, 0); Red = (0, 0, 255).
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    icon_only = annotate_elements(frame, [_icon(1, [0.1, 0.1, 0.3, 0.3])], 800, 600)
    text_only = annotate_elements(
        frame, [_text(2, [0.1, 0.1, 0.3, 0.3])], 800, 600, include_text=True
    )

    # Icon's rectangle border is green, no red anywhere on the canvas.
    assert (icon_only[:, :, 1] == 255).any()  # green channel saturated
    assert not (icon_only[:, :, 2] == 255).any()  # no saturated red

    # Text's rectangle border is red, no saturated green.
    assert (text_only[:, :, 2] == 255).any()
    assert not (text_only[:, :, 1] == 255).any()


def test_annotate_elements_scales_bbox_by_w_and_h() -> None:
    # bbox [0.1, 0.1, 0.3, 0.3] in a 1000×500 frame → pixel rect
    # (100, 50, 300, 150). The bbox-rectangle border must land inside
    # that range; columns to the right (x > 300) stay untouched.
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)

    out = annotate_elements(frame, [_icon(1, [0.1, 0.1, 0.3, 0.3])], 1000, 500)

    # Inside bbox region — green border drawn.
    assert (out[50:150, 100:300, 1] == 255).any()
    # Far-right of the frame — no annotation pixels.
    assert (out[:, 350:, :] == 0).all()


def test_annotate_elements_uses_bbox_2_for_right_edge_not_bbox_3() -> None:
    # An asymmetric bbox catches index-swap mutations on x2: original
    # `int(bbox[2] * w)` vs mutated `int(bbox[3] * w)` produce visibly
    # different right-edge column positions only when bbox[2] != bbox[3].
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)
    # Rectangle is 200 px tall and 600 px wide — bbox[2]=0.7 (right
    # edge at x=700) vs bbox[3]=0.3 (would shift right edge to x=300).
    out = annotate_elements(frame, [_icon(1, [0.1, 0.1, 0.7, 0.3])], 1000, 500)

    # Far-right of the rectangle border — green pixel must exist past
    # x=600 (where the bbox[3]-mutated right edge would have stopped).
    assert (out[50:150, 600:700, 1] == 255).any()


def test_annotate_elements_continues_past_text_to_draw_later_icon() -> None:
    # The icon-skip `continue` must NOT be `break`, otherwise a text
    # element appearing first would short-circuit the loop and an
    # icon after it would never be drawn.
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    out = annotate_elements(
        frame,
        [
            _text(1, [0.1, 0.1, 0.2, 0.2]),
            _icon(2, [0.5, 0.5, 0.7, 0.7]),
        ],
        800,
        600,
    )

    # The icon's green border lands in the (400-560, 300-420) region.
    assert (out[300:420, 400:560, 1] == 255).any()
