"""Tests for `physiclaw.core.bridge.nonce` — visual screenshot nonce.

`verify_nonce` samples each grey square's center pixel from a synthetic
image we build with the exact module constants (`NONCE_CSS_X`,
`NONCE_CSS_Y`, `NONCE_SQUARE_SIZE`). DPR=1 for most tests so the math
in setup matches the math under test character-for-character; one DPR
test exercises the scaling.

Image layout for DPR=1, no offsets (row-major grid, NONCE_GRID_COLS wide):
  - sample i column: x = NONCE_CSS_X + (i % NONCE_GRID_COLS)*step + step//2
  - sample i row:    y = NONCE_CSS_Y + (i // NONCE_GRID_COLS)*step + step//2
  where step = NONCE_SQUARE_SIZE (DPR=1).

We build BGR images sized just large enough to cover the last sample
row; pixels at sample centers carry the bit-encoded grey level. The
constants are pinned explicitly — both ends of the protocol must agree
on these values, so a constant-bump mutation has to break a test even
when the test's own image-builder uses the same module constants.

Accepted equivalent mutants (cannot be killed without violating the
input invariant):

  - Channel-index swaps in `b, g, r = int(img[cy, cx, 0..2])` — the
    luminance calculation is symmetric `(r + g + b) / 3` and the
    documented input is grey (B=G=R), so any permutation of the three
    reads produces the same luminance. Killing these would require
    a non-grey pixel, which violates the spec.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from physiclaw.core.bridge.nonce import (
    NONCE_COUNT,
    NONCE_CSS_X,
    NONCE_CSS_Y,
    NONCE_DARK,
    NONCE_GRID_COLS,
    NONCE_LIGHT,
    NONCE_SQUARE_SIZE,
    NONCE_THRESHOLD,
    generate_nonce,
    nonce_css_x,
    verify_nonce,
)
from physiclaw.core.calibration.transforms import ViewportShift


def _viewport(dpr: float = 1.0, offset_x: int = 0, offset_y: int = 0) -> ViewportShift:
    # screenshot dimensions large enough that the indices land in-bounds
    # for DPR=1 and offsets=0; tests that need different image sizes pass
    # them in directly to the fixture-builder below.
    return ViewportShift(
        offset_x=offset_x,
        offset_y=offset_y,
        dpr=dpr,
        screenshot_width=400,
        screenshot_height=700,
    )


def _build_image(
    bits: list[int],
    dpr: float = 1.0,
    offset_x: int = 0,
    offset_y: int = 0,
    size: tuple[int, int] = (700, 400),
    css_x: int = NONCE_CSS_X,
) -> np.ndarray:
    """Build a BGR image with sample-pixel grey levels matching `bits`.

    Mirrors the index math in `verify_nonce` so a built image always
    verifies cleanly (when bits match expected).
    """
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    step = int(NONCE_SQUARE_SIZE * dpr)
    base_x = int(css_x * dpr + offset_x)
    base_y = int(NONCE_CSS_Y * dpr + offset_y)
    for i, bit in enumerate(bits):
        cx = base_x + (i % NONCE_GRID_COLS) * step + step // 2
        cy = base_y + (i // NONCE_GRID_COLS) * step + step // 2
        if not (0 <= cy < h and 0 <= cx < w):
            # Caller is deliberately undersizing the image — leave the
            # pixel unwritten so the verifier hits the bounds check.
            continue
        level = NONCE_LIGHT if bit == 1 else NONCE_DARK
        img[cy, cx, :] = level
    return img


# ---------- protocol constants (must match the renderer at the other end) ----------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("NONCE_CSS_X", 180),
        ("NONCE_CSS_Y", 300),
        ("NONCE_COUNT", 64),
        ("NONCE_GRID_COLS", 8),
        ("NONCE_SQUARE_SIZE", 15),
        ("NONCE_DARK", 40),
        ("NONCE_LIGHT", 220),
        ("NONCE_THRESHOLD", 130),
    ],
)
def test_protocol_constant_pinned(name: str, expected: int) -> None:
    # Both the calibration page (renderer) and verify_nonce must agree
    # byte-for-byte on these. A drift in either side breaks every
    # screenshot's nonce — pin them so a bump in one place fails CI.
    from physiclaw.core.bridge import nonce as mod

    assert getattr(mod, name) == expected


# ---------- generate_nonce ----------


def test_generate_nonce_returns_NONCE_COUNT_bits() -> None:
    nonce = generate_nonce()

    assert len(nonce) == NONCE_COUNT


def test_generate_nonce_each_bit_is_zero_or_one() -> None:
    # Run several times so we exercise both branches of randint.
    for _ in range(20):
        nonce = generate_nonce()

        for bit in nonce:
            assert bit in (0, 1)


def test_generate_nonce_uses_random_randint_per_bit(mocker) -> None:
    randint = mocker.patch(
        "physiclaw.core.bridge.nonce.random.randint",
        side_effect=[i % 2 for i in range(NONCE_COUNT)],
    )

    nonce = generate_nonce()

    assert randint.call_count == NONCE_COUNT
    randint.assert_called_with(0, 1)
    assert nonce == [i % 2 for i in range(NONCE_COUNT)]


# ---------- verify_nonce: full match ----------


def test_verify_nonce_all_bits_match_returns_true_and_full_count() -> None:
    bits = [i % 2 for i in range(NONCE_COUNT)]
    img = _build_image(bits)

    ok, count = verify_nonce(img, _viewport(), bits)

    assert ok is True
    assert count == NONCE_COUNT


def test_verify_nonce_all_dark_against_all_dark_expected() -> None:
    bits = [0] * NONCE_COUNT
    img = _build_image(bits)

    ok, count = verify_nonce(img, _viewport(), bits)

    assert ok is True
    assert count == NONCE_COUNT


def test_verify_nonce_all_light_against_all_light_expected() -> None:
    bits = [1] * NONCE_COUNT
    img = _build_image(bits)

    ok, count = verify_nonce(img, _viewport(), bits)

    assert ok is True
    assert count == NONCE_COUNT


# ---------- verify_nonce: mismatch ----------


def test_verify_nonce_one_flipped_bit_returns_false_and_count_minus_one() -> None:
    expected = [0] * NONCE_COUNT
    actual = [0] * NONCE_COUNT
    actual[5] = 1  # flip one square in the rendered image
    img = _build_image(actual)

    ok, count = verify_nonce(img, _viewport(), expected)

    assert ok is False
    assert count == NONCE_COUNT - 1


def test_verify_nonce_mismatch_logs_per_square_diagnostic(
    caplog: pytest.LogCaptureFixture,
) -> None:
    expected = [0] * NONCE_COUNT
    actual = [0] * NONCE_COUNT
    actual[5] = 1
    img = _build_image(actual)

    with caplog.at_level(logging.INFO, logger="physiclaw.core.bridge.nonce"):
        verify_nonce(img, _viewport(), expected)

    # Anchor both halves of the mismatch line — kills the XX-wrapping
    # mutations on each f-string fragment. `endswith("MISMATCH")` denies
    # any trailing XX wrapper.
    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        m.startswith("  Nonce square 5: expected bit 0,")
        and " got bit 1 " in m
        and m.endswith("MISMATCH")
        for m in msgs
    )


def test_verify_nonce_summary_log_reports_pass_on_full_match(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bits = [0] * NONCE_COUNT
    img = _build_image(bits)

    with caplog.at_level(logging.INFO, logger="physiclaw.core.bridge.nonce"):
        verify_nonce(img, _viewport(), bits)

    # Anchored — kills XX-wrapping on the verification line and the
    # 'PASS' literal substitution.
    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        m.startswith(f"  Nonce verification: {NONCE_COUNT}/{NONCE_COUNT} bits matched")
        and m.endswith(" — PASS")
        for m in msgs
    )


def test_verify_nonce_summary_log_reports_fail_on_any_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bits = [0] * NONCE_COUNT
    actual = [0] * NONCE_COUNT
    actual[0] = 1
    img = _build_image(actual)

    with caplog.at_level(logging.INFO, logger="physiclaw.core.bridge.nonce"):
        verify_nonce(img, _viewport(), bits)

    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.endswith(" — FAIL") for m in msgs)


def test_verify_nonce_completely_wrong_returns_false_and_zero_matches() -> None:
    expected = [0] * NONCE_COUNT
    img = _build_image([1] * NONCE_COUNT)

    ok, count = verify_nonce(img, _viewport(), expected)

    assert ok is False
    assert count == 0


# ---------- verify_nonce: threshold boundary ----------


def test_verify_nonce_luminance_exactly_at_threshold_reads_as_bit_one() -> None:
    # Threshold check is `>=`, so luminance == NONCE_THRESHOLD must be bit 1.
    # Mutating `>=` to `>` would push this into bit-0 territory.
    bits = [1] + [0] * (NONCE_COUNT - 1)
    img = _build_image(bits)
    # Overwrite square 0's pixel exactly at the threshold value.
    step = NONCE_SQUARE_SIZE
    cx = NONCE_CSS_X + step // 2
    cy = NONCE_CSS_Y + 0 * step + step // 2
    img[cy, cx, :] = NONCE_THRESHOLD

    ok, count = verify_nonce(img, _viewport(), bits)

    assert ok is True
    assert count == NONCE_COUNT


def test_verify_nonce_luminance_one_below_threshold_reads_as_bit_zero() -> None:
    bits = [0] + [1] * (NONCE_COUNT - 1)
    img = _build_image(bits)
    step = NONCE_SQUARE_SIZE
    cx = NONCE_CSS_X + step // 2
    cy = NONCE_CSS_Y + 0 * step + step // 2
    img[cy, cx, :] = NONCE_THRESHOLD - 1

    ok, count = verify_nonce(img, _viewport(), bits)

    assert ok is True
    assert count == NONCE_COUNT


# ---------- verify_nonce: out-of-bounds handling ----------


def test_verify_nonce_pixel_out_of_bounds_logs_warning_with_actual_dimensions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bits = [1] * NONCE_COUNT
    # Non-square — kills the `shape[0]` ↔ `shape[1]` swap that hides
    # behind a square image. Last sample row falls past height (300).
    img = _build_image(bits, size=(300, 400))

    with caplog.at_level(logging.WARNING, logger="physiclaw.core.bridge.nonce"):
        verify_nonce(img, _viewport(), bits)

    msgs = [r.getMessage() for r in caplog.records]
    # End-anchor on the closing `)` — substring `"(400×300)"` would
    # still appear inside `"XX(400×300)XX"`, but `endswith` rules that
    # out and forces the message to terminate exactly here.
    assert any(
        m.startswith("  Nonce square ")
        and "out of bounds" in m
        and m.endswith("(400×300)")
        for m in msgs
    )


def test_verify_nonce_logs_a_warning_per_out_of_bounds_square(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # `continue` (skip this square, keep iterating) must NOT be `break`.
    # Build an image where many trailing squares are OOB and assert a
    # warning lands for each — a `break` would log only once.
    bits = [1] * NONCE_COUNT
    img = _build_image(bits, size=(350, 400))  # grid rows 3+ (i>=24) are OOB

    with caplog.at_level(logging.WARNING, logger="physiclaw.core.bridge.nonce"):
        verify_nonce(img, _viewport(), bits)

    oob_count = sum(
        1 for r in caplog.records if "out of bounds" in r.getMessage()
    )
    assert oob_count >= 2


def test_verify_nonce_pixel_out_of_bounds_does_not_count_as_match() -> None:
    bits = [1] * NONCE_COUNT
    # Image too small to fit even one sample row → all samples skipped.
    img = np.zeros((NONCE_CSS_Y - 1, 400, 3), dtype=np.uint8)

    ok, count = verify_nonce(img, _viewport(), bits)

    assert ok is False
    assert count == 0


def test_verify_with_image_height_exactly_at_first_sample_row_treats_it_as_oob() -> None:
    # The upper-bound check is `cy < img.shape[0]` (strict). A `<=`
    # mutation would let cy == h slip through and IndexError on the
    # pixel access.
    bits = [1] * NONCE_COUNT
    cy0 = NONCE_CSS_Y + NONCE_SQUARE_SIZE // 2  # = 307
    img = np.zeros((cy0, 400, 3), dtype=np.uint8)

    ok, count = verify_nonce(img, _viewport(), bits)

    assert ok is False
    assert count == 0


def test_verify_with_image_width_exactly_at_sample_column_treats_it_as_oob() -> None:
    bits = [1] * NONCE_COUNT
    cx = NONCE_CSS_X + NONCE_SQUARE_SIZE // 2  # = 187
    img = np.zeros((700, cx, 3), dtype=np.uint8)

    ok, count = verify_nonce(img, _viewport(), bits)

    assert ok is False
    assert count == 0


def test_verify_includes_pixel_at_y_equals_zero_lower_bound() -> None:
    # Negative offset_y pushes square 0's row to cy=0; the lower-bound
    # check is `<=`, so the pixel must be read (not skipped). Mutating
    # `0 <= cy` to `1 <= cy` or `0 < cy` would skip it and lose a match.
    cy0 = NONCE_CSS_Y + NONCE_SQUARE_SIZE // 2
    bits = [1] + [0] * (NONCE_COUNT - 1)
    img = _build_image(bits, offset_y=-cy0)

    ok, count = verify_nonce(img, _viewport(offset_y=-cy0), bits)

    assert ok is True
    assert count == NONCE_COUNT


def test_verify_includes_pixel_at_x_equals_zero_lower_bound() -> None:
    cx0 = NONCE_CSS_X + NONCE_SQUARE_SIZE // 2
    bits = [i % 2 for i in range(NONCE_COUNT)]
    img = _build_image(bits, offset_x=-cx0)

    ok, count = verify_nonce(img, _viewport(offset_x=-cx0), bits)

    assert ok is True
    assert count == NONCE_COUNT


# ---------- nonce_css_x + css_x override ----------


def test_nonce_css_x_centers_the_grid_on_the_viewport() -> None:
    # Left edge + half the grid width must land on the viewport midline.
    grid_w = NONCE_GRID_COLS * NONCE_SQUARE_SIZE
    for vw in (375, 390, 430):
        x = nonce_css_x(vw)

        assert x + grid_w / 2 == pytest.approx(vw / 2, abs=0.5)


def test_verify_nonce_samples_at_the_given_css_x() -> None:
    bits = [i % 2 for i in range(NONCE_COUNT)]
    x = nonce_css_x(390)  # 135 — differs from the NONCE_CSS_X fallback
    img = _build_image(bits, css_x=x)

    ok, count = verify_nonce(img, _viewport(), bits, css_x=x)

    assert ok is True
    assert count == NONCE_COUNT


def test_verify_nonce_css_x_none_falls_back_to_NONCE_CSS_X() -> None:
    bits = [1] * NONCE_COUNT
    img = _build_image(bits)  # rendered at the NONCE_CSS_X fallback

    ok, count = verify_nonce(img, _viewport(), bits, css_x=None)

    assert ok is True
    assert count == NONCE_COUNT


def test_verify_nonce_wrong_css_x_reads_unwritten_pixels() -> None:
    # Renderer and sampler disagreeing on x must FAIL the verification —
    # all-black pixels read as bit 0 against expected bit 1.
    bits = [1] * NONCE_COUNT
    img = _build_image(bits, css_x=135)

    ok, _ = verify_nonce(img, _viewport(), bits, css_x=NONCE_CSS_X)

    assert ok is False


# ---------- verify_nonce: DPR + offsets ----------


def test_verify_nonce_uses_dpr_to_scale_step_and_base() -> None:
    bits = [i % 2 for i in range(NONCE_COUNT)]
    img = _build_image(bits, dpr=2.0, size=(1400, 800))

    ok, count = verify_nonce(img, _viewport(dpr=2.0), bits)

    assert ok is True
    assert count == NONCE_COUNT


def test_verify_nonce_applies_viewport_offsets_to_base_coords() -> None:
    bits = [i % 2 for i in range(NONCE_COUNT)]
    img = _build_image(bits, offset_x=10, offset_y=20)

    ok, count = verify_nonce(img, _viewport(offset_x=10, offset_y=20), bits)

    assert ok is True
    assert count == NONCE_COUNT


# ---------- Hypothesis round-trip ----------


@given(
    bits=st.lists(
        st.sampled_from([0, 1]), min_size=NONCE_COUNT, max_size=NONCE_COUNT
    )
)
@settings(deadline=None, max_examples=50)
def test_verify_round_trip_holds_for_any_bit_sequence(bits: list[int]) -> None:
    img = _build_image(bits)

    ok, count = verify_nonce(img, _viewport(), bits)

    assert ok is True
    assert count == NONCE_COUNT
