"""Binary nonce grid — layout, generation, and verification.

The calibration page renders NONCE_COUNT grey squares (1 bit each) of
NONCE_SQUARE_SIZE CSS px in a NONCE_GRID_COLS-wide grid (row-major),
horizontally centered on the viewport (see :func:`nonce_css_x`), top
edge at NONCE_CSS_Y. The server computes the left edge and both serves
it to the page and passes it to :func:`verify_nonce`, so renderer and
verifier share one value by construction; NONCE_CSS_X is only the
fallback when the viewport width isn't known. Greys lie on the
achromatic axis where Display P3 and sRGB agree, so the iOS screenshot
ICC profile shift can't fool the verifier.
"""

import logging
import random

import numpy as np

from physiclaw.core.calibration.transforms import ViewportShift

log = logging.getLogger(__name__)

NONCE_CSS_X = 180  # fallback left edge — used only when viewport width unknown
NONCE_CSS_Y = 300
NONCE_GRID_COLS = 8  # squares per row — NONCE_COUNT/NONCE_GRID_COLS rows
NONCE_COUNT = 64  # 8×8 grid
NONCE_SQUARE_SIZE = 15  # CSS pixels per square


def nonce_css_x(viewport_width: float) -> int:
    """Left edge (CSS px) that centers the grid on the viewport's x axis."""
    return round(viewport_width / 2 - NONCE_GRID_COLS * NONCE_SQUARE_SIZE / 2)


NONCE_DARK = 40  # bit 0 — dark grey
NONCE_LIGHT = 220  # bit 1 — light grey
NONCE_THRESHOLD = 130  # luminance midpoint between dark and light


def generate_nonce() -> list[int]:
    """Generate a random binary sequence (NONCE_COUNT bits) for verification."""
    return [random.randint(0, 1) for _ in range(NONCE_COUNT)]


def verify_nonce(
    img: np.ndarray,
    t: ViewportShift,
    expected_bits: list[int],
    css_x: int | None = None,
) -> tuple[bool, int]:
    """Verify the binary nonce grid in a screenshot.

    Samples the center pixel of each square (scaled by DPR), thresholds its
    luminance to a bit, and compares to the expected sequence. `css_x` is
    the grid's left edge as served to the page (CalibrationState.nonce_x);
    None falls back to NONCE_CSS_X.

    Returns (all_matched, match_count).
    """
    dpr = t.dpr
    step = int(NONCE_SQUARE_SIZE * dpr)
    base_x = int((NONCE_CSS_X if css_x is None else css_x) * dpr + t.offset_x)
    base_y = int(NONCE_CSS_Y * dpr + t.offset_y)

    matched = 0
    for i, expected in enumerate(expected_bits):
        cx = base_x + (i % NONCE_GRID_COLS) * step + step // 2
        cy = base_y + (i // NONCE_GRID_COLS) * step + step // 2
        if not (0 <= cy < img.shape[0] and 0 <= cx < img.shape[1]):
            log.warning(
                f"  Nonce square {i}: pixel ({cx}, {cy}) out of bounds "
                f"({img.shape[1]}×{img.shape[0]})"
            )
            continue
        # OpenCV is BGR; mean of channels is fine since the square is grey.
        b, g, r = int(img[cy, cx, 0]), int(img[cy, cx, 1]), int(img[cy, cx, 2])
        luminance = (r + g + b) / 3
        actual = 1 if luminance >= NONCE_THRESHOLD else 0
        if actual == expected:
            matched += 1
        else:
            log.info(
                f"  Nonce square {i}: expected bit {expected}, "
                f"got bit {actual} (luminance {luminance:.0f}) — MISMATCH"
            )

    all_matched = matched == len(expected_bits)
    log.info(
        f"  Nonce verification: {matched}/{len(expected_bits)} bits matched"
        f" — {'PASS' if all_matched else 'FAIL'}"
    )
    return all_matched, matched
