"""Binary nonce barcode — layout, generation, and verification.

The calibration page renders NONCE_COUNT grey squares (1 bit each) of
NONCE_SQUARE_SIZE CSS px, starting at (NONCE_CSS_X, NONCE_CSS_Y). Greys
lie on the achromatic axis where Display P3 and sRGB agree, so the iOS
screenshot ICC profile shift can't fool the verifier.
"""

import logging
import random

import numpy as np

from physiclaw.calibration.transforms import ViewportShift

log = logging.getLogger(__name__)

NONCE_CSS_X = 180
NONCE_CSS_Y = 300
NONCE_COUNT = 20
NONCE_SQUARE_SIZE = 15  # CSS pixels per square
NONCE_DARK = 40  # bit 0 — dark grey
NONCE_LIGHT = 220  # bit 1 — light grey
NONCE_THRESHOLD = 130  # luminance midpoint between dark and light


def generate_nonce() -> list[int]:
    """Generate a random binary sequence (NONCE_COUNT bits) for verification."""
    return [random.randint(0, 1) for _ in range(NONCE_COUNT)]


def verify_nonce(
    img: np.ndarray, t: ViewportShift, expected_bits: list[int]
) -> tuple[bool, int]:
    """Verify the binary nonce barcode in a screenshot.

    Samples the center pixel of each square (scaled by DPR), thresholds its
    luminance to a bit, and compares to the expected sequence.

    Returns (all_matched, match_count).
    """
    dpr = t.dpr
    step = int(NONCE_SQUARE_SIZE * dpr)
    base_x = int(NONCE_CSS_X * dpr + t.offset_x)
    base_y = int(NONCE_CSS_Y * dpr + t.offset_y)

    matched = 0
    for i, expected in enumerate(expected_bits):
        cx = base_x + step // 2
        cy = base_y + i * step + step // 2
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
