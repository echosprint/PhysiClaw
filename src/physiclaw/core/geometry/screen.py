"""Shared screen-geometry contract — the render↔measure constants.

The phone page (``core/bridge``) RENDERS visual targets and the
calibration steps (``core/calibration``) MEASURE them; both sides must
agree on where each target sits, so the geometry lives in this leaf
package. This is what keeps bridge and calibration free of a mutual
dependency: bridge draws from this contract, calibration measures
against it.

Contents:
- ``ViewportShift`` — viewport CSS px → screenshot 0-1 mapping, the one
  hop from phone units into the canonical screenshot space;
- the pre-cal orange square, the calibration dot grid, the
  AssistiveTouch button, and the nonce-grid layout.

The affine point-math primitives live in the sibling ``affine`` module.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class ViewportShift:
    """Viewport CSS pixels → screenshot 0-1 coordinates.

    Produced by the pre-calibration step: server shows an orange square at
    a known CSS position, user uploads a screenshot, server detects the
    square and derives (dpr, offset_x, offset_y) from the mismatch.

    Fields:
        offset_x, offset_y: screenshot-pixel offset caused by iOS status
            bar / safe area (viewport origin is not at screenshot origin).
        dpr: device pixel ratio — CSS px → screenshot px scale factor.
        screenshot_width, screenshot_height: screenshot image size in px.
    """

    offset_x: float
    offset_y: float
    dpr: float
    screenshot_width: int
    screenshot_height: int

    def css_to_pct(self, css_x: float, css_y: float) -> tuple[float, float]:
        """Convert viewport CSS pixel coords to screenshot 0-1 coords."""
        sx = (css_x * self.dpr + self.offset_x) / self.screenshot_width
        sy = (css_y * self.dpr + self.offset_y) / self.screenshot_height
        return (sx, sy)


# Pre-cal square (phase "screenshot_cal"): CSS top-left + size. The page
# draws exactly this; viewport.py's screenshot mapping expects it here.
SQUARE_CSS_X, SQUARE_CSS_Y, SQUARE_CSS_SIZE = 100, 200, 50

# Calibration dot grid, viewport 0-1 (must match bridge.html and the
# calibration plan): 3 columns × 5 rows = 15 dots. Tuples on purpose —
# render and measurement share these objects, so an in-place mutation
# would silently rewrite the contract for both sides at once.
GRID_COLS_PCT = (0.25, 0.50, 0.75)
GRID_ROWS_PCT = (0.20, 0.40, 0.50, 0.60, 0.80)

# AssistiveTouch button position in CSS viewport pixels (iPhone left
# edge snap): 11pt edge margin + 28pt button radius; 56pt diameter.
AT_CSS_X = 39
AT_CSS_Y = 260  # hardcoded vertical position
AT_RADIUS = 28

# Nonce grid layout (see core/bridge/nonce.py for generation/verify):
# NONCE_COUNT grey squares of NONCE_SQUARE_SIZE CSS px in a
# NONCE_GRID_COLS-wide row-major grid, horizontally centered on the
# viewport (nonce_css_x), top edge at NONCE_CSS_Y. Greys lie on the
# achromatic axis where Display P3 and sRGB agree, so the iOS screenshot
# ICC profile shift can't fool the verifier.
NONCE_CSS_X = 180  # fallback left edge — used only when viewport width unknown
NONCE_CSS_Y = 300
NONCE_GRID_COLS = 8  # squares per row — NONCE_COUNT/NONCE_GRID_COLS rows
NONCE_COUNT = 64  # 8×8 grid
NONCE_SQUARE_SIZE = 15  # CSS pixels per square
NONCE_DARK = 40  # bit 0 — dark grey
NONCE_LIGHT = 220  # bit 1 — light grey
NONCE_THRESHOLD = 130  # luminance midpoint between dark and light


def nonce_css_x(viewport_width: float) -> int:
    """Left edge (CSS px) that centers the grid on the viewport's x axis."""
    return round(viewport_width / 2 - NONCE_GRID_COLS * NONCE_SQUARE_SIZE / 2)
