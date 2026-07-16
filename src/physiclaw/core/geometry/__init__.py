"""Coordinate geometry — the leaf package below bridge/calibration/hardware.

Two modules, one public surface (re-exported here):
- ``affine`` — domain-free point math (`apply_affine`, `invert_affine`);
- ``screen`` — the render↔measure contract (`ViewportShift` + the
  page-target constants both bridge and calibration must agree on).
"""

from physiclaw.core.geometry.affine import apply_affine, invert_affine
from physiclaw.core.geometry.screen import (
    AT_CSS_X,
    AT_CSS_Y,
    AT_RADIUS,
    GRID_COLS_PCT,
    GRID_ROWS_PCT,
    NONCE_COUNT,
    NONCE_CSS_X,
    NONCE_CSS_Y,
    NONCE_DARK,
    NONCE_GRID_COLS,
    NONCE_LIGHT,
    NONCE_SQUARE_SIZE,
    NONCE_THRESHOLD,
    SQUARE_CSS_SIZE,
    SQUARE_CSS_X,
    SQUARE_CSS_Y,
    ViewportShift,
    nonce_css_x,
)

__all__ = [
    "AT_CSS_X",
    "AT_CSS_Y",
    "AT_RADIUS",
    "GRID_COLS_PCT",
    "GRID_ROWS_PCT",
    "NONCE_COUNT",
    "NONCE_CSS_X",
    "NONCE_CSS_Y",
    "NONCE_DARK",
    "NONCE_GRID_COLS",
    "NONCE_LIGHT",
    "NONCE_SQUARE_SIZE",
    "NONCE_THRESHOLD",
    "SQUARE_CSS_SIZE",
    "SQUARE_CSS_X",
    "SQUARE_CSS_Y",
    "ViewportShift",
    "apply_affine",
    "invert_affine",
    "nonce_css_x",
]
