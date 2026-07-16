"""Affine point math — the coordinate-frame primitives.

Domain-free: nothing here knows about screens, arms, or cameras. The
sibling ``screen`` module holds the domain constants; both live in this
leaf package because ``hardware`` (the AssistiveTouch driver applies
the arm affine) must reach them without importing ``calibration``.
"""

import numpy as np


def apply_affine(affine: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Apply a (2, 3) affine to a point — THE point-mapping primitive.

    Every screen↔arm↔camera point conversion goes through here (mostly
    via ``ScreenTransforms``); nothing else in the codebase multiplies a
    coordinate by an affine, so a change to the convention (a y-flip, a
    perspective term) is a one-place edit."""
    pt = affine @ np.array([x, y, 1.0])
    return (float(pt[0]), float(pt[1]))


def invert_affine(affine: np.ndarray) -> np.ndarray:
    """The (2, 3) affine mapping the other direction.

    The single inverse implementation — e.g. Mapping B (screen 0-1 →
    camera 0-1) inverted for back-projection during validation."""
    A = affine[:, :2]
    b = affine[:, 2]
    A_inv = np.linalg.inv(A)
    return np.hstack([A_inv, (-A_inv @ b).reshape(2, 1)])
