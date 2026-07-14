"""Linear-stage idler bundle (RJ2) — a SHOULDER M4 × 20 mm axle
carrying a toothed GT2 idler over an M5×8×0.5 washer on top of an
M5×10×9 spacer, ready to bolt into the host's tapped M4 hole.

Same construction as linear_40_idler_lj1 (LI40IdlerLj1) — only the
idler tooth profile differs (toothed here vs smooth there). The
build logic is reused via inheritance; this class just overrides
the compound label and the tooth flag.

Stack height = 9 + 0.5 + 8.5 = 18 mm; shoulder 20 mm → 2 mm of free
shoulder above the thread for axial play.

See linear_40_idler_lj1 for the full stack model, variant
descriptions, and exploded-view conventions.

Run from the repo root:

    uv run --group cad python -m hardware step linear_46_idler_rj2
"""

from hardware.assembly.base import ASSEMBLED_CAM0, EXPLODED_CAM0
from hardware.assembly.procedures.linear_40_idler_lj1 import LI40IdlerLj1


class LI46IdlerRj2(LI40IdlerLj1):
    compound_label = "linear_46_idler_rj2"
    views = [EXPLODED_CAM0, ASSEMBLED_CAM0]
    toothed = True
