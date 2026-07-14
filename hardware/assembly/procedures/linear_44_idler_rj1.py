"""Linear-stage idler bundle (RJ1) — a SHOULDER M4 × 10 mm axle
carrying a smooth (no-teeth) GT2 idler over an M5×8×0.5 washer, no
spacer, ready to bolt into the host's tapped M4 hole.

Same construction as linear_42_idler_lj2 (LI42IdlerLj2) — only the
idler tooth profile differs (smooth here vs toothed there). The
build logic is reused via inheritance; this class just overrides
four class attributes.

Stack height = 0.5 + 8.5 = 9 mm; shoulder 10 mm → 1 mm of free
shoulder above the thread for axial play.

See linear_40_idler_lj1 for the full stack model, variant
descriptions, and exploded-view conventions.

Run from the repo root:

    uv run --group cad python -m hardware step linear_44_idler_rj1
"""

from hardware.assembly.base import ASSEMBLED_CAM0, EXPLODED_CAM0
from hardware.assembly.procedures.linear_40_idler_lj1 import LI40IdlerLj1


class LI44IdlerRj1(LI40IdlerLj1):
    compound_label = "linear_44_idler_rj1"
    views = [EXPLODED_CAM0, ASSEMBLED_CAM0]
    toothed = False
    include_spacer = False
    shoulder_len = 10  # mm — pairs with the 9 mm washer + idler stack
