"""Linear-stage idler bundle (LJ2) — a SHOULDER M4 × 10 mm axle
carrying a toothed GT2 idler over an M5×8×0.5 washer, no spacer,
ready to bolt into the host's tapped M4 hole.

Same construction as linear_40_idler_lj1 (LI40IdlerLj1) — only the
idler tooth profile, the absence of the spacer, and the matching
shoulder length differ. The build logic is reused via inheritance;
this class just overrides four class attributes.

Stack height = 0.5 + 8.5 = 9 mm; shoulder 10 mm → 1 mm of free
shoulder above the thread for axial play.

See linear_40_idler_lj1 for the full stack model, variant
descriptions, and exploded-view conventions.

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.linear_42_idler_lj2
"""

from hardware.assembly.procedures.linear_40_idler_lj1 import LI40IdlerLj1


class LI42IdlerLj2(LI40IdlerLj1):
    compound_label = "linear_42_idler_lj2"
    toothed = True
    include_spacer = False
    shoulder_len = 10  # mm — pairs with the 9 mm washer + idler stack


if __name__ == "__main__":
    for exploded in (True, False):
        asm = LI42IdlerLj2(exploded=exploded)
        asm.export()
        asm.render()
