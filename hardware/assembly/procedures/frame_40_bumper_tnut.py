"""Bumper fastening step — one cylindrical bumper attached by 1 M5×16
BHCS driven UP from below into 1 hammer T-nut catching the shank tip
above. Mirror of frame_30_bracket_tnut, but with the screw axis inverted: the
bumper's bottom-face cbore receives the head from below.

Two variants:

  * exploded — three part kinds stacked along the screw axis (world
               +Z): BHCS at the bottom (head down, shank up) with
               SCREW_GAP of air below the bumper bottom face; bumper
               in the middle with its cbore opening downward; T-nut
               above with TNUT_GAP between the bumper top and the
               boss top. Order of operations reads bottom-up.
  * assembled — loose-mate state, not yet fixed to the frame: BHCS
                head bottomed in the cbore (underhead at z = cbore_depth)
                and the t-nut hanging from the shank tip (boss top
                touches the tip), still floating above the bumper by
                the shank protrusion.

  * 1 x Bumper (Ø20 × 22 mm, Ø14 × 16 cbore on bottom, Ø5 through)
  * 1 x BHCS M5 × 16 — driven up from below
  * 1 x TNut "hammer" M5 — catches the shank tip from above (boss down)

Run from the repo root:

    uv run --group cad python -m hardware step frame_40_bumper_tnut
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import BaseAssembly, EXPLODED_CAM0
from hardware.assembly.projection import FRONT_LEFT_HIGH
from hardware.parts.standard.bumper import (
    Bumper,
    body_height as bumper_height,
    cbore_depth,
)
from hardware.parts.standard.screw import Screw
from hardware.parts.standard.t_nut import (
    HAMMER_TOTAL_HEIGHT,
    LENGTHS as TNUT_LENGTHS,
    TNut,
)

BHCS_LENGTH = 16  # mm — BHCS M5 underhead length
TNUT_GAP = 20  # mm — exploded: vertical gap, bumper top → t-nut boss top
SCREW_GAP = 8  # mm — exploded: vertical gap, bumper bottom → shank tip


class FR40BumperTnut(BaseAssembly):
    camera = FRONT_LEFT_HIGH
    views = [EXPLODED_CAM0]

    def _build(self) -> Compound:
        bumper = Bumper().build()
        nut = TNut("hammer", "M5").build()
        screw = Screw("BHCS", "M5", BHCS_LENGTH).build()

        # Bumper is built centered on its own Z (z ∈ [-h/2, +h/2]); lift
        # so the bottom face sits at world z=0. cbore opens at z=0 and
        # floors at z=cbore_depth (=16); the Ø5 through-hole spans the
        # full 22 mm.
        bumper.move(Location((0, 0, bumper_height / 2)))

        # Layout per variant. cbore_depth doubles as the world Z of the
        # cbore floor — where the BHCS head bottoms in assembled state.
        if self.exploded:
            shank_tip_z = -SCREW_GAP
            boss_top_z = bumper_height + TNUT_GAP
        else:
            # Loose-mate state — bumper not yet fastened to the frame.
            # BHCS head bottoms in the cbore; t-nut hangs from the shank
            # tip (boss top touches the tip), still clear of the bumper
            # top by the shank protrusion.
            shank_tip_z = cbore_depth + BHCS_LENGTH
            boss_top_z = shank_tip_z

        # Screw default: head at part +Z, shank at part −Z, underhead at
        # part z=0. Rotate 180° about X to flip head-down / shank-up,
        # then translate so the shank tip lands at shank_tip_z. In the
        # flipped frame the shank tip is +length above the seating plane,
        # so the seating plane (= translation Z) is shank_tip_z − length.
        screw.move(
            Location(
                (0, 0, shank_tip_z - BHCS_LENGTH),
                (180, 0, 0),
            )
        )

        # T-nut local: +Y is bore/boss axis, +Z is slide length. The
        # plane below sends local +Z → world +Y (slide axis along world
        # Y) and local +Y → world −Z (boss points down, toward the
        # bumper). Origin is set so the bore axis (local x=0, z=length/2)
        # lands on the world Z axis and the boss top (local
        # y=HAMMER_TOTAL_HEIGHT) lands at boss_top_z.
        slide_half = TNUT_LENGTHS["hammer"] / 2
        nut.move(
            Location(
                Plane(
                    origin=(0, -slide_half, boss_top_z + HAMMER_TOTAL_HEIGHT),
                    x_dir=(1, 0, 0),
                    z_dir=(0, 1, 0),
                )
            )
        )

        return Compound(
            label="frame_40_bumper_tnut",
            children=[
                bumper,
                nut,
                screw,
            ],
        )
