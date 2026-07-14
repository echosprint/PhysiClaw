"""Idler-mount bracket (left-up) — flat bracket tying the IdlerMountMotor
lu block to a frame extrusion via two M5 BHCS. One screw catches a
hammer T-nut in the frame slot; the other threads into the captive M5
square nut that idler_10_lu already seated in the lu block's top pocket.

Hole pair (along bracket local X, 20 mm apart):
  * LEFT  — frame side: BHCS M5 × 10 + hammer T-nut M5 (captured in
            the frame extrusion slot, not modelled here). 10 mm is
            plenty: 4 bracket + 4.5 nut thread = 8.5, tip in open slot.
  * RIGHT — block side: BHCS M5 × 12, no nut in this sub-assembly —
            its mate is the captive top-pocket square nut from
            idler_10_lu. 12 mm is required: 4 bracket + 3.5 face-to-
            pocket standoff + 4 nut = 11.5; an M5×10 would engage only
            2.5 mm of the nut.

Both variants follow frame_30_bracket_tnut's bracket convention so
this reads as the "block-tying twin" of that step:

  * exploded — chain along world +Z: t-nut at the bottom (LEFT hole
               only), bracket above with BRACKET_GAP of air, both
               screws above with SCREW_GAP air below their shank tips.
               The RIGHT shank tip shares the LEFT's line; its head
               rides 2 mm higher (the RIGHT screw is 2 mm longer).
  * assembled — loose-mate: both BHCS dropped through the bracket
                (underheads on bracket top); the LEFT shank carries
                the t-nut hanging from its tip (boss top touching the
                tip, then slid up by TNUT_LIFT to read as
                "threaded most of the way in"). The RIGHT shank just
                protrudes — its mate lives on the block side.

  * 1 x FlatBracket (40 × 18 × 4 mm, two M5 through-holes 20 mm apart)
  * 1 x BHCS M5 × 10 — LEFT (frame-slot side), driven down through the bracket
  * 1 x BHCS M5 × 12 — RIGHT (block side), driven down through the bracket
  * 1 x TNut "hammer" M5 — LEFT hole only (frame-slot side)

Run from the repo root:

    uv run --group cad python -m hardware step idler_11_lu
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import ASSEMBLED_CAM0, BaseAssembly, EXPLODED_CAM0
from hardware.assembly.projection import FRONT_LEFT_LOW
from hardware.parts.standard.bracket import (
    FlatBracket,
    flat_hole_spacing as hole_spacing,
    flat_plate_thick as plate_thick,
)
from hardware.parts.standard.screw import Screw
from hardware.parts.standard.t_nut import (
    HAMMER_TOTAL_HEIGHT,
    LENGTHS as TNUT_LENGTHS,
    TNut,
)

FRAME_BHCS_LENGTH = 10  # mm — LEFT, frame-slot side: into the hammer T-nut
BLOCK_BHCS_LENGTH = 12  # mm — RIGHT, block side: 4 bracket + 3.5 standoff
#      + 4 nut = 11.5 needed for full engagement
BRACKET_GAP = 20  # mm — exploded: visual gap between t-nut top and bracket bottom
SCREW_GAP = 8  # mm — exploded: visual gap between bracket top and screw shank tips
TNUT_LIFT = 5  # mm — assembled: nut slid up the shank toward the bracket
#      (closes the 6 mm shank-protrusion gap to ~1 mm,
#      reads as "screw threaded most of the way in")


class ID11Lu(BaseAssembly):
    # Subclasses share this build logic and only override
    # ``compound_label`` (e.g. ID21Ru reuses the same geometry for the
    # right-up corner). ``_module_stem()`` already derives the STEP/SVG
    # filename from the subclass's own module, so no other override is
    # needed.
    compound_label: str = "idler_11_lu"
    camera = FRONT_LEFT_LOW
    views = [EXPLODED_CAM0, ASSEMBLED_CAM0]

    def _build(self) -> Compound:
        bracket = FlatBracket().build()
        nut = TNut("hammer", "M5").build()
        lengths = (FRAME_BHCS_LENGTH, BLOCK_BHCS_LENGTH)  # LEFT, RIGHT
        screws = [Screw("BHCS", "M5", length).build() for length in lengths]

        half_hole = hole_spacing / 2
        hammer_half_length = TNUT_LENGTHS["hammer"] / 2

        # T-nut sits under the LEFT hole only (frame-slot side). Same
        # orientation as the LEFT nut in frame_30_bracket_tnut: local +Y
        # → world +Z (boss up, bore vertical, aligned with the screw
        # shank dropping from above), slide axis along world -Y. In
        # assembled mode the nut is slid TNUT_LIFT up the shank toward
        # the bracket; exploded keeps it on the floor at z=0.
        tnut_lift = 0 if self.exploded else TNUT_LIFT
        nut.move(
            Location(
                Plane(
                    origin=(-half_hole, hammer_half_length, tnut_lift),
                    x_dir=(1, 0, 0),
                    z_dir=(0, -1, 0),
                )
            )
        )

        # Layout per variant (mirrors frame_30_bracket_tnut so the two
        # files read identically apart from the missing RIGHT t-nut):
        #   exploded:  shank tips float SCREW_GAP above the bracket top
        #              (tips aligned, so each head sits at its own
        #              length above that line); bracket sits BRACKET_GAP
        #              above the t-nut top.
        #   assembled: both BHCS dropped through the bracket with their
        #              underheads on the bracket top; the LEFT t-nut
        #              hangs from the LEFT shank tip — bracket floats
        #              above the t-nut by (FRAME_BHCS_LENGTH − plate_thick).
        #              The 2 mm-longer RIGHT shank simply protrudes lower.
        if self.exploded:
            bracket_z = HAMMER_TOTAL_HEIGHT + BRACKET_GAP + plate_thick / 2
            tip_z = bracket_z + plate_thick / 2 + SCREW_GAP
            screw_zs = [tip_z + length for length in lengths]
        else:
            screw_z = HAMMER_TOTAL_HEIGHT + FRAME_BHCS_LENGTH
            bracket_z = screw_z - plate_thick / 2
            screw_zs = [screw_z, screw_z]

        bracket.move(Location((0, 0, bracket_z)))
        for screw, side, screw_z in zip(screws, (-half_hole, half_hole), screw_zs):
            screw.move(Location((side, 0, screw_z)))

        # Expose the bracket-bottom height so a host assembly can
        # flush-mount this sub-assembly — same hook as FR30BracketTnut.
        self.bracket_bottom_z = bracket_z - plate_thick / 2

        return Compound(
            label=self.compound_label,
            children=[
                bracket,
                nut,
                *screws,
            ],
        )
