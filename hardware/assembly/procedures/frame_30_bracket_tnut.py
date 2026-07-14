"""Bracket fastening step — one flat bracket attached by 2 M5×10 BHCS
into 2 hammer T-nuts.

Two variants:

  * exploded — three part kinds stacked along the screw axis (world
               +Z): t-nuts at the bottom, bracket above with BRACKET_GAP
               of air, screws above with SCREW_GAP air below their
               shank tips. Order of operations reads top-down.
  * assembled — loose-mate state: each BHCS dropped through the
                bracket (underhead on the bracket top), with the t-nut
                hanging from the shank tip (boss top touches the tip).
                The bracket floats (BHCS_LENGTH − plate_thick) above
                the t-nut — visible threading clearance, screw not
                tightened.

  * 1 x FlatBracket (40 × 18 × 4 mm, two M5 through-holes 20 mm apart)
  * 2 x BHCS M5 × 10 — driven down through the bracket
  * 2 x TNut "hammer" M5 — catches the screw from below

Run from the repo root:

    uv run --group cad python -m hardware step frame_30_bracket_tnut
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import BaseAssembly, EXPLODED_CAM0
from hardware.assembly.projection import FRONT_HIGH_L10
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

BHCS_LENGTH = 10  # mm — BHCS M5 underhead length
BRACKET_GAP = 20  # mm — exploded: visual gap between t-nut top and bracket bottom
SCREW_GAP = 8  # mm — exploded: visual gap between bracket top and screw shank tip
TNUT_LIFT = 5  # mm — assembled: nut slid up the shank toward the bracket
#      (closes the 6 mm shank-protrusion gap to ~1 mm,
#      reads as "screw threaded most of the way in")


class FR30BracketTnut(BaseAssembly):
    camera = FRONT_HIGH_L10
    views = [EXPLODED_CAM0]

    def _build(self) -> Compound:
        bracket = FlatBracket().build()
        nuts = [TNut("hammer", "M5").build() for _ in range(2)]
        screws = [Screw("BHCS", "M5", BHCS_LENGTH).build() for _ in range(2)]

        # T-nut local: +Y is bore/boss direction, +Z is slide length. Both
        # nuts have +Y → world +Z (boss up, bore vertical); their bores
        # land at world (±half_hole, 0, 0), directly under the matching
        # bracket hole. The two slide axes are 90° apart about world Z
        # — the left nut runs along world -Y, the right along world +X
        # — so the reader sees both installation orientations.
        half_hole = hole_spacing / 2
        hammer_half_length = TNUT_LENGTHS["hammer"] / 2
        # In assembled mode the nut is slid TNUT_LIFT up the shank
        # toward the bracket; exploded keeps the floor at z=0.
        tnut_lift = 0 if self.exploded else TNUT_LIFT
        nut_planes = (
            Plane(
                origin=(-half_hole, hammer_half_length, tnut_lift),
                x_dir=(1, 0, 0),
                z_dir=(0, -1, 0),  # slide axis along world -Y
            ),
            Plane(
                origin=(half_hole - hammer_half_length, 0, tnut_lift),
                x_dir=(0, 1, 0),
                z_dir=(1, 0, 0),  # slide axis along world +X
            ),
        )
        for nut, plane in zip(nuts, nut_planes):
            nut.move(Location(plane))

        # Screw default: head at part +Z, shank at part -Z. Identity
        # rotation keeps head up. Layout per variant:
        #   exploded:  shank tip floats SCREW_GAP above the bracket top
        #              (bracket itself sits BRACKET_GAP above the t-nut).
        #   assembled: BHCS dropped through the bracket (underhead on
        #              bracket top) with the t-nut hanging from its
        #              shank tip — bracket floats above the t-nut by
        #              the shank protrusion (BHCS_LENGTH − plate_thick).
        if self.exploded:
            bracket_z = HAMMER_TOTAL_HEIGHT + BRACKET_GAP + plate_thick / 2
            screw_z = bracket_z + plate_thick / 2 + SCREW_GAP + BHCS_LENGTH
        else:
            screw_z = HAMMER_TOTAL_HEIGHT + BHCS_LENGTH
            bracket_z = screw_z - plate_thick / 2

        bracket.move(Location((0, 0, bracket_z)))
        for screw, side in zip(screws, (-half_hole, half_hole)):
            screw.move(Location((side, 0, screw_z)))

        # Expose the bracket-bottom height (local Z) so a host assembly
        # can flush-mount this sub-assembly against a target surface —
        # e.g. frame_31_bracket puts bracket_bottom_z on the slot face.
        self.bracket_bottom_z = bracket_z - plate_thick / 2

        return Compound(
            label="frame_30_bracket_tnut",
            children=[
                bracket,
                *nuts,
                *screws,
            ],
        )
