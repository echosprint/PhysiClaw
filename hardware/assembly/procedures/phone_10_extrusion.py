"""Phone-bed cross-beam — sub-assembly of one 230 mm Extrusion1020 with the
hardware that fastens it under a frame Y extrusion.

At each of the 1020's two vertical end holes (Extrusion1020(hole=True): an M5
hole 10 mm in from each end, centered in width):

  * a BHCS M5 × 16 enters from the BOTTOM and points UP — its head bears on
    the 1020 underside, the shank passes up through the hole and protrudes
    above the top face.
  * a hammer M5 T-nut sits on top, UPSIDE DOWN (boss/mouth facing down toward
    the beam) — its threaded boss receives the screw shank. In the next step
    this nut slides into the downward-facing slot on the bottom of the frame Y
    extrusion, and the screw clamps the beam up against it.

Geometry in the 1020's native frame (matches Extrusion1020: width along X,
height native Y = 0 (bottom) → 9.9 (top), length along native Z = 0 → length).

Two variants:
  * exploded — screws dropped below the beam and nuts lifted above it, along
               the shared vertical (Y) install axis.
  * assembled — screws seated (heads on the underside) and nuts on the top
                face, threaded on the protruding shanks.

Run from the repo root:

    uv run --group cad python -m hardware step phone_10_extrusion
"""

from build123d import Compound, Location

from hardware.assembly.base import ASSEMBLED_CAM1, EXPLODED_CAM1, BaseAssembly
from hardware.assembly.projection import FRONT_LEFT_HIGH, Camera
from hardware.assembly.travel_ranges import PHONE_BED_BEAM_LENGTH
from hardware.parts.standard.extrusion import (
    Extrusion1020,
    end_hole_offset,
)
from hardware.parts.standard.extrusion import (
    half_x_1020 as BEAM_TOP_Y,  # section height (native Y of the top face)
)
from hardware.parts.standard.screw import Screw
from hardware.parts.standard.t_nut import HAMMER_TOTAL_HEIGHT, LENGTHS, TNut

BEAM_LENGTH = (
    PHONE_BED_BEAM_LENGTH  # mm — Extrusion1020 length — see assembly/travel_ranges.py
)
BHCS_LENGTH = 16  # mm — BHCS M5 underhead length
SCREW_EXPLODE = 55  # mm — exploded: screw drops below the beam
NUT_EXPLODE = 25  # mm — exploded: nut lifts above the beam

# Native Z of the two end holes (Extrusion1020 drills them at end_hole_offset
# from each end face).
HOLE_POSITIONS = (end_hole_offset, BEAM_LENGTH - end_hole_offset)


class PH10Extrusion(BaseAssembly):
    camera = [FRONT_LEFT_HIGH, Camera(138.99, 20.09, -98.56)]
    views = [EXPLODED_CAM1, ASSEMBLED_CAM1]

    def _build(self) -> Compound:
        beam = Extrusion1020(length=BEAM_LENGTH, hole=True).build()

        hammer_len = LENGTHS["hammer"]
        screws, nuts = [], []
        for hole_z in HOLE_POSITIONS:
            # BHCS, head down / shank up: rotation (90, 0, 0) maps the screw's
            # local +Z (head) to native -Y and its shank to +Y. Seating plane
            # at the beam underside (native Y = 0).
            screw = Screw("BHCS", "M5", BHCS_LENGTH).build()
            screw_y = -SCREW_EXPLODE if self.exploded else 0
            screw.move(Location((0, screw_y, hole_z), (90, 0, 0)))
            screws.append(screw)

            # Hammer T-nut, upside down: rotation (180, 0, 0) flips it so the
            # boss/mouth (native +Y, screw-entry side) faces down toward the
            # beam. Translate so the mouth sits on the top face and the bore
            # lands on the screw axis (native bore at z = length / 2).
            nut = TNut("hammer", "M5").build()
            nut_y = (
                BEAM_TOP_Y + HAMMER_TOTAL_HEIGHT + (NUT_EXPLODE if self.exploded else 0)
            )
            nut.move(Location((0, nut_y, hole_z + hammer_len / 2), (180, 0, 0)))
            nuts.append(nut)

        return Compound(label="phone_10_extrusion", children=[beam, *screws, *nuts])
