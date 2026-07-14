"""Linear Y rail sub-assembly — one MGN9H 240 mm guideway with six
M3 × 10 FHCS in the rail's mounting holes and six M3 hammer T-nuts
ready to engage them.

Hole layout (12 holes at 20 mm pitch on a 240 mm rail):
  Screws at 1-indexed hole positions 1, 3, 5, 8, 10, 12 — both ends
  included, symmetric about the centre (40 mm pitch with a 60 mm
  central span). 6 screws total.

Geometry in the rail's native frame (matches MGN9H — rail bottom
face at native Z = 0, length axis along native X):
  * Each FHCS sits with its head top flush with the rail top face
    (native Z = rail_height); its shank passes through the rail body
    and protrudes into the slot mouth where the hammer T-nut catches
    it.
  * Each hammer T-nut HANGS LOOSELY from its screw's shank tip —
    only TNUT_LOOSE_ENGAGEMENT mm of the shank is inside the bore.
    This is the pre-install state: the bundle (rail + screws +
    t-nuts) is ready to be pressed onto an extrusion slot. Final
    tightening on the slot pulls each t-nut up the shank until the
    plate top catches the slot lip (1.8 mm inside the slot face,
    derived from leg - slot_lip_under_y) — not modeled here.
  * The t-nut's placement plane maps its native +Y (plate depth /
    bore axis) → rail +Z so the bore aligns with the screw shank,
    and native +Z (slot length) → rail +X.

Two variants:
  * exploded — screws lifted SCREW_EXPLODE above the rail top face;
               t-nuts dropped TNUT_EXPLODE below their loose-hang
               position. Reads as "fasteners ready to assemble onto
               the rail."
  * assembled — screws seated in the rail with t-nuts hanging
                loosely from each shank tip.

Run from the repo root:

    uv run --group cad python -m hardware step linear_10_y
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import ASSEMBLED_CAM0, BaseAssembly, EXPLODED_CAM0
from hardware.assembly.projection import FRONT_LEFT_HIGH
from hardware.assembly.travel_ranges import Y_RAIL_LENGTH
from hardware.parts.standard.mgn9h import (
    MGN9H,
    rail_height,
    rail_hole_pitch,
    slider_position as default_slider_position,
)
from hardware.parts.standard.screw import FHCS_DIMS, Screw, head_skirt
from hardware.parts.standard.t_nut import (
    HAMMER_TOTAL_HEIGHT,
    LENGTHS as TNUT_LENGTHS,
    TNut,
)

RAIL_LENGTH = Y_RAIL_LENGTH  # mm — MGN9H rail length — see assembly/travel_ranges.py
FHCS_LENGTH = 10  # mm — M3 FHCS overall length
TNUT_LOOSE_ENGAGEMENT = 1  # mm — assembled: shank depth inside the bore at
#      loose hang (a few threads — the bundle is
#      ready to drop onto an extrusion slot)
SCREW_EXPLODE = 20  # mm — exploded: screws lifted above rail top
TNUT_EXPLODE = 15  # mm — exploded: t-nuts dropped below loose hang

# 1-indexed rail-hole positions that receive a screw — both ends plus
# interior, symmetric about the centre (12-hole / 240 mm rail).
SCREW_HOLE_INDICES = (1, 3, 5, 8, 10, 12)


class LI10Y(BaseAssembly):
    # Subclasses share this build logic and only override the four
    # class attributes below — ``compound_label`` retargets the
    # STEP / SVG filename, ``rail_length`` swaps in a different MGN9H
    # length, ``screw_hole_indices`` selects which of the rail's
    # mounting holes get fastened, and ``slider_position`` (0.0 = -X
    # end, 1.0 = +X end) moves the slider along the rail.
    # ``_module_stem()`` already derives the output filename from the
    # subclass's own module, so no other override is needed.
    compound_label: str = "linear_10_y"
    rail_length: float = RAIL_LENGTH
    screw_hole_indices: tuple = SCREW_HOLE_INDICES
    slider_position: float = default_slider_position
    camera = FRONT_LEFT_HIGH
    views = [EXPLODED_CAM0, ASSEMBLED_CAM0]

    def _build(self) -> Compound:
        mgn = MGN9H(
            rail_length=self.rail_length,
            slider_position=self.slider_position,
        ).build()

        # Reproduce MGN9H's hole grid: GridLocations(rail_hole_pitch,
        # 0, n_holes, 1) centered on rail native X = 0.
        n_holes = max(1, int(self.rail_length // rail_hole_pitch))
        first_hole_x = -((n_holes - 1) * rail_hole_pitch) / 2
        hole_xs = [first_hole_x + i * rail_hole_pitch for i in range(n_holes)]
        screw_xs = [hole_xs[i - 1] for i in self.screw_hole_indices]

        # FHCS head total height (cone + skirt rim) — used to seat the
        # head top flush with the rail top face.
        fhcs_head_height = FHCS_DIMS["M3"]["k"] + head_skirt
        tnut_length = TNUT_LENGTHS["hammer"]

        # T-nut hangs loosely from the shank tip with only
        # TNUT_LOOSE_ENGAGEMENT mm of shank inside the bore. The
        # bore's far end (boss top, at t-nut native Y =
        # HAMMER_TOTAL_HEIGHT) maps to rail Z = origin.z +
        # HAMMER_TOTAL_HEIGHT via the placement plane (native +Y →
        # rail +Z); solving for origin.z gives the loose-hang z.
        screw_z_seated = rail_height - fhcs_head_height
        shank_tip_z = screw_z_seated - (FHCS_LENGTH - fhcs_head_height)
        tnut_z_loose = shank_tip_z + TNUT_LOOSE_ENGAGEMENT - HAMMER_TOTAL_HEIGHT
        if self.exploded:
            screw_z = screw_z_seated + SCREW_EXPLODE
            tnut_z = tnut_z_loose - TNUT_EXPLODE
        else:
            screw_z = screw_z_seated
            tnut_z = tnut_z_loose

        attachments = []
        for sx in screw_xs:
            screw = Screw("FHCS", "M3", FHCS_LENGTH).build()
            screw.move(Location((sx, 0, screw_z)))
            attachments.append(screw)

            nut = TNut("hammer", "M3").build()
            nut.move(
                Location(
                    Plane(
                        origin=(sx - tnut_length / 2, 0, tnut_z),
                        x_dir=(0, 1, 0),
                        z_dir=(1, 0, 0),
                    )
                )
            )
            attachments.append(nut)

        return Compound(label=self.compound_label, children=[mgn, *attachments])
