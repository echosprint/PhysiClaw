"""Linear X crossbeam — sub-assembly with one 165 mm Extrusion1020
and two standard M5 T-nuts pre-loaded in its slot at the positions
where linear_31_x's joint screws will engage them.

Geometry in the 1020's native frame (matches Extrusion1020 — slot
face at native Y = 9.9, length along native Z = 0 to length):
  * T-nut native Z (length) → 1020 native Z; bore axis = native Y.
  * Wings (t-nut native Y = engagement_y = 3.3) catch the 1020
    cavity ceiling at native Y = 8 (the bottom of the narrow slot
    wall, where the 9 mm wings stop as they're pulled up against
    the slot).
  * Two t-nuts: centers at native Z = TNUT_POSITIONS, derived so
    each bore lands directly under the corresponding joint big-CSK
    hole when the sub-assembly is placed centered on world X = 0
    in linear_31_x. SCREW_SPACING (= 2 × big-CSK distance from
    frame centerline) is derived from frame width + XyJoint
    hole-grid offset + big-CSK offset in the joint native frame.

Two variants:
  * exploded — each t-nut slid TNUT_EXPLODE further along the slot
               toward the nearest open end of the 1020, reading as
               "inserted from this end and slid into position."
  * assembled — t-nuts at their final positions, wings against the
                cavity ceiling.

Run from the repo root:

    uv run --group cad python -m hardware step linear_30_x
"""

from build123d import Compound, Location

from hardware.assembly.base import ASSEMBLED_CAM1, BaseAssembly, EXPLODED_CAM1
from hardware.assembly.procedures.frame_10_extrusion_tnut import (
    EXT_THICKNESS,
    SHORT_LENGTH,
)
from hardware.assembly.projection import FRONT_LEFT_HIGH, Camera
from hardware.assembly.travel_ranges import X_BEAM_LENGTH
from hardware.parts.custom.xy_joint_left import (
    big_csk_dx_from_extra2,
    csk_hole_from_left,
    csk_x_spacing,
    extra_hole2_dx_from_csk,
    length as joint_length,
)
from hardware.parts.standard.extrusion import Extrusion1020
from hardware.parts.standard.t_nut import (
    ENGAGEMENT_Y as TNUT_ENGAGEMENT_Y,
    LENGTHS as TNUT_LENGTHS,
    TNut,
)

# X_BEAM_LENGTH (Extrusion1020 length) — see assembly/travel_ranges.py
TNUT_EXPLODE = 30  # mm — exploded: each t-nut slid further toward open end

# 1020 native Y of the cavity ceiling (where standard t-nut wings
# catch from below — the slot narrows above this).
BEAM_CAVITY_TOP_Y_NATIVE = 8

# SCREW_SPACING = distance between LEFT and RIGHT joint big-CSK
# world X centers in linear_31_x. Derived from frame half-width +
# XyJoint hole-grid X offset + big-CSK X offset in joint native:
#   right_csk_world_x = half_w + grid_center_x - big_csk_native_x
#   (where grid_center_x is XyJointLeft's, = -9 mm).
_half_w = SHORT_LENGTH / 2 + EXT_THICKNESS / 2
_csk_ll_x = -joint_length / 2 + csk_hole_from_left
_csk_ur_x = _csk_ll_x + csk_x_spacing
_grid_center_x = (_csk_ll_x + _csk_ur_x) / 2
_big_csk_x = _csk_ur_x + extra_hole2_dx_from_csk + big_csk_dx_from_extra2
SCREW_SPACING = 2 * (_half_w + _grid_center_x - _big_csk_x)

# T-nut centers in 1020 native Z (= world X after placement, with
# the 1020 centered on world X = 0).
_end_gap = (X_BEAM_LENGTH - SCREW_SPACING) / 2
TNUT_POSITIONS = (_end_gap, X_BEAM_LENGTH - _end_gap)


class LI30X(BaseAssembly):
    camera = [FRONT_LEFT_HIGH, Camera(138.99, 20.09, -98.56)]
    views = [EXPLODED_CAM1, ASSEMBLED_CAM1]

    def _build(self) -> Compound:
        beam = Extrusion1020(length=X_BEAM_LENGTH).build()

        # Origin Y so the t-nut wings (native Y = engagement_y) sit
        # at the 1020 cavity ceiling (where the slot narrows).
        tnut_engagement_y = TNUT_ENGAGEMENT_Y["standard"]
        tnut_origin_y = BEAM_CAVITY_TOP_Y_NATIVE - tnut_engagement_y
        tnut_length = TNUT_LENGTHS["standard"]

        tnuts = []
        for tnut_native_z in TNUT_POSITIONS:
            # Exploded: slide toward the nearest open end of the 1020.
            if self.exploded:
                if tnut_native_z < X_BEAM_LENGTH / 2:
                    z_center = tnut_native_z - TNUT_EXPLODE
                else:
                    z_center = tnut_native_z + TNUT_EXPLODE
            else:
                z_center = tnut_native_z
            # T-nut bore is at native (0, _, length/2); offset origin
            # so the bore center lands at z_center.
            nut = TNut("standard", "M5").build()
            nut.move(Location((0, tnut_origin_y, z_center - tnut_length / 2)))
            tnuts.append(nut)

        return Compound(label="linear_30_x", children=[beam, *tnuts])
