"""Linear X rail on the crossbeam — extends linear_31_x by mounting
the linear_32_x sub-assembly (130 mm MGN9H rail + 4 M3 × 10 FHCS +
4 hammer M3 T-nuts, hanging loosely from the screw shanks) onto
the 1020 crossbeam's slot.

The rail runs along world X (= 1020 length axis), centered on the
1020 (world X = 0). Rail bottom flush against the 1020's slot face
at world Y = beam_slot_face (= -30 in the current frame, exposed
by linear_31_x as a hook). Cross-section centered on world Z =
beam_center (= 187.5).

Joint orientation:
  * native +X (rail length) → world +X.
  * native +Z (rail top)    → world -Y (outboard, in front of 1020).
  * native +Y (rail width)  → world +Z (derived).

Two variants:
  * exploded — the rail (with its loose-hanging fasteners) lifted
               along world -Y by RAIL_EXPLODE, reading "rail
               pressed onto the 1020 slot from this direction".
  * assembled — rail bottom flush on the 1020 slot face.

The base (linear_31_x) is always shown assembled.

Run from the repo root:

    uv run --group cad python -m hardware step linear_33_x
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import BaseAssembly, EXPLODED_CAM1
from hardware.assembly.procedures.linear_31_x import LI31X
from hardware.assembly.procedures.linear_32_x import LI32X
from hardware.assembly.projection import MAIN_FRAME_VIEW, Camera

RAIL_EXPLODE = 30  # mm — exploded: rail lifted outward along world -Y


class LI33X(BaseAssembly):
    camera = [MAIN_FRAME_VIEW, Camera(-1.95, -64.14, -1.85)]
    views = [EXPLODED_CAM1]

    def _build(self) -> Compound:
        base = LI31X(exploded=False)
        base_compound = base.build()

        # Read the 1020's slot-face Y and cross-section center Z from
        # the LI31X hooks so this composition stays free of any
        # frame / joint placement math. Forward joint_base so further
        # downstream consumers (linear_41_idler_lj1) can reach the
        # LI20Joint feature-center hooks without re-deriving placement.
        slot_face_y = base.beam_slot_face_world_y
        rail_center_z = base.beam_center_world_z
        self.joint_base = base.joint_base

        if self.exploded:
            rail_bottom_y = slot_face_y - RAIL_EXPLODE
        else:
            rail_bottom_y = slot_face_y

        rail = LI32X(exploded=False).build()
        rail.move(
            Location(
                Plane(
                    origin=(0, rail_bottom_y, rail_center_z),
                    x_dir=(1, 0, 0),  # rail native +X (length) → world +X
                    z_dir=(0, -1, 0),  # rail native +Z (top)   → world -Y (outboard)
                )
            )
        )

        return Compound(label="linear_33_x", children=[base_compound, rail])
