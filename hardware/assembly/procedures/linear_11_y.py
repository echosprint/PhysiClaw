"""Linear Y rails on the frame — extends motor_30_pulley (frame +
4 corner idlers + both motors + pulleys) by mounting two LI10Y rail
sub-assemblies on the long extrusions' slot faces, one on long_left
and one on long_right.

Each LI10Y already bundles its own MGN9H + 6 M3 × 10 FHCS + 6 hammer
M3 T-nuts (built in the rail's native frame). This step just places
each instance onto the corresponding extrusion:

  * The rail is centered in the free world-Z range between the two
    corner-mount blocks on each long extrusion — LU/RU
    (IdlerMountMotor, 42 mm length) at the top and LD/RD
    (IdlerMountFront, 20 mm width) at the bottom. Centering on
    LONG_LENGTH/2 would put the rail top past the LU/RU block bottom
    and clash with it; the midpoint of [LD top, LU bottom] sits the
    rail cleanly between them.
  * Origin at (±half_w, slot_face_y, rail_center_z), rail bottom
    face flush on the slot_right face.
  * Rail native +X (length) → world +Z (along the long).
  * Rail native +Z (rail top) → world -Y (outboard, away from the
    slot face).

Two variants:
  * exploded — both rails pulled outboard along world -Y by
               RAIL_EXPLODE so the install path reads clearly.
  * assembled — rail bottom flush on the slot face.

The base (motor_30_pulley) is always shown assembled.

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.linear_11_y
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import BaseAssembly
from hardware.assembly.procedures.frame_10_extrusion_tnut import (
    EXT_THICKNESS,
    LONG_BOT_GAP,
    LONG_LENGTH,
    LONG_TOP_GAP,
    SHORT_LENGTH,
)
from hardware.assembly.procedures.linear_10_y import LI10Y, RAIL_LENGTH
from hardware.assembly.procedures.motor_30_pulley import MO30Pulley
from hardware.assembly.projection import MAIN_FRAME_VIEW, Camera
from hardware.parts.custom.idler_mount_front import (
    slot_center_y as ld_slot_center_y,
    width as ld_block_width,
)
from hardware.parts.custom.idler_mount_motor import length as lu_block_length
from hardware.parts.standard.mgn9h import (
    block_top_z as slider_top_z,
    slider_position,
)

RAIL_EXPLODE = 40  # mm — exploded: outboard air gap, slot face → rail bottom


class LI11Y(BaseAssembly):
    camera = [MAIN_FRAME_VIEW, Camera(-57.02, 34.80, -114.34)]

    def _build(self) -> Compound:
        # Base layer — always assembled.
        base_compound = MO30Pulley(exploded=False).build()

        half_w = SHORT_LENGTH / 2 + EXT_THICKNESS / 2
        slot_face_y = -EXT_THICKNESS

        # World-Z extents of the corner-mount blocks on each long.
        # LU/RU block (IdlerMountMotor) is centered on the top t-nut
        # at world z = LONG_LENGTH - LONG_TOP_GAP with its 42 mm
        # length axis along world Z, so its bottom edge sits length/2
        # below that. LD/RD block (IdlerMountFront) is centered at
        # world z = LONG_BOT_GAP - slot_center_y (off-centre BHCS)
        # with its 20 mm width axis along world Z, so its top edge
        # sits width/2 above that. Center the rail in the gap between.
        lu_block_bottom_z = (LONG_LENGTH - LONG_TOP_GAP) - lu_block_length / 2
        ld_block_top_z = (LONG_BOT_GAP - ld_slot_center_y) + ld_block_width / 2
        rail_center_z = (ld_block_top_z + lu_block_bottom_z) / 2

        if self.exploded:
            rail_bottom_y = slot_face_y - RAIL_EXPLODE
        else:
            rail_bottom_y = slot_face_y

        # Slider position along the rail (MGN9H places it at
        # slider_position * RAIL_LENGTH from the -X end of the rail).
        slider_x_offset = -RAIL_LENGTH / 2 + slider_position * RAIL_LENGTH

        rails = []
        # Hook for downstream consumers (linear_20_joint): world (x, y, z)
        # of each slider's M3 mount-hole grid center (which coincides
        # with the slider top-face center).
        self.slider_mount_centers = []
        for world_x in (-half_w, +half_w):
            rail_compound = LI10Y(exploded=False).build()
            rail_compound.move(
                Location(
                    Plane(
                        origin=(world_x, rail_bottom_y, rail_center_z),
                        x_dir=(0, 0, +1),  # rail native +X (length) → world +Z
                        z_dir=(0, -1, 0),  # rail native +Z (top)    → world -Y
                    )
                )
            )
            rails.append(rail_compound)
            self.slider_mount_centers.append(
                (
                    world_x,
                    rail_bottom_y - slider_top_z,
                    rail_center_z + slider_x_offset,
                )
            )

        return Compound(label="linear_11_y", children=[base_compound, *rails])


if __name__ == "__main__":
    for exploded in (True, False):
        asm = LI11Y(exploded=exploded)
        asm.export()
        asm.render()
