"""RD idler-mount + frame composition — extends idler_31_ld (frame +
LU/RU mounts + their brackets + LD mount) by adding the right-down
counterpart: idler_40_rd (single-idler IdlerMountFront block with the
spacer + washer + idler stack on a 20 mm M4 shoulder screw).

The RD block (20 × 20 × 18) mounts onto long_right's slot face at the
BOTTOM t-nut. Uses the same orientation as LD — the IdlerMountFront
block is symmetric across its own YZ plane and has no front-face
feature requiring a chiral flip, so mirroring to the right side only
changes the world X position (-half_w → +half_w).

Placement:

  * Base layer — ID31Ld(exploded=False) at world origin. Bundles the
    full frame plus both upper mounts and their brackets, and the
    left-down mount. Long extrusions run along world Z at
    x = ±half_w; long_right lives at x = +half_w (= +95 mm).
  * RD mount — ID40Rd(exploded=False) oriented the same way as LD:
    native +X → world +X, native +Z (top face) → world -Y (outboard),
    derived native +Y → world +Z (back face / captive-nut side toward
    the frame top). The block is translated so its bottom face sits
    flush on long_right's slot_right face (world y = -EXT_THICKNESS)
    and its M5 BHCS shank exits at world (x = +half_w,
    y = -EXT_THICKNESS, z = LONG_BOT_GAP) — threading into
    long_right's bottom standard M5 T-nut.

Two variants:
  * exploded — rd mount pulled outboard along world -Y by RD_EXPLODE
               so the BHCS install path reads clearly.
  * assembled — rd mount flush on the long's slot face.

The base ID31Ld layer is always shown assembled — it's the prior
step's finished state that this composition builds onto.

Run from the repo root:

    uv run --group cad python -m hardware step idler_41_rd
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import ASSEMBLED_CAM0, BaseAssembly, EXPLODED_CAM1
from hardware.assembly.procedures.frame_10_extrusion_tnut import (
    EXT_THICKNESS,
    LONG_BOT_GAP,
    SHORT_LENGTH,
)
from hardware.assembly.procedures.idler_31_ld import ID31Ld
from hardware.assembly.procedures.idler_40_rd import ID40Rd
from hardware.assembly.projection import MAIN_FRAME_VIEW, Camera
from hardware.parts.custom.idler_mount_front import (
    slot_center_y,
    thickness as block_thickness,
)

RD_EXPLODE = 40  # mm — exploded: outboard air gap, slot face → rd block bottom


class ID41Rd(BaseAssembly):
    camera = [MAIN_FRAME_VIEW, Camera(40.06, -53.35, 47.58)]
    views = [EXPLODED_CAM1, ASSEMBLED_CAM0]

    def _build(self) -> Compound:
        # Base layer — always assembled. Bundles frame + LU/RU/LD
        # mounts + their brackets; this step adds the RD mount on top.
        base_compound = ID31Ld(exploded=False).build()

        half_w = SHORT_LENGTH / 2 + EXT_THICKNESS / 2
        slot_face_y = -EXT_THICKNESS
        # Bottom standard t-nut on long_right — the RD block engages
        # this one. The frame layout puts the bottom tnut LONG_BOT_GAP
        # above the frame's bottom edge (world z = 0), so its center
        # is at world z = LONG_BOT_GAP.
        rd_tnut_world_z = LONG_BOT_GAP

        # RD pulley mount — same orientation as LD: native +X → world
        # +X, native +Z (top face) → world -Y (outboard), derived
        # native +Y → world +Z (back face / captive-nut side toward
        # the frame top). The block's frame-mount M5 BHCS sits at
        # native (0, slot_center_y, …) and exits the bottom face
        # along native -Z → world +Y; align that exit with the bottom
        # t-nut and the bottom face with the slot.
        if self.exploded:
            block_center_y = slot_face_y - block_thickness / 2 - RD_EXPLODE
        else:
            block_center_y = slot_face_y - block_thickness / 2
        # BHCS is off-centre in the block (native y = slot_center_y, < 0),
        # so the block origin sits above the t-nut by -slot_center_y in
        # world Z to land the BHCS on the t-nut.
        block_center_z = rd_tnut_world_z - slot_center_y

        rd_compound = ID40Rd(exploded=False).build()
        rd_compound.move(
            Location(
                Plane(
                    origin=(+half_w, block_center_y, block_center_z),
                    x_dir=(
                        1,
                        0,
                        0,
                    ),  # native +X → world +X (block X along extrusion narrow dir)
                    z_dir=(0, -1, 0),  # native +Z (top face) → world -Y (outboard)
                )
            )
        )  # → native +Y → world +Z (back face up)

        return Compound(
            label="idler_41_rd",
            children=[
                base_compound,
                rd_compound,
            ],
        )
