"""LD idler-mount + frame composition — extends idler_22_ru (frame +
LU mount + RU mount + their brackets) by adding the left-down
counterpart: idler_30_ld (single-idler IdlerMountFront block populated
with idler, ring, shoulder screw, captive M4 nut, and frame-mount
M5 BHCS).

The LD block (20 × 20 × 18) mounts onto long_left's slot face at the
BOTTOM t-nut. Unlike the LU/RU IdlerMountMotor blocks, the M5 BHCS in
the IdlerMountFront block is OFF-CENTER (at native y = slot_center_y
= -6) — pinning the BHCS to the t-nut forces the block to extend
asymmetrically in world Z. The chosen orientation puts the back face
(native +Y, with the captive M4 nut pocket) toward world +Z (the frame
top), so the block extends from the t-nut upward into the frame body
rather than off the bottom edge.

Placement:

  * Base layer — ID22Ru(exploded=False) at world origin. Bundles the
    full frame plus both upper mounts and their brackets. Long
    extrusions run along world Z at x = ±half_w; long_left lives at
    x = -half_w (= -95 mm).
  * LD mount — ID30Ld(exploded=False) oriented so the block's 18 mm
    thickness (native +Z, top face) points world -Y (outboard, away
    from the slot), the 20 mm native X axis runs along world X (the
    extrusion's narrow direction), and the 20 mm native Y axis runs
    along world Z. With the BHCS at native y = slot_center_y = -6 the
    block centerline lands LONG_BOT_GAP - slot_center_y above the
    frame's bottom edge, so the block extends upward from the t-nut
    in world Z. The block is translated so its bottom face sits flush
    on long_left's slot_right face (world y = -EXT_THICKNESS) and its
    M5 BHCS shank exits at world (x = -half_w, y = -EXT_THICKNESS,
    z = LONG_BOT_GAP) — threading into long_left's bottom standard
    M5 T-nut.

Two variants:
  * exploded — ld mount pulled outboard along world -Y by LD_EXPLODE
               so the BHCS install path reads clearly.
  * assembled — ld mount flush on the long's slot face.

The base ID22Ru layer is always shown assembled — it's the prior
step's finished state that this composition builds onto.

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.idler_31_ld
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import BaseAssembly
from hardware.assembly.procedures.frame_10_extrusion_tnut import (
    EXT_THICKNESS,
    LONG_BOT_GAP,
    SHORT_LENGTH,
)
from hardware.assembly.procedures.idler_22_ru import ID22Ru
from hardware.assembly.procedures.idler_30_ld import ID30Ld
from hardware.assembly.projection import MAIN_FRAME_VIEW, Camera
from hardware.parts.custom.idler_mount_front import (
    slot_center_y,
    thickness as block_thickness,
)

LD_EXPLODE = 40  # mm — exploded: outboard air gap, slot face → ld block bottom


class ID31Ld(BaseAssembly):
    camera = [MAIN_FRAME_VIEW, Camera(50.88, -36.72, 60.03)]

    def _build(self) -> Compound:
        # Base layer — always assembled. Bundles frame + LU/RU mounts
        # + their brackets; this step adds the LD mount on top.
        base_compound = ID22Ru(exploded=False).build()

        half_w = SHORT_LENGTH / 2 + EXT_THICKNESS / 2
        slot_face_y = -EXT_THICKNESS
        # Bottom standard t-nut on long_left — the LD block engages
        # this one. The frame layout puts the bottom tnut LONG_BOT_GAP
        # above the frame's bottom edge (world z = 0), so its center
        # is at world z = LONG_BOT_GAP.
        ld_tnut_world_z = LONG_BOT_GAP

        # LD pulley mount — orient native +X → world +X, native +Z
        # (top face) → world -Y (outboard, same as LU/RU), and derived
        # native +Y → world +Z (back face / captive-nut side toward the
        # frame top). The block's frame-mount M5 BHCS sits at native
        # (0, slot_center_y, …) and exits the bottom face along native
        # -Z → world +Y; align that exit with the bottom t-nut and the
        # bottom face with the slot.
        if self.exploded:
            block_center_y = slot_face_y - block_thickness / 2 - LD_EXPLODE
        else:
            block_center_y = slot_face_y - block_thickness / 2
        # BHCS is off-centre in the block (native y = slot_center_y, < 0),
        # so the block origin sits above the t-nut by -slot_center_y in
        # world Z to land the BHCS on the t-nut.
        block_center_z = ld_tnut_world_z - slot_center_y

        ld_compound = ID30Ld(exploded=False).build()
        ld_compound.move(
            Location(
                Plane(
                    origin=(-half_w, block_center_y, block_center_z),
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
            label="idler_31_ld",
            children=[
                base_compound,
                ld_compound,
            ],
        )


if __name__ == "__main__":
    for exploded in (True, False):
        asm = ID31Ld(exploded=exploded)
        asm.export()
        asm.render()
