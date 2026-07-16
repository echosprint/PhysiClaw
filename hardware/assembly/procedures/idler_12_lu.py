"""LU idler-mount + frame composition — combines frame_41_bumper (full
frame with corner brackets + bumper feet), idler_10_lu (the pulley-mount
block populated with idlers, shoulder screws and the captive top-pocket
M5 square nut), and idler_11_lu (the block-tying flat bracket).

Placement:

  * Frame layer — FR41Bumper(exploded=False) at world origin. Long
    extrusions run along world Z at x = ±half_w; long_left lives at
    x = -half_w (= -95 mm).
  * LU mount — ID10Lu(exploded=False) oriented so the block's 42 mm
    length axis (native +X) runs along world Z (the extrusion long
    direction), the 18 mm thickness (native +Z, top face) points
    world -Y (outboard, away from the slot), and the 20 mm width
    (native +Y) points world +X (toward the frame interior) — making
    the front face (native -Y) point world -X, i.e. outside the frame.
    The block is translated so its bottom face sits flush on
    long_left's slot_right face (world y = -EXT_THICKNESS) and its
    centerline lands on the TOP standard M5 T-nut of long_left at
    world z = LONG_LENGTH - LONG_TOP_GAP. The center M5 BHCS in the
    lu block engages that t-nut.
  * Block-tying bracket — ID11Lu(exploded=False) oriented so its BHCS
    shanks point world +X (into the lu block's front-face M5 hole at
    world x = -105). The bracket lies flat against the world -X face
    (in the Y-Z plane), with its 20 mm hole spacing along world Y:
    RIGHT hole at the lu front-face hole world y reaches the captive
    square nut in the block's top pocket via the front-face clearance
    hole; LEFT hole, 20 mm away in +Y, sits over a t-nut in the
    extrusion's -X-face slot (the slot at local +Y of the cell at
    local x = +cell_offset, mapped to world y = -10). The 20 mm
    bracket hole spacing matches the front-face-to-slot Y gap exactly
    in assembled mode (front face at world y = -30, slot at -10).

Two variants:
  * exploded — lu mount pulled outboard along world -Y; bracket
               follows the block in Y (preserving the front-face hole
               interface) and is additionally pulled outward along
               world -X by BRACKET_EXPLODE so the bracket-to-block
               interface reads clearly. The LEFT-hole-to-slot mate is
               only meaningful in assembled mode.
  * assembled — lu mount flush on the long's slot face, bracket flush
                against the block's front face / extrusion -X face.

The frame layer is always shown assembled — it's the prior step's
finished state that this composition builds onto.

Run from the repo root:

    uv run --group cad python -m hardware step idler_12_lu
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import ASSEMBLED_CAM0, EXPLODED_CAM1, BaseAssembly
from hardware.assembly.procedures.frame_10_extrusion_tnut import (
    EXT_THICKNESS,
    LONG_LENGTH,
    LONG_TOP_GAP,
    SHORT_LENGTH,
)
from hardware.assembly.procedures.frame_41_bumper import FR41Bumper
from hardware.assembly.procedures.idler_10_lu import ID10Lu
from hardware.assembly.procedures.idler_11_lu import ID11Lu
from hardware.assembly.projection import MAIN_FRAME_VIEW, Camera
from hardware.parts.custom.idler_mount_motor import (
    front_hole_center_z,
)
from hardware.parts.custom.idler_mount_motor import (
    thickness as block_thickness,
)
from hardware.parts.custom.idler_mount_motor import (
    width as block_width,
)
from hardware.parts.standard.bracket import flat_hole_spacing

LU_EXPLODE = 40  # mm — exploded: outboard air gap, slot face → lu block bottom
BRACKET_EXPLODE = 30  # mm — exploded: -X air gap, lu block front face → bracket


class ID12Lu(BaseAssembly):
    camera = [MAIN_FRAME_VIEW, Camera(-54.17, -69.62, -57.05)]
    views = [EXPLODED_CAM1, ASSEMBLED_CAM0]

    def _build(self) -> Compound:
        # Frame layer — always assembled.
        frame_compound = FR41Bumper(exploded=False).build()

        half_w = SHORT_LENGTH / 2 + EXT_THICKNESS / 2
        slot_face_y = -EXT_THICKNESS
        # Top standard t-nut on long_left — the LU block engages this
        # one. The frame layout puts the top tnut LONG_TOP_GAP below
        # the frame's top edge (world z = LONG_LENGTH), so its center
        # is at world z = LONG_LENGTH - LONG_TOP_GAP.
        lu_tnut_world_z = LONG_LENGTH - LONG_TOP_GAP

        # LU pulley mount — orient with the 42 mm length axis along
        # world Z (the extrusion long direction) and the front face
        # (native -Y) pointing world -X (outside the frame). The top
        # face (native +Z) still points world -Y (outboard) so the
        # M5 BHCS shank (native -Z) drives world +Y into the slot.
        # Origin offsets place the bottom face flush on the slot in
        # assembled mode, or floating LU_EXPLODE outboard in exploded.
        if self.exploded:
            block_center_y = slot_face_y - block_thickness / 2 - LU_EXPLODE
        else:
            block_center_y = slot_face_y - block_thickness / 2
        lu_compound = ID10Lu(exploded=False).build()
        lu_compound.move(
            Location(
                Plane(
                    origin=(-half_w, block_center_y, lu_tnut_world_z),
                    x_dir=(0, 0, -1),  # native +X (length=42) → world -Z (along long)
                    z_dir=(0, -1, 0),  # native +Z (top face)  → world -Y (outboard)
                )
            )
        )  # → native +Y → world +X, native -Y (front) → -X

        # Block-tying bracket — placed flat against the world -X face,
        # BHCS shanks pointing world +X. The plane axes map ID11Lu's
        # native frame as: +X (hole-spacing) → world -Y, +Y (width) →
        # world +Z, +Z (thickness / screw-head side) → world -X.
        # Resulting hole positions (with origin chosen below):
        #   RIGHT (native +half_hole) → world y = front-face hole y
        #          = block_center_y - front_hole_center_z → threads
        #          into the captive top-pocket nut.
        #   LEFT  (native -half_hole) → world y = RIGHT_y + 20 → in
        #          assembled mode that's y = -10 = extrusion -X-face
        #          slot t-nut (slot from the cell at local x =
        #          +cell_offset). The bracket tracks the block in Y
        #          when the block is exploded; the LEFT-hole-to-slot
        #          mate is only meaningful in assembled mode.
        # In X, the bracket bottom (native z = bracket_bottom_z) lands
        # on the front face at world x = front_face_x; exploded pulls
        # the bracket BRACKET_EXPLODE further in -X.
        front_face_x = -half_w - block_width / 2  # = -105
        front_face_hole_y = block_center_y - front_hole_center_z
        half_hole = flat_hole_spacing / 2

        bracket_asm = ID11Lu(exploded=False)
        bracket_compound = bracket_asm.build()
        bracket_origin_x = front_face_x + bracket_asm.bracket_bottom_z
        if self.exploded:
            bracket_origin_x -= BRACKET_EXPLODE
        bracket_origin_y = front_face_hole_y + half_hole  # RIGHT lands on hole
        bracket_compound.move(
            Location(
                Plane(
                    origin=(bracket_origin_x, bracket_origin_y, lu_tnut_world_z),
                    x_dir=(0, -1, 0),
                    z_dir=(-1, 0, 0),
                )
            )
        )

        return Compound(
            label="idler_12_lu",
            children=[
                frame_compound,
                lu_compound,
                bracket_compound,
            ],
        )
