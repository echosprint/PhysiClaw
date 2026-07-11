"""RU idler-mount + frame composition — extends idler_12_lu (frame +
LU mount + LU bracket) by adding the right-up counterparts: idler_20_ru
(the pulley-mount block populated with idlers, shoulder screws and the
captive top-pocket M5 square nut) and idler_21_ru (the block-tying flat
bracket).

The RU mount is the mirror of LU across the world YZ plane: same
outboard-Y placement, but on the right long extrusion. The block's
42 mm length still runs along world Z (extrusion long direction); the
front face (native -Y) points world +X — i.e. outside the frame on the
right side — and the 20 mm width (native +Y) points world -X (toward
the frame interior).

Placement:

  * Base layer — ID12Lu(exploded=False) at world origin. This already
    bundles the full frame, the LU mount on long_left, and the LU
    block-tying bracket. Long extrusions run along world Z at
    x = ±half_w; long_right lives at x = +half_w (= +95 mm).
  * RU mount — ID20Ru(exploded=False) oriented so the block's 42 mm
    length axis (native +X) runs along world Z (the extrusion long
    direction), the 18 mm thickness (native +Z, top face) points
    world -Y (outboard, away from the slot), and the 20 mm width
    (native +Y) points world -X (toward the frame interior) — making
    the front face (native -Y) point world +X, i.e. outside the
    frame. The block is translated so its bottom face sits flush on
    long_right's slot_right face (world y = -EXT_THICKNESS) and its
    centerline lands on the TOP standard M5 T-nut of long_right at
    world z = LONG_LENGTH - LONG_TOP_GAP. The center M5 BHCS in the
    ru block engages that t-nut.
  * Block-tying bracket — ID21Ru(exploded=False) oriented so its BHCS
    shanks point world -X (into the ru block's front-face M5 hole at
    world x = +105). The bracket lies flat against the world +X face
    (in the Y-Z plane), with its 20 mm hole spacing along world Y:
    RIGHT hole at the ru front-face hole world y reaches the captive
    square nut in the block's top pocket via the front-face clearance
    hole; LEFT hole, 20 mm away in +Y, sits over a t-nut in the
    extrusion's +X-face slot (the slot at local +Y of the cell at
    local x = +cell_offset, mapped to world y = -10). The 20 mm
    bracket hole spacing matches the front-face-to-slot Y gap exactly
    in assembled mode (front face at world y = -30, slot at -10).

Two variants:
  * exploded — ru mount pulled outboard along world -Y; bracket
               follows the block in Y (preserving the front-face hole
               interface) and is additionally pulled outward along
               world +X by BRACKET_EXPLODE so the bracket-to-block
               interface reads clearly. The LEFT-hole-to-slot mate is
               only meaningful in assembled mode.
  * assembled — ru mount flush on the long's slot face, bracket flush
                against the block's front face / extrusion +X face.

The base ID12Lu layer is always shown assembled — it's the prior
step's finished state that this composition builds onto.

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.idler_22_ru
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import BaseAssembly
from hardware.assembly.procedures.frame_10_extrusion_tnut import (
    EXT_THICKNESS,
    LONG_LENGTH,
    LONG_TOP_GAP,
    SHORT_LENGTH,
)
from hardware.assembly.procedures.idler_12_lu import ID12Lu
from hardware.assembly.procedures.idler_20_ru import ID20Ru
from hardware.assembly.procedures.idler_21_ru import ID21Ru
from hardware.assembly.projection import MAIN_FRAME_VIEW, Camera
from hardware.parts.custom.idler_mount_motor import (
    front_hole_center_z,
    thickness as block_thickness,
    width as block_width,
)
from hardware.parts.standard.bracket import flat_hole_spacing

RU_EXPLODE = 40  # mm — exploded: outboard air gap, slot face → ru block bottom
BRACKET_EXPLODE = 30  # mm — exploded: +X air gap, ru block front face → bracket


class ID22Ru(BaseAssembly):
    camera = [MAIN_FRAME_VIEW, Camera(61.32, -54.86, 67.24)]

    def _build(self) -> Compound:
        # Base layer — always assembled. Bundles frame + LU mount + LU
        # bracket; this step adds the RU mount + RU bracket on top.
        base_compound = ID12Lu(exploded=False).build()

        half_w = SHORT_LENGTH / 2 + EXT_THICKNESS / 2
        slot_face_y = -EXT_THICKNESS
        # Top standard t-nut on long_right — the RU block engages this
        # one. The frame layout puts the top tnut LONG_TOP_GAP below
        # the frame's top edge (world z = LONG_LENGTH), so its center
        # is at world z = LONG_LENGTH - LONG_TOP_GAP.
        ru_tnut_world_z = LONG_LENGTH - LONG_TOP_GAP

        # RU pulley mount — orient with the 42 mm length axis along
        # world Z (the extrusion long direction) and the front face
        # (native -Y) pointing world +X (outside the frame on the
        # right side). The top face (native +Z) still points world -Y
        # (outboard) so the M5 BHCS shank (native -Z) drives world +Y
        # into the slot. Origin offsets place the bottom face flush on
        # the slot in assembled mode, or floating RU_EXPLODE outboard
        # in exploded.
        if self.exploded:
            block_center_y = slot_face_y - block_thickness / 2 - RU_EXPLODE
        else:
            block_center_y = slot_face_y - block_thickness / 2
        ru_compound = ID20Ru(exploded=False).build()
        ru_compound.move(
            Location(
                Plane(
                    origin=(+half_w, block_center_y, ru_tnut_world_z),
                    x_dir=(0, 0, +1),  # native +X (length=42) → world +Z (along long)
                    z_dir=(0, -1, 0),  # native +Z (top face)  → world -Y (outboard)
                )
            )
        )  # → native +Y → world -X, native -Y (front) → +X

        # Block-tying bracket — placed flat against the world +X face,
        # BHCS shanks pointing world -X. The plane axes map ID21Ru's
        # native frame as: +X (hole-spacing) → world -Y, +Y (width) →
        # world -Z, +Z (thickness / screw-head side) → world +X.
        # Resulting hole positions (with origin chosen below):
        #   RIGHT (native +half_hole) → world y = front-face hole y
        #          = block_center_y - front_hole_center_z → threads
        #          into the captive top-pocket nut.
        #   LEFT  (native -half_hole) → world y = RIGHT_y + 20 → in
        #          assembled mode that's y = -10 = extrusion +X-face
        #          slot t-nut (slot from the cell at local x =
        #          +cell_offset). The bracket tracks the block in Y
        #          when the block is exploded; the LEFT-hole-to-slot
        #          mate is only meaningful in assembled mode.
        # In X, the bracket bottom (native z = bracket_bottom_z) lands
        # on the front face at world x = front_face_x; exploded pulls
        # the bracket BRACKET_EXPLODE further in +X.
        front_face_x = +half_w + block_width / 2  # = +105
        front_face_hole_y = block_center_y - front_hole_center_z
        half_hole = flat_hole_spacing / 2

        bracket_asm = ID21Ru(exploded=False)
        bracket_compound = bracket_asm.build()
        bracket_origin_x = front_face_x - bracket_asm.bracket_bottom_z
        if self.exploded:
            bracket_origin_x += BRACKET_EXPLODE
        bracket_origin_y = front_face_hole_y + half_hole  # RIGHT lands on hole
        bracket_compound.move(
            Location(
                Plane(
                    origin=(bracket_origin_x, bracket_origin_y, ru_tnut_world_z),
                    x_dir=(0, -1, 0),
                    z_dir=(+1, 0, 0),
                )
            )
        )

        return Compound(
            label="idler_22_ru",
            children=[
                base_compound,
                ru_compound,
                bracket_compound,
            ],
        )


if __name__ == "__main__":
    for exploded in (True, False):
        asm = ID22Ru(exploded=exploded)
        asm.export()
        asm.render()
