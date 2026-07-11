"""Idler mount (right-down) — single smooth idler raised on an M5×10×9
spacer, held to a IdlerMountFront block by a 20 mm M4 shoulder screw
through the block's top-face M4 hole (at world x = 0, y = top_hole_y,
on the +Y-half of the top face). The 5 mm shoulder seats on the block
top; the M4 thread reaches down into the 4.3 mm clearance hole and
threads into the captive M4 square nut sitting in the +Y-face back
pocket (bore vertical, aligned with the through-hole).

Stack, in the user's top → bottom order:
  idler (smooth) / ring M5×8×0.5 / ring M5×10×9
  Stack = 8.5 + 0.5 + 9 = 18 mm; shoulder 20 mm → 2 mm axial play.

Two variants:
  * exploded — stack laid out along +Z with EXPLODE_SEPARATION of air
               between adjacent parts. The M4 shoulder screw's thread
               tip floats SCREW_GAP above the stack top (same shank-tip
               convention as idler_10_lu / motor_10_bracket). The
               frame-mount M5 BHCS, sitting on the opposite -Y half of
               the top face at world y = slot_center_y, floats BHCS_GAP
               above the block top face — it has no stack to clear, so
               its shank-tip line sits lower than the shoulder's. The
               M4 square nut pulls outward NUT_GAP along +Y, the
               back-face install axis.
  * assembled — stack tight on block top; M4 nut bore vertical,
                centered in the back pocket and aligned with the
                shoulder thread; M5 BHCS head bottomed on the
                stadium-slot floor.

The block's stadium slot (with its M5 through-hole) is the
frame-mount interface — its BHCS M5 × 10 threads into a hammer t-nut
in the frame extrusion slot below (frame not modelled here).

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.idler_40_rd
"""

from build123d import Axis, Compound, Location

from hardware.assembly.base import BaseAssembly
from hardware.assembly.projection import FRONT_LEFT_HIGH
from hardware.parts.custom.idler_mount_front import (
    IdlerMountFront,
    back_pocket_center_z,
    slot_center_y,
    slot_depth,
    thickness as block_thickness,
    top_hole_y,
    width as block_width,
)
from hardware.parts.standard.nut import SPECS as NUT_SPECS, Nut
from hardware.parts.standard.pulley import Pulley2GT20T, flange_belt_h
from hardware.parts.standard.ring import SPECS as RING_SPECS, Ring
from hardware.parts.standard.screw import SHOULDER_DIMS, Screw

WASHER_SPEC = "M5x8x0.5"
SPACER_SPEC = "M5x10x9"
SHOULDER_LEN = 20  # mm — covers spacer + washer + idler (18 mm), 2 mm play
FRAME_BHCS_LEN = 10  # mm — BHCS M5 underhead length (stadium-slot frame mount)
EXPLODE_SEPARATION = 5  # mm — exploded: air between adjacent parts in the stack
SCREW_GAP = 12  # mm — exploded: stack top → shoulder thread tip
BHCS_GAP = 10  # mm — exploded: block top → BHCS shank tip (sits 2 mm
#      lower than SCREW_GAP since the BHCS has no stack to clear)
NUT_GAP = 15  # mm — exploded: back face → nut center along +Y install axis


class ID40Rd(BaseAssembly):
    camera = FRONT_LEFT_HIGH

    def _build(self) -> Compound:
        block = IdlerMountFront().build()
        washer_h = RING_SPECS[WASHER_SPEC]["height"]
        spacer_h = RING_SPECS[SPACER_SPEC]["height"]
        nut_thickness = NUT_SPECS["square"]["M4"]["thickness"]
        block_top_z = block_thickness / 2
        back_face_y = block_width / 2
        thread_len = SHOULDER_DIMS["M4"]["thread_len"]

        # Stack on the screw axis (x=0, y=top_hole_y), bottom → top.
        stack = [
            (Ring(SPACER_SPEC).build(), spacer_h),
            (Ring(WASHER_SPEC).build(), washer_h),
            (Pulley2GT20T(kind="idler", toothed=False).build(), flange_belt_h),
        ]

        sep = EXPLODE_SEPARATION if self.exploded else 0

        placed = []
        cursor_z = block_top_z + sep
        for part, h in stack:
            part.move(Location((0, top_hole_y, cursor_z)))
            placed.append(part)
            cursor_z += h + sep
        stack_top_z = cursor_z - sep

        # Shoulder screw: shoulder bottom seats on the block top face in
        # assembled mode. Exploded: thread tip floats SCREW_GAP above the
        # stack top (same convention as motor_10_bracket / idler_10_lu).
        if self.exploded:
            underhead_z = stack_top_z + SCREW_GAP + thread_len + SHOULDER_LEN
        else:
            underhead_z = block_top_z + SHOULDER_LEN
        screw = Screw("SHOULDER", "M4", SHOULDER_LEN).build()
        screw.move(Location((0, top_hole_y, underhead_z)))
        placed.append(screw)

        # Frame-mount M5 BHCS in the stadium slot's round end at world
        # (0, slot_center_y). Head bottoms on the slot floor (drilled
        # slot_depth into the top face); shank exits the block bottom
        # into a frame extrusion slot t-nut (frame not modelled here).
        # In exploded view the shank tip floats BHCS_GAP above the
        # block top face — the BHCS has no stack to clear (its install
        # axis sits on the -Y half of the top, opposite the shoulder).
        frame_screw = Screw("BHCS", "M5", FRAME_BHCS_LEN).build()
        if self.exploded:
            frame_underhead_z = block_top_z + BHCS_GAP + FRAME_BHCS_LEN
        else:
            frame_underhead_z = block_top_z - slot_depth
        frame_screw.move(Location((0, slot_center_y, frame_underhead_z)))
        placed.append(frame_screw)

        # Captive square nut in the +Y back pocket, flipped chamfer-down so
        # the chamfer eases the lead-in for the screw thread approaching
        # from above. Exploded: pulled outward NUT_GAP along +Y (the back
        # face it slides through).
        nut = Nut("square", "M4").build().rotate(Axis.X, 180)
        nut_y = back_face_y + NUT_GAP if self.exploded else top_hole_y
        nut_bottom_z = back_pocket_center_z - nut_thickness / 2
        # After the flip the part spans local z ∈ [-thickness, 0]; offset by
        # +thickness so the chamfered face lands at the pocket floor.
        nut.move(Location((0, nut_y, nut_bottom_z + nut_thickness)))
        placed.append(nut)

        return Compound(label="idler_40_rd", children=[block, *placed])


if __name__ == "__main__":
    for exploded in (True, False):
        asm = ID40Rd(exploded=exploded)
        asm.export()
        asm.render()
