"""Idler mount (left-up) — two shoulder-screw stacks of smooth idlers
and M5 thrust washers fastened into the IdlerMountMotor block's two
outer M4 through-holes (top face, at x = ±outer_hole_offset). M4
square nuts captured in the block's left/right side pockets engage
each shoulder screw's M4 thread.

Per stack: the 5 mm shoulder seats on the block's top face; the M4
thread reaches down into the 4.3 mm clearance hole and threads into
the captive square nut sitting in the side pocket (bore vertical,
aligned with the through-hole).

Stacks, in the user's top → bottom order:
  * LEFT (x = -outer_hole_offset), SHOULDER M4 × 20 mm
    idler (smooth) / ring M5×8×0.5 / idler (smooth) / ring M5×8×0.5
    Stack = 2 × 8.5 + 2 × 0.5 = 18 mm; shoulder 20 mm → 2 mm axial
    play so the idlers spin free.
  * RIGHT (x = +outer_hole_offset), SHOULDER M4 × 10 mm
    idler (smooth) / ring M5×8×0.5
    Stack = 8.5 + 0.5 = 9 mm; shoulder 10 mm → 1 mm play.

Each column also captures one M4 square nut in the block's side
pocket opening on the same side (LEFT nut in the -X-face pocket,
RIGHT nut in the +X-face pocket).

Two variants:
  * exploded — idler stacks laid out along +Z with EXPLODE_SEPARATION
               of air between adjacent parts. Both shoulder screws
               plus the center M5 BHCS share a single shank-tip line
               (SCREW_GAP above the taller stack) so all three tips
               align — heads sit at heights differing only by screw
               length. Same convention as motor_10_bracket. The two
               side-pocket M4 nuts pull outward by NUT_GAP along
               their ±X install axis; the top-pocket M5 nut lifts by
               TOP_NUT_LIFT along its +Z install axis (clear of the
               stack tops).
  * assembled — stacks tight on block top; side-pocket M4 nuts
                centered in their pockets, bores vertical and aligned
                with the shoulder-screw threads; center BHCS head
                bottomed in the cbore; top-pocket M5 nut bore aligned
                with the front-face M5 hole.

The center M5 counterbore takes a BHCS M5 × 10 that threads into a
hammer t-nut in the frame extrusion slot below (frame not modelled
here). The -Y face M5 hole receives a screw (not part of this
sub-assembly) that threads into the captive M5 square nut sitting in
the top -Y-edge pocket — the nut is added here so it ships pre-seated.

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.idler_10_lu
"""

from collections.abc import Callable

from build123d import Axis, Compound, Location

from hardware.assembly.base import BaseAssembly
from hardware.assembly.projection import FRONT_LEFT_HIGH
from hardware.parts.custom.idler_mount_motor import (
    IdlerMountMotor,
    front_hole_center_z,
    length as block_length,
    outer_hole_offset,
    side_pocket_center_z,
    thickness as block_thickness,
    top_counterbore_depth,
    top_pocket_center_y,
)
from hardware.parts.standard.nut import SPECS as NUT_SPECS, Nut
from hardware.parts.standard.pulley import Pulley2GT20T, flange_belt_h
from hardware.parts.standard.ring import SPECS as RING_SPECS, Ring
from hardware.parts.standard.screw import SHOULDER_DIMS, Screw

RING_SPEC = "M5x8x0.5"
LEFT_SHOULDER_LEN = 20  # mm — covers ring + idler + ring + idler (18 mm)
RIGHT_SHOULDER_LEN = 10  # mm — covers ring + idler (9 mm)
FRAME_BHCS_LEN = 10  # mm — BHCS M5 underhead length (center cbore, frame mount)
EXPLODE_SEPARATION = 5  # mm — exploded: air between adjacent parts in a stack
SCREW_GAP = 12  # mm — exploded: taller-stack top → shared shank-tip line
NUT_GAP = 15  # mm — exploded: block side face → side-pocket nut center, along ±X
TOP_NUT_LIFT = 44  # mm — exploded: block top face → top-pocket nut center, along +Z
#      (clears the idler stacks so the nut reads as a separate step)

# One shoulder-screw column: ``(x, shoulder_len, stack)``, with ``stack``
# the bottom→top build recipe as ``(part_factory, axial_height)`` pairs.
Column = tuple[float, float, list[tuple[Callable, float]]]


class ID10Lu(BaseAssembly):
    # Subclasses reuse this build by overriding ``compound_label`` and the
    # ``_columns()`` hook (e.g. ID20Ru); ``_module_stem()`` already derives
    # the STEP/SVG filename from the subclass's own module, so no other
    # override is needed.
    compound_label: str = "idler_10_lu"
    camera = FRONT_LEFT_HIGH

    @staticmethod
    def _idler():
        """Smooth (untoothed) idler pulley — stacked by every column recipe."""
        return Pulley2GT20T(kind="idler", toothed=False).build()

    def _columns(self) -> list[Column]:
        """The shoulder-screw columns to stack; everything else in
        ``_build`` (screw placement, captive nuts, explode conventions)
        is shared."""

        def ring():
            return Ring(RING_SPEC).build()

        ring_h = RING_SPECS[RING_SPEC]["height"]
        return [
            (
                -outer_hole_offset,
                LEFT_SHOULDER_LEN,
                [
                    (ring, ring_h),
                    (self._idler, flange_belt_h),
                    (ring, ring_h),
                    (self._idler, flange_belt_h),
                ],
            ),
            (
                +outer_hole_offset,
                RIGHT_SHOULDER_LEN,
                [
                    (ring, ring_h),
                    (self._idler, flange_belt_h),
                ],
            ),
        ]

    def _build(self) -> Compound:
        block = IdlerMountMotor().build()
        nut_thickness = NUT_SPECS["square"]["M4"]["thickness"]
        block_top_z = block_thickness / 2
        block_side_x = block_length / 2
        thread_len = SHOULDER_DIMS["M4"]["thread_len"]
        columns = self._columns()

        sep = EXPLODE_SEPARATION if self.exploded else 0

        placed = []
        column_tops = []
        for x, _, stack in columns:
            cursor_z = block_top_z + sep
            for factory, h in stack:
                p = factory()
                p.move(Location((x, 0, cursor_z)))
                placed.append(p)
                cursor_z += h + sep
            column_tops.append(cursor_z - sep)

        # Shared shank-tip line in exploded mode (motor_10_bracket convention):
        # both screws' thread tips align at thread_tip_z, so heads end up at
        # heights differing only by (LEFT - RIGHT) shoulder length.
        if self.exploded:
            thread_tip_z = max(column_tops) + SCREW_GAP

        for x, shoulder_len, _ in columns:
            screw = Screw("SHOULDER", "M4", shoulder_len).build()
            if self.exploded:
                underhead_z = thread_tip_z + thread_len + shoulder_len
            else:
                underhead_z = block_top_z + shoulder_len
            screw.move(Location((x, 0, underhead_z)))
            placed.append(screw)

        # Center M5 BHCS — frame-mount interface. Head bottoms on the
        # cbore floor (drilled top_counterbore_depth into the top face);
        # shank exits the block bottom into a frame extrusion slot t-nut
        # (frame not modelled here). In exploded view the shank tip
        # joins the shared shoulder-screw line.
        frame_screw = Screw("BHCS", "M5", FRAME_BHCS_LEN).build()
        if self.exploded:
            frame_underhead_z = thread_tip_z + FRAME_BHCS_LEN
        else:
            frame_underhead_z = block_top_z - top_counterbore_depth
        frame_screw.move(Location((0, 0, frame_underhead_z)))
        placed.append(frame_screw)

        # Captive square nut per side pocket, flipped chamfer-down so the
        # chamfer eases the lead-in for the screw thread approaching from
        # above. Exploded: nut pulled outward NUT_GAP along its ±X install
        # axis (the side face it slides through).
        nut_bottom_z = side_pocket_center_z - nut_thickness / 2
        for x, _, _ in columns:
            nut = Nut("square", "M4").build().rotate(Axis.X, 180)
            side_sign = 1 if x > 0 else -1
            nut_x = side_sign * (block_side_x + NUT_GAP) if self.exploded else x
            # After the flip the part spans local z ∈ [-thickness, 0]; offset
            # by +thickness so the chamfered face lands at the pocket floor.
            nut.move(Location((nut_x, 0, nut_bottom_z + nut_thickness)))
            placed.append(nut)

        # Captive M5 square nut in the top -Y-edge pocket — receives a
        # screw threaded through the front-face M5 hole (screw not part
        # of this sub-assembly). -90° X rotation maps local +Z (bore
        # axis) → world +Y, aligning the bore with the front hole; the
        # FLAT (un-chamfered) bottom face ends up on the screw-entry
        # side (world -Y), with the chamfered face at the back of the
        # pocket. Install axis is +Z (drops in from the top), so
        # exploded pulls the nut along +Z by TOP_NUT_LIFT above the
        # block top face — clear of the idler-stack tops.
        top_nut = Nut("square", "M5").build().rotate(Axis.X, -90)
        m5_nut_thickness = NUT_SPECS["square"]["M5"]["thickness"]
        # Rotation maps local z ∈ [0, thickness] → world y ∈ [0, thickness],
        # so the bore axis sits at world y = ty + thickness/2. Shift so
        # the bore lands on top_pocket_center_y.
        top_nut_y = top_pocket_center_y - m5_nut_thickness / 2
        if self.exploded:
            top_nut_z = block_top_z + TOP_NUT_LIFT
        else:
            top_nut_z = front_hole_center_z
        top_nut.move(Location((0, top_nut_y, top_nut_z)))
        placed.append(top_nut)

        return Compound(label=self.compound_label, children=[block, *placed])


if __name__ == "__main__":
    for exploded in (True, False):
        asm = ID10Lu(exploded=exploded)
        asm.export()
        asm.render()
