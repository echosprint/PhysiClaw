from build123d import *

from hardware.parts._fits import M3_NORMAL, M3_NUT_T, M3_NUT_W
from hardware.parts.base import BaseCustomPart

# ── Block dimensions ──────────────────────────────────────────────────────────
length = 20 * MM
width = 25 * MM
thickness = 16 * MM

# ── Top face: two stadium slots (round head + rect tail) ──────────────────────
# Slot 1: tail breaks the -Y edge.   Slot 2: 15 mm above slot 1, tail breaks +Y.
slot_radius = 3 * MM
slot_depth = 11 * MM
slot_overshoot = 1 * MM
slot_center_from_right = 8 * MM  # shared: circle center inset from +X edge
slot1_center_from_bottom = 7 * MM  # slot 1: circle center inset from -Y edge
slot2_above_slot1 = 15 * MM  # slot 2 center this much above slot 1 center

slot1_center_x = length / 2 - slot_center_from_right
slot1_center_y = -width / 2 + slot1_center_from_bottom
slot1_tail_y = -width / 2 - slot_overshoot
slot2_center_x = slot1_center_x  # same X as slot 1
slot2_center_y = slot1_center_y + slot2_above_slot1
slot2_tail_y = width / 2 + slot_overshoot

# ── Top face: through-holes concentric with the slot round ends ───────────────
top_hole_diameter = M3_NORMAL

# ── Top face: through-cut along the -Y edge, joins its mirror image ───────────
# Pre-mirror: left edge 9 mm in from the -X edge, right edge ON x = length/2
# (the mirror plane), so the mirror copy joins it into a single 22 × 3
# through-pocket. Bottom edge sits on the -Y face edge; cut goes full thickness.
top_center_cut_pre_left = 9 * MM  # rect left edge from -X edge of face
top_center_cut_pre_w = (
    length - top_center_cut_pre_left
)  # right edge sits on x = length/2
top_center_cut_h = 3 * MM  # along Y
top_center_cut_depth = thickness  # along Z — through cut
top_center_cut_center_x = length / 2 - top_center_cut_pre_w / 2
top_center_cut_center_y = -width / 2 + top_center_cut_h / 2

# ── Left face: two through-slots (2.2 mm wide), same X column ─────────────────
# Slot 1 cuts the top edge of the face; slot 2 (3.1 mm below slot 1) cuts the bottom.
side_slot_w = 2.2 * MM  # face-local X (= world -Y direction)
side_slot_from_slot2 = 9 * MM  # rect left edge from slot 2 projection
side_slot_break_overshoot = 1 * MM  # both slots' break-out past face edge
side_slot1_bottom_from_face = 9.3 * MM  # slot 1 rect bottom above face bottom edge
side_slot2_top_below_side_slot1 = 3.1 * MM

# ── Top + bottom faces: belt-routing polygon pocket ───────────────────────────
# Mirrored cuts on top and bottom leave a post in the middle.
belt_pocket_top_depth = 6.7 * MM
belt_pocket_bottom_depth = 6.2 * MM
belt_pocket_vertices = (  # CW from lower-left, world XY on the top face
    (-1.0 * MM, 0.4 * MM),  # 1: base of left 45° diagonal
    (3.5 * MM, 4.9 * MM),  # 2: top of left 45° diagonal
    (length / 2, 4.9 * MM),  # 3: top-right, ON +X edge (mirror plane)
    (length / 2, 0.4 * MM),  # 4: bottom-right, ON +X edge
    (7.5 * MM, 0.4 * MM),  # 5: notch right-bottom
    (7.5 * MM, 2.9 * MM),  # 6: notch right-top
    (4.3 * MM, 2.9 * MM),  # 7: notch left-top
    (1.8 * MM, 0.4 * MM),  # 8: top of inner 45° diagonal
)

# ── Top face: corner holes near the -X edge (4 mm in from edges) ──────────────
# -Y corner is a 9 mm blind hole; +Y corner is a through-hole.
corner_hole_diameter = M3_NORMAL
corner_hole_offset = 4 * MM  # from -X edge and from nearest Y edge
corner_hole_blind_depth = 9 * MM  # -Y corner only

# ── Left face: M3 square-nut sockets — horizontal pair, mirrored ──────────────
# Centers align with the corner holes' projections on the face.
left_rect1_w = M3_NUT_W
left_rect1_h = M3_NUT_T
left_rect1_top_offset = 3 * MM
left_rect1_depth = 7.5 * MM

# ── Front face: blind hole near the (-X, -Z) corner ───────────────────────────
front_hole_diameter = M3_NORMAL
front_hole_offset = 4 * MM  # from -X edge and from -Z edge
front_hole_depth = 9 * MM

# ── Left face: M3 square-nut socket — vertical, near the front edge ──────────
# Vertical center matches the front-face hole's Z; right edge 3 mm from front.
left_rect2_w = M3_NUT_T  # rotated 90° vs left_rect1: width = nut thickness
left_rect2_h = M3_NUT_W  # height = nut width-across-flats
left_rect2_right_from_front = 3 * MM
left_rect2_depth = 7.5 * MM

# ── Top face: PTFE-tube blind hole (on the mirror centerline) ─────────────────
# Receives the OD4 PTFE tube that passes through the SolenoidMount plate bolted
# on top, anchoring the solenoid wire's stiffener. Sits on the mirror centerline
# (x = length/2) in the solid band between the belt slots / belt pocket / center
# cut, so it threads through without touching any of them. Aligned with the
# mount's tube hole: mount native (0, 4) ↔ clamp native (length/2, -6). Blind.
tube_hole_diameter = 4.3 * MM  # OD4 PTFE tube, slip fit
tube_hole_y = -6 * MM  # clear of slots, belt pocket, center cut
tube_hole_depth = 14 * MM  # blind (block is 16 mm thick → 2 mm floor)

# ── Fillets (applied last, after mirror) ──────────────────────────────────────
cube_corner_fillet_radius = 1 * MM  # 4 outer vertical corners of the mirrored body
slot_cut_fillet_radius = 1.5 * MM  # 4 slot2 +Y break-out vertical edges (incl. mirror)


class BeltClamp(BaseCustomPart):
    def _build(self):
        with BuildPart() as my_part:
            Box(length, width, thickness)

            top_plane = Plane.XY.offset(thickness / 2)
            bottom_plane = Plane.XY.offset(-thickness / 2)

            # Top: both stadium slots in one sketch → one OCCT subtract
            with BuildSketch(top_plane):
                # Slot 1 (tail running -Y)
                with Locations((slot1_center_x, slot1_center_y)):
                    Circle(slot_radius)
                with Locations((slot1_center_x, (slot1_center_y + slot1_tail_y) / 2)):
                    Rectangle(2 * slot_radius, slot1_center_y - slot1_tail_y)
                # Slot 2 (tail running +Y)
                with Locations((slot2_center_x, slot2_center_y)):
                    Circle(slot_radius)
                with Locations((slot2_center_x, (slot2_center_y + slot2_tail_y) / 2)):
                    Rectangle(2 * slot_radius, slot2_tail_y - slot2_center_y)
            extrude(amount=-slot_depth, mode=Mode.SUBTRACT)

            # Top: half of the through-cut — sits flush against the mirror plane
            with BuildSketch(top_plane):
                with Locations((top_center_cut_center_x, top_center_cut_center_y)):
                    Rectangle(top_center_cut_pre_w, top_center_cut_h)
            extrude(amount=-top_center_cut_depth, mode=Mode.SUBTRACT)

            # Top: through-holes at both slot round ends (single Locations)
            with Locations(
                (slot1_center_x, slot1_center_y, thickness / 2),
                (slot2_center_x, slot2_center_y, thickness / 2),
            ):
                Hole(radius=top_hole_diameter / 2)

            # Left face plane: face-local X = -Y world, face-local Y = +Z world
            left_plane = Plane(
                origin=(-length / 2, width / 2, -thickness / 2),
                x_dir=(0, -1, 0),
                z_dir=(-1, 0, 0),
            )
            # Slot 2 projects onto the left face at face_X = width/2 - slot2_center_y
            side_slot_left_x = width / 2 - slot2_center_y + side_slot_from_slot2
            side_slot1_top_y = thickness + side_slot_break_overshoot
            side_slot2_top_y = (
                side_slot1_bottom_from_face - side_slot2_top_below_side_slot1
            )
            side_slot2_bottom_y = -side_slot_break_overshoot

            # Left: both through-slots in one sketch (cuts top edge and bottom edge)
            with BuildSketch(left_plane):
                # Side-slot 1 (cuts top edge)
                with Locations(
                    (
                        side_slot_left_x + side_slot_w / 2,
                        (side_slot1_bottom_from_face + side_slot1_top_y) / 2,
                    )
                ):
                    Rectangle(
                        side_slot_w, side_slot1_top_y - side_slot1_bottom_from_face
                    )
                # Side-slot 2 (cuts bottom edge)
                with Locations(
                    (
                        side_slot_left_x + side_slot_w / 2,
                        (side_slot2_bottom_y + side_slot2_top_y) / 2,
                    )
                ):
                    Rectangle(side_slot_w, side_slot2_top_y - side_slot2_bottom_y)
            extrude(amount=-length, mode=Mode.SUBTRACT)

            # Top: belt-routing polygon
            with BuildSketch(top_plane):
                with BuildLine():
                    Polyline(*belt_pocket_vertices, close=True)
                make_face()
            extrude(amount=-belt_pocket_top_depth, mode=Mode.SUBTRACT)

            # Bottom: same polygon, cut upward — leaves a post in the middle
            with BuildSketch(bottom_plane):
                with BuildLine():
                    Polyline(*belt_pocket_vertices, close=True)
                make_face()
            extrude(amount=belt_pocket_bottom_depth, mode=Mode.SUBTRACT)

            # Top: -Y corner blind hole
            with Locations(
                (
                    -length / 2 + corner_hole_offset,
                    -width / 2 + corner_hole_offset,
                    thickness / 2,
                )
            ):
                Hole(radius=corner_hole_diameter / 2, depth=corner_hole_blind_depth)

            # Top: +Y corner through-hole
            with Locations(
                (
                    -length / 2 + corner_hole_offset,
                    width / 2 - corner_hole_offset,
                    thickness / 2,
                )
            ):
                Hole(radius=corner_hole_diameter / 2)

            # Left: horizontal M3 nut-socket pair, mirrored about face's vertical centerline
            left_rect1_face_x = width - corner_hole_offset
            left_rect1_face_y = thickness - left_rect1_top_offset - left_rect1_h / 2
            with BuildSketch(left_plane):
                with Locations(
                    (left_rect1_face_x, left_rect1_face_y),  # above -Y corner hole
                    (
                        width - left_rect1_face_x,
                        left_rect1_face_y,
                    ),  # above +Y corner hole
                ):
                    Rectangle(left_rect1_w, left_rect1_h)
            extrude(amount=-left_rect1_depth, mode=Mode.SUBTRACT)

            # Front face plane: face-local (x, y) = (from -X edge, from -Z edge)
            front_plane = Plane(
                origin=(-length / 2, -width / 2, -thickness / 2),
                x_dir=(1, 0, 0),
                z_dir=(0, -1, 0),
            )

            # Front: blind hole near (-X, -Z) corner
            with BuildSketch(front_plane):
                with Locations((front_hole_offset, front_hole_offset)):
                    Circle(radius=front_hole_diameter / 2)
            extrude(amount=-front_hole_depth, mode=Mode.SUBTRACT)

            # Left: vertical M3 nut socket near the front edge
            left_rect2_face_x = width - left_rect2_right_from_front - left_rect2_w / 2
            left_rect2_face_y = front_hole_offset
            with BuildSketch(left_plane):
                with Locations((left_rect2_face_x, left_rect2_face_y)):
                    Rectangle(left_rect2_w, left_rect2_h)
            extrude(amount=-left_rect2_depth, mode=Mode.SUBTRACT)

            # Mirror the whole body across the right-face plane (x = +length/2)
            mirror(about=Plane.YZ.offset(length / 2))

            # Top: PTFE-tube blind hole on the mirror centerline (single feature,
            # so added after the mirror). Cut before the fillets so the fillet
            # edge-selection runs on the final cut geometry.
            with Locations((length / 2, tube_hole_y, thickness / 2)):
                Hole(radius=tube_hole_diameter / 2, depth=tube_hole_depth)

            # Fillet: 4 outer vertical corners (longest Z-parallel edges = thickness)
            cube_corners = my_part.edges().filter_by(Axis.Z).group_by(Edge.length)[-1]
            fillet(cube_corners, radius=cube_corner_fillet_radius)

            # Fillet: slot-tail break-out vertical edges (length = slot_depth).
            # Slot1's -Y-face break-outs are now consumed/extended by the center
            # through-cut, so they no longer match the length filter — only
            # slot2's +Y-face break-outs (and mirror copies) remain.
            slot_cut_edges = [
                e
                for e in my_part.edges().filter_by(Axis.Z)
                if abs(e.length - slot_depth) < 0.1 * MM
            ]
            fillet(slot_cut_edges, radius=slot_cut_fillet_radius)

        return my_part.part


if __name__ == "__main__":
    BeltClamp().export()
