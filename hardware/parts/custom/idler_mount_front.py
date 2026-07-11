from build123d import *

from hardware.parts._fits import M4_CLOSE, M4_NUT_T, M4_NUT_W, M5_NORMAL
from hardware.parts.base import BaseCustomPart

# ── Block dimensions ──────────────────────────────────────────────────────────
length = 20 * MM
width = 20 * MM
thickness = 18 * MM

# ── Top face: stadium slot pocket (round head + rect tail running -Y) ─────────
slot_radius = 5 * MM
slot_depth = 13 * MM
slot_center_y = -6 * MM  # arc center, 4 mm from -Y edge (face -Y edge at y = -10)
slot_bottom_y = -11 * MM  # tail extends 1 mm past the -Y edge

# ── Slot floor: through-hole concentric with the slot's round end ─────────────
hole_diameter = M5_NORMAL

# ── Top face: second through-hole, centered in X ──────────────────────────────
top_hole_diameter = M4_CLOSE
top_hole_y = 4 * MM  # 6 mm from +Y edge (face +Y edge at y = +10)

# ── Back face: M4 square-nut socket ───────────────────────────────────────────
back_pocket_w = M4_NUT_W
back_pocket_h = M4_NUT_T
back_pocket_top_offset = 4 * MM  # upper edge below +Z edge of face
back_pocket_depth = 10 * MM
back_pocket_center_z = (thickness / 2) - back_pocket_top_offset - back_pocket_h / 2

# ── Fillets (applied as R3 — last) ────────────────────────────────────────────
slot_break_fillet_radius = 2 * MM  # 2 slot-tail break-out edges on the front face
front_corner_fillet_radius = 1 * MM  # 2 outer vertical corners of the front face
back_corner_fillet_radius = 2 * MM  # 2 outer vertical corners of the back face


class IdlerMountFront(BaseCustomPart):
    def _build(self):
        with BuildPart() as my_part:
            Box(length, width, thickness)

            top_plane = Plane.XY.offset(thickness / 2)

            # Top: stadium slot — full circle + rectangle, unioned in the sketch
            with BuildSketch(top_plane):
                with Locations((0, slot_center_y)):
                    Circle(slot_radius)
                # Tail: runs from circle center down past the -Y edge for a clean break-out
                tail_h = slot_center_y - slot_bottom_y
                tail_center = (slot_center_y + slot_bottom_y) / 2
                with Locations((0, tail_center)):
                    Rectangle(2 * slot_radius, tail_h)
            extrude(amount=-slot_depth, mode=Mode.SUBTRACT)

            # Top: through-hole at slot's round end
            with Locations((0, slot_center_y)):
                Hole(radius=hole_diameter / 2)

            # Top: second through-hole, centered in X
            with Locations((0, top_hole_y)):
                Hole(radius=top_hole_diameter / 2)

            # Back face plane (face-local Y = world +Z)
            back_plane = Plane(
                origin=(0, width / 2, 0), x_dir=(-1, 0, 0), z_dir=(0, 1, 0)
            )
            with BuildSketch(back_plane):
                with Locations((0, back_pocket_center_z)):
                    Rectangle(back_pocket_w, back_pocket_h)
            extrude(amount=-back_pocket_depth, mode=Mode.SUBTRACT)

            # Fillet: 2 slot-tail break-out edges on the front face (shortest Z-parallel).
            # Re-fetch the face between fillets — its topology changes each pass.
            front_face = my_part.faces().sort_by(Axis.Y)[0]
            slot_break_edges = (
                front_face.edges().filter_by(Axis.Z).group_by(Edge.length)[0]
            )
            fillet(slot_break_edges, radius=slot_break_fillet_radius)

            # Fillet: 2 outer vertical corners of the front face (longest Z-parallel)
            front_face = my_part.faces().sort_by(Axis.Y)[0]
            front_outer_edges = (
                front_face.edges().filter_by(Axis.Z).group_by(Edge.length)[-1]
            )
            fillet(front_outer_edges, radius=front_corner_fillet_radius)

            # Fillet: 2 outer vertical corners of the back face
            back_face = my_part.faces().sort_by(Axis.Y)[-1]
            back_outer_edges = (
                back_face.edges().filter_by(Axis.Z).group_by(Edge.length)[-1]
            )
            fillet(back_outer_edges, radius=back_corner_fillet_radius)

        return my_part.part


if __name__ == "__main__":
    IdlerMountFront().export()
