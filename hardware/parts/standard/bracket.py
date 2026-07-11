from build123d import *

from hardware.parts.base import BaseStandardPart

# ── FlatBracket parameters ────────────────────────────────────────────────────
flat_plate_width = 40 * MM  # W, along X
flat_plate_height = 18 * MM  # H, along Y
flat_plate_thick = 4 * MM  # T, along Z

# Two inline through-holes along W, centered in H.
flat_hole_diameter = 5.5 * MM
flat_hole_spacing = 20 * MM  # center-to-center


# ── MotorBracket parameters ───────────────────────────────────────────────────
# NEMA 17 mounting plate. Long axis along X; the shaft pass-through is
# offset 20 mm from the LEFT (-X) edge, centered in Y. The four M3
# mounting holes sit at the corners of a 31 mm square (NEMA 17 bolt
# pattern), centered on the shaft hole.
motor_plate_length = 60 * MM  # along X
motor_plate_width = 42 * MM  # along Y
motor_plate_thick = 3 * MM  # along Z

motor_shaft_d = 25 * MM  # shaft / pilot pass-through
motor_shaft_x_offset = 20 * MM  # shaft hole center, from -X (left) edge

motor_mount_hole_d = 3.3 * MM  # M3 clearance
motor_mount_pitch = 31 * MM  # NEMA 17 square pattern, corner-to-corner

# Two M5 clearance holes inset from the +X (right) edge, aligned along
# Y with 25 mm pitch and centered top-to-bottom.
motor_m5_hole_d = 5.5 * MM  # M5 clearance
motor_m5_x_inset = 8 * MM  # from +X (right) edge
motor_m5_pitch = 25 * MM  # center-to-center, along Y

motor_corner_fillet = 3 * MM  # the four Z-parallel outer corners


# ── CornerBracket parameters ──────────────────────────────────────────────────
# L-shaped angle bracket: two faces meeting at 90°, their outer corner at the
# origin (the inner concave corner is one thickness in, at x=t, z=t).
# The bend (connection) edge runs along Y for corner_bend_length; each face
# extends corner_face_depth out from the bend — the horizontal face in +X
# (z 0..thickness), the vertical face in +Z (x 0..thickness). Each face carries
# one through-hole centered on the face (mid-bend, mid-depth).
corner_bend_length = 26 * MM  # the connection edge, along Y
corner_face_depth = 30 * MM  # how far each face reaches out from the bend
corner_thickness = 4 * MM  # plate thickness
corner_hole_d = 6.5 * MM  # one through-hole per face, centered
corner_bend_fillet = 4 * MM  # inner bend radius on the concave corner (x=t, z=t)
corner_end_fillet = (
    2 * MM
)  # round on the two inner free-end edges (x=d, z=t) and (x=t, z=d)
corner_edge_tol = 1e-3  # fuzz for matching the inner edges by center


# ── Geometry ──────────────────────────────────────────────────────────────────
class FlatBracket(BaseStandardPart):
    def name_suffix(self) -> str:
        return f"_flat_x{self.qty}"

    def bom_key(self):
        return ("FlatBracket",)

    def _build(self):
        with BuildPart() as p:
            Box(flat_plate_width, flat_plate_height, flat_plate_thick)
            with Locations((0, 0, flat_plate_thick / 2)):
                with GridLocations(flat_hole_spacing, 0, 2, 1):
                    Hole(radius=flat_hole_diameter / 2)
        return p.part


class MotorBracket(BaseStandardPart):
    """NEMA 17 motor mounting plate — 60 × 42 × 3 mm with a 25 mm
    shaft pass-through and a 31 mm square pattern of 4 × Ø3.3 M3
    mounting holes around the shaft hole."""

    def name_suffix(self) -> str:
        return f"_motor_x{self.qty}"

    def bom_key(self):
        return ("MotorBracket",)

    def _build(self):
        # Plate centered on the origin. Shaft hole offset in -X from
        # center so that x_offset measured from the left edge lands
        # at the user-spec'd 20 mm. M5 pair offset in +X from center
        # so the inset measures from the right edge.
        shaft_x = -motor_plate_length / 2 + motor_shaft_x_offset
        m5_x = motor_plate_length / 2 - motor_m5_x_inset
        with BuildPart() as p:
            Box(motor_plate_length, motor_plate_width, motor_plate_thick)
            with Locations((shaft_x, 0, motor_plate_thick / 2)):
                Hole(radius=motor_shaft_d / 2)
                with GridLocations(motor_mount_pitch, motor_mount_pitch, 2, 2):
                    Hole(radius=motor_mount_hole_d / 2)
            with Locations((m5_x, 0, motor_plate_thick / 2)):
                with GridLocations(0, motor_m5_pitch, 1, 2):
                    Hole(radius=motor_m5_hole_d / 2)
            # The 4 Z-parallel edges are the plate's outer corners; the
            # holes are cylindrical (no straight Z-edges), so filtering
            # by Axis.Z picks exactly the corners we want to round.
            fillet(p.edges().filter_by(Axis.Z), radius=motor_corner_fillet)
        return p.part


class CornerBracket(BaseStandardPart):
    """L-shaped angle bracket — a 26 mm bend edge with two 30 mm-deep faces
    at 90°, 4 mm thick. Each face carries one Ø6.5 mm through-hole centered on
    the face. The inner-face edges are filleted — the concave bend plus the
    two free-end edges."""

    def name_suffix(self) -> str:
        return f"_corner_x{self.qty}"

    def bom_key(self):
        return ("CornerBracket",)

    def _build(self):
        t, b, d = corner_thickness, corner_bend_length, corner_face_depth
        with BuildPart() as p:
            # Horizontal face: x 0..d, z 0..t. Vertical face: x 0..t, z 0..d.
            # Both span the bend length in Y; they fuse in the t×t corner.
            Box(d, b, t, align=(Align.MIN, Align.CENTER, Align.MIN))
            Box(t, b, d, align=(Align.MIN, Align.CENTER, Align.MIN))

            # Hole through the horizontal face (drill -Z from the top face),
            # centered on the face: mid-depth (x = d/2) and mid-bend (y = 0).
            with Locations(Plane.XY.offset(t)):
                with Locations((d / 2, 0)):
                    Hole(radius=corner_hole_d / 2)

            # Hole through the vertical face (drill -X from its +X face).
            # Local +Z = world +X; local +X maps to world +Y (the bend), local
            # +Y to world +Z (the face depth). Centered: mid-bend, mid-depth.
            vface = Plane(origin=(t, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0))
            with Locations(vface):
                with Locations((0, d / 2)):
                    Hole(radius=corner_hole_d / 2)

            # Fillet the inner face's three Y-parallel edges (holes are circular,
            # so the Axis.Y filter picks only straight edges; match each by its
            # center). The concave bend where the two faces meet takes the larger
            # radius; the two free-end edges far from it take a smaller round —
            # they sit on the 4 mm arms and can't carry the full bend radius.
            def y_edges_at(*centers):
                return [
                    e
                    for e in p.edges().filter_by(Axis.Y)
                    if any(
                        abs(e.center().X - cx) < corner_edge_tol
                        and abs(e.center().Z - cz) < corner_edge_tol
                        for cx, cz in centers
                    )
                ]

            fillet(y_edges_at((t, t)), radius=corner_bend_fillet)
            # re-query: the bend fillet rebuilt the solid
            fillet(y_edges_at((d, t), (t, d)), radius=corner_end_fillet)
        return p.part


if __name__ == "__main__":
    FlatBracket().export()
    MotorBracket().export()
    CornerBracket().export()
