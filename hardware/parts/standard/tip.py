from build123d import *

from hardware.parts.base import BaseStandardPart

# ── Parameters ────────────────────────────────────────────────────────────────
# Stylus tip stack, top → bottom.
seg1_diameter = 7.9 * MM
seg1_height = 3.5 * MM

seg2_diameter = 8.5 * MM
seg2_height = 4.1 * MM

seg3_diameter = 7 * MM
seg3_height = 1.4 * MM

ball_radius = 3.5 * MM  # lower hemisphere at the bottom

# M3 mounting hole on the top face, axial.
m3_hole_dia = 3 * MM
m3_hole_depth = 4 * MM

# Edge chamfer applied to seg2 top/bottom and the seg1 top face circles.
edge_chamfer = 0.2 * MM  # capped by 0.3 mm radial gap at seg1↔seg2 junction


# ── Geometry ──────────────────────────────────────────────────────────────────
class Tip(BaseStandardPart):
    def _build(self):
        segments = [
            (seg1_diameter, seg1_height),
            (seg2_diameter, seg2_height),
            (seg3_diameter, seg3_height),
        ]
        with BuildPart() as p:
            z = 0
            for diameter, height in segments:
                with Locations((0, 0, z - height / 2)):
                    Cylinder(radius=diameter / 2, height=height)
                z -= height
            # Lower hemisphere flush with the bottom cylinder face. Sphere's
            # local origin sits at the midpoint of flat-face and curve, so
            # shift down by half the radius to put the flat face at z.
            with Locations((0, 0, z - ball_radius / 2)):
                Sphere(radius=ball_radius, arc_size2=0)
            # M3 hole cut from the top face downward.
            with Locations((0, 0, -m3_hole_depth / 2)):
                Cylinder(
                    radius=m3_hole_dia / 2,
                    height=m3_hole_depth,
                    mode=Mode.SUBTRACT,
                )
            # Chamfer seg2's outer top and bottom edges (circle radius 4.25).
            seg2_top_z = -seg1_height
            seg2_bot_z = -(seg1_height + seg2_height)
            seg2_r = seg2_diameter / 2
            seg2_edges = [
                e
                for e in p.edges()
                if e.geom_type == GeomType.CIRCLE
                and abs(e.radius - seg2_r) < 0.01
                and (
                    abs(e.center().Z - seg2_top_z) < 0.01
                    or abs(e.center().Z - seg2_bot_z) < 0.01
                )
            ]
            chamfer(seg2_edges, length=edge_chamfer)
            # Chamfer both circle edges on the top face (Z=0).
            top_face = p.faces().sort_by(Axis.Z)[-1]
            chamfer(top_face.edges(), length=edge_chamfer)

            # Mating joint: M3 hole bottom (Z = -m3_hole_depth). The solenoid
            # rod tip touches this point at full insertion.
            RigidJoint(
                "solenoid_mount",
                joint_location=Location((0, 0, -m3_hole_depth)),
            )
        return p.part


if __name__ == "__main__":
    Tip().export()
