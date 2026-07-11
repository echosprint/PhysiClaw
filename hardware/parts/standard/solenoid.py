from build123d import *

from hardware.parts.base import BaseStandardPart

# ── Parameters ────────────────────────────────────────────────────────────────
outer_w = 15 * MM  # X, on XZ plane
outer_h = 30 * MM  # Z, on XZ plane
depth = 13 * MM  # extrusion along the plane normal (Y)
inner_w = 13 * MM  # X cavity — leaves 1 mm walls on each X side
inner_h = 28 * MM  # Z cavity — leaves 1 mm walls on each Z side

# Inner cylinder (coil core), axis along Z, centered in the cavity.
cyl_radius = 6 * MM
cyl_height = 28 * MM

# Plunger rod protruding above the shell top face, concentric with the coil.
rod_diameter = 5.8 * MM
rod_height = 20 * MM

# Slot cut across the rod top (along Y), e.g. for a horn/clevis pin.
slot_width = 1.6 * MM  # along X
slot_depth = 8 * MM  # down from the rod top face

# Cross-pin hole through the rod along X, measured down from the rod top.
hole_diameter = 2 * MM
hole_z_from_top = 2.8 * MM

# Retaining flange around the rod (id matches rod, forms a collar).
ring_id = 5.8 * MM
ring_od = 8 * MM
ring_thick = 1 * MM
ring_z_from_top = 9 * MM

# Compression spring between shell top face and ring bottom, around the rod.
spring_id = 6 * MM  # just over rod_diameter for slip fit
spring_wire_dia = 0.8 * MM
spring_pitch = 2 * MM

# Bottom pin extending in -Z from the shell bottom face.
bottom_rod_radius = 1.4 * MM
bottom_rod_height = 10 * MM

# Stacked annular grooves on the bottom rod (thread-like appearance).
groove_length_from_bottom = 4 * MM  # grooves fill the bottom 4 mm
groove_width = 0.2 * MM  # along Z
groove_depth = 0.1 * MM  # radial
groove_pitch = 0.5 * MM  # center-to-center between adjacent grooves

# Bushing ring just below the shell, around the bottom rod.
bottom_ring_id = 3 * MM
bottom_ring_od = 4 * MM
bottom_ring_thick = 2 * MM

# Two M3 mounting holes on the left face, drilled along +X.
mount_hole_dia = 3 * MM
mount_hole_depth = 1 * MM
mount_hole_z_spacing = 15 * MM

# Two lead wires exiting the shell back face (-Y), near the bottom.
# Each is an L: a circle extruded out along -Y, then bent down along -Z,
# with a sphere rounding the elbow.
wire_radius = 0.5 * MM
wire_length = 4 * MM  # protrusion along -Y from the back face
wire_drop = 8 * MM  # then bends down along -Z
wire_x_offset = 4 * MM  # half-spacing → wires at x = ±wire_x_offset
wire_z = -outer_h / 2 + 4 * MM  # height up from the shell bottom


# ── Geometry ──────────────────────────────────────────────────────────────────
class Solenoid(BaseStandardPart):
    def _build(self):
        with BuildPart() as p:
            # Outer body: 15 × 30 sketch on XZ, extruded 13 mm along the normal.
            with BuildSketch(Plane.XZ):
                Rectangle(outer_w, outer_h)
            extrude(amount=depth)
            # Cavity: 13 × 28 cut at the same depth → shell open on both Y faces.
            with BuildSketch(Plane.XZ):
                Rectangle(inner_w, inner_h)
            extrude(amount=depth, mode=Mode.SUBTRACT)
            # Coil cylinder, Z-axis, centered at the shell's Y midpoint.
            # Plane.XZ extrudes along -Y, so the shell's Y center is -depth/2.
            with Locations((0, -depth / 2, 0)):
                Cylinder(radius=cyl_radius, height=cyl_height)
            # Plunger rod above the shell top face (Z = +outer_h/2).
            top_z = outer_h / 2
            with Locations((0, -depth / 2, top_z + rod_height / 2)):
                Cylinder(radius=rod_diameter / 2, height=rod_height)
            # Slot across the rod top: Y-aligned, cut from Z=rod_top downward.
            rod_top_z = top_z + rod_height
            with Locations((0, -depth / 2, rod_top_z - slot_depth / 2)):
                Box(slot_width, rod_diameter * 2, slot_depth, mode=Mode.SUBTRACT)
            # Cross-pin hole along X, 2.8 mm below the rod top.
            hole_z = rod_top_z - hole_z_from_top
            with Locations((0, -depth / 2, hole_z)):
                Cylinder(
                    radius=hole_diameter / 2,
                    height=rod_diameter * 2,
                    rotation=(0, 90, 0),
                    mode=Mode.SUBTRACT,
                )
            # Retaining flange around the rod, 9 mm below the rod top.
            ring_z = rod_top_z - ring_z_from_top
            with Locations((0, -depth / 2, ring_z)):
                Cylinder(radius=ring_od / 2, height=ring_thick)
                Cylinder(radius=ring_id / 2, height=ring_thick, mode=Mode.SUBTRACT)
            # Two M3 blind holes on the left face, drilled into +X.
            # Centers symmetric about Z=0 (face midpoint), spaced 15 mm vertically.
            hole_center_x = -outer_w / 2 + mount_hole_depth / 2
            with Locations(
                (hole_center_x, -depth / 2, -mount_hole_z_spacing / 2),
                (hole_center_x, -depth / 2, mount_hole_z_spacing / 2),
            ):
                Cylinder(
                    radius=mount_hole_dia / 2,
                    height=mount_hole_depth,
                    rotation=(0, 90, 0),
                    mode=Mode.SUBTRACT,
                )
            # Bottom pin extending in -Z from the shell bottom face.
            bottom_z = -outer_h / 2
            with Locations((0, -depth / 2, bottom_z - bottom_rod_height / 2)):
                Cylinder(radius=bottom_rod_radius, height=bottom_rod_height)
            # Annular grooves on the bottom 4 mm of the rod. Per groove,
            # subtract a thin disc and add back the inner core so only the
            # outer ring is removed (cheaper than 3D booleans on a torus).
            rod_tip_z = bottom_z - bottom_rod_height
            n_grooves = int(groove_length_from_bottom / groove_pitch)
            for i in range(n_grooves):
                z = rod_tip_z + (i + 0.5) * groove_pitch
                with Locations((0, -depth / 2, z)):
                    Cylinder(
                        radius=bottom_rod_radius + 0.5 * MM,
                        height=groove_width,
                        mode=Mode.SUBTRACT,
                    )
                    Cylinder(
                        radius=bottom_rod_radius - groove_depth,
                        height=groove_width,
                    )

        # Helical spring between shell top and ring bottom. Built as a separate
        # solid in a Compound — fusing it into the main BuildPart breaks the
        # boolean union where the wire is tangent to the shell/ring faces.
        wire_r = spring_wire_dia / 2
        with BuildPart() as spring:
            spring_path = Helix(
                pitch=spring_pitch,
                height=(ring_z - ring_thick / 2) - top_z - 2 * wire_r,
                radius=(spring_id + spring_wire_dia) / 2,
                center=(0, -depth / 2, top_z + wire_r),
            )
            with BuildSketch(Plane(origin=spring_path @ 0, z_dir=spring_path % 0)):
                Circle(wire_r)
            sweep(path=spring_path)

        # Bushing ring just below the shell — separate solid so its inner
        # subtract doesn't carve into the bottom rod (id > rod diameter).
        with BuildPart() as bottom_ring:
            with Locations((0, -depth / 2, bottom_z - bottom_ring_thick / 2)):
                Cylinder(radius=bottom_ring_od / 2, height=bottom_ring_thick)
                Cylinder(
                    radius=bottom_ring_id / 2,
                    height=bottom_ring_thick,
                    mode=Mode.SUBTRACT,
                )

        # Two lead wires: circles drawn on the XZ plane (offset back to the
        # coil's Y center so each wire roots *inside* the coil) and extruded
        # out along -Y, through the open back face, to protrude as leads.
        # Plane.XZ normal is -Y, so a positive extrude amount runs outward
        # to the back; the tip sits wire_length beyond the back face.
        coil_y = -depth / 2
        tip_y = -depth - wire_length
        with BuildPart() as wires:
            with BuildSketch(Plane.XZ.offset(-coil_y)):
                with Locations((-wire_x_offset, wire_z), (wire_x_offset, wire_z)):
                    Circle(wire_radius)
            extrude(amount=wire_length - coil_y)
            # Bend: from each horizontal tip (Y=tip_y), drop along -Z.
            with BuildSketch(Plane.XY.offset(wire_z)):
                with Locations((-wire_x_offset, tip_y), (wire_x_offset, tip_y)):
                    Circle(wire_radius)
            extrude(amount=-wire_drop)
            # Round elbow: a sphere at each corner fills the outer notch the
            # two perpendicular cylinders leave, fusing them into one wire.
            with Locations(
                (-wire_x_offset, tip_y, wire_z),
                (wire_x_offset, tip_y, wire_z),
            ):
                Sphere(radius=wire_radius)

        body = Compound(
            label="Solenoid",
            children=[p.part, spring.part, bottom_ring.part, wires.part],
        )

        # Mating joint: the bottom rod tip. Identity orientation so a partner
        # joint placed inside its part (e.g. Tip.solenoid_mount at the M3 hole
        # bottom) lands the partner upright below the solenoid.
        RigidJoint(
            "tip_mount",
            to_part=body,
            joint_location=Location(
                (0, -depth / 2, -outer_h / 2 - bottom_rod_height),
            ),
        )
        return body


if __name__ == "__main__":
    Solenoid().export()
