from build123d import *

from hardware.parts.base import BaseStandardPart

# ── Rail parameters (HIWIN MGN9H spec, G99TE25-2602) ──────────────────────────
rail_width = 9 * MM  # W_R
rail_height = 6.5 * MM  # H_R
rail_corner_chamfer = 0.3 * MM  # 4 outer X-parallel corners

# Right half of the rail cross-section in (Y, Z); centerline at Y=0. The
# inset between (3.9, 4.25)→(3.3, 4.25)→(3.3, 5)→(3.9, 5) is the side
# ball-track notch on the upper shoulder. Mirrored about Y=0 at build time.
rail_half_profile = (
    (0, 0),
    (4.5, 0),
    (4.5, 3.5),
    (3.9, 4.25),
    (3.3, 4.25),
    (3.3, 5),
    (3.9, 5),
    (4.5, 5.75),
    (4.5, 6.5),
    (0, 6.5),
)

# Rail mounting holes — M3 clearance through-hole + counterbore on top face.
rail_hole_dia = 3.5 * MM  # d (M3 close)
rail_cbore_dia = 6 * MM  # D
rail_cbore_depth = 3.5 * MM  # h
rail_hole_pitch = 20 * MM  # P
rail_cbore_chamfer = 0.1 * MM  # break the cbore rim

# ── Slider parameters (HIWIN MGN9H spec) ──────────────────────────────────────
block_bottom_z = 2 * MM  # H1 (rail base → block bottom)
block_top_z = 10 * MM  # H (assembly height)
slider_width = 20 * MM  # W
slider_main_length = 29.9 * MM  # L1 (metal body, excluding end seals)
cap_length = 5 * MM  # tailer / header along X
cap_width_inset = 0.2 * MM  # cap is this much narrower than the main body
cap_top_drop = 0.4 * MM  # cap top sits this far below slider top
cavity_clearance = 0.1 * MM  # rail cavity offset wider than the rail outline
slider_position = 0.6  # slider center along rail (0 = -X end, 1 = +X end)

# Shallow recess on the main-body top face (mounting decoration / wire relief).
top_recess_width = 9 * MM  # along Y
top_recess_depth = 0.3 * MM  # along -Z

# 4× M3 mounting holes on the main-body top face.
mount_hole_dia = 3 * MM
mount_hole_depth = 3 * MM
mount_pitch_x = 16 * MM  # C (along rail)
mount_pitch_y = 15 * MM  # B (across rail)
mount_hole_chamfer = 0.5 * MM  # top rim
top_edge_chamfer = 0.3 * MM  # two outer X-parallel top edges of the main body
cap_top_chamfer = 0.8 * MM  # two outer X-parallel top edges of each cap
cap_outer_chamfer = 0.3 * MM  # 2 verticals + 1 top edge on each cap's outer YZ face

# Lubrication port on each cap's outer YZ face (X faces).
cap_port_dia = 1.4 * MM
cap_port_z_offset = 1.8 * MM  # hole center below the cap top
cap_port_depth = 2 * MM  # bore depth into the cap

# Decorative "screw head" bosses on each cap outer face — 2 per face, looks
# like Phillips screws flanking the port.
boss_dia = 2 * MM
boss_distance = 12 * MM  # Y distance between the two bosses on a face
boss_height = 0.5 * MM  # protrusion from the face along ±X
slot_width = 0.2 * MM  # cross-slot width
slot_depth = 0.2 * MM  # cross-slot depth into the boss

# ── Derived constants (independent of rail length) ────────────────────────────
half_rail_w = rail_width / 2
half_slider_w = slider_width / 2
block_body_height = block_top_z - block_bottom_z
cap_height = block_top_z - cap_top_drop - block_bottom_z
cap_width = slider_width - cap_width_inset
half_cap_w = cap_width / 2
cap_top_z = block_top_z - cap_top_drop
body_center_z = (block_bottom_z + block_top_z) / 2
cap_center_z = (block_bottom_z + cap_top_z) / 2
slider_total_length = slider_main_length + 2 * cap_length


# ── Geometry ──────────────────────────────────────────────────────────────────
class MGN9H(BaseStandardPart):
    """HIWIN MGN9H linear guideway — 9 mm rail + sliding carriage.

    Rail bottom face at Z=0, centered in X and Y. Carriage sits H1=2 mm
    above the rail base (block bottom Z=2, top Z=10). The carriage is a
    compound of a main body (L1=29.9 mm) bracketed by two 5 mm "end-seal"
    caps that are slightly narrower and shorter than the main body."""

    def __init__(
        self,
        rail_length: float = 150 * MM,
        slider_position: float = slider_position,
        qty: int = 1,
    ):
        super().__init__(qty=qty)
        self.rail_length = rail_length
        self.slider_position = slider_position

    def name_suffix(self) -> str:
        return f"_{int(self.rail_length)}mm_x{self.qty}"

    def bom_key(self):
        return ("MGN9H", int(self.rail_length))

    def geom_key(self):
        # slider_position affects shape but is irrelevant to BOM aggregation
        return ("MGN9H", self.rail_length, self.slider_position)

    def _build(self):
        rail_plane = Plane.YZ.offset(-self.rail_length / 2)
        # Slider compound shifted so its center sits at self.slider_position along the rail.
        slider_center_x = (
            -self.rail_length / 2 + self.slider_position * self.rail_length
        )
        slider_tail_x = slider_center_x - slider_total_length / 2
        slider_base = Plane.YZ.offset(slider_tail_x)

        # Rail
        with BuildPart() as rail_p:
            with BuildSketch(rail_plane):
                with BuildLine():
                    Polyline(*rail_half_profile, close=True)
                make_face()
                # Mirror about sketch local x=0 (world Y=0) → full cross-section.
                mirror(about=Plane.YZ)
            extrude(amount=self.rail_length)
            # Chamfer the 4 outer corners (X-parallel edges at Y=±half_rail_w,
            # Z=0 bottom and Z=rail_height top).
            corners = [
                e
                for e in rail_p.edges().filter_by(Axis.X)
                if abs(abs(e.center().Y) - half_rail_w) < 0.1
                and (abs(e.center().Z) < 0.1 or abs(e.center().Z - rail_height) < 0.1)
            ]
            chamfer(corners, length=rail_corner_chamfer)
            # Counterbored M3 mounting holes along the rail top face.
            n_holes = max(1, int(self.rail_length // rail_hole_pitch))
            with Locations((0, 0, rail_height)):
                with GridLocations(rail_hole_pitch, 0, n_holes, 1):
                    CounterBoreHole(
                        radius=rail_hole_dia / 2,
                        counter_bore_radius=rail_cbore_dia / 2,
                        counter_bore_depth=rail_cbore_depth,
                    )
            # Chamfer the counterbore top rims.
            cbore_edges = [
                e
                for e in rail_p.edges()
                if e.geom_type == GeomType.CIRCLE
                and abs(e.radius - rail_cbore_dia / 2) < 0.01
                and abs(e.center().Z - rail_height) < 0.01
            ]
            chamfer(cbore_edges, length=rail_cbore_chamfer)

        # Slider — main body bracketed by tailer (-X) and header (+X) caps.
        main_center_x = slider_center_x  # symmetric compound

        with BuildPart() as slider_p:
            # Main body (after the tailer, before the header).
            with BuildSketch(slider_base.offset(cap_length)):
                with Locations((0, body_center_z)):
                    Rectangle(slider_width, block_body_height)
            extrude(amount=slider_main_length)
            # Tailer (-X end) and header (+X end) caps — narrower, dropped top.
            for plane in (
                slider_base,
                slider_base.offset(cap_length + slider_main_length),
            ):
                with BuildSketch(plane):
                    with Locations((0, cap_center_z)):
                        Rectangle(cap_width, cap_height)
                extrude(amount=cap_length)
            # Rail-shaped cavity through the full slider length, offset wider
            # by cavity_clearance for a slip fit on the rail.
            with BuildSketch(slider_base):
                with BuildLine():
                    Polyline(*rail_half_profile, close=True)
                make_face()
                mirror(about=Plane.YZ)
                offset(amount=cavity_clearance)
            extrude(amount=slider_total_length, mode=Mode.SUBTRACT)
            # Shallow top-face recess on the main body.
            with Locations((main_center_x, 0, block_top_z - top_recess_depth / 2)):
                Box(
                    slider_main_length,
                    top_recess_width,
                    top_recess_depth,
                    mode=Mode.SUBTRACT,
                )
            # 4× M3 mounting holes on the main-body top face.
            with Locations((main_center_x, 0, block_top_z)):
                with GridLocations(mount_pitch_x, mount_pitch_y, 2, 2):
                    Hole(radius=mount_hole_dia / 2, depth=mount_hole_depth)
            # Chamfer the M3 hole top rims.
            m3_edges = [
                e
                for e in slider_p.edges()
                if e.geom_type == GeomType.CIRCLE
                and abs(e.radius - mount_hole_dia / 2) < 0.01
                and abs(e.center().Z - block_top_z) < 0.01
            ]
            chamfer(m3_edges, length=mount_hole_chamfer)
            # Chamfer the two outer main-body top X-edges (Y=±half_slider_w).
            top_outer = [
                e
                for e in slider_p.edges().filter_by(Axis.X)
                if abs(e.center().Z - block_top_z) < 0.01
                and abs(abs(e.center().Y) - half_slider_w) < 0.01
            ]
            chamfer(top_outer, length=top_edge_chamfer)
            # Chamfer the 4 outer cap top X-edges (2 per cap, at Y=±half_cap_w, Z=cap_top_z).
            cap_top_outer = [
                e
                for e in slider_p.edges().filter_by(Axis.X)
                if abs(e.center().Z - cap_top_z) < 0.01
                and abs(abs(e.center().Y) - half_cap_w) < 0.01
            ]
            chamfer(cap_top_outer, length=cap_top_chamfer)
            # Chamfer the outer YZ-face edges of each cap (2 verticals + 1 top
            # horizontal per cap) — the face far from the main body.
            tailer_outer_x = slider_tail_x
            header_outer_x = slider_tail_x + slider_total_length
            cap_outer_edges = []
            for x in (tailer_outer_x, header_outer_x):
                cap_outer_edges += [
                    e
                    for e in slider_p.edges().filter_by(Axis.Y)
                    if abs(e.center().Z - cap_top_z) < 0.01
                    and abs(e.center().X - x) < 0.01
                ]
                cap_outer_edges += [
                    e
                    for e in slider_p.edges().filter_by(Axis.Z)
                    if abs(abs(e.center().Y) - half_cap_w) < 0.01
                    and abs(e.center().X - x) < 0.01
                ]
            chamfer(cap_outer_edges, length=cap_outer_chamfer)
            # Lubrication port on each cap's outer YZ face — drilled inward
            # along X (tailer drills +X, header drills −X).
            port_z = cap_top_z - cap_port_z_offset
            with Locations(
                (tailer_outer_x + cap_port_depth / 2, 0, port_z),
                (header_outer_x - cap_port_depth / 2, 0, port_z),
            ):
                Cylinder(
                    radius=cap_port_dia / 2,
                    height=cap_port_depth,
                    rotation=(0, 90, 0),
                    mode=Mode.SUBTRACT,
                )
            # Two Ø3 × 1 mm bosses on each cap outer face (±6 mm in Y around
            # the port), each cross-slotted to look like a Phillips screw.
            half_boss_dist = boss_distance / 2
            for face_x, drill_dir in ((tailer_outer_x, -1), (header_outer_x, +1)):
                boss_center_x = face_x + drill_dir * boss_height / 2
                slot_center_x = face_x + drill_dir * (boss_height - slot_depth / 2)
                for y_sign in (-1, +1):
                    y = y_sign * half_boss_dist
                    with Locations((boss_center_x, y, port_z)):
                        Cylinder(
                            radius=boss_dia / 2,
                            height=boss_height,
                            rotation=(0, 90, 0),
                        )
                    with Locations((slot_center_x, y, port_z)):
                        Box(slot_depth, boss_dia, slot_width, mode=Mode.SUBTRACT)
                        Box(slot_depth, slot_width, boss_dia, mode=Mode.SUBTRACT)

        return Compound(label="MGN9H", children=[rail_p.part, slider_p.part])


if __name__ == "__main__":
    MGN9H().export()
