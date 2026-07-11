from build123d import *

from hardware.parts.base import BaseStandardPart

# ── Shared 2020-cell parameters ───────────────────────────────────────────────
leg = 10 * MM  # outer half-extent of one 2020 cell
default_length = 200 * MM  # default extrusion length (override per-instance)
bore_diameter = 5 * MM  # center through-hole of each cell
corner_fillet = 1.5 * MM  # outer vertical corner edges
rib_fillet = 2 * MM  # inner rib vertical edges near the slot/diagonal joint

# 1/8 cross-section outline (one slot in profile). Traces, in order:
# center → bottom edge → slot notch → right edge → top corner → hypotenuse back.
wedge_vertices = (
    (0, 0),
    (3.9 * MM, 0),
    (3.9 * MM, 2.84 * MM),
    (6.56 * MM, 5.5 * MM),
    (8.2 * MM, 5.5 * MM),
    (8.2 * MM, 3.1 * MM),
    (9.5 * MM, 3.1 * MM),
    (9.5 * MM, 3.6 * MM),
    (10 * MM, 3.6 * MM),
    (10 * MM, 10 * MM),
)

# Mirror plane through the hypotenuse (line y = x in the XY plane), extending
# up Z. x_dir along the hypotenuse; z_dir is the plane normal in XY.
hypotenuse_plane = Plane(origin=(0, 0, 0), x_dir=(1, 1, 0), z_dir=(1, -1, 0))

# ── 2040-specific parameters ──────────────────────────────────────────────────
cell_offset = leg  # 2040 cell centers at ±10 mm along X
slot_w = 6 * MM  # central through-channel width (X)
slot_h = 16.4 * MM  # central through-channel height (Y)
slot_lip_under_y = 8.2 * MM  # cavity belly top, cell-local — T-nut wings
# seat here against the slot lip underside

# End-counterbore screw access on the +Y (front) face, mirrored on each
# Z end. Two per end, aligned in X with the bores at ±cell_offset.
cb_end_offset = 10 * MM  # axial offset of CB from each end face
cb_head_d = 11 * MM  # counterbore (head pocket) diameter
cb_head_depth = 5.5 * MM  # counterbore depth
cb_shaft_d = 5.5 * MM  # through-hole diameter

# Joint labels for the four end counterbores on a 2040 with cb=True.
# Shared so callers can iterate without restating the names (typo risk).
CB_LABELS = (
    "cb_bot_left",
    "cb_bot_right",
    "cb_top_left",
    "cb_top_right",
)

# ── 1020-specific parameters ──────────────────────────────────────────────────
# Half cross-section outline (right half, x ≥ 0; mirrored across the Y axis
# for the full profile). Traces, in order:
# centerline bottom → bottom-right → top-right → top edge to slot lip →
# lip underside → cavity belly → rib slope → centerline → close.
half_vertices_1020 = (
    (0, 0),
    (9.9 * MM, 0),
    (9.9 * MM, 9.9 * MM),
    (3.5 * MM, 9.9 * MM),
    (3.5 * MM, 9.4 * MM),
    (3.2 * MM, 9.4 * MM),
    (3.2 * MM, 8 * MM),
    (5.6 * MM, 8 * MM),
    (5.6 * MM, 6.4 * MM),
    (2.4 * MM, 3.6 * MM),
    (0, 3.6 * MM),
)
half_x_1020 = 9.9 * MM  # half cross-section width (= section height too)
hole_1020_d = 4.2 * MM  # through-hole diameter
hole_1020_x_inset = 3 * MM  # hole center inset from right edge
hole_1020_y_inset = 3 * MM  # hole center inset from bottom edge

# Optional end-mounting holes (Extrusion1020(hole=True)): one M5 clearance
# hole drilled vertically (through Y) at the section center, set in from each
# end face — a bolt passes up through the bottom into the T-slot.
end_hole_d = 5.5 * MM  # M5 clearance
end_hole_offset = 10 * MM  # hole center from each end face, along Z


# ── Helpers ───────────────────────────────────────────────────────────────────
def _cell_face():
    """One full 2020 cross-section face — 1/8 wedge mirrored across the
    hypotenuse, then across YZ and XZ, into the 4-way symmetric profile.
    Bore is NOT subtracted here so the caller can place cells freely."""
    with BuildSketch() as sk:
        with BuildLine():
            Polyline(*wedge_vertices, close=True)
        make_face()
        mirror(about=hypotenuse_plane)
        mirror(about=Plane.YZ)
        mirror(about=Plane.XZ)
    return sk.sketch


def _near(a, b):
    return abs(abs(a) - b) < 0.1


def _outer_corner_edges(z_edges, half_x):
    """Z-parallel edges at (±half_x, ±leg) — the 4 true outer corners."""
    return [
        e for e in z_edges if _near(e.center().X, half_x) and _near(e.center().Y, leg)
    ]


def _rib_edges(z_edges, cell_offsets):
    """For each cell centered at X = ox, the 8 rib edges where the slot
    cut meets the diagonal rib, at local (±rx, ±ry) and (±ry, ±rx) with
    (rx, ry) = wedge_vertices[2]."""
    rx, ry = wedge_vertices[2]
    edges = []
    for ox in cell_offsets:
        for e in z_edges:
            dx = e.center().X - ox
            y = e.center().Y
            if (_near(dx, rx) and _near(y, ry)) or (_near(dx, ry) and _near(y, rx)):
                edges.append(e)
    return edges


# ── Geometry ──────────────────────────────────────────────────────────────────
class Extrusion2020(BaseStandardPart):
    def __init__(self, length: float = default_length, qty: int = 1):
        super().__init__(qty=qty)
        self.length = length

    def name_suffix(self) -> str:
        return f"_2020_{int(self.length)}mm_x{self.qty}"

    def bom_key(self):
        return ("Extrusion2020", int(self.length))

    def _build(self):
        face = _cell_face()
        with BuildPart() as p:
            with BuildSketch():
                add(face)
                Circle(bore_diameter / 2, mode=Mode.SUBTRACT)
            extrude(amount=self.length)

            z_edges = p.edges().filter_by(Axis.Z)
            fillet(_outer_corner_edges(z_edges, leg), radius=corner_fillet)
            fillet(_rib_edges(z_edges, (0,)), radius=rib_fillet)

        p.part.label = f"Extrusion2020_{int(self.length)}mm"
        return p.part


class Extrusion2040(BaseStandardPart):
    """2040 = two 2020 cross-sections unioned at sketch level, then
    extruded. Stacking happens before fillet so the inner contact at X=0
    disappears and only the four true outer corners get rounded."""

    def __init__(
        self,
        length: float = default_length,
        qty: int = 1,
        cb: bool = False,
    ):
        super().__init__(qty=qty)
        self.length = length
        self.cb = cb

    def name_suffix(self) -> str:
        cb = "_cb" if self.cb else ""
        return f"_2040_{int(self.length)}mm{cb}_x{self.qty}"

    def bom_key(self):
        return ("Extrusion2040", int(self.length), "cb" if self.cb else "plain")

    def _build(self):
        face = _cell_face()
        cell_centers = ((-cell_offset, 0), (cell_offset, 0))
        with BuildPart() as p:
            with BuildSketch():
                with Locations(*cell_centers):
                    add(face)
                with Locations(*cell_centers):
                    Circle(bore_diameter / 2, mode=Mode.SUBTRACT)
                Rectangle(slot_w, slot_h, mode=Mode.SUBTRACT)
            extrude(amount=self.length)

            z_edges = p.edges().filter_by(Axis.Z)
            fillet(_outer_corner_edges(z_edges, 2 * leg), radius=corner_fillet)
            fillet(_rib_edges(z_edges, (-cell_offset, cell_offset)), radius=rib_fillet)

            if self.cb:
                # ── End counterbores on the +Y (front) face ──
                # Workplane on +Y face; +Z axis of plane = world +Z, normal
                # points into the part (-Y), so positive extrude drills inward.
                front_plane = Plane(
                    origin=(0, leg, 0),
                    x_dir=(1, 0, 0),
                    z_dir=(0, -1, 0),
                )
                cb_named = dict(
                    zip(
                        CB_LABELS,
                        [
                            (-cell_offset, cb_end_offset),
                            (cell_offset, cb_end_offset),
                            (-cell_offset, self.length - cb_end_offset),
                            (cell_offset, self.length - cb_end_offset),
                        ],
                    )
                )
                cb_centers = list(cb_named.values())
                # Through shaft + head pocket — sketched on the same plane,
                # two separate extrudes so depths can differ.
                with BuildSketch(front_plane):
                    with Locations(*cb_centers):
                        Circle(cb_shaft_d / 2)
                extrude(amount=2 * leg, mode=Mode.SUBTRACT)
                with BuildSketch(front_plane):
                    with Locations(*cb_centers):
                        Circle(cb_head_d / 2)
                extrude(amount=cb_head_depth, mode=Mode.SUBTRACT)

                # LinearJoint per counterbore — slide axis along the
                # drilling direction (inward from the +Y face). A partner
                # RigidJoint (e.g. Screw "head") clamps to a slider
                # position: 0 = head at the mouth, position>0 inserts
                # deeper, position<0 explodes the screw outboard.
                for label, (x, z) in cb_named.items():
                    LinearJoint(
                        label,
                        axis=Axis(
                            origin=(x, leg, z),
                            direction=(0, -1, 0),
                        ),
                        linear_range=(-100, cb_head_depth),
                    )

            # Slot as a 1-DOF slide axis along Z, threaded through the +X
            # (right) face — the 20 mm-wide narrow side, single slot at Y=0.
            # Axis at the lip underside, X = cell_offset + slot_lip_under_y.
            LinearJoint(
                "slot_right",
                axis=Axis(
                    origin=(cell_offset + slot_lip_under_y, 0, 0),
                    direction=(0, 0, 1),
                ),
                linear_range=(0, self.length),
            )

        cb = "_cb" if self.cb else ""
        p.part.label = f"Extrusion2040_{int(self.length)}mm{cb}"
        return p.part


class Extrusion1020(BaseStandardPart):
    """1020 T-slot extrusion — 19.8 × 9.9 mm cross-section (nominal 20 × 10),
    slot opening on the +Y face. Two through-holes drilled along Z on the
    bottom rim, mirrored about the Y axis.

    With ``hole=True``, also drills two vertical M5 clearance holes — one set
    in 10 mm from each end face, centered in width."""

    def __init__(
        self,
        length: float = default_length,
        qty: int = 1,
        hole: bool = False,
    ):
        super().__init__(qty=qty)
        self.length = length
        self.hole = hole

    def name_suffix(self) -> str:
        h = "_h" if self.hole else ""
        return f"_1020_{int(self.length)}mm{h}_x{self.qty}"

    def bom_key(self):
        return ("Extrusion1020", int(self.length), "hole" if self.hole else "plain")

    def _build(self):
        with BuildPart() as p:
            with BuildSketch():
                with BuildLine():
                    Polyline(*half_vertices_1020, close=True)
                make_face()
                mirror(about=Plane.YZ)
            extrude(amount=self.length)

            # Fillet the 4 true outer corners — Z-parallel edges at
            # (±half_x_1020, 0) and (±half_x_1020, half_x_1020). Height
            # equals half-width, so both axes use the same constant.
            z_edges = p.edges().filter_by(Axis.Z)
            outer = [
                e
                for e in z_edges
                if _near(e.center().X, half_x_1020)
                and (_near(e.center().Y, 0) or _near(e.center().Y, half_x_1020))
            ]
            fillet(outer, radius=corner_fillet)

            # Two through-holes on the bottom rim, mirrored across the Y axis.
            hole_centers = [
                (half_x_1020 - hole_1020_x_inset, hole_1020_y_inset),
                (-half_x_1020 + hole_1020_x_inset, hole_1020_y_inset),
            ]
            with BuildSketch():
                with Locations(*hole_centers):
                    Circle(hole_1020_d / 2)
            extrude(amount=self.length, mode=Mode.SUBTRACT)

            # Optional vertical M5 holes, set in 10 mm from each end face,
            # centered in width (axis along Y, through the full section).
            if self.hole:
                hole_h = half_x_1020 + 4 * MM  # > section height + overshoot
                for z in (end_hole_offset, self.length - end_hole_offset):
                    with Locations(Location((0, half_x_1020 / 2, z), (90, 0, 0))):
                        Cylinder(end_hole_d / 2, hole_h, mode=Mode.SUBTRACT)

        p.part.label = f"Extrusion1020_{int(self.length)}mm"
        return p.part


if __name__ == "__main__":
    Extrusion2020().export()
    Extrusion2040().export()
    Extrusion1020().export()
