import math

from build123d import *

from hardware.parts.base import BaseStandardPart

# ── Parameters ────────────────────────────────────────────────────────────────
# GT2 timing belt geometry — shared between the motor pulley (driven) and a
# free-spinning idler (used to keep tension on the belt run).
flange_diameter = 18 * MM  # outer flange OD
bore_diameter = 5 * MM  # through-bore for the motor shaft
flange_belt_h = 8.5 * MM  # axial height of the flange + belt + flange stack

belt_width = 6 * MM  # toothed band axial length
tooth_pitch = 2 * MM  # 2GT
tooth_count = 20
tooth_depth = 0.76 * MM  # GT2 tooth-gap radial depth

# Derived: pitch diameter on which the belt sits (D = N·p / π).
pitch_diameter = tooth_count * tooth_pitch / math.pi

# Hub: only on the motor pulley. Adds shaft-clamp length and carries an
# M3 tapped set-screw that presses against the motor's D-flat. The hub
# sits below the flange/belt stack (z = 0 is the motor-facing face).
hub_diameter = 18 * MM
hub_height = 7.5 * MM
set_screw_d = 3 * MM  # M3 — threads aren't modeled; the screw body is a plain rod
set_screw_z = hub_height / 2
set_screw_length = 5 * MM  # M3 × 5 grub screw
set_screw_hex_s = 1.5 * MM  # hex drive across-flats (ISO 4026 for M3)
set_screw_hex_depth = 1.5 * MM

# Bearing-look ring grooves: thin annular indents on the idler's top and
# bottom faces — visual race-separator lines. Skipped on the pulley (its
# bottom face is hidden against the motor body).
ring_grooves = [
    (6.2 * MM, 6.4 * MM),
]
ring_depth = 0.3 * MM

# Lead-in chamfer on the bore openings (top + bottom faces) so the part
# slips onto its shaft without scraping.
bore_chamfer = 0.2 * MM

# Lead-in chamfer where each set-screw hole breaks through the hub OD.
set_screw_chamfer = 0.3 * MM

# Tolerances used by edge-picking filters in `_build`.
FLOAT_TOL = 0.01 * MM  # exact-match slack for face Z + radius
chamfer_z_tol = 1.5 * MM  # axial window around set_screw_z
chamfer_r_tol = 1.0 * MM  # radial slack inside hub_r when picking the entry curve

KINDS = ("pulley", "idler")


def _build_set_screw():
    """One M3 set screw on +Y — plain rod with a hex socket on the outer
    end. Threads aren't modeled; just the visible body + drive socket.
    Built along +Z, then rotated so its axis lies on +Y (the radial
    direction through the hub) and translated so the hex-socket end sits
    flush with the hub OD at z = set_screw_z."""
    hex_circumradius = set_screw_hex_s / (2 * math.cos(math.radians(30)))
    with BuildPart() as ss:
        Cylinder(
            radius=set_screw_d / 2,
            height=set_screw_length,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        with BuildSketch(Plane.XY.offset(set_screw_length)):
            RegularPolygon(radius=hex_circumradius, side_count=6)
        extrude(amount=-set_screw_hex_depth, mode=Mode.SUBTRACT)
    return ss.part.rotate(Axis.X, -90).translate(
        (0, hub_diameter / 2 - set_screw_length, set_screw_z)
    )


# ── Geometry ──────────────────────────────────────────────────────────────────
class Pulley2GT20T(BaseStandardPart):
    """GT2 20-tooth timing pulley or idler.

    `pulley` — sits on the motor: always toothed, with an extended hub and
    an M3 set screw aimed at the motor's D-flat (+Y face, per motor.py).
    `idler` — free-spinning tension wheel: no hub. The `toothed` flag picks
    between toothed (belt-tracking) and smooth (low-friction) idlers.
    Stacks upward from z = 0 (the motor-facing / mount-facing bottom face)."""

    def __init__(self, kind: str = "pulley", toothed: bool = True, qty: int = 1):
        super().__init__(qty=qty)
        if kind not in KINDS:
            raise ValueError(
                f"Pulley2GT20T kind {kind!r} not recognized; expected one of {KINDS}"
            )
        if kind == "pulley" and not toothed:
            raise ValueError(
                "pulley must be toothed (it drives the belt); use kind='idler' "
                "for a smooth wheel"
            )
        self.kind = kind
        self.toothed = toothed

    def name_suffix(self) -> str:
        smooth_tag = "" if self.toothed else "_smooth"
        return f"_{self.kind}{smooth_tag}_x{self.qty}"

    def bom_key(self):
        return ("Pulley2GT20T", self.kind, "toothed" if self.toothed else "smooth")

    def _build(self):
        is_pulley = self.kind == "pulley"
        hub_h = hub_height if is_pulley else 0.0
        flange_thick = (flange_belt_h - belt_width) / 2

        with BuildPart() as p:
            if is_pulley:
                Cylinder(
                    radius=hub_diameter / 2,
                    height=hub_h,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                )

            with Locations((0, 0, hub_h)):
                Cylinder(
                    radius=flange_diameter / 2,
                    height=flange_thick,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                )
            with Locations((0, 0, hub_h + flange_thick)):
                Cylinder(
                    radius=pitch_diameter / 2,
                    height=belt_width,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                )
            with Locations((0, 0, hub_h + flange_thick + belt_width)):
                Cylinder(
                    radius=flange_diameter / 2,
                    height=flange_thick,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                )

            # Belt teeth — always present on the pulley; optional on idlers.
            if self.toothed:
                overshoot = 0.5 * MM
                radial_center = pitch_diameter / 2 - tooth_depth / 2 + overshoot / 2
                chord = math.pi * pitch_diameter / tooth_count / 2  # half-pitch chord
                band_z_center = hub_h + flange_thick + belt_width / 2
                with Locations((0, 0, band_z_center)):
                    with PolarLocations(radial_center, tooth_count):
                        Box(
                            chord,
                            tooth_depth + overshoot,
                            belt_width,
                            mode=Mode.SUBTRACT,
                        )

            # Through-bore: 1 mm of overshoot above the top face for a
            # clean Boolean cut; the bottom face is hidden against the motor
            # body so coincident alignment there is fine.
            total_h = hub_h + flange_belt_h
            Cylinder(
                radius=bore_diameter / 2,
                height=total_h + 1 * MM,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
                mode=Mode.SUBTRACT,
            )

            # Set screws (pulley only): two M3 radial holes 90° apart in
            # top view. The +Y screw presses on the motor's D-flat; the +X
            # screw wedges the round side of the shaft against it.
            if is_pulley:
                hole_planes = [
                    Plane(
                        origin=(0, 0, set_screw_z), x_dir=(1, 0, 0), z_dir=(0, 1, 0)
                    ),  # +Y
                    Plane(
                        origin=(0, 0, set_screw_z), x_dir=(0, 1, 0), z_dir=(1, 0, 0)
                    ),  # +X
                ]
                for plane in hole_planes:
                    with BuildSketch(plane):
                        Circle(set_screw_d / 2)
                    extrude(amount=hub_diameter / 2 + 0.5 * MM, mode=Mode.SUBTRACT)

            if not is_pulley:
                for z in (0, flange_belt_h - ring_depth):
                    for inner_d, outer_d in ring_grooves:
                        with BuildSketch(Plane(origin=(0, 0, z))):
                            Circle(outer_d / 2)
                            Circle(inner_d / 2, mode=Mode.SUBTRACT)
                        extrude(amount=ring_depth, mode=Mode.SUBTRACT)

            # Bore lead-in chamfer: pick the two circular bore-mouth edges
            # by radius + Z. Other circles (flange OD, grooves, set-screw
            # hole) have different radii or different Z and are filtered out.
            bore_r = bore_diameter / 2
            bore_mouths = [
                e
                for e in p.edges().filter_by(GeomType.CIRCLE)
                if abs(e.radius - bore_r) < FLOAT_TOL
                and (
                    abs(e.center().Z) < FLOAT_TOL
                    or abs(e.center().Z - total_h) < FLOAT_TOL
                )
            ]
            chamfer(bore_mouths, length=bore_chamfer)

            # Chamfer where each set-screw hole exits the hub OD. The exit
            # edge is a cylinder-cylinder intersection (BSPLINE), so we
            # filter by curve type + axial proximity to set_screw_z +
            # radial distance close to hub_r. The inner edges where each
            # hole meets the bore sit at radius ≈ bore_r and are excluded.
            if is_pulley:
                hub_r = hub_diameter / 2
                entry_edges = []
                for e in p.edges().filter_by(GeomType.BSPLINE):
                    c = e.center()
                    if (
                        abs(c.Z - set_screw_z) < chamfer_z_tol
                        and (c.X**2 + c.Y**2) ** 0.5 > hub_r - chamfer_r_tol
                    ):
                        entry_edges.append(e)
                chamfer(entry_edges, length=set_screw_chamfer)

        if is_pulley:
            # Bundle the pulley body with two separate set-screw solids
            # (90° apart in top view) so the exported STEP shows both screws
            # seated in their holes.
            screw_y = _build_set_screw()
            screw_x = screw_y.rotate(Axis.Z, -90)  # +Y screw → +X screw
            return Compound(
                label="pulley",
                children=[p.part, screw_y, screw_x],
            )
        return p.part


if __name__ == "__main__":
    Pulley2GT20T(kind="pulley").export()
    Pulley2GT20T(kind="idler", toothed=True).export()
    Pulley2GT20T(kind="idler", toothed=False).export()
