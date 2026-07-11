import math

from build123d import *

from hardware.parts.base import BaseStandardPart

# ── Dimension tables (mm) ─────────────────────────────────────────────────────
# Hex nut: ISO 4032 / DIN 934 regular pattern, nominal dims.
# Square nut: DIN 557 regular pattern, nominal dims for M4/M5. DIN 557
# itself starts at M4; the M3 entry below is non-spec, sized to match the
# M3 hex nut (af / thickness / bore borrowed from ISO 4032) so M3 hex and
# square share hardware. DIN 557 doesn't spec a chamfer; the chamfer here
# is a visual touch so the part doesn't read as a raw CAD primitive.
# `af` is across-flats (hex) / side length (square). `bore` is the nominal
# thread Ø — threads aren't modeled, so the through-hole is a smooth
# cylinder of that diameter. The chamfer terminates at the inscribed
# circle (radius = af / 2) for both shapes — the cone never narrows past
# the flats, so only the corners get clipped.
SPECS = {
    "hex": {
        "M3": {"af": 5.5, "thickness": 2.4, "bore": 3.0},
        "M4": {"af": 7.0, "thickness": 3.2, "bore": 4.0},
        "M5": {"af": 8.0, "thickness": 4.7, "bore": 5.0},
        # 1/4"-20 UNC hex nut (ASME B18.2.2). af = 7/16", thickness = 7/32",
        # bore = nominal 1/4" — threads onto the Gooseneck male stud.
        "1/4-20": {"af": 11.11, "thickness": 5.56, "bore": 6.35},
    },
    "square": {
        "M3": {"af": 5.5, "thickness": 2.4, "bore": 3.0},
        "M4": {"af": 7.0, "thickness": 3.2, "bore": 4.0},
        "M5": {"af": 8.0, "thickness": 4.0, "bore": 5.0},
    },
}

# Polygon sides per shape — drives both the sketch and the corner-radius
# formula (corner_r = af / (2 cos(180° / n))).
SIDES = {"hex": 6, "square": 4}

# Sketch rotation per shape, in degrees. Hex stays vertex-up (the
# conventional orientation in fastener drawings); square rotates 45° so
# its edges land on the X/Y axes instead of the diagonal.
ROTATION_DEG = {"hex": 0, "square": 45}

# 30° from the bearing face (ISO 4032 figure). Same angle is reused for
# the square nut so both shapes share the chamfer envelope.
CHAMFER_ANGLE_DEG = 30.0


# ── Geometry ──────────────────────────────────────────────────────────────────
class Nut(BaseStandardPart):
    """Hex (ISO 4032) or square (DIN 557 / +M3) nut with a smooth through-bore.

    Centered on the Z axis with its bottom face at z = 0. Single-chamfered:
    the bottom face is the full polygon, and the top is a 30° conical
    chamfer that takes the corners down to the inscribed circle. Because
    the cone terminates exactly at the inscribed radius, it never intrudes
    on the flats — only the corners get clipped, and the top face is the
    inscribed circle."""

    def __init__(self, shape: str, size: str, qty: int = 1):
        super().__init__(qty=qty)
        if shape not in SPECS:
            raise ValueError(f"Nut shape must be one of {sorted(SPECS)}; got {shape!r}")
        if size not in SPECS[shape]:
            raise ValueError(
                f"{shape} nut has no entry for size {size!r}; "
                f"available: {sorted(SPECS[shape])}"
            )
        s = SPECS[shape][size]
        self.shape = shape
        self.size = size
        self.af = s["af"]
        self.thickness = s["thickness"]
        self.bore = s["bore"]

    def name_suffix(self) -> str:
        # Sanitize the size for the filename ("1/4-20" → "1-4-20").
        safe_size = self.size.replace("/", "-")
        return f"_{self.shape}_{safe_size}_x{self.qty}"

    def bom_key(self):
        return ("Nut", self.shape, self.size)

    def _build(self):
        af = self.af
        m = self.thickness

        n = SIDES[self.shape]
        # Circumradius of a regular n-gon with across-flats `af`. Reduces to
        # af/(2 cos30°) for hex (n=6) and af/√2 for square (n=4).
        corner_r = af / (2 * math.cos(math.radians(180 / n)))

        with BuildPart() as p:
            with BuildSketch():
                RegularPolygon(
                    radius=corner_r,
                    side_count=n,
                    rotation=ROTATION_DEG[self.shape],
                )
            extrude(amount=m)

            # Single-chamfer envelope: bottom face is flat (envelope at
            # corner_r from z=0 up to z=m-ch_h), then a 30° cone tapers
            # from corner_r to the inscribed circle (af / 2) at z=m.
            # Because the cone never narrows past the flats, INTERSECT
            # with the prism only clips corners — the side flats stay
            # full-height up to their tangent line at the top face.
            bearing_r = af / 2
            ch_h = (corner_r - bearing_r) * math.tan(math.radians(CHAMFER_ANGLE_DEG))
            with BuildSketch(Plane.XZ):
                with BuildLine():
                    Polyline(
                        (0, 0),
                        (corner_r, 0),
                        (corner_r, m - ch_h),
                        (bearing_r, m),
                        (0, m),
                        close=True,
                    )
                make_face()
            revolve(axis=Axis.Z, mode=Mode.INTERSECT)

            # Fillet the vertical corner edges between adjacent flats. These
            # are the only Z-parallel straight edges in the prism after the
            # chamfer cut (the bore is round, not yet present here).
            fillet(p.edges().filter_by(Axis.Z), radius=0.5 * MM)

            # Bore overshoots top and bottom by 0.5 mm for a clean through-cut.
            with BuildSketch(Plane.XY.offset(-0.5 * MM)):
                Circle(self.bore / 2)
            extrude(amount=m + 1 * MM, mode=Mode.SUBTRACT)

        return p.part


if __name__ == "__main__":
    for shape, sizes in SPECS.items():
        for size in sizes:
            Nut(shape, size).export()
