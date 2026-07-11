import math

from build123d import *

from hardware.parts.base import BaseStandardPart

# ── Dimension tables (mm) ─────────────────────────────────────────────────────
# Metric L-shape hex driver (Allen key), DIN 911 short-arm series. `s` is the
# across-flats hex dimension — must match the screw socket size. Arm lengths
# are from the bend centerline to the tip; the bend itself is a circular arc
# of radius ≈ s so the sweep stays tangent-continuous.
HEX_DIMS = {
    "2mm": {"s": 2.0, "short_arm": 16.0, "long_arm": 50.0},
    "3mm": {"s": 3.0, "short_arm": 20.0, "long_arm": 63.0},
    "4mm": {"s": 4.0, "short_arm": 25.0, "long_arm": 70.0},
    "5mm": {"s": 5.0, "short_arm": 28.0, "long_arm": 80.0},
}


# ── Geometry ──────────────────────────────────────────────────────────────────
class HexDriver(BaseStandardPart):
    """L-shape metric hex driver (Allen key).

    Long arm runs along −Z down to its tip, short arm runs along +X to its
    tip; both arm lengths are measured from the bend centerline. The hex
    cross-section is swept along a path that bends through a circular arc
    so the section stays perpendicular to the wire along its length."""

    def __init__(self, size: str = "2mm", qty: int = 1):
        super().__init__(qty=qty)
        if size not in HEX_DIMS:
            raise ValueError(
                f"HexDriver has no entry for size {size!r}; "
                f"available: {sorted(HEX_DIMS)}"
            )
        self.size = size

    def name_suffix(self) -> str:
        return f"_{self.size}_x{self.qty}"

    def bom_key(self):
        return ("HexDriver", self.size)

    def _build(self):
        dim = HEX_DIMS[self.size]
        s = dim["s"]
        L_short = dim["short_arm"]
        L_long = dim["long_arm"]
        r_bend = s  # bend radius ≈ across-flats — matches typical Allen keys
        hex_circumradius = s / (2 * math.cos(math.radians(30)))

        with BuildPart() as p:
            # Sweep path in Plane.XZ: long-arm tip at (0, −L_long) up to the
            # bend start (0, −r_bend), arc to (r_bend, 0), then out to the
            # short-arm tip (L_short, 0). RadiusArc with positive radius
            # bulges left of start→end (here toward the outer L corner at
            # the origin), so the arc center sits at the inside corner.
            with BuildLine(Plane.XZ) as path:
                Line((0, -L_long), (0, -r_bend))
                RadiusArc((0, -r_bend), (r_bend, 0), r_bend)
                Line((r_bend, 0), (L_short, 0))

            # Hex profile at the long-arm tip — Plane.XY normal is +Z, which
            # matches the path tangent there, so the sweep starts square.
            with BuildSketch(Plane.XY.offset(-L_long)):
                RegularPolygon(radius=hex_circumradius, side_count=6)

            sweep(path=path.line)

        return p.part


if __name__ == "__main__":
    for size in HEX_DIMS:
        HexDriver(size=size).export()
