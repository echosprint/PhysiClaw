from build123d import *

from hardware.parts.base import BaseStandardPart

# ── Dimension tables (mm) ─────────────────────────────────────────────────────
# Single-arg lookup table for `Ring(spec)`. Keys follow the common washer-
# marking format "M<screw>x<outer_d>x<thickness>"; each entry carries its
# own `kind` so callers can write Ring("M3x10x1") and get a fully-specified
# part. `kind` is for naming / BOM intent only — washer and spacer share
# the same annular-cylinder geometry. Inner bores follow DIN-125-style
# clearance for the named screw (M3 → 3.2, M5 → 5.3, M6 → 6.4).
SPECS = {
    "M3x10x1": {"kind": "washer", "inner_d": 3.2, "outer_d": 10.0, "height": 1.0},
    "M5x8x0.5": {"kind": "washer", "inner_d": 5.3, "outer_d": 8.0, "height": 0.5},
    "M5x10x9": {"kind": "spacer", "inner_d": 5.3, "outer_d": 10.0, "height": 9.0},
    "M5x15x12": {"kind": "spacer", "inner_d": 5.3, "outer_d": 15.0, "height": 12.0},
    "M6x12x8": {"kind": "spacer", "inner_d": 6.4, "outer_d": 12.0, "height": 8.0},
    "M6x12x12": {"kind": "spacer", "inner_d": 6.4, "outer_d": 12.0, "height": 12.0},
}


# ── Geometry ──────────────────────────────────────────────────────────────────
class Ring(BaseStandardPart):
    """Annular cylinder — washer / spacer.

    Selected by spec key (e.g. 'M3x10x1') from SPECS; kind and dimensions
    come from the table. Centered on the Z axis with its bottom face at
    z = 0."""

    def __init__(self, spec: str, qty: int = 1):
        super().__init__(qty=qty)
        if spec not in SPECS:
            raise ValueError(f"Ring has no spec {spec!r}; available: {sorted(SPECS)}")
        s = SPECS[spec]
        self.spec = spec
        self.kind = s["kind"]
        self.inner_d = s["inner_d"]
        self.outer_d = s["outer_d"]
        self.height = s["height"]

    def name_suffix(self) -> str:
        return f"_{self.kind}_{self.spec}_x{self.qty}"

    def bom_key(self):
        return ("Ring", self.kind, self.spec)

    def _build(self):
        with BuildPart() as p:
            Cylinder(
                radius=self.outer_d / 2,
                height=self.height,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            )
            # Bore overshoots top and bottom by 0.5 mm for a clean through-cut.
            Cylinder(
                radius=self.inner_d / 2,
                height=self.height + 1 * MM,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
                mode=Mode.SUBTRACT,
            )
        return p.part


if __name__ == "__main__":
    for spec in SPECS:
        Ring(spec).export()
