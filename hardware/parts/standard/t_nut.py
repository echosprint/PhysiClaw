from build123d import *

from hardware.parts.base import BaseStandardPart

# ── Dimension tables (mm) ─────────────────────────────────────────────────────
# 2020 T-slot nut half cross-section in XY (mirrored about x = 0). Each kind
# has its own profile because they engage the slot in different ways:
#
# "standard" — slides in from the open END of the extrusion. The profile
# traces centerline → bottom → trapezoidal flare → wings → step → neck →
# top, matching the 2020 slot envelope:
#   y ∈ [0,   2.8]: trapezoidal flare, half-width 3.25 → 4.5. Rides the 45°
#                   rib slope at the cavity floor.
#   y ∈ [2.8, 3.3]: wings, half-width 4.5 (full 9.0). Sits behind the slot
#                   lip — this is what locks the nut against pull-out.
#   y ∈ [3.3, 4.5]: neck, half-width 3.0 (full 6.0). Passes through the
#                   6.2 mm slot mouth.
#
# "hammer" — tilted-insert: slim flat plate (full width 10, height 1.5) with
# a chamfered bottom corner so it can be tipped on edge through the 6.2 mm
# slot mouth, then rotated flat to lock the 10 mm width behind the lip.
HALF_PROFILES = {
    "standard": (
        (0, 0),
        (3.25, 0),
        (4.5, 2.8),
        (4.5, 3.3),
        (3, 3.3),
        (3, 4.5),
        (0, 4.5),
    ),
    "hammer": (
        (0, 0),
        (4.3, 0),
        (5, 0.7),
        (5, 1.5),
        (0, 1.5),
    ),
}

# Length along the extrusion axis (Z).
LENGTHS = {
    "standard": 9.5,
    "hammer": 6.0,
}

# Through-bore diameter. Threads aren't modeled — the bore is the nominal
# thread Ø (cosmetic; the real part is tapped).
BORE = {
    "M3": 3.0,
    "M4": 4.0,
    "M5": 5.0,
}

# Y of the feature that catches the slot lip from below — the mating point
# with an extrusion slot LinearJoint. Standard: wing top. Hammer: plate top.
ENGAGEMENT_Y = {
    "standard": 3.3,
    "hammer": 1.5,
}

# Hammer t-nut boss — 6×6 mm pad extruded 3 mm above the plate top
# (see `_build` below). Exposed so callers placing the nut by hand can
# reason about the total bore-axis height without re-encoding magics.
HAMMER_BOSS_HEIGHT = 3
HAMMER_TOTAL_HEIGHT = HALF_PROFILES["hammer"][-1][1] + HAMMER_BOSS_HEIGHT  # 4.5


# ── Geometry ──────────────────────────────────────────────────────────────────
class TNut(BaseStandardPart):
    """2020 T-slot nut — standard (slide-in) or hammer (drop-and-twist).

    Cross-section in XY (x across slot, y depth from cavity floor at y = 0
    to slot mouth at y = 4.5) and runs along Z from 0 to length. The smooth
    through-bore is on the Y axis at x = 0, z = length / 2 — a screw enters
    from outside the extrusion along −Y."""

    def __init__(self, kind: str, size: str = "M5", qty: int = 1):
        super().__init__(qty=qty)
        if kind not in LENGTHS:
            raise ValueError(
                f"T-nut kind must be one of {sorted(LENGTHS)}; got {kind!r}"
            )
        if size not in BORE:
            raise ValueError(
                f"T-nut bore size must be one of {sorted(BORE)}; got {size!r}"
            )
        self.kind = kind
        self.size = size
        self.length = LENGTHS[kind]
        self.bore = BORE[size]
        self.half_profile = HALF_PROFILES[kind]
        self.engagement_y = ENGAGEMENT_Y[kind]

    def name_suffix(self) -> str:
        return f"_{self.kind}_{self.size}_x{self.qty}"

    def bom_key(self):
        return ("TNut", self.kind, self.size)

    def _build(self):
        height = self.half_profile[-1][1]
        top_y = height

        with BuildPart() as p:
            with BuildSketch():
                with BuildLine():
                    Polyline(*self.half_profile, close=True)
                make_face()
                mirror(about=Plane.YZ)
            extrude(amount=self.length)

            if self.kind == "hammer":
                mid_z = self.length / 2

                def diag(edges):
                    return [
                        e for e in edges if (e.center().X > 0) == (e.center().Z > mid_z)
                    ]

                # Fillet the (+X,+Z) / (-X,-Z) plate corners. Gives the
                # nut a preferred tilt direction so it drops into the
                # slot mouth without binding on opposite corners.
                fillet(diag(p.edges().filter_by(Axis.Y)), radius=2 * MM)

                # 6 mm square boss centered on the top face. Lifts the
                # threaded bore above the slot lip so a screw clamps work
                # against the boss rather than the thin 1.5 mm plate.
                pad_plane = Plane(
                    origin=(0, height, mid_z),
                    x_dir=(1, 0, 0),
                    z_dir=(0, 1, 0),
                )
                with BuildSketch(pad_plane):
                    Rectangle(6 * MM, 6 * MM)
                extrude(amount=HAMMER_BOSS_HEIGHT * MM)
                top_y = height + HAMMER_BOSS_HEIGHT

                # Fillet the same diagonal on the boss — the unbroken
                # curved diagonal from plate to boss reads as a visual
                # key for the tilt-insert orientation.
                boss_edges = [
                    e for e in p.edges().filter_by(Axis.Y) if e.center().Y > height
                ]
                fillet(diag(boss_edges), radius=2 * MM)

                # Friction grooves running across the wing tops in X
                # (y = 1.5, outside the boss footprint). The slot lip
                # bites into these as the nut tilts into place.
                groove_w = 0.3
                groove_d = 0.3
                wing_plane = Plane(
                    origin=(0, height, mid_z),
                    x_dir=(1, 0, 0),
                    z_dir=(0, -1, 0),
                )
                with BuildSketch(wing_plane):
                    z_offsets = (-2, -1, 0, 1, 2)
                    locs = [(x, z) for x in (-4, 4) for z in z_offsets]
                    with Locations(*locs):
                        Rectangle(2 * MM, groove_w * MM)
                extrude(amount=groove_d * MM, mode=Mode.SUBTRACT)

            # Bore along +Y, centered on (x=0, z=length/2). Sketch plane is
            # shifted into −Y by 0.5 mm so the extrude overshoots both faces
            # for a clean through-cut.
            bore_plane = Plane(
                origin=(0, -0.5 * MM, self.length / 2),
                x_dir=(1, 0, 0),
                z_dir=(0, 1, 0),
            )
            with BuildSketch(bore_plane):
                Circle(self.bore / 2)
            extrude(amount=top_y + 1 * MM, mode=Mode.SUBTRACT)

            # Mating joint: top of the engagement feature, centered along
            # the slide axis. Connects to an extrusion slot LinearJoint
            # whose slider frame is identity-oriented (the only frame
            # build123d's Axis derives from a +Z slide direction). The +90°
            # Z rotation orients the part so that TNut +Y (bore axis) ends
            # up along extrusion +X — i.e. mounted in a ±X face slot, with
            # the screw entering from outside the 20 mm-wide face.
            RigidJoint(
                "slot_mount",
                joint_location=Location(
                    (0, self.engagement_y, self.length / 2),
                    (0, 0, 90),
                ),
            )

        p.part.label = f"TNut_{self.kind}_{self.size}"
        return p.part


if __name__ == "__main__":
    TNut("standard", "M5").export()
    TNut("hammer", "M5").export()
