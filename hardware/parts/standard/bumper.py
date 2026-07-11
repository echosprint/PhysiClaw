from build123d import *

from hardware.parts.base import BaseStandardPart

# ── Parameters ────────────────────────────────────────────────────────────────
body_diameter = 20 * MM
body_height = 22 * MM

# Counterbore pocket on the bottom face (recess for a screw head).
cbore_diameter = 14 * MM
cbore_depth = 16 * MM

# Through-hole along the cylinder axis (Z).
hole_diameter = 5.3 * MM

# Fillet radius on the bottom face edges.
bottom_fillet = 0.5 * MM


# ── Geometry ──────────────────────────────────────────────────────────────────
class Bumper(BaseStandardPart):
    def bom_key(self):
        return ("Bumper", body_diameter, body_diameter, body_height)

    def bom_display(self):
        return f"Bumper {body_diameter:g}x{body_diameter:g}x{body_height:g}"

    def _build(self):
        with BuildPart() as p:
            Cylinder(radius=body_diameter / 2, height=body_height)
            bot_z = -body_height / 2
            # Counterbore pocket cut from the bottom face upward.
            with Locations((0, 0, bot_z + cbore_depth / 2)):
                Cylinder(
                    radius=cbore_diameter / 2,
                    height=cbore_depth,
                    mode=Mode.SUBTRACT,
                )
            # Through-hole along Z (1 mm overshoot for clean cut).
            Cylinder(
                radius=hole_diameter / 2,
                height=body_height + 1 * MM,
                mode=Mode.SUBTRACT,
            )
            # Fillet the bottom face edges (outer body edge + cbore opening).
            bottom_face = p.faces().sort_by(Axis.Z)[0]
            fillet(bottom_face.edges(), radius=bottom_fillet)
        return p.part


if __name__ == "__main__":
    Bumper().export()
