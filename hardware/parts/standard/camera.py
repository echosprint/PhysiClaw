"""Camera — a webcam-style rounded bar (82 × 26 × 25 mm).

  * Bottom face (−Z): the lens — a short barrel with a recessed glass disc.
  * Top: heavily filleted so the whole upper shell reads as a soft, rounded,
    hand-friendly form rather than a hard box.
  * Front face (+Y): a 1/4"-20 female socket for mounting on a 1/4-20 stud
    (e.g. the gooseneck's male tip). Threads are representational (a smooth
    tapped-minor bore, like the gooseneck socket).

Canonical frame: long axis along X, centered in X/Y, bottom face at z = 0
(lens looking down −Z), top at z = height.

Run from the repo root:

    uv run --group cad python -m hardware.parts.standard.camera
"""

from build123d import *

from hardware.parts.base import BaseStandardPart

# ── Body ──────────────────────────────────────────────────────────────────────
length = 82 * MM  # X — the long axis
width = 26 * MM  # Y
height = 25 * MM  # Z — bottom (z = 0) is the lens, top (z = height) is rounded

# ── Top rounding (a single large radius blends the whole top into a dome) ──────
top_fillet = 12 * MM  # all four top edges → near half-round upper shell
corner_fillet = 6 * MM  # the four vertical corners → no sharp edges to hold

# ── Lens on the bottom face (centered, facing −Z) ──────────────────────────────
lens_barrel_d = 20 * MM  # barrel protruding below the bottom face
lens_barrel_h = 3 * MM
lens_glass_d = 16 * MM  # glass disc, recessed into the barrel mouth
lens_glass_t = 1.2 * MM
lens_recess = 0.4 * MM  # how far the glass sits up inside the barrel

# ── 1/4"-20 female mount socket, centered on the front (+Y) face ────────────────
mount_bore = 5.2 * MM  # tapped 1/4-20 minor Ø (threads not modelled)
mount_depth = 12 * MM  # drilled −Y into the body

COL_BODY = Color(0.06, 0.06, 0.07)  # dark plastic shell
COL_GLASS = Color(0.02, 0.04, 0.10)  # lens glass


class Camera(BaseStandardPart):
    """82 × 26 × 25 mm rounded-top camera with a bottom lens and a 1/4-20
    female mount socket on the front face."""

    def name_suffix(self) -> str:
        return f"_{length:g}x{width:g}x{height:g}_x{self.qty}"

    def bom_key(self):
        return ("Camera",)

    def _build(self):
        with BuildPart() as p:
            Box(length, width, height, align=(Align.CENTER, Align.CENTER, Align.MIN))

            # Lens barrel protruding below the bottom face, then a recess
            # bored back up into its mouth to seat the glass.
            Cylinder(
                lens_barrel_d / 2,
                lens_barrel_h,
                align=(Align.CENTER, Align.CENTER, Align.MAX),
            )
            with Locations((0, 0, -lens_barrel_h)):
                Cylinder(
                    lens_glass_d / 2,
                    lens_glass_t + lens_recess,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                    mode=Mode.SUBTRACT,
                )

            # 1/4-20 female socket drilled −Y into the centre of the front face.
            with Locations(Plane(origin=(0, width / 2, height / 2), z_dir=(0, 1, 0))):
                Cylinder(
                    mount_bore / 2,
                    mount_depth,
                    align=(Align.CENTER, Align.CENTER, Align.MAX),
                    mode=Mode.SUBTRACT,
                )

            # Round the top: one large fillet on all four top edges blends the
            # upper shell into a soft dome; a smaller one breaks the vertical
            # corners. (Holes/barrel are cylindrical, so the axis filters pick
            # only the straight box edges.)
            top_face = p.faces().sort_by(Axis.Z)[-1]
            fillet(top_face.edges(), radius=top_fillet)
            fillet(p.edges().filter_by(Axis.Z), radius=corner_fillet)

        body = p.part
        body.color = COL_BODY
        body.label = "body"

        # Lens glass disc seated in the barrel mouth, recessed by lens_recess.
        with BuildPart() as g:
            with Locations((0, 0, -lens_barrel_h + lens_recess)):
                Cylinder(
                    lens_glass_d / 2 - 0.3 * MM,
                    lens_glass_t,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                )
        glass = g.part
        glass.color = COL_GLASS
        glass.label = "lens_glass"

        return Compound(label="Camera", children=[body, glass])


if __name__ == "__main__":
    Camera().export()
