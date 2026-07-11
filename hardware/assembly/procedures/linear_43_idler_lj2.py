"""Left-joint idler (LJ2) on the assembled X stage — extends
linear_41_idler_lj1 by mounting one linear_42_idler_lj2 bundle on
the LEFT XY joint's second M4 hole (extra_hole2, near the slant
cutout), plus an M4 square nut captured in the SLANT ("side") pocket
to receive the bundle's shoulder-bolt thread.

The bundle's install face lands on the joint top, centered on
extra_hole2; the shoulder rod sits OUTBOARD along world -Y (same
orientation as the LJ1 bundle). The M4 thread passes down through
extra_hole2 and engages the nut in the slant pocket. The slant
pocket opens on the joint's slant face, so the nut slides in along
the slant face outward normal — a world XZ-diagonal direction
specific to the LEFT joint.

Placement (read from the LI20Joint feature hooks forwarded through
LI31X / LI33X / LI41IdlerLj1 — no joint math repeated here):
  * Bundle origin = joint extra_hole2 world center. Bundle native +Z
    → world -Y (outboard).
  * Nut center XZ = bolt XZ (so the bolt thread passes through the
    bore center); nut center Y = slant_pocket Y - thickness/2 (so the
    nut centers in the pocket along the bore axis). Nut native +X
    aligns with the slant edge in world; nut native +Z (bore +
    chamfer) → world +Y, so the FLAT face lands outboard (world -Y,
    camera-facing) and the chamfered face faces inboard, matching
    the LJ1 nut convention.

Two variants:
  * exploded — bundle lifted along world -Y by BUNDLE_EXPLODE; nut
               shifted along the slant face outward normal by
               NUT_EXPLODE (the direction it slides into the pocket
               from). The bundle itself is shown assembled — its own
               fastener-explode reading belongs to linear_42_idler_lj2.
  * assembled — bundle install face flush on joint top, nut seated in
                the slant pocket, shoulder thread engaging the nut.

The base (linear_41_idler_lj1) is always shown assembled.

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.linear_43_idler_lj2
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import BaseAssembly
from hardware.assembly.procedures.linear_41_idler_lj1 import LI41IdlerLj1
from hardware.assembly.procedures.linear_42_idler_lj2 import LI42IdlerLj2
from hardware.assembly.projection import Camera, MAIN_FRAME_VIEW
from hardware.parts.standard.nut import SPECS as NUT_SPECS, Nut

BUNDLE_EXPLODE = 35  # mm — exploded: bundle lifted along world -Y (outboard)
NUT_EXPLODE = 25  # mm — exploded: nut slid along +slant_z_dir (pocket opening)


class LI43IdlerLj2(BaseAssembly):
    camera = [MAIN_FRAME_VIEW, Camera(49.74, -63.86, 54.41)]

    def _build(self) -> Compound:
        base = LI41IdlerLj1(exploded=False)
        base_compound = base.build()

        joint = base.joint_base
        self.joint_base = joint  # forwarded for downstream idler procedures.
        bundle_origin = joint.extra_hole2_world_centers[
            0
        ]  # LEFT joint top, on extra_hole2
        pocket_center = joint.slant_pocket_world_centers[0]  # LEFT joint slant pocket
        slant_x_w = joint.slant_x_dir_worlds[0]  # slant edge dir in world
        slant_z_w = joint.slant_z_dir_worlds[0]  # slant face normal in world
        nut_thickness = NUT_SPECS["square"]["M4"]["thickness"]

        # Bundle: install face (native z=0) on joint top at extra_hole2;
        # shoulder/idler stack along world -Y (outboard) — same
        # orientation as the LJ1 bundle.
        bundle_y = (
            bundle_origin[1] - BUNDLE_EXPLODE if self.exploded else bundle_origin[1]
        )
        bundle = LI42IdlerLj2(exploded=False).build()
        bundle.move(
            Location(
                Plane(
                    origin=(bundle_origin[0], bundle_y, bundle_origin[2]),
                    x_dir=(1, 0, 0),
                    z_dir=(0, -1, 0),  # bundle native +Z (shoulder-up) → world -Y
                )
            )
        )

        # M4 square nut in the slant pocket. Nut native frame: bore on
        # +Z, bottom face (z=0) is the full square, top face
        # (z=thickness) is chamfered. Placement plane z_dir = +Y maps
        # nut native +Z → world +Y, so chamfer lands on world +Y
        # (inboard) and the FLAT face lands on world -Y (outboard /
        # camera-facing). x_dir = slant_x_dir_world rotates the nut
        # about its bore so the square sides align with the slant edge
        # — derived y_dir then points along -slant_z_dir (into the
        # pocket depth), so the nut fits inside the rotated pocket.
        # Origin XZ = bolt XZ (bolt thread through bore center); origin
        # Y = pocket Y center - thickness/2 (nut centered in pocket
        # along the bore axis).
        nut_origin = [
            bundle_origin[0],
            pocket_center[1] - nut_thickness / 2,
            bundle_origin[2],
        ]
        if self.exploded:
            nut_origin[0] += NUT_EXPLODE * slant_z_w[0]
            nut_origin[1] += NUT_EXPLODE * slant_z_w[1]
            nut_origin[2] += NUT_EXPLODE * slant_z_w[2]

        nut = Nut("square", "M4").build()
        nut.move(
            Location(
                Plane(
                    origin=tuple(nut_origin),
                    x_dir=slant_x_w,
                    z_dir=(0, 1, 0),
                )
            )
        )

        return Compound(
            label="linear_43_idler_lj2",
            children=[
                base_compound,
                bundle,
                nut,
            ],
        )


if __name__ == "__main__":
    for exploded in (True, False):
        asm = LI43IdlerLj2(exploded=exploded)
        asm.export()
        asm.render()
