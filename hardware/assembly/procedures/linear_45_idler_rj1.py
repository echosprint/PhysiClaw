"""Right-joint idler (RJ1) on the assembled X stage — extends
linear_43_idler_lj2 by mounting one linear_44_idler_rj1 bundle on
the RIGHT XY joint's extra_hole (with the front pocket beneath it),
plus an M4 square nut captured in that pocket to receive the
bundle's shoulder-bolt thread.

This mirrors linear_41_idler_lj1 to the RIGHT joint: same hole +
pocket pair (extra_hole + front_pocket), same outboard install
direction (world -Y), same nut convention (flat face outboard /
camera-facing, chamfered face inboard). On the right joint,
extra_hole lands at the +X-side of the joint — toward the right Y
extrusion. The bundle itself differs (linear_44_idler_rj1 = smooth
idler, no spacer, 10 mm shoulder), but the joint-mount geometry is
identical to LJ1.

Placement (read from the LI20Joint feature hooks forwarded through
LI31X / LI33X / LI41IdlerLj1 / LI43IdlerLj2 — no joint math repeated
here):
  * Bundle origin = RIGHT joint extra_hole world center. Bundle native
    +Z → world -Y (outboard).
  * Nut center = RIGHT joint front_pocket world center. Nut native +Z
    (bore + chamfer) → world +Y. Origin Y shifted by -thickness/2 so
    the nut centers in the pocket along the bore axis.

Two variants:
  * exploded — bundle lifted along world -Y by BUNDLE_EXPLODE; nut
               lifted along world +Z by NUT_EXPLODE (the pocket
               opening direction). The bundle itself is shown
               assembled — its own fastener-explode reading belongs
               to linear_44_idler_rj1.
  * assembled — bundle install face flush on joint top, nut seated in
                the pocket, shoulder thread engaging the nut.

The base (linear_43_idler_lj2) is always shown assembled.

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.linear_45_idler_rj1
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import BaseAssembly
from hardware.assembly.procedures.linear_43_idler_lj2 import LI43IdlerLj2
from hardware.assembly.procedures.linear_44_idler_rj1 import LI44IdlerRj1
from hardware.assembly.projection import Camera, MAIN_FRAME_VIEW
from hardware.parts.standard.nut import SPECS as NUT_SPECS, Nut

BUNDLE_EXPLODE = 35  # mm — exploded: bundle lifted along world -Y (outboard)
NUT_EXPLODE = 25  # mm — exploded: nut lifted along world +Z (pocket opening)


class LI45IdlerRj1(BaseAssembly):
    camera = [MAIN_FRAME_VIEW, Camera(68.97, 36.43, 106.12)]

    def _build(self) -> Compound:
        base = LI43IdlerLj2(exploded=False)
        base_compound = base.build()

        joint = base.joint_base
        self.joint_base = joint  # forwarded for downstream idler procedures.
        bundle_origin = joint.extra_hole_world_centers[
            1
        ]  # RIGHT joint top, on extra_hole
        pocket_center = joint.front_pocket_world_centers[1]  # RIGHT joint front pocket
        nut_thickness = NUT_SPECS["square"]["M4"]["thickness"]

        # Bundle: install face (native z=0) on joint top at extra_hole;
        # shoulder/idler stack along world -Y (outboard).
        bundle_y = (
            bundle_origin[1] - BUNDLE_EXPLODE if self.exploded else bundle_origin[1]
        )
        bundle = LI44IdlerRj1(exploded=False).build()
        bundle.move(
            Location(
                Plane(
                    origin=(bundle_origin[0], bundle_y, bundle_origin[2]),
                    x_dir=(1, 0, 0),
                    z_dir=(0, -1, 0),  # bundle native +Z (shoulder-up) → world -Y
                )
            )
        )

        # M4 square nut in the front pocket — same orientation as the
        # LJ1 nut (see linear_41_idler_lj1 for the full read of the
        # nut frame). Origin sits on the nut's flat (full-square) face;
        # shift -thickness/2 along world +Y so the nut centers in the
        # pocket along the bore axis.
        nut_z = pocket_center[2] + NUT_EXPLODE if self.exploded else pocket_center[2]
        nut = Nut("square", "M4").build()
        nut.move(
            Location(
                Plane(
                    origin=(
                        pocket_center[0],
                        pocket_center[1] - nut_thickness / 2,
                        nut_z,
                    ),
                    x_dir=(1, 0, 0),
                    z_dir=(0, 1, 0),
                )
            )
        )

        return Compound(
            label="linear_45_idler_rj1",
            children=[
                base_compound,
                bundle,
                nut,
            ],
        )


if __name__ == "__main__":
    for exploded in (True, False):
        asm = LI45IdlerRj1(exploded=exploded)
        asm.export()
        asm.render()
