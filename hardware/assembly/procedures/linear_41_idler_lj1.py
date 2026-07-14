"""Left-joint idler (LJ1) on the assembled X stage — extends
linear_33_x by mounting one linear_40_idler_lj1 bundle on the LEFT
XY joint's extra_hole (with the front pocket beneath it), plus an
M4 square nut captured in that pocket to receive the bundle's
shoulder-bolt thread.

The bundle's install face (stack bottom) lands on the joint top
face, centered on extra_hole; the shoulder rod sits OUTBOARD along
world -Y; the M4 thread passes down through the joint's M4 clearance
hole and engages the captured nut. The pocket opens at the joint's
native -Y face which, after the joint placement, faces world +Z —
so the nut is dropped in from above.

Placement (read from the LI20Joint feature hooks forwarded through
LI31X / LI33X — no joint math repeated here):
  * Bundle origin = joint extra_hole world center (the joint-top
    point where the bundle's install face sits). Bundle native +Z →
    world -Y (outboard).
  * Nut center = joint front-pocket world center. Nut native +Z
    (bore + chamfer) → world +Y, so the FLAT full-square face lands
    outboard (world -Y, the camera-facing side of the joint) and the
    chamfered face faces inboard. Origin Y is shifted by half the
    nut thickness so the nut centers in the pocket along world Y.

Two variants:
  * exploded — bundle lifted along world -Y by BUNDLE_EXPLODE (the
               install direction the bundle came from); nut lifted
               along world +Z by NUT_EXPLODE (the pocket-opening
               direction the nut slides through). The bundle itself
               is shown assembled — its own fastener-explode reading
               belongs to linear_40_idler_lj1.
  * assembled — bundle install face flush on joint top, nut seated
                in the pocket, shoulder thread engaging the nut.

The base (linear_33_x) is always shown assembled.

Run from the repo root:

    uv run --group cad python -m hardware step linear_41_idler_lj1
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import ASSEMBLED_CAM0, BaseAssembly, EXPLODED_CAM1
from hardware.assembly.procedures.linear_33_x import LI33X
from hardware.assembly.procedures.linear_40_idler_lj1 import LI40IdlerLj1
from hardware.assembly.projection import Camera, MAIN_FRAME_VIEW
from hardware.parts.standard.nut import SPECS as NUT_SPECS, Nut

BUNDLE_EXPLODE = 35  # mm — exploded: bundle lifted along world -Y (outboard)
NUT_EXPLODE = 25  # mm — exploded: nut lifted along world +Z (pocket opening)


class LI41IdlerLj1(BaseAssembly):
    camera = [MAIN_FRAME_VIEW, Camera(-67.41, 31.24, -99.63)]
    views = [EXPLODED_CAM1, ASSEMBLED_CAM0]

    def _build(self) -> Compound:
        base = LI33X(exploded=False)
        base_compound = base.build()

        joint = base.joint_base
        self.joint_base = joint  # forwarded for downstream idler procedures.
        bundle_origin = joint.extra_hole_world_centers[
            0
        ]  # LEFT joint top, on extra_hole
        pocket_center = joint.front_pocket_world_centers[0]  # LEFT joint front pocket
        nut_thickness = NUT_SPECS["square"]["M4"]["thickness"]

        # Bundle: install face (native z=0) on joint top at extra_hole;
        # shoulder/idler stack along world -Y (outboard).
        bundle_y = (
            bundle_origin[1] - BUNDLE_EXPLODE if self.exploded else bundle_origin[1]
        )
        bundle = LI40IdlerLj1(exploded=False).build()
        bundle.move(
            Location(
                Plane(
                    origin=(bundle_origin[0], bundle_y, bundle_origin[2]),
                    x_dir=(1, 0, 0),
                    z_dir=(0, -1, 0),  # bundle native +Z (shoulder-up) → world -Y
                )
            )
        )

        # M4 square nut in the front pocket. Nut native frame: bore on
        # +Z, bottom face (z=0) is the full square, top face
        # (z=thickness) is chamfered. Placement plane z_dir = +Y maps
        # nut native +Z → world +Y, so the chamfer lands on the world
        # +Y (inboard) side and the FLAT full-square face lands on the
        # world -Y (outboard / camera-facing) side — flat side up from
        # the user's view of the joint. The plane origin sits on the
        # nut's bottom face (flat); shift -thickness/2 along world +Y
        # so the nut centers in the pocket along the bore axis.
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
            label="linear_41_idler_lj1",
            children=[
                base_compound,
                bundle,
                nut,
            ],
        )
