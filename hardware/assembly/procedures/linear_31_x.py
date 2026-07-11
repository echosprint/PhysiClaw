"""Linear X crossbeam on the frame — extends linear_20_joint by
mounting linear_30_x (an Extrusion1020 sub-assembly with two
standard M5 T-nuts pre-loaded in its slot) between the two XY
joints, then driving an M5 × 12 FHCS through each joint's big CSK
hole into the corresponding pre-loaded t-nut.

The t-nut positions are computed inside linear_30_x so that when
the sub-assembly is placed here (centered on world X = 0), each
t-nut's bore lands exactly at the joint's big-CSK world X.

Placement:
  * linear_30_x sub-assembly: centered on world X = 0, slot face
    flush against the joints' bottom faces at world Y = -30. Native
    +X (cross-section) → world +Z; native +Y (slot face) → world
    -Y; native +Z (length) → world +X.
  * Each FHCS placed at its joint's big-CSK world position, head
    flush with the joint top face; shank passes through the joint
    into the t-nut.

Two variants:
  * exploded — separation along the actual install axis (world Y):
               the linear_30_x sub-assembly (1020 + its t-nuts)
               shifts deeper into the frame along world +Y by
               X_BEAM_EXPLODE; each FHCS lifts outward along world
               -Y by FHCS_EXPLODE. Reads as "1020 mates from one
               side of the joint, screws drop in from the other".
  * assembled — 1020 against joint bottoms, t-nuts inside its slot,
                FHCS heads flush with the joint top face.

The base (linear_20_joint) is always shown assembled.

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.linear_31_x
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import BaseAssembly
from hardware.assembly.procedures.linear_20_joint import LI20Joint
from hardware.assembly.procedures.linear_30_x import LI30X, X_BEAM_LENGTH
from hardware.assembly.projection import MAIN_FRAME_VIEW, Camera
from hardware.parts.custom.xy_joint_left import thickness as joint_thickness
from hardware.parts.standard.screw import FHCS_DIMS, Screw, head_skirt

FHCS_LENGTH = 12  # mm — M5 FHCS overall length
X_BEAM_EXPLODE = 35  # mm — exploded: 1020 sub-assembly shifts along world +Y
#       (deeper into the frame, away from the joint bottoms)
FHCS_EXPLODE = 35  # mm — exploded: each FHCS lifts along world -Y
#       (outward, away from the joint top faces)

# 1020 slot face Y in its native frame (from extrusion.half_vertices_1020).
BEAM_SLOT_FACE_Y_NATIVE = 9.9


class LI31X(BaseAssembly):
    camera = [MAIN_FRAME_VIEW, Camera(-1.95, -64.14, -1.85)]

    def _build(self) -> Compound:
        base = LI20Joint(exploded=False)
        base_compound = base.build()

        # The two joints' big-CSK world centers (joint mid-thickness).
        big_csk_centers = base.big_csk_world_centers
        joint_top_world_y = big_csk_centers[0][1] - joint_thickness / 2
        joint_bottom_world_y = big_csk_centers[0][1] + joint_thickness / 2

        # Hooks for downstream consumers (linear_33_x, linear_41_idler_lj1):
        #   * beam_slot_face_world_y — world Y of the 1020's slot face
        #     (where another rail / part mounts on the 1020).
        #   * beam_center_world_z — world Z of the 1020 cross-section
        #     centerline.
        #   * joint_base — the LI20Joint instance, forwarded so further
        #     downstream steps can read its joint-feature world centers
        #     (extra_hole_world_centers, front_pocket_world_centers, …).
        self.beam_slot_face_world_y = joint_bottom_world_y
        self.beam_center_world_z = big_csk_centers[0][2]
        self.joint_base = base

        # ── 1020 sub-assembly (1020 + 2 standard M5 t-nuts) ─────────
        # Slot face touches joint bottom; cross-section centered on
        # the big-CSK world Z. Exploded: shifts along world +Y
        # (deeper into frame, away from joint bottom — the install
        # direction the 1020 came from).
        beam_origin_y = joint_bottom_world_y + BEAM_SLOT_FACE_Y_NATIVE
        beam_origin_x = -X_BEAM_LENGTH / 2
        beam_origin_z = big_csk_centers[0][2]
        if self.exploded:
            beam_origin_y += X_BEAM_EXPLODE

        beam_sub = LI30X(exploded=False).build()
        beam_sub.move(
            Location(
                Plane(
                    origin=(beam_origin_x, beam_origin_y, beam_origin_z),
                    x_dir=(0, 0, +1),  # 1020 native +X → world +Z
                    z_dir=(+1, 0, 0),  # 1020 native +Z (length) → world +X
                )
            )
        )  # → native +Y (slot face) → world -Y

        # ── 1 FHCS per joint, head flush with joint top ─────────────
        # Exploded: each FHCS lifts along world -Y (outward, the
        # direction it would be inserted from).
        fhcs_head_height = FHCS_DIMS["M5"]["k"] + head_skirt
        fhcs_under_y = joint_top_world_y + fhcs_head_height
        if self.exploded:
            fhcs_under_y -= FHCS_EXPLODE

        screws = []
        for csk_x, _, csk_z in big_csk_centers:
            screw = Screw("FHCS", "M5", FHCS_LENGTH).build()
            screw.move(
                Location(
                    Plane(
                        origin=(csk_x, fhcs_under_y, csk_z),
                        x_dir=(1, 0, 0),
                        z_dir=(0, -1, 0),  # native +Z (head) → world -Y
                    )
                )
            )
            screws.append(screw)

        return Compound(
            label="linear_31_x",
            children=[
                base_compound,
                beam_sub,
                *screws,
            ],
        )


if __name__ == "__main__":
    for exploded in (True, False):
        asm = LI31X(exploded=exploded)
        asm.export()
        asm.render()
