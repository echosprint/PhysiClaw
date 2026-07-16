"""Belt clamp on the X-rail slider — extends belt_10_motor_a by mounting
one BeltClamp part to the X-rail's MGN9H slider via four M3 × 8 BHCS
driven through the clamp's four mount holes (which form a 16 × 15 mm
grid matching the slider's M3 mount pattern).

Placement:
  * BeltClamp — oriented with native +X (length axis, also the mirror
    axis) along world +X (along the X-rail), native +Z (top face, where
    the screw heads bottom in the stadium-slot floors) outboard toward
    world -Y. The clamp's hole-grid center sits on the slider's M3
    mount-grid center; the bottom face lands flush on the slider top.
  * 4 × BHCS M3 × 8 — one per slot round-end. Underhead seats on the
    stadium-slot floor (slot_depth below the clamp top face); shank
    threads world +Y direction into the slider's M3 mount hole.

World position of the slider mount-grid center is derived from the
upstream assembly chain (no hardcoded numbers):
  * SLIDER_MOUNT_Y = -EXT_THICKNESS - 2·slider_top_z
    (slot face at -EXT_THICKNESS, joint bottom 10 mm inboard of that,
    1020 beam slot face there too, X-rail bottom on it, slider top
    one block-height further outboard).
  * SLIDER_MOUNT_Z = beam_center_world_z = LEFT joint big_csk world Z,
    available from the LI11Y / LI20Joint / LI31X chain.
  * SLIDER_MOUNT_X = 0 (LI32X has slider_position = 0.5 along a
    130 mm rail centered on world X = 0).

Two variants:
  * exploded — clamp pulled CLAMP_EXPLODE outboard along world -Y so
               the bracket-to-slider interface reads clearly; each
               BHCS shank-tip floats SCREW_EXPLODE further along the
               same axis (shared shank-tip line, motor_10_bracket
               convention).
  * assembled — clamp bottom flush on slider top, BHCS heads bottomed
                in the stadium-slot floors.

The base (belt_10_motor_a) is always shown assembled — the X-rail's
slider and the belt are already in place from that step.

Run from the repo root:

    uv run --group cad python -m hardware step belt_20_clamp
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import ASSEMBLED_CAM0, EXPLODED_CAM1, BaseAssembly
from hardware.assembly.procedures.belt_10_motor_a import BE10MotorA
from hardware.assembly.procedures.frame_10_extrusion_tnut import EXT_THICKNESS
from hardware.assembly.projection import MAIN_FRAME_VIEW, Camera
from hardware.parts.custom.belt_clamp import (
    BeltClamp,
    slot1_center_y,
    slot2_center_y,
    slot_depth,
)
from hardware.parts.custom.belt_clamp import (
    length as clamp_length,
)
from hardware.parts.custom.belt_clamp import (
    thickness as clamp_thickness,
)
from hardware.parts.standard.mgn9h import (
    block_top_z as slider_top_z,
)
from hardware.parts.standard.mgn9h import (
    mount_pitch_x,
    mount_pitch_y,
)
from hardware.parts.standard.screw import Screw

BHCS_LENGTH = 8  # mm — M3 BHCS underhead length
CLAMP_EXPLODE = 30  # mm — exploded: clamp pulled outboard along world -Y
SCREW_EXPLODE = 30  # mm — exploded: each BHCS shank tip floats this far
#      further outboard than the clamp top face, so
#      the screws read as separate parts that drop
#      into the stadium-slot floors

# ── Slider mount-grid center (X-rail's MGN9H slider top face) ─────────────────
# Chain: 1020 beam slot face at world Y = joint_bottom = -EXT - slider_top_z
# (because the LEFT joint sits on the Y carriage with bottom at slider top,
# which is one slider_top_z outboard of the 1020 slot face... — see LI31X /
# LI11Y for the full derivation). The X-rail bottom flushes on the 1020
# slot face, and its slider extends one more slider_top_z outboard.
# Net: SLIDER_MOUNT_Y = -EXT_THICKNESS - 2 * slider_top_z.
SLIDER_MOUNT_X = 0
SLIDER_MOUNT_Y = -EXT_THICKNESS - 2 * slider_top_z

# rail_center_z = beam_center_world_z = LEFT-joint big_csk world Z. Walk
# the chain (Y-rail center Z → Y-carriage slider Z → LEFT joint origin Z
# → big_csk world Z) so this file stays in sync with the upstream
# placement math without hardcoded numbers.
from hardware.assembly.procedures.frame_10_extrusion_tnut import (
    LONG_BOT_GAP,
    LONG_LENGTH,
    LONG_TOP_GAP,
)
from hardware.assembly.procedures.linear_10_y import RAIL_LENGTH
from hardware.parts.custom.idler_mount_front import (
    slot_center_y as ld_slot_center_y,
)
from hardware.parts.custom.idler_mount_front import (
    width as ld_block_width,
)
from hardware.parts.custom.idler_mount_motor import length as lu_block_length
from hardware.parts.custom.xy_joint_left import (
    big_csk_y as joint_big_csk_y_n,
)
from hardware.parts.custom.xy_joint_left import (
    csk_hole_from_bottom as joint_csk_hole_from_bottom,
)
from hardware.parts.custom.xy_joint_left import (
    csk_y_spacing as joint_csk_y_spacing,
)
from hardware.parts.custom.xy_joint_left import (
    width as joint_width,
)
from hardware.parts.standard.mgn9h import slider_position

_y_rail_center_z = (
    (LONG_LENGTH - LONG_TOP_GAP)
    - lu_block_length / 2
    + (LONG_BOT_GAP - ld_slot_center_y)
    + ld_block_width / 2
) / 2
_y_carriage_slider_z = _y_rail_center_z + (slider_position - 0.5) * RAIL_LENGTH
_joint_grid_y = -joint_width / 2 + joint_csk_hole_from_bottom + joint_csk_y_spacing / 2
_left_joint_origin_z = _y_carriage_slider_z + _joint_grid_y
SLIDER_MOUNT_Z = _left_joint_origin_z - joint_big_csk_y_n


class BE20Clamp(BaseAssembly):
    camera = [MAIN_FRAME_VIEW, Camera(-9.68, -51.45, -8.14)]
    views = [EXPLODED_CAM1, ASSEMBLED_CAM0]

    def _build(self) -> Compound:
        base_compound = BE10MotorA(exploded=False).build()

        # Belt clamp's M3-hole grid center in its native frame:
        # X = clamp_length / 2 (mirror axis); Y = midpoint of slot1 / slot2.
        clamp_grid_x = clamp_length / 2
        clamp_grid_y = (slot1_center_y + slot2_center_y) / 2

        # Origin: place the clamp's hole-grid center on the slider's mount
        # center, with the bottom face (native -Z, which maps to world +Y
        # under z_dir=(0,-1,0)) flush on the slider top at world Y =
        # SLIDER_MOUNT_Y. Origin Y must therefore be one half-thickness
        # OUTBOARD of the slider top.
        clamp_origin_x = SLIDER_MOUNT_X - clamp_grid_x
        clamp_origin_y = SLIDER_MOUNT_Y - clamp_thickness / 2
        clamp_origin_z = SLIDER_MOUNT_Z - clamp_grid_y
        if self.exploded:
            clamp_origin_y -= CLAMP_EXPLODE

        clamp = BeltClamp().build()
        clamp.move(
            Location(
                Plane(
                    origin=(clamp_origin_x, clamp_origin_y, clamp_origin_z),
                    x_dir=(1, 0, 0),  # native +X (length / mirror axis) → world +X
                    z_dir=(0, -1, 0),  # native +Z (top face)             → world -Y
                )
            )
        )  # → derived native +Y → world +Z

        # 4 × BHCS M3 × 8 — one per stadium-slot round end. Underhead
        # bottoms in the slot floor at world Y = SLIDER_MOUNT_Y -
        # (clamp_thickness - slot_depth); shank threads inboard (world
        # +Y) into the slider's M3 mount hole.
        slot_floor_y = SLIDER_MOUNT_Y - (clamp_thickness - slot_depth)
        if self.exploded:
            screw_underhead_y = slot_floor_y - CLAMP_EXPLODE - SCREW_EXPLODE
        else:
            screw_underhead_y = slot_floor_y

        screws = []
        for hole_dx in (-mount_pitch_x / 2, +mount_pitch_x / 2):
            for hole_dz in (-mount_pitch_y / 2, +mount_pitch_y / 2):
                screw = Screw("BHCS", "M3", BHCS_LENGTH).build()
                screw.move(
                    Location(
                        Plane(
                            origin=(
                                SLIDER_MOUNT_X + hole_dx,
                                screw_underhead_y,
                                SLIDER_MOUNT_Z + hole_dz,
                            ),
                            x_dir=(1, 0, 0),
                            z_dir=(0, -1, 0),  # native +Z (head) → world -Y
                        )
                    )
                )
                screws.append(screw)

        return Compound(
            label="belt_20_clamp",
            children=[
                base_compound,
                clamp,
                *screws,
            ],
        )
