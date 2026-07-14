"""Motor + bracket on the frame (right motor) — extends motor_11_frame
(frame + 4 corner idler mounts + their brackets + LEFT motor) by
adding motor_20_bracket (NEMA 17 motor without the 180° Z flip) onto
the RIGHT TWO standard M5 T-nuts on short_top.

Mirror of motor_11_frame across the world YZ plane: the two M5 BHCS
land 25 mm apart along world X, but on the right-side pair of
short_top t-nuts (at world x = SHORT_LENGTH/2 - SHORT_TOP_END_GAP and
SHORT_LENGTH/2 - SHORT_TOP_END_GAP - SHORT_TOP_INNER_GAP). The
bracket's frame-mount orientation is identical to motor_11_frame's —
only the origin's world X flips sign. The bracket sub-assembly is
swapped from MO10Bracket to MO20Bracket so the cable connector lands
on world +X (RIGHT side from the top view, outward from the frame
centre) rather than world -X.

Placement:

  * Base layer — MO11Frame(exploded=False) at world origin. Bundles
    the full frame + 4 corner idler mounts + their brackets + the
    left motor.
  * Motor bracket — MO20Bracket(exploded=False) translated so its two
    M5 BHCS shanks land at the RIGHT TWO short_top t-nuts, with the
    bracket bottom one ring-height in front of the slot face. Same
    orientation as motor_11_frame (x_dir=(0,0,+1), z_dir=(0,-1,0)),
    so the motor body extends INSIDE the frame rectangle.

Two variants:
  * exploded — motor assembly pulled outboard along world -Y by
               MOTOR_EXPLODE so the BHCS install path reads clearly.
  * assembled — rings flush against the slot face.

The base MO11Frame layer is always shown assembled — it's the prior
step's finished state that this composition builds onto.

Run from the repo root:

    uv run --group cad python -m hardware step motor_21_frame
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import ASSEMBLED_CAM0, BaseAssembly, EXPLODED_CAM1
from hardware.assembly.procedures.frame_10_extrusion_tnut import (
    EXT_THICKNESS,
    LONG_LENGTH,
    SHORT_LENGTH,
    SHORT_TOP_END_GAP,
    SHORT_TOP_INNER_GAP,
)
from hardware.assembly.procedures.motor_11_frame import MO11Frame
from hardware.assembly.procedures.motor_20_bracket import MO20Bracket
from hardware.assembly.projection import MAIN_FRAME_VIEW, Camera
from hardware.parts.standard.extrusion import cb_end_offset

MOTOR_EXPLODE = 40  # mm — exploded: outboard air gap, slot face → bracket bottom


class MO21Frame(BaseAssembly):
    camera = [MAIN_FRAME_VIEW, Camera(-41.17, 59.81, -134.40)]
    views = [EXPLODED_CAM1, ASSEMBLED_CAM0]

    def _build(self) -> Compound:
        # Base layer — always assembled. Bundles frame + all 4 idler
        # corner mounts + their brackets + left motor; this step adds
        # the right motor + bracket on top. Keep the base instance so
        # downstream consumers (motor_30_pulley) can read its
        # ``pulley_plane`` for the left motor.
        self.base = MO11Frame(exploded=False)
        base_compound = self.base.build()

        # Target t-nuts: short_top's RIGHT TWO standard M5 t-nuts.
        # Mirror of motor_11_frame's left pair across the world YZ
        # plane.
        right_t1_world_x = SHORT_LENGTH / 2 - SHORT_TOP_END_GAP - SHORT_TOP_INNER_GAP
        right_t2_world_x = SHORT_LENGTH / 2 - SHORT_TOP_END_GAP
        short_top_world_z = LONG_LENGTH - cb_end_offset

        # See motor_11_frame for the hook-based placement derivation —
        # the math is mirror-identical, only origin_x flips sign.
        bracket_asm = MO20Bracket(exploded=False)
        motor_compound = bracket_asm.build()

        origin_x = (right_t1_world_x + right_t2_world_x) / 2
        origin_y = (
            -EXT_THICKNESS - bracket_asm.stack_height + bracket_asm.bracket_bottom_z
        )
        if self.exploded:
            origin_y -= MOTOR_EXPLODE
        origin_z = short_top_world_z - bracket_asm.m5_native_x

        motor_compound.move(
            Location(
                Plane(
                    origin=(origin_x, origin_y, origin_z),
                    x_dir=(
                        0,
                        0,
                        +1,
                    ),  # native +X (bracket length) → world +Z (into frame)
                    z_dir=(0, -1, 0),  # native +Z (bracket / head side) → world -Y
                )
            )
        )  # → native +Y (M5 pair) → world -X (along short_top)

        # Mirror of MO11Frame.pulley_plane for the right motor.
        # MO20Bracket leaves the motor un-rotated so the D-flat is on
        # the motor's native +Y → world -X here; pick x_dir so the
        # pulley's native +Y set-screw points world -X.
        self.pulley_plane = Plane(
            origin=(origin_x, origin_y - bracket_asm.bracket_top_z, origin_z),
            x_dir=(0, 0, +1),
            z_dir=(0, -1, 0),
        )

        return Compound(
            label="motor_21_frame",
            children=[
                base_compound,
                motor_compound,
            ],
        )
