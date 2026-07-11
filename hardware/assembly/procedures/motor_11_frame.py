"""Motor + bracket on the frame (left motor) — extends idler_41_rd
(frame + 4 corner idler mounts + their brackets) by adding
motor_10_bracket (NEMA 17 motor with bracket plate + ring spacers +
4 M3 + 2 M5 BHCS) onto the LEFT TWO standard M5 T-nuts on short_top.

The two M5 BHCS span 25 mm along the bracket's native Y axis, which
matches the 25 mm spacing between short_top's outermost left t-nut
(at local Z = SHORT_TOP_END_GAP) and its inner-left neighbour
(at SHORT_TOP_END_GAP + SHORT_TOP_INNER_GAP) — so pinning the screws
to that pair fully determines the bracket's X position.

Orientation:
  * native +Z (bracket / screw-head side) → world -Y (in front of the
    frame, where the heads are accessible). The M5 shanks (native -Z)
    therefore point world +Y, into the slot and into the t-nut bores.
  * native +X (bracket extends out from the motor body) → world +Z
    (down into the frame interior), so the bracket's M5 holes — at
    native x = m5_native_x past the motor body — land on short_top
    one ring-height in front of its slot face, while the motor body
    itself drops INSIDE the frame rectangle (world z < short_top
    centerline).
  * derived native +Y (M5-pair direction) → world -X, so the two M5
    screws land 25 mm apart in world X along short_top's length.

Placement:

  * Base layer — ID41Rd(exploded=False) at world origin. Bundles the
    full frame + 4 corner idler mounts + their brackets.
  * Motor bracket — MO10Bracket(exploded=False) translated so its two
    M5 BHCS shanks land at the LEFT TWO short_top t-nuts (world x =
    -SHORT_LENGTH/2 + SHORT_TOP_END_GAP and
    -SHORT_LENGTH/2 + SHORT_TOP_END_GAP + SHORT_TOP_INNER_GAP, both at
    world z = LONG_LENGTH - cb_end_offset). The bracket bottom sits
    one spacer-stack height in front of the slot face (world y =
    -EXT_THICKNESS - stack_height); the rings span the gap to the
    slot face at world y = -EXT_THICKNESS.

Two variants:
  * exploded — motor assembly pulled outboard along world -Y by
               MOTOR_EXPLODE so the BHCS install path reads clearly.
  * assembled — rings flush against the slot face.

The base ID41Rd layer is always shown assembled — it's the prior
step's finished state that this composition builds onto.

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.motor_11_frame
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import BaseAssembly
from hardware.assembly.procedures.frame_10_extrusion_tnut import (
    EXT_THICKNESS,
    LONG_LENGTH,
    SHORT_LENGTH,
    SHORT_TOP_END_GAP,
    SHORT_TOP_INNER_GAP,
)
from hardware.assembly.procedures.idler_41_rd import ID41Rd
from hardware.assembly.procedures.motor_10_bracket import MO10Bracket
from hardware.assembly.projection import MAIN_FRAME_VIEW, Camera
from hardware.parts.standard.extrusion import cb_end_offset

MOTOR_EXPLODE = 40  # mm — exploded: outboard air gap, slot face → bracket bottom


class MO11Frame(BaseAssembly):
    camera = [MAIN_FRAME_VIEW, Camera(-32.50, 48.55, -133.86)]

    def _build(self) -> Compound:
        # Base layer — always assembled. Bundles frame + all 4 idler
        # corner mounts + their brackets; this step adds the left
        # motor + bracket on top.
        base_compound = ID41Rd(exploded=False).build()

        # Target t-nuts: short_top's LEFT TWO standard M5 t-nuts.
        # short_top's local Z position p maps to world x =
        # -SHORT_LENGTH/2 + p (its placement plane has x_dir=(0,-1,0),
        # z_dir=(1,0,0), so local +Z → world +X).
        left_t1_world_x = -SHORT_LENGTH / 2 + SHORT_TOP_END_GAP
        left_t2_world_x = -SHORT_LENGTH / 2 + SHORT_TOP_END_GAP + SHORT_TOP_INNER_GAP
        short_top_world_z = LONG_LENGTH - cb_end_offset

        # Bracket hooks (bracket_bottom_z, m5_native_x, stack_height)
        # let this composition flush-mount without re-deriving the
        # bracket's internal geometry. With our placement x_dir=
        # (0,0,+1) and z_dir=(0,-1,0): native X → world Z so
        # m5_native_x maps to world_z = origin_z + m5_native_x; and
        # native Z → world -Y so the bracket bottom sits at world_y =
        # origin_y - bracket_bottom_z. The rings span stack_height in
        # +Y from the bracket bottom to the slot face.
        bracket_asm = MO10Bracket(exploded=False)
        motor_compound = bracket_asm.build()

        origin_x = (left_t1_world_x + left_t2_world_x) / 2
        # Bracket bottom touches the rings at world y = -EXT_THICKNESS
        # - stack_height. Exploded pulls origin further in -Y.
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

        # Hook for motor_30_pulley — plane where a pulley would seat
        # on the motor shaft. Origin sits on the bracket top face
        # (world Y = origin_y - bracket_top_z); z_dir matches the
        # shaft direction (world -Y); y_dir = +X aligns the pulley's
        # native +Y set-screw with the motor D-flat (MO10Bracket spins
        # the motor 180° about Z, so the D-flat lands on world +X).
        self.pulley_plane = Plane(
            origin=(origin_x, origin_y - bracket_asm.bracket_top_z, origin_z),
            x_dir=(0, 0, -1),
            z_dir=(0, -1, 0),
        )

        return Compound(
            label="motor_11_frame",
            children=[
                base_compound,
                motor_compound,
            ],
        )


if __name__ == "__main__":
    for exploded in (True, False):
        asm = MO11Frame(exploded=exploded)
        asm.export()
        asm.render()
