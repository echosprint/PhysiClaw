"""Motor A belt on the assembled stage — extends linear_47_idler_rj2 by
routing the LEFT motor's GT2 belt across all of its pulleys and idlers.

The belt is one continuous strip, both ends clamped on the X-carriage
(modelled as 180° U-turns around 2 mm pins inside CLAMP_LEFT and
CLAMP_RIGHT). The full route, in path order:

    CLAMP_LEFT → PIN_LEFT (U-turn, r=2 mm)
        → LJ2     (X-gantry LEFT joint, lower idler — toothed)
        → LD      (front-LEFT corner idler)
        → LU.down2 (LU 1-idler stack)
        → MOTOR_A pulley
        → LU.down1 (LU 2-stack, BOTTOM idler)
        → RU.down2 (RU 1-idler stack)
        → RJ1     (X-gantry RIGHT joint, lower idler — smooth)
    → PIN_RIGHT (U-turn, r=2 mm) → CLAMP_RIGHT

Every waypoint sits at world Y = -42.75 (the lower belt plane), so the
belt centerline stays in a single XZ plane — no twist between motor
pulley and adjacent idlers. The motor pulley Y is set by the
LEFT_PULLEY_GAP constant in motor_30_pulley; coplanarity is part of
that assembly's calibration.

The base (linear_47_idler_rj2) is always shown assembled — the belt
itself has no exploded variant, since it's a single continuous strip
with no fasteners.

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.belt_10_motor_a
"""

from build123d import Compound, Location

from hardware.assembly.base import BaseAssembly
from hardware.assembly.procedures.linear_47_idler_rj2 import LI47IdlerRj2
from hardware.assembly.projection import FRONT
from hardware.parts.standard.belt import Belt, motor_a_path

BELT_EXPLODE = 30  # mm — exploded: shift the belt outboard along world -Y
#      so it lifts clear of the pulleys / idlers and the
#      routing reads cleanly from the camera


class BE10MotorA(BaseAssembly):
    camera = FRONT

    def _build(self) -> Compound:
        base_compound = LI47IdlerRj2(exploded=False).build()
        belt = Belt(path=motor_a_path, name="motor_a", motor="A").build()
        if self.exploded:
            belt.move(Location((0, -BELT_EXPLODE, 0)))
        return Compound(
            label="belt_10_motor_a",
            children=[
                base_compound,
                belt,
            ],
        )


if __name__ == "__main__":
    for exploded in (True, False):
        asm = BE10MotorA(exploded=exploded)
        asm.export()
        asm.render()
