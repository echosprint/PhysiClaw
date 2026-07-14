"""Motor pulleys — extends motor_21_frame (frame + 4 corner idlers +
both motors with brackets) by placing one GT2 20-tooth pulley on each
motor's shaft.

Pulley placement reads the ``pulley_plane`` hooks exposed by MO11Frame
(left motor) and MO21Frame (right motor) — each plane already encodes
the world position of the bracket-top seat (= motor pilot-pad top
face), the shaft axis (world -Y), and a rotation aligning the
pulley's native +Y set-screw with the motor's D-flat (world +X for
the left motor, world -X for the right). That keeps this composition
free of any motor-placement math.

Each pulley sits some distance above the pad top face along the
shaft — the gap differs per side so the two belt runs end up at
distinct heights and don't share a plane:

  * LEFT  — LEFT_PULLEY_GAP mm above the pad top
  * RIGHT — RIGHT_PULLEY_GAP mm above the pad top

Two variants:
  * exploded — pulleys pulled outboard along world -Y (the shaft
               axis) by PULLEY_EXPLODE so they read as separate
               pieces installed onto the motor shafts. The base
               (motor_21_frame) is always assembled.
  * assembled — pulleys seated on the motor shafts at their gaps.

Run from the repo root:

    uv run --group cad python -m hardware step motor_30_pulley
"""

from build123d import Compound, Location

from hardware.assembly.base import ASSEMBLED_CAM0, BaseAssembly, EXPLODED_CAM1
from hardware.assembly.procedures.motor_21_frame import MO21Frame
from hardware.assembly.projection import MAIN_FRAME_VIEW, Camera
from hardware.parts.standard.pulley import Pulley2GT20T

LEFT_PULLEY_GAP = 0  # mm — LEFT pulley belt-band lands on the LOWER idler
#      plane (Y = -42.75) so motor A's pulley is coplanar
#      with the LU/LD/LJ2/RJ1 idler belt-bands it drives.
#      0 = pulley hub seated flush on the bracket-top pad.
#      Tuned for motor A's 8 mm spacer (M6x12x8); a taller
#      spacer stands the motor off further and needs its
#      own gap (cf. RIGHT_PULLEY_GAP for the 12 mm side,
#      and the ring spec in motor_10_bracket).
RIGHT_PULLEY_GAP = 5  # mm — RIGHT pulley belt-band lands on the UPPER idler
#      plane (Y = -51.75) so motor B's pulley is coplanar
#      with the LU.top1 / RU.top1 idler belt-bands it
#      drives. Tuned for motor B's 12 mm spacer (M6x12x12).
#      The two belt runs sit 9 mm apart (lower -42.75 /
#      upper -51.75): 4 mm from the 8/12 mm spacer
#      difference plus the 5 mm gap difference here.
PULLEY_EXPLODE = 30  # mm — exploded: pull each pulley outboard along world
#      -Y (the motor shaft direction) so they read as
#      separate pieces installed onto the motor shafts.
#      Matches motor_21_frame's -Y explode convention.


class MO30Pulley(BaseAssembly):
    camera = [MAIN_FRAME_VIEW, Camera(-64.27, 40.48, -110.78)]
    views = [EXPLODED_CAM1, ASSEMBLED_CAM0]

    def _build(self) -> Compound:
        # Build the prior step and keep the instance so we can read
        # the pulley_plane hooks for both motors. The base is always
        # assembled — only the pulleys themselves move in exploded
        # mode (PULLEY_EXPLODE below).
        frame = MO21Frame(exploded=False)
        frame_compound = frame.build()

        pulleys = []
        for plane, gap in (
            (frame.base.pulley_plane, LEFT_PULLEY_GAP),
            (frame.pulley_plane, RIGHT_PULLEY_GAP),
        ):
            pulley = Pulley2GT20T(kind="pulley").build()
            # Lift the pulley along its native +Z (the shaft axis)
            # FIRST, then apply the plane so the gap maps to the
            # shaft direction in world coords.
            pulley.move(Location((0, 0, gap)))
            pulley.move(Location(plane))
            # Exploded: pull the pulley outboard along world -Y (the
            # shaft axis) — applied AFTER the plane, so the translation
            # is in world coords.
            if self.exploded:
                pulley.move(Location((0, -PULLEY_EXPLODE, 0)))
            pulleys.append(pulley)

        return Compound(
            label="motor_30_pulley",
            children=[
                frame_compound,
                *pulleys,
            ],
        )
