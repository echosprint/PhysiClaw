"""Linear-stage idler bundle (LJ1) — a shoulder-screw axle carrying
an idler (toothed or smooth) over an M5×8×0.5 washer, optionally
with an M5×10×9 spacer beneath the washer, ready to bolt into the
host's tapped M4 hole (host not modelled here).

Subclasses share this build logic and override these class attributes:
  * ``compound_label``  retargets the embedded compound label (and
                        the STEP / SVG filename via ``_module_stem``).
  * ``toothed``         picks toothed (belt-tracking) vs smooth
                        (low-friction) idler.
  * ``include_spacer``  adds an M5×10×9 spacer at the bottom of the
                        stack — needed when the idler must sit
                        higher off the host top.
  * ``shoulder_len``    SHOULDER M4 length; pick to match the
                        resulting stack height + a couple of mm of
                        free-shoulder play above the thread so the
                        idler spins free.

Stack composition (assembled, install face at native Z = 0, stack
growing up along +Z):
  * (optional) spacer (M5×10×9, 9 mm)        install face at z = 0
  * washer            (M5×8×0.5, 0.5 mm)     above the spacer
  * idler             (flange_belt_h = 8.5 mm)  above the washer
  * SHOULDER M4 × shoulder_len screw         underhead seated on the
                                              idler top

Stack heights and matching shoulder lengths:
  * with spacer    — 9 + 0.5 + 8.5 = 18 mm; shoulder 20 mm → 2 mm play
  * without spacer —     0.5 + 8.5 =  9 mm; shoulder 10 mm → 1 mm play
The M4 thread (7.2 mm) protrudes below the install face into the
host's tapped hole.

Two variants:
  * exploded — stack laid out along +Z with EXPLODE_SEPARATION of air
               between adjacent parts (bottom part fixed at z = 0);
               screw lifted so its thread tip sits SCREW_GAP above
               the idler's exploded top face.
  * assembled — stack tight, screw underhead touching the idler top,
                free shoulder above the thread, thread protruding
                below z = 0.

Run from the repo root:

    uv run --group cad python -m hardware step linear_40_idler_lj1
"""

from build123d import Compound, Location

from hardware.assembly.base import ASSEMBLED_CAM0, BaseAssembly, EXPLODED_CAM0
from hardware.assembly.projection import FRONT_LEFT_HIGH
from hardware.parts.standard.pulley import Pulley2GT20T, flange_belt_h
from hardware.parts.standard.ring import SPECS as RING_SPECS, Ring
from hardware.parts.standard.screw import SHOULDER_DIMS, Screw

WASHER_SPEC = "M5x8x0.5"
SPACER_SPEC = "M5x10x9"
EXPLODE_SEPARATION = 5  # mm — exploded: air between adjacent stack parts
SCREW_GAP = 12  # mm — exploded: idler exploded top → screw thread tip


class LI40IdlerLj1(BaseAssembly):
    compound_label: str = "linear_40_idler_lj1"
    toothed: bool = False
    include_spacer: bool = True
    shoulder_len: int = 20  # mm — pairs with the 18 mm spacer+washer+idler stack
    camera = FRONT_LEFT_HIGH
    views = [EXPLODED_CAM0, ASSEMBLED_CAM0]

    def _build(self) -> Compound:
        washer_h = RING_SPECS[WASHER_SPEC]["height"]
        spacer_h = RING_SPECS[SPACER_SPEC]["height"]
        thread_len = SHOULDER_DIMS["M4"]["thread_len"]

        # Bottom→top order. Each entry: (part, axial height). The
        # bottom face (z = 0) is the install face that lands on the host.
        stack: list = []
        if self.include_spacer:
            stack.append((Ring(SPACER_SPEC).build(), spacer_h))
        stack.append((Ring(WASHER_SPEC).build(), washer_h))
        stack.append(
            (
                Pulley2GT20T(kind="idler", toothed=self.toothed).build(),
                flange_belt_h,
            )
        )

        sep = EXPLODE_SEPARATION if self.exploded else 0

        placed = []
        cursor_z = 0  # bottom face of the next part to place
        for part, h in stack:
            part.move(Location((0, 0, cursor_z)))
            placed.append(part)
            cursor_z += h + sep
        idler_top_z = cursor_z - sep  # top face of the topmost stack part

        if self.exploded:
            thread_tip_z = idler_top_z + SCREW_GAP
            screw_z = thread_tip_z + thread_len + self.shoulder_len
        else:
            screw_z = idler_top_z

        screw = Screw("SHOULDER", "M4", self.shoulder_len).build()
        screw.move(Location((0, 0, screw_z)))
        placed.append(screw)

        return Compound(label=self.compound_label, children=placed)
