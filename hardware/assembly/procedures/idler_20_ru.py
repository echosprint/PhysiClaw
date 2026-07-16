"""Idler mount (right-up) — same IdlerMountMotor block, shoulder
screws, center BHCS, and captive nuts as idler_10_lu; only the two
idler-stack recipes and the shoulder lengths differ. The build logic
lives in idler_10_lu (ID10Lu) and is reused here via inheritance:
this class overrides the ``_columns()`` hook with its own stacks and
retargets the Compound label so the STEP / SVG outputs land at
``idler_20_ru_*`` instead of ``idler_10_lu_*``.

Stacks, in the user's top → bottom order:
  * LEFT (x = -outer_hole_offset), SHOULDER M4 × 20 mm
    idler (smooth) / ring M5×8×0.5 / ring M5×10×9
    Stack = 8.5 + 0.5 + 9 = 18 mm; shoulder 20 mm → 2 mm axial play.
  * RIGHT (x = +outer_hole_offset), SHOULDER M4 × 20 mm
    idler (smooth) / ring M5×8×0.5 / idler (smooth) / ring M5×8×0.5
    Stack = 2 × 8.5 + 2 × 0.5 = 18 mm; shoulder 20 mm → 2 mm play.

Equal shoulder lengths mean the two exploded screw heads share a Z on
the common shank-tip line. See idler_10_lu for the variant
descriptions, nut placement, and the rest of the placement math.

Run from the repo root:

    uv run --group cad python -m hardware step idler_20_ru
"""

from hardware.assembly.base import ASSEMBLED_CAM0, EXPLODED_CAM0
from hardware.assembly.procedures.idler_10_lu import Column, ID10Lu
from hardware.parts.custom.idler_mount_motor import outer_hole_offset
from hardware.parts.standard.pulley import flange_belt_h
from hardware.parts.standard.ring import SPECS as RING_SPECS
from hardware.parts.standard.ring import Ring

WASHER_SPEC = "M5x8x0.5"
SPACER_SPEC = "M5x10x9"
LEFT_SHOULDER_LEN = 20  # mm — covers spacer + washer + idler (18 mm)
RIGHT_SHOULDER_LEN = 20  # mm — covers washer + idler + washer + idler (18 mm)


class ID20Ru(ID10Lu):
    compound_label = "idler_20_ru"
    views = [EXPLODED_CAM0, ASSEMBLED_CAM0]

    def _columns(self) -> list[Column]:
        def washer():
            return Ring(WASHER_SPEC).build()

        def spacer():
            return Ring(SPACER_SPEC).build()

        washer_h = RING_SPECS[WASHER_SPEC]["height"]
        spacer_h = RING_SPECS[SPACER_SPEC]["height"]
        return [
            (
                -outer_hole_offset,
                LEFT_SHOULDER_LEN,
                [
                    (spacer, spacer_h),
                    (washer, washer_h),
                    (self._idler, flange_belt_h),
                ],
            ),
            (
                +outer_hole_offset,
                RIGHT_SHOULDER_LEN,
                [
                    (washer, washer_h),
                    (self._idler, flange_belt_h),
                    (washer, washer_h),
                    (self._idler, flange_belt_h),
                ],
            ),
        ]
