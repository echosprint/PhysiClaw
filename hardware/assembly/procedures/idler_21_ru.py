"""Idler-mount bracket (right-up) — geometrically identical to
idler_11_lu (1 flat bracket + 1 M5×10 BHCS (frame side) + 1 M5×12 BHCS
(block side) + 1 hammer M5 T-nut on the LEFT hole). The two files exist as separate sub-assemblies so each can
be placed independently at its own corner of the frame; the build
logic lives in idler_11_lu and is reused here via inheritance, with
just the Compound label retargeted so the STEP / SVG outputs land at
``idler_21_ru_*`` instead of ``idler_11_lu_*``.

See idler_11_lu for the full part list, variant descriptions, and
placement math.

Run from the repo root:

    uv run --group cad python -m hardware step idler_21_ru
"""

from hardware.assembly.base import ASSEMBLED_CAM0, EXPLODED_CAM0
from hardware.assembly.procedures.idler_11_lu import ID11Lu


class ID21Ru(ID11Lu):
    compound_label = "idler_21_ru"
    views = [EXPLODED_CAM0, ASSEMBLED_CAM0]
