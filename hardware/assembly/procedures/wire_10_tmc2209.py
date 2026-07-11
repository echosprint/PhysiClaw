"""TMC2209 stepper drivers on the board — plugs two TMC2209 carrier modules
into the MKS DLC32's driver slots.

Both are standard (purchased) parts: the bare ``MksBoard`` and the
``Tmc2209`` modules, shown on their own (no holder, no frame).

Each module's 2×8 male header drops into a slot's two 1×8 female sockets:
the carrier bottom (its ``mount`` plane, z = 0) seats on the socket tops
(z = pcb_th + HDR_HEIGHT), pins protruding -Z into the sockets. The board
lays its sockets out with the 8-pin rows along Y and the two rows
``DRIVER_ROW_PITCH`` apart along X (``_female_header`` rotated 90°), so each
module — canonically 8 pins along X, rows along Y — is rotated +90° about Z
to match. The pin pitch and row spacing are the shared ``_fits`` constants,
so the pins land in the sockets by construction.

Two variants:
  * exploded — the modules lifted off their slots along +Z.
  * assembled — modules seated, pins in the sockets.

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.wire_10_tmc2209
"""

from build123d import Compound, Location

from hardware.assembly.base import BaseAssembly
from hardware.assembly.projection import Camera, ISO
from hardware.parts.standard.board import (
    DRIVER_X,
    DRIVER_Y,
    HDR_HEIGHT,
    MksBoard,
    pcb_th,
)
from hardware.parts.standard.tmc2209 import Tmc2209

SOCKET_TOP = pcb_th + HDR_HEIGHT  # driver-socket top (carrier bottom seats here)
MODULE_EXPLODE = 30  # mm — exploded: modules lifted off their slots along +Z


class WI10Tmc2209(BaseAssembly):
    camera = [ISO, Camera(-21.21, 40.26, 7.02)]

    def _build(self) -> Compound:
        board = MksBoard().build()

        modules = []
        for dx in DRIVER_X[:2]:  # two modules, in the first two driver slots
            # +90° about Z aligns the module's 2×8 pins with the slot's two
            # 1×8 sockets; carrier bottom seats on the socket tops.
            module = Tmc2209().build()
            mz = SOCKET_TOP + (MODULE_EXPLODE if self.exploded else 0)
            module.move(Location((dx, DRIVER_Y, mz), (0, 0, 90)))
            modules.append(module)

        return Compound(label="wire_10_tmc2209", children=[board, *modules])


if __name__ == "__main__":
    for exploded in (True, False):
        asm = WI10Tmc2209(exploded=exploded)
        asm.export()
        asm.render()
