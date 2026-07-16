"""PCB on the holder — sets the MKS DLC32 board onto the board_20 holder and
fastens it through the four corners.

The PCB drops onto the holder's four standoffs (bottom face at z = thickness +
cyl_h, corner holes over the bores). Each corner is fastened with a BHCS M3 × 10
from the top — head on the PCB, shank down through the PCB + standoff + plate —
into an M3 square nut on the holder underside (seated in its locating recess).
Everything is built in the holder's native frame, then carried into the mounted
location by the same BOARD_PLACEMENT transform that positioned the holder.

Two variants:
  * exploded — PCB + screws lifted off the standoffs (screws floating above),
               nuts dropped below the plate, along the mount axis.
  * assembled — PCB seated, screws bottomed, nuts under the plate.

Run from the repo root:

    uv run --group cad python -m hardware step board_30_pcb
"""

from build123d import Compound, Location

from hardware.assembly.base import ASSEMBLED_CAM0, EXPLODED_CAM1, BaseAssembly
from hardware.assembly.procedures.board_20_frame import BOARD_PLACEMENT, BO20Frame
from hardware.assembly.projection import MAIN_FRAME_VIEW, Camera
from hardware.parts.custom.pcb_holder import cyl_h, standoff_xy, thickness
from hardware.parts.standard.board import MksBoard, pcb_th
from hardware.parts.standard.nut import SPECS as NUT_SPECS
from hardware.parts.standard.nut import Nut
from hardware.parts.standard.screw import Screw

STANDOFF_TOP = thickness + cyl_h  # holder standoff top (PCB bottom rests here)
PCB_TOP = STANDOFF_TOP + pcb_th  # PCB top face (BHCS head bears here)
BHCS_LENGTH = 10  # mm — M3 BHCS underhead length
NUT_THICK = NUT_SPECS["square"]["M3"]["thickness"]

PCB_EXPLODE = 35  # mm — exploded: PCB lifted off the standoffs
SCREW_EXTRA = 18  # mm — exploded: screws float this much above the PCB
NUT_EXPLODE = 25  # mm — exploded: square nuts dropped below the plate


class BO30Pcb(BaseAssembly):
    camera = [MAIN_FRAME_VIEW, Camera(55.44, 46.44, 119.69)]
    views = [EXPLODED_CAM1, ASSEMBLED_CAM0]

    def _build(self) -> Compound:
        base = BO20Frame(exploded=False).build()

        # PCB seated on the standoff tops (holder native frame).
        pcb = MksBoard().build()
        pcb.move(Location((0, 0, STANDOFF_TOP)))
        if self.exploded:
            pcb.move(Location((0, 0, PCB_EXPLODE)))

        screws, nuts = [], []
        for cx, cy in standoff_xy:
            # BHCS from the top: head on the PCB top (+Z), shank down through the
            # PCB + standoff + plate.
            screw = Screw("BHCS", "M3", BHCS_LENGTH).build()
            sz = PCB_TOP + (PCB_EXPLODE + SCREW_EXTRA if self.exploded else 0)
            screw.move(Location((cx, cy, sz)))
            screws.append(screw)

            # Square nut on the holder underside, flipped (180° about X) so its
            # FLAT face is up toward the screw (the chamfer faces away). Top
            # (flat) face at the plate underside (z = 0); solid extends to
            # z = -NUT_THICK.
            nut = Nut("square", "M3").build()
            nz = -(NUT_EXPLODE if self.exploded else 0)
            nut.move(Location((cx, cy, nz), (180, 0, 0)))
            nuts.append(nut)

        # Carry the PCB + fasteners into the mounted position.
        for part in (pcb, *screws, *nuts):
            part.move(BOARD_PLACEMENT)

        return Compound(label="board_30_pcb", children=[base, pcb, *screws, *nuts])
