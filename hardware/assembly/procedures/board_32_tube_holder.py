"""Tube holder mounted on the frame — sets the board_31 tube-holder sub-assembly
onto the top short 2040 (beside the control-board holder, on the -X side), with
the holder's two rib hammer-nuts dropped one into each of the two 20 mm-pitch
slots. The board_20 step for the PCB holder, repeated for the tube holder.

Extends board_30_pcb (the control board already in place). board_31's two nuts
are 20 mm apart along the rib (which runs along Y in the holder native frame);
the 180° about Z swings the socket boss to point along world -Y — which is
physically UP in the machine's use orientation — so the spigot stands up and the
tube drops into it. The holder is X-symmetric, so the nuts still drop into the
two slots at Y = ±10, and the translation seats the rib face on Z = SHORT_FACE_Z.

The TubeHolder's 4 mm socket anchors the fixed end of the PTFE stiffener tube;
the routed tube itself is added in the next step (board_40_teflon), which reads
``TUBE_HOLDER_PLACEMENT`` (exposed here) to route from the same placement.

Two variants:
  * exploded — the holder sub-assembly lifted off the frame along +Z.
  * assembled — rib seated on the +Z side face, nuts in the two slots.

Run from the repo root:

    uv run --group cad python -m hardware step board_32_tube_holder
"""

from build123d import MM, Compound, Location

from hardware.assembly.base import ASSEMBLED_CAM0, ASSEMBLED_CAM1, BaseAssembly
from hardware.assembly.procedures.board_20_frame import SHORT_FACE_Z
from hardware.assembly.procedures.board_30_pcb import BO30Pcb
from hardware.assembly.procedures.board_31_tube_tnut import BO31TubeTnut
from hardware.assembly.projection import Camera, MAIN_FRAME_VIEW
from hardware.parts.custom import tube_holder as TH

TH_X = (
    -65 * MM
)  # position along the extrusion (-X side; socket boss points world -Y = up)
EXPLODE = 60  # mm — exploded: holder sub-assembly lifted off along +Z

# board_31 sub-assembly (TubeHolder native frame) → mounted on the top short 2040's
# +Z side face. 180° about Z points the socket boss along world -Y (physically up),
# so the spigot stands up; the translation seats the rib face on Z = SHORT_FACE_Z.
# Exposed so board_40_teflon routes the tube from the same placement.
TUBE_HOLDER_PLACEMENT = Location((TH_X, 0, SHORT_FACE_Z + TH.tab_thick), (0, 0, 180))


class BO32TubeHolder(BaseAssembly):
    camera = [MAIN_FRAME_VIEW, Camera(-53.22, 43.33, -109.69)]
    views = [ASSEMBLED_CAM0, ASSEMBLED_CAM1]

    def _build(self) -> Compound:
        base = BO30Pcb(exploded=False).build()

        sub = BO31TubeTnut(exploded=False).build()
        sub.move(TUBE_HOLDER_PLACEMENT)
        if self.exploded:
            sub.move(Location((0, 0, EXPLODE)))

        return Compound(label="board_32_tube_holder", children=[base, sub])
