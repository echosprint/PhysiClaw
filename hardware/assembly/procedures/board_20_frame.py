"""PCB holder mounted to the frame — sets the board_10 holder sub-assembly onto
the phone_40 frame, on the TOP short extrusion's outward side face, with the
holder's two rib hammer-nuts dropped one into each of the two 20 mm-pitch slots.

In phone_40 world the top short 2040 runs along X (X ∈ [-85, 85]) at Z ∈
[315, 335]; its 40 mm side faces are the ±Z faces, each carrying two slots at
Y = ±10 (pitch 20). board_10's rib holes are also 20 mm apart, so the two
hammer nuts land one in each slot.

board_10 is rotated +90° about Z so its rib (and the two nuts, 20 mm apart along
the rib) line up across the two slots at Y = ±10 with each nut's length along X
(the slot run); the PCB/standoff side keeps facing +Z, out of the frame's top
end. Then translated so the rib face seats on the +Z side face (Z = 335) and the
nuts straddle Y = 0.

Two variants:
  * exploded — the holder sub-assembly pulled out along +Z off the side face.
  * assembled — rib seated on the +Z side face, nuts in the two slots.

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.board_20_frame
"""

from build123d import Compound, Location

from hardware.assembly.base import BaseAssembly
from hardware.assembly.procedures.board_10_holder_tnut import BO10HolderTnut
from hardware.assembly.procedures.frame_10_extrusion_tnut import LONG_LENGTH
from hardware.assembly.procedures.phone_40_bed_frame import PH40BedFrame
from hardware.assembly.projection import Camera, MAIN_FRAME_VIEW
from hardware.parts.custom.pcb_holder import rib_cx, tab_thick

SHORT_FACE_Z = LONG_LENGTH  # mm — top short 2040 outward (+Z) side face, = 335
HOLDER_X = 0  # mm — position along the short (centered)
EXPLODE = 60  # mm — exploded: holder pulled out along +Z off the face

# board_10 (holder native frame) → mounted on the top short's +Z side face.
# +90° about Z aligns the rib's two nuts across the two Y=±10 slots; the
# translation seats the rib face on Z = SHORT_FACE_Z and centers the nuts on
# Y = 0. Exposed so board_30 can place the PCB with the same transform.
BOARD_PLACEMENT = Location((HOLDER_X, -rib_cx, SHORT_FACE_Z + tab_thick), (0, 0, 90))


class BO20Frame(BaseAssembly):
    camera = [MAIN_FRAME_VIEW, Camera(39.94, 62.62, 133.88)]

    def _build(self) -> Compound:
        base = PH40BedFrame(exploded=False).build()

        board = BO10HolderTnut(exploded=False).build()
        board.move(BOARD_PLACEMENT)
        if self.exploded:
            board.move(Location((0, 0, EXPLODE)))

        return Compound(label="board_20_frame", children=[base, board])


if __name__ == "__main__":
    for exploded in (True, False):
        asm = BO20Frame(exploded=exploded)
        asm.export()
        asm.render()
