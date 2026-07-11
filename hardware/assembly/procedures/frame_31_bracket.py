"""Bracket-to-frame fastening — 4 flat brackets, one at each frame
corner, each spanning where a short extrusion meets a long. Fastened
by 2 BHCS M5 driven into 2 hammer T-nuts (one per slot) per bracket.
Reinforces the M6 SHCS end-tap joint already in place from frame_20_SHCS.

Composition:

  * Frame layer — FR20SHCS(exploded=False): 4 extrusions + 8 seated
    M6 SHCS + the preloaded standard T-nuts FR10ExtrusionTnut put inside
    the slots.
  * Bracket layer — one FR30BracketTnut sub-assembly per corner, oriented
    so its local +Z (the chain direction: t-nut → bracket → screw
    head) points outboard from the slot face (world -Y). Each
    FR30BracketTnut carries 1 bracket + 2 BHCS M5 + 2 hammer t-nuts.

Both variants use FR30BracketTnut(exploded=False) — the loose-mate
sub-assembly (BHCS dropped through bracket, t-nut hanging from shank
tip). Only the placement differs:

  * exploded  — chain pulled BRACKET_EXPLODE outboard of the slot face
                (t-nut floor sits in air, pre-install).
  * assembled — bracket BOTTOM flush on the slot face. The BHCS shank
                and t-nut reach inboard into the slot cavity.

The frame layer is always shown assembled — it's the prior step's
finished state that the brackets visibly reinforce.

Bracket-to-corner orientation: the bracket's two holes are along
FR30BracketTnut's local X axis. x_dir=(-sign_x, 0, 0) maps local -X (the
LEFT nut, whose slide axis runs along FR30BracketTnut's local Z direction)
onto the LONG-side hole at world ±half_w, and local +X (the RIGHT
nut, slide axis along FR30BracketTnut's local X) onto the SHORT-side hole
inboard from the corner. Slide-axis directions then fall out
correctly: long-side nut slides along world Z (the long's axis),
short-side nut slides along world X (the short's axis).

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.frame_31_bracket
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import BaseAssembly
from hardware.assembly.procedures.frame_10_extrusion_tnut import (
    EXT_THICKNESS,
    LONG_LENGTH,
    SHORT_LENGTH,
)
from hardware.assembly.procedures.frame_20_SHCS import FR20SHCS
from hardware.assembly.procedures.frame_30_bracket_tnut import FR30BracketTnut
from hardware.assembly.projection import MAIN_FRAME_VIEW
from hardware.parts.standard.extrusion import cb_end_offset

BRACKET_EXPLODE = 20  # mm — exploded: air gap, slot face → FR30BracketTnut t-nut floor


class FR31Bracket(BaseAssembly):
    camera = MAIN_FRAME_VIEW

    def _build(self) -> Compound:
        # Frame layer — always assembled (it's the prior step's result).
        frame = FR20SHCS(exploded=False)
        frame_compound = frame.build()

        top_z = LONG_LENGTH - cb_end_offset
        bot_z = cb_end_offset
        slot_face_y = -EXT_THICKNESS  # outboard face of the longs

        bracket_groups = []
        for sign_x in (-1, 1):
            corner_edge_x = sign_x * (SHORT_LENGTH / 2)
            for corner_z in (top_z, bot_z):
                bt = FR30BracketTnut(exploded=False)
                bt_compound = bt.build()
                # Place the chain relative to the slot face:
                # exploded → t-nut floor BRACKET_EXPLODE outboard (chain
                #            hovers above the slot face, pre-install).
                # assembled → bracket bottom flush with slot face (chain
                #             reaches inboard into the slot cavity).
                if self.exploded:
                    origin_y = slot_face_y - BRACKET_EXPLODE
                else:
                    origin_y = slot_face_y + bt.bracket_bottom_z
                bt_compound.move(
                    Location(
                        Plane(
                            origin=(corner_edge_x, origin_y, corner_z),
                            x_dir=(-sign_x, 0, 0),  # LEFT hole → long side
                            z_dir=(0, -1, 0),  # chain runs outboard
                        )
                    )
                )
                bracket_groups.append(bt_compound)

        return Compound(
            label="frame_31_bracket",
            children=[
                frame_compound,
                *bracket_groups,
            ],
        )


if __name__ == "__main__":
    for exploded in (True, False):
        asm = FR31Bracket(exploded=exploded)
        asm.export()
        asm.render()
