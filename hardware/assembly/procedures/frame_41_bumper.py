"""4 bumper feet at the frame corners — one per corner, on the +Y
face (opposite the brackets). Mirror of frame_31_bracket: same
composition pattern, opposite slot face.

Composition:

  * Frame layer — FR31Bracket(exploded=False): full frame + the four
    corner brackets reinforcing the SHCS M6 end-taps.
  * Bumper layer — one FR40BumperTnut sub-assembly per corner,
    sitting on the +Y slot face of the LONG extrusions near each
    end. The chain (BHCS head → bumper → t-nut) points world −Y
    into the slot; the t-nut slide axis (sub-asm +Y) maps to world
    +Z, tracking the long's length.

Both variants use FR40BumperTnut(exploded=False) — the loose-mate
state (head bottomed in cbore, t-nut hanging from shank tip). Only
the corner placement differs:

  * exploded — bumper top floats BUMPER_EXPLODE outboard of the +Y
               slot face; the t-nut floats with it.
  * assembled — bumper top flush on the +Y slot face; the t-nut hangs
                from the shank tip just inside the slot mouth.

The frame layer is always shown assembled — it's the prior step's
finished state that the bumpers visibly add feet to.

Camera mirrors frame_31_bracket's −30/25 about world Y so the +Y
face (where the bumpers live) is the one facing the camera.

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.frame_41_bumper
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import BaseAssembly
from hardware.assembly.procedures.frame_10_extrusion_tnut import (
    EXT_THICKNESS,
    LONG_LENGTH,
    SHORT_LENGTH,
)
from hardware.assembly.procedures.frame_31_bracket import FR31Bracket
from hardware.assembly.procedures.frame_40_bumper_tnut import FR40BumperTnut
from hardware.assembly.projection import MAIN_FRAME_VIEW
from hardware.parts.standard.bumper import body_height as bumper_height
from hardware.parts.standard.extrusion import cb_end_offset

BUMPER_EXPLODE = 25  # mm — exploded: outboard air gap, +Y slot face → bumper top


class FR41Bumper(BaseAssembly):
    camera = MAIN_FRAME_VIEW

    def _build(self) -> Compound:
        # Frame layer — always assembled (it's the prior step's result).
        frame = FR31Bracket(exploded=False)
        frame_compound = frame.build()

        top_z = LONG_LENGTH - cb_end_offset
        bot_z = cb_end_offset
        # +Y slot face on the LONGS — mirror of slot_face_y=−EXT_THICKNESS
        # that frame_31_bracket uses on the −Y side. The longs sit at
        # world x = ±half_w (separation=0 in the assembled frame).
        back_slot_face_y = EXT_THICKNESS
        half_w = SHORT_LENGTH / 2 + EXT_THICKNESS / 2

        bumpers = []
        for sign_x in (-1, 1):
            for corner_z in (top_z, bot_z):
                bm = FR40BumperTnut(exploded=False)
                bm_compound = bm.build()
                # FR40BumperTnut local: chain runs +Z' from z'=0 (bumper
                # bottom) to z'=36.5 (t-nut plate-back); the bumper top
                # is at z'=bumper_height (22); t-nut slide axis is +Y'.
                # The Plane below sends +Z' → world −Y (chain into the
                # slot) and +Y' → world +Z (slide along the long).
                # origin_y is chosen so sub-asm z'=bumper_height lands
                # on back_slot_face_y (world y = origin_y − z').
                if self.exploded:
                    origin_y = back_slot_face_y + bumper_height + BUMPER_EXPLODE
                else:
                    origin_y = back_slot_face_y + bumper_height
                bm_compound.move(
                    Location(
                        Plane(
                            origin=(sign_x * half_w, origin_y, corner_z),
                            x_dir=(1, 0, 0),
                            z_dir=(0, -1, 0),
                        )
                    )
                )
                bumpers.append(bm_compound)

        return Compound(
            label="frame_41_bumper",
            children=[
                frame_compound,
                *bumpers,
            ],
        )


if __name__ == "__main__":
    for exploded in (True, False):
        asm = FR41Bumper(exploded=exploded)
        asm.export()
        asm.render()
