"""Frame assembly step — 8 SHCS M6 close the four-member preload into
a rectangle. The preload itself (4 extrusions + seated T-nuts in the
populated members) comes from FR10ExtrusionTnut; this file just adds the
screws on top.

Two variants:

  * exploded — preload built with FR10ExtrusionTnut(separation=FRAME_GAP),
               so the longs sit FRAME_GAP outboard of the short ends
               (CB axes still collinear with the short end-cell bores).
               Screws floated further outboard along the entry axis at
               SCREW_EXPLODE.
  * assembled — preload built with FR10ExtrusionTnut(separation=0), longs
                flush against the short ends. Screws fully bottomed in
                the counterbores (position=cb_head_depth).

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.frame_20_SHCS
"""

from build123d import Compound

from hardware.assembly.base import BaseAssembly
from hardware.assembly.procedures.frame_10_extrusion_tnut import FR10ExtrusionTnut
from hardware.assembly.projection import MAIN_FRAME_VIEW
from hardware.parts.standard.extrusion import CB_LABELS, cb_head_depth
from hardware.parts.standard.screw import Screw

FRAME_GAP = 30  # mm — exploded: horizontal gap between long and short
SCREW_EXPLODE = -35  # mm — exploded: negative LinearJoint position = outboard
SHCS_LENGTH = 20  # mm — SHCS M6 underhead length


class FR20SHCS(BaseAssembly):
    camera = MAIN_FRAME_VIEW

    def _build(self) -> Compound:
        # Frame geometry (extrusions + seated nuts) comes from the
        # preload step. separation chooses the explode amount;
        # screw_position chooses outboard (exploded) vs seated.
        if self.exploded:
            separation, screw_position = FRAME_GAP, SCREW_EXPLODE
        else:
            separation, screw_position = 0, cb_head_depth

        preload = FR10ExtrusionTnut(separation=separation)
        preload_compound = preload.build()
        long_left_ext, _ = preload.frame_parts["long_left"]
        long_right_ext, _ = preload.frame_parts["long_right"]

        # 8 SHCS M6 — one per counterbore on each long. The longs have
        # already been moved as part of preload.build, so connect_to
        # here picks up their world transforms.
        screws = []
        for long_ext in (long_left_ext, long_right_ext):
            for cb in CB_LABELS:
                screw = Screw("SHCS", "M6", SHCS_LENGTH).build()
                long_ext.joints[cb].connect_to(
                    screw.joints["head"],
                    position=screw_position,
                )
                screws.append(screw)

        return Compound(
            label="frame_20_SHCS",
            children=[
                preload_compound,
                *screws,
            ],
        )


if __name__ == "__main__":
    for exploded in (True, False):
        asm = FR20SHCS(exploded=exploded)
        asm.export()
        asm.render()
