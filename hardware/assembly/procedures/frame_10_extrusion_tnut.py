"""Extrusion + T-nut preload — slide standard T-nuts into 2040 slots.

The two variants use intentionally different layouts:

  * exploded — four extrusions stacked in Y rows.
      Solid:  extrusion + loose T-nuts queued past its +Z end, ready
              to slide in.
      Ghost:  destination silhouettes inside the slot — the seat
              positions the loose nuts will end up at.
      Each row's prep queue extends along its OWN +Z direction with no
      neighbor in the way; the install motion (slide along +Z into
      slot) reads cleanly per row. Row layout is intentional — easier
      to follow than a frame layout where prep queues from different
      members would collide.

  * assembled — the four members arranged as a rectangular frame:
                longs flush against shorts (separation=0), counterbore
                faces outboard, each long CB axis collinear with the
                matching short's end-cell bore. Nuts seated at their
                destinations; single layer.

  * 2 x 335 mm long extrusion (cb), each with 2 standard M5 T-nuts
    (top nut LONG_TOP_GAP from the frame top edge, bottom nut
    LONG_BOT_GAP from the frame bottom edge)
  * 1 x 170 mm short extrusion (top) with 4 standard M5 T-nuts
  * 1 x 170 mm short extrusion (bottom), plain

Run from the repo root:

    uv run --group cad python -m hardware step frame_10_extrusion_tnut
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import BaseAssembly, EXPLODED_CAM0, GHOST_LABEL, SOLID_LABEL
from hardware.assembly.projection import BACK_RIGHT_LOW_R90, MAIN_FRAME_VIEW
from hardware.assembly.travel_ranges import Y_EXTRUSION_LENGTH, X_EXTRUSION_LENGTH
from hardware.parts.standard.extrusion import (
    Extrusion2040,
    cb_end_offset,
    leg,
)
from hardware.parts.standard.t_nut import LENGTHS as TNUT_LENGTHS, TNut

LONG_LENGTH = Y_EXTRUSION_LENGTH  # Y-axis frame span — see assembly/travel_ranges.py
SHORT_LENGTH = X_EXTRUSION_LENGTH  # frame width — see assembly/travel_ranges.py
EXT_THICKNESS = 2 * leg  # 2040 narrow cross-section (= 20 mm)

ROW_SPACING = 60  # mm — exploded: between extrusion centerlines along Y
PREP_START_GAP = 5  # mm — clearance between extrusion +Z end and first prep nut
PREP_PITCH = 15  # mm — Z pitch between adjacent prep nuts in the queue

# 4 nuts on the top short, arranged as two motor-bracket pairs:
#   positions [0, 1] (left pair)  — motor A bracket M5 holes
#   positions [2, 3] (right pair) — motor B bracket M5 holes
# Each pair sits SHORT_TOP_INNER_GAP apart, matching the motor
# bracket's M5 hole pitch (motor_m5_pitch in parts/standard/bracket.py).
# The outer nut of each pair is SHORT_TOP_END_GAP in from its end of
# the extrusion; the gap between the inner nuts of the two pairs
# leaves room for the motors' bodies hanging down inside the frame
# rect (see motor_11_frame / motor_21_frame).
SHORT_TOP_END_GAP = 35
SHORT_TOP_INNER_GAP = 25
SHORT_TOP_POSITIONS = [
    SHORT_TOP_END_GAP,
    SHORT_TOP_END_GAP + SHORT_TOP_INNER_GAP,
    SHORT_LENGTH - SHORT_TOP_END_GAP - SHORT_TOP_INNER_GAP,
    SHORT_LENGTH - SHORT_TOP_END_GAP,
]

# 2 nuts per long extrusion: top nut LONG_TOP_GAP from the frame's
# top edge, bottom nut LONG_BOT_GAP from the frame's bottom edge.
# Layout is chiral because long_left is placed with local Z flipped
# (z_dir=(0,0,-1)) so its local-Z=0 lands at the frame TOP, while
# long_right uses z_dir=(0,0,+1) so its local-Z=0 lands at the frame
# BOTTOM. The two _LEFT / _RIGHT position lists are reflections of
# each other across LONG_LENGTH/2 and yield identical world Z's on
# both sides (top tnut at world z = LONG_LENGTH - LONG_TOP_GAP,
# bottom tnut at world z = LONG_BOT_GAP).
#
# These two gaps also fix the idler-mount blocks that bolt to the
# nuts (LU/RU motor mount on the top nut, LD/RD front-idler mount on
# the bottom nut — see idler_12_lu / idler_31_ld), and the Y rail is
# centered in the free span between those blocks (linear_11_y). So the
# gap sum is constrained by the rail: free span = LONG_LENGTH -
# LONG_TOP_GAP - LONG_BOT_GAP - 37 must stay >= Y_RAIL_LENGTH (240) or
# the rail penetrates the blocks. With LONG_LENGTH 355->345 the sum had
# to drop 10 (76->67); split so neither the rail-end clearance nor
# either block's clearance to its corner joining plate (frame_31) goes
# to zero: span 241 (rail +0.5 mm/side), both brackets +2 mm.
LONG_TOP_GAP = 42
LONG_BOT_GAP = 25
LONG_POSITIONS_LEFT = [LONG_TOP_GAP, LONG_LENGTH - LONG_BOT_GAP]
LONG_POSITIONS_RIGHT = [LONG_BOT_GAP, LONG_LENGTH - LONG_TOP_GAP]


def _seat_nuts(ext, positions: list[float]) -> list:
    """Build and seat one TNut per position into ext's slot_right slot."""
    nuts = [TNut("standard", "M5").build() for _ in positions]
    for nut, pos in zip(nuts, positions):
        ext.joints["slot_right"].connect_to(
            nut.joints["slot_mount"],
            position=pos,
        )
    return nuts


def ext_with_nuts(
    length: float,
    positions: list[float],
    cb: bool = False,
    with_prep: bool = True,
) -> tuple:
    """Build one 2040 + T-nut sets. Returns (ext, destinations, prep):
    * ext            — the extrusion itself
    * destinations   — one nut per entry in ``positions``, seated inside
                       the +X slot at that local-Z position
    * prep           — same count, loose nuts queued past the +Z end,
                       solid; empty list when ``with_prep=False`` so
                       callers that only want the seated state don't
                       pay for unused TNut builds.

    Pass ``positions=[]`` for a bare extrusion with no nuts."""
    ext = Extrusion2040(length=length, cb=cb).build()
    destinations = _seat_nuts(ext, positions)
    if not with_prep:
        return ext, destinations, []
    prep = _seat_nuts(ext, positions)
    half_len = TNUT_LENGTHS["standard"] / 2
    for i, (nut, pos) in enumerate(zip(prep, positions)):
        target_z = length + PREP_START_GAP + i * PREP_PITCH
        seated_z = pos - half_len
        nut.move(Location((0, 0, target_z - seated_z)))
    return ext, destinations, prep


class FR10ExtrusionTnut(BaseAssembly):
    views = [EXPLODED_CAM0]

    def __init__(self, *, separation: float = 30, exploded: bool = False):
        """``separation`` (mm) — horizontal gap between each long and
        the short ends in the assembled view. Default 30 leaves the
        longs visibly apart from the shorts so the preload reads as
        "longs placed nearby but not yet joined to the shorts."
        No effect on the exploded view (which uses the row layout)."""
        super().__init__(exploded=exploded)
        self.separation = separation

        # Two layouts → two cameras. Exploded looks down the rows from
        # the side; assembled views the frame head-on from the front —
        # though only the exploded view renders (see ``views``).
        self.camera = BACK_RIGHT_LOW_R90 if exploded else MAIN_FRAME_VIEW

        # Populated by _build_assembled_frame after .build(): keyed by
        # "long_left" / "long_right" / "short_top" / "short_bot", each
        # value is (ext, destinations). Lets a downstream assembly
        # (e.g. frame_20_SHCS) reach the longs' CB joints for placing
        # screws onto the moved members. Empty in the exploded row
        # variant — joints there are decoupled from the next step.
        self.frame_parts: dict = {}

    def _build(self) -> Compound:
        return (
            self._build_exploded_rows()
            if self.exploded
            else self._build_assembled_frame()
        )

    def _build_exploded_rows(self) -> Compound:
        specs = [
            # (length, cb, positions). Both long rows share
            # LONG_POSITIONS_RIGHT for visual symmetry; the actual
            # chirality is baked in by the assembled-frame plane (one
            # side flips local Z). short_bot has no nuts → [].
            (LONG_LENGTH, True, LONG_POSITIONS_RIGHT),
            (LONG_LENGTH, True, LONG_POSITIONS_RIGHT),
            (SHORT_LENGTH, False, SHORT_TOP_POSITIONS),
            (SHORT_LENGTH, False, []),
        ]
        solid_shapes = []  # extrusion + loose prep nuts
        ghost_shapes = []  # destination silhouettes inside the slot
        for i, (length, cb, positions) in enumerate(specs):
            ext, destinations, prep = ext_with_nuts(
                length,
                positions,
                cb=cb,
            )
            # Bake row offset into each leaf — the solid/ghost wrappers
            # below sit at identity, so projection treats leaf locations
            # as world positions and no nested-parent ambiguity arises.
            row_offset = Location((0, i * ROW_SPACING, 0))
            for s in (ext, *destinations, *prep):
                s.move(row_offset)
            solid_shapes.append(ext)
            solid_shapes.extend(prep)
            ghost_shapes.extend(destinations)
        return Compound(
            label="frame_10_extrusion_tnut",
            children=[
                Compound(label=SOLID_LABEL, children=solid_shapes),
                Compound(label=GHOST_LABEL, children=ghost_shapes),
            ],
        )

    def _build_assembled_frame(self) -> Compound:
        # self.separation pulls each long outboard from the short ends
        # (0 = flush/closed).
        #
        # All members: slot face → world -Y (toward front camera) so
        # seated nuts are visible. The longs also need their +Y (CB)
        # face outboard horizontally — chirality forces one to be
        # flipped end-to-end: long_left uses z_dir=(0,0,-1) with origin
        # at Z=LONG_LENGTH; long_right uses z_dir=(0,0,1).
        half_w = SHORT_LENGTH / 2 + EXT_THICKNESS / 2 + self.separation
        top_z = LONG_LENGTH - cb_end_offset
        bot_z = cb_end_offset

        members = [
            # name, (length, cb, positions), placement
            (
                "long_left",
                (LONG_LENGTH, True, LONG_POSITIONS_LEFT),
                Plane(
                    origin=(-half_w, 0, LONG_LENGTH), x_dir=(0, -1, 0), z_dir=(0, 0, -1)
                ),
            ),  # CB → world -X
            (
                "long_right",
                (LONG_LENGTH, True, LONG_POSITIONS_RIGHT),
                Plane(origin=(half_w, 0, 0), x_dir=(0, -1, 0), z_dir=(0, 0, 1)),
            ),  # CB → world +X
            (
                "short_top",
                (SHORT_LENGTH, False, SHORT_TOP_POSITIONS),
                Plane(
                    origin=(-SHORT_LENGTH / 2, 0, top_z),
                    x_dir=(0, -1, 0),
                    z_dir=(1, 0, 0),
                ),
            ),
            (
                "short_bot",
                (SHORT_LENGTH, False, []),  # no nuts
                Plane(
                    origin=(-SHORT_LENGTH / 2, 0, bot_z),
                    x_dir=(0, -1, 0),
                    z_dir=(1, 0, 0),
                ),
            ),
        ]
        shapes = []
        for name, (length, cb, positions), plane in members:
            ext, destinations, _ = ext_with_nuts(
                length,
                positions,
                cb=cb,
                with_prep=False,
            )
            loc = Location(plane)
            for s in (ext, *destinations):
                s.move(loc)
            self.frame_parts[name] = (ext, destinations)
            shapes.append(ext)
            shapes.extend(destinations)

        return Compound(label="frame_10_extrusion_tnut", children=shapes)
