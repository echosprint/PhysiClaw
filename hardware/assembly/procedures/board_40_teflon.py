"""Teflon-tube service route — anchors the OD4 PTFE stiffener tube from a frame-
mounted TubeHolder (near the control board) to the BeltClamp on the moving
carriage.

The solenoid wire is spiral-wrapped to the tube ALONGSIDE it (it does not pass
through the bore); the tube only stiffens the bundle. The route is an inverted-U
(∩): both sockets open upward, so the tube drops down into each (the TubeHolder
socket on the frame and the BeltClamp blind hole on the carriage) and arcs up
and over between them. Because the tube anchors to the frame and the carriage —
never to the control board — carriage motion no longer bends the PCB.

Both port poses are read out of the built assembly so the route stays correct if
upstream placement changes:
  * End A — the TubeHolder socket mouth (placed here on the top short 2040).
  * End B — the BeltClamp blind-hole mouth, carried by the clamp's global
    location within the assembly.

Two variants:
  * exploded — tube lifted off (+Z) to show it separated from the ports.
  * assembled — tube seated between the two ports.

Run from the repo root:

    uv run --group cad python -m hardware step board_40_teflon
"""

from build123d import Compound, Location, Vector

from hardware.assembly.base import ASSEMBLED_CAM0, BaseAssembly
from hardware.assembly.procedures.board_32_tube_holder import (
    TUBE_HOLDER_PLACEMENT,
    BO32TubeHolder,
)
from hardware.assembly.projection import MAIN_FRAME_VIEW
from hardware.parts.custom import belt_clamp as BC
from hardware.parts.custom import tube_holder as TH
from hardware.parts.standard.teflon import Teflon

TUBE_OD = 4  # mm — OD4 PTFE (plugs the 4 mm sockets)
TUBE_ID = 2  # mm
TANGENT_SCALAR = 2.0  # Bézier-handle length — full enough for the ∩ apex to sweep
EXPLODE = 40  # mm — exploded: tube lifted off along +Z

# Socket mouth + boss axis in the TubeHolder native frame (placed by board_32).
_SOCKET_NATIVE = TH.socket_mouth
_SOCKET_BACK = tuple(m - a for m, a in zip(TH.socket_mouth, TH.socket_axis))


def _gloc(node, label, acc=None):
    """Composed world Location of the first leaf labeled ``label``."""
    acc = (acc or Location()) * node.location
    if getattr(node, "label", None) == label and not node.children:
        return acc
    for child in node.children:
        found = _gloc(child, label, acc)
        if found is not None:
            return found
    return None


def _wpt(loc: Location, p) -> Vector:
    return (loc * Location(Vector(*p))).position


def _wdir(loc: Location, p, q) -> Vector:
    """World direction from native point ``p`` to native point ``q``."""
    return (_wpt(loc, q) - _wpt(loc, p)).normalized()


class BO40Teflon(BaseAssembly):
    camera = MAIN_FRAME_VIEW
    views = [ASSEMBLED_CAM0]

    def _build(self) -> Compound:
        base = BO32TubeHolder(exploded=False).build()

        # End A: socket mouth — tube leaves UP along the boss axis.
        p1 = _wpt(TUBE_HOLDER_PLACEMENT, _SOCKET_NATIVE)
        d1 = _wdir(TUBE_HOLDER_PLACEMENT, _SOCKET_BACK, _SOCKET_NATIVE)

        # End B: BeltClamp blind-hole mouth on the carriage (opens upward). The
        # tube arcs over and drops DOWN into it, so it arrives heading into the
        # clamp; the opposing end tangents make Teflon raise the ∩ apex between.
        clamp = _gloc(base, "BeltClamp")
        hole_mouth = (BC.length / 2, BC.tube_hole_y, BC.thickness / 2)
        hole_in = (BC.length / 2, BC.tube_hole_y, BC.thickness / 2 - 1)
        p2 = _wpt(clamp, hole_mouth)
        d2 = _wdir(clamp, hole_mouth, hole_in)  # down into the clamp

        tube = Teflon(
            point1=tuple(p1),
            direction1=tuple(d1),
            point2=tuple(p2),
            direction2=tuple(d2),  # drop down into the clamp
            od=TUBE_OD,
            id=TUBE_ID,
            tangent_scalars=(TANGENT_SCALAR, TANGENT_SCALAR),
        ).build()
        if self.exploded:
            tube.move(Location((0, 0, EXPLODE)))

        return Compound(label="board_40_teflon", children=[base, tube])
