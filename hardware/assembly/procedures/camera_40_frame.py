"""Camera on the frame — mounts the camera_30 arm onto board_40_teflon's RIGHT
long extrusion, on its outward (+X) side face.

In phone_40 / board_40 world the two long 2040s run along Z at X = ±105; the
right long's outward side face is its +X face (at X = 115). That 40 mm face
carries two T-slots at world Y = ±10 (running along Z). The corner bracket is
installed HORIZONTALLY — its single hammer T-nut drops into one slot — so the
gooseneck arm reaches out over the phone bed and the camera looks down at it.

Placement (CAMERA_PLACEMENT) maps the bracket's local frame onto the face:
  bracket +Z (deck outward normal) → world +X   (deck seats on the +X face)
  bracket +Y (the bend line)       → world +Z   (the T-nut sits in the slot)
  bracket +X (deck depth)          → world +Y
The deck hole at bracket x = 15, y = 0 lands on the chosen slot (world Y =
SLOT_Y) at MOUNT_Z along Z.

Two variants:
  * exploded — the camera arm pulled out along +X off the side face.
  * assembled — the bracket seated on the face, T-nuts in the slot.

Run from the repo root:

    uv run --group cad python -m hardware step camera_40_frame
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import ASSEMBLED_CAM0, ASSEMBLED_CAM1, BaseAssembly
from hardware.assembly.procedures.board_40_teflon import BO40Teflon
from hardware.assembly.procedures.camera_30_mount import Camera30Mount
from hardware.assembly.projection import Camera, MAIN_FRAME_VIEW
from hardware.assembly.travel_ranges import X_EXTRUSION_LENGTH
from hardware.parts.standard.extrusion import leg
from hardware.parts.standard.t_nut import HAMMER_TOTAL_HEIGHT

# Right long 2040. The two longs butt flush against the ends of the
# X_EXTRUSION_LENGTH-wide short members, so each long's INNER face sits at
# X = ±X_EXTRUSION_LENGTH/2 and its center one leg further out. Derived (not a
# magic 95/105) so it tracks the frame width — a stale literal had the bracket
# seating 10 mm inside the extrusion. The long is 20 mm along X (two legs); its
# outward +X side face is the 40 mm face, carrying two slots at world Y = ±10.
LONG_CENTER_X = X_EXTRUSION_LENGTH / 2 + leg  # = 105 (inner face at 95, +leg)
FACE_X = LONG_CENTER_X + leg  # = 115, outward +X side face
SLOT_Y = 10  # the second (lower, +Y / bed-side) slot
MOUNT_Z = 147.5  # along the long, at the bed's mid-length
DECK_HOLE_X = 15  # bracket-local X of the deck hole
EXPLODE = 70  # mm — exploded: arm pulled out +X off the face

# Bracket frame → extrusion face. x_dir = image of bracket +X, z_dir = image of
# bracket +Z; the origin offsets bracket x=DECK_HOLE_X onto the slot (Y=SLOT_Y)
# and the single hole (bracket y=0) onto MOUNT_Z along Z.
#
# The bracket's deck is lifted HAMMER_TOTAL_HEIGHT off its own z=0 (it sits on
# the T-nut bosses in camera_10), so the deck-bottom — the L-face that must seat
# flat on the extrusion side face — is at bracket z = HAMMER_TOTAL_HEIGHT. Pull
# the origin in by that amount so the deck contacts the face at X = FACE_X and
# the T-nuts drop INTO the slot (X < FACE_X) instead of standing proud of it.
CAMERA_PLACEMENT = Plane(
    origin=(FACE_X - HAMMER_TOTAL_HEIGHT, SLOT_Y - DECK_HOLE_X, MOUNT_Z),
    x_dir=(0, 1, 0),  # bracket +X → world +Y
    z_dir=(1, 0, 0),  # bracket +Z → world +X
)


class Camera40Frame(BaseAssembly):
    compound_label = "camera_40_frame"
    camera = [MAIN_FRAME_VIEW, Camera(35.39, -56.69, 45.46)]
    views = [ASSEMBLED_CAM0, ASSEMBLED_CAM1]

    def _build(self) -> Compound:
        base = BO40Teflon(exploded=False).build()

        cam = Camera30Mount(exploded=False).build()
        cam.move(Location(CAMERA_PLACEMENT))
        if self.exploded:
            cam.move(Location((EXPLODE, 0, 0)))

        return Compound(label=self.compound_label, children=[base, cam])
