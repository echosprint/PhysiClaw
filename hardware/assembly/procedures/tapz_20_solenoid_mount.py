"""Solenoid mount on the BeltClamp — extends belt_30_motor_b by
placing the tapz_11_solenoid_attach sub-assembly onto the BeltClamp's
top face. The SolenoidMount's keyboard plate flushes on the clamp top;
its 32 × 17 corner-hole pattern aligns with the clamp's top corner
holes, and the wall wraps DOWN over the clamp's front face so the
wall's CSK pair engages the clamp's front-face holes.

Orientation in tapz_20 world (rooted in BE30MotorB / BE20Clamp frame —
clamp native +X → world +X, +Y → world +Z, +Z → world -Y):
  * SolenoidMount native +Z (wall direction) → world +Y — wall hangs
    DOWN from the plate, parallel to the clamp's front face (which
    spans clamp native +Z → -Z, i.e. world -Y → +Y).
  * SolenoidMount native +Y (toward the wall) → world -Z — the +Y end
    of the plate is on the clamp's front-face side (world -Z), so the
    wall's screen face (native Y = width/2 - wall_thickness) lands
    flush against the clamp's front face (world Z = clamp front face).
  * SolenoidMount native +X (length axis) → world +X (forced by
    right-handed composition of the two axes above).

A single Location is applied to the tapz_11_solenoid_attach compound
to bring its tapz_11-world placement into tapz_20 world; the solenoid,
tip, and wall-to-solenoid BHCS travel with the mount as one rigid
sub-assembly.

Fasteners added at this step:
  * 4 × BHCS M3 × 8 — through the SolenoidMount plate's corner holes,
    into the BeltClamp's top corner holes. Each engages an M3 square
    nut captive in one of the BeltClamp's horizontal side pockets
    (2 per side after mirror = 4 total).
  * 2 × FHCS M3 × 10 — heads sunk into the wall's corner CSK holes
    (CSK from back). The CSK pair sits 12 mm up the wall (mount native
    Z = +13); after the wall is flipped DOWN that lands at world Y =
    CLAMP_TOP_Y + 12, exactly on the clamp's front-face holes. The
    shanks pass through the 4 mm wall and thread into the M3 square
    nuts captive in the clamp's vertical side pockets.
  * 2 × M3 square nuts — captive in the BeltClamp's vertical side
    pockets (1 per side after mirror), aligned with the clamp's
    front-face holes.

Two variants:
  * exploded — TZ11SolenoidAttach sub-assembly lifted outboard along
               world -Y by EXPLODE_OUT (away from the clamp top); each
               BHCS shank tip floats SCREW_EXPLODE further along the
               same axis. The FHCS travel with the sub-assembly in -Y
               and additionally lift out of their CSK along world -Z
               by FHCS_EXPLODE (their own outboard axis). The 6 captive
               nuts pull out along their slot install axes (world ±X)
               by NUT_EXPLODE.
  * assembled — plate flush on clamp top, BHCS heads bottomed on the
                plate exposed face, FHCS heads sunk into the wall CSK
                (wide top flush with wall back face), nuts in their
                pockets.

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.tapz_20_solenoid_mount
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import BaseAssembly
from hardware.assembly.procedures.belt_20_clamp import (
    SLIDER_MOUNT_X,
    SLIDER_MOUNT_Y,
    SLIDER_MOUNT_Z,
)
from hardware.assembly.procedures.belt_30_motor_b import BE30MotorB
from hardware.assembly.procedures.tapz_11_solenoid_attach import (
    TZ11SolenoidAttach,
    MOUNT_ORIGIN_X as _TZ11_MOUNT_X,
    MOUNT_ORIGIN_Y as _TZ11_MOUNT_Y,
    MOUNT_ORIGIN_Z as _TZ11_MOUNT_Z,
)
from hardware.assembly.projection import Camera, MAIN_FRAME_VIEW
from hardware.parts._fits import M3_NUT_T
from hardware.parts.custom.belt_clamp import (
    corner_hole_offset,
    front_hole_offset,
    left_rect1_top_offset,
    left_rect2_right_from_front,
    length as clamp_length,
    slot1_center_y,
    slot2_center_y,
    thickness as clamp_thickness,
    width as clamp_width,
)
from hardware.parts.custom.solenoid_mount import (
    keyboard_rect_center_y as mount_keyboard_center_y,
    screen_corner_csk_hole_from_bottom,
    thickness as mount_plate_thickness,
    width as mount_width,
)
from hardware.parts.standard.nut import Nut, SPECS as NUT_SPECS
from hardware.parts.standard.screw import FHCS_DIMS, Screw, head_skirt

BHCS_LENGTH    = 8     # mm — M3 BHCS underhead length (plate corner → clamp top corner)
FHCS_LENGTH    = 10    # mm — M3 FHCS overall length (wall corner CSK)
EXPLODE_OUT    = 30    # mm — exploded: TZ11SolenoidAttach lifted outboard along world -Y
SCREW_EXPLODE  = 30    # mm — exploded: BHCS shank tip floats further along world -Y
FHCS_EXPLODE   = 30    # mm — exploded: FHCS lifted out of its CSK along world -Z
NUT_EXPLODE    = 15    # mm — exploded: each nut pulled out its install axis by this much

NUT_THICKNESS  = NUT_SPECS["square"]["M3"]["thickness"]
FHCS_HEAD_HEIGHT = FHCS_DIMS["M3"]["k"] + head_skirt   # cone + rim skirt

# ── BeltClamp feature positions in tapz_20 world ─────────────────────────────
# Clamp placement (BE20Clamp): origin = (-length/2, SLIDER_MOUNT_Y - thickness/2,
# SLIDER_MOUNT_Z - clamp_grid_y), x_dir = (1,0,0), z_dir = (0,-1,0). So clamp
# native +X → world +X, +Y → world +Z, +Z → world -Y.
_clamp_origin_x = SLIDER_MOUNT_X - clamp_length / 2
_clamp_grid_y   = (slot1_center_y + slot2_center_y) / 2
_clamp_origin_y = SLIDER_MOUNT_Y - clamp_thickness / 2
_clamp_origin_z = SLIDER_MOUNT_Z - _clamp_grid_y

# Convenience: clamp native (n_x, n_y, n_z) → tapz_20 world.
def _clamp_to_world(n_x, n_y, n_z):
    return (
        _clamp_origin_x + n_x,
        _clamp_origin_y - n_z,
        _clamp_origin_z + n_y,
    )

# Clamp top face (mating with plate's -Z face).
CLAMP_TOP_Y     = _clamp_origin_y - clamp_thickness / 2     # = SLIDER_MOUNT_Y - clamp_thickness
# Clamp top corner-hole X positions: pre-mirror hole at native X =
# -length/2 + corner_hole_offset (= -6); mirror about x = length/2 puts
# its copy at native X = 3·length/2 - corner_hole_offset (= +26). With
# _clamp_origin_x = -length/2, world X = ±(length - corner_hole_offset)
# = ±16.
_CORNER_NX_LEFT  = -clamp_length / 2 + corner_hole_offset
_CORNER_NX_RIGHT = +3 * clamp_length / 2 - corner_hole_offset
CORNER_HOLE_X    = [_clamp_origin_x + nx for nx in (_CORNER_NX_LEFT, _CORNER_NX_RIGHT)]

# Clamp top corner-hole Y rows (in clamp native Y, mapped to world Z).
_CORNER_NY_NEG = -clamp_width / 2 + corner_hole_offset      # -Y corner (= -8.5)
_CORNER_NY_POS = +clamp_width / 2 - corner_hole_offset      # +Y corner (= +8.5)
CORNER_HOLE_Z  = [_clamp_origin_z + ny for ny in (_CORNER_NY_NEG, _CORNER_NY_POS)]

# Horizontal-pocket nut center, in clamp native Z.
# Slot top is at face_local Y = clamp_thickness - left_rect1_top_offset; the
# nut socket rectangle has height M3_NUT_T (= face-local Y), so its center is
# half that below the top.
_HNUT_NZ = -clamp_thickness / 2 + (clamp_thickness - left_rect1_top_offset - M3_NUT_T / 2)
HNUT_Y   = _clamp_origin_y - _HNUT_NZ
# Horizontal nut Y in native Y matches the corner-hole Y row (centered on the
# screw axis at native Y = ±8.5).

# Vertical-pocket nut center, in clamp native (Y, Z).
# Face-local X = clamp_width - left_rect2_right_from_front - left_rect2_w/2,
# where left_rect2_w = M3_NUT_T. Face-local Y = front_hole_offset.
_VNUT_NY = +clamp_width / 2 - (clamp_width - left_rect2_right_from_front - M3_NUT_T / 2)
_VNUT_NZ = -clamp_thickness / 2 + front_hole_offset
VNUT_Y   = _clamp_origin_y - _VNUT_NZ
VNUT_Z   = _clamp_origin_z + _VNUT_NY

# ── SolenoidMount placement in tapz_20 world ──────────────────────────────────
# Mount native +X → world +X, +Y → world -Z, +Z → world +Y (right-handed).
# Constraints:
#   • plate native +Z face (top face of the box, from which the wall extrudes)
#     is the mating face — it lands flush on the clamp top face (world Y =
#     CLAMP_TOP_Y). Native +Z → world +Y, so the +Z face at native z =
#     +thickness/2 sits at world Y = origin_y + thickness/2; setting that
#     equal to CLAMP_TOP_Y gives origin_y = CLAMP_TOP_Y - mount_plate_thickness/2.
#   • plate keyboard hole-grid centre (native (0, mount_keyboard_center_y, 0))
#     lands on the clamp top corner-hole grid centre (world (SLIDER_MOUNT_X,
#     CLAMP_TOP_Y, _clamp_origin_z)). With native +Y → world -Z, the world Z
#     contribution of the grid centre is -mount_keyboard_center_y; setting
#     origin_z + (-mount_keyboard_center_y) = _clamp_origin_z gives origin_z
#     = _clamp_origin_z + mount_keyboard_center_y.
MOUNT_ORIGIN_X = SLIDER_MOUNT_X
MOUNT_ORIGIN_Y = CLAMP_TOP_Y - mount_plate_thickness / 2
MOUNT_ORIGIN_Z = _clamp_origin_z + mount_keyboard_center_y

# ── tapz_11-world → tapz_20-world transform ───────────────────────────────────
# In tapz_11 the SolenoidMount native origin sits at (_TZ11_MOUNT_X,
# _TZ11_MOUNT_Y, _TZ11_MOUNT_Z) with mount native +X → tapz_11 +Y, +Y →
# tapz_11 +X, +Z → tapz_11 -Z. The transform below maps tapz_11 world
# coordinates to the new placement so the entire TZ11SolenoidAttach compound
# lands in tapz_20 world without rebuilding any of its parts.
#
# Rotation R takes mount-native axes as expressed in tapz_11 world to the
# same axes as expressed in tapz_20 world:
#   tapz_11 world +X (= mount +Y) → tapz_20 world -Z
#   tapz_11 world +Y (= mount +X) → tapz_20 world +X
#   tapz_11 world +Z (= mount -Z) → tapz_20 world -Y
# i.e. R·(a, b, c) = (b, -c, -a), expressed as Plane(x_dir=(0,0,-1),
# z_dir=(0,-1,0)). Translation O = M_20_origin − R · M_11_origin lands the
# tapz_11 mount origin on (MOUNT_ORIGIN_X, MOUNT_ORIGIN_Y, MOUNT_ORIGIN_Z).
# Derived from the imported tapz_11 origin (not hardcoded), so it tracks the
# screen-pattern position automatically — e.g. shifting
# screen_pattern_base_from_bottom moves _TZ11_MOUNT_Z and the placement follows.
TAPZ11_TO_TAPZ20 = Location(Plane(
    origin=(MOUNT_ORIGIN_X - _TZ11_MOUNT_Y,
            MOUNT_ORIGIN_Y + _TZ11_MOUNT_Z,
            MOUNT_ORIGIN_Z + _TZ11_MOUNT_X),
    x_dir=(0, 0, -1),
    z_dir=(0, -1, 0),
))


class TZ20SolenoidMount(BaseAssembly):
    camera = [MAIN_FRAME_VIEW, Camera(-13.83, -31.54, -31.35)]
    def _build(self) -> Compound:
        base_compound = BE30MotorB(exploded=False).build()

        # SolenoidMount + solenoid + tip + wall BHCS — pulled in as a rigid
        # sub-assembly, then optionally lifted outboard along world -Y.
        sub = TZ11SolenoidAttach(exploded=False).build()
        sub.move(TAPZ11_TO_TAPZ20)
        if self.exploded:
            sub.move(Location((0, -EXPLODE_OUT, 0)))

        # ── 4 × BHCS M3 × 8 — plate corners → clamp top corner holes ──────────
        # Screw native +Z (head direction) → world -Y so the head sits on the
        # plate's exposed (outboard) face. Underhead seating plane lands at
        # the plate's exposed face Y; in exploded mode the screw moves with
        # the sub-assembly and floats SCREW_EXPLODE further outboard.
        plate_exposed_y = CLAMP_TOP_Y - mount_plate_thickness
        if self.exploded:
            bhcs_underhead_y = plate_exposed_y - EXPLODE_OUT - SCREW_EXPLODE
        else:
            bhcs_underhead_y = plate_exposed_y

        bhcs_screws = []
        for world_x in CORNER_HOLE_X:
            for world_z in CORNER_HOLE_Z:
                screw = Screw("BHCS", "M3", BHCS_LENGTH).build()
                screw.move(Location(Plane(
                    origin=(world_x, bhcs_underhead_y, world_z),
                    x_dir=(1, 0, 0),     # round screw — choice cosmetic
                    z_dir=(0, -1, 0),    # native +Z (head) → world -Y
                )))
                bhcs_screws.append(screw)

        # ── 4 × M3 square nuts — captive in the BeltClamp horizontal pockets ──
        # Bore along world Y direction (matches BHCS axis). The flat
        # (unchamfered) face is presented to the screw thread (world -Y, the
        # BHCS entrance side), so nut native -Z faces world -Y, i.e. native +Z
        # → world +Y. Each nut slides into its slot from world ±X (the LEFT
        # / RIGHT face of the clamp). Exploded: pull out along that axis by
        # NUT_EXPLODE.
        horizontal_nuts = []
        for world_x in CORNER_HOLE_X:
            install_dir = -1 if world_x < 0 else +1     # -X face for LEFT, +X for RIGHT
            nut_x = world_x + (install_dir * NUT_EXPLODE if self.exploded else 0)
            for world_z in CORNER_HOLE_Z:
                nut = Nut("square", "M3").build()
                nut.move(Location(Plane(
                    origin=(nut_x, HNUT_Y - NUT_THICKNESS / 2, world_z),
                    x_dir=(1, 0, 0),
                    z_dir=(0, +1, 0),    # nut native +Z (chamfer) → world +Y (away from screw)
                )))
                horizontal_nuts.append(nut)

        # ── 2 × FHCS M3 × 10 — heads sunk in the wall corner CSK holes ────────
        # The CSK pair sits screen_corner_csk_hole_from_bottom (= 12 mm) up
        # the wall (mount native Z = thickness/2 + 12 = +13) and 4 mm in from
        # each plate end. With native +Z → world +Y, the CSK pair lands at
        # world Y = MOUNT_ORIGIN_Y + 13 = CLAMP_TOP_Y + 12 — exactly the
        # clamp's front-face hole Y. The screw axis runs along world Z; the
        # head's wide top sits FLUSH with the wall BACK face (world Z =
        # MOUNT_ORIGIN_Z - mount_width/2) and the cone sinks into the CSK
        # toward world +Z. The shank exits the screen face and threads into
        # the clamp front-face hole, engaging the captive square nut in the
        # vertical pocket. Underhead seating plane = back face + head height
        # so the head ends up FLUSH (not standing proud of the surface).
        wall_back_z       = MOUNT_ORIGIN_Z - mount_width / 2
        csk_native_z      = mount_plate_thickness / 2 + screen_corner_csk_hole_from_bottom
        csk_world_y       = MOUNT_ORIGIN_Y + csk_native_z
        fhcs_under_z      = wall_back_z + FHCS_HEAD_HEIGHT
        if self.exploded:
            csk_world_y -= EXPLODE_OUT     # travel with sub-assembly along world -Y
            fhcs_under_z -= FHCS_EXPLODE   # plus lift out of the CSK along world -Z

        fhcs_screws = []
        for world_x in CORNER_HOLE_X:
            screw = Screw("FHCS", "M3", FHCS_LENGTH).build()
            screw.move(Location(Plane(
                origin=(world_x, csk_world_y, fhcs_under_z),
                x_dir=(1, 0, 0),
                z_dir=(0, 0, -1),    # native +Z (head) → world -Z (head sinks into CSK from back face)
            )))
            fhcs_screws.append(screw)

        # ── 2 × M3 square nuts — captive in the BeltClamp vertical pockets ────
        # Bore along world Z direction (matches the clamp's front-face hole
        # axis). The flat (unchamfered) face is presented to the screw thread
        # (world -Z, where the FHCS approaches from through the wall), so nut
        # native -Z faces world -Z, i.e. native +Z → world +Z. Each nut slides
        # into its slot from world ±X (LEFT / RIGHT face). Exploded: pull
        # along that axis.
        vertical_nuts = []
        for world_x in CORNER_HOLE_X:
            install_dir = -1 if world_x < 0 else +1
            nut_x = world_x + (install_dir * NUT_EXPLODE if self.exploded else 0)
            nut = Nut("square", "M3").build()
            nut.move(Location(Plane(
                origin=(nut_x, VNUT_Y, VNUT_Z - NUT_THICKNESS / 2),
                x_dir=(1, 0, 0),
                z_dir=(0, 0, +1),    # nut native +Z (chamfer) → world +Z (away from screw)
            )))
            vertical_nuts.append(nut)

        return Compound(label="tapz_20_solenoid_mount", children=[
            base_compound,
            sub,
            *bhcs_screws,
            *horizontal_nuts,
            *fhcs_screws,
            *vertical_nuts,
        ])


if __name__ == "__main__":
    for exploded in (True, False):
        asm = TZ20SolenoidMount(exploded=exploded)
        asm.export()
        asm.render()
