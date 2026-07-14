"""Solenoid + tip attached to the SolenoidMount — extends
tapz_10_solenoid_tip by adding the SolenoidMount custom part and 2× M3×6
BHCS that tie the solenoid's left-face M3 mount holes to the mount's
screen-face middle-column 15 mm-pitch pair.

The SolenoidMount hangs in its "ready to bolt onto the BeltClamp" pose:
keyboard plate on TOP, screen wall (the L's vertical leg) extending
DOWN. The solenoid keeps its native orientation so the bottom-rod + tip
point -Z (down), ready to strike the surface below; the top-rod
actuating end points +Z (up) and exits the open side of the L
(plate-relative +X), clear of the plate. The solenoid is translated
+Y by solenoid_depth so its -X face holes land on the wall's middle
column (which sits at world Y = +solenoid_depth/2 because of the
mount's native +X → world +Y mapping).

Mount-hole alignment:
  * Solenoid native left-face holes at (-outer_w/2, -depth/2,
    ±mount_hole_z_spacing/2). After +Y translation by depth: world
    (-outer_w/2, +depth/2, ±7.5) — 15 mm pitch.
  * Screen-face middle column: i=2 of the 5-base pattern, face_X =
    screen_pattern_base_from_left + 2 × screen_pattern_spacing = 20 mm.
    Base-row pair (without screen_pattern_row_shift): face_Y =
    base_from_bottom + y_offsets[2] = 15 mm  and  15 + pair_offset = 30 mm
    → native (X=0, Z=16 / 31), 15 mm pitch matching the solenoid.
  * Mount placed with native +Y → world +X, native +Z → world -Z;
    derived native +X → world +Y. The wall BACK face (native
    Y = mount_width/2) mates against the solenoid -X face at world
    X = -solenoid_outer_w/2; the SCREEN face (native Y = mount_width/2
    - wall_thickness) faces world -X, where the screw heads enter.
    Pair midpoint at native Z = 23.5 maps to world Z = 0, centered on
    the solenoid's mount holes.

Two variants:
  * exploded — mount pulled MOUNT_EXPLODE outboard along world -X (away
               from the solenoid); each BHCS shank-tip floats
               SCREW_EXPLODE further along the same axis.
  * assembled — wall back face flush against the solenoid -X face;
                BHCS heads bottomed on the screen face.

Run from the repo root:

    uv run --group cad python -m hardware step tapz_11_solenoid_attach
"""

from build123d import MM, Compound, Location, Plane

from hardware.assembly.base import ASSEMBLED_CAM0, BaseAssembly, EXPLODED_CAM1
from hardware.assembly.procedures.tapz_10_solenoid_tip import TZ10SolenoidTip
from hardware.assembly.projection import Camera
from hardware.parts.custom.solenoid_mount import (
    SolenoidMount,
    screen_pattern_base_from_bottom,
    screen_pattern_base_from_left,
    screen_pattern_pair_offset,
    screen_pattern_spacing,
    screen_pattern_y_offsets,
    thickness as mount_thickness,
    wall_thickness,
    width as mount_width,
)
from hardware.parts.standard.screw import Screw
from hardware.parts.standard.solenoid import (
    depth as solenoid_depth,
    outer_w as solenoid_outer_w,
)

BHCS_LENGTH = 6  # mm — M3 BHCS underhead length
MOUNT_EXPLODE = 30  # mm — exploded: mount pulled outboard along world -X
SCREW_EXPLODE = 30  # mm — exploded: each BHCS shank-tip floats this far
#      further outboard along -X, so screws read as
#      separate parts that drop into the wall holes

# ── Middle-column 15 mm-pitch pair in mount native frame ─────────────────────
_MIDDLE_IDX = 2
_face_x = screen_pattern_base_from_left + _MIDDLE_IDX * screen_pattern_spacing
_face_y_lo = (
    screen_pattern_base_from_bottom + screen_pattern_y_offsets[_MIDDLE_IDX] * MM
)
_face_y_hi = _face_y_lo + screen_pattern_pair_offset
_hole_native_z_lo = mount_thickness / 2 + _face_y_lo
_hole_native_z_hi = mount_thickness / 2 + _face_y_hi
_hole_native_z_mid = (_hole_native_z_lo + _hole_native_z_hi) / 2

# ── Mount placement ──────────────────────────────────────────────────────────
# Native +Y → world +X (wall back-face normal points to solenoid), native +Z →
# world -Z (wall extends down from the plate). The solenoid (translated +Y by
# solenoid_depth) has mount holes at world (-outer_w/2, +depth/2, ±7.5); the
# constants below put the wall's middle-column pair at the same coordinates.
MOUNT_ORIGIN_X = -solenoid_outer_w / 2 - mount_width / 2
MOUNT_ORIGIN_Y = +solenoid_depth / 2
MOUNT_ORIGIN_Z = +_hole_native_z_mid

# Screen-face world X in the assembled state — where each BHCS underhead seats.
SCREEN_FACE_X = MOUNT_ORIGIN_X + (mount_width / 2 - wall_thickness)


class TZ11SolenoidAttach(BaseAssembly):
    camera = [Camera(50.69, 32.63, 0.47), Camera(-36.53, 5.28, 3.80)]
    views = [EXPLODED_CAM1, ASSEMBLED_CAM0]

    def _build(self) -> Compound:
        base_compound = TZ10SolenoidTip(exploded=False).build()
        # Solenoid + tip stay in their native orientation (tip → world -Z);
        # shift +Y by solenoid_depth so the -X face mount holes (native
        # Y = -depth/2) land at world Y = +depth/2, matching the wall's
        # middle-column pair after the mount's native +X → world +Y mapping.
        base_compound.move(Location((0, solenoid_depth, 0)))

        if self.exploded:
            mount_origin_x = MOUNT_ORIGIN_X - MOUNT_EXPLODE
            screw_underhead_x = SCREEN_FACE_X - MOUNT_EXPLODE - SCREW_EXPLODE
        else:
            mount_origin_x = MOUNT_ORIGIN_X
            screw_underhead_x = SCREEN_FACE_X

        mount = SolenoidMount().build()
        mount.move(
            Location(
                Plane(
                    origin=(mount_origin_x, MOUNT_ORIGIN_Y, MOUNT_ORIGIN_Z),
                    x_dir=(0, 1, 0),  # native +X → world +Y
                    z_dir=(0, 0, -1),  # native +Z → world -Z (wall hangs down)
                )
            )
        )  # → derived native +Y → world +X

        # 2 × BHCS M3 × 6 — head bottoms on the screen face; shank drives
        # world +X through the wall and into the solenoid's left-face M3
        # blind holes (1 mm deep).
        screws = []
        for hole_native_z in (_hole_native_z_lo, _hole_native_z_hi):
            screw = Screw("BHCS", "M3", BHCS_LENGTH).build()
            screw.move(
                Location(
                    Plane(
                        origin=(
                            screw_underhead_x,
                            MOUNT_ORIGIN_Y,
                            MOUNT_ORIGIN_Z - hole_native_z,
                        ),
                        x_dir=(0, 0, 1),  # round screw — x_dir choice cosmetic
                        z_dir=(-1, 0, 0),  # native +Z (head) → world -X (screen side)
                    )
                )
            )
            screws.append(screw)

        return Compound(
            label="tapz_11_solenoid_attach",
            children=[
                base_compound,
                mount,
                *screws,
            ],
        )
