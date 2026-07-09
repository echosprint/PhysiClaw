import math

from build123d import *

from hardware.assembly.procedures.frame_10_extrusion_tnut import (
    EXT_THICKNESS,
    LONG_BOT_GAP,
    LONG_LENGTH,
    LONG_TOP_GAP,
    SHORT_LENGTH,
    SHORT_TOP_END_GAP,
    SHORT_TOP_INNER_GAP,
)
from hardware.assembly.procedures.linear_10_y import RAIL_LENGTH
from hardware.assembly.procedures.motor_10_bracket import MO10Bracket
from hardware.assembly.procedures.motor_30_pulley import LEFT_PULLEY_GAP
from hardware.parts.base import BaseStandardPart
from hardware.parts.custom.idler_mount_front import (
    slot_center_y as ld_slot_center_y,
    top_hole_y as ld_top_hole_y,
    width as ld_block_width,
)
from hardware.parts.custom.idler_mount_motor import (
    length as lu_block_length,
    outer_hole_offset,
    thickness as lu_block_thickness,
)
from hardware.parts.custom.xy_joint_left import (
    csk_hole_from_bottom as joint_csk_hole_from_bottom,
    csk_hole_from_left as joint_csk_hole_from_left,
    csk_x_spacing as joint_csk_x_spacing,
    csk_y_spacing as joint_csk_y_spacing,
    extra_hole_x as joint_extra_hole_x_n,
    extra_hole_y as joint_extra_hole_y_n,
    extra_hole2_x as joint_extra_hole2_x_n,
    extra_hole2_y as joint_extra_hole2_y_n,
    length as joint_length,
    thickness as joint_thickness,
    width as joint_width,
)
from hardware.parts.standard.bracket import (
    motor_m5_x_inset,
    motor_plate_length,
    motor_plate_thick,
    motor_shaft_x_offset,
)
from hardware.parts.standard.extrusion import cb_end_offset
from hardware.parts.standard.mgn9h import (
    block_top_z as slider_top_z,
    slider_position,
)
from hardware.parts.standard.motor import default_height as motor_default_height
from hardware.parts.standard.pulley import (
    belt_width as pulley_belt_width,
    flange_belt_h,
    hub_height as pulley_hub_height,
    pitch_diameter as pulley_pitch_diameter,
)
from hardware.parts.standard.ring import SPECS as RING_SPECS

# ── GT2 belt parameters ───────────────────────────────────────────────────────
belt_width      = pulley_belt_width                  # along pulley / idler axes
belt_thickness  = 1.38 * MM                          # GT2 total thickness (backing + tooth)
idler_pitch_r   = pulley_pitch_diameter / 2          # ≈ 6.37 mm
wrap_r          = idler_pitch_r + belt_thickness / 2  # belt centerline radius at wrap


def _expand_wraps(waypoints, step_deg=10):
    """Convert a list of (position, radius) waypoints into a denser polyline
    that has straight tangent segments between waypoints and short
    polyline approximations of the wrap arcs at each interior waypoint.

    Each interior waypoint is wrapped with its own radius (so clamp
    "pins" with r=2 mm coexist with idler wraps at the GT2 pitch
    radius). Endpoints have radius None — no wrap. Wrap direction (CCW
    vs CW in XZ) is picked from the path's turn at that waypoint —
    positive cross product of (entry-direction × exit-direction) ⇒ CCW.

    Y is preserved per waypoint; the connecting segments carry any
    Y-twist between waypoints.
    """
    def _norm_xz(vx, vz):
        n = math.sqrt(vx * vx + vz * vz)
        return (vx / n, vz / n) if n > 1e-9 else (0.0, 0.0)

    n = len(waypoints)
    if n < 2:
        return [p for p, _ in waypoints]

    out = [waypoints[0][0]]
    for i in range(1, n - 1):
        A = waypoints[i - 1][0]
        C, r = waypoints[i]
        B = waypoints[i + 1][0]
        D_in_x,  D_in_z  = _norm_xz(C[0] - A[0], C[2] - A[2])
        D_out_x, D_out_z = _norm_xz(B[0] - C[0], B[2] - C[2])
        cross = D_in_x * D_out_z - D_in_z * D_out_x
        ccw = cross >= 0   # default CCW for the 180° (cross ≈ 0) case
        if ccw:
            in_rx, in_rz   = ( D_in_z,  -D_in_x)
            out_rx, out_rz = ( D_out_z, -D_out_x)
        else:
            in_rx, in_rz   = (-D_in_z,   D_in_x)
            out_rx, out_rz = (-D_out_z,  D_out_x)

        theta_in  = math.atan2(in_rz,  in_rx)
        theta_out = math.atan2(out_rz, out_rx)
        if ccw:
            sweep = theta_out - theta_in
            if sweep <= 1e-6:
                sweep += 2 * math.pi
        else:
            sweep = theta_in - theta_out
            if sweep <= 1e-6:
                sweep += 2 * math.pi

        n_steps = max(2, int(math.degrees(sweep) / step_deg))
        for k in range(n_steps + 1):
            t = k / n_steps
            theta = theta_in + (sweep if ccw else -sweep) * t
            ax = C[0] + r * math.cos(theta)
            az = C[2] + r * math.sin(theta)
            ay = C[1]
            out.append((ax, ay, az))

    out.append(waypoints[-1][0])
    return out


def _perp_y(z_dir):
    """Direction perpendicular to z_dir with maximum +Y projection — used as
    the cross-section x-axis so the belt's width axis tracks world Y. For
    segments lying in world XZ this returns (0, 1, 0) exactly; segments
    that shift in Y get a small tilt so the section stays perpendicular
    to the centerline. Falls back to +X for the degenerate case where
    z_dir is parallel to world Y."""
    zx, zy, zz = z_dir
    px, py, pz = -zy * zx, 1 - zy * zy, -zy * zz
    length = math.sqrt(px * px + py * py + pz * pz)
    if length < 1e-6:
        return (1, 0, 0)
    return (px / length, py / length, pz / length)


class Belt(BaseStandardPart):
    """GT2 belt — rectangular cross-section (width × thickness) extruded
    along each segment of a polyline centerline, then unioned. Width axis
    tracks world Y (the pulley / idler axis direction); thickness is in
    the orthogonal in-plane direction.

    The ``motor`` argument selects which of the two CoreXY belts to draw:
      * ``"A"`` — driven by the LEFT motor (at world x ≈ -37.5 on
                  short_top). This is the path authored in
                  ``motor_a_path``: clamp_left → LJ2 → LD → LU.down2 →
                  motor_A → LU.down1 → RU.down2 → RJ1 → clamp_right.
      * ``"B"`` — driven by the RIGHT motor (at world x ≈ +37.5). The
                  path is the world-X mirror of motor A's path plus a
                  Y shift down to the UPPER belt plane, since CoreXY
                  needs the two belts in distinct Y planes to avoid
                  interference. Motor A rides Y = -42.75 (LU.down1/2,
                  LD, LJ2, RJ1, motor A pulley); motor B rides
                  Y = -51.75 (LU.top1, RU.top1, LJ1, RJ2, motor B pulley)
                  — 9 mm offset. The mirror+shift maps the joint
                  waypoints to their physical counterparts exactly:
                  LJ2 ↔ RJ2 (both on extra_hole2), RJ1 ↔ LJ1 (both on
                  extra_hole), LU.down2 ↔ RU.top1 and RU.down2 ↔
                  LU.top1 (the 1-idler / 2-stack columns swap Z between
                  LU and RU because the blocks are placed with opposite
                  x_dir). LD / LU.down1 have no physical upper-plane
                  counterpart (the bottom corner and the LU 2-stack
                  column at Z=304 don't have upper twins), so motor B
                  routes through empty space at those waypoints — fine
                  for illustration, not a faithful CoreXY motor B path."""

    def __init__(self, path, name="", qty=1, motor="A"):
        super().__init__(qty=qty)
        if motor not in ("A", "B"):
            raise ValueError(f"Belt motor must be 'A' or 'B', got {motor!r}")
        if motor == "B":
            # Motor B's belt is the world-X mirror of motor A's belt
            # PLUS a Y shift down to the upper belt plane. CoreXY needs
            # the two belts in distinct Y planes so they don't interfere.
            # Motor A rides Y = _lower_belt_y (LU.down1 / down2, LD, LJ2,
            # RJ1, motor A pulley); motor B rides Y = _upper_belt_y
            # (LU.top1, RU.top1, LJ1, RJ2, motor B pulley).
            y_shift = _upper_belt_y - _lower_belt_y   # -9 mm
            path = [
                ((-p[0], p[1] + y_shift, p[2]), r) for p, r in path
            ]
        self.path = path
        self.name = name
        self.motor = motor

    def name_suffix(self) -> str:
        return f"_{self.name}_x{self.qty}" if self.name else f"_x{self.qty}"

    def bom_key(self):
        # Belt is a continuous strip cut to length; one BOM line per machine.
        return ("Belt", "GT2", "loop")

    def geom_key(self):
        # bom_key collapses A/B belts into one purchasable line, but the
        # geometry differs per path + motor (motor='B' mirrors and Y-shifts
        # the path inside __init__). Without this override the cache would
        # return motor A's geometry for a motor B build.
        return (
            "Belt",
            self.motor,
            tuple((tuple(p), r) for p, r in self.path),
        )

    def _build(self):
        # One rectangular section extruded per centerline segment, then
        # unioned. Extruding each segment *inside a single BuildPart* booleans
        # it into the running solid — ~120 sequential fuses, O(n²), ~19 s.
        # Collecting the segment solids and doing ONE n-ary fuse at the end is
        # geometrically identical (same faces / volume) but ~50× faster.
        expanded = _expand_wraps(self.path)
        segments = []
        for i in range(len(expanded) - 1):
            p0 = expanded[i]
            p1 = expanded[i + 1]
            dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
            length = math.sqrt(dx * dx + dy * dy + dz * dz)
            if length < 1e-6:
                continue
            z_dir = (dx / length, dy / length, dz / length)
            section_plane = Plane(origin=p0, x_dir=_perp_y(z_dir), z_dir=z_dir)
            with BuildSketch(section_plane) as section:
                Rectangle(belt_width, belt_thickness)
            segments.append(extrude(section.sketch, amount=length))
        return segments[0].fuse(*segments[1:]) if len(segments) > 1 else segments[0]


# ── Motor A belt route — left motor, LOWER belt plane ─────────────────────────
# Open path: both ends clamped on the X-carriage (the belt is one
# continuous strip; the motor pulls one end and releases the other to
# move the carriage). Each waypoint sits at the center of the
# pulley / idler the belt wraps; straight-line segments between, no arc
# detail at the wraps.
#
# ── Derived geometry ──────────────────────────────────────────────────────────
# Every position below is computed from imported module constants — no
# hand-tuned numbers. See the docstrings on each block for the chain
# of derivations.

_half_w           = SHORT_LENGTH / 2 + EXT_THICKNESS / 2
_pulley_flange_h  = (flange_belt_h - pulley_belt_width) / 2
_lower_ring_h     = RING_SPECS["M5x8x0.5"]["height"]
_motor_ring_h     = RING_SPECS[MO10Bracket.RING_SPEC]["height"]   # Motor A spacer

# Idler belt-band center offset (along the shoulder bolt) from the block
# top face. The corner-idler stack on each block is ring (M5×8×0.5) +
# idler; belt-band center sits at ring + flange + belt_width/2 above the
# block top. The upper idler in a 2-stack column is one full idler +
# one more ring above that.
_idler_band_offset_lower = _lower_ring_h + _pulley_flange_h + pulley_belt_width / 2
_idler_band_offset_upper = _idler_band_offset_lower + flange_belt_h + _lower_ring_h

# Lower-belt-plane Y, shared by every corner idler. Block centerline
# Y = -EXT_THICKNESS - block_thickness/2 (block bottom flush on slot);
# block top face Y = block_centerline_y - block_thickness/2 (native +Z →
# world -Y); idler band Y subtracts the stack offset.
_lower_belt_y = (
    -EXT_THICKNESS - lu_block_thickness - _idler_band_offset_lower
)
_upper_belt_y = _lower_belt_y - flange_belt_h - _lower_ring_h

# LU block placement (idler_12_lu): origin Z on long_left's top t-nut;
# native +X → world -Z so the column at native x=-outer_hole_offset
# lands at world z = LU_block_z + outer_hole_offset.
_lu_block_z       = LONG_LENGTH - LONG_TOP_GAP
_lu_left_col_z    = _lu_block_z + outer_hole_offset    # 2-stack column
_lu_right_col_z   = _lu_block_z - outer_hole_offset    # 1-idler column

# RU mirrors: native +X → world +Z (idler_22_ru), so the columns swap
# world-Z assignment relative to LU.
_ru_left_col_z    = _lu_block_z - outer_hole_offset    # 2-stack column
_ru_right_col_z   = _lu_block_z + outer_hole_offset    # 1-idler column

LU_down1 = (-_half_w, _lower_belt_y, _lu_left_col_z)
LU_top1  = (-_half_w, _upper_belt_y, _lu_left_col_z)
LU_down2 = (-_half_w, _lower_belt_y, _lu_right_col_z)

RU_down1 = (+_half_w, _lower_belt_y, _ru_left_col_z)
RU_top1  = (+_half_w, _upper_belt_y, _ru_left_col_z)
RU_down2 = (+_half_w, _lower_belt_y, _ru_right_col_z)

# LD / RD: IdlerMountFront block at the bottom t-nut; idler sits on the
# block's top-face M4 hole at native (0, top_hole_y, …) — y_dir=(0,0,1)
# in placement maps native +Y → world +Z, so idler Z = block_z + top_hole_y.
_ld_block_z = LONG_BOT_GAP - ld_slot_center_y
_ld_idler_z = _ld_block_z + ld_top_hole_y
LD = (-_half_w, _lower_belt_y, _ld_idler_z)
RD = (+_half_w, _lower_belt_y, _ld_idler_z)

# Motor A pulley — left motor on short_top, LEFT_PULLEY_GAP above the
# bracket pad. Walk the placement chain (motor_10_bracket / motor_11_frame
# / motor_30_pulley) to land the belt-band center.
_left_t1_x      = -SHORT_LENGTH / 2 + SHORT_TOP_END_GAP
_left_t2_x      = _left_t1_x + SHORT_TOP_INNER_GAP
_motor_a_x      = (_left_t1_x + _left_t2_x) / 2
_motor_top_z_n  = motor_default_height / 2
_bracket_top_n  = _motor_top_z_n + motor_plate_thick
_bracket_dx     = motor_plate_length / 2 - motor_shaft_x_offset
_motor_m5_x_n   = _bracket_dx + motor_plate_length / 2 - motor_m5_x_inset
_motor_a_z      = (LONG_LENGTH - cb_end_offset) - _motor_m5_x_n
# Motor placement origin Y (motor_11_frame) → pulley_plane origin Y by
# stepping over the bracket thickness; pulley belt-band sits one
# LEFT_PULLEY_GAP + hub + flange + belt/2 further along world -Y.
_motor_origin_y = -EXT_THICKNESS - _motor_ring_h + _motor_top_z_n
_pulley_plane_y = _motor_origin_y - _bracket_top_n
_pulley_band_n  = pulley_hub_height + _pulley_flange_h + pulley_belt_width / 2
_motor_a_y      = _pulley_plane_y - LEFT_PULLEY_GAP - _pulley_band_n
MOTOR_A         = (_motor_a_x, _motor_a_y, _motor_a_z)

# Joint idlers (LJ2, RJ1) — both bundles are LI4x_idler_lj/rj, no spacer,
# so the belt-band center sits _idler_band_offset_lower above the joint
# top face (along world -Y after the bundle's z_dir=(0,-1,0) placement).
# Walk linear_11_y → linear_20_joint to land the joint top hole world XYZ.
_joint_csk_ll_x   = -joint_length / 2 + joint_csk_hole_from_left
_joint_csk_ur_x   = _joint_csk_ll_x + joint_csk_x_spacing
_joint_csk_ll_y   = -joint_width / 2 + joint_csk_hole_from_bottom
_joint_csk_ur_y   = _joint_csk_ll_y + joint_csk_y_spacing
_joint_grid_y     = (_joint_csk_ll_y + _joint_csk_ur_y) / 2
_joint_grid_x_abs = abs((_joint_csk_ll_x + _joint_csk_ur_x) / 2)
# (mirrored joint contributes +abs to origin X, non-mirrored contributes -abs)

# Slider mount center (linear_11_y) — depends on rail-center Z (sandwiched
# between LU bottom edge and LD top edge) plus slider_position along the
# 220 mm rail.
_rail_center_z   = ((LONG_LENGTH - LONG_TOP_GAP) - lu_block_length / 2
                    + _ld_block_z + ld_block_width / 2) / 2
_slider_x_offset = -RAIL_LENGTH / 2 + slider_position * RAIL_LENGTH
_slider_z        = _rail_center_z + _slider_x_offset
_slider_y        = -EXT_THICKNESS - slider_top_z

# LEFT joint (mirrored): native +X flips, so grid_center_x = +abs.
# native_to_world for mirrored: world = (origin_x + nx, origin_y - nz, origin_z - ny).
# Joint top face is at native z = +joint_thickness/2, so its world Y =
# origin_y - joint_thickness/2 (joint origin_y = slider_y - joint_thickness/2).
_left_joint_origin_x = -_half_w + _joint_grid_x_abs
_left_joint_origin_y = _slider_y - joint_thickness / 2
_left_joint_origin_z = _slider_z + _joint_grid_y
_lj_extra2_world = (
    _left_joint_origin_x + joint_extra_hole2_x_n,
    _left_joint_origin_y - joint_thickness / 2,
    _left_joint_origin_z - joint_extra_hole2_y_n,
)
LJ2 = (
    _lj_extra2_world[0],
    _lj_extra2_world[1] - _idler_band_offset_lower,
    _lj_extra2_world[2],
)

# RIGHT joint (non-mirrored): grid_center_x = -abs (since CSK X's are
# both negative). native_to_world: world = (origin_x - nx, origin_y - nz,
# origin_z - ny).
_right_joint_origin_x = +_half_w - _joint_grid_x_abs
_right_joint_origin_y = _slider_y - joint_thickness / 2
_right_joint_origin_z = _slider_z + _joint_grid_y
_rj_extra_world = (
    _right_joint_origin_x - joint_extra_hole_x_n,
    _right_joint_origin_y - joint_thickness / 2,
    _right_joint_origin_z - joint_extra_hole_y_n,
)
RJ1 = (
    _rj_extra_world[0],
    _rj_extra_world[1] - _idler_band_offset_lower,
    _rj_extra_world[2],
)

# Clamp pin radius — small 1.8 mm pin inside each belt clamp; the belt
# loops around it 180° to lock the belt's tip into the clamp.
clamp_pin_r = 2 * MM

# Belt clamp pins and tips. The belt tip enters the clamp going +X (left
# clamp) / -X (right clamp), wraps 180° around the pin, then exits going
# back the opposite way to the adjacent joint idler.
# Z geometry chains: clamp tip Z - clamp_pin_r = pin Z (clamp tip rides
# ABOVE the pin so the U bends up in top view), and pin Z - clamp_pin_r
# = adjacent idler's tangent Z (= LJ2.z + wrap_r on the left / RJ1.z -
# wrap_r on the right). Solve → pin Z = idler_tangent_z + r.
_pin_left_z  = LJ2[2] + wrap_r + clamp_pin_r
_pin_right_z = RJ1[2] - wrap_r + clamp_pin_r

PIN_LEFT     = (-4.0, _lower_belt_y, _pin_left_z)
PIN_RIGHT    = (+4.0, _lower_belt_y, _pin_right_z)

CLAMP_LEFT   = (-20.0, _lower_belt_y, _pin_left_z  + clamp_pin_r)
CLAMP_RIGHT  = (+20.0, _lower_belt_y, _pin_right_z + clamp_pin_r)

# Path: list of (position, wrap_radius) tuples. radius=None for endpoints.
motor_a_path = [
    (CLAMP_LEFT, None),
    (PIN_LEFT,   clamp_pin_r),
    (LJ2,        wrap_r),
    (LD,         wrap_r),
    (LU_down2,   wrap_r),
    (MOTOR_A,    wrap_r),
    (LU_down1,   wrap_r),
    (RU_down2,   wrap_r),
    (RJ1,        wrap_r),
    (PIN_RIGHT,  clamp_pin_r),
    (CLAMP_RIGHT, None),
]


if __name__ == "__main__":
    Belt(path=motor_a_path, name="motor_a", motor="A").export()
    Belt(path=motor_a_path, name="motor_b", motor="B").export()
