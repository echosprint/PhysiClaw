import math

from build123d import *

from hardware.parts._fits import (
    CSK_ANGLE,
    M3_NORMAL,
    M3_NUT_W,
    M5_CSK_HEAD,
    M5_NORMAL,
)
from hardware.parts.base import BaseCustomPart

# ── Plate ─────────────────────────────────────────────────────────────────────
thickness = 3 * MM

# ── PCB mounting pattern ──────────────────────────────────────────────────────
# PCB is 90 × 70 with four corner M3 holes spaced 82 × 62 (i.e. 4 mm in from
# each PCB edge). Holder mirrors that pattern about its center.
hole_pitch_x = 82 * MM
hole_pitch_y = 62 * MM
hole_x = hole_pitch_x / 2  # 41
hole_y = hole_pitch_y / 2  # 31
standoff_xy = [(sx, sy) for sx in (-hole_x, hole_x) for sy in (-hole_y, hole_y)]

# Standoff: the PCB rests on top of these; the screw passes through the bore
# and the plate into a square nut held against the underside.
cyl_d = 8 * MM
cyl_h = 4 * MM

# Square-nut locating mark: a shallow square recess on the plate's underside
# (opposite the cylinder side), centered under each bore — shows where to seat
# the M3 square nut. A locating mark, not a deep capture pocket.
nut_mark_size = M3_NUT_W  # square nut width across flats
nut_mark_depth = 1.0 * MM

# Plate sized to enclose the standoffs with a small margin.
plate_margin = 3 * MM
plate_half_x = hole_x + cyl_d / 2 + plate_margin  # 48 → 96 mm wide
plate_half_y = hole_y + cyl_d / 2 + plate_margin  # 38 → 76 mm deep

# Mounting rib on the underside (the square-nut side): runs along X, set in
# rib_gap from the +X short edge, centered in Y. Protrudes 3 mm.
tab_x = 38 * MM  # along X — the rib's long side
tab_y = 10 * MM  # along Y — the rib's short side
tab_thick = 3 * MM  # protrudes below the plate (-Z)
rib_gap = 9 * MM  # gap between the rib's +X end and the plate short edge
rib_cx = plate_half_x - rib_gap - tab_x / 2  # rib center in X
rib_cy = 0 * MM  # rib center in Y

# 2040 mount: two M5 countersink holes through plate + rib, recessed on the
# cylinder side (top), threading into the extrusion T-slot below the rib.
mount_hole_pitch = 20 * MM  # along X, centered on the rib

# ── Lightening: hexagonal honeycomb ───────────────────────────────────────────
# The PCB rests on the four standoffs, not the plate, so the web between them
# carries almost nothing — honeycomb it aggressively (matching the phone bed).
# Pointy-top hex holes on an offset grid (uniform `hex_wall` between neighbours),
# filtered against the standoff rings, the rib footprint and a plate-edge
# border, so those stay solid.
hex_flat = 9 * MM  # hole flat-to-flat (small cells — the plate is
#                                  crowded with standoffs, so fine cells pack best)
hex_wall = 1 * MM  # wall between holes (thin PA12 rib)
cap_margin = 4 * MM  # plate-edge border
cap_keepout = cyl_d / 2 + 2.5 * MM  # clearance ring around each standoff
rib_keepout = 1.5 * MM  # clearance around the rib footprint (thin wall only)

# Relief slots flanking the rib along X. The hex grid can't seat a cell in the
# band just beside the rib (those rows overlap the rib/M5-countersink keep-out and
# get dropped), leaving a solid strip. Two stadium slots — one each side, in the
# plate beside the rib (clear of the rib's y∈[-tab_y/2, tab_y/2] and the y=0
# countersinks) — open that strip up.
rib_slot_len = 32 * MM
rib_slot_wid = 5 * MM
rib_slot_y = tab_y / 2 + rib_slot_wid / 2 + 1 * MM  # ± offset (≈8.5): 1 mm wall to rib

# Round the four plate corners (d = 4 mm → radius 2 mm).
corner_d = 4 * MM
edge_tol = 0.5 * MM  # fuzz for matching fillet edges by center


def standoff_locs(z):
    """The four standoff (x, y) positions lifted to height z."""
    return [(sx, sy, z) for sx, sy in standoff_xy]


def _lightening_centers():
    pitch = hex_flat + hex_wall  # nearest-neighbour centre distance
    hx = hex_flat / 2  # horizontal half-extent (apothem)
    hy = hex_flat / math.sqrt(3)  # vertical half-extent (vertex, pointy-top)
    dx, dy = pitch, pitch * math.sqrt(3) / 2
    # Solid keep-outs (rects): standoff rings and the rib footprint.
    keepouts = [
        (sx - cap_keepout, sx + cap_keepout, sy - cap_keepout, sy + cap_keepout)
        for sx, sy in standoff_xy
    ]
    keepouts.append(
        (
            rib_cx - tab_x / 2 - rib_keepout,
            rib_cx + tab_x / 2 + rib_keepout,
            rib_cy - tab_y / 2 - rib_keepout,
            rib_cy + tab_y / 2 + rib_keepout,
        )
    )
    nx = int(plate_half_x / dx) + 2
    ny = int(plate_half_y / dy) + 2
    centers = []
    for j in range(-ny, ny + 1):
        cy = j * dy
        x_off = pitch / 2 if j % 2 else 0  # honeycomb row offset
        for i in range(-nx, nx + 1):
            cx = i * dx + x_off
            if (
                cx - hx < -plate_half_x + cap_margin
                or cx + hx > plate_half_x - cap_margin
                or cy - hy < -plate_half_y + cap_margin
                or cy + hy > plate_half_y - cap_margin
            ):
                continue
            if any(
                cx - hx < x1 and cx + hx > x0 and cy - hy < y1 and cy + hy > y0
                for x0, x1, y0, y1 in keepouts
            ):
                continue
            centers.append((cx, cy))
    return centers


class PcbHolder(BaseCustomPart):
    def _build(self):
        with BuildPart() as my_part:
            # Base plate: z = 0 .. thickness.
            Box(
                2 * plate_half_x,
                2 * plate_half_y,
                thickness,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            )

            # Mounting rib on the underside, centered at (rib_cx, rib_cy),
            # protruding -Z by tab_thick.
            with Locations((rib_cx, rib_cy, 0)):
                Box(
                    tab_x,
                    tab_y,
                    tab_thick,
                    align=(Align.CENTER, Align.CENTER, Align.MAX),
                )

            # Standoff cylinders on top: z = thickness .. thickness + cyl_h.
            with Locations(*standoff_locs(thickness)):
                Cylinder(
                    cyl_d / 2, cyl_h, align=(Align.CENTER, Align.CENTER, Align.MIN)
                )

            # M3 clearance bore through each standoff + plate.
            with Locations(*standoff_locs(thickness + cyl_h)):
                Hole(radius=M3_NORMAL / 2)

            # Square-nut locating recess on the underside, centered under each bore.
            with Locations(*standoff_locs(0)):
                Box(
                    nut_mark_size,
                    nut_mark_size,
                    nut_mark_depth,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                    mode=Mode.SUBTRACT,
                )

            # 2040 mount countersinks — recessed on the cylinder-side (top) face,
            # bored down through plate + rib. Centered along the rib, 20 mm apart.
            with Locations(
                (rib_cx - mount_hole_pitch / 2, rib_cy, thickness),
                (rib_cx + mount_hole_pitch / 2, rib_cy, thickness),
            ):
                CounterSinkHole(
                    radius=M5_NORMAL / 2,
                    counter_sink_radius=M5_CSK_HEAD / 2,
                    counter_sink_angle=CSK_ANGLE,
                )

            # Honeycomb lightening through the plate (one sketch, one subtract).
            # Pointy-top hexes: apothem = hex_flat/2, rotated 30° from the
            # build123d default so flats face left/right.
            with BuildSketch(Plane.XY):
                with Locations(*_lightening_centers()):
                    RegularPolygon(
                        radius=hex_flat / 2,
                        side_count=6,
                        major_radius=False,
                        rotation=30,
                    )
                # Relief slots beside the rib (along X).
                with Locations((rib_cx, rib_slot_y), (rib_cx, -rib_slot_y)):
                    SlotOverall(rib_slot_len, rib_slot_wid)
            extrude(amount=thickness, mode=Mode.SUBTRACT)

            # Round the four plate corners.
            plate_corners = [
                e
                for e in my_part.edges().filter_by(Axis.Z)
                if abs(abs(e.center().X) - plate_half_x) < edge_tol
                and abs(abs(e.center().Y) - plate_half_y) < edge_tol
            ]
            fillet(plate_corners, radius=corner_d / 2)

        return my_part.part


if __name__ == "__main__":
    PcbHolder().export()
