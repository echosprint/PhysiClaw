import math

from build123d import *

from hardware.parts._fits import CSK_ANGLE, M3_CSK_HEAD, M3_NORMAL
from hardware.parts.base import BaseCustomPart

# ── Plate ─────────────────────────────────────────────────────────────────────
thickness = 3 * MM

# Key plate dimensions — the quarter outline below is built from these, so the
# ear length (how far the ±X mounting tabs jut out) is the single knob
# `ear_x_outer`. body_half_x/y bound the main body (the lightening region).
body_half_x = 45 * MM  # body half-width in X (excludes the ears)
body_half_y = 87.5 * MM  # body half-height in Y
ear_y_lo = 40 * MM  # each ear spans y ∈ [ear_y_lo, ear_y_hi]
ear_y_hi = 60 * MM
ear_x_outer = 57 * MM  # ear outer edge in X (juts out ear_x_outer − body_half_x)

# Quarter outline in the +X/+Y quadrant (XY plane), mirrored across XZ then
# YZ to build the full plate. Edges on x=0 and y=0 sit on the mirror lines.
quarter_vertices = (
    (0, 0),
    (body_half_x, 0),
    (body_half_x, ear_y_lo),
    (ear_x_outer, ear_y_lo),
    (ear_x_outer, ear_y_hi),
    (body_half_x, ear_y_hi),
    (body_half_x, body_half_y),
    (0, body_half_y),
)

# ── Lightening cuts: hexagonal honeycomb ──────────────────────────────────────
# The bed is screwed to two 1020 cross-beams that sit under the ear rows (y≈±50);
# a rigid, light phone bridges the gap between them, so the body needs very
# little material — only enough to register and carry the phone to those beams.
# A honeycomb gives the highest void fraction behind thin walls (so it's cheap to
# print) while staying stiff and reading as an intentional, "designed" surface.
# The 3 mm walls bear straight down onto the beams, so support is unaffected.
#
# Pointy-top hex holes on an offset grid: every nearest neighbour sits one
# `pitch` away, so the wall between any two holes is exactly `hex_wall`. Centres
# whose hole would breach the body margin are dropped, leaving a solid perimeter
# (where the alignment walls and ears live).
hex_flat = 18 * MM  # hole flat-to-flat (the horizontal width of each cell)
hex_wall = 2 * MM  # wall thickness between holes (PA12 rib)
edge_margin = 5 * MM  # clear border kept around every hole (leaves solid under
#                          the alignment walls: top-vertex clearance ≈ 4.8 mm)

# ── Alignment walls ───────────────────────────────────────────────────────────
# Upstanding fences on the top (+Y) and left (-X) body edges. The phone rests
# on the bed's top surface and is pushed into the top-left corner; its two
# edges register against the inner faces of these walls. Each wall is flush
# with the outer perimeter and grows inward by its thickness + upward by its
# height; they overlap at the corner and fuse into a clean L.
wall_thickness = 3 * MM
wall_height = 10 * MM
low_wall_height = 5 * MM  # short retaining lips on the open (right + bottom) sides
wall_align = (Align.CENTER, Align.CENTER, Align.MIN)  # walls/lips seat on the top face

# Truss the tall walls: stadium windows through the wall, leaving continuous
# top/bottom rails + posts between windows — lighter, still aligns the phone.
window_length = 20 * MM  # stadium length along the wall
window_height = 5 * MM  # stadium height (≈2.5 mm rail above and below)
top_window_x = (-30, 0, 30)  # window centers on the top wall
left_window_y = (-60, -30, 0, 30, 60)  # window centers on the left wall

# Ear-derived positions (ear_y_lo/hi, ear_x_outer defined with the outline above).
ear_y_center = (ear_y_lo + ear_y_hi) / 2  # 50
ear_y_width = ear_y_hi - ear_y_lo  # 20, same as the ear
ear_x_center = (body_half_x + ear_x_outer) / 2  # ear-tab (mounting hole) center in X
ear_fillet_radius = 3 * MM  # round the outer ear corners
br_fillet_radius = 5 * MM  # round the bottom-right body corner
wall_fillet_radius = 1.0 * MM  # wall + lip vertical corners (capped
#   at 1.0: 1.5 overruns the narrow face where the upper lip meets the top wall)
edge_tol = 0.5 * MM  # fuzz for matching fillet edges by center

# Bottom-left lip: runs +X from the left corner along the bottom edge,
# fusing to the foot of the left wall.
bottom_lip_length = 10 * MM  # from x=-45 to x=-35

# Lower-right lip (the +X edge segment at the bottom ear): 16 mm, shorter than
# the 20 mm ear so its ends sit 2 mm inside the ear corners — otherwise the
# lip's vertical fillet runs into the ear corner and leaves an artifact. Stays
# centered on the ear (y = -ear_y_center).
right_lip_length = 16 * MM


def _honeycomb_centers():
    pitch = hex_flat + hex_wall  # nearest-neighbour centre distance
    apothem = hex_flat / 2  # half flat-to-flat (horizontal half-extent)
    circum = hex_flat / math.sqrt(3)  # vertex radius (vertical half-extent, pointy-top)
    dx = pitch  # columns within a row
    dy = pitch * math.sqrt(3) / 2  # row-to-row, alternating half-pitch offset
    # Drop any hole whose footprint would breach the body margin → solid border.
    x_lim = body_half_x - edge_margin - apothem
    y_lim = body_half_y - edge_margin - circum
    rows = int(body_half_y / dy) + 2
    cols = int(body_half_x / dx) + 2
    centers = []
    for j in range(-rows, rows + 1):
        cy = j * dy
        x_off = pitch / 2 if (j & 1) else 0.0
        for i in range(-cols, cols + 1):
            cx = i * dx + x_off
            if abs(cx) <= x_lim and abs(cy) <= y_lim:
                centers.append((cx, cy))
    return centers


class PhoneBed(BaseCustomPart):
    def _build(self):
        with BuildPart() as my_part:
            with BuildSketch(Plane.XY):
                with BuildLine():
                    Polyline(*quarter_vertices, close=True)
                make_face()
                mirror(about=Plane.XZ)  # reflect +Y quarter to -Y → half
                mirror(about=Plane.YZ)  # reflect +X half to -X → full
            extrude(amount=thickness)

            # Lightening honeycomb — one sketch, one boolean subtract. Pointy-top
            # hexes (vertex up): apothem = hex_flat/2, rotated 30° from the
            # build123d default (which is flat-top) so flats face left/right.
            with BuildSketch(Plane.XY):
                with Locations(*_honeycomb_centers()):
                    RegularPolygon(
                        radius=hex_flat / 2,
                        side_count=6,
                        major_radius=False,
                        rotation=30,
                    )
            extrude(amount=thickness, mode=Mode.SUBTRACT)

            # Alignment walls — sit ON the top surface (align MIN in Z) so they
            # rise from z=thickness to z=thickness+wall_height.
            top_face = thickness
            # Top wall along the +Y edge, inner face at y = body_half_y - wall_thickness
            with Locations((0, body_half_y - wall_thickness / 2, top_face)):
                Box(2 * body_half_x, wall_thickness, wall_height, align=wall_align)
            # Left wall along the -X edge, inner face at x = -body_half_x + wall_thickness
            with Locations((-body_half_x + wall_thickness / 2, 0, top_face)):
                Box(wall_thickness, 2 * body_half_y, wall_height, align=wall_align)

            # Capsule windows through the tall walls (cut on each wall's
            # mid-plane, extruded both ways through the 3 mm thickness only —
            # so the cut can't reach any other feature).
            wall_z_mid = thickness + wall_height / 2
            top_plane = Plane(
                origin=(0, body_half_y - wall_thickness / 2, 0),
                x_dir=(1, 0, 0),
                z_dir=(0, -1, 0),
            )
            left_plane = Plane(
                origin=(-body_half_x + wall_thickness / 2, 0, 0),
                x_dir=(0, 1, 0),
                z_dir=(1, 0, 0),
            )
            for plane, centers in (
                (top_plane, top_window_x),
                (left_plane, left_window_y),
            ):
                with BuildSketch(plane):
                    with Locations(*[(c, wall_z_mid) for c in centers]):
                        SlotOverall(window_length, window_height)
                extrude(amount=wall_thickness, both=True, mode=Mode.SUBTRACT)

            # Low lips on the open sides — right edge and the bottom-left corner.
            # Lower-right ear segment on the +X edge, 16 mm (inset from the ear).
            with Locations((body_half_x - wall_thickness / 2, -ear_y_center, top_face)):
                Box(wall_thickness, right_lip_length, low_wall_height, align=wall_align)
            # Upper segment moved up to the top edge so it fuses with the top wall.
            with Locations(
                (
                    body_half_x - wall_thickness / 2,
                    body_half_y - ear_y_width / 2,
                    top_face,
                )
            ):
                Box(wall_thickness, ear_y_width, low_wall_height, align=wall_align)
            # Bottom-left lip on the -Y edge, fused to the foot of the left wall.
            with Locations(
                (
                    -body_half_x + bottom_lip_length / 2,
                    -body_half_y + wall_thickness / 2,
                    top_face,
                )
            ):
                Box(
                    bottom_lip_length, wall_thickness, low_wall_height, align=wall_align
                )

            # M3 countersunk through-hole in each ear — recess on the top face
            # so a flat-head screw seats flush with the phone-resting surface.
            with Locations(
                (ear_x_center, ear_y_center, thickness),
                (ear_x_center, -ear_y_center, thickness),
                (-ear_x_center, ear_y_center, thickness),
                (-ear_x_center, -ear_y_center, thickness),
            ):
                CounterSinkHole(
                    radius=M3_NORMAL / 2,
                    counter_sink_radius=M3_CSK_HEAD / 2,
                    counter_sink_angle=CSK_ANGLE,
                )

            # Round the four outer corners of each ear (the convex |x|=65 tabs).
            ear_corners = [
                e
                for e in my_part.edges().filter_by(Axis.Z)
                if abs(abs(e.center().X) - ear_x_outer) < edge_tol
                and min(
                    abs(abs(e.center().Y) - ear_y_lo), abs(abs(e.center().Y) - ear_y_hi)
                )
                < edge_tol
            ]
            fillet(ear_corners, radius=ear_fillet_radius)

            # Round the bottom-right body corner (+X / -Y).
            br_corner = [
                e
                for e in my_part.edges().filter_by(Axis.Z)
                if abs(e.center().X - body_half_x) < edge_tol
                and abs(e.center().Y + body_half_y) < edge_tol
            ]
            fillet(br_corner, radius=br_fillet_radius)

            # Round every vertical corner that rises into the wall band (the
            # tall walls' free ends + the retaining-lip corners), leaving the
            # plate-only edges untouched.
            wall_vedges = [
                e
                for e in my_part.edges().filter_by(Axis.Z)
                if e.bounding_box().max.Z > thickness + 0.1
            ]
            fillet(wall_vedges, radius=wall_fillet_radius)

        return my_part.part


if __name__ == "__main__":
    PhoneBed().export()
