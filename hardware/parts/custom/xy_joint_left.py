import functools

from build123d import *

from hardware.parts._fits import M3_NORMAL, M4_CLOSE, M4_NUT_T, M4_NUT_W, M5_NORMAL
from hardware.parts.base import BaseCustomPart

# ── Block dimensions ──────────────────────────────────────────────────────────
length = 41 * MM
width = 26 * MM
thickness = 8 * MM

# ── Top face: polygonal cutout, through the plate (upper-right corner area) ───
# Vertices (h, v) relative to the (-X, -Y) corner of the top face.
cutout_corner_vertices = (
    (22 * MM, 26 * MM),
    (41 * MM, 26 * MM),
    (41 * MM, 16 * MM),
    (32 * MM, 16 * MM),
)

# ── Top face: 2×2 grid of countersunk through-holes ───────────────────────────
csk_hole_diameter = M3_NORMAL
csk_head_diameter = 6.5 * MM
csk_hole_from_left = 4 * MM  # lower-left (LL) hole, from -X edge
csk_hole_from_bottom = 6 * MM  # lower-left (LL) hole, from -Y edge
csk_x_spacing = 15 * MM
csk_y_spacing = 16 * MM
csk_angle = 90  # degrees, all CSK holes

# ── Top face: extra through-hole, offset from the LL CSK hole ─────────────────
extra_hole_diameter = M4_CLOSE
extra_hole_dx_from_csk = 7.5 * MM
extra_hole_dy_from_csk = -2 * MM

# ── Top face: second extra through-hole, offset from the UR CSK hole ──────────
extra_hole2_diameter = M4_CLOSE
extra_hole2_dx_from_csk = 5.4 * MM
extra_hole2_dy_from_csk = -5.1 * MM

# ── Top face: bigger CSK hole, offset from the second extra hole ──────────────
big_csk_hole_diameter = M5_NORMAL
big_csk_head_diameter = 11 * MM
big_csk_dx_from_extra2 = 8.8 * MM
big_csk_dy_from_extra2 = -8.4 * MM

# ── Front + slant faces: rect pocket (same dimensions; depth differs) ────────
# Front pocket X-center = first extra hole's X.
# Slant pocket X-center = second extra hole projected onto the slant.
pocket_w = M4_NUT_W  # along each face's horizontal direction
pocket_h = M4_NUT_T  # along world Z
pocket_top_offset = 3.2 * MM  # rect top edge below top face
front_pocket_depth = 8 * MM
slant_pocket_depth = 9 * MM

# ── Bottom face: two extrude pads, mirrored about the big CSK hole ────────────
# Each pad is centered vertically on the hole's Y.
pad_w = 2.8 * MM  # along world X
pad_h = 6.3 * MM  # along world Y
pad_near_from_csk = 5 * MM  # near edge of each pad to big CSK center (in ±X)
pad_height = 2 * MM  # extrudes this far below the bottom face

# ── Fillets ───────────────────────────────────────────────────────────────────
outer_fillet_radius = 2 * MM  # 6 outer vertical edges (3 cube + 3 from cutout)
pad_fillet_radius = 0.6 * MM  # 8 pad vertical edges (4 per pad × 2 pads)

# ── Derived feature positions (centered native frame) ─────────────────────────
# Exposed at module level so downstream consumers (LI20Joint, etc.) can place
# parts at these features without re-deriving the layout math.
csk_ll_x = -length / 2 + csk_hole_from_left
csk_ll_y = -width / 2 + csk_hole_from_bottom
csk_ur_x = csk_ll_x + csk_x_spacing
csk_ur_y = csk_ll_y + csk_y_spacing

extra_hole_x = csk_ll_x + extra_hole_dx_from_csk
extra_hole_y = csk_ll_y + extra_hole_dy_from_csk
extra_hole2_x = csk_ur_x + extra_hole2_dx_from_csk
extra_hole2_y = csk_ur_y + extra_hole2_dy_from_csk
big_csk_x = extra_hole2_x + big_csk_dx_from_extra2
big_csk_y = extra_hole2_y + big_csk_dy_from_extra2

# Front + slant pockets share the same face-local Y center.
pocket_face_y = thickness / 2 - pocket_top_offset - pocket_h / 2

# Slant face — geometry derived from the cutout's 45° diagonal
# (cutout vertex [3] → [0]). slant_z_dir is the face outward normal
# (= 90° CW from x_dir, pointing into the polygon cutout).
slant_start = Vector(
    cutout_corner_vertices[3][0] - length / 2,
    cutout_corner_vertices[3][1] - width / 2,
    0,
)
slant_end = Vector(
    cutout_corner_vertices[0][0] - length / 2,
    cutout_corner_vertices[0][1] - width / 2,
    0,
)
slant_origin = (slant_start + slant_end) * 0.5
slant_x_dir = (slant_end - slant_start).normalized()
slant_z_dir = Vector(slant_x_dir.Y, -slant_x_dir.X, 0)

# Slant-pocket face-X — projection of the second extra hole onto slant_x_dir.
slant_pocket_face_x = (Vector(extra_hole2_x, extra_hole2_y, 0) - slant_origin).dot(
    slant_x_dir
)


@functools.cache
def _build_shape():
    """Cached so XyJointRight can mirror the same Part without rebuilding."""
    with BuildPart() as my_part:
        Box(length, width, thickness)

        # Top: polygonal cutout, through the full thickness.
        top_plane = Plane.XY.offset(thickness / 2)
        cutout_face_pts = [
            (h - length / 2, v - width / 2) for h, v in cutout_corner_vertices
        ]
        with BuildSketch(top_plane):
            with BuildLine():
                Polyline(*cutout_face_pts, close=True)
            make_face()
        extrude(amount=-thickness, mode=Mode.SUBTRACT)

        # Top: 2×2 grid of CSK holes. The lower-left (LL) hole of the grid sits at
        # corner-relative (csk_hole_from_left, csk_hole_from_bottom); the upper-right
        # (UR) hole is one spacing diagonally above.
        with Locations(
            ((csk_ll_x + csk_ur_x) / 2, (csk_ll_y + csk_ur_y) / 2, thickness / 2)
        ):
            with GridLocations(csk_x_spacing, csk_y_spacing, 2, 2):
                CounterSinkHole(
                    radius=csk_hole_diameter / 2,
                    counter_sink_radius=csk_head_diameter / 2,
                    counter_sink_angle=csk_angle,
                )

        # Top: extra through-hole, offset from LL CSK
        with Locations((extra_hole_x, extra_hole_y, thickness / 2)):
            Hole(radius=extra_hole_diameter / 2)

        # Top: second extra through-hole, offset from UR CSK
        with Locations((extra_hole2_x, extra_hole2_y, thickness / 2)):
            Hole(radius=extra_hole2_diameter / 2)

        # Top: bigger CSK, offset from the second extra hole
        with Locations((big_csk_x, big_csk_y, thickness / 2)):
            CounterSinkHole(
                radius=big_csk_hole_diameter / 2,
                counter_sink_radius=big_csk_head_diameter / 2,
                counter_sink_angle=csk_angle,
            )

        # Front face: rect pocket, X-center = first extra hole's X. Both pockets
        # share pocket_face_y as the face-local Y center (both planes have
        # origin Z = 0 and y_dir = world +Z, so face_y = world Z).
        front_plane = Plane(
            origin=(0, -width / 2, 0),
            x_dir=(1, 0, 0),
            z_dir=(0, -1, 0),
        )
        with BuildSketch(front_plane):
            with Locations((extra_hole_x, pocket_face_y)):
                Rectangle(pocket_w, pocket_h)
        extrude(amount=-front_pocket_depth, mode=Mode.SUBTRACT)

        # Slant face: rect pocket, X-center = second extra hole's projection
        # onto slant_x_dir (already computed at module level as slant_pocket_face_x).
        slant_plane = Plane(origin=slant_origin, x_dir=slant_x_dir, z_dir=slant_z_dir)
        with BuildSketch(slant_plane):
            with Locations((slant_pocket_face_x, pocket_face_y)):
                Rectangle(pocket_w, pocket_h)
        extrude(amount=-slant_pocket_depth, mode=Mode.SUBTRACT)

        # Bottom: two pads mirrored about the big CSK hole, centered on the hole's Y.
        bottom_plane = Plane.XY.offset(-thickness / 2)
        with BuildSketch(bottom_plane):
            with Locations(
                (big_csk_x - pad_near_from_csk - pad_w / 2, big_csk_y),
                (big_csk_x + pad_near_from_csk + pad_w / 2, big_csk_y),
            ):
                Rectangle(pad_w, pad_h)
        extrude(amount=-pad_height)

        # Fillet: 6 outer vertical edges (longest Z-parallel = thickness).
        outer_vertical_edges = (
            my_part.edges().filter_by(Axis.Z).group_by(Edge.length)[-1]
        )
        fillet(outer_vertical_edges, radius=outer_fillet_radius)

        # Fillet: 8 pad vertical edges — Z-parallel edges whose midpoint sits
        # below the cube's bottom face (i.e., inside the protruding pad volume).
        pad_vertical_edges = (
            my_part.edges()
            .filter_by(Axis.Z)
            .filter_by_position(
                Axis.Z,
                minimum=-thickness / 2 - pad_height,
                maximum=-thickness / 2,
            )
        )
        fillet(pad_vertical_edges, radius=pad_fillet_radius)

    return my_part.part


class XyJointLeft(BaseCustomPart):
    def _build(self):
        return _build_shape()


if __name__ == "__main__":
    XyJointLeft().export()
