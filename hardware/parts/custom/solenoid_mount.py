from build123d import *

from hardware.parts._fits import M3_NORMAL
from hardware.parts.base import BaseCustomPart

# ── Plate dimensions (the "keyboard") ─────────────────────────────────────────
length = 40 * MM
width = 29 * MM
thickness = 2 * MM

# ── Back wall (the "screen"), extruded up from the +Y edge of the plate ───────
wall_thickness = 4 * MM  # along Y
wall_height = 45 * MM  # extruded up (+Z); tall enough for the +10 mm-shifted
# pattern (top hole ≈ 37 mm) with the top slots above it
wall_center_y = width / 2 - wall_thickness / 2

# ── Keyboard face: 4 corner through-holes ─────────────────────────────────────
keyboard_hole_diameter = M3_NORMAL
keyboard_rect_w = length - 8 * MM  # 32 mm
keyboard_rect_h = width - 12 * MM  # 17 mm
keyboard_rect_bottom_offset = 4 * MM  # rect bottom above -Y edge of face
keyboard_rect_center_y = -width / 2 + keyboard_rect_bottom_offset + keyboard_rect_h / 2

# Keyboard face: rectangular through-slot at the screen base — its +Y edge sits
# on the keyboard↔screen connection inner edge; cut clean through the plate.
keyboard_slot_w = 8 * MM  # along X (centered)
keyboard_slot_h = 3 * MM  # along Y, from the connection edge toward -Y

# Keyboard face: Ø4.3 through-hole for the PTFE tube (OD4), just below the wire
# slot. The tube plugs through here and continues into the BeltClamp blind hole
# beneath; the solenoid wire (spiral-wrapped to the tube) runs through the slot.
tube_hole_diameter = 4.3 * MM  # OD4 PTFE tube, slip fit
tube_hole_x = 0 * MM  # centered, below the keyboard_slot
tube_hole_y = 4 * MM  # face-local Y (keyboard_slot bottom ≈ 6.5)

# ── Screen face: paired hole pattern (N bases × 2 row-shifts × 2 paired rows) ─
# Hole count = 4 × len(screen_pattern_y_offsets).
screen_pattern_hole_diameter = M3_NORMAL
screen_pattern_spacing = 4 * MM  # face-local X between consecutive circles
screen_pattern_base_from_left = (
    12 * MM
)  # X of first circle (= base) from left edge of face
screen_pattern_base_from_bottom = (
    13 * MM
)  # base-row Y from bottom edge of face (= keyboard join)
screen_pattern_y_offsets = (
    3,
    0,
    2,
    4,
    1,
)  # mm above base-row, one per circle (i=0 is first)
screen_pattern_pair_offset = 15 * MM  # paired hole sits this far above each circle
screen_pattern_row_shift = 5 * MM  # second 10-hole row sits this far above the first

# ── Screen face: 2 corner mount holes, CSK from the back (solenoid-mating) face ─
screen_corner_csk_hole_diameter = M3_NORMAL
screen_corner_csk_hole_from_side = 4 * MM  # face-local X from each side edge
screen_corner_csk_hole_from_bottom = 12 * MM  # face-local Y from bottom edge
screen_corner_csk_head_diameter = 6.5 * MM  # 90° CSK head Ø (FHCS M3)
screen_corner_csk_angle = 90  # degrees

# ── Screen face: 2 rectangular through-slots near the top corners ─────────────
# (top = far from the keyboard plate). Centers 4 mm in from each top corner.
screen_top_slot_w = 3 * MM  # face-local X (width)
screen_top_slot_h = 5 * MM  # face-local Y (height, vertical)
screen_top_slot_from_side = 4 * MM  # face-local X from each side edge
screen_top_slot_from_top = 4 * MM  # face-local Y below the top edge

# ── Fillets ───────────────────────────────────────────────────────────────────
x_edge_fillet_radius = 0.5 * MM  # all X-parallel edges except the keyboard↔screen join


class SolenoidMount(BaseCustomPart):
    def _build(self):
        with BuildPart() as my_part:
            Box(length, width, thickness)

            # Back wall — extruded up from the +Y strip of the top face
            top_plane = Plane.XY.offset(thickness / 2)
            with BuildSketch(top_plane):
                with Locations((0, wall_center_y)):
                    Rectangle(length, wall_thickness)
            extrude(amount=wall_height)

            # Keyboard face: 4 corner through-holes
            with BuildSketch(top_plane):
                with Locations((0, keyboard_rect_center_y)):
                    with GridLocations(keyboard_rect_w, keyboard_rect_h, 2, 2):
                        Circle(radius=keyboard_hole_diameter / 2)
            extrude(amount=-thickness, mode=Mode.SUBTRACT)

            # Screen face plane (-Y normal). Origin at bottom-left corner of the face;
            # x_dir = +X forces y_dir = z_dir × x_dir = (0, 0, +1) = world +Z, so
            # face-local (x, y) reads as (from-left, above-bottom).
            screen_plane = Plane(
                origin=(-length / 2, width / 2 - wall_thickness, thickness / 2),
                x_dir=(1, 0, 0),
                z_dir=(0, -1, 0),
            )

            # Screen face: paired hole pattern (N bases × 2 row-shifts × 2 paired rows)
            screen_pattern_locations = [
                (
                    screen_pattern_base_from_left + i * screen_pattern_spacing,
                    screen_pattern_base_from_bottom + screen_pattern_y_offsets[i] * MM,
                )
                for i in range(len(screen_pattern_y_offsets))
            ]
            with BuildSketch(screen_plane):
                with Locations(*screen_pattern_locations):
                    with Locations((0, 0), (0, screen_pattern_row_shift)):
                        with Locations((0, 0), (0, screen_pattern_pair_offset)):
                            Circle(radius=screen_pattern_hole_diameter / 2)
            extrude(amount=-wall_thickness, mode=Mode.SUBTRACT)

            # Screen face: 2 corner mount holes, CSK from the BACK face (the
            # side the solenoid mates against). Head sinks flush into the back
            # so the solenoid sits flat; through-hole exits on the screen face.
            # Rotation (-90, 0, 0) puts the Location's local +Z along world +Y,
            # so the CSK opens at world Y = width/2 and drills in -Y direction.
            csk_world_z = thickness / 2 + screen_corner_csk_hole_from_bottom
            csk_axis_rot = (-90, 0, 0)
            with Locations(
                Location(
                    (
                        -length / 2 + screen_corner_csk_hole_from_side,
                        width / 2,
                        csk_world_z,
                    ),
                    csk_axis_rot,
                ),
                Location(
                    (
                        +length / 2 - screen_corner_csk_hole_from_side,
                        width / 2,
                        csk_world_z,
                    ),
                    csk_axis_rot,
                ),
            ):
                CounterSinkHole(
                    radius=screen_corner_csk_hole_diameter / 2,
                    counter_sink_radius=screen_corner_csk_head_diameter / 2,
                    counter_sink_angle=screen_corner_csk_angle,
                )

            join_y = width / 2 - wall_thickness

            # Screen face: 2 rectangular through-slots, centers 4 mm in from each
            # top corner (top = far from the keyboard plate).
            top_y = wall_height - screen_top_slot_from_top
            with BuildSketch(screen_plane):
                with Locations(
                    (screen_top_slot_from_side, top_y),
                    (length - screen_top_slot_from_side, top_y),
                ):
                    Rectangle(screen_top_slot_w, screen_top_slot_h)
            extrude(amount=-wall_thickness, mode=Mode.SUBTRACT)

            # Keyboard-face through-slot: +Y edge on the keyboard↔screen inner
            # edge (join_y), centered in X.
            with BuildSketch(top_plane):
                with Locations((0, join_y - keyboard_slot_h / 2)):
                    Rectangle(keyboard_slot_w, keyboard_slot_h)
            extrude(amount=-thickness, mode=Mode.SUBTRACT)

            # Keyboard-face Ø4.3 PTFE-tube through-hole, just below the wire slot.
            with BuildSketch(top_plane):
                with Locations((tube_hole_x, tube_hole_y)):
                    Circle(radius=tube_hole_diameter / 2)
            extrude(amount=-thickness, mode=Mode.SUBTRACT)

            # Fillet last: break the long X-parallel body edges (0.5 mm). Keep
            # only full-span (length == `length`) edges, which drops the short
            # edges of the three rect slots; also skip the keyboard↔screen join.
            join_z = thickness / 2
            x_fillet_edges = [
                e
                for e in my_part.edges().filter_by(Axis.X)
                if abs(e.length - length) < 1e-3
                and not (
                    abs(e.center().Y - join_y) < 1e-3
                    and abs(e.center().Z - join_z) < 1e-3
                )
            ]
            fillet(x_fillet_edges, radius=x_edge_fillet_radius)

        return my_part.part


if __name__ == "__main__":
    SolenoidMount().export()
