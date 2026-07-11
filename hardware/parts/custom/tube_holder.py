"""PTFE-tube holder — a frame-mounted anchor for the solenoid wire's stiffener
tube.

Carries the same rib mount as the PCB holder (two M5 countersinks at
``mount_hole_pitch``, each dropping a hammer T-nut into a 2040 top slot). The
plate and the underside rib both run along Y out to the tube spigot's tip, so
they back the full length of the tall cylinder boss on the +Y end. The boss
holds a Ø4.3 coaxial bore that the OD4 PTFE tube plugs into. The wire is
spiral-wrapped to the tube alongside it (it does not pass through the bore); the
tube just gives the bundle stiffness and this socket anchors its fixed end to
the frame — so carriage motion never loads the control board.

Run from the repo root:

    uv run --group cad python -m hardware.parts.custom.tube_holder
"""

from build123d import *

from hardware.parts._fits import CSK_ANGLE, M5_CSK_HEAD, M5_NORMAL
from hardware.parts.base import BaseCustomPart

# ── Plate ─────────────────────────────────────────────────────────────────────
thickness = 3 * MM
plate_half_x = 8 * MM  # ±X (narrow)
plate_half_y = 20 * MM  # -Y half-extent; the +Y side runs out to the boss tip

# ── Mounting rib (same design as the PCB holder) ──────────────────────────────
# Runs along Y from the plate's -Y edge out to the boss tip (so the rib and the
# tube share their +Y end, bracing the full length of the spigot). Both M5 nuts
# drop one into each 20 mm-pitch slot.
tab_x = 10 * MM  # along X — short side
tab_thick = 3 * MM  # protrudes below the plate (-Z)
rib_cx = 0 * MM
rib_cy = 0 * MM  # M5 mount-hole center (rib geometry extends to +Y)
mount_hole_pitch = 20 * MM  # along Y, one nut per slot

# ── Tube socket: cylinder boss along the +Y region with a Ø4.3 coaxial bore ────
# The PTFE tube (OD4) plugs into the bore. Axis along +Y (outward), centered in
# X at mid-thickness — same boss/bore idiom the PCB holder used to carry.
boss_d = 8 * MM  # → ~1.85 mm wall around the 4.3 mm bore
boss_len = 25 * MM  # overall cylinder length (tall spigot for the tube)
boss_embed = 0.5 * MM  # how far the boss base sits inboard of the plate's +Y datum
bore_d = 4.3 * MM  # OD4 PTFE tube, slip fit
bore_depth = 28 * MM  # tube insertion depth from the tip; stops short of the M5 holes
boss_tip_y = plate_half_y - boss_embed + boss_len  # +Y end of the boss (= socket mouth)

# Socket mouth (native point) + outward axis (native +Y). board_32 installs the
# holder with a 180° turn about Z, so this axis points WORLD -Y — physically up
# (spigot stands up) in the machine's use orientation — once mounted.
socket_mouth = (0, boss_tip_y, thickness / 2)
socket_axis = (0, 1, 0)

# ── Fillets ────────────────────────────────────────────────────────────────────
corner_d = 4 * MM  # plate corner rounding
edge_tol = 0.5 * MM


class TubeHolder(BaseCustomPart):
    def _build(self):
        with BuildPart() as my_part:
            # Base plate: runs from the -Y edge out to the boss tip (same +Y
            # extent as the rib), so it fully backs the boss. z = 0 .. thickness.
            plate_len_y = boss_tip_y + plate_half_y
            with Locations((0, -plate_half_y, 0)):
                Box(
                    2 * plate_half_x,
                    plate_len_y,
                    thickness,
                    align=(Align.CENTER, Align.MIN, Align.MIN),
                )

            # Mounting rib on the underside, protruding -Z, running from the
            # plate's -Y edge out to the boss tip (shares the boss's +Y end).
            rib_len_y = boss_tip_y + plate_half_y
            with Locations((rib_cx, -plate_half_y, 0)):
                Box(
                    tab_x,
                    rib_len_y,
                    tab_thick,
                    align=(Align.CENTER, Align.MIN, Align.MAX),
                )

            # 2040 mount countersinks — recessed on top, bored through plate +
            # rib, along the rib (Y), 20 mm apart.
            with Locations(
                (rib_cx, rib_cy - mount_hole_pitch / 2, thickness),
                (rib_cx, rib_cy + mount_hole_pitch / 2, thickness),
            ):
                CounterSinkHole(
                    radius=M5_NORMAL / 2,
                    counter_sink_radius=M5_CSK_HEAD / 2,
                    counter_sink_angle=CSK_ANGLE,
                )

            # Cylinder boss along the +Y region: axis along +Y (outward — points
            # world +Y once installed), centered in X at mid-thickness, embedded
            # in the plate that backs it.
            boss_face = Plane(
                origin=(0, plate_half_y - boss_embed, thickness / 2),
                x_dir=(1, 0, 0),
                z_dir=(0, 1, 0),
            )
            with Locations(boss_face):
                Cylinder(
                    boss_d / 2, boss_len, align=(Align.CENTER, Align.CENTER, Align.MIN)
                )

            # Coaxial Ø4.3 bore from the boss tip, -Y inward through the boss and
            # into the plate — stops short of the M5 mount holes.
            bore_face = Plane(
                origin=(0, boss_tip_y, thickness / 2), x_dir=(1, 0, 0), z_dir=(0, -1, 0)
            )
            with Locations(bore_face):
                Cylinder(
                    bore_d / 2,
                    bore_depth,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                    mode=Mode.SUBTRACT,
                )

            # Round the four plate corners — last, after all cuts. The +Y corners
            # are now at the boss tip (the plate runs out there).
            plate_corners = [
                e
                for e in my_part.edges().filter_by(Axis.Z)
                if abs(abs(e.center().X) - plate_half_x) < edge_tol
                and (
                    abs(e.center().Y + plate_half_y) < edge_tol
                    or abs(e.center().Y - boss_tip_y) < edge_tol
                )
            ]
            fillet(plate_corners, radius=corner_d / 2)

        return my_part.part


if __name__ == "__main__":
    TubeHolder().export()
