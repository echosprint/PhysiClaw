"""Simplified representation of the MKS DLC32 V2.0 CNC controller board.

A purchased part, modeled as a positional/visual stand-in. "Simplified"
means only the major components are included (many small headers are
omitted) — but every component that *is* modeled uses its real-world
dimensions and shape, including the cavities/pins a mating connector
plugs into. Component specs are sourced from the connector datasheets
(see the comments by each builder).

Footprint and mounting-hole pattern match what ``pcb_holder.py`` is
designed against (90 × 70 board, four M3 corner holes on an 82 × 62
pitch) so the board drops onto the holder's standoffs.

Coordinate convention (matches ``pcb_holder``):
  * Board centered on the XY origin; long axis (90 mm) along X.
  * PCB spans z = 0 (bottom) .. ``pcb_th`` (top). Components sit on +Z.
  * +Y is the motor-connector edge (image top), -Y is the USB/power edge
    (image bottom), -X is the spindle-terminal edge (image left).

Components modeled (from the "MKS DLC32 Interface introduction" diagram):
  chip (ESP32-WROOM-32), DC power input, USB-PC, CNC spindle terminal,
  three stepper-driver slots, X/Y1/Y2/Z motor connectors, four corner
  mounting holes, and the electrolytic capacitors.

Run from the repo root:

    uv run --group cad python -m hardware.parts.standard.board
"""

from build123d import *

from hardware.parts._fits import DRIVER_ROW_PITCH, HDR_PITCH, M3_NORMAL
from hardware.parts.base import BaseStandardPart

# ── PCB ───────────────────────────────────────────────────────────────────────
board_x = 90 * MM  # long axis (left↔right, X)
board_y = 70 * MM  # short axis (front↔back, Y)
pcb_th = 1.6 * MM  # standard PCB thickness
corner_r = 3 * MM  # rounded PCB corners
edge_tol = 0.5 * MM  # fuzz for matching corner edges by center

# Four M3 corner mounting holes — 82 × 62 pitch (4 mm in from each edge),
# the same pattern pcb_holder's standoffs use.
hole_pitch_x = 82 * MM
hole_pitch_y = 62 * MM
hole_dia = M3_NORMAL

# Silkscreen-style board label, engraved into an empty strip of the top
# face (above the USB port, below the bulk cap, left of the ESP32).
label_text = "MKS DLC32 V2.1"
label_size = 3.0 * MM  # cap height
label_depth = 0.5 * MM  # engrave depth
label_cx = 24 * MM  # center X
label_cy = -17 * MM  # center Y

# Connector pitches and sizes. HDR_PITCH / DRIVER_ROW_PITCH come from _fits
# (shared with the TMC2209 driver module so the two mate).
XH_PITCH = 2.5 * MM  # JST XH series (motor connectors)
HDR_HEIGHT = 8.5 * MM  # female-header (driver-slot) height above the PCB

# Colors (visual aid in STEP viewers; the SVG render pipeline ignores them).
COL_PCB = Color(0.05, 0.32, 0.13)  # green solder mask
COL_BLACK = Color(0.10, 0.10, 0.12)  # plastic connector bodies / caps
COL_METAL = Color(0.62, 0.64, 0.66)  # USB shield / ESP32 RF can
COL_BLUE = Color(0.10, 0.22, 0.60)  # screw terminal
COL_RED = Color(0.70, 0.10, 0.10)  # DIP switches


def hole_locations(z: float = 0):
    """The four corner mounting-hole (x, y) centers, lifted to height z."""
    return [
        (sx, sy, z)
        for sx in (-hole_pitch_x / 2, hole_pitch_x / 2)
        for sy in (-hole_pitch_y / 2, hole_pitch_y / 2)
    ]


# ── Component builders (canonical orientation, base on z = 0) ──────────────────
def _xh_header(n: int) -> Part:
    """JST XH top-entry shrouded header, ``n`` ways at 2.5 mm pitch.

    Real B*B-XH-A: pin span = (n-1)·2.5, overall housing ≈ span + 4.9 mm
    long × 5.75 mm wide, ~6 mm shroud above the PCB. The shroud opens
    upward (+Z) — that cavity is where the wire housing plugs in. Square
    pins sit inside it.
    """
    span = (n - 1) * XH_PITCH
    length = span + 4.9 * MM
    width = 5.75 * MM
    h = 6.0 * MM
    with BuildPart() as p:
        Box(length, width, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
        # Plug cavity: open top, ~1 mm walls all round.
        with Locations((0, 0, h)):
            Box(
                span + 1.5 * MM,
                width - 1.5 * MM,
                h - 1.5 * MM,
                align=(Align.CENTER, Align.CENTER, Align.MAX),
                mode=Mode.SUBTRACT,
            )
        # Square header pins standing in the cavity.
        with Locations((0, 0, h - 1.5 * MM)):
            with GridLocations(XH_PITCH, 0, n, 1):
                Box(
                    0.64 * MM,
                    0.64 * MM,
                    4.0 * MM,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                )
    return p.part


def _female_header(n: int) -> Part:
    """A single 1×``n`` 0.1" female header strip (length along X).

    Black body ~2.54 mm wide × 8.5 mm tall with a square socket hole per
    pin on the top face — the slot a driver carrier's pin plugs into.
    """
    span = (n - 1) * HDR_PITCH
    length = span + HDR_PITCH
    h = HDR_HEIGHT
    with BuildPart() as p:
        Box(length, HDR_PITCH, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
        with Locations((0, 0, h)):
            with GridLocations(HDR_PITCH, 0, n, 1):
                Box(
                    1.0 * MM,
                    1.0 * MM,
                    3.5 * MM,
                    align=(Align.CENTER, Align.CENTER, Align.MAX),
                    mode=Mode.SUBTRACT,
                )
    return p.part


def _dc_jack() -> Part:
    """DC-005 barrel jack (5.5 mm barrel / 2.1 mm center pin).

    Body 14 (insertion, Y) × 9 (X) × 11 (Z). The round socket opens on
    the -Y face: an 8 mm front recess with the 2.1 mm center pin inside.
    """
    w, l, h = 9.0 * MM, 14.0 * MM, 11.0 * MM
    with BuildPart() as p:
        Box(w, l, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
        front = Plane(origin=(0, -l / 2, h / 2), x_dir=(1, 0, 0), z_dir=(0, 1, 0))
        with Locations(front):
            Cylinder(
                8.0 / 2 * MM,
                9.0 * MM,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
                mode=Mode.SUBTRACT,
            )
        with Locations(front):
            Cylinder(
                2.1 / 2 * MM, 7.0 * MM, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
    return p.part


def _usb_b() -> Part:
    """USB Type-B (printer-style) receptacle.

    Rectangular metal shell 12 (X) × 16.4 (Y) × 11 (Z). The front (-Y)
    opening is the connector's characteristic near-square profile with the
    two TOP corners beveled (~45°) — the keying that lets the plug enter
    only one way. A contact tongue hangs in the upper part of the cavity.
    """
    w, l, h = 12.0 * MM, 16.4 * MM, 11.0 * MM
    op_w, op_h = 8.0 * MM, 8.0 * MM  # opening width / height
    bevel = 2.2 * MM  # top-corner chamfer
    depth = 9.0 * MM  # opening cut depth into the shell
    cz = h / 2  # opening vertical center
    with BuildPart() as p:
        Box(w, l, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
        # Front-face plane: local +y → world +Z so the beveled corners land
        # on top; normal +Y so the profile cuts into the shell.
        front = Plane(origin=(0, -l / 2, cz), x_dir=(-1, 0, 0), z_dir=(0, 1, 0))
        with BuildSketch(front):
            with BuildLine():
                Polyline(
                    (-op_w / 2, -op_h / 2),
                    (op_w / 2, -op_h / 2),
                    (op_w / 2, op_h / 2 - bevel),
                    (op_w / 2 - bevel, op_h / 2),
                    (-op_w / 2 + bevel, op_h / 2),
                    (-op_w / 2, op_h / 2 - bevel),
                    close=True,
                )
            make_face()
        extrude(amount=depth, mode=Mode.SUBTRACT)
        # Contact tongue in the upper part of the opening.
        tongue = Plane(
            origin=(0, -l / 2, cz + op_h / 2 - 2.0 * MM),
            x_dir=(1, 0, 0),
            z_dir=(0, 1, 0),
        )
        with Locations(tongue):
            Box(
                5.0 * MM,
                1.6 * MM,
                6.0 * MM,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            )
    return p.part


def _screw_terminal(poles: int = 2, pitch: float = 5.08 * MM) -> Part:
    """KF301-style PCB screw terminal, ``poles`` ways (poles along Y).

    Body ~7.5 (X) × span+5 (Y) × 9 (Z). Wire-entry bores face -X; screw
    heads sit on top.
    """
    span = (poles - 1) * pitch
    w, l, h = 7.5 * MM, span + 5.0 * MM, 9.0 * MM
    screw_r, screw_h = 2.0 * MM, 2.0 * MM  # raised clamp-screw heads
    slot_w, slot_depth = 0.7 * MM, 1.0 * MM  # flat-head screwdriver slot
    with BuildPart() as p:
        Box(w, l, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
        # Wire-entry bores facing -X (one per pole).
        side = Plane(origin=(-w / 2, 0, h * 0.4), x_dir=(0, 1, 0), z_dir=(1, 0, 0))
        with Locations(side):
            with GridLocations(pitch, 0, poles, 1):
                Cylinder(
                    3.0 / 2 * MM,
                    5.0 * MM,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                    mode=Mode.SUBTRACT,
                )
        # Raised cylindrical screw heads on the top face, one per pole.
        with Locations((0, 0, h)):
            with GridLocations(0, pitch, 1, poles):
                Cylinder(
                    screw_r, screw_h, align=(Align.CENTER, Align.CENTER, Align.MIN)
                )
        # Straight screwdriver slot across the top of each screw head.
        with Locations((0, 0, h + screw_h)):
            with GridLocations(0, pitch, 1, poles):
                Box(
                    2 * screw_r,
                    slot_w,
                    slot_depth,
                    align=(Align.CENTER, Align.CENTER, Align.MAX),
                    mode=Mode.SUBTRACT,
                )
    return p.part


def _esp32(side: float, notch: float) -> Part:
    """ESP32-WROOM-32 module — a square RF-shielded can with a small square
    notch removed from its bottom-right corner (board +X / -Y after
    placement), matching how the module reads on the board."""
    h = 3.1 * MM
    with BuildPart() as p:
        Box(side, side, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
        # Bottom-right corner notch (+X, -Y corner in the canonical frame).
        with Locations((side / 2, -side / 2, 0)):
            Box(
                notch,
                notch,
                h,
                align=(Align.MAX, Align.MIN, Align.MIN),
                mode=Mode.SUBTRACT,
            )
    return p.part


def _dip3() -> Part:
    """3-position microstep DIP switch: red body with three white sliders."""
    body_l, body_w, body_h = 10.0 * MM, 4.0 * MM, 3.0 * MM
    with BuildPart() as p:
        Box(body_l, body_w, body_h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return p.part


def _cap(dia: float, h: float) -> Part:
    """Radial electrolytic capacitor (can), base on z = 0."""
    with BuildPart() as p:
        Cylinder(dia / 2, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
        top = p.faces().sort_by(Axis.Z)[-1]
        chamfer(top.edges(), length=0.5 * MM)
    return p.part


# ── Layout (board-frame placement of each component) ──────────────────────────
# Motor connectors: XH 4-pin, packed along the +Y edge.
MOTOR_X = (-31 * MM, -17.5 * MM, -4 * MM, 9.5 * MM)  # X centers (X, Y1, Y2, Z)
MOTOR_Y = 27 * MM  # Y center, inboard of the +Y corner holes

# Stepper-driver slots: each is 2× 1×8 female headers, pins running along Y,
# the two rows ``DRIVER_ROW_PITCH`` apart along X. One DIP-3 below each.
DRIVER_X = (-29 * MM, -11 * MM, 5 * MM)  # X centers of the three slots
DRIVER_Y = 12 * MM  # Y center of the socket pair
DIP_Y = -1 * MM  # microstep DIP just below each slot

# ESP32 module: square can, right face up against the board's right edge.
esp_side = 22 * MM
esp_notch = 4 * MM  # bottom-right corner cut-out
esp_margin = 2.5 * MM  # gap from the +X board edge
esp_cx = board_x / 2 - esp_margin - esp_side / 2
esp_cy = 4 * MM  # vertical center

# Electrolytic capacitors: one 100 µF per driver (nestled between its two
# socket rows) plus the bulk power-section caps. (cx, cy, dia, h). Heights
# stay below the driver-slot sockets so no can stands proud.
cap_h = HDR_HEIGHT - 0.5 * MM
CAPS = [
    (-29 * MM, 18 * MM, 8 * MM, cap_h),  # 100 µF 35 V — driver X
    (-11 * MM, 18 * MM, 8 * MM, cap_h),  # 100 µF 35 V — driver Y
    (5 * MM, 18 * MM, 8 * MM, cap_h),  # 100 µF 35 V — driver Z
    (-2 * MM, -8 * MM, 10 * MM, cap_h),  # 220 µF 16 V — bulk
    (-20 * MM, -8 * MM, 8 * MM, cap_h),  # 100 µF 35 V — power section
]


class MksBoard(BaseStandardPart):
    def _build(self):
        children = []

        # ── PCB ───────────────────────────────────────────────────────────────
        with BuildPart() as pcb:
            Box(board_x, board_y, pcb_th, align=(Align.CENTER, Align.CENTER, Align.MIN))
            corners = [
                e
                for e in pcb.edges().filter_by(Axis.Z)
                if abs(abs(e.center().X) - board_x / 2) < edge_tol
                and abs(abs(e.center().Y) - board_y / 2) < edge_tol
            ]
            fillet(corners, radius=corner_r)
            with Locations(*hole_locations(pcb_th)):
                Hole(radius=hole_dia / 2)
            # Engrave the board label into the top face.
            with BuildSketch(Plane.XY.offset(pcb_th)):
                with Locations((label_cx, label_cy)):
                    Text(label_text, font_size=label_size)
            extrude(amount=-label_depth, mode=Mode.SUBTRACT)
        pcb_part = pcb.part
        pcb_part.color = COL_PCB
        pcb_part.label = "pcb"
        children.append(pcb_part)

        def place(part, x, y, rz=0, color=None, label=""):
            moved = part.moved(Location((x, y, pcb_th), (0, 0, rz)))
            if color is not None:
                moved.color = color
            moved.label = label
            children.append(moved)

        # ── Motor connectors (XH 4-pin) along the +Y edge ─────────────────────
        xh4 = _xh_header(4)
        for x, lbl in zip(MOTOR_X, ("X-Motor", "Y1-Motor", "Y2-Motor", "Z-Motor")):
            place(xh4, x, MOTOR_Y, color=COL_BLACK, label=lbl)

        # ── Stepper-driver slots (2× 1×8 female headers) + DIP switches ───────
        sock8 = _female_header(8)
        dip = _dip3()
        for dx in DRIVER_X:
            for off in (-DRIVER_ROW_PITCH / 2, DRIVER_ROW_PITCH / 2):
                place(
                    sock8,
                    dx + off,
                    DRIVER_Y,
                    rz=90,
                    color=COL_BLACK,
                    label="driver_slot",
                )
            place(dip, dx, DIP_Y, color=COL_RED, label="dip3")

        # ── ESP32-WROOM-32 (the "chip") — right edge, square, corner-notched ──
        place(
            _esp32(esp_side, esp_notch), esp_cx, esp_cy, color=COL_METAL, label="esp32"
        )

        # ── Bottom-edge power / data ──────────────────────────────────────────
        place(
            _dc_jack(), -23 * MM, -30 * MM, color=COL_BLACK, label="dc_input"
        )  # 12/24 V in
        place(_usb_b(), -8 * MM, -28.8 * MM, color=COL_METAL, label="usb_pc")  # USB-PC

        # ── CNC principal-axis (spindle) screw terminal, left edge ────────────
        # Sits low on the left edge, clear of the first driver slot above it.
        place(_screw_terminal(2), -40 * MM, -6 * MM, color=COL_BLUE, label="spindle")

        # ── Electrolytic capacitors ───────────────────────────────────────────
        for cx, cy, dia, h in CAPS:
            place(_cap(dia, h), cx, cy, color=COL_BLACK, label="cap")

        board = Compound(label="MksBoard", children=children)

        # Placement reference: board center on the bottom (mounting) face.
        RigidJoint("mount", to_part=board, joint_location=Location((0, 0, 0)))
        return board


if __name__ == "__main__":
    MksBoard().export()
