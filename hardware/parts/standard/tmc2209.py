"""TMC2209 stepper-driver module (StepStick / Pololu carrier form factor).

A purchased part, modeled as a positional/visual stand-in: the black
carrier PCB, the exposed gold chip thermal pad, the black plastic header
bodies that brace the pins, the teardrop solder joints on the top face,
and the 2×8 male header pins that plug down into the board's driver slots.

Dimensions follow the standard StepStick carrier (e.g. BIGTREETECH
TMC2209 V1.3): 20.32 × 15.24 mm PCB (0.8" × 0.6"), 16 pins on a 2.54 mm
pitch arranged 2×8, the two rows 12.7 mm apart. The pin pitch and
row-to-row spacing are the shared ``_fits`` interface constants, so this
module is guaranteed to mate with the board's ``_female_header`` sockets.

Canonical frame: carrier PCB bottom on z = 0 (the mating plane), pins
protrude in -Z (into the sockets). The 8-pin rows run along X; the two
rows straddle Y. A ``mount`` joint at the carrier-bottom center seats the
module on top of a driver slot.

Run from the repo root:

    uv run --group cad python -m hardware.parts.standard.tmc2209
"""

from build123d import *

from hardware.parts._fits import DRIVER_ROW_PITCH, HDR_PITCH
from hardware.parts.base import BaseStandardPart

# ── Carrier PCB ───────────────────────────────────────────────────────────────
carrier_x = 20.32 * MM  # along the 8-pin rows
carrier_y = 15.24 * MM  # across the rows
carrier_th = 1.6 * MM

# ── Pins (2×8 male headers) ───────────────────────────────────────────────────
pin_w = 0.64 * MM  # square header pin
pin_protrude = 8 * MM  # exposed pin length below the carrier (into socket)
pin_stickout = 0.6 * MM  # how far the pin tip pokes above the solder mound
pin_rows = 2
pin_cols = 8

# ── Header plastic ─ the black spacer strip that braces each pin row. Sits on
# the underside of the carrier; the pins pass through it.
hdr_strip_len = pin_cols * HDR_PITCH  # 20.32 mm — full 8-pin span + end margins
hdr_strip_w = 2.54 * MM
hdr_plastic_h = 2.5 * MM  # depth below the carrier

# ── Solder joints ─ shiny squat mounds on the TOP face, one per pin,
# anchoring the pins to the board. A flat-top frustum (not spiky, not domed).
solder_base_r = 1.0 * MM
solder_tip_r = 0.55 * MM
solder_h = 0.9 * MM

# ── Chip thermal pad ─ the exposed gold copper square under the QFN driver.
pad_x, pad_y, pad_h = 9 * MM, 7 * MM, 0.1 * MM

# Colors (black solder mask, black header plastic, silver metal, gold pad).
COL_PCB = Color(0.08, 0.08, 0.09)
COL_PLASTIC = Color(0.05, 0.05, 0.06)
COL_METAL = Color(0.75, 0.75, 0.78)
COL_PAD = Color(0.83, 0.68, 0.21)

# Locations shared by the pins and their top-face solder joints.
_PIN_GRID = (HDR_PITCH, DRIVER_ROW_PITCH, pin_cols, pin_rows)


def _carrier() -> Part:
    with BuildPart() as p:
        Box(
            carrier_x,
            carrier_y,
            carrier_th,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
    return p.part


def _pad() -> Part:
    """Exposed gold thermal pad on the top face."""
    with BuildPart() as p:
        with Locations(Plane.XY.offset(carrier_th)):
            Box(pad_x, pad_y, pad_h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return p.part


def _pins() -> Part:
    """2×8 male pins: tips poke just above the solder mounds, then run down
    through the carrier and out -Z into the sockets."""
    pin_top = carrier_th + solder_h + pin_stickout
    with BuildPart() as p:
        with Locations(Plane.XY.offset(pin_top)):
            with GridLocations(*_PIN_GRID):
                Box(
                    pin_w,
                    pin_w,
                    pin_top + pin_protrude,
                    align=(Align.CENTER, Align.CENTER, Align.MAX),
                )
    return p.part


def _header_plastic() -> Part:
    """Black plastic spacer strip per pin row, on the carrier underside."""
    with BuildPart() as p:
        with Locations((0, -DRIVER_ROW_PITCH / 2, 0), (0, DRIVER_ROW_PITCH / 2, 0)):
            Box(
                hdr_strip_len,
                hdr_strip_w,
                hdr_plastic_h,
                align=(Align.CENTER, Align.CENTER, Align.MAX),
            )
    return p.part


def _solder() -> Part:
    """Squat flat-top solder mound (frustum) on the top face at each pin."""
    with BuildPart() as p:
        with Locations(Plane.XY.offset(carrier_th)):
            with GridLocations(*_PIN_GRID):
                Cone(
                    solder_base_r,
                    solder_tip_r,
                    solder_h,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                )
    return p.part


class Tmc2209(BaseStandardPart):
    def _build(self):
        parts = [
            ("carrier", _carrier(), COL_PCB),
            ("pad", _pad(), COL_PAD),
            ("plastic", _header_plastic(), COL_PLASTIC),
            ("pins", _pins(), COL_METAL),
            ("solder", _solder(), COL_METAL),
        ]
        children = []
        for label, part, color in parts:
            part.color = color
            part.label = label
            children.append(part)

        module = Compound(label="Tmc2209", children=children)

        # Seats on top of a driver slot; pins drop into the sockets below.
        RigidJoint("mount", to_part=module, joint_location=Location((0, 0, 0)))
        return module


if __name__ == "__main__":
    Tmc2209().export()
