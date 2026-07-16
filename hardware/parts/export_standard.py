"""Export every standard part — one .step file per part written to
hardware/output/step/. Run: `uv run python -m hardware.parts.export_standard`."""

from build123d import MM

from hardware.assembly.travel_ranges import X_RAIL_LENGTH, Y_RAIL_LENGTH
from hardware.parts.base import export_all
from hardware.parts.standard.board import MksBoard
from hardware.parts.standard.bracket import CornerBracket, FlatBracket, MotorBracket
from hardware.parts.standard.bumper import Bumper
from hardware.parts.standard.driver import HexDriver
from hardware.parts.standard.extrusion import Extrusion2020
from hardware.parts.standard.gooseneck import Gooseneck
from hardware.parts.standard.mgn9h import MGN9H
from hardware.parts.standard.motor import Nema17Motor
from hardware.parts.standard.nut import Nut
from hardware.parts.standard.pulley import Pulley2GT20T
from hardware.parts.standard.ring import Ring
from hardware.parts.standard.screw import Screw
from hardware.parts.standard.solenoid import Solenoid
from hardware.parts.standard.t_nut import TNut
from hardware.parts.standard.teflon import Teflon
from hardware.parts.standard.tip import Tip
from hardware.parts.standard.tmc2209 import Tmc2209

ALL_PARTS = [
    Extrusion2020(length=200 * MM, qty=2),
    Extrusion2020(length=100 * MM, qty=2),
    Nema17Motor(qty=2),
    FlatBracket(qty=6),
    MotorBracket(qty=2),
    CornerBracket(qty=2),
    Pulley2GT20T(kind="pulley", qty=2),
    Pulley2GT20T(kind="idler", toothed=True, qty=2),
    Pulley2GT20T(kind="idler", toothed=False, qty=2),
    MGN9H(rail_length=X_RAIL_LENGTH * MM, qty=1),  # X rail
    MGN9H(rail_length=Y_RAIL_LENGTH * MM, qty=2),  # Y rails
    MksBoard(qty=1),
    Tmc2209(qty=2),
    Screw("BHCS", "M3", 8 * MM),
    Screw("SHOULDER", "M4", 20 * MM),
    Screw("SHCS", "1/4-20", 16 * MM),
    HexDriver("2mm", qty=1),
    Ring("M3x10x1"),
    Ring("M5x8x0.5"),
    Ring("M5x15x12"),
    Ring("M6x12x8"),
    Ring("M6x12x12"),
    Nut("hex", "M3"),
    Nut("hex", "M4"),
    Nut("hex", "M5"),
    Nut("hex", "1/4-20"),
    Nut("square", "M3"),
    Nut("square", "M4"),
    Nut("square", "M5"),
    TNut("standard", "M3"),
    TNut("standard", "M4"),
    TNut("standard", "M5"),
    TNut("hammer", "M3"),
    TNut("hammer", "M4"),
    TNut("hammer", "M5"),
    Solenoid(),
    Bumper(),
    Tip(),
    Teflon(),
    Gooseneck(),
]


if __name__ == "__main__":
    export_all(ALL_PARTS)
