"""Linear X rail sub-assembly (short) — MGN9H 150 mm guideway with
M3 × 10 FHCS in the rail's mounting holes and hammer M3 T-nuts
hanging loosely from each shank tip, ready to engage.

Same construction as linear_10_y (LI10Y) — only the rail length,
screw-hole pattern, and slider position differ. The build logic is
reused via inheritance; this class just overrides four class
attributes.

Hole layout (7 holes at 20 mm pitch on a 150 mm rail):
  Screws at 1-indexed hole positions 1, 3, 5, 7 — every other hole
  (40 mm pitch), both ends included. 4 screws total.

Slider position:
  slider_position = 0.5 puts the slider centered along the rail
  (= world X = 0 after LI33X places the rail centered on world X = 0),
  matching the X-axis carriage's natural home position.

See linear_10_y for the full part list, geometry derivation, and
variant descriptions.

Run from the repo root:

    uv run --group cad python -m hardware step linear_32_x
"""

from hardware.assembly.base import ASSEMBLED_CAM0, EXPLODED_CAM0
from hardware.assembly.procedures.linear_10_y import LI10Y
from hardware.assembly.travel_ranges import X_RAIL_LENGTH


class LI32X(LI10Y):
    compound_label = "linear_32_x"
    views = [EXPLODED_CAM0, ASSEMBLED_CAM0]
    rail_length = X_RAIL_LENGTH  # mm — see assembly/travel_ranges.py
    screw_hole_indices = (1, 3, 5, 7)  # 7-hole (150 mm) rail — both ends, 40 mm pitch
    slider_position = 0.5  # slider centered along the rail
