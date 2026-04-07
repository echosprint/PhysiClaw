"""Calibration workflow for PhysiClaw.

The 7-step calibration pipeline that maps screen coordinates (0-1) to
GRBL mm and camera pixels. Uses hardware (arm + camera) and vision
(red/orange dot detection) to compute affine transforms.
"""

from physiclaw.calibration.grid_calibrate import GridCalibration

__all__ = ["GridCalibration"]
