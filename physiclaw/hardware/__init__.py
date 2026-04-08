"""Physical device control for PhysiClaw.

GRBL stylus arm, OpenCV camera, and AssistiveTouch screenshot pipeline.
Knows nothing about computer vision or calibration.
"""

from physiclaw.hardware.arm import StylusArm
from physiclaw.hardware.camera import Camera, SNAPSHOT_DIR
from physiclaw.hardware.iphone import AssistiveTouch
from physiclaw.hardware.grbl import detect_grbl

__all__ = [
    "StylusArm",
    "Camera",
    "SNAPSHOT_DIR",
    "AssistiveTouch",
    "detect_grbl",
]
