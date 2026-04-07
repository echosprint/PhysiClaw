"""Physical device control for PhysiClaw.

GRBL stylus arm, OpenCV camera, and AssistiveTouch screenshot pipeline.
Knows nothing about computer vision or calibration.
"""

from physiclaw.hardware.stylus_arm import StylusArm
from physiclaw.hardware.camera import Camera, SNAPSHOT_DIR
from physiclaw.hardware.screenshot import PhoneScreenshot
from physiclaw.hardware.serial_probe import detect_grbl

__all__ = [
    "StylusArm",
    "Camera",
    "SNAPSHOT_DIR",
    "PhoneScreenshot",
    "detect_grbl",
]
