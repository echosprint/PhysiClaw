"""Physical device control for PhysiClaw.

GRBL stylus arm, OpenCV camera, and AssistiveTouch screenshot pipeline.
Knows nothing about computer vision or calibration.
"""

from physiclaw.core.hardware.arm import StylusArm
from physiclaw.core.hardware.camera import Camera
from physiclaw.core.hardware.grbl import detect_grbl
from physiclaw.core.hardware.iphone import AssistiveTouch
from physiclaw.core.hardware.solenoid import Solenoid

__all__ = [
    "StylusArm",
    "Solenoid",
    "Camera",
    "AssistiveTouch",
    "detect_grbl",
]
