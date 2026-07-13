"""Camera frame calibration — physical setup check + software rotation.

One overhead frame drives both halves of the step: a physical
camera-setup diagnostic (phone shape, coverage, edge straightness via
``check_phone_in_frame``) and the software frame-rotation code. The
page shows a blue UP and a red RIGHT marker; their relative positions
in the raw frame tell us which cv2 rotation uprights the camera view,
so every later frame grab (camera mapping, validation, runtime vision)
can rotate first and treat the image as canonical portrait.
"""

import logging
import time

import cv2
import numpy as np

from physiclaw.core.bridge import CalibrationState
from physiclaw.core.hardware.camera import Camera
from physiclaw.core.vision.blobs import find_largest_hsv_blob
from physiclaw.core.vision.colors import red_ranges
from physiclaw.core.vision.util import check_phone_in_frame

log = logging.getLogger(__name__)


def _pick_rotation_from_markers(frame: np.ndarray) -> tuple[int, str]:
    """Locate blue UP and red RIGHT markers, derive the cv2 rotation code.

    Returns (rotation_code, human_label). Rotation is -1 (none) or one of
    cv2.ROTATE_{90_CLOCKWISE, 180, 90_COUNTERCLOCKWISE}. Raises if either
    marker is missing.
    """

    def _blob(
        lower: list[int], upper: list[int] | None = None
    ) -> tuple[float, float] | None:
        return find_largest_hsv_blob(
            frame,
            lower,
            upper,
            min_area=500,
            morph_op=cv2.MORPH_CLOSE,
            morph_kernel=(15, 15),
        )

    # UP = blue (#2563eb → H≈110). RIGHT = red (#ef4444), which straddles the
    # 0/180 hue seam — red_ranges covers both ends, since the camera commonly
    # renders the on-screen red near the high end (H≈175), not near 0.
    up = _blob([100, 80, 80], [130, 255, 255])
    if up is None:
        raise RuntimeError("UP (blue) marker not found")
    right = _blob(red_ranges(80, 80))
    if right is None:
        raise RuntimeError("RIGHT (red) marker not found")
    up_x, up_y = up
    right_x, right_y = right

    log.info(
        f"  Blue UP at ({up_x:.0f}, {up_y:.0f}), red RIGHT at ({right_x:.0f}, {right_y:.0f})"
    )

    if up_y < right_y and abs(up_x - right_x) < abs(up_y - right_y):
        return -1, "0° — no rotation needed"
    if up_x < right_x and abs(up_y - right_y) < abs(up_x - right_x):
        return cv2.ROTATE_90_CLOCKWISE, "90° clockwise"
    if up_y > right_y and abs(up_x - right_x) < abs(up_y - right_y):
        return cv2.ROTATE_180, "180°"
    return cv2.ROTATE_90_COUNTERCLOCKWISE, "90° counter-clockwise"


def calibrate_camera_frame(cam: Camera, cal: CalibrationState) -> dict:
    """Camera frame calibration — physical setup check + rotation code.

    One overhead frame drives both:
      - physical camera-setup diagnostic (shape, coverage, edge straightness),
      - software rotation code picked from the UP/RIGHT markers.

    Returns ``{"rotation", "rotation_name", "setup_ok", "issues",
              "coverage", "aspect_ratio", "image_size", "phone_region"}``.
    """
    log.info("═══ Camera frame calibration ═══")
    cal.set_phase("markers")
    time.sleep(1.0)

    frame = cam.raw_frame()
    if frame is None:
        raise RuntimeError("Camera read failed — is the camera connected?")

    checks = check_phone_in_frame(frame)
    rotation, rot_label = _pick_rotation_from_markers(frame)
    log.info(f"  ✓ Camera frame: rotation={rot_label}, setup_ok={checks['ok']}")
    return {
        "rotation": rotation,
        "rotation_name": rot_label,
        "setup_ok": checks["ok"],
        "issues": checks["issues"],
        "coverage": checks["coverage"],
        "aspect_ratio": checks["aspect_ratio"],
        "image_size": checks["image_size"],
        "phone_region": checks["phone_region"],
    }
