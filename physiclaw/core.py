"""
PhysiClaw orchestrator — central hardware lifecycle manager.

Owns the stylus arm, cameras, and calibration state.
Creating an instance connects hardware, identifies cameras, and runs calibration.
"""

import cv2

from physiclaw.camera import Camera
from physiclaw.stylus_arm import StylusArm
from physiclaw.vision import PhoneDetector


class PhysiClaw:
    """Central orchestrator — owns all hardware lifecycle.

    Construction connects the arm, identifies cameras, and runs
    the full calibration workflow. Ready to use immediately after.
    """

    def __init__(self):
        self._arm: StylusArm | None = None
        self._top_cam: Camera | None = None
        self._side_cam: Camera | None = None
        self._setup()

    # ─── Setup ────────────────────────────────────────────────

    def _setup(self):
        """Connect arm, identify cameras, calibrate.

        1. Connect GRBL arm (auto-detect port)
        2. Identify top vs side cameras via PhoneDetector
        3. Open cameras
        4. Run calibration
        """
        self._arm = StylusArm()
        self._arm.setup()

        detector = PhoneDetector()
        cameras = detector.identify_cameras()

        if 'top' not in cameras:
            raise RuntimeError("Top camera not found — is the phone under the camera?")

        self._top_cam = Camera(cameras['top'])
        if 'side' in cameras:
            self._side_cam = Camera(cameras['side'])

        self.calibrate()

    # ─── Calibration ───────────────────────────────────────────

    def calibrate(self):
        """Run the full 5-phase calibration workflow.

        Uses the side camera to detect green flashes during probing.
        Sets Z depth and axis mapping directly on the arm instance.
        """
        from physiclaw.calibrate import (
            phase1_z, phase2_right, phase3_down,
            phase4_long_press, phase5_swipe,
        )

        if self._side_cam is None:
            raise RuntimeError("Side camera not found — needed for calibration")

        arm = self._arm
        cam = self._side_cam

        # Phase 1 — Z depth
        z_tap = phase1_z(arm, cam)
        if z_tap is None:
            raise RuntimeError("Phase 1 failed — no Z contact detected")
        arm.Z_DOWN = z_tap

        # Phase 2 — find phone-right direction
        right_result = phase2_right(arm, cam, z_tap)
        if right_result is None:
            raise RuntimeError("Phase 2 failed — could not find phone-right")

        # Phase 3 — find phone-down direction
        right_vec = (right_result[0], right_result[1])
        down_result = phase3_down(arm, cam, z_tap, right_vec)
        if down_result is None:
            raise RuntimeError("Phase 3 failed — could not find phone-down")

        # Phase 4 — long press verification
        phase4_long_press(arm, cam)

        # Phase 5 — swipe verification
        rax, ray, r_dist = right_result
        dax, day, d_dist = down_result
        arm.set_direction_mapping((rax, ray), (dax, day))
        phase5_swipe(arm, cam)

        print("\n" + "=" * 50)
        print("CALIBRATION COMPLETE")
        print("=" * 50)
        print(f"  Z tap depth:    {z_tap} mm")
        print(f"  Phone right:    arm ({rax}, {ray}) × {r_dist} mm")
        print(f"  Phone down:     arm ({dax}, {day}) × {d_dist} mm")

    # ─── Properties ────────────────────────────────────────────

    @property
    def arm(self) -> StylusArm:
        return self._arm

    @property
    def top_cam(self) -> Camera:
        return self._top_cam

    @property
    def side_cam(self) -> Camera:
        if self._side_cam is None:
            raise RuntimeError("Side camera not available")
        return self._side_cam

    # ─── Snapshot helpers ──────────────────────────────────────

    def snapshot_top(self):
        """Capture a frame from the top camera. Returns BGR numpy array."""
        frame = self.top_cam.snapshot()
        if frame is None:
            raise RuntimeError("Top camera capture failed")
        return frame

    def snapshot_side(self):
        """Capture a frame from the side camera. Returns BGR numpy array."""
        frame = self.side_cam.snapshot()
        if frame is None:
            raise RuntimeError("Side camera capture failed")
        return frame

    @staticmethod
    def frame_to_jpeg(frame, quality=85) -> bytes:
        """Encode a BGR frame to JPEG bytes."""
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return jpeg.tobytes()

    # ─── Lifecycle ─────────────────────────────────────────────

    def shutdown(self):
        if self._arm:
            self._arm._pen_up()
            self._arm._fast_move(0, 0)
            self._arm.close()
        if self._top_cam:
            self._top_cam.close()
        if self._side_cam:
            self._side_cam.close()
