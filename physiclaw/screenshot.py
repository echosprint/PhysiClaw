"""
Phone screenshot via AssistiveTouch — tap to screenshot, double-tap to upload.

Manages the AT button position, tapping, and screenshot verification
via a color nonce barcode.
"""

import logging
import random
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)

# AssistiveTouch button position (CSS viewport pixels, iPhone left edge snap)
AT_CSS_X = 38       # 10pt edge margin + 28pt button radius
AT_CSS_Y = 200      # hardcoded vertical position
AT_RADIUS = 28      # matches AT button (56pt diameter)

# Color nonce barcode position and size
NONCE_CSS_X = 180
NONCE_CSS_Y = 300
NONCE_COUNT = 20
NONCE_SQUARE_SIZE = 15  # CSS pixels per square
NONCE_COLOR_MIN = 50    # avoid near-black
NONCE_COLOR_MAX = 230   # avoid near-white
NONCE_MAX_DIST = 40     # max Euclidean RGB distance for a match


class PhoneScreenshot:
    """AssistiveTouch screenshot manager.

    Knows where the AT button is in screen 0-1 coordinates.
    Single-tap: iOS takes a screenshot (saved to Photos).
    Double-tap: iOS Shortcut gets the latest screenshot from Photos and uploads it.
    Verifies uploaded screenshots via a color nonce barcode.

    Usage:
        ps = PhoneScreenshot()
        ps.compute_at_screen_pos(cal.screenshot_transform)  # after pre-cal
        ps.tap(arm, pct_to_grbl)              # iOS screenshot
        ps.double_tap(arm, pct_to_grbl)       # screenshot + upload
        img_bytes = ps.take_screenshot(arm, bridge, pct_to_grbl)
    """

    def __init__(self):
        self.at_screen: tuple[float, float] | None = None  # screenshot 0-1
        self.at_radius_screen: tuple[float, float] | None = None  # (rx, ry) in 0-1

    @property
    def ready(self) -> bool:
        """True when AT position is known and verified."""
        return self.at_screen is not None

    def overlaps_at(self, sx: float, sy: float) -> bool:
        """Check if a screen 0-1 position overlaps the AssistiveTouch button.

        Use before tapping to avoid accidentally hitting AT when aiming
        for a nearby UI element. Returns False if AT position is not set.
        """
        if self.at_screen is None or self.at_radius_screen is None:
            return False
        ax, ay = self.at_screen
        rx, ry = self.at_radius_screen
        # Ellipse test: ((sx-ax)/rx)^2 + ((sy-ay)/ry)^2 < 1
        return ((sx - ax) / rx) ** 2 + ((sy - ay) / ry) ** 2 < 1.0

    def compute_at_screen_pos(self, screenshot_transform: dict) -> tuple[float, float]:
        """Convert AT CSS position to screenshot 0-1 using pre-cal transform.

        Must be called after step_screenshot_cal sets screenshot_transform.
        Stores the result in self.at_screen.
        """
        t = screenshot_transform
        # CSS viewport → screenshot pixel
        px_x = AT_CSS_X * t['dpr'] + t['offset_x']
        px_y = AT_CSS_Y * t['dpr'] + t['offset_y']
        # Screenshot pixel → screenshot 0-1
        sx = px_x / t['screenshot_width']
        sy = px_y / t['screenshot_height']
        self.at_screen = (sx, sy)
        # AT button radius in screenshot 0-1 (different for x/y due to aspect ratio)
        rx = AT_RADIUS * t['dpr'] / t['screenshot_width']
        ry = AT_RADIUS * t['dpr'] / t['screenshot_height']
        self.at_radius_screen = (rx, ry)
        log.info(f"AT screen position: CSS ({AT_CSS_X}, {AT_CSS_Y}) → "
                 f"screenshot 0-1 ({sx:.3f}, {sy:.3f}), "
                 f"radius ({rx:.3f}, {ry:.3f})")
        return self.at_screen

    def _move_to_at(self, arm, pct_to_grbl: np.ndarray):
        """Move arm to AT button position."""
        if self.at_screen is None:
            raise RuntimeError("AT position not set — call compute_at_screen_pos first")
        sx, sy = self.at_screen
        grbl = pct_to_grbl @ np.array([sx, sy, 1.0])
        arm._fast_move(float(grbl[0]), float(grbl[1]))
        arm.wait_idle()

    def tap(self, arm, pct_to_grbl: np.ndarray):
        """Single-tap AT — iOS takes a screenshot (saved to Photos)."""
        if self.at_screen is None:
            raise RuntimeError("AT position not set — call compute_at_screen_pos first")
        self._move_to_at(arm, pct_to_grbl)
        arm.tap()
        log.info(f"AT single-tap at screen ({self.at_screen[0]:.3f}, {self.at_screen[1]:.3f})")

    def double_tap(self, arm, pct_to_grbl: np.ndarray):
        """Double-tap AT — iOS Shortcut gets latest screenshot and uploads it."""
        if self.at_screen is None:
            raise RuntimeError("AT position not set — call compute_at_screen_pos first")
        self._move_to_at(arm, pct_to_grbl)
        arm.double_tap()
        log.info(f"AT double-tap at screen ({self.at_screen[0]:.3f}, {self.at_screen[1]:.3f})")

    def take_screenshot(self, arm, bridge, pct_to_grbl: np.ndarray,
                        timeout: float = 10.0) -> bytes | None:
        """Single-tap (take screenshot) + double-tap (upload latest), return image bytes."""
        bridge.clear_screenshot()
        self.tap(arm, pct_to_grbl)
        time.sleep(5.0)
        self.double_tap(arm, pct_to_grbl)
        data = bridge.wait_screenshot(timeout=timeout)
        if data is None:
            log.warning("Screenshot upload timed out")
        return data

    # ─── Color nonce ──────────────────────────────────────────

    @staticmethod
    def generate_nonce() -> list[list[int]]:
        """Generate 20 random RGB colors for screenshot verification."""
        return [[random.randint(NONCE_COLOR_MIN, NONCE_COLOR_MAX)
                 for _ in range(3)]
                for _ in range(NONCE_COUNT)]

    @staticmethod
    def verify_nonce(img: np.ndarray, screenshot_transform: dict,
                     expected_colors: list[list[int]]) -> tuple[bool, int]:
        """Verify color nonce barcode in a screenshot.

        Samples the center pixel of each 3×3 CSS square (scaled by DPR)
        and compares to expected RGB values.

        Args:
            img: decoded screenshot (BGR, from cv2.imdecode)
            screenshot_transform: pre-cal transform with dpr, offset_x/y
            expected_colors: list of [r, g, b] values (20 entries)

        Returns:
            (all_matched, match_count) — all_matched is True only if 20/20.
        """
        t = screenshot_transform
        dpr = t['dpr']
        step = int(NONCE_SQUARE_SIZE * dpr)  # pixel spacing between squares
        base_x = int(NONCE_CSS_X * dpr + t['offset_x'])
        base_y = int(NONCE_CSS_Y * dpr + t['offset_y'])

        matched = 0
        for i, expected in enumerate(expected_colors):
            # Center of each square
            cx = base_x + step // 2
            cy = base_y + i * step + step // 2
            if not (0 <= cy < img.shape[0] and 0 <= cx < img.shape[1]):
                log.warning(f"  Nonce square {i}: pixel ({cx}, {cy}) out of bounds "
                            f"({img.shape[1]}×{img.shape[0]})")
                continue
            # OpenCV is BGR
            b, g, r = int(img[cy, cx, 0]), int(img[cy, cx, 1]), int(img[cy, cx, 2])
            dist = ((r - expected[0])**2 + (g - expected[1])**2 +
                    (b - expected[2])**2) ** 0.5
            if dist < NONCE_MAX_DIST:
                matched += 1
            else:
                log.info(f"  Nonce square {i}: expected RGB({expected[0]}, {expected[1]}, {expected[2]}), "
                         f"got RGB({r}, {g}, {b}), dist={dist:.1f} — MISMATCH")

        all_matched = matched == len(expected_colors)
        log.info(f"  Nonce verification: {matched}/{len(expected_colors)} matched"
                 f" — {'PASS' if all_matched else 'FAIL'}")
        return all_matched, matched

    # ─── Step 7 setup flow ────────────────────────────────────

    def setup(self, arm, bridge, cal_state, pct_to_grbl: np.ndarray) -> dict:
        """Full step 7: single-tap, wait 5s, double-tap, verify nonce.

        Requires:
        - Phase "assistive_touch" already set on phone with nonce colors
        - User has positioned AT at the orange circle
        - cal_state.screenshot_transform is set (from pre-cal)
        - arm.Z_DOWN is set (from step 0)

        Returns dict with passed, matched, total.
        """
        if self.at_screen is None:
            raise RuntimeError("AT position not set — call compute_at_screen_pos first")

        log.info("═══ Step 7: AssistiveTouch screenshot verification ═══")
        log.info(f"  AT position: screen 0-1 ({self.at_screen[0]:.3f}, {self.at_screen[1]:.3f})")

        # Generate nonce (caller should have already set it on the phone)
        nonce = cal_state._screenshot_nonce
        if nonce is None:
            raise RuntimeError("No nonce set — call step7_show first")

        # Clear any stale screenshot from previous steps
        bridge.clear_screenshot()

        # 1. Single-tap AT → iOS takes screenshot
        log.info("  Single-tap AT (iOS screenshot)...")
        self.tap(arm, pct_to_grbl)

        # 2. Wait for screenshot animation to complete
        log.info("  Waiting 5s for screenshot animation...")
        time.sleep(5.0)

        # 3. Double-tap AT → iOS Shortcut: screenshot + upload
        log.info("  Double-tap AT (screenshot + upload)...")
        self.double_tap(arm, pct_to_grbl)

        # 4. Wait for screenshot upload
        log.info("  Waiting for screenshot upload...")
        data = bridge.wait_screenshot(timeout=10.0)
        if data is None:
            log.warning("  Screenshot upload timed out")
            return {"passed": False, "matched": 0, "total": NONCE_COUNT}

        # 5. Decode and verify
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            log.warning("  Failed to decode screenshot")
            return {"passed": False, "matched": 0, "total": NONCE_COUNT}

        log.info(f"  Screenshot received: {img.shape[1]}×{img.shape[0]}px")

        t = cal_state.screenshot_transform
        if t is None:
            raise RuntimeError("screenshot_transform not set — run pre-cal first")

        passed, matched = self.verify_nonce(img, t, nonce)

        if passed:
            log.info(f"  ✓ Step 7 done: AT verified, screenshot pipeline working")
        else:
            log.warning(f"  ✗ Step 7 failed: {matched}/{NONCE_COUNT} colors matched")

        return {"passed": passed, "matched": matched, "total": NONCE_COUNT}
