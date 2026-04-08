"""
AssistiveTouch driver — control the iOS AssistiveTouch floating button.

Knows where the AT button sits on the phone screen and taps it via the arm
(single-tap → iOS screenshot, double-tap → iOS Shortcut runs and uploads
the latest screenshot to the server). Pure driver — calibration and
screenshot-pipeline verification logic live in `physiclaw.calibration`.
"""

import logging
import time

import numpy as np

log = logging.getLogger(__name__)


class AssistiveTouch:
    """AssistiveTouch button driver.

    Knows where the AT button is in screen 0-1 coordinates.
    Single-tap: iOS takes a screenshot (saved to Photos).
    Double-tap: iOS Shortcut gets the latest screenshot from Photos and uploads it.

    Usage:
        at = AssistiveTouch()
        at.compute_at_screen_pos(cal.screenshot_transform)  # after pre-cal
        at.tap(arm, pct_to_grbl)              # iOS screenshot
        at.double_tap(arm, pct_to_grbl)       # screenshot + upload
        img_bytes = at.take_screenshot(arm, bridge, pct_to_grbl)
    """

    # AT button position in CSS viewport pixels (iPhone left edge snap).
    AT_CSS_X = 38  # 10pt edge margin + 28pt button radius
    AT_CSS_Y = 200  # hardcoded vertical position
    AT_RADIUS = 28  # 56pt diameter

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

        Must be called after the screenshot_transform calibration step has set
        the viewport→screenshot transform on CalibrationState.
        Stores the result in self.at_screen.
        """
        t = screenshot_transform
        # CSS viewport → screenshot pixel
        px_x = self.AT_CSS_X * t["dpr"] + t["offset_x"]
        px_y = self.AT_CSS_Y * t["dpr"] + t["offset_y"]
        # Screenshot pixel → screenshot 0-1
        sx = px_x / t["screenshot_width"]
        sy = px_y / t["screenshot_height"]
        self.at_screen = (sx, sy)
        # AT button radius in screenshot 0-1 (different for x/y due to aspect ratio)
        rx = self.AT_RADIUS * t["dpr"] / t["screenshot_width"]
        ry = self.AT_RADIUS * t["dpr"] / t["screenshot_height"]
        self.at_radius_screen = (rx, ry)
        log.info(
            f"AT screen position: CSS ({self.AT_CSS_X}, {self.AT_CSS_Y}) → "
            f"screenshot 0-1 ({sx:.3f}, {sy:.3f}), "
            f"radius ({rx:.3f}, {ry:.3f})"
        )
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
        log.info(
            f"AT single-tap at screen ({self.at_screen[0]:.3f}, {self.at_screen[1]:.3f})"
        )

    def double_tap(self, arm, pct_to_grbl: np.ndarray):
        """Double-tap AT — iOS Shortcut gets latest screenshot and uploads it."""
        if self.at_screen is None:
            raise RuntimeError("AT position not set — call compute_at_screen_pos first")
        self._move_to_at(arm, pct_to_grbl)
        arm.double_tap()
        log.info(
            f"AT double-tap at screen ({self.at_screen[0]:.3f}, {self.at_screen[1]:.3f})"
        )

    def long_press(self, arm, pct_to_grbl: np.ndarray):
        """Long-press AT — iOS Shortcut fetches bridge text to clipboard."""
        if self.at_screen is None:
            raise RuntimeError("AT position not set — call compute_at_screen_pos first")
        self._move_to_at(arm, pct_to_grbl)
        arm.long_press()
        log.info(
            f"AT long-press at screen ({self.at_screen[0]:.3f}, {self.at_screen[1]:.3f})"
        )

    def take_screenshot(
        self, arm, bridge, pct_to_grbl: np.ndarray, timeout: float = 10.0
    ) -> bytes | None:
        """Single-tap (take screenshot) + double-tap (upload latest), return image bytes."""
        bridge.clear_screenshot()
        self.tap(arm, pct_to_grbl)
        time.sleep(5.0)
        self.double_tap(arm, pct_to_grbl)
        data = bridge.wait_screenshot(timeout=timeout)
        if data is None:
            log.warning("Screenshot upload timed out")
        return data


