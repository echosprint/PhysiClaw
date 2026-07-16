"""
AssistiveTouch driver — control the iOS AssistiveTouch floating button.

Knows where the AT button sits on the phone screen and taps it via the arm
(single-tap → iOS screenshot, double-tap → iOS Shortcut runs and uploads
the latest screenshot to the server). Pure driver — calibration and
screenshot-pipeline verification logic live in `physiclaw.core.calibration`.
"""

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from physiclaw.core import geometry
from physiclaw.core.hardware.arm import StylusArm

if TYPE_CHECKING:
    from physiclaw.core.bridge import BridgeState

log = logging.getLogger(__name__)


class AssistiveTouch:
    """AssistiveTouch button driver.

    Knows where the AT button is in screen 0-1 coordinates.
    Single-tap: iOS takes a screenshot (saved to Photos).
    Double-tap: iOS Shortcut gets the latest screenshot from Photos and uploads it.

    Usage:
        at = AssistiveTouch()
        at.compute_at_screen_pos(cal.viewport_shift)  # after pre-cal
        at.tap(arm, pct_to_grbl)              # iOS screenshot
        at.double_tap(arm, pct_to_grbl)       # screenshot + upload
        img_bytes = at.take_screenshot(arm, bridge, pct_to_grbl)
    """

    # AT button geometry (CSS viewport px) — from the shared
    # render↔measure contract, kept as class attrs for callers.
    AT_CSS_X = geometry.AT_CSS_X
    AT_CSS_Y = geometry.AT_CSS_Y
    AT_RADIUS = geometry.AT_RADIUS

    def __init__(self) -> None:
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

    def swipe_crosses_at(self, cx: float, cy: float, ex: float, ey: float) -> bool:
        """True if the swipe segment (cx, cy) → (ex, ey), screen 0-1, passes
        over the AssistiveTouch button.

        Segment-vs-ellipse: scaling each axis by the button's radius maps
        the ellipse to the unit circle (and the segment to a segment), so
        the test is the segment's closest approach to the circle center.
        Checking the actual travel span — not just the start's row/column
        band — is what keeps a legitimate swipe elsewhere in the button's
        band from being blocked. Returns False if AT position is not set.
        """
        if self.at_screen is None or self.at_radius_screen is None:
            return False
        ax, ay = self.at_screen
        rx, ry = self.at_radius_screen
        if rx <= 0.0 or ry <= 0.0:
            return False  # degenerate radius — treat like an unset button
        px, py = (cx - ax) / rx, (cy - ay) / ry
        qx, qy = (ex - ax) / rx, (ey - ay) / ry
        dx, dy = qx - px, qy - py
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq == 0.0:
            return px * px + py * py < 1.0
        # Closest point on the segment to the (normalized) button center.
        t = max(0.0, min(1.0, -(px * dx + py * dy) / seg_len_sq))
        nx, ny = px + t * dx, py + t * dy
        return nx * nx + ny * ny < 1.0

    def compute_at_screen_pos(self, t: geometry.ViewportShift) -> tuple[float, float]:
        """Convert AT CSS position to screenshot 0-1 using the viewport shift.

        Must be called after the measure-viewport-shift pre-cal step has set
        cal.viewport_shift. Stores the result in self.at_screen.
        """
        # CSS viewport → screenshot 0-1 (center of AT button)
        sx, sy = t.css_to_pct(self.AT_CSS_X, self.AT_CSS_Y)
        self.at_screen = (sx, sy)
        # AT button radius in screenshot 0-1 (different for x/y due to aspect ratio)
        rx = self.AT_RADIUS * t.dpr / t.screenshot_width
        ry = self.AT_RADIUS * t.dpr / t.screenshot_height
        self.at_radius_screen = (rx, ry)
        log.info(
            f"AT screen position: CSS ({self.AT_CSS_X}, {self.AT_CSS_Y}) → "
            f"screenshot 0-1 ({sx:.3f}, {sy:.3f}), "
            f"radius ({rx:.3f}, {ry:.3f})"
        )
        return self.at_screen

    def _move_to_at(self, arm: StylusArm, pct_to_grbl: np.ndarray) -> None:
        """Move arm to AT button position."""
        if self.at_screen is None:
            raise RuntimeError("AT position not set — call compute_at_screen_pos first")
        sx, sy = self.at_screen
        gx, gy = geometry.apply_affine(pct_to_grbl, sx, sy)
        arm.rapid_to(gx, gy)
        arm.wait_idle()

    def tap(self, arm: StylusArm, pct_to_grbl: np.ndarray) -> None:
        """Single-tap AT — iOS takes a screenshot (saved to Photos)."""
        if self.at_screen is None:
            raise RuntimeError("AT position not set — call compute_at_screen_pos first")
        self._move_to_at(arm, pct_to_grbl)
        arm.tap()
        log.info(
            f"AT single-tap at screen ({self.at_screen[0]:.3f}, {self.at_screen[1]:.3f})"
        )

    def double_tap(self, arm: StylusArm, pct_to_grbl: np.ndarray) -> None:
        """Double-tap AT — iOS Shortcut gets latest screenshot and uploads it."""
        if self.at_screen is None:
            raise RuntimeError("AT position not set — call compute_at_screen_pos first")
        self._move_to_at(arm, pct_to_grbl)
        arm.double_tap()
        log.info(
            f"AT double-tap at screen ({self.at_screen[0]:.3f}, {self.at_screen[1]:.3f})"
        )

    def long_press(self, arm: StylusArm, pct_to_grbl: np.ndarray) -> None:
        """Long-press AT — iOS Shortcut fetches bridge text to clipboard."""
        if self.at_screen is None:
            raise RuntimeError("AT position not set — call compute_at_screen_pos first")
        self._move_to_at(arm, pct_to_grbl)
        arm.long_press()
        log.info(
            f"AT long-press at screen ({self.at_screen[0]:.3f}, {self.at_screen[1]:.3f})"
        )

    def take_screenshot(
        self,
        arm: StylusArm,
        bridge: "BridgeState",
        pct_to_grbl: np.ndarray,
        timeout: float = 10.0,
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
