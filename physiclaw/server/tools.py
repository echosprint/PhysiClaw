"""
Core MCP tools — screenshot, gestures, targeting, detection.

Tools that operate on the phone via the stylus arm and camera. Each tool
acquires the hardware lock, runs its operation, and releases.
"""

import logging
import time

import cv2
from mcp.server.fastmcp import Image

log = logging.getLogger(__name__)


def register(mcp, physiclaw):
    """Register core MCP tools on the given FastMCP instance."""

    @mcp.tool()
    def screenshot() -> Image:
        """Take a screenshot of the phone screen.

        Use this to read screen content, check stylus position, or verify results.
        The stylus may be visible in the frame — call park() first if you need
        an unobstructed view of the screen.
        """
        physiclaw.require_hardware()
        physiclaw.acquire()
        try:
            frame = physiclaw.screenshot()
            return Image(data=physiclaw.frame_to_jpeg(frame), format="jpeg")
        finally:
            physiclaw.release()

    @mcp.tool()
    def park() -> str:
        """Park the stylus out of the camera frame so it doesn't occlude the screen.

        Call this before screenshot() when you need a clear view of the full screen
        (e.g. to read text or identify UI elements). The stylus moves 100mm away
        and will need to be repositioned with move() afterward.
        """
        physiclaw.require_hardware()
        physiclaw.acquire()
        try:
            physiclaw.park()
            return "Stylus parked out of frame"
        finally:
            physiclaw.release()

    @mcp.tool()
    def detect_elements() -> list:
        """Detect all interactable UI elements on the phone screen.

        Parks the stylus, takes a clean screenshot, and runs three detectors:
        1. Color segmentation: finds colored buttons, icons, tags, content images
           (no ML model needed — pure OpenCV HSV pipeline)
        2. Icon detection: finds all UI elements including gray/colorless ones
        3. OCR: reads all visible text (labels, keys, prices, etc.)

        Returns a text listing of all elements with bounding boxes as 0-1
        decimals [left, top, right, bottom], plus three annotated images
        (color blocks + icon boxes + OCR boxes).

        Color segmentation always works. Icon detection and OCR require vision
        models (run /setup-vision-models). Missing models show "unavailable".
        """
        physiclaw.require_hardware()
        physiclaw.acquire()
        try:
            physiclaw.park()
            time.sleep(1.5)
            elements_text, color_frame, icon_frame, ocr_frame = physiclaw.detect_elements()
            return [
                elements_text,
                Image(data=physiclaw.frame_to_jpeg(color_frame), format="jpeg"),
                Image(data=physiclaw.frame_to_jpeg(icon_frame), format="jpeg"),
                Image(data=physiclaw.frame_to_jpeg(ocr_frame), format="jpeg"),
            ]
        finally:
            physiclaw.release()

    @mcp.tool()
    def check_screen(reference_path: str) -> str:
        """Compare the current phone screen against a reference screenshot.

        Takes a camera screenshot and matches it against the reference image
        using ORB feature matching. Use this during skill execution to confirm
        the phone is showing the expected screen before acting.

        Also detects dark overlays (popups/modals) that might need dismissal.

        Args:
            reference_path: path to the reference screenshot (e.g.,
                "data/app-skills/meituan/screens/01_home.png")
        """
        from pathlib import Path
        from physiclaw.vision.screen_match import match_screen, detect_dark_overlay

        ref_path = Path(reference_path)
        if not ref_path.exists():
            return f"Reference image not found: {reference_path}"

        ref = cv2.imread(str(ref_path))
        if ref is None:
            return f"Failed to read reference image: {reference_path}"

        physiclaw.require_hardware()
        physiclaw.acquire()
        try:
            physiclaw.park()
            time.sleep(1.0)
            frame = physiclaw.screenshot()
        finally:
            physiclaw.release()

        result = match_screen(frame, ref)
        has_overlay = detect_dark_overlay(frame)

        lines = [f"Screen match: {'YES' if result.matched else 'NO'}"]
        lines.append(f"Confidence: {result.confidence:.1%} "
                     f"({result.good_matches} matches, {result.inliers} inliers)")
        if has_overlay:
            lines.append("WARNING: dark overlay detected — possible popup or modal dialog")
        lines.append(f"Reference: {reference_path}")
        return "\n".join(lines)

    @mcp.tool()
    def check_screen_changed() -> list:
        """Take two screenshots 1 second apart and check if the screen changed.

        Use this after a gesture (tap, swipe) to verify the action had an effect.
        Returns the second screenshot plus a text description of whether the
        screen changed.
        """
        from physiclaw.vision.screen_match import frames_differ

        physiclaw.require_hardware()
        physiclaw.acquire()
        try:
            physiclaw.park()
            time.sleep(0.5)
            frame_a = physiclaw.screenshot()
            time.sleep(1.0)
            frame_b = physiclaw.screenshot()
        finally:
            physiclaw.release()

        changed = frames_differ(frame_a, frame_b)
        status = "Screen CHANGED — the gesture had an effect" if changed \
            else "Screen UNCHANGED — the gesture may not have registered, try again"
        return [status, Image(data=physiclaw.frame_to_jpeg(frame_b), format="jpeg")]

    @mcp.tool()
    def grid_overlay(density: str = "normal", color: str = "green") -> Image:
        """Show the phone screen with a coordinate reference grid (0-1 scale).

        Draws numbered grid lines on a fresh screenshot so you can estimate
        coordinates for any target element. Call this before bbox_target()
        to get your bearings.

        To find a target: look at which grid lines it falls between, then
        estimate the value. For example, if a button is halfway between
        the 0.40 and 0.60 vertical lines, its x-coordinate is ~0.50.

        If the target falls between lines and you need more precision,
        call again with density="dense".

        Args:
            density: "sparse" (2x4 lines, coarse), "normal" (4x9 lines, default),
                     or "dense" (9x19 lines, 0.05 spacing for precise targeting)
            color: line color — "green", "red", or "yellow"
        """
        density_map = {
            "sparse": (4, 2),
            "normal": (9, 4),
            "dense": (19, 9),
        }
        rows, cols = density_map.get(density, density_map["normal"])
        physiclaw.require_hardware()
        physiclaw.acquire()
        try:
            physiclaw.park()
            time.sleep(1.5)
            frame = physiclaw.screenshot_with_grid(color, rows, cols)
            return Image(data=physiclaw.frame_to_jpeg(frame), format="jpeg")
        finally:
            physiclaw.release()

    @mcp.tool()
    def bbox_target(bbox: list[float]) -> Image:
        """Target a screen region by bounding box using 0-1 decimals.

        Takes a fresh screenshot and draws a green rectangle at the specified position.

        VERIFICATION REQUIRED: Name the UI element INSIDE the rectangle.
        - If the rectangle covers your target → confirm_bbox() → gesture.
        - If it does NOT cover your target → call bbox_target() again with
          corrected coordinates. Shift toward the target by the gap you observe.

        2-3 attempts is normal. bbox_target() is cheap (just a photo).
        tap() is expensive — a wrong tap can send a wrong message, transfer the
        wrong amount, or trigger an irreversible action.

        Args:
            bbox: [left, top, right, bottom] as 0-1 decimals
                  (0=left/top edge, 1=right/bottom edge)
        """
        physiclaw.require_hardware()
        physiclaw.acquire()
        try:
            physiclaw.park()
            time.sleep(1.5)
            physiclaw.set_pending_bbox(bbox)
            frame = physiclaw.screenshot_with_bboxes()
            return Image(data=physiclaw.frame_to_jpeg(frame), format="jpeg")
        finally:
            physiclaw.release()

    @mcp.tool()
    def confirm_bbox() -> str:
        """Confirm the bounding box from the last bbox_target() call.

        BEFORE CONFIRMING, ask yourself:
        "Does the green rectangle COVER the target element?"
        If not → do NOT confirm. Call bbox_target() with corrected coordinates.

        After confirmation, the next gesture (tap, double_tap, long_press, swipe)
        will auto-move to the bbox center before executing.
        """
        physiclaw.require_hardware()
        physiclaw.acquire()
        try:
            physiclaw.confirm_bbox()
            return "Bbox confirmed — next gesture will target this location"
        finally:
            physiclaw.release()

    def _maybe_move_to_bbox():
        """If a bbox is confirmed, move arm to its center."""
        if physiclaw._confirmed_bbox is not None:
            physiclaw.move_to_bbox_center(physiclaw._confirmed_bbox)

    @mcp.tool()
    def tap() -> str:
        """Single tap — like a finger tap on the screen.

        Use for: pressing buttons, selecting items, opening apps, following links, dismissing dialogs.
        Call bbox_target() + confirm_bbox() first to set the target location.
        After tapping, use park() + screenshot() to verify the result.

        If the screen didn't change, the stylus may not have registered.
        Just call tap() again — the confirmed bbox is retained. No need to
        re-confirm. This applies to all gestures (tap, double_tap, etc.).
        """
        physiclaw.require_hardware()
        physiclaw.acquire()
        try:
            _maybe_move_to_bbox()
            physiclaw.arm.tap()
            return "Tapped"
        finally:
            physiclaw.release()

    @mcp.tool()
    def double_tap() -> str:
        """Double tap — two quick taps in succession.

        Use for: zooming in (maps, photos, web pages), selecting a word in text.
        Call bbox_target() + confirm_bbox() first to set the target location.
        """
        physiclaw.require_hardware()
        physiclaw.acquire()
        try:
            _maybe_move_to_bbox()
            physiclaw.arm.double_tap()
            return "Double tapped"
        finally:
            physiclaw.release()

    @mcp.tool()
    def long_press() -> str:
        """Long press — holds contact for ~1.2 seconds.

        Use for: opening context menus, entering edit/selection mode, selecting text,
        rearranging home screen icons, or any action that requires a sustained press.
        Call bbox_target() + confirm_bbox() first to set the target location.
        """
        physiclaw.require_hardware()
        physiclaw.acquire()
        try:
            _maybe_move_to_bbox()
            physiclaw.arm.long_press()
            return "Long pressed"
        finally:
            physiclaw.release()

    @mcp.tool()
    def swipe(direction: str, speed: str = "medium") -> str:
        """Swipe in a cardinal direction — the stylus touches down, slides, and lifts.

        Use for: scrolling content, switching pages, pulling down notifications,
        unlocking the phone, navigating between screens.
        Call bbox_target() + confirm_bbox() first to set the starting position.

        Args:
            direction: 'top', 'bottom', 'left', 'right'
            speed: 'slow' (gentle scroll), 'medium' (normal), 'fast' (fling)
        """
        physiclaw.require_hardware()
        physiclaw.acquire()
        try:
            _maybe_move_to_bbox()
            physiclaw.arm.swipe(direction, speed)
            return f"Swiped {direction} {speed}"
        finally:
            physiclaw.release()
