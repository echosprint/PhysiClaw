"""
All MCP tools — gestures, detection, bridge, and annotation workflows.

Every @mcp.tool() in the PhysiClaw server lives in this file. The
server/{bridge,annotation,calibration,hardware}.py modules only register
HTTP routes. Splitting tools across multiple files would make discovery
harder; this single file is the canonical agent surface.

Each tool acquires the hardware lock when it touches arm/camera state,
runs its operation, and releases. Tools that only read or only mutate
state objects (annotation queue, bridge text) skip the hardware lock.
"""

import logging
import time
from datetime import datetime

import cv2
from mcp.server.fastmcp import FastMCP, Image

from physiclaw.annotation import AnnotationState
from physiclaw.bridge import BridgeState, CalibrationState, get_lan_ip
from physiclaw.core import PhysiClaw
from physiclaw.hardware.camera import SNAPSHOT_DIR
from physiclaw.vision.render import (
    draw_bbox,
    draw_grid_overlay,
    encode_jpeg,
)

log = logging.getLogger(__name__)


# Lazy-cached vision detectors. Created on first detect_elements() call,
# kept for the process lifetime so subsequent calls don't re-load the
# (slow-to-init) ONNX/Paddle models.
_detector_cache: dict = {"icon": None, "ocr": None}


def _save_frame(frame, suffix: str) -> None:
    """Write a BGR frame into data/snapshot/ with a timestamp + suffix."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    cv2.imwrite(str(SNAPSHOT_DIR / f"{ts}_{suffix}.jpg"), frame)


def register(
    mcp: FastMCP,
    physiclaw: PhysiClaw,
    bridge: BridgeState,
    calib: CalibrationState,
    ann: AnnotationState,
):
    """Register every MCP tool on the given FastMCP instance.

    Args:
        mcp: FastMCP server instance.
        physiclaw: PhysiClaw orchestrator (hardware lifecycle + bbox state).
        bridge: BridgeState for text/clipboard transfer + screenshot upload.
        calib: CalibrationState for the bridge_status screen-dimension lookup.
        ann: AnnotationState for the propose/confirm bbox workflow.
    """

    @mcp.tool()
    def camera_view() -> Image:
        """Capture the overhead camera's view of the phone.

        Use this to read screen content, check stylus position, or verify
        results. The stylus may be visible in the frame — call park() first
        if you need an unobstructed view of the screen.

        This is the camera looking down at the phone. For a pixel-perfect
        screenshot taken by the phone itself, use phone_screenshot() instead.
        """
        physiclaw.require_hardware()
        physiclaw.acquire()
        try:
            frame = physiclaw.camera_view()
            return Image(data=encode_jpeg(frame), format="jpeg")
        finally:
            physiclaw.release()

    @mcp.tool()
    def park() -> str:
        """Park the stylus out of the camera frame so it doesn't occlude the screen.

        Call this before camera_view() when you need a clear view of the full
        screen (e.g. to read text or identify UI elements). The stylus moves
        100mm away and will need to be repositioned with move() afterward.
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
        from physiclaw.vision.detect import detect_all_elements

        physiclaw.require_hardware()
        physiclaw.acquire()
        try:
            physiclaw.park()
            time.sleep(1.5)
            frame = physiclaw.camera_view()

            # Lazy-create cached detectors on first use
            if _detector_cache["icon"] is None:
                try:
                    from physiclaw.vision.icon_detect import IconDetector

                    _detector_cache["icon"] = IconDetector()
                except (ImportError, FileNotFoundError):
                    pass  # detect_all_elements handles missing detector
            if _detector_cache["ocr"] is None:
                try:
                    from physiclaw.vision.ocr import OCRReader

                    _detector_cache["ocr"] = OCRReader()
                except ImportError:
                    pass

            elements_text, color_frame, icon_frame, ocr_frame = detect_all_elements(
                frame,
                physiclaw.transforms,
                icon_detector=_detector_cache["icon"],
                ocr_reader=_detector_cache["ocr"],
            )

            _save_frame(color_frame, "colors")
            _save_frame(icon_frame, "icons")
            _save_frame(ocr_frame, "ocr")

            return [
                elements_text,
                Image(data=encode_jpeg(color_frame), format="jpeg"),
                Image(data=encode_jpeg(icon_frame), format="jpeg"),
                Image(data=encode_jpeg(ocr_frame), format="jpeg"),
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
            frame = physiclaw.camera_view()
        finally:
            physiclaw.release()

        result = match_screen(frame, ref)
        has_overlay = detect_dark_overlay(frame)

        lines = [f"Screen match: {'YES' if result.matched else 'NO'}"]
        lines.append(
            f"Confidence: {result.confidence:.1%} "
            f"({result.good_matches} matches, {result.inliers} inliers)"
        )
        if has_overlay:
            lines.append(
                "WARNING: dark overlay detected — possible popup or modal dialog"
            )
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
            frame_a = physiclaw.camera_view()
            time.sleep(1.0)
            frame_b = physiclaw.camera_view()
        finally:
            physiclaw.release()

        changed = frames_differ(frame_a, frame_b)
        status = (
            "Screen CHANGED — the gesture had an effect"
            if changed
            else "Screen UNCHANGED — the gesture may not have registered, try again"
        )
        return [status, Image(data=encode_jpeg(frame_b), format="jpeg")]

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
            frame = physiclaw.camera_view()
            out = draw_grid_overlay(frame, physiclaw.transforms, color, rows, cols)
            _save_frame(out, "overlay")
            return Image(data=encode_jpeg(out), format="jpeg")
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
            frame = physiclaw.camera_view()
            out = draw_bbox(frame, bbox, physiclaw.transforms)
            _save_frame(out, "bbox")
            return Image(data=encode_jpeg(out), format="jpeg")
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
        bbox = physiclaw.confirmed_bbox
        if bbox is not None:
            physiclaw.move_to_bbox_center(bbox)

    @mcp.tool()
    def tap() -> str:
        """Single tap — like a finger tap on the screen.

        Use for: pressing buttons, selecting items, opening apps, following links, dismissing dialogs.
        Call bbox_target() + confirm_bbox() first to set the target location.
        After tapping, use park() + camera_view() to verify the result.

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

    # ─── LAN bridge tools ───────────────────────────────────

    @mcp.tool()
    def bridge_status() -> str:
        """Check the LAN bridge and get the URL for the phone to open.

        The bridge enables fast text transfer via clipboard. The user opens the
        URL on the phone in Safari/Chrome. Then the agent can send text and
        tap the screen to copy it to the phone's clipboard — much faster than
        typing on the keyboard character by character.

        Returns the URL, connection status, and device info if available.
        """
        ip = get_lan_ip()
        port = mcp.settings.port
        phone_url = f"http://{ip}:{port}/bridge"

        lines = [f"Phone URL: {phone_url}"]
        if bridge.connected:
            lines.append("Status: connected ✓")
        else:
            lines.append(
                "Status: not connected — ask the user to open the URL on their phone"
            )

        if calib.screen_dimension:
            d = calib.screen_dimension
            lines.append(f"Screen: {d['width']}×{d['height']}pt")
        return "\n".join(lines)

    @mcp.tool()
    def bridge_send_text(text: str) -> str:
        """Send text to the phone clipboard.

        Two ways to copy:
        1. Phone has /bridge page open → call bridge_tap() to tap and copy
        2. Phone runs "PhysiClaw Clipboard" shortcut (long-press AssistiveTouch)
           → shortcut GETs /api/bridge/clipboard → text copied directly

        Either way, the text ends up in the phone's clipboard.
        Paste into any app: long_press on a text field → tap "Paste".
        """
        bridge.send_text(text)
        return f"Text '{text}' ready. Copy via bridge_tap() or AT long-press shortcut."

    @mcp.tool()
    def bridge_tap() -> str:
        """Tap the phone screen center to copy bridge text to clipboard.

        The phone page fills the screen. Tapping anywhere triggers the
        JavaScript clipboard copy. This tool taps screen center (0.5, 0.5),
        then waits up to 5 seconds for the phone to confirm.

        Call bridge_send_text() first to set the text.
        After this tool returns, the text is in the phone's clipboard.
        """
        physiclaw.require_hardware()
        physiclaw.acquire()
        try:
            physiclaw.tap_at_pct(0.5, 0.5)
        finally:
            physiclaw.release()

        if bridge.wait_clipboard(timeout=5.0):
            return (
                f"Clipboard ready — '{bridge.current_text()}' copied to phone clipboard"
            )
        return (
            "Tap sent but clipboard copy not confirmed. "
            "Is the phone page open in the foreground on the phone?"
        )

    @mcp.tool()
    def phone_screenshot() -> Image:
        """Take a pixel-perfect screenshot via AssistiveTouch.

        Single-tap AT takes a screenshot (saved to Photos), then double-tap AT
        triggers the iOS Shortcut to get the latest screenshot and upload it.

        Much sharper than camera screenshots — use for skill building, OCR,
        and detailed UI analysis.

        Requires: /setup completed (step 7 verifies AT position).
        """
        physiclaw.require_hardware()
        if not physiclaw.assistive_touch.ready:
            raise RuntimeError("AssistiveTouch not set up — run /setup first (step 7)")

        pct_to_grbl = physiclaw._cal.get("screen_to_grbl")
        if pct_to_grbl is None:
            raise RuntimeError("Calibration incomplete — run /setup first")

        physiclaw.acquire()
        try:
            data = physiclaw.assistive_touch.take_screenshot(
                physiclaw._arm, bridge, pct_to_grbl
            )
        finally:
            physiclaw.release()

        if data is None:
            raise RuntimeError(
                "Timeout — no screenshot received. Check iOS Shortcut is configured."
            )

        fmt = "png" if data[:4] == b"\x89PNG" else "jpeg"
        return Image(data=data, format=fmt)

    # ─── Annotation tools ───────────────────────────────────

    def _format_confirmed(boxes: list[dict], header: str) -> str:
        """Render a list of confirmed annotations as markdown."""
        lines = [f"# {header} ({len(boxes)} items)\n"]
        for i, box in enumerate(boxes):
            b = box["bbox"]
            box_type = box.get("type", "box")
            label = box.get("label", "")
            source = box.get("source", "user")
            src = f" [{source}]" if source != "user" else ""
            desc = f" — {label}" if label else ""
            coords = ", ".join(str(v) for v in b)
            type_tag = f" ({box_type})" if box_type != "box" else ""
            lines.append(f"- {i + 1}{type_tag}{src}: [{coords}]{desc}")
        return "\n".join(lines)

    @mcp.tool()
    def get_user_annotations() -> list:
        """Get confirmed annotations from the annotation UI.

        Returns the confirmed boxes with coordinates and labels, plus the
        frozen screenshot. The user must click Confirm in the annotation UI
        before this returns data.

        Use wait_for_confirmation() instead if you want to block until
        the user confirms. This tool returns immediately — it returns
        whatever was last confirmed, or "no annotations" if nothing was confirmed.
        """
        with ann.lock:
            confirmed = list(ann.confirmed_annotations)
        frozen_frame = ann.get_frozen_frame()
        if not confirmed:
            return [
                "No confirmed annotations. "
                f"Ask the user to draw boxes at http://{mcp.settings.host}:{mcp.settings.port}/annotate and click Confirm."
            ]

        text = _format_confirmed(confirmed, "Confirmed Annotations")
        if frozen_frame is not None:
            return [text, Image(data=encode_jpeg(frozen_frame), format="jpeg")]
        return [text]

    @mcp.tool()
    def propose_bboxes(proposals: list[dict]) -> str:
        """Propose bounding boxes for the user to review in the annotation UI.

        Sends your coordinate guesses to the annotation web UI at /annotate.
        The user can move, resize, delete, relabel, or add new boxes.
        After the user confirms, call wait_for_confirmation() to get the result.

        Parks the arm and takes a fresh screenshot automatically.

        Args:
            proposals: list of {"bbox": [left, top, right, bottom], "label": "element name"}
                       Coordinates are 0-1 decimals (phone screen).
        """
        physiclaw.require_hardware()
        physiclaw.acquire()
        try:
            physiclaw.park()
            time.sleep(1.5)

            frame = physiclaw.cam.peek()
            if frame is None:
                return "Camera capture failed"

            snapshot_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            ann.freeze(frame, snapshot_id)
            ann.push_agent_proposals(proposals)

            url = f"http://{mcp.settings.host}:{mcp.settings.port}/annotate"
            return (
                f"{len(proposals)} proposals sent to annotation UI. "
                f"Ask the user to review and confirm at {url}"
            )
        finally:
            physiclaw.release()

    @mcp.tool()
    def wait_for_confirmation(timeout: int = 120) -> list:
        """Wait for the user to confirm bounding boxes in the annotation UI.

        Blocks until the user clicks Confirm at /annotate, or until timeout.
        Returns the confirmed boxes with user-corrected coordinates and labels.

        Call this after propose_bboxes() or after asking the user to draw boxes.

        Args:
            timeout: seconds to wait before giving up (default 120)
        """
        result = ann.wait_confirmed(timeout=float(timeout))
        if result is None:
            return [
                "Timeout — the user hasn't confirmed yet. "
                f"Ask them if they need help at http://{mcp.settings.host}:{mcp.settings.port}/annotate"
            ]

        frozen_frame = ann.get_frozen_frame()
        ann.clear_confirmation()

        text = _format_confirmed(result, "Confirmed Annotations")
        if frozen_frame is not None:
            return [text, Image(data=encode_jpeg(frozen_frame), format="jpeg")]
        return [text]
