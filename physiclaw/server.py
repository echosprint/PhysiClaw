"""
PhysiClaw MCP Server — tool definitions, routes, and setup endpoints.

Started by physiclaw.main. Server starts instantly without hardware.
Run /setup to connect and calibrate.
"""

import logging

import cv2
from mcp.server.fastmcp import FastMCP, Image

from physiclaw.core import PhysiClaw

# ─── MCP server ─────────────────────────────────────────────

mcp = FastMCP(
    "physiclaw",
    instructions="""PhysiClaw gives you a physical finger (robotic stylus arm) and an eye (camera) to operate any phone.

You control a real phone sitting on a desk — a camera sees the screen from directly above, and a 3-axis arm moves and taps a capacitive stylus.

## Before every tap, classify the target

**Fixed UI element** (button, icon, nav control, text field, keyboard key — same position every visit):
- In preset (.claude/ui-presets/) → use preset coordinates with bbox_target(bbox).
- Not in preset → propose_bboxes() → wait_for_confirmation() → save to preset.
- You CANNOT guess coordinates for fixed UI. Your estimates are unreliable.

**Dynamic content** (list item, menu entry, product card — large, changes every visit):
- Visual targeting OK: grid_overlay() → bbox_target(bbox) → label test → confirm_bbox() → tap.

## Operation cycle (dynamic content / preset path)

1. Check UI presets — if the target has known coordinates, use them directly.
2. If no preset: park() + screenshot(). Optionally detect_elements() to find icons + text with coordinates. Or grid_overlay() to estimate manually.
3. bbox_target(bbox) — bbox = [left, top, right, bottom] as 0-1 decimals.
4. **Label test:** name the element INSIDE each rectangle.
   - Covers the target → confirm_bbox()
   - Misses → call bbox_target() with corrected coordinates. 2-3 attempts is normal.
5. tap() / double_tap() / long_press() / swipe() — executes at the bbox center.
6. park() + screenshot() — verify the result.

## Propose-confirm cycle (fixed UI without preset)

1. park() + screenshot() — reason about visible elements.
2. propose_bboxes([{"bbox": [l,t,r,b], "label": "..."}]) — sends guesses to /annotate.
3. Tell user to review and confirm at /annotate.
4. wait_for_confirmation() — blocks until user confirms.
5. Use confirmed coordinates: bbox_target(bbox) → confirm_bbox() → gesture.
6. Save to preset for future autonomous use.

## Setup

All tools require hardware to be set up first. If you get "Hardware not set up",
tell the user to run /setup. Do not attempt to call setup endpoints yourself.

## CRITICAL

- bbox_target() is cheap (just a photo). tap() is expensive (physical arm, irreversible).
- Never guess coordinates for fixed UI elements — propose and let the user confirm.
- Before confirming, ask: "Am I choosing this because it COVERS the target, or because it's the closest option?" If closest → reject, re-bbox.
""",
)

# ─── Tools ──────────────────────────────────────────────────

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
    import time
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
    import time
    from pathlib import Path
    from physiclaw.screen_match import match_screen, detect_dark_overlay

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
    import time
    from physiclaw.screen_match import frames_differ

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
    import time
    density_map = {
        "sparse": (4, 2),    # rows, cols — lines at 0.20/0.40/0.60/0.80 x 0.25/0.50/0.75
        "normal": (9, 4),    # lines at 0.10..0.90 x 0.20/0.40/0.60/0.80
        "dense": (19, 9),    # lines at 0.05..0.95 x 0.10..0.90
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
    import time
    physiclaw.require_hardware()
    physiclaw.acquire()
    try:
        physiclaw.park()
        time.sleep(1.5)  # let arm settle after parking
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
    """If a bbox is confirmed, move arm to its center.
    The bbox is kept for retry — calling tap() again reuses the same target.
    Cleared automatically on the next bbox_target() call.
    """
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


# ─── Module-level state ───────────────────────────────────────

physiclaw = PhysiClaw()


def shutdown():
    """Clean up hardware resources."""
    physiclaw.shutdown()

# ─── LAN Bridge routes + tools ────────────────────────────────

from physiclaw.bridge import (
    BridgeState, CalibrationState, PhoneState, get_lan_ip,
    serve_bridge_page, serve_qr_page,
    handle_clipboard_copied, handle_screen_dimension,
    handle_screenshot_upload, handle_phone_state,
    handle_calib_touch,
)

_bridge = BridgeState()
_calib = CalibrationState()
_phone = PhoneState(_bridge, _calib)

# Phone opens this page — shows bridge UI (text clipboard) or calibration UI
@mcp.custom_route("/bridge", methods=["GET"])
async def _phone_page(request):
    return await serve_bridge_page(request)

# Phone polls every 250ms — returns current mode (bridge/calibrate) and mode-specific data
@mcp.custom_route("/api/bridge/state", methods=["GET"])
async def _phone_state(request):
    return await handle_phone_state(request, _phone)

# Displays a QR code linking to /bridge for easy phone scanning
@mcp.custom_route("/api/bridge/qr", methods=["GET"])
async def _qr(request):
    return await serve_qr_page(request)

# Phone confirms tap-to-copy succeeded — unblocks bridge_tap() tool
@mcp.custom_route("/api/bridge/tapped", methods=["POST"])
async def _bridge_tapped(request):
    return await handle_clipboard_copied(request, _bridge)

# Phone sends screen dimensions, pixel ratio, safe area on page load
@mcp.custom_route("/api/bridge/screen-dimension", methods=["POST"])
async def _bridge_screen_dimension(request):
    return await handle_screen_dimension(request, _calib)

# iOS Shortcut uploads a screenshot (raw PNG/JPEG body)
@mcp.custom_route("/api/bridge/screenshot", methods=["POST"])
async def _bridge_screenshot(request):
    return await handle_screenshot_upload(request, _bridge)


# ─── Calibration routes ──────────────────────────────────────

# Switch phone page between "bridge" and "calibrate" mode
@mcp.custom_route("/api/bridge/switch", methods=["POST"])
async def _bridge_switch(request):
    from starlette.responses import JSONResponse
    body = await request.json()
    mode = body.get("mode")
    if mode not in ("bridge", "calibrate"):
        return JSONResponse({"error": "mode must be 'bridge' or 'calibrate'"}, status_code=400)
    if mode == "calibrate":
        phase = body.get("phase")
        if not phase:
            return JSONResponse({"error": "phase required for calibrate mode"}, status_code=400)
        kwargs = {k: v for k, v in body.items() if k not in ("mode", "phase")}
        try:
            _phone.set_mode(mode, phase, **kwargs)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"ok": True, "mode": mode, "phase": phase})
    _phone.set_mode(mode)
    return JSONResponse({"ok": True, "mode": mode})

# Phone reports a touch event during calibration (tap/long-press/swipe with coords)
@mcp.custom_route("/api/bridge/touch", methods=["POST"])
async def _calib_touch(request):
    return await handle_calib_touch(request, _calib)



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
    if _bridge.connected:
        lines.append("Status: connected ✓")
    else:
        lines.append("Status: not connected — ask the user to open the URL on their phone")

    if _calib.screen_dimension:
        d = _calib.screen_dimension
        lines.append(f"Screen: {d['width']}×{d['height']}pt")
    return "\n".join(lines)


@mcp.tool()
def bridge_send_text(text: str) -> str:
    """Send text to the phone page for clipboard transfer.

    The text appears large on the bridge page. Next step: call bridge_tap()
    to physically tap the screen — this triggers the JavaScript clipboard
    copy on the phone.

    After bridge_tap() confirms, the text is in the phone's clipboard.
    Paste it into any app: long_press on a text field → tap "Paste".

    Phone must have the phone page open (call bridge_status() for URL).
    """
    if not _bridge.connected:
        return ("Phone not connected to bridge. "
                "Call bridge_status() to get the URL, then ask the user "
                "to open it on their phone in Safari or Chrome.")
    _bridge.send_text(text)
    return f"Text '{text}' sent to phone. Call bridge_tap() to copy to clipboard."


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

    if _bridge.wait_clipboard(timeout=5.0):
        return f"Clipboard ready — '{_bridge.text}' copied to phone clipboard"
    return ("Tap sent but clipboard copy not confirmed. "
            "Is the phone page open in the foreground on the phone?")


@mcp.tool()
def phone_screenshot(assistive_touch_bbox: list[float] | None = None) -> Image:
    """Take a pixel-perfect screenshot via AssistiveTouch.

    Taps the AssistiveTouch button on the phone. An iOS Shortcut captures a
    screenshot and uploads it to the server. Returns the full-resolution PNG.

    Much sharper than camera screenshots — use this for skill building, OCR,
    and detailed UI analysis.

    Requires setup first (run /setup-screenshot):
    1. AssistiveTouch enabled with single-tap → Run Shortcut
    2. Shortcut: Take Screenshot → Get Contents of URL (POST to server)

    Args:
        assistive_touch_bbox: [left, top, right, bottom] as 0-1 decimals for
            the AssistiveTouch button. If None, reads from the
            .claude/ui-presets/system.md preset file.
    """
    import json
    from pathlib import Path

    # Resolve AssistiveTouch button position
    bbox = assistive_touch_bbox
    if bbox is None:
        preset_path = Path(".claude/ui-presets/system.md")
        if not preset_path.exists():
            return "AssistiveTouch position unknown. Run /setup-screenshot first, " \
                   "or pass assistive_touch_bbox=[left, top, right, bottom]."
        # Parse preset for AssistiveTouch entry
        content = preset_path.read_text()
        for line in content.splitlines():
            if "AssistiveTouch" in line and "|" in line:
                # Extract position from table row: | Element | Position | Action |
                parts = [p.strip() for p in line.split("|")]
                for p in parts:
                    if p.startswith("[") and p.endswith("]"):
                        bbox = json.loads(p)
                        break
                if bbox:
                    break
        if bbox is None:
            return "AssistiveTouch position not found in system.md preset. " \
                   "Run /setup-screenshot to configure."

    physiclaw.require_hardware()

    # Clear any pending screenshot
    _bridge._screenshot_ready.clear()

    # Tap the AssistiveTouch button
    physiclaw.acquire()
    try:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        physiclaw.tap_at_pct(cx, cy)
    finally:
        physiclaw.release()

    # Wait for the screenshot to arrive via HTTP POST
    data = _bridge.wait_screenshot(timeout=10.0)
    if data is None:
        return "Timeout — no screenshot received. Check that the iOS Shortcut " \
               "is configured to POST to this server."

    # Save to data/screenshot/
    from datetime import datetime
    from physiclaw.camera import SNAPSHOT_DIR
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    ext = "png" if data[:4] == b'\x89PNG' else "jpg"
    save_path = SNAPSHOT_DIR / f'{ts}_phone.{ext}'
    save_path.write_bytes(data)

    # Detect format for MCP Image
    fmt = "png" if ext == "png" else "jpeg"
    return Image(data=data, format=fmt)


# ─── Annotation routes + tool ──────────────────────────────────

from physiclaw.annotation import (
    AnnotationState, freeze_snapshot, get_frozen_snapshot,
    handle_annotations, handle_confirm, serve_annotate_page,
)

_ann = AnnotationState()

@mcp.custom_route("/annotate", methods=["GET"])
async def _annotate(request):
    return await serve_annotate_page(request)

@mcp.custom_route("/api/snapshot", methods=["GET", "POST"])
async def _snapshot(request):
    if request.method == "GET":
        return await get_frozen_snapshot(request, physiclaw, _ann)
    physiclaw.acquire()
    try:
        return await freeze_snapshot(request, physiclaw, _ann)
    finally:
        physiclaw.release()

@mcp.custom_route("/api/annotations", methods=["GET", "DELETE"])
async def _annotations(request):
    return await handle_annotations(request, _ann)

@mcp.custom_route("/api/confirm", methods=["POST"])
async def _confirm(request):
    return await handle_confirm(request, _ann, physiclaw)

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
    with _ann.lock:
        confirmed = list(_ann.confirmed_annotations)
    frozen_frame = _ann.get_frozen_frame()
    if not confirmed:
        return ["No confirmed annotations. "
                f"Ask the user to draw boxes at http://{mcp.settings.host}:{mcp.settings.port}/annotate and click Confirm."]

    lines = [f"# Confirmed Annotations ({len(confirmed)} items)\n"]
    for i, box in enumerate(confirmed):
        b = box['bbox']
        box_type = box.get('type', 'box')
        label = box.get('label', '')
        source = box.get('source', 'user')
        src = f" [{source}]" if source != 'user' else ""
        desc = f" — {label}" if label else ""
        coords = ", ".join(str(v) for v in b)
        type_tag = f" ({box_type})" if box_type != 'box' else ""
        lines.append(f"- {i+1}{type_tag}{src}: [{coords}]{desc}")
    text = "\n".join(lines)

    if frozen_frame is not None:
        return [text, Image(data=physiclaw.frame_to_jpeg(frozen_frame),
                            format="jpeg")]
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
    import time
    physiclaw.require_hardware()
    physiclaw.acquire()
    try:
        physiclaw.park()
        time.sleep(1.5)

        # Freeze a fresh snapshot for the annotation UI
        frame = physiclaw.cam._fresh_frame()
        if frame is None:
            return "Camera capture failed"
        import cv2
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        from datetime import datetime
        snapshot_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        _ann.freeze(frame, snapshot_id)

        # Push proposals to staging area for UI to pick up
        _ann.push_agent_proposals(proposals)

        url = f"http://{mcp.settings.host}:{mcp.settings.port}/annotate"
        return (f"{len(proposals)} proposals sent to annotation UI. "
                f"Ask the user to review and confirm at {url}")
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
    result = _ann.wait_confirmed(timeout=float(timeout))
    if result is None:
        return ["Timeout — the user hasn't confirmed yet. "
                f"Ask them if they need help at http://{mcp.settings.host}:{mcp.settings.port}/annotate"]

    frozen_frame = _ann.get_frozen_frame()
    _ann.clear_confirmation()

    lines = [f"# Confirmed Annotations ({len(result)} items)\n"]
    for i, box in enumerate(result):
        b = box['bbox']
        box_type = box.get('type', 'box')
        label = box.get('label', '')
        source = box.get('source', 'user')
        src = f" [{source}]" if source != 'user' else ""
        desc = f" — {label}" if label else ""
        coords = ", ".join(str(v) for v in b)
        type_tag = f" ({box_type})" if box_type != 'box' else ""
        lines.append(f"- {i+1}{type_tag}{src}: [{coords}]{desc}")
    text = "\n".join(lines)

    if frozen_frame is not None:
        return [text, Image(data=physiclaw.frame_to_jpeg(frozen_frame),
                            format="jpeg")]
    return [text]

# ─── Setup endpoints (called by /setup skill) ────────────────────

@mcp.custom_route("/api/status", methods=["GET"])
async def _status(request):
    from starlette.responses import JSONResponse
    return JSONResponse(physiclaw.status())


@mcp.custom_route("/api/connect-arm", methods=["POST"])
async def _connect_arm(request):
    import asyncio
    from starlette.responses import JSONResponse

    def _do():
        physiclaw.acquire()
        try:
            physiclaw.connect_arm()
        finally:
            physiclaw.release()

    try:
        await asyncio.get_event_loop().run_in_executor(None, _do)
        return JSONResponse({"status": "ok", "message": "Arm connected"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)},
                            status_code=500)


@mcp.custom_route("/api/connect-camera", methods=["POST"])
async def _connect_camera(request):
    import asyncio
    from starlette.responses import JSONResponse
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    index = body.get("index")  # None = auto-detect

    def _do():
        physiclaw.acquire()
        try:
            physiclaw.connect_camera(index)
        finally:
            physiclaw.release()

    try:
        await asyncio.get_event_loop().run_in_executor(None, _do)
        return JSONResponse({"status": "ok",
                             "message": f"Camera {physiclaw._cam.index} connected",
                             "index": physiclaw._cam.index})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)},
                            status_code=500)


@mcp.custom_route("/api/camera-preview/{index}", methods=["GET"])
async def _camera_preview(request):
    import asyncio
    import base64
    from starlette.responses import JSONResponse
    index = int(request.path_params["index"])
    watermark = request.query_params.get("watermark", "0") == "1"
    try:
        jpeg = await asyncio.get_event_loop().run_in_executor(
            None, PhysiClaw.camera_preview, index, watermark)
        return JSONResponse({"status": "ok", "index": index,
                             "image": base64.b64encode(jpeg).decode()})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)},
                            status_code=404)


# ─── Plan calibration endpoints (touch + camera) ─────────────
# Matches architecture plan Steps 0-6. No green flash.
# Touch coordinates from /calibrate page. Camera for markers/dots.

from physiclaw.plan_calibrate import (
    step0_z_depth, step1_alignment, step2_camera_rotation,
    step3_software_rotation, step4_grbl_screen, step5_camera_screen,
    step6_validate,
)


@mcp.custom_route("/api/calibrate/step0-z-depth", methods=["POST"])
async def _step0(request):
    import asyncio
    from starlette.responses import JSONResponse
    def _do():
        if physiclaw._arm is None:
            raise RuntimeError("Arm not connected")
        _phone.set_mode("calibrate")
        physiclaw.acquire()
        try:
            z_tap = step0_z_depth(physiclaw._arm, _calib)
            physiclaw._arm.Z_DOWN = z_tap
            physiclaw._cal['z_tap'] = z_tap
            return {"z_tap": z_tap}
        finally:
            physiclaw.release()
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _do)
        return JSONResponse({"status": "ok", **result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@mcp.custom_route("/api/calibrate/step1-alignment", methods=["POST"])
async def _step1(request):
    import asyncio
    from starlette.responses import JSONResponse
    def _do():
        if physiclaw._arm is None:
            raise RuntimeError("Arm not connected")
        z_tap = physiclaw._cal.get('z_tap')
        if z_tap is None:
            raise RuntimeError("Run step0 first")
        physiclaw.acquire()
        try:
            tilt = step1_alignment(physiclaw._arm, _calib, z_tap)
            return {"tilt_ratio": round(tilt, 4), "aligned": tilt < 0.02}
        finally:
            physiclaw.release()
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _do)
        return JSONResponse({"status": "ok", **result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@mcp.custom_route("/api/calibrate/step2-camera-rotation", methods=["POST"])
async def _step2(request):
    import asyncio
    from starlette.responses import JSONResponse
    def _do():
        if physiclaw._cam is None:
            raise RuntimeError("Camera not connected")
        result = step2_camera_rotation(physiclaw._cam)
        return result
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _do)
        return JSONResponse({"status": "ok", **result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@mcp.custom_route("/api/calibrate/step3-sw-rotation", methods=["POST"])
async def _step3(request):
    import asyncio
    from starlette.responses import JSONResponse
    def _do():
        if physiclaw._cam is None:
            raise RuntimeError("Camera not connected")
        rotation = step3_software_rotation(physiclaw._cam, _calib)
        physiclaw._cal['rotation'] = rotation
        name = {-1: "none", 0: "90° CW", 1: "180°", 2: "90° CCW"}.get(rotation, str(rotation))
        return {"rotation": rotation, "rotation_name": name}
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _do)
        return JSONResponse({"status": "ok", **result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@mcp.custom_route("/api/calibrate/step4-mapping-a", methods=["POST"])
async def _step4(request):
    import asyncio
    from starlette.responses import JSONResponse
    def _do():
        if physiclaw._arm is None:
            raise RuntimeError("Arm not connected")
        z_tap = physiclaw._cal.get('z_tap')
        if z_tap is None:
            raise RuntimeError("Run step0 first")
        physiclaw.acquire()
        try:
            pct_to_grbl, touches = step4_grbl_screen(physiclaw._arm, _calib, z_tap)
            physiclaw._cal['screen_to_grbl'] = pct_to_grbl
            return {"ok": True, "pairs": len(touches)}
        finally:
            physiclaw.release()
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _do)
        return JSONResponse({"status": "ok", **result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@mcp.custom_route("/api/calibrate/step5-mapping-b", methods=["POST"])
async def _step5(request):
    import asyncio
    from starlette.responses import JSONResponse
    def _do():
        if physiclaw._cam is None:
            raise RuntimeError("Camera not connected")
        rotation = physiclaw._cal.get('rotation', cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Park arm if possible
        if physiclaw._arm and physiclaw._arm.MOVE_DIRECTIONS:
            ux, uy = physiclaw._arm.MOVE_DIRECTIONS['top']
            physiclaw._arm._fast_move(ux * 100, uy * 100)
            physiclaw._arm.wait_idle()
        pct_to_pixel = step5_camera_screen(physiclaw._cam, _calib, rotation)
        physiclaw._cal['pct_to_pixel'] = pct_to_pixel
        return {"ok": True, "dots": 15}
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _do)
        return JSONResponse({"status": "ok", **result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@mcp.custom_route("/api/calibrate/step6-validate", methods=["POST"])
async def _step6(request):
    import asyncio
    from starlette.responses import JSONResponse
    def _do():
        if physiclaw._arm is None:
            raise RuntimeError("Arm not connected")
        z_tap = physiclaw._cal.get('z_tap')
        rotation = physiclaw._cal.get('rotation', cv2.ROTATE_90_COUNTERCLOCKWISE)
        pct_to_grbl = physiclaw._cal.get('screen_to_grbl')
        pct_to_pixel = physiclaw._cal.get('pct_to_pixel')
        if not all([z_tap, pct_to_grbl is not None, pct_to_pixel is not None]):
            raise RuntimeError("Run steps 0-5 first")
        physiclaw.acquire()
        try:
            results = step6_validate(physiclaw._arm, physiclaw._cam, _calib,
                                     z_tap, rotation, pct_to_grbl, pct_to_pixel)
            passed = sum(1 for r in results if r['passed'])
            # If passed, build and store GridCalibration
            if passed >= 2:
                from physiclaw.grid_calibrate import GridCalibration
                # Step 4 already set origin at screen center and adjusted affine
                physiclaw._grid_cal = GridCalibration(
                    pct_to_grbl=pct_to_grbl, pct_to_pixel=pct_to_pixel)
            if passed >= 2:
                _phone.set_mode("bridge")
            return {"results": results, "passed": passed, "total": len(results),
                    "calibrated": passed >= 2}
        finally:
            physiclaw.release()
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _do)
        return JSONResponse({"status": "ok", **result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@mcp.custom_route("/api/calibrate/verify-edge", methods=["POST"])
async def _verify_edge(request):
    import asyncio
    from starlette.responses import JSONResponse

    def _do():
        physiclaw.acquire()
        try:
            result = physiclaw.verify_edge_trace()
            _phone.set_mode("bridge")
            return result
        finally:
            physiclaw.release()

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _do)
        return JSONResponse({"status": "ok", **result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)},
                            status_code=500)


