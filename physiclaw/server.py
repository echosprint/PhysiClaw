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

    Parks the stylus, takes a clean screenshot, and runs two detectors:
    - Icon detection: finds buttons, icons, and interactive elements
    - OCR: reads all visible text (labels, keys, prices, etc.)

    Returns a text listing of all elements with bounding boxes as 0-1
    decimals [left, top, right, bottom], plus two annotated images
    (icon boxes + OCR boxes). Use the coordinates to call bbox_target().

    The detection models are lightweight (<100MB) so bounding boxes may not
    be pixel-perfect. Use them as estimates to narrow down your target's
    position, then refine with bbox_target() for precise targeting.

    Requires vision models to be set up first (run /setup-vision-models).
    If a model isn't installed, its section shows "unavailable" instead of results.
    """
    import time
    physiclaw.require_hardware()
    physiclaw.acquire()
    try:
        physiclaw.park()
        time.sleep(1.5)
        elements_text, icon_frame, ocr_frame = physiclaw.detect_elements()
        return [
            elements_text,
            Image(data=physiclaw.frame_to_jpeg(icon_frame), format="jpeg"),
            Image(data=physiclaw.frame_to_jpeg(ocr_frame), format="jpeg"),
        ]
    finally:
        physiclaw.release()


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
    try:
        jpeg = await asyncio.get_event_loop().run_in_executor(
            None, PhysiClaw.camera_preview, index)
        return JSONResponse({"status": "ok", "index": index,
                             "image": base64.b64encode(jpeg).decode()})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)},
                            status_code=404)


_CALIBRATE_STEPS = [
    ("z-depth", "calibrate_z_depth"),
    ("find-right", "calibrate_find_right"),
    ("find-down", "calibrate_find_down"),
    ("long-press", "calibrate_long_press"),
    ("swipe", "calibrate_swipe"),
    ("grid", "calibrate_grid"),
]

for _slug, _method_name in _CALIBRATE_STEPS:
    def _make_handler(method_name):
        async def _handler(request):
            import asyncio
            from starlette.responses import JSONResponse

            def _do():
                physiclaw.acquire()
                try:
                    return getattr(physiclaw, method_name)()
                finally:
                    physiclaw.release()

            try:
                result = await asyncio.get_event_loop().run_in_executor(None, _do)
                return JSONResponse({"status": "ok", **result})
            except Exception as e:
                return JSONResponse({"status": "error", "message": str(e)},
                                    status_code=500)
        return _handler

    mcp.custom_route(f"/api/calibrate/{_slug}", methods=["POST"])(
        _make_handler(_method_name))


@mcp.custom_route("/api/calibrate/verify-edge", methods=["POST"])
async def _verify_edge(request):
    import asyncio
    from starlette.responses import JSONResponse

    def _do():
        physiclaw.acquire()
        try:
            return physiclaw.verify_edge_trace()
        finally:
            physiclaw.release()

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _do)
        return JSONResponse({"status": "ok", **result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)},
                            status_code=500)


