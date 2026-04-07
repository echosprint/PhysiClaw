"""Starlette route handlers for the LAN bridge.

These are async functions invoked by the MCP server to serve the phone
bridge page, accept screenshot uploads, handle clipboard confirmations,
and report calibration touch events.
"""

import logging
from pathlib import Path

from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse

from physiclaw.bridge.lan import get_lan_ip
from physiclaw.bridge.state import BridgeState
from physiclaw.bridge.calib import CalibrationState
from physiclaw.bridge.page import PageState

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent.parent / "static"


async def serve_bridge_page(request):
    """Serve the bridge page for the phone browser."""
    return HTMLResponse((STATIC_DIR / "bridge.html").read_text())


async def handle_phone_state(request, phone: PageState):
    """GET /api/bridge/state — unified poll endpoint for the phone page."""
    phone.bridge.poll()  # keep connected status alive in both modes
    return JSONResponse(phone.get_state())


async def handle_clipboard_copied(request, bridge: BridgeState):
    """POST /api/bridge/tapped — phone confirms text was copied to clipboard."""
    bridge.mark_clipboard_copied()
    log.info(f"Bridge: clipboard copied — '{bridge.current_text()}'")
    return JSONResponse({"ok": True})


async def handle_screen_dimension(request, cal: CalibrationState):
    """POST /api/bridge/screen-dimension — phone sends screen dimensions on load."""
    body = await request.json()
    with cal.lock:
        cal.screen_dimension = {
            "width": int(body.get("screen_width", 0)),
            "height": int(body.get("screen_height", 0)),
            "viewport_width": int(body.get("viewport_width", 0)),
            "viewport_height": int(body.get("viewport_height", 0)),
        }
    dim = cal.screen_dimension
    log.info(
        f"Bridge device: {dim['width']}×{dim['height']}pt, "
        f"viewport {dim['viewport_width']}×{dim['viewport_height']}pt"
    )
    return JSONResponse({"ok": True})


async def handle_screenshot_upload(request, bridge: BridgeState):
    """POST /api/bridge/screenshot — iOS Shortcut uploads a screenshot.

    Accepts raw image body (PNG or JPEG). The iOS Shortcut action is:
    "Get Contents of URL" with method POST, body = screenshot image.
    """
    data = await request.body()
    if not data:
        return JSONResponse({"error": "empty body"}, status_code=400)
    bridge.receive_screenshot(data)
    return JSONResponse({"ok": True, "size": len(data)})


async def handle_clipboard_fetch(request, bridge: BridgeState):
    """GET /api/bridge/clipboard — iOS Shortcut fetches the current bridge text.

    Returns the text as plain text (so the Shortcut can write it directly to
    the clipboard) and marks the bridge as copied so bridge_tap() unblocks.
    Returns 204 if no text is queued.
    """
    text = bridge.fetch_text()
    if text is None:
        return PlainTextResponse("", status_code=204)
    log.info(f"Bridge: clipboard fetched — '{text}'")
    return PlainTextResponse(text)


async def handle_mode_switch(request, phone: PageState):
    """POST /api/bridge/switch — switch the phone page between bridge and calibrate modes.

    Body: {"mode": "bridge"} or {"mode": "calibrate", "phase": "<phase>", ...phase_kwargs}
    """
    body = await request.json()
    mode = body.get("mode")
    if mode not in ("bridge", "calibrate"):
        return JSONResponse(
            {"error": "mode must be 'bridge' or 'calibrate'"}, status_code=400
        )
    if mode == "calibrate":
        phase_name = body.get("phase")
        if not phase_name:
            return JSONResponse(
                {"error": "phase required for calibrate mode"}, status_code=400
            )
        kwargs = {k: v for k, v in body.items() if k not in ("mode", "phase")}
        try:
            phone.set_mode(mode, phase_name, **kwargs)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"ok": True, "mode": mode, "phase": phase_name})
    phone.set_mode(mode)
    return JSONResponse({"ok": True, "mode": mode})


async def serve_qr_page(request):
    """Serve a page with a single QR code for the unified phone page."""
    ip = get_lan_ip()
    port = request.url.port or 8048
    phone_url = f"http://{ip}:{port}/bridge"
    html = (STATIC_DIR / "qr.html").read_text().replace("__PHONE_URL__", phone_url)
    return HTMLResponse(html)


async def handle_calib_touch(request, cal: CalibrationState):
    """POST /api/bridge/touch — page reports a touch event.

    Body: {"clientX": float, "clientY": float} viewport CSS coords from
    bridge.html. The screenshot transform is set during pre-cal (which
    always runs before any touch-driven step), so we convert directly to
    screenshot 0-1 coords and attach them as x, y.
    """
    body = await request.json()
    body["x"], body["y"] = cal.viewport_to_screenshot_pct(
        body["clientX"], body["clientY"]
    )
    cal.report_touch(body)
    return JSONResponse({"ok": True})
