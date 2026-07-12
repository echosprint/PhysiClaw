"""Starlette route handlers for the LAN bridge.

These are async functions invoked by the MCP server to serve the phone
bridge page, accept screenshot uploads, handle clipboard confirmations,
and report calibration touch events.
"""

import logging
from pathlib import Path

from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse

from physiclaw.core.bridge.lan import bridge_base_urls
from physiclaw.core.bridge.state import BridgeState
from physiclaw.core.bridge.calib import CalibrationState
from physiclaw.core.bridge.page import PageState
from physiclaw.text import read_text

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent.parent / "static"

# Upload guard rails (see handle_screenshot_upload). An iPhone screenshot
# PNG is 2–8 MiB; 10 MiB is generous headroom, small enough that a
# malicious poster can't balloon the process.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_IMAGE_MAGICS = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")  # PNG, JPEG


def render_phone_page_html(filename: str, port: int) -> str:
    """Read a static page and substitute the phone bridge URLs into it.

    ``port`` is the CONTROL port serving the page; the substituted URLs
    point at the LAN bridge listener (`bridge_port`). Shared by every
    page that embeds the bridge URL (the QR page and the setup wizard)
    so the placeholder convention lives in one place. Callers wrap the
    result in their own ``HTMLResponse`` (so each picks its own cache
    headers).
    """
    primary, fallback = bridge_base_urls(port)
    return (
        read_text(STATIC_DIR / filename)
        .replace("__PHONE_URL__", f"{primary}/bridge")
        .replace("__PHONE_URL_FALLBACK__", f"{fallback}/bridge")
    )


async def serve_bridge_page(request):
    """Serve the bridge page for the phone browser.

    ``Cache-Control: no-store`` so Safari always pulls the latest JS —
    otherwise a stale cached copy can silently miss newly-added phases
    and the phone renders nothing.
    """
    return HTMLResponse(
        read_text(STATIC_DIR / "bridge.html"),
        headers={"Cache-Control": "no-store"},
    )


async def handle_phone_state(request, phone: PageState):
    """GET /api/bridge/state — unified poll endpoint for the phone page."""
    phone.bridge.poll()  # keep connected status alive in both modes
    if request.client:
        # A live poll is the freshest "this is the phone" signal — lets
        # the upload IP pin follow a DHCP change.
        phone.bridge.note_phone_ip(request.client.host)
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

    Guarded three ways (defence for the one LAN route that injects data
    into the vision pipeline):
      - arming window — only accepted while a consumer is expecting an
        upload (`BridgeState.upload_armed`), so an idle server ignores it;
      - TOFU IP pin — must come from the pinned phone IP (loopback exempt);
      - size cap + magic bytes — streamed with a hard cap so an oversized
        body is cut off instead of buffered, and must look like PNG/JPEG.
    """
    ip = request.client.host if request.client else None
    if not bridge.upload_armed():
        log.warning(f"Bridge: screenshot upload rejected — not armed (from {ip})")
        return JSONResponse({"error": "no screenshot expected"}, status_code=403)
    if not bridge.upload_ip_allowed(ip):
        log.warning(f"Bridge: screenshot upload rejected — unpinned IP {ip}")
        return JSONResponse({"error": "unrecognized sender"}, status_code=403)

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "body too large"}, status_code=413)
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            log.warning(f"Bridge: screenshot upload rejected — >{MAX_UPLOAD_BYTES}B")
            return JSONResponse({"error": "body too large"}, status_code=413)
        chunks.append(chunk)
    data = b"".join(chunks)

    if not data:
        return JSONResponse({"error": "empty body"}, status_code=400)
    if not data.startswith(_IMAGE_MAGICS):
        log.warning("Bridge: screenshot upload rejected — not PNG/JPEG")
        return JSONResponse({"error": "not a PNG/JPEG image"}, status_code=415)
    bridge.receive_screenshot(data)
    return JSONResponse({"ok": True, "size": len(data)})


async def handle_recent_screenshots(request, bridge: BridgeState):
    """GET /api/bridge/recent-screenshots?n=10 — last N raw uploads as base64,
    oldest→newest."""
    import base64

    try:
        n = int(request.query_params.get("n", "10"))
    except ValueError:
        n = 10
    shots = bridge.recent_screenshots(n)
    return JSONResponse(
        {"screenshots": [base64.b64encode(s).decode("ascii") for s in shots]}
    )


async def handle_clipboard_fetch(request, bridge: BridgeState):
    """GET /api/bridge/clipboard — iOS Shortcut fetches the current bridge text.

    Returns the text as plain text (so the Shortcut can write it directly to
    the clipboard) and marks the bridge as copied so bridge_tap() unblocks.
    Returns 204 if no text is queued.
    """
    text = bridge.fetch_text()
    if text is None:
        return PlainTextResponse("", status_code=204)
    # Defence-in-depth: some upstream paths (mis-typed sequence args) could
    # slip a non-string through. str()-coerce rather than crash the handler.
    if not isinstance(text, str):
        log.warning("Bridge: non-string clipboard value %r — coercing", text)
        text = str(text)
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
    """Serve a QR code page for the phone bridge URL."""
    return HTMLResponse(render_phone_page_html("qr.html", request.url.port or 8048))


async def handle_calib_touch(request, cal: CalibrationState):
    """POST /api/bridge/touch — page reports a touch event.

    Body: {"clientX": float, "clientY": float} viewport CSS coords from
    bridge.html. The viewport shift is measured during pre-cal (which
    always runs before any touch-driven step), so we convert directly to
    screenshot 0-1 coords and attach them as x, y.
    """
    body = await request.json()
    body["x"], body["y"] = cal.viewport_to_screenshot_pct(
        body["clientX"], body["clientY"]
    )
    cal.report_touch(body)
    return JSONResponse({"ok": True})
