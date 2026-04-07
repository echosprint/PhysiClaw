"""Starlette route handlers for the LAN bridge.

These are async functions invoked by the MCP server to serve the phone
bridge page, accept screenshot uploads, handle clipboard confirmations,
and report calibration touch events.
"""

import logging
from pathlib import Path

from physiclaw.bridge.lan import get_lan_ip
from physiclaw.bridge.state import BridgeState
from physiclaw.bridge.phase import CalibrationState
from physiclaw.bridge.phone import PhoneState

log = logging.getLogger(__name__)


async def serve_bridge_page(request):
    """Serve the bridge page for the phone browser."""
    from starlette.responses import HTMLResponse
    html_path = Path(__file__).parent.parent / "static" / "bridge.html"
    return HTMLResponse(html_path.read_text())


async def handle_phone_state(request, phone: PhoneState):
    """GET /api/bridge/state — unified poll endpoint for the phone page."""
    from starlette.responses import JSONResponse
    phone.bridge.poll()  # keep connected status alive in both modes
    return JSONResponse(phone.get_state())


async def handle_clipboard_copied(request, bridge: BridgeState):
    """POST /api/bridge/tapped — phone confirms text was copied to clipboard."""
    from starlette.responses import JSONResponse
    bridge.mark_clipboard_copied()
    log.info(f"Bridge: clipboard copied — '{bridge.text}'")
    return JSONResponse({"ok": True})


async def handle_screen_dimension(request, cal: CalibrationState):
    """POST /api/bridge/screen-dimension — phone sends screen dimensions on load."""
    from starlette.responses import JSONResponse
    body = await request.json()
    with cal.lock:
        cal.screen_dimension = {
            "width": int(body.get('screen_width', 0)),
            "height": int(body.get('screen_height', 0)),
            "viewport_width": int(body.get('viewport_width', 0)),
            "viewport_height": int(body.get('viewport_height', 0)),
        }
    dim = cal.screen_dimension
    log.info(f"Bridge device: {dim['width']}×{dim['height']}pt, "
             f"viewport {dim['viewport_width']}×{dim['viewport_height']}pt")
    return JSONResponse({"ok": True})


async def handle_screenshot_upload(request, bridge: BridgeState):
    """POST /api/bridge/screenshot — iOS Shortcut uploads a screenshot.

    Accepts raw image body (PNG or JPEG). The iOS Shortcut action is:
    "Get Contents of URL" with method POST, body = screenshot image.
    """
    from starlette.responses import JSONResponse
    data = await request.body()
    if not data:
        return JSONResponse({"error": "empty body"}, status_code=400)
    bridge.receive_screenshot(data)
    return JSONResponse({"ok": True, "size": len(data)})


async def serve_qr_page(request):
    """Serve a page with a single QR code for the unified phone page."""
    from starlette.responses import HTMLResponse
    ip = get_lan_ip()
    port = request.url.port or 8048
    phone_url = f"http://{ip}:{port}/bridge"
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>PhysiClaw QR</title>
<script src="https://cdn.jsdelivr.net/npm/qrcode-generator@1.4.4/qrcode.min.js"></script>
<style>
body{{font-family:system-ui;text-align:center;padding:40px;background:#f9fafb}}
.qr{{display:inline-block;margin:40px;padding:30px;background:#fff;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.1)}}
.qr h3{{margin:0 0 10px;color:#374151}}
.qr p{{font-size:14px;color:#6b7280;word-break:break-all;max-width:300px}}
canvas{{display:block;margin:10px auto}}
</style></head><body>
<h2>PhysiClaw — Scan with Phone</h2>
<div class="qr"><h3>Open on Phone</h3><canvas id="qr1"></canvas><p>{phone_url}</p></div>
<script>
function drawQR(id, text){{
  var qr=qrcode(0,'M');qr.addData(text);qr.make();
  var canvas=document.getElementById(id);var ctx=canvas.getContext('2d');
  var size=240;var cells=qr.getModuleCount();var cell=size/cells;
  canvas.width=canvas.height=size;
  ctx.fillStyle='#fff';ctx.fillRect(0,0,size,size);
  ctx.fillStyle='#000';
  for(var r=0;r<cells;r++)for(var c=0;c<cells;c++)
    if(qr.isDark(r,c))ctx.fillRect(c*cell,r*cell,cell,cell);
}}
drawQR('qr1','{phone_url}');
</script></body></html>"""
    return HTMLResponse(html)


async def handle_calib_touch(request, cal: CalibrationState):
    """POST /api/bridge/touch — page reports a touch event.

    Body: {"clientX": float, "clientY": float} (viewport CSS coords).
    If screenshot_transform is set, converts to screenshot 0-1 as x, y.
    """
    from starlette.responses import JSONResponse
    body = await request.json()
    if 'clientX' in body:
        if cal.screenshot_transform:
            # Convert viewport CSS coords → screenshot 0-1
            sx, sy = cal.viewport_to_screenshot_pct(body['clientX'], body['clientY'])
        else:
            # Fallback: normalize to viewport 0-1 (less accurate but won't crash)
            dim = cal.screen_dimension
            vw = dim['viewport_width'] if dim and dim.get('viewport_width') else 1
            vh = dim['viewport_height'] if dim and dim.get('viewport_height') else 1
            sx = body['clientX'] / vw
            sy = body['clientY'] / vh
        body['x'] = sx
        body['y'] = sy
    cal.report_touch(body)
    return JSONResponse({"ok": True})


async def handle_calib_touches(request, cal: CalibrationState):
    """GET /api/calibrate/touches — server reads accumulated touch events."""
    from starlette.responses import JSONResponse
    touches = cal.flush_touches()
    return JSONResponse({"touches": touches})
