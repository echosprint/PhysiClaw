"""
LAN Bridge — text, screenshot, and calibration page for the phone.

Three data flows:

1. Text → clipboard: Agent sends text → phone /message page displays it →
   tap copies to clipboard → POSTs confirmation.

2. Screenshot upload: iOS Shortcut takes screenshot → POSTs to server.

3. Calibration: Server controls which visual targets the /calibrate page
   shows. Page reports touch coordinates and flashes green on valid
   interaction (camera detects the green flash for backward compatibility).

Architecture plan channel 2: "LAN bridge for text/data transfer."
"""

import logging
import socket
import threading
import time

log = logging.getLogger(__name__)


def get_lan_ip() -> str:
    """Detect this machine's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class BridgeState:
    """Shared state between the phone's /message page and agent MCP tools.

    Thread-safe: accessed from async route handlers and blocking MCP tool
    threads concurrently.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.text: str | None = None
        self.version: int = 0
        self.device_info: dict | None = None
        self.last_seen: float = 0
        self._clipboard_ready = threading.Event()
        # Screenshot upload (from iOS Shortcut)
        self._screenshot_data: bytes | None = None
        self._screenshot_ready = threading.Event()

    @property
    def connected(self) -> bool:
        """True if phone polled within the last 5 seconds."""
        return time.time() - self.last_seen < 5.0

    def send_text(self, text: str):
        """Set text for the phone to display and copy on tap."""
        with self.lock:
            self.text = text
            self.version += 1
            self._clipboard_ready.clear()

    def mark_clipboard_ready(self):
        """Phone confirms the tap-to-copy succeeded."""
        self._clipboard_ready.set()

    def wait_clipboard(self, timeout: float = 30.0) -> bool:
        """Block until phone confirms clipboard copy, or timeout."""
        return self._clipboard_ready.wait(timeout=timeout)

    def touch(self):
        """Update last-seen timestamp (called on every phone poll)."""
        self.last_seen = time.time()

    # ─── Screenshot upload ────────────────────────────────────

    def receive_screenshot(self, data: bytes):
        """Store an uploaded screenshot and signal waiters."""
        with self.lock:
            self._screenshot_data = data
        self._screenshot_ready.set()
        log.info(f"Bridge: screenshot received ({len(data)} bytes)")

    def wait_screenshot(self, timeout: float = 10.0) -> bytes | None:
        """Block until a screenshot arrives, or timeout. Returns PNG/JPEG bytes."""
        if self._screenshot_ready.wait(timeout=timeout):
            self._screenshot_ready.clear()
            with self.lock:
                return self._screenshot_data
        return None


# ─── Calibration state ────────────────────────────────────────

# Grid dot positions (must match calibrate.html and grid_calibrate.py)
GRID_COLS_PCT = [0.25, 0.50, 0.75]
GRID_ROWS_PCT = [0.20, 0.40, 0.50, 0.60, 0.80]

# Valid calibration phases (server → page display commands)
CALIB_PHASES = {
    "idle",         # blank, waiting
    "center",       # orange circle at center (Steps 0, 1, 4)
    "markers",      # UP/RIGHT blue markers for camera rotation (Steps 2-3)
    "grid",         # 15 red dots at known positions (Step 5)
    "dot",          # single orange dot at custom position (Step 6)
}


class CalibrationState:
    """Server-controlled calibration page state.

    The server sets the phase (what the page displays). The page reports
    touch events back. The phase controls which visual targets appear and
    what interactions trigger a green flash.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.phase: str = "idle"
        self.phase_version: int = 0
        self.dot_position: tuple[float, float] | None = None  # for "dot" phase
        self.swipe_direction: str | None = None  # for "swipe" phase
        self.touches: list[dict] = []
        self._touch_event = threading.Event()

    def set_phase(self, phase: str, **kwargs):
        """Set the calibration display phase.

        Args:
            phase: one of CALIB_PHASES
            dot_x, dot_y: position for "dot" phase (0-1 decimals)
            direction: expected direction for "swipe" phase
        """
        if phase not in CALIB_PHASES:
            raise ValueError(f"Unknown phase: {phase}. Must be one of {CALIB_PHASES}")
        with self.lock:
            self.phase = phase
            self.phase_version += 1
            self.dot_position = None
            self.swipe_direction = None
            self.touches = []
            self._touch_event.clear()
            if phase == "dot":
                self.dot_position = (kwargs.get("dot_x", 0.5),
                                     kwargs.get("dot_y", 0.5))
            if phase == "swipe":
                self.swipe_direction = kwargs.get("direction", "top")

    def report_touch(self, touch: dict):
        """Page reports a touch event."""
        with self.lock:
            self.touches.append(touch)
        self._touch_event.set()
        log.debug(f"Calibration touch: ({touch.get('x')}, {touch.get('y')})")

    def wait_touch(self, timeout: float = 10.0) -> dict | None:
        """Block until a touch event arrives. Returns the touch or None.

        Caller must call get_touches() first to clear stale events.
        This method waits for the NEXT report_touch() call.
        """
        if self._touch_event.wait(timeout=timeout):
            self._touch_event.clear()
            with self.lock:
                if self.touches:
                    return self.touches[-1]
        return None

    def get_touches(self) -> list[dict]:
        """Get and clear all accumulated touch events."""
        with self.lock:
            touches = list(self.touches)
            self.touches = []
            self._touch_event.clear()
        return touches

    def get_display(self) -> dict:
        """Get current display command for the page to render."""
        with self.lock:
            d = {"phase": self.phase, "version": self.phase_version}
            if self.dot_position:
                d["dot_x"], d["dot_y"] = self.dot_position
            if self.swipe_direction:
                d["direction"] = self.swipe_direction
            # Always include grid positions so the page has them
            d["grid_cols"] = GRID_COLS_PCT
            d["grid_rows"] = GRID_ROWS_PCT
            return d


# ─── Phone state (unified page controller) ──────────────────

class PhoneState:
    """Coordinates mode switching between calibration and bridge on one page.

    The phone runs a single page that can display calibration UI or bridge UI.
    The server controls which mode is active.
    """

    def __init__(self, bridge: BridgeState, cal: CalibrationState):
        self.bridge = bridge
        self.cal = cal
        self.lock = threading.Lock()
        self.mode: str = "bridge"  # "calibrate" or "bridge"
        self.mode_version: int = 0

    def set_mode(self, mode: str):
        with self.lock:
            if self.mode != mode:
                self.mode = mode
                self.mode_version += 1
                log.info(f"Phone mode → {mode}")

    def get_state(self) -> dict:
        """Unified state for the phone page poll."""
        with self.lock:
            mode = self.mode
            mv = self.mode_version

        state = {"mode": mode, "mode_version": mv}

        if mode == "calibrate":
            state.update(self.cal.get_display())
        else:
            state["text"] = self.bridge.text
            state["text_version"] = self.bridge.version

        return state


# ─── Route handlers ──────────────────────────────────────────


async def serve_message_page(request):
    """Serve the bridge /message page for the phone browser."""
    from pathlib import Path
    from starlette.responses import HTMLResponse
    html_path = Path(__file__).parent / "static" / "bridge.html"
    return HTMLResponse(html_path.read_text())


async def handle_phone_state(request, phone: PhoneState):
    """GET /api/bridge/state — unified poll endpoint for the phone page."""
    from starlette.responses import JSONResponse
    phone.bridge.touch()  # keep connected status alive in both modes
    return JSONResponse(phone.get_state())


async def handle_bridge_text(request, bridge: BridgeState):
    """GET /api/bridge/text — phone polls for pending text to display."""
    from starlette.responses import JSONResponse
    bridge.touch()
    return JSONResponse({
        "text": bridge.text,
        "version": bridge.version,
    })


async def handle_bridge_tapped(request, bridge: BridgeState):
    """POST /api/bridge/tapped — phone confirms text was copied to clipboard."""
    from starlette.responses import JSONResponse
    bridge.mark_clipboard_ready()
    log.info(f"Bridge: clipboard ready — '{bridge.text}'")
    return JSONResponse({"ok": True})


async def handle_device_info(request, bridge: BridgeState):
    """POST /api/bridge/device-info — phone sends screen/device info on load."""
    from starlette.responses import JSONResponse
    body = await request.json()
    with bridge.lock:
        bridge.device_info = body
    log.info(f"Bridge device: {body.get('css_width')}×{body.get('css_height')}pt, "
             f"{body.get('pixel_ratio')}x")
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


# ─── Calibration route handlers ──────────────────────────────


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


async def serve_calibrate_page(request):
    """Serve the unified phone page (same as /message and /bridge)."""
    return await serve_message_page(request)


async def handle_calib_phase(request, cal: CalibrationState):
    """GET /api/calibrate/phase — page polls for current display command."""
    from starlette.responses import JSONResponse
    return JSONResponse(cal.get_display())


async def handle_calib_set_phase(request, cal: CalibrationState):
    """POST /api/calibrate/set-phase — server sets calibration display.

    Body: {"phase": "center"} or {"phase": "dot", "dot_x": 0.3, "dot_y": 0.7}
    """
    from starlette.responses import JSONResponse
    body = await request.json()
    phase = body.get("phase", "idle")
    try:
        cal.set_phase(phase, **{k: v for k, v in body.items() if k != "phase"})
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, "phase": phase})


async def handle_calib_touch(request, cal: CalibrationState):
    """POST /api/calibrate/touch — page reports a touch event.

    Body: {"x": css_x, "y": css_y, "type": "tap"|"long-press"|"swipe",
           "direction": "top"|...}
    """
    from starlette.responses import JSONResponse
    body = await request.json()
    cal.report_touch(body)
    return JSONResponse({"ok": True})


async def handle_calib_touches(request, cal: CalibrationState):
    """GET /api/calibrate/touches — server reads accumulated touch events."""
    from starlette.responses import JSONResponse
    touches = cal.get_touches()
    return JSONResponse({"touches": touches})
