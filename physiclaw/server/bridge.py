"""LAN bridge tools and routes — text/clipboard transfer + screenshot upload."""

import logging

from mcp.server.fastmcp import Image

from physiclaw.bridge import BridgeState, CalibrationState, PhoneState, get_lan_ip
from physiclaw.bridge.routes import (
    serve_bridge_page,
    serve_qr_page,
    handle_clipboard_copied,
    handle_screen_dimension,
    handle_screenshot_upload,
    handle_phone_state,
    handle_calib_touch,
)

log = logging.getLogger(__name__)


def register(mcp, physiclaw, bridge: BridgeState, calib: CalibrationState, phone: PhoneState):
    """Register bridge tools and routes."""

    # ─── Routes ─────────────────────────────────────────────

    @mcp.custom_route("/bridge", methods=["GET"])
    async def _phone_page(request):
        return await serve_bridge_page(request)

    @mcp.custom_route("/api/bridge/state", methods=["GET"])
    async def _phone_state(request):
        return await handle_phone_state(request, phone)

    @mcp.custom_route("/api/bridge/qr", methods=["GET"])
    async def _qr(request):
        return await serve_qr_page(request)

    @mcp.custom_route("/api/bridge/tapped", methods=["POST"])
    async def _bridge_tapped(request):
        return await handle_clipboard_copied(request, bridge)

    @mcp.custom_route("/api/bridge/screen-dimension", methods=["POST"])
    async def _bridge_screen_dimension(request):
        return await handle_screen_dimension(request, calib)

    @mcp.custom_route("/api/bridge/screenshot", methods=["POST"])
    async def _bridge_screenshot(request):
        return await handle_screenshot_upload(request, bridge)

    @mcp.custom_route("/api/bridge/clipboard", methods=["GET"])
    async def _bridge_clipboard(request):
        from starlette.responses import PlainTextResponse
        text = bridge.text
        if text is None:
            return PlainTextResponse("", status_code=204)
        bridge.mark_clipboard_copied()
        log.info(f"Bridge: clipboard fetched — '{text}'")
        return PlainTextResponse(text)

    @mcp.custom_route("/api/bridge/switch", methods=["POST"])
    async def _bridge_switch(request):
        from starlette.responses import JSONResponse
        body = await request.json()
        mode = body.get("mode")
        if mode not in ("bridge", "calibrate"):
            return JSONResponse({"error": "mode must be 'bridge' or 'calibrate'"}, status_code=400)
        if mode == "calibrate":
            phase_name = body.get("phase")
            if not phase_name:
                return JSONResponse({"error": "phase required for calibrate mode"}, status_code=400)
            kwargs = {k: v for k, v in body.items() if k not in ("mode", "phase")}
            try:
                phone.set_mode(mode, phase_name, **kwargs)
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
            return JSONResponse({"ok": True, "mode": mode, "phase": phase_name})
        phone.set_mode(mode)
        return JSONResponse({"ok": True, "mode": mode})

    @mcp.custom_route("/api/bridge/touch", methods=["POST"])
    async def _calib_touch(request):
        return await handle_calib_touch(request, calib)

    # ─── Tools ──────────────────────────────────────────────

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
            lines.append("Status: not connected — ask the user to open the URL on their phone")

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
            return f"Clipboard ready — '{bridge.text}' copied to phone clipboard"
        return ("Tap sent but clipboard copy not confirmed. "
                "Is the phone page open in the foreground on the phone?")

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
        if not physiclaw._screenshot.ready:
            raise RuntimeError("AssistiveTouch not set up — run /setup first (step 7)")

        pct_to_grbl = physiclaw._cal.get('screen_to_grbl')
        if pct_to_grbl is None:
            raise RuntimeError("Calibration incomplete — run /setup first")

        physiclaw.acquire()
        try:
            data = physiclaw._screenshot.take_screenshot(
                physiclaw._arm, bridge, pct_to_grbl)
        finally:
            physiclaw.release()

        if data is None:
            raise RuntimeError("Timeout — no screenshot received. "
                               "Check iOS Shortcut is configured.")

        fmt = "png" if data[:4] == b'\x89PNG' else "jpeg"
        return Image(data=data, format=fmt)
