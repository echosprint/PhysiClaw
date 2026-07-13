"""LAN bridge HTTP routes — register thin wrappers around bridge/handler.py.

The MCP tools that talk to the bridge live in server/tools.py — every
tool in the project is registered there. This module only wires up HTTP
routes, split across the two planes (see core/server/planes.py):

- ``register`` — control-plane routes (loopback): the QR page, mode
  switching, and the recent-uploads fetch. Callers are the desktop
  browser and local tools, never the phone.
- ``register_phone`` — the LAN-facing routes the phone actually touches:
  the bridge page, its poll/touch/dimension XHRs, and the two iOS
  Shortcut endpoints (screenshot upload, clipboard fetch).
"""

import logging

from physiclaw.core.bridge import BridgeState, CalibrationState, PageState
from physiclaw.core.bridge.handler import (
    handle_calib_touch,
    handle_clipboard_copied,
    handle_clipboard_fetch,
    handle_mode_switch,
    handle_phone_state,
    handle_recent_screenshots,
    handle_screen_dimension,
    handle_screenshot_upload,
    serve_bridge_page,
    serve_qr_page,
)

log = logging.getLogger(__name__)


def register(
    mcp, physiclaw, bridge: BridgeState, calib: CalibrationState, phone: PageState
):
    """Register the control-plane bridge routes (loopback callers only)."""

    @mcp.custom_route("/api/bridge/qr", methods=["GET"])
    async def _qr(request):
        return await serve_qr_page(request)

    @mcp.custom_route("/api/bridge/recent-screenshots", methods=["GET"])
    async def _bridge_recent_screenshots(request):
        return await handle_recent_screenshots(request, bridge)

    @mcp.custom_route("/api/bridge/switch", methods=["POST"])
    async def _bridge_switch(request):
        return await handle_mode_switch(request, phone)


def register_phone(
    app, physiclaw, bridge: BridgeState, calib: CalibrationState, phone: PageState
):
    """Register the phone-facing routes on the LAN bridge app.

    ``app`` exposes the same ``custom_route`` decorator shape as FastMCP
    (see ``apps.PhoneApp``) so both planes register identically.
    """

    @app.custom_route("/bridge", methods=["GET"])
    async def _phone_page(request):
        return await serve_bridge_page(request)

    @app.custom_route("/api/bridge/state", methods=["GET"])
    async def _phone_state(request):
        return await handle_phone_state(request, phone)

    @app.custom_route("/api/bridge/tapped", methods=["POST"])
    async def _bridge_tapped(request):
        return await handle_clipboard_copied(request, bridge)

    @app.custom_route("/api/bridge/screen-dimension", methods=["POST"])
    async def _bridge_screen_dimension(request):
        return await handle_screen_dimension(request, calib)

    @app.custom_route("/api/bridge/screenshot", methods=["POST"])
    async def _bridge_screenshot(request):
        return await handle_screenshot_upload(request, bridge)

    @app.custom_route("/api/bridge/clipboard", methods=["GET"])
    async def _bridge_clipboard(request):
        return await handle_clipboard_fetch(request, bridge)

    @app.custom_route("/api/bridge/touch", methods=["POST"])
    async def _calib_touch(request):
        return await handle_calib_touch(request, calib)
