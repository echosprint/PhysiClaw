"""LAN bridge HTTP routes — register thin wrappers around bridge/routes.py.

The MCP tools that talk to the bridge live in server/tools.py — every
tool in the project is registered there. This module only wires up the
HTTP routes used by the phone bridge page and the iOS Shortcut.
"""

import logging

from physiclaw.bridge import BridgeState, CalibrationState, PhoneState
from physiclaw.bridge.routes import (
    serve_bridge_page,
    serve_qr_page,
    handle_clipboard_copied,
    handle_clipboard_fetch,
    handle_mode_switch,
    handle_screen_dimension,
    handle_screenshot_upload,
    handle_phone_state,
    handle_calib_touch,
)

log = logging.getLogger(__name__)


def register(mcp, physiclaw,
             bridge: BridgeState,
             calib: CalibrationState,
             phone: PhoneState):
    """Register bridge HTTP routes."""

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
        return await handle_clipboard_fetch(request, bridge)

    @mcp.custom_route("/api/bridge/switch", methods=["POST"])
    async def _bridge_switch(request):
        return await handle_mode_switch(request, phone)

    @mcp.custom_route("/api/bridge/touch", methods=["POST"])
    async def _calib_touch(request):
        return await handle_calib_touch(request, calib)
