"""Hardware setup HTTP routes — register thin wrappers around hardware_setup.py."""

import logging
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from physiclaw.common.ready import STATUS_PATH
from physiclaw.core.bridge import PageState
from physiclaw.core.server.hardware_setup import (
    handle_camera_preview,
    handle_connect_arm,
    handle_connect_camera,
    handle_disconnect_camera,
    handle_setup_page,
    handle_status,
)

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

    from physiclaw.core.orchestration import HardwareRig

log = logging.getLogger(__name__)


def register(mcp: "MCPServer", rig: "HardwareRig", phone: PageState) -> None:
    """Register hardware setup routes."""

    @mcp.custom_route("/setup-hardware", methods=["GET"])
    async def _setup_page(request: Request) -> HTMLResponse:
        return await handle_setup_page(request)

    # The path constant is shared with every client-side reader
    # (`common.ready`) — route and probes can't drift apart.
    @mcp.custom_route(STATUS_PATH, methods=["GET"])
    async def _status(request: Request) -> JSONResponse:
        return await handle_status(request, rig)

    @mcp.custom_route("/api/connect-arm", methods=["POST"])
    async def _connect_arm(request: Request) -> JSONResponse:
        return await handle_connect_arm(request, rig)

    @mcp.custom_route("/api/connect-camera", methods=["POST"])
    async def _connect_camera(request: Request) -> JSONResponse:
        return await handle_connect_camera(request, rig, phone)

    @mcp.custom_route("/api/disconnect-camera", methods=["POST"])
    async def _disconnect_camera(request: Request) -> JSONResponse:
        return await handle_disconnect_camera(request, rig)

    @mcp.custom_route("/api/camera-preview/{index}", methods=["GET"])
    async def _camera_preview(request: Request) -> JSONResponse:
        return await handle_camera_preview(request)
