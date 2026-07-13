"""Hardware setup HTTP routes — register thin wrappers around hardware/handler.py."""

import logging
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from physiclaw.core.bridge import PageState
from physiclaw.core.hardware.handler import (
    handle_camera_preview,
    handle_connect_arm,
    handle_connect_camera,
    handle_disconnect_camera,
    handle_setup_page,
    handle_status,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from physiclaw.core.orchestration import PhysiClaw

log = logging.getLogger(__name__)


def register(mcp: "FastMCP", physiclaw: "PhysiClaw", phone: PageState) -> None:
    """Register hardware setup routes."""

    @mcp.custom_route("/setup-hardware", methods=["GET"])
    async def _setup_page(request: Request) -> HTMLResponse:
        return await handle_setup_page(request)

    @mcp.custom_route("/api/status", methods=["GET"])
    async def _status(request: Request) -> JSONResponse:
        return await handle_status(request, physiclaw)

    @mcp.custom_route("/api/connect-arm", methods=["POST"])
    async def _connect_arm(request: Request) -> JSONResponse:
        return await handle_connect_arm(request, physiclaw)

    @mcp.custom_route("/api/connect-camera", methods=["POST"])
    async def _connect_camera(request: Request) -> JSONResponse:
        return await handle_connect_camera(request, physiclaw, phone)

    @mcp.custom_route("/api/disconnect-camera", methods=["POST"])
    async def _disconnect_camera(request: Request) -> JSONResponse:
        return await handle_disconnect_camera(request, physiclaw)

    @mcp.custom_route("/api/camera-preview/{index}", methods=["GET"])
    async def _camera_preview(request: Request) -> JSONResponse:
        return await handle_camera_preview(request)
