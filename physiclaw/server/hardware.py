"""Hardware setup HTTP routes — register thin wrappers around hardware/handler.py."""

import logging

from physiclaw.hardware.handler import (
    handle_status,
    handle_connect_arm,
    handle_connect_camera,
    handle_camera_preview,
)

log = logging.getLogger(__name__)


def register(mcp, physiclaw):
    """Register hardware setup routes."""

    @mcp.custom_route("/api/status", methods=["GET"])
    async def _status(request):
        return await handle_status(request, physiclaw)

    @mcp.custom_route("/api/connect-arm", methods=["POST"])
    async def _connect_arm(request):
        return await handle_connect_arm(request, physiclaw)

    @mcp.custom_route("/api/connect-camera", methods=["POST"])
    async def _connect_camera(request):
        return await handle_connect_camera(request, physiclaw)

    @mcp.custom_route("/api/camera-preview/{index}", methods=["GET"])
    async def _camera_preview(request):
        return await handle_camera_preview(request)
