"""HTTP route handlers for hardware setup.

Handlers for connecting and inspecting the physical hardware (arm + camera)
during the /setup skill flow. Each handler runs blocking work in a thread
executor so the Starlette event loop stays responsive.
"""

import asyncio
import base64
import logging

from starlette.responses import JSONResponse

log = logging.getLogger(__name__)


async def handle_status(request, physiclaw):
    """GET /api/status — current hardware + calibration status."""
    return JSONResponse(physiclaw.status())


async def handle_connect_arm(request, physiclaw):
    """POST /api/connect-arm — auto-detect and connect the GRBL arm."""

    def _do():
        physiclaw.acquire()
        try:
            physiclaw.connect_arm()
        finally:
            physiclaw.release()

    try:
        await asyncio.get_event_loop().run_in_executor(None, _do)
        return JSONResponse({"status": "ok", "message": "Arm connected"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)},
                            status_code=500)


async def handle_connect_camera(request, physiclaw):
    """POST /api/connect-camera — open a camera by index, or auto-detect.

    Body: {"index": int} or empty for auto-detect.
    """
    body = (await request.json()
            if request.headers.get("content-type", "").startswith("application/json")
            else {})
    index = body.get("index")  # None = auto-detect

    def _do():
        physiclaw.acquire()
        try:
            physiclaw.connect_camera(index)
        finally:
            physiclaw.release()

    try:
        await asyncio.get_event_loop().run_in_executor(None, _do)
        return JSONResponse({"status": "ok",
                             "message": f"Camera {physiclaw._cam.index} connected",
                             "index": physiclaw._cam.index})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)},
                            status_code=500)


async def handle_camera_preview(request, physiclaw_cls):
    """GET /api/camera-preview/{index} — capture one frame from a camera index.

    Used during /setup to let the user pick the right camera by previewing
    each available index without committing to a connection.
    """
    index = int(request.path_params["index"])
    watermark = request.query_params.get("watermark", "0") == "1"
    try:
        jpeg = await asyncio.get_event_loop().run_in_executor(
            None, physiclaw_cls.camera_preview, index, watermark)
        return JSONResponse({"status": "ok", "index": index,
                             "image": base64.b64encode(jpeg).decode()})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)},
                            status_code=404)
