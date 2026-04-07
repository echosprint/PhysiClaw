"""HTTP route handlers for hardware setup.

Used by the /setup skill to query status, connect the GRBL arm, enumerate
cameras, and connect a chosen camera. Each handler runs blocking work in
a thread executor so the Starlette event loop stays responsive.
"""

import asyncio
import base64
import logging

from starlette.responses import JSONResponse

from physiclaw.hardware.camera import Camera
from physiclaw.vision.render import encode_jpeg, watermark_index

log = logging.getLogger(__name__)


# ─── Status ─────────────────────────────────────────────────


async def handle_status(request, physiclaw):
    """GET /api/status — current hardware + calibration status.

    Returns whether the arm and camera are connected, intermediate
    calibration progress (z_tap, rotation, mappings, etc.), and whether
    the full chain is calibrated and ready for tap operations.
    """
    return JSONResponse(physiclaw.status())


# ─── Stylus arm ─────────────────────────────────────────────


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ─── Camera ─────────────────────────────────────────────────


def camera_preview(index: int, watermark: bool = False) -> bytes:
    """Capture one frame from a camera, optionally watermark the index.

    Opens the camera, grabs a frame, closes the camera, returns JPEG bytes.
    Used by /api/camera-preview/{index} during /setup so the user can pick
    the right camera index by previewing each one without committing to a
    connection.

    Raises RuntimeError if the camera can't be opened or returns no frame.
    """
    cam = Camera(index)
    frame = cam.snapshot()
    cam.close()
    if frame is None:
        raise RuntimeError(f"Camera {index} returned no frame")

    if watermark:
        frame = watermark_index(frame, index)
    return encode_jpeg(frame, quality=80)


async def handle_connect_camera(request, physiclaw):
    """POST /api/connect-camera — open a camera by index.

    Body: {"index": int}. The user picks the index after previewing each
    one via /api/camera-preview/{index}.
    """
    body = await request.json()
    index = body.get("index")
    if index is None:
        return JSONResponse(
            {
                "status": "error",
                "message": "index is required (preview each camera first to choose)",
            },
            status_code=400,
        )

    def _do():
        physiclaw.acquire()
        try:
            physiclaw.connect_camera(int(index))
        finally:
            physiclaw.release()

    try:
        await asyncio.get_event_loop().run_in_executor(None, _do)
        return JSONResponse(
            {
                "status": "ok",
                "message": f"Camera {physiclaw.cam.index} connected",
                "index": physiclaw.cam.index,
            }
        )
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def handle_camera_preview(request):
    """GET /api/camera-preview/{index} — capture one frame from a camera index."""
    index = int(request.path_params["index"])
    watermark = request.query_params.get("watermark", "0") == "1"
    try:
        jpeg = await asyncio.get_event_loop().run_in_executor(
            None, camera_preview, index, watermark
        )
        return JSONResponse(
            {"status": "ok", "index": index, "image": base64.b64encode(jpeg).decode()}
        )
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=404)
