"""HTTP route handlers for hardware setup.

Used by the /setup skill to query status, connect the GRBL arm, enumerate
cameras, and connect a chosen camera. Each handler runs blocking work in
a thread executor so the Starlette event loop stays responsive.
"""

import asyncio
import base64
import logging

from starlette.responses import JSONResponse

from physiclaw.calibration.calibrate import _check_phone_in_frame
from physiclaw.hardware.camera import Camera
from physiclaw.vision.render import watermark_index
from physiclaw.vision.util import encode_jpeg

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


def _auto_pick_camera_index() -> int | None:
    """Return the index of the first camera whose frame looks like a phone.

    Iterates 0..3, captures one frame from each, runs the phone-in-frame
    diagnostic (aspect ≈ 2:1, coverage ≥ 30%). Returns the first passing
    index, or None if no camera looked right. The caller falls back to
    asking the user to pick manually.
    """
    for idx in range(4):
        cam = Camera(idx)
        try:
            frame = cam.snapshot()
        except Exception:
            frame = None
        finally:
            cam.close()
        if frame is None:
            continue
        try:
            result = _check_phone_in_frame(frame)
        except Exception:
            continue
        if result["ok"]:
            log.info(f"Auto-picked camera {idx} (coverage {result['coverage']:.0%})")
            return idx
    return None


async def handle_connect_camera(request, physiclaw):
    """POST /api/connect-camera — open a camera by index.

    Body: ``{"index": int}`` — connect that camera.
    Body: ``{"index": "auto"}`` (or body omitted) — iterate 0..3 and pick
    the one whose frame matches a phone shape. Returns the chosen index.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    index = body.get("index")

    def _do():
        nonlocal index
        if index is None or index == "auto":
            picked = _auto_pick_camera_index()
            if picked is None:
                raise RuntimeError(
                    "auto-pick found no camera with a phone-shaped bright region; "
                    "pass an explicit index"
                )
            index = picked
        physiclaw.acquire()
        try:
            physiclaw.connect_camera(int(index))
            physiclaw.calibration.cam_index = int(index)
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
