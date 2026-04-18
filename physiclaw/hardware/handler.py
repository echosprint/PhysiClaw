"""HTTP route handlers for hardware setup.

Used by the /setup skill to query status, connect the GRBL arm, enumerate
cameras, and connect a chosen camera. Each handler runs blocking work in
a thread executor so the Starlette event loop stays responsive.
"""

import asyncio
import base64
import logging

from starlette.responses import JSONResponse

from pathlib import Path

import cv2

from physiclaw.calibration.calibrate import (
    CAMERA_REF_FILE,
    check_phone_in_frame,
    frame_similarity,
)
from physiclaw.hardware.camera import Camera
from physiclaw.vision.render import watermark_index
from physiclaw.vision.util import encode_jpeg

SIMILARITY_MIN = 0.3  # floor for "this is the same camera we calibrated"

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


def _capture_raw(idx: int):
    """Open camera ``idx``, return one raw unrotated frame or None.

    Logs the reason on failure so a silent None doesn't mask a real issue.
    """
    cam = Camera(idx)
    try:
        return cam.raw_frame()
    except (OSError, RuntimeError) as e:
        log.warning(f"  cam {idx}: capture failed — {e}")
        return None
    finally:
        cam.close()


def _pick_by_similarity(ref_path: Path) -> int | None:
    """Iterate cameras, return the index whose frame best matches the
    saved reference (if any score clears SIMILARITY_MIN). Identifies the
    overhead camera by scene rather than by shape heuristic."""
    ref = cv2.imread(str(ref_path))
    if ref is None:
        return None
    best_idx, best_score = None, -1.0
    for idx in range(4):
        frame = _capture_raw(idx)
        if frame is None:
            continue
        score = frame_similarity(ref, frame)
        log.info(f"  cam {idx}: similarity to reference = {score:.3f}")
        if score > best_score:
            best_idx, best_score = idx, score
    if best_idx is not None and best_score >= SIMILARITY_MIN:
        log.info(
            f"Auto-picked camera {best_idx} by similarity (score {best_score:.3f})"
        )
        return best_idx
    return None


def _pick_by_shape() -> int | None:
    """Fallback when no reference exists: return the first camera whose
    frame shows a phone-shaped bright region (aspect ≈ 2, coverage ≥ 30%)."""
    for idx in range(4):
        frame = _capture_raw(idx)
        if frame is None:
            continue
        try:
            result = check_phone_in_frame(frame)
        except RuntimeError:
            # no bright region — not this camera
            continue
        if result["ok"]:
            log.info(
                f"Auto-picked camera {idx} by shape "
                f"(coverage {result['coverage']:.0%})"
            )
            return idx
    return None


def _auto_pick_camera_index() -> int | None:
    """Identify the overhead camera automatically.

    Prefers reference-similarity matching — ``data/calibration/cache/
    camera_ref.jpg`` is saved during calibrate_camera_frame, so after
    a successful setup we can always pick the same USB index on warm
    restart even if other cameras (room, webcam, etc.) are attached.

    Falls back to the phone-shape heuristic on first-ever setup when
    no reference exists yet.
    """
    ref_path = Path(CAMERA_REF_FILE)
    if ref_path.exists():
        picked = _pick_by_similarity(ref_path)
        if picked is not None:
            return picked
        log.warning(
            f"No camera matched {ref_path} (all below {SIMILARITY_MIN}); "
            f"falling back to shape heuristic"
        )
    return _pick_by_shape()


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
