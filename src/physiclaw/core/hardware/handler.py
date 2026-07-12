"""HTTP route handlers for hardware setup.

Used by the /setup skill to query status, connect the GRBL arm, enumerate
cameras, and connect a chosen camera. Each handler runs blocking work in
a thread executor so the Starlette event loop stays responsive.
"""

import asyncio
import base64
import logging
import time

from starlette.responses import HTMLResponse, JSONResponse

from physiclaw.core.bridge.handler import render_phone_page_html
from physiclaw.core.hardware.camera import Camera
from physiclaw.core.vision.render import watermark_index
from physiclaw.core.vision.grid_detect import detect_bridge_corners
from physiclaw.core.vision.util import encode_jpeg

from physiclaw.common.config import CONFIG

log = logging.getLogger(__name__)


# ─── Setup wizard page ──────────────────────────────────────


async def handle_setup_page(request):
    """GET /setup-hardware — serve the browser-based hardware-setup wizard.

    A single-file app that drives the same ``/api/*`` endpoints as
    ``physiclaw setup hardware``. Shares ``render_phone_page_html`` with the
    QR page so the bridge-URL substitution lives in one place; the inline QR
    renders from those URLs. ``Cache-Control: no-store`` so the browser
    always pulls the latest build after an upgrade.
    """
    return HTMLResponse(
        render_phone_page_html("setup-hardware.html", request.url.port or 8048),
        headers={"Cache-Control": "no-store"},
    )


# Timeouts for the auto-pick wait-for-bridge gate (mirrors the
# warm-start pattern; see core/server/warm_start.py). Values come from
# `CONFIG.auto_pick.bridge_*` so operators can tune them via
# `physiclaw config edit`.
AUTO_PICK_BRIDGE_TIMEOUT = CONFIG.auto_pick.bridge_wait_timeout_seconds
AUTO_PICK_BRIDGE_SETTLE = CONFIG.auto_pick.bridge_settle_seconds


# ─── Status ─────────────────────────────────────────────────


async def handle_status(request, physiclaw):
    """GET /api/status — current hardware + calibration status.

    Returns whether the arm and camera are connected, intermediate
    calibration progress (rotation, mappings, etc.), whether the full
    chain is calibrated and ready for tap operations, and whether the
    first-run screen layout has already been learned (so the setup page
    can skip the learning note on a recalibration).
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


def _auto_pick_camera_index() -> int | None:
    """Identify the overhead camera by the RGBM corner markers on /bridge.

    Caller must first put the phone page into the ``corners`` phase so
    bridge.html draws the four colored squares. We then iterate USB
    indices 0..3 and pick the camera whose frame contains all four
    markers arranged clockwise — only the camera actually pointing at
    the phone can possibly see them, so the match is unambiguous.
    """
    # macOS Continuity Camera + iPhone Webcam can occupy low indices,
    # pushing the actual USB camera to 4+. Probing 0..7 covers the
    # typical configurations without measurably slowing the loop —
    # missing-index opens fail fast on macOS.
    for idx in range(8):
        frame = _capture_raw(idx)
        if frame is None:
            continue
        corners = detect_bridge_corners(frame)
        if corners is None:
            log.info(f"  cam {idx}: corners not detected")
            continue
        log.info(f"Auto-picked camera {idx} — all four RGBM corners detected")
        return idx
    return None


async def handle_connect_camera(request, physiclaw, phone):
    """POST /api/connect-camera — open a camera by index.

    Body: ``{"index": int}`` — connect that camera directly.
    Body: ``{"index": "auto"}`` (or body omitted) — set the phone page
    to the ``corners`` phase and iterate 0..3 to find the camera that
    sees all four RGBM corner markers. The phone is restored to bridge
    mode before returning.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    index = body.get("index")

    def _do():
        nonlocal index
        if index is None or index == "auto":
            # Wait for the phone /bridge tab to be actively polling
            # before flipping to the corners phase. If the screen is
            # asleep or the tab is backgrounded, set_mode would update
            # server state but the canvas would never paint the RGBM
            # markers — and auto-pick would always fail. (warm-start
            # uses the same gate.)
            if not physiclaw._bridge.wait_for_connection(
                AUTO_PICK_BRIDGE_TIMEOUT, AUTO_PICK_BRIDGE_SETTLE
            ):
                raise RuntimeError(
                    f"auto-pick: /bridge page not polling within "
                    f"{AUTO_PICK_BRIDGE_TIMEOUT}s — open or refresh /bridge "
                    f"on the phone, then retry."
                )
            phone.set_mode("calibrate", phase="corners")
            time.sleep(0.5)  # give bridge.html time to render the corners
            try:
                picked = _auto_pick_camera_index()
            finally:
                phone.set_mode("bridge")
            if picked is None:
                raise RuntimeError(
                    "auto-pick found no camera with all four RGBM corners; "
                    "is /bridge open on the phone? Pass an explicit index to fall back."
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


async def handle_disconnect_camera(request, physiclaw):
    """POST /api/disconnect-camera — release the camera device handle.

    Used by `setup hardware` step 8 so the OS camera-preview app can
    claim the device while the user adjusts the camera angle. Idempotent
    — returns ``released=False`` if no camera was connected.
    """

    def _do() -> bool:
        physiclaw.acquire()
        try:
            return physiclaw.disconnect_camera()
        finally:
            physiclaw.release()

    try:
        released = await asyncio.get_event_loop().run_in_executor(None, _do)
        return JSONResponse({"status": "ok", "released": released})
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
