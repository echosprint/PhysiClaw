"""HTTP route handlers for hardware setup.

Used by the /setup skill to query status, connect the GRBL arm, enumerate
cameras, and connect a chosen camera. Each handler runs blocking work in
a thread executor so the Starlette event loop stays responsive. The
camera identification/preview algorithms live in
``core/orchestration/camera_pick.py`` — this module is HTTP marshalling
over the rig and those functions, registered by ``core/server/hardware.py``.
"""

import asyncio
import base64
import logging
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from physiclaw.core.bridge import PageState
from physiclaw.core.bridge.handler import json_or_none, render_phone_page_html
from physiclaw.core.hardware.device import DeviceNotFound
from physiclaw.core.orchestration.camera_pick import camera_preview, resolve_auto_index

if TYPE_CHECKING:
    from physiclaw.core.orchestration import HardwareRig

log = logging.getLogger(__name__)


# ─── Setup wizard page ──────────────────────────────────────


async def handle_setup_page(request: Request) -> HTMLResponse:
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


# ─── Status ─────────────────────────────────────────────────


async def handle_status(request: Request, rig: "HardwareRig") -> JSONResponse:
    """GET /api/status — current hardware + calibration status.

    Returns whether the arm and camera are connected, intermediate
    calibration progress (rotation, mappings, etc.), whether the full
    chain is calibrated and ready for tap operations, and whether the
    first-run screen layout has already been learned (so the setup page
    can skip the learning note on a recalibration).
    """
    return JSONResponse(rig.status())


# ─── Stylus arm ─────────────────────────────────────────────


async def handle_connect_arm(request: Request, rig: "HardwareRig") -> JSONResponse:
    """POST /api/connect-arm — auto-detect and connect the GRBL arm."""

    def _do() -> None:
        with rig.locked():
            rig.connect_arm()

    try:
        await asyncio.to_thread(_do)
        return JSONResponse({"status": "ok", "message": "Arm connected"})
    except DeviceNotFound as e:
        # Client state (plug it in / power it), not a server fault — the
        # same 409 convention the calibration preconditions use.
        return JSONResponse({"status": "error", "message": str(e)}, status_code=409)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ─── Camera ─────────────────────────────────────────────────


async def handle_connect_camera(
    request: Request, rig: "HardwareRig", phone: PageState
) -> JSONResponse:
    """POST /api/connect-camera — open a camera by index.

    Body: ``{"index": int}`` — connect that camera directly.
    Body: ``{"index": "auto"}`` (or body omitted) — auto-pick via the
    RGBM corner markers (see ``camera_pick.resolve_auto_index``).
    """
    body = (await json_or_none(request)) or {}
    index = body.get("index")

    def _do() -> None:
        nonlocal index
        if index is None or index == "auto":
            index = resolve_auto_index(rig, phone)
        with rig.locked():
            rig.connect_camera(int(index))
            rig.calibration.cam_index = int(index)

    try:
        await asyncio.to_thread(_do)
        cam = rig.cam
        if cam is None:  # _do just connected it, so the handle is set
            raise RuntimeError("camera missing after connect")
        return JSONResponse(
            {
                "status": "ok",
                "message": f"Camera {cam.index} connected",
                "index": cam.index,
            }
        )
    except DeviceNotFound as e:
        # Same 409 convention as connect-arm: a missing device is the
        # operator's to fix, not a server fault.
        return JSONResponse({"status": "error", "message": str(e)}, status_code=409)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def handle_disconnect_camera(
    request: Request, rig: "HardwareRig"
) -> JSONResponse:
    """POST /api/disconnect-camera — release the camera device handle.

    Used by `setup hardware` step 8 so the OS camera-preview app can
    claim the device while the user adjusts the camera angle. Idempotent
    — returns ``released=False`` if no camera was connected.
    """

    def _do() -> bool:
        with rig.locked():
            return rig.disconnect_camera()

    try:
        released = await asyncio.to_thread(_do)
        return JSONResponse({"status": "ok", "released": released})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def handle_camera_preview(request: Request) -> JSONResponse:
    """GET /api/camera-preview/{index} — capture one frame from a camera index."""
    watermark = request.query_params.get("watermark", "0") == "1"
    try:
        # Inside the try: the route has no :int converter, so a
        # non-numeric index must produce this module's JSON error shape,
        # not a bare 500.
        index = int(request.path_params["index"])
        jpeg = await asyncio.to_thread(camera_preview, index, watermark)
        return JSONResponse(
            {"status": "ok", "index": index, "image": base64.b64encode(jpeg).decode()}
        )
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=404)
