"""Starlette route handlers for the annotation UI.

Two workflows:
  1. User draws boxes manually in the browser
  2. Agent proposes boxes → user reviews/edits/confirms → agent acts
"""

import asyncio
import base64
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from physiclaw.annotation.bbox import classify_bbox
from physiclaw.annotation.state import AnnotationState

if TYPE_CHECKING:
    from physiclaw.core import PhysiClaw

STATIC_DIR = Path(__file__).parent.parent / "static"


async def serve_annotate_page(request):
    """Serve the annotation web UI."""
    from starlette.responses import HTMLResponse
    return HTMLResponse((STATIC_DIR / "annotate.html").read_text())


async def freeze_snapshot(request, physiclaw: "PhysiClaw",
                          state: AnnotationState):
    """Capture current camera frame and freeze it for annotation."""
    from starlette.responses import JSONResponse

    def _capture():
        frame = physiclaw.cam._fresh_frame()
        if frame is None:
            return None
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    loop = asyncio.get_event_loop()
    frame = await loop.run_in_executor(None, _capture)
    if frame is None:
        return JSONResponse({"error": "capture failed"}, status_code=500)

    snapshot_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    state.freeze(frame, snapshot_id)
    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(jpeg.tobytes()).decode()

    # Phone screen corners in image pixels (for drawing boundary)
    phone_bounds = None
    cal = physiclaw._grid_cal
    if cal is not None:
        phone_bounds = [
            list(cal.pct_to_cam_pixel(0, 0)),  # top-left
            list(cal.pct_to_cam_pixel(1, 0)),  # top-right
            list(cal.pct_to_cam_pixel(1, 1)),  # bottom-right
            list(cal.pct_to_cam_pixel(0, 1)),  # bottom-left
        ]

    return JSONResponse({
        "image": b64,
        "width": frame.shape[1],
        "height": frame.shape[0],
        "phone_bounds": phone_bounds,
        "snapshot_id": snapshot_id,
    })


async def get_frozen_snapshot(request, physiclaw: "PhysiClaw",
                              state: AnnotationState):
    """Return the already-frozen snapshot without capturing a new one."""
    from starlette.responses import JSONResponse

    frame = state.get_frozen_frame()
    if frame is None:
        return JSONResponse({"error": "no frozen snapshot"}, status_code=404)

    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(jpeg.tobytes()).decode()

    phone_bounds = None
    cal = physiclaw._grid_cal
    if cal is not None:
        phone_bounds = [
            list(cal.pct_to_cam_pixel(0, 0)),
            list(cal.pct_to_cam_pixel(1, 0)),
            list(cal.pct_to_cam_pixel(1, 1)),
            list(cal.pct_to_cam_pixel(0, 1)),
        ]

    return JSONResponse({
        "image": b64,
        "width": frame.shape[1],
        "height": frame.shape[0],
        "phone_bounds": phone_bounds,
        "snapshot_id": state.snapshot_id,
    })


async def handle_annotations(request, state: AnnotationState):
    """GET: poll for agent proposals. DELETE: clear state."""
    from starlette.responses import JSONResponse

    if request.method == "DELETE":
        state.clear()
        return JSONResponse({"ok": True})
    else:  # GET — UI polls this for agent proposals
        agent_proposals = state.pop_agent_proposals()
        return JSONResponse({
            "agent_proposals": agent_proposals,
            "snapshot_id": state.snapshot_id,
        })


async def handle_confirm(request, state: AnnotationState,
                         physiclaw: "PhysiClaw"):
    """User confirms final boxes from the annotation UI.

    Receives boxes in image pixel coords, converts to 0-1 phone coords,
    and stores them for the agent to consume via wait_for_confirmation().
    """
    from starlette.responses import JSONResponse

    body = await request.json()
    raw_boxes = body.get("annotations", [])

    cal = physiclaw._grid_cal
    confirmed = []
    for box in raw_boxes:
        if cal is not None:
            l, t = cal.pixel_to_pct(int(box['left']), int(box['top']))
            r, b = cal.pixel_to_pct(int(box['right']), int(box['bottom']))
            full_bbox = [
                max(0.0, min(1.0, round(l, 3))),
                max(0.0, min(1.0, round(t, 3))),
                max(0.0, min(1.0, round(r, 3))),
                max(0.0, min(1.0, round(b, 3))),
            ]
        else:
            # No calibration — normalize by image dimensions
            frame = state.frozen_frame
            h, w = frame.shape[:2] if frame is not None else (1, 1)
            full_bbox = [
                round(box['left'] / w, 3),
                round(box['top'] / h, 3),
                round(box['right'] / w, 3),
                round(box['bottom'] / h, 3),
            ]

        box_type, coords = classify_bbox(full_bbox)
        confirmed.append({
            "type": box_type,
            "bbox": coords,
            "label": box.get("label", ""),
            "source": box.get("source", "user"),
            "id": box.get("id", ""),
        })

    state.confirm(confirmed)
    return JSONResponse({"ok": True, "count": len(confirmed)})
