"""
Annotation system — browser-based UI element labeling.

Provides web route handlers and shared state for the annotation UI.
Coordinate conversion logic lives in PhysiClaw.process_annotations().
server.py wires routes and the MCP tool.
"""

import asyncio
import base64
import threading
from pathlib import Path

import cv2

from physiclaw.core import PhysiClaw

STATIC_DIR = Path(__file__).parent / "static"


class AnnotationState:
    """Shared state for the web annotation UI."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frozen_frame = None  # BGR numpy array
        self.annotations: list[dict] = []

    def freeze(self, frame):
        with self.lock:
            self.frozen_frame = frame.copy()
            self.annotations = []

    def set_annotations(self, annotations: list[dict]):
        with self.lock:
            self.annotations = annotations

    def get_all(self):
        with self.lock:
            return self.frozen_frame, list(self.annotations)

    def clear(self):
        with self.lock:
            self.frozen_frame = None
            self.annotations = []


# ─── Route handlers ────────────────────────────────────────────


async def mjpeg_stream(request, physiclaw: PhysiClaw):
    """MJPEG video stream from the camera."""
    from starlette.responses import StreamingResponse

    async def generate():
        loop = asyncio.get_event_loop()
        gen = physiclaw.cam.mjpeg_generator()
        try:
            while True:
                chunk = await loop.run_in_executor(None, next, gen)
                yield chunk
        except StopIteration:
            pass

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


async def serve_annotate_page(request):
    """Serve the annotation web UI."""
    from starlette.responses import HTMLResponse
    return HTMLResponse((STATIC_DIR / "annotate.html").read_text())


async def freeze_snapshot(request, physiclaw: PhysiClaw,
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

    state.freeze(frame)
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
    })


async def handle_annotations(request, state: AnnotationState):
    """CRUD for annotation boxes."""
    from starlette.responses import JSONResponse

    if request.method == "POST":
        body = await request.json()
        state.set_annotations(body.get("annotations", []))
        return JSONResponse({"ok": True})
    elif request.method == "DELETE":
        state.clear()
        return JSONResponse({"ok": True})
    else:  # GET
        _, annotations = state.get_all()
        return JSONResponse({"annotations": annotations,
                             "count": len(annotations)})
