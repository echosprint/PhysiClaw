"""
Annotation system — browser-based UI element labeling.

Provides web route handlers and shared state for the annotation UI.
Coordinate conversion happens in handle_confirm() (pixel → 0-1 phone coords).
server.py wires routes and MCP tools.

Supports two workflows:
  1. User draws boxes manually in the browser
  2. Agent proposes boxes → user reviews/edits/confirms → agent acts
"""

import asyncio
import base64
import threading
import uuid
from pathlib import Path

import cv2

from physiclaw.core import PhysiClaw

STATIC_DIR = Path(__file__).parent / "static"

# Agent-proposed boxes use this color (deep orange — not in user palette)
AGENT_COLOR = "#ff6d00"

# Aspect ratio threshold for classifying boxes as column/row
LINE_ASPECT_RATIO = 10


def classify_bbox(bbox: list[float]) -> tuple[str, list[float]]:
    """Classify a bbox by aspect ratio and strip the irrelevant axis.

    Args:
        bbox: [left, top, right, bottom] as 0-1 decimals

    Returns:
        (type, coords) where:
        - "box"    → [left, top, right, bottom]
        - "column" → [left, right]   (X interval, tall thin)
        - "row"    → [top, bottom]   (Y interval, wide thin)
    """
    w = abs(bbox[2] - bbox[0])
    h = abs(bbox[3] - bbox[1])
    if h > 0 and w > 0:
        if h / w > LINE_ASPECT_RATIO:
            return "column", [bbox[0], bbox[2]]
        if w / h > LINE_ASPECT_RATIO:
            return "row", [bbox[1], bbox[3]]
    return "box", bbox


class AnnotationState:
    """Shared state for the web annotation UI and agent propose-confirm loop."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frozen_frame = None          # BGR numpy array
        self.snapshot_id: str = ""        # timestamp of frozen snapshot

        # Agent → UI staging area
        self.agent_proposals: list[dict] = []

        # Confirmation flow
        self._confirmed = threading.Event()
        self.confirmed_annotations: list[dict] = []  # final boxes (typed, 0-1 coords)

    def freeze(self, frame, snapshot_id: str = ""):
        with self.lock:
            self.frozen_frame = frame.copy()
            self.snapshot_id = snapshot_id
            self.agent_proposals = []
            self._confirmed.clear()
            self.confirmed_annotations = []

    def get_frozen_frame(self):
        with self.lock:
            return self.frozen_frame

    def clear(self):
        with self.lock:
            self.frozen_frame = None
            self.snapshot_id = ""
            self.agent_proposals = []
            self._confirmed.clear()
            self.confirmed_annotations = []

    # ─── Agent proposals ──────────────────────────────────────

    def push_agent_proposals(self, proposals: list[dict]):
        """Stage agent-proposed boxes for the UI to pick up.

        Each proposal: {"bbox": [l,t,r,b], "label": "..."}
        Coordinates are in 0-1 phone screen decimals.
        Replaces any previous proposals (agent sends fresh set each time).
        """
        enriched = []
        for p in proposals:
            enriched.append({
                "id": str(uuid.uuid4())[:8],
                "bbox": p["bbox"],
                "label": p.get("label", ""),
                "source": "agent",
                "color": AGENT_COLOR,
            })
        with self.lock:
            self.agent_proposals = enriched
            self._confirmed.clear()
            self.confirmed_annotations = []

    def pop_agent_proposals(self) -> list[dict]:
        """Return and clear pending agent proposals. Called by UI polling."""
        with self.lock:
            proposals = self.agent_proposals
            self.agent_proposals = []
            return proposals

    # ─── Confirmation flow ────────────────────────────────────

    def confirm(self, annotations: list[dict]):
        """Called when user clicks Confirm in the UI.

        annotations: final boxes with 0-1 phone coords, labels, sources.
        """
        with self.lock:
            self.confirmed_annotations = annotations
            self._confirmed.set()

    def wait_confirmed(self, timeout: float = 120.0) -> list[dict] | None:
        """Block until user confirms or timeout. Returns confirmed boxes or None."""
        if self._confirmed.wait(timeout=timeout):
            with self.lock:
                return list(self.confirmed_annotations)
        return None

    def clear_confirmation(self):
        """Reset confirmation state for next cycle."""
        with self.lock:
            self._confirmed.clear()
            self.confirmed_annotations = []


# ─── Route handlers ────────────────────────────────────────────


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

    from datetime import datetime
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


async def get_frozen_snapshot(request, physiclaw: PhysiClaw,
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
                         physiclaw: PhysiClaw):
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
