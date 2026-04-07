"""Annotation MCP tools and HTTP routes — propose/confirm bbox workflow."""

import logging
from datetime import datetime

import cv2
from mcp.server.fastmcp import Image

from physiclaw.annotation import (
    AnnotationState,
    freeze_snapshot,
    get_frozen_snapshot,
    handle_annotations,
    handle_confirm,
    serve_annotate_page,
)

log = logging.getLogger(__name__)


def register(mcp, physiclaw, ann: AnnotationState):
    """Register annotation tools and routes."""

    @mcp.custom_route("/annotate", methods=["GET"])
    async def _annotate(request):
        return await serve_annotate_page(request)

    @mcp.custom_route("/api/snapshot", methods=["GET", "POST"])
    async def _snapshot(request):
        if request.method == "GET":
            return await get_frozen_snapshot(request, physiclaw, ann)
        physiclaw.acquire()
        try:
            return await freeze_snapshot(request, physiclaw, ann)
        finally:
            physiclaw.release()

    @mcp.custom_route("/api/annotations", methods=["GET", "DELETE"])
    async def _annotations(request):
        return await handle_annotations(request, ann)

    @mcp.custom_route("/api/confirm", methods=["POST"])
    async def _confirm(request):
        return await handle_confirm(request, ann, physiclaw)

    @mcp.tool()
    def get_user_annotations() -> list:
        """Get confirmed annotations from the annotation UI.

        Returns the confirmed boxes with coordinates and labels, plus the
        frozen screenshot. The user must click Confirm in the annotation UI
        before this returns data.

        Use wait_for_confirmation() instead if you want to block until
        the user confirms. This tool returns immediately — it returns
        whatever was last confirmed, or "no annotations" if nothing was confirmed.
        """
        with ann.lock:
            confirmed = list(ann.confirmed_annotations)
        frozen_frame = ann.get_frozen_frame()
        if not confirmed:
            return ["No confirmed annotations. "
                    f"Ask the user to draw boxes at http://{mcp.settings.host}:{mcp.settings.port}/annotate and click Confirm."]

        lines = [f"# Confirmed Annotations ({len(confirmed)} items)\n"]
        for i, box in enumerate(confirmed):
            b = box['bbox']
            box_type = box.get('type', 'box')
            label = box.get('label', '')
            source = box.get('source', 'user')
            src = f" [{source}]" if source != 'user' else ""
            desc = f" — {label}" if label else ""
            coords = ", ".join(str(v) for v in b)
            type_tag = f" ({box_type})" if box_type != 'box' else ""
            lines.append(f"- {i+1}{type_tag}{src}: [{coords}]{desc}")
        text = "\n".join(lines)

        if frozen_frame is not None:
            return [text, Image(data=physiclaw.frame_to_jpeg(frozen_frame),
                                format="jpeg")]
        return [text]

    @mcp.tool()
    def propose_bboxes(proposals: list[dict]) -> str:
        """Propose bounding boxes for the user to review in the annotation UI.

        Sends your coordinate guesses to the annotation web UI at /annotate.
        The user can move, resize, delete, relabel, or add new boxes.
        After the user confirms, call wait_for_confirmation() to get the result.

        Parks the arm and takes a fresh screenshot automatically.

        Args:
            proposals: list of {"bbox": [left, top, right, bottom], "label": "element name"}
                       Coordinates are 0-1 decimals (phone screen).
        """
        import time
        physiclaw.require_hardware()
        physiclaw.acquire()
        try:
            physiclaw.park()
            time.sleep(1.5)

            frame = physiclaw.cam._fresh_frame()
            if frame is None:
                return "Camera capture failed"
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            snapshot_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            ann.freeze(frame, snapshot_id)
            ann.push_agent_proposals(proposals)

            url = f"http://{mcp.settings.host}:{mcp.settings.port}/annotate"
            return (f"{len(proposals)} proposals sent to annotation UI. "
                    f"Ask the user to review and confirm at {url}")
        finally:
            physiclaw.release()

    @mcp.tool()
    def wait_for_confirmation(timeout: int = 120) -> list:
        """Wait for the user to confirm bounding boxes in the annotation UI.

        Blocks until the user clicks Confirm at /annotate, or until timeout.
        Returns the confirmed boxes with user-corrected coordinates and labels.

        Call this after propose_bboxes() or after asking the user to draw boxes.

        Args:
            timeout: seconds to wait before giving up (default 120)
        """
        result = ann.wait_confirmed(timeout=float(timeout))
        if result is None:
            return ["Timeout — the user hasn't confirmed yet. "
                    f"Ask them if they need help at http://{mcp.settings.host}:{mcp.settings.port}/annotate"]

        frozen_frame = ann.get_frozen_frame()
        ann.clear_confirmation()

        lines = [f"# Confirmed Annotations ({len(result)} items)\n"]
        for i, box in enumerate(result):
            b = box['bbox']
            box_type = box.get('type', 'box')
            label = box.get('label', '')
            source = box.get('source', 'user')
            src = f" [{source}]" if source != 'user' else ""
            desc = f" — {label}" if label else ""
            coords = ", ".join(str(v) for v in b)
            type_tag = f" ({box_type})" if box_type != 'box' else ""
            lines.append(f"- {i+1}{type_tag}{src}: [{coords}]{desc}")
        text = "\n".join(lines)

        if frozen_frame is not None:
            return [text, Image(data=physiclaw.frame_to_jpeg(frozen_frame),
                                format="jpeg")]
        return [text]
