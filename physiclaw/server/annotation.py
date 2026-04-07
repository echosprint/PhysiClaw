"""Annotation HTTP routes — register thin wrappers around annotation/handler.py.

The MCP tools for the propose/confirm bbox workflow live in server/tools.py —
every tool in the project is registered there. This module only wires up
the HTTP routes used by the browser annotation UI.
"""

import logging

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
    """Register annotation HTTP routes."""

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
