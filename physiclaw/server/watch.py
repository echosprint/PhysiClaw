"""/api/phone/watch — HTTP route for phone-screen wake detection.

Thin wiring layer: detection logic lives in `PhysiClaw.watch()`,
backed by `physiclaw.vision.watchdog.Watchdog`.

Designed to be polled by `agent.runtime.Runtime`. See that module for
the client-side loop.
"""

import asyncio
import logging

from starlette.responses import JSONResponse

log = logging.getLogger(__name__)


def register(mcp, physiclaw):
    """Wire the /api/phone/watch route onto the FastMCP server."""

    @mcp.custom_route("/api/phone/watch", methods=["GET"])
    async def _watch(request):  # noqa: ARG001
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, physiclaw.watch
            )
            return JSONResponse(result)
        except RuntimeError as e:
            log.debug("watch skipped: %s", e)
            return JSONResponse({"wake": False, "reason": ""})
        except Exception as e:
            log.exception("watch poll failed")
            return JSONResponse({"error": str(e)}, status_code=503)
