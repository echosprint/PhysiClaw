"""/api/phone/watch — HTTP route for phone-screen wake detection.

Thin wiring layer: detection logic lives in `Perception.watch()`,
backed by `physiclaw.core.vision.watchdog.Watchdog`. These routes cut
across the orchestration slices (perception polls, rig flips ready,
the facade sends the phone home), so they take the facade.

Designed to be polled by `physiclaw.agent.runtime.Runtime`. See that module for
the client-side loop.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import JSONResponse

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from physiclaw.core.orchestration import PhysiClaw

log = logging.getLogger(__name__)


def register(mcp: "FastMCP", physiclaw: "PhysiClaw") -> None:
    """Wire the /api/phone/watch route onto the FastMCP server."""

    @mcp.custom_route("/api/phone/watch", methods=["GET"])
    async def _watch(request: Request) -> JSONResponse:  # noqa: ARG001
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, physiclaw.perception.watch
            )
            return JSONResponse(result)
        except RuntimeError as e:
            log.debug("watch skipped: %s", e)
            return JSONResponse({"wake": False, "reason": ""})
        except Exception as e:
            log.exception("watch poll failed")
            return JSONResponse({"error": str(e)}, status_code=503)

    @mcp.custom_route("/api/phone/home", methods=["POST"])
    async def _home(request: Request) -> JSONResponse:  # noqa: ARG001
        try:
            await asyncio.get_event_loop().run_in_executor(None, physiclaw.home_screen)
            return JSONResponse({"ok": True})
        except Exception as e:
            log.exception("home_screen failed")
            return JSONResponse({"error": str(e)}, status_code=503)

    @mcp.custom_route("/api/ready", methods=["POST"])
    async def _mark_ready(request: Request) -> JSONResponse:  # noqa: ARG001
        physiclaw.rig.mark_ready()
        # Fire-and-forget: setup just parked the phone on the home screen
        # (the dark scene that exposes AE failure), so verify/converge
        # exposure now — without stalling the wizard's finish screen.
        # tune_exposure is internally fail-open, so the future can't
        # carry an exception that matters.
        asyncio.get_event_loop().run_in_executor(
            None, physiclaw.perception.tune_exposure
        )
        return JSONResponse({"ok": True, "ready": physiclaw.rig.ready})
