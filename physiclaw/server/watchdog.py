"""/api/phone/watch — stateful phone-screen watchdog.

The watchdog keeps the previous camera frame and runs a wake check against
each new frame. The HTTP route is stateless from the client's perspective:
every GET captures a fresh frame, diffs it against the stored baseline,
advances the baseline, and returns {"event": bool}.

Designed to be polled by `agent.runtime.Runtime`. See that module for
the client-side loop.
"""

import asyncio
import logging
import threading
from typing import Callable

import numpy as np
from starlette.responses import JSONResponse

from physiclaw.vision.wake import detect_wake

log = logging.getLogger(__name__)

WakeCheck = Callable[[np.ndarray, np.ndarray], bool]


class Watchdog:
    """Stateful phone-screen wake detector.

    Thread-safe: a single lock protects the baseline frame. `poll()` is the
    only mutator and it's atomic.
    """

    def __init__(self, check: WakeCheck = detect_wake):
        self._check = check
        self._prev: np.ndarray | None = None
        self._lock = threading.Lock()

    def poll(self, frame: np.ndarray) -> bool:
        """Feed a fresh frame. First call establishes the baseline and
        returns False. Subsequent calls compare against the previous frame,
        advance the baseline, and return True iff the check fired."""
        with self._lock:
            prev = self._prev
            self._prev = frame
        if prev is None:
            return False
        return self._check(prev, frame)

    def reset(self) -> None:
        """Drop the baseline so the next poll re-establishes it."""
        with self._lock:
            self._prev = None


def register(mcp, physiclaw, watchdog: Watchdog):
    """Wire the /api/phone/watch route onto the FastMCP server."""

    @mcp.custom_route("/api/phone/watch", methods=["GET"])
    async def _watch(request):  # noqa: ARG001
        def _do() -> bool:
            if physiclaw._cam is None:
                raise RuntimeError("camera not connected")
            physiclaw.acquire()
            try:
                frame = physiclaw._cam.peek()
            finally:
                physiclaw.release()
            if frame is None:
                raise RuntimeError("camera peek returned no frame")
            return watchdog.poll(frame)

        try:
            event = await asyncio.get_event_loop().run_in_executor(None, _do)
            return JSONResponse({"event": event})
        except Exception as e:
            log.exception("watchdog poll failed")
            return JSONResponse({"error": str(e)}, status_code=503)
