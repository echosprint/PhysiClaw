"""HTTP marshalling for the studio — starlette routes over a
`StudioSession` (the `hardware_setup` handler precedent: thin JSON
shells, the brains live one module down).

Error convention, read by the browser's banner (`_refused` is its one
home): 400 — request the studio refuses (unknown tool, unreadable
body); 409 — a call is in flight (one rig); 502 — the MCP server is
unreachable; 500 — the tool ran and reported failure.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from physiclaw.common import gesture_vocab
from physiclaw.common.text import read_text
from physiclaw.studio.session import Session

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent

START_HINT = "Start it first: physiclaw mcp -H"
BUSY_MSG = "a call is already in flight — one rig"


def _error(code: int, message: str) -> JSONResponse:
    # Same wire shape as the core calibration handlers' `_err` — a
    # deliberate copy, not an import: the studio is an independent
    # process and must not couple to the core server package.
    return JSONResponse({"status": "error", "message": message}, status_code=code)


def _refused(e: Exception) -> JSONResponse:
    """The ONE exception → HTTP mapping; anything unrecognized
    re-raises (a bug, not a refusal)."""
    if isinstance(e, ValueError):
        return _error(400, str(e))
    if isinstance(e, ConnectionError):
        return _error(502, f"{e}. {START_HINT}")
    if isinstance(e, RuntimeError):
        return _error(500, str(e))
    raise e


async def _json_body(request: Request) -> dict:
    # Local on purpose (not `core.bridge.handler.json_or_none`): the
    # studio process must not import the core server's modules.
    try:
        body = json.loads(await request.body())
    except json.JSONDecodeError:
        raise ValueError("body must be JSON") from None
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON mapping")
    return body


async def handle_page(request: Request) -> HTMLResponse:
    """GET / — the single-file studio app. `no-store` so the browser
    always pulls the latest build after an upgrade (the setup-wizard
    convention)."""
    return HTMLResponse(
        read_text(STATIC_DIR / "studio.html"),
        headers={"Cache-Control": "no-store"},
    )


async def handle_state(request: Request, session: Session) -> JSONResponse:
    """GET /api/state — connection + surface, without dialing out.
    `swipe` carries the stroke ladder the page maps a drag onto, so JS
    never re-spells it."""
    return JSONResponse(
        {
            "status": "ok",
            **session.state(),
            "swipe": {
                "sizes": gesture_vocab.SWIPE_DISTANCES,
                "speeds": list(gesture_vocab.SWIPE_SPEEDS),
            },
        }
    )


async def handle_act(request: Request, session: Session) -> JSONResponse:
    """POST /api/act {tool, args} — one published tool, verbatim args.
    Arg validation is the server's job; its error text comes back in
    `message` untouched."""
    try:
        body = await _json_body(request)
        tool = body.get("tool")
        args = body.get("args") or {}
        if not isinstance(tool, str) or not isinstance(args, dict):
            return _error(400, "`tool` is a string, `args` a mapping")
        if session.busy:
            return _error(409, BUSY_MSG)
        view = await session.act(tool, args)
    except Exception as e:
        return _refused(e)
    return JSONResponse({"status": "ok", "tool": tool, **view})


def build_app(session: Session) -> Starlette:
    def bind(handler):
        async def route(request: Request):
            return await handler(request, session)

        return route

    @asynccontextmanager
    async def lifespan(app: Starlette):
        yield
        await session.close()

    return Starlette(
        routes=[
            Route("/", handle_page),
            Route("/api/state", bind(handle_state)),
            Route("/api/act", bind(handle_act), methods=["POST"]),
        ],
        lifespan=lifespan,
    )
