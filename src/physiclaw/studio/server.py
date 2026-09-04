"""HTTP marshalling for the studio — starlette routes over a `Session`
and one `Job` (the `hardware_setup` handler precedent: thin JSON
shells, the brains live one module down).

Three surfaces, one page: `/api/act` drives the phone by hand (one
tool call, verbatim args); `/api/step` steps a playbook node and
`/api/macro` runs a macro's gesture range, both through the stepping
driver (`debug/stepping.py`) — the same core, the same position file,
as `physiclaw playbooks step` and `macros run`, so the page and a
terminal can take turns. A step or a macro run is one background
`Job` holding the session's one-rig lock; the page polls `/api/step`
for its progress lines and the latest phone view.

Error convention, read by the browser's banner (`_refused` is its one
home): 400 — request the studio refuses (unknown tool, unreadable
body, a bad playbook ref or input); 409 — a call or a step is in
flight (one rig); 502 — the MCP server is unreachable; 500 — the tool
ran and reported failure.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from physiclaw.common import gesture_vocab
from physiclaw.common.ready import START_HINT
from physiclaw.common.text import read_text
from physiclaw.conductor.spec.pack import split_ref
from physiclaw.conductor.spec.specfile import SpecError
from physiclaw.debug import stepping
from physiclaw.macros.model import MacroError
from physiclaw.studio.session import Session, view_reply

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent

BUSY_MSG = "a call is already in flight — one rig"
STEPPING_MSG = "a step is already running — one rig"

# What a job runs — the snapshot's `kind`.
KIND_STEP = "step"
KIND_MACRO = "macro"

# The step request's string fields, passed to the driver by name.
_STEP_STRINGS = ("at", "to", "reply", "start_at", "stop_after")


def _error(code: int, message: str) -> JSONResponse:
    # Same wire shape as the core calibration handlers' `_err` — a
    # deliberate copy, not an import: the studio is an independent
    # process and must not couple to the core server package.
    return JSONResponse({"status": "error", "message": message}, status_code=code)


def _refused(e: Exception) -> JSONResponse:
    """The ONE exception → HTTP mapping; anything unrecognized
    re-raises (a bug, not a refusal)."""
    if isinstance(e, ValueError):  # SpecError and MacroError are ValueErrors
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


def _ref(body: dict) -> tuple[str, str]:
    ref = body.get("ref")
    if not isinstance(ref, str):
        raise ValueError("`ref` is <app>/<playbook>")
    return split_ref(ref)


def _str_field(body: dict, key: str) -> str:
    """An optional string field; absent or null reads as ""."""
    value = body.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"`{key}` is a string")
    return value


def _str_map(value, what: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise ValueError(f"`{what}` is a mapping of strings")
    return dict(value)


class Job:
    """One driver run in flight (or the last one) — a stepped playbook
    node or a macro's gesture range — and what it produced: the
    progress lines, the latest real phone view, and the driver's result
    or the error it raised. The page polls `snapshot`; a `since` older
    than `seq` gets the view again."""

    def __init__(self) -> None:
        self.kind: str | None = None
        self.ref: str | None = None
        self.task: asyncio.Task | None = None
        self.lines: list[str] = []
        # Every model round-trip of a stepped node (`rehearsal.exchanges`
        # records): what went to the provider and what came back.
        self.exchanges: list[dict] = []
        self.result: dict | None = None
        self.error: str | None = None
        self.view: dict | None = None
        self.seq = 0

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def start(self, session: Session, kind: str, ref: str, driver) -> None:
        """Run `driver(client, emit, emit_warn, observe) -> dict` on the
        session's lent client, in the background."""
        self.kind = kind
        self.ref = ref
        self.lines = []
        self.exchanges = []
        self.result = None
        self.error = None
        self.task = asyncio.create_task(self._run(session, driver))

    async def _run(self, session: Session, driver) -> None:
        async def lent(client):
            return await driver(
                client,
                self.lines.append,
                lambda line: self.lines.append(f"! {line}"),
                self.observe,
            )

        try:
            self.result = await session.drive(lent)
        except ConnectionError as e:
            self.error = f"{e}. {START_HINT}"
        except (SpecError, MacroError, RuntimeError, ValueError) as e:
            self.error = str(e)
        except Exception as e:  # the page must learn of a bug, not hang
            log.exception("studio job crashed")
            self.error = f"{self.kind} crashed: {e}"

    def observe(self, call, blocks) -> None:
        # Only a result that carries a view is worth parsing: the walk
        # reads the listing itself right after.
        if any(b.get("type") == "image" for b in blocks):
            self.view = {**view_reply(blocks), "tool": call.name}
            self.seq += 1

    def snapshot(
        self, since: int, lines_from: int = 0, exchanges_from: int = 0
    ) -> dict:
        """The job as the page polls it: the lines and exchanges past
        the counts it already holds, and the view when newer than
        `since`."""
        out = {
            "kind": self.kind,
            "ref": self.ref,
            "running": self.running,
            "lines": self.lines[lines_from:],
            "exchanges": self.exchanges[exchanges_from:],
            "result": self.result,
            "error": self.error,
            "seq": self.seq,
        }
        if since < self.seq and self.view is not None:
            out["view"] = self.view
        return out


def _launch(job: Job, session: Session, kind: str, ref: str, driver) -> JSONResponse:
    """The one-rig gate every driver launch passes: a running job or a
    manual call in flight answers 409; else the job starts and the
    page is told to poll."""
    if job.running:
        return _error(409, STEPPING_MSG)
    if session.busy:
        return _error(409, BUSY_MSG)
    job.start(session, kind, ref, driver)
    return JSONResponse({"status": "ok", "ref": ref})


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


async def handle_playbooks(request: Request) -> JSONResponse:
    """GET /api/playbooks — every pack's playbooks, each with its route
    and its stored position (the panel's whole model)."""
    return JSONResponse({"status": "ok", "packs": stepping.catalog()})


async def handle_step_state(request: Request, job: Job) -> JSONResponse:
    """GET /api/step?since=N — the job's progress; the view rides only
    when newer than `since`."""
    q = request.query_params
    try:
        since, lines, exchanges = (
            int(q.get("since", "0")),
            int(q.get("lines", "0")),
            int(q.get("exchanges", "0")),
        )
    except ValueError:
        return _error(400, "`since`, `lines`, `exchanges` are integers")
    return JSONResponse({"status": "ok", **job.snapshot(since, lines, exchanges)})


async def handle_step(request: Request, session: Session, job: Job) -> JSONResponse:
    """POST /api/step {ref, inputs, outputs, at, to, reply, start_at,
    stop_after, verbose} — start one step (`stepping.step`'s knobs by
    name). Answers at once; poll GET /api/step for progress."""
    try:
        body = await _json_body(request)
        app, name = _ref(body)
        opts: dict = {
            "values": _str_map(body.get("inputs"), "inputs"),
            "outputs": _str_map(body.get("outputs"), "outputs"),
            "verbose": bool(body.get("verbose")),
        }
        for key in _STEP_STRINGS:
            if value := _str_field(body, key):
                opts[key] = value

        async def driver(client, emit, emit_warn, observe):
            result = await stepping.step(
                app,
                name,
                client,
                emit=emit,
                emit_warn=emit_warn,
                observe=observe,
                on_exchange=job.exchanges.append,
                **opts,
            )
            return result.to_dict()

        return _launch(job, session, KIND_STEP, f"{app}/{name}", driver)
    except Exception as e:
        return _refused(e)


async def handle_macros(request: Request) -> JSONResponse:
    """GET /api/macros — every macro runnable by name, with its steps
    (the Macro tab's whole model)."""
    return JSONResponse({"status": "ok", "macros": stepping.macro_catalog()})


async def handle_macro(request: Request, session: Session, job: Job) -> JSONResponse:
    """POST /api/macro {name, inputs, start_at, stop_after} — run a
    macro's step range with no page checks (`stepping.run_macro`, the
    `macros run` core). Answers at once; poll GET /api/step."""
    try:
        body = await _json_body(request)
        name = _str_field(body, "name")
        if not name:
            raise ValueError("`name` is a macro name")
        values = _str_map(body.get("inputs"), "inputs")
        span = {k: _str_field(body, k) for k in ("start_at", "stop_after")}
        spec = stepping.find_macro(name)

        async def driver(client, emit, emit_warn, observe):
            return await stepping.run_macro(
                spec,
                values,
                client,
                emit=emit,
                observe=observe,
                caller="studio",
                **span,
            )

        return _launch(job, session, KIND_MACRO, name, driver)
    except Exception as e:
        return _refused(e)


async def handle_step_reset(request: Request, job: Job) -> JSONResponse:
    """POST /api/step/reset {ref} — forget the stored position."""
    try:
        app, name = _ref(await _json_body(request))
        if job.running:
            return _error(409, STEPPING_MSG)
        message = stepping.reset(app, name)
    except Exception as e:
        return _refused(e)
    return JSONResponse({"status": "ok", "message": message})


def build_app(session: Session) -> Starlette:
    job = Job()

    def bind(handler, *deps):
        async def route(request: Request):
            return await handler(request, *deps)

        return route

    @asynccontextmanager
    async def lifespan(app: Starlette):
        yield
        if job.task is not None and not job.task.done():
            job.task.cancel()
        await session.close()

    return Starlette(
        routes=[
            Route("/", handle_page),
            Route("/api/state", bind(handle_state, session)),
            Route("/api/act", bind(handle_act, session), methods=["POST"]),
            Route("/api/playbooks", handle_playbooks),
            Route("/api/step", bind(handle_step_state, job)),
            Route("/api/step", bind(handle_step, session, job), methods=["POST"]),
            Route("/api/step/reset", bind(handle_step_reset, job), methods=["POST"]),
            Route("/api/macros", handle_macros),
            Route("/api/macro", bind(handle_macro, session, job), methods=["POST"]),
        ],
        lifespan=lifespan,
    )
