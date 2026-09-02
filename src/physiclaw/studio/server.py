"""HTTP marshalling for the studio — starlette routes over a
`StudioSession` (the `hardware_setup` handler precedent: thin JSON
shells, the brains live one module down).

Error convention, read by the browser's banner (`_refused` is its one
home): 400 — request the studio refuses (unknown tool, unreadable
body, any brains-layer refusal); 409 — a call is in flight (one rig);
502 — the MCP server is unreachable; 500 — the tool ran and reported
failure, or a model call has no model configured.
"""

import base64
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.routing import Route

from physiclaw.common.listing import Screen
from physiclaw.common.text import read_text
from physiclaw.conductor.capture import MIN_FREQ
from physiclaw.macros.model import MAX_STEPS
from physiclaw.studio import curate, record, route
from physiclaw.studio import draft as draft_store
from physiclaw.studio.curate import LOOKALIKE_FRACTION
from physiclaw.studio.draft import DraftError
from physiclaw.studio.session import StudioSession, view_reply

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent

START_HINT = "Start it first: physiclaw mcp -H"
BUSY_MSG = "a call is already in flight — one rig"


class _Busy(Exception):
    """The rig is mid-call — mapped to 409 by `_refused`."""


def _error(code: int, message: str) -> JSONResponse:
    # Same wire shape as the core calibration handlers' `_err` — a
    # deliberate copy, not an import: the studio is an independent
    # process and must not couple to the core server package.
    return JSONResponse({"status": "error", "message": message}, status_code=code)


def _refused(e: Exception) -> JSONResponse:
    """The ONE exception → HTTP mapping. Every brains-layer refusal
    (DraftError/PagesError/MacroError/PlaybookError — all ValueError)
    is a 400 with its verbatim user-facing text; anything unrecognized
    re-raises (a bug, not a refusal)."""
    if isinstance(e, _Busy):
        return _error(409, BUSY_MSG)
    if isinstance(e, KeyError):
        return _error(400, f"missing field {e}")
    if isinstance(e, ValueError):
        return _error(400, str(e))
    if isinstance(e, ConnectionError):
        return _error(502, f"{e}. {START_HINT}")
    if isinstance(e, RuntimeError):
        return _error(500, str(e))
    raise e


async def _json_body(request: Request) -> dict:
    # Local on purpose (not `core.bridge.handler.json_or_none`): the
    # studio process must not import the core server's modules, and a
    # draft edit wants a loud refusal, not a silent None.
    try:
        body = json.loads(await request.body())
    except json.JSONDecodeError:
        raise DraftError("body must be JSON") from None
    if not isinstance(body, dict):
        raise DraftError("body must be a JSON mapping")
    return body


async def _act(session: StudioSession, tool: str, args: dict) -> dict:
    """One phone action as a view DICT — the busy gate plus
    `session.act`; serialization to a JSONResponse happens once, at
    the edge (handlers that enrich the view never re-parse it)."""
    if session.busy:
        raise _Busy()
    return await session.act(tool, args)


# ---------- pages ----------


async def handle_page(request: Request) -> HTMLResponse:
    """GET / — the single-file studio app. `no-store` so the browser
    always pulls the latest build after an upgrade (the setup-wizard
    convention)."""
    return HTMLResponse(
        read_text(STATIC_DIR / "studio.html"),
        headers={"Cache-Control": "no-store"},
    )


async def handle_state(request: Request, session: StudioSession) -> JSONResponse:
    """GET /api/state — connection + surface, without dialing out.
    `limits` carries the server-side constants the UI renders (gauge
    caps, evidence bars) so JS never re-spells them."""
    return JSONResponse(
        {
            "status": "ok",
            **session.state(),
            "limits": {
                "max_steps": MAX_STEPS,
                "min_freq": MIN_FREQ,
                "lookalike": LOOKALIKE_FRACTION,
            },
        }
    )


async def handle_act(request: Request, session: StudioSession) -> JSONResponse:
    """POST /api/act {tool, args} — one published tool, verbatim args.
    Arg validation is the server's job; its error text comes back in
    `message` untouched. With REC armed, a recordable gesture appends a
    draft step (checked BEFORE the arm moves — a press without its
    label refuses here), its after-view becomes the step snapshot, and
    the reply carries the fresh draft feedback (no second round-trip)."""
    try:
        body = await _json_body(request)
        tool = body["tool"]
        args = body.get("args") or {}
        if not isinstance(tool, str) or not isinstance(args, dict):
            return _error(400, "`tool` is a string, `args` a mapping")

        draft = None
        recording = False
        app = session.app
        if app:
            draft = draft_store.load_draft(app)
            recording = record.check_recordable(draft, tool, args)
        view = await _act(session, tool, args)
        if not recording or not view.get("image"):
            # No after-view, nothing to snapshot — not recorded.
            return JSONResponse({"status": "ok", "tool": tool, **view})

        assert draft is not None and app is not None
        macro_name = draft["recording"]["macro"]
        snap = draft_store.save_snap(draft, view["image"]["data"], view["listing"])
        index = record.record_step(draft, tool, args, snap)
        try:
            page = curate.page_guess(draft, view["listing"])
        except DraftError:
            page = None
        steps = draft["macros"][macro_name]["steps"]
        steps[index]["page"] = page
        prev_page = steps[index - 1].get("page") if index > 0 else None
        draft_store.save_draft(app, draft)
    except Exception as e:
        return _refused(e)
    return JSONResponse(
        {
            "status": "ok",
            "tool": tool,
            **view,
            **_feedback(draft),
            "recorded": {
                "macro": macro_name,
                "index": index,
                "steps": len(steps),
                "armed": draft["recording"] is not None,
                "page": page,
                # The split hint: the screen crossed into another
                # captured page — consider cutting the macro here.
                "page_changed": bool(page and prev_page and page != prev_page),
            },
        }
    )


# ---------- draft curation ----------
# All local-file work — no busy gate except where the arm moves.
# Draft/pack parser errors come back 400 with their verbatim text.


def _draft_view(draft: dict) -> dict:
    """The draft for the browser — shot listings stay server-side (the
    rail renders rows via /api/shot/{id}/view when one is opened)."""
    return {
        **draft,
        "shots": {sid: {"page": s["page"]} for sid, s in draft["shots"].items()},
    }


def _app_or_refuse(session: StudioSession) -> str:
    if not session.app:
        raise DraftError(
            "no app pack selected — start with: physiclaw studio --app <name>"
        )
    return session.app


def _feedback(draft: dict) -> dict:
    """Everything a draft-mutating reply carries: the fresh draft,
    per-anchor evidence, and the route editor's validation + pack
    pickers (one pack load — `route.editor_feedback`)."""
    return {
        "draft": _draft_view(draft),
        "evidence": curate.evidence(draft),
        **route.editor_feedback(draft),
    }


def _ok_draft(draft: dict) -> JSONResponse:
    return JSONResponse({"status": "ok", **_feedback(draft)})


# One draft mutation per op — each row is exactly one brains call, so
# op ownership stays readable against the module split (draft_store =
# pages/shots/landmarks, record = macros/steps, route = playbooks).
_DRAFT_OPS = {
    "page_add": lambda d, b: draft_store.add_page(d, b["name"]),
    "page_update": lambda d, b: draft_store.update_page(
        d,
        b["page"],
        anchors=b.get("anchors"),
        forbid=b.get("forbid"),
        scrollable=b.get("scrollable"),
    ),
    "page_delete": lambda d, b: draft_store.delete_page(d, b["page"]),
    "shot_delete": lambda d, b: draft_store.delete_shot(d, b["shot"]),
    "landmark_set": lambda d, b: draft_store.set_landmark(
        d, b["name"], b["label"], b["bbox"]
    ),
    "landmark_clear": lambda d, b: draft_store.clear_landmark(d, b["name"]),
    "macro_add": lambda d, b: record.add_macro(d, b["name"]),
    "macro_delete": lambda d, b: record.delete_macro(d, b["name"]),
    "macro_update": lambda d, b: record.update_macro(
        d,
        b["macro"],
        description=b.get("description"),
        inputs=b.get("inputs"),
        target=b.get("target"),
    ),
    "record_start": lambda d, b: record.start_recording(
        d, b["macro"], b.get("replace")
    ),
    "record_stop": lambda d, b: record.stop_recording(d),
    "step_update": lambda d, b: record.update_step(
        d, b["macro"], b["index"], b["fields"]
    ),
    "step_delete": lambda d, b: record.delete_step(d, b["macro"], b["index"]),
    "step_expect": lambda d, b: record.insert_step(
        d, b["macro"], b.get("index"), record.expect_step(b["text"], b["bbox"])
    ),
    "step_dismissal": lambda d, b: record.mark_dismissal(d, b["macro"], b["index"]),
    "playbook_add": lambda d, b: route.add_playbook(d, b["name"]),
    "playbook_delete": lambda d, b: route.delete_playbook(d, b["name"]),
    "playbook_update": lambda d, b: route.update_playbook(
        d, b["playbook"], b["fields"]
    ),
    "route_insert": lambda d, b: route.entry_insert(
        d, b["playbook"], b.get("index"), b["entry"]
    ),
    "route_embed": lambda d, b: route.entry_insert(
        d, b["playbook"], b.get("index"), route.entry_from_macro(d, b["macro"])
    ),
    "route_update": lambda d, b: route.entry_update(
        d, b["playbook"], b["index"], b["entry"]
    ),
    "route_delete": lambda d, b: route.entry_delete(d, b["playbook"], b["index"]),
    "route_move": lambda d, b: route.entry_move(
        d, b["playbook"], b["index"], b["delta"]
    ),
}


async def handle_draft(request: Request, session: StudioSession) -> JSONResponse:
    """GET /api/draft — the draft plus evidence and route feedback."""
    try:
        return _ok_draft(draft_store.load_draft(_app_or_refuse(session)))
    except Exception as e:
        return _refused(e)


async def handle_draft_edit(request: Request, session: StudioSession) -> JSONResponse:
    """POST /api/draft/edit {op, ...} — one local draft mutation via
    the `_DRAFT_OPS` table: load → mutate → save, and the reply is
    always the fresh draft feedback."""
    try:
        app = _app_or_refuse(session)
        body = await _json_body(request)
        op = _DRAFT_OPS.get(str(body.get("op")))
        if op is None:
            return _error(400, f"unknown draft op {body.get('op')!r}")
        draft = draft_store.load_draft(app)
        op(draft, body)
        draft_store.save_draft(app, draft)
        return _ok_draft(draft)
    except Exception as e:
        return _refused(e)


async def handle_draft_shot(request: Request, session: StudioSession) -> JSONResponse:
    """POST /api/draft/shot {page} — peek NOW and attach the view to the
    page as one observation."""
    try:
        app = _app_or_refuse(session)
        body = await _json_body(request)
        page = body["page"]
        draft = draft_store.load_draft(app)
        if page not in draft["pages"]:  # refuse before moving the arm
            return _error(400, f"no drafted page {page!r}")
        view = await _act(session, "peek", {})
        if not view.get("image"):
            return _error(500, "peek returned no view — nothing to attach")
        shot_id = draft_store.add_shot(
            draft, page, view["listing"], view["image"]["data"]
        )
        draft_store.save_draft(app, draft)
    except Exception as e:
        return _refused(e)
    return JSONResponse(
        {
            "status": "ok",
            "shot": shot_id,
            **_feedback(draft),
            **{k: view[k] for k in ("text", "image", "rows")},
        }
    )


async def handle_shot_jpeg(request: Request, session: StudioSession):
    try:
        p = draft_store.shot_jpeg(_app_or_refuse(session), request.path_params["shot"])
    except DraftError as e:
        return _error(404, str(e))
    return FileResponse(p, media_type="image/jpeg")


async def handle_shot_view(request: Request, session: StudioSession) -> JSONResponse:
    """GET /api/shot/{shot}/view — a stored snapshot (page shot or
    macro-step snap — one registry) rendered exactly like a live peek,
    so curation works offline: click a shot, click its rows."""
    try:
        app = _app_or_refuse(session)
        shot_id = request.path_params["shot"]
        shot = draft_store.load_draft(app)["shots"].get(shot_id)
        if shot is None:
            return _error(404, f"no shot {shot_id!r}")
        jpeg = draft_store.shot_jpeg(app, shot_id).read_bytes()
    except DraftError as e:
        return _error(404, str(e))
    return JSONResponse(
        {
            "status": "ok",
            "shot": shot_id,
            "page": shot["page"],
            "text": "",
            "image": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(jpeg).decode(),
            },
            "rows": [el.to_dict() for el in Screen.read(shot["listing"]).rows],
        }
    )


async def handle_matrix(request: Request, session: StudioSession) -> JSONResponse:
    """GET /api/draft/matrix — mine + score, writing nothing."""
    try:
        draft = draft_store.load_draft(_app_or_refuse(session))
        return JSONResponse({"status": "ok", **curate.dry_run(draft)})
    except Exception as e:
        return _refused(e)


async def handle_commit(request: Request, session: StudioSession) -> JSONResponse:
    """POST /api/draft/commit — write the pack sections + merged learned."""
    try:
        draft = draft_store.load_draft(_app_or_refuse(session))
        return JSONResponse({"status": "ok", **curate.commit(draft)})
    except Exception as e:
        return _refused(e)


# ---------- macros ----------


async def handle_macro_save(request: Request, session: StudioSession) -> JSONResponse:
    """POST /api/macro/save {macro} — emit MACRO.yml (parse-validated)
    to the draft's target: the pack's macros/ or the global store."""
    try:
        app = _app_or_refuse(session)
        body = await _json_body(request)
        name = body["macro"]
        m = draft_store.load_draft(app).get("macros", {}).get(name)
        if m is None:
            return _error(400, f"no drafted macro {name!r}")
        path = record.save_macro(app, name, m)
    except Exception as e:
        return _refused(e)
    return JSONResponse(
        {"status": "ok", "path": str(path), "yaml": record.emit_macro_yaml(name, m)}
    )


async def handle_macro_rehearse(
    request: Request, session: StudioSession
) -> JSONResponse:
    """POST /api/macro/rehearse {macro, values?, start_at?} — replay the
    draft over this session via `run_and_record` (the same core `macros
    run` drives). A mid-run gesture failure is a RESULT: 200 with
    ok=false and the composed step log."""
    try:
        app = _app_or_refuse(session)
        body = await _json_body(request)
        name = body["macro"]
        m = draft_store.load_draft(app).get("macros", {}).get(name)
        if m is None:
            return _error(400, f"no drafted macro {name!r}")
        spec = record.macro_spec(name, m)
        if session.busy:
            raise _Busy()
        result = await session.run_macro(
            spec, body.get("values") or {}, start_at=body.get("start_at") or ""
        )
    except Exception as e:
        return _refused(e)
    return JSONResponse(
        {
            "status": "ok",
            "ok": result.ok,
            "run_id": result.run_id,
            "aborted_step": result.aborted_step,
            "reason": result.reason,
            **view_reply(result.blocks),
        }
    )


# ---------- playbooks ----------


async def handle_playbook_commit(
    request: Request, session: StudioSession
) -> JSONResponse:
    """POST /api/playbook/commit {playbook} — one key under `playbooks:`
    (compile-refused; hand-authored siblings survive byte-for-byte)."""
    try:
        app = _app_or_refuse(session)
        body = await _json_body(request)
        name = body["playbook"]
        p = draft_store.load_draft(app).get("playbooks", {}).get(name)
        if p is None:
            return _error(400, f"no drafted playbook {name!r}")
        result = route.commit_playbook(app, name, p)
    except Exception as e:
        return _refused(e)
    return JSONResponse({"status": "ok", **result})


async def handle_playbook_rehearse(
    request: Request, session: StudioSession
) -> JSONResponse:
    """POST /api/playbook/rehearse {playbook, values?} — arm from the
    COMMITTED pack file (bad specs fail before the phone moves), then
    drive the real walk over this session; each synthesized turn's
    summary comes back as a log line."""
    from physiclaw.conductor import rehearsal

    lines: list[str] = []
    try:
        app = _app_or_refuse(session)
        body = await _json_body(request)
        name = body["playbook"]
        if session.busy:
            raise _Busy()
        program, registry = rehearsal.arm(
            app, name, body.get("values") or {}, emit_warn=lines.append
        )
        outcome = await session.run_walk(program, registry, emit=lines.append)
    except Exception as e:
        return _refused(e)
    return JSONResponse({"status": "ok", "outcome": outcome, "lines": lines})


def build_app(session: StudioSession) -> Starlette:
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
            Route("/api/draft", bind(handle_draft)),
            Route("/api/draft/edit", bind(handle_draft_edit), methods=["POST"]),
            Route("/api/draft/shot", bind(handle_draft_shot), methods=["POST"]),
            Route("/api/draft/matrix", bind(handle_matrix)),
            Route("/api/draft/commit", bind(handle_commit), methods=["POST"]),
            Route("/api/macro/save", bind(handle_macro_save), methods=["POST"]),
            Route("/api/macro/rehearse", bind(handle_macro_rehearse), methods=["POST"]),
            Route(
                "/api/playbook/commit", bind(handle_playbook_commit), methods=["POST"]
            ),
            Route(
                "/api/playbook/rehearse",
                bind(handle_playbook_rehearse),
                methods=["POST"],
            ),
            Route("/api/shot/{shot}", bind(handle_shot_jpeg)),
            Route("/api/shot/{shot}/view", bind(handle_shot_view)),
        ],
        lifespan=lifespan,
    )
