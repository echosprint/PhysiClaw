"""Local (engine-side) tools, exposed via the same tools= API as MCP tools.

Local handlers run in-process and may mutate `Session` (the engine reads
`sentinel_status`, `sentinel_turn_created_job`, and `plan` after each
turn). Other handlers are stateless. Source of truth for the registry is
`build_registry()` at the bottom; insertion order there determines wire
order in `tools[]` and exploits LLM position bias.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from physiclaw.agent.engine import (
    jobs,
    memory,
    pitfalls,
    scratchpad,
    screen_layout,
    skill,
)
from physiclaw.agent.engine.job_store import (
    KIND_ONE_TIME,
    KIND_PERIODIC,
    NEVER,
    load_jobs,
)
from physiclaw.agent.engine.session import Session
from physiclaw.agent.macros import runner as macro_runner
from physiclaw.agent.macros.model import Macro
from physiclaw.agent.runtime.sentinel import IDLE, STATUSES
from physiclaw.common import gesture_vocab

# A handler returns either plain text, or raw MCP-style blocks (run_macro's
# step log + final view) — dispatch routes a block list through the same
# verdict/observer path as an MCP result.
Handler = Callable[[Session, dict], Awaitable[str | list[dict]]]


@dataclass(frozen=True)
class LocalTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Handler
    # Whether the handler returns raw MCP-style blocks rather than text, and
    # so takes the gesture-result path in `dispatch` (verdict parse, result
    # observers, screen supersede). DECLARED here rather than sniffed from
    # the returned value: the two pipelines differ in what they let a result
    # do to session state, and that is a property of the tool, not of one
    # call's return type. `dispatch` enforces the declaration in both
    # directions — a mismatched return type becomes a normal tool error.
    returns_blocks: bool = False


# ---------- handlers ----------


async def _handle_note(session: Session, args: dict) -> str:
    # The note's text lives in the assistant message's tool_calls arguments
    # (in history), not in session state. Handler pairs the call with a
    # tool_result (principle 6) and — if `scratchpad` was passed — also
    # mutates `session.scratchpad`. Merging scratchpad into note avoids
    # spending a `[note, one-other]` action slot on a write-down.
    summary = (args.get("summary") or "").strip()
    msg = f"noted: {summary}"
    sp_arg = args.get("scratchpad")
    if sp_arg is not None:
        try:
            msg += f" | {scratchpad.write(session, sp_arg)}"
        except ValueError as e:
            msg += f" | scratchpad rejected: {e}"
    return msg


async def _handle_update_progress(session: Session, args: dict) -> str:
    try:
        session.plan.update(**args)
    except ValueError as e:
        return f"update_progress rejected: {e}"
    return "progress updated"


async def _handle_append_log(_session: Session, args: dict) -> str:
    memory.append_log(args["entry"])
    return "log appended"


async def _handle_save_memory(session: Session, args: dict) -> str:
    memory.save_fact(args["text"])
    session.saved_memory = True  # satisfies the pre-close memory-cue gate
    return memory.render_result(f"saved: {args['text'].strip()!r}")


async def _handle_create_job(session: Session, args: dict) -> str:
    jobs.create_job(
        id=args["id"],
        description=args["description"],
        schedule=args["schedule"],
        context=args["context"],
        kind=args.get("kind", "one-time"),
    )
    session.sentinel_turn_created_job = True
    return f"scheduled job {args['id']!r}"


async def _handle_get_job(_session: Session, args: dict) -> str:
    job = jobs.get_job(args["id"])
    return (
        f"## {job.id}\n"
        f"{job.description}\n\n"
        f"- Type: {job.kind}\n"
        f"- Status: {job.status}\n"
        f"- Schedule: `{job.schedule}`\n"
        f"- Context: {job.context}\n"
        f"- Next fire time: {job.next_fire_time or NEVER}\n"
        f"- Last fire time: {job.last_fire_time or NEVER}\n"
        f"- Execution time: {job.execution_time or NEVER}\n"
        f"- Execution result: {job.execution_result or NEVER}\n"
    )


async def _handle_read_memory(_session: Session, _args: dict) -> str:
    out = memory.load_persistent()
    return out if out else "(memory.md is empty)"


async def _handle_read_logs(_session: Session, args: dict) -> str:
    n = args.get("entries", memory.DEFAULT_LOG_ENTRIES)
    out = memory.load_recent_entries(n)
    return out if out else "(no log entries found)"


async def _handle_update_memory(session: Session, args: dict) -> str:
    memory.update_fact(args["old"], args["new"])
    session.saved_memory = True  # satisfies the pre-close memory-cue gate
    return memory.render_result("memory.md updated")


async def _handle_list_jobs(_session: Session, args: dict) -> str:
    want = (args.get("status") or "all").lower()
    rows = load_jobs()
    if want != "all":
        rows = [j for j in rows if j.status == want]
    if not rows:
        suffix = f" with status={want!r}" if want != "all" else ""
        return f"no jobs{suffix}"
    lines = [f"{len(rows)} job(s):"]
    for j in rows:
        nxt = j.next_fire_time or NEVER
        lines.append(
            f"  [{j.kind}] [{j.status}] {j.id} — {j.description} (next: {nxt})"
        )
    return "\n".join(lines)


async def _handle_finish_job(_session: Session, args: dict) -> str:
    # finish_job composes the reply — a periodic done/fail RE-ARMS the
    # job, and the agent must see that in the tool result.
    return jobs.finish_job(id=args["id"], status=args["status"], recap=args["recap"])


async def _handle_wait(_session: Session, args: dict) -> str:
    seconds = args["seconds"]
    await asyncio.sleep(seconds)
    return f"waited {seconds}s — `peek` now to see what changed."


async def _handle_end_session(session: Session, args: dict) -> str:
    status = args["status"]
    if status not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}, got {status!r}")
    session.sentinel_status = status
    session.sentinel_recap = args.get("recap", "").strip()
    return f"session closing: {status}"


async def _handle_add_pitfall(session: Session, args: dict) -> str:
    # Prepend up to 3 traps (append-only), then mark the session so the pre-close
    # gate lets `end_session` through and the post-session curator runs. Setting
    # the flag even on a no-op is intentional — re-forcing an empty add is
    # pointless (the agent judged there was nothing to bank).
    res = pitfalls.add(args.get("items") or [])
    session.added_pitfalls = True
    return f"{res['added']} pitfall(s) added — {res['total']} total"


async def _handle_report_screen_layout(session: Session, args: dict) -> str:
    was_complete = screen_layout.is_learned()
    result = screen_layout.record(
        args.get("page", ""),
        args.get("field", ""),
        args.get("bbox") or [],
        args.get("app"),
    )
    # If THIS call finished first-run setup, close the session cleanly and ask
    # the engine to restart — the fresh SYSTEM prompt then carries the layout
    # so the agent handles the original request with it loaded.
    if not was_complete and screen_layout.is_learned():
        session.sentinel_status = IDLE
        session.sentinel_recap = (
            "first-run screen layout captured — restarting to load it"
        )
        session.restart_for_setup = True
    return result


def _handle_skill_factory(skill_registry: dict[str, skill.Skill]) -> Handler:
    async def _handle(_session: Session, args: dict) -> str:
        return skill.dispatch(skill_registry, args)

    return _handle


def _handle_run_macro_factory(macro_registry: dict[str, Macro]) -> Handler:
    """Execute one enabled macro over the process MCP singleton.
    `run_and_record` owns the stats fold (shared with CLI rehearsal, so a
    successful rehearsal resets `consecutive_aborts` too). Raises on an
    unknown name or bad inputs (dispatch marks is_error); a mid-run abort
    returns normally — the step log + current screen are a substantive
    result the model must read, not an error to retry."""

    async def _handle(session: Session, args: dict) -> str | list[dict]:
        # Local import: mcp_tool pulls the mcp SDK; keep it off the module's
        # import path so offline assembly (`physiclaw prompt`) stays light.
        from physiclaw.agent.engine.mcp_tool import get_mcp

        name = (args.get("name") or "").strip()
        spec = macro_registry.get(name)
        if spec is None:
            available = ", ".join(sorted(macro_registry)) or "(none)"
            raise ValueError(f"unknown macro {name!r}. Available: {available}")
        result = await macro_runner.run_and_record(
            spec,
            args.get("inputs") or {},
            await get_mcp(),
            start_at=(args.get("start_at") or "").strip(),
        )
        if not result.ok:
            # One strike. The rehearsed path stopped matching this screen, so
            # every later call would replay completed steps against a state
            # we already know diverged — which is exactly what the abort
            # header tells the agent not to do. `policy.BurnedMacro` enforces
            # it from here; bad_input can't reach this line (it raised).
            session.failed_macros.add(spec.name)
        return result.blocks

    return _handle


# ---------- tool definitions ----------


_NOTE = LocalTool(
    name="note",
    description=(
        "MUST be called every turn alongside whatever other tool you call. "
        "`summary` is one line (≤20 words): the LAST result + this turn's "
        "action (`qty still 2 after retry — opening product page`), never "
        "intent alone. **It is the ONLY part of the turn that survives "
        "compaction** — once a turn ages out, the screen, the tap, every "
        "other tool_result is gone; your `summary` alone represents that "
        "turn, and intent-only summaries hide a retry loop from your future "
        "self.\n"
        "\n"
        "Optional `scratchpad` field — see CONVENTION § Scratchpad. "
        "REQUIRED on a ⚠ pre-compression checkpoint turn (a tail notice "
        "tells you when)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "One line, ≤20 words: last result + this turn's action.",
            },
            "scratchpad": {
                "type": "string",
                "description": (
                    "Optional. Replace the scratchpad's full content. "
                    "Reissue the full text to update; pass empty string to clear."
                ),
            },
        },
        "required": ["summary"],
    },
    handler=_handle_note,
)


_UPDATE_PROGRESS = LocalTool(
    name="update_progress",
    description=(
        "Mutate the plan pinned at the tail of every request. Pass "
        "any subset of `user_said` (IM verbatim), `understanding` (one-line "
        "read), `steps` (full ordered list of `{content, status}`). Status is "
        "`pending` / `in_progress` / `completed`; **exactly one step may be "
        "`in_progress`** — engine rejects violators.\n"
        "\n"
        "Rules — when to call (draft once → tick after each step → re-plan "
        "on shift), step granularity (one objective per step), and the "
        "skip-if-≤4-steps shortcut — all in CONVENTION § The plan."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "user_said": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1000,
                "description": "What the user literally said — quote the IM message.",
            },
            "understanding": {
                "type": "string",
                "minLength": 5,
                "maxLength": 1000,
                "description": "One sentence: what you think they want and why.",
            },
            "steps": {
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 200,
                            "description": "One concrete imperative action.",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": (
                                "pending = not started; in_progress = "
                                "currently doing (max ONE at a time); "
                                "completed = finished."
                            ),
                        },
                    },
                    "required": ["content", "status"],
                },
                "description": (
                    "Full ordered step list. Exactly one step may be "
                    "in_progress at a time; the engine rejects plans that "
                    "violate this."
                ),
            },
        },
        "required": [],
    },
    handler=_handle_update_progress,
)


_APPEND_LOG = LocalTool(
    name="append_log",
    description=(
        "Append one line to today's `memory/YYYY-MM-DD.md` daily log. See "
        "PERSISTENCE § When to write for trigger rules and § Format for the "
        "line shape."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "entry": {
                "type": "string",
                "description": "Format: `[HH:MM] app: page -> page - what you did`",
            },
        },
        "required": ["entry"],
    },
    handler=_handle_append_log,
)


_SAVE_MEMORY = LocalTool(
    name="save_memory",
    description=(
        "Append a durable fact or preference to `memory/memory.md`. Use only "
        "when the user says 'remember this' or you learn a lasting preference "
        "— not for session detail (that's `append_log`)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
        },
        "required": ["text"],
    },
    handler=_handle_save_memory,
)


_CREATE_JOB = LocalTool(
    name="create_job",
    description=(
        "Schedule a follow-up in `jobs/jobs.md`. Use on WAIT to set the "
        "resume check, or when the user asks for a recurring task. See "
        "JOBS for lifecycle and id format."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "lowercase / digits / hyphens"},
            "description": {"type": "string"},
            "schedule": {
                "type": "string",
                "description": "5-field cron (min hour dom mon dow)",
            },
            "context": {"type": "string", "description": "at least 10 chars"},
            "kind": {"type": "string", "enum": [KIND_ONE_TIME, KIND_PERIODIC]},
        },
        "required": ["id", "description", "schedule", "context"],
    },
    handler=_handle_create_job,
)


_GET_JOB = LocalTool(
    name="get_job",
    description=(
        "Return all fields of one job (description, type, status, schedule, "
        "context, fire times). Use when `list_jobs`' one-line summary isn't "
        "enough. Raises if the id does not exist."
    ),
    input_schema={
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
    },
    handler=_handle_get_job,
)


_READ_MEMORY = LocalTool(
    name="read_memory",
    description=(
        "Re-read `memory/memory.md` from disk. SYSTEM already shows it under "
        "`## memory.md` at session start — call this only after a "
        "`save_memory` / `update_memory` mid-session, before another "
        "`update_memory` whose `old` must match byte-exact."
    ),
    input_schema={"type": "object", "properties": {}, "required": []},
    handler=_handle_read_memory,
)


_READ_LOGS = LocalTool(
    name="read_logs",
    description=(
        "Fetch the last N log entries across `memory/YYYY-MM-DD.md` files, "
        "most recent first. Recent entries are auto-injected at wake — call "
        "this only when you need MORE history. Walks back through prior days "
        "if today's file has fewer than N. Each `[HH:MM]` is rewritten to "
        "`[YYYY-MM-DD HH:MM]` for unambiguous cross-day order."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "entries": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "description": (
                    f"Number of entries to return (default "
                    f"{memory.DEFAULT_LOG_ENTRIES})."
                ),
            },
        },
        "required": [],
    },
    handler=_handle_read_logs,
)


_UPDATE_MEMORY = LocalTool(
    name="update_memory",
    description=(
        "Replace the single occurrence of `old` with `new` in "
        "`memory/memory.md`. Empty `new` deletes the line. Errors if `old` is "
        "not found or matches more than once — narrow with surrounding text. "
        "After any prior `save_memory` / `update_memory` this session, the "
        "SYSTEM snapshot is stale; call `read_memory` first."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "old": {"type": "string", "description": "Exact substring to replace."},
            "new": {
                "type": "string",
                "description": "Replacement; empty string deletes the line.",
            },
        },
        "required": ["old", "new"],
    },
    handler=_handle_update_memory,
)


_LIST_JOBS = LocalTool(
    name="list_jobs",
    description=(
        "List jobs from `jobs/jobs.md` as one-line summaries. Optional "
        "`status` filter: all (default) / pend / fired / cancel / done / fail."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["all", "pend", "fired", "cancel", "done", "fail"],
            },
        },
        "required": [],
    },
    handler=_handle_list_jobs,
)


_FINISH_JOB = LocalTool(
    name="finish_job",
    description=(
        "Mark a fired job as terminated. `status` is `done` (work complete), "
        "`fail` (blocked / impossible), or `cancel` (no longer needed — user "
        "changed mind, task already happened, or rescheduling). One per fired "
        "job per wake — engine never auto-marks. Periodic jobs reset to "
        "`pend` on done/fail; `cancel` is permanent. Raises on already-"
        "terminal jobs."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "status": {"type": "string", "enum": ["done", "fail", "cancel"]},
            "recap": {"type": "string", "description": "One-line outcome summary."},
        },
        "required": ["id", "status", "recap"],
    },
    handler=_handle_finish_job,
)


_WAIT = LocalTool(
    name="wait",
    description=(
        "Block 1–60s, then return. For short in-session waits when the "
        "user is actively engaged. **Your next call must be `peek`** — "
        "chaining `wait → wait` without observing is a bug. See CONVENTION "
        "§ Wait-retry for the full pattern (max 3 attempts, escalate via "
        "`end_session(WAIT)` + `create_job`)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "seconds": {"type": "integer", "minimum": 1, "maximum": 60},
        },
        "required": ["seconds"],
    },
    handler=_handle_wait,
)


_ADD_PITFALL = LocalTool(
    name="add_pitfall",
    description=(
        "Bank up to 3 traps you hit AND got past this session, so a future run "
        "skips them (append-only, joins the always-on `## Learned pitfalls`). "
        "Each: **lead with the app**, then `trap → the fix that worked`, one "
        "terse line. Empty list if none worth banking. Required before a long "
        "DONE close."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "maxItems": 3,
                "items": {"type": "string", "maxLength": 150},
                "description": "Up to 3 traps, each `<app>: <trap> → <fix>`, terse.",
            },
        },
        "required": ["items"],
    },
    handler=_handle_add_pitfall,
)


_END_SESSION = LocalTool(
    name="end_session",
    description=(
        "Close the session. `status` is one of: DONE (complete, verified), "
        "STUCK (blocker outside your control), FAIL (task cannot succeed), "
        "IDLE (wake triggered, no work needed), WAIT (paused for user reply "
        "— pair with `create_job` to auto-resume)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": sorted(STATUSES)},
            "recap": {"type": "string", "description": "One-line outcome summary."},
        },
        "required": ["status", "recap"],
    },
    handler=_handle_end_session,
)


_REPORT_SCREEN_LAYOUT = LocalTool(
    name="report_screen_layout",
    description=(
        "First-run setup: save ONE input / keyboard / Paste / Send box you read "
        "off a screenshot — one call per box. Follow the `screen-layout` skill "
        "for which pages and fields to capture; it returns the layout so far "
        "plus what's left. `app` is required for the chat pages (labels the "
        "chat boxes). Once all fields are in they load into your context next "
        "wake."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "page": {
                "type": "string",
                "enum": ["spotlight", "chat-no-keyboard", "chat-keyboard"],
                "description": "Which page the latest screenshot shows.",
            },
            "field": {
                "type": "string",
                "enum": list(screen_layout.ALL_FIELDS),
                "description": "Which box this bbox is — must be a field valid "
                "for `page` (see the `screen-layout` skill).",
            },
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": "[left, top, right, bottom] in 0–1 coords, copied "
                "verbatim from the screenshot element.",
            },
            "app": {
                "type": "string",
                "description": "The chat app you opened (any app, e.g. 'wechat', "
                "'whatsapp', 'telegram', 'signal', 'messenger'). Required for "
                "the chat pages; ignored for spotlight.",
            },
        },
        "required": ["page", "field", "bbox"],
    },
    handler=_handle_report_screen_layout,
)


_RUN_MACRO_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": (
                "Macro name, exactly as listed under `## Available Macros`."
            ),
        },
        "inputs": {
            "type": "object",
            "description": (
                "Values for the macro's declared inputs — strings only. "
                "Omit an optional input to use its default."
            ),
        },
        "start_at": {
            "type": "string",
            "description": (
                "Optional. Begin at this step NAME (as listed under the "
                "macro's `steps:` in `## Available Macros`) instead of the "
                "first, for when you already did the leading steps by hand. "
                "The skipped steps are NOT executed. Omit to run the whole "
                "macro."
            ),
        },
    },
    "required": ["name"],
}


_SKILL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Skill name, e.g. `wechat`.",
        },
        "reference": {
            "type": "string",
            "description": (
                "Optional path under the skill's `references/` directory. "
                "Loads that file's text instead of SKILL.md — use only when "
                "SKILL.md explicitly points at a reference."
            ),
        },
    },
    "required": ["name"],
}


def schemas(registry: dict[str, LocalTool]) -> list[dict]:
    """Flatten a local-tool registry into wire-shape dicts (the same shape
    as MCP `list_tools()` entries)."""
    return [
        {
            "name": lt.name,
            "description": lt.description,
            "input_schema": lt.input_schema,
        }
        for lt in registry.values()
    ]


def build_registry(
    skill_registry: dict[str, skill.Skill],
    macro_registry: dict[str, Macro] | None = None,
) -> dict[str, LocalTool]:
    """All local tools keyed by name. Skill / run_macro are included iff
    skills / enabled macros were discovered (keeps the tool surface minimal
    when none exist).

    **Insertion order matters.** `schemas()` iterates `.values()`, and the
    engine concatenates MCP tools + these to build `tool_schemas`, so the
    order here directly determines the order of `tools[]` in the provider
    request. LLMs show mild position bias — the turn-shape primitives
    (`note`, `update_progress`) are listed first so they sit near the top
    of the array. Don't reshuffle, and don't refactor this dict into a
    set or comprehension.
    """
    tools: dict[str, LocalTool] = {
        _NOTE.name: _NOTE,
        _UPDATE_PROGRESS.name: _UPDATE_PROGRESS,
        _APPEND_LOG.name: _APPEND_LOG,
        _SAVE_MEMORY.name: _SAVE_MEMORY,
        _READ_MEMORY.name: _READ_MEMORY,
        _READ_LOGS.name: _READ_LOGS,
        _UPDATE_MEMORY.name: _UPDATE_MEMORY,
        _CREATE_JOB.name: _CREATE_JOB,
        _GET_JOB.name: _GET_JOB,
        _LIST_JOBS.name: _LIST_JOBS,
        _FINISH_JOB.name: _FINISH_JOB,
        _WAIT.name: _WAIT,
        _REPORT_SCREEN_LAYOUT.name: _REPORT_SCREEN_LAYOUT,
        _ADD_PITFALL.name: _ADD_PITFALL,
        _END_SESSION.name: _END_SESSION,
    }
    # First-run only: once the layout is learned, drop the setup tool — the
    # `screen-layout` skill it points to is pruned too, so keeping the tool
    # would dangle and clutter the tool surface every wake. Resetting the
    # layout brings both back.
    if screen_layout.is_learned():
        tools.pop(_REPORT_SCREEN_LAYOUT.name, None)
    if skill_registry:
        tools["Skill"] = LocalTool(
            name="Skill",
            description=(
                f"Load a skill's workflow ({'/'.join(sorted(skill_registry))}) "
                "before acting in that app — don't tap blind. Default returns "
                "the skill's SKILL.md body. Pass `reference=<path>` to load a "
                "details file from the skill's `references/` directory when "
                "SKILL.md points at one."
            ),
            input_schema=_SKILL_SCHEMA,
            handler=_handle_skill_factory(skill_registry),
        )
    if macro_registry:
        # The constant, not a literal: policy.BurnedMacro, policy._GESTURES
        # and KeyboardTracker all key off gesture_vocab.RUN_MACRO, and every
        # one of them fails OPEN on a mismatch.
        tools[gesture_vocab.RUN_MACRO] = LocalTool(
            name=gesture_vocab.RUN_MACRO,
            returns_blocks=True,
            description=(
                f"Execute a rehearsed gesture macro "
                f"({'/'.join(sorted(macro_registry))}) as ONE step — see "
                "`## Available Macros` for what each does and its inputs. "
                "Use this instead of gesturing a covered stretch turn by "
                "turn: one call replaces every one of those turns' round "
                "trips. Aborts safely the moment the screen stops matching "
                "the rehearsed path; the result shows the completed steps "
                "plus the current screen, so you can continue manually. A "
                "macro that aborted this session cannot be called again, "
                "with or without `start_at`."
            ),
            input_schema=_RUN_MACRO_SCHEMA,
            handler=_handle_run_macro_factory(macro_registry),
        )
    return tools
