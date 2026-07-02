"""Engine — native tool-call loop. Follows OpenClaw's 7 principles.

  1. Structure via provider API: tools=[...] + message.tool_calls, not JSON.
  2. Normalized at the boundary: Provider returns AssistantMessage; raw
     provider-specific fields (reasoning_content) stripped before echo.
  3. Real finish_reason preserved and routed (length / content_filter / stop).
  4. Arguments validated against JSONSchema before dispatch.
  5. Errors marked: failed tool_calls get synthetic tool_result(is_error=True);
     finish_reason="error" / "length" / "content_filter" drive recovery.
  6. Transcript stays API-legal: each ToolCall gets exactly one ToolResult
     with matching tool_call_id in the very next message.
  7. Loop is the driver: model → tool_calls → dispatch → tool_results → model.
"""
import asyncio
import datetime as dt
import logging
import time
from dataclasses import dataclass

from physiclaw.agent.engine import builtin_tool, compact, jobs, memory, plan, prompt, scratchpad, screen_layout, skill
from physiclaw.agent.engine.builtin_tool import LocalTool, Session
from physiclaw.agent.engine.mcp_tool import McpClient, get_mcp, list_tools_cached
from physiclaw.agent.engine.dto import (
    AssistantMessage,
    FinishReason,
    Message,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from physiclaw.agent.provider import (
    Provider,
    ProviderTransientError,
    make_provider,
    mcp_blocks_to_content_blocks,
)
from physiclaw.agent.engine.trace import RawLog, Trace, brief_content, format_call_args, format_call_result
from physiclaw.agent.engine.validator import ValidationError, validate_arguments
from physiclaw.agent.runtime.hook import Trigger
from physiclaw.agent.runtime.sentinel import FAIL, STUCK, WAIT
from physiclaw.config import CONFIG

log = logging.getLogger(__name__)

# Runaway-loop backstop, not a context-safety limit. Prompt tokens grow
# ~624·t + 13k empirically (R²=0.97); at 1M context (Qwen3.6-plus) the
# hard wall is ~1,580 turns, so 300 leaves ample headroom.
MAX_TURNS = CONFIG.engine.max_turns
MAX_ATTEMPTS = CONFIG.engine.max_attempts
RETRY_BACKOFF = CONFIG.engine.retry_backoff_seconds
WAIT_DEFAULT_MINUTES = CONFIG.engine.wait_default_minutes


async def run(
    triggers: list[Trigger], *, model_ref: str
) -> None:
    """Run an engine session for `triggers`, retrying on STUCK.

    STUCK happens when the loop hit MAX_TURNS without a clean close, the
    provider exhausted its retries, or the session crashed. Up to
    MAX_ATTEMPTS fresh attempts run before we accept the STUCK outcome.
    DONE / FAIL / IDLE / WAIT are final on first occurrence — no retry.

    One extra restart is allowed when a session completes first-run
    screen-layout setup (`session.restart_for_setup`): the layout only
    reaches the SYSTEM prompt on a fresh render, so we re-run the same
    triggers once with the layout loaded to handle the original request.
    It doesn't consume a STUCK attempt, and fires at most once (the layout
    is complete from then on). This restart only helps when there's a real
    request to resume — a synthetic first-run wake (source="first-run") has
    none, so it ends instead and the layout loads on the next wake.

    `model_ref` is a `provider/model` string (e.g. `"qwen/qwen3.6-plus"`).
    Parsed inside `_run_session`; the provider is instantiated via
    `provider.make_provider(provider_id, model_id)`. Required — every
    caller goes through the launcher, which resolves `PHYSICLAW_MODEL`.
    """
    setup_restart_used = False
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        session = Session()
        await _run_session(triggers, model_ref=model_ref, session=session)
        if session.restart_for_setup and not setup_restart_used:
            setup_restart_used = True
            # Re-running is only useful when there's a real request to resume
            # with the layout now loaded. A synthetic first-run wake
            # (source="first-run") has no request, and the layout loads on the
            # next wake anyway — restarting would just replay the stale "learn
            # the layout" trigger and send the agent back into first-run setup.
            if any(t.source != "first-run" for t in triggers):
                attempt -= 1  # a setup restart isn't a STUCK retry
                log.info("screen layout learned during setup — restarting to load it")
                continue
            log.info("screen layout learned on first-run wake — no request to resume")
        if session.sentinel_status != STUCK:
            break
        if attempt < MAX_ATTEMPTS:
            log.warning(
                "session STUCK (attempt %d/%d): %r — retrying",
                attempt, MAX_ATTEMPTS, session.sentinel_recap,
            )


@dataclass(slots=True)
class PromptBundle:
    """The turn-0 request pieces assembled offline — no MCP or provider needed.
    Shared by `_run_session` and the `physiclaw prompt` dump so the dump matches
    the real request. `local_registry`/`local_schemas` are the engine's local
    tools; `layout_incomplete` gates the first-run reminder; the skill counts
    are for the session's tools-loaded log line."""
    system_prompt: str
    local_registry: dict[str, LocalTool]
    local_schemas: list[dict]
    layout_incomplete: bool
    builtin_skill_count: int
    user_skill_count: int


def build_prompt_bundle(provider_id: str) -> PromptBundle:
    """Discover skills, build the local tool registry, and render the SYSTEM
    prompt for a session — the offline half of `_run_session`'s setup."""
    layout_incomplete = not screen_layout.is_learned()
    builtin_skills = screen_layout.prune_builtin_skills(skill.discover_builtin_skills())
    user_skills = skill.discover_user_skills()
    local_registry = builtin_tool.build_registry(user_skills)
    local_schemas = builtin_tool.schemas(local_registry)
    system_prompt = prompt.render_system_prompts(
        local_tool_schemas=local_schemas,
        memory_ctx=memory.load_persistent(),
        builtin_skills_ctx=skill.render_builtin(builtin_skills),
        user_skills_ctx=skill.render_section(user_skills),
        provider_id=provider_id,
    )
    return PromptBundle(
        system_prompt=system_prompt,
        local_registry=local_registry,
        local_schemas=local_schemas,
        layout_incomplete=layout_incomplete,
        builtin_skill_count=len(builtin_skills),
        user_skill_count=len(user_skills),
    )


def build_initial_messages(triggers: list[Trigger], system_prompt: str) -> list[Message]:
    """The message array a session starts with: cached SYSTEM prompt, the
    wake-trigger user message (date anchor + fired-job context), and the three
    pre-allocated compaction slots (summary / memory / skills)."""
    return [
        SystemMessage(content=system_prompt),
        UserMessage(content=_format_triggers(
            triggers, cron_ctx=jobs.format_fired(triggers),
        )),
        compact.new_summary_placeholder(),
        compact.new_memory_placeholder(),
        compact.new_skills_placeholder(),
    ]


def apply_request_tails(
    messages: list[Message], session: Session, *, layout_incomplete: bool
) -> list[Message]:
    """Pin the per-turn tail slots to the request — scratchpad, plan, and
    (while first-run setup is pending) the layout reminder — the LAST things the
    model sees. Shared by `_loop` and the `prompt` dump so a turn-0 dump matches
    the wire the engine sends. Does not mutate `messages`."""
    out = scratchpad.inject_tail(messages, session.scratchpad)
    out = plan.inject_tail(out, session.plan)
    if layout_incomplete:
        out = screen_layout.inject_tail(out)
    return out


async def _run_session(
    triggers: list[Trigger],
    *,
    model_ref: str,
    session: Session,
) -> None:
    """One session attempt. Fresh sid / Trace / RawLog / MCP / Provider
    per call. Writes outcome to `session.sentinel_*`; never raises."""
    from physiclaw.config import parse_model_ref
    provider_id, model_id = parse_model_ref(model_ref)

    sid = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    provider: Provider | None = None
    tr: Trace | None = None
    rlog: RawLog | None = None
    try:
        # Open inside the try so the finally block's close() runs even
        # if construction fails midway (disk full, perms, etc.).
        tr = Trace(sid)
        rlog = RawLog(sid)
        tr.write({
            "event": "wake", "session": sid, "model_ref": model_ref,
            "triggers": [
                {"source": t.source, "description": t.description} for t in triggers
            ],
        })
        log.info(
            "wake session=%s model=%s triggers=%s",
            sid, model_ref, [t.source or "?" for t in triggers],
        )
        mcp = await get_mcp()
        mcp_tools = await list_tools_cached()
        # Built-in skills are inlined full-text into SYSTEM; user skills are
        # indexed and loaded on demand via the Skill tool — so only user
        # skills go into the local registry. The first-run screen-layout skill
        # is dropped once the layout is learned (dead weight after setup).
        # `build_prompt_bundle` is the offline half, shared with the CLI dump.
        bundle = build_prompt_bundle(provider_id)
        local_registry = bundle.local_registry
        local_schemas = bundle.local_schemas
        layout_incomplete = bundle.layout_incomplete
        system_prompt = bundle.system_prompt
        # Full merged list goes to provider.chat(tools=) for invocation;
        # the inline `## Tooling` card pulls MCP names from AST so it
        # stays complete even offline. Each source has one consumer.
        tool_schemas = list(mcp_tools) + local_schemas
        schema_by_name = {s["name"]: s for s in tool_schemas}
        tr.write({
            "event": "tools_loaded",
            "mcp": [s["name"] for s in mcp_tools],
            "local": sorted(local_registry.keys()),
        })
        log.info(
            "tools loaded: %d MCP + %d local + %d built-in + %d user skills",
            len(mcp_tools), len(local_registry),
            bundle.builtin_skill_count, bundle.user_skill_count,
        )

        messages: list[Message] = build_initial_messages(triggers, system_prompt)

        provider = make_provider(provider_id, model_id)
        prompt_hash = prompt.prefix_hash(system_prompt)
        rlog.write_session_start(
            provider=provider_id,
            model=provider.model,
            prompt_hash=prompt_hash,
            tools=tool_schemas,
        )
        await _loop(
            mcp=mcp,
            provider=provider,
            messages=messages,
            tool_schemas=tool_schemas,
            schema_by_name=schema_by_name,
            local_registry=local_registry,
            session=session,
            prompt_hash=prompt_hash,
            tr=tr,
            rlog=rlog,
            layout_incomplete=layout_incomplete,
        )

        log.info(
            "session done: status=%s recap=%r",
            session.sentinel_status, session.sentinel_recap,
        )
        if session.sentinel_status == WAIT and not session.sentinel_turn_created_job:
            log.warning("WAIT with no create_job — auto-scheduling %d-min follow-up", WAIT_DEFAULT_MINUTES)
            _auto_schedule_wait_check(tr)

        tr.write({
            "event": "done",
            "sentinel": session.sentinel_status,
            "recap": session.sentinel_recap,
        })
    except asyncio.CancelledError:
        raise
    except Exception as e:
        # Crashes count as STUCK so the retry loop gives it another shot.
        log.exception("engine session crashed")
        if tr is not None:
            tr.write({"event": "crashed"})
        session.sentinel_status = STUCK
        session.sentinel_recap = f"session crashed: {e}"
    finally:
        if provider is not None:
            await provider.aclose()
        if tr is not None:
            tr.close()
        if rlog is not None:
            rlog.close()


# ---------- core loop ----------


async def _loop(
    *,
    mcp: McpClient,
    provider: Provider,
    messages: list[Message],
    tool_schemas: list[dict],
    schema_by_name: dict[str, dict],
    local_registry: dict[str, LocalTool],
    session: Session,
    prompt_hash: str,
    tr: Trace,
    rlog: RawLog,
    layout_incomplete: bool = False,
) -> None:
    tr.write({"event": "prefix_pinned", "hash": prompt_hash})

    for turn in range(MAX_TURNS):
        # Plan + scratchpad are just-in-time tails — keeps `messages[]`
        # cache-stable across writes. Plan goes last so the model reads
        # "what to do next" last — except during first-run screen-layout
        # setup, when the setup reminder rides one step further out so the
        # blocker is the very last thing the model sees (absent once learned).
        session.plan.tick_turn()
        # First-run reminder is empty once learned, and a session that started
        # learned stays learned — so skip the per-turn disk read entirely unless
        # setup was still pending at session start.
        request_messages = apply_request_tails(
            messages, session, layout_incomplete=layout_incomplete,
        )
        # Cache markers + the actual wire format are the provider's
        # business now; engine logs the wire form for debugging by asking
        # the provider to serialize once.
        wire_for_log = provider.serialize_history(request_messages)
        tr.write({"event": "request", "turn": turn, "message_count": len(request_messages)})
        rlog.write_request(turn, wire_for_log)
        log.info("turn %d: %d messages → provider", turn + 1, len(request_messages))

        t0 = time.perf_counter()
        try:
            asst = await _chat_with_retry(provider, request_messages, tool_schemas)
        except Exception as e:
            log.exception("provider exhausted retries")
            tr.write({"event": "provider_failed", "turn": turn, "error": str(e)})
            session.sentinel_status = STUCK
            session.sentinel_recap = f"provider error: {e}"
            return

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        rlog.write_response(turn, asst.raw, elapsed_ms=elapsed_ms)
        tr.write({
            "event": "response",
            "turn": turn,
            "finish_reason": asst.finish_reason,
            "content_len": len(asst.content or ""),
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in asst.tool_calls
            ],
        })
        cache_summary = _log_usage(turn, asst, tr)
        called = asst.tool_names()
        # Provider round-trip time first, then token/cache — all provider-call
        # metrics grouped in the trailing segment.
        metrics = f"{elapsed_ms / 1000:.1f}s"
        if cache_summary:
            metrics += f", {cache_summary}"
        log.info(
            "turn %d: finish=%s calls=%s — %s",
            turn + 1, asst.finish_reason, called or None, metrics,
        )

        # Principle 2: AssistantMessage already has provider-specific
        # fields (reasoning_content, thinking blocks) stripped at parse
        # time inside the provider — append directly.
        messages.append(asst)

        # Principle 3: route on finish_reason.
        if asst.finish_reason == FinishReason.CONTENT_FILTER:
            log.error("content_filter — stopping session")
            session.sentinel_status = FAIL
            session.sentinel_recap = "content filter blocked response"
            return

        # Shape checks may reject the turn. The rejected assistant message
        # must come back out of history: leaving orphan tool_calls behind
        # breaks providers' "tool_calls → matching tool messages" rule and
        # anchors the model on its own failure on retry.
        if not asst.tool_calls:
            messages.pop()
            messages.append(UserMessage(content=(
                "Your last turn had no tool_calls. Every turn must "
                "either call tools or call end_session(status, recap) "
                "to close. Reply again as a tool call — and include "
                "note(summary=...) alongside."
            )))
            continue

        # Turn shape: exactly [note, one-other]. `note` keeps a permanent
        # trace even after image tool_results are compacted away; the
        # one-other cap forces one action per turn.
        if len(called) != 2 or called.count("note") != 1:
            log.warning("turn %d: bad turn shape tool_calls=%s — injecting corrective", turn, called)
            tr.write({"event": "bad_turn_shape", "turn": turn, "tool_calls": called})
            messages.pop()
            messages.append(UserMessage(content=_corrective_for_bad_shape(called)))
            continue

        # Principle 6: each tool_call gets exactly one ToolResult, in order,
        # in the very next messages. Also mark truncation: if finish=length,
        # the final tool_call's arguments may be cut off — the validator
        # catches it and pairs an error result, same path as any bad args.
        if asst.finish_reason == FinishReason.LENGTH:
            log.warning("turn %d: finish=length; last tool_call args may be truncated", turn)
            tr.write({"event": "finish_length_warning", "turn": turn})

        for call in asst.tool_calls:
            result = await _dispatch(
                call=call,
                schema_by_name=schema_by_name,
                mcp=mcp,
                local_registry=local_registry,
                session=session,
                tr=tr,
                turn=turn,
            )
            messages.append(result)

        compact.drop_stale_screens(messages)
        compact.collapse_old_turns(
            messages,
            first_at=provider.COLLAPSE_FIRST_AT_TURN,
            interval=provider.COLLAPSE_INTERVAL_TURNS,
            keep=provider.KEEP_RECENT_TURNS,
        )

        if session.sentinel_status:
            return

    log.warning("engine hit max turns (%d)", MAX_TURNS)
    session.sentinel_status = STUCK
    session.sentinel_recap = f"max turns ({MAX_TURNS}) reached"


# ---------- dispatch ----------


async def _dispatch(
    *,
    call: ToolCall,
    schema_by_name: dict[str, dict],
    mcp: McpClient,
    local_registry: dict[str, LocalTool],
    session: Session,
    tr: Trace,
    turn: int,
) -> ToolResultMessage:
    """Validate, then route to local handler or MCP. Always returns a
    ToolResultMessage — never raises (principle 5 + principle 6 require
    that every ToolCall is paired with a ToolResult even on failure)."""
    log.info("  → %s(%s)", call.name, format_call_args(call.name, call.arguments))

    schema = schema_by_name.get(call.name)
    if schema is None:
        tr.write({"event": "tool_unknown", "turn": turn, "name": call.name, "id": call.id})
        log.warning("  ✗ %s: unknown tool", call.name)
        return ToolResultMessage(
            tool_call_id=call.id,
            content=f"unknown tool: {call.name!r}",
            is_error=True,
        )

    # Principle 4: validate arguments before executing.
    try:
        validate_arguments(call.arguments, schema.get("input_schema") or {})
    except ValidationError as e:
        tr.write({
            "event": "tool_invalid_args", "turn": turn,
            "name": call.name, "id": call.id,
            "arguments": call.arguments, "error": str(e),
        })
        log.warning("  ✗ %s: invalid args — %s", call.name, e)
        return ToolResultMessage(
            tool_call_id=call.id,
            content=f"invalid arguments for {call.name}: {e}",
            is_error=True,
        )

    local = local_registry.get(call.name)
    try:
        if local is not None:
            text = await local.handler(session, call.arguments)
            tr.write({
                "event": "tool_result", "turn": turn,
                "name": call.name, "id": call.id,
                "arguments": call.arguments, "text": text,
            })
            log.info("  ✓ %s → %s", call.name, format_call_result(call.name, text))
            return ToolResultMessage(tool_call_id=call.id, content=text)

        blocks = await mcp.call_tool(call.name, call.arguments)
        content = mcp_blocks_to_content_blocks(blocks)
        tr.write({
            "event": "tool_result", "turn": turn,
            "name": call.name, "id": call.id,
            "arguments": call.arguments, "blocks": blocks,
        })
        log.info("  ✓ %s → %s", call.name, brief_content(content))
        return ToolResultMessage(tool_call_id=call.id, content=content)

    except Exception as e:
        log.error("  ✗ %s failed: %s", call.name, e)
        tr.write({"event": "tool_error", "turn": turn, "name": call.name, "error": str(e)})
        return ToolResultMessage(
            tool_call_id=call.id,
            content=f"{call.name} failed: {e}",
            is_error=True,
        )


# ---------- helpers ----------


async def _chat_with_retry(
    provider: Provider, messages: list[Message], tools: list[dict],
) -> AssistantMessage:
    """Retry transient errors only (principle 3: permanent 4xx fails fast)."""
    last_err: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return await provider.chat(messages, tools)
        except ProviderTransientError as e:
            last_err = e
            if attempt < MAX_ATTEMPTS:
                log.warning("provider transient (attempt %d/%d): %s", attempt, MAX_ATTEMPTS, e)
                await asyncio.sleep(RETRY_BACKOFF)
    raise RuntimeError(f"provider failed after {MAX_ATTEMPTS} attempts: {last_err}")


def _log_usage(turn: int, asst: AssistantMessage, tr: Trace) -> str:
    """Read normalized `AssistantMessage.usage`, emit a trace event,
    return a short `token: X.Xk, cache: YY%` summary for the per-turn
    log line. Empty string when the provider didn't report usage (zero
    prompt_tokens) — readers treat that as 'no data'.

    Each provider populates `Usage` from its native usage block at parse
    time, so this code is provider-agnostic."""
    u = asst.usage
    total = u.prompt_tokens
    new = max(0, total - u.cached_tokens - u.cache_creation_tokens)
    tr.write({
        "event": "cache",
        "turn": turn,
        "hit": u.cached_tokens,
        "create": u.cache_creation_tokens,
        "new": new,
        "total": total,
    })
    if not total:
        return ""
    return f"token: {total / 1000:.1f}k, cache: {100 * u.cached_tokens / total:.0f}%"


def _corrective_for_bad_shape(called: list[str]) -> str:
    """Build a turn-specific corrective for a [note, one-other] shape
    violation.

    Naming the rejected tool nudges the model to retry the same action
    on the next turn instead of switching to whatever fits the shape.
    Without this, models often interpret the generic "re-issue as
    [note, one-other]" as licence to pick a different second tool —
    silently dropping the rejected action entirely.
    """
    # Each branch tests (n_notes, len(others)) explicitly so the
    # coverage matrix is readable top-to-bottom: no implicit fallthrough,
    # no overlap. Caller filters out (0,0) and (1,1) — every other
    # non-empty shape lands in exactly one branch below.
    n_notes = called.count("note")
    others = [c for c in called if c != "note"]
    n_others = len(others)

    if n_notes == 0 and n_others == 1:
        # Most common: agent forgot note alongside its action.
        return (
            f"Your last turn called `{others[0]}` without `note`. Every "
            f"turn = exactly `[note, one-other]`. Re-issue as "
            f"`[note(summary=...), {others[0]}(...)]` with the same arguments."
        )

    if n_notes == 0 and n_others >= 2:
        return (
            f"Your last turn called {called!r} without `note` and with "
            f"too many action tools. Every turn = exactly `[note, one-other]`. "
            f"Keep `[note(summary=...), {others[0]}(...)]` for this turn; "
            f"split {others[1:]!r} into later turns."
        )

    if n_notes == 1 and n_others == 0:
        # Naming `peek()` as the default breaks the anchoring loop when
        # the model has no concrete action in mind (e.g. ambient warm-
        # start triggers): without a suggestion, models pick `note` again
        # because every other tool feels unjustified.
        return (
            "Your last turn called `note` alone with no action tool. "
            "Every turn = exactly `[note, one-other]`. If unsure what to "
            "do next, `peek()` is the safe default — re-issue as "
            "`[note(summary=...), peek()]`."
        )

    if n_notes == 1 and n_others >= 2:
        return (
            f"Your last turn called {called!r} — `note` plus {n_others} "
            f"action tools. Every turn = exactly `[note, one-other]`. "
            f"Keep `[note(summary=...), {others[0]}(...)]` for this turn; "
            f"split {others[1:]!r} into later turns."
        )

    if n_notes >= 2:
        return (
            f"Your last turn called `note` {n_notes} times. Every turn = "
            f"exactly `[note, one-other]`. Re-issue with `note(summary=...)` "
            f"once plus one action tool."
        )

    raise AssertionError(f"unreachable: called={called!r}")


def _format_triggers(triggers: list[Trigger], *, cron_ctx: str = "") -> str:
    # Leading `Now:` line is the absolute-date anchor — memory logs use
    # relative dates, and the model burns turns triangulating today without
    # it. Each trigger then uses a uniform `[stamp] source: body` envelope.
    # cron_ctx (jobs firing now) rides in this user message, not the system,
    # so the system stays byte-stable across wakes for cross-session caching.
    now = dt.datetime.now()
    stamp = now.strftime("%Y-%m-%d %a %H:%M")
    lines = [
        f"Now: {stamp}",
        "",
        "[Current wake — act on this]",
    ]
    for t in triggers:
        source = t.source or "manual"
        lines.append(f"[{stamp}] {source}: {t.description}")
    text = "\n".join(lines)
    if cron_ctx:
        text += "\n\n" + cron_ctx
    return text


def _auto_schedule_wait_check(tr: Trace) -> None:
    """Schedule the singleton auto-WAIT-check job to fire in
    WAIT_DEFAULT_MINUTES. Reuses one canonical job id across sessions
    (see `jobs.upsert_auto_wait_check`) so jobs.md doesn't grow one
    entry per WAIT close.
    """
    at = dt.datetime.now() + dt.timedelta(minutes=WAIT_DEFAULT_MINUTES)
    try:
        jobs.upsert_auto_wait_check(at)
        tr.write({
            "event": "wait_auto_scheduled",
            "job_id": jobs.AUTO_WAIT_JOB_ID,
            "at": at.isoformat(timespec="minutes"),
        })
    except Exception as e:
        log.exception("failed to auto-schedule WAIT follow-up")
        tr.write({"event": "wait_auto_schedule_failed", "error": str(e)})
