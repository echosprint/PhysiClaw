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

Layering: this module owns the protocol invariants above plus the
*mechanics* of every interception (pop-and-corrective, blocked
ToolResults, trace events). The *judgment* — when to reject a turn, when
to refuse a call, what to warn about — lives in `policy.py` as policy
objects on `EngineRun.policies`; request assembly (prompt bundle, initial
messages, tails) lives in `assemble.py`, shared with the CLI dump.
"""

import asyncio
import datetime as dt
import enum
import logging
import time
from dataclasses import dataclass

from physiclaw import verdict
from physiclaw.agent.engine import assemble, compact, curate, jobs, prompt, trajectory
from physiclaw.agent.engine.builtin_tool import LocalTool
from physiclaw.agent.engine.mcp_tool import McpClient, get_mcp, list_tools_cached
from physiclaw.agent.engine.dto import (
    AssistantMessage,
    ContentBlock,
    FinishReason,
    Message,
    TextBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from physiclaw.agent.engine.policy import DispatchGuard, Policies, default_policies
from physiclaw.agent.engine.session import Session
from physiclaw.agent.provider import (
    Provider,
    ProviderTransientError,
    make_provider,
    mcp_blocks_to_content_blocks,
)
from physiclaw.agent.engine.trace import (
    RawLog,
    Trace,
    brief_content,
    format_call_args,
    format_call_result,
    new_sid,
)
from physiclaw.agent.engine.validator import ValidationError, validate_arguments
from physiclaw.agent.runtime.hook import Trigger
from physiclaw.agent.runtime.sentinel import FAIL, STUCK, WAIT
from physiclaw.config import CONFIG

log = logging.getLogger(__name__)

# A model that can't hold the [note, one-other] shape burns a full
# provider round-trip per corrective; after this many CONSECUTIVE
# failures, end STUCK instead of paying up to max_turns of them.
CORRECTIVE_LIMIT = 5


@dataclass(frozen=True, slots=True)
class Settings:
    """Engine knobs, read from CONFIG at `run()` time — not import time —
    so tests construct them explicitly and a config write between wakes
    takes effect without a process restart.

    `max_turns` is a runaway-loop backstop, not a context-safety limit.
    Prompt tokens grow ~624·t + 13k empirically (R²=0.97); at 1M context
    (Qwen3.6-plus) the hard wall is ~1,580 turns, so 300 leaves ample
    headroom. `max_session_attempts` is session-level STUCK retries;
    `provider_retry_attempts` is per-call transient-error retries — two
    different knobs, deliberately not shared.
    """

    max_turns: int
    max_session_attempts: int
    provider_retry_attempts: int
    retry_backoff_seconds: float
    wait_default_minutes: int

    @classmethod
    def from_config(cls) -> "Settings":
        e = CONFIG.engine
        return cls(
            max_turns=e.max_turns,
            max_session_attempts=e.max_attempts,
            provider_retry_attempts=e.provider_retry_attempts,
            retry_backoff_seconds=e.retry_backoff_seconds,
            wait_default_minutes=e.wait_default_minutes,
        )


@dataclass(slots=True)
class EngineRun:
    """One session's immutable wiring: provider, MCP client, tool surface,
    logging sinks, settings, and the policy set. Threaded through the loop
    instead of a parameter list so adding a dependency is a one-line change."""

    provider: Provider
    mcp: McpClient
    tool_schemas: list[dict]
    schema_by_name: dict[str, dict]
    local_registry: dict[str, LocalTool]
    tr: Trace
    rlog: RawLog
    settings: Settings
    policies: Policies
    layout_incomplete: bool = False


async def run(triggers: list[Trigger], *, model_ref: str) -> None:
    """Run an engine session for `triggers`, retrying on STUCK.

    STUCK happens when the loop hit max_turns without a clean close, the
    provider exhausted its retries, or the session crashed. Up to
    `max_session_attempts` fresh attempts run before we accept the STUCK
    outcome. DONE / FAIL / IDLE / WAIT are final on first occurrence — no
    retry.

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
    settings = Settings.from_config()
    setup_restart_used = False
    attempt = 0
    while attempt < settings.max_session_attempts:
        attempt += 1
        session = Session()
        await _run_session(
            triggers,
            model_ref=model_ref,
            session=session,
            settings=settings,
        )
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
        if attempt < settings.max_session_attempts:
            log.warning(
                "session STUCK (attempt %d/%d): %r — retrying",
                attempt,
                settings.max_session_attempts,
                session.sentinel_recap,
            )


async def _run_session(
    triggers: list[Trigger],
    *,
    model_ref: str,
    session: Session,
    settings: Settings | None = None,
) -> None:
    """One session attempt. Fresh sid / Trace / RawLog / MCP / Provider /
    policy set per call. Writes outcome to `session.sentinel_*`; never
    raises."""
    from physiclaw.config import parse_model_ref

    provider_id, model_id = parse_model_ref(model_ref)
    settings = settings or Settings.from_config()

    sid = new_sid()
    provider: Provider | None = None
    tr: Trace | None = None
    rlog: RawLog | None = None
    try:
        # Open inside the try so the finally block's close() runs even
        # if construction fails midway (disk full, perms, etc.).
        tr = Trace(sid)
        rlog = RawLog(sid)
        tr.write(
            {
                "event": "wake",
                "session": sid,
                "model_ref": model_ref,
                "triggers": [
                    {"source": t.source, "description": t.description} for t in triggers
                ],
            }
        )
        log.info(
            "wake session=%s model=%s triggers=%s",
            sid,
            model_ref,
            [t.source or "?" for t in triggers],
        )
        mcp = await get_mcp()
        mcp_tools = await list_tools_cached()
        # Built-in skills are inlined full-text into SYSTEM; user skills are
        # indexed and loaded on demand via the Skill tool — so only user
        # skills go into the local registry. The first-run screen-layout skill
        # is dropped once the layout is learned (dead weight after setup).
        # `assemble.build_prompt_bundle` is the offline half, shared with the
        # CLI dump.
        bundle = assemble.build_prompt_bundle(provider_id)
        # Full merged list goes to provider.chat(tools=) for invocation;
        # the inline `## Tooling` card pulls MCP names from AST so it
        # stays complete even offline. Each source has one consumer.
        tool_schemas = list(mcp_tools) + bundle.local_schemas
        schema_by_name = {s["name"]: s for s in tool_schemas}
        tr.write(
            {
                "event": "tools_loaded",
                "mcp": [s["name"] for s in mcp_tools],
                "local": sorted(bundle.local_registry.keys()),
            }
        )
        log.info(
            "tools loaded: %d MCP + %d local + %d built-in + %d user skills "
            "+ %d pitfalls",
            len(mcp_tools),
            len(bundle.local_registry),
            bundle.builtin_skill_count,
            bundle.user_skill_count,
            bundle.pitfall_count,
        )

        messages: list[Message] = assemble.build_initial_messages(
            triggers,
            bundle.system_prompt,
        )

        provider = make_provider(provider_id, model_id)
        prompt_hash = prompt.prefix_hash(bundle.system_prompt)
        rlog.write_session_start(
            provider=provider_id,
            model=provider.model,
            prompt_hash=prompt_hash,
            tools=tool_schemas,
        )
        engine_run = EngineRun(
            provider=provider,
            mcp=mcp,
            tool_schemas=tool_schemas,
            schema_by_name=schema_by_name,
            local_registry=bundle.local_registry,
            tr=tr,
            rlog=rlog,
            settings=settings,
            policies=default_policies(layout_incomplete=bundle.layout_incomplete),
            layout_incomplete=bundle.layout_incomplete,
        )
        tr.write({"event": "prefix_pinned", "hash": prompt_hash})
        await _loop(engine_run, session, messages)

        log.info(
            "session done: status=%s recap=%r",
            session.sentinel_status,
            session.sentinel_recap,
        )
        # Post-session pitfalls curation: a separate LLM pass (still-open
        # provider) that consolidates the list + enforces the cap when the agent
        # added traps this session. Fail-open — never affects the outcome.
        if CONFIG.pitfalls.curate_enabled and session.added_pitfalls:
            await curate.curate(provider, tr=tr)
        if session.sentinel_status == WAIT and not session.sentinel_turn_created_job:
            log.warning(
                "WAIT with no create_job — auto-scheduling %d-min follow-up",
                settings.wait_default_minutes,
            )
            _auto_schedule_wait_check(tr, minutes=settings.wait_default_minutes)

        tr.write(
            {
                "event": "done",
                "sentinel": session.sentinel_status,
                "recap": session.sentinel_recap,
            }
        )
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


class _Turn(enum.Enum):
    """What `_loop` does after a turn's finish_reason + shape checks."""

    PROCEED = enum.auto()  # well-formed — go run the turn's tools
    RETRY = enum.auto()  # rejected, corrective injected — re-loop this index
    ABORT = enum.auto()  # terminal sentinel set — return from `_loop`


@dataclass
class _LoopState:
    """Loop-local counters that persist across turns within one `_loop` run.

    consecutive_correctives: malformed turns in a row — STUCK past the cap.
    (Per-gate one-shot state lives on the gate objects in `run.policies`.)
    """

    consecutive_correctives: int = 0


async def _loop(run: EngineRun, session: Session, messages: list[Message]) -> None:
    """The driver (principle 7): model → tool_calls → dispatch → results → model.

    Each turn runs a fixed pipeline of phase helpers; the control flow between
    them stays here so it's auditable in one place:
      1. `_prepare_request`    — advance clocks, pin tails, log the request.
      2. `_call_provider`      — chat with retry; None ⇒ STUCK, return.
      3. `_enforce_shape`      — finish_reason + [note, one-other]; may RETRY/ABORT.
      4. gates observe_turn    — accumulate per-turn evidence (memory cues).
      5. `_apply_turn_gates`   — the policy gates in declared order; may RETRY.
      6. `_dispatch_turn`      — run the tool_calls, append their results.
      7. `_finalize_turn`      — snapshot trajectory, compact, reset gates.
    """
    state = _LoopState()
    for turn in range(run.settings.max_turns):
        request_messages, compaction_imminent = _prepare_request(
            run,
            session,
            messages,
            turn,
        )

        asst = await _call_provider(run, session, request_messages, turn)
        if asst is None:
            return  # provider exhausted its retries — STUCK already set

        # Principle 2: reasoning_content / thinking blocks were stripped at
        # parse time inside the provider — append the assistant turn directly.
        messages.append(asst)
        called = asst.tool_names()

        shape = _enforce_shape(
            messages, asst, called, state, session=session, turn=turn, tr=run.tr
        )
        if shape is _Turn.ABORT:
            return
        if shape is _Turn.RETRY:
            continue

        for gate in run.policies.turn_gates:
            gate.observe_turn(session, asst)

        if _apply_turn_gates(
            run,
            session,
            messages,
            asst,
            called,
            turn=turn,
            compaction_imminent=compaction_imminent,
        ):
            continue

        await _dispatch_turn(run, session, messages, asst, turn)
        _finalize_turn(run, session, messages, asst, called, turn)

        if session.sentinel_status:
            return
    else:
        log.warning("engine hit max turns (%d)", run.settings.max_turns)
        session.sentinel_status = STUCK
        session.sentinel_recap = f"max turns ({run.settings.max_turns}) reached"


# ---------- per-turn phases ----------


def _prepare_request(
    run: EngineRun,
    session: Session,
    messages: list[Message],
    turn: int,
) -> tuple[list[Message], bool]:
    """Assemble and log this turn's request; return the wire-ready message list
    and whether a collapse is imminent this turn.

    Plan + scratchpad are just-in-time tails — keeps `messages[]` cache-stable
    across writes. Plan goes last so the model reads "what to do next" last —
    except during first-run screen-layout setup, when the setup reminder rides
    one step further out so the blocker is the very last thing the model sees.
    """
    session.plan.tick_turn()
    # Stuck-guard counters key on the in_progress step: a step change (via last
    # turn's update_progress) wipes the same-target history.
    session.guard.observe_step(session.plan.current_step())
    # Pre-compression checkpoint: when this turn's completion will collapse old
    # turns to their note-summary lines, this is the model's last look at them —
    # the request carries a ⚠ tail and the turn's note must bank state into the
    # scratchpad (enforced by policy.CompactionCheckpoint).
    compaction_imminent = compact.collapse_pending(
        messages,
        first_at=run.provider.COLLAPSE_FIRST_AT_TURN,
        interval=run.provider.COLLAPSE_INTERVAL_TURNS,
        keep=run.provider.KEEP_RECENT_TURNS,
    )
    # First-run reminder is empty once learned, and a session that started
    # learned stays learned — so the per-turn disk read is skipped entirely
    # unless setup was still pending at session start (`layout_incomplete`).
    request_messages = assemble.apply_request_tails(
        messages,
        session,
        layout_incomplete=run.layout_incomplete,
        compaction_keep=(
            run.provider.KEEP_RECENT_TURNS if compaction_imminent else None
        ),
    )
    # Cache markers + the actual wire format are the provider's business now;
    # log the wire form for debugging by asking the provider to serialize once.
    wire_for_log = run.provider.serialize_history(request_messages)
    run.tr.write(
        {"event": "request", "turn": turn, "message_count": len(request_messages)}
    )
    run.rlog.write_request(turn, wire_for_log)
    log.info("turn %d: %d messages → provider", turn + 1, len(request_messages))
    return request_messages, compaction_imminent


async def _call_provider(
    run: EngineRun,
    session: Session,
    request_messages: list[Message],
    turn: int,
) -> AssistantMessage | None:
    """Call the provider (transient-retry), log the response, and return the
    AssistantMessage. When the provider exhausts its retries: mark the session
    STUCK, trace it, and return None — the caller returns from the loop."""
    t0 = time.perf_counter()
    try:
        asst = await _chat_with_retry(
            run.provider,
            request_messages,
            run.tool_schemas,
            attempts=run.settings.provider_retry_attempts,
            backoff=run.settings.retry_backoff_seconds,
        )
    except Exception as e:
        log.exception("provider exhausted retries")
        run.tr.write({"event": "provider_failed", "turn": turn, "error": str(e)})
        session.sentinel_status = STUCK
        session.sentinel_recap = f"provider error: {e}"
        return None

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    run.rlog.write_response(turn, asst.raw, elapsed_ms=elapsed_ms)
    run.tr.write(
        {
            "event": "response",
            "turn": turn,
            "finish_reason": asst.finish_reason,
            "content_len": len(asst.content or ""),
            # Provider round-trip time — the session summary sums these
            # into provider_time_ms.
            "elapsed_ms": elapsed_ms,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in asst.tool_calls
            ],
        }
    )
    cache_summary = _log_usage(turn, asst, run.tr)
    # Provider round-trip time first, then token/cache — all provider-call
    # metrics grouped in the trailing segment.
    metrics = f"{elapsed_ms / 1000:.1f}s"
    if cache_summary:
        metrics += f", {cache_summary}"
    log.info(
        "turn %d: finish=%s calls=%s — %s",
        turn + 1,
        asst.finish_reason,
        asst.tool_names() or None,
        metrics,
    )
    return asst


def _enforce_shape(
    messages: list[Message],
    asst: AssistantMessage,
    called: list[str],
    state: _LoopState,
    *,
    session: Session,
    turn: int,
    tr: Trace,
) -> _Turn:
    """Route finish_reason and enforce the [note, one-other] turn shape.

    - content_filter → FAIL, ABORT.
    - no tool_calls, or a shape other than exactly [note, one-other] → pop the
      rejected assistant turn (leaving orphan tool_calls breaks providers'
      "tool_calls → matching tool messages" rule and anchors the model on its
      own failure), inject a corrective, RETRY. After CORRECTIVE_LIMIT in a
      row, give up STUCK rather than pay correctives up to max_turns.
    - well-formed → reset the corrective streak, PROCEED.
    """
    # Principle 3: route on finish_reason.
    if asst.finish_reason == FinishReason.CONTENT_FILTER:
        log.error("content_filter — stopping session")
        session.sentinel_status = FAIL
        session.sentinel_recap = "content filter blocked response"
        return _Turn.ABORT

    if not asst.tool_calls:
        # Trace it like the other shape violation below, so the session
        # summary's corrective count stays honest.
        tr.write({"event": "bad_turn_shape", "turn": turn, "tool_calls": []})
        return _reject_shape(
            messages,
            state,
            session,
            (
                "Your last turn had no tool_calls. Every turn must "
                "either call tools or call end_session(status, recap) "
                "to close. Reply again as a tool call — and include "
                "note(summary=...) alongside."
            ),
        )

    # Turn shape: exactly [note, one-other]. `note` keeps a permanent trace
    # even after image tool_results are compacted away; the one-other cap
    # forces one action per turn.
    if len(called) != 2 or called.count("note") != 1:
        log.warning(
            "turn %d: bad turn shape tool_calls=%s — injecting corrective",
            turn + 1,
            called,
        )
        tr.write({"event": "bad_turn_shape", "turn": turn, "tool_calls": called})
        return _reject_shape(
            messages, state, session, _corrective_for_bad_shape(called)
        )

    state.consecutive_correctives = 0
    return _Turn.PROCEED


def _reject_shape(
    messages: list[Message],
    state: _LoopState,
    session: Session,
    corrective: str,
) -> _Turn:
    """Count one malformed turn and RETRY it: past CORRECTIVE_LIMIT in a row,
    give up STUCK; otherwise pop the rejected assistant turn (leaving orphan
    tool_calls breaks providers' "tool_calls → matching tool messages" rule and
    anchors the model on its own failure) and inject `corrective`."""
    state.consecutive_correctives += 1
    if state.consecutive_correctives >= CORRECTIVE_LIMIT:
        return _give_up_on_shape(session, state)
    messages.pop()
    messages.append(UserMessage(content=corrective))
    return _Turn.RETRY


def _give_up_on_shape(session: Session, state: _LoopState) -> _Turn:
    """Terminal STUCK when the model can't hold the turn shape after
    CORRECTIVE_LIMIT consecutive rejections. The offending turn is left in
    history (not popped) as the recap's evidence."""
    log.warning(
        "%d consecutive turn-shape correctives — giving up",
        state.consecutive_correctives,
    )
    session.sentinel_status = STUCK
    session.sentinel_recap = (
        f"{state.consecutive_correctives} consecutive malformed turns — "
        "model cannot hold the [note, one-other] shape"
    )
    return _Turn.ABORT


def _apply_turn_gates(
    run: EngineRun,
    session: Session,
    messages: list[Message],
    asst: AssistantMessage,
    called: list[str],
    *,
    turn: int,
    compaction_imminent: bool,
) -> bool:
    """Run the policy turn gates in declared order; the first Rejection wins.
    The gate owns the judgment and its one-shot state; this owns the shared
    mechanic — log, trace, pop the rejected assistant turn, inject the
    corrective. Returns True if a gate rejected the turn (the caller
    re-loops), False to proceed to dispatch."""
    for gate in run.policies.turn_gates:
        rej = gate.check(
            session,
            asst,
            called,
            turn=turn,
            compaction_imminent=compaction_imminent,
        )
        if rej is None:
            continue
        log.info("turn %d: %s", turn + 1, rej.log_msg)
        run.tr.write({"event": rej.event, "turn": turn, **rej.extra})
        messages.pop()
        messages.append(UserMessage(content=rej.corrective))
        return True
    return False


async def _dispatch_turn(
    run: EngineRun,
    session: Session,
    messages: list[Message],
    asst: AssistantMessage,
    turn: int,
) -> None:
    """Run the turn's tool_calls — principle 6: exactly one ToolResult per
    ToolCall, in order, in the very next messages. Flags a length-truncation
    warning first: if finish=length the final tool_call's arguments may be cut
    off — the validator catches it and pairs an error result, same path as any
    bad args."""
    if asst.finish_reason == FinishReason.LENGTH:
        log.warning(
            "turn %d: finish=length; last tool_call args may be truncated", turn + 1
        )
        run.tr.write({"event": "finish_length_warning", "turn": turn})

    for call in asst.tool_calls:
        result = await _dispatch(run, session, call, turn)
        messages.append(result)


def _finalize_turn(
    run: EngineRun,
    session: Session,
    messages: list[Message],
    asst: AssistantMessage,
    called: list[str],
    turn: int,
) -> None:
    """Post-dispatch bookkeeping. Snapshot this turn's plan/scratchpad so a hard
    close can reflect on the whole run's evolution, not just the final state —
    the plan on every update_progress and the scratchpad on every
    note(scratchpad=...) write (each explicit act is its own turn-tagged
    entry). Then compact (drop stale screens, collapse old turns) and tell the
    gates the turn completed so per-event one-shot state renews."""
    note_call = next((tc for tc in asst.tool_calls if tc.name == "note"), None)
    scratchpad_written = bool(
        note_call
        and isinstance(note_call.arguments, dict)
        and note_call.arguments.get("scratchpad") is not None
    )
    trajectory.record(
        session,
        turn,
        plan_updated="update_progress" in called,
        scratchpad_written=scratchpad_written,
    )
    compact.drop_stale_screens(messages)
    compact.collapse_old_turns(
        messages,
        first_at=run.provider.COLLAPSE_FIRST_AT_TURN,
        interval=run.provider.COLLAPSE_INTERVAL_TURNS,
        keep=run.provider.KEEP_RECENT_TURNS,
    )
    for gate in run.policies.turn_gates:
        gate.on_turn_complete()


# ---------- dispatch ----------


async def _dispatch(
    run: EngineRun,
    session: Session,
    call: ToolCall,
    turn: int,
) -> ToolResultMessage:
    """Guards, validate, execute, observe. Always returns a ToolResultMessage —
    never raises (principle 5 + principle 6 require that every ToolCall is
    paired with a ToolResult even on failure)."""
    log.info("  → %s(%s)", call.name, format_call_args(call.name, call.arguments))

    blocked = _run_guards(run, session, call, turn, run.policies.pre_validation_guards)
    if blocked is not None:
        return blocked

    schema = run.schema_by_name.get(call.name)
    if schema is None:
        run.tr.write(
            {"event": "tool_unknown", "turn": turn, "name": call.name, "id": call.id}
        )
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
        run.tr.write(
            {
                "event": "tool_invalid_args",
                "turn": turn,
                "name": call.name,
                "id": call.id,
                "arguments": call.arguments,
                "error": str(e),
            }
        )
        log.warning("  ✗ %s: invalid args — %s", call.name, e)
        return ToolResultMessage(
            tool_call_id=call.id,
            content=f"invalid arguments for {call.name}: {e}",
            is_error=True,
        )

    blocked = _run_guards(run, session, call, turn, run.policies.post_validation_guards)
    if blocked is not None:
        return blocked

    local = run.local_registry.get(call.name)
    try:
        if local is not None:
            text = await local.handler(session, call.arguments)
            run.tr.write(
                {
                    "event": "tool_result",
                    "turn": turn,
                    "name": call.name,
                    "id": call.id,
                    "arguments": call.arguments,
                    "text": text,
                }
            )
            log.info("  ✓ %s → %s", call.name, format_call_result(call.name, text))
            return ToolResultMessage(tool_call_id=call.id, content=text)

        blocks = await run.mcp.call_tool(call.name, call.arguments)
        content = mcp_blocks_to_content_blocks(blocks)
        changed = verdict.parse(_action_text(blocks))
        content = _observe_result(
            run,
            session,
            call,
            content,
            turn=turn,
            changed=changed,
            failed=False,
        )
        run.tr.write(
            {
                "event": "tool_result",
                "turn": turn,
                "name": call.name,
                "id": call.id,
                "arguments": call.arguments,
                "blocks": blocks,
            }
        )
        log.info("  ✓ %s → %s", call.name, brief_content(content))
        return ToolResultMessage(tool_call_id=call.id, content=content)

    except Exception as e:
        log.error("  ✗ %s failed: %s", call.name, e)
        run.tr.write(
            {"event": "tool_error", "turn": turn, "name": call.name, "error": str(e)}
        )
        content = _observe_result(
            run,
            session,
            call,
            f"{call.name} failed: {e}",
            turn=turn,
            changed=None,
            failed=True,
        )
        return ToolResultMessage(
            tool_call_id=call.id,
            content=content,
            is_error=True,
        )


def _run_guards(
    run: EngineRun,
    session: Session,
    call: ToolCall,
    turn: int,
    guards: tuple[DispatchGuard, ...],
) -> ToolResultMessage | None:
    """Run one phase's dispatch guards (pre- or post-validation, split once
    at Policies construction) in declared order; the first Block wins and
    becomes an error ToolResult."""
    for guard in guards:
        block = guard.check(session, call, turn=turn)
        if block is None:
            continue
        run.tr.write(
            {
                "event": block.event,
                "turn": turn,
                "name": call.name,
                "id": call.id,
                "arguments": call.arguments,
            }
        )
        log.warning("  ✗ %s: %s", call.name, block.log_msg)
        return ToolResultMessage(
            tool_call_id=call.id,
            content=block.content,
            is_error=True,
        )
    return None


def _observe_result(
    run: EngineRun,
    session: Session,
    call: ToolCall,
    content: str | list[ContentBlock],
    *,
    turn: int,
    changed: bool | None,
    failed: bool,
) -> str | list[ContentBlock]:
    """Run the result observers in declared order; each may append one
    advisory line to the tool-result content (traced under its event)."""
    for obs in run.policies.result_observers:
        advisory = obs.observe(session, call, changed=changed, failed=failed)
        if advisory is None:
            continue
        run.tr.write(
            {
                "event": advisory.event,
                "turn": turn,
                "name": call.name,
                "id": call.id,
            }
        )
        content = _append_text(content, advisory.text)
    return content


# ---------- helpers ----------


def _action_text(blocks: list[dict]) -> str:
    """The FIRST text block of a raw MCP tool result — the core-composed
    action text carrying the verdict marker, and the ONLY safe haystack
    for it. Later text blocks hold the OCR listing, i.e. whatever the
    phone displays — scanning those would let on-screen text forge a
    verdict (the agent IM'ing the user "…screen: no visible change…"
    about a blocker, then operating in that chat, is enough). Must run
    on the raw blocks: `mcp_blocks_to_content_blocks` fuses all text
    blocks into one, erasing the boundary."""
    for b in blocks:
        if b.get("type") == "text":
            return b.get("text") or ""
    return ""


def _append_text(
    content: str | list[ContentBlock], extra: str
) -> str | list[ContentBlock]:
    """Append a line of text to a tool-result content, whatever its shape."""
    if isinstance(content, str):
        return f"{content}\n{extra}"
    return [*content, TextBlock(text=extra)]


async def _chat_with_retry(
    provider: Provider,
    messages: list[Message],
    tools: list[dict],
    *,
    attempts: int,
    backoff: float,
) -> AssistantMessage:
    """Retry transient errors only (principle 3: permanent 4xx fails fast)."""
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await provider.chat(messages, tools)
        except ProviderTransientError as e:
            last_err = e
            if attempt < attempts:
                log.warning(
                    "provider transient (attempt %d/%d): %s", attempt, attempts, e
                )
                await asyncio.sleep(backoff)
    raise RuntimeError(f"provider failed after {attempts} attempts: {last_err}")


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
    tr.write(
        {
            "event": "cache",
            "turn": turn,
            "hit": u.cached_tokens,
            "create": u.cache_creation_tokens,
            "new": new,
            "total": total,
            # Output tokens — the session summary sums these into
            # usage.output_tokens.
            "out": u.completion_tokens,
        }
    )
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


def _auto_schedule_wait_check(tr: Trace, *, minutes: int) -> None:
    """Schedule the singleton auto-WAIT-check job to fire in `minutes`.
    Reuses one canonical job id across sessions (see
    `jobs.upsert_auto_wait_check`) so jobs.md doesn't grow one entry per
    WAIT close.
    """
    at = dt.datetime.now() + dt.timedelta(minutes=minutes)
    try:
        jobs.upsert_auto_wait_check(at)
        tr.write(
            {
                "event": "wait_auto_scheduled",
                "job_id": jobs.AUTO_WAIT_JOB_ID,
                "at": at.isoformat(timespec="minutes"),
            }
        )
    except Exception as e:
        log.exception("failed to auto-schedule WAIT follow-up")
        tr.write({"event": "wait_auto_schedule_failed", "error": str(e)})
