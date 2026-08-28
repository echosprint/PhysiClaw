"""Turn loop — the driver (principle 7): model → tool_calls → dispatch →
tool_results → model.

Owns the turn-level protocol invariants — real finish_reason routing
(principle 3), the [note, one-other] turn shape — plus the *mechanics*
of every interception: pop-and-corrective, trace events, the corrective
streak. The *judgment* — when to reject a turn, what to warn about —
lives in `policy.py` as the turn gates on `EngineRun.policies`;
tool-call execution lives in `dispatch.py`.
"""

import asyncio
import datetime as dt
import enum
import logging
import time
from dataclasses import dataclass

from physiclaw.agent.engine import assemble, compact, memory, trajectory
from physiclaw.agent.engine.dispatch import dispatch
from physiclaw.agent.engine.runspec import EngineRun
from physiclaw.agent.engine.session import Session
from physiclaw.agent.runtime.sentinel import FAIL, STUCK
from physiclaw.agent.trace import Trace
from physiclaw.contract.dto import (
    AssistantMessage,
    FinishReason,
    Message,
    UserMessage,
)
from physiclaw.provider import ProviderError, ProviderTransientError

log = logging.getLogger(__name__)

# A model that can't hold the [note, one-other] shape burns a full
# provider round-trip per corrective; after this many CONSECUTIVE
# failures, end STUCK instead of paying up to max_turns of them.
CORRECTIVE_LIMIT = 5


class _Turn(enum.Enum):
    """What `drive` does after a turn's finish_reason + shape checks."""

    PROCEED = enum.auto()  # well-formed — go run the turn's tools
    RETRY = enum.auto()  # rejected, corrective injected — re-loop this index
    ABORT = enum.auto()  # terminal sentinel set — return from `drive`


@dataclass
class _LoopState:
    """Loop-local counters that persist across turns within one `drive` run.

    consecutive_correctives: malformed turns in a row — STUCK past the cap.
    (Per-gate one-shot state lives on the gate objects in `run.policies`.)
    """

    consecutive_correctives: int = 0


async def drive(run: EngineRun, session: Session, messages: list[Message]) -> None:
    """The driver (principle 7): model → tool_calls → dispatch → results → model.

    Each turn runs a fixed pipeline of phase helpers; the control flow between
    them stays here so it's auditable in one place:
      1. `_prepare_request`    — advance clocks, pin tails, log the request.
      2. `_call_provider`      — plugin arbitration, then the provider
                                 with retry; None ⇒ STUCK, return.
      3. `_enforce_shape`      — finish_reason + [note, one-other]; may RETRY/ABORT.
      4. gates observe_turn    — accumulate per-turn evidence (memory cues).
      5. `_apply_turn_gates`   — the policy gates in declared order; may RETRY.
      6. `_dispatch_turn`      — run the tool_calls, append their results.
      7. `_finalize_turn`      — snapshot trajectory, compact, reset gates.

    A plugin-synthesized turn (e.g. the conductor's playbook walk) skips
    3–5 — those judge model output — and dispatches under
    `session.synthesized_turn`.
    """
    state = _LoopState()
    for turn in range(run.settings.max_turns):
        if run.deadline is not None and time.monotonic() >= run.deadline:
            _close_budget_exhausted(run, session, turn)
            return
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

        # A synthesized turn is a plugin's own output, not the
        # model's: the shape is correct by construction and the turn
        # gates' judgments (plan discipline, checkpoints, stuck
        # reflection) address model behavior — skip both. Dispatch
        # guards still see every call, under `session.synthesized_turn`
        # so the plan gate stands down while phone protection holds.
        if not asst.synthesized:
            # The plan gate's meter: turns the MODEL produced. Synthesized
            # turns must not advance it — a long playbook would otherwise
            # leave the model instantly plan-gated on its first real turn.
            session.model_turns += 1
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

        session.synthesized_turn = asst.synthesized
        try:
            await _dispatch_turn(run, session, messages, asst, turn)
        finally:
            session.synthesized_turn = False
        _finalize_turn(run, session, messages, asst, called, turn)

        if session.sentinel_status:
            return
    else:
        log.warning("engine hit max turns (%d)", run.settings.max_turns)
        session.sentinel_status = STUCK
        session.sentinel_recap = f"max turns ({run.settings.max_turns}) reached"


def _close_budget_exhausted(run: EngineRun, session: Session, turn: int) -> None:
    """Close the session STUCK when the wall-clock budget runs out — the
    time counterpart of the max-turns backstop. Writes the mid-task
    recovery breadcrumb first (the guard needs the sentinel still unset),
    and flags `budget_exhausted` so the outcome is not retried: a slow
    environment would burn every retry the same way."""
    budget = run.settings.max_session_seconds
    log.warning(
        "session wall-clock budget (%ds) exhausted at turn %d — closing STUCK",
        budget,
        turn + 1,
    )
    run.tr.write({"event": "budget_exhausted", "turn": turn, "budget_seconds": budget})
    log_external_stop(session, run.tr, cause="wall-clock budget exhausted", event=None)
    session.budget_exhausted = True
    session.sentinel_status = STUCK
    session.sentinel_recap = f"wall-clock budget ({budget}s) exhausted"


def log_external_stop(
    session: Session,
    tr: Trace | None,
    *,
    cause: str = "stopped externally",
    event: str | None = "stopped_externally",
) -> None:
    """Deterministic mid-task checkpoint when a session is cut short —
    external stop (Ctrl-C, runtime shutdown) or the wall-clock budget.
    The model gets no final turn, so the harness writes the daily-log
    line the doctrine would have required — the auto-injected recovery
    breadcrumb for the next wake. Sync file I/O only (this runs on the
    cancellation path); best effort, never raises."""
    if session.sentinel_status is not None or not session.plan.is_drafted():
        return  # closed cleanly, or nothing recoverable yet
    try:
        step = session.plan.current_step() or "(none in_progress)"
        done, total = session.plan.progress()
        memory.append_log(
            f"[{dt.datetime.now():%H:%M}] engine: {cause} mid-task "
            f"({done}/{total} steps done) — in-progress: "
            f"{step}. Verify app state before resuming."
        )
        if tr is not None and event:
            tr.write({"event": event})
        log.warning("%s mid-task — recovery line logged", cause)
    except Exception:
        log.exception("mid-task recovery log failed")


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
        policy=run.collapse,
        collapsed_once=session.collapsed_once,
    )
    # First-run reminder is empty once learned, and a session that started
    # learned stays learned — so the per-turn disk read is skipped entirely
    # unless setup was still pending at session start (`layout_incomplete`).
    request_messages = assemble.apply_request_tails(
        messages,
        session,
        layout_incomplete=run.layout_incomplete,
        compaction_keep=(run.collapse.keep if compaction_imminent else None),
    )
    # The wire-form request record is NOT written here: only turns that
    # actually reach the provider pay the full-transcript serialization
    # (`_call_provider`) — a playbook walk of synthesized turns would
    # otherwise serialize and log the growing history once per leg.
    run.tr.write(
        {"event": "request", "turn": turn, "message_count": len(request_messages)}
    )
    log.info("turn %d: %d messages → conductor", turn + 1, len(request_messages))
    return request_messages, compaction_imminent


async def _call_provider(
    run: EngineRun,
    session: Session,
    request_messages: list[Message],
    turn: int,
) -> AssistantMessage | None:
    """Ask the plugins, then the provider, for the turn (transient-retry
    on the provider only) — a synthesized plugin turn or a provider call
    — log the response, and return the AssistantMessage. When retries
    exhaust: mark the session STUCK, trace it, and return None — the
    caller returns from the loop."""
    t0 = time.perf_counter()
    try:
        asst = await _advance_with_retry(
            run.plugins,
            run.chat,
            request_messages,
            run.tool_schemas,
            attempts=run.settings.provider_retry_attempts,
            backoff=run.settings.retry_backoff_seconds,
        )
    except Exception as e:
        if isinstance(e, ProviderError):
            # Expected terminal state (retries exhausted, permanent 4xx)
            # — already handled as STUCK + session retry. One readable
            # line; a traceback here is noise, not signal.
            log.error("provider failed: %s", e)
        else:
            log.exception("provider call crashed")
        # The failed request's wire form is the forensic record — write
        # it now (deferred from _prepare_request; see below).
        run.rlog.write_request(turn, run.serialize_wire(request_messages))
        run.tr.write({"event": "provider_failed", "turn": turn, "error": str(e)})
        session.sentinel_status = STUCK
        session.sentinel_recap = f"provider error: {e}"
        return None

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    if not asst.synthesized:
        # Cache markers + the actual wire format are the provider's
        # business; log the wire form by asking it to serialize once.
        # Deferred to AFTER the advance so a synthesized turn — which
        # sent no request — never pays the full-transcript serialization.
        run.rlog.write_request(turn, run.serialize_wire(request_messages))
    run.rlog.write_response(
        turn, asst.raw, elapsed_ms=elapsed_ms, synthesized=asst.synthesized
    )
    run.tr.write(
        {
            "event": "response",
            "turn": turn,
            "finish_reason": asst.finish_reason,
            "content_len": len(asst.content or ""),
            # Provider round-trip time — the session summary sums these
            # into provider_time_ms (synthesized turns count into
            # conductor_turns instead; no request was sent).
            "elapsed_ms": elapsed_ms,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in asst.tool_calls
            ],
            **({"synthesized": True} if asst.synthesized else {}),
        }
    )
    if asst.synthesized:
        # No provider round-trip: no usage to log, and the token/cache
        # metrics would report zeros that read as data.
        log.info("turn %d: conductor synthesized calls=%s", turn + 1, asst.tool_names())
        return asst
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
        result = await dispatch(run, session, call, turn)
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
    if compact.collapse_old_turns(
        messages,
        policy=run.collapse,
        collapsed_once=session.collapsed_once,
    ):
        session.collapsed_once = True
    for gate in run.policies.turn_gates:
        gate.on_turn_complete()


# ---------- helpers ----------


async def _advance_with_retry(
    plugins,
    chat,
    messages: list[Message],
    tools: list[dict],
    *,
    attempts: int,
    backoff: float,
) -> AssistantMessage:
    """Plugin arbitration, then the provider with transient-retry.

    Plugins are asked in registration order; the first to return a turn
    has spoken and the provider is never called (arbitration, not a
    chain — see `contract.plugin`). The protocol says a plugin never
    raises; the wrapper here is the seam's own belt: a plugin bug must
    degrade to "the LLM speaks", not kill the session. Only the provider
    call retries — transient errors only (principle 3: permanent 4xx
    fails fast)."""
    for plug in plugins:
        try:
            turn = await plug.advance(messages)
        except Exception:
            log.exception("turn plugin advance crashed — passing to the next")
            turn = None
        if turn is not None:
            return turn
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await chat(messages, tools)
        except ProviderTransientError as e:
            last_err = e
            if attempt < attempts:
                log.warning(
                    "provider transient (attempt %d/%d): %s", attempt, attempts, e
                )
                await asyncio.sleep(backoff)
    # Typed, so _call_provider can tell this expected exhaustion (one
    # clean log line) from an engine bug (full traceback).
    raise ProviderError(f"gave up after {attempts} attempts: {last_err}")


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
