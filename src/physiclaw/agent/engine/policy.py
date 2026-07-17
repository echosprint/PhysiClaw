"""Behavioral policies — the judgment layer of the engine loop.

`loop.py` and `dispatch.py` own the protocol invariants (API-legal
transcript, one ToolResult per ToolCall, finish_reason routing, retries)
and the *mechanics* of every interception (pop-and-corrective, blocked
ToolResults, trace events). Everything that is a product decision — WHEN
to reject a turn, WHEN to refuse a call, WHAT to warn about — lives here
as a policy object. Three interception shapes:

- `TurnGate` — inspects a well-formed assistant turn before dispatch and
  may reject it (the engine pops the turn, injects the gate's corrective
  as a UserMessage, and re-runs the turn). Gates own their one-shot
  state: constructed fresh per session, "reject once then fail open" is
  structural rather than a flag someone must remember to check.
- `DispatchGuard` — inspects a single tool call before execution and may
  block it (the engine returns the guard's text as an error ToolResult
  without actuating).
- `ResultObserver` — sees each executed call's outcome and may return an
  advisory the engine appends to the tool result.

Ordering is load-bearing and declared exactly once, in
`default_policies()`: gates run in list order and the first rejection
wins; observers run in list order and share the parsed screen-change
verdict (keyboard belief must update before the stuck guard records).
A new behavior = one new class here + one line in `default_policies()`;
`loop.py` and `dispatch.py` stay untouched.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from physiclaw.agent.engine import (
    memory,
    pitfalls,
    screen_layout,
    trace,
    trajectory,
)
from physiclaw.agent.engine.dto import AssistantMessage, ToolCall
from physiclaw.common.config import CONFIG
from physiclaw.common.gesture_vocab import NAV_TOOLS, PRESS_TOOLS, SEQUENCE, SWIPE

if TYPE_CHECKING:
    from physiclaw.agent.engine.session import Session

log = logging.getLogger(__name__)


# ---------- interception results ----------


@dataclass(frozen=True)
class Rejection:
    """A TurnGate verdict: pop the assistant turn, inject `corrective`,
    re-run the turn. `event` must stay stable — the session summary counts
    rejection kinds by trace event name (see trace._CORRECTIVE_EVENTS)."""

    corrective: str
    event: str
    log_msg: str
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Block:
    """A DispatchGuard verdict: refuse the call pre-actuation and pair it
    with an error ToolResult carrying `content`. `event` names the trace
    event (trace maps tool_blocked_* into per-kind summary counters)."""

    content: str
    event: str
    log_msg: str


@dataclass(frozen=True)
class Advisory:
    """A ResultObserver verdict: append `text` to the tool result and
    emit `event` to the trace."""

    text: str
    event: str = "stuck_warning"


# ---------- policy shapes ----------


class TurnGate:
    """May reject a well-formed turn; owns its own one-shot state."""

    def observe_turn(self, session: "Session", asst: AssistantMessage) -> None:
        """Called once per well-formed turn, before `check` — for gates
        that accumulate evidence across turns (e.g. memory cues)."""

    def check(
        self,
        session: "Session",
        asst: AssistantMessage,
        called: list[str],
        *,
        turn: int,
        compaction_imminent: bool,
    ) -> Rejection | None:
        return None

    def on_turn_complete(self) -> None:
        """Called after a turn dispatches cleanly — for gates whose
        one-shot state renews per event (not per session)."""


class DispatchGuard:
    """May block one tool call before it executes.

    `PRE_VALIDATION` guards run before the unknown-tool / JSONSchema
    checks (a gated call must be told to fix the *plan*, not its
    arguments); the rest run after, so they see validated arguments.
    """

    PRE_VALIDATION = False

    def check(self, session: "Session", call: ToolCall, *, turn: int) -> Block | None:
        return None


class ResultObserver:
    """Sees each executed call's outcome; may return an advisory line.

    `changed` is the parsed screen-change verdict (None = couldn't judge
    or not a gesture); `failed` marks a call whose execution raised.
    """

    def observe(
        self,
        session: "Session",
        call: ToolCall,
        *,
        changed: bool | None,
        failed: bool,
    ) -> Advisory | None:
        return None


# ---------- turn gates ----------


# Pre-compression checkpoint corrective. Mirrors the ⚠ tail the rejected
# turn already saw; names the exact re-issue shape.
_CHECKPOINT_CORRECTIVE = (
    "Rejected — context compresses after this turn: older turns keep "
    "only their note summaries (scratchpad and plan survive in full). "
    "Re-issue the SAME turn with `note` carrying BOTH `summary` and "
    "`scratchpad` — rewrite the scratchpad with the facts and values "
    "collected, the current situation, and what remains."
)


class CompactionCheckpoint(TurnGate):
    """Pre-compression checkpoint: when this turn's completion will collapse
    old turns to their note-summary lines, require the scratchpad write-down
    while the folding turns are still readable. One retry per collapse event,
    then fail open — losing detail beats stalling. A turn that closes the
    session skips it (nothing left to preserve)."""

    def __init__(self) -> None:
        self._retried = False

    def check(self, session, asst, called, *, turn, compaction_imminent):
        if not compaction_imminent or self._retried or "end_session" in called:
            return None
        note_args = next(
            (tc.arguments for tc in asst.tool_calls if tc.name == "note"),
            None,
        )
        sp = note_args.get("scratchpad") if isinstance(note_args, dict) else None
        if isinstance(sp, str) and sp.strip():
            return None
        self._retried = True
        return Rejection(
            corrective=_CHECKPOINT_CORRECTIVE,
            event="checkpoint_corrective",
            log_msg="checkpoint note lacks scratchpad — corrective",
        )

    def on_turn_complete(self) -> None:
        # A completed turn closes any collapse event that was pending on it —
        # the NEXT pending collapse gets a fresh corrective. Reset here (not
        # on "not pending") because with a small interval every turn can be
        # pending, which would otherwise burn the single retry on the first
        # event forever.
        self._retried = False


# Phone gestures — the calls StuckReflection treats as "still pressing".
# swipe/sequence extend the press family: a stuck step continued by a
# scroll or a batched run is the same loop.
_GESTURES = PRESS_TOOLS | {SWIPE, SEQUENCE}


class StuckReflection(TurnGate):
    """Forced re-plan when a step is stuck past the urgent threshold.

    The plan tail's step-stuck ⚠ is advisory and models ignore it. Research
    (MobileUse's trajectory reflector, VeriSafe's hard feedback) says the
    trajectory-level signal must escalate to enforcement: reject the next
    gesture turn once and demand a re-plan. One rejection per step
    identity, then fail open — same conservative shape as the other
    gates. Non-gesture turns pass untouched — the `[note, one-other]`
    shape (enforced before gates run) means a gesture turn can never
    also carry the update_progress / end_session way out."""

    def __init__(self, *, urgent: int) -> None:
        self._urgent = urgent
        self._rejected_step: str | None = None

    def check(self, session, asst, called, *, turn, compaction_imminent):
        p = session.plan
        if p.step_turns < self._urgent:
            return None
        step = p.current_step()  # None on an undrafted plan — seed step is pending
        if (
            step is None
            or step == self._rejected_step
            or not any(c in _GESTURES for c in called)
        ):
            return None
        self._rejected_step = step
        shown = trace.brief(step)
        return Rejection(
            corrective=(
                f"Rejected — step ▸ {shown!r} has run {p.step_turns} turns "
                "with no tick; another gesture continues the loop. Re-issue "
                "as [note, update_progress]: split the step or insert a "
                "recovery step — change METHOD, not coordinates. Truly "
                "blocked → message the user + close WAIT, or "
                "end_session(STUCK) (CONVENTION § Stuck)."
            ),
            event="stuck_reflection",
            log_msg=f"step stuck {p.step_turns} turns — forced re-plan",
            extra={"step": shown, "step_turns": p.step_turns},
        )


class PitfallCheckpoint(TurnGate):
    """Pre-close pitfalls checkpoint: on a long DONE (a run that length
    almost always hit a trap), force `add_pitfall(...)` first so the traps —
    with the fix that worked — land for next time. The agent may add 0. One
    forced retry, then fail open."""

    def __init__(self) -> None:
        self._retried = False

    def check(self, session, asst, called, *, turn, compaction_imminent):
        if "end_session" not in called or session.added_pitfalls or self._retried:
            return None
        end_status = next(
            (
                tc.arguments.get("status", "")
                for tc in asst.tool_calls
                if tc.name == "end_session" and isinstance(tc.arguments, dict)
            ),
            "",
        )
        do_capture, seed = pitfalls.should_capture(end_status, turn, session)
        if not do_capture:
            return None
        self._retried = True
        # Capture the closing turn's state too — this turn rejects before
        # the post-dispatch record, so record here (dedups) so a plan that
        # never completed a turn still reaches the trajectory.
        trajectory.record(session, turn)
        return Rejection(
            corrective=pitfalls.corrective(seed, trajectory.render(session)),
            event="pitfall_checkpoint",
            log_msg=f"pre-close pitfalls checkpoint ({seed})",
            extra={"seed": seed},
        )


class MemoryCueCheckpoint(TurnGate):
    """Memory-cue gate, both halves of the concern in one place:
    `observe_turn` accumulates distinct "remember this" / "记住" cues from
    each turn's own note / scratchpad / plan text (scanned every turn, not
    just at close, so a durable fact survives even after compaction folds
    the turn away); `check` forces one `save_memory` nudge at close when
    cues accumulated but nothing was saved. One retry, then fail open."""

    # Cap on accumulated cue snippets — the first N distinct cues win by
    # append-order.
    MAX_CUES = 5

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._retried = False

    def observe_turn(self, session, asst) -> None:
        if not (self._enabled and len(session.memory_cues) < self.MAX_CUES):
            return
        for snip in memory.scan_cues(_cue_text(asst)):
            if snip not in session.memory_cues:
                session.memory_cues.append(snip)
        del session.memory_cues[self.MAX_CUES :]

    def check(self, session, asst, called, *, turn, compaction_imminent):
        if (
            "end_session" not in called
            or not session.memory_cues
            or session.saved_memory
            or self._retried
        ):
            return None
        self._retried = True
        return Rejection(
            corrective=memory.cue_corrective(session.memory_cues),
            event="memory_cue_checkpoint",
            log_msg=(
                f"pre-close memory-cue checkpoint ({len(session.memory_cues)} cue(s))"
            ),
            extra={"cues": list(session.memory_cues)},
        )


def _cue_text(asst: AssistantMessage) -> str:
    """The turn's own text worth scanning for memory cues: the `note`
    summary + scratchpad, plus any `update_progress` plan text. These are
    exactly the fields that survive compaction, so a cue banked here isn't
    lost when the turn folds away."""
    parts: list[str] = []
    for tc in asst.tool_calls:
        args = tc.arguments if isinstance(tc.arguments, dict) else {}
        if tc.name == "note":
            parts.append(args.get("summary") or "")
            parts.append(args.get("scratchpad") or "")
        elif tc.name == "update_progress":
            parts.append(args.get("user_said") or "")
            parts.append(args.get("understanding") or "")
            for s in args.get("steps") or []:
                if isinstance(s, dict):
                    parts.append(s.get("content") or "")
    return "\n".join(p for p in parts if p)


# ---------- dispatch guards ----------


# Tools that still run while the plan gate is closed: `note` is the
# mandatory turn shape, `update_progress` is the way out, `end_session`
# lets a legitimately plan-less wake (idle check-in) close instead of
# being trapped.
PLAN_GATE_EXEMPT = frozenset({"note", "update_progress", "end_session"})


class PlanGate(DispatchGuard):
    """Drafting the plan is mandatory: past `required_after` turns with no
    plan drafted, every tool is blocked except the way out. Keyed on the
    raw turn index (not `turns_since_update`, which any update_progress
    call resets) so it can't be dodged; off during first-run setup, whose
    screen-layout flow legitimately runs many plan-less turns."""

    # Fires before schema lookup / validation — a gated call must be told
    # to draft the plan, not to fix its arguments.
    PRE_VALIDATION = True

    def __init__(self, *, layout_incomplete: bool, required_after: int):
        self._layout_incomplete = layout_incomplete
        self._required_after = required_after

    def overdue(self, session: "Session", turn: int) -> bool:
        return (
            not self._layout_incomplete
            and turn >= self._required_after
            and not session.plan.is_drafted()
        )

    def check(self, session, call, *, turn):
        if call.name in PLAN_GATE_EXEMPT or not self.overdue(session, turn):
            return None
        return Block(
            content=(
                f"BLOCKED — not executed: turn {turn + 1} and no plan drafted. "
                "Call update_progress NOW (CONVENTION § The plan). Read the IM "
                "already? user_said verbatim + the full step list through "
                "end_session. Still navigating to it? Draft the steps you know "
                "(open app → read IM → re-plan) — steps alone open the gate; "
                "fill user_said once read. Until then only note / "
                "update_progress / end_session run."
            ),
            event="tool_blocked_no_plan",
            log_msg=f"blocked by plan gate (no plan by turn {turn})",
        )


class LayoutLint(DispatchGuard):
    """Refuse a gesture that targets a learned box which cannot work in the
    current keyboard state (e.g. long_press on the keyboard-HIDDEN input box
    with the keyboard up presses the keyboard itself, and the stuck guard is
    blind to it — every press "changes" the screen). Sequences carry
    in-batch evidence; the session's KeyboardTracker carries it across calls
    (a raising tap one turn, the wrong long_press the next). Fail-open on
    any internal error."""

    def check(self, session, call, *, turn):
        if call.name not in (SEQUENCE, "long_press"):
            return None
        try:
            lint = screen_layout.lint_gesture(
                call.name,
                call.arguments,
                keyboard_up=session.kb.state == "up",
            )
        except Exception:
            log.exception("  layout lint failed — skipping")
            return None
        if lint is None:
            return None
        return Block(
            content=lint,
            event="tool_blocked_layout",
            log_msg="blocked by layout lint",
        )


class StuckBlock(DispatchGuard):
    """Stuck guard tier 2: refuse a looping call BEFORE actuating —
    exhausted target, repeated action cycle, or identical failing call.
    Models can ignore warnings; they can't ignore an action that doesn't
    execute."""

    def check(self, session, call, *, turn):
        blocked = session.guard.should_block(call.name, call.arguments)
        if blocked is None:
            return None
        session.stuck_events += 1  # inefficiency signal for the session summary
        return Block(
            content=blocked,
            event="tool_blocked_stuck",
            log_msg="blocked by stuck guard",
        )


# ---------- result observers ----------


# Tools that physically touch the screen — only their failures leave the
# keyboard state unproven. A failed local tool (note, Skill, jobs) says
# nothing about the phone, and demoting belief on it would disarm the
# layout lint mid-typing.
_SCREEN_TOOLS = _GESTURES | NAV_TOOLS | {"unlock_phone"}


class KeyboardBelief(ResultObserver):
    """Feed each gesture outcome into the session's KeyboardTracker — the
    belief the layout lint reads on LATER dispatches. A failed gesture
    leaves the screen state unproven."""

    def observe(self, session, call, *, changed, failed):
        if failed:
            if call.name in _SCREEN_TOOLS:
                session.kb.state = "unknown"
        else:
            session.kb.observe(call.name, call.arguments, changed)
        return None


class StuckRecorder(ResultObserver):
    """Stuck guard tier 1: feed the gesture's screen-change verdict back in;
    any detector crossing its warn threshold (target / cycle / position
    orbit) appends a ⚠ to the result — the highest-adherence steering
    channel. Identical failing calls are loop evidence too (e.g. a target
    overlapping the AssistiveTouch button raises on every retry)."""

    def observe(self, session, call, *, changed, failed):
        if failed:
            warning = session.guard.record_error(call.name, call.arguments)
        else:
            warning = session.guard.record(call.name, call.arguments, changed)
        if warning is None:
            return None
        session.stuck_events += 1  # inefficiency signal for the session summary
        return Advisory(text=warning)


# ---------- assembly ----------


@dataclass
class Policies:
    turn_gates: list[TurnGate]
    dispatch_guards: list[DispatchGuard]
    result_observers: list[ResultObserver]
    # Guards split by phase once at construction, so dispatch iterates only
    # the relevant slice instead of filtering on PRE_VALIDATION per call.
    pre_validation_guards: tuple[DispatchGuard, ...] = field(init=False)
    post_validation_guards: tuple[DispatchGuard, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.pre_validation_guards = tuple(
            g for g in self.dispatch_guards if g.PRE_VALIDATION
        )
        self.post_validation_guards = tuple(
            g for g in self.dispatch_guards if not g.PRE_VALIDATION
        )


def default_policies(*, layout_incomplete: bool) -> Policies:
    """One session's policy set — also the single place policy config knobs
    are read from CONFIG (constructors take plain values, so unit tests
    never monkeypatch config). Ordering is load-bearing:

    - turn gates: compaction checkpoint first (bank state before anything
      else argues), stuck reflection before the close gates (a stuck close
      turn is still a close), pitfalls before memory cues (both one-shot;
      a rejected close re-enters the list from the top).
    - dispatch guards: plan gate first (and pre-validation), lint before
      the stuck block (a linted call must not feed loop counters).
    - observers: keyboard belief updates before the stuck guard records.
    """
    return Policies(
        turn_gates=[
            CompactionCheckpoint(),
            StuckReflection(urgent=CONFIG.engine.step_stuck_urgent),
            PitfallCheckpoint(),
            MemoryCueCheckpoint(enabled=CONFIG.engine.memory_cue_enabled),
        ],
        dispatch_guards=[
            PlanGate(
                layout_incomplete=layout_incomplete,
                required_after=CONFIG.engine.plan_required_after,
            ),
            LayoutLint(),
            StuckBlock(),
        ],
        result_observers=[
            KeyboardBelief(),
            StuckRecorder(),
        ],
    )
