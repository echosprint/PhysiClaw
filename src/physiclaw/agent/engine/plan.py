"""The session's working plan — pinned at the tail of every request.

The plan lives on `Session`, never in `messages[]`. Before each
`conductor.advance(...)` call the engine appends `plan.render()` as a final
`UserMessage` so the model sees its current plan on every turn without
being lost in the scroll of tool_results. After the model responds, the
plan message is discarded; only the assistant reply lands in history.
This keeps the prefix byte-stable across turns — prefix cache still
hits everything above; only the short tail recomputes.

Mutation funnels through the `update_progress` tool (handler in
`builtin_tool.py`). Each step has a typed status (pending / in_progress
/ completed); Plan.update() enforces the invariant that at most ONE
step is `in_progress` at a time — matching Claude Code's TodoWrite rule.
"""

from dataclasses import dataclass, field

from physiclaw.common.config import CONFIG
from physiclaw.contract.dto import Message, UserMessage

PENDING = "pending"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"
STATUSES: frozenset[str] = frozenset({PENDING, IN_PROGRESS, COMPLETED})

STATUS_ICON = {PENDING: "- ", IN_PROGRESS: "▸ ", COMPLETED: "✓ "}

DEFAULT_USER_SAID = "(not yet read)"

DEFAULT_UNDERSTANDING = (
    "Unknown — open IM, read the latest message, then call update_progress."
)

DEFAULT_SEED_STEP = (
    "(no plan yet — after reading the user's IM, call update_progress "
    "with the full step list through end_session; see CONVENTION § "
    "'The plan' for rules, update_progress docstring for the worked "
    "example)"
)


@dataclass
class Step:
    content: str
    status: str = PENDING


# Stay-silent window after a successful update_progress call. A legit
# multi-tap step (e.g. add-to-cart) can run 10-15 tap+peek turns, so a
# tip at turn 8 may fire mid-step — intentional: the tip is advisory,
# not rejecting, and reminding the model to re-check beats missing a
# real forgot-to-flip.
STALE_TICK_AFTER = CONFIG.engine.stale_tick_threshold
# When the plan is still in its default state this long into a wake, it is
# time to draft. Reaching the IM costs ~3 turns of navigation (wake peek →
# open app → open thread), so the default fires at the natural draft turn —
# earlier would nag during legitimate navigation and teach the model to
# ignore ⚠ tips.
DEFAULT_STATE_AFTER = CONFIG.engine.state_decay_turns
# One in_progress step running this long = likely stuck. The engine
# tracks it (the model can't count across compaction — folded turns
# leave nothing to count) and escalates the tip text at each threshold.
STEP_STUCK_WARN = CONFIG.engine.step_stuck_warn
STEP_STUCK_URGENT = CONFIG.engine.step_stuck_urgent
# From this turn the engine's plan gate blocks action tools while the plan
# is undrafted (policy.PlanGate enforces at dispatch; the default-state tip warns).
PLAN_REQUIRED_AFTER = CONFIG.engine.plan_required_after


@dataclass
class Plan:
    user_said: str = DEFAULT_USER_SAID
    understanding: str = DEFAULT_UNDERSTANDING
    steps: list[Step] = field(default_factory=lambda: [Step(DEFAULT_SEED_STEP)])
    turns_since_update: int = 0
    step_turns: int = 0  # turns the current in_progress step has run

    def current_step(self) -> str | None:
        """Content of the in_progress step, or None. Doubles as the
        step identity the stuck guard and `step_turns` key on — so two
        ADJACENT steps with identical content don't reset the counters
        when the tick moves between them. Conservative direction (an
        earlier stuck warning, never a missed one), and content beats
        index as identity: inserting a recovery step above the current
        one must not reset either."""
        for s in self.steps:
            if s.status == IN_PROGRESS:
                return s.content
        return None

    def is_drafted(self) -> bool:
        """True once the model has drafted the plan — quoted the IM into
        `user_said` OR emitted real steps.

        OR, not AND: an agent gated mid-navigation can't quote an IM it
        hasn't read (and cron wakes have none) — its escape is a
        steps-only draft; AND would deadlock it. The gate forces
        engagement; completeness is the plan tail's job. Blank
        `user_said` / empty steps don't count (defense in depth —
        `update` rejects blanks too)."""
        if self.user_said != DEFAULT_USER_SAID and self.user_said.strip():
            return True
        # A real steps-only draft needs at least one step with actual
        # content — not the default seed, and not blank/whitespace (the
        # schema's minLength 1 lets a lone space through, which would
        # otherwise open the gate on an effectively empty plan).
        return any(
            s.content.strip() and s.content != DEFAULT_SEED_STEP for s in self.steps
        )

    def tick_turn(self) -> None:
        """Engine calls this once per turn (before `inject_tail`) so
        `render()` can surface a staleness tip when the model forgets to
        call update_progress."""
        self.turns_since_update += 1
        self.step_turns += 1

    def update(
        self,
        *,
        user_said: str | None = None,
        understanding: str | None = None,
        steps: list[dict] | None = None,
    ) -> None:
        if user_said is None and understanding is None and steps is None:
            raise ValueError(
                "update needs at least one of user_said / understanding / steps"
            )
        # A whitespace-only quote would open the plan gate with a blank
        # `user_said` (the schema's minLength counts the spaces).
        if user_said is not None and not user_said.strip():
            raise ValueError("user_said must not be blank")
        # Validate everything before mutating — partial updates on failure
        # leave the plan in a confusing mixed state.
        parsed_steps: list[Step] | None = None
        if steps is not None:
            parsed_steps = [
                Step(content=s["content"], status=s["status"]) for s in steps
            ]
            active = [s for s in parsed_steps if s.status == IN_PROGRESS]
            if len(active) > 1:
                names = ", ".join(repr(s.content) for s in active)
                raise ValueError(
                    f"{len(active)} steps in_progress ({names}); "
                    "exactly one step may be in_progress at a time — "
                    "mark the others pending or completed"
                )
        before = self.current_step()
        if user_said is not None:
            self.user_said = user_said.strip()
        if understanding is not None:
            self.understanding = understanding.strip()
        if parsed_steps is not None:
            self.steps = parsed_steps
        if self.current_step() != before:
            self.step_turns = 0
        self.turns_since_update = 0

    def progress(self) -> tuple[int, int]:
        """(completed, total) step counts."""
        return sum(1 for s in self.steps if s.status == COMPLETED), len(self.steps)

    def render(self) -> str:
        done, total = self.progress()
        step_lines = [f"  {STATUS_ICON[s.status]}{s.content}" for s in self.steps]
        if not step_lines:
            step_lines = ["  (none)"]
        lines = [
            "<plan>",
            f"User said: {self.user_said}",
            f"My understanding: {self.understanding}",
            f"Progress: {done}/{total}",
            "Steps:",
            *step_lines,
        ]
        tip = self._tip()
        if tip:
            lines.append(tip)
        lines.append("</plan>")
        return "\n".join(lines)

    def snapshot(self) -> str:
        """A compact, single-line view of the plan's semantic content for the
        session trajectory (trajectory.record). Excludes the volatile `_tip()`
        and the derivable progress count — so a snapshot changes only when the
        model actually re-plans, not when a turn-counter ticks."""
        steps = "; ".join(f"{STATUS_ICON[s.status]}{s.content}" for s in self.steps)
        return f"{self.understanding} || {steps}"

    def _tip(self) -> str | None:
        """Contextual reminder line appended to render() when a signal
        fires. Silent when the plan is fresh — the model learns to ignore
        messages that always appear. One tip at a time, most urgent
        first: a stuck step matters more than a stale tick."""
        is_default = not self.is_drafted()
        step = self.current_step()
        if not is_default and step is not None:
            # Steps may be up to 200 chars; the tip re-renders every
            # turn once stuck, so quote a bounded prefix.
            shown = step if len(step) <= 80 else step[:79] + "…"
            if self.step_turns >= STEP_STUCK_URGENT:
                return (
                    f"⚠ Step ▸ {shown!r} at {self.step_turns} turns — STOP "
                    "retrying. Report the blocker to the user with options "
                    "and close WAIT (+ create_job the resume), or "
                    "end_session(STUCK)."
                )
            if self.step_turns >= STEP_STUCK_WARN:
                return (
                    f"⚠ Step ▸ {shown!r} has run {self.step_turns} turns with "
                    "no tick — you are likely stuck. Escalate per CONVENTION "
                    "§ Stuck: re-plan → back out → force-quit → ask the user."
                )
        if is_default and self.turns_since_update >= DEFAULT_STATE_AFTER:
            return (
                f"⚠ Plan still default after {self.turns_since_update} "
                "turns — read the user's IM and call update_progress. "
                f"After {PLAN_REQUIRED_AFTER} turns with no plan the engine "
                "blocks every tool except note / update_progress / "
                "end_session."
            )
        if not is_default and self.turns_since_update >= STALE_TICK_AFTER:
            return (
                f"⚠ {self.turns_since_update} turns since last "
                "update_progress. If the current step's intent is "
                "achieved, flip its status now; if you're stuck, re-plan."
            )
        return None


def inject_tail(messages: list[Message], plan: Plan) -> list[Message]:
    """Return `messages + [plan-tail UserMessage]`. Original list untouched."""
    return messages + [UserMessage(content=plan.render())]
