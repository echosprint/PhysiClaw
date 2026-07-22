"""Session — ephemeral per-session state shared by the engine and local tools.

One `Session` lives exactly as long as one engine session attempt
(`engine._run_session`). Local tool handlers (`builtin_tool`) mutate it —
`end_session` sets the sentinel, `add_pitfall` / `save_memory` flip the
close-gate inputs — and the engine reads it between turns.

Only genuinely *shared* state belongs here. State scoped to a single
behavioral policy (e.g. a gate's one-shot corrective flag) lives on the
policy object instead (`policy.py`), where fresh-per-session construction
makes the one-shot semantics structural.
"""

from dataclasses import dataclass, field

from physiclaw.agent.engine import screen_layout
from physiclaw.agent.engine.plan import Plan
from physiclaw.agent.engine.stuck import StuckGuard


@dataclass
class Session:
    """Ephemeral state the engine and local tools share for one session."""

    sentinel_status: str | None = None
    sentinel_recap: str = ""
    sentinel_turn_created_job: bool = False
    # Set by report_screen_layout when it completes first-run setup: the
    # engine ends this session and re-runs the same triggers from scratch so
    # the fresh SYSTEM prompt carries the learned layout for the real task.
    restart_for_setup: bool = False
    # Set by `loop._close_budget_exhausted` when the session's wall-clock
    # budget runs out: the STUCK it closes with must not be retried (the
    # outcome contract reads this as `retryable=False`).
    budget_exhausted: bool = False
    # Set by `loop._finalize_turn` when `compact.collapse_old_turns`
    # actually folds; threads back into the collapse threshold so
    # first-vs-subsequent cadence is explicit state, never inferred from
    # slot byte-forms (an all-empty-summaries fold re-renders the summary
    # slot to its placeholder bytes, which would look like "never
    # collapsed" to a content sniff).
    collapsed_once: bool = False
    # Set by `add_pitfall` (also triggers post-session curation); read by the
    # pre-close pitfalls gate (policy.PitfallCheckpoint). `stuck_events`
    # (loop-guard tally) only enriches the capture corrective's seed.
    added_pitfalls: bool = False
    stuck_events: int = 0
    # Memory-cue gate inputs (policy.MemoryCueCheckpoint): `memory_cues` =
    # "remember this"/"记住" snippets scanned per turn; `saved_memory` set by
    # save/update_memory.
    memory_cues: list[str] = field(default_factory=list)
    saved_memory: bool = False
    plan: Plan = field(default_factory=Plan)
    scratchpad: str = ""
    # Turn-tagged plan + scratchpad history (trajectory.record) fed to the
    # reflect corrective at a hard close — reflect on the whole run, not just
    # the final state.
    plan_log: list[tuple[int, str]] = field(default_factory=list)
    scratchpad_log: list[tuple[int, str]] = field(default_factory=list)
    guard: StuckGuard = field(default_factory=StuckGuard)
    # Cross-call keyboard belief for the layout lint (see KeyboardTracker).
    kb: screen_layout.KeyboardTracker = field(
        default_factory=screen_layout.KeyboardTracker
    )
