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

from physiclaw.agent import layout
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
    # Macros that aborted this session — `run_macro` refuses them from then
    # on (policy.BurnedMacro). A macro is a REHEARSED path: once the screen
    # stops matching it, the rehearsal no longer describes what is in front
    # of us, and a re-run replays every already-completed step. One strike,
    # then the agent finishes by hand. Written by the run_macro handler;
    # bad_input never lands here — it raises before a result exists and says
    # nothing about the macro.
    failed_macros: set[str] = field(default_factory=set)
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
    kb: layout.KeyboardTracker = field(default_factory=layout.KeyboardTracker)
    # True while the loop dispatches a conductor-synthesized turn. The
    # plan gate stands down (the armed playbook IS the plan) and the
    # run_macro handler resolves pack-private `app/name` macros only
    # under it — the model can never borrow the conductor's hands.
    synthesized_turn: bool = False
    # Turns the MODEL produced (synthesized turns excluded) — the meter
    # gates like PlanGate read, so a long playbook prefix never leaves
    # the model instantly overdue on its first real turn.
    model_turns: int = 0
