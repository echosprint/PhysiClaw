"""Tests for `physiclaw.agent.engine.plan` — the session working plan.

Three public surfaces:

  - `Step` dataclass — content + status
  - `Plan` dataclass — defaults, `tick_turn`, `update`, `render`
  - `inject_tail(messages, plan)` — appends a plan-tail UserMessage

The staleness tip thresholds (`STALE_TICK_AFTER`,
`DEFAULT_STATE_AFTER`) are loaded from CONFIG at import; tests pin
them via the live module attributes so a config change shows up here.

Accepted equivalent mutants:

  - `parsed_steps: list[Step] | None = None` → `list[Step] & None = None` —
    a local-variable annotation, never evaluated at runtime; the
    mutation is invisible.
"""

from __future__ import annotations

import pytest

from physiclaw.agent.engine import plan as plan_mod
from physiclaw.agent.engine.dto import UserMessage
from physiclaw.agent.engine.plan import (
    COMPLETED,
    DEFAULT_SEED_STEP,
    DEFAULT_UNDERSTANDING,
    DEFAULT_USER_SAID,
    IN_PROGRESS,
    PENDING,
    STATUS_ICON,
    STATUSES,
    Plan,
    Step,
    inject_tail,
)

# ---------- protocol constants ----------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("PENDING", "pending"),
        ("IN_PROGRESS", "in_progress"),
        ("COMPLETED", "completed"),
    ],
)
def test_status_constant_pinned(name: str, expected: str) -> None:
    assert getattr(plan_mod, name) == expected


def test_default_user_said_is_pinned_literal() -> None:
    assert DEFAULT_USER_SAID == "(not yet read)"


def test_default_understanding_is_pinned_literal() -> None:
    assert DEFAULT_UNDERSTANDING == (
        "Unknown — open IM, read the latest message, then call update_progress."
    )


def test_default_seed_step_is_pinned_literal() -> None:
    assert DEFAULT_SEED_STEP == (
        "(no plan yet — after reading the user's IM, call update_progress "
        "with the full step list through end_session; see CONVENTION § "
        "'The plan' for rules, update_progress docstring for the worked "
        "example)"
    )


def test_statuses_frozenset_is_exactly_three_members() -> None:
    assert STATUSES == frozenset({PENDING, IN_PROGRESS, COMPLETED})


@pytest.mark.parametrize(
    "status, icon",
    [(PENDING, "- "), (IN_PROGRESS, "▸ "), (COMPLETED, "✓ ")],
)
def test_status_icon_mapping(status: str, icon: str) -> None:
    assert STATUS_ICON[status] == icon


# ---------- Step ----------


def test_step_defaults_to_pending_status() -> None:
    s = Step(content="x")

    assert s.status == PENDING


# ---------- Plan defaults ----------


def test_plan_defaults_to_one_seed_step_and_unknown_user_said() -> None:
    p = Plan()

    assert p.user_said == DEFAULT_USER_SAID
    assert p.understanding == DEFAULT_UNDERSTANDING
    assert len(p.steps) == 1
    assert p.steps[0].content == DEFAULT_SEED_STEP
    assert p.steps[0].status == PENDING
    assert p.turns_since_update == 0


def test_plan_default_steps_factory_does_not_share_list_between_instances() -> None:
    a = Plan()
    b = Plan()

    a.steps.append(Step(content="a-only"))

    assert len(b.steps) == 1


# ---------- tick_turn ----------


def test_tick_turn_increments_turns_since_update_each_call() -> None:
    p = Plan()

    p.tick_turn()
    p.tick_turn()
    p.tick_turn()

    assert p.turns_since_update == 3


# ---------- update ----------


def test_update_with_no_arguments_raises_value_error() -> None:
    p = Plan()

    with pytest.raises(
        ValueError,
        match=r"^update needs at least one of user_said / understanding / steps$",
    ):
        p.update()


def test_update_strips_whitespace_around_user_said_and_understanding() -> None:
    p = Plan()

    p.update(user_said="  hello  ", understanding="\n\tcontext\n")

    assert p.user_said == "hello"
    assert p.understanding == "context"


def test_update_replaces_steps_with_parsed_step_objects() -> None:
    p = Plan()

    p.update(
        steps=[
            {"content": "first", "status": "in_progress"},
            {"content": "second", "status": "pending"},
        ],
    )

    assert [s.content for s in p.steps] == ["first", "second"]
    assert [s.status for s in p.steps] == ["in_progress", "pending"]


def test_update_resets_turns_since_update_to_zero() -> None:
    p = Plan()
    p.tick_turn()
    p.tick_turn()

    p.update(user_said="something")

    assert p.turns_since_update == 0


def test_update_with_two_in_progress_steps_raises_with_both_names() -> None:
    p = Plan()
    bad_steps = [
        {"content": "step-A", "status": "in_progress"},
        {"content": "step-B", "status": "in_progress"},
    ]

    with pytest.raises(
        ValueError,
        match=(
            r"^2 steps in_progress \('step-A', 'step-B'\); exactly one step "
            r"may be in_progress at a time — mark the others pending or completed$"
        ),
    ):
        p.update(steps=bad_steps)


def test_update_without_steps_arg_leaves_existing_steps_unchanged() -> None:
    # parsed_steps is initialized to None so the post-validation
    # `if parsed_steps is not None` skips assigning self.steps. A
    # truthy-but-not-None initialization would clobber self.steps.
    p = Plan()
    initial_steps = p.steps

    p.update(user_said="hello")

    assert p.steps is initial_steps


def test_update_validates_before_mutating_so_failure_leaves_plan_intact() -> None:
    p = Plan()
    p.update(user_said="prior", understanding="prior")

    bad_steps = [
        {"content": "a", "status": "in_progress"},
        {"content": "b", "status": "in_progress"},
    ]

    with pytest.raises(ValueError):
        p.update(user_said="should not stick", steps=bad_steps)

    # user_said unchanged because steps validation failed first.
    assert p.user_said == "prior"


def test_update_allows_exactly_one_in_progress_step() -> None:
    p = Plan()

    p.update(
        steps=[
            {"content": "a", "status": "in_progress"},
            {"content": "b", "status": "pending"},
        ],
    )

    assert sum(1 for s in p.steps if s.status == IN_PROGRESS) == 1


# ---------- render ----------


def test_render_default_plan_emits_exact_expected_block() -> None:
    # Exact-string match anchors all the f-string template literals at
    # once: "<plan>", "User said: ", "My understanding: ", "Progress: ",
    # "Steps:", the per-step indent, and the closing "</plan>".
    expected = "\n".join(
        [
            "<plan>",
            f"User said: {DEFAULT_USER_SAID}",
            f"My understanding: {DEFAULT_UNDERSTANDING}",
            "Progress: 0/1",
            "Steps:",
            f"  - {DEFAULT_SEED_STEP}",
            "</plan>",
        ]
    )

    assert Plan().render() == expected


def test_render_progress_counter_reflects_completed_steps() -> None:
    p = Plan()
    p.update(
        steps=[
            {"content": "a", "status": "completed"},
            {"content": "b", "status": "completed"},
            {"content": "c", "status": "in_progress"},
            {"content": "d", "status": "pending"},
        ],
    )

    assert "Progress: 2/4" in p.render()


def test_render_with_no_steps_emits_exact_none_marker_line() -> None:
    # The else-branch under `if not step_lines` — only reachable when
    # the plan has zero steps. Exact "  (none)" line; XX-wrap mutation
    # would change it.
    p = Plan()
    p.steps = []

    lines = p.render().splitlines()

    assert "  (none)" in lines
    assert "Progress: 0/0" in lines


def test_render_uses_correct_icon_per_status() -> None:
    p = Plan()
    p.update(
        steps=[
            {"content": "p-step", "status": "pending"},
            {"content": "i-step", "status": "in_progress"},
            {"content": "c-step", "status": "completed"},
        ],
    )

    out = p.render()
    assert "  - p-step" in out
    assert "  ▸ i-step" in out
    assert "  ✓ c-step" in out


def test_render_appends_default_state_tip_after_decay_turns() -> None:
    # Default user_said + turns >= DEFAULT_STATE_AFTER → tip fires.
    # Exact tip line — anchors both halves of the f-string against
    # XX-wrap mutations on either fragment.
    p = Plan()
    for _ in range(plan_mod.DEFAULT_STATE_AFTER):
        p.tick_turn()

    expected_tip = (
        f"⚠ Plan still default after {plan_mod.DEFAULT_STATE_AFTER} "
        "turns — read the user's IM and call update_progress. "
        f"After {plan_mod.PLAN_REQUIRED_AFTER} turns with no plan the engine "
        "blocks every tool except note / update_progress / "
        "end_session."
    )

    assert expected_tip in p.render().splitlines()


def test_render_omits_default_state_tip_before_decay_threshold() -> None:
    p = Plan()
    # One turn shy of the threshold.
    for _ in range(plan_mod.DEFAULT_STATE_AFTER - 1):
        p.tick_turn()

    out = p.render()

    assert "Plan still default" not in out


def test_render_appends_stale_tick_tip_for_non_default_plan() -> None:
    # Non-default state — user has spoken — and STALE_TICK_AFTER turns
    # since last update_progress. Exact-line match anchors both halves.
    p = Plan()
    p.update(user_said="buy bananas")
    for _ in range(plan_mod.STALE_TICK_AFTER):
        p.tick_turn()

    expected_tip = (
        f"⚠ {plan_mod.STALE_TICK_AFTER} turns since last update_progress. "
        "If the current step's intent is achieved, flip its status now; "
        "if you're stuck, re-plan."
    )

    assert expected_tip in p.render().splitlines()


def test_render_omits_stale_tick_tip_when_recently_updated() -> None:
    p = Plan()
    p.update(user_said="buy bananas")
    p.tick_turn()  # only 1 turn since update

    out = p.render()

    assert "turns since last update_progress" not in out


# ---------- step-stuck tips ----------


def _stuck_plan(step_turns: int) -> Plan:
    """Non-default plan with one in_progress step that has run
    `step_turns` turns, kept fresh via periodic no-op updates so the
    stale-tick tip can't mask the step-stuck ones."""
    p = Plan()
    p.update(
        user_said="buy yogurt",
        steps=[{"content": "update quantity to 4", "status": "in_progress"}],
    )
    for i in range(step_turns):
        p.tick_turn()
        if i % 4 == 3:
            p.update(understanding="still working the quantity step")
    return p


def test_step_stuck_warn_tip_fires_at_threshold() -> None:
    p = _stuck_plan(plan_mod.STEP_STUCK_WARN)

    out = p.render()

    assert f"has run {plan_mod.STEP_STUCK_WARN} turns" in out
    assert "CONVENTION" in out


def test_step_stuck_urgent_tip_supersedes_warn() -> None:
    p = _stuck_plan(plan_mod.STEP_STUCK_URGENT)

    out = p.render()

    assert "STOP" in out
    assert "has run" not in out  # only the most urgent tip renders


def test_step_stuck_tip_truncates_long_step_content() -> None:
    # The schema caps a step at 200 chars, but Plan.update doesn't enforce it
    # — so defensively, an over-long step must still not balloon the tip that
    # re-renders every turn once stuck; only a bounded prefix is quoted.
    p = Plan()
    long_step = "verify " * 140  # ~980 chars
    p.update(user_said="x", steps=[{"content": long_step, "status": "in_progress"}])
    for _ in range(plan_mod.STEP_STUCK_WARN):
        p.tick_turn()

    tip = p.render().splitlines()[-2]

    assert "…" in tip
    assert len(tip) < 300
    assert tip.startswith("⚠ Step ▸ 'verify ")


def test_step_stuck_tip_silent_below_threshold() -> None:
    p = _stuck_plan(plan_mod.STEP_STUCK_WARN - 1)

    out = p.render()

    assert "likely stuck" not in out and "STOP" not in out


def test_step_turns_reset_when_in_progress_step_changes() -> None:
    p = _stuck_plan(plan_mod.STEP_STUCK_WARN)
    p.update(
        steps=[
            {"content": "update quantity to 4", "status": "completed"},
            {"content": "pay for the order", "status": "in_progress"},
        ]
    )

    assert p.step_turns == 0
    assert "stuck" not in p.render()


def test_step_turns_survive_unrelated_updates() -> None:
    # Ticking understanding/user_said must NOT reset the step counter —
    # that would let a chatty agent dodge the watchdog forever.
    p = _stuck_plan(plan_mod.STEP_STUCK_WARN)

    assert p.step_turns == plan_mod.STEP_STUCK_WARN


def test_current_step_returns_in_progress_content() -> None:
    p = Plan()
    assert p.current_step() is None  # the seed step is pending
    p.update(
        steps=[
            {"content": "a", "status": "completed"},
            {"content": "b", "status": "in_progress"},
        ]
    )
    assert p.current_step() == "b"


# ---------- is_drafted ----------


def test_is_drafted_false_on_fresh_plan() -> None:
    assert Plan().is_drafted() is False


def test_is_drafted_true_once_user_said_set() -> None:
    p = Plan()
    p.update(user_said="buy yogurt")
    assert p.is_drafted() is True


def test_is_drafted_true_on_real_steps_without_user_said() -> None:
    # Drafting steps counts even if the model never quoted the IM — the
    # plan gate must not punish a structurally-complete plan.
    p = Plan()
    p.update(steps=[{"content": "reply to user", "status": "in_progress"}])
    assert p.is_drafted() is True


def test_is_drafted_false_for_empty_steps() -> None:
    # Gate integrity: `update_progress(steps=[])` must NOT count as a
    # draft — a gated model can't escape with a hollow plan.
    p = Plan()
    p.update(steps=[])
    assert p.is_drafted() is False


def test_is_drafted_false_for_blank_user_said_direct() -> None:
    # Defense in depth: even if a future path stores a blank user_said
    # (bypassing update()'s rejection), the predicate must not open the
    # gate on it.
    p = Plan()
    p.user_said = "   "
    assert p.is_drafted() is False


def test_update_rejects_blank_user_said() -> None:
    # Gate integrity: whitespace passes the schema's minLength but must
    # not open the gate with an empty quote.
    p = Plan()
    with pytest.raises(ValueError, match="blank"):
        p.update(user_said="   ")
    assert p.is_drafted() is False


def test_step_turns_survive_same_steps_reemit() -> None:
    # Re-emitting the plan with the SAME in_progress step must not reset
    # the stuck counter — else re-sending the plan dodges the watchdog.
    p = Plan()
    p.update(steps=[{"content": "update qty", "status": "in_progress"}])
    for _ in range(5):
        p.tick_turn()
    p.update(
        steps=[
            {"content": "update qty", "status": "in_progress"},
            {"content": "pay", "status": "pending"},
        ]
    )
    assert p.step_turns == 5


def test_is_drafted_false_after_understanding_only_update() -> None:
    p = Plan()
    p.update(understanding="probably wants the weather")
    assert p.is_drafted() is False


# ---------- inject_tail ----------


def test_inject_tail_appends_rendered_plan_as_user_message() -> None:
    p = Plan()

    out = inject_tail([], p)

    assert len(out) == 1
    assert isinstance(out[0], UserMessage)
    assert out[0].content == p.render()


def test_inject_tail_does_not_mutate_original_messages_list() -> None:
    msgs: list = []

    inject_tail(msgs, Plan())

    assert msgs == []


def test_inject_tail_preserves_existing_messages_in_order() -> None:
    earlier = UserMessage(content="earlier")

    out = inject_tail([earlier], Plan())

    assert out[0] is earlier
    assert isinstance(out[1], UserMessage)
