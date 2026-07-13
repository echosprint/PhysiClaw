"""Unit tests for `physiclaw.agent.engine.policy` — the policy objects in
isolation. Their loop-visible behavior (rejection mechanics, blocked
ToolResults, appended advisories) is covered through `test_engine_loop.py`;
this file owns the judgment edges that don't need a loop.
"""

from __future__ import annotations

from physiclaw.agent.engine import policy as policy_mod
from physiclaw.agent.engine.dto import AssistantMessage, FinishReason, ToolCall, Usage
from physiclaw.agent.engine.session import Session


def _asst(tool_calls: list[ToolCall]) -> AssistantMessage:
    return AssistantMessage(
        content="",
        tool_calls=tool_calls,
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(),
    )


def _tc(name: str, args: dict | None = None) -> ToolCall:
    return ToolCall(id=f"tc_{name}", name=name, arguments=args or {})


# ---------- base shapes default to no-ops ----------


def test_base_shapes_are_permissive_no_ops() -> None:
    session = Session()
    asst = _asst([_tc("note", {"summary": "x"}), _tc("peek")])

    gate = policy_mod.TurnGate()
    gate.observe_turn(session, asst)
    assert (
        gate.check(
            session,
            asst,
            ["note", "peek"],
            turn=0,
            compaction_imminent=False,
        )
        is None
    )
    gate.on_turn_complete()

    assert policy_mod.DispatchGuard().check(session, _tc("tap"), turn=0) is None
    assert (
        policy_mod.ResultObserver().observe(
            session,
            _tc("tap"),
            changed=True,
            failed=False,
        )
        is None
    )


# ---------- CompactionCheckpoint ----------


def test_compaction_checkpoint_passes_a_note_with_scratchpad() -> None:
    gate = policy_mod.CompactionCheckpoint()
    asst = _asst(
        [
            _tc("note", {"summary": "x", "scratchpad": "cart=3"}),
            _tc("peek"),
        ]
    )

    rej = gate.check(
        Session(),
        asst,
        ["note", "peek"],
        turn=5,
        compaction_imminent=True,
    )

    assert rej is None
    assert gate._retried is False  # the one-shot wasn't consumed


def test_compaction_checkpoint_rejects_once_then_renews_on_turn_complete() -> None:
    gate = policy_mod.CompactionCheckpoint()
    bare = _asst([_tc("note", {"summary": "x"}), _tc("peek")])

    first = gate.check(
        Session(), bare, ["note", "peek"], turn=5, compaction_imminent=True
    )
    second = gate.check(
        Session(), bare, ["note", "peek"], turn=5, compaction_imminent=True
    )

    assert first is not None and first.event == "checkpoint_corrective"
    assert second is None  # fail open within the same collapse event

    gate.on_turn_complete()
    third = gate.check(
        Session(), bare, ["note", "peek"], turn=6, compaction_imminent=True
    )
    assert third is not None  # renewed for the next collapse event


# ---------- StuckReflection ----------


def _stuck_session(step_turns: int) -> Session:
    session = Session()
    session.plan.update(
        user_said="buy tape",
        steps=[{"content": "add tape to cart", "status": "in_progress"}],
    )
    session.plan.step_turns = step_turns
    return session


def test_stuck_reflection_rejects_gesture_turn_at_urgent() -> None:
    gate = policy_mod.StuckReflection(urgent=12)
    asst = _asst([_tc("note", {"summary": "x"}), _tc("tap")])

    rej = gate.check(
        _stuck_session(12), asst, ["note", "tap"], turn=20, compaction_imminent=False
    )

    assert rej is not None and rej.event == "stuck_reflection"
    assert "add tape to cart" in rej.corrective


def test_stuck_reflection_passes_replan_close_and_non_gesture_turns() -> None:
    gate = policy_mod.StuckReflection(urgent=12)
    session = _stuck_session(12)

    for called in (
        ["note", "update_progress"],  # the demanded way out
        ["note", "end_session"],  # closing is escalation, not looping
        ["note", "peek"],  # not a gesture — already changing method
    ):
        asst = _asst([_tc(n) for n in called])
        assert (
            gate.check(session, asst, called, turn=20, compaction_imminent=False)
            is None
        )


def test_stuck_reflection_one_shot_per_step_identity() -> None:
    gate = policy_mod.StuckReflection(urgent=12)
    session = _stuck_session(12)
    asst = _asst([_tc("note"), _tc("tap")])
    called = ["note", "tap"]

    first = gate.check(session, asst, called, turn=20, compaction_imminent=False)
    second = gate.check(session, asst, called, turn=21, compaction_imminent=False)
    assert first is not None
    assert second is None  # fail open on the same step

    # A different step that gets stuck re-arms the gate.
    session.plan.update(steps=[{"content": "search for glue", "status": "in_progress"}])
    session.plan.step_turns = 12
    assert (
        gate.check(session, asst, called, turn=40, compaction_imminent=False)
        is not None
    )


def test_stuck_reflection_silent_below_urgent_and_on_undrafted_plan() -> None:
    gate = policy_mod.StuckReflection(urgent=12)
    asst = _asst([_tc("note"), _tc("tap")])
    called = ["note", "tap"]

    assert (
        gate.check(_stuck_session(11), asst, called, turn=20, compaction_imminent=False)
        is None
    )
    undrafted = Session()  # default seed plan — no in_progress step
    undrafted.plan.step_turns = 30
    assert (
        gate.check(undrafted, asst, called, turn=20, compaction_imminent=False) is None
    )


# ---------- MemoryCueCheckpoint ----------


def test_memory_cue_scan_reads_update_progress_fields() -> None:
    gate = policy_mod.MemoryCueCheckpoint()
    session = Session()
    asst = _asst(
        [
            _tc("note", {"summary": "ok"}),
            _tc(
                "update_progress",
                {
                    "user_said": "记住 gate code is 4021",
                    "understanding": "save the code",
                    "steps": [
                        {"content": "reply", "status": "in_progress"},
                        "not-a-dict",
                    ],
                },
            ),
        ]
    )

    gate.observe_turn(session, asst)

    assert any("4021" in c for c in session.memory_cues)


def test_memory_cue_scan_skipped_when_disabled(monkeypatch) -> None:
    from physiclaw.common import config

    # The knob is wired through default_policies (the single CONFIG read).
    monkeypatch.setattr(config.CONFIG.engine, "memory_cue_enabled", False)
    gate = next(
        g
        for g in policy_mod.default_policies(layout_incomplete=False).turn_gates
        if isinstance(g, policy_mod.MemoryCueCheckpoint)
    )
    session = Session()

    gate.observe_turn(session, _asst([_tc("note", {"summary": "记住 this"})]))

    assert session.memory_cues == []


def test_memory_cue_scan_skipped_once_cap_is_full() -> None:
    gate = policy_mod.MemoryCueCheckpoint()
    session = Session()
    session.memory_cues = [f"cue {i}" for i in range(gate.MAX_CUES)]

    gate.observe_turn(session, _asst([_tc("note", {"summary": "记住 another"})]))

    assert len(session.memory_cues) == gate.MAX_CUES
    assert not any("another" in c for c in session.memory_cues)


# ---------- LayoutLint ----------


def test_layout_lint_passes_when_lint_finds_nothing(mocker) -> None:
    from physiclaw.agent.engine import screen_layout

    mocker.patch.object(screen_layout, "lint_gesture", return_value=None)

    block = policy_mod.LayoutLint().check(
        Session(),
        _tc("long_press", {"bbox": [0, 0, 1, 1]}),
        turn=0,
    )

    assert block is None


def test_layout_lint_ignores_non_gesture_tools(mocker) -> None:
    from physiclaw.agent.engine import screen_layout

    spy = mocker.patch.object(screen_layout, "lint_gesture")

    block = policy_mod.LayoutLint().check(Session(), _tc("tap"), turn=0)

    assert block is None
    spy.assert_not_called()


# ---------- default_policies ----------


def test_default_policies_declares_the_documented_order() -> None:
    p = policy_mod.default_policies(layout_incomplete=False)

    assert [type(g).__name__ for g in p.turn_gates] == [
        "CompactionCheckpoint",
        "StuckReflection",
        "PitfallCheckpoint",
        "MemoryCueCheckpoint",
    ]
    assert [type(g).__name__ for g in p.dispatch_guards] == [
        "PlanGate",
        "LayoutLint",
        "StuckBlock",
    ]
    assert [type(o).__name__ for o in p.result_observers] == [
        "KeyboardBelief",
        "StuckRecorder",
    ]
    # Only the plan gate runs pre-validation; the phase split is
    # precomputed at construction.
    assert [type(g).__name__ for g in p.pre_validation_guards] == ["PlanGate"]
    assert [type(g).__name__ for g in p.post_validation_guards] == [
        "LayoutLint",
        "StuckBlock",
    ]


def test_default_policies_builds_fresh_gate_state_per_call() -> None:
    a = policy_mod.default_policies(layout_incomplete=False)
    b = policy_mod.default_policies(layout_incomplete=False)

    assert a.turn_gates[0] is not b.turn_gates[0]


# ---------- KeyboardBelief ----------


def test_keyboard_belief_failed_gesture_degrades_to_unknown() -> None:
    session = Session()
    session.kb.state = "up"

    policy_mod.KeyboardBelief().observe(
        session, _tc("tap", {"bbox": [0.1, 0.1, 0.2, 0.2]}), changed=None, failed=True
    )

    assert session.kb.state == "unknown"


def test_keyboard_belief_failed_local_tool_preserves_belief() -> None:
    """A failed note/Skill/jobs call never touched the screen — it must
    not demote the keyboard belief the layout lint depends on."""
    session = Session()
    session.kb.state = "up"

    policy_mod.KeyboardBelief().observe(
        session, _tc("note", {"summary": "x"}), changed=None, failed=True
    )

    assert session.kb.state == "up"


def test_keyboard_belief_success_routes_to_tracker(mocker) -> None:
    session = Session()
    spy = mocker.patch.object(session.kb, "observe")

    policy_mod.KeyboardBelief().observe(
        session, _tc("tap", {"bbox": [0.1, 0.1, 0.2, 0.2]}), changed=True, failed=False
    )

    spy.assert_called_once_with("tap", {"bbox": [0.1, 0.1, 0.2, 0.2]}, True)
