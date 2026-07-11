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
    from physiclaw import config

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
