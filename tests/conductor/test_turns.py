"""Tests for `physiclaw.conductor.turns` — the synthesized-turn
shape and the call-id convention two drivers depend on."""

from __future__ import annotations

from physiclaw.conductor import views
from physiclaw.conductor.turns import Turnsmith
from physiclaw.contract.dto import FinishReason, TextBlock, ToolResultMessage


def test_synth_is_a_note_plus_one_other_marked_synthesized() -> None:
    asst = Turnsmith("walk").synth("peek", "looking", "peek", {})

    assert [c.name for c in asst.tool_calls] == ["note", "peek"]
    assert asst.tool_calls[0].arguments == {"summary": "looking"}
    assert asst.finish_reason is FinishReason.TOOL_CALLS
    # The loop keys its gate-skipping off this flag.
    assert asst.synthesized is True


def test_pending_points_at_the_action_call_not_the_note() -> None:
    # The coupling that makes result lookup work: a driver reads the
    # ACTION's result, and the note's result must never satisfy it.
    t = Turnsmith("walk")
    asst = t.synth("leg", "running", "run_macro", {"name": "app/x"})

    assert t.pending is not None
    assert t.pending.call_id == asst.tool_calls[1].id
    assert t.pending.call_id != asst.tool_calls[0].id
    assert t.pending.kind == "leg"


def test_pending_call_id_resolves_against_real_history() -> None:
    # End-to-end with the reader the drivers actually use.
    t = Turnsmith("walk")
    asst = t.synth("peek", "looking", "peek", {})
    assert t.pending is not None
    history = [
        asst,
        ToolResultMessage(
            tool_call_id=t.pending.call_id,
            content=[TextBlock(text="a screen")],
        ),
    ]

    found = views.result_for(history, t.pending.call_id)

    assert found is not None and views.text_of(found) == "a screen"


def test_call_ids_are_unique_per_turn() -> None:
    t = Turnsmith("walk")

    ids = {c.id for _ in range(3) for c in t.synth("peek", "s", "peek", {}).tool_calls}

    assert len(ids) == 6  # 3 turns × (note, act), none colliding


def test_latest_synth_replaces_the_pending_action() -> None:
    t = Turnsmith("walk")
    t.synth("peek", "s", "peek", {})
    second = t.synth("tap", "s", "tap", {"bbox": [0, 0, 1, 1]})

    assert t.pending is not None
    assert t.pending.kind == "tap"
    assert t.pending.call_id == second.tool_calls[1].id


def test_channel_flag_is_declared_at_the_synth_site() -> None:
    t = Turnsmith("walk")

    assert t.synth("peek", "s", "peek", {}) and t.pending is not None
    assert t.pending.channel is False
    assert t.synth("gate-peek", "s", "peek", {}, channel=True) and t.pending is not None
    assert t.pending.channel is True


def test_driver_scopes_keep_ids_unique_across_smiths() -> None:
    # The boot and the program it activates each run a private Turnsmith; the
    # scope is what keeps their id sequences disjoint — without it a
    # driver whose result never landed could adopt the OTHER driver's
    # stale one (result_for is exact-match, newest-first).
    boot = Turnsmith("boot").synth("peek", "s", "peek", {})
    walk = Turnsmith("walk").synth("peek", "s", "peek", {})

    boot_ids = {c.id for c in boot.tool_calls}
    walk_ids = {c.id for c in walk.tool_calls}

    assert boot_ids.isdisjoint(walk_ids)
    assert all(i.startswith("conductor-") for i in boot_ids | walk_ids)
