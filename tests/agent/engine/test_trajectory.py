"""Tests for `physiclaw.agent.engine.trajectory` — the turn-tagged history of
plan + scratchpad states that the reflect corrective feeds the agent.
"""

from __future__ import annotations

import pytest

from physiclaw.agent.engine import trajectory
from physiclaw.agent.engine.session import Session


def _drafted_plan(session: Session, understanding: str, steps: list[str]) -> None:
    session.plan.update(
        understanding=understanding,
        steps=[{"content": c, "status": "pending"} for c in steps],
    )


# ---------- Plan.snapshot ----------


def test_plan_snapshot_is_single_line_and_ignores_tip_and_progress() -> None:
    s = Session()
    _drafted_plan(s, "buy milk", ["open app", "search"])
    snap = s.plan.snapshot()
    assert "\n" not in snap
    assert "buy milk" in snap and "open app" in snap and "search" in snap
    # Ticking the turn counter (which changes render()'s tip) must NOT change
    # the snapshot — it moves only on a real re-plan.
    before = s.plan.snapshot()
    for _ in range(30):
        s.plan.tick_turn()
    assert s.plan.snapshot() == before


# ---------- record ----------


def test_record_logs_every_update_progress_even_if_unchanged() -> None:
    # plan_updated=True is an explicit re-plan — capture each call, no dedup.
    s = Session()
    _drafted_plan(s, "u1", ["a"])
    trajectory.record(s, turn=0, plan_updated=True)
    trajectory.record(s, turn=1, plan_updated=True)  # same content, still logged
    assert [t for t, _ in s.plan_log] == [1, 2]  # both calls captured

    _drafted_plan(s, "u2", ["a", "b"])
    trajectory.record(s, turn=5, plan_updated=True)
    assert [t for t, _ in s.plan_log] == [1, 2, 6]


def test_record_safety_net_logs_changed_plan_without_update_deduped() -> None:
    # Without plan_updated (e.g. the closing turn) a drafted plan that changed
    # is logged once, deduped — so the final state still reaches the trajectory.
    s = Session()
    _drafted_plan(s, "u1", ["a"])
    trajectory.record(s, turn=2)  # no update_progress this turn
    trajectory.record(s, turn=3)  # unchanged → not re-logged
    assert [t for t, _ in s.plan_log] == [3]


def test_record_skips_undrafted_plan_and_blank_scratchpad() -> None:
    s = Session()  # default plan is undrafted, scratchpad ""
    trajectory.record(s, turn=3)  # no update_progress, plan undrafted
    assert s.plan_log == [] and s.scratchpad_log == []

    s.scratchpad = "   "
    trajectory.record(s, turn=4)
    assert s.scratchpad_log == []  # whitespace-only never logged


def test_record_logs_scratchpad_changes() -> None:
    s = Session()
    s.scratchpad = "v1"
    trajectory.record(s, turn=0)
    trajectory.record(s, turn=1)  # unchanged, no write → safety net skips
    s.scratchpad = "v2"
    trajectory.record(s, turn=2)
    assert s.scratchpad_log == [(1, "v1"), (3, "v2")]


def test_record_logs_every_scratchpad_write_even_if_unchanged() -> None:
    # scratchpad_written=True is an explicit re-write — capture each, no dedup.
    s = Session()
    s.scratchpad = "same"
    trajectory.record(s, turn=0, scratchpad_written=True)
    trajectory.record(s, turn=1, scratchpad_written=True)  # same content, still logged
    assert s.scratchpad_log == [(1, "same"), (2, "same")]

    # A write that clears to blank is not logged (nothing to reflect on).
    s.scratchpad = "   "
    trajectory.record(s, turn=2, scratchpad_written=True)
    assert [t for t, _ in s.scratchpad_log] == [1, 2]


def test_record_caps_retained_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    from physiclaw.common import config

    monkeypatch.setattr(config.CONFIG.engine, "trajectory_max_snapshots", 3)
    s = Session()
    for i in range(6):
        s.scratchpad = f"v{i}"
        trajectory.record(s, turn=i)
    # Only the newest 3 retained (oldest evicted).
    assert [text for _, text in s.scratchpad_log] == ["v3", "v4", "v5"]


# ---------- render ----------


def test_render_empty_when_no_history() -> None:
    assert trajectory.render(Session()) == ""


def test_render_merges_into_one_ascending_timeline() -> None:
    s = Session()
    _drafted_plan(s, "u1", ["a"])
    trajectory.record(s, turn=0)  # plan @ t1
    _drafted_plan(s, "u2", ["a", "b"])
    trajectory.record(s, turn=9)  # plan @ t10
    s.scratchpad = "notes-late"
    trajectory.record(s, turn=9, scratchpad_written=True)  # scratchpad @ t10

    out = trajectory.render(s)
    assert out.startswith("<session-trajectory>") and out.endswith(
        "</session-trajectory>"
    )
    # One merged timeline, ascending by turn regardless of kind; every entry is
    # `[tN] <kind>\n<content>`.
    assert (
        out.index("[t1] plan\n")
        < out.index("[t10] plan\n")
        < out.index("[t10] scratchpad\n")
    )
    assert out.index("u1") < out.index("u2")  # oldest→newest


def test_render_elides_over_budget_and_says_so() -> None:
    s = Session()
    for i in range(8):
        s.scratchpad = f"snapshot-{i}-" + "x" * 200
        trajectory.record(s, turn=i, scratchpad_written=True)

    out = trajectory.render(s, budget=400)
    assert "elided" in out
    # Newest survives even under a tight budget; the oldest is dropped.
    assert "snapshot-7-" in out and "snapshot-0-" not in out


def test_render_applies_count_cap_from_newest_keeping_full_text() -> None:
    # max_count stops accumulation from the newest backward, regardless of type;
    # every kept entry retains its FULL text (no mid-entry trimming).
    s = Session()
    for i in range(6):
        s.scratchpad = f"note-{i}-" + "y" * 50  # each well under budget
        trajectory.record(s, turn=i, scratchpad_written=True)

    out = trajectory.render(s, budget=10**9, max_count=2)
    assert "(4 earlier entries elided to fit)" in out
    # Newest 2 kept in full; older 4 elided.
    assert ("note-5-" + "y" * 50) in out and ("note-4-" + "y" * 50) in out
    assert "note-3-" not in out


def test_render_edge_entry_is_never_trimmed() -> None:
    # The oldest KEPT entry that sits right at the budget edge keeps full text.
    s = Session()
    s.scratchpad = "A" * 300  # oldest
    trajectory.record(s, turn=0, scratchpad_written=True)
    s.scratchpad = "B" * 300  # newest
    trajectory.record(s, turn=1, scratchpad_written=True)

    # Budget fits both — the edge (oldest) entry appears in full, not sliced.
    out = trajectory.render(s, budget=10**9, max_count=300)
    assert "A" * 300 in out and "B" * 300 in out
