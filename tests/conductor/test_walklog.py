"""Tests for `physiclaw.conductor.walklog` — the per-walk runs.jsonl
writer and the aggregation `physiclaw playbooks stats` renders."""

from __future__ import annotations

import json

import pytest

from physiclaw.conductor import walklog


def _record(**overrides) -> None:
    base = dict(
        app="demo",
        playbook="flow",
        outcome="handover",
        idx=1,
        nodes=3,
        node="search",
        reason="leg did not land",
        micros=2,
    )
    walklog.record(**{**base, **overrides})


def test_record_appends_line_with_fields() -> None:
    _record()

    rows = walklog.load()

    assert len(rows) == 1
    assert rows[0]["app"] == "demo"
    assert rows[0]["playbook"] == "flow"
    assert rows[0]["outcome"] == "handover"
    assert rows[0]["node"] == "search"
    assert rows[0]["idx"] == 1
    assert rows[0]["nodes"] == 3
    assert rows[0]["reason"] == "leg did not land"
    assert rows[0]["micros"] == 2
    assert rows[0]["ts"]


def test_record_without_session_marker_records_null_session() -> None:
    _record()

    rows = walklog.load()

    assert rows[0]["session"] is None


def test_record_rejects_unknown_outcome() -> None:
    with pytest.raises(ValueError, match="unknown walk outcome"):
        _record(outcome="exploded")


def test_record_clips_long_reason() -> None:
    _record(reason="x" * 500)

    rows = walklog.load()

    assert len(rows[0]["reason"]) == 200
    assert rows[0]["reason"].endswith("…")


def test_load_returns_empty_when_missing() -> None:
    assert walklog.load() == []


def test_load_skips_unparseable_lines() -> None:
    _record()
    with walklog.runs_file().open("a", encoding="utf-8") as f:
        f.write("not json\n")
    _record(outcome="completed", node=None, reason="")

    rows = walklog.load()

    assert [r["outcome"] for r in rows] == ["handover", "completed"]


@pytest.mark.parametrize(
    ("outcome", "attr"),
    [
        ("completed", "completed"),
        ("suspended", "suspended"),
        ("handover", "handover"),
        ("crashed", "crashed"),
        ("abandoned", "abandoned"),
    ],
)
def test_summarize_counts_each_outcome(outcome: str, attr: str) -> None:
    rows = [{"app": "demo", "playbook": "flow", "outcome": outcome}]

    stats = walklog.summarize(rows)["demo/flow"]

    assert stats.runs == 1
    assert getattr(stats, attr) == 1


def test_summarize_escalation_rate_counts_handover_and_crash_only() -> None:
    # Suspensions are the walk working as designed, and abandonments
    # spent no model session — neither skews the KPI.
    rows = [
        {"app": "demo", "playbook": "flow", "outcome": "completed"},
        {"app": "demo", "playbook": "flow", "outcome": "handover", "node": "a"},
        {"app": "demo", "playbook": "flow", "outcome": "crashed", "node": "a"},
        {"app": "demo", "playbook": "flow", "outcome": "suspended"},
        {"app": "demo", "playbook": "flow", "outcome": "abandoned", "node": "b"},
    ]

    stats = walklog.summarize(rows)["demo/flow"]

    assert stats.escalation_rate == pytest.approx(0.4)


def test_summarize_ranks_hot_nodes_with_latest_reason() -> None:
    rows = [
        {
            "app": "demo",
            "playbook": "flow",
            "outcome": "handover",
            "node": "search",
            "reason": "old reason",
        },
        {
            "app": "demo",
            "playbook": "flow",
            "outcome": "handover",
            "node": "search",
            "reason": "new reason",
        },
        {
            "app": "demo",
            "playbook": "flow",
            "outcome": "handover",
            "node": "open",
            "reason": "once",
        },
    ]

    hot = walklog.summarize(rows)["demo/flow"].hot_nodes()

    assert hot == [("search", 2, "new reason"), ("open", 1, "once")]


def test_summarize_sums_micros_across_runs() -> None:
    rows = [
        {"app": "demo", "playbook": "flow", "outcome": "completed", "micros": 2},
        {"app": "demo", "playbook": "flow", "outcome": "handover", "micros": 3},
    ]

    stats = walklog.summarize(rows)["demo/flow"]

    assert stats.micros == 5


def test_record_and_summarize_carry_rescues() -> None:
    _record(rescues=3)

    rows = walklog.load()
    stats = walklog.summarize(rows)["demo/flow"]

    assert rows[0]["rescues"] == 3
    assert stats.rescues == 3


def test_record_carries_history_fields() -> None:
    _record(
        outcome="completed",
        node=None,
        reason="",
        values={"keyword": "milk"},
        total=45.0,
    )

    (row,) = walklog.load()

    assert row["values"] == {"keyword": "milk"}
    assert row["total"] == 45.0


def test_escalation_sites_rank_and_carry_the_evidence() -> None:
    rows = [
        {
            "app": "a",
            "playbook": "p",
            "outcome": "handover",
            "node": "x",
            "reason": "old",
        },
        {
            "app": "a",
            "playbook": "p",
            "outcome": "handover",
            "node": "x",
            "reason": "new",
            "session": "sid-2",
        },
        {
            "app": "a",
            "playbook": "p",
            "outcome": "crashed",
            "node": "y",
            "reason": "boom",
        },
        {"app": "a", "playbook": "p", "outcome": "completed", "node": None},
    ]

    sites = walklog.escalation_sites(rows)

    assert (sites[0].node, sites[0].count, sites[0].reason) == ("x", 2, "new")
    assert sites[0].sessions == ("sid-2",)
    assert (sites[1].node, sites[1].count) == ("y", 1)


def test_escalation_sites_empty_without_escalations() -> None:
    rows = [{"app": "a", "playbook": "p", "outcome": "completed"}]

    assert walklog.escalation_sites(rows) == []


def test_record_lines_are_valid_json_per_line() -> None:
    _record()
    _record(outcome="completed", node=None, reason="")

    lines = walklog.runs_file().read_text(encoding="utf-8").strip().splitlines()

    assert [json.loads(line)["outcome"] for line in lines] == [
        "handover",
        "completed",
    ]
