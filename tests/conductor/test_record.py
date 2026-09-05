"""Tests for `walk.record` — the walk's terminal record: the `walk`
event every listening session gets (dry walks included — the boot's
hand-on is the session's story), the runs.jsonl row a live walk adds,
the first-terminal-moment latch, and the fail-open sink."""

from __future__ import annotations

from conductor_fakes import Sink

from physiclaw.conductor.walk import walklog
from physiclaw.conductor.walk.record import Record
from physiclaw.conductor.walk.walklog import Outcome


class _Broken:
    def write(self, event: dict) -> None:
        raise OSError("disk gone")


def _run(
    record: Record,
    outcome: Outcome = Outcome.HANDOVER,
    *,
    idx: int = 2,
    nodes: int = 5,
    node: str | None = "search",
    reason: str = "move did not land",
) -> None:
    record.run(
        outcome,
        idx=idx,
        nodes=nodes,
        node=node,
        reason=reason,
        micros=1,
        rescues=0,
        values={"keyword": "milk"},
        total=None,
    )


def test_a_live_walk_writes_the_event_and_the_runs_row() -> None:
    sink = Sink()
    _run(Record("demo", "flow", dry=False, events=sink))

    assert sink.events == [
        {
            "event": "walk",
            "app": "demo",
            "playbook": "flow",
            "outcome": "handover",
            "node": "search",
            "idx": 2,
            "nodes": 5,
            "reason": "move did not land",
            "micros": 1,
            "rescues": 0,
            "total": None,
        }
    ]
    assert [r["outcome"] for r in walklog.load()] == ["handover"]


def test_a_dry_walk_still_tells_the_session_but_writes_no_row() -> None:
    sink = Sink()
    _run(
        Record("channel", "boot", dry=True, events=sink),
        Outcome.COMPLETED,
        reason="hands over to demo/flow",
    )

    assert [e["reason"] for e in sink.events] == ["hands over to demo/flow"]
    assert walklog.load() == []


def test_no_sink_means_no_event_and_the_row_still_lands() -> None:
    _run(Record("demo", "flow", dry=False))

    assert len(walklog.load()) == 1


def test_the_first_terminal_moment_wins() -> None:
    sink = Sink()
    record = Record("demo", "flow", dry=False, events=sink)
    _run(record, Outcome.SUSPENDED, reason="asked")
    _run(record, Outcome.HANDOVER, reason="end_session blocked")

    assert record.outcome is Outcome.SUSPENDED
    assert [e["outcome"] for e in sink.events] == ["suspended"]
    assert len(walklog.load()) == 1


def test_a_failing_sink_never_stops_the_record() -> None:
    record = Record("demo", "flow", dry=False, events=_Broken())

    _run(record)

    assert record.outcome is Outcome.HANDOVER
    assert len(walklog.load()) == 1
