"""Tests for `physiclaw.conductor.brief` — the handover/completion/boot
report renderings the drivers' final [note, peek] turns carry."""

from __future__ import annotations

import pytest

from physiclaw.conductor import brief
from physiclaw.conductor.ledger import LedgerItem


def _walk(**overrides) -> str:
    base = dict(
        app="demo",
        playbook="flow",
        node="search",
        idx=2,
        nodes=9,
        outputs={},
        ledger_items=None,
        consented=None,
    )
    return brief.walk_brief("leg did not land", **{**base, **overrides})


def test_walk_brief_carries_reason_and_position() -> None:
    text = _walk()

    assert "conductor handing over: leg did not land." in text
    assert "Walk demo/flow stopped at node search (3/9)." in text
    assert "Verify state before acting." in text


def test_walk_brief_past_the_last_node_names_the_end() -> None:
    text = _walk(node=None, idx=9)

    assert "past the last node (9/9)" in text


def test_walk_brief_includes_recorded_decisions() -> None:
    text = _walk(outputs={"choose.pick": "milk 1L"})

    assert "Decisions so far: choose.pick='milk 1L'." in text


def test_walk_brief_includes_ledger_state() -> None:
    items = [LedgerItem(query="eggs", qty=2, status="picked", label="farm eggs")]

    text = _walk(ledger_items=items)

    assert "Buying list: eggs×2(picked)." in text


def test_walk_brief_consent_line_says_payment_did_not_fire() -> None:
    text = _walk(consented=58.0)

    assert "consented to ¥58" in text
    assert "has NOT been made" in text


@pytest.mark.parametrize(
    "absent",
    ["Decisions so far", "Buying list", "consented"],
)
def test_walk_brief_omits_empty_sections(absent: str) -> None:
    text = _walk()

    assert absent not in text


def test_completion_brief_reports_done_and_wrap_up() -> None:
    text = brief.completion_brief("demo", "flow", 9, None)

    assert "walk demo/flow completed (9/9 nodes)." in text
    assert "Report the outcome to the user" in text


def test_completion_brief_includes_ledger_state() -> None:
    items = [LedgerItem(query="eggs", qty=2)]

    text = brief.completion_brief("demo", "flow", 9, items)

    assert "Buying list: eggs×2(pending)." in text


def test_boot_brief_carries_reason_and_instruction() -> None:
    text = brief.boot_brief("phone still locked after 2 unlock attempts")

    assert "conductor handing over: phone still locked" in text
    assert "Take the session from there." in text
