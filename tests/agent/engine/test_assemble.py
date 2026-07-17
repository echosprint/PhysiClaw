"""Tests for `physiclaw.agent.engine.assemble` — trigger formatting.
(The request-assembly halves are covered through the loop and lifecycle
suites; this file owns the pure units.)"""

from __future__ import annotations

from freezegun import freeze_time

from physiclaw.agent.engine.assemble import format_triggers
from physiclaw.agent.runtime.hook import Trigger


def test_format_triggers_includes_now_and_each_trigger() -> None:
    triggers = [
        Trigger(description="phone IM arrived", source="phone"),
        Trigger(description="cron fired", source="cron:user-greet"),
    ]

    with freeze_time("2026-04-28T14:30:00"):
        out = format_triggers(triggers)

    assert out.startswith("Now: 2026-04-28")
    assert "[Current wake — act on this]" in out
    assert "phone: phone IM arrived" in out
    assert "cron:user-greet: cron fired" in out


def test_format_triggers_uses_manual_for_empty_source() -> None:
    triggers = [Trigger(description="user typed", source="")]

    with freeze_time("2026-04-28T14:30:00"):
        out = format_triggers(triggers)

    assert "manual: user typed" in out


def test_format_triggers_appends_cron_context_when_provided() -> None:
    triggers = [Trigger(description="x", source="phone")]

    with freeze_time("2026-04-28T14:30:00"):
        out = format_triggers(
            triggers, cron_ctx="## Scheduled jobs firing now\n\n### foo"
        )

    assert out.endswith("### foo")


def test_format_triggers_omits_cron_section_when_blank() -> None:
    out = format_triggers([Trigger(description="x", source="phone")])

    assert "Scheduled jobs" not in out
