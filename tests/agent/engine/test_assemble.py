"""Tests for `physiclaw.agent.engine.assemble` — trigger formatting and
the macro wiring of the prompt bundle. (The other request-assembly halves
are covered through the loop and lifecycle suites; this file owns the
pure units.)"""

from __future__ import annotations

from freezegun import freeze_time

from physiclaw.agent.engine.assemble import build_prompt_bundle, format_triggers
from physiclaw.agent.runtime.hook import Trigger
from physiclaw.common import paths

MACRO = """
name: notify-user
description: Ping the user
enabled: true

steps:
  - name: home-screen-1
    tool: home_screen
"""


def _write_macro(text: str = MACRO, name: str = "notify-user") -> None:
    d = paths.macros_dir() / name
    d.mkdir(parents=True)
    (d / "MACRO.yml").write_text(text, encoding="utf-8")


def test_build_prompt_bundle_wires_enabled_macros() -> None:
    _write_macro()

    bundle = build_prompt_bundle("")

    assert bundle.macro_count == 1
    assert "run_macro" in bundle.local_registry
    assert "## Available Macros" in bundle.system_prompt
    assert "**notify-user** — Ping the user" in bundle.system_prompt
    assert "## MACRO.md" in bundle.system_prompt  # doctrine rides along


def test_build_prompt_bundle_without_macros_carries_zero_macro_bytes() -> None:
    # Pay only for what's used: no section, no tool, no MACRO.md doctrine.
    bundle = build_prompt_bundle("")

    assert bundle.macro_count == 0
    assert "run_macro" not in bundle.local_registry
    assert "## Available Macros" not in bundle.system_prompt
    assert "## MACRO.md" not in bundle.system_prompt


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
