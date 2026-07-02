"""Tests for `physiclaw.agent.hooks.first_run`."""
from __future__ import annotations

import pytest

from physiclaw.agent.hooks import first_run
from physiclaw.agent.runtime.hook import Trigger


@pytest.fixture(autouse=True)
def _reset_fired(monkeypatch):
    """The hook fires at most once per process — give each test a fresh flag."""
    monkeypatch.setattr(first_run, "_fired", False)


def test_fires_once_when_layout_not_learned(monkeypatch) -> None:
    monkeypatch.setattr(first_run.screen_layout, "is_learned", lambda: False)

    first = first_run.first_run_layout()
    second = first_run.first_run_layout()

    assert isinstance(first, Trigger)
    assert first.source == "first-run"
    assert "screen-layout" in first.description
    assert second is None  # already fired — don't re-wake every tick


def test_silent_when_layout_already_learned(monkeypatch) -> None:
    monkeypatch.setattr(first_run.screen_layout, "is_learned", lambda: True)

    assert first_run.first_run_layout() is None
