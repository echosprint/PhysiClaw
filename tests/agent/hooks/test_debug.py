"""Tests for `physiclaw.agent.hooks.debug` — the debug wake: gated on
debug mode's env var, one-shot on fire."""

from __future__ import annotations

import json

import pytest

from physiclaw.agent.hooks.debug import debug_wake, wake_path
from physiclaw.common.config import DEBUG_ENV_VAR


def _arm(payload: str) -> None:
    wake_path().parent.mkdir(parents=True, exist_ok=True)
    wake_path().write_text(payload, encoding="utf-8")


@pytest.fixture()
def _enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DEBUG_ENV_VAR, "1")


def test_off_by_default_even_with_a_wake_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DEBUG_ENV_VAR, raising=False)
    _arm(json.dumps({"description": "x"}))

    assert debug_wake() is None
    assert wake_path().exists()  # not consumed: production ignores it whole


def test_no_file_no_trigger(_enabled) -> None:
    assert debug_wake() is None


def test_fires_with_the_description_and_consumes_the_file(_enabled) -> None:
    _arm(json.dumps({"description": "debug: user sent a message"}))

    trigger = debug_wake()

    assert trigger is not None
    assert trigger.description == "debug: user sent a message"
    assert trigger.source == "debug"
    assert not wake_path().exists()  # one-shot


def test_unreadable_file_still_fires_and_is_consumed(_enabled) -> None:
    _arm("not json")

    trigger = debug_wake()

    assert trigger is not None and trigger.description == "debug wake"
    assert not wake_path().exists()
