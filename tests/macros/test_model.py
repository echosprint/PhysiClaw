"""Tests for `physiclaw.macros.model` — the shapes: display
methods, the required-input rule, and the shared name check."""

from __future__ import annotations

import pytest

from physiclaw.macros.model import (
    MacroError,
    MacroInput,
    OrClause,
    TextClause,
    check_name,
)
from physiclaw.macros.steps import GestureStep

# ---------- display ----------


def test_step_display_shows_verb_and_object() -> None:
    # The step log reads like the file: the verb and what it acts on.
    step = GestureStep(mcp_tool="send_to_clipboard", args={"text": "hi"}, name="3")
    press = GestureStep(
        mcp_tool="tap", args={"label": ["A", "B"], "bbox": []}, name="1"
    )

    assert step.display() == "send_to_clipboard 'hi'"
    assert press.display() == "tap 'A' / 'B'"
    assert GestureStep(mcp_tool="home_screen", name="2").display() == "home_screen"


def test_require_clause_display_plain_and_group() -> None:
    plain = TextClause(text="微信")
    group = OrClause(children=(TextClause(text="微信"), TextClause(text="WeChat")))

    assert plain.display() == "'微信'"
    assert group.display() == "('微信' or 'WeChat')"


def test_require_clause_display_with_region() -> None:
    clause = TextClause(text="微信", within=(0.2, 0.2, 0.8, 0.3))

    assert clause.display() == "'微信' within [0.2,0.2,0.8,0.3]"


# ---------- MacroInput.required ----------


def test_input_without_default_is_required() -> None:
    assert MacroInput(name="msg", description="d").required is True


def test_input_with_default_is_optional() -> None:
    assert MacroInput(name="msg", description="d", default="hi").required is False


# ---------- check_name ----------


def test_check_name_accepts_kebab_case() -> None:
    check_name("notify-user-wechat")  # must not raise


@pytest.mark.parametrize(
    "bad", ["Bad-Case", "a--b", "-lead", "trail-", "under_score", "a" * 65]
)
def test_check_name_rejects_bad_names(bad: str) -> None:
    with pytest.raises(MacroError, match="lowercase"):
        check_name(bad)
