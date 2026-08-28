"""Tests for `physiclaw.agent.engine.scratchpad` — agent working memory.

Two public functions:
  - `write(session, content)` — set/clear `session.scratchpad`, status string
  - `inject_tail(messages, content)` — append a `<scratchpad>` UserMessage

`MAX_CHARS` is a config-driven cap (CONFIG.engine.scratchpad_max_chars) — it
ships in every prompt tail, so it's a per-turn token ceiling — and it appears
in the reject message, so it gets a freeze test against config.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from physiclaw.agent.engine.scratchpad import MAX_CHARS, inject_tail, write
from physiclaw.common.config import CONFIG
from physiclaw.contract.dto import UserMessage


def test_max_chars_tracks_config() -> None:
    assert MAX_CHARS == CONFIG.engine.scratchpad_max_chars == 8192


# ---------- write ----------


def test_write_stores_content_and_returns_updated_status() -> None:
    session = SimpleNamespace(scratchpad="")

    result = write(session, "draft of an idea")

    assert session.scratchpad == "draft of an idea"
    assert result == f"scratchpad updated ({len('draft of an idea')} chars)"


def test_write_replaces_prior_content() -> None:
    session = SimpleNamespace(scratchpad="old")

    write(session, "new")

    assert session.scratchpad == "new"


@pytest.mark.parametrize("content", ["", "   ", "\n\n  \t\n"])
def test_write_whitespace_only_clears_scratchpad_and_reports_cleared(
    content: str,
) -> None:
    session = SimpleNamespace(scratchpad="prior content")

    result = write(session, content)

    assert session.scratchpad == ""
    assert result == "scratchpad cleared"


def test_write_at_exactly_MAX_CHARS_is_accepted() -> None:
    # `> MAX_CHARS` (strict). Exactly MAX_CHARS must pass; mutating to
    # `>= MAX_CHARS` would reject this boundary case.
    session = SimpleNamespace(scratchpad="")
    content = "x" * MAX_CHARS

    write(session, content)

    assert session.scratchpad == content


def test_write_above_MAX_CHARS_raises_with_size_in_message() -> None:
    session = SimpleNamespace(scratchpad="")
    too_big = "x" * (MAX_CHARS + 1)

    with pytest.raises(
        ValueError,
        match=rf"^{MAX_CHARS + 1} chars > {MAX_CHARS} cap\. Summarize before writing\.$",
    ):
        write(session, too_big)


def test_write_oversize_does_not_mutate_session() -> None:
    session = SimpleNamespace(scratchpad="kept")
    too_big = "x" * (MAX_CHARS + 1)

    with pytest.raises(ValueError):
        write(session, too_big)

    assert session.scratchpad == "kept"


# ---------- inject_tail ----------


@pytest.mark.parametrize("content", ["", "   ", "\n\n"])
def test_inject_tail_with_whitespace_only_returns_messages_unchanged(
    content: str,
) -> None:
    msgs: list = []

    out = inject_tail(msgs, content)

    assert out is msgs  # identity-preserved when no change


def test_inject_tail_appends_user_message_wrapping_content() -> None:
    msgs: list = []

    out = inject_tail(msgs, "draft text")

    assert len(out) == 1
    assert isinstance(out[0], UserMessage)
    assert out[0].content == "<scratchpad>\ndraft text\n</scratchpad>"


def test_inject_tail_does_not_mutate_original_list() -> None:
    msgs: list = []

    inject_tail(msgs, "draft text")

    assert msgs == []  # original untouched


def test_inject_tail_appends_after_existing_messages() -> None:
    earlier = UserMessage(content="hello")

    out = inject_tail([earlier], "draft")

    assert len(out) == 2
    assert out[0] is earlier
    assert out[1].content == "<scratchpad>\ndraft\n</scratchpad>"
