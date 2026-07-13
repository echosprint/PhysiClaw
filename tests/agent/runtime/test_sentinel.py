"""Tests for `physiclaw.agent.runtime.sentinel`."""

from __future__ import annotations

import pytest

from physiclaw.agent.runtime.sentinel import (
    STATUSES,
    parse_sentinel,
)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("DONE", "DONE"),
        ("STUCK", "STUCK"),
        ("FAIL", "FAIL"),
        ("IDLE", "IDLE"),
        ("WAIT", "WAIT"),
    ],
)
def test_status_constants(name: str, expected: str) -> None:
    from physiclaw.agent.runtime import sentinel

    assert getattr(sentinel, name) == expected


def test_statuses_frozenset_complete() -> None:
    assert STATUSES == frozenset({"DONE", "STUCK", "FAIL", "IDLE", "WAIT"})


def test_parse_sentinel_none_returns_none_empty() -> None:
    assert parse_sentinel(None) == (None, "")


def test_parse_sentinel_empty_string_returns_none_empty() -> None:
    assert parse_sentinel("") == (None, "")


def test_parse_sentinel_canonical_form_with_recap() -> None:
    assert parse_sentinel(">> DONE - completed task") == ("DONE", "completed task")


def test_parse_sentinel_single_caret() -> None:
    assert parse_sentinel("> WAIT - waiting") == ("WAIT", "waiting")


def test_parse_sentinel_lowercase_status() -> None:
    assert parse_sentinel(">> done - x") == ("DONE", "x")


def test_parse_sentinel_no_hyphen_separator() -> None:
    assert parse_sentinel(">> STUCK no hyphen") == ("STUCK", "no hyphen")


def test_parse_sentinel_bare_status_word() -> None:
    assert parse_sentinel("DONE") == ("DONE", "")


def test_parse_sentinel_lowercase_bare_status() -> None:
    assert parse_sentinel("idle") == ("IDLE", "")


def test_parse_sentinel_no_match_returns_none_and_stripped_text() -> None:
    assert parse_sentinel("nothing matches here") == (None, "nothing matches here")


def test_parse_sentinel_recap_strips_whitespace() -> None:
    assert parse_sentinel(">> FAIL -    too many retries   ") == (
        "FAIL",
        "too many retries",
    )


def test_parse_sentinel_finds_match_in_multiline_text() -> None:
    multi = "thinking out loud\n>> DONE - all good\nextra trailing"
    status, recap = parse_sentinel(multi)
    assert status == "DONE"
    assert "all good" in recap


def test_parse_sentinel_unknown_status_word_falls_through() -> None:
    assert parse_sentinel("MAYBE") == (None, "MAYBE")


def test_parse_sentinel_status_prefix_word_is_not_a_match() -> None:
    """`DONEDEAL` must not parse as DONE — the status needs a word boundary."""
    assert parse_sentinel(">> DONEDEAL - shipped") == (None, ">> DONEDEAL - shipped")


def test_parse_sentinel_waiting_prose_is_not_wait() -> None:
    assert parse_sentinel("> waiting on user") == (None, "> waiting on user")
