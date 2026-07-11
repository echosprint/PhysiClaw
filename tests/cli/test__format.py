"""Tests for `physiclaw.cli._format`."""

from __future__ import annotations


from physiclaw.cli._format import info, next_hint, ok, section, warn


def test_ok_includes_check_prefix_and_message() -> None:
    out = ok("hello")

    assert "✓" in out
    assert "hello" in out


def test_warn_includes_bang_prefix_and_message() -> None:
    out = warn("careful")

    assert "!" in out
    assert "careful" in out


def test_next_hint_includes_next_prefix() -> None:
    out = next_hint("run something")

    assert "Next:" in out
    assert "run something" in out


def test_info_indents_with_two_spaces() -> None:
    out = info("a note")

    assert out.startswith("  ")
    assert "a note" in out


def test_section_includes_title_text() -> None:
    out = section("Status")

    assert "Status" in out


def test_ok_and_warn_use_different_prefixes() -> None:
    assert "✓" in ok("x") and "✓" not in warn("x")
    assert "!" in warn("x") and "!" not in ok("x")
