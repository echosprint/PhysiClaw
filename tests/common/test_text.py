"""Tests for `physiclaw.common.text` — UTF-8 (+ LF) text I/O helpers."""

from __future__ import annotations

from pathlib import Path

from physiclaw.common.text import append_text, read_text, write_text


def test_write_then_read_roundtrips_utf8(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    write_text(path, "伦敦 — 14°")

    assert read_text(path) == "伦敦 — 14°"
    # Bytes are UTF-8, not a locale codepage.
    assert path.read_bytes().decode("utf-8") == "伦敦 — 14°"


def test_append_text_utf8(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    append_text(path, "a\n")
    append_text(path, "b\n")

    assert read_text(path) == "a\nb\n"


def test_write_text_pins_lf_newlines(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    write_text(path, "line1\nline2\n")

    raw = path.read_bytes()
    assert b"\r\n" not in raw  # never CRLF, even on Windows
    assert raw == b"line1\nline2\n"
