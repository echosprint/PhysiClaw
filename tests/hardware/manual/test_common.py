"""Tests for hardware/manual/common.py — shared builder helpers."""

from __future__ import annotations

import pytest

from hardware.manual import common

# ── loc ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "lang", "expected"),
    [
        ({"en": "Frame", "zh": "框架"}, "zh", "框架"),
        ({"en": "Frame", "zh": "框架"}, "en", "Frame"),
        ({"en": "Frame", "zh": ""}, "zh", "Frame"),
        ({"en": "Frame"}, "zh", "Frame"),
        ("M6×16", "zh", "M6×16"),
    ],
    ids=["zh", "en", "empty-zh-falls-back", "missing-zh-falls-back", "plain-string"],
)
def test_loc_resolves_language_with_english_fallback(value, lang, expected):
    assert common.loc(value, lang) == expected


# ── _rowspans ─────────────────────────────────────────────────────────────────


def test_rowspans_marks_group_starts_with_size_and_rest_with_zero():
    rows = [{"k": "a"}, {"k": "a"}, {"k": "b"}, {"k": "a"}]

    assert common._rowspans(rows, lambda r: r["k"]) == [2, 0, 1, 1]


def test_rowspans_of_empty_rows_is_empty():
    assert common._rowspans([], lambda r: r) == []


# ── load_pages ────────────────────────────────────────────────────────────────


def test_load_pages_concatenates_content_files_in_filename_order(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "CONTENT_DIR", tmp_path)
    (tmp_path / "01_a.json").write_text('[{"page": "a"}]')
    (tmp_path / "00_front.json").write_text('[{"page": "front"}]')

    assert common.load_pages() == [{"page": "front"}, {"page": "a"}]
