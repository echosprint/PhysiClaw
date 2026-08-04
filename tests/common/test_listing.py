"""Tests for `physiclaw.common.listing` — the element-listing grammar.

The contract lives in one module now, so the round-trip guarantee
(`format_row` output must parse under the matching regex and never the
other kind's) is tested next to the grammar. Three doc surfaces quote
the header verbatim for a model to read; the header stays literal
prose (not a doctrine `{{token}}` — two of the surfaces have no render
pass), so a parametrized pin holds each copy to the constant instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

import physiclaw
from physiclaw.common.listing import (
    ICON_ROW_RE,
    KINDS,
    LISTING_HEADER,
    ROW_PARSE_RE,
    ROW_RE,
    TEXT_ROW_RE,
    format_row,
)
from physiclaw.common.text import read_text

_PKG = Path(physiclaw.__file__).parent

# Every file that quotes LISTING_HEADER verbatim for a model to read.
# Grammar changes must update all of them; this pin is what enforces it.
HEADER_COPIES = [
    _PKG / "agent" / "context" / "PHYSICLAW.md",  # doctrine + MCP instructions
    _PKG / "agent" / "claude" / "CLAUDE.md",  # claude-engine doctrine
    _PKG / "core" / "server" / "tools.py",  # `peek` tool docstring
]

# ---------- header ----------


def test_listing_header_bytes_pinned() -> None:
    # Agent-visible bytes — the doctrine explains this exact line.
    assert LISTING_HEADER == 'id [kind] "label" [left,top,right,bottom] conf'


@pytest.mark.parametrize("path", HEADER_COPIES, ids=lambda p: p.name)
def test_listing_header_quoted_verbatim_in_doc_surfaces(path: Path) -> None:
    """Each doc surface quotes the header as literal prose — it is not
    a doctrine `{{token}}` (two of the three surfaces have no render
    pass) — so this pin is what keeps every copy true to the constant."""
    assert LISTING_HEADER in read_text(path)


# ---------- format_row bytes ----------


def test_format_row_text_bytes_pinned() -> None:
    # 3-decimal bbox, 2-decimal conf — agent-visible bytes.
    row = format_row(7, "text", "加入购物车", [0.5, 0.8, 0.65, 0.9], 0.987)
    assert row == '7 [text] "加入购物车" [0.500,0.800,0.650,0.900] 0.99'


def test_format_row_icon_bytes_pinned() -> None:
    row = format_row(0, "icon", "", [0.1, 0.2, 0.3, 0.4], 0.9)
    assert row == '0 [icon] "" [0.100,0.200,0.300,0.400] 0.90'


def test_row_parse_re_round_trips_format_row() -> None:
    # Region-scoped macro guards parse label + bbox back out of rows —
    # a formatter change that breaks this fails here, not mid-macro.
    row = format_row(7, "text", 'He said "hi" [ok]', [0.1, 0.2, 0.3, 0.4], 0.93)

    m = ROW_PARSE_RE.match(row)

    assert m is not None
    assert m.group(1) == 'He said "hi" [ok]'
    assert [float(v) for v in m.group(2).split(",")] == [0.1, 0.2, 0.3, 0.4]


def test_format_row_rejects_unknown_kind() -> None:
    # The kind vocabulary is closed: a new kind must extend KINDS (and
    # gain a full-shape regex) in the same edit, or composition fails
    # loudly here instead of parsers silently missing the new rows.
    with pytest.raises(ValueError, match="unknown kind 'input'"):
        format_row(1, "input", "Search", [0.1, 0.2, 0.3, 0.4], 0.9)


def test_row_re_alternation_derives_from_kinds() -> None:
    assert all(kind in ROW_RE.pattern for kind in KINDS)


# ---------- composer × parser round-trip ----------


def test_text_row_matches_text_regex_and_captures_label() -> None:
    row = format_row(3, "text", 'He said "hi" [ok]', [0.1, 0.3, 0.4, 0.35], 0.80)
    m = TEXT_ROW_RE.match(row)
    assert m is not None
    assert m.group(1) == 'He said "hi" [ok]'
    assert ICON_ROW_RE.match(row) is None
    assert ROW_RE.match(row) is not None


def test_icon_row_matches_icon_regex_only() -> None:
    row = format_row(1, "icon", "", [0.7, 0.1, 0.8, 0.2], 0.90)
    assert ICON_ROW_RE.match(row) is not None
    assert TEXT_ROW_RE.match(row) is None  # kind tag keeps the two disjoint
    assert ROW_RE.match(row) is not None


def test_header_is_not_a_row() -> None:
    assert ROW_RE.match(LISTING_HEADER) is None
    assert TEXT_ROW_RE.match(LISTING_HEADER) is None
    assert ICON_ROW_RE.match(LISTING_HEADER) is None


@given(
    label=st.text(
        alphabet=st.characters(
            blacklist_categories=("Cs",), blacklist_characters="\n\r"
        ),
        max_size=80,
    )
)
def test_text_row_roundtrips_any_single_line_label(label: str) -> None:
    """Whatever OCR emits on one line — quotes, brackets, CJK — the
    parser peels the exact label back out of the composed row."""
    row = format_row(12, "text", label, [0.111, 0.222, 0.333, 0.444], 0.55)
    m = TEXT_ROW_RE.match(row)
    assert m is not None
    assert m.group(1) == label
