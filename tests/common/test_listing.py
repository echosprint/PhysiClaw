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
    ROW_RE,
    TEXT_ROW_RE,
    Element,
    decode_elements,
    format_elements,
    format_row,
    parse_row,
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


def test_parse_row_round_trips_format_row() -> None:
    # Region-scoped macro guards parse label + bbox back out of rows
    # (`Screen.read` → `parse_row`) — a formatter change that breaks
    # this fails here, not mid-macro.
    row = format_row(7, "text", 'He said "hi" [ok]', [0.1, 0.2, 0.3, 0.4], 0.93)

    el = parse_row(row)

    assert el is not None
    assert el.label == 'He said "hi" [ok]'
    assert el.bbox == (0.1, 0.2, 0.3, 0.4)


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


# One definition of "a legal single-line label", shared by every
# strategy here — `splitlines`-clean, exactly the `Element` invariant
# (hypothesis found \x0c when this was a mere \n\r blacklist).
_LABELS = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)), max_size=80
).filter(lambda s: "".join(s.splitlines()) == s)


@given(label=_LABELS)
def test_text_row_roundtrips_any_single_line_label(label: str) -> None:
    """Whatever OCR emits on one line — quotes, brackets, CJK — the
    parser peels the exact label back out of the composed row."""
    row = format_row(12, "text", label, [0.111, 0.222, 0.333, 0.444], 0.55)
    m = TEXT_ROW_RE.match(row)
    assert m is not None
    assert m.group(1) == label


# ---------- Element codec (text ↔ Element ↔ dict) ----------

# The `ui_elements` docstring example, plus the gnarly-label case the
# regex tests above already care about.
_ELEMENTS = [
    Element(0, "icon", "", (0.02, 0.06, 0.11, 0.10), 0.64),
    Element(1, "text", "加入购物车", (0.37, 0.54, 0.51, 0.56), 0.91),
    Element(2, "text", 'He said "hi" [ok]', (0.1, 0.2, 0.3, 0.4), 0.93),
]


def test_element_is_canonical_by_construction() -> None:
    # bbox rounds to 3 decimals, conf to 2 — the precision the text side
    # renders, which is what makes every conversion below an identity.
    e = Element(7, "text", "x", (0.12345, 0.2, 0.3, 0.4), 0.987)
    assert e.bbox == (0.123, 0.2, 0.3, 0.4)
    assert e.conf == 0.99
    assert e.row() == format_row(7, "text", "x", e.bbox, e.conf)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"kind": "input"}, "unknown kind"),
        ({"kind": "icon", "label": "settings"}, "icon's label"),
        ({"label": "two\nlines"}, "single-line"),
        ({"label": "page\x0cbreak"}, "single-line"),  # splitlines splits here too
        ({"bbox": (0.1, 0.2, 0.3)}, "left, top, right, bottom"),
        # The numeric grammar closure: either would compose a row the
        # parsers cannot match back (`^\d+` / `[0-9.]+`).
        ({"id": -1}, "non-negative integer"),
        ({"conf": -0.5}, "finite and ≥ 0"),
        ({"conf": float("nan")}, "finite and ≥ 0"),
        ({"bbox": (0.1, 0.2, 0.3, float("inf"))}, "finite"),
    ],
)
def test_element_rejects_row_grammar_violations(override: dict, match: str) -> None:
    # The constraints the row grammar implies hold by construction, so
    # the composer can never emit a row shape the parsers silently miss.
    base: dict = {
        "id": 1,
        "kind": "text",
        "label": "ok",
        "bbox": (0.1, 0.2, 0.3, 0.4),
        "conf": 0.9,
    }
    with pytest.raises(ValueError, match=match):
        Element(**{**base, **override})


def test_element_dict_round_trip() -> None:
    for e in _ELEMENTS:
        assert Element.from_dict(e.to_dict()) == e


def test_from_dict_reads_missing_or_none_label_as_empty() -> None:
    # Icon dicts historically carry `label: None`; absent works too.
    d = {"id": 2, "kind": "icon", "label": None, "bbox": [0, 0, 1, 1], "conf": 0.9}
    assert Element.from_dict(d).label == ""


def test_row_label_needs_no_escaping() -> None:
    """The grammar never interprets label bytes — no escape processing on
    either side — so backslashes, regex metacharacters, quotes, brackets,
    even a fake row tail inside the label round-trip verbatim: the greedy
    label capture plus the end anchor peel off the one real tail."""
    labels = [
        r"12 \ \n ^534*",  # what OCR reads when the screen SHOWS `\n`
        'x" [0.100,0.200,0.300,0.400] 0.99',  # tail-mimic inside the label
        '"',  # lone double-quote
        "a\\",  # backslash right before the closing quote
        "[0.1,0.2] .. ]] [[",  # bracket soup
    ]
    for label in labels:
        e = Element(3, "text", label, (0.1, 0.2, 0.3, 0.4), 0.9)
        assert parse_row(e.row()) == e


def test_parse_row_returns_none_on_non_rows() -> None:
    assert parse_row(LISTING_HEADER) is None
    assert parse_row("Payment successful") is None  # plain screen prose
    assert parse_row('3 [text] "x" [0.1,0.2] 0.90') is None  # 2-coord bbox
    assert parse_row('3 [icon] "labelled" [0.1,0.2,0.3,0.4] 0.90') is None


def test_format_elements_of_nothing_is_just_the_header() -> None:
    assert format_elements([]) == LISTING_HEADER


def test_decode_elements_is_strict() -> None:
    with pytest.raises(ValueError, match="listing header"):
        decode_elements('0 [icon] "" [0.1,0.2,0.3,0.4] 0.90')
    with pytest.raises(ValueError, match="line 2"):
        decode_elements(f"{LISTING_HEADER}\nPayment successful")


_UNIT = st.floats(min_value=0, max_value=1)
_BBOXES = st.tuples(_UNIT, _UNIT, _UNIT, _UNIT)
# Icon and text elements built separately: an icon's label is "" by the
# grammar, a text label is any legal single line.
_ELEMENTS_ST = st.lists(
    st.one_of(
        st.builds(
            Element,
            id=st.integers(min_value=0, max_value=99),
            kind=st.just("icon"),
            label=st.just(""),
            bbox=_BBOXES,
            conf=_UNIT,
        ),
        st.builds(
            Element,
            id=st.integers(min_value=0, max_value=99),
            kind=st.just("text"),
            label=_LABELS,
            bbox=_BBOXES,
            conf=_UNIT,
        ),
    ),
    max_size=5,
)


@given(elements=_ELEMENTS_ST)
def test_codec_round_trips_arbitrary_elements(elements: list[Element]) -> None:
    """decode ∘ encode is the identity on any canonical element list —
    the codec-level statement of the round-trip guarantee."""
    assert decode_elements(format_elements(elements)) == elements


# ---------- production / consumption contract pins ----------


def test_core_producer_converts_to_the_canonical_element() -> None:
    """`UIElement.to_element` — the boundary icon detection + OCR output
    crosses — lands on the canonical shape with the canonical rounding,
    both kinds. If the producer's fields drift, production and
    consumption stop agreeing here, not mid-macro."""
    # Imported here: `ui_elements` pulls the cv2-backed render module.
    from physiclaw.core.vision.ui_elements import UIElement

    icon = UIElement(0, "icon", "", [0.0213, 0.06, 0.1149, 0.10], 0.641)
    text = UIElement(1, "text", "$29.9", [0.37, 0.5407, 0.51, 0.56], 0.912)

    assert icon.to_element() == Element(0, "icon", "", (0.021, 0.06, 0.115, 0.1), 0.64)
    assert text.to_element() == Element(1, "text", "$29.9", (0.37, 0.541, 0.51, 0.56), 0.91)


def test_macro_screen_consumption_agrees_with_the_codec() -> None:
    """`Screen.read` — what region-scoped macro guards match against —
    pulls the same labels and bboxes out of a composed listing that the
    codec decodes: consumer and producer share one format."""
    from physiclaw.agent.macros.model import Screen

    screen = Screen.read(format_elements(_ELEMENTS))

    assert screen.rows == tuple(_ELEMENTS)
    assert screen.content.splitlines() == [e.label for e in _ELEMENTS]
