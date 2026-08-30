"""Tests for `physiclaw.conductor.cards` — 2D card clustering: stacked
rows merge, columns stay apart, titles stay verbatim screen labels."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from physiclaw.common.listing import Element
from physiclaw.conductor import cards


def _el(label: str, x0: float, y0: float, x1: float, y1: float, kind: str = "text"):
    return Element(id=0, kind=kind, label=label, bbox=(x0, y0, x1, y1), conf=0.9)


TITLE = _el("acme fresh milk 1L carton", 0.05, 0.30, 0.45, 0.34)
PRICE = _el("¥12.9", 0.05, 0.35, 0.15, 0.38)
SALES = _el("1000+ sold", 0.25, 0.35, 0.45, 0.38)


def test_lone_row_is_a_single_card() -> None:
    (card,) = cards.group_cards((TITLE,))

    assert card.title is TITLE
    assert card.meta == ()


def test_stacked_rows_form_one_card_titled_by_the_longest_label() -> None:
    (card,) = cards.group_cards((TITLE, PRICE, SALES))

    assert card.title is TITLE
    assert card.meta == ("¥12.9", "1000+ sold")  # reading order


def test_side_by_side_meta_rows_join_through_the_spanning_title() -> None:
    # Price (left) and sales (right) never overlap each other — they
    # join the card via the title above them spanning both.
    (card,) = cards.group_cards((PRICE, SALES, TITLE))

    assert card.title is TITLE
    assert len(card.meta) == 2


def test_two_columns_at_the_same_height_stay_separate() -> None:
    left = _el("left column item title", 0.02, 0.30, 0.48, 0.34)
    right = _el("right column item title", 0.52, 0.30, 0.98, 0.34)

    got = cards.group_cards((left, right))

    assert [c.title.label for c in got] == [
        "left column item title",
        "right column item title",
    ]


def test_cards_split_at_the_image_gap() -> None:
    below = _el("other brand milk 1L carton", 0.05, 0.50, 0.45, 0.54)

    got = cards.group_cards((TITLE, PRICE, below))

    assert [c.title.label for c in got] == [TITLE.label, below.label]


def test_meta_is_capped_and_clipped() -> None:
    long_title = _el("t" * 70, 0.05, 0.30, 0.45, 0.34)
    fragments = tuple(
        _el(f"fact {i} " + "x" * 60, 0.05, 0.35 + i * 0.03, 0.45, 0.36 + i * 0.03)
        for i in range(6)
    )

    (card,) = cards.group_cards((long_title,) + fragments)

    assert card.title is long_title
    assert len(card.meta) == cards.MAX_META
    assert all(len(m) <= cards.META_CLIP for m in card.meta)
    assert card.meta[0].endswith("…")


def test_icon_and_unlabeled_rows_are_ignored() -> None:
    icon = _el("", 0.05, 0.31, 0.08, 0.33, kind="icon")
    blank = _el("   ", 0.05, 0.35, 0.45, 0.38)

    (card,) = cards.group_cards((TITLE, icon, blank))

    assert card.title is TITLE and card.meta == ()


def test_cards_ordered_top_to_bottom_then_left_to_right() -> None:
    top_right = _el("top right item title", 0.52, 0.10, 0.98, 0.14)
    mid_left = _el("mid left item title", 0.02, 0.30, 0.48, 0.34)
    top_left = _el("top left item title", 0.02, 0.10, 0.48, 0.14)

    got = cards.group_cards((top_right, mid_left, top_left))

    assert [c.title.label for c in got] == [
        "top left item title",
        "top right item title",
        "mid left item title",
    ]


@given(
    st.lists(
        st.builds(
            lambda label, x0, y0, w, h: _el(label, x0, y0, x0 + w, y0 + h),
            label=st.text(
                alphabet=st.characters(whitelist_categories=("L", "N")),
                min_size=1,
                max_size=8,
            ),
            x0=st.floats(0, 0.8),
            y0=st.floats(0, 0.9),
            w=st.floats(0.05, 0.2),
            h=st.floats(0.01, 0.05),
        ),
        max_size=8,
    )
)
def test_every_title_is_a_verbatim_input_row(rows) -> None:
    got = cards.group_cards(tuple(rows))

    titles = [c.title for c in got]
    assert all(t in rows for t in titles)  # identity, never a rewrite
    assert len(titles) == len(set(id(t) for t in titles))  # no row twice
