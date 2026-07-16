"""Tests for hardware/manual/paginate.py — page numbering + BOM pagination."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hardware.manual import paginate


def row(cls: str, component: str, **extra) -> dict:
    return {"cls": {"en": cls}, "component": {"en": component}, **extra}


def component_rows(*group_sizes: int) -> list[dict]:
    """Rows forming consecutive components of the given sizes (one class)."""
    return [
        row("Frame", f"component-{g}")
        for g, size in enumerate(group_sizes)
        for _ in range(size)
    ]


# ── assign_page_numbers ───────────────────────────────────────────────────────


def test_assign_page_numbers_numbers_pages_by_position():
    pages = [{"type": "cover", "page": "cover"}, {"type": "back", "page": "back"}]

    paginate.assign_page_numbers(pages)

    assert [p["_pageno"] for p in pages] == [1, 2]


def test_assign_page_numbers_resolves_toc_rows_to_referenced_positions():
    pages = [
        {"type": "toc", "page": "toc", "rows": [{"page": "frame_10"}]},
        {"type": "solo", "page": "frame_10"},
    ]

    paginate.assign_page_numbers(pages)

    assert pages[0]["rows"][0]["_pgno"] == 2


def test_assign_page_numbers_without_page_id_raises():
    with pytest.raises(ValueError, match="missing a 'page' id"):
        paginate.assign_page_numbers([{"type": "solo"}])


def test_assign_page_numbers_with_duplicate_page_id_raises():
    pages = [{"type": "solo", "page": "dup"}, {"type": "solo", "page": "dup"}]

    with pytest.raises(ValueError, match="duplicate page id 'dup'"):
        paginate.assign_page_numbers(pages)


def test_assign_page_numbers_with_unknown_toc_reference_raises():
    pages = [{"type": "toc", "page": "toc", "rows": [{"page": "ghost"}]}]

    with pytest.raises(ValueError, match="unknown page id 'ghost'"):
        paginate.assign_page_numbers(pages)


# ── _balanced_split / _paginate_bom ───────────────────────────────────────────


def test_balanced_split_of_empty_rows_returns_one_empty_page():
    assert paginate._balanced_split([], 16) == [[]]


def test_balanced_split_fitting_rows_returns_one_page():
    rows = component_rows(2, 2)

    assert paginate._balanced_split(rows, 16) == [rows]


def test_balanced_split_balances_instead_of_filling_greedily():
    rows = component_rows(*[2] * 10)  # 20 rows, boundaries every 2

    chunks = paginate._balanced_split(rows, 16)

    assert [len(c) for c in chunks] == [10, 10]


def test_balanced_split_never_splits_a_component():
    rows = component_rows(7, 7, 7)  # boundaries only at 7 and 14

    chunks = paginate._balanced_split(rows, 16)

    assert [len(c) for c in chunks] == [7, 14]


@given(sizes=st.lists(st.integers(1, 5), min_size=1, max_size=12))
def test_balanced_split_preserves_rows_and_component_grouping(sizes):
    rows = component_rows(*sizes)

    chunks = paginate._balanced_split(rows, 6)

    boundary_ok = all(
        chunk[0]["component"] != prev[-1]["component"]
        for prev, chunk in zip(chunks, chunks[1:])
        if prev and chunk
    )
    assert ([r for c in chunks for r in c], boundary_ok) == (rows, True)


def test_paginate_bom_break_before_forces_a_new_page():
    rows = component_rows(2, 2)
    rows[2]["break_before"] = True

    chunks = paginate._paginate_bom(rows, 16)

    assert [len(c) for c in chunks] == [2, 2]


# ── paginate_bom_pages ────────────────────────────────────────────────────────


def bom_page(n_rows: int) -> dict:
    return {
        "type": "bom",
        "page": "bom",
        "head": {"title": {"en": "Bill of materials", "zh": "物料清单"}},
        "rows": component_rows(*[1] * n_rows),
    }


def test_paginate_bom_pages_without_a_bom_page_is_a_noop():
    pages = [{"type": "solo", "page": "x"}]

    paginate.paginate_bom_pages(pages)

    assert pages == [{"type": "solo", "page": "x"}]


def test_paginate_bom_pages_splices_continuation_pages_after_the_template():
    pages = [bom_page(paginate.BOM_ROWS_PER_PAGE * 2), {"type": "back", "page": "back"}]

    paginate.paginate_bom_pages(pages)

    assert [p["page"] for p in pages] == ["bom", "bom-2", "back"]


def test_paginate_bom_pages_tags_continuation_titles_per_language():
    pages = [bom_page(paginate.BOM_ROWS_PER_PAGE * 2)]

    paginate.paginate_bom_pages(pages)

    assert pages[1]["head"]["title"] == {
        "en": "Bill of materials (cont.)",
        "zh": "物料清单（续）",
    }


def test_paginate_bom_pages_keeps_every_row_across_the_split():
    template = bom_page(paginate.BOM_ROWS_PER_PAGE * 2 + 3)
    all_rows = list(template["rows"])
    pages = [template]

    paginate.paginate_bom_pages(pages)

    assert [r for p in pages for r in p["rows"]] == all_rows
