"""Tests for hardware/manual/build_sourcing_guide.py — the pure data logic."""

from __future__ import annotations

import pytest

from hardware.manual import BuildError
from hardware.manual import build_sourcing_guide as bsg


def bom_row(pid: str) -> dict:
    return {
        "part_id": pid,
        "component": {"en": pid},
        "spec": {"en": "spec"},
        "cls": {"en": "Frame"},
    }


# ── sync_entries ──────────────────────────────────────────────────────────────


def test_sync_entries_adds_bare_entries_for_missing_rows():
    rows = [bom_row("p1"), bom_row("p2")]

    synced, added, stale = bsg.sync_entries(rows, [{"part_id": "p1", "ref": "¥5"}])

    assert (synced, added, stale) == (
        [{"part_id": "p1", "ref": "¥5"}, {"part_id": "p2"}],
        ["p2"],
        [],
    )


def test_sync_entries_reorders_authored_entries_to_bom_order():
    rows = [bom_row("p1"), bom_row("p2")]
    entries = [{"part_id": "p2", "ref": "b"}, {"part_id": "p1", "ref": "a"}]

    synced, _, _ = bsg.sync_entries(rows, entries)

    assert [e["part_id"] for e in synced] == ["p1", "p2"]


def test_sync_entries_drops_and_reports_stale_ids():
    synced, _, stale = bsg.sync_entries([bom_row("p1")], [{"part_id": "gone"}])

    assert (synced, stale) == ([{"part_id": "p1"}], ["gone"])


def test_sync_entries_with_duplicate_entry_ids_raises():
    entries = [{"part_id": "p1"}, {"part_id": "p1"}]

    with pytest.raises(BuildError, match="duplicate part_id"):
        bsg.sync_entries([bom_row("p1")], entries)


# ── ditto_walk / supplier_columns ─────────────────────────────────────────────


def test_ditto_walk_merges_consecutive_dittos_into_the_anchor_span():
    spans, resolved = bsg.ditto_walk(
        ["¥5", "Ditto", "Ditto", "¥9"], "ref", ["a", "b", "c", "d"]
    )

    assert (spans, resolved) == ([3, 0, 0, 1], ["¥5", "¥5", "¥5", "¥9"])


def test_ditto_walk_on_the_first_row_raises_with_the_entry_id():
    with pytest.raises(BuildError, match="'ref' of entry 'p1'"):
        bsg.ditto_walk(["Ditto"], "ref", ["p1"])


def test_supplier_columns_expands_a_whole_row_ditto_to_every_slot():
    columns = bsg.supplier_columns([{"suppliers": "Ditto"}])

    assert [col[0] for col in columns] == ["Ditto", "Ditto", "Ditto"]


def test_supplier_columns_pads_short_slots_with_none():
    columns = bsg.supplier_columns([{"suppliers": [{"name": "shop"}]}])

    assert [col[0] for col in columns] == [{"name": "shop"}, None, None]


# ── clean_taobao_url ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://item.taobao.com/item.htm?abbucket=14&id=12345&mi_id=xyz"
            "&skuId=67890&spm=a21xtw.123&xxc=ad",
            "https://item.taobao.com/item.htm?id=12345&skuId=67890",
        ),
        (
            "https://item.taobao.com/item.htm?ns=1&id=98765",
            "https://item.taobao.com/item.htm?id=98765",
        ),
        ("https://shop123.taobao.com/", "https://shop123.taobao.com/"),
        (
            "https://example.com/listing?id=1&color=red",
            "https://example.com/listing?id=1&color=red",
        ),
    ],
    ids=["strips-tracking", "keeps-id-only", "shop-homepage", "non-taobao"],
)
def test_clean_taobao_url_keeps_only_identifying_params(url, expected):
    assert bsg.clean_taobao_url(url) == expected


# ── load_bom_rows ─────────────────────────────────────────────────────────────


def test_load_bom_rows_collects_rows_from_bom_pages(monkeypatch):
    pages = [{"type": "solo"}, {"type": "bom", "rows": [bom_row("p1")]}]
    monkeypatch.setattr(bsg, "load_pages", lambda: pages)

    assert bsg.load_bom_rows() == [bom_row("p1")]


def test_load_bom_rows_without_a_bom_page_raises(monkeypatch):
    monkeypatch.setattr(bsg, "load_pages", lambda: [{"type": "solo"}])

    with pytest.raises(BuildError, match="no 'bom' page rows"):
        bsg.load_bom_rows()


def test_load_bom_rows_with_missing_part_id_raises(monkeypatch):
    row = {"component": {"en": "Rail"}, "spec": {"en": "MGN9H"}}
    monkeypatch.setattr(bsg, "load_pages", lambda: [{"type": "bom", "rows": [row]}])

    with pytest.raises(BuildError, match="missing a part_id"):
        bsg.load_bom_rows()


def test_load_bom_rows_with_duplicate_part_ids_raises(monkeypatch):
    pages = [{"type": "bom", "rows": [bom_row("p1"), bom_row("p1")]}]
    monkeypatch.setattr(bsg, "load_pages", lambda: pages)

    with pytest.raises(BuildError, match="duplicate part_id"):
        bsg.load_bom_rows()
