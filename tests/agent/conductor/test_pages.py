"""Tests for `physiclaw.agent.conductor.pages` — declaration parsing and
the learned store."""

from __future__ import annotations

import pytest

from physiclaw.agent.conductor import pages
from physiclaw.agent.conductor.pages import (
    LearnedAnchor,
    LearnedPage,
    PagesError,
    parse_pages,
)
from physiclaw.common import paths

VALID = """\
results:
  anchors:
    - "综合"
    - {text: "搜索", region: top}
  forbid: ["直播中"]
  scrollable: true
item-detail:
  anchors: ["加入购物车", "立即购买"]
"""


# ---------- parse ----------


def test_parse_valid_pages() -> None:
    out = parse_pages(VALID, "taobao")

    assert set(out) == {"results", "item-detail"}
    r = out["results"]
    assert r.anchors[0].text == "综合" and r.anchors[0].region is None
    assert r.anchors[1].region == "top"
    assert r.forbid == ("直播中",)
    assert r.scrollable is True
    assert out["item-detail"].scrollable is False


def test_parse_empty_file_is_no_pages() -> None:
    assert parse_pages("", "taobao") == {}


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("results: []", "must be a mapping"),
        ("results:\n  anchors: []", "non-empty list"),
        ("results:\n  anchors: ['a']\n  bogus: 1", "unknown key"),
        ("Results:\n  anchors: ['a']", "lowercase"),
        ("results:\n  anchors: [{text: 'a', region: middle}]", "region"),
        ("results:\n  anchors: ['x']", "needs a `region`"),
        ("results:\n  anchors: [123]", "string or"),
        ("results:\n  anchors: ['ok']\n  forbid: 'nope'", "list of strings"),
        ("results:\n  anchors: ['ok']\n  scrollable: 1", "true or false"),
    ],
)
def test_parse_rejects_with_named_error(text: str, fragment: str) -> None:
    with pytest.raises(PagesError, match=fragment):
        parse_pages(text, "taobao")


def test_single_char_anchor_allowed_with_region() -> None:
    out = parse_pages("results:\n  anchors: [{text: 'x', region: top}]", "app")

    assert out["results"].anchors[0].text == "x"


def test_reserved_app_rejected_by_scan() -> None:
    with pytest.raises(PagesError, match="reserved"):
        pages.scan_app_decls("ios")


# ---------- discovery + learned store ----------


def _write_pack(app: str, text: str) -> None:
    d = paths.playbooks_dir() / app
    d.mkdir(parents=True)
    (d / "pages.yml").write_text(text, encoding="utf-8")


def test_scan_app_decls_reads_pack_or_empty() -> None:
    _write_pack("taobao", VALID)

    assert set(pages.scan_app_decls("taobao")) == {"results", "item-detail"}
    assert pages.scan_app_decls("absent") == {}


def test_scan_app_decls_validates_name_before_touching_paths() -> None:
    with pytest.raises(pages.PagesError, match="app name"):
        pages.scan_app_decls("../escape")


def test_parse_wraps_any_loader_error_as_pages_error(mocker) -> None:
    # The YAML loader does not confine itself to YAMLError (deep nesting
    # surfaces as RecursionError) — the contract is PagesError-only.
    mocker.patch.object(pages._yaml, "load", side_effect=RecursionError("deep"))

    with pytest.raises(pages.PagesError, match="invalid YAML"):
        pages.parse_pages("x: {anchors: ['a']}", "app")


def test_learned_round_trip_and_merge() -> None:
    _write_pack("taobao", VALID)
    learned = LearnedPage(
        anchors={
            "综合": LearnedAnchor(
                text="综合",
                cx=0.2,
                cy=0.11,
                pos_tol=0.02,
                freq=1.0,
                weight=0.9,
                variants=("综台",),
            )
        },
        threshold=0.55,
        observations=7,
    )
    pages.save_learned("taobao", {"results": learned})

    prints = {p.decl.name: p for p in pages.prints_for_app("taobao")}

    r = prints["results"]
    assert r.learned is not None and r.learned.observations == 7
    assert r.threshold == 0.55
    assert r.learned.anchors["综合"].variants == ("综台",)
    d = prints["item-detail"]
    assert d.learned is None
    assert d.threshold == pages.DECL_ONLY_THRESHOLD


def test_load_learned_missing_or_garbage_is_empty() -> None:
    assert pages.load_learned("nothing") == {}

    paths.learned_pages_dir().mkdir(parents=True, exist_ok=True)
    (paths.learned_pages_dir() / "bad.json").write_text("{nope", encoding="utf-8")

    assert pages.load_learned("bad") == {}
