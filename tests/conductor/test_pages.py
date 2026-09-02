"""Tests for `physiclaw.conductor.pages` — declaration parsing and
the learned store."""

from __future__ import annotations

import pytest

from physiclaw.common import paths
from physiclaw.conductor import pages
from physiclaw.conductor.pages import (
    Landmark,
    LearnedAnchor,
    LearnedPage,
    PagesError,
    parse_landmarks,
    parse_pages,
)

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
        ("results:\n  anchors: []", "non-empty"),
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


# ---------- anchor alternates ----------


def test_bare_string_anchor_has_no_alternates() -> None:
    # Every pages.yml written before alternates existed must parse to
    # exactly what it did before.
    a = parse_pages(VALID, "taobao")["results"].anchors[0]

    assert (a.text, a.alts) == ("综合", ())
    assert a.readings == ("综合",)


def test_anchor_alternates_parse_as_one_anchor() -> None:
    out = parse_pages(
        'lock:\n  anchors: [{text: ["Swipe up", "轻扫以打开"], region: top}]', "ios2"
    )
    (a,) = out["lock"].anchors

    # First reading is canonical — the learned-geometry key; the rest are
    # alternates, mirroring LearnedAnchor's text/variants split.
    assert a.text == "Swipe up"
    assert a.alts == ("轻扫以打开",)
    assert a.readings == ("Swipe up", "轻扫以打开")
    assert a.region == "top"


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("p:\n  anchors: [{text: []}]", "list is empty"),
        (
            "p:\n  anchors: [{text: ['a1','b2','c3','d4','e5']}]",
            "5 readings > max 4",
        ),
        ("p:\n  anchors: [{text: ['dup', 'dup']}]", "repeats the reading"),
        # The single-char rule is per reading — one loose alternate opens
        # the same door as one loose anchor.
        ("p:\n  anchors: [{text: ['Search', 'x']}]", "needs a `region`"),
        ("p:\n  anchors: [{text: ['ok', 123]}]", "must be a string"),
    ],
)
def test_alternates_rejected_with_named_error(text: str, fragment: str) -> None:
    with pytest.raises(PagesError, match=fragment):
        parse_pages(text, "app")


# ---------- the ios pack (scaffolded like any other) ----------


def test_ios_pack_is_scaffolded_then_read_from_disk() -> None:
    # `ios` used to be reserved-and-unloadable. It is now an ordinary
    # pack directory the user owns — scaffolded by the CLI, read from
    # disk like every other, never shipped in the wheel.
    from physiclaw.conductor import scaffold

    assert pages.scan_app_decls(pages.IOS_APP) == {}  # nothing until scaffolded

    scaffold.init_pack(pages.IOS_APP)
    decls = pages.scan_app_decls(pages.IOS_APP)

    assert "locked" in decls
    assert decls["locked"].anchors  # semantics only — geometry is captured


def test_scaffolded_ios_pages_are_matchable_prints_without_geometry() -> None:
    from physiclaw.conductor import scaffold

    scaffold.init_pack(pages.IOS_APP)

    prints = pages.prints_for_app(pages.IOS_APP)

    assert [p.page_id for p in prints] == ["ios.locked"]
    # No learned file until `calibrate` runs — declaration-only threshold.
    assert prints[0].learned is None
    assert prints[0].threshold == pages.DECL_ONLY_THRESHOLD


# ---------- discovery + learned store ----------


def _write_pack(app: str, text: str) -> None:
    from conductor_fakes import compose_pack_doc

    d = paths.playbooks_dir() / app
    d.mkdir(parents=True)
    (d / "PLAYBOOK.yml").write_text(compose_pack_doc(app, text), encoding="utf-8")


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
    from physiclaw.conductor import _spec

    mocker.patch.object(_spec.yaml_loader, "load", side_effect=RecursionError("deep"))

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


def test_parse_pages_rejects_unpopulated_placeholder() -> None:
    with pytest.raises(PagesError, match="unpopulated template placeholder.*CONTACT"):
        parse_pages('thread:\n  anchors: ["<<CONTACT>>"]\n', "channel")


# ---------- `landmarks:` (declared fixed spots) ----------


def test_parse_landmarks_happy_path() -> None:
    out = parse_landmarks(
        {
            "back": {"label": "back chevron", "bbox": [0.02, 0.05, 0.1, 0.1]},
            "dismiss": {"label": ["scrim", "empty area"], "bbox": [0.3, 0.1, 0.7, 0.2]},
            "cart": {"label": "cart", "bbox": [0.8, 0.0, 0.9, 0.1]},  # open vocabulary
        }
    )

    assert out["back"] == Landmark(label=("back chevron",), bbox=(0.02, 0.05, 0.1, 0.1))
    assert out["dismiss"].label == ("scrim", "empty area")
    assert set(out) == {"back", "dismiss", "cart"}


def test_parse_landmarks_page_scope_is_checked_against_the_pack() -> None:
    spec = {"cart": {"label": "cart", "bbox": [0.2, 0.9, 0.3, 0.95], "page": "detail"}}

    assert parse_landmarks(spec)["cart"].page == "detail"  # unchecked door
    assert parse_landmarks(spec, {"detail"})["cart"].page == "detail"
    with pytest.raises(PagesError, match="not a declared page"):
        parse_landmarks(spec, {"home"})
    with pytest.raises(PagesError, match="must be a"):
        parse_landmarks({"cart": {**spec["cart"], "extra": 1}})


def test_parse_landmarks_names_and_count_are_bounded() -> None:
    with pytest.raises(PagesError):
        parse_landmarks({"Cart Tab": {"label": "cart", "bbox": [0.8, 0.0, 0.9, 0.1]}})
    many = {f"spot{i}": {"label": "x", "bbox": [0.1, 0.1, 0.2, 0.2]} for i in range(13)}
    with pytest.raises(PagesError, match="max"):
        parse_landmarks(many)


def test_parse_landmarks_rejects_bad_shapes() -> None:
    with pytest.raises(PagesError, match="label, bbox"):
        parse_landmarks({"back": {"label": "x"}})
    with pytest.raises(PagesError, match="left < right"):
        parse_landmarks({"back": {"label": "x", "bbox": [0.9, 0.1, 0.2, 0.2]}})
    with pytest.raises(PagesError, match="duplicate"):
        parse_landmarks({"back": {"label": ["x", "x"], "bbox": [0.1, 0.1, 0.2, 0.2]}})


def test_collect_page_decls_skips_a_dotted_route_page() -> None:
    # `page: ios.locked` with anchors beside it is a playbook error (a
    # reserved built-in cannot be declared) — reported by that route's
    # own parse, never by taking the whole pack down.
    doc = {
        "playbooks": {
            "walk": {
                "route": [
                    {"page": "home", "anchors": ["Files"]},
                    {"page": "ios.locked", "anchors": ["x"]},
                ]
            }
        }
    }

    assert list(pages.collect_page_decls(doc)) == ["home"]
