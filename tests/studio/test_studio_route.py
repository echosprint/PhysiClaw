"""Tests for `physiclaw.studio.route` — route assembly, live validation
through the real compiler, and playbook commit/splice."""

from __future__ import annotations

import pytest

from physiclaw.common import paths
from physiclaw.common.paths import PACK_FILENAME
from physiclaw.common.text import read_text, write_text
from physiclaw.studio import curate, record, route
from physiclaw.studio import draft as ds
from physiclaw.studio.draft import DraftError

PACK_TEXT = (
    "app: shopdemo\n"
    "description: test pack\n"
    "\n"
    "pages:\n"
    "  home:\n"
    '    anchors: ["首页"]\n'
    "  results:\n"
    '    anchors: ["综合"]\n'
    "\n"
    "playbooks:\n"
    "  handmade:\n"
    "    description: keep me byte-for-byte\n"
    "    enabled: false\n"
    "    route:\n"
    "      - page: home\n"
    "      - tell: done\n"
    '        message: "done"\n'
)

GO = {
    "do": "go",
    "macro": {"steps": [{"name": "s1", "tool": "home_screen"}]},
}


def _pack() -> None:
    d = paths.playbooks_dir() / "shopdemo"
    d.mkdir(parents=True, exist_ok=True)
    write_text(d / PACK_FILENAME, PACK_TEXT)


def _draft_with_route() -> dict:
    d = ds.load_draft("shopdemo")
    route.add_playbook(d, "walk")
    route.entry_insert(d, "walk", None, {"page": "home"})
    route.entry_insert(d, "walk", None, dict(GO))
    route.entry_insert(d, "walk", None, {"page": "results"})
    route.entry_insert(
        d, "walk", None, {"tell": "done", "message": "done — reply to continue"}
    )
    return d


# ---------- draft mutations ----------


def test_add_playbook_refuses_duplicates_and_bad_names() -> None:
    d = ds.load_draft("shopdemo")
    route.add_playbook(d, "walk")

    with pytest.raises(DraftError, match="already drafted"):
        route.add_playbook(d, "walk")
    with pytest.raises(DraftError):
        route.add_playbook(d, "Bad Name")


def test_entry_ops_insert_update_move_delete() -> None:
    d = ds.load_draft("shopdemo")
    route.add_playbook(d, "walk")
    route.entry_insert(d, "walk", None, {"page": "home"})
    route.entry_insert(d, "walk", None, {"tell": "done", "message": "x"})

    route.entry_update(d, "walk", 1, {"tell": "bye", "message": "y"})
    route.entry_move(d, "walk", 1, -1)
    r = d["playbooks"]["walk"]["route"]
    assert r[0]["tell"] == "bye" and r[1]["page"] == "home"

    route.entry_delete(d, "walk", 0)
    assert len(r) == 1
    with pytest.raises(DraftError, match="cannot move"):
        route.entry_move(d, "walk", 0, -1)


def test_entry_from_macro_embeds_inline_with_prefilled_inputs() -> None:
    d = ds.load_draft("shopdemo")
    record.add_macro(d, "search")
    record.update_macro(
        d, "search", inputs={"query": {"description": "q", "example": "x"}}
    )
    record.start_recording(d, "search")
    snap = ds.save_snap(d, "aGk=", "l")
    record.record_step(d, "send_to_clipboard", {"text": "{query}"}, snap)

    entry = route.entry_from_macro(d, "search")

    assert entry["do"] == "search"
    assert entry["macro"]["steps"][0]["tool"] == "send_to_clipboard"
    assert "snap" not in entry["macro"]["steps"][0]  # draft keys stripped
    assert entry["with"] == {"query": "{inputs.query}"}


# ---------- live validation ----------


def test_validation_compiles_a_good_route_and_names_a_bad_one() -> None:
    _pack()
    d = _draft_with_route()
    route.add_playbook(d, "broken")
    route.entry_insert(d, "broken", None, dict(GO))  # must start at a page

    v = route.validation(d)

    assert v["walk"] is None
    assert "needs a page waypoint" in v["broken"]


def test_validation_without_a_pack_says_commit_pages_first() -> None:
    d = _draft_with_route()

    v = route.validation(d)

    assert "commit pages first" in v["walk"]


# ---------- commit ----------


def test_commit_playbook_splices_one_key_and_keeps_siblings() -> None:
    _pack()
    d = _draft_with_route()

    result = route.commit_playbook("shopdemo", "walk", d["playbooks"]["walk"])

    text = read_text(paths.playbooks_dir() / "shopdemo" / PACK_FILENAME)
    assert result["check"] == "ok"
    assert "keep me byte-for-byte" in text  # the hand-authored sibling
    assert "  walk:\n" in text and "enabled: false" in text
    # Re-commit replaces in place, no duplicate key.
    route.commit_playbook("shopdemo", "walk", d["playbooks"]["walk"])
    assert (
        read_text(paths.playbooks_dir() / "shopdemo" / PACK_FILENAME).count("  walk:\n")
        == 1
    )


def test_commit_playbook_refuses_a_route_that_does_not_compile() -> None:
    _pack()
    d = ds.load_draft("shopdemo")
    route.add_playbook(d, "walk")
    route.entry_insert(d, "walk", None, dict(GO))

    with pytest.raises(DraftError, match="commit refused"):
        route.commit_playbook("shopdemo", "walk", d["playbooks"]["walk"])

    assert "walk" not in read_text(paths.playbooks_dir() / "shopdemo" / PACK_FILENAME)


@pytest.mark.parametrize(
    ("original", "must_have", "must_not"),
    [
        # replace the studio's own key, sibling untouched
        (
            "playbooks:\n  walk:\n    old: 1\n  handmade:\n    keep: 1\n",
            ["NEW-BODY", "handmade:"],
            ["old: 1"],
        ),
        # add beside a sibling
        ("playbooks:\n  handmade:\n    keep: 1\n", ["NEW-BODY", "keep: 1"], []),
        # no playbooks section at all
        ("app: a\n", ["playbooks:\n  walk:"], []),
        # the empty-stub spelling reopens as a block mapping
        ("app: a\nplaybooks: {}\n", ["playbooks:\n  walk:"], ["{}"]),
    ],
)
def test_splice_playbook(original, must_have, must_not) -> None:
    out = curate.splice_playbook(original, "walk", "walk:\n  NEW-BODY: 1\n")

    for frag in must_have:
        assert frag in out
    for frag in must_not:
        assert frag not in out


def test_splice_playbook_refuses_a_flow_style_section() -> None:
    with pytest.raises(DraftError, match="flow-style"):
        curate.splice_playbook(
            "playbooks: {other: {enabled: false}}\n", "walk", "walk:\n  x: 1\n"
        )
