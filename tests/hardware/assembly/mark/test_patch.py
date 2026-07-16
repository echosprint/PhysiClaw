"""Tests for hardware/assembly/mark/patch.py — patch persistence + op ids."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hardware import scheme
from hardware.assembly.mark import patch


@pytest.fixture
def patch_dir(tmp_path, monkeypatch):
    """Point the module's PATCH_DIR at a per-test directory."""
    d = tmp_path / "patch"
    d.mkdir()
    monkeypatch.setattr(patch, "PATCH_DIR", d)
    return d


# ── path scheme ───────────────────────────────────────────────────────────────


def test_snapshot_path_appends_op_id_before_suffix():
    out = patch.snapshot_path(Path("/out/belt_20_clamp_exploded_cam0.svg"), "abcd")

    assert out == Path("/out/belt_20_clamp_exploded_cam0_abcd.svg")


def test_snapshot_path_name_matches_scheme_snapshot_regex():
    out = patch.snapshot_path(
        scheme.svg_path_for("belt_20_clamp", exploded=True), "abcd"
    )

    assert scheme.snapshot_svg_re("belt_20_clamp").match(out.name)


def test_patch_path_maps_svg_stem_to_patch_json(patch_dir):
    out = patch.patch_path(Path("/out/belt_20_clamp_exploded_cam0.svg"))

    assert out == patch_dir / "belt_20_clamp_exploded_cam0.json"


def test_source_for_patch_inverts_patch_path(patch_dir):
    src = Path("/out/belt_20_clamp_exploded_cam0.svg")

    back = patch.source_for_patch(patch.patch_path(src))

    assert back == patch.SVG_DIR / src.name


# ── op ids ────────────────────────────────────────────────────────────────────


def test_new_id_skips_the_orig_sentinel(tmp_path, mocker):
    mocker.patch.object(
        patch.random, "choices", side_effect=[list("orig"), list("abcd")]
    )

    assert patch.new_id(tmp_path / "x.svg") == "abcd"


def test_new_id_skips_taken_ids(tmp_path, mocker):
    mocker.patch.object(
        patch.random, "choices", side_effect=[list("aaaa"), list("bbbb")]
    )

    assert patch.new_id(tmp_path / "x.svg", taken={"aaaa"}) == "bbbb"


def test_new_id_skips_ids_with_an_existing_snapshot(tmp_path, mocker):
    src = tmp_path / "x.svg"
    patch.snapshot_path(src, "aaaa").write_text("")
    mocker.patch.object(
        patch.random, "choices", side_effect=[list("aaaa"), list("bbbb")]
    )

    assert patch.new_id(src) == "bbbb"


@pytest.mark.parametrize("raw", ["orig", "abcd", "zzzz"])
def test_validate_preop_accepts_sentinel_and_four_lowercase(raw):
    assert patch.validate_preop(raw) == raw


@pytest.mark.parametrize("raw", ["abc", "abcde", "ABCD", "ab1d", None, 4])
def test_validate_preop_rejects_malformed_input(raw):
    with pytest.raises(ValueError, match="four lowercase letters"):
        patch.validate_preop(raw)


# ── load / write ──────────────────────────────────────────────────────────────


def test_load_patch_without_file_returns_empty_list(patch_dir):
    assert patch.load_patch(Path("belt_20_clamp_exploded_cam0.svg")) == []


def test_load_patch_with_empty_file_returns_empty_list(patch_dir):
    (patch_dir / "x.json").write_text("")

    assert patch.load_patch(Path("x.svg")) == []


def test_load_patch_round_trips_write_patch(patch_dir):
    entries = [{"id": "abcd", "preop": "orig", "shapes": [], "viewBox": None}]
    patch.write_patch(Path("x.svg"), entries)

    assert patch.load_patch(Path("x.svg")) == entries


def test_load_patch_with_invalid_json_raises(patch_dir):
    (patch_dir / "x.json").write_text("{nope")

    with pytest.raises(ValueError, match="isn't valid JSON"):
        patch.load_patch(Path("x.svg"))


def test_load_patch_with_non_array_json_raises(patch_dir):
    (patch_dir / "x.json").write_text('{"id": "abcd"}')

    with pytest.raises(ValueError, match="not a JSON array"):
        patch.load_patch(Path("x.svg"))


# ── entry construction ────────────────────────────────────────────────────────


def test_make_entry_carries_id_preop_shapes_viewbox():
    entry = patch.make_entry("abcd", "orig", [{"type": "rect", "geom": {}}], "0 0 1 1")

    assert entry == {
        "id": "abcd",
        "preop": "orig",
        "shapes": [{"type": "rect", "geom": {}}],
        "viewBox": "0 0 1 1",
    }


def test_make_entry_copies_polygon_points_instead_of_aliasing():
    points = [(1, 2), (3, 4)]
    shape = {"type": "polygon", "geom": {"points": points}}

    entry = patch.make_entry("abcd", "orig", [shape], None)
    points[0] = (9, 9)

    assert entry["shapes"][0]["geom"]["points"] == [[1, 2], [3, 4]]


def test_make_entry_polygon_points_survive_json_round_trip():
    shape = {"type": "polygon", "geom": {"points": [(1, 2)]}}

    entry = patch.make_entry("abcd", "orig", [shape], None)

    assert json.loads(json.dumps(entry)) == entry


def test_upsert_entry_with_edit_id_replaces_in_place():
    entries = [
        patch.make_entry("aaaa", "orig", [], None),
        patch.make_entry("bbbb", "aaaa", [], None),
    ]

    op_id, updated = patch.upsert_entry(
        Path("x.svg"), entries, "aaaa", "orig", [{"type": "rect", "geom": {}}], None
    )

    assert (op_id, updated[0]["shapes"], updated[1]) == (
        "aaaa",
        [{"type": "rect", "geom": {}}],
        entries[1],
    )


def test_upsert_entry_without_edit_id_appends_with_fresh_id(tmp_path):
    entries = [patch.make_entry("aaaa", "orig", [], None)]

    op_id, updated = patch.upsert_entry(
        tmp_path / "x.svg", entries, None, "aaaa", [], None
    )

    assert (len(updated), updated[-1]["id"], updated[-1]["preop"]) == (2, op_id, "aaaa")


@given(op_id=st.from_regex(r"[a-z]{4}", fullmatch=True))
def test_validate_preop_accepts_any_four_lowercase_letters(op_id):
    assert patch.validate_preop(op_id) == op_id
