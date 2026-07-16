"""Tests for hardware/assembly/mark/replay.py — op-graph walking + snapshots."""

from __future__ import annotations

import pytest

from hardware.assembly.mark import patch, replay


def op(op_id: str, preop: str, shapes: list | None = None) -> dict:
    return {"id": op_id, "preop": preop, "shapes": shapes or [], "viewBox": None}


# ── find_leaves ───────────────────────────────────────────────────────────────


def test_find_leaves_of_linear_chain_returns_last_op():
    entries = [op("aaaa", "orig"), op("bbbb", "aaaa")]

    assert replay.find_leaves(entries) == [entries[1]]


def test_find_leaves_of_branching_history_returns_every_tip():
    entries = [op("aaaa", "orig"), op("bbbb", "aaaa"), op("cccc", "aaaa")]

    assert replay.find_leaves(entries) == [entries[1], entries[2]]


def test_find_leaves_of_empty_history_returns_nothing():
    assert replay.find_leaves([]) == []


# ── chain_to ──────────────────────────────────────────────────────────────────


def test_chain_to_returns_ops_in_application_order():
    entries = [op("bbbb", "aaaa"), op("aaaa", "orig"), op("cccc", "bbbb")]

    chain = replay.chain_to(entries, "cccc")

    assert [e["id"] for e in chain] == ["aaaa", "bbbb", "cccc"]


def test_chain_to_unknown_op_raises():
    with pytest.raises(ValueError, match="unknown op id"):
        replay.chain_to([op("aaaa", "orig")], "zzzz")


def test_chain_to_cycle_raises():
    entries = [op("aaaa", "bbbb"), op("bbbb", "aaaa")]

    with pytest.raises(ValueError, match="cycle detected"):
        replay.chain_to(entries, "aaaa")


# ── apply_upto / replay_one ───────────────────────────────────────────────────

SRC_SVG = b'<svg viewBox="0 0 100 100"><g id="art"/></svg>'
RECT = {
    "type": "rect",
    "geom": {"x": 10.0, "y": 10.0, "w": 5.0, "h": 5.0},
    "color": {"fill": "#ff0000", "opacity": 0.5},
    "outlined": False,
}


def test_apply_upto_orig_returns_source_unchanged():
    assert replay.apply_upto([op("aaaa", "orig")], SRC_SVG, "orig") == SRC_SVG


def test_apply_upto_leaf_composites_every_op_in_the_chain():
    entries = [op("aaaa", "orig", [RECT]), op("bbbb", "aaaa", [RECT])]

    out = replay.apply_upto(entries, SRC_SVG, "bbbb")

    assert out.count(b"<rect x=") == 2


def test_replay_one_writes_one_snapshot_per_leaf(tmp_path, monkeypatch):
    monkeypatch.setattr(patch, "PATCH_DIR", tmp_path)
    src = tmp_path / "belt_20_clamp_exploded_cam0.svg"
    src.write_bytes(SRC_SVG)
    entries = [op("aaaa", "orig", [RECT]), op("bbbb", "orig", [RECT])]
    patch.write_patch(src, entries)

    written = replay.replay_one(src)

    assert sorted(p.name for p in written) == [
        "belt_20_clamp_exploded_cam0_aaaa.svg",
        "belt_20_clamp_exploded_cam0_bbbb.svg",
    ]


def test_replay_one_snapshot_contains_the_marked_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(patch, "PATCH_DIR", tmp_path)
    src = tmp_path / "belt_20_clamp_exploded_cam0.svg"
    src.write_bytes(SRC_SVG)
    patch.write_patch(src, [op("aaaa", "orig", [RECT])])

    (snapshot,) = replay.replay_one(src)

    assert b'<rect x="10.0000"' in snapshot.read_bytes()


def test_replay_one_without_ops_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(patch, "PATCH_DIR", tmp_path)
    src = tmp_path / "x.svg"
    src.write_bytes(SRC_SVG)

    assert replay.replay_one(src) == []
