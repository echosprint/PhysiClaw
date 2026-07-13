"""Tests for `physiclaw.agent.engine.pitfalls` — the always-on, append-only
learned-pitfalls list.

The autouse `physiclaw_home` fixture points `paths.HOME` at a per-test tmp dir,
so `paths.pitfalls_dir()` resolves under it.
"""

from __future__ import annotations

import json

import pytest

from physiclaw.agent.engine import pitfalls
from physiclaw.common import config, paths


def _history() -> list[dict]:
    path = paths.pitfalls_dir() / "history.jsonl"
    return [
        json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln
    ]


# ---------- helpers ----------


def test_clean_clamps_to_item_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config.CONFIG.pitfalls, "max_item_chars", 20)
    out = pitfalls._clean("x" * 100)
    assert len(out) == 20 and out.endswith("…")


def test_dedup_drops_blanks_and_normalized_dupes() -> None:
    assert pitfalls._dedup([" A ", "a", "", "  ", "B"]) == ["A", "B"]


# ---------- add: append-only, newest on top ----------


def test_add_prepends_newest_on_top() -> None:
    pitfalls.add(["京东: trap one"])
    res = pitfalls.add(["wechat: trap two"])
    assert res["added"] == 1 and res["total"] == 2
    assert pitfalls.read() == ["wechat: trap two", "京东: trap one"]  # newest first


def test_add_caps_to_three_per_call() -> None:
    res = pitfalls.add(["a", "b", "c", "d", "e"])
    assert res["added"] == 3
    assert pitfalls.read() == ["a", "b", "c"]


def test_add_dedups_against_existing() -> None:
    pitfalls.add(["京东: avoid Ai搜索"])
    res = pitfalls.add(
        ["京东: Avoid  AI搜索".replace("AI", "Ai"), "new one"]
    )  # first is a dupe
    assert res["added"] == 1  # only "new one"
    assert pitfalls.read().count("京东: avoid Ai搜索") == 1


def test_add_hard_caps_from_the_bottom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config.CONFIG.pitfalls, "max_items", 3)
    pitfalls.add(["a", "b", "c"])  # a on top, c at the bottom (oldest)
    pitfalls.add(["d"])  # prepends d → [d,a,b,c], cut top 3 drops the bottom (c)
    assert pitfalls.read() == ["d", "a", "b"]


def test_add_empty_is_noop() -> None:
    pitfalls.add(["real"])
    res = pitfalls.add(["   ", ""])
    assert res["added"] == 0 and res["total"] == 1


# ---------- replace: curator path ----------


def test_replace_swaps_list_and_cuts_top(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config.CONFIG.pitfalls, "max_items", 2)
    pitfalls.add(["old"])
    res = pitfalls.replace(["x", "y", "z"])  # 3 → cut to top 2
    assert res == {"total": 2, "dropped": 1}
    assert pitfalls.read() == ["x", "y"]


def test_replace_empty_does_not_wipe() -> None:
    pitfalls.add(["keep me"])
    pitfalls.replace([])
    assert pitfalls.read() == ["keep me"]


# ---------- history + render ----------


def test_every_write_snapshots_history() -> None:
    pitfalls.add(["a"])
    pitfalls.replace(["a", "b"])
    hist = _history()
    assert [h["op"] for h in hist] == ["add", "curate"]
    assert hist[-1]["items"] == ["a", "b"]


def test_render_section_is_empty_when_no_pitfalls() -> None:
    assert pitfalls.render_section() == ""


def test_render_section_has_header_doctrine_and_items() -> None:
    pitfalls.add(["京东: avoid Ai搜索"])
    out = pitfalls.render_section()
    assert out.startswith("## Learned pitfalls")
    assert "don't repeat" in out
    assert "- 京东: avoid Ai搜索" in out
