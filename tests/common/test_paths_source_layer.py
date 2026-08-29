"""Tests for the source-tree pack layer (`paths.source_root` and the
search paths built on it) — the dev convenience: running physiclaw from
a checkout makes the repo's `playbooks/` (and `macros/`, if present)
loadable without an install step, while setting `PHYSICLAW_HOME` turns the
layer off at import (`paths.SOURCE_LAYER`) so tests and probes see
exactly the home they built."""

from __future__ import annotations

from physiclaw.common import paths


def test_source_layer_is_suppressed_under_physiclaw_home() -> None:
    # The whole suite runs with PHYSICLAW_HOME set (conftest), so this
    # is the suite's own isolation guarantee.
    assert paths.SOURCE_LAYER is False
    assert paths.source_root() is None
    assert paths.playbooks_dirs() == [paths.playbooks_dir()]
    assert paths.macros_dirs() == [paths.macros_dir()]


def test_source_root_finds_the_checkout_when_the_layer_is_on(monkeypatch) -> None:
    monkeypatch.setattr(paths, "SOURCE_LAYER", True)

    root = paths.source_root()

    # Tests always run from the checkout: the repo root carries both
    # markers the detector requires.
    assert root is not None
    assert (root / "pyproject.toml").exists() and (root / "playbooks").is_dir()
    assert paths.playbooks_dirs() == [paths.playbooks_dir(), root / "playbooks"]


def test_pack_root_prefers_home_then_falls_back_to_the_tree(monkeypatch) -> None:
    monkeypatch.setattr(paths, "SOURCE_LAYER", True)
    root = paths.source_root()
    assert root is not None and (root / "playbooks" / "taobao").is_dir()

    # Not installed at home → the shipped tree pack resolves.
    assert paths.pack_root("taobao") == root / "playbooks" / "taobao"

    # Installed at home → home shadows the tree.
    home_pack = paths.playbooks_dir() / "taobao"
    home_pack.mkdir(parents=True)
    (home_pack / paths.PACK_FILENAME).write_text("name: taobao\n", encoding="utf-8")
    assert paths.pack_root("taobao") == home_pack

    # Nowhere → the home dir (the write target), so error messages point
    # at where the pack WOULD live.
    assert paths.pack_root("ghost") == paths.playbooks_dir() / "ghost"
