"""Tests for hardware/assembly/build_procedures.py — dispatcher/worker glue.

The real module drives CAD geometry (build123d / OCCT) in worker
subprocesses; these tests monkeypatch the heavy pieces (``load_class``,
``_run_one``, the ``stepcache`` module) and pin the wiring around them:
BOM failure accounting, the ``_incomplete`` completeness predicate, and
the patch-replay branch's stale-snapshot clearing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# The module under test imports the CAD stack at module level; that's
# the optional `cad` extra (uv sync --extra cad), absent on CI runners.
pytest.importorskip("build123d", reason="CAD extra not installed")

from hardware.assembly import build_procedures as bp  # noqa: E402

STEM = "frame_10_base"


# ── _build_stems: BOM failure accounting (worker mode) ────────────────────────


@pytest.fixture
def worker(monkeypatch):
    """_build_stems with the CAD-heavy pieces stubbed to instant successes."""
    monkeypatch.setattr(bp, "load_class", lambda stem: object)
    monkeypatch.setattr(
        bp,
        "_run_one",
        lambda cls, exploded, *, only_missing=False: (0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(bp, "_replay_patches_for", lambda stem, exploded: 0)


def test_bom_write_failure_is_recorded_so_the_build_returns_nonzero(
    worker, monkeypatch
):
    # A swallowed BOM failure would return 0 and let `make hw-release`
    # package output with BOM pages silently missing.
    def boom(stem, *, cumulative=False, want_delta=False):
        raise RuntimeError("disk full")

    monkeypatch.setattr(bp, "write_bom", boom)

    assert bp._build_stems([STEM], bom=True) == 1


def test_bom_write_success_keeps_the_build_green(worker, monkeypatch):
    monkeypatch.setattr(
        bp, "write_bom", lambda stem, *, cumulative=False, want_delta=False: None
    )

    assert bp._build_stems([STEM], bom=True) == 0


# ── _incomplete: STEP/SVG variants + requested BOMs ───────────────────────────


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    """A fake output tree that _missing_variants / _missing_boms resolve
    against, with the CAD-loading view-index lookup pinned to one camera."""
    dirs = SimpleNamespace(
        step=tmp_path / "step", svg=tmp_path / "svg", bom=tmp_path / "bom"
    )
    for d in vars(dirs).values():
        d.mkdir()
    monkeypatch.setattr(bp, "BOM_DIR", dirs.bom)
    monkeypatch.setattr(
        bp,
        "step_path_for",
        lambda stem, exploded: dirs.step / f"{stem}{suffix(exploded)}.step",
    )
    monkeypatch.setattr(
        bp,
        "svg_path_for",
        lambda stem, exploded, index=0: (
            dirs.svg / f"{stem}{suffix(exploded)}_cam{index}.svg"
        ),
    )
    monkeypatch.setattr(bp, "_view_indices", lambda stem, exploded: (0,))
    return dirs


def suffix(exploded: bool) -> str:
    return "_exploded" if exploded else "_assembled"


def write_variants(outputs) -> None:
    """Materialize a complete STEP + SVG set for both variants of STEM."""
    for variant in ("exploded", "assembled"):
        (outputs.step / f"{STEM}_{variant}.step").write_text("step")
        (outputs.svg / f"{STEM}_{variant}_cam0.svg").write_text("svg")


def test_incomplete_flags_a_missing_requested_bom(outputs):
    # STEP + SVG landed but the BOM never did (a write_bom-phase SIGSEGV):
    # the stem still owes its BOM, so the retry machinery must see it.
    write_variants(outputs)

    assert bp._incomplete(STEM, ("--bom",)) == ["bom"]


def test_incomplete_without_a_bom_request_ignores_the_absent_bom(outputs):
    write_variants(outputs)

    assert bp._incomplete(STEM, ()) == []


def test_incomplete_with_every_output_present_is_empty(outputs):
    write_variants(outputs)
    (outputs.bom / f"{STEM}.md").write_text("| bom |")

    assert bp._incomplete(STEM, ("--bom",)) == []


def test_incomplete_flags_a_missing_delta_bom(outputs):
    write_variants(outputs)
    (outputs.bom / f"{STEM}.md").write_text("| bom |")

    assert bp._incomplete(STEM, ("--bom", "--bom-delta")) == ["bom"]


def test_incomplete_still_reports_a_missing_variant(outputs):
    write_variants(outputs)
    (outputs.bom / f"{STEM}.md").write_text("| bom |")
    (outputs.step / f"{STEM}_exploded.step").unlink()

    assert bp._incomplete(STEM, ("--bom",)) == ["exploded"]


# ── _dispatch: the cached-geometry patch-replay branch ────────────────────────


def test_patch_replay_branch_clears_stale_snapshots_before_replaying(
    tmp_path, monkeypatch
):
    # A leaf op the patch edit removed/renamed leaves its `_<opid>.svg` on
    # disk; without clear_snapshots first, store_snapshots re-manifests it.
    events: list[tuple[str, str]] = []

    def record(name):
        return lambda stem, **kwargs: events.append((name, stem))

    fake_cache = SimpleNamespace(
        CACHE_DIR=tmp_path / "hardware" / "output" / ".cache",
        prune=lambda stems: 0,
        source_cached=lambda stem, *, want_bom: True,
        snapshots_cached=lambda stem: False,  # patch edited → re-replay
        restore_source=record("restore_source"),
        restore_snapshots=record("restore_snapshots"),
        clear_snapshots=record("clear_snapshots"),
        store_snapshots=record("store_snapshots"),
        clear_outputs=record("clear_outputs"),
        store_source=record("store_source"),
    )
    monkeypatch.setattr(bp, "stepcache", fake_cache)
    monkeypatch.setattr(bp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bp, "CRASH_LOG", tmp_path / "crash.log")
    monkeypatch.setattr(bp, "STEP_DIR", tmp_path / "step")
    monkeypatch.setattr(bp, "SVG_DIR", tmp_path / "svg")
    monkeypatch.setattr(bp, "_batches", lambda batch_size: [[STEM]])
    monkeypatch.setattr(bp, "_incomplete", lambda stem, bom_flags: [])
    monkeypatch.setattr(
        bp,
        "_replay_patches_for",
        lambda stem, exploded: events.append(("replay", stem)),
    )

    rc = bp._dispatch(1, (), use_cache=True)

    assert rc == 0
    assert [name for name, _ in events] == [
        "restore_source",
        "clear_snapshots",  # stale layer dropped BEFORE the replay …
        "replay",  # … which rewrites both variants …
        "replay",
        "store_snapshots",  # … and only then is the fresh layer stored.
    ]
