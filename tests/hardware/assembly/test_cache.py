"""Tests for hardware/assembly/cache.py — keys, layers, manifest integrity.

All directory constants are re-pointed at a per-test fake repo, so the
cache's hashing (which walks the procedure's ``hardware.*`` import closure
under ``REPO_ROOT``) and its store/restore run against tmp files only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hardware.assembly import cache

STEM = "frame_10_base"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fake repo tree: one procedure importing one helper, plus every
    output/cache directory the module writes to."""
    dirs = {
        "PROCEDURES_DIR": tmp_path / "hardware" / "assembly" / "procedures",
        "PATCH_DIR": tmp_path / "hardware" / "assembly" / "patch",
        "MARK_DIR": tmp_path / "hardware" / "assembly" / "mark",
        "STEP_DIR": tmp_path / "hardware" / "output" / "step",
        "SVG_DIR": tmp_path / "hardware" / "output" / "svg",
        "BOM_DIR": tmp_path / "hardware" / "output" / "bom",
        "CACHE_DIR": tmp_path / "hardware" / "output" / ".cache",
    }
    monkeypatch.setattr(cache, "REPO_ROOT", tmp_path)
    for name, d in dirs.items():
        d.mkdir(parents=True)
        monkeypatch.setattr(cache, name, d)

    procedure = dirs["PROCEDURES_DIR"] / f"{STEM}.py"
    procedure.write_text("import hardware.parts.helper\n")
    helper = tmp_path / "hardware" / "parts" / "helper.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("SIZE = 1\n")
    (dirs["MARK_DIR"] / "replay.py").write_text("ENGINE = 1\n")
    return SimpleNamespace(procedure=procedure, helper=helper, **dirs)


def write_outputs(repo, *, bom: bool = True, cams: tuple[int, ...] = (0,)):
    """Materialize a complete source-layer output set for STEM."""
    for variant in ("exploded", "assembled"):
        (repo.STEP_DIR / f"{STEM}_{variant}.step").write_text(f"step {variant}")
        for cam in cams:
            (repo.SVG_DIR / f"{STEM}_{variant}_cam{cam}.svg").write_text(
                f"svg {variant} {cam}"
            )
    if bom:
        (repo.BOM_DIR / f"{STEM}.md").write_text("| bom |")


# ── keys ──────────────────────────────────────────────────────────────────────


def test_source_key_is_stable_across_calls(repo):
    assert cache.source_key(STEM) == cache.source_key(STEM)


def test_source_key_changes_when_procedure_source_changes(repo):
    before = cache.source_key(STEM)

    repo.procedure.write_text("import hardware.parts.helper  # edited\n")

    assert cache.source_key(STEM) != before


def test_source_key_changes_when_imported_dependency_changes(repo):
    before = cache.source_key(STEM)

    repo.helper.write_text("SIZE = 2\n")

    assert cache.source_key(STEM) != before


def test_patch_key_changes_when_patch_json_changes(repo):
    before = cache.patch_key(STEM)

    (repo.PATCH_DIR / f"{STEM}_exploded_cam0.json").write_text("[]")

    assert cache.patch_key(STEM) != before


def test_patch_key_changes_when_replay_engine_changes(repo):
    before = cache.patch_key(STEM)

    (repo.MARK_DIR / "replay.py").write_text("ENGINE = 2\n")

    assert cache.patch_key(STEM) != before


# ── source layer: store / hit / restore ───────────────────────────────────────


def test_source_cached_after_store_is_true(repo):
    write_outputs(repo)

    cache.store_source(STEM)

    assert cache.source_cached(STEM, want_bom=True)


def test_source_cached_without_any_store_is_false(repo):
    assert not cache.source_cached(STEM, want_bom=False)


def test_restore_source_reconstructs_deleted_outputs(repo):
    write_outputs(repo)
    cache.store_source(STEM)
    (repo.STEP_DIR / f"{STEM}_exploded.step").unlink()
    (repo.SVG_DIR / f"{STEM}_exploded_cam0.svg").unlink()

    cache.restore_source(STEM)

    assert (repo.STEP_DIR / f"{STEM}_exploded.step").read_text() == "step exploded"


def test_restore_source_without_want_bom_skips_the_bom(repo):
    write_outputs(repo)
    cache.store_source(STEM)
    (repo.BOM_DIR / f"{STEM}.md").unlink()

    cache.restore_source(STEM, want_bom=False)

    assert not (repo.BOM_DIR / f"{STEM}.md").exists()


def test_source_cached_after_source_edit_is_false(repo):
    write_outputs(repo)
    cache.store_source(STEM)

    repo.procedure.write_text("import hardware.parts.helper  # edited\n")

    assert not cache.source_cached(STEM, want_bom=True)


def test_source_cached_with_stored_file_deleted_is_false(repo):
    write_outputs(repo)
    cache.store_source(STEM)

    (cache.CACHE_DIR / STEM / f"{STEM}_exploded.step").unlink()

    assert not cache.source_cached(STEM, want_bom=True)


def test_source_cached_with_stored_file_corrupted_is_false(repo):
    write_outputs(repo)
    cache.store_source(STEM)

    (cache.CACHE_DIR / STEM / f"{STEM}_exploded.step").write_text("tampered")

    assert not cache.source_cached(STEM, want_bom=True)


def test_source_cached_with_unreadable_manifest_is_false(repo):
    write_outputs(repo)
    cache.store_source(STEM)

    (cache.CACHE_DIR / STEM / "source.key").write_text("{broken")

    assert not cache.source_cached(STEM, want_bom=True)


def test_source_cached_wanting_a_bom_the_entry_lacks_is_false(repo):
    write_outputs(repo, bom=False)

    cache.store_source(STEM)

    assert not cache.source_cached(STEM, want_bom=True)


def test_source_cached_not_wanting_the_missing_bom_is_true(repo):
    write_outputs(repo, bom=False)

    cache.store_source(STEM)

    assert cache.source_cached(STEM, want_bom=False)


def test_store_source_replaces_the_layer_wholesale(repo):
    write_outputs(repo, cams=(0, 1))
    cache.store_source(STEM)
    (repo.SVG_DIR / f"{STEM}_exploded_cam1.svg").unlink()
    (repo.SVG_DIR / f"{STEM}_assembled_cam1.svg").unlink()

    cache.store_source(STEM)

    assert not (cache.CACHE_DIR / STEM / f"{STEM}_exploded_cam1.svg").exists()


# ── snapshot layer ────────────────────────────────────────────────────────────


def test_snapshots_cached_after_store_is_true(repo):
    (repo.SVG_DIR / f"{STEM}_exploded_cam0_abcd.svg").write_text("snap")

    cache.store_snapshots(STEM)

    assert cache.snapshots_cached(STEM)


def test_snapshots_cached_with_no_snapshots_is_trivially_true(repo):
    cache.store_snapshots(STEM)

    assert cache.snapshots_cached(STEM)


def test_snapshots_cached_after_patch_edit_is_false(repo):
    cache.store_snapshots(STEM)

    (repo.PATCH_DIR / f"{STEM}_exploded_cam0.json").write_text("[]")

    assert not cache.snapshots_cached(STEM)


def test_clear_snapshots_drops_snapshots_but_keeps_source(repo):
    # A patch re-replay must drop the old snapshot layer (a leaf op the
    # edit removed) without touching the cached .step / raw .svg.
    raw = repo.SVG_DIR / f"{STEM}_exploded_cam0.svg"
    raw.write_text("render")
    step = repo.STEP_DIR / f"{STEM}_exploded.step"
    step.write_text("geom")
    stale = repo.SVG_DIR / f"{STEM}_exploded_cam0_gone.svg"
    stale.write_text("stale snapshot")

    cache.clear_snapshots(STEM)

    assert not stale.exists()
    assert raw.read_text() == "render"
    assert step.read_text() == "geom"


def test_restore_snapshots_reconstructs_deleted_snapshot(repo):
    snap = repo.SVG_DIR / f"{STEM}_exploded_cam0_abcd.svg"
    snap.write_text("snap")
    cache.store_snapshots(STEM)
    snap.unlink()

    cache.restore_snapshots(STEM)

    assert snap.read_text() == "snap"


# ── output clearing & pruning ─────────────────────────────────────────────────


def test_clear_outputs_removes_only_this_stems_files(repo):
    write_outputs(repo)
    (repo.SVG_DIR / f"{STEM}_exploded_cam0.tmp").write_text("partial")
    other = repo.STEP_DIR / "belt_20_clamp_exploded.step"
    other.write_text("other")

    cache.clear_outputs(STEM)

    assert other.exists() and not list(repo.SVG_DIR.iterdir())


def test_prune_drops_entries_for_unknown_stems(repo):
    write_outputs(repo)
    cache.store_source(STEM)
    (cache.CACHE_DIR / "ghost_10_gone").mkdir()

    removed = cache.prune({STEM})

    assert (removed, (cache.CACHE_DIR / STEM).is_dir()) == (1, True)
