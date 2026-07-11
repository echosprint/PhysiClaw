"""Tests for `physiclaw.paths` — the data-root resolver.

Per-test isolation comes from conftest's autouse `physiclaw_home`
fixture, which monkeypatches `paths.HOME` and `paths.LOG_DIR` to a
fresh `tmp_path` subdir. The helpers reference `HOME` / `LOG_DIR` via
the module's global namespace, so the patch lands at call time.

The four import-time tests at the bottom (`test_HOME_*`,
`test_LOG_DIR_*`) intentionally `importlib.reload(paths)` so the
module re-evaluates `HOME = Path(os.environ.get(...))` and
`LOG_DIR = HOME / "log"` against a freshly-set env var. Without a
reload, mutations on those literals are masked by the autouse
fixture's monkeypatch.

`load_calibration_bundle` swallows OSError and JSONDecodeError into
None; tests exercise each branch explicitly so a `raise` ↔ `return`
mutation can't slip past.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from physiclaw import paths


# ---------- module-level invariants ----------


def test_HOME_resolves_to_per_test_tmp_path(physiclaw_home: Path) -> None:
    assert paths.HOME == physiclaw_home


def test_LOG_DIR_is_HOME_slash_log(physiclaw_home: Path) -> None:
    assert paths.LOG_DIR == physiclaw_home / "log"


# ---------- ensure_dirs ----------


def test_ensure_dirs_creates_HOME_and_LOG_DIR(physiclaw_home: Path) -> None:
    # Per-test home doesn't have LOG_DIR pre-created; ensure_dirs adds it.
    assert not (physiclaw_home / "log").exists()

    paths.ensure_dirs()

    assert physiclaw_home.is_dir()
    assert (physiclaw_home / "log").is_dir()


def test_ensure_dirs_is_idempotent(physiclaw_home: Path) -> None:
    paths.ensure_dirs()

    # No exception even though both dirs already exist.
    paths.ensure_dirs()

    assert physiclaw_home.is_dir()
    assert (physiclaw_home / "log").is_dir()


def test_ensure_dirs_creates_intermediate_parents_for_HOME(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `mkdir(parents=True)` — required because the deep parent chain
    # doesn't exist yet. parents=False would FileNotFoundError.
    deep_home = tmp_path / "a" / "b" / "c" / "physiclaw"
    monkeypatch.setattr(paths, "HOME", deep_home)
    monkeypatch.setattr(paths, "LOG_DIR", deep_home / "log")

    paths.ensure_dirs()

    assert deep_home.is_dir()


def test_ensure_dirs_creates_intermediate_parents_for_LOG_DIR(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # LOG_DIR's chain is independent of HOME's — mutations on the second
    # mkdir's parents= flag are only caught when LOG_DIR has its own
    # missing parent chain.
    home = tmp_path / "home"
    detached_log = tmp_path / "log_root" / "nested" / "log"
    monkeypatch.setattr(paths, "HOME", home)
    monkeypatch.setattr(paths, "LOG_DIR", detached_log)

    paths.ensure_dirs()

    assert detached_log.is_dir()


# ---------- HOME-rooted helpers ----------


@pytest.mark.parametrize(
    "func_name, suffix",
    [
        ("model_cache", "models"),
        ("calibration_dir", "calibration"),
        ("calibration_cache_dir", "calibration/cache"),
        ("snapshots_dir", "snapshots"),
        ("screenshots_dir", "screenshots"),
        ("tool_calls_dir", "tool_calls"),
        ("memory_dir", "memory"),
        ("skills_dir", "skills"),
        ("ui_presets_dir", "ui-presets"),
    ],
)
def test_home_rooted_helpers_resolve_under_HOME(
    physiclaw_home: Path, func_name: str, suffix: str
) -> None:
    func = getattr(paths, func_name)

    assert func() == physiclaw_home / suffix


def test_omniparser_onnx_lives_under_model_cache(physiclaw_home: Path) -> None:
    assert paths.omniparser_onnx() == (
        physiclaw_home / "models" / "omniparser_icon_detect" / "model.onnx"
    )


def test_calibration_bundle_lives_under_calibration_dir(
    physiclaw_home: Path,
) -> None:
    assert paths.calibration_bundle() == physiclaw_home / "calibration" / "bundle.json"


def test_jobs_file_lives_under_jobs_dir(physiclaw_home: Path) -> None:
    assert paths.jobs_file() == physiclaw_home / "jobs" / "jobs.md"


def test_runtime_state_file_lives_under_run_dir(physiclaw_home: Path) -> None:
    assert paths.runtime_state_file() == physiclaw_home / "run" / "server.json"


# ---------- LOG_DIR-rooted helpers ----------


@pytest.mark.parametrize(
    "func_name, suffix",
    [("claude_log_dir", "claude"), ("engine_log_dir", "engine")],
)
def test_log_rooted_helpers_resolve_under_LOG_DIR(
    physiclaw_home: Path, func_name: str, suffix: str
) -> None:
    func = getattr(paths, func_name)

    assert func() == physiclaw_home / "log" / suffix


# ---------- load_calibration_bundle ----------


def test_load_calibration_bundle_returns_none_when_file_missing(
    physiclaw_home: Path,
) -> None:
    # bundle path doesn't exist yet on a fresh per-test home.
    assert paths.load_calibration_bundle() is None


def test_load_calibration_bundle_returns_none_on_malformed_json(
    physiclaw_home: Path,
) -> None:
    bundle = paths.calibration_bundle()
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_text("{not valid json")

    assert paths.load_calibration_bundle() is None


def test_load_calibration_bundle_returns_none_when_root_is_not_dict(
    physiclaw_home: Path,
) -> None:
    bundle = paths.calibration_bundle()
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_text(json.dumps([1, 2, 3]))  # JSON valid, but a list

    assert paths.load_calibration_bundle() is None


def test_load_calibration_bundle_returns_parsed_dict_on_valid_json(
    physiclaw_home: Path,
) -> None:
    bundle = paths.calibration_bundle()
    bundle.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "screen": {"width": 1170, "height": 2532}}
    bundle.write_text(json.dumps(payload))

    assert paths.load_calibration_bundle() == payload


def test_load_calibration_bundle_returns_none_when_read_text_raises_oserror(
    physiclaw_home: Path, mocker
) -> None:
    bundle = paths.calibration_bundle()
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_text("{}")

    mocker.patch.object(Path, "read_text", side_effect=OSError("permission denied"))

    assert paths.load_calibration_bundle() is None


# ---------- import-time HOME / LOG_DIR resolution ----------


def test_HOME_at_import_reads_PHYSICLAW_HOME_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reload the module under a custom env var to verify the import-time
    # `os.environ.get("PHYSICLAW_HOME", ...)` lookup uses *that* exact key.
    custom = tmp_path / "custom_root"
    monkeypatch.setenv("PHYSICLAW_HOME", str(custom))

    importlib.reload(paths)

    assert paths.HOME == custom


def test_HOME_at_import_falls_back_to_tilde_physiclaw_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Default `~/.physiclaw` — must expanduser-expand the leading tilde.
    monkeypatch.delenv("PHYSICLAW_HOME", raising=False)

    importlib.reload(paths)

    assert paths.HOME == Path("~/.physiclaw").expanduser()


def test_LOG_DIR_at_import_is_HOME_slash_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Anchors the literal "log" subdir name and rules out a None
    # initialization.
    monkeypatch.setenv("PHYSICLAW_HOME", str(tmp_path))

    importlib.reload(paths)

    assert paths.LOG_DIR == tmp_path / "log"
