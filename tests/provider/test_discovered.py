"""Tests for `physiclaw.provider.discovered` — model-list cache.

Module-level `_DIR = paths.HOME / "discovered"` is captured at import
time; the autouse `_dir_fixture` re-points it to a per-test directory
so writes don't bleed. The two import-time tests at the bottom use
`importlib.reload` to validate the literal "discovered" subdir name.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from physiclaw.provider import discovered


@pytest.fixture(autouse=True)
def _dir_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test cache root — replaces `discovered._DIR`."""
    cache_root = tmp_path / "discovered"
    monkeypatch.setattr(discovered, "_DIR", cache_root)
    return cache_root


# ---------- cache_path ----------


def test_cache_path_lives_under_underscore_DIR(_dir_fixture: Path) -> None:
    assert discovered.cache_path("openai") == _dir_fixture / "openai.json"


# ---------- save ----------


def test_save_writes_models_with_fetched_at_timestamp(_dir_fixture: Path) -> None:
    models = [{"id": "gpt-5"}, {"id": "gpt-4"}]

    discovered.save("openai", models)

    payload = json.loads((_dir_fixture / "openai.json").read_text())
    assert payload["models"] == models
    # ISO 8601 with seconds precision and UTC timezone
    assert payload["fetched_at"].endswith("+00:00")
    assert "T" in payload["fetched_at"]


def test_save_creates_parent_directory_when_missing(_dir_fixture: Path) -> None:
    assert not _dir_fixture.exists()

    discovered.save("openai", [])

    assert _dir_fixture.is_dir()


def test_save_creates_intermediate_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deep = tmp_path / "a" / "b" / "c" / "discovered"
    monkeypatch.setattr(discovered, "_DIR", deep)

    discovered.save("openai", [{"id": "gpt-5"}])

    assert (deep / "openai.json").is_file()


def test_save_serializes_with_two_space_indent(_dir_fixture: Path) -> None:
    discovered.save("openai", [{"id": "gpt-5"}])

    text = (_dir_fixture / "openai.json").read_text()
    # Byte-for-byte equality with json.dumps(indent=2) — kills any
    # mutation on the indent argument.
    import json as _json

    expected = _json.dumps(_json.loads(text), indent=2)
    assert text == expected


def test_save_is_idempotent_when_called_twice(_dir_fixture: Path) -> None:
    # `mkdir(parents=True, exist_ok=True)` — exist_ok=True must hold so
    # a second save doesn't raise FileExistsError on the existing dir.
    discovered.save("openai", [{"id": "first"}])

    discovered.save("openai", [{"id": "second"}])  # must not raise

    assert discovered.load("openai") == [{"id": "second"}]


def test_save_swallows_oserror_and_logs_warning(
    _dir_fixture: Path,
    mocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mocker.patch.object(Path, "write_text", side_effect=OSError("disk full"))

    with caplog.at_level(logging.WARNING, logger="physiclaw.provider.discovered"):
        # Must not raise.
        discovered.save("openai", [{"id": "gpt-5"}])

    assert any(
        r.getMessage().startswith("failed to write discovered cache for openai:")
        for r in caplog.records
    )


# ---------- load ----------


def test_load_returns_empty_list_when_cache_file_missing() -> None:
    assert discovered.load("openai") == []


def test_load_returns_models_list_when_cache_present(_dir_fixture: Path) -> None:
    discovered.save("openai", [{"id": "gpt-5"}, {"id": "gpt-4"}])

    assert discovered.load("openai") == [{"id": "gpt-5"}, {"id": "gpt-4"}]


def test_load_returns_empty_list_when_models_key_absent(_dir_fixture: Path) -> None:
    _dir_fixture.mkdir(parents=True)
    (_dir_fixture / "openai.json").write_text(
        json.dumps({"fetched_at": "2026-04-28T00:00:00+00:00"})
    )

    assert discovered.load("openai") == []


def test_load_returns_empty_when_models_key_is_null(_dir_fixture: Path) -> None:
    # `payload.get("models") or []` — null in JSON becomes None in Python,
    # falsy, so the `or []` fallback fires.
    _dir_fixture.mkdir(parents=True)
    (_dir_fixture / "openai.json").write_text(
        json.dumps({"fetched_at": "2026-04-28T00:00:00+00:00", "models": None})
    )

    assert discovered.load("openai") == []


def test_load_returns_empty_and_logs_on_malformed_json(
    _dir_fixture: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _dir_fixture.mkdir(parents=True)
    (_dir_fixture / "openai.json").write_text("{not valid json")

    with caplog.at_level(logging.WARNING, logger="physiclaw.provider.discovered"):
        out = discovered.load("openai")

    assert out == []
    assert any(
        r.getMessage().startswith("failed to read discovered cache for openai:")
        for r in caplog.records
    )


def test_load_returns_empty_and_logs_on_oserror(
    _dir_fixture: Path,
    mocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _dir_fixture.mkdir(parents=True)
    (_dir_fixture / "openai.json").write_text("{}")
    mocker.patch.object(Path, "read_text", side_effect=OSError("permission denied"))

    with caplog.at_level(logging.WARNING, logger="physiclaw.provider.discovered"):
        out = discovered.load("openai")

    assert out == []
    assert any(
        r.getMessage().startswith("failed to read discovered cache for openai:")
        for r in caplog.records
    )


# ---------- model_ids ----------


def test_model_ids_returns_set_of_ids(_dir_fixture: Path) -> None:
    discovered.save("openai", [{"id": "gpt-5"}, {"id": "gpt-4"}])

    assert discovered.model_ids("openai") == {"gpt-5", "gpt-4"}


def test_model_ids_filters_empty_string_ids(_dir_fixture: Path) -> None:
    # Defensive — if a provider returns a model with empty id, it must
    # not pollute the membership check.
    discovered.save("openai", [{"id": "gpt-5"}, {"id": ""}, {"name": "no-id-key"}])

    assert discovered.model_ids("openai") == {"gpt-5"}


def test_model_ids_returns_empty_set_when_no_cache() -> None:
    assert discovered.model_ids("openai") == set()


# ---------- is_cached ----------


def test_is_cached_true_when_model_in_cache(_dir_fixture: Path) -> None:
    discovered.save("openai", [{"id": "gpt-5"}])

    assert discovered.is_cached("openai", "gpt-5") is True


def test_is_cached_false_when_model_absent(_dir_fixture: Path) -> None:
    discovered.save("openai", [{"id": "gpt-5"}])

    assert discovered.is_cached("openai", "gpt-9000") is False


def test_is_cached_false_when_no_cache_at_all() -> None:
    assert discovered.is_cached("openai", "gpt-5") is False


# ---------- import-time _DIR resolution ----------


def test_dir_at_import_lives_under_paths_HOME(physiclaw_home: Path) -> None:
    # Reload the module to recompute `_DIR = paths.HOME / "discovered"`
    # against the per-test paths.HOME. Pin the literal "discovered"
    # subdir name — kills `"discovered"` ↔ `"XXdiscoveredXX"` and
    # `_DIR = None` mutations.
    import importlib

    importlib.reload(discovered)

    assert discovered._DIR == physiclaw_home / "discovered"
    assert discovered._DIR.name == "discovered"
