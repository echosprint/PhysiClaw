"""Tests for `physiclaw.cli._update_check` — soft PyPI version check.

Network is mocked at the ``urllib.request.urlopen`` boundary; the
``physiclaw_home`` autouse fixture (see ``tests/conftest.py``) gives
each test a clean ``paths.HOME`` so the cache file is per-test.
"""
from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from physiclaw.cli import _update_check as uc


# ---------- _is_newer comparator ----------


@pytest.mark.parametrize(
    "current,latest,expected",
    [
        ("0.0.5", "0.1.0", True),
        ("0.0.5", "0.0.10", True),     # numeric, not lexical
        ("0.0.5", "0.0.5", False),
        ("0.1.0", "0.0.9", False),
        ("1.0.0", "1.0.0", False),
        ("0.0.5", "1.0.0", True),
    ],
)
def test_is_newer_dotted_versions(current: str, latest: str, expected: bool) -> None:
    assert uc._is_newer(current, latest) is expected


@pytest.mark.parametrize(
    "current,latest",
    [
        ("0.0.5", "0.1.0a1"),       # pre-release suffix → bail
        ("0.0.5", "0.1.0+local"),   # local version → bail
        ("0.0.5", ""),
    ],
)
def test_is_newer_returns_false_on_non_numeric(current: str, latest: str) -> None:
    # Conservative: don't nudge on versions we can't safely compare.
    assert uc._is_newer(current, latest) is False


# ---------- env disable ----------


@pytest.mark.parametrize("val", ["1", "true", "yes", "TRUE", "Yes"])
def test_env_disable_skips_check(val: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHYSICLAW_DISABLE_UPDATE_CHECK", val)
    assert uc._disabled_via_env() is True


@pytest.mark.parametrize("val", ["", "0", "false", "no", " "])
def test_env_disable_inactive_for_other_values(
    val: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHYSICLAW_DISABLE_UPDATE_CHECK", val)
    assert uc._disabled_via_env() is False


def test_env_disable_short_circuits_banner(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setenv("PHYSICLAW_DISABLE_UPDATE_CHECK", "1")
    # Even if PyPI would say "newer", the env disable wins before any I/O.
    fake_resolve = MagicMock(return_value="99.99.99")
    monkeypatch.setattr(uc, "_resolve_latest", fake_resolve)

    uc.maybe_print_update_banner()

    fake_resolve.assert_not_called()
    assert capsys.readouterr().out == ""


# ---------- TTY guard ----------


def test_non_tty_skips_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHYSICLAW_DISABLE_UPDATE_CHECK", raising=False)
    fake_stdout = io.StringIO()  # io.StringIO.isatty() returns False
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    fake_resolve = MagicMock(return_value="99.99.99")
    monkeypatch.setattr(uc, "_resolve_latest", fake_resolve)

    uc.maybe_print_update_banner()

    fake_resolve.assert_not_called()
    assert fake_stdout.getvalue() == ""


# ---------- cache freshness ----------


def test_cache_is_fresh_within_ttl() -> None:
    recent = datetime.now(timezone.utc) - timedelta(days=3)
    assert uc._cache_is_fresh(recent.isoformat()) is True


def test_cache_is_stale_past_ttl() -> None:
    old = datetime.now(timezone.utc) - timedelta(days=8)
    assert uc._cache_is_fresh(old.isoformat()) is False


def test_cache_is_fresh_handles_naive_timestamp() -> None:
    naive = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None)
    # Treats naive as UTC (forward-compatible).
    assert uc._cache_is_fresh(naive.isoformat()) is True


def test_cache_is_fresh_returns_false_on_garbage() -> None:
    assert uc._cache_is_fresh("not-a-timestamp") is False
    assert uc._cache_is_fresh("") is False


# ---------- cache read / write ----------


def test_write_then_read_cache_roundtrip(physiclaw_home: Path) -> None:
    uc._write_cache("1.2.3")
    cache = uc._read_cache()
    assert cache is not None
    assert cache["latest_version"] == "1.2.3"
    # checked_at is ISO-format; parses cleanly.
    assert datetime.fromisoformat(cache["checked_at"]).tzinfo is not None


def test_read_cache_returns_none_when_missing(physiclaw_home: Path) -> None:
    # No cache file written.
    assert uc._read_cache() is None


def test_read_cache_returns_none_on_corrupt_json(physiclaw_home: Path) -> None:
    cache_file = physiclaw_home / "run" / "version-check.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("not json {")
    assert uc._read_cache() is None


def test_read_cache_returns_none_when_top_level_is_not_object(
    physiclaw_home: Path,
) -> None:
    cache_file = physiclaw_home / "run" / "version-check.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps([1, 2, 3]))
    assert uc._read_cache() is None


def test_write_cache_swallows_oserror(
    physiclaw_home: Path, mocker,
) -> None:
    mocker.patch.object(Path, "write_text", side_effect=OSError("read-only"))
    # Must not raise.
    uc._write_cache("9.9.9")


# ---------- _resolve_latest: cache vs. network ----------


def test_resolve_latest_uses_fresh_cache(
    physiclaw_home: Path, mocker,
) -> None:
    uc._write_cache("0.0.7")
    fetch_spy = mocker.patch.object(uc, "_fetch_pypi_version")

    assert uc._resolve_latest() == "0.0.7"
    fetch_spy.assert_not_called()


def test_resolve_latest_refetches_when_stale(
    physiclaw_home: Path, mocker,
) -> None:
    # Manually plant a stale cache.
    cache_file = physiclaw_home / "run" / "version-check.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({
        "checked_at": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
        "latest_version": "0.0.4",
    }))
    mocker.patch.object(uc, "_fetch_pypi_version", return_value="0.0.9")

    assert uc._resolve_latest() == "0.0.9"
    # Cache has been updated with the new fetch.
    refreshed = uc._read_cache()
    assert refreshed["latest_version"] == "0.0.9"


def test_resolve_latest_returns_none_when_pypi_unreachable(
    physiclaw_home: Path, mocker,
) -> None:
    mocker.patch.object(uc, "_fetch_pypi_version", return_value=None)
    assert uc._resolve_latest() is None


def test_resolve_latest_falls_back_to_network_on_corrupt_cache(
    physiclaw_home: Path, mocker,
) -> None:
    cache_file = physiclaw_home / "run" / "version-check.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("garbage")
    mocker.patch.object(uc, "_fetch_pypi_version", return_value="2.0.0")

    assert uc._resolve_latest() == "2.0.0"


# ---------- _fetch_pypi_version ----------


def test_fetch_pypi_version_extracts_info_version(mocker) -> None:
    fake_resp = MagicMock()
    fake_resp.read.return_value = json.dumps({"info": {"version": "0.1.5"}}).encode()
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda self, *a: None
    mocker.patch("urllib.request.urlopen", return_value=fake_resp)

    assert uc._fetch_pypi_version() == "0.1.5"


def test_fetch_pypi_version_returns_none_on_http_error(mocker) -> None:
    mocker.patch(
        "urllib.request.urlopen",
        side_effect=OSError("network unreachable"),
    )
    assert uc._fetch_pypi_version() is None


def test_fetch_pypi_version_returns_none_on_unexpected_payload(mocker) -> None:
    fake_resp = MagicMock()
    fake_resp.read.return_value = b'{"unexpected": "shape"}'
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda self, *a: None
    mocker.patch("urllib.request.urlopen", return_value=fake_resp)

    assert uc._fetch_pypi_version() is None


# ---------- end-to-end maybe_print_update_banner ----------
#
# We patch ``sys.stdout.isatty`` and ``builtins.print`` directly because
# pytest's default fd-level output capture intercepts real print() calls
# even when ``sys.stdout`` is monkeypatched at the Python level — there's
# no clean way to verify a print's content from the captured side
# without disabling capture (-s).


@pytest.fixture
def banner_env(monkeypatch: pytest.MonkeyPatch, mocker) -> MagicMock:
    """Disable env override, force isatty=True, and intercept builtins.print.

    Yields the print mock so each test can inspect call args.
    """
    monkeypatch.delenv("PHYSICLAW_DISABLE_UPDATE_CHECK", raising=False)
    mocker.patch.object(sys.stdout, "isatty", return_value=True)
    return mocker.patch("builtins.print")


def test_banner_prints_when_newer(
    banner_env: MagicMock, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(uc, "_pkg_version", "0.0.5")
    monkeypatch.setattr(uc, "_resolve_latest", lambda: "1.2.3")

    uc.maybe_print_update_banner()

    banner_env.assert_called_once()
    msg = banner_env.call_args.args[0]
    assert "physiclaw 0.0.5 → 1.2.3 available" in msg
    assert "uv tool upgrade physiclaw" in msg


def test_banner_silent_when_at_latest(
    banner_env: MagicMock, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(uc, "_pkg_version", "1.2.3")
    monkeypatch.setattr(uc, "_resolve_latest", lambda: "1.2.3")

    uc.maybe_print_update_banner()

    banner_env.assert_not_called()


def test_banner_silent_when_pypi_unreachable(
    banner_env: MagicMock, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(uc, "_resolve_latest", lambda: None)

    uc.maybe_print_update_banner()

    banner_env.assert_not_called()


def test_banner_silent_when_local_is_newer_than_pypi(
    banner_env: MagicMock, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Dev install or pre-release ahead of public PyPI.
    monkeypatch.setattr(uc, "_pkg_version", "9.9.9")
    monkeypatch.setattr(uc, "_resolve_latest", lambda: "0.0.5")

    uc.maybe_print_update_banner()

    banner_env.assert_not_called()
