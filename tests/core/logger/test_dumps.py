"""Tests for `physiclaw.core.logger.dumps`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from physiclaw.core.logger import dumps


@pytest.fixture(autouse=True)
def _clear_ensured_cache() -> None:
    """The module memoizes which directories it has mkdir'd. Reset
    between tests so each one tests its own creation logic."""
    dumps._ENSURED.clear()


@pytest.fixture
def fake_dirs(tmp_path: Path, mocker) -> dict[str, Path]:
    tc = tmp_path / "tool_calls"
    sn = tmp_path / "snapshots"
    ss = tmp_path / "screenshots"
    mocker.patch.object(dumps.paths, "tool_calls_dir", return_value=tc)
    mocker.patch.object(dumps.paths, "snapshots_dir", return_value=sn)
    mocker.patch.object(dumps.paths, "screenshots_dir", return_value=ss)
    return {"tool_calls": tc, "snapshots": sn, "screenshots": ss}


# ---------- _stamp ----------


def test_stamp_format() -> None:
    s = dumps._stamp()

    # YYYYMMDD_HHMMSS_NNN — 19 chars (8 + 1 + 6 + 1 + 3).
    assert len(s) == 19
    assert s[8] == "_"
    assert s[15] == "_"


def test_stamp_strips_to_milliseconds() -> None:
    """`%f` is microseconds (6 digits); we slice to 3 → milliseconds."""
    s = dumps._stamp()

    # Last segment is the millisecond fragment (3 digits).
    assert len(s.split("_")[-1]) == 3


# ---------- _mkdir ----------


def test_mkdir_creates_directory(tmp_path: Path) -> None:
    d = tmp_path / "newdir"

    out = dumps._mkdir(d)

    assert out is d
    assert d.exists()
    assert d.is_dir()


def test_mkdir_caches_ensured_dirs(tmp_path: Path) -> None:
    d = tmp_path / "x"
    dumps._mkdir(d)

    assert d in dumps._ENSURED


def test_mkdir_idempotent(tmp_path: Path) -> None:
    d = tmp_path / "x"
    dumps._mkdir(d)
    dumps._mkdir(d)  # second call must not raise

    assert d in dumps._ENSURED


# ---------- save_tool_call ----------


def test_save_tool_call_no_op_when_env_unset(
    fake_dirs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHYSICLAW_SAVE_TOOL_CALLS", raising=False)

    dumps.save_tool_call("peek", "listing")

    assert not fake_dirs["tool_calls"].exists()


def test_save_tool_call_writes_listing(
    fake_dirs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHYSICLAW_SAVE_TOOL_CALLS", "1")

    dumps.save_tool_call("peek", "abc")

    files = list(fake_dirs["tool_calls"].iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".txt"
    assert files[0].read_text() == "abc"
    assert "_peek." in files[0].name


def test_save_tool_call_writes_jpeg_when_provided(
    fake_dirs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHYSICLAW_SAVE_TOOL_CALLS", "1")

    dumps.save_tool_call("screenshot", "list", jpeg=b"\xff\xd8")

    by_suffix = {p.suffix: p for p in fake_dirs["tool_calls"].iterdir()}
    assert ".txt" in by_suffix
    assert ".jpg" in by_suffix
    assert by_suffix[".jpg"].read_bytes() == b"\xff\xd8"


def test_save_tool_call_skips_jpeg_when_none(
    fake_dirs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHYSICLAW_SAVE_TOOL_CALLS", "1")

    dumps.save_tool_call("peek", "list", jpeg=None)

    suffixes = {p.suffix for p in fake_dirs["tool_calls"].iterdir()}
    assert suffixes == {".txt"}


# ---------- save_snapshot ----------


def test_save_snapshot_no_op_when_env_unset(
    fake_dirs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHYSICLAW_SAVE_SNAPSHOTS", raising=False)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    dumps.save_snapshot(frame)

    assert not fake_dirs["snapshots"].exists()


def test_save_snapshot_writes_jpeg(
    fake_dirs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHYSICLAW_SAVE_SNAPSHOTS", "1")
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    dumps.save_snapshot(frame)

    files = list(fake_dirs["snapshots"].iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".jpg"
    assert files[0].stat().st_size > 0  # cv2 actually wrote


# ---------- save_screenshot ----------


def test_save_screenshot_no_op_when_env_unset(
    fake_dirs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHYSICLAW_SAVE_SCREENSHOTS", raising=False)

    dumps.save_screenshot(b"data")

    assert not fake_dirs["screenshots"].exists()


def test_save_screenshot_writes_bytes(
    fake_dirs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHYSICLAW_SAVE_SCREENSHOTS", "1")

    dumps.save_screenshot(b"\x89PNG-fake-data")

    files = list(fake_dirs["screenshots"].iterdir())
    assert len(files) == 1
    assert files[0].read_bytes() == b"\x89PNG-fake-data"
    assert files[0].suffix == ".jpg"


def test_save_screenshot_filenames_unique_per_call(
    fake_dirs,
    monkeypatch: pytest.MonkeyPatch,
    mocker,
) -> None:
    """Stamps differ across rapid-fire calls. Without ms precision, two
    calls in the same second would collide; mock _stamp to verify each
    call gets a fresh filename."""
    monkeypatch.setenv("PHYSICLAW_SAVE_SCREENSHOTS", "1")
    stamps = iter(["t1", "t2", "t3"])
    mocker.patch.object(dumps, "_stamp", side_effect=lambda: next(stamps))

    for _ in range(3):
        dumps.save_screenshot(b"x")

    assert {p.stem for p in fake_dirs["screenshots"].iterdir()} == {"t1", "t2", "t3"}
