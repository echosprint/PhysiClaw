"""Tests for `physiclaw.runtime_state` — server pid/host/port file.

The autouse `physiclaw_home` fixture handles per-test home isolation,
so `paths.runtime_state_file()` lands under tmp_path automatically.

`_pid_alive` is private but its branches (ProcessLookupError,
PermissionError, OSError, success) are observable through `read_live`,
so tests drive it via `mocker.patch("os.kill", side_effect=…)`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from physiclaw import paths, runtime_state


# ---------- write ----------


def test_write_creates_file_with_required_fields(physiclaw_home: Path) -> None:
    runtime_state.write("127.0.0.1", 8048)

    path = paths.runtime_state_file()
    state = json.loads(path.read_text())
    assert state["pid"] == os.getpid()
    assert state["host"] == "127.0.0.1"
    assert state["port"] == 8048
    assert state["model_ref"] is None
    assert state["model_source"] is None
    assert isinstance(state["started_at"], float)


def test_write_records_started_at_close_to_current_time(physiclaw_home: Path) -> None:
    before = time.time()
    runtime_state.write("h", 1)
    after = time.time()

    state = json.loads(paths.runtime_state_file().read_text())
    assert before <= state["started_at"] <= after


def test_write_records_model_ref_and_source_when_supplied(
    physiclaw_home: Path,
) -> None:
    runtime_state.write(
        "h", 1, model_ref="anthropic/claude-opus-4-7", model_source="config"
    )

    state = json.loads(paths.runtime_state_file().read_text())
    assert state["model_ref"] == "anthropic/claude-opus-4-7"
    assert state["model_source"] == "config"


def test_write_creates_parent_directory_when_missing(
    physiclaw_home: Path,
) -> None:
    # Per-test home doesn't have the `run/` subdir yet.
    assert not paths.runtime_state_file().parent.exists()

    runtime_state.write("h", 1)

    assert paths.runtime_state_file().is_file()


def test_write_creates_intermediate_parents_when_deep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, physiclaw_home: Path
) -> None:
    # `parents=True` is required because the deep parent chain doesn't
    # exist yet — `parents=False` would FileNotFoundError on the first
    # missing dir.
    deep = tmp_path / "a" / "b" / "c" / "server.json"
    monkeypatch.setattr(paths, "runtime_state_file", lambda: deep)

    runtime_state.write("h", 1)

    assert deep.is_file()


def test_write_overwrites_prior_file(physiclaw_home: Path) -> None:
    runtime_state.write("first", 1111)

    runtime_state.write("second", 2222)

    state = json.loads(paths.runtime_state_file().read_text())
    assert state["host"] == "second"
    assert state["port"] == 2222


# ---------- clear ----------


def test_clear_removes_file_when_pid_matches_current_process(
    physiclaw_home: Path,
) -> None:
    runtime_state.write("h", 1)

    runtime_state.clear()

    assert not paths.runtime_state_file().exists()


def test_clear_refuses_to_remove_file_owned_by_other_pid(
    physiclaw_home: Path,
) -> None:
    # Stand-in for a stale file from a different process.
    path = paths.runtime_state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getpid() + 1, "host": "h", "port": 1}))

    runtime_state.clear()

    assert path.exists()


def test_clear_silently_passes_when_file_missing(physiclaw_home: Path) -> None:
    # No file at all — `read_text` raises FileNotFoundError, swallowed.
    runtime_state.clear()  # must not raise


def test_clear_silently_passes_on_malformed_json(physiclaw_home: Path) -> None:
    path = paths.runtime_state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")

    runtime_state.clear()  # JSONDecodeError swallowed

    # File remains because we never reached the unlink branch.
    assert path.exists()


def test_clear_silently_passes_when_file_disappears_before_unlink(
    physiclaw_home: Path, mocker
) -> None:
    # Race: file exists when read_text runs, gone by the time unlink does.
    path = paths.runtime_state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getpid(), "host": "h", "port": 1}))

    mocker.patch.object(Path, "unlink", side_effect=FileNotFoundError)

    runtime_state.clear()  # must not raise


# ---------- read_live ----------


def test_pid_alive_calls_os_kill_with_signal_zero(physiclaw_home: Path, mocker) -> None:
    # Run BEFORE any test that calls real `os.kill` — the mutated form
    # `os.kill(pid, 1)` would deliver SIGHUP to the pytest process and
    # crash the test runner before this assertion gets a chance to fail.
    # Pytest runs tests in file order; keep this first in the read_live
    # section so the mutation is detected via mock-arg inspection,
    # never via real signal delivery.
    runtime_state.write("h", 1)
    kill = mocker.patch("os.kill")

    runtime_state.read_live()

    kill.assert_called_once_with(os.getpid(), 0)


def test_read_live_returns_state_dict_when_pid_is_alive(
    physiclaw_home: Path,
) -> None:
    runtime_state.write("h", 8048, model_ref="x/y")

    state = runtime_state.read_live()

    assert state is not None
    assert state["host"] == "h"
    assert state["port"] == 8048
    assert state["model_ref"] == "x/y"


def test_read_live_returns_none_when_file_missing(physiclaw_home: Path) -> None:
    assert runtime_state.read_live() is None


def test_read_live_returns_none_on_malformed_json(physiclaw_home: Path) -> None:
    path = paths.runtime_state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")

    assert runtime_state.read_live() is None


def test_read_live_returns_none_when_read_text_raises_oserror(
    physiclaw_home: Path, mocker
) -> None:
    runtime_state.write("h", 1)

    mocker.patch.object(Path, "read_text", side_effect=OSError("permission denied"))

    assert runtime_state.read_live() is None


@pytest.mark.parametrize("bad_pid", ["string", None, 1.5, [42]])
def test_read_live_returns_none_when_pid_field_is_not_int(
    physiclaw_home: Path, bad_pid: object
) -> None:
    path = paths.runtime_state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": bad_pid, "host": "h", "port": 1}))

    assert runtime_state.read_live() is None


def test_read_live_returns_none_when_pid_lookup_raises_process_lookup_error(
    physiclaw_home: Path, mocker
) -> None:
    runtime_state.write("h", 1)

    mocker.patch("os.kill", side_effect=ProcessLookupError)

    assert runtime_state.read_live() is None


def test_read_live_returns_state_when_pid_owned_by_other_user(
    physiclaw_home: Path, mocker
) -> None:
    # _pid_alive treats PermissionError as "exists, owned by another user"
    # — the process is alive, just not visible-as-killable to us.
    runtime_state.write("h", 1)

    mocker.patch("os.kill", side_effect=PermissionError)

    state = runtime_state.read_live()
    assert state is not None
    assert state["host"] == "h"


def test_read_live_returns_none_when_pid_lookup_raises_oserror(
    physiclaw_home: Path, mocker
) -> None:
    # Generic OSError (e.g. EINVAL) — falls through to "not alive".
    runtime_state.write("h", 1)

    mocker.patch("os.kill", side_effect=OSError("EINVAL"))

    assert runtime_state.read_live() is None
