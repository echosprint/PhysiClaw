"""Tests for `physiclaw.macros.stats` — the single stats.json:
counter folding, consecutive-abort streaks, pruning, and fail-open IO."""

from __future__ import annotations

import json

from freezegun import freeze_time

from physiclaw.common import paths
from physiclaw.macros import stats


def _record_ok(name: str = "demo", known: set[str] | None = None) -> None:
    stats.record(name, ok=True, known_names=known if known is not None else {name})


def _record_abort(name: str = "demo", known: set[str] | None = None) -> None:
    stats.record(
        name,
        ok=False,
        known_names=known if known is not None else {name},
        step=2,
        reason="guard_failed",
        detail="'WeChat' not found",
    )


# ---------- recording ----------


@freeze_time("2026-08-02 10:11:00")
def test_record_success_writes_counters_and_timestamps() -> None:
    _record_ok()

    entry = stats.load()["demo"]

    assert entry["total_runs"] == 1
    assert entry["total_successes"] == 1
    assert entry["total_aborts"] == 0
    assert entry["consecutive_aborts"] == 0
    assert entry["last_run_at"] == "2026-08-02T10:11:00"
    assert entry["last_success_at"] == "2026-08-02T10:11:00"


@freeze_time("2026-08-02 10:11:00")
def test_record_abort_writes_last_abort_details() -> None:
    _record_abort()

    entry = stats.load()["demo"]

    assert entry["total_aborts"] == 1
    assert entry["consecutive_aborts"] == 1
    assert entry["last_abort"] == {
        "ts": "2026-08-02T10:11:00",
        "step": 2,
        "reason": "guard_failed",
        "detail": "'WeChat' not found",
        "run": None,
    }


def test_record_abort_carries_the_run_id() -> None:
    stats.record(
        "demo",
        ok=False,
        known_names={"demo"},
        step=1,
        reason="tool_error",
        detail="d",
        run_id="macro-run-a3f9c1",
    )

    assert stats.load()["demo"]["last_abort"]["run"] == "macro-run-a3f9c1"


def test_record_success_resets_consecutive_aborts() -> None:
    _record_abort()
    _record_abort()

    _record_ok()

    entry = stats.load()["demo"]
    assert entry["consecutive_aborts"] == 0
    assert entry["total_aborts"] == 2
    assert entry["total_runs"] == 3


def test_record_aborts_accumulate_streak() -> None:
    _record_abort()
    _record_abort()

    entry = stats.load()["demo"]

    assert entry["consecutive_aborts"] == 2


def test_record_prunes_keys_without_a_macro_dir() -> None:
    _record_ok("gone", known={"gone"})

    _record_ok("demo", known={"demo"})  # 'gone' no longer exists on disk

    assert set(stats.load()) == {"demo"}


def test_record_keeps_other_known_macros() -> None:
    _record_ok("a", known={"a", "b"})
    _record_ok("b", known={"a", "b"})

    assert set(stats.load()) == {"a", "b"}


# ---------- fail-open IO ----------


def test_load_missing_file_returns_empty() -> None:
    assert stats.load() == {}


def test_load_corrupt_file_returns_empty() -> None:
    paths.macros_dir().mkdir(parents=True)
    stats.stats_file().write_text("{not json", encoding="utf-8")

    assert stats.load() == {}


def test_load_non_dict_payload_returns_empty() -> None:
    paths.macros_dir().mkdir(parents=True)
    stats.stats_file().write_text("[1, 2]", encoding="utf-8")

    assert stats.load() == {}


def test_record_over_corrupt_file_starts_fresh() -> None:
    paths.macros_dir().mkdir(parents=True)
    stats.stats_file().write_text("{not json", encoding="utf-8")

    _record_ok()

    assert stats.load()["demo"]["total_runs"] == 1


def test_record_over_a_scalar_entry_resets_that_key_instead_of_raising() -> None:
    # A hand-edited or truncated file can leave a scalar under a macro key.
    # `record` runs AFTER the gestures physically fired, so raising here
    # would cost the agent its step log and its screen.
    paths.macros_dir().mkdir(parents=True)
    stats.stats_file().write_text('{"demo": 3}', encoding="utf-8")

    _record_ok()

    assert stats.load()["demo"]["total_runs"] == 1


def test_bad_input_does_not_move_the_re_rehearse_streak() -> None:
    # `consecutive_aborts` means "the app layout changed under this macro".
    # A bad_input run never reached the screen, so it must not say that —
    # three typos in a row would make a healthy macro read as decayed.
    for _ in range(3):
        stats.record(
            "demo",
            ok=False,
            known_names={"demo"},
            step=0,
            reason="bad_input",
            detail="missing required input 'msg'",
        )

    entry = stats.load()["demo"]

    assert entry["total_runs"] == 3
    assert entry["total_aborts"] == 3  # still a failed run
    assert entry["consecutive_aborts"] == 0  # but not a layout signal
    assert entry["last_abort"]["reason"] == "bad_input"


def test_bad_input_does_not_clear_a_real_streak_either() -> None:
    _record_abort()
    _record_abort()
    stats.record("demo", ok=False, known_names={"demo"}, reason="bad_input")

    assert stats.load()["demo"]["consecutive_aborts"] == 2


def test_record_write_failure_does_not_raise(mocker) -> None:
    mocker.patch(
        "physiclaw.macros.stats.write_json_atomic",
        side_effect=OSError("disk full"),
    )

    _record_ok()  # must not raise


def test_stats_file_is_valid_json_on_disk() -> None:
    _record_ok()

    data = json.loads(stats.stats_file().read_text(encoding="utf-8"))

    assert "demo" in data
