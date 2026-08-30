"""Tests for `physiclaw.common.daylog` — the shared daily-log file
format (engine and conductor both write and read it; the engine's
tool-facing API delegates here)."""

from __future__ import annotations

import datetime as dt
import re

from freezegun import freeze_time

from physiclaw.common import daylog, paths


def _today_file():
    return paths.memory_dir() / f"{dt.date.today().isoformat()}.md"


def test_append_creates_the_file_with_its_date_header() -> None:
    daylog.append_log("[11:00] demo: did a thing")

    text = _today_file().read_text(encoding="utf-8")

    assert text.startswith(f"# {dt.date.today().isoformat()}\n\n")
    assert text.endswith("[11:00] demo: did a thing\n")


def test_append_skips_empty_entries() -> None:
    daylog.append_log("   ")

    assert not _today_file().exists()


@freeze_time("2026-08-30 09:41")
def test_stamped_carries_the_doctrine_time_prefix() -> None:
    assert daylog.stamped("conductor: paid") == "[09:41] conductor: paid"


def test_load_recent_entries_newest_first_with_date_rewrite() -> None:
    daylog.append_log("[10:00] first")
    daylog.append_log("[11:00] second")

    entries = daylog.load_recent_entries(5)

    today = dt.date.today().isoformat()
    assert entries.splitlines() == [
        f"[{today} 11:00] second",
        f"[{today} 10:00] first",
    ]


def test_load_recent_entries_walks_back_across_days() -> None:
    yesterday = dt.date.today() - dt.timedelta(days=1)
    paths.memory_dir().mkdir(parents=True, exist_ok=True)
    (paths.memory_dir() / f"{yesterday.isoformat()}.md").write_text(
        f"# {yesterday.isoformat()}\n\n[23:00] older entry\n", encoding="utf-8"
    )
    daylog.append_log("[08:00] newer entry")

    entries = daylog.load_recent_entries(5)

    assert entries.splitlines() == [
        f"[{dt.date.today().isoformat()} 08:00] newer entry",
        f"[{yesterday.isoformat()} 23:00] older entry",
    ]


def test_load_recent_entries_respects_the_window() -> None:
    daylog.append_log("[10:00] one")
    daylog.append_log("[11:00] two")
    daylog.append_log("[12:00] three")

    entries = daylog.load_recent_entries(2)

    assert len(entries.splitlines()) == 2
    assert "three" in entries and "one" not in entries


def test_load_recent_entries_empty_without_files() -> None:
    assert daylog.load_recent_entries(5) == ""


def test_unstamped_lines_pass_through_unchanged() -> None:
    daylog.append_log("a bare note without a clock")

    entries = daylog.load_recent_entries(5)

    assert entries == "a bare note without a clock"
    assert not re.match(r"^\[\d{4}-", entries)
