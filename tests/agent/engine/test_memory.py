"""Tests for `physiclaw.agent.engine.memory` — read + write helpers.

Module-level paths (`MEMORY_DIR`, `MEMORY_FILE`, `USER_FILE`) are bound
at import time from `paths.memory_dir()` — captured against the session
tmp dir. The autouse `_memory_paths` fixture below re-points them to a
per-test directory so writes don't bleed across tests.

`freezegun` is used for `append_log` and `load_recent_entries` because
both read `dt.date.today()` directly. The `_LOOKBACK_DAYS_CEILING` test
makes 365 days of accumulation deterministic by freezing the day.

Accepted equivalent mutants:

  - `update_fact` delete-line branch flag mutations (`removed = False/
    None` initial, `True` → `False/None` post-match) — the `count > 1`
    pre-check guarantees `old` matches in exactly one line, so the
    flag never has to gate a second iteration.
  - `text.replace(old, new, 1)` ↔ `count=2` — same: pre-check ensures
    only one occurrence, so 1 and 2 produce identical output.
  - Outer-loop `break` → `continue` (after `len(collected) >= n`) —
    behavior matches `break` because `continue` re-checks the guard
    immediately; only the iteration count differs.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from freezegun import freeze_time

from physiclaw.agent.engine import memory


@pytest.fixture(autouse=True)
def _memory_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Per-test memory dir under `tmp_path`. Module-level constants get
    monkeypatched to point here for the duration of the test."""
    mem = tmp_path / "memory"
    monkeypatch.setattr(memory, "MEMORY_DIR", mem)
    monkeypatch.setattr(memory, "MEMORY_FILE", mem / "memory.md")
    monkeypatch.setattr(memory, "USER_FILE", mem / "USER.md")
    return mem


# ---------- module-level constants ----------


def test_memory_file_basename_is_memory_md(tmp_path: Path) -> None:
    # Reload to recompute against a known parent so the basename is
    # observable; my autouse fixture otherwise overrides MEMORY_FILE.
    import os
    os.environ["PHYSICLAW_HOME"] = str(tmp_path)
    importlib.reload(memory)

    assert memory.MEMORY_FILE.name == "memory.md"
    assert memory.MEMORY_FILE.parent == memory.MEMORY_DIR


def test_user_file_basename_is_USER_md(tmp_path: Path) -> None:
    import os
    os.environ["PHYSICLAW_HOME"] = str(tmp_path)
    importlib.reload(memory)

    assert memory.USER_FILE.name == "USER.md"
    assert memory.USER_FILE.parent == memory.MEMORY_DIR


def test_lookback_days_ceiling_is_365() -> None:
    assert memory._LOOKBACK_DAYS_CEILING == 365


def test_bootstrap_log_entries_is_a_positive_integer() -> None:
    # Loaded from CONFIG; mutating to None would AttributeError elsewhere.
    assert isinstance(memory.BOOTSTRAP_LOG_ENTRIES, int)
    assert memory.BOOTSTRAP_LOG_ENTRIES > 0


# ---------- load_user ----------


def test_load_user_returns_empty_string_when_file_missing() -> None:
    assert memory.load_user() == ""


def test_load_user_returns_stripped_content(_memory_paths: Path) -> None:
    _memory_paths.mkdir(parents=True)
    (_memory_paths / "USER.md").write_text("\n\n  Profile: power user  \n\n")

    assert memory.load_user() == "Profile: power user"


def test_load_user_returns_empty_string_for_whitespace_only_file(
    _memory_paths: Path,
) -> None:
    _memory_paths.mkdir(parents=True)
    (_memory_paths / "USER.md").write_text("   \n\n\t  \n")

    assert memory.load_user() == ""


# ---------- load_persistent ----------


def test_load_persistent_returns_empty_string_when_file_missing() -> None:
    assert memory.load_persistent() == ""


def test_load_persistent_returns_stripped_content(_memory_paths: Path) -> None:
    _memory_paths.mkdir(parents=True)
    (_memory_paths / "memory.md").write_text("\n  fact one\nfact two  \n\n")

    assert memory.load_persistent() == "fact one\nfact two"


# ---------- load_recent_entries ----------


def test_load_recent_entries_returns_empty_string_when_no_logs() -> None:
    with freeze_time("2026-04-28"):
        assert memory.load_recent_entries() == ""


def test_load_recent_entries_reads_today_with_most_recent_first(
    _memory_paths: Path,
) -> None:
    _memory_paths.mkdir(parents=True)
    (_memory_paths / "2026-04-28.md").write_text(
        "# 2026-04-28\n\n"
        "[09:00] first\n"
        "[10:00] second\n"
        "[11:00] third\n"
    )

    with freeze_time("2026-04-28"):
        out = memory.load_recent_entries(n=3)

    # Reversed within file — most recent first.
    assert out == (
        "[2026-04-28 11:00] third\n"
        "[2026-04-28 10:00] second\n"
        "[2026-04-28 09:00] first"
    )


def test_load_recent_entries_walks_back_when_today_underfilled(
    _memory_paths: Path,
) -> None:
    _memory_paths.mkdir(parents=True)
    (_memory_paths / "2026-04-28.md").write_text("# 2026-04-28\n[09:00] today-1\n")
    (_memory_paths / "2026-04-27.md").write_text(
        "# 2026-04-27\n[14:00] yesterday-1\n[15:00] yesterday-2\n"
    )

    with freeze_time("2026-04-28"):
        out = memory.load_recent_entries(n=3)

    # Today's lone entry first, then yesterday's two in reverse order.
    assert out == (
        "[2026-04-28 09:00] today-1\n"
        "[2026-04-27 15:00] yesterday-2\n"
        "[2026-04-27 14:00] yesterday-1"
    )


def test_load_recent_entries_skips_header_and_blank_lines(
    _memory_paths: Path,
) -> None:
    _memory_paths.mkdir(parents=True)
    (_memory_paths / "2026-04-28.md").write_text(
        "# 2026-04-28\n"
        "\n"
        "[09:00] real entry\n"
        "  \n"  # whitespace-only
        "## a sub-header — also skipped\n"
        "[10:00] another\n"
    )

    with freeze_time("2026-04-28"):
        out = memory.load_recent_entries(n=10)

    assert out == "[2026-04-28 10:00] another\n[2026-04-28 09:00] real entry"


def test_load_recent_entries_passes_through_lines_with_no_time_prefix(
    _memory_paths: Path,
) -> None:
    _memory_paths.mkdir(parents=True)
    (_memory_paths / "2026-04-28.md").write_text(
        "# 2026-04-28\n[09:00] timed\nfreeform note\n"
    )

    with freeze_time("2026-04-28"):
        out = memory.load_recent_entries(n=10)

    # The freeform line has no [HH:MM] prefix — passes through unchanged.
    # Reversed order: freeform first, then timed.
    assert out == "freeform note\n[2026-04-28 09:00] timed"


def test_load_recent_entries_stops_at_n_even_with_more_available(
    _memory_paths: Path,
) -> None:
    _memory_paths.mkdir(parents=True)
    (_memory_paths / "2026-04-28.md").write_text(
        "# 2026-04-28\n"
        + "".join(f"[09:0{i}] entry-{i}\n" for i in range(5))
    )

    with freeze_time("2026-04-28"):
        out = memory.load_recent_entries(n=2)

    # Only 2 lines collected — most recent first.
    assert out == "[2026-04-28 09:04] entry-4\n[2026-04-28 09:03] entry-3"


def test_load_recent_entries_outer_loop_does_not_overshoot_when_today_fills_n(
    _memory_paths: Path,
) -> None:
    # Today exactly fills n=2; the outer-loop guard `>= n` must break,
    # not `> n` (which would let one extra entry from yesterday slip in).
    _memory_paths.mkdir(parents=True)
    (_memory_paths / "2026-04-28.md").write_text(
        "# 2026-04-28\n[09:00] today-a\n[10:00] today-b\n"
    )
    (_memory_paths / "2026-04-27.md").write_text(
        "# 2026-04-27\n[10:00] yesterday-extra\n"
    )

    with freeze_time("2026-04-28"):
        out = memory.load_recent_entries(n=2)

    assert "yesterday" not in out
    assert out.count("\n") == 1  # exactly two lines


def test_load_recent_entries_skips_missing_day_to_continue_walking_back(
    _memory_paths: Path,
) -> None:
    # Today has 1 entry; yesterday's file is MISSING; day-before has 2.
    # FileNotFoundError handler must `continue` (not `break`), otherwise
    # day-before's entries are never read.
    _memory_paths.mkdir(parents=True)
    (_memory_paths / "2026-04-28.md").write_text("# 2026-04-28\n[09:00] today\n")
    # 2026-04-27.md intentionally missing
    (_memory_paths / "2026-04-26.md").write_text(
        "# 2026-04-26\n[14:00] day-before-a\n[15:00] day-before-b\n"
    )

    with freeze_time("2026-04-28"):
        out = memory.load_recent_entries(n=3)

    assert "today" in out
    assert "day-before-a" in out
    assert "day-before-b" in out


def test_load_recent_entries_uses_default_n_when_omitted(
    _memory_paths: Path,
) -> None:
    # Default is `DEFAULT_LOG_ENTRIES`. Just verify the path is reachable
    # — exact value depends on user config.
    _memory_paths.mkdir(parents=True)
    (_memory_paths / "2026-04-28.md").write_text(
        "# 2026-04-28\n[09:00] entry\n"
    )

    with freeze_time("2026-04-28"):
        out = memory.load_recent_entries()

    assert out == "[2026-04-28 09:00] entry"


# ---------- append_log ----------


def test_append_log_creates_file_with_date_header(_memory_paths: Path) -> None:
    with freeze_time("2026-04-28T10:30:00"):
        memory.append_log("[10:30] first entry")

    path = _memory_paths / "2026-04-28.md"
    assert path.read_text() == "# 2026-04-28\n\n[10:30] first entry\n"


def test_append_log_does_not_repeat_header_on_existing_file(
    _memory_paths: Path,
) -> None:
    with freeze_time("2026-04-28T10:30:00"):
        memory.append_log("[10:30] first")
        memory.append_log("[11:00] second")

    text = (_memory_paths / "2026-04-28.md").read_text()
    assert text.count("# 2026-04-28") == 1
    assert text.endswith("[10:30] first\n[11:00] second\n")


def test_append_log_creates_parent_directory_when_missing(
    _memory_paths: Path,
) -> None:
    assert not _memory_paths.exists()

    with freeze_time("2026-04-28"):
        memory.append_log("[10:00] x")

    assert _memory_paths.is_dir()


def test_append_log_creates_intermediate_parents_when_chain_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `mkdir(parents=True)` — required when the chain doesn't exist.
    deep = tmp_path / "a" / "b" / "c" / "memory"
    monkeypatch.setattr(memory, "MEMORY_DIR", deep)
    monkeypatch.setattr(memory, "MEMORY_FILE", deep / "memory.md")

    with freeze_time("2026-04-28"):
        memory.append_log("[10:00] x")

    assert (deep / "2026-04-28.md").is_file()


def test_append_log_strips_entry_before_writing(_memory_paths: Path) -> None:
    with freeze_time("2026-04-28T10:30:00"):
        memory.append_log("  [10:30] padded entry  \n\n")

    text = (_memory_paths / "2026-04-28.md").read_text()
    # Stripped — no surrounding whitespace, single trailing newline.
    assert "[10:30] padded entry\n" in text
    assert "  [10:30]" not in text


@pytest.mark.parametrize("entry", ["", "   ", "\n\n\t"])
def test_append_log_empty_or_whitespace_only_is_noop(
    _memory_paths: Path, entry: str
) -> None:
    with freeze_time("2026-04-28"):
        memory.append_log(entry)

    assert not _memory_paths.exists()


# ---------- save_fact ----------


def test_save_fact_appends_to_memory_md(_memory_paths: Path) -> None:
    memory.save_fact("user prefers metric units")
    memory.save_fact("user uses AssistiveTouch")

    text = (_memory_paths / "memory.md").read_text()
    assert text == "user prefers metric units\nuser uses AssistiveTouch\n"


def test_save_fact_creates_parent_directory_when_missing(
    _memory_paths: Path,
) -> None:
    assert not _memory_paths.exists()

    memory.save_fact("first fact")

    assert (_memory_paths / "memory.md").is_file()


def test_save_fact_creates_intermediate_parents_when_chain_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deep = tmp_path / "x" / "y" / "z" / "memory"
    monkeypatch.setattr(memory, "MEMORY_DIR", deep)
    monkeypatch.setattr(memory, "MEMORY_FILE", deep / "memory.md")

    memory.save_fact("first")

    assert (deep / "memory.md").is_file()


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_save_fact_empty_or_whitespace_only_is_noop(
    _memory_paths: Path, text: str
) -> None:
    memory.save_fact(text)

    assert not _memory_paths.exists()


def test_save_fact_strips_text_before_writing(_memory_paths: Path) -> None:
    memory.save_fact("  surrounded  ")

    assert (_memory_paths / "memory.md").read_text() == "surrounded\n"


# ---------- update_fact ----------


def test_update_fact_replaces_single_occurrence(_memory_paths: Path) -> None:
    _memory_paths.mkdir(parents=True)
    (_memory_paths / "memory.md").write_text(
        "user is qiaoqian\nuser uses metric units\n"
    )

    memory.update_fact("metric", "imperial")

    assert (_memory_paths / "memory.md").read_text() == (
        "user is qiaoqian\nuser uses imperial units\n"
    )


def test_update_fact_with_empty_new_deletes_the_containing_line(
    _memory_paths: Path,
) -> None:
    _memory_paths.mkdir(parents=True)
    (_memory_paths / "memory.md").write_text(
        "fact-one\nfact-to-remove\nfact-three\n"
    )

    memory.update_fact("fact-to-remove", "")

    assert (_memory_paths / "memory.md").read_text() == "fact-one\nfact-three\n"


def test_update_fact_raises_FileNotFoundError_when_memory_md_missing() -> None:
    with pytest.raises(
        FileNotFoundError, match=r"^.+/memory\.md does not exist$"
    ):
        memory.update_fact("anything", "anything")


def test_update_fact_raises_ValueError_when_old_text_not_present(
    _memory_paths: Path,
) -> None:
    _memory_paths.mkdir(parents=True)
    (_memory_paths / "memory.md").write_text("present text\n")

    with pytest.raises(
        ValueError, match=r"^old text not found in memory\.md: 'absent'$"
    ):
        memory.update_fact("absent", "new")


def test_update_fact_raises_ValueError_when_old_text_matches_exactly_two_places(
    _memory_paths: Path,
) -> None:
    # `count > 1` boundary — exactly 2 occurrences must raise. Mutating
    # to `count > 2` would incorrectly let this case through.
    _memory_paths.mkdir(parents=True)
    (_memory_paths / "memory.md").write_text(
        "user uses bot\nuser trusts bot\n"
    )

    with pytest.raises(
        ValueError,
        match=(
            r"^old text matched 2 places in memory\.md — "
            r"narrow the string so it matches exactly once$"
        ),
    ):
        memory.update_fact("bot", "PhysiClaw")


def test_update_fact_raises_ValueError_when_old_text_matches_three_places(
    _memory_paths: Path,
) -> None:
    _memory_paths.mkdir(parents=True)
    (_memory_paths / "memory.md").write_text(
        "user uses bot\nuser likes bot\nuser trusts bot\n"
    )

    with pytest.raises(
        ValueError,
        match=(
            r"^old text matched 3 places in memory\.md — "
            r"narrow the string so it matches exactly once$"
        ),
    ):
        memory.update_fact("bot", "PhysiClaw")


def test_update_fact_with_empty_new_only_removes_first_match_and_keeps_others(
    _memory_paths: Path,
) -> None:
    # Even though `count > 1` would normally raise, the helper's
    # delete-mode would only zap the first containing line if reached.
    # Pre-validate: only one occurrence, on its own line.
    _memory_paths.mkdir(parents=True)
    (_memory_paths / "memory.md").write_text(
        "keep this line\nremove unique-marker\nkeep this too\n"
    )

    memory.update_fact("unique-marker", "")

    assert (_memory_paths / "memory.md").read_text() == (
        "keep this line\nkeep this too\n"
    )


# ---------- memory cues (scan_cues / cue_corrective) ----------


def test_scan_cues_matches_english_and_chinese_with_context() -> None:
    # Cues far enough apart (> 2×window) to yield two distinct snippets.
    text = "记住我不吃辣。" + "x" * 120 + " remember this: code 4021."
    cues = memory.scan_cues(text)

    assert len(cues) == 2
    assert any("记住我不吃辣" in c for c in cues)
    assert any("remember this: code 4021" in c for c in cues)


def test_scan_cues_none_when_no_signal() -> None:
    assert memory.scan_cues("tapped the button, screen changed, added to cart") == []
    assert memory.scan_cues("") == []


def test_scan_cues_variants_match() -> None:
    for t in ("don't forget the gate code", "keep in mind she prefers oat milk",
              "别忘了买牛奶", "for next time: use the side door"):
        assert memory.scan_cues(t), t


def test_scan_cues_dedups_and_caps() -> None:
    text = " ".join(["记住 X"] * 10)  # same snippet repeated
    assert len(memory.scan_cues(text)) == 1  # deduped

    many = " ".join(f"记住 item{i} zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz" for i in range(9))
    assert len(memory.scan_cues(many, window=5)) <= 5  # capped


def test_cue_corrective_lists_snippets_and_names_the_fix() -> None:
    out = memory.cue_corrective(["记住我不吃辣", "remember: code 4021"])
    assert "记住我不吃辣" in out and "remember: code 4021" in out
    assert "save_memory" in out


# ---------- soft cap (over_soft_cap) ----------


def test_over_soft_cap_reports_size_only_when_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory, "SOFT_CAP_CHARS", 10)
    assert memory.over_soft_cap() is None  # empty store

    memory.save_fact("x" * 40)
    over = memory.over_soft_cap()
    assert over == len(memory.load_persistent()) and over > 10

    monkeypatch.setattr(memory, "SOFT_CAP_CHARS", 10_000)
    assert memory.over_soft_cap() is None  # now comfortably under
