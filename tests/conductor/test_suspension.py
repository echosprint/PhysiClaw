"""Tests for `physiclaw.conductor.suspension` — the one cross-wake file.

Unit pins for the module's own surface: the cheap `suspended_ref` peek
(the CLI/jobs-facing read that skips walk-state validation) and the
one-shot `clear_suspended`. The WAKE path — `setup.load_suspended`'s
corrupt-file and consumed-on-load behavior — has its own tests in
test_program.py; nothing here duplicates them.
"""

from __future__ import annotations

import json

from physiclaw.conductor import suspension


def _write(text: str) -> None:
    p = suspension.suspended_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_clear_reports_whether_anything_was_there() -> None:
    assert suspension.clear_suspended() is False  # nothing to clear
    _write("{}")
    assert suspension.clear_suspended() is True
    assert not suspension.suspended_path().exists()


def test_ref_is_none_when_nothing_is_suspended() -> None:
    assert suspension.suspended_ref() is None


def test_ref_reads_the_pair_without_validating_the_rest() -> None:
    # `suspended_ref` is the cheap peek (jobs/CLI); it must not demand
    # the full walk-state schema the loader validates.
    _write(json.dumps({"app": "demo", "playbook": "buy"}))
    assert suspension.suspended_ref() == ("demo", "buy")


def test_ref_swallows_garbage_and_half_written_files() -> None:
    # A crash mid-write (or a hand-edit) must read as "nothing
    # suspended" — the peek is used from surfaces that must never
    # crash on a bad file.
    for text in ("not json{", "", json.dumps({"app": "demo"})):
        _write(text)
        assert suspension.suspended_ref() is None, text
