"""Tests for `physiclaw.core.orchestration.clipboard` — ClipboardSyncState.

The state machine takes an injectable clock, so the decay rules are
tested with a plain fake instead of mocking `time.monotonic`.
"""

from __future__ import annotations

from physiclaw.core.orchestration.clipboard import ClipboardSyncState


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _state() -> tuple[ClipboardSyncState, _Clock]:
    clock = _Clock()
    return ClipboardSyncState(now=clock), clock


# ---------- timeout policy ----------


def test_first_attempt_gets_full_window() -> None:
    state, _ = _state()

    assert state.begin() == ClipboardSyncState.CONFIRM_SECONDS


def test_attempt_after_miss_gets_short_window() -> None:
    state, _ = _state()
    state.begin()
    state.record_miss()

    assert state.begin() == ClipboardSyncState.RETRY_CONFIRM_SECONDS


def test_confirm_restores_full_window() -> None:
    state, _ = _state()
    state.begin()
    state.record_miss()
    state.confirm()

    assert state.begin() == ClipboardSyncState.CONFIRM_SECONDS


# ---------- escalation ----------


def test_first_miss_warns_without_escalation() -> None:
    state, _ = _state()

    msg = state.record_miss()

    assert "do NOT paste" in msg
    assert "Miss #" not in msg


def test_record_miss_appends_connectivity_hint() -> None:
    state, _ = _state()

    msg = state.record_miss(connectivity_hint="phone not reaching the server")

    assert "do NOT paste (phone not reaching the server)" in msg


def test_record_miss_without_hint_has_no_parenthetical() -> None:
    state, _ = _state()

    msg = state.record_miss(connectivity_hint=None)

    assert "(" not in msg


def test_second_miss_escalates_with_count() -> None:
    state, _ = _state()
    state.record_miss()

    msg = state.record_miss()

    assert "Miss #2 in a row" in msg
    assert "STOP retrying" in msg


def test_streak_keeps_counting() -> None:
    state, _ = _state()
    state.record_miss()
    state.record_miss()

    assert "Miss #3 in a row" in state.record_miss()


def test_confirm_resets_escalation() -> None:
    state, _ = _state()
    state.record_miss()
    state.confirm()

    assert "Miss #" not in state.record_miss()


# ---------- decay ----------


def test_stale_miss_state_decays_at_begin() -> None:
    # The server process spans sessions — a miss hours ago must not
    # shorten the window or claim "in a row" for today's first sync.
    state, clock = _state()
    state.record_miss()
    clock.t += ClipboardSyncState.MISS_DECAY_SECONDS + 1

    assert state.begin() == ClipboardSyncState.CONFIRM_SECONDS
    assert "Miss #" not in state.record_miss()


def test_recent_miss_state_does_not_decay() -> None:
    state, clock = _state()
    state.record_miss()
    clock.t += ClipboardSyncState.MISS_DECAY_SECONDS - 1

    assert state.begin() == ClipboardSyncState.RETRY_CONFIRM_SECONDS
    assert "Miss #2" in state.record_miss()
