"""Tests for physiclaw.core.orchestration.unlock — the stateful
mechanical passcode entry, driven with a recording ``execute`` fake and a
stub perception (no rig / lock / observation bracket needed)."""

from __future__ import annotations

from unittest.mock import MagicMock

from physiclaw.core.orchestration import gestures, unlock


class _Perception:
    """Minimal perception stub: warms OCR (no-op) and returns a fixed
    keypad bbox (or None) from the poll."""

    def __init__(self, bbox: list[float] | None):
        self.ocr_reader = MagicMock()
        self.wait_for_numpad_digit = MagicMock(return_value=bbox)


def _run(mocker, unlocker, *, bbox, monotonic=None):
    """Run one unlock_phone on ``unlocker`` with sleeps patched out;
    monotonic optionally faked. Returns (result, recorded gestures,
    perception stub)."""
    mocker.patch.object(unlock.time, "sleep")
    if monotonic is not None:
        mocker.patch.object(unlock.time, "monotonic", side_effect=monotonic)
    recorded: list = []

    def execute(step):
        recorded.append(step)
        return "ok"

    per = _Perception(bbox)
    result = unlocker.unlock_phone(execute, per)
    return result, recorded, per


def _taps(recorded) -> list:
    return [g for g in recorded if isinstance(g, gestures.Tap)]


def _swipes(recorded) -> list:
    return [g for g in recorded if isinstance(g, gestures.Swipe)]


def test_taps_six_times_when_keypad_found(mocker) -> None:
    result, recorded, per = _run(
        mocker, unlock.PhoneUnlock(), bbox=[0.1, 0.1, 0.2, 0.2]
    )

    assert result == "Passcode entered"
    per.ocr_reader.assert_called_once()  # OCR warmed before the wake
    # 1 wake-tap (WAKE_SCREEN is a Tap) + 6 digit-taps = 7.
    assert len(_taps(recorded)) == 7
    assert len(_swipes(recorded)) == 1  # single swipe, no re-arm


def test_fails_when_keypad_not_found(mocker) -> None:
    result, recorded, _ = _run(mocker, unlock.PhoneUnlock(), bbox=None)

    assert result == "Failed to find passcode keypad"
    # Only the wake tap ran; no digit taps after the failed poll.
    assert len(_taps(recorded)) == 1


def test_rearms_keypad_when_locate_is_slow(mocker) -> None:
    # Locating the "1" ran past STALE_SECONDS → re-swipe a fresh keypad
    # and blind-tap the found bbox, with NO second OCR. monotonic feeds
    # swipe_at=0 then two reads past the threshold.
    result, recorded, per = _run(
        mocker,
        unlock.PhoneUnlock(),
        bbox=[0.1, 0.1, 0.2, 0.2],
        monotonic=[0.0, 10.0, 10.0],
    )

    assert result == "Passcode entered"
    per.wait_for_numpad_digit.assert_called_once()  # no second OCR
    assert len(_swipes(recorded)) == 2  # initial swipe + re-arm
    assert len(_taps(recorded)) == 7


def test_second_unlock_uses_cached_bbox_without_ocr(mocker) -> None:
    unlocker = unlock.PhoneUnlock()
    # First unlock: OCR locates and caches the "1".
    _run(mocker, unlocker, bbox=[0.1, 0.1, 0.2, 0.2])

    # Second unlock: the poll would return None, but it must never be
    # called — the cached bbox is blind-tapped instead.
    result, recorded, per2 = _run(mocker, unlocker, bbox=None)

    assert result == "Passcode entered"
    per2.ocr_reader.assert_not_called()  # cached — no OCR warm
    per2.wait_for_numpad_digit.assert_not_called()  # cached — no OCR poll
    assert len(_taps(recorded)) == 7  # 1 wake + 6 digit taps
    assert len(_swipes(recorded)) == 1
