"""Tests for `physiclaw.core.hardware.frame_reader` — the frame pump +
self-healing supervisor, exercised with scripted callables (no threads
except where the thread itself is under test, no cv2)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from physiclaw.core.hardware import frame_reader as fr_mod
from physiclaw.core.hardware.frame_reader import FrameReader


def _frame(h: int = 4, w: int = 4) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _reader(reads=None, reopen=None, raise_on_read: bool = False) -> FrameReader:
    """A FrameReader over a scripted read_frame; exhausted → (False, None)."""
    queue = list(reads or [])
    calls = {"reopen": 0}

    def read_frame():
        if raise_on_read:
            raise RuntimeError("hardware error")
        return queue.pop(0) if queue else (False, None)

    def _reopen():
        calls["reopen"] += 1
        if reopen is not None:
            reopen()

    r = FrameReader(read_frame, _reopen, label="test-cam")
    r._reopen_calls = calls  # test-side visibility
    return r


def _run_one_tick(r: FrameReader, mocker) -> None:
    """Drive one loop iteration: the backoff wait stops the loop."""
    r._stopped.clear()
    mocker.patch.object(r._stopped, "wait", side_effect=lambda t: r._stopped.set())
    r._loop()


# ---------- publish / consume ----------


def test_publish_and_fresh_frame_roundtrip() -> None:
    r = _reader()
    f = _frame()

    r.publish(f)

    assert r.fresh_frame() is f


def test_fresh_frame_returns_none_when_nothing_published(mocker) -> None:
    r = _reader()
    mocker.patch.object(FrameReader, "FRAME_WAIT_SECONDS", 0.01)

    assert r.fresh_frame() is None


def test_wait_frames_counts_new_publishes() -> None:
    import threading

    r = _reader()

    def publish(n: int) -> None:
        for _ in range(n):
            time.sleep(0.01)
            r.publish(_frame())

    t = threading.Thread(target=publish, args=(3,))
    t.start()
    try:
        assert r.wait_frames(3, timeout=2.0) is True
    finally:
        t.join(timeout=1.0)


def test_wait_frames_times_out_without_publisher() -> None:
    r = _reader()

    assert r.wait_frames(1, timeout=0.05) is False


# ---------- the loop ----------


def test_loop_publishes_good_frame(mocker) -> None:
    f = _frame()
    r = _reader(reads=[(True, f)])
    # Second tick has no frame → backoff wait stops the loop.
    _run_one_tick(r, mocker)

    assert r._frame is f
    assert r._frame_seq == 1


def test_loop_survives_read_exception(mocker) -> None:
    r = _reader(raise_on_read=True)

    _run_one_tick(r, mocker)  # must exit cleanly, not raise

    assert r._stopped.is_set()


def test_loop_reopens_after_stale(mocker) -> None:
    r = _reader()  # every read fails
    r._frame_time = 0.0  # very old → stale branch fires

    _run_one_tick(r, mocker)

    assert r._reopen_calls["reopen"] == 1


def test_loop_fatal_after_long_drought(mocker) -> None:
    r = _reader()
    r._first_fail_time = time.monotonic() - FrameReader.FATAL_AFTER_SECONDS - 1
    interrupt_spy = mocker.patch.object(fr_mod._thread, "interrupt_main")
    r._stopped.clear()

    r._loop()

    interrupt_spy.assert_called_once_with()


def test_loop_good_frame_resets_fail_clock() -> None:
    calls = {"n": 0}

    def read_frame():
        calls["n"] += 1
        if calls["n"] >= 2:
            r._stopped.set()
        return (True, _frame())

    r = FrameReader(read_frame, lambda: None, label="test-cam")
    r._first_fail_time = time.monotonic() - 30
    r._stopped.clear()

    r._loop()

    assert r._first_fail_time is None


# ---------- lifecycle / health ----------


def _endless_reader() -> FrameReader:
    """A reader whose pump always has a good frame (no exhaustion race)."""
    return FrameReader(lambda: (True, _frame()), lambda: None, label="test-cam")


def test_start_stop_lifecycle() -> None:
    r = _endless_reader()

    r.start()
    try:
        assert r.wait_frames(1, timeout=2.0)
        assert r.alive
    finally:
        r.stop()

    assert not r.alive


def test_healthy_requires_running_thread_and_recent_frame() -> None:
    r = _endless_reader()
    assert not r.healthy()  # not started

    r.start()
    try:
        assert r.wait_frames(1, timeout=2.0)
        assert r.healthy()
    finally:
        r.stop()
    assert not r.healthy()


def test_note_reopened_resets_stale_clock() -> None:
    r = _reader()
    r._frame_time = 0.0

    r.note_reopened()

    assert time.monotonic() - r._frame_time < 1.0


@pytest.mark.slow
def test_loop_backs_off_on_fail_path() -> None:
    # Real (short) backoff: two failing ticks must take at least one
    # READER_BACKOFF_SECONDS interval, not spin at full CPU.
    r = _reader()
    r.READER_BACKOFF_SECONDS = 0.05
    r._frame_time = time.monotonic()  # not stale → no reopen

    ticks = {"n": 0}
    orig_wait = r._stopped.wait

    def counting_wait(t):
        ticks["n"] += 1
        if ticks["n"] >= 2:
            r._stopped.set()
        return orig_wait(t)

    r._stopped.wait = counting_wait
    start = time.monotonic()
    r._loop()

    assert time.monotonic() - start >= 0.05
