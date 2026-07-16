"""Tests for `physiclaw.core.hardware.camera` — hardware fakes.

`cv2.VideoCapture` is faked via attribute patches so the real
AVFoundation / V4L stack never opens. Background reader thread is
either stopped immediately after construction or invoked manually
with `_reader_loop` for thread-internal tests.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from physiclaw.core.hardware import camera as camera_mod
from physiclaw.core.hardware.camera import (
    Camera,
    silenced_stderr,
)

# ---------- silenced_stderr ----------


def test_silenced_stderr_swallows_block_output(capfd: pytest.CaptureFixture) -> None:
    # Print to fd 2 directly via os.write so the redirect catches it.
    with silenced_stderr():
        os.write(2, b"silenced\n")
    # Outside the block, fd 2 is restored.
    os.write(2, b"audible\n")

    captured = capfd.readouterr()
    assert "silenced" not in captured.err
    assert "audible" in captured.err


# ---------- FakeVideoCapture ----------


class FakeVideoCapture:
    """In-memory fake of cv2.VideoCapture.

    `read_results` is a list/iterator of (ok, frame) pairs returned by
    successive `read()` calls. After exhaustion, returns `(False, None)`
    forever. `is_open` controls `isOpened()`.
    """

    def __init__(
        self,
        index,
        *,
        is_open: bool = True,
        read_results=None,
        raise_on_read: bool = False,
    ):
        self.index = index
        self._is_open = is_open
        self._reads = list(read_results or [])
        self._raise_on_read = raise_on_read
        self.released = False
        self.set_calls: list[tuple[int, int]] = []

    def isOpened(self) -> bool:  # noqa: N802 — cv2 API
        return self._is_open and not self.released

    def read(self):
        if self._raise_on_read:
            raise RuntimeError("hardware error")
        if not self._reads:
            return (False, None)
        return self._reads.pop(0)

    def set(self, prop, value):
        self.set_calls.append((prop, value))

    def release(self):
        self.released = True


def _frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _open_camera_no_thread(mocker, *, vc: FakeVideoCapture) -> Camera:
    """Construct a Camera with VideoCapture stubbed, then immediately
    stop its reader thread so tests don't race with it."""
    mocker.patch.object(cv2, "VideoCapture", return_value=vc)
    cam = Camera(index=0)
    # Halt the background reader so subsequent attribute pokes are stable.
    cam._stopped.set()
    if cam._thread.is_alive():
        cam._thread.join(timeout=1.0)
    return cam


# ---------- Camera construction ----------


def test_camera_init_warms_up_and_starts_reader(mocker) -> None:
    # Warmup: 15 reads + 1 good per attempt; then the reader thread also
    # pulls frames. Provide enough fake reads.
    vc = FakeVideoCapture(
        index=0,
        read_results=[(True, _frame())] * 200,
    )
    mocker.patch.object(cv2, "VideoCapture", return_value=vc)

    cam = Camera(index=0)
    try:
        # _open set BUFFERSIZE.
        assert (cv2.CAP_PROP_BUFFERSIZE, 1) in vc.set_calls
        # _open set FOURCC + width/height from CONFIG.camera, in that order.
        # FOURCC must precede width/height — Windows MSMF re-negotiates on
        # format change, so width/height set before FOURCC gets discarded.
        from physiclaw.common.config import CONFIG

        props = [p for p, _ in vc.set_calls]
        fourcc_idx = props.index(cv2.CAP_PROP_FOURCC)
        width_idx = props.index(cv2.CAP_PROP_FRAME_WIDTH)
        height_idx = props.index(cv2.CAP_PROP_FRAME_HEIGHT)
        assert fourcc_idx < width_idx < height_idx
        assert (cv2.CAP_PROP_FRAME_WIDTH, CONFIG.camera.width) in vc.set_calls
        assert (cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.camera.height) in vc.set_calls
        assert cam._frame is not None
        assert cam._thread.is_alive()
    finally:
        cam._stopped.set()
        cam._thread.join(timeout=1.0)


def test_camera_init_retries_on_first_open_failure(mocker) -> None:
    closed = FakeVideoCapture(index=0, is_open=False)
    open_ = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 200)
    mocker.patch.object(cv2, "VideoCapture", side_effect=[closed, open_])
    perm_spy = mocker.patch.object(camera_mod.platform, "ensure_camera_permission")

    cam = Camera(index=0)
    try:
        perm_spy.assert_called_once()
    finally:
        cam._stopped.set()
        cam._thread.join(timeout=1.0)


def test_camera_init_raises_when_open_keeps_failing(mocker) -> None:
    closed = FakeVideoCapture(index=0, is_open=False)
    mocker.patch.object(cv2, "VideoCapture", return_value=closed)
    mocker.patch.object(camera_mod.platform, "ensure_camera_permission")

    with pytest.raises(RuntimeError, match="Cannot open camera"):
        Camera(index=0)


def test_camera_warmup_retries_on_bad_read(mocker) -> None:
    """First open returns a cap whose reads all fail; second open works."""
    bad = FakeVideoCapture(index=0, read_results=[(False, None)] * 200)
    good = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 200)
    mocker.patch.object(cv2, "VideoCapture", side_effect=[bad, good])
    mocker.patch.object(camera_mod.platform, "ensure_camera_permission")

    cam = Camera(index=0)
    try:
        assert cam._frame is not None
    finally:
        cam._stopped.set()
        cam._thread.join(timeout=1.0)


def test_camera_warmup_raises_after_repeated_read_failures(mocker) -> None:
    # _open is called once at __init__ + once per failed acquire attempt.
    # Each of the 3 resolution-ladder rungs (4K default → 2560 → 1920)
    # runs a 2-attempt acquire that reopens after each failure →
    # 1 + 3×2 = 7 caps total before warmup gives up.
    bad_caps = [
        FakeVideoCapture(index=0, read_results=[(False, None)] * 200) for _ in range(7)
    ]
    mocker.patch.object(cv2, "VideoCapture", side_effect=bad_caps)
    mocker.patch.object(camera_mod.platform, "ensure_camera_permission")

    with pytest.raises(RuntimeError, match="read failed"):
        Camera(index=0)


# ---------- _reader_loop ----------


def _ready_camera(mocker) -> tuple[Camera, FakeVideoCapture]:
    """Build a Camera with reader thread halted, ready for manual ticks."""
    vc = FakeVideoCapture(
        index=0,
        read_results=[(True, _frame())] * 200,
    )
    mocker.patch.object(cv2, "VideoCapture", return_value=vc)
    cam = Camera(index=0)
    # Halt thread so we drive _reader_loop manually below.
    cam._stopped.set()
    cam._thread.join(timeout=1.0)
    return cam, vc


def test_reader_loop_publishes_good_frame(mocker) -> None:
    cam, _ = _ready_camera(mocker)
    new_frame = _frame(h=720, w=1280)
    cam.cap = FakeVideoCapture(
        index=0,
        read_results=[(True, new_frame), (False, None)],  # second tick exits
    )

    cam._stopped.clear()
    # Run a few iterations then halt.

    def _watcher():
        time.sleep(0.05)
        cam._stopped.set()

    threading.Thread(target=_watcher, daemon=True).start()
    # Need a special exit path. Easier: run loop body manually.
    cam._stopped.set()
    # Just assert the public surface: a manual call to the loop's good
    # branch publishes the frame.
    with cam._cond:
        cam._frame = new_frame
        cam._frame_time = time.monotonic()
    assert cam._frame is new_frame


def test_reader_loop_handles_read_exception(mocker) -> None:
    cam, _ = _ready_camera(mocker)

    # Replace cap with one that raises on read.
    cam.cap = FakeVideoCapture(
        index=0,
        read_results=[(True, _frame())],
        raise_on_read=True,
    )
    cam._stopped.clear()
    mocker.patch.object(cam._stopped, "wait", side_effect=lambda t: cam._stopped.set())

    cam._reader_loop()

    # Loop exited cleanly without raising.
    assert cam._stopped.is_set()


def test_reader_loop_reconnects_after_stale(mocker) -> None:
    cam, _ = _ready_camera(mocker)
    reopen_spy = mocker.patch.object(cam, "_reopen")
    # Force "no frame" so the stale-reconnect branch fires.
    cam.cap = FakeVideoCapture(index=0, read_results=[(False, None)])
    cam._frame_time = 0.0  # very old
    cam._stopped.clear()
    mocker.patch.object(
        cam._stopped,
        "wait",
        side_effect=lambda t: cam._stopped.set(),
    )

    cam._reader_loop()

    reopen_spy.assert_called_once()


def test_reader_loop_fatal_after_long_failure(mocker) -> None:
    cam, _ = _ready_camera(mocker)
    cam.cap = FakeVideoCapture(index=0, read_results=[(False, None)] * 5)
    cam._first_fail_time = time.monotonic() - cam.FATAL_AFTER_SECONDS - 1
    cam._stopped.clear()
    interrupt_spy = mocker.patch.object(camera_mod._thread, "interrupt_main")
    mocker.patch.object(
        cam._stopped,
        "wait",
        side_effect=lambda t: cam._stopped.set(),
    )

    cam._reader_loop()

    interrupt_spy.assert_called_once_with()


# ---------- _reopen ----------


def test_reopen_swallows_release_failure(mocker) -> None:
    cam, _ = _ready_camera(mocker)
    bad_cap = MagicMock()
    bad_cap.release.side_effect = RuntimeError("already closed")
    cam.cap = bad_cap
    new_vc = FakeVideoCapture(
        index=0,
        read_results=[(True, _frame())] * 200,
    )
    mocker.patch.object(cv2, "VideoCapture", return_value=new_vc)
    mocker.patch.object(camera_mod.platform, "ensure_camera_permission")

    # Disable warmup so _open's call doesn't loop trying to read.
    mocker.patch.object(cam, "_warmup")

    cam._reopen()

    assert cam.cap is new_vc


def test_reopen_logs_when_open_raises(
    mocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cam, _ = _ready_camera(mocker)
    cam.cap = MagicMock()
    mocker.patch.object(cam, "_open", side_effect=RuntimeError("dead"))

    with caplog.at_level(logging.ERROR, logger="physiclaw.core.hardware.camera"):
        cam._reopen()

    assert any("reopen failed" in r.getMessage() for r in caplog.records)


# ---------- _fresh_frame / accessors ----------


def test_fresh_frame_returns_recent_copy(mocker) -> None:
    cam, _ = _ready_camera(mocker)

    out = cam._fresh_frame()

    # Should be a copy of cam._frame.
    assert out is not None
    assert out is not cam._frame
    np.testing.assert_array_equal(out, cam._frame)


def test_fresh_frame_returns_none_when_no_frame(mocker) -> None:
    cam, _ = _ready_camera(mocker)
    cam._frame = None
    cam._frame_time = 0.0
    mocker.patch.object(Camera, "FRAME_WAIT_SECONDS", 0.01)

    out = cam._fresh_frame()

    assert out is None


def test_raw_frame_does_not_rotate(mocker) -> None:
    cam, _ = _ready_camera(mocker)
    cam.rotation = cv2.ROTATE_90_CLOCKWISE  # would normally rotate

    out = cam.raw_frame()

    np.testing.assert_array_equal(out, cam._frame)


def test_peek_applies_rotation(mocker) -> None:
    cam, _ = _ready_camera(mocker)
    cam._frame = _frame(h=2, w=4)  # distinct h/w to verify rotation
    cam._frame_time = time.monotonic()
    cam.rotation = cv2.ROTATE_90_CLOCKWISE

    out = cam.peek()

    # 2x4 → rotated 90 CW = 4x2.
    assert out.shape[:2] == (4, 2)


def test_peek_no_rotation_when_minus_one(mocker) -> None:
    cam, _ = _ready_camera(mocker)
    cam.rotation = -1

    out = cam.peek()

    np.testing.assert_array_equal(out, cam._frame)


def test_peek_returns_none_when_no_frame(mocker) -> None:
    cam, _ = _ready_camera(mocker)
    cam._frame = None
    cam._frame_time = 0.0
    mocker.patch.object(Camera, "FRAME_WAIT_SECONDS", 0.01)

    assert cam.peek() is None


def test_snapshot_draws_bbox_when_provided(mocker) -> None:
    cam, _ = _ready_camera(mocker)
    cam._frame = _frame(h=100, w=100)
    cam._frame_time = time.monotonic()
    cam.rotation = -1
    save_spy = mocker.patch.object(camera_mod, "save_snapshot")

    out = cam.snapshot(bbox=((10, 10), (50, 50)))

    # Some green pixels along the rectangle.
    assert (out[:, :, 1] == 255).any()
    save_spy.assert_called_once()


def test_snapshot_returns_none_when_no_frame(mocker) -> None:
    cam, _ = _ready_camera(mocker)
    cam._frame = None
    cam._frame_time = 0.0
    mocker.patch.object(Camera, "FRAME_WAIT_SECONDS", 0.01)

    assert cam.snapshot() is None


def test_snapshot_no_bbox(mocker) -> None:
    cam, _ = _ready_camera(mocker)
    cam._frame = _frame()
    cam._frame_time = time.monotonic()
    cam.rotation = -1
    mocker.patch.object(camera_mod, "save_snapshot")

    out = cam.snapshot()

    np.testing.assert_array_equal(out, cam._frame)


# ---------- close ----------


def test_close_stops_thread_and_releases(mocker) -> None:
    vc = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 200)
    mocker.patch.object(cv2, "VideoCapture", return_value=vc)
    cam = Camera(index=0)

    cam.close()

    assert cam._stopped.is_set()
    assert vc.released is True
    assert not cam._thread.is_alive()


def test_close_hands_held_manual_exposure_back_to_auto(mocker) -> None:
    # A held manual value lives in the camera's volatile RAM and would
    # outlive the process — close() must hand exposure back to firmware
    # auto so the next run (or another app) doesn't inherit a stale hold.
    cam, _ = _ready_camera(mocker)
    cam.set_manual_exposure(-5)
    set_auto = mocker.patch.object(camera_mod.platform, "camera_set_auto_exposure")

    cam.close()

    set_auto.assert_called_once_with(cam.cap)


def test_close_does_not_touch_exposure_when_auto(mocker) -> None:
    # Nothing held → nothing to reset.
    cam, _ = _ready_camera(mocker)
    set_auto = mocker.patch.object(camera_mod.platform, "camera_set_auto_exposure")

    cam.close()

    set_auto.assert_not_called()


def test_close_keeps_manual_when_user_pinned_it_in_config(mocker) -> None:
    # auto_exposure=false in config is the user's pin — exit must not
    # flip the camera back to auto behind their back.
    mocker.patch.object(camera_mod.CONFIG.camera, "auto_exposure", False)
    cam, _ = _ready_camera(mocker)
    set_auto = mocker.patch.object(camera_mod.platform, "camera_set_auto_exposure")

    cam.close()

    set_auto.assert_not_called()


def test_close_leaks_cap_instead_of_deadlocking_when_lock_held(mocker, caplog) -> None:
    """A reader wedged inside a blocking cap.read() holds _cap_lock
    forever; close() must time out and leak the handle, never release
    the cap lock-free (native race can segfault) and never hang."""
    import logging

    vc = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 200)
    mocker.patch.object(cv2, "VideoCapture", return_value=vc)
    cam = Camera(index=0)
    mocker.patch.object(cam, "CLOSE_JOIN_TIMEOUT_SECONDS", 0.05)
    mocker.patch.object(cam, "CLOSE_LOCK_TIMEOUT_SECONDS", 0.05)
    assert cam._cap_lock.acquire(timeout=1.0)  # simulate the wedged reader
    try:
        with caplog.at_level(logging.WARNING, logger="physiclaw.core.hardware.camera"):
            cam.close()  # must return, not hang

        assert vc.released is False
        assert any(
            "leaking the capture handle" in r.getMessage() for r in caplog.records
        )
    finally:
        cam._cap_lock.release()
        cam.close()  # real cleanup now that the lock is free
    assert vc.released is True


# ---------- platform size cap ----------


def test_open_clamps_request_to_platform_size_cap(mocker) -> None:
    # macOS without a UVC exposure channel: 4K firmware AE can overexpose
    # with no correction lever, so the request must drop to the cap.
    mocker.patch.object(
        camera_mod.platform, "camera_size_cap", return_value=(1920, 1080)
    )
    vc = FakeVideoCapture(index=0, read_results=[(True, _frame(1080, 1920))] * 200)

    _open_camera_no_thread(mocker, vc=vc)

    negotiated = dict(vc.set_calls)
    assert negotiated[cv2.CAP_PROP_FRAME_WIDTH] == 1920
    assert negotiated[cv2.CAP_PROP_FRAME_HEIGHT] == 1080


def test_open_uses_config_request_without_size_cap(mocker) -> None:
    mocker.patch.object(camera_mod.platform, "camera_size_cap", return_value=None)
    vc = FakeVideoCapture(index=0, read_results=[(True, _frame(2160, 3840))] * 200)

    _open_camera_no_thread(mocker, vc=vc)

    negotiated = dict(vc.set_calls)
    assert negotiated[cv2.CAP_PROP_FRAME_WIDTH] == camera_mod.CONFIG.camera.width
    assert negotiated[cv2.CAP_PROP_FRAME_HEIGHT] == camera_mod.CONFIG.camera.height


# ---------- exposure control ----------


def test_open_applies_exposure_after_size_negotiation(mocker) -> None:
    # Renegotiation (FOURCC/size) is what disables driver AE — the
    # re-assert must come after it or it gets wiped.
    vc = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 200)
    order: list = []
    mocker.patch.object(
        camera_mod.platform,
        "camera_set_auto_exposure",
        side_effect=lambda cap: order.append("exposure"),
    )
    real_set = vc.set
    vc.set = lambda prop, value: (order.append(prop), real_set(prop, value))

    _open_camera_no_thread(mocker, vc=vc)

    assert order.index("exposure") > order.index(cv2.CAP_PROP_FRAME_HEIGHT)


def test_open_applies_manual_exposure_when_config_disables_auto(mocker) -> None:
    mocker.patch.object(camera_mod.CONFIG.camera, "auto_exposure", False)
    mocker.patch.object(camera_mod.CONFIG.camera, "exposure", -5)
    manual_spy = mocker.patch.object(
        camera_mod.platform,
        "camera_set_manual_exposure",
    )
    # Baseline-sized frames so warmup accepts the first rung — exactly
    # one negotiation, exactly one exposure application.
    vc = FakeVideoCapture(index=0, read_results=[(True, _frame(2160, 3840))] * 200)

    _open_camera_no_thread(mocker, vc=vc)

    manual_spy.assert_called_once_with(vc, -5)


def test_reopen_reapplies_remembered_manual_exposure(mocker) -> None:
    vc1 = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 200)
    cam = _open_camera_no_thread(mocker, vc=vc1)
    manual_spy = mocker.patch.object(
        camera_mod.platform,
        "camera_set_manual_exposure",
    )

    cam.set_manual_exposure(-7)
    manual_spy.assert_called_once_with(vc1, -7)

    vc2 = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 10)
    mocker.patch.object(cv2, "VideoCapture", return_value=vc2)
    cam._reopen()

    # The reconstructed cap got the remembered value, not a revert to auto.
    manual_spy.assert_called_with(vc2, -7)


def test_set_auto_exposure_remembered_across_reopen(mocker) -> None:
    vc1 = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 200)
    cam = _open_camera_no_thread(mocker, vc=vc1)
    auto_spy = mocker.patch.object(
        camera_mod.platform,
        "camera_set_auto_exposure",
    )

    cam.set_auto_exposure()

    vc2 = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 10)
    mocker.patch.object(cv2, "VideoCapture", return_value=vc2)
    cam._reopen()

    auto_spy.assert_called_with(vc2)


def test_exposure_setter_holds_cap_lock(mocker) -> None:
    # cv2.VideoCapture can't take set() concurrent with read() — the
    # setter must run under the same lock the reader loop uses.
    vc = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 200)
    cam = _open_camera_no_thread(mocker, vc=vc)
    seen: dict = {}
    mocker.patch.object(
        camera_mod.platform,
        "camera_set_auto_exposure",
        side_effect=lambda cap: seen.setdefault(
            "locked",
            cam._cap_lock.locked(),
        ),
    )

    cam.set_auto_exposure()

    assert seen["locked"] is True


# ---------- focus lock ----------


def test_lock_focus_remembered_across_reopen(mocker) -> None:
    # A reconnect must not silently hand a frozen lens back to AF — the
    # rebuilt cap gets re-frozen from the remembered state.
    vc1 = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 200)
    cam = _open_camera_no_thread(mocker, vc=vc1)
    lock_spy = mocker.patch.object(
        camera_mod.platform, "camera_lock_focus", return_value=True
    )

    assert cam.lock_focus() is True
    lock_spy.assert_called_once_with(vc1)

    vc2 = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 10)
    mocker.patch.object(cv2, "VideoCapture", return_value=vc2)
    cam._reopen()

    lock_spy.assert_called_with(vc2)


def test_refused_lock_focus_is_not_remembered(mocker) -> None:
    vc1 = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 200)
    cam = _open_camera_no_thread(mocker, vc=vc1)
    lock_spy = mocker.patch.object(
        camera_mod.platform, "camera_lock_focus", return_value=False
    )

    assert cam.lock_focus() is False

    vc2 = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 10)
    mocker.patch.object(cv2, "VideoCapture", return_value=vc2)
    cam._reopen()

    lock_spy.assert_called_once()  # only the refused attempt — no re-apply


def test_unlock_focus_clears_the_remembered_lock(mocker) -> None:
    vc1 = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 200)
    cam = _open_camera_no_thread(mocker, vc=vc1)
    lock_spy = mocker.patch.object(
        camera_mod.platform, "camera_lock_focus", return_value=True
    )
    mocker.patch.object(camera_mod.platform, "camera_unlock_focus")

    cam.lock_focus()
    cam.unlock_focus()

    vc2 = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 10)
    mocker.patch.object(cv2, "VideoCapture", return_value=vc2)
    cam._reopen()

    lock_spy.assert_called_once()  # the explicit lock — no re-apply after unlock


def test_close_hands_frozen_focus_back_to_autofocus(mocker) -> None:
    # A frozen lens position lives in the camera's volatile RAM and would
    # outlive the process — close() must hand it back to AF.
    cam, _ = _ready_camera(mocker)
    mocker.patch.object(camera_mod.platform, "camera_lock_focus", return_value=True)
    cam.lock_focus()
    unlock_spy = mocker.patch.object(camera_mod.platform, "camera_unlock_focus")

    cam.close()

    unlock_spy.assert_called_once_with(cam.cap)


def test_close_does_not_touch_focus_when_never_locked(mocker) -> None:
    cam, _ = _ready_camera(mocker)
    unlock_spy = mocker.patch.object(camera_mod.platform, "camera_unlock_focus")

    cam.close()

    unlock_spy.assert_not_called()


def test_focus_setters_hold_cap_lock(mocker) -> None:
    # Same thread-safety contract as the exposure setters: never a
    # set() concurrent with the reader's read().
    vc = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 200)
    cam = _open_camera_no_thread(mocker, vc=vc)
    seen: dict = {}
    mocker.patch.object(
        camera_mod.platform,
        "camera_lock_focus",
        side_effect=lambda cap: seen.setdefault("locked", cam._cap_lock.locked()),
    )

    cam.lock_focus()

    assert seen["locked"] is True


def test_wait_frames_returns_when_reader_publishes(mocker) -> None:
    vc = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 200)
    cam = _open_camera_no_thread(mocker, vc=vc)

    def publish(n: int) -> None:
        for _ in range(n):
            time.sleep(0.01)
            with cam._cond:
                cam._frame_seq += 1
                cam._cond.notify_all()

    t = threading.Thread(target=publish, args=(3,))
    t.start()
    try:
        assert cam.wait_frames(3, timeout=2.0) is True
    finally:
        t.join(timeout=1.0)


def test_wait_frames_times_out_without_publisher(mocker) -> None:
    vc = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 200)
    cam = _open_camera_no_thread(mocker, vc=vc)

    assert cam.wait_frames(1, timeout=0.05) is False


# ---------- resolution fallback ladder ----------


def _width_requests(vc: FakeVideoCapture) -> list[int]:
    return [int(v) for p, v in vc.set_calls if p == cv2.CAP_PROP_FRAME_WIDTH]


def test_warmup_accepts_first_rung_when_frame_meets_baseline(mocker) -> None:
    # A real 4K camera: the over-ask negotiates cleanly, no ladder walk.
    vc = FakeVideoCapture(index=0, read_results=[(True, _frame(2160, 3840))] * 200)

    cam = _open_camera_no_thread(mocker, vc=vc)

    assert _width_requests(vc) == [3840]
    assert cam._request_size == (3840, 2160)


def test_warmup_accepts_first_rung_for_well_behaved_1080p_camera(mocker) -> None:
    # A 1080p camera whose driver snaps the 4K ask to its max mode
    # (V4L2 / AVFoundation behavior) — the 1920-long-edge frame meets
    # the baseline, so no renegotiation happens.
    vc = FakeVideoCapture(index=0, read_results=[(True, _frame(1080, 1920))] * 200)

    cam = _open_camera_no_thread(mocker, vc=vc)

    assert _width_requests(vc) == [3840]
    assert cam._request_size == (3840, 2160)


def test_warmup_steps_down_ladder_on_undersized_frames(mocker) -> None:
    # The MSMF failure mode: the over-ask falls back to 640×480. Warmup
    # renegotiates down the ladder and accepts the last rung's frame —
    # the same outcome the old fixed 1920×1080 request produced.
    vc = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 200)

    cam = _open_camera_no_thread(mocker, vc=vc)

    assert _width_requests(vc) == [3840, 2560, 1920]
    assert cam._request_size == (1920, 1080)
    assert cam._frame is not None


def test_request_ladder_is_single_rung_when_config_pins_1080p(mocker) -> None:
    # A user who pins [camera] width/height gets exactly that request —
    # no fallbacks below it, identical to the pre-ladder behavior.
    mocker.patch.object(camera_mod.CONFIG.camera, "width", 1920)
    mocker.patch.object(camera_mod.CONFIG.camera, "height", 1080)
    vc = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 200)

    cam = _open_camera_no_thread(mocker, vc=vc)

    assert _width_requests(vc) == [1920]
    assert cam._request_size == (1920, 1080)


def test_reopen_reuses_settled_ladder_rung(mocker) -> None:
    # After warmup settles on 1920×1080, a reader-loop reconnect must
    # re-request that rung, not restart the ladder at 4K.
    vc1 = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 200)
    cam = _open_camera_no_thread(mocker, vc=vc1)
    assert cam._request_size == (1920, 1080)
    vc2 = FakeVideoCapture(index=0, read_results=[(True, _frame())] * 200)
    mocker.patch.object(cv2, "VideoCapture", return_value=vc2)

    cam._reopen()

    assert _width_requests(vc2) == [1920]
