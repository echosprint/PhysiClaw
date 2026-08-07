"""Tests for `physiclaw.core.hardware.transport` — SerialTransport.

Protocol behavior (ok/error/alarm/optional) is exercised end-to-end
through the arm in `test_arm.py`; this file pins what the transport
adds: the typed error taxonomy and the port lock.
"""

from __future__ import annotations

from collections import deque

import pytest

from physiclaw.core.hardware.device import (
    DeviceAlarm,
    DeviceNotFound,
    DeviceTimeout,
    HardwareError,
    ProtocolError,
)
from physiclaw.core.hardware.transport import SerialTransport


class _FakeSerial:
    def __init__(self, responses=()):
        self.writes: list[bytes] = []
        self.responses = deque(responses)
        self.in_waiting = 0
        self.closed = False
        self.locked_during_write: list[bool] = []
        self._transport: SerialTransport | None = None

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        if self._transport is not None:
            # Record whether the transport lock is held at wire time.
            acquired = self._transport._lock.acquire(blocking=False)
            if acquired:
                self._transport._lock.release()
            # RLock: same-thread acquire always succeeds, so probe via
            # a helper thread instead.
        return len(data)

    def readline(self) -> bytes:
        return self.responses.popleft() if self.responses else b""

    def read(self, n: int) -> bytes:
        return b""

    def reset_input_buffer(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def test_send_raises_typed_timeout() -> None:
    t = SerialTransport(_FakeSerial(responses=[b""] * 10))

    with pytest.raises(DeviceTimeout, match="GRBL not responding"):
        t.send("G0 X0 Y0")


def test_send_raises_typed_protocol_error() -> None:
    t = SerialTransport(_FakeSerial(responses=[b"error:23\n"]))

    with pytest.raises(ProtocolError, match="GRBL error"):
        t.send("G0 X0 Y0")


def test_send_raises_typed_alarm() -> None:
    t = SerialTransport(_FakeSerial(responses=[b"ALARM:1\n"]))

    with pytest.raises(DeviceAlarm, match="GRBL alarm"):
        t.send("G0 X0 Y0")


def test_hierarchy_is_runtime_error() -> None:
    """Pre-existing `except RuntimeError` handlers up the stack must keep
    catching driver failures — the taxonomy refines, never escapes."""
    for exc in (DeviceNotFound, DeviceTimeout, ProtocolError, DeviceAlarm):
        assert issubclass(exc, HardwareError)
        assert issubclass(exc, RuntimeError)


def test_send_serializes_against_concurrent_sender() -> None:
    """Two threads interleaving writes would desync GRBL's one-reply-per-
    command stream — the transport lock forces one wire interaction at a
    time."""
    import threading

    order: list[str] = []

    class _SlowSerial(_FakeSerial):
        def write(self, data: bytes) -> int:
            order.append(f"write:{data!r}")
            return len(data)

        def readline(self) -> bytes:
            # First reader parks until the main thread releases it.
            evt = events.popleft() if events else None
            if evt is not None:
                evt.wait(timeout=2.0)
            order.append("read")
            return b"ok\n"

    events: deque[threading.Event] = deque()
    gate = threading.Event()
    events.append(gate)
    t = SerialTransport(_SlowSerial())

    first = threading.Thread(target=lambda: t.send("G0 X1 Y1"))
    first.start()
    # Give the first send time to enter the locked region and park.
    for _ in range(100):
        if order and order[0].startswith("write"):
            break
        first.join(timeout=0.01)

    second = threading.Thread(target=lambda: t.send("G0 X2 Y2"))
    second.start()
    second.join(timeout=0.05)
    assert second.is_alive()  # blocked on the transport lock
    assert len([o for o in order if o.startswith("write")]) == 1

    gate.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)
    assert not second.is_alive()
    # Full serialization: write/read of the first completes before the
    # second's write hits the wire.
    assert order == ["write:b'G0 X1 Y1\\n'", "read", "write:b'G0 X2 Y2\\n'", "read"]


def test_close_closes_port() -> None:
    fake = _FakeSerial()
    t = SerialTransport(fake)

    t.close()

    assert fake.closed is True


# ---------- send_bounded (teardown path) ----------


def test_send_bounded_sends_when_wire_is_free() -> None:
    ser = _FakeSerial(responses=[b"ok\n"])
    t = SerialTransport(ser)

    assert t.send_bounded("M5", lock_timeout=0.05) is True
    assert b"M5\n" in ser.writes


def test_send_bounded_skips_when_wire_lock_is_held() -> None:
    # An abandoned holder (daemon killed mid-read at interpreter exit)
    # must not hang an atexit teardown — skip the command; the port
    # close is the backstop.
    import threading

    ser = _FakeSerial(responses=[b"ok\n"])
    t = SerialTransport(ser)
    held = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with t._lock:
            held.set()
            release.wait(timeout=5)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    held.wait(timeout=5)
    try:
        assert t.send_bounded("M5", lock_timeout=0.05) is False
        assert b"M5\n" not in ser.writes
    finally:
        release.set()
        holder.join(timeout=5)
