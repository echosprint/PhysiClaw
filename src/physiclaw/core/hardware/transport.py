"""SerialTransport — the GRBL line protocol over one owned serial port.

Owns the send/ok handshake, the ``?`` status query, and the port lock.
The lock makes reply-stream integrity a property of the transport
itself: two threads interleaving writes would desync GRBL's one-reply-
per-command stream (a later command then reads a stale error), so every
wire interaction serializes here instead of relying on each caller to
hold the rig-level busy lock.

``StylusArm`` composes this the same way it composes ``Solenoid`` —
gesture/G-code logic stays in the arm, wire mechanics live here.
"""

import logging
import threading
import time

from physiclaw.core.hardware.device import DeviceAlarm, DeviceTimeout, ProtocolError

log = logging.getLogger(__name__)

# Error codes that mean "this firmware doesn't take this setting at
# runtime" — the value lives in YAML config (FluidNC) or isn't supported.
# Swallowed only when a command is sent with optional=True. Matches
# scripts/grbl_solenoid_test.py.
_OPTIONAL_ERRORS = frozenset({"error:3", "error:162"})


class SerialTransport:
    """One GRBL wire: an open ``serial.Serial`` plus the line protocol."""

    def __init__(self, ser) -> None:
        self.ser = ser
        # RLock, not Lock: cheap insurance if a caller ever nests a send
        # inside a send-driven callback; never a behavioral difference on
        # the current straight-line call paths.
        self._lock = threading.RLock()

    def send(self, cmd: str, wait_ok: bool = True, optional: bool = False) -> None:
        """Send a single command, block until 'ok'.

        optional=True swallows the "setting not accepted at runtime" error
        codes (error:3 / error:162) so PWM-config writes ($32/$30) work on
        FluidNC (which takes them from YAML) and a bare GRBL board alike.
        """
        with self._lock:
            log.debug(f"[GRBL] sent {cmd}")
            # LF only — NOT CRLF. FluidNC v4 treats the trailing `\r` + `\n`
            # as two line endings (the command + an empty line) and acks BOTH
            # with `ok`; that extra `ok` lags every reply by one and desyncs
            # the stream (a later command then reads a stale error). `\n` is
            # the canonical GRBL terminator and yields exactly one reply per
            # command.
            self.ser.write((cmd + "\n").encode())

            if not wait_ok:
                return

            retries = 0
            while True:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    retries = 0
                    log.debug(f"[GRBL] got {line}")
                else:
                    retries += 1
                    if retries > 3:
                        raise DeviceTimeout(f"GRBL not responding, command: {cmd}")
                    continue
                if line == "ok":
                    break
                if line.startswith("error"):
                    if optional and line.replace(" ", "") in _OPTIONAL_ERRORS:
                        # FluidNC follows a rejected setting with a trailing
                        # `[MSG:ERR: ...]` line (no `ok`). It's a
                        # non-terminator, so the next command's read loop
                        # skips it harmlessly — no draining needed now that
                        # we send LF (single reply each).
                        log.debug(
                            f"[GRBL] {cmd}: {line} — not accepted at runtime, skipping"
                        )
                        break
                    raise ProtocolError(f"GRBL error: {line}  command: {cmd}")
                if line.startswith("ALARM"):
                    raise DeviceAlarm(f"GRBL alarm: {line}, call unlock() first")

    def query_status(self) -> str:
        """Query current status (`?`), return the `<...>` status line or ''."""
        with self._lock:
            self.ser.write(b"?")
            time.sleep(0.1)
            resp = self.ser.read(self.ser.in_waiting or 64).decode(
                "utf-8", errors="ignore"
            )
        for line in resp.splitlines():
            if line.startswith("<"):
                log.debug(f"[GRBL] got {line}")
                return line
        return ""

    def read_startup(self, max_bytes: int = 256) -> str:
        """Drain and return whatever the firmware printed at boot."""
        with self._lock:
            return self.ser.read(self.ser.in_waiting or max_bytes).decode(
                "utf-8", errors="ignore"
            )

    def reset_input_buffer(self) -> None:
        with self._lock:
            self.ser.reset_input_buffer()

    def send_bounded(self, cmd: str, *, lock_timeout: float) -> bool:
        """``send``, but give up if the wire lock can't be won in
        ``lock_timeout`` — for teardown, where a holder abandoned inside a
        blocking read (a daemon killed at interpreter exit) must not hang
        an atexit handler (``Camera.close``'s bounded-acquire precedent).
        Returns whether the command was sent. The port close remains the
        true backstop: it unblocks the stuck reader, and the DTR drop
        resets the board."""
        if not self._lock.acquire(timeout=lock_timeout):
            log.warning(
                "[GRBL] wire lock held > %.1fs — skipping %r", lock_timeout, cmd
            )
            return False
        try:
            self.send(cmd)
            return True
        finally:
            self._lock.release()

    def close(self) -> None:
        self.ser.close()
        log.debug("[GRBL] serial port closed")
