"""FrameReader — keep a capture stream warm and self-healing.

A background daemon thread continuously pulls frames via an injected
``read_frame`` callable and publishes the latest one; callers get it
through ``fresh_frame()`` / ``wait_frames()`` without blocking on the
native capture. The supervisor policy lives here too: reconnect (via
the injected ``reopen``) when frames go stale, give up and interrupt
the main thread when the drought outlasts ``FATAL_AFTER_SECONDS`` —
reopen can't revive a stream the OS has cut (display sleep, bus
suspend), and dying cleanly beats spamming the log forever.

Nothing in here knows about cv2 or PhysiClaw semantics — ``Camera``
composes a FrameReader with a lock-wrapped ``cap.read`` and its own
``_reopen``, which is what makes this unit testable with scripted
callables instead of thread-timing-sensitive fakes.
"""

import _thread
import logging
import threading
import time

log = logging.getLogger(__name__)


class FrameReader:
    """Background frame pump + self-healing supervisor over one stream."""

    # If the reader gets no frame for this long, ask the owner to
    # close+reopen the stream. Recovers from real disconnects.
    STALE_RECONNECT_SECONDS = 5.0

    # If the reader gets no frame for this long, give up and raise
    # KeyboardInterrupt in the main thread (via _thread.interrupt_main —
    # cross-platform; os.kill(SIGINT) on Windows would TerminateProcess
    # and skip atexit).
    FATAL_AFTER_SECONDS = 60.0

    # Max time fresh_frame() waits for the loop to produce a frame
    # before returning whatever it last had (or None).
    FRAME_WAIT_SECONDS = 2.0

    # A frame older than this is treated as "not yet fresh" by
    # fresh_frame() — it'll wait for the loop to publish a newer one.
    FRESH_MAX_AGE_SECONDS = 1.0

    # Backoff after a failed read or reconnect, so a permanently broken
    # stream doesn't spin-loop the reader thread.
    READER_BACKOFF_SECONDS = 0.5

    # stop(): how long to wait for the thread. A reader wedged inside a
    # blocking native read can't be joined; the owner handles the
    # consequences (see Camera.close's lock-timeout leak).
    JOIN_TIMEOUT_SECONDS = 3.0

    def __init__(self, read_frame, reopen, *, label: str = "camera") -> None:
        """``read_frame``: () -> (ok, frame) — the owner serializes it
        against its own setters/reopen/close. ``reopen``: () -> None —
        close and reopen the underlying stream; called on staleness."""
        self._read_frame = read_frame
        self._reopen = reopen
        self._label = label
        self._frame = None
        self._frame_time = 0.0
        # Monotonic count of published frames — the settle primitive for
        # exposure tuning (`wait_frames`): after a property set, "wait N
        # frames" is deterministic where "sleep N/fps" is not.
        self._frame_seq = 0
        # Separate from _frame_time because a reopen resets _frame_time —
        # which would otherwise postpone the FATAL_AFTER_SECONDS check
        # indefinitely. _first_fail_time resets only on a real good frame.
        self._first_fail_time: float | None = None
        self._cond = threading.Condition()
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name=f"{label}-reader", daemon=True
        )

    # ─── Lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.JOIN_TIMEOUT_SECONDS)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def healthy(self) -> bool:
        """Live check: loop running, a frame published recently, and no
        ongoing read drought. The drought check matters because a reopen
        resets the stale clock without producing a frame — a dead stream
        that keeps nominally reopening would otherwise read healthy for
        most of every stale cycle."""
        with self._cond:
            age = time.monotonic() - self._frame_time
        return (
            self._thread.is_alive()
            and not self._stopped.is_set()
            and self._first_fail_time is None
            and age < self.STALE_RECONNECT_SECONDS
        )

    # ─── Publish / consume ──────────────────────────────────────

    def publish(self, frame) -> None:
        """Publish a frame — the loop's good path, and the owner's warmup
        seed before the thread starts."""
        with self._cond:
            self._frame = frame
            self._frame_time = time.monotonic()
            self._frame_seq += 1
            self._cond.notify_all()

    def note_reopened(self) -> None:
        """Reset the stale clock after the owner rebuilt the stream, so
        one reopen isn't immediately followed by another."""
        with self._cond:
            self._frame_time = time.monotonic()

    def wait_frames(self, n: int, timeout: float = 5.0) -> bool:
        """Block until `n` MORE frames are published (or timeout).

        The settle primitive for exposure tuning: drivers apply property
        changes with latency, so "wait N fresh frames" is the
        deterministic version of "sleep a bit". Returns False on timeout
        (reader stalled) — callers treat that as a failed meter."""
        with self._cond:
            target = self._frame_seq + n
            return self._cond.wait_for(
                lambda: self._frame_seq >= target,
                timeout=timeout,
            )

    def fresh_frame(self):
        """The latest published frame, or ``None``.

        Waits up to ``FRAME_WAIT_SECONDS`` for a frame fresher than
        ``FRESH_MAX_AGE_SECONDS``; otherwise returns whatever the loop
        last had. Returns the internal reference — the owner copies
        outside the lock so the copy can't stall the next publish."""
        with self._cond:
            self._cond.wait_for(
                lambda: (
                    self._frame is not None
                    and time.monotonic() - self._frame_time < self.FRESH_MAX_AGE_SECONDS
                ),
                timeout=self.FRAME_WAIT_SECONDS,
            )
            return self._frame

    # ─── The loop ───────────────────────────────────────────────

    def _loop(self) -> None:
        """Pull frames continuously so the native pipeline never goes idle.

        ``read_frame`` blocks for the next native-FPS frame, so the loop
        self-paces — no explicit sleep needed in the steady state.
        """
        while not self._stopped.is_set():
            try:
                ok, frame = self._read_frame()
            except Exception as e:
                log.warning(f"{self._label}: read raised {e!r}")
                ok, frame = False, None

            now = time.monotonic()

            if ok and frame is not None:
                self._first_fail_time = None
                self.publish(frame)
                continue

            if self._first_fail_time is None:
                self._first_fail_time = now
            fail_duration = now - self._first_fail_time
            if fail_duration >= self.FATAL_AFTER_SECONDS:
                log.error(
                    f"{self._label}: no frames for {fail_duration:.0f}s "
                    "— giving up and exiting process "
                    "(display sleep / bus suspend / hardware gone)"
                )
                _thread.interrupt_main()
                return

            with self._cond:
                stale = now - self._frame_time
            if stale > self.STALE_RECONNECT_SECONDS:
                self._reopen()
            # Unconditional on the fail path: caps iteration rate if
            # read_frame raises or returns empty every tick (e.g. display
            # asleep), otherwise the loop would spin at full CPU.
            self._stopped.wait(self.READER_BACKOFF_SECONDS)
