"""
Camera module — reusable Camera class and CLI test utilities.

Usage as library:
    from physiclaw.core.hardware.camera import Camera
    cam = Camera(index=0)
    frame = cam.snapshot()
    cam.close()

Usage as CLI:
    uv run python -m physiclaw.camera              # scan all cameras
    uv run python -m physiclaw.camera --index 0    # live preview (q=quit, s=save)
    uv run python -m physiclaw.camera --snap 0     # save one frame

Note: On macOS, OpenCV won't trigger the camera permission dialog.
If the camera returns blank frames, run `imagesnap` once first to
grant camera access to your terminal app, then retry.
"""

import _thread
import contextlib
import logging
import os
import sys
import threading
import time
from collections.abc import Iterator

import cv2
import numpy as np

from physiclaw.common import platform
from physiclaw.common.config import CONFIG
from physiclaw.common.dumps import save_raw_camera, save_snapshot

log = logging.getLogger(__name__)


@contextlib.contextmanager
def silenced_stderr() -> Iterator[None]:
    """Redirect OS-level stderr (fd 2) to /dev/null for the block.

    OpenCV/AVFoundation/ffmpeg print things like ``out device of bound``
    and ``camera failed to properly initialize`` via C-level fprintf,
    bypassing Python's logging — only an fd-level redirect catches them.
    Wrap any code that probes or opens a ``cv2.VideoCapture`` whose index
    might not exist.

    Process-global: fd 2 is shared, so any other thread logging to
    stderr inside the ``with`` block drops to /dev/null too. Keep the
    wrapped region short.
    """
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        saved = os.dup(2)
        try:
            os.dup2(devnull, 2)
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)
    finally:
        os.close(devnull)


def apply_size_cap(size: tuple[int, int]) -> tuple[int, int]:
    """Clamp a capture request to the platform's safe maximum, if any.

    On macOS without a UVC exposure channel, high-res modes risk a
    firmware AE that overexposes with no way to correct it, so requests
    drop to the largest mode whose AE behaves. Called both where
    `Camera._request_size` is seeded (so remembered state and warmup
    logs stay honest) and inside `configure_capture` (the choke point,
    so the fallback ladder and reconnects can't sneak past it)."""
    size_cap = platform.camera_size_cap()
    if size_cap is None or size[0] * size[1] <= size_cap[0] * size_cap[1]:
        return size
    log.info(
        "capture request %dx%d capped to %dx%d — exposure is not "
        "tunable on this rig and higher modes risk uncorrectable "
        "overexposure",
        *size,
        *size_cap,
    )
    return size_cap


def configure_capture(
    cap: cv2.VideoCapture,
    *,
    exposure_auto: bool,
    exposure: int,
    size: tuple[int, int] | None = None,
) -> None:
    """Apply PhysiClaw's capture properties to an open cv2.VideoCapture.

    Order is load-bearing: FOURCC must be set before width/height —
    Windows MSMF re-negotiates on format change, and YUY2 (the default)
    caps at 640×480 over USB even for 4K cameras; MJPG-compressed makes
    high resolutions actually reachable. Exposure comes AFTER the size
    negotiation, because renegotiation is exactly what can leave the
    driver's auto-exposure off/misconfigured (the Windows blown-frames
    root cause). Drivers snap to the nearest supported mode; the caller
    logs the truth after the first frame.

    ``size`` overrides the CONFIG request — `Camera._warmup` uses it to
    walk `RESOLUTION_FALLBACKS` when a negotiation misbehaves.

    The request passes through `apply_size_cap` — see its docstring
    for the macOS exposure-safety rationale.

    Shared by `Camera._open` and the doctor's deep camera probe so the
    probe meters the same negotiation the server runs.
    """
    width, height = apply_size_cap(
        size
        if size is not None
        else (
            CONFIG.camera.width,
            CONFIG.camera.height,
        )
    )
    # AVFoundation (macOS) ignores this; V4L (Linux) honors it.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CONFIG.camera.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if exposure_auto:
        platform.camera_set_auto_exposure(cap)
    else:
        platform.camera_set_manual_exposure(cap, exposure)


# Fallback requests when a high-resolution negotiation misbehaves.
# Linux V4L2 guarantees nearest-match adjustment (VIDIOC_S_FMT rounds
# down and reports back) and macOS AVFoundation snaps to the closest
# supported mode, but Windows MSMF can fail stream selection outright
# or fall back to 640×480 when asked for a mode the camera lacks
# (opencv/opencv#25243, #12822). `Camera._warmup` steps down this
# ladder until a frame arrives at ≥ BASELINE_LONG_EDGE; the last rung
# equals the pre-4K-default fixed request, so no camera negotiates
# worse than it did before the default moved.
RESOLUTION_FALLBACKS = [(2560, 1440), (1920, 1080)]
BASELINE_LONG_EDGE = 1920


# ─── Reusable Camera class ──────────────────────────────────────


class Camera:
    """Persistent camera handle for fast repeated frame grabs.

    A background daemon thread continuously calls ``cap.read()`` so the
    macOS AVFoundation pipeline never goes idle (cv2 stalls indefinitely
    on the next read after tens of seconds of inactivity — reproduced on
    the bench rig; no matching issue exists on OpenCV's tracker).
    Callers get the latest frame via ``peek()`` /
    ``snapshot()`` / ``_fresh_frame()`` without blocking on cv2.

    Holds the software rotation code applied to raw frames. Default is
    ``-1`` (no rotation) — calibration step 3 (`pick_frame_rotation`)
    writes the detected ``cv2.ROTATE_*`` code via ``cam.rotation = code``.
    Callers that need a rotated frame should always use
    ``peek()``/``snapshot()`` rather than calling ``cv2.rotate`` themselves.
    """

    # If the reader gets no frame for this long, force close+reopen of
    # cv2.VideoCapture. Recovers from real disconnects.
    STALE_RECONNECT_SECONDS = 5.0

    # If the reader gets no frame for this long, give up and raise
    # KeyboardInterrupt in the main thread (via _thread.interrupt_main —
    # cross-platform; os.kill(SIGINT) on Windows would TerminateProcess
    # and skip atexit). Reopen can't revive a stream the OS has cut
    # (display sleep, bus suspend) — better to die cleanly than spam
    # the log forever.
    FATAL_AFTER_SECONDS = 60.0

    # Max time _fresh_frame() waits for the reader to produce a frame
    # before returning whatever it last had (or None).
    FRAME_WAIT_SECONDS = 2.0

    # A frame older than this is treated as "not yet fresh" by
    # _fresh_frame() — it'll wait for the reader to publish a newer one.
    FRESH_MAX_AGE_SECONDS = 1.0

    # Backoff after a failed read or reconnect, so a permanently broken
    # camera doesn't spin-loop the reader thread.
    READER_BACKOFF_SECONDS = 0.5

    # close(): how long to wait for the reader thread, then for the cap
    # lock. A reader wedged inside a blocking native cap.read() (USB bus
    # suspend) can hold the lock indefinitely; close() must not hang
    # the caller on it.
    CLOSE_JOIN_TIMEOUT_SECONDS = 3.0
    CLOSE_LOCK_TIMEOUT_SECONDS = 2.0

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.rotation: int = -1  # no rotation until calibration step 3 sets it
        self._frame = None
        self._frame_time = 0.0
        # Monotonic count of published frames — the settle primitive for
        # exposure tuning (`wait_frames`): after a property set, "wait N
        # frames" is deterministic where "sleep N/fps" is not.
        self._frame_seq = 0
        # Separate from _frame_time because _reopen() resets _frame_time —
        # which would otherwise postpone the FATAL_AFTER_SECONDS check
        # indefinitely. _first_fail_time resets only on a real good frame.
        self._first_fail_time: float | None = None
        self._cond = threading.Condition()
        self._stopped = threading.Event()
        # cv2.VideoCapture is not thread-safe for concurrent read()+set():
        # this lock serializes the reader's cap.read() against exposure
        # setters, _reopen, and close. Non-reentrant, never nested; _open
        # does NOT take it (callers hold it, or run before the thread).
        self._cap_lock = threading.Lock()
        # Exposure state remembered on the instance so _reopen() re-applies
        # a converged manual value instead of silently reverting to auto.
        self._exposure_auto: bool = CONFIG.camera.auto_exposure
        self._exposure_value: int = CONFIG.camera.exposure
        # The resolution request, remembered so _reopen() re-applies the
        # rung _warmup settled on instead of re-failing at the top of the
        # ladder on every reconnect.
        self._request_size: tuple[int, int] = apply_size_cap(
            (CONFIG.camera.width, CONFIG.camera.height)
        )

        self._open()
        self._warmup()

        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"Camera-{index}-reader",
            daemon=True,
        )
        self._thread.start()

    # ─── cv2 lifecycle ──────────────────────────────────────────

    def _open(self) -> None:
        """Open the underlying ``cv2.VideoCapture``. Retries with macOS perm
        prompt. Caller must hold ``_cap_lock`` or run before the reader
        thread starts (``__init__``/``_warmup``)."""
        backend = platform.camera_backend()
        with silenced_stderr():
            self.cap = cv2.VideoCapture(self.index, backend)
            if not self.cap.isOpened():
                platform.ensure_camera_permission()
                self.cap = cv2.VideoCapture(self.index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.index}")
        # FOURCC → size → exposure, in that order (see configure_capture).
        # Reads the REMEMBERED exposure state, so a converged manual value
        # survives _reopen()'s reconstruction of the cap.
        configure_capture(
            self.cap,
            exposure_auto=self._exposure_auto,
            exposure=self._exposure_value,
            size=self._request_size,
        )

    def _request_ladder(self) -> list[tuple[int, int]]:
        """The configured request first, then each strictly-smaller fallback."""
        first = self._request_size
        return [first] + [
            s for s in RESOLUTION_FALLBACKS if s[0] * s[1] < first[0] * first[1]
        ]

    def _acquire_first_frame(self) -> np.ndarray | None:
        """Discard initial auto-exposure frames and read one; on failure
        release + perm-prompt + reopen once. Returns the frame or None."""
        for _ in range(2):
            for _ in range(15):
                self.cap.read()
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
            # Read returned no frame — likely macOS perm denied silently.
            self.cap.release()
            platform.ensure_camera_permission()
            self._open()
        return None

    def _warmup(self) -> None:
        """Verify reads work, stepping the resolution request down when
        the negotiation misbehaves.

        A frame smaller than BASELINE_LONG_EDGE while a larger mode was
        requested — or no frame at all — means the driver mishandled the
        over-ask (Windows MSMF; see RESOLUTION_FALLBACKS). Renegotiate
        down the ladder; the last rung accepts whatever arrives, which
        matches the behavior of the old fixed 1920×1080 request.
        """
        ladder = self._request_ladder()
        for rung, size in enumerate(ladder):
            if size != self._request_size:
                log.warning(
                    f"Camera {self.index}: negotiation misbehaved — "
                    f"retrying at {size[0]}x{size[1]}"
                )
                self._request_size = size
                configure_capture(
                    self.cap,
                    exposure_auto=self._exposure_auto,
                    exposure=self._exposure_value,
                    size=size,
                )
            frame = self._acquire_first_frame()
            if frame is None:
                continue
            h, w = frame.shape[:2]
            if max(w, h) < BASELINE_LONG_EDGE and rung < len(ladder) - 1:
                log.warning(
                    f"Camera {self.index}: {size[0]}x{size[1]} request "
                    f"negotiated only {w}x{h}"
                )
                continue
            log.info(f"Camera {self.index} ready  ({w}x{h})")
            with self._cond:
                self._frame = frame
                self._frame_time = time.monotonic()
                self._frame_seq += 1
            return
        raise RuntimeError(f"Camera {self.index}: read failed")

    def _reopen(self) -> None:
        """Close and reopen the cap. Called by the reader on stale frames.
        `_open` re-applies the remembered exposure state, so a converged
        manual exposure survives the reconnect."""
        log.warning(f"Camera {self.index}: reconnecting cv2.VideoCapture")
        with self._cap_lock:
            try:
                self.cap.release()
            except Exception:
                pass
            try:
                self._open()
            except Exception as e:
                log.error(f"Camera {self.index}: reopen failed: {e!r}")
                return
        with self._cond:
            self._frame_time = time.monotonic()  # reset stale clock

    # ─── Background reader ──────────────────────────────────────

    def _reader_loop(self) -> None:
        """Pull frames continuously so AVFoundation never goes idle.

        ``cap.read()`` blocks for the next native-FPS frame, so the loop
        self-paces — no explicit sleep needed in the steady state.
        """
        while not self._stopped.is_set():
            try:
                # Serialized against exposure setters / _reopen / close —
                # cv2.VideoCapture can't take a concurrent set()+read().
                with self._cap_lock:
                    ok, frame = self.cap.read()
            except Exception as e:
                log.warning(f"Camera {self.index}: cap.read() raised {e!r}")
                ok, frame = False, None

            now = time.monotonic()

            if ok and frame is not None:
                self._first_fail_time = None
                with self._cond:
                    self._frame = frame
                    self._frame_time = now
                    self._frame_seq += 1
                    self._cond.notify_all()
                continue

            if self._first_fail_time is None:
                self._first_fail_time = now
            fail_duration = now - self._first_fail_time
            if fail_duration >= self.FATAL_AFTER_SECONDS:
                log.error(
                    f"Camera {self.index}: no frames for {fail_duration:.0f}s "
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
            # cap.read() raises or returns empty every tick (e.g. display
            # asleep), otherwise the loop would spin at full CPU.
            self._stopped.wait(self.READER_BACKOFF_SECONDS)

    # ─── Exposure control ───────────────────────────────────────

    @property
    def exposure_tunable(self) -> bool:
        """Whether exposure can be controlled on this rig. Always true on
        Linux/Windows (V4L2/MSMF); on macOS true only when the vendored
        UVC helper compiled and unambiguously controls the camera —
        AVFoundation itself ignores exposure properties."""
        return platform.camera_exposure_tunable()

    def set_auto_exposure(self) -> None:
        """Hand exposure back to the camera firmware. Remembered, so a
        later `_reopen` keeps the choice."""
        self._exposure_auto = True
        with self._cap_lock:
            platform.camera_set_auto_exposure(self.cap)

    def set_manual_exposure(self, value: int) -> None:
        """Hold a fixed exposure (backend-native units). Remembered, so a
        later `_reopen` re-applies it. Verification is the caller's job —
        measured frame brightness, never the driver's word."""
        self._exposure_auto = False
        self._exposure_value = value
        with self._cap_lock:
            platform.camera_set_manual_exposure(self.cap, value)

    def wait_frames(self, n: int, timeout: float = 5.0) -> bool:
        """Block until the reader publishes `n` MORE frames (or timeout).

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

    # ─── Frame accessors ────────────────────────────────────────

    def _fresh_frame(self) -> np.ndarray | None:
        """Return the latest raw (unrotated) BGR frame, or ``None``.

        Waits up to ``FRAME_WAIT_SECONDS`` for the reader to publish a
        frame fresher than ``FRESH_MAX_AGE_SECONDS``; otherwise returns
        whatever the reader last had.
        """
        with self._cond:
            self._cond.wait_for(
                lambda: (
                    self._frame is not None
                    and time.monotonic() - self._frame_time < self.FRESH_MAX_AGE_SECONDS
                ),
                timeout=self.FRAME_WAIT_SECONDS,
            )
            frame = self._frame
        # Copy outside the lock — a frame copy (~1 ms at 1080p, ~8 ms at
        # 4K) would otherwise stall the reader's next publish for that long.
        if frame is None:
            return None
        out = frame.copy()
        save_raw_camera(out)  # no-op unless --save-raw-camera
        return out

    def raw_frame(self) -> np.ndarray | None:
        """Return a fresh BGR frame without applying calibration rotation.

        Used during camera identification (warm-start auto-pick) where
        rotation isn't known yet. For normal use see ``peek`` and
        ``snapshot`` which apply ``self.rotation`` before returning.
        """
        return self._fresh_frame()

    def _rotate(self, frame: np.ndarray) -> np.ndarray:
        """Apply ``self.rotation`` to a raw frame. No-op when rotation is -1."""
        if self.rotation == -1:
            return frame
        return cv2.rotate(frame, self.rotation)

    def peek(self) -> np.ndarray | None:
        """Return a fresh BGR frame with the calibrated rotation applied.

        Used for high-frequency polling (e.g. the phone-watch runtime) where
        writing a JPEG to disk every tick would be wasteful.
        """
        frame = self._fresh_frame()
        if frame is None:
            return None
        return self._rotate(frame)

    def snapshot(
        self, bbox: tuple[tuple[int, int], tuple[int, int]] | None = None
    ) -> np.ndarray | None:
        """Return a fresh BGR frame with the calibrated rotation applied.

        If ``bbox`` is provided as ``((x1,y1), (x2,y2))``, a green rectangle
        is drawn on the returned frame. When ``PHYSICLAW_SAVE_SNAPSHOTS``
        is set, every frame is also written to ``data/snapshots/``.
        """
        frame = self.peek()
        if frame is None:
            return None
        if bbox is not None:
            cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 2)
        save_snapshot(frame)
        return frame

    def close(self) -> None:
        self._stopped.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.CLOSE_JOIN_TIMEOUT_SECONDS)
        # Never release the cap without holding the lock — a native
        # release() concurrent with the reader's cap.read() can segfault
        # the whole process. If the reader is wedged holding the lock,
        # leak the handle instead of deadlocking the caller.
        if self._cap_lock.acquire(timeout=self.CLOSE_LOCK_TIMEOUT_SECONDS):
            try:
                # Hand a held manual exposure back to firmware auto on the
                # way out: the value lives in the camera's volatile RAM, so
                # it would outlive this process and poison the next run (or
                # any other app) until a re-tune or a USB power cycle.
                # Skipped when the user pinned manual in config — that pin
                # is theirs to keep. Best-effort: never blocks the close.
                if CONFIG.camera.auto_exposure and not self._exposure_auto:
                    try:
                        platform.camera_set_auto_exposure(self.cap)
                    except Exception:
                        log.warning("exposure hand-back on close failed", exc_info=True)
                self.cap.release()
            finally:
                self._cap_lock.release()
        else:
            log.warning(
                f"Camera {self.index}: close() couldn't take the cap lock "
                "(reader wedged in cap.read()?) — leaking the capture handle"
            )
