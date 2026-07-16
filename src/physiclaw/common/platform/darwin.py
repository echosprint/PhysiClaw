"""macOS implementations of platform-specific helpers.

Imported by ``physiclaw.common.platform`` on Darwin only. Callers should
never import this module directly — go through ``physiclaw.common.platform``.
"""

import logging
import socket
import subprocess
import time

log = logging.getLogger(__name__)

# urllib's ProxyHandler / httpx's `trust_env` consult the system proxy
# config for loopback HTTP. macOS exposes its bypass list (which usually
# includes 127.0.0.1, localhost, *.local) via getproxies_macosx_sysconf,
# and urllib/httpx honor it — so trusting env on darwin is safe.
TRUST_PROXY_ENV = True


def ensure_camera_permission() -> None:
    """Trigger the macOS camera permission dialog via ``imagesnap``.

    OpenCV's AVFoundation backend won't surface the prompt itself, so the
    first ``cv2.VideoCapture.read()`` silently returns blank frames until
    the user grants access. ``imagesnap`` forces TCC to prompt. No-ops if
    imagesnap isn't installed or hangs.
    """
    try:
        subprocess.run(
            ["imagesnap", "-w", "0", "/dev/null"],
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def local_hostname() -> str | None:
    """Return the short hostname suitable for ``<name>.local`` mDNS, or None.

    Prefers ``scutil --get LocalHostName`` (the user-editable Bonjour name);
    falls back to ``socket.gethostname()`` stripped of any DNS suffix.
    """
    try:
        result = subprocess.run(
            ["scutil", "--get", "LocalHostName"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        if result.returncode == 0:
            name = result.stdout.strip()
            if name:
                return name
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        return socket.gethostname().split(".")[0] or None
    except Exception:
        return None


def open_camera_aim_app() -> None:
    """Open Photo Booth so the user can position the phone under the camera."""
    subprocess.run(["open", "-a", "Photo Booth"])


def quit_camera_aim_app() -> None:
    """Quit Photo Booth so AVFoundation releases the camera.

    Without the quit + 0.5s settle, the next ``Camera(...)`` open hits a
    still-exclusive AVCaptureSession and surfaces as "Camera not connected"
    downstream. Graceful AppleScript quit (not ``killall``) so macOS tears
    the session down cleanly.
    """
    subprocess.run(
        ["osascript", "-e", 'tell application "Photo Booth" to quit'],
        capture_output=True,
    )
    time.sleep(0.5)


def open_image_files(paths: list[str]) -> None:
    """Open one or more image files in the user's default viewer (Preview)."""
    if not paths:
        return
    subprocess.run(["open", *paths])


# ─── camera exposure ────────────────────────────────────────
#
# AVFoundation exposes no exposure properties through OpenCV — set() is
# ignored / returns False on macOS — and Apple's manual-exposure API
# doesn't apply to external UVC webcams. But the camera itself honors
# standard UVC control requests over USB, a channel Apple's driver
# doesn't block: the sibling `uvc` module sends them through IOKit in
# pure ctypes (no compiler, no privileges, works mid-stream). Everything
# is probed, never assumed: no UVC device, an ambiguous multi-device
# bus, or a camera that won't answer all degrade to exactly the old
# behavior (firmware AE untouched, tuning skipped).
#
# Why this matters: some cameras' firmware AE overexposes in high-res
# modes (stretching shutter past the frame interval — brighter frames
# AND a lower fps than the mode advertises). The UVC path is the only
# macOS lever that lets exposure.py's `converge` correct that.

# Probe result cache: None = not probed yet, False = unavailable this
# process, else the live CameraTerminal. One verdict per process — a
# camera hot-plugged mid-session picks it up on the next restart.
_uvc_terminal = None


def _camera_terminal():
    global _uvc_terminal
    if _uvc_terminal is None:
        from physiclaw.common.platform import uvc

        _uvc_terminal = uvc.camera_terminal() or False
        if _uvc_terminal:
            log.info("UVC camera-terminal control live (IOKit, in-process)")
    return _uvc_terminal or None


def camera_exposure_tunable() -> bool:
    """Whether UVC exposure control is live (probed once per process).

    True only when exactly one UVC device is on the bus and it answers
    for its exposure controls — see `uvc.camera_terminal` for why the
    single-device rule exists. (Built-in FaceTime/Continuity cameras are
    not UVC and don't count.)"""
    return _camera_terminal() is not None


def camera_size_cap() -> tuple[int, int] | None:
    """Largest safe capture request, or None for no cap.

    Without a live UVC channel, high-resolution modes are dangerous on
    macOS: some cameras' firmware AE stretches shutter there — frames
    ~2× too bright at a fraction of the advertised fps — and with
    AVFoundation exposing no exposure API there is nothing to correct
    it with. The same cameras' ≤1080p modes meter sanely, and 1080p is
    the pre-4K default the vision quality thresholds were calibrated
    on. With a live channel there is no cap: `exposure.converge` can
    pull any mode back into band."""
    return None if camera_exposure_tunable() else (1920, 1080)


def camera_backend() -> int:
    """Capture backend for cv2.VideoCapture: let OpenCV pick (AVFoundation)."""
    import cv2

    return cv2.CAP_ANY


def camera_set_auto_exposure(cap) -> None:
    """Hand exposure back to the camera firmware, via UVC (`cap` unused —
    the control channel is USB, not the capture session; it takes effect
    mid-stream). No-op when the probe found no usable UVC channel —
    firmware AE was already in charge."""
    channel = _camera_terminal()
    if channel is None:
        return
    if not channel.set_auto():
        log.warning("UVC: could not set auto-exposure mode")


def camera_set_manual_exposure(cap, exposure: int) -> None:
    """Hold a fixed exposure via UVC (`cap` unused — see above).

    `exposure` arrives on the shared log2-seconds scale; the tick
    conversion lives in `uvc.ticks_from_log2_seconds` (shared with
    linux.py). The channel clamps to the device's published range; a
    refused write is logged and `converge`'s luma-stall check reverts
    to auto, so it degrades, never wedges."""
    from physiclaw.common.platform import uvc

    channel = _camera_terminal()
    if channel is None:
        return
    ticks = uvc.ticks_from_log2_seconds(exposure)
    if not channel.set_manual(ticks):
        log.warning("UVC: could not set exposure-time-abs=%d", ticks)


def camera_focus_lockable() -> bool:
    """Whether the lens can be frozen on this rig: needs the live UVC
    channel (AVFoundation exposes no focus API, same story as exposure)
    AND a camera terminal that answers for its focus controls —
    fixed-focus cameras don't, and need no lock."""
    channel = _camera_terminal()
    return channel is not None and channel.answers_focus()


def camera_lock_focus(cap) -> bool:
    """Freeze autofocus at its current position via UVC (`cap` unused —
    the control channel is USB, it takes effect mid-stream). Returns
    whether the freeze was accepted; the caller verifies by measured
    sharpness, never by this return alone."""
    channel = _camera_terminal()
    return channel is not None and channel.lock_focus()


def camera_unlock_focus(cap) -> None:
    """Hand the lens back to continuous autofocus via UVC."""
    channel = _camera_terminal()
    if channel is None:
        return
    if not channel.unlock_focus():
        log.warning("UVC: could not re-enable autofocus")


# ─── doctor diagnostics ─────────────────────────────────────


def camera_denied_hint() -> str:
    """Guidance when a camera opens but yields no frame, or none are detected."""
    return (
        "likely denied Camera permission — System Settings → "
        "Privacy & Security → Camera"
    )


def camera_aim_hint() -> str | None:
    """No extra instruction needed — ``open_camera_aim_app`` reliably launches
    Photo Booth on macOS."""
    return None


def opencv_import_hint(_exc: ImportError) -> str | None:
    """The macOS OpenCV wheel is self-contained — no system-lib remediation."""
    return None


def hardware_permission_hints() -> list[str]:
    """macOS gates camera access via TCC prompts, not Unix groups — nothing to
    advise here (``ensure_camera_permission`` triggers the prompt)."""
    return []
