"""macOS implementations of platform-specific helpers.

Imported by ``physiclaw.core.platform`` on Darwin only. Callers should
never import this module directly — go through ``physiclaw.core.platform``.
"""

import socket
import subprocess
import time

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

# AVFoundation exposes no exposure properties through OpenCV — set() is
# ignored / returns False on macOS, so there is nothing to tune. The
# camera firmware's own auto-exposure runs untouched (and works well).
CAMERA_EXPOSURE_TUNABLE = False


def camera_backend() -> int:
    """Capture backend for cv2.VideoCapture: let OpenCV pick (AVFoundation)."""
    import cv2

    return cv2.CAP_ANY


def camera_set_auto_exposure(cap) -> None:
    """No-op — AVFoundation ignores CAP_PROP_AUTO_EXPOSURE; firmware AE
    is always in charge on macOS."""


def camera_set_manual_exposure(cap, exposure: int) -> None:
    """No-op — AVFoundation ignores CAP_PROP_EXPOSURE; manual exposure
    cannot be set through OpenCV on macOS."""


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
