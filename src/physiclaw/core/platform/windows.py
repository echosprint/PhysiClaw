"""Windows implementations of platform-specific helpers.

Imported by ``physiclaw.core.platform`` on win32 only. Callers should
never import this module directly — go through ``physiclaw.core.platform``.
"""

import os
import socket
import subprocess
import time

# Windows registry's ProxyOverride is unreliable for loopback bypass —
# corporate ProxyOverride often lacks `<local>` / `localhost`, and an
# `HTTP_PROXY` env var bypasses no-proxy rules entirely. Tell urllib /
# httpx to skip system proxy when calling our own server.
TRUST_PROXY_ENV = False


def ensure_camera_permission() -> None:
    """No-op on Windows — MediaFoundation surfaces the camera-access prompt
    itself when the device is opened."""


def local_hostname() -> str | None:
    """Return the short hostname suitable for ``<name>.local`` mDNS, or None.

    Windows doesn't have a separate Bonjour ``LocalHostName`` concept; the
    Bonjour-for-Windows service publishes ``socket.gethostname()`` as-is.
    """
    try:
        return socket.gethostname().split(".")[0] or None
    except Exception:
        return None


def open_camera_aim_app() -> None:
    """Open the built-in Camera app so the user can position the phone."""
    # `start` is a cmd builtin (not an exe); the empty "" is the window
    # title that `start` consumes when the first arg is quoted.
    subprocess.run(
        ["cmd", "/c", "start", "", "microsoft.windows.camera:"],
        capture_output=True,
    )


def quit_camera_aim_app() -> None:
    """Close the Camera app so MediaFoundation releases the camera.

    Without the close + 0.5s settle, the next ``Camera(...)`` open can hit
    a still-held device handle.
    """
    subprocess.run(
        ["taskkill", "/F", "/IM", "WindowsCamera.exe"],
        capture_output=True,
    )
    time.sleep(0.5)


def open_image_files(paths: list[str]) -> None:
    """Open one or more image files in the user's default viewer."""
    for p in paths:
        try:
            os.startfile(p)  # type: ignore[attr-defined]
        except OSError:
            pass


# ─── camera exposure ────────────────────────────────────────
#
# cv2 is imported lazily inside each function: the doctor CLI imports this
# package on its cv2-import-failure path, so a top-level import would mask
# the very error it reports.

# MSMF exposes AE toggling via IAMCameraControl; DSHOW can't re-enable AE
# once off (opencv#17019).
CAMERA_EXPOSURE_TUNABLE = True


def camera_backend() -> int:
    """Capture backend for cv2.VideoCapture: explicit MSMF.

    Modern OpenCV defaults to MSMF on Windows already — the explicit flag
    documents intent and guards against DSHOW-default builds, where
    auto-exposure could never be re-enabled programmatically."""
    import cv2

    return cv2.CAP_MSMF


def camera_set_auto_exposure(cap) -> None:
    """Ask the driver for firmware auto-exposure.

    MSMF maps CAP_PROP_AUTO_EXPOSURE nonzero → VideoProcAmp_Flags_Auto,
    0 → Manual (cap_msmf.cpp). set()/get() are unreliable across drivers —
    verification is measured frame brightness, never the return value."""
    import cv2

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)


def camera_set_manual_exposure(cap, exposure: int) -> None:
    """Hold a fixed exposure. Value is log2 seconds per the DirectShow
    CameraControl_Exposure spec (-6 ≈ 1/64s; indoors -4..-8) — though
    drivers may deviate, which the caller's measured-brightness stall
    check catches. Setting CAP_PROP_EXPOSURE already carries the Manual
    flag on MSMF; the explicit AE-off keeps intent obvious."""
    import cv2

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)


# ─── doctor diagnostics ─────────────────────────────────────


def camera_denied_hint() -> str:
    """Guidance when a camera opens but yields no frame, or none are detected."""
    return (
        "camera access may be off — Settings → Privacy & security → Camera, "
        "and allow desktop apps to access the camera"
    )


def camera_aim_hint() -> str | None:
    """No extra instruction needed — ``open_camera_aim_app`` launches the
    built-in Camera app on Windows."""
    return None


def opencv_import_hint(_exc: ImportError) -> str | None:
    """The Windows OpenCV wheel is self-contained — no system-lib remediation."""
    return None


def hardware_permission_hints() -> list[str]:
    """Windows surfaces device-access prompts itself — no group advice."""
    return []
