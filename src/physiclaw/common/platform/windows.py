"""Windows implementations of platform-specific helpers.

Imported by ``physiclaw.common.platform`` on win32 only. Callers should
never import this module directly — go through ``physiclaw.common.platform``.
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


def camera_exposure_tunable() -> bool:
    """MSMF exposes AE toggling via IAMCameraControl — always tunable.
    (DSHOW couldn't re-enable AE once off, opencv#17019 — we pin MSMF.)"""
    return True


def camera_size_cap() -> tuple[int, int] | None:
    """No cap — exposure is always tunable here, so any capture mode a
    bad firmware AE picks can be corrected by `exposure.converge`."""
    return None


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


def camera_focus_lockable() -> bool:
    """MSMF maps CAP_PROP_AUTOFOCUS onto the CameraControl_Focus flags —
    the lock is always worth ATTEMPTING here. Whether this camera honors
    it is judged at lock time: the set() return, then (as always)
    measured frame sharpness."""
    return True


def camera_lock_focus(cap) -> bool:
    """Freeze autofocus at its current position: read the converged
    position, then re-pin it through `camera_apply_focus` — the AF-off
    ordering lives there. No readable position (see
    `camera_read_focus`) → AF off alone, rather than risk racking an
    unsupported lens to its stop; disabling AF holds the lens on most
    cameras, and the caller's measured-sharpness verify covers the
    rest."""
    import cv2

    cur = camera_read_focus(cap)
    if cur is None:
        return bool(cap.set(cv2.CAP_PROP_AUTOFOCUS, 0))
    return camera_apply_focus(cap, cur)


def camera_unlock_focus(cap) -> None:
    """Hand the lens back to continuous autofocus."""
    import cv2

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)


def camera_read_focus(cap) -> float | None:
    """Current absolute focus position, or None when the driver reports
    none. get() returns 0.0 when CAP_PROP_FOCUS is unsupported and 0
    is also a legal focus value — <= 0 reads as "no usable position"."""
    import cv2

    cur = cap.get(cv2.CAP_PROP_FOCUS)
    return cur if cur > 0 else None


def camera_apply_focus(cap, value: float) -> bool:
    """Drive the lens to a caller-supplied absolute position. AF off
    first, then the absolute write — MSMF maps both onto
    CameraControl_Focus."""
    import cv2

    if not cap.set(cv2.CAP_PROP_AUTOFOCUS, 0):
        return False
    return bool(cap.set(cv2.CAP_PROP_FOCUS, value))


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
