"""Linux implementations of platform-specific helpers.

Imported by ``physiclaw.core.platform`` on linux only. Callers should
never import this module directly — go through ``physiclaw.core.platform``.

The browser setup wizard (``/setup-hardware``) drives Linux entirely over
HTTP and needs none of the GUI helpers below — only ``ensure_camera_permission``
(a no-op here) is on its hot path. ``open_camera_aim_app`` / ``quit_camera_aim_app``
/ ``open_image_files`` are conveniences for the terminal ``physiclaw setup
hardware`` flow and degrade gracefully when no desktop app is available.
"""

import glob
import grp
import os
import shutil
import socket
import subprocess
import time

# urllib/httpx read proxy config from env vars on Linux and honor `no_proxy`
# for loopback, so trusting env is safe when calling our own 127.0.0.1 server.
TRUST_PROXY_ENV = True

# Webcam viewers tried in preference order. GNOME Snapshot superseded Cheese
# as the default in Ubuntu 24.04 / Fedora Workstation; guvcview is the common
# non-GNOME fallback. Used only by the CLI aim step — best-effort.
_AIM_APPS = ("snapshot", "cheese", "guvcview")


def ensure_camera_permission() -> None:
    """No-op on Linux.

    V4L2 has no interactive permission prompt — access to ``/dev/video*`` is
    governed by ``video`` group membership. ``doctor`` detects a non-member
    and advises ``sudo usermod -aG video $USER``.
    """


def local_hostname() -> str | None:
    """Return the short hostname suitable for ``<name>.local`` mDNS, or None.

    Linux has no separate Bonjour ``LocalHostName`` concept; Avahi publishes
    ``socket.gethostname()`` as-is.
    """
    try:
        return socket.gethostname().split(".")[0] or None
    except Exception:
        return None


def open_camera_aim_app() -> None:
    """Launch a webcam viewer so the user can position the phone.

    Best-effort: tries Snapshot → Cheese → guvcview and launches the first
    one installed. No-ops if none is present — the CLI step prints a manual
    instruction and still waits for the user.
    """
    for app in _AIM_APPS:
        if shutil.which(app):
            subprocess.Popen(
                [app], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return


def quit_camera_aim_app() -> None:
    """Close the webcam viewer so V4L2 releases the device, then settle.

    pkills every candidate app name (only one was launched) and waits 0.5s
    so the device handle is free before the next ``Camera(...)`` open.
    """
    for app in _AIM_APPS:
        subprocess.run(["pkill", "-x", app], capture_output=True)
    time.sleep(0.5)


def open_image_files(paths: list[str]) -> None:
    """Open one or more image files in the user's default viewer via xdg-open."""
    for p in paths:
        try:
            subprocess.run(["xdg-open", p], capture_output=True)
        except FileNotFoundError:
            pass


# ─── camera exposure ────────────────────────────────────────
#
# cv2 is imported lazily inside each function: the doctor CLI imports this
# package on its cv2-import-failure path, so a top-level import would mask
# the very error it reports.

CAMERA_EXPOSURE_TUNABLE = True


def camera_backend() -> int:
    """Capture backend for cv2.VideoCapture: explicit V4L2.

    The exposure setters below encode raw V4L2 menu values — pinning the
    backend makes that invariant real instead of hoping OpenCV doesn't
    pick GStreamer (symmetric with windows.py pinning MSMF)."""
    import cv2

    return cv2.CAP_V4L2


def camera_set_auto_exposure(cap) -> None:
    """Ask the driver for auto-exposure.

    Modern OpenCV 4.x V4L2 passes raw menu values: 3 =
    V4L2_EXPOSURE_APERTURE_PRIORITY (the usual UVC "auto"), 1 = manual.
    (Older builds normalized to 0.75/0.25 — not our regime, we require
    opencv>=4.8.) Verification is measured frame brightness, never the
    set() return value."""
    import cv2

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)


def camera_set_manual_exposure(cap, exposure: int) -> None:
    """Hold a fixed exposure. `exposure` arrives on the shared
    log2-seconds scale (-11 ≈ 0.5ms … -2 = 250ms, exposure.py's bounds);
    V4L2's EXPOSURE control wants ~100µs ticks, so convert — one log2
    stop is one halving of ticks, keeping `converge`'s integer stepping
    a real one-stop move on Linux. (Passing the raw negative through —
    the old behavior — got clamped by the driver, which the luma stall
    check read as "driver ignores exposure writes", reverting every
    tune to blown-out auto.) Tick ranges vary per device; out-of-range
    values are the driver's to clamp."""
    import cv2

    ticks = max(1, round((2.0 ** exposure) * 10_000))
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, ticks)


# ─── doctor diagnostics ─────────────────────────────────────


def camera_denied_hint() -> str:
    """Guidance when a camera opens but yields no frame, or none are detected."""
    return (
        "no access to /dev/video* — add yourself to the 'video' group: "
        "sudo usermod -aG video $USER (then log out and back in)"
    )


def camera_aim_hint() -> str | None:
    """The aim-app launch is best-effort on Linux; tell the user to open their
    own viewer if none popped up."""
    return "If no camera app opened, open one (e.g. Snapshot or Cheese) to aim."


def opencv_import_hint(exc: ImportError) -> str | None:
    """Actionable remediation when ``import cv2`` fails, else None.

    The manylinux OpenCV wheel can't load without libGL/glib on a minimal
    Linux; turn that cryptic ImportError into an apt/dnf line. (PhysiClaw uses
    no cv2 GUI — these are pure load-time libs.)
    """
    if "libGL" not in str(exc):
        return None
    return (
        "\n    Install the system libs:\n"
        "      sudo apt install libgl1 libglib2.0-0   # Debian/Ubuntu\n"
        "      sudo dnf install mesa-libGL glib2        # Fedora/RHEL"
    )


def _in_group(name: str) -> bool:
    """True if the current user belongs to group ``name``. False if the group
    doesn't exist on this distro."""
    try:
        gid = grp.getgrnam(name).gr_gid
    except KeyError:
        return False
    return gid in os.getgroups()


def hardware_permission_hints() -> list[str]:
    """Warn when device nodes exist but the user lacks the group that grants
    access — the common Linux "permission denied" cause for camera/serial."""
    hints: list[str] = []
    if glob.glob("/dev/video*") and not _in_group("video"):
        hints.append(
            "camera: you're not in the 'video' group — /dev/video* access is "
            "denied.\n    sudo usermod -aG video $USER   (then log out and back in)"
        )
    serial_nodes = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    # dialout on Debian/Ubuntu, uucp on Arch — flag whichever exists.
    serial_group = next(
        (g for g in ("dialout", "uucp") if _group_exists(g)), None
    )
    if serial_nodes and serial_group and not _in_group(serial_group):
        hints.append(
            f"serial: you're not in the '{serial_group}' group — arm access is "
            f"denied.\n    sudo usermod -aG {serial_group} $USER   "
            "(then log out and back in)"
        )
    return hints


def _group_exists(name: str) -> bool:
    try:
        grp.getgrnam(name)
        return True
    except KeyError:
        return False
