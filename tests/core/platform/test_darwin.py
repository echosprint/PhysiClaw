"""Tests for `physiclaw.core.platform.darwin` — macOS-specific helpers.

Tests run on every platform: subprocess and socket calls are mocked, so
we exercise the dispatch logic regardless of where the suite is run.
"""
from __future__ import annotations

import socket
import subprocess
from unittest.mock import MagicMock

from physiclaw.core.platform import darwin


# ---------- TRUST_PROXY_ENV ----------


def test_trust_proxy_env_is_true_on_darwin() -> None:
    # macOS proxy bypass list reliably excludes localhost, so urllib /
    # httpx can trust env-derived proxy settings on loopback.
    assert darwin.TRUST_PROXY_ENV is True


# ---------- ensure_camera_permission ----------


def test_ensure_camera_permission_calls_imagesnap(mocker) -> None:
    spy = mocker.patch.object(darwin.subprocess, "run")

    darwin.ensure_camera_permission()

    spy.assert_called_once()
    args = spy.call_args.args[0]
    assert args[0] == "imagesnap"


def test_ensure_camera_permission_swallows_missing_imagesnap(mocker) -> None:
    mocker.patch.object(darwin.subprocess, "run", side_effect=FileNotFoundError)

    darwin.ensure_camera_permission()  # must not raise


def test_ensure_camera_permission_swallows_timeout(mocker) -> None:
    mocker.patch.object(
        darwin.subprocess, "run",
        side_effect=subprocess.TimeoutExpired(cmd="imagesnap", timeout=5),
    )

    darwin.ensure_camera_permission()  # must not raise


# ---------- local_hostname ----------


def test_local_hostname_returns_scutil_value_when_present(mocker) -> None:
    fake = MagicMock(returncode=0, stdout="My-Mac\n")
    mocker.patch.object(darwin.subprocess, "run", return_value=fake)

    assert darwin.local_hostname() == "My-Mac"


def test_local_hostname_falls_back_to_socket_when_scutil_empty(mocker) -> None:
    fake = MagicMock(returncode=0, stdout="\n")
    mocker.patch.object(darwin.subprocess, "run", return_value=fake)
    mocker.patch.object(socket, "gethostname", return_value="fallback-name")

    assert darwin.local_hostname() == "fallback-name"


def test_local_hostname_falls_back_when_scutil_missing(mocker) -> None:
    mocker.patch.object(darwin.subprocess, "run", side_effect=FileNotFoundError)
    mocker.patch.object(socket, "gethostname", return_value="other.example")

    # Strips DNS suffix.
    assert darwin.local_hostname() == "other"


def test_local_hostname_falls_back_when_scutil_times_out(mocker) -> None:
    mocker.patch.object(
        darwin.subprocess, "run",
        side_effect=subprocess.TimeoutExpired(cmd="scutil", timeout=1),
    )
    mocker.patch.object(socket, "gethostname", return_value="host")

    assert darwin.local_hostname() == "host"


def test_local_hostname_returns_none_when_socket_raises(mocker) -> None:
    fake = MagicMock(returncode=1, stdout="")
    mocker.patch.object(darwin.subprocess, "run", return_value=fake)
    mocker.patch.object(socket, "gethostname", side_effect=OSError)

    assert darwin.local_hostname() is None


def test_local_hostname_returns_none_when_socket_returns_empty(mocker) -> None:
    fake = MagicMock(returncode=1, stdout="")
    mocker.patch.object(darwin.subprocess, "run", return_value=fake)
    mocker.patch.object(socket, "gethostname", return_value="")

    assert darwin.local_hostname() is None


# ---------- open_camera_aim_app / quit_camera_aim_app ----------


def test_open_camera_aim_app_runs_open_photo_booth(mocker) -> None:
    spy = mocker.patch.object(darwin.subprocess, "run")

    darwin.open_camera_aim_app()

    spy.assert_called_once()
    assert spy.call_args.args[0] == ["open", "-a", "Photo Booth"]


def test_quit_camera_aim_app_runs_osascript_and_settles(mocker) -> None:
    run_spy = mocker.patch.object(darwin.subprocess, "run")
    sleep_spy = mocker.patch.object(darwin.time, "sleep")

    darwin.quit_camera_aim_app()

    run_spy.assert_called_once()
    args = run_spy.call_args.args[0]
    assert args[0] == "osascript"
    assert "Photo Booth" in args[-1]
    sleep_spy.assert_called_once_with(0.5)


# ---------- open_image_files ----------


def test_open_image_files_runs_open_with_paths(mocker) -> None:
    spy = mocker.patch.object(darwin.subprocess, "run")

    darwin.open_image_files(["/tmp/a.jpg", "/tmp/b.jpg"])

    spy.assert_called_once()
    assert spy.call_args.args[0] == ["open", "/tmp/a.jpg", "/tmp/b.jpg"]


def test_open_image_files_noop_on_empty_list(mocker) -> None:
    spy = mocker.patch.object(darwin.subprocess, "run")

    darwin.open_image_files([])

    spy.assert_not_called()


# ---------- camera exposure ----------


def test_camera_exposure_not_tunable_on_macos() -> None:
    # AVFoundation ignores exposure props through OpenCV entirely.
    assert darwin.CAMERA_EXPOSURE_TUNABLE is False


def test_camera_backend_lets_opencv_pick() -> None:
    import cv2

    assert darwin.camera_backend() == cv2.CAP_ANY


def test_exposure_setters_are_noops() -> None:
    cap = MagicMock()

    darwin.camera_set_auto_exposure(cap)
    darwin.camera_set_manual_exposure(cap, -6)

    cap.set.assert_not_called()
