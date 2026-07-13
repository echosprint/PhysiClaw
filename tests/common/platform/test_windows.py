"""Tests for `physiclaw.common.platform.windows` — Windows-specific helpers.

These run on every platform: subprocess, socket, and os.startfile are
mocked, so the windows module's dispatch logic is exercised regardless
of where the suite is run.
"""

from __future__ import annotations

import socket
import sys
from unittest.mock import MagicMock, call

import pytest

from physiclaw.common.platform import windows

# ---------- TRUST_PROXY_ENV ----------


def test_trust_proxy_env_is_false_on_windows() -> None:
    # Windows registry / env proxies don't reliably bypass loopback,
    # so urllib / httpx must skip system proxy on this platform.
    assert windows.TRUST_PROXY_ENV is False


# ---------- ensure_camera_permission ----------


def test_ensure_camera_permission_is_noop() -> None:
    # Returns None and never raises — MediaFoundation prompts natively.
    assert windows.ensure_camera_permission() is None


# ---------- local_hostname ----------


def test_local_hostname_returns_short_name(mocker) -> None:
    mocker.patch.object(socket, "gethostname", return_value="DESKTOP-AB12.example")

    assert windows.local_hostname() == "DESKTOP-AB12"


def test_local_hostname_returns_none_on_empty(mocker) -> None:
    mocker.patch.object(socket, "gethostname", return_value="")

    assert windows.local_hostname() is None


def test_local_hostname_returns_none_on_failure(mocker) -> None:
    mocker.patch.object(socket, "gethostname", side_effect=OSError)

    assert windows.local_hostname() is None


# ---------- open_camera_aim_app / quit_camera_aim_app ----------


def test_open_camera_aim_app_runs_camera_uri(mocker) -> None:
    spy = mocker.patch.object(windows.subprocess, "run")

    windows.open_camera_aim_app()

    spy.assert_called_once()
    args = spy.call_args.args[0]
    assert args[0] == "cmd"
    assert "microsoft.windows.camera:" in args


def test_quit_camera_aim_app_runs_taskkill_and_settles(mocker) -> None:
    run_spy = mocker.patch.object(windows.subprocess, "run")
    sleep_spy = mocker.patch.object(windows.time, "sleep")

    windows.quit_camera_aim_app()

    run_spy.assert_called_once()
    args = run_spy.call_args.args[0]
    assert args[0] == "taskkill"
    assert "WindowsCamera.exe" in args
    sleep_spy.assert_called_once_with(0.5)


# ---------- open_image_files ----------


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="os.startfile only exists on Windows; can't import it on macOS/Linux.",
)
def test_open_image_files_calls_startfile_for_each(mocker) -> None:  # pragma: no cover
    spy = mocker.patch.object(windows.os, "startfile", create=True)

    windows.open_image_files(["a.jpg", "b.jpg"])

    assert spy.call_count == 2


def test_open_image_files_swallows_oserror_when_startfile_available(mocker) -> None:
    fake_startfile = MagicMock(side_effect=OSError)
    mocker.patch.object(windows.os, "startfile", fake_startfile, create=True)

    windows.open_image_files(["missing.jpg"])  # must not raise


def test_open_image_files_noop_on_empty_list(mocker) -> None:
    spy = MagicMock()
    mocker.patch.object(windows.os, "startfile", spy, create=True)

    windows.open_image_files([])

    spy.assert_not_called()


# ---------- camera exposure ----------


def test_camera_exposure_tunable_on_windows() -> None:
    assert windows.CAMERA_EXPOSURE_TUNABLE is True


def test_camera_backend_is_msmf() -> None:
    # MSMF is the only Windows backend where AE can be re-enabled.
    import cv2

    assert windows.camera_backend() == cv2.CAP_MSMF


def test_set_auto_exposure_msmf_encoding() -> None:
    # cap_msmf.cpp: nonzero → VideoProcAmp_Flags_Auto.
    import cv2

    cap = MagicMock()
    windows.camera_set_auto_exposure(cap)

    assert cap.set.call_args_list == [call(cv2.CAP_PROP_AUTO_EXPOSURE, 1)]


def test_set_manual_exposure_msmf_encoding() -> None:
    import cv2

    cap = MagicMock()
    windows.camera_set_manual_exposure(cap, -6)

    assert cap.set.call_args_list == [
        call(cv2.CAP_PROP_AUTO_EXPOSURE, 0),
        call(cv2.CAP_PROP_EXPOSURE, -6),
    ]
