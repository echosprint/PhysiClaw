"""Tests for `physiclaw.common.platform.darwin` — macOS-specific helpers.

Tests run on every platform: subprocess and socket calls are mocked, so
we exercise the dispatch logic regardless of where the suite is run.
"""

from __future__ import annotations

import socket
import subprocess
from unittest.mock import MagicMock

import pytest

from physiclaw.common.platform import darwin

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
        darwin.subprocess,
        "run",
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
        darwin.subprocess,
        "run",
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
#
# Exposure control on macOS rides the pure-ctypes `uvc` module (IOKit ->
# USB control requests), never OpenCV. Tests pin the probe/degrade logic
# and the shared log2-seconds -> 100µs-ticks contract; the IOKit layer
# itself is always mocked so the suite runs on every platform.


class _FakeChannel:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.auto_calls = 0
        self.manual_ticks: list[int] = []

    def set_auto(self) -> bool:
        self.auto_calls += 1
        return self.ok

    def set_manual(self, ticks: int) -> bool:
        self.manual_ticks.append(ticks)
        return self.ok


def _reset_probe(monkeypatch, channel) -> None:
    from physiclaw.common.platform import uvc

    monkeypatch.setattr(darwin, "_uvc_channel", None)
    monkeypatch.setattr(uvc, "exposure_channel", lambda: channel)


def test_exposure_not_tunable_without_uvc_channel(monkeypatch) -> None:
    # No UVC device / ambiguous bus / silent camera -> degrade to the old
    # firmware-AE behavior.
    _reset_probe(monkeypatch, None)

    assert darwin.camera_exposure_tunable() is False


@pytest.mark.parametrize(
    "channel, tunable",
    [(_FakeChannel(), True), (None, False)],
    ids=["live-channel", "failed-probe"],
)
def test_probe_verdict_is_cached_either_way(monkeypatch, channel, tunable) -> None:
    from physiclaw.common.platform import uvc

    calls = 0

    def probe():
        nonlocal calls
        calls += 1
        return channel

    monkeypatch.setattr(darwin, "_uvc_channel", None)
    monkeypatch.setattr(uvc, "exposure_channel", probe)

    assert darwin.camera_exposure_tunable() is tunable
    assert darwin.camera_exposure_tunable() is tunable
    assert calls == 1


def test_manual_exposure_converts_log2_seconds_to_uvc_ticks(monkeypatch) -> None:
    # Same contract as linux.py: 2^e seconds in 100µs ticks, so one
    # integer step of `converge` stays one real stop.
    channel = _FakeChannel()
    _reset_probe(monkeypatch, channel)

    darwin.camera_set_manual_exposure(MagicMock(), -6)  # 2^-6 s ≈ 15.6 ms

    assert channel.manual_ticks == [156]


def test_manual_exposure_never_sends_zero_ticks(monkeypatch) -> None:
    channel = _FakeChannel()
    _reset_probe(monkeypatch, channel)

    darwin.camera_set_manual_exposure(MagicMock(), -20)  # 2^-20 s ≈ 1 µs

    assert channel.manual_ticks == [1]


def test_auto_exposure_delegates_to_channel(monkeypatch) -> None:
    channel = _FakeChannel()
    _reset_probe(monkeypatch, channel)

    darwin.camera_set_auto_exposure(MagicMock())

    assert channel.auto_calls == 1


def test_exposure_setters_never_touch_the_capture_handle(monkeypatch) -> None:
    # The UVC channel is USB — AVFoundation's cap object stays untouched
    # (and when not tunable, the setters are complete no-ops).
    _reset_probe(monkeypatch, None)
    cap = MagicMock()

    darwin.camera_set_auto_exposure(cap)
    darwin.camera_set_manual_exposure(cap, -6)

    cap.set.assert_not_called()


def test_size_cap_1080p_when_exposure_not_tunable(monkeypatch) -> None:
    # No UVC channel -> high-res firmware AE can overexpose with no
    # correction lever -> stay in the sane <=1080p mode family.
    _reset_probe(monkeypatch, None)

    assert darwin.camera_size_cap() == (1920, 1080)


def test_no_size_cap_when_exposure_tunable(monkeypatch) -> None:
    _reset_probe(monkeypatch, _FakeChannel())

    assert darwin.camera_size_cap() is None


def test_camera_backend_lets_opencv_pick() -> None:
    import cv2

    assert darwin.camera_backend() == cv2.CAP_ANY
