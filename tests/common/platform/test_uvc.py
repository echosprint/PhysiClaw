"""Tests for `physiclaw.common.platform.uvc` — IOKit UVC exposure control.

The IOKit layer is mocked throughout: these tests pin the fail-soft
policy of `exposure_channel` (when it must return None) and the UVC
request framing of `ExposureChannel`, not the ctypes ABI itself — that
is exercised on real hardware by the darwin integration checks.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from physiclaw.common.platform import uvc

# ---------- exposure_channel fail-soft policy ----------


def test_channel_none_when_iokit_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(uvc, "_load", lambda: False)

    assert uvc.exposure_channel() is None


def test_channel_none_when_no_uvc_device(monkeypatch) -> None:
    monkeypatch.setattr(uvc, "_load", lambda: True)
    monkeypatch.setattr(uvc, "_video_control_interfaces", lambda: [])

    assert uvc.exposure_channel() is None


def test_channel_none_and_released_with_multiple_devices(monkeypatch) -> None:
    # OpenCV indexes cameras via AVFoundation, this module via IOKit;
    # with two devices the orders may disagree, so refuse to tune.
    released = []
    monkeypatch.setattr(uvc, "_load", lambda: True)
    monkeypatch.setattr(uvc, "_video_control_interfaces", lambda: ["intf-a", "intf-b"])
    monkeypatch.setattr(
        uvc, "_call", lambda ptr, name, *a: released.append((ptr, name))
    )

    assert uvc.exposure_channel() is None
    assert released == [("intf-a", "Release"), ("intf-b", "Release")]


def test_channel_none_when_probe_fails(monkeypatch) -> None:
    # Device enumerates but won't answer for its exposure controls (the
    # Monterey+ empty-controls failure mode) -> degrade, don't tune blind.
    released = []
    monkeypatch.setattr(uvc, "_load", lambda: True)
    monkeypatch.setattr(uvc, "_video_control_interfaces", lambda: ["intf"])
    monkeypatch.setattr(uvc, "_call", lambda ptr, name, *a: released.append(name) or 0)

    channel = MagicMock()
    channel.probe.return_value = False
    monkeypatch.setattr(uvc, "ExposureChannel", lambda intf: channel)

    assert uvc.exposure_channel() is None
    assert "Release" in released


def test_channel_returned_for_single_answering_device(monkeypatch) -> None:
    monkeypatch.setattr(uvc, "_load", lambda: True)
    monkeypatch.setattr(uvc, "_video_control_interfaces", lambda: ["intf"])
    monkeypatch.setattr(uvc, "_call", lambda *a: 0)

    channel = MagicMock()
    channel.probe.return_value = True
    monkeypatch.setattr(uvc, "ExposureChannel", lambda intf: channel)

    assert uvc.exposure_channel() is channel


# ---------- ExposureChannel request framing ----------


def _bare_channel(monkeypatch, **attrs) -> uvc.ExposureChannel:
    """ExposureChannel without the IOKit-touching __init__, with `attrs`
    set on the instance. The shared base for every framing test."""
    monkeypatch.setattr(uvc.ExposureChannel, "__init__", lambda self, intf: None)
    channel = uvc.ExposureChannel(None)
    for name, value in attrs.items():
        setattr(channel, name, value)
    return channel


def _channel_with_recorder(monkeypatch, results: dict[tuple, int]):
    """Build an ExposureChannel whose _request is replaced by a recorder.

    `results` maps (direction, request, selector) to an int the fake
    device "returns" (loaded into the data buffer little-endian); a
    missing key means the request fails.
    """
    channel = _bare_channel(monkeypatch, _range=None)
    sent = []

    def request(direction, request_code, selector, data):
        sent.append((direction, request_code, selector, bytes(data)))
        key = (direction, request_code, selector)
        if key not in results:
            return False
        data[:] = results[key].to_bytes(len(data), "little")
        return True

    channel._request = request
    return channel, sent


def test_set_manual_clamps_to_device_range(monkeypatch) -> None:
    channel, sent = _channel_with_recorder(
        monkeypatch,
        {
            (uvc._OUT, uvc._SET_CUR, uvc._CT_AE_MODE): 0,
            (uvc._OUT, uvc._SET_CUR, uvc._CT_EXPOSURE_ABS): 0,
            (uvc._IN, uvc._GET_MIN, uvc._CT_EXPOSURE_ABS): 10,
            (uvc._IN, uvc._GET_MAX, uvc._CT_EXPOSURE_ABS): 2000,
        },
    )

    assert channel.set_manual(5000) is True

    # mode=1 first, then the clamped 4-byte exposure write.
    assert sent[0][:3] == (uvc._OUT, uvc._SET_CUR, uvc._CT_AE_MODE)
    assert sent[0][3] == (1).to_bytes(1, "little")
    assert sent[-1][:3] == (uvc._OUT, uvc._SET_CUR, uvc._CT_EXPOSURE_ABS)
    assert sent[-1][3] == (2000).to_bytes(4, "little")


def test_set_manual_passes_ticks_through_without_published_range(
    monkeypatch,
) -> None:
    channel, sent = _channel_with_recorder(
        monkeypatch,
        {
            (uvc._OUT, uvc._SET_CUR, uvc._CT_AE_MODE): 0,
            (uvc._OUT, uvc._SET_CUR, uvc._CT_EXPOSURE_ABS): 0,
        },
    )

    assert channel.set_manual(156) is True
    assert sent[-1][3] == (156).to_bytes(4, "little")


def test_set_manual_fails_when_mode_write_refused(monkeypatch) -> None:
    channel, sent = _channel_with_recorder(monkeypatch, {})

    assert channel.set_manual(156) is False
    # Never writes an exposure value on top of an unknown AE mode.
    assert all(s[2] != uvc._CT_EXPOSURE_ABS for s in sent)


def test_set_auto_falls_back_to_full_auto_mode(monkeypatch) -> None:
    # Mode 8 (aperture priority) refused -> retry with mode 2.
    channel, sent = _channel_with_recorder(monkeypatch, {})

    assert channel.set_auto() is False
    assert [s[3] for s in sent] == [
        (8).to_bytes(1, "little"),
        (2).to_bytes(1, "little"),
    ]


# ---------- camera-terminal discovery + transfer-length guard ----------


def _channel_with_device(monkeypatch, terminals: dict[int, dict[int, int]]):
    """ExposureChannel over a fake bus: `terminals` maps entity id ->
    {selector: control byte size}. GET requests to a listed control
    "transfer" min(its real size, requested length) bytes — mimicking a
    same-numbered selector of the wrong size — and anything else STALLs.
    """
    channel = _bare_channel(monkeypatch, _terminal=1)

    def request(direction, request_code, selector, data):
        controls = terminals.get(channel._terminal, {})
        if direction != uvc._IN or selector not in controls:
            return False
        done = min(controls[selector], len(data))
        # Mirrors the real _request: partial transfer = failure.
        return done == len(data)

    channel._request = request
    return channel


def test_probe_settles_on_conventional_terminal_1(monkeypatch) -> None:
    channel = _channel_with_device(
        monkeypatch, {1: {uvc._CT_AE_MODE: 1, uvc._CT_EXPOSURE_ABS: 4}}
    )

    assert channel.probe() is True
    assert channel._terminal == 1


def test_probe_discovers_nonstandard_terminal_id(monkeypatch) -> None:
    channel = _channel_with_device(
        monkeypatch, {3: {uvc._CT_AE_MODE: 1, uvc._CT_EXPOSURE_ABS: 4}}
    )

    assert channel.probe() is True
    assert channel._terminal == 3


def test_probe_rejects_processing_unit_by_control_size(monkeypatch) -> None:
    # A PU's selectors 0x02/0x04 are brightness/contrast — 2 bytes each.
    # The size mismatch must reject it, not silently mis-drive it.
    channel = _channel_with_device(
        monkeypatch, {2: {uvc._CT_AE_MODE: 2, uvc._CT_EXPOSURE_ABS: 2}}
    )

    assert channel.probe() is False


def test_probe_false_when_no_terminal_answers(monkeypatch) -> None:
    channel = _channel_with_device(monkeypatch, {})

    assert channel.probe() is False


@pytest.mark.parametrize(
    "done, expected",
    [(2, None), (4, 0)],
    ids=["short-transfer-rejected", "full-transfer-accepted"],
)
def test_request_requires_full_transfer(monkeypatch, done, expected) -> None:
    # kr == 0 alone is not success: wLenDone must equal wLength.
    channel = _bare_channel(monkeypatch, _intf="intf", _terminal=1, _ifnum=0)

    def fake_call(ptr, name, pipe, req_ref):
        req_ref._obj.wLenDone = done
        return 0

    monkeypatch.setattr(uvc, "_call", fake_call)

    assert channel._get(uvc._GET_CUR, uvc._CT_EXPOSURE_ABS, 4) == expected
