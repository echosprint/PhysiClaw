"""Tests for `physiclaw.core.hardware.grbl` — serial-port GRBL detection.

`serial.tools.list_ports.comports()` and `serial.Serial` are mocked
so tests don't touch real hardware.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock


from physiclaw.core.hardware import grbl
from physiclaw.core.hardware.grbl import (
    GRBL_BAUDRATE,
    candidate_ports,
    detect_grbl,
)


def _port_info(
    device: str, description: str = "", vid: int | None = 0x9999
) -> SimpleNamespace:
    # vid defaults to a generic USB vid (NOT a known serial-bridge VID), so
    # priority is driven by the name keyword unless a test sets vid explicitly.
    # Use vid=None for a built-in/virtual UART (e.g. /dev/ttyS0), or 0x1A86
    # (CH340) to model a real GRBL board's bridge chip by VID alone.
    return SimpleNamespace(device=device, description=description, vid=vid)


# ---------- constants ----------


def test_grbl_baudrate_is_115200() -> None:
    assert GRBL_BAUDRATE == 115200


# ---------- candidate_ports ----------


def test_candidate_ports_filters_out_skip_keywords(mocker) -> None:
    fake_ports = [
        _port_info("/dev/cu.Bluetooth-Incoming", "Bluetooth"),
        _port_info("/dev/cu.usbserial-CH340", "CH340 USB Serial"),
        _port_info("/dev/cu.AirPods-WirelessiAP", "AirPods"),
    ]
    mocker.patch.object(
        grbl.serial.tools.list_ports, "comports", return_value=fake_ports
    )

    out = candidate_ports()

    assert "/dev/cu.usbserial-CH340" in out
    assert "/dev/cu.Bluetooth-Incoming" not in out
    assert "/dev/cu.AirPods-WirelessiAP" not in out


def test_candidate_ports_prioritizes_known_usb_serial_vid(mocker) -> None:
    # A CH340 (VID 0x1A86) is "likely" even when its name carries no keyword —
    # names vary by OS/driver, the VID is the same everywhere.
    fake_ports = [
        _port_info("/dev/cu.someUSBthing", "", vid=0x9999),  # generic USB device
        _port_info("/dev/cu.blankname", "", vid=0x1A86),  # CH340, by VID only
    ]
    mocker.patch.object(
        grbl.serial.tools.list_ports, "comports", return_value=fake_ports
    )

    assert candidate_ports()[0] == "/dev/cu.blankname"


def test_candidate_ports_excludes_builtin_uart_without_vid(mocker) -> None:
    # Linux's built-in /dev/ttyS0 (no USB vid) is never the GRBL board.
    fake_ports = [
        _port_info("/dev/ttyS0", "", vid=None),
        _port_info("/dev/ttyUSB0", "USB Serial"),
    ]
    mocker.patch.object(
        grbl.serial.tools.list_ports, "comports", return_value=fake_ports
    )

    assert candidate_ports() == ["/dev/ttyUSB0"]


def test_candidate_ports_empty_when_only_builtin_uart(mocker) -> None:
    # Board not connected → only /dev/ttyS0 present → report nothing, so flash
    # says "No board found" instead of auto-picking the phantom port.
    fake_ports = [_port_info("/dev/ttyS0", "", vid=None)]
    mocker.patch.object(
        grbl.serial.tools.list_ports, "comports", return_value=fake_ports
    )

    assert candidate_ports() == []


def test_candidate_ports_sorts_likely_first(mocker) -> None:
    fake_ports = [
        _port_info("/dev/cu.unknown1", "Unknown"),
        _port_info("/dev/cu.usbserial-CP210", "CP2102 USB UART"),
        _port_info("/dev/cu.unknown2", "Random"),
    ]
    mocker.patch.object(
        grbl.serial.tools.list_ports, "comports", return_value=fake_ports
    )

    out = candidate_ports()

    # CP210 (likely) sorts before unknowns.
    assert out[0] == "/dev/cu.usbserial-CP210"


def test_candidate_ports_returns_empty_when_no_serial_devices(mocker) -> None:
    mocker.patch.object(grbl.serial.tools.list_ports, "comports", return_value=[])

    assert candidate_ports() == []


# ---------- detect_grbl ----------


def test_detect_grbl_returns_none_when_no_candidates(mocker) -> None:
    mocker.patch.object(grbl.serial.tools.list_ports, "comports", return_value=[])

    assert detect_grbl() is None


def test_detect_grbl_returns_port_on_successful_probe(mocker) -> None:
    fake_ports = [_port_info("/dev/cu.usbserial-X", "CH340")]
    mocker.patch.object(
        grbl.serial.tools.list_ports, "comports", return_value=fake_ports
    )

    fake_serial = MagicMock()
    fake_serial.in_waiting = 64
    fake_serial.read.return_value = b"Grbl 1.1h ['$' for help]\r\n"
    fake_serial.__enter__ = MagicMock(return_value=fake_serial)
    fake_serial.__exit__ = MagicMock(return_value=None)
    mocker.patch.object(grbl.serial, "Serial", return_value=fake_serial)
    mocker.patch.object(grbl.time, "sleep")

    out = detect_grbl()

    assert out == "/dev/cu.usbserial-X"


def test_detect_grbl_returns_port_when_response_contains_ver_marker(mocker) -> None:
    fake_ports = [_port_info("/dev/cu.usbserial-Y", "FTDI USB Serial")]
    mocker.patch.object(
        grbl.serial.tools.list_ports, "comports", return_value=fake_ports
    )

    fake_serial = MagicMock()
    fake_serial.in_waiting = 64
    fake_serial.read.return_value = b"[VER:1.1h.20190825:]\r\n[OPT:V,15,128]\r\n"
    fake_serial.__enter__ = MagicMock(return_value=fake_serial)
    fake_serial.__exit__ = MagicMock(return_value=None)
    mocker.patch.object(grbl.serial, "Serial", return_value=fake_serial)
    mocker.patch.object(grbl.time, "sleep")

    out = detect_grbl()

    assert out == "/dev/cu.usbserial-Y"


def test_detect_grbl_skips_port_when_response_is_not_grbl(mocker) -> None:
    fake_ports = [
        _port_info("/dev/cu.fake", "FTDI"),
        _port_info("/dev/cu.real", "CH340"),
    ]
    mocker.patch.object(
        grbl.serial.tools.list_ports, "comports", return_value=fake_ports
    )

    # First port returns junk; second returns Grbl banner.
    fake_first = MagicMock()
    fake_first.in_waiting = 8
    fake_first.read.return_value = b"random output"
    fake_first.__enter__ = MagicMock(return_value=fake_first)
    fake_first.__exit__ = MagicMock(return_value=None)

    fake_second = MagicMock()
    fake_second.in_waiting = 32
    fake_second.read.return_value = b"Grbl 1.1h\r\n"
    fake_second.__enter__ = MagicMock(return_value=fake_second)
    fake_second.__exit__ = MagicMock(return_value=None)

    mocker.patch.object(grbl.serial, "Serial", side_effect=[fake_first, fake_second])
    mocker.patch.object(grbl.time, "sleep")

    out = detect_grbl()

    assert out == "/dev/cu.real"


def test_detect_grbl_continues_after_serial_exception_on_a_port(mocker) -> None:
    import serial as pyserial

    fake_ports = [
        _port_info("/dev/cu.dead", "FTDI"),
        _port_info("/dev/cu.live", "CH340"),
    ]
    mocker.patch.object(
        grbl.serial.tools.list_ports, "comports", return_value=fake_ports
    )

    fake_live = MagicMock()
    fake_live.in_waiting = 32
    fake_live.read.return_value = b"Grbl 1.1h\r\n"
    fake_live.__enter__ = MagicMock(return_value=fake_live)
    fake_live.__exit__ = MagicMock(return_value=None)

    mocker.patch.object(
        grbl.serial,
        "Serial",
        side_effect=[pyserial.SerialException("port busy"), fake_live],
    )
    mocker.patch.object(grbl.time, "sleep")

    out = detect_grbl()

    assert out == "/dev/cu.live"


def test_detect_grbl_returns_none_when_all_probes_fail(mocker) -> None:
    fake_ports = [_port_info("/dev/cu.x", "CH340")]
    mocker.patch.object(
        grbl.serial.tools.list_ports, "comports", return_value=fake_ports
    )

    fake_serial = MagicMock()
    fake_serial.in_waiting = 8
    fake_serial.read.return_value = b"random response"
    fake_serial.__enter__ = MagicMock(return_value=fake_serial)
    fake_serial.__exit__ = MagicMock(return_value=None)
    mocker.patch.object(grbl.serial, "Serial", return_value=fake_serial)
    mocker.patch.object(grbl.time, "sleep")

    assert detect_grbl() is None
