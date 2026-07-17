"""Tests for `physiclaw.cli.flash` — FluidNC firmware flashing command.

The firmware download is mocked with an in-memory zip and esptool runs are
intercepted at `subprocess.run`, so no network or serial hardware is touched.
"""

from __future__ import annotations

import importlib
import io
import subprocess
import zipfile
from unittest.mock import MagicMock
from urllib.error import URLError

import typer
from typer.testing import CliRunner

flash_mod = importlib.import_module("physiclaw.cli.flash")

app = typer.Typer()
app.command()(flash_mod.flash)
runner = CliRunner()


def _bundle_bytes(*, skip: str | None = None, prefix: str = "") -> bytes:
    """A flat (or wrapper-folder) firmware bundle zip, built in memory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        names = [name for _offset, name in flash_mod.FLASH_LAYOUT]
        names.append("fluidnc_version.txt")
        for name in names:
            if name == skip:
                continue
            # Arbitrary version — the CLI just echoes whatever the bundle says.
            data = b"v9.9.9\n" if name == "fluidnc_version.txt" else b"\x01"
            zf.writestr(prefix + name, data)
    return buf.getvalue()


def _mock_download(mocker, data: bytes | None = None) -> None:
    """Route `http_get`/`stream` (as imported in flash.py) to `data`,
    defaulting to a complete bundle."""
    if data is None:
        data = _bundle_bytes()
    mocker.patch.object(flash_mod, "http_get", return_value=MagicMock())
    mocker.patch.object(
        flash_mod,
        "stream",
        side_effect=lambda resp, write, label: write(data),
    )


# ---------- dry run ----------


def test_flash_dry_run_lists_images_without_flashing(mocker) -> None:
    _mock_download(mocker)
    run_spy = mocker.patch.object(flash_mod.subprocess, "run")

    result = runner.invoke(app, ["--port", "/dev/cu.usbserial-X", "--dry-run"])

    assert result.exit_code == 0
    assert "Dry run" in result.output
    assert "firmware.bin" in result.output
    assert "littlefs.bin" in result.output
    run_spy.assert_not_called()


def test_flash_dry_run_without_board_uses_autodetect_placeholder(mocker) -> None:
    _mock_download(mocker)
    mocker.patch("physiclaw.core.hardware.grbl.candidate_ports", return_value=[])

    result = runner.invoke(app, ["--dry-run"])

    assert result.exit_code == 0
    assert "Board: auto-detect" in result.output


# ---------- port detection ----------


def test_flash_no_port_found_exits_with_checklist(mocker) -> None:
    _mock_download(mocker)
    mocker.patch("physiclaw.core.hardware.grbl.candidate_ports", return_value=[])

    result = runner.invoke(app, [])

    assert result.exit_code == 1
    assert "No board found" in result.output
    assert "--port" in result.output


def test_flash_auto_detect_single_port_is_used_silently(mocker) -> None:
    _mock_download(mocker)
    mocker.patch(
        "physiclaw.core.hardware.grbl.candidate_ports",
        return_value=["/dev/cu.usbserial-A"],
    )

    result = runner.invoke(app, ["--dry-run"])

    assert result.exit_code == 0
    assert "Board: /dev/cu.usbserial-A" in result.output
    assert "several ports found" not in result.output


def test_flash_auto_detect_uses_first_of_several_ports(mocker) -> None:
    _mock_download(mocker)
    mocker.patch(
        "physiclaw.core.hardware.grbl.candidate_ports",
        return_value=["/dev/cu.usbserial-A", "/dev/cu.usbserial-B"],
    )

    result = runner.invoke(app, ["--dry-run"])

    assert result.exit_code == 0
    assert "several ports found" in result.output
    assert "Board: /dev/cu.usbserial-A" in result.output


# ---------- bundle fetch ----------


def test_flash_download_failure_exits_nonzero(mocker) -> None:
    mocker.patch.object(flash_mod, "http_get", side_effect=URLError("boom"))

    result = runner.invoke(app, ["--port", "/dev/cu.usbserial-X", "--dry-run"])

    assert result.exit_code == 1
    assert "Download failed" in result.output
    assert flash_mod.FIRMWARE_URL in result.output


def test_flash_incomplete_bundle_exits_nonzero(mocker) -> None:
    _mock_download(mocker, _bundle_bytes(skip="littlefs.bin"))

    result = runner.invoke(app, ["--port", "/dev/cu.usbserial-X", "--dry-run"])

    assert result.exit_code == 1
    assert "littlefs.bin missing from the firmware bundle" in result.output


def test_flash_bundle_with_wrapper_folder_is_tolerated(mocker) -> None:
    _mock_download(mocker, _bundle_bytes(prefix="wrapper-folder/"))

    result = runner.invoke(app, ["--port", "/dev/cu.usbserial-X", "--dry-run"])

    assert result.exit_code == 0
    assert "firmware.bin" in result.output
    assert "Firmware: FluidNC v9.9.9" in result.output


def test_flash_reports_bundle_firmware_version(mocker) -> None:
    _mock_download(mocker)

    result = runner.invoke(app, ["--port", "/dev/cu.usbserial-X", "--dry-run"])

    assert result.exit_code == 0
    assert "Firmware: FluidNC v9.9.9" in result.output


def test_flash_bundle_without_version_file_stays_quiet(mocker) -> None:
    _mock_download(mocker, _bundle_bytes(skip="fluidnc_version.txt"))

    result = runner.invoke(app, ["--port", "/dev/cu.usbserial-X", "--dry-run"])

    assert result.exit_code == 0
    assert "Firmware: FluidNC" not in result.output


# ---------- flashing ----------


def test_flash_writes_all_images_in_layout_order(mocker) -> None:
    _mock_download(mocker)
    run_spy = mocker.patch.object(flash_mod.subprocess, "run")

    result = runner.invoke(app, ["--port", "/dev/cu.usbserial-X"])

    assert result.exit_code == 0
    assert "Done" in result.output
    assert run_spy.call_count == 1
    cmd = run_spy.call_args.args[0]
    assert "write_flash" in cmd
    assert "erase_flash" not in cmd
    positions = [cmd.index(offset) for offset, _name in flash_mod.FLASH_LAYOUT]
    assert positions == sorted(positions)


def test_flash_erase_runs_before_write(mocker) -> None:
    _mock_download(mocker)
    run_spy = mocker.patch.object(flash_mod.subprocess, "run")

    result = runner.invoke(app, ["--port", "/dev/cu.usbserial-X", "--erase"])

    assert result.exit_code == 0
    assert run_spy.call_count == 2
    first, second = (c.args[0] for c in run_spy.call_args_list)
    assert "erase_flash" in first
    assert "write_flash" not in first
    assert "write_flash" in second


def test_flash_esptool_failure_exits_with_power_hint(mocker) -> None:
    _mock_download(mocker)
    mocker.patch.object(
        flash_mod.subprocess,
        "run",
        side_effect=subprocess.CalledProcessError(2, "uv"),
    )

    result = runner.invoke(app, ["--port", "/dev/cu.usbserial-X"])

    assert result.exit_code == 2
    assert "Couldn't reach the board" in result.output
    assert "12 V power adaptor" in result.output


def test_flash_missing_uv_exits_nonzero(mocker) -> None:
    _mock_download(mocker)
    mocker.patch.object(flash_mod.subprocess, "run", side_effect=FileNotFoundError())

    result = runner.invoke(app, ["--port", "/dev/cu.usbserial-X"])

    assert result.exit_code == 1
    assert "`uv` not found" in result.output
