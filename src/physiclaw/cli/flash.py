"""``physiclaw flash`` — flash FluidNC firmware onto the MKS DLC32 V2.1 (ESP32).

Downloads a prebuilt firmware bundle and writes it with esptool at the fixed
offsets FluidNC uses. Nothing to choose — the board, baud, offsets, and the
firmware itself are all decided by the bundle, to match the firmware guide:
https://docs.physiclaw.ai/en/set-up/firmware/

The bundle (``FIRMWARE_URL``) is a flat zip of ready-to-flash images:

    bootloader.bin   -> 0x1000
    partitions.bin   -> 0x8000
    boot_app0.bin    -> 0xe000
    firmware.bin     -> 0x10000    (FluidNC noradio app)
    littlefs.bin     -> 0x3d0000   (filesystem image holding config.yaml)

To rebuild the bundle from a FluidNC release: take ``common/boot_app0.bin`` and
``wifi/{bootloader,partitions}.bin`` from ``fluidnc-<ver>-posix.zip`` (the
bootloader + partition table are radio-independent), convert the noradio app
with ``esptool elf2image esp32-noradio-firmware.elf -o firmware.bin``, build a
192K LittleFS image of the spiffs partition (offset 0x3d0000) containing the
PhysiClaw config.yaml, and zip them flat (config.yaml is added for reference).

esptool runs in an ephemeral ``uv`` environment, so it isn't a permanent
dependency of the CLI.

NOTE — disconnect the stepper motors before flashing: the board can back-feed
odd voltages during the bootloader reset. Standard MKS DLC32 boards flash
fine; a board with secure-boot / flash-encryption eFuses burned would brick,
so this command does not touch those.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated, Optional
from urllib.error import URLError

import typer

from physiclaw.cli._format import ok
from physiclaw.cli._http import http_get, stream

FIRMWARE_URL = "https://physiclaw.ai/downloads/firmware/fluidnc_4_0_3.zip"
CHIP = "esp32"  # MKS DLC32 V2.1
BAUD = 115200  # matches the firmware guide's install speed

# (esptool offset, filename inside the bundle).
FLASH_LAYOUT = [
    ("0x1000", "bootloader.bin"),
    ("0x8000", "partitions.bin"),
    ("0xe000", "boot_app0.bin"),
    ("0x10000", "firmware.bin"),
    ("0x3d0000", "littlefs.bin"),  # filesystem — carries config.yaml
]

# Run esptool without making it a permanent dependency. Catch its errors in
# the child so a FatalError (e.g. "No serial data received") exits cleanly with
# a one-line reason instead of dumping a Python traceback to the terminal —
# the parent then prints the friendly checklist. SystemExit (esptool's own
# clean exits) is a BaseException, so it passes straight through.
_ESPTOOL_RUN = (
    "import esptool, sys\n"
    "try:\n"
    "    esptool.main(sys.argv[1:])\n"
    "except Exception as e:\n"
    "    print(f'esptool: {e}', file=sys.stderr)\n"
    "    sys.exit(1)\n"
)


def _fetch_bundle(into: Path) -> list[tuple[str, Path]]:
    """Download + unzip the firmware bundle. Returns [(offset, path), ...]."""
    buf = io.BytesIO()
    try:
        with http_get(FIRMWARE_URL) as r:
            stream(r, buf.write, "  firmware")
    except URLError as e:
        typer.secho(f"Download failed: {e}\n  {FIRMWARE_URL}", fg="red")
        raise typer.Exit(1)

    with zipfile.ZipFile(buf) as zf:
        zf.extractall(into)

    files = []
    for offset, name in FLASH_LAYOUT:
        p = into / name
        if not p.exists():  # tolerate a single wrapper folder inside the zip
            p = next(into.rglob(name), None)
        if p is None:
            typer.secho(f"{name} missing from the firmware bundle.", fg="red")
            raise typer.Exit(1)
        files.append((offset, p))
    return files


def _detect_port() -> Optional[str]:
    """Best-guess USB-serial port. Doesn't probe (a blank board won't reply)."""
    from physiclaw.core.hardware.grbl import candidate_ports

    ports = candidate_ports()
    if not ports:
        return None
    if len(ports) > 1:
        typer.echo(f"  (several ports found — using {ports[0]}; override with --port)")
    return ports[0]


def _uv_esptool(args: list[str]) -> None:
    cmd = [
        "uv",
        "run",
        "--no-project",
        "--with",
        "esptool>=4,<5",
        "python",
        "-c",
        _ESPTOOL_RUN,
        *args,
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        typer.secho("`uv` not found on PATH — esptool runs inside uv.", fg="red")
        raise typer.Exit(1)
    except subprocess.CalledProcessError as e:
        typer.secho(
            "\n✗ Couldn't reach the board. The 12 V power adaptor MUST be switched on.",
            fg="red",
        )
        raise typer.Exit(e.returncode or 1)


def flash(
    port: Annotated[
        Optional[str],
        typer.Option("--port", help="Serial port (default: auto-detect)."),
    ] = None,
    erase: Annotated[
        bool,
        typer.Option(
            "--erase", help="Erase the chip before writing (fixes boot loops)."
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the esptool command without flashing."),
    ] = False,
) -> None:
    """Flash FluidNC firmware onto the MKS DLC32 control board."""
    typer.secho(
        "Flash FluidNC firmware to the MKS DLC32 V2.1 control board\n", bold=True
    )

    with tempfile.TemporaryDirectory(prefix="physiclaw-fw-") as td:
        typer.echo("→ Downloading firmware …")
        files = _fetch_bundle(Path(td))

        dev = port or _detect_port() or ("auto-detect" if dry_run else None)
        if not dev:
            typer.secho(
                "\n✗ No board found. Switch on the 12 V power adaptor and connect "
                "the MKS DLC32 over USB — install the CH340 driver if needed, or pass --port.",
                fg="red",
            )
            raise typer.Exit(1)
        typer.echo(f"→ Board: {dev}")

        if dry_run:
            typer.secho("\nDry run — would flash these images:", bold=True)
            for offset, p in files:
                typer.echo(f"    {offset:>9}  {p.name}")
            return

        base = ["--chip", CHIP, "--port", dev, "--baud", str(BAUD)]
        if erase:
            typer.echo("→ Erasing flash …")
            _uv_esptool([*base, "erase_flash"])

        write = [
            *base,
            "--before",
            "default_reset",
            "--after",
            "hard_reset",
            "write_flash",
            "-z",
            "--flash_mode",
            "dio",
            "--flash_freq",
            "80m",
            "--flash_size",
            "detect",
        ]
        for offset, p in files:
            write += [offset, str(p)]

        typer.echo("→ Flashing … takes about a minute; don't unplug the board.\n")
        _uv_esptool(write)

    typer.echo("")
    typer.echo(ok("Done — the board now runs as a PhysiClaw machine."))
    typer.echo("  FluidNC firmware and the config.yml are both on the board.\n")
