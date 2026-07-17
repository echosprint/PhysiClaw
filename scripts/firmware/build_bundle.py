#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["esptool>=4,<5", "littlefs-python>=0.12"]
# ///
"""Build the ``physiclaw flash`` firmware bundle from a FluidNC release.

Produces the zip that ``physiclaw flash`` downloads (published at
https://physiclaw.ai/downloads/firmware/ and on the ``firmware_fluidNC``
GitHub release) from upstream FluidNC release assets plus
``scripts/firmware/config.yaml``.

    uv run scripts/firmware/build_bundle.py                # DEFAULT_TAG
    uv run scripts/firmware/build_bundle.py --tag v4.1.0   # a new release

Provenance of each file in the bundle:

    bootloader.bin   copied from fluidnc-<tag>-posix.zip  wifi/bootloader.bin
    partitions.bin   copied from fluidnc-<tag>-posix.zip  wifi/partitions.bin
    boot_app0.bin    copied from fluidnc-<tag>-posix.zip  common/boot_app0.bin
    firmware.bin     esptool elf2image --flash_mode dio --flash_freq 80m
                     --flash_size 4MB  of the esp32-noradio-firmware.elf
                     release asset (the posix zip has no noradio build)
    littlefs.bin     LittleFS image of the data partition, containing only
                     /config.yaml (littlefs-python, default disk version)
    config.yaml      scripts/firmware/config.yaml (also zipped for reference)
    fluidnc_version.txt  the FluidNC release tag, one line — every name in
                     the chain is version-free, this records the version

The littlefs offset/size are read from the release's partition table and
checked against ``LITTLEFS_OFFSET`` — if a FluidNC release moves the
partition, this script fails instead of producing a bundle the flash
command would write to the wrong address. The constant mirrors
``FLASH_LAYOUT`` in ``physiclaw.cli.flash`` (this script stays standalone
and can't import it); ``tests/test_firmware_consistency.py`` keeps the
pair aligned.

The zip is written deterministically (fixed entry order and timestamps,
no platform extra fields), so the same inputs give the same bytes on any
machine; SHA-256s are printed for the release notes. FluidNC assets are
cached in ~/.cache/physiclaw-firmware/<tag>/.

Publishing the built bundle:

    gh release upload firmware_fluidNC dist/firmware/fluidnc.zip \\
        --clobber -R physiclaw/PhysiClaw

then redeploy PhysiClaw-site — its ``scripts/fetch-downloads.mjs`` copies
the asset to https://physiclaw.ai/downloads/firmware/fluidnc.zip, which
``physiclaw flash`` downloads. Asset, path, and CLI are all version-free;
a new FluidNC version is just build → upload → redeploy.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import struct
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_TAG = "v4.0.3"
BUNDLE_NAME = "fluidnc.zip"  # version-free; fluidnc_version.txt inside records it
CACHE_DIR = Path.home() / ".cache" / "physiclaw-firmware"
CONFIG_YAML = Path(__file__).with_name("config.yaml")

# Mirrors FLASH_LAYOUT in physiclaw.cli.flash (littlefs.bin offset) —
# pinned to it by tests/test_firmware_consistency.py.
LITTLEFS_OFFSET = 0x3D0000
LFS_BLOCK_SIZE = 4096

ORDER = [
    "bootloader.bin",
    "partitions.bin",
    "boot_app0.bin",
    "firmware.bin",
    "littlefs.bin",
    "config.yaml",
    "fluidnc_version.txt",
]
# Informational entries — everything else in ORDER is flashed by the CLI.
REFERENCE_FILES = {"config.yaml", "fluidnc_version.txt"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, dest: Path) -> None:
    """Download with resume + retry (GitHub is flaky from some networks)."""
    done = dest.with_name(dest.name + ".done")
    if done.exists():  # complete from a previous run — skip the round trip
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 9):
        have = dest.stat().st_size if dest.exists() else 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status == 200:  # server ignored Range
                    have = 0
                with open(dest, "ab" if have else "wb") as f:
                    while chunk := r.read(1 << 16):
                        f.write(chunk)
            done.touch()
            return
        except Exception as e:  # noqa: BLE001 — any transport error: resume
            if getattr(e, "code", None) == 416:  # Range past EOF: complete
                done.touch()
                return
            print(f"    retry {attempt}/8: {e}", file=sys.stderr)
            time.sleep(2)
    sys.exit(f"download failed: {url}")


def elf2image(elf: Path, out: Path) -> None:
    import esptool  # type: ignore[import-not-found]  # PEP 723 script dep

    args = [
        "--chip", "esp32",
        "elf2image",
        "--flash_mode", "dio",
        "--flash_freq", "80m",
        "--flash_size", "4MB",
        "-o", str(out),
        str(elf),
    ]  # fmt: skip
    try:
        esptool.main(args)
    except SystemExit as e:
        if e.code:
            raise


def data_partition(partitions: bytes) -> tuple[int, int]:
    """(offset, size) of the spiffs/littlefs data partition in the table."""
    for i in range(0, len(partitions) - 32, 32):
        entry = partitions[i : i + 32]
        if entry[:2] != b"\xaaP":  # end of table
            break
        ptype, subtype = entry[2], entry[3]
        offset, size = struct.unpack_from("<II", entry, 4)
        if ptype == 1 and subtype == 0x82:  # data / spiffs
            return offset, size
    sys.exit("no spiffs/littlefs data partition found in partitions.bin")


def build_littlefs(config: bytes, block_count: int, out: Path) -> None:
    from littlefs import LittleFS  # type: ignore[import-not-found]  # PEP 723 dep

    fs = LittleFS(block_size=LFS_BLOCK_SIZE, block_count=block_count)
    with fs.open("/config.yaml", "wb") as f:
        f.write(config)
    fs.unmount()
    out.write_bytes(bytes(fs.context.buffer))


def write_bundle(files: dict[str, bytes], out: Path) -> None:
    """Deterministic zip: fixed order and timestamps, no extra fields."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w") as zf:
        for name in ORDER:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, files[name])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default=DEFAULT_TAG, help="FluidNC release tag")
    ap.add_argument("--out", type=Path, help="output zip (default dist/firmware/)")
    ap.add_argument("--posix-zip", type=Path, help="local fluidnc posix zip")
    ap.add_argument("--elf", type=Path, help="local esp32-noradio-firmware.elf")
    args = ap.parse_args()

    tag = args.tag
    out = args.out or Path("dist/firmware") / BUNDLE_NAME
    dl = f"https://github.com/bdring/FluidNC/releases/download/{tag}"

    posix_zip = args.posix_zip or CACHE_DIR / tag / f"fluidnc-{tag}-posix.zip"
    elf = args.elf or CACHE_DIR / tag / "esp32-noradio-firmware.elf"
    if not args.posix_zip:
        print(f"→ FluidNC posix zip … ({posix_zip})")
        fetch(f"{dl}/fluidnc-{tag}-posix.zip", posix_zip)
    if not args.elf:
        print(f"→ FluidNC noradio elf … ({elf})")
        fetch(f"{dl}/esp32-noradio-firmware.elf", elf)

    with tempfile.TemporaryDirectory(prefix="physiclaw-fw-build-") as td:
        work = Path(td)
        prefix = f"fluidnc-{tag}-posix"
        with zipfile.ZipFile(posix_zip) as zf:
            for member, name in [
                (f"{prefix}/wifi/bootloader.bin", "bootloader.bin"),
                (f"{prefix}/wifi/partitions.bin", "partitions.bin"),
                (f"{prefix}/common/boot_app0.bin", "boot_app0.bin"),
            ]:
                (work / name).write_bytes(zf.read(member))

        offset, size = data_partition((work / "partitions.bin").read_bytes())
        if offset != LITTLEFS_OFFSET:
            sys.exit(
                f"data partition moved to {offset:#x} in {tag} — "
                f"physiclaw.cli.flash writes littlefs.bin at "
                f"{LITTLEFS_OFFSET:#x}; update FLASH_LAYOUT first"
            )

        print("→ firmware.bin (esptool elf2image) …")
        elf2image(elf, work / "firmware.bin")
        print(f"→ littlefs.bin ({size // 1024}K config.yaml filesystem) …")
        build_littlefs(
            CONFIG_YAML.read_bytes(), size // LFS_BLOCK_SIZE, work / "littlefs.bin"
        )
        shutil.copyfile(CONFIG_YAML, work / "config.yaml")
        (work / "fluidnc_version.txt").write_text(f"{tag}\n")

        files = {name: (work / name).read_bytes() for name in ORDER}
        for name in ORDER:
            print(f"    {sha256(files[name])}  {name}")
        write_bundle(files, out)

    print(f"    {sha256(out.read_bytes())}  {BUNDLE_NAME} ({tag})")
    print(f"\n✓ bundle written → {out}")


if __name__ == "__main__":
    main()
