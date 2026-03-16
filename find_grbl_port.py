"""Auto-detect GRBL devices on serial ports."""

import sys
import time

import serial
import serial.tools.list_ports


GRBL_BAUDRATE = 115200

# Skip ports that are never GRBL devices
_SKIP_KEYWORDS = {"bluetooth", "bt-", "debug", "wlan", "wifi", "airpods"}

# GRBL boards typically use CH340/CP210x/FTDI USB-serial chips
_LIKELY_KEYWORDS = {"ch340", "cp210", "ftdi", "usbserial", "usbmodem", "wch", "serial"}


def _probe_port(port: str, baudrate: int = GRBL_BAUDRATE) -> str | None:
    """Try to connect to a port and identify a GRBL device.

    Uses $I query (no reset side-effect, faster than soft-reset).
    Returns the version string if GRBL is detected, None otherwise.
    """
    try:
        with serial.Serial(port, baudrate, timeout=2) as ser:
            time.sleep(2)
            ser.reset_input_buffer()
            ser.write(b'$I\r\n')
            time.sleep(0.5)
            resp = ser.read(ser.in_waiting or 256).decode('utf-8', errors='ignore')
            if 'grbl' in resp.lower() or '[VER:' in resp:
                # Extract version line
                for line in resp.splitlines():
                    if 'grbl' in line.lower() or '[VER:' in line:
                        return line.strip()
                return 'GRBL'
    except (serial.SerialException, OSError):
        pass
    return None


def _port_priority(port_info) -> int:
    """Lower value = probe first. Likely USB-serial chips first, unlikely ports last."""
    name = (port_info.device + " " + (port_info.description or "")).lower()
    if any(kw in name for kw in _SKIP_KEYWORDS):
        return 99  # skip these entirely
    if any(kw in name for kw in _LIKELY_KEYWORDS):
        return 0   # most likely GRBL
    return 50      # unknown, probe after likely ones


def find_grbl_port() -> str | None:
    """Scan serial ports and return the first GRBL device port, or None.

    Skips Bluetooth/debug ports and probes likely USB-serial ports first.
    """
    all_ports = serial.tools.list_ports.comports()
    ports = sorted(
        [p for p in all_ports if _port_priority(p) < 99],
        key=lambda p: (_port_priority(p), p.device),
    )
    skipped = len(all_ports) - len(ports)

    if not ports:
        print("No candidate serial ports found.")
        return None

    msg = f"Scanning {len(ports)} serial port(s) for GRBL"
    if skipped:
        msg += f" (skipped {skipped} Bluetooth/debug)"
    print(msg + "...\n")

    for port_info in ports:
        desc = port_info.description or ""
        print(f"  Probing {port_info.device}  ({desc})  ...", end=" ", flush=True)

        version = _probe_port(port_info.device)
        if version:
            print(f"GRBL detected!  [{version}]")
            return port_info.device
        else:
            print("not GRBL")

    return None


def main() -> None:
    port = find_grbl_port()
    print()
    if port:
        print(f"GRBL port: {port}")
    else:
        print("No GRBL device found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
