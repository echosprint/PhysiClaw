"""Jog GRBL through four phases — confirms X/Y wiring on a fresh board.

Phases run in order:

    1. X-only         X+ then X-
    2. Y-only         Y+ then Y-
    3. diagonal       (X+,Y+) then (X-,Y-)
    4. anti-diagonal  (X+,Y-) then (X-,Y+)

Each phase loops --loops times. Forward + back leaves the machine back
at origin before the next phase, so the run ends at WPos 0,0,0.

Self-contained — only depends on pyserial.

Usage:
    .venv/bin/python scripts/grbl_jog.py
    .venv/bin/python scripts/grbl_jog.py --dist 15 --loops 2
    .venv/bin/python scripts/grbl_jog.py --port /dev/tty.usbserial-XXXX
"""

import argparse
import sys
import time

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    raise SystemExit("pyserial not installed.")


# ─── Shared GRBL helpers (identical in scripts/grbl_solenoid_test.py) ─

BAUD = 115200
_SKIP = ("bluetooth", "bt-", "debug", "wlan", "wifi", "airpods")
_LIKELY = (
    "ch340",
    "cp210",
    "ftdi",
    "usbserial",
    "usbmodem",
    "wch",
    "arduino",
    "prolific",
)


def find_port() -> str | None:
    """Probe likely USB-serial ports for a GRBL board, return first hit."""
    candidates = []
    for p in serial.tools.list_ports.comports():
        name = (p.device + " " + (p.description or "")).lower()
        if any(k in name for k in _SKIP):
            continue
        priority = 0 if any(k in name for k in _LIKELY) else 1
        candidates.append((priority, p.device))
    candidates.sort()

    for _, dev in candidates:
        print(f"  probing {dev} ...", end=" ", flush=True)
        try:
            with serial.Serial(dev, BAUD, timeout=2) as ser:
                time.sleep(2)
                ser.reset_input_buffer()
                ser.write(b"$I\r\n")
                time.sleep(0.5)
                resp = ser.read(ser.in_waiting or 256).decode("utf-8", "ignore")
                if "grbl" in resp.lower() or "[VER:" in resp:
                    first = resp.strip().splitlines()[0] if resp.strip() else "ok"
                    print(f"GRBL ({first})")
                    return dev
                print("not GRBL")
        except (serial.SerialException, OSError) as e:
            print(f"open failed ({e})")
    return None


def connect(port: str | None) -> serial.Serial:
    """Resolve port (auto-detect if None), open, drain the reset banner."""
    if port is None:
        print("Scanning serial ports for a GRBL board...")
        port = find_port()
        if port is None:
            raise SystemExit("No GRBL device found. Pass --port explicitly.")
    print(f"\nOpening {port} @ {BAUD}")
    ser = serial.Serial(port, BAUD, timeout=3)
    time.sleep(2)  # wait for board reset (DTR pulse on open)
    banner = ser.read(ser.in_waiting or 256).decode("utf-8", "ignore").strip()
    if banner:
        print(f"  banner: {banner!r}")
    ser.reset_input_buffer()
    return ser


_OPTIONAL_ERRORS = frozenset(
    {
        "error:3",  # setting/command not recognized (GRBL + FluidNC)
        "error:162",  # FluidNC: setting disabled (now lives in YAML)
    }
)


def send(ser: serial.Serial, cmd: str, *, optional: bool = False) -> None:
    """Send one line, block until 'ok'. Raise on error/alarm.

    optional=True swallows error codes that mean "this firmware doesn't
    take this setting at runtime" (`error:3`, `error:162`) — the value
    lives in YAML config (FluidNC) or simply isn't supported. Only the
    error codes we've observed empirically are swallowed.
    """
    print(f"  >>> {cmd}")
    ser.write((cmd + "\r\n").encode())
    empty = 0
    while True:
        line = ser.readline().decode("utf-8", "ignore").strip()
        if not line:
            empty += 1
            if empty > 3:
                raise RuntimeError(f"no reply to: {cmd}")
            continue
        empty = 0
        print(f"  <<< {line}")
        if line == "ok":
            return
        if line.startswith("error"):
            if optional and line.replace(" ", "") in _OPTIONAL_ERRORS:
                print(f"  !!! {cmd} not supported by this firmware — skipping")
                return
            raise RuntimeError(f"GRBL error on {cmd!r}: {line}")
        if line.startswith("ALARM"):
            raise RuntimeError(f"GRBL alarm on {cmd!r}: {line} (try $X first)")


def query_status(ser: serial.Serial) -> str:
    """Send '?' and return the first '<...>' status line."""
    ser.write(b"?")
    time.sleep(0.1)
    resp = ser.read(ser.in_waiting or 64).decode("utf-8", "ignore")
    for line in resp.splitlines():
        if line.startswith("<"):
            return line
    return ""


def unlock_if_alarmed(ser: serial.Serial) -> None:
    """Print status; if Alarm, send $X to clear it."""
    status = query_status(ser)
    print(f"  status: {status}")
    if "Alarm" in status:
        print("  alarm state — sending $X")
        send(ser, "$X")


# ─── Script-specific helpers ─────────────────────────────────────────


def wait_idle(ser: serial.Serial, timeout: float = 10.0) -> None:
    """Poll '?' until status reports 'Idle', or raise."""
    time.sleep(0.05)  # let GRBL transition Idle→Run after buffering
    deadline = time.time() + timeout
    while time.time() < deadline:
        if "Idle" in query_status(ser):
            return
        time.sleep(0.1)
    raise RuntimeError(f"not idle after {timeout}s")


# Phase table: (name, (x_sign, y_sign)). Forward jog = signs × dist,
# back jog = -forward. Signs are -1 / 0 / +1.
PHASES: tuple[tuple[str, tuple[int, int]], ...] = (
    ("X-only", (+1, 0)),
    ("Y-only", (0, +1)),
    ("diagonal", (+1, +1)),
    ("anti-diagonal", (+1, -1)),
)


# ─── Main ────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Jog GRBL through X, Y, and diagonal phases."
    )
    ap.add_argument("--port", help="serial device (default: auto-detect)")
    ap.add_argument(
        "--dist", type=float, default=15.0, help="distance in mm (default: 15)"
    )
    ap.add_argument(
        "--feed", type=int, default=1000, help="feed rate mm/min (default: 1000)"
    )
    ap.add_argument(
        "--loops",
        type=int,
        default=2,
        help="forward/back cycles per phase (default: 2)",
    )
    args = ap.parse_args()

    with connect(args.port) as ser:
        send(ser, "$I")
        unlock_if_alarmed(ser)
        send(ser, "G21")  # mm
        send(ser, "G90")  # absolute
        send(ser, "G92 X0 Y0 Z0")  # zero work coords here
        send(ser, "$10=0")  # report WPos in '?'

        def jog(dx: float, dy: float) -> None:
            parts = []
            if dx:
                parts.append(f"X{dx:.3f}")
            if dy:
                parts.append(f"Y{dy:.3f}")
            move = " ".join(parts)
            print(f"\n  jog {move} at F{args.feed}")
            send(ser, f"G91 G0 {move} F{args.feed}")
            send(ser, "G90")
            wait_idle(ser, timeout=15)

        for name, (sx, sy) in PHASES:
            fx, fy = sx * args.dist, sy * args.dist
            print(f"\n### phase: {name} ###")
            for i in range(args.loops):
                print(f"\n=== loop {i + 1}/{args.loops} ({name}) ===")
                jog(+fx, +fy)
                jog(-fx, -fy)

        print(f"\nFinal status: {query_status(ser)}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
