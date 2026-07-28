"""``physiclaw testdrive`` — take the board for a spin after flashing.

The post-flash bring-up check: connect to the board, show the machine's
axes, then hand the user the controls — ``X20Y20`` / ``Y-10`` style
moves plus ``tap`` and ``longpress`` — so they watch the motors and
solenoid do what they typed. No pass/fail bookkeeping: the user's own
eyes are the verdict. Every command routes through ``StylusArm``, so
moves are idle-synced and the solenoid keeps its safety profile.
(``scripts/grbl_jog.py`` keeps scripted per-motor phases for deeper
diagnosis.)

Runs before calibration, so directions are the arm's own axes — the
diagram defines them; the screen mapping comes later in
``setup hardware``.
"""

from __future__ import annotations

import re
from typing import Annotated, Optional

import typer

from physiclaw.cli._format import exit_error, info, ok

# Bird's-eye view of the frame plus the legend, printed once connected —
# it defines X/Y/A/B/origin before anything moves. Motors A/B sit on the
# top beam just inside the corners. Same orientation the screen mapping
# uses later: origin top-left, X growing right, Y growing down. The help
# below asks the user to verify real moves match these directions and
# names the wiring fix if not.
AXES_DIAGRAM = """\
  Looking down at the machine:

     origin
        ●──[A]───── X → ─────[B]──┐
        │                         │
        │                         │
        Y        (phone           │
        ↓         bed)            │
        │                         │
        └─────────────────────────┘

  Motors A and B sit on the top beam — the X axis.
  The other direction is Y; the top-left corner is the origin.
"""

TRY_IT_HELP = """\
  Try the machine — type a command, q to finish:

    X20Y20     move 20 mm in X+ and 20 mm in Y+
    Y-10       move 10 mm in Y-
    tap        click the stylus
    longpress  press and hold ~1 s

  Check each move matches the diagram's directions — if not, make sure
  motor A is plugged into the board's Y1 port and motor B into X.

  Phone off the bed — keep hands clear.
"""

# Per-command cap (inclusive): no homing or limit switches, so any one
# move stays small enough that a typo can't run the carriage hard into
# the frame. Up to ±100 mm per axis is allowed.
_MAX_JOG_MM = 100

# `X20`, `Y-10`, `X1.5Y-2.5` — case and spaces already normalized away.
_MOVE_RE = re.compile(r"^(?:x(?P<x>-?\d+(?:\.\d+)?))?(?:y(?P<y>-?\d+(?:\.\d+)?))?$")


def _parse_move(cmd: str) -> tuple[float, float] | None:
    """(dx, dy) mm for a move command, or None if it isn't one."""
    m = _MOVE_RE.fullmatch(cmd)
    if not m or (m.group("x") is None and m.group("y") is None):
        return None
    return float(m.group("x") or 0), float(m.group("y") or 0)


def testdrive(
    port: Annotated[
        Optional[str],
        typer.Option("--port", help="Serial port (default: auto-detect)."),
    ] = None,
) -> None:
    """Take the board for a spin — drive the motors and solenoid yourself.

    Shows the machine's axes, then reads commands in a loop: `X20Y20` /
    `Y-10` moves the carriage, `tap` / `longpress` fires the stylus, `q`
    finishes. Run it right after `physiclaw flash` or any re-wiring, with
    the phone off the bed.
    """
    from physiclaw.common import runtime_state

    live = runtime_state.read_live()
    if live:
        exit_error(
            f"physiclaw server is running (pid {live.get('pid')}) and owns "
            "the serial port — stop it first, then re-run testdrive."
        )

    # Lazy: importing StylusArm runs the core.hardware package __init__,
    # which pulls in the camera stack (cv2) — too heavy for CLI import
    # time. Same rule as `flash` and `server`.
    from physiclaw.core.hardware.arm import StylusArm
    from physiclaw.core.hardware.device import DeviceNotFound, HardwareError

    typer.secho("Test-drive the control board: motors + solenoid\n", bold=True)
    typer.echo("→ Looking for the control board …")
    try:
        arm = StylusArm(port)
    except DeviceNotFound as e:
        exit_error(str(e))

    try:
        arm.setup()
        typer.echo(ok(f"Board answered on {arm.port} — the firmware is alive.\n"))

        typer.echo(AXES_DIAGRAM)
        typer.echo(TRY_IT_HELP)

        while True:
            raw = typer.prompt("testdrive", prompt_suffix="> ")
            cmd = "".join(raw.split()).lower()
            if cmd in ("q", "quit", "exit"):
                break
            if cmd == "tap":
                arm.tap()
                continue
            if cmd == "longpress":
                arm.long_press()
                continue
            move = _parse_move(cmd)
            if move is None:
                typer.echo(
                    info("? try X10, Y-10, X5Y5, tap, longpress — or q to finish")
                )
                continue
            dx, dy = move
            if abs(dx) > _MAX_JOG_MM or abs(dy) > _MAX_JOG_MM:
                typer.echo(info(f"keep each move within ±{_MAX_JOG_MM} mm per axis"))
                continue
            arm.jog(dx, dy)
    except HardwareError as e:
        # Board answered at connect but died mid-run (cable yanked, alarm,
        # serial silence) — a plain exit, not a traceback.
        exit_error(f"the board stopped responding mid-run: {e}")
    finally:
        # Coil released and port freed on every path, including Ctrl-C.
        arm.close()
