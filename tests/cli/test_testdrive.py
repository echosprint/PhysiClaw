"""Tests for `physiclaw.cli.testdrive` — the post-flash bring-up check.

The arm is faked at the lazy-import boundary (`StylusArm` is imported
inside the command body) and the interactive command loop is fed through
CliRunner stdin — no serial hardware, no wall-clock waits.
"""

from __future__ import annotations

import importlib

import pytest
import typer
from typer.testing import CliRunner

from physiclaw.core.hardware.device import DeviceNotFound, ProtocolError

testdrive_mod = importlib.import_module("physiclaw.cli.testdrive")

app = typer.Typer()
app.command()(testdrive_mod.testdrive)
runner = CliRunner()


class FakeArm:
    """Records the command's hardware calls; no serial underneath."""

    def __init__(self) -> None:
        self.port = "/dev/cu.fake"
        self.jogs: list[tuple[float, float]] = []
        self.taps = 0
        self.long_presses = 0
        self.setup_called = False
        self.closed = False

    def setup(self) -> None:
        self.setup_called = True

    def jog(self, dx: float, dy: float) -> None:
        self.jogs.append((dx, dy))

    def tap(self) -> None:
        self.taps += 1

    def long_press(self) -> None:
        self.long_presses += 1

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_arm(mocker) -> FakeArm:
    """A FakeArm wired in place of StylusArm; no live server recorded."""
    arm = FakeArm()
    mocker.patch("physiclaw.core.hardware.arm.StylusArm", return_value=arm)
    mocker.patch("physiclaw.common.runtime_state.read_live", return_value=None)
    return arm


def _cmds(*commands: str) -> str:
    """Stdin feed for the interactive loop, one command per prompt."""
    return "".join(c + "\n" for c in commands)


# ---------- the command loop ----------


def test_moves_and_actions_execute(fake_arm) -> None:
    result = runner.invoke(
        app, [], input=_cmds("X20Y20", "Y-10", "tap", "longpress", "q")
    )

    assert result.exit_code == 0
    assert fake_arm.setup_called
    assert fake_arm.jogs == [(20.0, 20.0), (0.0, -10.0)]
    assert fake_arm.taps == 1
    assert fake_arm.long_presses == 1
    assert fake_arm.closed


def test_commands_ignore_case_and_spaces(fake_arm) -> None:
    result = runner.invoke(app, [], input=_cmds("x10 Y5", "TAP", "LongPress", "Q"))

    assert result.exit_code == 0
    assert fake_arm.jogs == [(10.0, 5.0)]
    assert fake_arm.taps == 1
    assert fake_arm.long_presses == 1


def test_decimal_and_single_axis_moves(fake_arm) -> None:
    result = runner.invoke(app, [], input=_cmds("X1.5Y-2.5", "X-10", "q"))

    assert result.exit_code == 0
    assert fake_arm.jogs == [(1.5, -2.5), (-10.0, 0.0)]


def test_unknown_input_shows_help_line(fake_arm) -> None:
    result = runner.invoke(app, [], input=_cmds("wat", "q"))

    assert result.exit_code == 0
    assert "try X10, Y10, X-10Y-10" in result.output
    assert fake_arm.jogs == []


def test_oversize_move_rejected(fake_arm) -> None:
    # Caps are inclusive and per-axis: exactly ±20 in X / ±50 in Y pass,
    # anything beyond either cap is refused (X20.5 shows X's cap is its
    # own, not Y's).
    result = runner.invoke(
        app, [], input=_cmds("X500", "X20.5", "Y50.5", "X20", "Y-50", "q")
    )

    assert result.exit_code == 0
    assert "within ±20 mm in X and ±50 mm in Y" in result.output
    assert fake_arm.jogs == [(20.0, 0.0), (0.0, -50.0)]


def test_axes_diagram_and_help_shown(fake_arm) -> None:
    result = runner.invoke(app, [], input=_cmds("q"))

    assert "origin" in result.output
    assert "[A]" in result.output
    assert "[B]" in result.output
    assert "Motors A and B" in result.output
    assert "q to finish" in result.output
    assert fake_arm.jogs == []
    assert fake_arm.closed  # port released on quit


# ---------- guards and error paths ----------


def test_refuses_when_server_live(mocker) -> None:
    mocker.patch("physiclaw.common.runtime_state.read_live", return_value={"pid": 42})
    arm_spy = mocker.patch("physiclaw.core.hardware.arm.StylusArm")

    result = runner.invoke(app, [])

    assert result.exit_code == 1
    assert "stop it first" in result.output
    arm_spy.assert_not_called()


def test_exits_plainly_when_board_missing(mocker) -> None:
    mocker.patch("physiclaw.common.runtime_state.read_live", return_value=None)
    mocker.patch(
        "physiclaw.core.hardware.arm.StylusArm",
        side_effect=DeviceNotFound("GRBL device not found — no serial ports"),
    )

    result = runner.invoke(app, [])

    assert result.exit_code == 1
    assert "GRBL device not found" in result.output


def test_board_dying_midrun_closes_port(fake_arm, mocker) -> None:
    mocker.patch.object(
        fake_arm, "jog", side_effect=ProtocolError("GRBL error: error:9")
    )

    result = runner.invoke(app, [], input=_cmds("x10"))

    assert result.exit_code == 1
    assert "stopped responding" in result.output
    assert fake_arm.closed
