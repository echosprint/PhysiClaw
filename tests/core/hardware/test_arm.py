"""Tests for `physiclaw.core.hardware.arm` — hardware fakes.

`serial.Serial` is replaced with a `FakeSerial` that records writes and
returns scripted replies. GRBL's textual line protocol is small enough
that a few patterns cover every public method:

  - `?`          → status reply, e.g. `<Idle|WPos:0.000,0.000>`
  - any G-code   → `ok\n` or `error:N\n` or `ALARM:N\n`
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import pytest

from physiclaw.core.hardware import arm as arm_mod
from physiclaw.core.hardware.arm import StylusArm

# ---------- FakeSerial ----------


class FakeSerial:
    """Minimal in-memory replacement for `serial.Serial`.

    `responses` is an iterable of byte sequences returned by successive
    `readline()` calls. `?` queries pull from `status_replies`. Status
    queries use `read(in_waiting)` rather than `readline`, so we keep
    those bytes separate.
    """

    def __init__(
        self,
        *,
        responses: Iterable[bytes] = (),
        status_replies: Iterable[bytes] = (),
        startup_bytes: bytes = b"",
    ):
        self.writes: list[bytes] = []
        self.responses = deque(responses)
        self.status_replies = deque(status_replies)
        self._startup = startup_bytes
        self._startup_consumed = False
        self.in_waiting = len(startup_bytes)
        self.closed = False

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        # `?` is a status query — caller will then read(in_waiting) for the response.
        if data == b"?":
            if self.status_replies:
                self._next_status = self.status_replies.popleft()
            else:
                self._next_status = b""
            self.in_waiting = len(self._next_status)
        return len(data)

    def readline(self) -> bytes:
        if self.responses:
            return self.responses.popleft()
        return b""

    def read(self, n: int) -> bytes:
        if not self._startup_consumed and self._startup:
            self._startup_consumed = True
            self.in_waiting = 0
            return self._startup
        out = getattr(self, "_next_status", b"")
        self.in_waiting = 0
        self._next_status = b""
        return out

    def reset_input_buffer(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _arm(
    mocker,
    *,
    responses: Iterable[bytes] = (),
    status_replies: Iterable[bytes] = (),
) -> tuple[StylusArm, FakeSerial]:
    """Build a StylusArm with FakeSerial as its serial backend, no setup()."""
    fake = FakeSerial(
        responses=list(responses),
        status_replies=list(status_replies),
    )
    mocker.patch.object(arm_mod, "detect_grbl", return_value="/dev/cu.fake")
    mocker.patch.object(arm_mod.serial, "Serial", return_value=fake)
    mocker.patch.object(arm_mod.time, "sleep")  # collapse the 2s settle
    return StylusArm(), fake


# ---------- __init__ ----------


def test_init_uses_detect_grbl_when_port_omitted(mocker) -> None:
    fake = FakeSerial()
    detect_spy = mocker.patch.object(
        arm_mod,
        "detect_grbl",
        return_value="/dev/cu.test",
    )
    serial_spy = mocker.patch.object(
        arm_mod.serial,
        "Serial",
        return_value=fake,
    )
    mocker.patch.object(arm_mod.time, "sleep")

    arm = StylusArm()

    assert arm.port == "/dev/cu.test"
    detect_spy.assert_called_once()
    serial_spy.assert_called_once_with("/dev/cu.test", 115200, timeout=3)


def test_init_raises_when_no_grbl_detected(mocker) -> None:
    # Ports exist but none answered as GRBL — manual port choice is the
    # right hint.
    mocker.patch.object(arm_mod, "detect_grbl", return_value=None)
    mocker.patch.object(arm_mod, "candidate_ports", return_value=["/dev/cu.usb-A"])

    with pytest.raises(Exception, match="specify port manually"):
        StylusArm()


def test_init_hints_unplugged_board_when_no_ports_at_all(mocker) -> None:
    # No serial ports on the bus almost always means the control board
    # isn't plugged in or isn't powered — say that, not "specify port
    # manually".
    mocker.patch.object(arm_mod, "detect_grbl", return_value=None)
    mocker.patch.object(arm_mod, "candidate_ports", return_value=[])

    with pytest.raises(Exception, match="plugged into USB"):
        StylusArm()


def test_init_uses_explicit_port_when_provided(mocker) -> None:
    fake = FakeSerial()
    serial_spy = mocker.patch.object(
        arm_mod.serial,
        "Serial",
        return_value=fake,
    )
    mocker.patch.object(arm_mod.time, "sleep")
    detect_spy = mocker.patch.object(arm_mod, "detect_grbl")

    arm = StylusArm(port="/dev/cu.explicit", baudrate=9600)

    assert arm.port == "/dev/cu.explicit"
    detect_spy.assert_not_called()
    serial_spy.assert_called_once_with("/dev/cu.explicit", 9600, timeout=3)


# ---------- _send ----------


def test_send_writes_command_with_lf_not_crlf(mocker) -> None:
    # LF only: a trailing CR makes FluidNC ack an extra empty line, desyncing
    # the reply stream. Lock in the single-LF terminator.
    arm, fake = _arm(mocker, responses=[b"ok\n"])

    arm._send("G0 X1 Y1")

    assert fake.writes[-1] == b"G0 X1 Y1\n"
    assert b"\r" not in fake.writes[-1]


def test_send_returns_immediately_when_wait_ok_false(mocker) -> None:
    arm, fake = _arm(mocker)  # no responses queued

    arm._send("G0 X0 Y0", wait_ok=False)


def test_send_skips_blank_lines_until_ok(mocker) -> None:
    arm, _ = _arm(mocker, responses=[b"\n", b"ok\n"])

    arm._send("G0 X0 Y0")  # must not raise


def test_send_raises_after_too_many_empty_lines(mocker) -> None:
    arm, _ = _arm(mocker, responses=[b""] * 10)

    with pytest.raises(Exception, match="GRBL not responding"):
        arm._send("G0 X0 Y0")


def test_send_raises_on_error_response(mocker) -> None:
    arm, _ = _arm(mocker, responses=[b"error:23\n"])

    with pytest.raises(Exception, match="GRBL error"):
        arm._send("G0 X0 Y0")


def test_send_raises_on_alarm_response(mocker) -> None:
    arm, _ = _arm(mocker, responses=[b"ALARM:1\n"])

    with pytest.raises(Exception, match="GRBL alarm"):
        arm._send("G0 X0 Y0")


def test_send_optional_error_does_not_desync_next_command(mocker) -> None:
    # FluidNC rejects a read-only setting with `error:162` then a trailing
    # `[MSG:ERR: ...]` line (LF terminator → no extra ok). After swallowing the
    # optional error, the stray `[MSG]` is a non-terminator the next command's
    # read loop skips, so M5 still reads its OWN ok and never sees a stale one.
    arm, _ = _arm(
        mocker,
        responses=[
            b"error:162\n",
            b"[MSG:ERR: Read-only setting]\n",  # $32=0 reply (no ok)
            b"ok\n",  # M5's own reply
        ],
    )

    arm._send("$32=0", optional=True)
    arm._send("M5")  # must not raise — skips the stray [MSG], reads its ok


# ---------- _query_status ----------


def test_query_status_returns_status_line(mocker) -> None:
    arm, _ = _arm(
        mocker,
        status_replies=[b"<Idle|WPos:0.000,0.000>\r\n"],
    )

    line = arm._query_status()

    assert line.startswith("<")


def test_query_status_returns_empty_when_no_response(mocker) -> None:
    arm, _ = _arm(mocker, status_replies=[b""])

    assert arm._query_status() == ""


# ---------- wait_idle ----------


def test_wait_idle_returns_when_idle(mocker) -> None:
    arm, _ = _arm(
        mocker,
        status_replies=[b"<Idle|WPos:0.000,0.000>\n"],
    )

    arm.wait_idle()  # no raise


def test_wait_idle_polls_until_idle(mocker) -> None:
    arm, _ = _arm(
        mocker,
        status_replies=[b"<Run|WPos:1.0,1.0>\n", b"<Idle|WPos:1.0,1.0>\n"],
    )

    arm.wait_idle()


def test_wait_idle_raises_on_timeout(mocker) -> None:
    arm, _ = _arm(
        mocker,
        status_replies=[b"<Run|WPos:0,0>\n"] * 50,
    )
    # Force time.time() to advance past the deadline.
    times = iter([0.0, 100.0, 100.0])
    mocker.patch.object(arm_mod.time, "time", side_effect=lambda: next(times))

    with pytest.raises(RuntimeError, match="Arm not idle"):
        arm.wait_idle(timeout=1)


# ---------- position ----------


def test_position_parses_wpos(mocker) -> None:
    arm, _ = _arm(
        mocker,
        status_replies=[b"<Idle|WPos:1.500,2.500|FS:0,0>\n"],
    )

    x, y = arm.position()

    assert x == 1.5 and y == 2.5


def test_position_falls_back_to_mpos_minus_wco(mocker) -> None:
    arm, _ = _arm(
        mocker,
        status_replies=[b"<Idle|MPos:5.000,7.000|WCO:1.000,2.000>\n"],
    )

    x, y = arm.position()

    assert x == 4.0 and y == 5.0


def test_position_uses_mpos_only_with_warning(
    mocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    arm, _ = _arm(
        mocker,
        status_replies=[b"<Idle|MPos:3.000,4.000>\n"],
    )

    with caplog.at_level(logging.WARNING, logger="physiclaw.core.hardware.arm"):
        x, y = arm.position()

    assert (x, y) == (3.0, 4.0)
    assert any("not reporting WPos" in r.getMessage() for r in caplog.records)


def test_position_raises_when_unparseable(mocker) -> None:
    arm, _ = _arm(mocker, status_replies=[b"<Idle>\n"])

    with pytest.raises(RuntimeError, match="Cannot parse position"):
        arm.position()


# ---------- setup / unlock / set_origin / return_to_origin ----------


def test_setup_unlocks_when_alarm_present(mocker) -> None:
    # Setup runs: VERSION → status → SET_ORIGIN → MM → ABS → F → IDLE →
    # $10=0 → $32 → $30 → M5. If alarm detected, also unlock between
    # status and SET_ORIGIN.
    arm, fake = _arm(
        mocker,
        # $I + $X (unlock) + 6 base + $32 + $30 + M5 = 11 ok's
        responses=[b"ok\n"] * 11,
        status_replies=[b"<Alarm|WPos:0,0>\n"],
    )

    arm.setup()

    written = b"".join(fake.writes)
    assert b"$X" in written
    assert b"$I" in written
    assert b"G92" in written
    assert b"$10=0" in written


def test_setup_configures_solenoid_pwm(mocker) -> None:
    arm, fake = _arm(
        mocker,
        responses=[b"ok\n"] * 10,
        status_replies=[b"<Idle|WPos:0,0>\n"],
    )

    arm.setup()

    written = b"".join(fake.writes)
    assert b"$32=0" in written  # spindle mode
    assert b"$30=1000" in written  # S-range ceiling
    assert fake.writes[-1] == b"M5\n"  # failsafe coil-off last


def test_setup_skips_unlock_when_no_alarm(mocker) -> None:
    arm, fake = _arm(
        mocker,
        responses=[b"ok\n"] * 10,  # no $X
        status_replies=[b"<Idle|WPos:0,0>\n"],
    )

    arm.setup()

    assert b"$X\n" not in fake.writes


def test_setup_swallows_optional_pwm_errors(mocker) -> None:
    # FluidNC rejects live $32/$30 writes (error:162) — setup must not raise.
    arm, fake = _arm(
        mocker,
        # $I, G92, G21, G90, F, $1, $10 ok (7) → $32 error:162 →
        # $30 error:3 → M5 ok
        responses=[b"ok\n"] * 7 + [b"error:162\n", b"error:3\n", b"ok\n"],
        status_replies=[b"<Idle|WPos:0,0>\n"],
    )

    arm.setup()  # must not raise

    assert fake.writes[-1] == b"M5\n"


def test_unlock_sends_X(mocker) -> None:
    arm, fake = _arm(mocker, responses=[b"ok\n"])

    arm.unlock()

    assert any(b"$X" in w for w in fake.writes)


def test_set_origin_sends_G92(mocker) -> None:
    arm, fake = _arm(mocker, responses=[b"ok\n"])

    arm.set_origin()

    assert any(b"G92" in w for w in fake.writes)


def test_set_work_position_sends_G92_with_coords(mocker) -> None:
    arm, fake = _arm(mocker, responses=[b"ok\n"])

    arm.set_work_position(-1.0, -1.0)

    assert any(b"G92 X-1.000 Y-1.000 Z0" in w for w in fake.writes)


def test_return_to_origin_fast_moves_and_waits(mocker) -> None:
    arm, fake = _arm(
        mocker,
        responses=[b"ok\n"],
        status_replies=[b"<Idle|WPos:0,0>\n"],
    )

    arm.return_to_origin()

    assert any(b"X0.000Y0.000" in w for w in fake.writes)


# ---------- direction mapping ----------


def test_set_direction_mapping_builds_8_directions() -> None:
    arm = StylusArm.__new__(StylusArm)  # bypass __init__
    arm.set_direction_mapping(right_vec=(1.0, 0.0), down_vec=(0.0, 1.0))

    # 8 cardinal + diagonal entries.
    assert set(arm.MOVE_DIRECTIONS.keys()) == {
        "right",
        "left",
        "bottom",
        "top",
        "top-left",
        "top-right",
        "bottom-left",
        "bottom-right",
    }
    assert arm.MOVE_DIRECTIONS["right"] == (1.0, 0.0)
    assert arm.MOVE_DIRECTIONS["left"] == (-1.0, -0.0)
    assert arm.MOVE_DIRECTIONS["bottom"] == (0.0, 1.0)
    assert arm.MOVE_DIRECTIONS["top"] == (-0.0, -1.0)


# ---------- solenoid / motion primitives ----------


# Solenoid contact mechanics (strike / hold / release) are unit-tested in
# test_solenoid.py. Here we only verify the arm composes a Solenoid bound to
# its own transport and that the gestures drive it onto the wire.


def test_arm_owns_a_solenoid_bound_to_its_transport(mocker) -> None:
    arm, _ = _arm(mocker)

    assert isinstance(arm.solenoid, arm_mod.Solenoid)
    # Bound to this arm's send/dwell so solenoid G-code flows over its serial.
    assert arm.solenoid._send == arm._send
    assert arm.solenoid._dwell == arm._dwell


def test_lift_stylus_releases_the_solenoid(mocker) -> None:
    arm, _ = _arm(mocker)
    arm.solenoid = mocker.MagicMock()

    arm.lift_stylus()

    arm.solenoid.release.assert_called_once_with()


def test_dwell_emits_G4(mocker) -> None:
    arm, fake = _arm(mocker, responses=[b"ok\n"])

    arm._dwell(0.5)

    assert any(b"G4 P0.5" in w for w in fake.writes)


def test_rapid_to_emits_G0(mocker) -> None:
    arm, fake = _arm(mocker, responses=[b"ok\n"])

    arm.rapid_to(1.5, 2.5, speed=5000)

    cmd = next(w for w in fake.writes if b"G0" in w)
    assert b"X1.500" in cmd
    assert b"Y2.500" in cmd
    assert b"F5000" in cmd


def test_linear_move_emits_G1(mocker) -> None:
    arm, fake = _arm(mocker, responses=[b"ok\n"])

    arm._linear_move(0.0, 1.0)

    assert any(b"G1 X0.000Y1.000" in w for w in fake.writes)


def test_jog_emits_relative_G0_then_restores_absolute(mocker) -> None:
    arm, fake = _arm(
        mocker,
        responses=[b"ok\n"] * 2,  # G91G0 jog + G90 restore
        status_replies=[b"<Idle|WPos:0,0>\n"],  # wait_idle
    )

    arm.jog(15.0, -15.0)

    written = b"".join(fake.writes)
    assert b"G91G0 X15.000Y-15.000" in written
    # G90 must follow the jog so relative mode can't leak into later moves.
    assert written.index(b"G91G0") < written.index(b"G90\n")
    assert b"?" in written  # settled via wait_idle


# ---------- public gestures ----------


def test_tap_strikes_solenoid(mocker) -> None:
    arm, fake = _arm(
        mocker,
        # _strike (M3 S1000, G4, M5, G4 rebound) = 4 ok's, then wait_idle.
        responses=[b"ok\n"] * 4,
        status_replies=[b"<Idle|WPos:0,0>\n"],
    )

    arm.tap()

    written = b"".join(fake.writes)
    assert b"M3 S1000" in written
    assert b"G4 P0.08" in written  # TAP_DURATION
    assert b"M5" in written
    assert b"G4 P0.2" in written  # spring-rebound dwell before any next move


def test_double_tap_fires_two_strikes(mocker) -> None:
    arm, fake = _arm(
        mocker,
        # M3/dwell/M5/dwell × 2 = 8 ok's, then status.
        responses=[b"ok\n"] * 8,
        status_replies=[b"<Idle|WPos:0,0>\n"],
    )

    arm.double_tap()

    # Two strikes, a short contact-breaking gap between them, and one full
    # spring-clear release at the end (so a following move can't drag).
    strike_count = sum(1 for w in fake.writes if w == b"M3 S1000\n")
    gap_count = sum(1 for w in fake.writes if w == b"G4 P0.1\n")
    release_count = sum(1 for w in fake.writes if w == b"G4 P0.2\n")
    assert strike_count == 2
    assert (
        gap_count == 1
    )  # brief inter-tap lift (DOUBLE_TAP_GAP_MS), not a full release
    assert release_count == 1  # full spring-clear only after the second strike


def test_long_press_strikes_and_holds(mocker) -> None:
    arm, fake = _arm(
        mocker,
        responses=[b"ok\n"] * 6,
        status_replies=[b"<Idle|WPos:0,0>\n"],
    )

    arm.long_press()

    written = b"".join(fake.writes)
    assert b"M3 S1000" in written  # strike
    assert b"M3 S750" in written  # hold current (HOLD_S)
    assert b"G4 P1.2" in written  # LONG_PRESS_DURATION
    assert b"G4 P0.2" in written  # spring-rebound dwell on release


def test_swipe_holds_solenoid_across_linear_move(mocker) -> None:
    arm, fake = _arm(mocker, responses=[b"ok\n"] * 7)
    arm.set_direction_mapping(right_vec=(1.0, 0.0), down_vec=(0.0, 1.0))

    arm.swipe("right", speed="fast")

    written = b"".join(fake.writes)
    # Swipe strikes then drops to the unified HOLD_S (S750) for the slide.
    assert b"M3 S1000" in written  # strike to make contact
    assert b"M3 S750" in written  # held at HOLD_S during the slide
    assert b"G91G1" in written
    assert b"F10000" in written  # fast speed
    assert written.rfind(b"M5") > written.rfind(b"G91G1")  # release after slide
    assert written.rfind(b"G4 P0.2") > written.rfind(b"M5")  # rebound before park


def test_swipe_raises_when_directions_unset(mocker) -> None:
    arm, _ = _arm(mocker, responses=[])

    with pytest.raises(RuntimeError, match="MOVE_DIRECTIONS not set"):
        arm.swipe("right")


def test_swipe_to_presses_slides_to_endpoint_releases(mocker) -> None:
    # press(3) + G1 linear(1) + release(2) = 6 ok's, then wait_idle.
    arm, fake = _arm(
        mocker,
        responses=[b"ok\n"] * 6,
        status_replies=[b"<Idle|WPos:0,0>\n"],
    )

    arm.swipe_to(12.5, -4.0, speed="slow")

    written = b"".join(fake.writes)
    assert b"M3 S1000" in written  # strike to make contact
    assert b"M3 S750" in written  # held at HOLD_S during the slide
    assert b"G1 X12.500Y-4.000" in written  # slide to the mm endpoint
    assert b"F3000" in written  # slow speed
    assert written.rfind(b"M5") > written.rfind(b"G1 X12.500")  # release after slide
    assert written.rfind(b"G4 P0.2") > written.rfind(b"M5")  # rebound before park


def test_swipe_to_start_dwell_anchors_touch_before_slide(mocker) -> None:
    # press(3) + start-dwell G4(1) + G1 linear(1) + release(2) = 7 ok's.
    arm, fake = _arm(
        mocker,
        responses=[b"ok\n"] * 7,
        status_replies=[b"<Idle|WPos:0,0>\n"],
    )

    arm.swipe_to(12.5, -4.0, speed="fast", start_dwell=0.08)

    written = b"".join(fake.writes)
    # The anchoring dwell holds contact at the start BEFORE the slide begins —
    # so iOS registers the edge touch-down before the drag arms the pop.
    assert b"G4 P0.08" in written
    assert written.find(b"G4 P0.08") < written.rfind(b"G1 X12.500")


def test_swipe_to_end_dwell_holds_endpoint_before_release(mocker) -> None:
    # press(3) + G1 linear(1) + end-dwell G4(1) + release(2) = 7 ok's.
    arm, fake = _arm(
        mocker,
        responses=[b"ok\n"] * 7,
        status_replies=[b"<Idle|WPos:0,0>\n"],
    )

    arm.swipe_to(12.5, -4.0, speed="medium", end_dwell=0.6)

    written = b"".join(fake.writes)
    # The endpoint hold sits between the slide and the release — the
    # pause iOS reads as "open the app switcher" instead of "go home".
    assert b"G4 P0.6" in written
    assert written.find(b"G4 P0.6") > written.find(b"G1 X12.500")
    assert written.find(b"G4 P0.6") < written.rfind(b"M5")


# ---------- move ----------


def test_move_emits_relative_fast_move(mocker) -> None:
    arm, fake = _arm(
        mocker,
        responses=[b"ok\n"] * 2,
        status_replies=[b"<Idle|WPos:0,0>\n"],
    )
    arm.set_direction_mapping(right_vec=(1.0, 0.0), down_vec=(0.0, 1.0))

    arm.move("right", "small")

    written = b"".join(fake.writes)
    assert b"G91G0" in written
    assert b"X3" in written  # small distance = 3mm


def test_move_normalizes_diagonal(mocker) -> None:
    arm, fake = _arm(
        mocker,
        responses=[b"ok\n"] * 2,
        status_replies=[b"<Idle|WPos:0,0>\n"],
    )
    arm.set_direction_mapping(right_vec=(1.0, 0.0), down_vec=(0.0, 1.0))

    arm.move("top-right", "medium")

    # Diagonal distance should be normalized to 8mm magnitude.
    written = b"".join(fake.writes)
    # Each axis gets 8/sqrt(2) ≈ 5.657 mm.
    assert b"5.657" in written or b"-5.657" in written


def test_move_raises_when_directions_unset(mocker) -> None:
    arm, _ = _arm(mocker, responses=[])

    with pytest.raises(RuntimeError, match="MOVE_DIRECTIONS not set"):
        arm.move("right")


# ---------- close ----------


def test_close_releases_coil_and_closes_serial(mocker) -> None:
    arm, fake = _arm(mocker, responses=[b"ok\n"] * 2)

    arm.close()

    assert any(w == b"M5\n" for w in fake.writes)  # coil released
    assert fake.closed is True


def test_close_swallows_coil_release_failure(mocker) -> None:
    arm, fake = _arm(
        mocker,
        responses=[b"error:1\n"],  # M5 release errors
    )

    # Even if the release errors, close() must not raise.
    arm.close()

    assert fake.closed is True
