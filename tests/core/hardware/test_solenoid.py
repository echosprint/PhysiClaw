"""Tests for `physiclaw.core.hardware.solenoid` — the Z-actuator driver.

`Solenoid` depends only on two transport callables (`send` + `dwell`), so we
exercise it with a tiny recorder that captures the exact command sequence —
no serial port, no GRBL. Wire-level integration (solenoid G-code reaching the
serial port through StylusArm) is covered by the gesture tests in test_arm.py.
"""

from __future__ import annotations

import pytest

from physiclaw.core.hardware.solenoid import Solenoid


class Boom(RuntimeError):
    """Injected transport failure (e.g. GRBL alarm / serial timeout)."""


class Recorder:
    """Captures the ordered (op, *args) calls a Solenoid makes.

    ``fail_on`` is an optional predicate ``(op, payload) -> bool``; when it
    returns True the corresponding ``send``/``dwell`` raises :class:`Boom`,
    simulating a mid-gesture GRBL error so we can prove the coil is forced off.
    """

    def __init__(self, fail_on=None) -> None:
        self.calls: list[tuple] = []
        self._fail_on = fail_on

    @property
    def sends(self) -> list[str]:
        return [c[1] for c in self.calls if c[0] == "send"]

    def send(self, cmd: str, optional: bool = False) -> None:
        self.calls.append(("send", cmd, optional))
        if self._fail_on and self._fail_on("send", cmd):
            raise Boom(f"transport failed on {cmd!r}")

    def dwell(self, seconds: float) -> None:
        self.calls.append(("dwell", seconds))
        if self._fail_on and self._fail_on("dwell", seconds):
            raise Boom(f"transport failed on dwell {seconds}")


@pytest.fixture
def rec() -> Recorder:
    return Recorder()


@pytest.fixture
def sol(rec: Recorder) -> Solenoid:
    return Solenoid(send=rec.send, dwell=rec.dwell)


# ---------- configure ----------


def test_configure_sets_pwm_mode_and_range_optionally_then_coil_off(
    rec: Recorder, sol: Solenoid
) -> None:
    sol.configure()

    assert rec.calls == [
        ("send", f"$32={Solenoid.SPINDLE_MODE}", True),  # spindle mode, optional
        ("send", f"$30={Solenoid.PWM_MAX}", True),  # S-range, optional
        ("send", Solenoid.GCODE_OFF, False),  # failsafe coil-off
    ]


# ---------- press / release ----------


def test_press_strikes_settles_then_holds(rec: Recorder, sol: Solenoid) -> None:
    sol.press()

    assert rec.calls == [
        ("send", "M3 S1000", False),  # strike at HIT_S
        ("dwell", Solenoid.SETTLE_MS / 1000.0),  # settle
        ("send", "M3 S750", False),  # drop to HOLD_S, coil left energized
    ]


def test_release_cuts_then_waits_for_rebound(rec: Recorder, sol: Solenoid) -> None:
    sol.release()

    assert rec.calls == [
        ("send", "M5", False),  # cut the coil
        ("dwell", Solenoid.RELEASE_MS / 1000.0),  # wait for the spring to lift
    ]


# ---------- tap / press_and_hold ----------


def test_tap_fires_for_duration_then_releases(rec: Recorder, sol: Solenoid) -> None:
    sol.tap(0.08)

    assert rec.calls == [
        ("send", "M3 S1000", False),  # strike at HIT_S
        ("dwell", 0.08),  # contact duration
        ("send", "M5", False),  # release (via release())
        ("dwell", Solenoid.RELEASE_MS / 1000.0),  # spring rebound
    ]
    # No hold step for a momentary tap.
    assert ("send", "M3 S750", False) not in rec.calls


def test_double_tap_uses_short_gap_then_full_release(
    rec: Recorder, sol: Solenoid
) -> None:
    sol.double_tap(0.08)

    assert rec.calls == [
        ("send", "M3 S1000", False),  # first strike
        ("dwell", 0.08),  # contact duration
        ("send", "M5", False),  # lift
        (
            "dwell",
            Solenoid.DOUBLE_TAP_GAP_MS / 1000.0,
        ),  # brief gap — break contact only
        ("send", "M3 S1000", False),  # second strike
        ("dwell", 0.08),  # contact duration
        ("send", "M5", False),  # release
        ("dwell", Solenoid.RELEASE_MS / 1000.0),  # full spring-clear after the pair
    ]
    # The inter-tap gap must stay below a full release, or down-to-down drifts
    # past the iOS double-tap window and the pair reads as two single taps.
    assert Solenoid.DOUBLE_TAP_GAP_MS < Solenoid.RELEASE_MS


def test_double_tap_forces_coil_off_when_a_strike_fails(rec: Recorder) -> None:
    # A transport failure mid-double-tap must still force the coil off.
    rec._fail_on = lambda op, p: op == "send" and p == "M3 S1000"
    sol = Solenoid(send=rec.send, dwell=rec.dwell)

    with pytest.raises(Boom, match="M3 S1000"):
        sol.double_tap(0.08)

    assert rec.sends[-1] == "M5"  # emergency release fired


def test_press_and_hold_drops_to_hold_current_for_duration(
    rec: Recorder, sol: Solenoid
) -> None:
    sol.press_and_hold(1.2)

    assert rec.calls == [
        ("send", "M3 S1000", False),  # strike at HIT_S
        ("dwell", Solenoid.SETTLE_MS / 1000.0),  # settle
        ("send", "M3 S750", False),  # hold current
        ("dwell", 1.2),  # hold duration
        ("send", "M5", False),  # release
        ("dwell", Solenoid.RELEASE_MS / 1000.0),  # spring rebound
    ]


# ---------- formatting uses the configured S-values ----------


def test_duty_levels_track_class_constants(rec: Recorder) -> None:
    class HotSolenoid(Solenoid):
        HIT_S = 800
        HOLD_S = 120

    HotSolenoid(send=rec.send, dwell=rec.dwell).press_and_hold(0.5)

    assert "M3 S800" in rec.sends  # HIT_S override honored
    assert "M3 S120" in rec.sends  # HOLD_S override honored


# ---------- construction guard: hold current must be safe ----------


@pytest.mark.parametrize("hold", [1000, 1500, 0, -50])
def test_init_rejects_unsafe_hold_current(rec: Recorder, hold: int) -> None:
    """A hold at/above the strike duty (or non-positive) would burn the coil —
    reject it at construction so a misconfigured rig never energizes."""

    class BadSolenoid(Solenoid):
        HOLD_S = hold  # HIT_S stays 1000

    with pytest.raises(ValueError, match="HOLD_S"):
        BadSolenoid(send=rec.send, dwell=rec.dwell)


def test_init_accepts_hold_below_hit(rec: Recorder) -> None:
    class OkSolenoid(Solenoid):
        HOLD_S = 999  # just under HIT_S — allowed (thermal tuning is the user's)

    OkSolenoid(send=rec.send, dwell=rec.dwell)  # no raise


# ---------- coil is forced off on any mid-gesture failure ----------


def test_tap_forces_coil_off_when_dwell_raises(rec: Recorder) -> None:
    rec._fail_on = lambda op, _p: op == "dwell"  # GRBL stalls during the tap
    sol = Solenoid(send=rec.send, dwell=rec.dwell)

    with pytest.raises(Boom):
        sol.tap(0.08)

    assert rec.sends[0] == "M3 S1000"  # coil was energized
    assert rec.sends[-1] == "M5"  # ...and forced off despite the failure


def test_press_forces_coil_off_if_hold_send_raises(rec: Recorder) -> None:
    # Fail the drop-to-hold write, after the peak strike already turned the coil on.
    rec._fail_on = lambda op, p: op == "send" and p == "M3 S750"
    sol = Solenoid(send=rec.send, dwell=rec.dwell)

    with pytest.raises(Boom):
        sol.press()

    # Strike on, hold attempt failed, then forced off — never stuck at peak.
    assert rec.sends == ["M3 S1000", "M3 S750", "M5"]


def test_held_forces_coil_off_when_block_raises() -> None:
    rec = Recorder()
    sol = Solenoid(send=rec.send, dwell=rec.dwell)

    with pytest.raises(ValueError, match="slide failed"):
        with sol.held():
            assert rec.sends[-1] == "M3 S750"  # held at hold current inside block
            raise ValueError("slide failed")

    assert rec.sends[-1] == "M5"  # released despite the error
    # Error path uses the immediate force-off, not the rebound-dwell release.
    assert ("dwell", Solenoid.RELEASE_MS / 1000.0) not in rec.calls


def test_held_releases_normally_on_clean_exit(rec: Recorder) -> None:
    sol = Solenoid(send=rec.send, dwell=rec.dwell)

    with sol.held():
        pass

    # Clean exit takes the normal release: M5 + spring-rebound dwell.
    assert rec.sends[-1] == "M5"
    assert ("dwell", Solenoid.RELEASE_MS / 1000.0) in rec.calls


def test_force_off_failure_is_swallowed_and_original_error_survives(
    rec: Recorder,
) -> None:
    # The tap dwell fails AND the emergency M5 also fails: the original error
    # must still propagate (not be masked by the release failure).
    rec._fail_on = lambda op, p: op == "dwell" or (op == "send" and p == "M5")
    sol = Solenoid(send=rec.send, dwell=rec.dwell)

    with pytest.raises(Boom, match="dwell"):
        sol.tap(0.08)

    assert "M5" in rec.sends  # release was still attempted
