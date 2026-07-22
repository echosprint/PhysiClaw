"""Solenoid stylus-tip actuator — the Z executor, driven via GRBL spindle PWM.

The stylus tip is pushed onto the glass by a solenoid (not a stepper Z axis):
the coil is fired through the GRBL spindle PWM pin (``M3 S<duty>`` / ``M5``),
and a return spring lifts the tip when the coil de-energizes. The tip sits at
a fixed mechanical height, so there is no Z depth to calibrate — only timing.

Hit-and-keep current profile (a solenoid burns its coil out if held at peak):
  - strike at ``HIT_S`` to pull the iron core in,
  - settle ``SETTLE_MS``, then drop to ``HOLD_S`` to keep it seated for a
    sustained press (long-press / swipe),
  - release with ``M5``.

On release the coil cuts instantly, but the spring rebound is slow
(~``RELEASE_MS``). We dwell that long after every ``M5`` so the tip is clear
of the glass before any following XY move — otherwise it drags a line across
the screen. Defaults were tuned on the MKS DLC32 bench rig; see
``scripts/grbl_solenoid_test.py``.

This driver depends only on two GRBL-transport callables so it stays decoupled
from the arm and is testable with fakes:
  - ``send(cmd, *, optional=False)`` — write one G-code line, block until ``ok``
  - ``dwell(seconds)`` — emit a planner-side ``G4`` dwell

SAFETY — a solenoid burns its coil out if left energized at the strike current.
Two guarantees protect the hardware:
  1. ``HOLD_S`` must be below ``HIT_S`` (validated at construction). The hold
     duty is the only level safe to *sustain*; the strike duty is for the brief
     pull-in only.
  2. Every method that energizes the coil de-energizes it before returning,
     including on any exception or KeyboardInterrupt (via ``try/finally`` +
     :meth:`_force_off`), so an error mid-tap can never leave the coil hot.
     :meth:`press` is the sole exception — it deliberately hands a held coil
     to the caller, who must release it; :meth:`held` enforces that.
The firmware (FluidNC ``off_on_alarm: true``) is a third, hardware-level net:
GRBL drops the spindle PWM whenever it enters an alarm state.
"""

import logging
from contextlib import contextmanager
from typing import Callable, Iterator

log = logging.getLogger(__name__)

DwellFn = Callable[[float], None]
SendFn = Callable[..., None]


class Solenoid:
    """Hit-and-keep solenoid driver over a GRBL spindle-PWM channel."""

    # Electrical / mechanical parameters (PWM duty on a 0..PWM_MAX scale).
    HIT_S = 1000  # peak strike duty — pulls the iron core in (brief only!)
    # Hold duty after the strike — the level the coil drops to once the core is
    # pulled in, for both a long-press and the slide of a swipe. Must stay below
    # HIT_S (validated in __init__) or a sustained hold overheats the coil.
    # S750 (~75% duty) keeps the tip pressed firmly enough that a *stationary*
    # long-press holds continuous capacitive contact. At the cooler S500 the tip
    # eased off at the strike→hold transition and the panel read the touch as
    # lifting, so a long-press registered as a single tap; S500 was fine for a
    # swipe (the slide keeps re-establishing contact) but not for a still hold.
    # Lower it only if your coil keeps a still touch registered at less.
    HOLD_S = 750
    SETTLE_MS = 80  # dwell after the strike before dropping to the hold level
    # Spring rebound after M5: the coil de-energizes instantly but the spring
    # takes ~200ms to lift the tip clear of the glass. We dwell this long on
    # release so the next XY move can't start while the tip is still down
    # (which would drag a line across the screen). Increase if you still see
    # dragging; decrease only if your spring snaps back faster.
    RELEASE_MS = 200
    # Lift between the two strikes of a double-tap. Only long enough to break
    # capacitive contact so the screen sees two distinct touches — NOT the full
    # RELEASE_MS spring-clear dwell (that exists to stop the tip dragging during
    # a following XY move, of which there is none mid-double-tap). Short keeps
    # down-to-down (the tap duration + this) well under the iOS ~300ms double-tap
    # window; chaining two full taps pushed it to ~280ms, so timing jitter
    # intermittently split the gesture into two single taps.
    DOUBLE_TAP_GAP_MS = 100
    SPINDLE_MODE = 0  # $32=0 — spindle mode; laser mode emits PWM only while moving
    PWM_MAX = 1000  # $30 — S-value range ceiling

    GCODE_ON = "M3 S{s}"  # energize at PWM duty S
    GCODE_OFF = "M5"  # release (coil off)

    def __init__(self, send: SendFn, dwell: DwellFn) -> None:
        # The hold current is sustained, so it must stay below the strike
        # current — a hold at/above HIT_S would cook the coil.
        if not 0 < self.HOLD_S < self.HIT_S:
            raise ValueError(
                f"unsafe solenoid hold current: HOLD_S={self.HOLD_S} must be >0 "
                f"and < HIT_S={self.HIT_S} — a sustained hold at/above the "
                f"strike current burns out the coil"
            )
        self._send = send
        self._dwell = dwell

    def configure(self) -> None:
        """Set spindle PWM mode + range so ``M3 S`` works regardless of the
        firmware's defaults, then leave the coil off.

        ``optional=True`` because FluidNC takes ``$32``/``$30`` from YAML and
        rejects live writes (``error:162``), while a bare GRBL board accepts
        them — we try anyway so both work. Called once from ``StylusArm.setup``.
        """
        self._send(f"$32={self.SPINDLE_MODE}", optional=True)  # spindle, not laser
        self._send(f"$30={self.PWM_MAX}", optional=True)  # S-range ceiling
        self._send(self.GCODE_OFF)  # failsafe — start with the coil off

    def _force_off(self) -> None:
        """Best-effort coil-off for error paths — never raises.

        Called from the ``finally`` of every energized window so a release can
        never mask the original error or, far worse, leave the coil hot. If
        even this ``M5`` fails the transport is likely dead; we log and move on
        (the firmware's ``off_on_alarm`` is the last line of defense).

        The trailing ``RELEASE_MS`` dwell matters here too: the error path
        unwinds into ``rig.engaged()``'s auto-park, and without the dwell the
        park G0 starts while the spring is still rebounding — ~200 ms of
        tip-on-glass drag registering phantom touches mid-error-recovery.
        """
        try:
            self._send(self.GCODE_OFF)
            self._dwell(self.RELEASE_MS / 1000.0)
        except Exception:
            log.exception("solenoid: emergency release (M5) failed")

    @contextmanager
    def _energized(self) -> Iterator[None]:
        """Guard an energized sequence: if the body doesn't run to completion —
        any exception, including KeyboardInterrupt — force the coil off so it
        can never sit at peak current after a fault. This is the single home of
        the "never leave the coil hot" invariant; every energizing method runs
        its sequence inside it.
        """
        done = False
        try:
            yield
            done = True
        finally:
            if not done:
                self._force_off()

    # ─── Press / release ─────────────────────────────────────

    def press(self) -> None:
        """Press the tip onto the glass and keep it there (hit-and-keep).

        Strike at ``HIT_S`` to pull the core in, settle, then drop to ``HOLD_S``
        so the coil stays seated without sitting at peak current. Leaves the
        solenoid energized — the caller MUST :meth:`release` it (use
        :meth:`held` to guarantee that). If anything fails before the hold level
        is reached, the coil is forced off so a half-finished press can't sit at
        peak current.

        Buffers commands only — the caller must ``dwell``/``wait_idle`` to
        ensure contact before proceeding.
        """
        with self._energized():
            self._send(self.GCODE_ON.format(s=self.HIT_S))
            self._dwell(self.SETTLE_MS / 1000.0)
            self._send(self.GCODE_ON.format(s=self.HOLD_S))

    def release(self) -> None:
        """Lift the tip off the glass and wait for the spring to clear it.

        ``M5`` cuts the coil instantly, but the spring rebound takes
        ~``RELEASE_MS`` to pull the tip off the glass. The trailing dwell
        keeps the motion planner busy that long, so any XY move queued after
        this can't start until the tip is airborne — otherwise it drags.
        """
        self._send(self.GCODE_OFF)
        self._dwell(self.RELEASE_MS / 1000.0)

    @contextmanager
    def held(self) -> Iterator[None]:
        """Hold the tip down for the duration of the ``with`` block, then lift
        it — **even if the block raises**. Use to bracket a swipe slide or any
        held gesture; the exception-safe way to pair :meth:`press` with
        :meth:`release`. On a clean exit the tip lifts with the normal
        spring-rebound dwell; on error the coil is forced off immediately.
        """
        self.press()
        with self._energized():
            yield
            self.release()

    # ─── Taps ────────────────────────────────────────────────

    def tap(self, duration: float) -> None:
        """Momentary tap: press at ``HIT_S``, hold ``duration`` s, release.

        Short enough that the reduced hold current isn't needed — ``HIT_S`` for
        ~80ms doesn't cook the coil and gives a crisp contact. The coil is
        forced off if the dwell or release fails, so a tap can never leave it
        hot. :meth:`release` adds the spring-rebound dwell on the clean path.
        """
        with self._energized():
            self._send(self.GCODE_ON.format(s=self.HIT_S))
            self._dwell(duration)
            self.release()

    def double_tap(self, duration: float) -> None:
        """Two momentary taps registered as a single double-tap.

        Strike, hold ``duration``, lift briefly (``DOUBLE_TAP_GAP_MS`` — only
        enough to break capacitive contact, not the full spring-clear), strike
        again, then a normal :meth:`release` so a following XY move can't drag.
        The whole pair runs in one ``_energized`` window, so a fault anywhere
        forces the coil off. Keeping the inter-tap gap short (vs. chaining two
        :meth:`tap` calls, each with its 200ms release) holds down-to-down
        under the iOS double-tap window so the strikes don't read as two taps.
        """
        with self._energized():
            self._send(self.GCODE_ON.format(s=self.HIT_S))
            self._dwell(duration)
            self._send(self.GCODE_OFF)
            self._dwell(self.DOUBLE_TAP_GAP_MS / 1000.0)
            self._send(self.GCODE_ON.format(s=self.HIT_S))
            self._dwell(duration)
            self.release()

    def press_and_hold(self, duration: float) -> None:
        """Press and hold for ``duration`` s, then release: strike, settle,
        drop to ``HOLD_S``, hold, lift. For a long-press, where the coil must
        stay seated past the brief strike window. Release is guaranteed via
        :meth:`held`."""
        with self.held():
            self._dwell(duration)
