"""Stuck guard — engine-enforced loop detection.

The failure class: the model repeats itself for half an hour because a
prompt rule ("stop after N attempts") cannot work — compaction erases
the evidence the model would need to count. So the ENGINE counts.
Three detectors, one shared pair of tiers (production norms per
OpenHands / browser-use):

  tier 1 (warn)   the WARN_AT-th occurrence appends a ⚠ to the tool
                  result — the highest-adherence channel.
  tier 2 (block)  the BLOCK_AT-th is refused pre-dispatch with an error
                  result. Models sometimes ignore warnings; they cannot
                  ignore an action that doesn't execute.

The detectors, by the loop they catch:

  1. SAME TARGET  press-family gestures (tap / double_tap / long_press
     — switching gesture type on a dead element is not "changing
     method") whose results carry `screen: no visible change`
     (`physiclaw.verdict`). A silent refusal — stock-limit toast the
     camera never sees — looks exactly like this.
  2. ACTION CYCLES  repeating the same 2–3-action cycle (checkout ↔
     cart, tab ↔ tab ↔ back) — each action CHANGES the screen, so
     detector 1 is blind to it; identity comes from action signatures.
  3. SAME FAILING CALL  identical calls that raise on execution (e.g. a
     target overlapping the AssistiveTouch button) — no verdict ever
     exists, so detectors 1–2 never see them.
  4. POSITION ORBIT (warn-only)  BLOCK_AT of the last 2×BLOCK_AT presses
     landing on one spot, regardless of gesture type, verdict, or what
     ran in between — catches erratic circling that slips between 1–3.
     Advisory only: five presses on one spot is also what a legitimate
     "press + five times" task looks like, so the model judges from the
     warning; blocking stays with the evidence-based detectors.

What never counts: `screen: changed` presses (a qty stepper that works
RESETS its target), missing verdicts (camera hiccup → fail open),
swipes in detector 1 (repeated scrolls are normal; direction-alternating
swipes are caught by detector 2), and learned keyboard keys everywhere
(backspace-clearing a field is 10–20 legitimate presses).

Counters are per-step: they reset when the plan's `in_progress` step
changes, so a new objective gets a clean slate on the same coordinates.
False positives stay cheap by design (SWE-agent abandoned stuck
detection over them): tier 1 is advisory, and tier 2 needs four
camera-verified no-ops / four full cycles / four identical errors.
"""
from dataclasses import dataclass, field

from physiclaw.agent.engine import screen_layout
from physiclaw.config import CONFIG

# Gestures that count toward same-target tiers. Swipes are deliberately
# absent (see module docstring).
PRESS_TOOLS = frozenset({"tap", "double_tap", "long_press"})

# Two press centers within this L∞ distance are the same target —
# covers the coordinate jitter of re-transcribed bboxes without
# swallowing a genuinely different neighbor element.
MATCH_TOLERANCE = 0.02

WARN_AT = CONFIG.engine.same_target_warn
BLOCK_AT = CONFIG.engine.same_target_block


def _center(bbox: list) -> tuple[float, float] | None:
    try:
        left, top, right, bottom = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    return (left + right) / 2, (top + bottom) / 2


def _near(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return abs(a[0] - b[0]) <= MATCH_TOLERANCE and abs(a[1] - b[1]) <= MATCH_TOLERANCE


def _press_centers(name: str, arguments: dict) -> list[tuple[float, float]]:
    """Extract the press-family target centers of a gesture call.

    A plain press yields one center; a `sequence` yields one per
    press-family step (so a blocked element can't be smuggled through a
    batch); anything else yields none.
    """
    if name in PRESS_TOOLS:
        c = _center(arguments.get("bbox") or [])
        return [c] if c else []
    if name == "sequence":
        actions = arguments.get("actions")
        if not isinstance(actions, list):
            return []
        centers = []
        for step in actions:
            if not isinstance(step, dict) or step.get("tool_name") not in PRESS_TOOLS:
                continue
            c = _center(step.get("arg") or [])
            if c:
                centers.append(c)
        return centers
    return []


# Action signatures (detectors 2–3): presses = (tool, quantized
# center); swipes = (tool, direction); nav gestures = (tool,). Views,
# local tools, and exempt keyboard keys have NO signature — they
# neither extend nor break a cycle.

_NAV_TOOLS = frozenset({"go_back", "home_screen", "force_quit"})
# Longest lookback any detection needs: BLOCK_AT repeats of a period-3
# cycle (+1 headroom for the pre-append should_block probe). Derived so
# a config change can't silently outgrow the history.
_HISTORY_MAX = 3 * BLOCK_AT + 1

# Position-orbit window (detector 4): warn when BLOCK_AT of the last
# _ORBIT_WINDOW presses hit one quantized spot.
_ORBIT_WINDOW = 2 * BLOCK_AT


def _quant(center: tuple[float, float]) -> tuple[int, int]:
    """Quantize a center to MATCH_TOLERANCE cells so jittered re-taps
    share a signature (border-straddling jitter splits — conservative)."""
    return (round(center[0] / MATCH_TOLERANCE), round(center[1] / MATCH_TOLERANCE))


def _trailing_cycles(history: list[tuple]) -> int:
    """Max count of consecutive repeats of the trailing action cycle,
    for cycle periods 2 and 3. The cycle must contain ≥2 distinct
    signatures (A,A,A… is repetition, not a cycle — pagination and
    feed-scrolling stay untouched). ABAB → 2; ABCABC → 2; ABC → 0."""
    best = 0
    for period in (2, 3):
        if len(history) < 2 * period:
            continue
        block = history[-period:]
        if len(set(block)) < 2:
            continue
        n = 0
        i = len(history)
        while i >= period and history[i - period:i] == block:
            n += 1
            i -= period
        best = max(best, n)
    return best


@dataclass
class _Target:
    center: tuple[float, float]
    misses: int = 0  # camera-verified no-change presses on this spot


@dataclass
class StuckGuard:
    """Per-session loop detectors (see module docstring). Engine calls,
    in order: `observe_step` at each turn start, `should_block` before
    dispatching a gesture, then `record` (success) or `record_error`
    (execution raised) when its result comes back."""

    _targets: list[_Target] = field(default_factory=list)
    _history: list[tuple] = field(default_factory=list)  # action signatures
    _error_counts: dict[tuple, int] = field(default_factory=dict)
    _presses: list[tuple[int, int]] = field(default_factory=list)  # quantized press positions
    _orbit_warned: set[tuple[int, int]] = field(default_factory=set)
    _step_identity: str | None = None
    # Lazily read on the first press (None = not loaded yet) so Session
    # construction stays I/O-free; the layout file changes only during
    # first-run setup, which restarts the session anyway.
    _exempt: list[tuple[float, float]] | None = None

    def observe_step(self, identity: str | None) -> None:
        """Reset counters when the plan's in_progress step changes — a
        new objective gets a clean slate on the same coordinates."""
        if identity != self._step_identity:
            self._step_identity = identity
            self._targets.clear()
            self._history.clear()
            self._error_counts.clear()
            self._presses.clear()
            self._orbit_warned.clear()

    def should_block(self, name: str, arguments: dict) -> str | None:
        """Blocking message if this call would be the BLOCK_AT-th press on
        an exhausted target, the BLOCK_AT-th repeat of an action cycle, or
        the BLOCK_AT-th identical failing call — else None. Blocked calls
        are never dispatched and never recorded, so repeated attempts stay
        blocked without extending state."""
        for center in self._press_centers_counted(name, arguments):
            t = self._find(center)
            if t is not None and t.misses >= BLOCK_AT - 1:
                return (
                    f"BLOCKED — not executed: press #{t.misses + 1} on the same "
                    f"target this step, {t.misses} camera-verified no-ops so far. "
                    "This element is exhausted (refusing or dead). Pick a "
                    "different element or method, or report the blocker to the "
                    "user and close (CONVENTION § Stuck)."
                )
        sig = self._signature(name, arguments)
        if sig is not None:
            if self._error_counts.get(sig, 0) >= BLOCK_AT - 1:
                return (
                    f"BLOCKED — not executed: this exact call has already "
                    f"failed {BLOCK_AT - 1}×. Retrying identical arguments "
                    "won't fix the error — change the target or the method "
                    "(CONVENTION § Stuck)."
                )
            if _trailing_cycles(self._history + [sig]) >= BLOCK_AT:
                return (
                    f"BLOCKED — not executed: {BLOCK_AT}th repeat of the same "
                    "action cycle (A→B→…→A→B→…). Bouncing between the same "
                    "screens/elements is a loop even though each action "
                    "changes the screen. Change method or escalate "
                    "(CONVENTION § Stuck)."
                )
        return None

    def record_error(self, name: str, arguments: dict) -> str | None:
        """Feed a FAILED execution back in (MCP raised — e.g. a target
        overlapping the AssistiveTouch button). Identical errors are
        stronger loop evidence than no-ops: the same signature failing
        WARN_AT times returns a warning to append to the error result."""
        warnings: list[str] = []
        for center in self._press_centers_counted(name, arguments):
            orbit = self._track_orbit(center)
            if orbit:
                warnings.append(orbit)
        sig = self._signature(name, arguments)
        if sig is not None:
            n = self._error_counts.get(sig, 0) + 1
            self._error_counts[sig] = n
            if n == WARN_AT:
                warnings.append(
                    f"⚠ this exact call has failed {n}× with the same error. "
                    "Retrying identical arguments won't fix it — change the "
                    f"target or method (CONVENTION § Stuck). The engine blocks "
                    f"attempt #{BLOCK_AT}."
                )
        return "\n".join(warnings) if warnings else None

    def record(self, name: str, arguments: dict, changed: bool | None) -> str | None:
        """Feed a dispatched gesture's screen-change verdict back in.
        Returns tier-1 warning text to append to the tool result when a
        threshold is crossed, else None."""
        warnings: list[str] = []
        for center in self._press_centers_counted(name, arguments):
            orbit = self._track_orbit(center)
            if orbit:
                warnings.append(orbit)
            t = self._find(center)
            if changed is True:
                if t is not None:
                    self._targets.remove(t)
                continue
            if changed is None:
                continue  # no verdict → fail open
            if t is None:
                t = _Target(center=center)
                self._targets.append(t)
            t.misses += 1
            if t.misses == WARN_AT:
                warnings.append(
                    f"⚠ press #{t.misses} on this target this step, screen "
                    "unchanged every time. The element is refusing (a toast "
                    "you can never see — stock limit, cap, disabled) or dead. "
                    "Don't re-press: read the value you're changing, look for "
                    "a limit label, or switch method (CONVENTION § Stuck). "
                    f"The engine blocks press #{BLOCK_AT}."
                )
        # Cycle history extends regardless of the verdict — the loop it
        # detects is screen-CHANGING by nature. A successful execution
        # also clears the signature's error count (the error was fixed).
        sig = self._signature(name, arguments)
        if sig is not None:
            self._error_counts.pop(sig, None)
            self._history.append(sig)
            del self._history[:-_HISTORY_MAX]
            if _trailing_cycles(self._history) == WARN_AT:
                warnings.append(
                    f"⚠ you've repeated the same action cycle {WARN_AT}× "
                    "(A→B→…→A→B→…). Each action changes the screen, but "
                    "nothing progresses — this is a loop. Change method "
                    f"(CONVENTION § Stuck); the engine blocks repeat "
                    f"#{BLOCK_AT}."
                )
        return "\n".join(warnings) if warnings else None

    # ---------- internals ----------

    def _track_orbit(self, center: tuple[float, float]) -> str | None:
        """Detector 4: append this executed press to the position window;
        warn (once per position per step) when BLOCK_AT of the last
        _ORBIT_WINDOW presses share its quantized spot."""
        pos = _quant(center)
        self._presses.append(pos)
        del self._presses[:-_ORBIT_WINDOW]
        if self._presses.count(pos) < BLOCK_AT or pos in self._orbit_warned:
            return None
        self._orbit_warned.add(pos)
        return (
            f"⚠ {BLOCK_AT} of your last {len(self._presses)} presses hit "
            "this same spot. If its value/state is genuinely progressing "
            "(a counter stepping up), carry on; otherwise you are circling "
            "one element — change method (CONVENTION § Stuck)."
        )

    def _press_centers_counted(self, name: str, arguments: dict) -> list[tuple[float, float]]:
        if self._exempt is None:
            self._exempt = [
                c for c in (
                    _center(b) for b in screen_layout.repeatable_key_boxes()
                ) if c
            ]
        return [
            c for c in _press_centers(name, arguments)
            if not any(_near(c, e) for e in self._exempt)
        ]

    def _signature(self, name: str, arguments: dict) -> tuple | None:
        """Ping-pong identity of a gesture call, or None for calls that
        neither extend nor break a pattern (views, local tools, exempt
        keyboard keys — typing two letters alternately must not trip)."""
        if name in PRESS_TOOLS:
            counted = self._press_centers_counted(name, arguments)
            return (name, _quant(counted[0])) if counted else None
        if name == "swipe":
            direction = arguments.get("direction")
            return ("swipe", direction) if direction else None
        if name in _NAV_TOOLS:
            return (name,)
        if name == "sequence":
            counted = self._press_centers_counted(name, arguments)
            return ("sequence", _quant(counted[0]) if counted else None)
        return None

    def _find(self, center: tuple[float, float]) -> _Target | None:
        for t in self._targets:
            if _near(center, t.center):
                return t
        return None
