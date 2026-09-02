"""Typed gesture values and their validation.

`sequence` steps arrive as string-keyed dicts (`{"tool_name": "swipe",
"arg": {...}}`); the public gesture tools arrive as loose args. Both
funnel through `GestureValidator`, which resolves them into the frozen
value types below and owns every check that doesn't need hardware:
bbox shape/range, swipe enums, and the AssistiveTouch geometry guards.
The orchestrator then dispatches on gesture type — a switch the type
checker can see, instead of per-tool dict unpacking.

Error texts here are agent-facing and byte-pinned by tests. The
sequence-path messages deliberately avoid `repr()`-ing raw args: they
join the batch action text the engine scans for the screen-change
verdict marker (`physiclaw.common.verdict`), and echoed free-form content
could carry marker-like text.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, get_args

from physiclaw.common import gesture_vocab
from physiclaw.core.vision.util import validate_bbox

if TYPE_CHECKING:
    from physiclaw.core.calibration import ScreenTransforms
    from physiclaw.core.hardware.iphone import AssistiveTouch

# One vocabulary, two views: the Literal aliases type every swipe
# signature (here and in the orchestrator), and the runtime tuples the
# range checks scan are derived from them — the annotations can't drift
# from what `validate_swipe` accepts.
Direction = Literal["up", "down", "left", "right"]
Size = Literal["s", "m", "l", "xl", "xxl"]
Speed = Literal["slow", "medium", "fast"]

# The ladder itself lives in the dependency-free vocabulary; the Literal
# above types it, and tests pin the two spellings to each other.
SWIPE_DISTANCES: dict[str, float] = gesture_vocab.SWIPE_DISTANCES
SWIPE_DIRS: tuple[Direction, ...] = get_args(Direction)
SWIPE_SPEEDS: tuple[Speed, ...] = get_args(Speed)


@dataclass(frozen=True)
class Tap:
    bbox: list[float]


@dataclass(frozen=True)
class DoubleTap:
    bbox: list[float]


@dataclass(frozen=True)
class LongPress:
    bbox: list[float]


@dataclass(frozen=True)
class Swipe:
    bbox: list[float]
    direction: Direction
    size: Size = "m"
    speed: Speed = "medium"
    # Hold the tip stationary this long AFTER contact, before the slide —
    # see arm.swipe_to. Only the iOS edge-pan macros below need it; the
    # agent-facing swipe paths always leave it 0.
    start_dwell: float = 0.0
    # Hold the tip pressed this long at the endpoint, before release —
    # see arm.swipe_to. Only the switcher-open macro below needs it.
    end_dwell: float = 0.0


@dataclass(frozen=True)
class SendToClipboard:
    text: str


Gesture = Tap | DoubleTap | LongPress | Swipe | SendToClipboard


# ─── iOS macro recipes ─────────────────────────────────────────
#
# The system-gesture step lists the orchestrator's macro tools replay.
# They are iOS policy (edge zones, switcher choreography), kept here as
# data so the orchestrator stays a coordinator: a different phone bed
# someday means a new recipe set, not orchestrator surgery. Each step
# runs through the same validated dispatch as an agent gesture.

# Hold the tip at the very edge this long after contact, before the slide,
# so iOS registers a touch-down inside the narrow interactive-pop zone —
# without it, a fast slide starting on contact skips past the edge and the
# gesture reads as a content scroll (no back). ~9 display frames; 0.08
# (~1.5 frames) was too tight and intermittently missed the arming window.
BACK_EDGE_DWELL_SECONDS = 0.15

# Shared anchor zones — every recipe that touches the same screen region
# names the same bbox, so a phone-bed change edits one line.
BOTTOM_EDGE = [0.4, 0.96, 0.6, 0.98]
# Back-swipe start: ~2% in from the physical edge, NOT x≈0. iOS's edge-pan
# zone is ~30pt (~7% of screen width), so 2% is still deep inside it — but
# a start hugging the glass edge (the old x≈0.005) left the arm <0.5mm of
# outward calibration tolerance before the tip landed on the bezel and no
# touch registered at all: the silent go_back miss. Landing too far INWARD
# is the other cliff (past ~7% the app owns the drag — WhatsApp reads it
# as swipe-to-reply), so keep the start well under half the zone.
LEFT_EDGE = [0.015, 0.4, 0.025, 0.6]

# Home: bottom-edge swipe up.
HOME_SCREEN: tuple[Gesture, ...] = (Swipe(BOTTOM_EDGE, "up", "xl", "fast"),)

# Back: left-edge swipe right. Starts just inside the edge zone (see
# LEFT_EDGE) and dwells briefly on contact so the touch is seen inside
# iOS's edge-pan zone before the slide arms the interactive pop.
# "slow", not "fast": on-rig A/B in a WhatsApp thread (2026-07-28) —
# fast (~167mm/s) failed to pop 2/2 while slow (~50mm/s) popped 2/2,
# same start point and travel. At stylus speeds the fast slide loses
# the pan (tip skip / tracking); commit doesn't need velocity anyway —
# releasing past half the screen commits at any speed.
GO_BACK: tuple[Gesture, ...] = (
    Swipe(LEFT_EDGE, "right", "xxl", "slow", BACK_EDGE_DWELL_SECONDS),
)

# Force-quit, four steps. The switcher opens with the current app's
# card clipped at the RIGHT edge and the previous app centered — so:
# open, drag the clipped card itself to center (the card follows the
# stylus; a center-anchored momentum scroll scatters and used to quit
# the wrong app), fling it, tap the wallpaper below the cards to
# dismiss (flinging the last card lands home by itself).

# Hold at the open-swipe's end this long before lifting — the pause is
# what iOS reads as "open the app switcher"; releasing on arrival reads
# as "go home". (start_dwell anchors the touch-down; this anchors the
# stop.)
SWITCHER_HOLD_SECONDS = 0.6

# Where the current app's clipped card sits when the switcher opens.
RIGHT_CENTER = [0.8, 0.4, 0.9, 0.6]

FORCE_QUIT: tuple[Gesture, ...] = (
    # Short slow swipe-up-and-hold: the hold does the talking, the
    # slide stays small and gentle to keep go-home momentum out.
    Swipe(BOTTOM_EDGE, "up", "s", "slow", end_dwell=SWITCHER_HOLD_SECONDS),
    # Drag the clipped current-app card from the right edge to center.
    Swipe(RIGHT_CENTER, "left", "l", "medium"),
    # Fling it: start on the centered card's lower half for max travel.
    Swipe([0.4, 0.7, 0.6, 0.8], "up", "xl", "fast"),
    # Dismiss the switcher via the wallpaper strip below the cards.
    Tap([0.4, 0.92, 0.6, 0.96]),
)

# Unlock, mechanical pieces: wake the screen, then swipe up from the
# bottom edge to leave the lock-screen cover. Two named gestures rather
# than a recipe tuple — the orchestrator settles the camera on the
# raised keypad after the swipe (the passcode keypad lives only seconds
# once the swipe opens it), and the passcode entry that follows is
# imperative (OCR poll loop) and stays in the orchestrator.

# The wake tap lands on dead space on BOTH screens a retry can hit: on
# the lock-screen cover it's empty wallpaper (clear of the flashlight /
# camera buttons); on the passcode screen it's the gap between the
# entry dots (bottom ~0.26) and digit row 1 (top ~0.35). NOT screen
# center: that is the "5" key, and a prepended 5 turns 111111 into a
# wrong passcode — repeated wrong entries walk toward the iOS lockout.
WAKE_SCREEN: Gesture = Tap([0.4, 0.28, 0.6, 0.32])
# "xl" travel, same as HOME_SCREEN's: an "l" swipe released mid-screen
# can read as a partial peek that iOS settles back onto the cover
# instead of committing to the passcode keypad. Caveat: on an
# already-open keypad (a fast retry catching one still up) a committed
# bottom-edge swipe is iOS's CANCEL gesture and dismisses it — the cold
# wake this flow assumes always shows the cover, where the swipe is
# always correct.
UNLOCK_SWIPE: Gesture = Swipe(BOTTOM_EDGE, "up", "xl", "fast")

# The tool-phone passcode — a throwaway code, never the user's real one. The
# single source of truth for the unlock tap loop: it taps ONE digit key N
# times, so the code must be a repdigit (every digit the same). To change it,
# edit this line only; the orchestrator derives the key and the tap count.
UNLOCK_PASSCODE = "111111"
# A raise, not an assert: asserts vanish under `python -O`, and this guard
# is the only thing keeping a mixed-digit edit from silently mis-typing.
if len(set(UNLOCK_PASSCODE)) != 1:
    raise ValueError(
        "UNLOCK_PASSCODE must be a repdigit — the unlock tap loop enters one "
        "key repeatedly, so mixed digits can't be typed"
    )


class GestureValidator:
    """Argument validation + AssistiveTouch geometry for every gesture path.

    Holds accessor callables rather than the objects themselves: the
    rig's AssistiveTouch and transforms are replaced at calibration
    time, so each check must read the current one.
    """

    def __init__(
        self,
        assistive_touch: Callable[[], "AssistiveTouch"],
        transforms: Callable[[], "ScreenTransforms"],
    ):
        self._assistive_touch = assistive_touch
        self._transforms = transforms

    # ─── AssistiveTouch guards ─────────────────────────────────

    def require_no_at_overlap(self, bbox: list[float], gesture: str):
        """Raise if the bbox center would hit the AssistiveTouch button."""
        cx, cy = self._transforms().bbox_center_pct(bbox)
        if self._assistive_touch().overlaps_at(cx, cy):
            raise ValueError(
                f"{gesture} target {bbox} overlaps AssistiveTouch button — aim aside"
            )

    def require_no_at_crossing(self, bbox: list[float], direction: str, size: Size):
        """Raise if the swipe's actual travel segment would cross
        AssistiveTouch — same start/end geometry the rig executes
        (`swipe_end_pct`, clamped to screen), so a swipe whose bounded
        path never reaches the button isn't blocked for sharing its
        row/column band."""
        t = self._transforms()
        cx, cy = t.bbox_center_pct(bbox)
        ex, ey = t.swipe_end_pct(bbox, direction, SWIPE_DISTANCES[size])
        if self._assistive_touch().swipe_crosses_at(cx, cy, ex, ey):
            raise ValueError(
                f"swipe {direction} at {bbox} crosses AssistiveTouch button — aim aside"
            )

    # ─── Range checks ──────────────────────────────────────────

    def validate_swipe(self, bbox, direction, size, speed):
        """Raises ValueError if any swipe arg is out of range."""
        validate_bbox(bbox)
        if direction not in SWIPE_DIRS:
            raise ValueError(
                f"direction must be one of {SWIPE_DIRS}, got {direction!r}"
            )
        if size not in SWIPE_DISTANCES:
            raise ValueError(
                f"size must be one of {list(SWIPE_DISTANCES)}, got {size!r}"
            )
        if speed not in SWIPE_SPEEDS:
            raise ValueError(f"speed must be one of {SWIPE_SPEEDS}, got {speed!r}")

    # ─── Sequence-step parsing ─────────────────────────────────

    def parse_step(self, tool: str, arg) -> Gesture:
        """Resolve one string-keyed sequence step into a typed, range-
        validated gesture value.

        Error messages here avoid repr-ing the raw arg — they join the
        batch action text the engine scans for the verdict marker (see
        module docstring)."""
        if tool == "tap":
            validate_bbox(arg)
            return Tap(arg)
        if tool == "double_tap":
            validate_bbox(arg)
            return DoubleTap(arg)
        if tool == "long_press":
            validate_bbox(arg)
            return LongPress(arg)
        if tool == "swipe":
            if not isinstance(arg, dict) or "bbox" not in arg or "direction" not in arg:
                raise ValueError(
                    f"swipe arg needs a dict with bbox + direction, got "
                    f"{type(arg).__name__}"
                )
            bbox, direction = arg["bbox"], arg["direction"]
            size, speed = arg.get("size", "m"), arg.get("speed", "medium")
            self.validate_swipe(bbox, direction, size, speed)
            return Swipe(bbox, direction, size, speed)
        if tool == "send_to_clipboard":
            if not isinstance(arg, str):
                raise ValueError(
                    f"send_to_clipboard arg must be a string, got {type(arg).__name__}"
                )
            return SendToClipboard(arg)
        raise ValueError(f"tool {tool!r} not allowed in sequence")
