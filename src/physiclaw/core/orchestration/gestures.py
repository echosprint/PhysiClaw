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
verdict marker (`physiclaw.verdict`), and echoed free-form content
could carry marker-like text.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, get_args

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

SWIPE_DISTANCES: dict[Size, float] = {
    "s": 0.1,
    "m": 0.3,
    "l": 0.5,
    "xl": 0.75,
    "xxl": 0.90,
}
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


@dataclass(frozen=True)
class SendToClipboard:
    text: str


Gesture = Tap | DoubleTap | LongPress | Swipe | SendToClipboard


class GestureValidator:
    """Argument validation + AssistiveTouch geometry for every gesture path.

    Holds accessor callables rather than the objects themselves: the
    orchestrator replaces its AssistiveTouch and transforms at
    calibration time, so each check must read the current one.
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

    def require_no_at_crossing(self, bbox: list[float], direction: str):
        """Raise if a swipe from bbox center in `direction` would cross AssistiveTouch."""
        cx, cy = self._transforms().bbox_center_pct(bbox)
        if self._assistive_touch().swipe_crosses_at(cx, cy, direction):
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
