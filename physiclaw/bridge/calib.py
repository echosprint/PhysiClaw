"""CalibrationState — server-controlled calibration page state.

The server sets the phase (what the page displays). The page reports
touch events back. The phase controls which visual targets appear and
what interactions trigger a green flash.
"""

import logging
import threading

from physiclaw.bridge.nonce import (
    NONCE_CSS_X,
    NONCE_CSS_Y,
    NONCE_DARK,
    NONCE_LIGHT,
    NONCE_SQUARE_SIZE,
)
from physiclaw.calibration.transforms import ViewportShift
from physiclaw.hardware.iphone import AssistiveTouch

log = logging.getLogger(__name__)


class CalibrationState:
    """Server-controlled calibration page state.

    The server sets the phase (what the page displays). The page reports
    touch events back. The phase controls which visual targets appear and
    what interactions trigger a green flash.
    """

    # Grid dot positions (must match bridge.html and the calibration plan)
    GRID_COLS_PCT = [0.25, 0.50, 0.75]
    GRID_ROWS_PCT = [0.20, 0.40, 0.50, 0.60, 0.80]

    # Valid calibration phases (server → page display commands).
    # Phases are visual primitives — each renders one thing on the
    # page; they're reused across different calibration steps.
    PHASES = {
        "idle",              # blank — waiting
        "screenshot_cal",    # orange square at viewport (100, 200) for pre-cal
        "center",            # orange circle at screen center
        "markers",           # blue UP + red RIGHT labels for camera orientation
        "grid",              # 15 red dots at known viewport positions
        "dot",               # single orange dot at a given (x, y) in 0-1
        "assistive_touch",   # AT circle + color nonce barcode
    }

    def __init__(self):
        self.lock = threading.Lock()  # protects shared fields across threads
        self.phase: str = "idle"  # current display phase (one of PHASES)
        self.dot_position: tuple[float, float] | None = (
            None  # (x, y) as 0-1 for "dot" phase
        )
        self.touches: list[dict] = []  # accumulated touch events from the phone
        self._touch_event = threading.Event()  # set when a new touch event arrives
        self.screen_dimension: dict | None = (
            None  # {"width", "height", "dpr", "viewport_width", "viewport_height"}
        )
        self.viewport_shift: ViewportShift | None = (
            None  # viewport→screenshot offset + DPR from pre-cal step
        )
        self._screenshot_nonce: list[int] | None = (
            None  # NONCE_COUNT bits for Step 7
        )

    def set_phase(self, phase: str, **kwargs):
        """Set the calibration display phase.

        Args:
            phase: one of self.PHASES
            dot_x, dot_y: position for "dot" phase (0-1 decimals)
            direction: expected direction for "swipe" phase
        """
        if phase not in self.PHASES:
            raise ValueError(f"Unknown phase: {phase}. Must be one of {self.PHASES}")
        with self.lock:
            self.phase = phase
            self.dot_position = None
            self.touches = []
            self._touch_event.clear()
            if phase == "dot":
                self.dot_position = (kwargs.get("dot_x", 0.5), kwargs.get("dot_y", 0.5))
            if phase == "assistive_touch":
                self._screenshot_nonce = kwargs.get("nonce_bits")

    def report_touch(self, touch: dict):
        """Page reports a touch event. x, y are 0-1 percentages relative to screen."""
        with self.lock:
            self.touches.append(touch)
        self._touch_event.set()
        log.debug(f"Calibration touch: ({touch.get('x')}, {touch.get('y')})")

    def wait_touch(self, timeout: float = 10.0) -> dict | None:
        """Block until a touch event arrives. Returns the touch or None.

        Caller must call flush_touches() first to clear stale events.
        This method waits for the NEXT report_touch() call.
        """
        if self._touch_event.wait(timeout=timeout):
            self._touch_event.clear()
            with self.lock:
                if self.touches:
                    return self.touches[-1]
        return None

    def flush_touches(self) -> list[dict]:
        """Drain and return all accumulated touch events, clearing the queue."""
        with self.lock:
            touches = list(self.touches)
            self.touches = []
            self._touch_event.clear()
        return touches

    def viewport_to_screenshot_pct(
        self, client_x: float, client_y: float
    ) -> tuple[float, float]:
        """Convert viewport CSS coords (clientX/clientY) to screenshot 0-1.

        Requires viewport_shift to be set via the pre-calibration step.
        """
        t = self.viewport_shift
        if t is None:
            raise RuntimeError(
                "Viewport shift not measured — run viewport-shift first"
            )
        return t.css_to_pct(client_x, client_y)

    def viewport_pct_to_screenshot_pct(
        self, vx: float, vy: float
    ) -> tuple[float, float]:
        """Convert viewport 0-1 percentages to screenshot 0-1.

        Used for converting grid dot positions (GRID_COLS_PCT/GRID_ROWS_PCT)
        from viewport space to screenshot space.
        """
        dim = self.screen_dimension
        if dim is None:
            raise RuntimeError("Screen dimension not set")
        return self.viewport_to_screenshot_pct(
            vx * dim["viewport_width"], vy * dim["viewport_height"]
        )

    def get_state(self) -> dict:
        """Get current display command for the page to render.

        Returns a nested dict — fields are grouped by feature so the schema
        is self-documenting:
            {
              phase, screen_dimension,
              grid:  {cols, rows},
              dot:   {x, y},                       # only when phase=="dot"
              at:    {x, y, r},                    # only when phase=="assistive_touch"
              nonce: {colors, x, y, size},         # only when phase=="assistive_touch"
            }
        """
        with self.lock:
            d = {
                "phase": self.phase,
                "screen_dimension": self.screen_dimension,
                "grid": {
                    "cols": self.GRID_COLS_PCT,
                    "rows": self.GRID_ROWS_PCT,
                },
            }
            if self.dot_position:
                d["dot"] = {"x": self.dot_position[0], "y": self.dot_position[1]}
            if self.phase == "assistive_touch" and self._screenshot_nonce is not None:
                d["at"] = {
                    "x": AssistiveTouch.AT_CSS_X,
                    "y": AssistiveTouch.AT_CSS_Y,
                    "r": AssistiveTouch.AT_RADIUS,
                }
                d["nonce"] = {
                    "colors": [
                        [NONCE_LIGHT] * 3 if b else [NONCE_DARK] * 3
                        for b in self._screenshot_nonce
                    ],
                    "x": NONCE_CSS_X,
                    "y": NONCE_CSS_Y,
                    "size": NONCE_SQUARE_SIZE,
                }
            return d
