"""PageState — coordinates mode switching between bridge and calibration."""

import logging
import threading

from physiclaw.bridge.state import BridgeState
from physiclaw.bridge.calib import CalibrationState

log = logging.getLogger(__name__)


class PageState:
    """Coordinates mode switching between calibration and bridge on one page.

    The phone runs a single page that can display calibration UI or bridge UI.
    The server controls which mode is active.
    """

    def __init__(self, bridge: BridgeState, cal: CalibrationState):
        self.bridge = bridge
        self.cal = cal
        self.lock = threading.Lock()
        self.mode: str = "bridge"  # "calibrate" or "bridge"

    def set_mode(self, mode: str, phase: str | None = None, **phase_kwargs):
        with self.lock:
            if self.mode != mode:
                self.mode = mode
                log.info(f"Phone mode → {mode}")
            if mode == "calibrate" and phase:
                self.cal.set_phase(phase, **phase_kwargs)

    def get_state(self) -> dict:
        """Unified state for the phone page poll."""
        with self.lock:
            mode = self.mode

        state = {"mode": mode}
        state["has_device_info"] = self.cal.screen_dimension is not None

        if mode == "calibrate":
            state.update(self.cal.get_state())
        else:
            state["text"] = self.bridge.current_text()

        return state
