"""PageState — coordinates mode switching between bridge and calibration."""

import logging
import threading

from physiclaw.core.bridge.calib import CalibrationState
from physiclaw.core.bridge.state import CAL_UPLOAD_WINDOW_SECONDS, BridgeState

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
            if mode == "calibrate":
                if phase:
                    self.cal.set_phase(phase, **phase_kwargs)
                # Calibration asks the user to double-tap AssistiveTouch;
                # open the upload window right away so an eager upload —
                # even one sent before the step's button is pressed —
                # is accepted instead of 403'd.
                self.bridge.arm_upload(CAL_UPLOAD_WINDOW_SECONDS)
            else:
                # Returning to bridge mode re-arms the touch gate: the
                # phase gate on /api/bridge/touch only rejects while
                # "idle", and without this reset the last step's phase
                # ("center", "dot", …) would leave the gate open for the
                # rest of the server's life.
                self.cal.set_phase("idle")

    def get_state(self) -> dict:
        """Unified state for the phone page poll."""
        with self.lock:
            mode = self.mode

        state = {"mode": mode}
        state["has_device_info"] = self.cal.screen_dimension is not None

        if mode == "calibrate":
            state.update(self.cal.get_state())
        else:
            # The queued text rides the poll ON PURPOSE: the phone must
            # display it so the operator can see what's about to be pasted
            # and verify the paste matches. It's the agent reporting its
            # own activity, not a secret — same trust model as the
            # clipboard-fetch endpoint (see core/server/planes.py).
            state["text"] = self.bridge.current_text()
            state["copied"] = self.bridge.is_copied()

        return state
