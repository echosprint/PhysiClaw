"""LAN bridge — phone ↔ server state for text, screenshots, and calibration.

Three data flows:
1. Text → clipboard: Agent sends text → phone displays it → tap copies → confirms.
2. Screenshot upload: iOS Shortcut takes screenshot → POSTs to server.
3. Calibration: Server controls page display, page reports touch events.
"""

from physiclaw.bridge.lan import get_lan_ip
from physiclaw.bridge.state import BridgeState
from physiclaw.bridge.phase import CalibrationState
from physiclaw.bridge.phone import PhoneState

__all__ = [
    "get_lan_ip",
    "BridgeState",
    "CalibrationState",
    "PhoneState",
]
