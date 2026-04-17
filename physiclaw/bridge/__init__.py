"""LAN bridge — phone ↔ server state for text, screenshots, and calibration.

Three data flows:
1. Text → clipboard: Agent sends text → phone displays it → tap copies → confirms.
2. Screenshot upload: iOS Shortcut takes screenshot → POSTs to server.
3. Calibration: Server controls page display, page reports touch events.
"""

from physiclaw.bridge.lan import bridge_base_urls, get_lan_ip, get_mdns_host
from physiclaw.bridge.state import BridgeState
from physiclaw.bridge.calib import CalibrationState
from physiclaw.bridge.page import PageState

__all__ = [
    "bridge_base_urls",
    "get_lan_ip",
    "get_mdns_host",
    "BridgeState",
    "CalibrationState",
    "PageState",
]
