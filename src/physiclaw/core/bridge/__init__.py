"""LAN bridge — phone ↔ server state for text, screenshots, and calibration.

Three data flows:
1. Text → clipboard: Agent sends text → phone displays it → tap copies → confirms.
2. Screenshot upload: iOS Shortcut takes screenshot → POSTs to server.
3. Calibration: Server controls page display, page reports touch events.
"""

from physiclaw.core.bridge.calib import CalibrationState
from physiclaw.core.bridge.lan import bridge_base_urls, get_lan_ip, get_mdns_host
from physiclaw.core.bridge.page import PageState
from physiclaw.core.bridge.state import BridgeState

__all__ = [
    "bridge_base_urls",
    "get_lan_ip",
    "get_mdns_host",
    "BridgeState",
    "CalibrationState",
    "PageState",
]
