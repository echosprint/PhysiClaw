"""Application assembly: construct singletons and wire registrations.

The MCP instance lives in `physiclaw.core.server.mcp`. This module owns the
hardware/state singletons and binds every tool/route module to that
instance. Importing this module has the side effect of fully wiring the
server — `physiclaw.core.server.__init__` re-exports the public surface.
"""

import logging

from physiclaw.core.bridge import BridgeState, CalibrationState, PageState
from physiclaw.core import PhysiClaw
from physiclaw.core.server.planes import PhoneApp, build_bridge_app, build_control_app
from physiclaw.core.server.bridge import register as _register_bridge
from physiclaw.core.server.bridge import register_phone as _register_bridge_phone
from physiclaw.core.server.calibration import register as _register_calibration
from physiclaw.core.server.hardware import register as _register_hardware
from physiclaw.core.server.mcp import mcp
from physiclaw.core.server.tools import register as _register_tools
from physiclaw.core.server.watch import register as _register_watch

log = logging.getLogger(__name__)

# ─── Singletons ─────────────────────────────────────────────

physiclaw = PhysiClaw()
_bridge = BridgeState()
_calib = CalibrationState()
_phone = PageState(_bridge, _calib)
physiclaw.attach_bridge(_bridge)

# Calibration starts empty. The on-disk bundle at
# `~/.physiclaw/calibration/bundle.json` is loaded ONLY by
# `--warm-start` (in `core/server/warm_start.py:try_resume`). A plain
# `physiclaw server` boot ignores the bundle so a stale calibration
# can't silently leak into a fresh setup. `cli/status.py` and
# `cli/doctor.py` still read the bundle directly from disk for display
# purposes — that's unaffected by this change.


def shutdown():
    """Clean up hardware resources."""
    physiclaw.shutdown()


# ─── Wire tools and routes ──────────────────────────────────
# Control plane on `mcp` (loopback), phone-facing routes on `_phone_app`
# (LAN) — see core/server/planes.py for the split and its gates.

_phone_app = PhoneApp()

_register_tools(mcp, physiclaw)
_register_bridge(mcp, physiclaw, _bridge, _calib, _phone)
_register_bridge_phone(_phone_app, physiclaw, _bridge, _calib, _phone)
_register_hardware(mcp, physiclaw, _phone)
_register_calibration(mcp, physiclaw, _bridge, _calib, _phone)
_register_watch(mcp, physiclaw)


def build_apps(host: str = "127.0.0.1"):
    """(control_asgi, bridge_asgi) — built on demand by `cli/server.py`.
    `streamable_http_app()` is lazy-init inside FastMCP, so this is safe
    to call once serving is about to start. `host` is the control bind."""
    return build_control_app(mcp, host), build_bridge_app(_phone_app)
