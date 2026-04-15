"""Application assembly: construct singletons and wire registrations.

The MCP instance lives in `physiclaw.server.mcp`. This module owns the
hardware/state singletons and binds every tool/route module to that
instance. Importing this module has the side effect of fully wiring the
server — `physiclaw.server.__init__` re-exports the public surface.
"""

from physiclaw.bridge import BridgeState, CalibrationState, PageState
from physiclaw.core import PhysiClaw
from physiclaw.server.bridge import register as _register_bridge
from physiclaw.server.calibration import register as _register_calibration
from physiclaw.server.hardware import register as _register_hardware
from physiclaw.server.mcp import mcp
from physiclaw.server.tools import register as _register_tools
from physiclaw.server.watch import register as _register_watch

# ─── Singletons ─────────────────────────────────────────────

physiclaw = PhysiClaw()
_bridge = BridgeState()
_calib = CalibrationState()
_phone = PageState(_bridge, _calib)


def shutdown():
    """Clean up hardware resources."""
    physiclaw.shutdown()


# ─── Wire tools and routes ──────────────────────────────────

_register_tools(mcp, physiclaw)
_register_bridge(mcp, physiclaw, _bridge, _calib, _phone)
_register_hardware(mcp, physiclaw)
_register_calibration(mcp, physiclaw, _bridge, _calib, _phone)
_register_watch(mcp, physiclaw)
