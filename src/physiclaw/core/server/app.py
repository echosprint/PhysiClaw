"""Application assembly: construct singletons and wire registrations.

``build_server()`` is the one assembly function: it constructs a fresh
set of state objects (orchestrator, bridge, calibration page, phone
page, MCP instance, phone-plane app) and binds every tool/route module
to them. The module-level ``default_bundle = build_server(...)`` keeps the
historical import-time singletons — `physiclaw.core.server.__init__`
re-exports the public surface — while tests can call ``build_server()``
for an isolated wiring.
"""

import dataclasses
import logging
from typing import TYPE_CHECKING

from starlette.applications import Starlette

from physiclaw.core import PhysiClaw
from physiclaw.core.bridge import BridgeState, CalibrationState, PageState
from physiclaw.core.server.bridge import register as _register_bridge
from physiclaw.core.server.bridge import register_phone as _register_bridge_phone
from physiclaw.core.server.calibration import register as _register_calibration
from physiclaw.core.server.hardware import register as _register_hardware
from physiclaw.core.server.mcp import build_mcp, mcp
from physiclaw.core.server.planes import (
    ControlGate,
    PhoneApp,
    build_bridge_app,
    build_control_app,
)
from physiclaw.core.server.tools import register as _register_tools
from physiclaw.core.server.watch import register as _register_watch

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ServerBundle:
    """One fully-wired server: state objects + the two route collectors."""

    physiclaw: PhysiClaw
    bridge: BridgeState
    calib: CalibrationState
    phone: PageState
    mcp: "FastMCP"
    phone_app: PhoneApp


def build_server(mcp_instance: "FastMCP | None" = None) -> ServerBundle:
    """Construct and wire a complete server assembly.

    Each call returns a fresh, isolated wiring (state objects + tool and
    route registrations), so tests can assemble a server without
    touching the module-level default. ``mcp_instance`` lets the default
    assembly reuse the ``core.server.mcp`` singleton; omitted, a fresh
    FastMCP is built (registering twice on one instance would collide).

    Calibration starts empty. The on-disk bundle at
    `~/.physiclaw/calibration/bundle.json` is loaded ONLY by
    `--warm-start` (in `core/server/warm_start.py:try_resume`). A plain
    `physiclaw server` boot ignores the bundle so a stale calibration
    can't silently leak into a fresh setup. `cli/status.py` and
    `cli/doctor.py` still read the bundle directly from disk for
    display purposes.
    """
    mcp_ = mcp_instance if mcp_instance is not None else build_mcp()
    pl = PhysiClaw()
    bridge = BridgeState()
    calib = CalibrationState()
    phone = PageState(bridge, calib)
    pl.rig.attach_bridge(bridge)

    # Control plane on `mcp_` (loopback), phone-facing routes on
    # `phone_app` (LAN) — see core/server/planes.py for the split and
    # its gates.
    phone_app = PhoneApp()
    _register_tools(mcp_, pl)
    _register_bridge(mcp_, bridge, phone)
    _register_bridge_phone(phone_app, bridge, calib, phone)
    _register_hardware(mcp_, pl.rig, phone)
    _register_calibration(mcp_, pl.rig, bridge, calib, phone)
    _register_watch(mcp_, pl)
    return ServerBundle(
        physiclaw=pl,
        bridge=bridge,
        calib=calib,
        phone=phone,
        mcp=mcp_,
        phone_app=phone_app,
    )


# ─── Default assembly (import-time singletons) ──────────────
# `default_bundle` is the public handle — external consumers (the CLI's
# warm-start thread) read the assembly through it instead of reaching
# for this module's underscore aliases.

default_bundle = build_server(mcp_instance=mcp)
physiclaw = default_bundle.physiclaw
_bridge = default_bundle.bridge
_calib = default_bundle.calib
_phone = default_bundle.phone
_phone_app = default_bundle.phone_app


def shutdown() -> None:
    """Clean up hardware resources."""
    physiclaw.shutdown()


def build_apps(host: str = "127.0.0.1") -> tuple[ControlGate, Starlette]:
    """(control_asgi, bridge_asgi) — built on demand by `cli/server.py`.
    `streamable_http_app()` is lazy-init inside FastMCP, so this is safe
    to call once serving is about to start. `host` is the control bind."""
    return build_control_app(mcp, host), build_bridge_app(_phone_app)
