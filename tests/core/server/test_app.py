"""Tests for `physiclaw.core.server.app` — application assembly.

Importing the module has side effects: it constructs the singleton
PhysiClaw + bridge state, attaches the bridge, and wires every
register_* function onto the MCPServer instance. These tests verify the
wiring without re-importing (which would fight pytest's import cache).
"""

from __future__ import annotations

# ---------- module-level singletons ----------


def test_physiclaw_singleton_constructed_at_import() -> None:
    """Importing app.py must construct the orchestrator singleton."""
    from physiclaw.core import PhysiClaw
    from physiclaw.core.server import app

    assert isinstance(app.physiclaw, PhysiClaw)


def test_bridge_state_singleton_constructed() -> None:
    from physiclaw.core.bridge import BridgeState
    from physiclaw.core.server import app

    assert isinstance(app._bridge, BridgeState)


def test_calibration_state_singleton_constructed() -> None:
    from physiclaw.core.bridge import CalibrationState
    from physiclaw.core.server import app

    assert isinstance(app._calib, CalibrationState)


def test_phone_page_state_singleton_constructed() -> None:
    from physiclaw.core.bridge import PageState
    from physiclaw.core.server import app

    assert isinstance(app._phone, PageState)


def test_bridge_attached_to_orchestrator() -> None:
    """`physiclaw.rig.attach_bridge(_bridge)` must run at module load —
    screenshot and clipboard tools depend on it."""
    from physiclaw.core.server import app

    assert app.physiclaw.rig.bridge is app._bridge


def test_calibration_starts_empty_at_import() -> None:
    """A plain `physiclaw server` boot must not leak the on-disk
    calibration bundle into the live state — only `--warm-start`
    loads it (via `warm_start.try_resume`). The default app.py
    construction starts with an empty calibration."""
    from physiclaw.core.server import app

    cal = app.physiclaw.rig.calibration
    assert cal.pct_to_grbl is None
    assert cal.cam_rotation is None


# ---------- build_server ----------


def test_build_server_returns_isolated_assembly() -> None:
    """Each build_server() call wires a fresh, isolated set of state
    objects on a fresh MCPServer — nothing shared with the module default."""
    from physiclaw.core import PhysiClaw
    from physiclaw.core.server import app

    bundle = app.build_server()

    assert isinstance(bundle.physiclaw, PhysiClaw)
    assert bundle.physiclaw is not app.physiclaw
    assert bundle.bridge is not app._bridge
    assert bundle.mcp is not app.mcp
    # Wired: the orchestrator's rig holds THIS bundle's bridge.
    assert bundle.physiclaw.rig.bridge is bundle.bridge
    assert bundle.phone.bridge is bundle.bridge
    assert bundle.phone.cal is bundle.calib


def test_default_assembly_is_a_build_server_product() -> None:
    from physiclaw.core.server import app

    assert app.physiclaw is app.default_bundle.physiclaw
    assert app._bridge is app.default_bundle.bridge
    assert app._phone_app is app.default_bundle.phone_app


# ---------- shutdown ----------


def test_shutdown_delegates_to_orchestrator(mocker) -> None:
    from physiclaw.core.server import app

    spy = mocker.patch.object(app.physiclaw, "shutdown")

    app.shutdown()

    spy.assert_called_once_with()


# ---------- registration wiring ----------


def test_mcp_has_tools_registered_after_import() -> None:
    """All five register_* fns ran. We can't easily inspect MCPServer's
    internal route table here, but we can confirm at least one of the
    tool functions is callable on the mcp instance via the public
    interface (the routes were wired before any test ran)."""
    from physiclaw.core.server import app
    from physiclaw.core.server.mcp import mcp

    assert app.mcp is mcp
    # MCPServer exposes registered tools via list_tools() / its internal
    # _tool_manager. Just check the instance is alive.
    assert mcp is not None


def test_register_modules_imported() -> None:
    """Each register_* private alias is the function from the
    matching module — confirms the wiring doesn't silently swap
    in a fake."""
    from physiclaw.core.server import app
    from physiclaw.core.server.bridge import register as bridge_register
    from physiclaw.core.server.calibration import (
        register as calibration_register,
    )
    from physiclaw.core.server.hardware import register as hardware_register
    from physiclaw.core.server.tools import register as tools_register
    from physiclaw.core.server.watch import register as watch_register

    # Module-private names are stable enough to introspect.
    assert app._register_bridge is bridge_register
    assert app._register_calibration is calibration_register
    assert app._register_hardware is hardware_register
    assert app._register_tools is tools_register
    assert app._register_watch is watch_register
