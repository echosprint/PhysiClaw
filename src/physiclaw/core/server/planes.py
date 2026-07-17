"""Two ASGI planes, one process — the bind split that isolates the arm.

The control plane (``/mcp`` + every setup/calibrate/watch route) binds
``127.0.0.1`` only: the LAN cannot reach the surface that drives the
stylus. The phone bridge (the handful of routes Safari and the iOS
Shortcuts touch) binds ``0.0.0.0`` on the next port up.

There is deliberately no credential on either plane. Nothing secret
crosses the bridge — screenshots and clipboard text are the agent
reporting its own activity, and calibration is routine rig work — so a
token would only add setup friction (hand-typing it into iOS Shortcut
URLs) without moving the threat model: the bind split keeps the
arm-driving surface off the LAN (structurally — see
``PHONE_PLANE_PATHS``), the data-injecting routes are gated (screenshot
upload: arming window + IP pin + size cap + magic bytes; touch
reports: phase gate — accepted only while a calibration step has a
visual on screen; see ``bridge/handler.py``), and hostile-Wi-Fi safety
is a transport problem (Tailscale/WireGuard), not a token problem.

FastMCP's own ``transport_security`` covers only the ``/mcp`` route —
``@mcp.custom_route`` routes are explicitly exempt from SDK auth. So
instead of ``mcp.run()`` we take ``streamable_http_app()`` (a plain
Starlette app) and wrap it in ``ControlGate``, which validates the Host
header on *all* control routes: a DNS-rebinding page reaches 127.0.0.1
but presents the attacker's hostname, so it is refused before routing.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from physiclaw.common.config import WILDCARD_HOSTS

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

log = logging.getLogger(__name__)

# Host values a local browser/client legitimately sends to a loopback bind.
_LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

# The complete set of LAN-exposed paths — the phone page's XHRs and the
# two iOS Shortcut endpoints, nothing else. ``PhoneApp.custom_route``
# REFUSES any path not listed here, so a copy-paste error that would
# put a control route (arm-driving, setup, calibrate) on the LAN plane
# fails at assembly time instead of silently exposing it. Adding a
# phone route is a deliberate two-step: register it AND list it here.
PHONE_PLANE_PATHS = frozenset(
    {
        "/bridge",
        "/api/bridge/state",
        "/api/bridge/tapped",
        "/api/bridge/screen-dimension",
        "/api/bridge/screenshot",
        "/api/bridge/clipboard",
        "/api/bridge/touch",
    }
)


class PhoneApp:
    """Route collector with FastMCP's ``custom_route`` decorator shape.

    Lets ``register_phone`` in each feature module read identically to its
    control-plane sibling; ``build()`` turns the collected routes into the
    LAN-facing Starlette app. Registration is gated by
    ``PHONE_PLANE_PATHS`` — see that constant for why.
    """

    def __init__(self) -> None:
        self._routes: list[Route] = []

    def custom_route(self, path: str, methods: list[str]) -> Callable:
        if path not in PHONE_PLANE_PATHS:
            raise RuntimeError(
                f"refusing to expose {path!r} on the LAN phone plane — "
                "phone routes must be listed in planes.PHONE_PLANE_PATHS"
            )

        def deco(fn: Callable) -> Callable:
            self._routes.append(Route(path, endpoint=fn, methods=methods))
            return fn

        return deco

    def build(self) -> Starlette:
        return Starlette(routes=self._routes)


def _host_name(scope: Scope) -> str:
    """Hostname from the Host header, port stripped (IPv6-bracket aware)."""
    raw = ""
    for k, v in scope.get("headers", []):
        if k == b"host":
            raw = v.decode("latin-1")
            break
    if raw.startswith("["):
        return raw.split("]", 1)[0] + "]"
    return raw.rsplit(":", 1)[0]


class ControlGate:
    """ASGI wrapper for the loopback plane: Host-header allowlist.

    ``host`` is the bind address — admitted as a Host value so a user
    exposing the control plane via a non-default ``--host`` still passes.
    """

    def __init__(self, app: ASGIApp, host: str = "127.0.0.1") -> None:
        self.app = app
        # A wildcard bind is the user's explicit opt-in to LAN exposure:
        # clients then send Host: <whichever interface IP they dialed>,
        # which can't be enumerated here — so the gate (a defense for the
        # loopback bind's rebinding hole) stands down instead of 421-ing
        # every real client. Concrete-IP binds keep the allowlist.
        self._wildcard = host in WILDCARD_HOSTS
        self._allowed = _LOCAL_HOSTNAMES | {host}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if not self._wildcard and _host_name(scope) not in self._allowed:
            reject = JSONResponse({"error": "unrecognized Host"}, status_code=421)
            return await reject(scope, receive, send)
        return await self.app(scope, receive, send)


def build_control_app(mcp: "FastMCP", host: str = "127.0.0.1") -> ControlGate:
    """The loopback ASGI app: FastMCP's Starlette app (``/mcp`` + every
    ``custom_route``) behind the ControlGate."""
    return ControlGate(mcp.streamable_http_app(), host)


def build_bridge_app(phone_app: PhoneApp) -> Starlette:
    """The LAN ASGI app: just the collected phone routes."""
    return phone_app.build()
