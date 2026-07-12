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
arm-driving surface off the LAN, the upload guards gate the one
data-injecting route (arming window, IP pin, size cap, magic bytes —
see ``bridge/handler.py``), and hostile-Wi-Fi safety is a transport
problem (Tailscale/WireGuard), not a token problem.

FastMCP's own ``transport_security`` covers only the ``/mcp`` route —
``@mcp.custom_route`` routes are explicitly exempt from SDK auth. So
instead of ``mcp.run()`` we take ``streamable_http_app()`` (a plain
Starlette app) and wrap it in ``ControlGate``, which validates the Host
header on *all* control routes: a DNS-rebinding page reaches 127.0.0.1
but presents the attacker's hostname, so it is refused before routing.
"""

import logging

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

log = logging.getLogger(__name__)

# Host values a local browser/client legitimately sends to a loopback bind.
_LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


class PhoneApp:
    """Route collector with FastMCP's ``custom_route`` decorator shape.

    Lets ``register_phone`` in each feature module read identically to its
    control-plane sibling; ``build()`` turns the collected routes into the
    LAN-facing Starlette app.
    """

    def __init__(self):
        self._routes: list[Route] = []

    def custom_route(self, path: str, methods: list[str]):
        def deco(fn):
            self._routes.append(Route(path, endpoint=fn, methods=methods))
            return fn

        return deco

    def build(self) -> Starlette:
        return Starlette(routes=self._routes)


def _host_name(scope) -> str:
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

    def __init__(self, app, host: str = "127.0.0.1"):
        self.app = app
        self._allowed = _LOCAL_HOSTNAMES | {host}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if _host_name(scope) not in self._allowed:
            reject = JSONResponse({"error": "unrecognized Host"}, status_code=421)
            return await reject(scope, receive, send)
        return await self.app(scope, receive, send)


def build_control_app(mcp, host: str = "127.0.0.1") -> ControlGate:
    """The loopback ASGI app: FastMCP's Starlette app (``/mcp`` + every
    ``custom_route``) behind the ControlGate."""
    return ControlGate(mcp.streamable_http_app(), host)


def build_bridge_app(phone_app: PhoneApp) -> Starlette:
    """The LAN ASGI app: just the collected phone routes."""
    return phone_app.build()
