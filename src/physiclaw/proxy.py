"""Normalize proxy env vars before any HTTP client reads them.

Linux desktop tooling (GNOME's proxy settings, Clash Verge, v2rayA…)
exports ``all_proxy=socks://host:port`` — bare ``socks``, meaning
SOCKS5. httpx rejects that scheme outright (``ValueError: Unknown
scheme for proxy URL``) when it constructs a ``trust_env=True`` client,
which killed every provider call the moment such a VPN was on. httpx
only speaks ``socks5://``, so rewrite the scheme in place; the
``socksio`` dependency (httpx's ``[socks]`` extra) makes the rewritten
proxy actually usable.

Called from the two process entry points — the CLI root callback and
the runtime launcher (which covers a direct ``python -m
physiclaw.agent.runtime``) — before anything builds a client, so every
env-honouring httpx client (provider clients, the Anthropic SDK's
internal one, the loopback poll/MCP clients on Linux/macOS) sees only
schemes it understands.
Schemeless values (``127.0.0.1:7897``) need no help: httpx already
assumes ``http://`` for those. The urllib paths (CLI downloads, PyPI
update check) ignore ``all_proxy`` entirely, so the rewrite is
invisible to them.
"""

import os

_PROXY_VARS = ("all_proxy", "http_proxy", "https_proxy")


def normalize_proxy_env() -> None:
    """Rewrite ``socks://`` proxy schemes to ``socks5://`` in-place."""
    for key, value in os.environ.items():
        if key.lower() not in _PROXY_VARS:
            continue
        scheme, sep, rest = value.partition("://")
        if sep and scheme.lower() == "socks":
            os.environ[key] = f"socks5://{rest}"
