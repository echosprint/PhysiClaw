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

The second job is loopback exemption: with an env proxy set and no
``NO_PROXY``, every loopback client — the runtime's 1s status poll, the
phone watch, the MCP client — transits the proxy, and local proxies
answer 502 for loopback upstreams, so the runtime never sees ``ready``.
It is fixed HERE, in the env, and not by per-client ``trust_env``,
because one loopback client is not ours to construct: the claude-code
subprocess's MCP client dials our loopback URL with whatever env it
inherited (``spawn._child_env`` passes ``NO_PROXY`` through). The env
is the process-wide floor that children inherit; ``TRUST_PROXY_ENV``
stays the per-client belt for platforms whose proxy config is not
env-visible at all (Windows registry — and a darwin proxy set only in
Network Settings keeps relying on macOS's own bypass list).
"""

import os

_PROXY_VARS = ("all_proxy", "http_proxy", "https_proxy")
# Every spelling `config.url_host` can produce for a loopback server —
# `[server] host = "::1"` is a supported bind, so IPv6 is not optional.
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def normalize_proxy_env() -> None:
    """Rewrite ``socks://`` proxy schemes to ``socks5://`` and exempt
    loopback from proxying — both in-place, before any client reads
    the env."""
    for key, value in os.environ.items():
        if key.lower() in _PROXY_VARS:
            scheme, sep, rest = value.partition("://")
            if sep and scheme.lower() == "socks":
                os.environ[key] = f"socks5://{rest}"
    if any(v for k, v in os.environ.items() if k.lower() in _PROXY_VARS):
        _ensure_loopback_no_proxy()


def _ensure_loopback_no_proxy() -> None:
    current = os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or ""
    entries = [h.strip() for h in current.split(",")]
    missing = [h for h in _LOOPBACK_HOSTS if h not in entries]
    if missing:
        merged = ",".join(filter(None, [current, *missing]))
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = merged
