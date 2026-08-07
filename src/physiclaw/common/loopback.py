"""Loopback identity — every spelling of "this very host", in one place.

Three surfaces need to recognize loopback and each speaks a different
dialect; defining the forms side by side keeps a new spelling (or a
dropped one) a single reviewed edit instead of three drifting sets:

  LOOPBACK_HOSTS         hostname spellings clients dial — the NO_PROXY
                         entries (`common.proxy`)
  LOOPBACK_IPS           socket peer addresses — the bridge's
                         trust-on-first-use exemption; a peer address
                         is never a hostname
  LOOPBACK_HOST_HEADERS  what a local browser/client sends as the Host
                         header — the control plane's allowlist; HTTP
                         brackets a bare IPv6 literal

Two of the three are security allowlists (DNS-rebinding gate, TOFU
exemption): derive narrower forms from the tuple, never widen a site
to "share a constant".
"""

# In NO_PROXY append order (pinned by tests). `[server] host = "::1"`
# is a supported bind, so IPv6 is not optional.
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")

LOOPBACK_IPS = frozenset(LOOPBACK_HOSTS) - {"localhost"}

LOOPBACK_HOST_HEADERS = frozenset(LOOPBACK_HOSTS) | {"[::1]"}
