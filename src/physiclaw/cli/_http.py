"""One HTTP layer for the ``physiclaw`` CLI — two policies, one module.

LOCAL SERVER (loopback proxy policy, no User-Agent):
  ``api()`` — fail-soft JSON exchange with the local PhysiClaw server
  (None on any failure); ``fetch_json()`` — the raising flavor, for
  probes that surface the failure detail. Local-server only: both route
  through the loopback-tuned opener below.

PUBLIC DOWNLOADS (env proxy, User-Agent pinned):
  ``http_get()`` / ``stream()`` — used by every CLI download (vision
  model, firmware, official skills) so fetch behaviour and UX stay
  uniform.

Deliberate exception: ``_update_check``'s PyPI probe stays on raw
``urlopen`` — it needs env-proxy AND the default UA, a combination
neither shape here offers, and changing its wire behavior isn't worth
folding it in.
"""

import json
import urllib.error
import urllib.request

import typer

from physiclaw.common import platform

# Cloudflare's WAF 403s the default Python-urllib User-Agent, so every request
# to the physiclaw.ai mirror must set one. Applied to all CLI downloads for
# uniformity (and so flash.py's firmware fetch keeps working too).
USER_AGENT = "physiclaw"

# Trust the system proxy for loopback only on platforms where the bypass
# list reliably excludes localhost (see physiclaw.common.platform).
_OPENER = (
    urllib.request.build_opener()
    if platform.TRUST_PROXY_ENV
    else urllib.request.build_opener(urllib.request.ProxyHandler({}))
)


def api(base: str, method: str, path: str, body=None, timeout=60):
    """JSON request against the local PhysiClaw server at ``base``.

    Fail-soft: returns the parsed JSON reply, or None on any failure
    (connection refused, timeout, non-JSON body). An HTTP error's body
    is parsed too — the server sends structured error responses."""
    data = json.dumps(body).encode() if body else (b"" if method == "POST" else None)
    hdrs = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(base + path, data=data, method=method, headers=hdrs)
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:
            return None
    except Exception:
        return None


def fetch_json(url: str, timeout: float = 60):
    """GET a LOCAL-SERVER ``url`` through the same loopback-tuned opener
    as ``api`` and parse the JSON body. Raises (``OSError`` for
    transport, ``ValueError`` for bad JSON) — for probes that report the
    failure detail rather than failing soft. Not for internet URLs:
    the opener bypasses env proxies on most platforms and sets no
    User-Agent — use ``http_get`` for public fetches."""
    with _OPENER.open(urllib.request.Request(url), timeout=timeout) as r:
        return json.loads(r.read())


def http_get(url: str, timeout: int = 120):
    """urlopen with a User-Agent set — the CDN's WAF blocks the default one."""
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": USER_AGENT}),
        timeout=timeout,
    )


def stream(resp, write, label: str, *, progress: bool = True) -> None:
    """Read ``resp`` in 64 KiB chunks into ``write``, drawing a progress bar
    sized from Content-Length (quietly streams if the length is unknown, or if
    ``progress=False`` — for small downloads where only the result matters).

    Raises ``urllib.error.ContentTooShortError`` (a ``URLError``, so every
    caller's fallback path catches it) when the body ends short of the
    declared Content-Length: a CDN dropping the connection mid-transfer just
    makes ``read()`` return early with NO error, and the truncated bytes
    would otherwise flow on into decoding/extraction as if complete."""
    raw = resp.getheader("Content-Length")
    total = int(raw) if raw and raw.isdigit() else None
    received = 0
    if not progress or total is None:
        for chunk in iter(lambda: resp.read(1 << 16), b""):
            write(chunk)
            received += len(chunk)
    else:
        # width=0 auto-fits the bar to the terminal. The default fixed width
        # (36) plus the label and percent renders an ~80-column line, which
        # wraps on a default-width window (iTerm2, Terminal); once wrapped,
        # the redraw's \r only returns to the start of the wrapped row, so
        # every update spills onto a new line instead of overwriting in place.
        with typer.progressbar(
            length=total, label=f"{label} ({total / 1048576:.1f} MiB)", width=0
        ) as bar:
            for chunk in iter(lambda: resp.read(1 << 16), b""):
                write(chunk)
                received += len(chunk)
                bar.update(len(chunk))
    if total is not None and received < total:
        raise urllib.error.ContentTooShortError(
            f"retrieval incomplete: got only {received} out of {total} bytes",
            None,
        )
