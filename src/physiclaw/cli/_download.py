"""Shared HTTP download helpers for the CLI — a UA-pinned fetch and a
chunked reader that draws a progress bar (or streams quietly). Used by every
CLI download (vision model, firmware, official skills) so the fetch behaviour
and UX stay uniform.
"""

import urllib.request

import typer

# Cloudflare's WAF 403s the default Python-urllib User-Agent, so every request
# to the physiclaw.ai mirror must set one. Applied to all CLI downloads for
# uniformity (and so flash.py's firmware fetch keeps working too).
USER_AGENT = "physiclaw"


def http_get(url: str, timeout: int = 120):
    """urlopen with a User-Agent set — the CDN's WAF blocks the default one."""
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": USER_AGENT}),
        timeout=timeout,
    )


def stream(resp, write, label: str, *, progress: bool = True) -> None:
    """Read ``resp`` in 64 KiB chunks into ``write``, drawing a progress bar
    sized from Content-Length (quietly streams if the length is unknown, or if
    ``progress=False`` — for small downloads where only the result matters)."""
    raw = resp.getheader("Content-Length")
    total = int(raw) if raw and raw.isdigit() else None
    if not progress or total is None:
        for chunk in iter(lambda: resp.read(1 << 16), b""):
            write(chunk)
        return
    # width=0 auto-fits the bar to the terminal. The default fixed width (36)
    # plus the label and percent renders an ~80-column line, which wraps on a
    # default-width window (iTerm2, Terminal); once wrapped, the redraw's \r
    # only returns to the start of the wrapped row, so every update spills onto
    # a new line instead of overwriting in place.
    with typer.progressbar(
        length=total, label=f"{label} ({total / 1048576:.1f} MiB)", width=0
    ) as bar:
        for chunk in iter(lambda: resp.read(1 << 16), b""):
            write(chunk)
            bar.update(len(chunk))
