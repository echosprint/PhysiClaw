"""UTF-8 by default for every text I/O.

``Path.read_text`` / ``write_text`` / ``open`` without an explicit
``encoding=`` fall back to ``locale.getencoding()`` — GBK on Chinese
Windows, cp1252 on Western Windows. That's bricked PhysiClaw twice
already (config.toml in 58aa00d, SKILL.md after). Route every short-
lived text I/O through these helpers so UTF-8 is the only default.

Long-lived append handles (daily log loops in agent/claude/spawn.py,
agent/trace, agent/engine/jobs.py) keep an explicit
``open(path, "a", encoding="utf-8")`` inline — wrapping a stateful
file handle in a helper would be more indirection than it's worth.
"""

import json
from pathlib import Path
from typing import Any


def json_span(text: str, opener: str, closer: str) -> Any | None:
    """Parse the outermost JSON span (`opener`..`closer`) out of an LLM
    reply — tolerant of ```json fences and surrounding prose. None on
    anything malformed; payload validation stays with the caller. The one
    home for the idiom (curate's list replies, the conductor's decision
    objects)."""
    start, end = text.find(opener), text.rfind(closer)
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, data: str) -> None:
    # newline="\n" pins LF: the default (newline=None) translates \n to
    # os.linesep — CRLF on Windows — breaking the "UTF-8 with LF on every
    # platform" invariant the config and session-log artifacts promise.
    path.write_text(data, encoding="utf-8", newline="\n")


def append_text(path: Path, data: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(data)


def clip(text: str, limit: int) -> str:
    """Truncate with a trailing ellipsis — the one spelling of the
    log-line trimmer (several modules used to hand-roll it)."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
