"""Sentinel line parsing — shared between the `claude -p` runtime and the engine.

Both agent runtimes close each session with a sentinel line of the form
`>> DONE - <recap>`. `parse_sentinel` recovers the status word + recap from
the assistant's final text; `STATUSES` is the canonical set.
"""

import re

DONE = "DONE"
STUCK = "STUCK"
FAIL = "FAIL"
IDLE = "IDLE"
WAIT = "WAIT"
STATUSES: frozenset[str] = frozenset({DONE, STUCK, FAIL, IDLE, WAIT})

_SENTINEL_RE = re.compile(
    r">+\s*(DONE|STUCK|FAIL|IDLE|WAIT)\b\s*-?\s*(.*)", re.IGNORECASE
)


def parse_sentinel(text: str | None) -> tuple[str | None, str]:
    """Return (status, recap) parsed from `text`, or (None, original) if no match.

    Accepts `>> DONE - recap`, bare `DONE`, lowercase, missing hyphen.
    """
    if not text:
        return None, ""
    m = _SENTINEL_RE.search(text)
    if m:
        return m.group(1).upper(), m.group(2).strip()
    word = text.strip().upper()
    if word in STATUSES:
        return word, ""
    return None, text.strip()
