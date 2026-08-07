"""Engine session logs — a daily human narrative + a per-session artifact dir.

`Trace` writes the per-day human log (through the shared
`DailyLogWriter`, the same one `agent/claude/session_log.py` uses, so
operators scan either runtime the same way) plus the session's
events.jsonl and summary.json; `RawLog` captures the provider
round-trips into wire.jsonl, extracting inline base64 images to
turn-tagged files. Everything lands under log/engine/sessions/<sid>/ —
self-contained (image refs are relative), so "share the bad session"
is one directory copy. The on-disk format is documented once, in
`store.SESSIONS_README` (shipped into the sessions dir and every
`physiclaw logs --save` zip); retention sweeps on session bootstrap
(`store._purge_old`).

One submodule per concern: `store` (artifact dirs, session ids,
retention, the shipped README), `format` (event/value rendering for
the human surfaces), `trace` (`Trace` + the summary accumulator),
`rawlog` (`RawLog`).
"""

from physiclaw.agent.trace.format import (
    brief,
    brief_args,
    brief_content,
    fmt_tokens,
    format_call_args,
    format_call_result,
    summarize_event,
)
from physiclaw.agent.trace.rawlog import RawLog
from physiclaw.agent.trace.store import SESSIONS_README, new_sid
from physiclaw.agent.trace.trace import Trace

__all__ = [
    "SESSIONS_README",
    "RawLog",
    "Trace",
    "brief",
    "brief_args",
    "brief_content",
    "fmt_tokens",
    "format_call_args",
    "format_call_result",
    "new_sid",
    "summarize_event",
]
