"""Per-step forensic log for macro runs — the debugging trail.

Every run (engine and CLI rehearsal alike — emission lives in
`runner.run_and_record`, so callers can't diverge) gets its own directory
named by its id, the engine-session layout in miniature:

    log/macros/
    └── macro-run-<hex6>/
        ├── events.jsonl     start / one line per step / end
        └── images/          each step's resulting view, one JPEG per step

The id joins the surfaces: it rides in the result header the agent sees
(so engine ``events.jsonl`` records it), in ``stats.json``'s
``last_abort``, and in the CLI's post-run output — ``physiclaw macros
runs <hex6>`` renders the trail.

Event lines share ``ts``/``run``/``macro``/``event``:

    start  caller ("engine"|"cli"), session (live sid or null), inputs
    step   i, tool, name, outcome (ok|skipped|guard_failed|expect_failed
           |tool_error|timeout),
           args (post-substitution), verdict, guard_polls, ms, image
           (filename under images/); on a guard failure also `detail` +
           `screen_text` (what the OCR haystack actually contained — the
           number-one debug question)
    end    ok, aborted_step, reason, detail, ms

Fail-open everywhere: a logging failure must never abort a physical run.
Run dirs are purged on the artifact window (`retention.trace_days`, like
engine session dirs) at RunLogger construction.
"""

import json
import logging
import secrets
import time
from pathlib import Path
from typing import Any

from physiclaw.common import paths
from physiclaw.common.config import CONFIG
from physiclaw.common.logger import iso_now, save_image
from physiclaw.common.logger.retention import purge_old_sessions
from physiclaw.common.text import append_text, clip

log = logging.getLogger(__name__)

RUN_ID_PREFIX = "macro-run-"
# Retention is swept once per process, like `trace.RawLog` and the claude
# session log. It used to run per RUN, and `purge_old_sessions` stats every
# file under every retained run dir (~23 per run) — so the cost grew with
# usage and blocked the event loop before each macro.
_purged = False
_TRUNCATE_INPUT = 120
_TRUNCATE_SCREEN = 500


def new_run_id() -> str:
    return RUN_ID_PREFIX + secrets.token_hex(3)


def run_dir(run_id: str) -> Path:
    return paths.macros_log_dir() / run_id


class RunLogger:
    """One macro run's event emitter. Construction mints the run id,
    creates the run dir, and purges old run dirs (the bootstrap-moment
    convention of the other log writers); every method is fail-open."""

    def __init__(self, macro: str, caller: str):
        self.run_id = new_run_id()
        self.macro = macro
        self.caller = caller
        self.dir = run_dir(self.run_id)
        self._t0 = time.monotonic()
        global _purged
        try:
            (self.dir / "images").mkdir(parents=True, exist_ok=True)
            if not _purged:
                _purged = True
                purge_old_sessions(
                    paths.macros_log_dir(), days=CONFIG.retention.trace_days
                )
        except OSError:
            log.warning("macros run log dir unavailable", exc_info=True)

    def start(self, inputs: dict[str, str], start_at: str = "") -> None:
        self._emit(
            "start",
            caller=self.caller,
            session=paths.live_session_id(),
            inputs={k: clip(v, _TRUNCATE_INPUT) for k, v in inputs.items()},
            start_at=start_at,
        )

    def step(
        self,
        i: int,
        tool: str,
        name: str,
        outcome: str,
        *,
        args: dict[str, Any] | None = None,
        verdict: bool | None = None,
        guard_polls: int = 0,
        ms: int = 0,
        detail: str = "",
        screen_text: str = "",
        view: list[dict] | None = None,
    ) -> None:
        fields: dict[str, Any] = {
            "i": i,
            "tool": tool,
            "name": name,
            "outcome": outcome,
            "args": args or {},
            "verdict": _verdict_str(verdict),
            "guard_polls": guard_polls,
            "ms": ms,
        }
        if detail:
            fields["detail"] = detail
        if screen_text:
            fields["screen_text"] = clip(screen_text, _TRUNCATE_SCREEN)
        image = self._save_view(i, view) if view else None
        if image:
            fields["image"] = image
        self._emit("step", **fields)

    def end(
        self,
        *,
        ok: bool,
        aborted_step: int | None = None,
        reason: str | None = None,
        detail: str = "",
    ) -> None:
        self._emit(
            "end",
            ok=ok,
            aborted_step=aborted_step,
            reason=reason,
            detail=detail,
            ms=int((time.monotonic() - self._t0) * 1000),
        )

    def _save_view(self, i: int, view: list[dict]) -> str | None:
        """The step's first image block → images/, named by step number.
        Pixel-level 'what did the camera see at this step' for later
        debugging; fail-open like everything here."""
        block = next((b for b in view if b.get("type") == "image"), None)
        if block is None:
            return None
        try:
            return save_image(
                self.dir / "images",
                i,
                block.get("mime_type") or "image/jpeg",
                block.get("data") or "",
            )
        except OSError:
            log.warning("macros run image write failed", exc_info=True)
            return None

    def _emit(self, event: str, **fields: Any) -> None:
        line = {
            "ts": iso_now(),
            "run": self.run_id,
            "macro": self.macro,
            "event": event,
            **fields,
        }
        try:
            append_text(
                self.dir / "events.jsonl",
                json.dumps(line, ensure_ascii=False) + "\n",
            )
        except OSError:
            log.warning("macros run log write failed", exc_info=True)


def _verdict_str(changed: bool | None) -> str | None:
    if changed is None:
        return None
    return "changed" if changed else "unchanged"
