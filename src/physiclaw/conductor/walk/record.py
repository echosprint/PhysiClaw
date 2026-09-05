"""The walk's record — what a terminal moment writes.

One `runs.jsonl` line per walk (`walklog`, the escalation KPI) at the
FIRST terminal moment, whatever fires later — so a suspension whose
end_session is then blocked stays recorded as suspended; one `walk`
event into the session's events.jsonl (the summary lists them, so a
session's playbook story is readable without the runtime log); and the
daily-log lines the agent reads at wake (`common.daylog`): a
suspension, a fired payment, a walk cut short. A dry walk (the replay,
the boot, a stepping checkpoint) writes no runs row and no daily-log
line but still its `walk` event when a session is listening — the boot
handing its baton on IS the session's story. Fail-open: a write
failure logs and the walk goes on.
"""

import logging
from dataclasses import dataclass
from typing import Any

from physiclaw.common import daylog
from physiclaw.conductor.walk import walklog
from physiclaw.conductor.walk.walklog import Outcome
from physiclaw.contract.plugin import EventSink

log = logging.getLogger(__name__)


@dataclass
class Record:
    app: str
    playbook: str
    dry: bool
    # The session's event stream (`Trace.write`-shaped); None outside a
    # wake (a rehearsal, a replay, a stepping tool).
    events: EventSink | None = None
    # The walk's recorded outcome, None while it runs — set by the
    # first terminal moment and never again.
    outcome: Outcome | None = None

    def run(
        self,
        outcome: Outcome,
        *,
        idx: int,
        nodes: int,
        node: str | None,
        reason: str = "",
        micros: int = 0,
        rescues: int = 0,
        values: dict | None = None,
        total: float | None = None,
    ) -> None:
        """The walk's one runs.jsonl line — first terminal moment wins.
        The keyword fields are the walk's position and counts
        (`walklog.record`'s own)."""
        if self.outcome is not None:
            return
        self.outcome = outcome
        # One set of fields, two readers: the session's event and the
        # runs row (which adds its stamp and the input values).
        fields: dict[str, Any] = dict(
            app=self.app,
            playbook=self.playbook,
            outcome=outcome,
            node=node,
            idx=idx,
            nodes=nodes,
            reason=reason,
            micros=micros,
            rescues=rescues,
            total=total,
        )
        if self.events is not None:
            try:
                self.events.write({"event": "walk", **fields})
            except Exception:
                log.warning("conductor walk event write failed", exc_info=True)
        if not self.dry:
            walklog.record(**fields, values=values)

    def day(self, entry: str) -> None:
        """One daily-log line in the agent's own convention, stamped —
        the walk's activity lands in the record the model reads at wake."""
        if self.dry:
            return
        try:
            daylog.append_log(daylog.stamped(entry))
        except Exception:
            log.warning("conductor daily-log write failed", exc_info=True)
