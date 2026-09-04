"""The walk's record — the two files a terminal moment writes.

One `runs.jsonl` line per walk (`walklog`, the escalation KPI) at the
FIRST terminal moment, whatever fires later — so a suspension whose
end_session is then blocked stays recorded as suspended — and the
daily-log lines the agent reads at wake (`common.daylog`): a
suspension, a fired payment, a walk cut short. A dry walk (the replay,
the boot, a stepping checkpoint) records nothing and keeps only the
outcome latch. Fail-open: a write failure logs and the walk goes on.
"""

import logging
from dataclasses import dataclass

from physiclaw.common import daylog
from physiclaw.conductor.walk import walklog
from physiclaw.conductor.walk.walklog import Outcome

log = logging.getLogger(__name__)


@dataclass
class Record:
    app: str
    playbook: str
    dry: bool
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
        if self.dry:
            return
        walklog.record(
            app=self.app,
            playbook=self.playbook,
            outcome=outcome,
            idx=idx,
            nodes=nodes,
            node=node,
            reason=reason,
            micros=micros,
            rescues=rescues,
            values=values,
            total=total,
        )

    def day(self, entry: str) -> None:
        """One daily-log line in the agent's own convention, stamped —
        the walk's activity lands in the record the model reads at wake."""
        if self.dry:
            return
        try:
            daylog.append_log(daylog.stamped(entry))
        except Exception:
            log.warning("conductor daily-log write failed", exc_info=True)
