"""Session-outcome contract — engine-neutral close policy for one wake.

Both engines (the in-process tool-call loop in ``agent/engine/`` and the
``claude -p`` subprocess in ``agent/claude/``) speak the same outcome
vocabulary (``runtime.sentinel``): DONE / STUCK / FAIL / IDLE / WAIT, or
no clean close at all. What must HAPPEN on each outcome — which ones get
a fresh attempt, what a jobless WAIT schedules, when a first-run setup
session earns a restart — is product policy, not engine mechanics. It
lives here, once, so an engine cannot silently diverge from it (the
claude path shipped without the WAIT follow-up for exactly that reason:
the doctrine promised it, but the policy was buried in the other
engine's retry loop).

``drive()`` owns the attempt loop; an engine shrinks to "run one
attempt, report a ``SessionOutcome``". Attempts keep their own error
containment — the engine converts crashes to STUCK internally, the
claude path lets construction errors propagate (a broken spawn is an
operator problem, not a retry case) — so ``drive()`` never catches.

One divergence is deliberate, not drift: the in-process engine runs
turn plugins (`[agent] plugins` — the conductor's armed playbooks, and
the overture's boot to the user's thread at wake); the claude path runs
none. A plugin's payoff is eliminating metered provider calls, and that
path is subscription-metered — so it keeps the doctrine-in-prose
version, executed by the model.

Retry semantics stay per-engine via ``retry_on``: the engine retries
STUCK (its harness-detected dead ends — max turns, provider exhausted,
crash — are worth a fresh session), while the claude path retries only
UNDONE (``None``): a model-declared STUCK there is a considered close,
and a re-spawn would replay the same doctrine against the same blocker.
"""

import asyncio
import datetime as dt
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Collection

from physiclaw.agent.runtime.hook import Trigger
from physiclaw.agent.runtime.sentinel import DONE, WAIT

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SessionOutcome:
    """What one session attempt reports back to ``drive()``.

    ``status`` is a ``runtime.sentinel`` status, or ``None`` when the
    session produced no clean close (crash, kill, timeout — the claude
    path's "UNDONE"). ``created_job`` matters only on WAIT: True when
    the session scheduled its own resume job, which suppresses the
    generic follow-up. ``restart_requested`` is the engine's one-shot
    first-run-setup restart. ``retryable=False`` marks an outcome that
    must not burn further attempts even when ``status`` is in
    ``retry_on`` (e.g. a wall-clock budget exhaustion — a slow
    environment would eat every retry the same way).
    """

    status: str | None = None
    recap: str = ""
    created_job: bool = False
    restart_requested: bool = False
    retryable: bool = True


def productive(outcome: SessionOutcome) -> bool:
    """Close-policy predicate for the runtime's wake backoff: DONE and
    WAIT are earned closes (the task moved, or a resume is scheduled),
    and a requested restart is progress by definition. Everything else —
    STUCK, FAIL, IDLE, no clean close — is an unproductive session the
    wake cadence should back off from. Lives HERE with the rest of the
    close policy, so the runtime loop cannot silently diverge from it."""
    return outcome.status in (DONE, WAIT) or outcome.restart_requested


# One attempt: (triggers, attempt_number) → outcome. The attempt number
# is 1-based and only for the engine's own log lines.
Attempt = Callable[[list[Trigger], int], Awaitable[SessionOutcome]]


async def drive(
    attempt: Attempt,
    triggers: list[Trigger],
    *,
    max_attempts: int,
    wait_default_minutes: int,
    retry_on: Collection[str | None],
    retry_backoff_seconds: float = 0.0,
) -> SessionOutcome:
    """Run ``attempt`` until a final outcome; enforce the close policy.

    - An outcome whose status is in ``retry_on`` (and is ``retryable``)
      gets a fresh attempt, up to ``max_attempts`` total, spaced by
      ``retry_backoff_seconds``. Anything else is final on first
      occurrence. ``retry_on`` has no default on purpose — which
      statuses earn a fresh attempt is the one genuinely per-engine
      policy (see module docstring), so every caller states it.
    - One extra restart is allowed when an attempt completes first-run
      screen-layout setup: the layout only reaches the SYSTEM prompt on
      a fresh render, so the same triggers re-run once with it loaded.
      It doesn't consume an attempt, fires at most once, and is skipped
      when the wake carries nothing but the synthetic first-run trigger
      (no request to resume — the layout loads on the next real wake).
    - A final WAIT with no session-created resume job auto-schedules
      the singleton follow-up job ``wait_default_minutes`` out, so a
      WAIT close can never strand the loop with nothing to wake it.

    Exceptions from ``attempt`` propagate — containment is the
    attempt's own business (see module docstring).
    """
    setup_restart_used = False
    outcome = SessionOutcome()
    n = 0
    while n < max_attempts:
        n += 1
        outcome = await attempt(triggers, n)
        if outcome.restart_requested and not setup_restart_used:
            setup_restart_used = True
            if any(t.source != "first-run" for t in triggers):
                n -= 1  # a setup restart isn't a retry
                log.info("screen layout learned during setup — restarting to load it")
                continue
            log.info("screen layout learned on first-run wake — no request to resume")
        if outcome.status not in retry_on or not outcome.retryable:
            break
        if n < max_attempts:
            log.warning(
                "session %s (attempt %d/%d): %r — retrying",
                outcome.status or "UNDONE",
                n,
                max_attempts,
                outcome.recap,
            )
            if retry_backoff_seconds > 0:
                await asyncio.sleep(retry_backoff_seconds)
    else:
        # Attempts exhausted without a final close. STUCK exhaustion
        # already logged its per-retry warnings; a silent UNDONE would
        # look like a wake that never happened.
        if outcome.status is None:
            log.error("giving up after %d UNDONE attempts", max_attempts)
    if outcome.status == WAIT and not outcome.created_job:
        log.warning(
            "WAIT with no create_job — auto-scheduling %d-min follow-up",
            wait_default_minutes,
        )
        _schedule_wait_followup(minutes=wait_default_minutes)
    return outcome


def _schedule_wait_followup(*, minutes: int) -> None:
    """Upsert the singleton auto-WAIT-check job to fire in ``minutes``.

    One canonical job id across sessions (``jobs.upsert_auto_wait_check``)
    so jobs.md doesn't grow one entry per WAIT close. Best effort — a
    scheduling failure is logged, never raised: the WAIT outcome stands
    either way. Imported lazily so this module stays importable without
    pulling the engine package's dependency tree.
    """
    from physiclaw.agent.engine import jobs

    at = dt.datetime.now() + dt.timedelta(minutes=minutes)
    try:
        jobs.upsert_auto_wait_check(at)
        log.info(
            "auto-WAIT follow-up scheduled at %s", at.isoformat(timespec="minutes")
        )
    except Exception:
        log.exception("failed to auto-schedule WAIT follow-up")
