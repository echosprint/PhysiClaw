"""Tests for the session-outcome contract (`runtime.contract.drive`).

The attempt callable is faked throughout — these tests own the policy
layer only: retry selection, the setup restart, the jobless-WAIT
follow-up, and backoff. Engine-side reporting (which Session fields
feed which SessionOutcome fields) is covered by the engine and claude
suites.
"""

from __future__ import annotations

import datetime as dt
import logging

import pytest
from freezegun import freeze_time

from physiclaw.agent.runtime import contract
from physiclaw.agent.runtime.contract import SessionOutcome
from physiclaw.agent.runtime.hook import Trigger
from physiclaw.agent.runtime.sentinel import DONE, IDLE, STUCK, WAIT

_TRIGGERS = [Trigger(description="t", source="phone")]


def _attempts(*outcomes: SessionOutcome):
    """An attempt fake yielding `outcomes` in order, recording its calls."""
    seq = iter(outcomes)
    calls: list[int] = []

    async def attempt(triggers, n):
        calls.append(n)
        return next(seq)

    return attempt, calls


def _no_upsert(mocker):
    """Stub the jobs upsert; returns the mock for call assertions."""
    return mocker.patch("physiclaw.agent.engine.jobs.upsert_auto_wait_check")


# ---------- retry selection ----------


@pytest.mark.asyncio
async def test_final_status_returns_after_one_attempt(mocker) -> None:
    _no_upsert(mocker)
    attempt, calls = _attempts(SessionOutcome(status=DONE))

    out = await contract.drive(
        attempt, _TRIGGERS, max_attempts=5, wait_default_minutes=15, retry_on=(STUCK,)
    )

    assert calls == [1]
    assert out.status == DONE


@pytest.mark.asyncio
async def test_retry_on_stuck_until_final(mocker) -> None:
    _no_upsert(mocker)
    attempt, calls = _attempts(
        SessionOutcome(status=STUCK, recap="a"),
        SessionOutcome(status=STUCK, recap="b"),
        SessionOutcome(status=DONE),
    )

    out = await contract.drive(
        attempt, _TRIGGERS, max_attempts=3, wait_default_minutes=15, retry_on=(STUCK,)
    )

    assert calls == [1, 2, 3]
    assert out.status == DONE


@pytest.mark.asyncio
async def test_stuck_exhaustion_returns_last_outcome(mocker) -> None:
    _no_upsert(mocker)
    attempt, calls = _attempts(
        SessionOutcome(status=STUCK, recap="x"),
        SessionOutcome(status=STUCK, recap="y"),
    )

    out = await contract.drive(
        attempt, _TRIGGERS, max_attempts=2, wait_default_minutes=15, retry_on=(STUCK,)
    )

    assert calls == [1, 2]
    assert out.status == STUCK
    assert out.recap == "y"


@pytest.mark.asyncio
async def test_undone_is_final_when_not_in_retry_on(mocker) -> None:
    # Engine semantics: retry_on=(STUCK,) — a None status ends the wake.
    _no_upsert(mocker)
    attempt, calls = _attempts(SessionOutcome(status=None))

    out = await contract.drive(
        attempt, _TRIGGERS, max_attempts=3, wait_default_minutes=15, retry_on=(STUCK,)
    )

    assert calls == [1]
    assert out.status is None


@pytest.mark.asyncio
async def test_stuck_is_final_when_retry_on_undone_only(mocker) -> None:
    # Claude semantics: retry_on=(None,) — a model-declared STUCK stands.
    _no_upsert(mocker)
    attempt, calls = _attempts(SessionOutcome(status=STUCK))

    out = await contract.drive(
        attempt, _TRIGGERS, max_attempts=3, wait_default_minutes=15, retry_on=(None,)
    )

    assert calls == [1]
    assert out.status == STUCK


@pytest.mark.asyncio
async def test_undone_exhaustion_logs_give_up(
    mocker, caplog: pytest.LogCaptureFixture
) -> None:
    _no_upsert(mocker)
    attempt, calls = _attempts(SessionOutcome(status=None), SessionOutcome(status=None))

    with caplog.at_level(logging.ERROR, logger="physiclaw.agent.runtime.contract"):
        await contract.drive(
            attempt,
            _TRIGGERS,
            max_attempts=2,
            wait_default_minutes=15,
            retry_on=(None,),
        )

    assert calls == [1, 2]
    assert any("giving up after 2 UNDONE" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_non_retryable_outcome_stops_despite_retry_on(mocker) -> None:
    # e.g. the engine's wall-clock budget exhaustion: STUCK, but final.
    _no_upsert(mocker)
    attempt, calls = _attempts(SessionOutcome(status=STUCK, retryable=False))

    out = await contract.drive(
        attempt, _TRIGGERS, max_attempts=3, wait_default_minutes=15, retry_on=(STUCK,)
    )

    assert calls == [1]
    assert out.status == STUCK


@pytest.mark.asyncio
async def test_backoff_sleeps_between_retries_only(mocker) -> None:
    _no_upsert(mocker)
    sleep = mocker.patch("asyncio.sleep")
    attempt, _ = _attempts(
        SessionOutcome(status=None),
        SessionOutcome(status=None),
        SessionOutcome(status=DONE),
    )

    await contract.drive(
        attempt,
        _TRIGGERS,
        max_attempts=3,
        wait_default_minutes=15,
        retry_on=(None,),
        retry_backoff_seconds=5.0,
    )

    # Between attempts 1→2 and 2→3, never before the first.
    assert sleep.call_count == 2


@pytest.mark.asyncio
async def test_attempt_exception_propagates(mocker) -> None:
    # Containment is the attempt's own business (engine catches, claude
    # lets construction errors escape) — drive must not swallow.
    _no_upsert(mocker)
    calls: list[int] = []

    async def attempt(triggers, n):
        calls.append(n)
        raise RuntimeError("can't fork")

    with pytest.raises(RuntimeError, match="can't fork"):
        await contract.drive(
            attempt,
            _TRIGGERS,
            max_attempts=3,
            wait_default_minutes=15,
            retry_on=(STUCK,),
        )
    assert calls == [1]


# ---------- setup restart ----------


@pytest.mark.asyncio
async def test_setup_restart_reruns_without_consuming_attempt(mocker) -> None:
    _no_upsert(mocker)
    attempt, calls = _attempts(
        SessionOutcome(status=None, restart_requested=True),
        SessionOutcome(status=DONE),
    )

    out = await contract.drive(
        attempt, _TRIGGERS, max_attempts=1, wait_default_minutes=15, retry_on=(STUCK,)
    )

    assert calls == [1, 1]  # restart re-runs the same attempt number
    assert out.status == DONE


@pytest.mark.asyncio
async def test_setup_restart_skipped_on_first_run_only_wake(mocker) -> None:
    # A synthetic first-run wake has no request to resume: the layout
    # loads on the next real wake instead of replaying the stale trigger.
    _no_upsert(mocker)
    attempt, calls = _attempts(SessionOutcome(status=IDLE, restart_requested=True))

    out = await contract.drive(
        attempt,
        [Trigger(description="learn the layout", source="first-run")],
        max_attempts=3,
        wait_default_minutes=15,
        retry_on=(STUCK,),
    )

    assert calls == [1]
    assert out.status == IDLE


@pytest.mark.asyncio
async def test_setup_restart_fires_at_most_once(mocker) -> None:
    _no_upsert(mocker)
    attempt, calls = _attempts(
        SessionOutcome(status=DONE, restart_requested=True),
        SessionOutcome(status=DONE, restart_requested=True),  # pathological
    )

    await contract.drive(
        attempt, _TRIGGERS, max_attempts=3, wait_default_minutes=15, retry_on=(STUCK,)
    )

    assert calls == [1, 1]  # second restart request ignored — DONE is final


# ---------- WAIT follow-up ----------


@pytest.mark.asyncio
async def test_wait_without_created_job_schedules_followup(mocker) -> None:
    upsert = _no_upsert(mocker)
    attempt, _ = _attempts(SessionOutcome(status=WAIT))

    with freeze_time("2026-04-28T14:00:00"):
        await contract.drive(
            attempt,
            _TRIGGERS,
            max_attempts=3,
            wait_default_minutes=15,
            retry_on=(STUCK,),
        )

    upsert.assert_called_once()
    target = upsert.call_args.args[0]
    assert isinstance(target, dt.datetime)
    # Target time = now + wait_default_minutes.
    assert (target.hour, target.minute) == (14, 15)


@pytest.mark.asyncio
async def test_wait_with_created_job_skips_followup(mocker) -> None:
    upsert = _no_upsert(mocker)
    attempt, _ = _attempts(SessionOutcome(status=WAIT, created_job=True))

    await contract.drive(
        attempt, _TRIGGERS, max_attempts=3, wait_default_minutes=15, retry_on=(STUCK,)
    )

    upsert.assert_not_called()


@pytest.mark.asyncio
async def test_wait_followup_failure_is_swallowed(
    mocker, caplog: pytest.LogCaptureFixture
) -> None:
    mocker.patch(
        "physiclaw.agent.engine.jobs.upsert_auto_wait_check",
        side_effect=OSError("disk full"),
    )
    attempt, _ = _attempts(SessionOutcome(status=WAIT))

    with caplog.at_level(logging.ERROR, logger="physiclaw.agent.runtime.contract"):
        out = await contract.drive(
            attempt,
            _TRIGGERS,
            max_attempts=3,
            wait_default_minutes=15,
            retry_on=(STUCK,),
        )

    assert out.status == WAIT  # the outcome stands
    assert any("failed to auto-schedule" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_done_never_schedules_followup(mocker) -> None:
    upsert = _no_upsert(mocker)
    attempt, _ = _attempts(SessionOutcome(status=DONE))

    await contract.drive(
        attempt, _TRIGGERS, max_attempts=3, wait_default_minutes=15, retry_on=(STUCK,)
    )

    upsert.assert_not_called()
