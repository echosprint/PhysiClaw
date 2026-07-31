"""Tests for `physiclaw.agent.engine.loop` — the turn driver (`drive`),
shape enforcement, the policy turn gates as exercised through the loop
(their judgment is engine-visible behavior), and the pure helpers
(`_chat_with_retry`, `_log_usage`, `_corrective_for_bad_shape`,
`log_external_stop`). Tool-call execution lives in `test_dispatch.py`;
session lifecycle in `test_engine.py`."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from engine_fakes import (
    FakeProvider,
    _asst,
    _mk_run,
    _registry,
    _schemas,
    _settings,
    _tc,
    patch_loop_tails,
)

from physiclaw.agent.engine import compact as compact_mod
from physiclaw.agent.engine import memory as memory_mod
from physiclaw.agent.engine.builtin_tool import LocalTool
from physiclaw.agent.engine.dto import (
    AssistantMessage,
    CollapsePolicy,
    FinishReason,
    SystemMessage,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from physiclaw.agent.engine.loop import (
    CORRECTIVE_LIMIT,
    _call_provider,
    _chat_with_retry,
    _corrective_for_bad_shape,
    _log_usage,
    drive,
    log_external_stop,
)
from physiclaw.agent.engine.runspec import EngineRun
from physiclaw.agent.engine.session import Session
from physiclaw.agent.engine.trace import Trace
from physiclaw.agent.provider.provider_base import (
    ProviderError,
    ProviderTransientError,
)
from physiclaw.agent.runtime.sentinel import DONE, FAIL, IDLE, STUCK

# ---------- drive ----------


@pytest.fixture
def patched_loop_deps(mocker) -> dict:
    """Stub out compact / scratchpad / plan tail injection so `drive` runs
    in isolation; returns the mocks keyed by target for spying."""
    return patch_loop_tails(mocker)


def _loop_run(provider, registry, **over) -> EngineRun:
    return _mk_run(
        provider=provider,
        tool_schemas=_schemas(registry),
        local_registry=registry,
        **over,
    )


@pytest.mark.asyncio
async def test_loop_closes_cleanly_on_end_session(patched_loop_deps) -> None:
    registry = _registry()

    asst = _asst(
        tool_calls=[
            _tc("note", {"summary": "closing"}),
            _tc("end_session", {"status": DONE, "recap": "all done"}),
        ],
        finish=FinishReason.TOOL_CALLS,
    )
    provider = FakeProvider([asst])

    session = Session()
    messages: list = [
        SystemMessage(content="sys"),
        UserMessage(content="trig"),
    ]

    await drive(_loop_run(provider, registry), session, messages)

    assert session.sentinel_status == DONE
    assert session.sentinel_recap == "all done"
    # Last messages: assistant + tool_result for note + tool_result for end.
    assert isinstance(messages[-3], AssistantMessage)
    assert isinstance(messages[-2], ToolResultMessage)
    assert isinstance(messages[-1], ToolResultMessage)


@pytest.mark.asyncio
async def test_loop_skips_layout_reminder_when_complete(patched_loop_deps) -> None:
    # Default (layout_incomplete=False) must NOT call screen_layout.inject_tail
    # — that avoids a per-turn disk read once setup is done.
    spy = patched_loop_deps["screen_layout"]

    registry = _registry()
    asst = _asst(
        tool_calls=[
            _tc("note", {"summary": "x"}),
            _tc("end_session", {"status": DONE, "recap": "done"}),
        ]
    )
    session = Session()

    await drive(
        _loop_run(FakeProvider([asst]), registry),  # layout_incomplete defaults False
        session,
        [SystemMessage(content="s")],
    )
    spy.assert_not_called()

    # With layout_incomplete=True it IS injected each turn.
    session2 = Session()
    await drive(
        _loop_run(FakeProvider([asst]), registry, layout_incomplete=True),
        session2,
        [SystemMessage(content="s")],
    )
    spy.assert_called()


@pytest.mark.asyncio
async def test_loop_stucks_after_consecutive_correctives(patched_loop_deps) -> None:
    # A model that never holds the [note, one-other] shape must end
    # STUCK after CORRECTIVE_LIMIT consecutive correctives instead of
    # burning max_turns of round-trips.
    registry = _registry()

    bad = [_asst(tool_calls=[_tc("peek")]) for _ in range(CORRECTIVE_LIMIT + 3)]
    provider = FakeProvider(bad)
    session = Session()

    await drive(
        _loop_run(provider, registry),
        session,
        [SystemMessage(content="sys"), UserMessage(content="trig")],
    )

    assert session.sentinel_status == STUCK
    assert "malformed turns" in session.sentinel_recap
    assert len(provider.calls) == CORRECTIVE_LIMIT


# ---------- pre-compression checkpoint ----------


class CheckpointProvider(FakeProvider):
    """Collapse knobs small enough to hit the checkpoint turn in a short
    scripted session; records every request for tail assertions."""

    COLLAPSE = CollapsePolicy(first_at=2, keep=1, interval=100)

    def __init__(self, responses):
        super().__init__(responses)
        self.requests: list[list] = []

    async def chat(self, messages, tools):
        self.requests.append(list(messages))
        return await super().chat(messages, tools)


def _scaffold_messages() -> list:
    """[system, trigger] + the three compaction slots, as bootstrap builds."""
    return [
        SystemMessage(content="sys"),
        UserMessage(content="trig"),
        compact_mod.new_summary_placeholder(),
        UserMessage(content=compact_mod.MEMORY_INITIAL),
        compact_mod.new_skills_placeholder(),
    ]


def _req_texts(request: list) -> str:
    return "\n".join(
        m.content
        for m in request
        if isinstance(m, UserMessage) and isinstance(m.content, str)
    )


async def _run_checkpoint_session(responses, provider_cls=CheckpointProvider):
    registry = _registry()
    provider = provider_cls(responses)
    session = Session()
    messages = _scaffold_messages()
    await drive(_loop_run(provider, registry), session, messages)
    return provider, session, messages


@pytest.mark.asyncio
async def test_checkpoint_requires_scratchpad_then_collapses() -> None:
    # Turn before the first collapse: the request carries the ⚠ tail; a
    # scratchpad-less note is rejected once; the compliant retry passes
    # and the collapse folds turn 0 to its summary line.
    provider, session, messages = await _run_checkpoint_session(
        [
            _asst(tool_calls=[_tc("note", {"summary": "a"}), _tc("peek")]),
            _asst(tool_calls=[_tc("note", {"summary": "b"}), _tc("peek")]),
            _asst(
                tool_calls=[
                    _tc("note", {"summary": "b", "scratchpad": "cart=3; addr saved"}),
                    _tc("peek"),
                ]
            ),
            _asst(
                tool_calls=[
                    _tc("note", {"summary": "closing"}),
                    _tc("end_session", {"status": DONE, "recap": "done"}),
                ]
            ),
        ]
    )

    assert session.sentinel_status == DONE
    assert len(provider.calls) == 4
    # ⚠ tail rides the imminent turn's requests only (1 = first attempt,
    # 2 = retry) — not turn 0, not the post-collapse turn.
    tails = ["newest 1 folds" in _req_texts(r) for r in provider.requests]
    assert tails == [False, True, True, False]
    # The retry request carries the checkpoint corrective.
    assert "Re-issue the SAME turn" in _req_texts(provider.requests[2])
    # The collapse fired: turn 0's summary folded into the slot.
    assert "- a" in messages[2].content


@pytest.mark.asyncio
async def test_checkpoint_fails_open_when_reminder_ignored() -> None:
    # A model that never adds the scratchpad is corrected ONCE, then its
    # summary-only note is accepted — compression proceeds on summaries
    # alone rather than stalling the session.
    provider, session, messages = await _run_checkpoint_session(
        [
            _asst(tool_calls=[_tc("note", {"summary": "a"}), _tc("peek")]),
            _asst(tool_calls=[_tc("note", {"summary": "b"}), _tc("peek")]),
            _asst(
                tool_calls=[_tc("note", {"summary": "b"}), _tc("peek")]
            ),  # still none
            _asst(
                tool_calls=[
                    _tc("note", {"summary": "closing"}),
                    _tc("end_session", {"status": DONE, "recap": "done"}),
                ]
            ),
        ]
    )

    assert session.sentinel_status == DONE
    assert len(provider.calls) == 4
    correctives = sum(
        "Re-issue the SAME turn" in _req_texts(r) for r in provider.requests
    )
    assert correctives == 1  # exactly one nudge, no reject loop
    assert "- a" in messages[2].content  # collapse still fired


@pytest.mark.asyncio
async def test_checkpoint_corrective_renews_per_collapse_event() -> None:
    # With a tight interval EVERY turn is collapse-pending — the single
    # retry must renew per collapse event (completed turn), not once per
    # session, or every compression after the first gets no checkpoint.
    class TightProvider(CheckpointProvider):
        COLLAPSE = CollapsePolicy(first_at=2, keep=1, interval=1)

    def _plain_turn():
        return _asst(tool_calls=[_tc("note", {"summary": "s"}), _tc("peek")])

    provider, session, _ = await _run_checkpoint_session(
        [
            _plain_turn(),  # t0: not pending
            _plain_turn(),  # t1: pending → corrective #1
            _plain_turn(),  # t1 retry: fail open → collapse #1
            _plain_turn(),  # t2: pending again → corrective #2
            _plain_turn(),  # t2 retry: fail open → collapse #2
            _asst(
                tool_calls=[
                    _tc("note", {"summary": "closing"}),
                    _tc("end_session", {"status": DONE, "recap": "done"}),
                ]
            ),  # t3: pending but end_session-exempt
        ],
        provider_cls=TightProvider,
    )

    assert session.sentinel_status == DONE
    assert len(provider.calls) == 6
    correctives = sum(
        "Re-issue the SAME turn" in _req_texts(r) for r in provider.requests
    )
    assert correctives == 2  # one per collapse event, renewed after each


@pytest.mark.asyncio
async def test_checkpoint_skipped_on_end_session_turn() -> None:
    # Closing on the checkpoint turn: nothing to preserve — no corrective.
    provider, session, _ = await _run_checkpoint_session(
        [
            _asst(tool_calls=[_tc("note", {"summary": "a"}), _tc("peek")]),
            _asst(
                tool_calls=[
                    _tc("note", {"summary": "closing"}),
                    _tc("end_session", {"status": DONE, "recap": "done"}),
                ]
            ),
        ]
    )

    assert session.sentinel_status == DONE
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_loop_corrective_counter_resets_on_good_turn(patched_loop_deps) -> None:
    # Interleaved good turns reset the counter — occasional shape slips
    # never accumulate into a STUCK.
    registry = _registry()

    def good(i):
        return _asst(
            tool_calls=[
                _tc("note", {"summary": f"turn {i}"}),
                _tc("peek"),
            ]
        )

    responses = []
    for i in range(CORRECTIVE_LIMIT - 1):
        responses.append(_asst(tool_calls=[_tc("peek")]))  # bad shape
    responses.append(good(0))  # resets the counter
    for i in range(CORRECTIVE_LIMIT - 1):
        responses.append(_asst(tool_calls=[_tc("peek")]))  # bad again
    responses.append(
        _asst(
            tool_calls=[
                _tc("note", {"summary": "closing"}),
                _tc("end_session", {"status": DONE, "recap": "ok"}),
            ]
        )
    )
    provider = FakeProvider(responses)
    session = Session()

    await drive(
        _loop_run(provider, registry),
        session,
        [SystemMessage(content="sys"), UserMessage(content="trig")],
    )

    assert session.sentinel_status == DONE


@pytest.mark.asyncio
async def test_loop_routes_content_filter_to_fail(patched_loop_deps) -> None:
    registry = _registry()
    asst = _asst(finish=FinishReason.CONTENT_FILTER)
    provider = FakeProvider([asst])
    session = Session()

    await drive(
        _loop_run(provider, registry),
        session,
        [SystemMessage(content="s")],
    )

    assert session.sentinel_status == FAIL
    assert "content filter" in session.sentinel_recap


@pytest.mark.asyncio
async def test_loop_provider_failure_marks_stuck(patched_loop_deps) -> None:
    registry = _registry()
    provider = FakeProvider([RuntimeError("network")])
    session = Session()

    await drive(
        _loop_run(provider, registry),
        session,
        [SystemMessage(content="s")],
    )

    assert session.sentinel_status == STUCK
    assert "network" in session.sentinel_recap


@pytest.mark.asyncio
async def test_loop_no_tool_calls_injects_corrective(patched_loop_deps) -> None:
    registry = _registry()
    asst_no_calls = _asst(content="just talking")
    asst_close = _asst(
        tool_calls=[
            _tc("note", {"summary": "x"}),
            _tc("end_session", {"status": IDLE, "recap": "nothing"}),
        ]
    )
    provider = FakeProvider([asst_no_calls, asst_close])
    session = Session()
    messages: list = [SystemMessage(content="s")]

    await drive(_loop_run(provider, registry), session, messages)

    correctives = [
        m
        for m in messages
        if isinstance(m, UserMessage) and "no tool_calls" in str(m.content)
    ]
    assert len(correctives) == 1
    assert session.sentinel_status == IDLE


@pytest.mark.asyncio
async def test_loop_bad_turn_shape_injects_corrective(
    patched_loop_deps,
) -> None:
    registry = _registry()
    bad = _asst(tool_calls=[_tc("peek")])
    good = _asst(
        tool_calls=[
            _tc("note", {"summary": "y"}),
            _tc("end_session", {"status": DONE, "recap": "ok"}),
        ]
    )
    provider = FakeProvider([bad, good])
    session = Session()
    messages: list = [SystemMessage(content="s")]

    await drive(_loop_run(provider, registry), session, messages)

    correctives = [
        m
        for m in messages
        if isinstance(m, UserMessage) and "without `note`" in str(m.content)
    ]
    assert len(correctives) == 1
    assert session.sentinel_status == DONE


@pytest.mark.asyncio
async def test_loop_max_turns_marks_stuck(patched_loop_deps) -> None:
    registry = _registry()
    provider = FakeProvider([_asst() for _ in range(2)])
    session = Session()

    await drive(
        _loop_run(provider, registry, settings=_settings(max_turns=2)),
        session,
        [SystemMessage(content="s")],
    )

    assert session.sentinel_status == STUCK
    assert "max turns" in session.sentinel_recap


@pytest.mark.asyncio
async def test_loop_finish_length_logs_warning(patched_loop_deps) -> None:
    registry = _registry()
    asst = _asst(
        tool_calls=[
            _tc("note", {"summary": "x"}),
            _tc("end_session", {"status": DONE, "recap": "ok"}),
        ],
        finish=FinishReason.LENGTH,
    )
    provider = FakeProvider([asst])
    session = Session()
    tr = MagicMock()

    await drive(
        _loop_run(provider, registry, tr=tr),
        session,
        [SystemMessage(content="s")],
    )

    events = [c.args[0].get("event") for c in tr.write.call_args_list]
    assert "finish_length_warning" in events


# ---------- pre-close pitfalls gate ----------


def _pitfall_tool() -> LocalTool:
    from physiclaw.agent.engine.builtin_tool import _handle_add_pitfall

    return LocalTool(
        name="add_pitfall",
        description="add_pitfall",
        input_schema={
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            "required": ["items"],
        },
        handler=_handle_add_pitfall,
    )


def _registry_with_pitfall() -> dict[str, LocalTool]:
    reg = _registry()
    pt = _pitfall_tool()
    reg[pt.name] = pt
    return reg


def _close(status: str) -> AssistantMessage:
    return _asst(
        tool_calls=[
            _tc("note", {"summary": "closing"}),
            _tc("end_session", {"status": status, "recap": "r"}),
        ]
    )


def _pitfall_asst() -> AssistantMessage:
    return _asst(
        tool_calls=[
            _tc("note", {"summary": "banking traps"}),
            _tc("add_pitfall", {"items": ["京东: avoid Ai搜索"]}),
        ]
    )


async def _run_close_loop(provider, session, registry):
    messages: list = [SystemMessage(content="s")]
    await drive(_loop_run(provider, registry), session, messages)
    return messages


def _pitfall_correctives(messages) -> list:
    return [
        m
        for m in messages
        if isinstance(m, UserMessage) and "add_pitfall(" in str(m.content)
    ]


def _floor0(monkeypatch) -> None:
    # Drop capture_turn_floor to 0 so a short scripted DONE (turn 0) trips the gate.
    from physiclaw.common import config

    monkeypatch.setattr(config.CONFIG.pitfalls, "capture_turn_floor", 0)


@pytest.mark.asyncio
async def test_loop_forces_pitfall_on_long_done(patched_loop_deps, monkeypatch) -> None:
    # A long DONE → force add_pitfall, then the re-issued close passes.
    from physiclaw.agent.engine import pitfalls

    _floor0(monkeypatch)
    registry = _registry_with_pitfall()
    provider = FakeProvider([_close(DONE), _pitfall_asst(), _close(DONE)])
    session = Session()

    messages = await _run_close_loop(provider, session, registry)

    assert len(_pitfall_correctives(messages)) == 1
    assert session.added_pitfalls is True
    assert session.sentinel_status == DONE
    assert pitfalls.read() == ["京东: avoid Ai搜索"]


@pytest.mark.asyncio
async def test_loop_pitfall_corrective_carries_trajectory(
    patched_loop_deps, monkeypatch
) -> None:
    # The turn-marked trajectory rides the corrective so the agent mines the
    # real turn-wasters even after compaction folds early turns.
    _floor0(monkeypatch)
    registry = _registry_with_pitfall()
    session = Session()
    session.plan.update(
        understanding="buy milk",
        steps=[{"content": "search via 搜索 not Ai搜索", "status": "completed"}],
    )
    provider = FakeProvider([_close(DONE), _pitfall_asst(), _close(DONE)])

    messages = await _run_close_loop(provider, session, registry)

    corrective = _pitfall_correctives(messages)[0].content
    assert "<session-trajectory>" in corrective
    assert "search via 搜索 not Ai搜索" in corrective


@pytest.mark.asyncio
async def test_loop_pitfall_gate_fails_open_after_one_retry(
    patched_loop_deps, monkeypatch
) -> None:
    # Agent ignores the corrective and re-issues end_session — closes on the
    # second attempt (one-shot), never adding.
    _floor0(monkeypatch)
    registry = _registry_with_pitfall()
    provider = FakeProvider([_close(DONE), _close(DONE)])
    session = Session()

    messages = await _run_close_loop(provider, session, registry)

    assert len(_pitfall_correctives(messages)) == 1
    assert session.added_pitfalls is False
    assert session.sentinel_status == DONE


@pytest.mark.asyncio
async def test_loop_no_pitfall_gate_on_stuck(patched_loop_deps, monkeypatch) -> None:
    # A STUCK close never captures — the agent never escaped the trap, so it
    # can't write a real fix. Even with the floor at 0.
    _floor0(monkeypatch)
    registry = _registry_with_pitfall()
    provider = FakeProvider([_close(STUCK)])
    session = Session()

    messages = await _run_close_loop(provider, session, registry)

    assert not _pitfall_correctives(messages)
    assert session.sentinel_status == STUCK


@pytest.mark.asyncio
async def test_loop_no_pitfall_gate_on_short_done(patched_loop_deps) -> None:
    # A short DONE (below capture_turn_floor) sailed through → no gate.
    registry = _registry_with_pitfall()
    provider = FakeProvider([_close(DONE)])  # closes at turn 0, far below the floor
    session = Session()

    messages = await _run_close_loop(provider, session, registry)

    assert not _pitfall_correctives(messages)
    assert session.sentinel_status == DONE


@pytest.mark.asyncio
async def test_loop_no_pitfall_gate_on_idle(patched_loop_deps, monkeypatch) -> None:
    _floor0(monkeypatch)
    registry = _registry_with_pitfall()
    provider = FakeProvider([_close(IDLE)])
    session = Session()

    messages = await _run_close_loop(provider, session, registry)

    assert not _pitfall_correctives(messages)
    assert session.sentinel_status == IDLE


# ---------- pre-close memory-cue gate ----------


def _save_memory_tool() -> LocalTool:
    async def _h(session, _args):
        session.saved_memory = True
        return "saved"

    return LocalTool(
        name="save_memory",
        description="save",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=_h,
    )


def _registry_with_save() -> dict[str, LocalTool]:
    reg = _registry()
    st = _save_memory_tool()
    reg[st.name] = st
    return reg


def _note(
    summary: str, other_name: str, other_args: dict | None = None
) -> AssistantMessage:
    return _asst(
        tool_calls=[
            _tc("note", {"summary": summary}),
            _tc(other_name, other_args or {}),
        ]
    )


def _memory_correctives(messages) -> list:
    return [
        m
        for m in messages
        if isinstance(m, UserMessage)
        and "flagged something to remember" in str(m.content)
    ]


@pytest.mark.asyncio
async def test_loop_scans_cue_each_turn_and_forces_save_before_close(
    patched_loop_deps,
) -> None:
    registry = _registry_with_save()
    # turn 0 banks a cue (note summary) alongside an action; close is IDLE so
    # the pitfalls gate stays out of the way — only the memory gate fires.
    provider = FakeProvider(
        [
            _note("记住 user hates cilantro", "peek"),
            _note("closing", "end_session", {"status": IDLE, "recap": "r"}),
            _note("saving", "save_memory", {"text": "no cilantro"}),
            _note("closing", "end_session", {"status": IDLE, "recap": "r"}),
        ]
    )
    session = Session()
    messages = await _run_close_loop(provider, session, registry)

    assert any("cilantro" in c for c in session.memory_cues)  # scanned turn 0
    assert len(_memory_correctives(messages)) == 1
    assert session.saved_memory is True
    assert session.sentinel_status == IDLE


@pytest.mark.asyncio
async def test_loop_no_memory_gate_when_already_saved(patched_loop_deps) -> None:
    registry = _registry_with_save()
    provider = FakeProvider(
        [
            _note("记住 no cilantro", "save_memory", {"text": "no cilantro"}),
            _note("closing", "end_session", {"status": IDLE, "recap": "r"}),
        ]
    )
    session = Session()
    messages = await _run_close_loop(provider, session, registry)

    assert not _memory_correctives(messages)  # cue present but already saved
    assert session.sentinel_status == IDLE


@pytest.mark.asyncio
async def test_loop_no_memory_gate_without_a_cue(patched_loop_deps) -> None:
    registry = _registry_with_save()
    provider = FakeProvider(
        [_note("tapped the button", "end_session", {"status": IDLE, "recap": "r"})]
    )
    session = Session()
    messages = await _run_close_loop(provider, session, registry)

    assert not session.memory_cues
    assert not _memory_correctives(messages)
    assert session.sentinel_status == IDLE


@pytest.mark.asyncio
async def test_loop_memory_gate_fails_open_after_one_retry(patched_loop_deps) -> None:
    registry = _registry_with_save()
    provider = FakeProvider(
        [
            _note("记住 the gate code 4021", "peek"),
            _note("closing", "end_session", {"status": IDLE, "recap": "r"}),
            _note("closing anyway", "end_session", {"status": IDLE, "recap": "r"}),
        ]
    )
    session = Session()
    messages = await _run_close_loop(provider, session, registry)

    assert len(_memory_correctives(messages)) == 1  # one nudge, then closes
    assert session.saved_memory is False
    assert session.sentinel_status == IDLE


# ---------- wall-clock budget ----------


@pytest.mark.asyncio
async def test_loop_closes_stuck_when_budget_exhausted() -> None:
    # Deadline already in the past: the loop must close before any
    # provider round-trip (FakeProvider([]) raises if chatted with).
    # Every other drive test runs with deadline=None, covering the
    # disabled-budget path.
    session = Session()
    run = _mk_run(
        FakeProvider([]),
        settings=_settings(max_session_seconds=10),
        deadline=time.monotonic() - 1,
    )
    messages: list = []

    await drive(run, session, messages)

    assert session.sentinel_status == STUCK
    assert session.budget_exhausted is True
    assert "budget" in session.sentinel_recap
    assert messages == []  # no turn ran


# ---------- log_external_stop ----------


def test_log_external_stop_writes_recovery_line(mem_paths) -> None:
    import datetime as dt

    session = Session()
    session.plan.update(
        user_said="buy oil",
        steps=[
            {"content": "open JD", "status": "completed"},
            {"content": "fix cart qty", "status": "in_progress"},
        ],
    )

    log_external_stop(session, None)

    text = (mem_paths / f"{dt.date.today().isoformat()}.md").read_text(encoding="utf-8")
    assert "stopped externally mid-task" in text
    assert "fix cart qty" in text
    assert "(1/2 steps done)" in text


def test_log_external_stop_skips_closed_and_undrafted_sessions(mem_paths) -> None:
    closed = Session()
    closed.plan.update(user_said="buy oil")
    closed.sentinel_status = DONE
    log_external_stop(closed, None)

    undrafted = Session()  # default seed plan — nothing recoverable
    log_external_stop(undrafted, None)

    assert not mem_paths.exists()  # neither wrote a daily log


def test_log_external_stop_never_raises(mem_paths, monkeypatch) -> None:
    monkeypatch.setattr(
        memory_mod, "append_log", MagicMock(side_effect=OSError("disk full"))
    )
    session = Session()
    session.plan.update(
        user_said="buy oil",
        steps=[{"content": "open JD", "status": "in_progress"}],
    )

    log_external_stop(session, None)  # must not raise


# ---------- _chat_with_retry ----------


@pytest.mark.asyncio
async def test_chat_with_retry_returns_immediately_on_success(mocker) -> None:
    provider = mocker.MagicMock()
    asst = AssistantMessage(
        content="ok", tool_calls=[], finish_reason=FinishReason.STOP
    )
    provider.chat = mocker.AsyncMock(return_value=asst)

    out = await _chat_with_retry(provider, [], [], attempts=3, backoff=0.0)

    assert out is asst
    assert provider.chat.await_count == 1


@pytest.mark.asyncio
async def test_chat_with_retry_retries_then_succeeds(mocker) -> None:
    mocker.patch("asyncio.sleep")
    provider = mocker.MagicMock()
    asst = AssistantMessage(
        content="ok", tool_calls=[], finish_reason=FinishReason.STOP
    )
    provider.chat = mocker.AsyncMock(
        side_effect=[ProviderTransientError("transient"), asst]
    )

    out = await _chat_with_retry(provider, [], [], attempts=3, backoff=0.0)

    assert out is asst
    assert provider.chat.await_count == 2


@pytest.mark.asyncio
async def test_chat_with_retry_raises_provider_error_after_max_attempts(
    mocker,
) -> None:
    mocker.patch("asyncio.sleep")
    provider = mocker.MagicMock()
    provider.chat = mocker.AsyncMock(side_effect=ProviderTransientError("nope"))

    # Typed (ProviderError, not bare RuntimeError) so _call_provider logs
    # expected exhaustion as one line instead of a traceback.
    with pytest.raises(ProviderError, match=r"^gave up after 2 attempts:"):
        await _chat_with_retry(provider, [], [], attempts=2, backoff=0.0)


@pytest.mark.asyncio
async def test_chat_with_retry_does_not_catch_permanent_errors(mocker) -> None:
    provider = mocker.MagicMock()
    provider.chat = mocker.AsyncMock(side_effect=RuntimeError("permanent"))

    with pytest.raises(RuntimeError, match=r"^permanent$"):
        await _chat_with_retry(provider, [], [], attempts=3, backoff=0.0)


# ---------- _log_usage ----------


@pytest.fixture
def trace_stub(mocker):
    t = mocker.MagicMock(spec=Trace)
    return t


def test_log_usage_returns_empty_string_when_no_usage_data(trace_stub) -> None:
    asst = AssistantMessage(
        content="x",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        usage=Usage(),
    )

    out = _log_usage(turn=1, asst=asst, tr=trace_stub)

    assert out == ""
    trace_stub.write.assert_called_once()


def test_log_usage_emits_cache_event_with_derived_new_count(trace_stub) -> None:
    asst = AssistantMessage(
        content="x",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        usage=Usage(prompt_tokens=200, cached_tokens=120, cache_creation_tokens=30),
    )

    _log_usage(turn=5, asst=asst, tr=trace_stub)

    trace_stub.write.assert_called_once()
    payload = trace_stub.write.call_args.args[0]
    assert payload["event"] == "cache"
    assert payload["turn"] == 5
    assert payload["hit"] == 120
    assert payload["create"] == 30
    # new = total - cached - created = 200 - 120 - 30 = 50
    assert payload["new"] == 50
    assert payload["total"] == 200


def test_log_usage_returns_token_summary_with_cache_pct(trace_stub) -> None:
    asst = AssistantMessage(
        content="",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        usage=Usage(prompt_tokens=10000, cached_tokens=5000, cache_creation_tokens=0),
    )

    out = _log_usage(turn=1, asst=asst, tr=trace_stub)

    assert out == "token: 10.0k, cache: 50%"


def test_log_usage_clamps_new_at_zero_when_cached_exceeds_total(trace_stub) -> None:
    # Defensive: if a provider reports cached > total (shouldn't happen
    # but...), `new` floors at 0.
    asst = AssistantMessage(
        content="",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        usage=Usage(prompt_tokens=100, cached_tokens=200, cache_creation_tokens=0),
    )

    _log_usage(turn=1, asst=asst, tr=trace_stub)

    payload = trace_stub.write.call_args.args[0]
    assert payload["new"] == 0


def test_log_usage_emits_output_tokens_for_the_session_summary(trace_stub) -> None:
    # The session summary sums `cache.out` into usage.output_tokens.
    asst = AssistantMessage(
        content="",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        usage=Usage(
            prompt_tokens=100,
            cached_tokens=0,
            cache_creation_tokens=0,
            completion_tokens=77,
        ),
    )

    _log_usage(turn=1, asst=asst, tr=trace_stub)

    payload = trace_stub.write.call_args.args[0]
    assert payload["out"] == 77


@pytest.mark.asyncio
async def test_call_provider_response_event_carries_elapsed_ms(
    mocker,
    trace_stub,
) -> None:
    # The session summary sums `response.elapsed_ms` into provider_time_ms.
    asst = AssistantMessage(
        content="",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        usage=Usage(),
    )
    provider = mocker.MagicMock()
    provider.chat = mocker.AsyncMock(return_value=asst)
    rlog = mocker.MagicMock()
    run = _mk_run(provider=provider, tr=trace_stub, rlog=rlog)

    out = await _call_provider(run, Session(), [], turn=0)

    assert out is asst
    response_event = next(
        c.args[0]
        for c in trace_stub.write.call_args_list
        if c.args[0].get("event") == "response"
    )
    assert isinstance(response_event["elapsed_ms"], int)
    # RawLog got the same number.
    assert (
        rlog.write_response.call_args.kwargs["elapsed_ms"]
        == response_event["elapsed_ms"]
    )


# ---------- _corrective_for_bad_shape ----------


def test_corrective_for_action_without_note() -> None:
    out = _corrective_for_bad_shape(["peek"])

    assert "called `peek` without `note`" in out
    assert "[note(summary=...), peek(...)]" in out


def test_corrective_for_action_without_note_with_extras() -> None:
    out = _corrective_for_bad_shape(["peek", "tap"])

    assert "without `note`" in out
    assert "too many action tools" in out
    assert "['tap']" in out


def test_corrective_for_note_alone() -> None:
    out = _corrective_for_bad_shape(["note"])

    assert "`note` alone with no action tool" in out
    assert "peek()" in out  # default suggestion


def test_corrective_for_note_with_too_many_actions() -> None:
    out = _corrective_for_bad_shape(["note", "peek", "tap"])

    assert "`note` plus 2 action tools" in out


def test_corrective_for_multiple_notes() -> None:
    out = _corrective_for_bad_shape(["note", "note"])

    assert "called `note` 2 times" in out


def test_corrective_for_three_notes() -> None:
    out = _corrective_for_bad_shape(["note", "note", "note"])

    assert "called `note` 3 times" in out
