"""Tests for `physiclaw.agent.engine.builtin_tool` — local tool handlers.

The engine's local tools call out to memory/jobs/scratchpad/skill —
those modules are mocked so handler tests stay focused on the tool
surface (input parsing, return strings, session mutation).

`schemas()` and `build_registry()` are exercised at the bottom for
the wire-format and ordering contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from physiclaw.agent.engine import memory, scratchpad
from physiclaw.agent.engine.builtin_tool import (
    LocalTool,
    Session,
    _handle_add_pitfall,
    _handle_append_log,
    _handle_create_job,
    _handle_end_session,
    _handle_finish_job,
    _handle_get_job,
    _handle_list_jobs,
    _handle_note,
    _handle_read_logs,
    _handle_read_memory,
    _handle_save_memory,
    _handle_skill_factory,
    _handle_update_memory,
    _handle_update_progress,
    _handle_wait,
    build_registry,
    schemas,
)
from physiclaw.agent.engine.plan import Plan
from physiclaw.agent.engine.skill import Skill

# ---------- Session ----------


def test_session_default_state() -> None:
    s = Session()

    assert s.sentinel_status is None
    assert s.sentinel_recap == ""
    assert s.sentinel_turn_created_job is False
    assert isinstance(s.plan, Plan)
    assert s.scratchpad == ""


def test_session_default_plans_not_shared() -> None:
    a = Session()
    b = Session()

    a.plan.user_said = "hello"

    assert b.plan.user_said != "hello"


# ---------- _handle_note ----------


@pytest.mark.asyncio
async def test_note_handler_returns_summary_message() -> None:
    s = Session()

    out = await _handle_note(s, {"summary": "looking for Send button"})

    assert out == "noted: looking for Send button"


@pytest.mark.asyncio
async def test_note_handler_writes_scratchpad_when_provided() -> None:
    s = Session()

    out = await _handle_note(s, {"summary": "x", "scratchpad": "remember this"})

    assert "scratchpad updated" in out
    assert s.scratchpad == "remember this"


@pytest.mark.asyncio
async def test_note_handler_records_scratchpad_rejection_on_oversize() -> None:
    s = Session()
    too_big = "x" * (scratchpad.MAX_CHARS + 1)

    out = await _handle_note(s, {"summary": "x", "scratchpad": too_big})

    assert "scratchpad rejected" in out
    assert "Summarize before writing" in out


@pytest.mark.asyncio
async def test_note_handler_no_scratchpad_if_arg_omitted() -> None:
    s = Session()

    await _handle_note(s, {"summary": "x"})

    assert s.scratchpad == ""


# ---------- _handle_update_progress ----------


@pytest.mark.asyncio
async def test_update_progress_calls_plan_update() -> None:
    s = Session()

    out = await _handle_update_progress(s, {"user_said": "buy bananas"})

    assert out == "progress updated"
    assert s.plan.user_said == "buy bananas"


@pytest.mark.asyncio
async def test_update_progress_returns_rejected_message_on_validation_error() -> None:
    s = Session()

    out = await _handle_update_progress(s, {})  # no args raises ValueError

    assert out.startswith("update_progress rejected:")


# ---------- _handle_append_log / _handle_save_memory ----------


@pytest.mark.asyncio
async def test_append_log_calls_memory_append_log(mocker) -> None:
    spy = mocker.patch("physiclaw.agent.engine.memory.append_log")

    out = await _handle_append_log(Session(), {"entry": "did stuff"})

    assert out == "log appended"
    spy.assert_called_once_with("did stuff")


@pytest.mark.asyncio
async def test_save_memory_calls_memory_save_fact(mocker) -> None:
    spy = mocker.patch("physiclaw.agent.engine.memory.save_fact")

    mocker.patch("physiclaw.agent.engine.memory.over_soft_cap", return_value=None)
    s = Session()
    out = await _handle_save_memory(s, {"text": "user prefers metric"})

    assert out.startswith("saved: 'user prefers metric'")
    assert "## memory.md (now)" in out  # echoes the full current store
    spy.assert_called_once_with("user prefers metric")
    assert s.saved_memory is True  # satisfies the pre-close memory-cue gate


@pytest.mark.asyncio
async def test_save_memory_echoes_full_updated_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No mocking — a real write, then the result echoes the whole store so the
    # agent sees the current memory without a re-read (wake snapshot is stale).
    # memory.MEMORY_* are import-bound module constants, so re-point them to tmp
    # (like test_memory's _memory_paths) — else the real write leaks across tests.
    mem = tmp_path / "mem"
    mem.mkdir()
    monkeypatch.setattr(memory, "MEMORY_DIR", mem)
    monkeypatch.setattr(memory, "MEMORY_FILE", mem / "memory.md")

    s = Session()
    await _handle_save_memory(s, {"text": "fact one"})
    out = await _handle_save_memory(s, {"text": "fact two"})

    assert "## memory.md (now)" in out
    assert "fact one" in out and "fact two" in out  # full store, both facts


@pytest.mark.asyncio
async def test_save_memory_nudges_to_consolidate_over_soft_cap(mocker) -> None:
    mocker.patch("physiclaw.agent.engine.memory.save_fact")
    mocker.patch("physiclaw.agent.engine.memory.over_soft_cap", return_value=2500)

    out = await _handle_save_memory(Session(), {"text": "x"})

    assert "update_memory" in out and "2500" in out  # curation nudge surfaced


# ---------- _handle_create_job ----------


@pytest.mark.asyncio
async def test_create_job_calls_jobs_create_and_marks_session(mocker) -> None:
    spy = mocker.patch("physiclaw.agent.engine.jobs.create_job")
    s = Session()

    out = await _handle_create_job(
        s,
        {
            "id": "user-greet",
            "description": "d",
            "schedule": "0 7 * * *",
            "context": "ten chars at least",
        },
    )

    assert out == "scheduled job 'user-greet'"
    assert s.sentinel_turn_created_job is True
    spy.assert_called_once_with(
        id="user-greet",
        description="d",
        schedule="0 7 * * *",
        context="ten chars at least",
        kind="one-time",
    )


@pytest.mark.asyncio
async def test_create_job_uses_explicit_kind_when_provided(mocker) -> None:
    spy = mocker.patch("physiclaw.agent.engine.jobs.create_job")

    await _handle_create_job(
        Session(),
        {
            "id": "x",
            "description": "d",
            "schedule": "* * * * *",
            "context": "ten chars at least",
            "kind": "periodic",
        },
    )

    assert spy.call_args.kwargs["kind"] == "periodic"


# ---------- _handle_get_job ----------


@pytest.mark.asyncio
async def test_get_job_renders_full_job_block(mocker) -> None:
    fake_job = type("J", (), {})()
    fake_job.id = "x"
    fake_job.description = "Say hi"
    fake_job.kind = "periodic"
    fake_job.status = "pend"
    fake_job.schedule = "0 7 * * *"
    fake_job.context = "morning context"
    fake_job.next_fire_time = "2026-04-29T07:00"
    fake_job.last_fire_time = ""
    fake_job.execution_time = ""
    fake_job.execution_result = ""
    mocker.patch("physiclaw.agent.engine.jobs.get_job", return_value=fake_job)

    out = await _handle_get_job(Session(), {"id": "x"})

    assert "## x" in out
    assert "Say hi" in out
    assert "- Schedule: `0 7 * * *`" in out
    # Empty fields render as NEVER ("(never)").
    assert "- Last fire time: (never)" in out


# ---------- _handle_read_memory / _handle_read_logs ----------


@pytest.mark.asyncio
async def test_read_memory_returns_persistent_or_placeholder(mocker) -> None:
    mocker.patch(
        "physiclaw.agent.engine.memory.load_persistent",
        return_value="user prefers metric\nuses AT",
    )

    out = await _handle_read_memory(Session(), {})

    assert out == "user prefers metric\nuses AT"


@pytest.mark.asyncio
async def test_read_memory_placeholder_when_empty(mocker) -> None:
    mocker.patch("physiclaw.agent.engine.memory.load_persistent", return_value="")

    out = await _handle_read_memory(Session(), {})

    assert out == "(memory.md is empty)"


@pytest.mark.asyncio
async def test_read_logs_uses_default_entries_when_omitted(mocker) -> None:
    spy = mocker.patch(
        "physiclaw.agent.engine.memory.load_recent_entries",
        return_value="line",
    )

    await _handle_read_logs(Session(), {})

    # Default = memory.DEFAULT_LOG_ENTRIES (positive int).
    assert spy.call_args.args[0] > 0


@pytest.mark.asyncio
async def test_read_logs_uses_explicit_entries(mocker) -> None:
    spy = mocker.patch(
        "physiclaw.agent.engine.memory.load_recent_entries",
        return_value="entries",
    )

    await _handle_read_logs(Session(), {"entries": 5})

    spy.assert_called_once_with(5)


@pytest.mark.asyncio
async def test_read_logs_placeholder_when_empty(mocker) -> None:
    mocker.patch("physiclaw.agent.engine.memory.load_recent_entries", return_value="")

    out = await _handle_read_logs(Session(), {})

    assert out == "(no log entries found)"


# ---------- _handle_update_memory ----------


@pytest.mark.asyncio
async def test_update_memory_calls_memory_update_fact(mocker) -> None:
    spy = mocker.patch("physiclaw.agent.engine.memory.update_fact")

    out = await _handle_update_memory(Session(), {"old": "metric", "new": "imperial"})

    assert out.startswith("memory.md updated")
    assert "## memory.md (now)" in out  # echoes the full current store
    spy.assert_called_once_with("metric", "imperial")


# ---------- _handle_list_jobs ----------


@pytest.mark.asyncio
async def test_list_jobs_returns_no_jobs_message_when_empty(mocker) -> None:
    mocker.patch("physiclaw.agent.engine.builtin_tool.load_jobs", return_value=[])

    out = await _handle_list_jobs(Session(), {})

    assert out == "no jobs"


@pytest.mark.asyncio
async def test_list_jobs_filters_by_status(mocker) -> None:
    fake = [type("J", (), {})(), type("J", (), {})()]
    fake[0].id, fake[0].kind, fake[0].status = "a", "periodic", "pend"
    fake[1].id, fake[1].kind, fake[1].status = "b", "one-time", "done"
    for j in fake:
        j.description = "d"
        j.next_fire_time = ""

    mocker.patch("physiclaw.agent.engine.builtin_tool.load_jobs", return_value=fake)

    out = await _handle_list_jobs(Session(), {"status": "pend"})

    assert "1 job(s)" in out
    assert " a — " in out  # specific match: id `a` not the letter elsewhere
    assert " b — " not in out


@pytest.mark.asyncio
async def test_list_jobs_no_match_message_when_filter_empty(mocker) -> None:
    fake = [type("J", (), {})()]
    fake[0].id, fake[0].kind, fake[0].status = "a", "periodic", "done"
    fake[0].description = "d"
    fake[0].next_fire_time = ""
    mocker.patch("physiclaw.agent.engine.builtin_tool.load_jobs", return_value=fake)

    out = await _handle_list_jobs(Session(), {"status": "fail"})

    assert out == "no jobs with status='fail'"


@pytest.mark.asyncio
async def test_list_jobs_renders_job_lines(mocker) -> None:
    fake = type("J", (), {})()
    fake.id, fake.kind, fake.status = "x", "periodic", "pend"
    fake.description = "do things"
    fake.next_fire_time = "2026-04-28T07:00"
    mocker.patch("physiclaw.agent.engine.builtin_tool.load_jobs", return_value=[fake])

    out = await _handle_list_jobs(Session(), {})

    assert "[periodic] [pend] x — do things (next: 2026-04-28T07:00)" in out


# ---------- _handle_finish_job ----------


@pytest.mark.asyncio
async def test_finish_job_calls_jobs_finish(mocker) -> None:
    # The handler passes finish_job's reply through verbatim — for a
    # periodic job that reply carries the RE-ARMED warning the agent
    # must see (see jobs.finish_job).
    spy = mocker.patch(
        "physiclaw.agent.engine.jobs.finish_job",
        return_value="finished job 'x' as done",
    )

    out = await _handle_finish_job(
        Session(), {"id": "x", "status": "done", "recap": "ok"}
    )

    assert out == "finished job 'x' as done"
    spy.assert_called_once_with(id="x", status="done", recap="ok")


# ---------- _handle_wait ----------


@pytest.mark.asyncio
async def test_wait_sleeps_for_given_seconds(mocker) -> None:
    sleep = mocker.patch("asyncio.sleep")

    out = await _handle_wait(Session(), {"seconds": 5})

    assert out == "waited 5s — `peek` now to see what changed."
    sleep.assert_awaited_once_with(5)


# ---------- _handle_end_session ----------


@pytest.mark.asyncio
async def test_end_session_marks_status_and_recap() -> None:
    s = Session()

    out = await _handle_end_session(s, {"status": "DONE", "recap": "ok"})

    assert out == "session closing: DONE"
    assert s.sentinel_status == "DONE"
    assert s.sentinel_recap == "ok"


@pytest.mark.asyncio
async def test_end_session_strips_recap() -> None:
    s = Session()

    await _handle_end_session(s, {"status": "DONE", "recap": "  ok  "})

    assert s.sentinel_recap == "ok"


@pytest.mark.asyncio
async def test_end_session_recap_default_empty_string() -> None:
    s = Session()

    await _handle_end_session(s, {"status": "DONE"})

    assert s.sentinel_recap == ""


@pytest.mark.asyncio
async def test_end_session_raises_on_invalid_status() -> None:
    s = Session()

    with pytest.raises(ValueError, match=r"^status must be one of"):
        await _handle_end_session(s, {"status": "MAYBE", "recap": ""})


# ---------- _handle_skill_factory ----------


@pytest.mark.asyncio
async def test_skill_handler_dispatches_via_skill_module(mocker) -> None:
    fake_dispatch = mocker.patch(
        "physiclaw.agent.engine.skill.dispatch", return_value="skill body"
    )
    handler = _handle_skill_factory({})

    out = await handler(Session(), {"name": "wechat"})

    assert out == "skill body"
    fake_dispatch.assert_called_once_with({}, {"name": "wechat"})


async def test_report_screen_layout_handler_delegates(mocker) -> None:
    from physiclaw.agent.engine.builtin_tool import (
        _handle_report_screen_layout,
    )

    rec = mocker.patch(
        "physiclaw.agent.layout.record",
        return_value="LAYOUT SAVED",
    )
    # Not the completing call → no session-ending / restart side effect.
    mocker.patch("physiclaw.agent.layout.is_learned", return_value=False)

    session = Session()
    bbox = [0.03, 0.08, 0.88, 0.13]
    out = await _handle_report_screen_layout(
        session, {"page": "spotlight", "field": "spotlight_input", "bbox": bbox}
    )

    assert out == "LAYOUT SAVED"
    rec.assert_called_once_with("spotlight", "spotlight_input", bbox, None)
    assert session.restart_for_setup is False
    assert session.sentinel_status is None


async def test_report_screen_layout_handler_restarts_on_completion(mocker) -> None:
    from physiclaw.agent.engine.builtin_tool import (
        _handle_report_screen_layout,
    )
    from physiclaw.agent.runtime.sentinel import IDLE

    mocker.patch(
        "physiclaw.agent.layout.record",
        return_value="done",
    )
    # is_learned: False before the call, True after → this call completed setup.
    mocker.patch(
        "physiclaw.agent.layout.is_learned",
        side_effect=[False, True],
    )

    session = Session()
    await _handle_report_screen_layout(
        session,
        {
            "page": "chat-keyboard",
            "app": "wechat",
            "field": "send",
            "bbox": [0.75, 0.87, 0.99, 0.92],
        },
    )

    assert session.restart_for_setup is True
    assert session.sentinel_status == IDLE


async def test_report_screen_layout_handler_no_restart_when_already_complete(
    mocker,
) -> None:
    from physiclaw.agent.engine.builtin_tool import (
        _handle_report_screen_layout,
    )

    mocker.patch("physiclaw.agent.layout.record", return_value="updated")
    # Already complete before AND after → a correction, not a completing call.
    mocker.patch("physiclaw.agent.layout.is_learned", return_value=True)

    session = Session()
    await _handle_report_screen_layout(
        session,
        {
            "page": "chat-keyboard",
            "app": "wechat",
            "field": "send",
            "bbox": [0.76, 0.87, 0.99, 0.92],
        },
    )

    assert session.restart_for_setup is False
    assert session.sentinel_status is None


# ---------- schemas ----------


def test_schemas_flattens_registry_to_wire_dicts() -> None:
    registry = build_registry({})

    out = schemas(registry)

    assert all(set(d) == {"name", "description", "input_schema"} for d in out)
    assert {d["name"] for d in out} >= {
        "note",
        "update_progress",
        "wait",
        "end_session",
    }


# ---------- build_registry ----------


def test_build_registry_first_two_tools_are_note_and_update_progress() -> None:
    reg = build_registry({})
    keys = list(reg.keys())

    assert keys[0] == "note"
    assert keys[1] == "update_progress"


def test_build_registry_omits_skill_when_no_skills_discovered() -> None:
    reg = build_registry({})

    assert "Skill" not in reg


def test_build_registry_includes_skill_tool_when_skills_present(
    tmp_path: Path,
) -> None:
    skill_registry = {
        "wechat": Skill(name="wechat", description="d", body="", dir=tmp_path),
    }

    reg = build_registry(skill_registry)

    assert "Skill" in reg
    assert "wechat" in reg["Skill"].description


def test_build_registry_returns_local_tool_instances() -> None:
    reg = build_registry({})

    assert all(isinstance(t, LocalTool) for t in reg.values())


# ---------- run_macro ----------


def _macro_spec(name: str = "demo"):
    from physiclaw.macros.parse import parse_macro

    return parse_macro(
        f"name: {name}\ndescription: d\nenabled: true\n"
        "inputs:\n  msg:\n    description: text\n"
        "steps:\n  - name: send-to-clipboard-1\n    tool: send_to_clipboard\n    with:\n"
        '      text: "{msg}"\n',
        name,
    )


def test_build_registry_omits_run_macro_when_no_macros() -> None:
    reg = build_registry({}, {})

    assert "run_macro" not in reg


def test_build_registry_includes_run_macro_when_macros_present() -> None:
    reg = build_registry({}, {"demo": _macro_spec()})

    assert "run_macro" in reg
    assert "demo" in reg["run_macro"].description


@pytest.mark.asyncio
async def test_run_macro_handler_unknown_name_raises_with_available() -> None:
    reg = build_registry({}, {"demo": _macro_spec()})

    with pytest.raises(ValueError, match="unknown macro 'nope'.*demo"):
        await reg["run_macro"].handler(Session(), {"name": "nope"})


@pytest.mark.asyncio
async def test_run_macro_handler_runs_and_records_success(mocker) -> None:
    from physiclaw.macros import stats as macro_stats

    class OkMcp:
        async def call_tool(self, name, args=None):
            return [{"type": "text", "text": f"{name} ok | screen: changed"}]

    mocker.patch(
        "physiclaw.agent.engine.mcp_tool.get_mcp",
        new=mocker.AsyncMock(return_value=OkMcp()),
    )
    reg = build_registry({}, {"demo": _macro_spec()})

    session = Session()
    blocks = await reg["run_macro"].handler(
        session, {"name": "demo", "inputs": {"msg": "hi"}}
    )

    assert isinstance(blocks, list)
    assert "all 1 steps completed" in blocks[0]["text"]
    assert macro_stats.load()["demo"]["total_successes"] == 1
    assert session.failed_macros == set()  # a good run burns nothing


@pytest.mark.asyncio
async def test_run_macro_handler_records_abort(mocker) -> None:
    from physiclaw.macros import stats as macro_stats

    class DownMcp:
        async def call_tool(self, name, args=None):
            raise RuntimeError("arm busy")

    mocker.patch(
        "physiclaw.agent.engine.mcp_tool.get_mcp",
        new=mocker.AsyncMock(return_value=DownMcp()),
    )
    reg = build_registry({}, {"demo": _macro_spec()})

    session = Session()
    blocks = await reg["run_macro"].handler(
        session, {"name": "demo", "inputs": {"msg": "hi"}}
    )

    assert "ABORTED" in blocks[0]["text"]
    entry = macro_stats.load()["demo"]
    assert entry["total_aborts"] == 1
    assert entry["last_abort"]["reason"] == "tool_error"
    # The handler is the ONLY writer of this set; `policy.BurnedMacro` is the
    # only reader. Without this assertion, dropping the write leaves the
    # suite green and silently turns the one-strike rule into dead code.
    assert session.failed_macros == {"demo"}


@pytest.mark.asyncio
async def test_run_macro_handler_bad_input_records_and_raises(mocker) -> None:
    from physiclaw.macros import stats as macro_stats
    from physiclaw.macros.model import MacroError

    mocker.patch(
        "physiclaw.agent.engine.mcp_tool.get_mcp",
        new=mocker.AsyncMock(return_value=object()),
    )
    reg = build_registry({}, {"demo": _macro_spec()})

    with pytest.raises(MacroError, match="missing required input"):
        await reg["run_macro"].handler(Session(), {"name": "demo"})

    assert macro_stats.load()["demo"]["last_abort"]["reason"] == "bad_input"


@pytest.mark.asyncio
async def test_add_pitfall_handler_appends_and_sets_flag() -> None:
    from physiclaw.agent.engine import pitfalls

    s = Session()
    out = await _handle_add_pitfall(
        s,
        {
            "items": ["京东: Ai搜索 opens AI chat → use right-side 搜索"],
        },
    )

    assert s.added_pitfalls is True
    assert "1 pitfall" in out
    assert pitfalls.read() == ["京东: Ai搜索 opens AI chat → use right-side 搜索"]


@pytest.mark.asyncio
async def test_add_pitfall_handler_sets_flag_even_when_nothing_added() -> None:
    # An empty add still counts — re-forcing an empty capture is pointless.
    s = Session()
    out = await _handle_add_pitfall(s, {"items": ["   ", ""]})

    assert s.added_pitfalls is True
    assert "0 pitfall" in out


def test_build_registry_includes_all_tool_categories(mocker) -> None:
    # First run (layout not learned): the setup tool is present.
    mocker.patch("physiclaw.agent.layout.is_learned", return_value=False)
    keys = set(build_registry({}).keys())

    assert keys == {
        "note",
        "update_progress",
        "append_log",
        "save_memory",
        "read_memory",
        "read_logs",
        "update_memory",
        "create_job",
        "get_job",
        "list_jobs",
        "finish_job",
        "wait",
        "report_screen_layout",
        "add_pitfall",
        "end_session",
    }


def test_build_registry_drops_report_screen_layout_once_learned(mocker) -> None:
    mocker.patch("physiclaw.agent.layout.is_learned", return_value=True)
    keys = set(build_registry({}).keys())

    assert "report_screen_layout" not in keys
    assert "note" in keys and "end_session" in keys  # everything else intact


# ---------- run_macro: pack macros (the conductor's hands) ----------


def test_build_registry_registers_run_macro_for_pack_macros_alone() -> None:
    reg = build_registry({}, {}, {"demo/leg": _macro_spec("leg")})

    assert "run_macro" in reg
    # The model-facing description never names pack macros.
    assert "demo/leg" not in reg["run_macro"].description
    assert "none available" in reg["run_macro"].description


@pytest.mark.asyncio
async def test_run_macro_pack_macro_only_runs_on_a_synthesized_turn(mocker) -> None:
    from physiclaw.macros import stats as macro_stats

    class OkMcp:
        async def call_tool(self, name, args=None):
            return [{"type": "text", "text": f"{name} ok | screen: changed"}]

    mocker.patch(
        "physiclaw.agent.engine.mcp_tool.get_mcp",
        new=mocker.AsyncMock(return_value=OkMcp()),
    )
    reg = build_registry({}, {}, {"demo/leg": _macro_spec("leg")})

    # A model turn naming the qualified macro gets the same unknown-macro
    # error as any typo — pack macros are uncallable by the model.
    with pytest.raises(ValueError, match="unknown macro 'demo/leg'"):
        await reg["run_macro"].handler(Session(), {"name": "demo/leg"})

    session = Session()
    session.synthesized_turn = True
    blocks = await reg["run_macro"].handler(
        session, {"name": "demo/leg", "inputs": {"msg": "hi"}}
    )

    assert "all 1 steps completed" in blocks[0]["text"]
    # Stats fold under the qualified name — the pack macro's own decay
    # signal, never shadowing a plain user macro named "leg".
    assert macro_stats.load()["demo/leg"]["total_successes"] == 1
    assert "leg" not in macro_stats.load()


@pytest.mark.asyncio
async def test_run_macro_pack_macro_abort_burns_the_called_name(mocker) -> None:
    class DownMcp:
        async def call_tool(self, name, args=None):
            raise RuntimeError("arm busy")

    mocker.patch(
        "physiclaw.agent.engine.mcp_tool.get_mcp",
        new=mocker.AsyncMock(return_value=DownMcp()),
    )
    reg = build_registry({}, {}, {"demo/leg": _macro_spec("leg")})

    session = Session()
    session.synthesized_turn = True
    blocks = await reg["run_macro"].handler(
        session, {"name": "demo/leg", "inputs": {"msg": "hi"}}
    )

    assert "ABORTED" in blocks[0]["text"]
    # Burned under the CALLED name, so a healthy user macro named "leg"
    # is never shadowed by the pack macro's failure.
    assert session.failed_macros == {"demo/leg"}
