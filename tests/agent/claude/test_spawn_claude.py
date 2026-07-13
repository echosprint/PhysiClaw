"""Tests for `spawn_claude` — async subprocess + retry loop.

Helper coverage lives in `test_spawn.py`; this file owns the
spawn_claude flow with a fake `asyncio.create_subprocess_exec`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from physiclaw.agent.claude import spawn as spawn_mod
from physiclaw.agent.runtime.hook import Trigger

# ---------- Fakes ----------


class _FakeStdout:
    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)

    async def readline(self) -> bytes:
        if not self._lines:
            return b""
        return self._lines.pop(0)


class _FakeProc:
    def __init__(self, *, lines: list[bytes], returncode: int = 0):
        self.stdout = _FakeStdout(lines)
        self.returncode = returncode
        self.killed = False

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.killed = True


def _result_line(
    status_line: str = ">> DONE - all good", *, num_turns: int = 5
) -> bytes:
    """Build a fake stream-json sequence ending in a result event with
    a final assistant text containing the sentinel."""
    asst = {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": status_line}]},
    }
    return [
        json.dumps(asst).encode() + b"\n",
        json.dumps(
            {
                "type": "result",
                "num_turns": num_turns,
                "result": "all good",
            }
        ).encode()
        + b"\n",
        b"",
    ]


@pytest.fixture
def patch_environment(mocker, tmp_path: Path):
    """Stub everything spawn_claude touches except create_subprocess_exec."""
    mocker.patch.object(spawn_mod, "_warn_stray_context")
    mocker.patch.object(spawn_mod, "_mcp_tools", return_value=[])
    mocker.patch.object(spawn_mod.skill, "discover", return_value={})
    mocker.patch.object(
        spawn_mod,
        "_render_system_prompt",
        return_value="SYSTEM",
    )
    mocker.patch.object(spawn_mod, "_child_env", return_value={})

    # CLAUDE.md must "exist" for _build_cmd not to raise.
    fake_md = tmp_path / "CLAUDE.md"
    fake_md.write_text("doctrine")
    mocker.patch.object(spawn_mod, "CLAUDE_MD", fake_md)
    mocker.patch.object(spawn_mod, "_mcp_config", return_value="{}")
    mocker.patch.object(spawn_mod, "PROJECT_ROOT", tmp_path)

    log_dir = tmp_path / "logs"
    mocker.patch.object(spawn_mod, "LOG_DIR", log_dir)

    # prepare_plugin_dir returns a dir we can rmtree without trouble.
    plugin = tmp_path / "plugin"

    def _prep(sid, skills=None):
        plugin.mkdir(parents=True, exist_ok=True)
        return plugin

    mocker.patch.object(spawn_mod, "prepare_plugin_dir", side_effect=_prep)
    mocker.patch.object(spawn_mod.asyncio, "sleep")  # collapse RETRY_BACKOFF
    return {"plugin": plugin, "log_dir": log_dir}


# ---------- spawn_claude ----------


@pytest.mark.asyncio
async def test_spawn_claude_done_on_first_attempt(
    mocker,
    patch_environment,
) -> None:
    proc = _FakeProc(lines=_result_line(">> DONE - ok"))

    async def _exec(*args, **kwargs):
        return proc

    mocker.patch.object(
        spawn_mod.asyncio,
        "create_subprocess_exec",
        side_effect=_exec,
    )
    spawn_spy = mocker.spy(spawn_mod, "prepare_plugin_dir")

    await spawn_mod.spawn_claude(
        [Trigger(description="t")],
        model_id="opus",
    )

    spawn_spy.assert_called_once()
    # Plugin dir cleaned up.
    assert not patch_environment["plugin"].exists()


@pytest.mark.asyncio
async def test_spawn_claude_retries_on_undone(
    mocker,
    patch_environment,
) -> None:
    """Two UNDONE attempts (no sentinel), third returns DONE."""
    mocker.patch.object(spawn_mod, "MAX_ATTEMPTS", 3)

    procs = [
        _FakeProc(lines=[b"\n", b""]),  # no result, no sentinel → UNDONE
        _FakeProc(lines=[b"\n", b""]),  # UNDONE
        _FakeProc(lines=_result_line(">> DONE - finally")),
    ]
    procs_iter = iter(procs)

    async def _exec(*args, **kwargs):
        return next(procs_iter)

    mocker.patch.object(
        spawn_mod.asyncio,
        "create_subprocess_exec",
        side_effect=_exec,
    )

    await spawn_mod.spawn_claude(
        [Trigger(description="t")],
        model_id="opus",
    )

    # All three attempts ran.
    sleep_spy = spawn_mod.asyncio.sleep
    # Backoff was awaited twice (between attempts 1→2 and 2→3).
    assert sleep_spy.call_count == 2


@pytest.mark.asyncio
async def test_spawn_claude_gives_up_after_max_undone(
    mocker,
    patch_environment,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    mocker.patch.object(spawn_mod, "MAX_ATTEMPTS", 2)

    proc = _FakeProc(lines=[b"\n", b""])

    async def _exec(*args, **kwargs):
        return proc

    mocker.patch.object(
        spawn_mod.asyncio,
        "create_subprocess_exec",
        side_effect=_exec,
    )

    with caplog.at_level(logging.ERROR, logger="physiclaw.agent.claude.spawn"):
        await spawn_mod.spawn_claude(
            [Trigger(description="t")],
            model_id="opus",
        )

    assert any("giving up after 2 UNDONE" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_spawn_claude_logs_nonzero_exit(
    mocker,
    patch_environment,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    proc = _FakeProc(lines=_result_line(">> DONE - x"), returncode=1)

    async def _exec(*args, **kwargs):
        return proc

    mocker.patch.object(
        spawn_mod.asyncio,
        "create_subprocess_exec",
        side_effect=_exec,
    )

    with caplog.at_level(logging.ERROR, logger="physiclaw.agent.claude.spawn"):
        await spawn_mod.spawn_claude(
            [Trigger(description="t")],
            model_id="opus",
        )

    assert any("claude exited 1" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_spawn_claude_kills_on_timeout(
    mocker,
    patch_environment,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    proc = _FakeProc(lines=[b"\n", b""])

    async def _exec(*args, **kwargs):
        return proc

    mocker.patch.object(
        spawn_mod.asyncio,
        "create_subprocess_exec",
        side_effect=_exec,
    )
    mocker.patch.object(
        spawn_mod,
        "_stream",
        side_effect=asyncio.TimeoutError,
    )
    mocker.patch.object(spawn_mod, "MAX_ATTEMPTS", 1)

    with caplog.at_level(logging.ERROR, logger="physiclaw.agent.claude.spawn"):
        await spawn_mod.spawn_claude(
            [Trigger(description="t")],
            model_id="opus",
        )

    assert proc.killed is True
    assert any("killed after" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_spawn_claude_cleans_plugin_dir_on_failure(
    mocker,
    patch_environment,
) -> None:
    """Even when subprocess creation fails, plugin dir gets rmtree'd."""

    async def _bad_exec(*args, **kwargs):
        raise RuntimeError("can't fork")

    mocker.patch.object(
        spawn_mod.asyncio,
        "create_subprocess_exec",
        side_effect=_bad_exec,
    )
    mocker.patch.object(spawn_mod, "MAX_ATTEMPTS", 1)

    with pytest.raises(RuntimeError, match="can't fork"):
        await spawn_mod.spawn_claude(
            [Trigger(description="t")],
            model_id="opus",
        )

    assert not patch_environment["plugin"].exists()


@pytest.mark.asyncio
async def test_spawn_claude_logs_done_summary_on_clean_exit(
    mocker,
    patch_environment,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    proc = _FakeProc(
        lines=_result_line(">> DONE - turns ok", num_turns=12),
    )

    async def _exec(*args, **kwargs):
        return proc

    mocker.patch.object(
        spawn_mod.asyncio,
        "create_subprocess_exec",
        side_effect=_exec,
    )

    with caplog.at_level(logging.INFO, logger="physiclaw.agent.claude.spawn"):
        await spawn_mod.spawn_claude(
            [Trigger(description="t")],
            model_id="opus",
        )

    assert any("turns=12" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_spawn_claude_first_attempt_no_backoff(
    mocker,
    patch_environment,
) -> None:
    """Backoff sleep only fires on retry, never before the first attempt."""
    proc = _FakeProc(lines=_result_line(">> DONE - x"))

    async def _exec(*args, **kwargs):
        return proc

    mocker.patch.object(
        spawn_mod.asyncio,
        "create_subprocess_exec",
        side_effect=_exec,
    )

    sleep_spy = spawn_mod.asyncio.sleep  # already patched in fixture

    await spawn_mod.spawn_claude(
        [Trigger(description="t")],
        model_id="opus",
    )

    sleep_spy.assert_not_called()


@pytest.mark.asyncio
async def test_spawn_claude_kills_proc_when_stream_raises_unexpectedly(
    mocker,
    patch_environment,
) -> None:
    """An unexpected error mid-stream must never orphan a live claude
    subprocess driving the phone — kill before propagating."""

    class _LiveProc(_FakeProc):
        def __init__(self):
            super().__init__(lines=[b""])
            self.returncode = None  # still running

        async def wait(self) -> int:
            self.returncode = -9
            return self.returncode

    proc = _LiveProc()

    async def _exec(*args, **kwargs):
        return proc

    mocker.patch.object(spawn_mod.asyncio, "create_subprocess_exec", side_effect=_exec)
    mocker.patch.object(spawn_mod, "_stream", side_effect=RuntimeError("boom"))
    mocker.patch.object(spawn_mod, "MAX_ATTEMPTS", 1)

    with pytest.raises(RuntimeError, match="boom"):
        await spawn_mod.spawn_claude([Trigger(description="t")], model_id="opus")

    assert proc.killed is True


@pytest.mark.asyncio
async def test_spawn_claude_kills_on_oversized_output_line(
    mocker,
    patch_environment,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """readline raises ValueError past the buffer limit — treated like a
    timeout: kill + 'killed' status, not a crash."""
    import logging

    proc = _FakeProc(lines=[b""])

    async def _exec(*args, **kwargs):
        return proc

    mocker.patch.object(spawn_mod.asyncio, "create_subprocess_exec", side_effect=_exec)
    mocker.patch.object(
        spawn_mod, "_stream", side_effect=ValueError("Separator is not found")
    )
    mocker.patch.object(spawn_mod, "MAX_ATTEMPTS", 1)

    with caplog.at_level(logging.ERROR, logger="physiclaw.agent.claude.spawn"):
        await spawn_mod.spawn_claude([Trigger(description="t")], model_id="opus")

    assert proc.killed is True
    assert any("stream buffer" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_spawn_claude_bounds_post_eof_wait(
    mocker,
    patch_environment,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A child that closes stdout but never exits must be killed after
    the EXIT_WAIT_SECONDS grace, not awaited forever."""
    import logging

    class _HangingProc(_FakeProc):
        def __init__(self):
            super().__init__(lines=[b""])
            self.returncode = None
            self._waits = 0

        async def wait(self) -> int:
            self._waits += 1
            if self.killed:
                self.returncode = -9
                return self.returncode
            await asyncio.sleep(3600)  # never exits on its own

    proc = _HangingProc()

    async def _exec(*args, **kwargs):
        return proc

    mocker.patch.object(spawn_mod.asyncio, "create_subprocess_exec", side_effect=_exec)
    mocker.patch.object(spawn_mod, "EXIT_WAIT_SECONDS", 0.01)
    mocker.patch.object(spawn_mod, "MAX_ATTEMPTS", 1)

    with caplog.at_level(logging.ERROR, logger="physiclaw.agent.claude.spawn"):
        await spawn_mod.spawn_claude([Trigger(description="t")], model_id="opus")

    assert proc.killed is True
