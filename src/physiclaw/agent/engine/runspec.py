"""Run wiring — the settings and per-session context dataclasses shared
by `engine` (session lifecycle), `loop` (turn driver), and `dispatch`
(tool-call execution).

Leaf of the engine's internal split: the three sibling modules all
depend on this one, never on each other through it, so it must not
import any of them.
"""

from dataclasses import dataclass

from physiclaw.agent.engine.builtin_tool import LocalTool
from physiclaw.agent.engine.mcp_tool import McpClient
from physiclaw.agent.engine.policy import Policies
from physiclaw.agent.engine.trace import RawLog, Trace
from physiclaw.agent.provider import Provider
from physiclaw.common.config import CONFIG


@dataclass(frozen=True, slots=True)
class Settings:
    """Engine knobs, read from CONFIG at `run()` time — not import time —
    so tests construct them explicitly and a config write between wakes
    takes effect without a process restart.

    `max_turns` is a runaway-loop backstop, not a context-safety limit.
    Prompt tokens grow ~624·t + 13k empirically (R²=0.97); at 1M context
    (Qwen3.6-plus) the hard wall is ~1,580 turns, so 300 leaves ample
    headroom. `max_session_attempts` is session-level STUCK retries;
    `provider_retry_attempts` is per-call transient-error retries — two
    different knobs, deliberately not shared.

    `max_session_seconds` is the wall-clock counterpart of `max_turns`:
    the turn cap can't bound time (one hung MCP call holds a turn for
    minutes), so the loop also watches the clock. 0 disables.
    """

    max_turns: int
    max_session_attempts: int
    provider_retry_attempts: int
    retry_backoff_seconds: float
    wait_default_minutes: int
    max_session_seconds: int

    @classmethod
    def from_config(cls) -> "Settings":
        e = CONFIG.engine
        return cls(
            max_turns=e.max_turns,
            max_session_attempts=e.max_attempts,
            provider_retry_attempts=e.provider_retry_attempts,
            retry_backoff_seconds=e.retry_backoff_seconds,
            wait_default_minutes=e.wait_default_minutes,
            max_session_seconds=e.max_session_seconds,
        )


@dataclass(slots=True)
class EngineRun:
    """One session's immutable wiring: provider, MCP client, tool surface,
    logging sinks, settings, and the policy set. Threaded through the loop
    instead of a parameter list so adding a dependency is a one-line change."""

    provider: Provider
    mcp: McpClient
    tool_schemas: list[dict]
    schema_by_name: dict[str, dict]
    local_registry: dict[str, LocalTool]
    tr: Trace
    rlog: RawLog
    settings: Settings
    policies: Policies
    layout_incomplete: bool = False
    # Wall-clock cutoff (time.monotonic() value) for this session, or None
    # when `settings.max_session_seconds` is 0 — see `loop.drive`.
    deadline: float | None = None
