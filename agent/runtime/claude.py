"""Spawn `claude -p` when any hook triggers.

Streams tool calls and responses to log/claude/claude-YYYY-MM-DD.log.
"""

import asyncio
import datetime as dt
import json
import logging
import os
import re
from pathlib import Path

from agent.runtime.hook import Trigger

log = logging.getLogger(__name__)

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLAUDE_MD = PROJECT_ROOT / ".claude" / "CLAUDE.md"
TOOLS_PY = PROJECT_ROOT / "physiclaw" / "server" / "tools.py"
LOG_DIR = PROJECT_ROOT / "log" / "claude"

TIMEOUT = 180  # 3min, per-line inactivity timeout
STREAM_BUFFER = 10 * 1024 * 1024  # 10MB readline limit (default 64KB blows up on screenshot base64)

# --- Tool permissions ---
_ALLOWED = [
    "Read", "Glob", "Grep", "Skill",
    "Write(memory/*)", "Write(jobs/*)",
    "Edit(memory/*)", "Edit(jobs/*)",
]
_DISALLOWED = [
    "Skill(setup)", "Skill(phone-setup)",
    "Skill(calibrate-keyboard)", "Skill(setup-vision-models)",
]


def _discover_mcp_tools() -> list[str]:
    """Auto-detect MCP tool names from @mcp.tool decorators in tools.py."""
    if not TOOLS_PY.exists():
        return []
    names = re.findall(
        r"@mcp\.tool(?:\([^)]*\))?\s*\n(?:\s*#[^\n]*\n)*\s+(?:async\s+)?def\s+(\w+)\(",
        TOOLS_PY.read_text(),
    )
    return [f"mcp__physiclaw__{n}" for n in names]


def _mcp_config() -> str:
    url = os.environ.get("PHYSICLAW_SERVER", "http://127.0.0.1:8048")
    return json.dumps({"mcpServers": {"physiclaw": {"type": "http", "url": f"{url}/mcp"}}})


def _build_prompt(triggers: list[Trigger]) -> str:
    lines = ["The following events were detected:"]
    for t in triggers:
        tag = f"[{t.source}] " if t.source else ""
        lines.append(f"- {tag}{t.description}")
    lines.append("\nFollow the Loop workflow to decide what to do next.")
    return "\n".join(lines)


# --- Logging ---

class _SessionLog:
    """Append-only log for a single claude session to a daily file."""

    def __init__(self, sources: list[str]):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._f = open(LOG_DIR / f"claude-{dt.datetime.now():%Y-%m-%d}.log", "a")
        self._f.write(f"\n{'='*60}\n")
        self._write(f"SPAWN triggers={sources}")

    def event(self, data: dict) -> dict | None:
        """Log a stream-json event. Returns the data if it's a result."""
        summary = self._summarize(data)
        if summary:
            self._write(summary)
        return data if data.get("type") == "result" else None

    def raw(self, text: str) -> None:
        self._write(f"raw: {text[:500]}")

    def done(self, returncode: int | str) -> None:
        self._write(f"DONE exit={returncode}")
        self._f.write(f"{'='*60}\n\n")

    def close(self) -> None:
        self._f.close()

    def _write(self, msg: str) -> None:
        self._f.write(f"[{dt.datetime.now():%H:%M:%S}] {msg}\n")
        self._f.flush()

    @staticmethod
    def _summarize(data: dict) -> str | None:
        t = data.get("type", "")

        if t == "assistant":
            parts = []
            for b in data.get("message", {}).get("content", []):
                if b.get("type") == "tool_use":
                    parts.append(f"tool_use: {b['name']} {str(b.get('input', ''))[:200]}")
                elif b.get("type") == "text" and b.get("text", "").strip():
                    parts.append(f"text: {b['text'][:300]}")
            return " | ".join(parts) if parts else None

        if t == "user":
            for b in data.get("message", {}).get("content", []):
                if b.get("type") == "tool_result":
                    return f"tool_result: {str(b.get('content', ''))[:200]}"

        if t == "result":
            return f"result: turns={data.get('num_turns', '?')} {str(data.get('result', ''))[:300]}"

        return None


# --- Main ---

def _build_cmd(triggers: list[Trigger]) -> list[str]:
    if not CLAUDE_MD.exists():
        raise FileNotFoundError(f"CLAUDE.md not found: {CLAUDE_MD}")
    allowed = _discover_mcp_tools() + _ALLOWED
    return [
        "claude",
        "-p", _build_prompt(triggers),
        "--permission-mode", "acceptEdits",
        "--output-format", "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--strict-mcp-config",
        "--mcp-config", _mcp_config(),
        "--allowedTools", ",".join(allowed),
        "--disallowedTools", ",".join(_DISALLOWED),
        "--append-system-prompt-file", str(CLAUDE_MD),
    ]


async def _stream(proc, slog: _SessionLog) -> dict | None:
    """Read stream-json lines until EOF. Returns the result event or None."""
    result_data = None
    while True:
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=TIMEOUT)
        if not line:
            break
        text = line.decode(errors="replace").strip()
        if not text:
            continue
        try:
            result_data = slog.event(json.loads(text)) or result_data
        except json.JSONDecodeError:
            slog.raw(text)
    return result_data


async def spawn_claude(triggers: list[Trigger]) -> None:
    cmd = _build_cmd(triggers)
    sources = [t.source or "?" for t in triggers]
    log.info("spawning claude (triggers=%s)", sources)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        limit=STREAM_BUFFER,
    )

    slog = _SessionLog(sources)
    try:
        result_data = await _stream(proc, slog)
        await proc.wait()

        if proc.returncode != 0:
            log.error("claude exited %s (see log for details)", proc.returncode)
        elif result_data:
            log.info("claude done (turns=%s): %s",
                     result_data.get("num_turns", "?"),
                     str(result_data.get("result", ""))[:200])
        slog.done(proc.returncode)

    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        slog.done("killed")
        log.error("claude killed after %ds timeout", TIMEOUT)
    finally:
        slog.close()
