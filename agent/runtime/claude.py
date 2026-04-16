"""Spawn Claude Code when any hook triggers.

Not a hook itself — this is the default `react` callable passed to
`Runtime`. The loop calls `spawn_claude(triggers)` whenever
`check_hooks()` returns a non-empty list, and the prompt is built from
the triggers that fired so Claude knows what woke it up and from where.

Runs `claude -p <prompt>` with explicit context:
  --strict-mcp-config          Only use MCP servers we specify (not user config).
  --mcp-config                 Running PhysiClaw server (streamable-http).
  --allowedTools               MCP tools (auto-detected) + file tools + Skill.
  --permission-mode            acceptEdits — no interactive prompts.
  --append-system-prompt-file  CLAUDE.md workflow (if present).
  --output-format stream-json  Streams tool calls and responses as they happen.
  --verbose                    Required for stream-json in print mode.

Logs every step to log/claude/claude-YYYY-MM-DD.log.
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

CLAUDE_BIN = "claude"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLAUDE_MD = PROJECT_ROOT / ".claude" / "CLAUDE.md"
TOOLS_PY = PROJECT_ROOT / "physiclaw" / "server" / "tools.py"
LOG_DIR = PROJECT_ROOT / "log" / "claude"
TIMEOUT = 600  # 10 min hard kill

# Built-in tools the agent may use.  Read/Glob/Grep are unrestricted;
# Write/Edit are scoped to memory/ and jobs/ only.
_BUILTIN_TOOLS = [
    "Read", "Glob", "Grep", "Skill",
    "Write(memory/*)", "Write(jobs/*)",
    "Edit(memory/*)", "Edit(jobs/*)",
]


def _open_log() -> tuple[Path, object]:
    """Open today's log file for appending. Returns (path, file handle)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"claude-{dt.datetime.now():%Y-%m-%d}.log"
    return path, open(path, "a")


def _log_line(f, msg: str) -> None:
    """Write a timestamped line to the log file."""
    ts = dt.datetime.now().strftime("%H:%M:%S")
    f.write(f"[{ts}] {msg}\n")
    f.flush()


def _summarize_event(data: dict) -> str | None:
    """Turn a stream-json event into a short log line, or None to skip."""
    t = data.get("type", "")

    if t == "assistant":
        parts = []
        for block in data.get("message", {}).get("content", []):
            btype = block.get("type")
            if btype == "tool_use":
                inp = str(block.get("input", ""))
                if len(inp) > 200:
                    inp = inp[:200] + "..."
                parts.append(f"tool_use: {block.get('name')} {inp}")
            elif btype == "text":
                text = block.get("text", "")
                if text.strip():
                    parts.append(f"text: {text[:300]}")
        return " | ".join(parts) if parts else None

    if t == "user":
        # tool_result — log tool name and truncated output
        for block in data.get("message", {}).get("content", []):
            if block.get("type") == "tool_result":
                content = str(block.get("content", ""))
                if len(content) > 200:
                    content = content[:200] + "..."
                return f"tool_result: {content}"
        return None

    if t == "result":
        result = data.get("result", "")
        turns = data.get("num_turns", "?")
        duration = data.get("duration_ms", "?")
        return f"result: turns={turns} duration={duration}ms {str(result)[:300]}"

    return None


def _discover_mcp_tools() -> list[str]:
    """Parse tool names from tools.py via regex on @mcp.tool decorated fns."""
    if not TOOLS_PY.exists():
        log.warning("tools.py not found at %s — no MCP tools allowed", TOOLS_PY)
        return []
    source = TOOLS_PY.read_text()
    # Handles @mcp.tool(), @mcp.tool, @mcp.tool(name=..., ...),
    # with optional blank/comment lines between decorator and def.
    names = re.findall(
        r"@mcp\.tool(?:\([^)]*\))?\s*\n(?:\s*#[^\n]*\n)*\s+(?:async\s+)?def\s+(\w+)\(",
        source,
    )
    return [f"mcp__physiclaw__{n}" for n in names]


def _mcp_config_json() -> str:
    """Build an MCP config JSON string pointing at the running server."""
    server_url = os.environ.get("PHYSICLAW_SERVER", "http://127.0.0.1:8048")
    return json.dumps({
        "mcpServers": {
            "physiclaw": {
                "type": "http",
                "url": f"{server_url}/mcp",
            }
        }
    })


def _build_prompt(triggers: list[Trigger]) -> str:
    lines = ["The following events were detected:"]
    for t in triggers:
        tag = f"[{t.source}] " if t.source else ""
        lines.append(f"- {tag}{t.description}")
    lines.append("")
    lines.append("Follow the Loop workflow to decide what to do next.")
    return "\n".join(lines)


async def spawn_claude(triggers: list[Trigger]) -> None:
    prompt = _build_prompt(triggers)
    allowed = _discover_mcp_tools() + _BUILTIN_TOOLS

    cmd = [
        CLAUDE_BIN,
        "-p", prompt,
        "--permission-mode", "acceptEdits",
        "--output-format", "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--strict-mcp-config",
        "--mcp-config", _mcp_config_json(),
        "--allowedTools", ",".join(allowed),
    ]

    if CLAUDE_MD.exists():
        cmd.extend(["--append-system-prompt-file", str(CLAUDE_MD)])

    sources = [t.source or "?" for t in triggers]
    log.info("spawning claude (triggers=%s)", sources)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(PROJECT_ROOT),
    )

    _, f = _open_log()
    _log_line(f, f"--- spawn (triggers={sources}) ---")

    try:
        result_data = None
        while True:
            line = await asyncio.wait_for(
                proc.stdout.readline(), timeout=TIMEOUT
            )
            if not line:
                break
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            try:
                data = json.loads(text)
                summary = _summarize_event(data)
                if summary:
                    _log_line(f, summary)
                if data.get("type") == "result":
                    result_data = data
            except json.JSONDecodeError:
                _log_line(f, f"raw: {text[:500]}")

        await proc.wait()

        if proc.returncode != 0:
            stderr = (await proc.stderr.read()).decode(errors="replace").strip()
            _log_line(f, f"ERROR exit={proc.returncode}: {stderr}")
            log.error("claude exited %s: %s", proc.returncode, stderr)
        elif result_data:
            text = result_data.get("result", "")
            turns = result_data.get("num_turns", "?")
            log.info("claude done (turns=%s): %s", turns, text[:200])
        _log_line(f, f"--- done (exit={proc.returncode}) ---\n")

    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        _log_line(f, f"TIMEOUT after {TIMEOUT}s")
        _log_line(f, f"--- done (exit=killed) ---\n")
        log.error("claude killed after %ds timeout", TIMEOUT)
    finally:
        f.close()
