"""Spawn `claude -p` when any hook triggers.

Streams tool calls and responses to log/claude/claude-YYYY-MM-DD.log.
"""

import ast
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


def _is_mcp_tool_decorator(dec: ast.expr) -> bool:
    """True for `@mcp.tool` or `@mcp.tool(...)`."""
    call = dec.func if isinstance(dec, ast.Call) else dec
    return (
        isinstance(call, ast.Attribute)
        and call.attr == "tool"
        and isinstance(call.value, ast.Name)
        and call.value.id == "mcp"
    )


def _discover_mcp_tools() -> list[str]:
    """Auto-detect MCP tool names from @mcp.tool decorators in tools.py."""
    try:
        tree = ast.parse(TOOLS_PY.read_text())
    except FileNotFoundError:
        return []
    # ast.walk (not tree.body) because tools are nested inside register().
    names = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(_is_mcp_tool_decorator(d) for d in node.decorator_list)
    ]
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
    # Claude Code interprets "think" / "think hard" / "ultrathink" in the
    # prompt as thinking-budget triggers. Keep it to "think" — tasks here
    # are observe→decide→tap loops, not deep reasoning. Bump for harder
    # tasks per-trigger if needed.
    lines.append("think")
    return "\n".join(lines)


# --- Logging ---

SENTINEL = re.compile(r">+ (DONE|STUCK|IDLE|WAIT) - (.+)")


def _redact_images(content):
    """Replace base64 image data with a length placeholder so logs stay readable."""
    if not isinstance(content, list):
        return content
    out = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "image":
            src = item.get("source") or {}
            data = src.get("data", "")
            out.append({**item, "source": {**src, "data": f"<{len(data)}b elided>"}})
        else:
            out.append(item)
    return out


class _SessionLog:
    """Append-only log for a single claude session to a daily file."""

    def __init__(self, sources: list[str]):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._date = dt.datetime.now().strftime("%Y-%m-%d")
        self._last_text = ""  # most recent assistant text block, for sentinel check
        self._f = open(LOG_DIR / f"claude-{self._date}.log", "a")
        self._f.write(f"\n{'='*60}\n")
        self._write(f"WAKE triggers={sources}")

    def event(self, data: dict) -> dict | None:
        """Log a stream-json event. Returns the data if it's a result."""
        summary = self._summarize(data)
        if summary:
            self._write(summary)
        return data if data.get("type") == "result" else None

    def raw(self, text: str) -> None:
        self._write(f"raw: {text[:500]}")

    def done(self, returncode: int | str) -> str:
        """Write OUTCOME + EXIT bookends. Returns the OUTCOME status.

        Trust the sentinel only when the process exited cleanly (code 0);
        otherwise the run crashed even if the agent claimed DONE earlier.
        """
        last_line = next(
            (l for l in reversed(self._last_text.splitlines()) if l.strip()), ""
        )
        m = SENTINEL.match(last_line.strip()) if returncode == 0 else None
        if m:
            status, recap = m.group(1), m.group(2).strip()
        else:
            status = "UNDONE"
            recap = (last_line or "(no text)").strip()[:200]
        self._write(f"OUTCOME: {status} - {recap}")
        self._write(f"EXIT code={returncode}")
        self._f.write(f"{'='*60}\n\n")
        return status

    def close(self) -> None:
        self._f.close()

    def _write(self, msg: str) -> None:
        now = dt.datetime.now()
        today = now.strftime("%Y-%m-%d")
        if today != self._date:
            # Crossed midnight — close current file, continue in today's file.
            # Markers in both files let a reader follow the session across days.
            self._f.write(f"[{now:%H:%M:%S}] ROLLOVER → claude-{today}.log\n")
            self._f.flush()
            self._f.close()
            self._date = today
            self._f = open(LOG_DIR / f"claude-{today}.log", "a")
            self._f.write(f"\n[{now:%H:%M:%S}] ROLLOVER ← continued from previous day\n")
        self._f.write(f"[{now:%H:%M:%S}] {msg}\n")
        self._f.flush()

    def _summarize(self, data: dict) -> str | None:
        t = data.get("type", "")

        if t == "assistant":
            parts = []
            for b in data.get("message", {}).get("content", []):
                if b.get("type") == "tool_use":
                    parts.append(f"tool_use: {b['name']} {str(b.get('input', ''))[:1000]}")
                elif b.get("type") == "text" and b.get("text", "").strip():
                    self._last_text = b["text"]  # for sentinel check in done()
                    parts.append(f"text: {b['text'][:1000]}")
                elif b.get("type") == "thinking" and b.get("thinking", "").strip():
                    parts.append(f"thinking: {b['thinking'][:2000]}")
            return " | ".join(parts) if parts else None

        if t == "user":
            for b in data.get("message", {}).get("content", []):
                if b.get("type") == "tool_result":
                    return f"tool_result: {str(_redact_images(b.get('content', '')))[:1000]}"

        if t == "result":
            return f"result: turns={data.get('num_turns', '?')} {str(data.get('result', ''))[:2000]}"

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


MAX_ATTEMPTS = 3
RETRY_BACKOFF = 5.0  # seconds between retry attempts


async def spawn_claude(triggers: list[Trigger]) -> None:
    sources = [t.source or "?" for t in triggers]
    cmd = _build_cmd(triggers)  # identical across retries — read tools.py once

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            log.warning("retry %d/%d after %.0fs backoff", attempt, MAX_ATTEMPTS, RETRY_BACKOFF)
            await asyncio.sleep(RETRY_BACKOFF)

        log.info("spawning claude (attempt=%d/%d, triggers=%s)", attempt, MAX_ATTEMPTS, sources)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            limit=STREAM_BUFFER,
        )

        slog = _SessionLog(sources)
        status = "UNDONE"
        try:
            result_data = await _stream(proc, slog)
            await proc.wait()
            if proc.returncode != 0:
                log.error("claude exited %s (see log for details)", proc.returncode)
            elif result_data:
                log.info("claude done (turns=%s): %s",
                         result_data.get("num_turns", "?"),
                         str(result_data.get("result", ""))[:200])
            status = slog.done(proc.returncode)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            status = slog.done("killed")
            log.error("claude killed after %ds timeout", TIMEOUT)
        finally:
            slog.close()

        if status != "UNDONE":
            return  # DONE, STUCK, IDLE, or WAIT — agent finished cleanly, no retry

    log.error("giving up after %d UNDONE attempts", MAX_ATTEMPTS)
