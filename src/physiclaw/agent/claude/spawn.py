"""Spawn `claude -p` when any hook triggers.

System prompt, MCP config, and the --plugin-dir skill tree are built
in-process per wake from neutral shared modules — no AGENT.md on disk.
The subprocess streams stream-json back; every event is summarized to
log/claude/claude-YYYY-MM-DD.log.

Engine-neutrality: this module lives under agent/claude/ and imports
freely from agent/engine/ utilities (skill discovery, MCP inventory).
The reverse direction is forbidden — agent/engine/ must not learn
about Claude Code. Deleting agent/claude/ leaves the engine intact.
"""

import asyncio
import datetime as dt
import json
import logging
import os
import shutil
from pathlib import Path

from physiclaw import paths
from physiclaw.agent.claude.plugin import prepare_plugin_dir
from physiclaw.agent.engine import skill
from physiclaw.agent.engine.trace import new_sid
from physiclaw.agent.engine.mcp_inventory import discover_mcp_tools
from physiclaw.text import read_text
from physiclaw.agent.engine.skill import Skill
from physiclaw.agent.runtime.hook import Trigger
from physiclaw.agent.runtime.sentinel import parse_sentinel
from physiclaw.config import CONFIG
from physiclaw.core.logger.retention import purge_daily_logs

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
CLAUDE_MD = _HERE.parent / "CLAUDE.md"
LOG_DIR = paths.claude_log_dir()
# cwd for the spawned `claude` subprocess — PHYSICLAW_HOME, our data
# root. A pip install has no repo root to inherit, and the `Write(memory/*)`
# allowlist + CLAUDE.md's `memory/memory.md` references both resolve
# relative to this cwd.
#
# Side effect: `claude -p` auto-loads `CLAUDE.md` and discovers
# `.claude/skills/` walking up from cwd, so a stray file here would
# silently mix into the doctrine we just appended.
# `_warn_stray_context()` runs on every spawn and logs any drift.
PROJECT_ROOT = paths.HOME

TIMEOUT = CONFIG.claude.timeout_seconds  # per-line inactivity timeout
MAX_ATTEMPTS = CONFIG.claude.max_attempts
RETRY_BACKOFF = CONFIG.claude.retry_backoff_seconds

# --- Tool permissions ---
#
# Filesystem scope for the child is deliberately narrow: read/write/edit
# restricted to `memory/**` (with `jobs/**` readable only via the jobs
# skill scripts, not Claude's Read tool). Bash is narrowed to `python*`
# so the jobs skill's CLI wrappers can run, without handing the child a
# general shell — no `curl`, no `rm`, no `ssh`, no `cat /etc/...`.
#
# Everything else:
#   Phone control → physiclaw MCP (added dynamically below).
#   Skill bodies  → via Skill tool + --plugin-dir, not Read.
#   Jobs         → force through the jobs skill; scripts import engine
#                   modules directly and bypass the Claude tool layer,
#                   so tight allowlisting here is compatible with full
#                   job-format access under the covers.
#
# Patterns are relative to cwd = PROJECT_ROOT (~/.physiclaw). CLAUDE.md
# instructs the model to use relative paths so absolute-path drift
# doesn't sneak past the pattern match.
_ALLOWED_STATIC = [
    # Read-only into the data root — narrow to memory/ only.
    "Read(memory/**)",
    "Glob(memory/**)",
    "Grep(memory/**)",
    # Memory is the only mutable on-disk surface Claude has direct access to.
    "Write(memory/**)",
    "Edit(memory/**)",
    # `uv run` only — matches the project's install discipline
    # (PhysiClaw installs via `uv tool install`), and keeps the child
    # off bare `python` that could pick up the wrong interpreter.
    # Narrow to `uv run` rather than all of `uv` so `uv pip`, `uv sync`,
    # `uv add`, and other env-mutating subcommands stay out of reach.
    "Bash(uv run:*)",
    # Skill tool (plugin-dir skills); individual skills can still be
    # denied via Skill(<name>) in _DISALLOWED.
    "Skill",
]
_DISALLOWED = [
    # Belt-and-suspenders on jobs mutation: even if a future allowlist
    # edit loosens Write/Edit, these explicit denials still block direct
    # jobs.md edits. The cron parser is regex-based; one malformed field
    # breaks every scheduled job.
    "Write(jobs/**)",
    "Edit(jobs/**)",
    # Hardware setup skills are interactive — not something to run
    # inside an autonomous wake.
    "Skill(setup)",
    "Skill(phone-setup)",
    "Skill(calibrate-keyboard)",
    "Skill(setup-vision-models)",
]


def _mcp_tools() -> list[dict]:
    """MCP tools with the Claude-prefixed name + first-line description.

    Single source — callers either use the full dict (for the tooling
    card) or pick `.name` out for the `--allowedTools` list. Calling
    `discover_mcp_tools()` once per spawn beats three times.
    """
    return [
        {
            "name": f"mcp__physiclaw__{t['name']}",
            "description": (t.get("description") or "").split("\n", 1)[0].strip(),
        }
        for t in discover_mcp_tools()
    ]


def _mcp_config() -> str:
    url = os.environ.get("PHYSICLAW_SERVER", "http://127.0.0.1:8048")
    return json.dumps(
        {"mcpServers": {"physiclaw": {"type": "http", "url": f"{url}/mcp"}}}
    )


def _render_system_prompt(mcp_tools: list[dict], skills: dict[str, Skill]) -> str:
    """Compose the system prompt appended to Claude's own.

    Layout:
      1. CLAUDE.md body — hand-authored Claude-idiomatic doctrine.
      2. ## Tooling — one-line-per-tool MCP catalog (Claude otherwise
         discovers tools only from the `tools=` payload; the card is a
         redundant anchor that helps tool recall, and surfaces names
         with their `mcp__physiclaw__` prefix so the model writes them
         correctly the first time).
      3. ## Available skills — merged metadata from the plugin dir's
         content; written as Tier-1 triggers so Claude knows which
         skill to invoke before acting in which app.
    """
    parts = [read_text(CLAUDE_MD).rstrip()]
    card = _tooling_card(mcp_tools)
    if card:
        parts.append(card)
    cat = skill.render_section(skills)
    if cat:
        parts.append(cat)
    return "\n\n".join(parts)


def _tooling_card(tools: list[dict]) -> str:
    if not tools:
        return ""
    lines = [
        "## Tooling",
        "Phone control is on the physiclaw MCP server. Names are "
        "case-sensitive; call them with the `mcp__physiclaw__` prefix "
        "shown below.",
        "",
    ]
    for t in tools:
        lines.append(f"- **{t['name']}** — {t['description']}")
    return "\n".join(lines)


def _build_trigger_prompt(triggers: list[Trigger]) -> str:
    lines = ["The following events were detected:"]
    for t in triggers:
        tag = f"[{t.source}] " if t.source else ""
        lines.append(f"- {tag}{t.description}")
    lines.append("\nFollow the Loop in CLAUDE.md to decide what to do next.")
    # Claude Code interprets "think" / "think hard" / "ultrathink" as
    # thinking-budget triggers. Keep it to "think" — observe→decide→tap
    # loops don't need deep reasoning. Bump per-trigger if needed.
    lines.append("think")
    return "\n".join(lines)


# --- Logging ---


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
        purge_daily_logs(LOG_DIR, "claude", CONFIG.retention.log_days)
        self._date = dt.datetime.now().strftime("%Y-%m-%d")
        self._last_text = ""  # most recent assistant text block, for sentinel check
        self._f = open(
            LOG_DIR / f"claude-{self._date}.log",
            "a",
            encoding="utf-8",
            newline="\n",
        )
        self._f.write(f"\n{'=' * 60}\n")
        self._write(f"WAKE triggers={sources}")

    def event(self, data: dict) -> dict | None:
        """Log a stream-json event. Returns the data if it's a result.

        Every event is summarized to the daily file. Assistant text
        chunks additionally forward a one-line narration to the runtime
        logger so the operator sees Claude's reasoning / intent live
        without tailing the detail log.
        """
        summary = self._summarize(data)
        if summary:
            self._write(summary)
        self._forward_to_runtime(data)
        return data if data.get("type") == "result" else None

    def raw(self, text: str) -> None:
        self._write(f"raw: {text[:500]}")

    def done(self, returncode: int | str) -> str:
        """Write OUTCOME + EXIT bookends. Returns the OUTCOME status.

        Trust the sentinel only when the process exited cleanly (code 0);
        otherwise the run crashed even if the agent claimed DONE earlier.
        """
        last_line = next(
            (line for line in reversed(self._last_text.splitlines()) if line.strip()),
            "",
        )
        status, recap = parse_sentinel(last_line) if returncode == 0 else (None, "")
        if not status:
            status = "UNDONE"
            recap = (last_line or "(no text)").strip()[:200]
        self._write(f"OUTCOME: {status} - {recap}")
        self._write(f"EXIT code={returncode}")
        self._f.write(f"{'=' * 60}\n\n")
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
            self._f = open(
                LOG_DIR / f"claude-{today}.log",
                "a",
                encoding="utf-8",
                newline="\n",
            )
            self._f.write(
                f"\n[{now:%H:%M:%S}] ROLLOVER ← continued from previous day\n"
            )
        self._f.write(f"[{now:%H:%M:%S}] {msg}\n")
        self._f.flush()

    def _forward_to_runtime(self, data: dict) -> None:
        """Forward the high-signal subset of events to runtime stderr so
        the daemon log is followable without tailing the detail file.

        Only assistant TEXT blocks are forwarded — tool_use / tool_result
        are already visible in the MCP server's own log, and the final
        `result` event is logged from `spawn_claude`'s exit path.
        """
        if data.get("type") != "assistant":
            return
        for b in data.get("message", {}).get("content", []):
            if b.get("type") == "text" and b.get("text", "").strip():
                first = b["text"].strip().splitlines()[0][:200]
                log.info("claude: %s", first)
                return

    def _summarize(self, data: dict) -> str | None:
        t = data.get("type", "")

        if t == "assistant":
            parts = []
            for b in data.get("message", {}).get("content", []):
                if b.get("type") == "tool_use":
                    parts.append(
                        f"tool_use: {b['name']} {str(b.get('input', ''))[:1000]}"
                    )
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


# --- Environment sanitization ---

# Env-var prefixes stripped from the child's environment before spawn.
# Rationale: a user-level shell config (e.g. `.zshrc` exporting
# ANTHROPIC_API_KEY or CLAUDE_CONFIG_DIR) silently changes where the
# child looks for config, auth, and telemetry — something we can't see
# from this side. Wipe the whole namespace so our flags are the sole
# source of truth.
#
# Not stripped: HOME, PATH, LANG, TERM, and the PHYSICLAW_* vars our
# own tools read.
_ENV_STRIP_PREFIXES = ("ANTHROPIC_", "CLAUDE_", "OTEL_")


def _child_env() -> dict[str, str]:
    """Return the env the `claude` subprocess should inherit.

    Strips CLAUDE_* / ANTHROPIC_* / OTEL_* so inherited shell config
    can't redirect the child, and pins PWD to our cwd so any tool in
    the child that trusts $PWD over getcwd() still sees our anchor
    (Claude Code itself uses getcwd, but this is a cheap hedge).
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(k.startswith(p) for p in _ENV_STRIP_PREFIXES)
    }
    env["PWD"] = str(PROJECT_ROOT)
    return env


# --- Context-pollution guard ---


def _warn_stray_context() -> None:
    """Log a warning if stray `CLAUDE.md` or `.claude/` lives inside
    PROJECT_ROOT. Claude Code auto-loads those from cwd + ancestors, so
    anything there silently joins our `--append-system-prompt` doctrine.

    Scope: only PROJECT_ROOT itself (~/.physiclaw). `~/CLAUDE.md` and
    `~/.claude/` are the user's across-all-invocations config — their
    intent, not our concern.
    """
    for name in ("CLAUDE.md", ".claude"):
        stray = PROJECT_ROOT / name
        if stray.exists():
            log.warning(
                "stray %s — `claude -p` auto-loads this and mixes it with our "
                "system prompt. Move or delete it to keep the spawn deterministic.",
                stray,
            )


# --- Main ---

_CLAUDE_ALIASES = ("opus", "sonnet", "haiku")


def _normalize_claude_model_id(model_id: str) -> str:
    """Coerce common user-input forms into what `claude --model` accepts.

    The CLI accepts bare aliases (`opus` / `sonnet` / `haiku`) and full
    ids prefixed with `claude-` (`claude-opus-4-7`, `claude-haiku-4-5-20251001`).
    It rejects de-prefixed forms (`opus-4-7`, `sonnet-4-6`) — a common
    mistake when users copy a version off Anthropic's docs without the
    brand prefix. Add it back so an obvious typo doesn't cost a wake."""
    if model_id in _CLAUDE_ALIASES or model_id.startswith("claude-"):
        return model_id
    return f"claude-{model_id}"


def _build_cmd(
    triggers: list[Trigger],
    *,
    plugin_dir: Path,
    system_prompt: str,
    mcp_tools: list[dict],
    model_id: str,
) -> list[str]:
    """Assemble argv from pre-computed pieces. Callers (spawn_claude,
    preview) build the plugin dir + system prompt + tool list once per
    wake and reuse them; this keeps `_build_cmd` pure (no side effects)
    and makes the retry loop cheap.

    `model_id` is the second segment of `[agent] model = "claude-code/<id>"`,
    normalized via `_normalize_claude_model_id` before passing as `--model`."""
    if not CLAUDE_MD.exists():
        raise FileNotFoundError(f"CLAUDE.md not found: {CLAUDE_MD}")
    allowed = [t["name"] for t in mcp_tools] + _ALLOWED_STATIC
    return [
        "claude",
        "-p",
        _build_trigger_prompt(triggers),
        "--model",
        _normalize_claude_model_id(model_id),
        "--append-system-prompt",
        system_prompt,
        "--plugin-dir",
        str(plugin_dir),
        "--setting-sources",
        "user",
        "--permission-mode",
        "acceptEdits",
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--strict-mcp-config",
        "--mcp-config",
        _mcp_config(),
        "--allowedTools",
        ",".join(allowed),
        "--disallowedTools",
        ",".join(_DISALLOWED),
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


async def spawn_claude(triggers: list[Trigger], *, model_id: str) -> None:
    sources = [t.source or "?" for t in triggers]
    _warn_stray_context()

    # Hoisted out of the retry loop — none of this varies per attempt.
    # Skills and MCP tools are scanned once; the rendered prompt + env
    # are identical across UNDONE retries.
    mcp_tools = _mcp_tools()
    skills = skill.discover()
    system_prompt = _render_system_prompt(mcp_tools, skills)
    env = _child_env()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            log.warning(
                "retry %d/%d after %.0fs backoff", attempt, MAX_ATTEMPTS, RETRY_BACKOFF
            )
            await asyncio.sleep(RETRY_BACKOFF)

        # Plugin dir is the only per-attempt artifact. Fresh sid keeps
        # retries from overlapping (the random suffix matters: a retry
        # can start within the same second) and, if we ever debug one,
        # the tmp dir is uniquely labelled.
        sid = new_sid()
        plugin_dir = prepare_plugin_dir(sid, skills=skills)
        cmd = _build_cmd(
            triggers,
            plugin_dir=plugin_dir,
            system_prompt=system_prompt,
            mcp_tools=mcp_tools,
            model_id=model_id,
        )

        log.info(
            "spawning claude (attempt=%d/%d, triggers=%s) — detail log: %s",
            attempt,
            MAX_ATTEMPTS,
            sources,
            LOG_DIR / f"claude-{dt.datetime.now():%Y-%m-%d}.log",
        )
        status = "UNDONE"
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env=env,
                # Default 64KB readline limit blows up on screenshot base64 lines.
                limit=CONFIG.claude.stream_buffer_mb * 1024 * 1024,
            )
            slog = _SessionLog(sources)
            try:
                result_data = await _stream(proc, slog)
                await proc.wait()
                if proc.returncode != 0:
                    log.error("claude exited %s (see log for details)", proc.returncode)
                elif result_data:
                    log.info(
                        "claude done (turns=%s): %s",
                        result_data.get("num_turns", "?"),
                        str(result_data.get("result", ""))[:200],
                    )
                status = slog.done(proc.returncode)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                status = slog.done("killed")
                log.error("claude killed after %ds timeout", TIMEOUT)
            finally:
                slog.close()
        finally:
            # Plugin dir holds only symlinks + one JSON file — no user
            # data worth keeping for post-mortem. Clean up regardless of
            # outcome so TMPDIR doesn't accumulate across retries and
            # wakes.
            shutil.rmtree(plugin_dir, ignore_errors=True)

        if status != "UNDONE":
            return  # DONE, STUCK, IDLE, or WAIT — agent finished cleanly, no retry

    log.error("giving up after %d UNDONE attempts", MAX_ATTEMPTS)
