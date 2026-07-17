"""Spawn `claude -p` when any hook triggers.

System prompt, MCP config, and the --plugin-dir skill tree are built
in-process per wake from neutral shared modules — no AGENT.md on disk.
The subprocess streams stream-json back; `session_log.py` writes every
event to the daily narrative plus the per-session artifact dir.

Engine-neutrality: this module lives under agent/claude/ and imports
freely from agent/engine/ utilities (skill discovery, MCP inventory).
The reverse direction is forbidden — agent/engine/ must not learn
about Claude Code. Deleting agent/claude/ leaves the engine intact.
"""

import asyncio
import datetime as dt
import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from physiclaw.agent.claude.plugin import prepare_plugin_dir
from physiclaw.agent.claude.session_log import _SessionLog
from physiclaw.agent.engine import jobs as engine_jobs
from physiclaw.agent.engine import screen_layout, skill
from physiclaw.agent.engine.mcp_inventory import discover_mcp_tools
from physiclaw.agent.engine.skill import Skill
from physiclaw.agent.engine.trace import new_sid
from physiclaw.agent.runtime import contract
from physiclaw.agent.runtime.hook import Trigger
from physiclaw.agent.runtime.sentinel import WAIT
from physiclaw.common import paths
from physiclaw.common.config import CONFIG, server_url
from physiclaw.common.text import read_text

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
# Post-EOF grace: a child that closed stdout but won't exit must not
# hang the runtime forever — past this, it's killed like a timeout.
EXIT_WAIT_SECONDS = 30

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
    url = server_url()
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
      4. ## Screen layout — learned input/keyboard bboxes, once first-run
         capture is complete. The `{{box}}` tokens in the built-in skill
         templates and any `§ Screen layout` reference resolve against it;
         empty (section omitted) until learned. Mirrors the native engine.
    """
    parts = [read_text(CLAUDE_MD).rstrip()]
    card = _tooling_card(mcp_tools)
    if card:
        parts.append(card)
    cat = skill.render_section(skills)
    if cat:
        parts.append(cat)
    if screen_layout.is_learned():
        layout = screen_layout.load_layout_md()
        if layout:
            parts.append(f"## Screen layout\n\n{layout}")
    else:
        # First run: no layout yet. Surface the same "[First-run setup needed]"
        # notice the native engine pins, so Claude knows to run the
        # `screen-layout` skill (which saves boxes via its screen_layout.py CLI)
        # before it tries to open apps by search or send messages.
        reminder = screen_layout.tail_reminder()
        if reminder:
            parts.append(reminder)
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
    # Point the child's `uv run` (the jobs + screen-layout skill CLIs) at the
    # venv PhysiClaw is installed in — this process's own prefix. cwd is
    # ~/.physiclaw with no project, and a `uv tool install` launch sets no
    # VIRTUAL_ENV, so without this `uv run python …` builds an empty ephemeral
    # env and `import physiclaw` fails. Overriding (not just inheriting) also
    # steers the child away from any unrelated venv active in the parent shell.
    env["VIRTUAL_ENV"] = sys.prefix
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
    """Assemble argv from pre-computed pieces. `spawn_claude` builds the
    system prompt + tool list once per wake and reuses them across
    retries (the plugin dir is the per-attempt artifact); this keeps
    `_build_cmd` pure (no side effects) and makes the retry loop cheap.

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
    # The native `screen-layout` skill saves boxes via `report_screen_layout` —
    # a local engine tool Claude doesn't have. Drop it so the claude-only
    # `screen-layout` skill (agent/claude/skills/, which saves through its
    # screen_layout.py CLI under `Bash(uv run)`) is the only one Claude sees.
    # Then fill the `{{box}}` tokens in the remaining built-in skill code
    # templates with the learned per-device bboxes (no-op until capture done).
    discovered = skill.discover()
    discovered.pop(screen_layout.SKILL_NAME, None)
    skills = screen_layout.fill_builtin_boxes(discovered)
    system_prompt = _render_system_prompt(mcp_tools, skills)
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    model_ref = f"claude-code/{model_id}"
    env = _child_env()

    async def attempt(trigs: list[Trigger], n: int) -> contract.SessionOutcome:
        # WAIT-close job detection needs the pre-attempt id set (see
        # engine_jobs.created_since) — this path has no in-process
        # created-a-job signal, unlike the native engine's session flag.
        # Snapshotted per attempt, not per wake: a job written by an
        # earlier UNDONE attempt must not suppress this attempt's WAIT
        # follow-up (a redundant check-in beats a stranded WAIT).
        jobs_before = engine_jobs.job_ids()
        # Plugin dir is the only per-attempt artifact. Fresh sid keeps
        # retries from overlapping (the random suffix matters: a retry
        # can start within the same second) and, if we ever debug one,
        # the tmp dir is uniquely labelled.
        sid = new_sid()
        plugin_dir = prepare_plugin_dir(sid, skills=skills)
        cmd = _build_cmd(
            trigs,
            plugin_dir=plugin_dir,
            system_prompt=system_prompt,
            mcp_tools=mcp_tools,
            model_id=model_id,
        )

        log.info(
            "spawning claude (attempt=%d/%d, triggers=%s) — detail log: %s",
            n,
            MAX_ATTEMPTS,
            sources,
            LOG_DIR / f"claude-{dt.datetime.now():%Y-%m-%d}.log",
        )
        status = "UNDONE"
        proc = None
        try:
            # Log construction BEFORE the spawn: if it raises, there is
            # no live claude subprocess left driving the phone unwatched.
            slog = _SessionLog(sid, trigs, model_ref=model_ref, prompt_hash=prompt_hash)
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
                result_data = await _stream(proc, slog)
                # Bounded: stdout EOF usually means exit, but a child
                # that closed stdout while alive must not hang us.
                returncode = await asyncio.wait_for(
                    proc.wait(), timeout=EXIT_WAIT_SECONDS
                )
                if returncode != 0:
                    log.error("claude exited %s (see log for details)", returncode)
                elif result_data:
                    log.info(
                        "claude done (turns=%s): %s",
                        result_data.get("num_turns", "?"),
                        str(result_data.get("result", ""))[:200],
                    )
                status = slog.done(returncode)
            except asyncio.TimeoutError:
                if proc is not None:
                    proc.kill()
                    await proc.wait()
                status = slog.done("killed")
                log.error("claude killed after %ds timeout", TIMEOUT)
            except ValueError:
                # readline raises ValueError when one stream-json line
                # exceeds the buffer limit — kill rather than orphan.
                if proc is not None:
                    proc.kill()
                    await proc.wait()
                status = slog.done("killed")
                log.error(
                    "claude killed — output line exceeded the %dMB stream buffer",
                    CONFIG.claude.stream_buffer_mb,
                )
            finally:
                # Any OTHER escape route (unexpected error, cancellation):
                # never leave a live claude with nobody reading it.
                if proc is not None and proc.returncode is None:
                    proc.kill()
                    await proc.wait()
                slog.close()
        finally:
            # Plugin dir holds only symlinks + one JSON file — no user
            # data worth keeping for post-mortem. Clean up regardless of
            # outcome so TMPDIR doesn't accumulate across retries and
            # wakes.
            shutil.rmtree(plugin_dir, ignore_errors=True)

        # `slog.done` returns a sentinel status on a clean close, else
        # "UNDONE" (crash, kill, timeout) — the contract's retry case.
        final = None if status == "UNDONE" else status
        return contract.SessionOutcome(
            status=final,
            created_job=final == WAIT and engine_jobs.created_since(jobs_before),
        )

    # A model-declared STUCK is a considered close on this path — only
    # UNDONE (no clean close at all) earns a re-spawn.
    await contract.drive(
        attempt,
        triggers,
        max_attempts=MAX_ATTEMPTS,
        wait_default_minutes=CONFIG.engine.wait_default_minutes,
        retry_on=(None,),
        retry_backoff_seconds=RETRY_BACKOFF,
    )
