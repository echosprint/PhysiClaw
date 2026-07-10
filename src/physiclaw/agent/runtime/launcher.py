"""Runtime launcher — wires args + env + engine choice, starts the Runtime loop.

Invoked via `python -m physiclaw.agent.runtime` (the package's `__main__.py`
shim imports `launch` from here).

Spawned as a subprocess by `physiclaw.main` so the hook loop runs
out-of-process from the MCP server. This isolates long-running hook work
(claude -p subprocess, or the in-process engine) from the FastMCP event
loop.

Two engines, one hook loop, one config knob:

  - `physiclaw`   — in-process tool-call loop (agent/engine/). Drives one
                    of the providers in agent/provider/.
  - `claude-code` — subprocess to Anthropic's `claude` CLI. Lives under
                    agent/claude/. Loaded lazily — if the package is
                    deleted, any `claude-code/...` ref errors out; the
                    engine path keeps working.

Single config field `[agent] model` (or `PHYSICLAW_MODEL` env) selects
the whole stack via a `provider/model` ref:
  qwen/qwen3.6-plus            → physiclaw engine + Qwen
  claude-code/claude-sonnet-4-6 → agent/claude/ subprocess

Each in-process provider reads its credential at startup (e.g.
QWEN_API_KEY env var or [provider] qwen_api_key in config.toml) and
fails loudly if missing.
"""

import argparse
import asyncio
import logging
import os
import sys
from functools import partial

from physiclaw.agent.engine.mcp_tool import close_mcp
from physiclaw.agent.provider import (
    CLAUDE_CODE_ID,
    in_process_provider_ids,
)
from physiclaw.agent.runtime import Runtime
from physiclaw.config import model_ref_with_source, parse_model_ref
from physiclaw import paths
from physiclaw.core.logger import setup_logging

log = logging.getLogger(__name__)


def _claude_available() -> bool:
    """Whether the claude-code subprocess engine is installed.

    Pure availability check via importlib — does not execute the
    package, so a missing plugin dir or other side-effect failure is
    surfaced later when `spawn_claude` actually imports it."""
    from importlib.util import find_spec
    return find_spec("physiclaw.agent.claude") is not None


def engine_label(ref: str) -> str:
    """Human-readable engine label for a model ref. `claude-code` is a
    whole engine (the subprocess runner); other providers run inside the
    in-process physiclaw engine."""
    provider_id, model_id = parse_model_ref(ref)
    if provider_id == CLAUDE_CODE_ID:
        return f"engine=claude-code, model={model_id}"
    return f"engine=physiclaw, provider={provider_id}, model={model_id}"


def resolve() -> tuple[str, str]:
    """Return `(ref, source)` for the active model ref.

    `source` describes where the value came from so log lines and error
    messages can point users at the right knob. Validates that the
    provider id is selectable; the model id is passed through verbatim
    — provider APIs reject unknown ids on the first chat."""
    ref, source = model_ref_with_source()  # raises if not configured
    provider_id, _ = parse_model_ref(ref)
    known_in_process = in_process_provider_ids()
    if provider_id == CLAUDE_CODE_ID:
        if not _claude_available():
            raise RuntimeError(
                f"model ref {ref!r} (from {source}) selects claude-code "
                "but agent/claude/ is not installed."
            )
    elif provider_id not in known_in_process:
        choices = (*known_in_process,) + ((CLAUDE_CODE_ID,) if _claude_available() else ())
        raise RuntimeError(
            f"unknown provider {provider_id!r} in ref {ref!r} (from {source}); "
            f"known: {choices}"
        )
    return ref, source


def launch() -> None:
    parser = argparse.ArgumentParser(description="PhysiClaw runtime loop")
    parser.add_argument("--server", default="http://127.0.0.1:8048")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    try:
        ref, source = resolve()
    except RuntimeError as e:
        # Friendly exit: print the (already actionable) message to stderr
        # and bail. No Python stack trace bleeds into the parent shell.
        print(f"physiclaw runtime: {e}", file=sys.stderr)
        sys.exit(1)
    provider_id, model_id = parse_model_ref(ref)

    # Hooks read this to know where the MCP server lives. Must be set
    # before load_hooks() imports them.
    os.environ.setdefault("PHYSICLAW_SERVER", args.server)

    # Mirror to a daily file so wake decisions / poll errors survive for
    # post-mortems — the runtime's stderr is gone once the terminal is.
    setup_logging(
        "runtime", logging.DEBUG if args.verbose else logging.INFO,
        file_dir=paths.runtime_log_dir(),
    )
    # Silence noisy per-request logs from httpx/httpcore.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if provider_id == CLAUDE_CODE_ID:
        from physiclaw.agent.claude import spawn_claude
        react = partial(spawn_claude, model_id=model_id)
    else:
        from physiclaw.agent.engine.engine import run as engine_run
        react = partial(engine_run, model_ref=ref)

    label = engine_label(ref)
    log.info("%s [%s]", label, source)

    async def _main():
        try:
            await Runtime(
                react=react, interval=args.interval, label=label,
            ).start()
        finally:
            await close_mcp()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
