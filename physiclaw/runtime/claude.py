"""Spawn Claude Code when any hook triggers.

Not a hook itself — this is the default `react` callable passed to
`Runtime`. The loop calls `spawn_claude(triggers)` whenever
`check_hooks()` returns a non-empty list, and the prompt is built from
the triggers that fired so Claude knows what woke it up and from where.
"""

import asyncio
import logging

from physiclaw.runtime.hook import Trigger

log = logging.getLogger(__name__)

CLAUDE_BIN = "claude"


def _build_prompt(triggers: list[Trigger]) -> str:
    lines = ["The following events were detected:"]
    for t in triggers:
        tag = f"[{t.source}] " if t.source else ""
        lines.append(f"- {tag}{t.description}")
    lines.append("")
    lines.append("Follow the workflow in CLAUDE.md to decide what to do next.")
    return "\n".join(lines)


async def spawn_claude(triggers: list[Trigger]) -> None:
    prompt = _build_prompt(triggers)
    proc = await asyncio.create_subprocess_exec(
        CLAUDE_BIN,
        "-p",
        prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        log.error(
            "claude exited %s: %s",
            proc.returncode,
            stderr.decode(errors="replace").strip(),
        )
        return
    log.info("claude said: %s", stdout.decode(errors="replace").strip())
