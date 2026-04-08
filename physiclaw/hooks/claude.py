"""Hook that invokes Claude Code on a phone event.

Spawns `claude` as a subprocess in non-interactive (`-p`) mode with a fixed
prompt. Auto-discovered by `Runtime.start()` because this module lives under
`physiclaw/hooks/` — no manual import or install step required.
"""

import asyncio
import logging

from physiclaw.runtime.hook import register

log = logging.getLogger(__name__)

CLAUDE_BIN = "claude"
PROMPT = "The phone just woke up. Take a screenshot and decide what to do next."


@register
async def call_claude_code() -> None:
    proc = await asyncio.create_subprocess_exec(
        CLAUDE_BIN,
        "-p",
        PROMPT,
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
