"""Standalone runtime entry point — `python -m physiclaw.runtime`.

Spawned as a subprocess by `physiclaw.main` so the hook loop runs
out-of-process from the MCP server. This isolates long-running hook
work (e.g. shelling out to `claude`) from the FastMCP event loop.

Each tick: run every hook under `physiclaw/hooks/` (auto-discovered via
`load_hooks`). Each hook returns a `Trigger` if it fired or `None` if
it didn't. If any hook fired, the collected triggers are passed to
`spawn_claude`, which launches `claude -p <prompt>` as a subprocess.
"""

import argparse
import asyncio
import logging
import os

from physiclaw.runtime import Runtime
from physiclaw.runtime.claude import spawn_claude

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="PhysiClaw runtime loop")
    parser.add_argument("--server", default="http://127.0.0.1:8048")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Hooks read this to know where the MCP server lives. Must be set
    # before load_hooks() imports them.
    os.environ.setdefault("PHYSICLAW_SERVER", args.server)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[runtime] %(message)s",
    )
    try:
        asyncio.run(
            Runtime(react=spawn_claude, interval=args.interval).start()
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
