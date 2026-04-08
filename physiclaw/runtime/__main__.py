"""Standalone runtime entry point — `python -m physiclaw.runtime`.

Spawned as a subprocess by `physiclaw.main` so the poll/dispatch loop runs
out-of-process from the MCP server. This isolates long-running hook work
(e.g. shelling out to `claude`) from the FastMCP event loop.

Polls the server's /api/phone/watch endpoint and fires registered hooks
(auto-discovered from `physiclaw.hooks/`) whenever it returns event=True.
"""

import argparse
import asyncio
import logging

import httpx

from physiclaw.runtime import Runtime, dispatch

log = logging.getLogger(__name__)


async def _amain(base_url: str, interval: float) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:

        async def poll() -> bool:
            try:
                r = await client.get("/api/phone/watch")
                if r.status_code != 200:
                    return False
                return bool(r.json().get("event"))
            except Exception:
                log.debug("watchdog poll failed", exc_info=True)
                return False

        await Runtime(poll=poll, dispatch=dispatch, interval=interval).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="PhysiClaw runtime loop")
    parser.add_argument("--server", default="http://127.0.0.1:8048")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[runtime] %(message)s",
    )
    try:
        asyncio.run(_amain(args.server, args.interval))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
