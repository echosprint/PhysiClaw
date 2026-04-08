"""PhysiClaw Runtime — poll a watchdog endpoint, dispatch hooks on events.

See `runtime.py` for the loop and `hook.py` for the registry.

Typical use:

    from physiclaw.runtime import Runtime, register, dispatch
    import httpx

    @register
    async def on_phone_event():
        print("phone changed")

    async def main():
        async with httpx.AsyncClient(base_url="http://localhost:8048") as c:
            async def poll():
                r = await c.get("/api/phone/watch")
                return r.json()["event"]
            await Runtime(poll=poll, dispatch=dispatch, interval=1.0).start()
"""

from physiclaw.runtime.runtime import Runtime
from physiclaw.runtime.hook import register, dispatch, clear

__all__ = ["Runtime", "register", "dispatch", "clear"]
