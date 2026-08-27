"""Engine — native tool-call loop. Follows OpenClaw's 7 principles.

  1. Structure via provider API: tools=[...] + message.tool_calls, not JSON.
  2. Normalized at the boundary: Provider returns AssistantMessage; raw
     provider-specific fields (reasoning_content) stripped before echo.
  3. Real finish_reason preserved and routed (length / content_filter / stop).
  4. Arguments validated against JSONSchema before dispatch.
  5. Errors marked: failed tool_calls get synthetic tool_result(is_error=True);
     finish_reason="error" / "length" / "content_filter" drive recovery.
  6. Transcript stays API-legal: each ToolCall gets exactly one ToolResult
     with matching tool_call_id in the very next message.
  7. Loop is the driver: model → tool_calls → dispatch → tool_results → model.

Layering: the mechanics split across four sibling modules — this one
owns the session lifecycle (wire provider/MCP/tools/policies into an
`EngineRun`, drive attempts under the session-outcome contract);
`loop.py` owns the turn driver, finish_reason routing, and the
[note, one-other] shape (principles 3, 7); `dispatch.py` owns tool-call
execution (principles 4, 5, 6); `runspec.py` holds the shared wiring
dataclasses. The *judgment* — when to reject a turn, when to refuse a
call, what to warn about — lives in `policy.py` as policy objects on
`EngineRun.policies`; request assembly (prompt bundle, initial
messages, tails) lives in `assemble.py`, shared with the CLI dump.
"""

import asyncio
import logging
import time
from functools import partial

from physiclaw.agent.conductor import Conductor
from physiclaw.agent.conductor import setup as conductor_setup
from physiclaw.agent.conductor.micro import MicroCaller
from physiclaw.agent.engine import assemble, curate, loop, prompt
from physiclaw.agent.engine.dto import Message
from physiclaw.agent.engine.mcp_tool import get_mcp, list_tools_cached
from physiclaw.agent.engine.policy import default_policies
from physiclaw.agent.engine.runspec import EngineRun, Settings
from physiclaw.agent.engine.session import Session
from physiclaw.agent.provider import Provider, make_provider
from physiclaw.agent.runtime import contract
from physiclaw.agent.runtime.hook import Trigger
from physiclaw.agent.runtime.sentinel import STUCK
from physiclaw.agent.trace import RawLog, Trace, new_sid
from physiclaw.common.config import CONFIG, parse_model_ref

log = logging.getLogger(__name__)


async def run(triggers: list[Trigger], *, model_ref: str) -> contract.SessionOutcome:
    """Run engine sessions for `triggers` under the session-outcome contract.

    One attempt = one fresh `Session` through `_run_session` (which never
    raises — crashes close STUCK). Everything above the attempt belongs to
    `contract.drive`: STUCK retries up to `max_attempts` (STUCK here is
    mostly harness-detected — max turns, provider exhausted, crash — so a
    fresh session is worth it; DONE / FAIL / IDLE / WAIT are final), the
    one-shot first-run setup restart, and the jobless-WAIT follow-up. A
    session that exhausted its wall-clock budget reports `retryable=False`
    — a slow environment would burn every retry the same way.

    `model_ref` is a `provider/model` string (e.g. `"qwen/qwen3.6-plus"`).
    Parsed inside `_run_session`; the provider is instantiated via
    `provider.make_provider(provider_id, model_id)`. Required — every
    caller goes through the launcher, which resolves `PHYSICLAW_MODEL`.
    """
    settings = Settings.from_config()

    async def attempt(trigs: list[Trigger], n: int) -> contract.SessionOutcome:
        session = Session()
        await _run_session(
            trigs,
            model_ref=model_ref,
            session=session,
            settings=settings,
        )
        return contract.SessionOutcome(
            status=session.sentinel_status,
            recap=session.sentinel_recap,
            created_job=session.sentinel_turn_created_job,
            restart_requested=session.restart_for_setup,
            retryable=not session.budget_exhausted,
        )

    # The final outcome flows back to the runtime loop, which backs off
    # its wake cadence on unproductive streaks (a dead phone must not
    # burn a full session per watchdog blip).
    return await contract.drive(
        attempt,
        triggers,
        max_attempts=settings.max_session_attempts,
        wait_default_minutes=settings.wait_default_minutes,
        retry_on=(STUCK,),
    )


def _wire_micro(program, activation, provider, tr, rlog):
    """The micro-caller for the conductor's decision calls — None when
    nothing can need one: no program and no activation trigger, or a
    pure-LEG program (which must not pay for a second provider client).
    `[conductor] micro_model` selects the cheap decision tier; the
    caller builds that client lazily on the FIRST call (an activation
    trigger usually never fires) and owns it — the session's finally
    block closes it via `MicroCaller.aclose()`. Fail-open to the session
    provider on any problem, at parse time or build time."""
    if activation is None and (program is None or not program.needs_micro):
        return None
    factory = None
    ref = CONFIG.conductor.micro_model
    if ref:
        try:
            pid, mid = parse_model_ref(ref)
            factory = partial(make_provider, pid, mid)
        except Exception as e:
            log.warning(
                "conductor micro_model %r unusable (%s) — using the session model",
                ref,
                e,
            )
    return MicroCaller(
        provider,
        confidence_floor=CONFIG.conductor.micro_confidence,
        tr=tr,
        rlog=rlog,
        owned_factory=factory,
    )


async def _run_session(
    triggers: list[Trigger],
    *,
    model_ref: str,
    session: Session,
    settings: Settings | None = None,
) -> None:
    """One session attempt. Fresh sid / Trace / RawLog / MCP / Provider /
    policy set per call. Writes outcome to `session.sentinel_*`; never
    raises."""
    provider_id, model_id = parse_model_ref(model_ref)
    settings = settings or Settings.from_config()

    sid = new_sid()
    provider: Provider | None = None
    micro_caller: MicroCaller | None = None
    tr: Trace | None = None
    rlog: RawLog | None = None
    try:
        # Open inside the try so the finally block's close() runs even
        # if construction fails midway (disk full, perms, etc.).
        tr = Trace(sid)
        rlog = RawLog(sid)
        tr.write(
            {
                "event": "wake",
                "session": sid,
                "model_ref": model_ref,
                "triggers": [
                    {"source": t.source, "description": t.description} for t in triggers
                ],
            }
        )
        log.info(
            "wake session=%s model=%s triggers=%s",
            sid,
            model_ref,
            [t.source or "?" for t in triggers],
        )
        mcp = await get_mcp()
        mcp_tools = await list_tools_cached()
        # Conductor wake-time setup, fail-open throughout: a suspended or
        # armed program, the activation trigger (parse_task fires once
        # if a screen matches the channel thread), and the hidden
        # qualified macro registry (every pack + channel on the
        # activation path; narrower otherwise — see setup.session_setup).
        # Nothing model-visible changes.
        program, activation, hidden_macros = conductor_setup.session_setup()
        # Built-in skills are inlined full-text into SYSTEM; user skills are
        # indexed and loaded on demand via the Skill tool — so only user
        # skills go into the local registry. The first-run screen-layout skill
        # is dropped once the layout is learned (dead weight after setup).
        # `assemble.build_prompt_bundle` is the offline half, shared with the
        # CLI dump.
        bundle = assemble.build_prompt_bundle(
            provider_id,
            pack_macros=hidden_macros or None,
        )
        # Full merged list goes to conductor.advance(tools=) for invocation;
        # the inline `## Tooling` card pulls MCP names from AST so it
        # stays complete even offline. Each source has one consumer.
        tool_schemas = list(mcp_tools) + bundle.local_schemas
        schema_by_name = {s["name"]: s for s in tool_schemas}
        tr.write(
            {
                "event": "tools_loaded",
                "mcp": [s["name"] for s in mcp_tools],
                "local": sorted(bundle.local_registry.keys()),
            }
        )
        log.info(
            "tools loaded: %d MCP + %d local + %d built-in + %d user skills "
            "+ %d pitfalls + %d macros",
            len(mcp_tools),
            len(bundle.local_registry),
            bundle.builtin_skill_count,
            bundle.user_skill_count,
            bundle.pitfall_count,
            bundle.macro_count,
        )

        messages: list[Message] = assemble.build_initial_messages(
            triggers,
            bundle.system_prompt,
        )

        provider = make_provider(provider_id, model_id)
        micro_caller = _wire_micro(program, activation, provider, tr, rlog)
        prompt_hash = prompt.prefix_hash(bundle.system_prompt)
        rlog.write_session_start(
            provider=provider_id,
            model=provider.model,
            prompt_hash=prompt_hash,
            tools=tool_schemas,
        )
        engine_run = EngineRun(
            # The conductor owns only the turn loop's provider call:
            # curate's off-transcript pass and the finally-close below
            # stay on the raw handle, so the conductor never intercepts
            # side-calls of the very kind its playbooks emit.
            # Session-management wiring (collapse cadence, wire logging)
            # comes straight off the provider — not the conductor's job.
            conductor=Conductor(
                provider,
                program=program,
                micro=micro_caller,
                activation=activation,
            ),
            collapse=provider.COLLAPSE,
            serialize_wire=provider.serialize_history,
            mcp=mcp,
            tool_schemas=tool_schemas,
            schema_by_name=schema_by_name,
            local_registry=bundle.local_registry,
            tr=tr,
            rlog=rlog,
            settings=settings,
            policies=default_policies(layout_incomplete=bundle.layout_incomplete),
            layout_incomplete=bundle.layout_incomplete,
            deadline=(
                time.monotonic() + settings.max_session_seconds
                if settings.max_session_seconds > 0
                else None
            ),
        )
        tr.write({"event": "prefix_pinned", "hash": prompt_hash})
        await loop.drive(engine_run, session, messages)

        log.info(
            "session done: session=%s status=%s recap=%r",
            sid,
            session.sentinel_status,
            session.sentinel_recap,
        )
        # Post-session pitfalls curation: a separate LLM pass (still-open
        # provider) that consolidates the list + enforces the cap when the agent
        # added traps this session. Fail-open — never affects the outcome.
        if CONFIG.pitfalls.curate_enabled and session.added_pitfalls:
            await curate.curate(provider, tr=tr)

        tr.write(
            {
                "event": "done",
                "sentinel": session.sentinel_status,
                "recap": session.sentinel_recap,
            }
        )
    except asyncio.CancelledError:
        loop.log_external_stop(session, tr)
        raise
    except Exception as e:
        # Crashes count as STUCK so the retry loop gives it another shot.
        log.exception("engine session crashed: session=%s", sid)
        if tr is not None:
            tr.write({"event": "crashed"})
        session.sentinel_status = STUCK
        session.sentinel_recap = f"session crashed: {e}"
    finally:
        closers = [provider.aclose] if provider is not None else []
        if micro_caller is not None:
            closers.append(micro_caller.aclose)
        for close in closers:
            try:
                await close()
            except Exception:
                # A close failure (cancellation landing here, transport
                # error) must not skip the trace closes below — losing
                # summary.json mislabels the session as killed.
                log.warning("provider close failed", exc_info=True)
        if tr is not None:
            tr.close()
        if rlog is not None:
            rlog.close()
