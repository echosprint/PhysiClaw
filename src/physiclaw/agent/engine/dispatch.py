"""Dispatch — one ToolCall in, exactly one ToolResultMessage out.

Owns the per-call half of the engine's protocol invariants:

  4. Arguments validated against JSONSchema before dispatch.
  5. Errors marked: failed tool_calls get synthetic tool_result(is_error=True).
  6. Transcript stays API-legal: each ToolCall gets exactly one ToolResult
     with matching tool_call_id.

The *judgment* — when to refuse a call, what to warn about — lives in
`policy.py` as the guards/observers on `EngineRun.policies`; this module
owns the mechanics of running them in declared order and turning a Block
into an error ToolResult.
"""

import logging

from physiclaw.agent.engine.dto import (
    ContentBlock,
    TextBlock,
    ToolCall,
    ToolResultMessage,
)
from physiclaw.agent.engine.policy import DispatchGuard
from physiclaw.agent.engine.runspec import EngineRun
from physiclaw.agent.engine.session import Session
from physiclaw.agent.engine.trace import (
    brief_content,
    format_call_args,
    format_call_result,
)
from physiclaw.agent.engine.validator import ValidationError, validate_arguments
from physiclaw.agent.provider import mcp_blocks_to_content_blocks
from physiclaw.common import verdict

log = logging.getLogger(__name__)


async def dispatch(
    run: EngineRun,
    session: Session,
    call: ToolCall,
    turn: int,
) -> ToolResultMessage:
    """Guards, validate, execute, observe. Always returns a ToolResultMessage —
    never raises (principle 5 + principle 6 require that every ToolCall is
    paired with a ToolResult even on failure)."""
    log.info("  → %s(%s)", call.name, format_call_args(call.name, call.arguments))

    blocked = _run_guards(run, session, call, turn, run.policies.pre_validation_guards)
    if blocked is not None:
        return blocked

    schema = run.schema_by_name.get(call.name)
    if schema is None:
        run.tr.write(
            {"event": "tool_unknown", "turn": turn, "name": call.name, "id": call.id}
        )
        log.warning("  ✗ %s: unknown tool", call.name)
        return ToolResultMessage(
            tool_call_id=call.id,
            content=f"unknown tool: {call.name!r}",
            is_error=True,
        )

    # Principle 4: validate arguments before executing.
    try:
        validate_arguments(call.arguments, schema.get("input_schema") or {})
    except ValidationError as e:
        run.tr.write(
            {
                "event": "tool_invalid_args",
                "turn": turn,
                "name": call.name,
                "id": call.id,
                "arguments": call.arguments,
                "error": str(e),
            }
        )
        log.warning("  ✗ %s: invalid args — %s", call.name, e)
        return ToolResultMessage(
            tool_call_id=call.id,
            content=f"invalid arguments for {call.name}: {e}",
            is_error=True,
        )

    blocked = _run_guards(run, session, call, turn, run.policies.post_validation_guards)
    if blocked is not None:
        return blocked

    local = run.local_registry.get(call.name)
    try:
        if local is not None:
            text = await local.handler(session, call.arguments)
            run.tr.write(
                {
                    "event": "tool_result",
                    "turn": turn,
                    "name": call.name,
                    "id": call.id,
                    "arguments": call.arguments,
                    "text": text,
                }
            )
            log.info("  ✓ %s → %s", call.name, format_call_result(call.name, text))
            return ToolResultMessage(tool_call_id=call.id, content=text)

        blocks = await run.mcp.call_tool(call.name, call.arguments)
        content = mcp_blocks_to_content_blocks(blocks)
        changed = verdict.parse(_action_text(blocks))
        content = _observe_result(
            run,
            session,
            call,
            content,
            turn=turn,
            changed=changed,
            failed=False,
        )
        run.tr.write(
            {
                "event": "tool_result",
                "turn": turn,
                "name": call.name,
                "id": call.id,
                "arguments": call.arguments,
                "blocks": blocks,
            }
        )
        log.info("  ✓ %s → %s", call.name, brief_content(content))
        return ToolResultMessage(tool_call_id=call.id, content=content)

    except Exception as e:
        log.error("  ✗ %s failed: %s", call.name, e)
        run.tr.write(
            {"event": "tool_error", "turn": turn, "name": call.name, "error": str(e)}
        )
        content = _observe_result(
            run,
            session,
            call,
            f"{call.name} failed: {e}",
            turn=turn,
            changed=None,
            failed=True,
        )
        return ToolResultMessage(
            tool_call_id=call.id,
            content=content,
            is_error=True,
        )


def _run_guards(
    run: EngineRun,
    session: Session,
    call: ToolCall,
    turn: int,
    guards: tuple[DispatchGuard, ...],
) -> ToolResultMessage | None:
    """Run one phase's dispatch guards (pre- or post-validation, split once
    at Policies construction) in declared order; the first Block wins and
    becomes an error ToolResult."""
    for guard in guards:
        block = guard.check(session, call, turn=turn)
        if block is None:
            continue
        run.tr.write(
            {
                "event": block.event,
                "turn": turn,
                "name": call.name,
                "id": call.id,
                "arguments": call.arguments,
            }
        )
        log.warning("  ✗ %s: %s", call.name, block.log_msg)
        return ToolResultMessage(
            tool_call_id=call.id,
            content=block.content,
            is_error=True,
        )
    return None


def _observe_result(
    run: EngineRun,
    session: Session,
    call: ToolCall,
    content: str | list[ContentBlock],
    *,
    turn: int,
    changed: bool | None,
    failed: bool,
) -> str | list[ContentBlock]:
    """Run the result observers in declared order; each may append one
    advisory line to the tool-result content (traced under its event)."""
    for obs in run.policies.result_observers:
        advisory = obs.observe(session, call, changed=changed, failed=failed)
        if advisory is None:
            continue
        run.tr.write(
            {
                "event": advisory.event,
                "turn": turn,
                "name": call.name,
                "id": call.id,
            }
        )
        content = _append_text(content, advisory.text)
    return content


def _action_text(blocks: list[dict]) -> str:
    """The FIRST text block of a raw MCP tool result — the core-composed
    action text carrying the verdict marker, and the ONLY safe haystack
    for it. Later text blocks hold the OCR listing, i.e. whatever the
    phone displays — scanning those would let on-screen text forge a
    verdict (the agent IM'ing the user "…screen: no visible change…"
    about a blocker, then operating in that chat, is enough). Must run
    on the raw blocks: `mcp_blocks_to_content_blocks` fuses all text
    blocks into one, erasing the boundary."""
    for b in blocks:
        if b.get("type") == "text":
            return b.get("text") or ""
    return ""


def _append_text(
    content: str | list[ContentBlock], extra: str
) -> str | list[ContentBlock]:
    """Append a line of text to a tool-result content, whatever its shape."""
    if isinstance(content, str):
        return f"{content}\n{extra}"
    return [*content, TextBlock(text=extra)]
