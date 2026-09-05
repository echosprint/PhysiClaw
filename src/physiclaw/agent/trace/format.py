"""Rendering events and values for the human-facing log surfaces.

The brief/args helpers are shared with `dispatch.py` (its per-call log
lines) and the `physiclaw logs` / `doctor` CLIs; `summarize_event`
turns one structured event into the daily log's one-liner — the CLI
renders a session's narrative from its events.jsonl with the same
formatter the daily log uses.
"""

from typing import Any

# Events that are internal bookkeeping — don't surface in the human log.
# Add here when silencing a new event is cheaper than adding a dedicated
# summary branch.
_SILENT_EVENTS = frozenset({"prefix_pinned", "finish_length_warning"})


def brief(value: Any, limit: int = 80) -> str:
    """One-line truncated repr for log output."""
    s = value if isinstance(value, str) else repr(value)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def brief_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={brief(v, 40)}" for k, v in args.items())


def _full_args(args: dict[str, Any]) -> str:
    """Like `brief_args` but no per-value truncation — for tools whose
    args carry irreplaceable planning/decision context (`update_progress`
    steps, etc.) that get hidden by 40-char truncation."""
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def format_call_args(tool_name: str, args: dict[str, Any]) -> str:
    """Render tool-call args for the human log. `update_progress`
    bypasses the default 40-char truncation — the plan content IS the
    point of the call, and it never appears in the result line (which
    is just "progress updated"). Other tools use the brief default."""
    if tool_name == "update_progress":
        return _full_args(args)
    return brief_args(args)


def format_call_result(tool_name: str, text: str) -> str:
    """Render a tool's result text for the human log. `note` bypasses
    the default 80-char truncation — its result is `noted: <summary>`,
    a literal echo of the summary that's the sole turn-survivor under
    compaction (CONVENTION § Compaction); truncating the result hides
    the canonical record of what the agent committed to."""
    if tool_name == "note":
        return text
    return brief(text, 80)


def brief_content(content: Any) -> str:
    """Compact summary of a `ToolResultMessage.content` (DTO) or an MCP
    blocks list (raw dicts). Handles both because `dispatch.dispatch`
    summarizes after MCP→DTO conversion, but tools that bypass that path
    still pass raw blocks through."""
    from physiclaw.contract.dto import ImageBlock, TextBlock

    if isinstance(content, str):
        return brief(content, 80)
    if not isinstance(content, list):
        return brief(repr(content), 80)
    parts: list[str] = []
    for b in content:
        if isinstance(b, TextBlock):
            parts.append(brief(b.text, 80))
        elif isinstance(b, ImageBlock):
            parts.append(f"<image {len(b.data_b64)}b>")
        elif isinstance(b, dict):
            t = b.get("type")
            if t == "text":
                parts.append(brief(b.get("text", ""), 80))
            elif t == "image":
                parts.append(f"<image {len(b.get('data', ''))}b>")
            elif t == "image_url":
                url = (b.get("image_url") or {}).get("url", "")
                _, _, data = url.partition(",")
                parts.append(f"<image {len(data)}b>")
            else:
                parts.append(t or "?")
        else:
            parts.append("?")
    return " + ".join(parts) or "(empty)"


def fmt_tokens(n: int) -> str:
    """Human token count: 980 → '980', 9_800 → '9.8k', 1_200_000 → '1.2M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


# Event kinds whose rendered line ALSO goes to the process log
# (runtime.log), beside the daily log: a model call's account and a
# conductor decision are each printed in ONE place, so neither the
# engine's turn line nor the micro-caller repeats them.
MIRRORED_EVENTS = frozenset({"usage", "micro_call"})


def usage_text(event: dict[str, Any]) -> str:
    """THE rendering of one model call's account — the daily log, the
    process log and `physiclaw logs` all read it from here, so the
    numbers can never disagree. Every bucket by name; reasoning and
    error only when there is something to say."""

    def n(key: str) -> int:
        return int(event.get(key) or 0)

    tag = f"{event.get('call') or '?'} {event.get('model') or '?'}"
    text = (
        f"usage[{tag}] {n('elapsed_ms') / 1000:.1f}s "
        f"in={fmt_tokens(n('input'))} (read {fmt_tokens(n('cache_read'))}, "
        f"write {fmt_tokens(n('cache_write'))}, new {fmt_tokens(n('input_new'))}) "
        f"out={fmt_tokens(n('output'))}"
    )
    if n("reasoning"):
        text += f" (reasoning {fmt_tokens(n('reasoning'))})"
    if event.get("error"):
        text += f" error={event['error']}"
    return text


def _end_footer(summary: dict[str, Any]) -> str:
    """The daily-log session footer — headline metrics on one greppable line."""
    u = summary["usage"]
    return (
        f"END session={summary['sid']} "
        f"outcome={summary['outcome']['sentinel'] or '(none)'} "
        f"turns={summary['turns']} duration={summary['duration_s']:.0f}s "
        f"tokens={fmt_tokens(u['input_tokens'])}/{fmt_tokens(u['output_tokens'])} "
        f"cache={u['cache_hit_pct']:.0f}% "
        f"tools={sum(summary['tool_calls'].values())}"
    )


def summarize_event(event: dict[str, Any]) -> str | None:  # noqa: C901 — flat dispatch
    name = event.get("event", "")
    t = event.get("turn")
    pfx = f"turn {t}: " if t is not None else ""

    if name == "wake":
        triggers = event.get("triggers") or []
        sources = [x.get("source") or "?" for x in triggers]
        return (
            f"WAKE session={event.get('session', '?')} "
            f"model={event.get('model_ref', '?')} triggers={sources}"
        )
    if name == "env":
        return (
            f"env physiclaw={event.get('physiclaw', '?')} "
            f"python={event.get('python', '?')} {event.get('platform', '?')} "
            f"host={event.get('host', '?')} utc{event.get('utc_offset', '?')}"
        )
    if name == "tools_loaded":
        return (
            f"tools: {len(event.get('mcp') or [])} MCP + "
            f"{len(event.get('local') or [])} local"
        )
    if name == "request":
        return f"{pfx}request ({event.get('message_count', '?')} messages)"
    if name == "response":
        calls = [c.get("name") for c in event.get("tool_calls") or []]
        return f"{pfx}response finish={event.get('finish_reason', '?')} calls={calls}"
    if name == "usage":
        return pfx + usage_text(event)
    if name == "tool_result":
        tool_name = event.get("name", "?")
        args = format_call_args(tool_name, event.get("arguments") or {})
        if "text" in event:
            result = format_call_result(tool_name, event["text"])
        else:
            result = brief_content(event.get("blocks") or [])
        return f"{pfx}{tool_name}({args}) → {result}"
    if name == "tool_invalid_args":
        return f"{pfx}{event.get('name', '?')} invalid args: {brief(event.get('error', ''), 200)}"
    if name == "tool_unknown":
        return f"{pfx}{event.get('name', '?')} unknown tool"
    if name == "tool_error":
        return f"{pfx}{event.get('name', '?')} failed: {brief(event.get('error', ''), 200)}"
    if name == "violations":
        return f"{pfx}violations {event.get('codes') or []}"
    if name == "log_append":
        return f"{pfx}log: {brief(event.get('entry', ''), 200)}"
    if name == "memory_save":
        return f"{pfx}memory: {brief(event.get('text', ''), 200)}"
    if name == "sentinel":
        return f"{pfx}SENTINEL {event.get('name', '?')} — {event.get('recap', '')}"
    if name == "done":
        return (
            f"OUTCOME: {event.get('sentinel') or '(none)'} — {event.get('recap', '')}"
        )
    if name == "crashed":
        return "CRASHED"
    if name == "provider_failed":
        return f"{pfx}provider failed: {brief(event.get('error', ''), 200)}"
    if name == "prefix_drift":
        return (
            f"{pfx}!! PREFIX DRIFT "
            f"expected={event.get('expected', '')[:12]}… "
            f"actual={event.get('actual', '')[:12]}…"
        )
    if name == "micro_call":
        c = event.get("confidence")
        out = event.get("out") or "escalate"
        conf = f" ({c:.2f})" if c is not None else ""
        return (
            f"{pfx}micro {event.get('call', '?')} ({event.get('node', '?')}) → "
            f"{out}{conf} — {brief(event.get('detail') or '', 120)} "
            f"[{event.get('attempts', '?')} attempt(s), {event.get('elapsed_ms', '?')}ms]"
        )
    if name == "walk":
        reason = event.get("reason")
        tail = f" — {brief(reason, 200)}" if reason else ""
        return (
            f"WALK {event.get('app', '?')}/{event.get('playbook', '?')} "
            f"{event.get('outcome', '?')} at node {event.get('node') or '(end)'} "
            f"({event.get('idx', '?')}/{event.get('nodes', '?')}){tail}"
        )
    if name in _SILENT_EVENTS:
        return None
    # Fallback — compact repr so nothing disappears silently.
    return f"event {name}: {brief(repr(event), 200)}"
