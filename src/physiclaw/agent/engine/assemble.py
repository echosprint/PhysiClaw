"""Request assembly — the offline half of a session's setup.

Everything here runs without an MCP connection or a provider: discover
skills, build the local tool registry, render the SYSTEM prompt, compose
the wake user-message, and pin the per-turn tail slots. Shared by the
engine (`engine._run_session` / `loop._prepare_request`) and the
`physiclaw prompt` dump (`cli/prompt.py`), so a dump can never drift from
the wire the running agent actually sends.
"""

import datetime as dt
from dataclasses import dataclass

from physiclaw.agent.engine import (
    builtin_tool,
    compact,
    jobs,
    memory,
    pitfalls,
    plan,
    prompt,
    scratchpad,
    screen_layout,
    skill,
)
from physiclaw.agent.engine.builtin_tool import LocalTool
from physiclaw.agent.engine.dto import Message, SystemMessage, UserMessage
from physiclaw.agent.engine.session import Session
from physiclaw.agent.runtime.hook import Trigger


@dataclass(slots=True)
class PromptBundle:
    """The turn-0 request pieces assembled offline — no MCP or provider needed.
    Shared by `engine._run_session` and the `physiclaw prompt` dump so the dump
    matches the real request. `local_registry`/`local_schemas` are the engine's
    local tools; `layout_incomplete` gates the first-run reminder; the skill
    counts are for the session's tools-loaded log line."""

    system_prompt: str
    local_registry: dict[str, LocalTool]
    local_schemas: list[dict]
    layout_incomplete: bool
    builtin_skill_count: int
    user_skill_count: int
    pitfall_count: int = 0


def build_prompt_bundle(provider_id: str) -> PromptBundle:
    """Discover skills, build the local tool registry, and render the SYSTEM
    prompt for a session — the offline half of `engine._run_session`'s setup."""
    layout_incomplete = not screen_layout.is_learned()
    builtin_skills = screen_layout.prune_builtin_skills(skill.discover_builtin_skills())
    # Inline the learned bboxes into the skill bodies (e.g. im's send sequence)
    # so the agent uses coordinates directly instead of cross-referencing.
    builtin_skills = screen_layout.fill_builtin_boxes(builtin_skills)
    user_skills = skill.discover_user_skills()
    local_registry = builtin_tool.build_registry(user_skills)
    local_schemas = builtin_tool.schemas(local_registry)
    pitfall_items = pitfalls.read()  # read once; render + count both use it
    system_prompt = prompt.render_system_prompts(
        local_tool_schemas=local_schemas,
        memory_ctx=memory.load_persistent(),
        builtin_skills_ctx=skill.render_builtin(builtin_skills),
        user_skills_ctx=skill.render_section(user_skills),
        # Agent-flagged traps, always-on right after user skills (no Skill() load).
        pitfalls_ctx=pitfalls.render_section(pitfall_items),
        provider_id=provider_id,
    )
    return PromptBundle(
        system_prompt=system_prompt,
        local_registry=local_registry,
        local_schemas=local_schemas,
        layout_incomplete=layout_incomplete,
        builtin_skill_count=len(builtin_skills),
        user_skill_count=len(user_skills),
        pitfall_count=len(pitfall_items),
    )


def build_initial_messages(
    triggers: list[Trigger], system_prompt: str
) -> list[Message]:
    """The message array a session starts with: cached SYSTEM prompt, the
    wake-trigger user message (date anchor + fired-job context), and the three
    pre-allocated compaction slots (summary / memory / skills)."""
    return [
        SystemMessage(content=system_prompt),
        UserMessage(
            content=format_triggers(
                triggers,
                cron_ctx=jobs.format_fired(triggers),
            )
        ),
        compact.new_summary_placeholder(),
        compact.new_memory_placeholder(),
        compact.new_skills_placeholder(),
    ]


def apply_request_tails(
    messages: list[Message],
    session: Session,
    *,
    layout_incomplete: bool,
    compaction_keep: int | None = None,
) -> list[Message]:
    """Pin the per-turn tail slots to the request — scratchpad, plan, the
    pre-compression checkpoint notice (`compaction_keep` set = this is a
    `collapse_pending` turn), and (while first-run setup is pending) the
    layout reminder — the LAST things the model sees. Shared by
    `loop._prepare_request` and the `prompt` dump so a turn-0 dump matches
    the wire the engine sends. Does not mutate `messages`."""
    out = scratchpad.inject_tail(messages, session.scratchpad)
    out = plan.inject_tail(out, session.plan)
    if compaction_keep is not None:
        out = compact.inject_checkpoint_tail(out, keep=compaction_keep)
    if layout_incomplete:
        out = screen_layout.inject_tail(out)
    return out


def format_triggers(triggers: list[Trigger], *, cron_ctx: str = "") -> str:
    # Leading `Now:` line is the absolute-date anchor — memory logs use
    # relative dates, and the model burns turns triangulating today without
    # it. Each trigger then uses a uniform `[stamp] source: body` envelope.
    # cron_ctx (jobs firing now) rides in this user message, not the system,
    # so the system stays byte-stable across wakes for cross-session caching.
    now = dt.datetime.now()
    stamp = now.strftime("%Y-%m-%d %a %H:%M")
    lines = [
        f"Now: {stamp}",
        "",
        "[Current wake — act on this]",
    ]
    for t in triggers:
        source = t.source or "manual"
        lines.append(f"[{stamp}] {source}: {t.description}")
    text = "\n".join(lines)
    if cron_ctx:
        text += "\n\n" + cron_ctx
    return text
