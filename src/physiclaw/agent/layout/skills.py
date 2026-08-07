"""Feeding the learned layout into the built-in skills."""

from dataclasses import replace

from physiclaw.agent.layout import store
from physiclaw.agent.layout.store import _fmt, is_learned

# The built-in skill that drives first-run capture — dropped from the prompt
# once the layout is learned (its body is only useful during setup).
SKILL_NAME = "screen-layout"


def prune_builtin_skills(skills: dict) -> dict:
    """Drop the first-run `screen-layout` skill once the layout is learned —
    after setup its body is dead weight in `## Built-in Skills`. No-op (and a
    fresh dict) while still incomplete, so the skill stays available for
    capture."""
    if is_learned():
        return {k: v for k, v in skills.items() if k != SKILL_NAME}
    return skills


# Built-in skill markdown placeholder -> learned layout field. Keyed by skill:
# the same token means different fields in different skills (im's <paste-button>
# is the chat paste; open-app's is the Spotlight paste). Only `im` is filled —
# it has a copy-paste `sequence` template the agent runs verbatim. Just the
# tokens that appear in that fenced template (filling is code-block-only, so a
# prose-only token like <backspace> would never substitute).
# Skill placeholders use the same `{{token}}` syntax as doctrine's
# config tokens (common.doctrine.DOCTRINE_TOKENS) — one placeholder
# style everywhere. Two domains though: doctrine tokens are ALL substituted
# (leftovers warn), while skill element tokens are substituted only
# here, only inside code blocks, only once the layout is learned —
# elsewhere they stay symbolic and the agent resolves them from
# SYSTEM § Screen layout.
_SKILL_BOX_TOKENS = {
    "im": {
        "{{input-hidden}}": "chat_input_kb_hidden",
        "{{input-visible}}": "chat_input_kb_visible",
        "{{paste-button}}": "chat_paste",
        "{{send}}": "send",
    },
    # Same token name, different field per skill: {{paste-button}} is the
    # CHAT paste box in `im` but the SPOTLIGHT paste box here.
    "open-app": {
        "{{search-field}}": "spotlight_input",
        "{{backspace}}": "backspace",
        "{{paste-button}}": "spotlight_paste",
    },
}


def _sub_in_code_blocks(body: str, subs: dict) -> str:
    """Apply `subs` (token -> replacement) only INSIDE fenced ``` code blocks.
    Splitting on the fence, the odd-indexed segments are the code bodies."""
    parts = body.split("```")
    for i in range(1, len(parts), 2):
        for token, val in subs.items():
            parts[i] = parts[i].replace(token, val)
    return "```".join(parts)


def fill_builtin_boxes(skills: dict) -> dict:
    """Once the layout is learned, swap the bbox placeholders
    (`{{input-hidden}}`, `{{send}}`, ...) for concrete coordinates INSIDE the
    skill's fenced code template (im's send `sequence`) so the agent runs it
    verbatim. Prose keeps the readable placeholder names — they reference
    § Screen layout. No-op (a fresh dict) while incomplete. Inputs aren't
    mutated; filled skills are copies."""
    if not is_learned():
        return dict(skills)
    layout = store._load()
    out = {}
    for name, s in skills.items():
        subs = {
            token: _fmt(layout[field])
            for token, field in _SKILL_BOX_TOKENS.get(name, {}).items()
            if layout.get(field)
        }
        body = _sub_in_code_blocks(s.body, subs) if subs else s.body
        out[name] = replace(s, body=body) if body != s.body else s
    return out
