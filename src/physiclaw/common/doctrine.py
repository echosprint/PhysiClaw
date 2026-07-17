"""Doctrine `{{token}}` rendering — shared by every doctrine delivery path.

Doctrine files quote engine-enforced thresholds via `{{token}}`
placeholders so the prose can never drift from the config the engine
actually enforces. Two paths deliver doctrine and both must render:
the engine's SYSTEM composer (`agent.engine.prompt`) and the MCP
`initialize` instructions (`core.server.mcp`, which ships
`PHYSICLAW.md` to external clients and the claude subprocess).

Lives in common because core must not import agent (see
`tests/test_architecture.py`) — before this module existed the renderer
was private to `agent.engine.prompt`, and the MCP path shipped a
`{{token}}` in `PHYSICLAW.md` to external clients as literal braces.

Values come from CONFIG, which is process-constant, so rendered bytes
are stable across calls — both the engine's prompt cache and the MCP
instructions computed once at import stay byte-stable.
"""

import logging
import re

from physiclaw.common.config import CONFIG

log = logging.getLogger(__name__)

# Engine-enforced thresholds quoted by the doctrine files.
DOCTRINE_TOKENS = {
    "{{same_target_warn}}": str(CONFIG.engine.same_target_warn),
    "{{same_target_block}}": str(CONFIG.engine.same_target_block),
    "{{step_stuck_warn}}": str(CONFIG.engine.step_stuck_warn),
    "{{step_stuck_urgent}}": str(CONFIG.engine.step_stuck_urgent),
    "{{plan_required_after}}": str(CONFIG.engine.plan_required_after),
}


def fill_tokens(name: str, body: str) -> str:
    """Substitute `{{token}}` placeholders; warn on leftovers (a typo'd
    token would otherwise reach the model as literal braces). `name` is
    the source label for the warning (e.g. the doctrine filename).

    NOTE: skill `{{box}}` placeholders are a different token system
    (learned screen elements, resolved by `agent.engine.screen_layout`
    / `agent.claude.spawn`) — never pass skill bodies through this.
    """
    for token, value in DOCTRINE_TOKENS.items():
        body = body.replace(token, value)
    leftover = re.search(r"\{\{[^}\n]*\}\}|\{\{", body)
    if leftover:
        log.warning(
            "doctrine %s: unresolved placeholder %r",
            name,
            leftover.group(),
        )
    return body
