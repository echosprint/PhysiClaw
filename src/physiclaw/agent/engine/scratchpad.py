"""Scratchpad — agent's free-form working memory.

Lives on `Session.scratchpad` (string), never in `messages[]`. Before
each `provider.chat(...)` call the engine appends a
`<scratchpad>...</scratchpad>` `UserMessage` via `inject_tail`, sitting
just before the plan tail. Stays out of `messages[]` for the same
reason the plan does: a write invalidates only the volatile tail, not
the cached body of the transcript.

Mutation funnels through the optional `scratchpad` field on the `note`
tool — every turn already calls `note`, so a write costs no turn slot.
The note handler delegates to `write()` here.
"""

from physiclaw.agent.engine.dto import Message, UserMessage
from physiclaw.common.config import CONFIG

# Scratchpad ships in every prompt for the rest of the session, so its size is
# a per-turn token cost — this cap is a working-memory ceiling, not a dumping
# ground (a cart list, target bboxes, a short draft fit; a whole screen dump
# does not). Over it, `write` rejects with "summarize first". Config knob.
MAX_CHARS = CONFIG.engine.scratchpad_max_chars


def write(session, content: str) -> str:
    """Replace `session.scratchpad`. Returns the model-facing status
    string. Raises `ValueError` on oversize input. Whitespace-only
    content normalizes to empty — the stored value and the reported
    status stay in sync."""
    if len(content) > MAX_CHARS:
        raise ValueError(
            f"{len(content)} chars > {MAX_CHARS} cap. Summarize before writing."
        )
    if not content.strip():
        session.scratchpad = ""
        return "scratchpad cleared"
    session.scratchpad = content
    return f"scratchpad updated ({len(content)} chars)"


def inject_tail(messages: list[Message], content: str) -> list[Message]:
    """Return `messages + [<scratchpad>...</scratchpad>]` when content is
    non-empty; else `messages` unchanged. Original list is not mutated."""
    if not content.strip():
        return messages
    return messages + [UserMessage(content=f"<scratchpad>\n{content}\n</scratchpad>")]
