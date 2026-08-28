"""Reading tool results out of the transcript — the conductor's eyes.

The conductor never actuates and never holds an MCP client; everything
it knows about the world arrives as ordinary ``ToolResultMessage``s in
the session history. These four readers are the whole vocabulary: the
latest result of any call (what the boot watches when it has no hand to
navigate with), the result of one specific synthesized call (every
driver's view, via `Turnsmith.settle`), and a result's text/screen
extraction. One home, because Program and Overture must read history
identically.
"""

from physiclaw.common.listing import Screen
from physiclaw.contract.dto import Message, TextBlock, ToolResultMessage


def last_result(history: list[Message]) -> ToolResultMessage | None:
    """The most recent tool result of ANY call — what activation reads
    (model turns included; the conductor is not driving yet)."""
    for msg in reversed(history):
        if isinstance(msg, ToolResultMessage):
            return msg
    return None


def result_for(history: list[Message], call_id: str) -> ToolResultMessage | None:
    for msg in reversed(history):
        if isinstance(msg, ToolResultMessage) and msg.tool_call_id == call_id:
            return msg
    return None


def screen_of(result: ToolResultMessage) -> Screen:
    """The screen a tool result carries — its text blocks parsed as a
    listing (macro results and peeks both end with the current view)."""
    return Screen.read(text_of(result))


def text_of(result: ToolResultMessage) -> str:
    """All text of a tool result, joined."""
    if isinstance(result.content, str):
        return result.content
    return "\n".join(b.text for b in result.content if isinstance(b, TextBlock))
