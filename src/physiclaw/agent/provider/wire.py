"""Wire-format adapters: DTOs ↔ OpenAI `/chat/completions` shape.

Two transformation directions:

  - **MCP → DTO**: `mcp_blocks_to_content_blocks` converts MCP tool-result
    content blocks into engine-native `ContentBlock`s (TextBlock,
    ImageBlock). Used by `engine.dispatch` when wrapping an MCP response
    into a `ToolResultMessage`. Image bytes get scaled here once, on the
    way into history.

  - **DTO → OpenAI wire**: `assistant_to_wire`, `tool_result_to_wire`,
    `user_content_to_openai`, `tool_to_wire` produce OpenAI dicts.
    `OpenAICompatibleProvider._encode_message` calls them per DTO; the
    inherited `BaseProvider.serialize_history` template assembles the
    list and places cache markers. Anthropic's message translators live
    in `provider/anthropic_compat.py`, but its ContentBlock dispatch
    runs through the shared `encode_content` here — TextBlock/ImageBlock
    handling (and any future block type or scaling change) lands once,
    with only the per-shape part dicts differing.

Provider-specific response fields (Qwen's `reasoning_content`,
Anthropic's `thinking` blocks) are stripped at parse time inside the
respective provider — they MUST NOT appear in the assistant content
that gets re-serialized back to wire on the next turn (would break the
prefix cache and confuse the model).
"""

import base64
import json
import logging
from collections.abc import Callable
from typing import Any

from physiclaw.agent.engine import compact
from physiclaw.agent.engine.dto import (
    AssistantMessage,
    ContentBlock,
    ImageBlock,
    TextBlock,
    ToolResultMessage,
)

log = logging.getLogger(__name__)


# ---------- request-side: tool schema → wire ----------


def tool_to_wire(tool: dict) -> dict:
    """Local tool schema → OpenAI `tools=` wire format."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema")
            or {
                "type": "object",
                "properties": {},
            },
        },
    }


# ---------- DTO → OpenAI history wire ----------


def assistant_to_wire(asst: AssistantMessage) -> dict[str, Any]:
    """`AssistantMessage` (DTO) → OpenAI assistant message wire format.
    Drops provider-specific leakage — `reasoning_content` etc. never
    ride along in history."""
    msg: dict[str, Any] = {"role": "assistant", "content": asst.content or ""}
    if asst.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            for tc in asst.tool_calls
        ]
    return msg


def tool_result_to_wire(result: ToolResultMessage) -> dict[str, Any]:
    """`ToolResultMessage` (DTO) → OpenAI `role: tool` message. Principle 6:
    `tool_call_id` pairs the result with its originating call."""
    return {
        "role": "tool",
        "tool_call_id": result.tool_call_id,
        "content": user_content_to_openai(result.content),
    }


def encode_content(
    content: str | list[ContentBlock] | None,
    *,
    image_part: Callable[[ImageBlock], dict],
    empty: str | list[dict],
    label: str,
) -> str | list[dict]:
    """Shared ContentBlock dispatch for every wire shape. Plain strings
    pass through unchanged so text-only messages stay cheap; unknown
    block types are dropped with a warning (tagged `label` so the log
    names the caller); an all-dropped/empty list falls back to `empty`
    (each shape's legal empty content). TextBlock's part is inlined —
    OpenAI and Anthropic agree on `{type: text, text}` — so callers
    supply only what actually differs (`image_part`, `empty`), and
    ImageBlock/scaling changes land here once."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ""
    parts: list[dict] = []
    for block in content:
        if isinstance(block, TextBlock):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            parts.append(image_part(block))
        else:
            log.warning("%s: dropping unknown block %r", label, type(block).__name__)
    return parts or empty


def _openai_image_part(block: ImageBlock) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{block.media_type};base64,{block.data_b64}"},
    }


def user_content_to_openai(content: str | list[ContentBlock]) -> str | list[dict]:
    """Engine ContentBlocks → OpenAI multipart content array."""
    return encode_content(
        content,
        image_part=_openai_image_part,
        empty="",
        label="user_content_to_openai",
    )


# ---------- MCP response → DTO ContentBlocks ----------


def mcp_blocks_to_content_blocks(blocks: list[dict]) -> str | list[ContentBlock]:
    """MCP tool-result content blocks → engine `ContentBlock`s. Image bytes
    are scaled (JPEG + long-edge cap) on the way through; smaller payload
    means smaller history footprint thereafter.

    Returns a bare string when no images are present so common text-only
    tool results stay lightweight (providers fast-path string content).
    """
    # Fast path: single text block
    if len(blocks) == 1 and blocks[0].get("type") == "text":
        return blocks[0].get("text") or ""

    texts: list[str] = []
    images: list[ImageBlock] = []
    for b in blocks:
        kind = b.get("type")
        if kind == "text":
            texts.append(b.get("text") or "")
        elif kind == "image":
            try:
                raw = base64.b64decode(b["data"])
                scaled, mime = compact.scale_image_bytes(raw)
                new_b64 = base64.b64encode(scaled).decode()
            except Exception:
                log.exception("scale failed; forwarding original image")
                new_b64 = b["data"]
                mime = b.get("mime_type") or "image/jpeg"
            images.append(ImageBlock(media_type=mime, data_b64=new_b64))

    if images:
        out: list[ContentBlock] = []
        if texts:
            out.append(TextBlock(text="\n".join(texts)))
        out.extend(images)
        return out
    return "\n".join(texts) if texts else "(empty tool result)"
