"""Data-transfer objects — the single shape the engine loop operates on.

Every provider serializes messages, tool_calls, finish reasons, and
assistant content differently. These DTOs normalize that boundary: the
engine deals only in DTOs; provider classes own translation to/from
their native wire format (`AnthropicProvider` → `/v1/messages` blocks,
`OpenAICompatibleProvider` → `/chat/completions` dicts, etc.). Provider-
specific leakage (e.g. Qwen's `reasoning_content`, Anthropic's
`thinking` blocks) is stripped before assistant content lands in
history — those fields don't survive re-serialization.

The DTO hierarchy mirrors what every chat-completion API exposes — four
message kinds and two content-block kinds — without committing to any
one API's encoding:

  Message kinds:
    - `SystemMessage`     — the system prompt
    - `UserMessage`       — user-role content (text, optionally + images)
    - `AssistantMessage`  — model response (text + tool_calls + usage)
    - `ToolResultMessage` — paired result of a `ToolCall` by id

  Content blocks (inside Message.content when multipart):
    - `TextBlock`   — plain text
    - `ImageBlock`  — base64-encoded image bytes + media_type

`Usage` is a normalized cache-aware token count populated by each
provider from its native usage block. The engine reads it via
`AssistantMessage.usage` for the per-turn cache-summary log; the raw
provider response is preserved on `AssistantMessage.raw` for trace
replay / debugging.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Union


class FinishReason(StrEnum):
    """Provider-normalized stop cause. Principle 3: preserve the real
    cause — do not derive it from content. Values equal their string form
    so equality checks against OpenAI-style strings still work."""

    TOOL_CALLS = "tool_calls"
    STOP = "stop"
    LENGTH = "length"  # truncated — arguments may be incomplete
    CONTENT_FILTER = "content_filter"
    ERROR = "error"  # our own: parse / transport failures


# ---------- content blocks ----------


@dataclass(frozen=True)
class TextBlock:
    """Plain text content. Used inside `UserMessage.content` and
    `ToolResultMessage.content` when multipart."""

    text: str


@dataclass(frozen=True)
class ImageBlock:
    """Image content carried as base64 bytes (no data URL wrapper). The
    provider re-encodes per its wire format — OpenAI emits `image_url`
    with a `data:` URL, Anthropic emits `image` with `source.base64`,
    etc. Engine code (compaction, cache markers) stores `ImageBlock`
    directly; the encoding lives entirely in the provider."""

    media_type: str  # e.g. "image/jpeg", "image/png"
    data_b64: str  # base64-encoded image bytes


ContentBlock = Union[TextBlock, ImageBlock]


# ---------- tool call / result primitives ----------


@dataclass(frozen=True)
class ToolCall:
    """One call the model made. Arguments already parsed from JSON."""

    id: str
    name: str
    arguments: dict[str, Any]


# ---------- token usage ----------


@dataclass(frozen=True)
class Usage:
    """Normalized per-call token accounting.

    Each provider parses its native usage block into this shape:
      - DashScope/OpenAI: `usage.prompt_tokens`,
        `usage.completion_tokens`,
        `usage.prompt_tokens_details.cached_tokens`,
        `usage.prompt_tokens_details.cache_creation_input_tokens`
      - Anthropic: `usage.input_tokens`, `usage.output_tokens`,
        `usage.cache_read_input_tokens`,
        `usage.cache_creation_input_tokens`
      - Google: TBD (different again)

    All zero = "provider didn't report" (e.g. local models, error
    responses). The engine treats zeros as "no data" rather than 0%
    cache hit.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # cache hit (read)
    cache_creation_tokens: int = 0  # cache write (first-time)


# ---------- messages (engine history entries) ----------


@dataclass(frozen=True)
class SystemMessage:
    """The system prompt. One per session, always at index 0."""

    content: str


@dataclass(frozen=True)
class UserMessage:
    """User-role content. May be a plain string (most messages) or a list
    of content blocks (when carrying images, e.g. tool-call corrective
    messages don't have images, but plan-tail messages might in the
    future)."""

    content: str | list[ContentBlock]


@dataclass
class AssistantMessage:
    """A single turn's assistant output, normalized across providers.

    `content` may be empty when the model only called tools. `usage` is
    the per-call token accounting (cache-aware); each provider populates
    it from its own response shape. `raw` retains the provider's
    original response dict for trace replay / debugging.

    `vendor_extra` carries opaque tokens a provider needs to round-trip
    verbatim across turns to preserve hidden state — e.g. Gemini 3's
    `thought_signature` (links the model's prior internal reasoning to
    the next turn). Keyed by `PROVIDER_ID` so multiple vendors can
    coexist in mixed-history scenarios. Engine code never reads this;
    only the originating provider does, on serialize.
    """

    content: str
    tool_calls: list[ToolCall]
    finish_reason: FinishReason
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] = field(default_factory=dict)
    vendor_extra: dict[str, Any] = field(default_factory=dict)

    def tool_names(self) -> list[str]:
        return [tc.name for tc in self.tool_calls]


@dataclass(frozen=True)
class ToolResultMessage:
    """Result of one `ToolCall`. Principle 6: every tool_call must have
    exactly one result with the same id in the very next message.

    `is_superseded` marks a result that `compact.drop_stale_screens`
    rewrote into a text stub (the original screen capture is now stale —
    a more recent peek/screenshot has landed). Providers use this flag,
    not content parsing, to find the deepest byte-stable point for the
    second cache-control marker.
    """

    tool_call_id: str
    content: str | list[ContentBlock]
    is_error: bool = False
    is_superseded: bool = False


Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolResultMessage]


# ---------- ToolResult (legacy alias for ToolResultMessage) ----------
# Pre-DTO-refactor short name. Engine + provider code migrated to the
# fully-qualified `ToolResultMessage`; only test fixtures still
# construct `ToolResult(...)`. Drop when those tests are updated.
ToolResult = ToolResultMessage


__all__ = [
    "FinishReason",
    "TextBlock",
    "ImageBlock",
    "ContentBlock",
    "ToolCall",
    "Usage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "ToolResult",  # alias — see note above
    "Message",
]
