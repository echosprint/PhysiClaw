"""Google Gemini — via Google's OpenAI-compatible endpoint.

Google maintains a first-party OpenAI-compatible shim at
`/v1beta/openai/`. We use that rather than the bespoke
`generativelanguage.googleapis.com` REST API; the shim shares the same
`chat()` flow as every other vendor here, with two Gemini-3 quirks
routed around at the wire boundary:

  1. **`image_url` is rejected in `role: tool`** ('Invalid content part
     type: image_url'), but accepted in `role: user`. We split a
     `ToolResultMessage` carrying images into [text-only tool,
     synthetic user with images]; the engine layer is unchanged.

  2. **Tool flows require `extra_content.google.thought_signature`** on
     the first tool_call of every assistant message that called a tool
     ('Function call is missing a thought_signature'). The signature is
     an opaque server-signed token Gemini emits on responses to anchor
     its prior internal reasoning across turns. We round-trip it via
     `AssistantMessage.vendor_extra["google"]["thought_signature"]`. A
     synthetic tool_call with no captured signature (cold start, test
     fixtures) falls back to the documented bypass token —
     `"skip_thought_signature_validator"` — which costs thinking-state
     continuity for that one turn but keeps the request legal.

Auth: `GOOGLE_API_KEY` env, or `[provider] google_api_key` in
`~/.physiclaw/config.toml`.

Model ref examples:  `google/gemini-3.1-pro-preview`,
`google/gemini-3-flash-preview`, `google/gemini-2.5-pro`.

`gemini-2.5-flash` is omitted — it CAN reason via the `thinkingBudget`
parameter, but defaults to off. Until we wire that param, it would
return non-reasoning responses, which violates PhysiClaw's requirement.
Gemini 3 series (`gemini-3.x-*-preview`) uses dynamic thinking by default.
"""

from physiclaw.contract.dto import (
    AssistantMessage,
    ContentBlock,
    ImageBlock,
    Message,
    ToolResultMessage,
)
from physiclaw.provider.openai_compat import OpenAICompatibleProvider
from physiclaw.provider.provider_base import NO_CACHE_MARKERS
from physiclaw.provider.wire import (
    assistant_to_wire,
    user_content_to_openai,
)

# Documented escape hatch from Google's thought-signatures doc. Used
# when no captured signature is available for an outbound tool_call.
_SIG_BYPASS = "skip_thought_signature_validator"


class GoogleProvider(OpenAICompatibleProvider):
    PROVIDER_ID = "google"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

    # Cache markers: disabled. Gemini's shim ignores Anthropic-style
    # `cache_control`. The bigger reason for stripping is that the stub
    # marker tracks the *latest* superseded tool_result, whose position
    # shifts forward every screen-obs turn — even the additive key the
    # base `mark_stub` now toggles on/off is a byte delta at exactly
    # the prefix position Gemini's implicit cache wants to anchor on,
    # with zero upside when the shim ignores the field anyway. Hits
    # surface via standard
    # `usage.prompt_tokens_details.cached_tokens`; the base `_parse_usage`
    # already reads it. Implicit caching itself is non-deterministic
    # across both Flash and Pro (intermittent zero-hit turns on stable
    # prefixes; see issue googleapis/python-genai#2064 and
    # tests/probe_google_cache_*) — don't bank on the theoretical 75%.
    CACHE_MARKERS = NO_CACHE_MARKERS

    async def list_models(self) -> list[dict]:
        """Strip Gemini's `models/` id prefix so the user-facing form
        is uniform across providers (`gemini-2.5-flash`, not
        `models/gemini-2.5-flash`). The OpenAI shim accepts both forms
        on the chat side, so no re-prefixing is needed when sending."""
        raw = await super().list_models()
        return [{**m, "id": m.get("id", "").removeprefix("models/")} for m in raw]

    # ---------- request side: shim quirk overrides ----------

    def _encode_message(self, msg: Message) -> dict | list[dict] | None:
        if isinstance(msg, ToolResultMessage):
            return _encode_tool_result(msg)
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            entry = assistant_to_wire(msg)
            sig = (msg.vendor_extra.get("google") or {}).get(
                "thought_signature"
            ) or _SIG_BYPASS
            entry["tool_calls"][0]["extra_content"] = {
                "google": {"thought_signature": sig},
            }
            return entry
        return super()._encode_message(msg)

    # ---------- response side: capture signature for next turn ----------

    def _parse_response(self, raw: dict) -> AssistantMessage:
        asst = super()._parse_response(raw)
        sig = _extract_thought_signature(raw)
        if sig:
            asst.vendor_extra.setdefault("google", {})["thought_signature"] = sig
        return asst


# ---------- helpers ----------


def _encode_tool_result(result: ToolResultMessage) -> list[dict]:
    """Encode a `ToolResultMessage` for the Google shim, splitting if
    needed.

    The shim rejects `image_url` parts inside `role: tool`. Text-only or
    superseded results pass through as a single `role: tool` entry; a
    fresh result carrying images splits into two entries — text portion
    on `role: tool` (preserving `tool_call_id` pairing) followed by a
    synthetic `role: user` carrying the image parts. The engine sees
    one DTO; the wire sees two messages.
    """
    text_blocks: list[ContentBlock] = []
    image_blocks: list[ContentBlock] = []
    if isinstance(result.content, list):
        for b in result.content:
            (image_blocks if isinstance(b, ImageBlock) else text_blocks).append(b)
    if not image_blocks:
        return [
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": user_content_to_openai(result.content),
            }
        ]
    return [
        {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "content": user_content_to_openai(text_blocks)
            if text_blocks
            else "(image attached in next message)",
        },
        {"role": "user", "content": user_content_to_openai(image_blocks)},
    ]


def _extract_thought_signature(raw: dict) -> str | None:
    """Locate the thought_signature in a Gemini OpenAI-shim response.

    When the response carries tool_calls the signature rides on the
    first one (`tool_calls[0].extra_content.google.thought_signature`);
    text-only responses put it on the message itself
    (`message.extra_content.google.thought_signature`).

    Also probes `tool_calls[0].function.thought_signature` as a
    defensive fallback — OpenClaw's `extractGoogleThoughtSignature`
    checks both paths, suggesting some shim variants stash it nested
    under `function`. Costs nothing and saves us from a silent miss
    if the shape changes.
    """
    msg = (raw.get("choices") or [{}])[0].get("message") or {}
    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        tc = tool_calls[0]
        extra = (tc.get("extra_content") or {}).get("google") or {}
        if sig := extra.get("thought_signature"):
            return sig
        if sig := (tc.get("function") or {}).get("thought_signature"):
            return sig
    extra = (msg.get("extra_content") or {}).get("google") or {}
    return extra.get("thought_signature")
