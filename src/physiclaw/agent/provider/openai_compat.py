"""OpenAI-compatible base — `/chat/completions` wire format.

Vendors that speak this shape (Qwen/DashScope, OpenAI, Moonshot, Google)
inherit from `OpenAICompatibleProvider` and declare `BASE_URL` plus
any auth quirks. This file owns:

  - the request/response flow (`chat`)
  - DTO → wire serialization (`_encode_message`, delegating block-
    level encoding to `wire.py`; the inherited `serialize_history`
    template assembles the list)
  - cache-control marker mechanics (`OpenAICacheMarkers`; the base
    template owns placement — system + latest stubbed tool_result)
  - response parsing (`_parse_response` + `_parse_usage` methods).
    `_parse_usage` tolerates the known shape drift among OpenAI-compat
    endpoints (top-level `cached_tokens` when the nested location is
    empty); a vendor whose `usage` shape differs beyond that overrides
    `_parse_usage` on its own class.

Cache-control marker layout — two anchors, chosen so cached bytes
survive turns AND wakes:

  1. System message (index 0). Session-stable; one cache_creation per
     `cache_key` ever; cache hits the first turn of every wake within
     the 5-min TTL.
  2. Latest superseded screen-obs `tool_result` — found via the typed
     `is_superseded` flag on the source `ToolResultMessage` (set by
     `compact.drop_stale_screens`). Deepest byte-stable point before
     the live multipart image; marking through the image would cache
     bytes that mutate away.

Provider-specific response leakage (Qwen's `reasoning_content`) is
stripped here at parse time so it never re-rides the wire on the next
turn (would break the prefix cache).
"""

import json
import logging
import uuid
from typing import Any, assert_never

import httpx

from physiclaw.agent.engine.dto import (
    AssistantMessage,
    FinishReason,
    Message,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from physiclaw.agent.provider.provider_base import (
    EPHEMERAL_CACHE_CONTROL,
    BaseProvider,
    CacheMarkers,
    ProviderPermanentError,
    ProviderTransientError,
)
from physiclaw.agent.provider.wire import (
    assistant_to_wire,
    tool_result_to_wire,
    tool_to_wire,
    user_content_to_openai,
)

log = logging.getLogger(__name__)


class OpenAICacheMarkers(CacheMarkers):
    """OpenAI-shape marker mechanics: both anchors (system entry,
    stubbed tool_result) carry string content on the wire, so both
    wrap it in a single cache-controlled text block the same way."""

    def mark_system(self, entry: dict) -> dict:
        return _with_cache_marker(entry)

    def mark_stub(self, entry: dict) -> dict:
        return _with_cache_marker(entry)


class OpenAICompatibleProvider(BaseProvider):
    """Base for providers that speak the OpenAI `/chat/completions` wire
    format. See `BaseProvider` for the auth declarations vendors are
    expected to set; this class plugs the wire-shape encoder
    (`_encode_message`) and the `OpenAICacheMarkers` factory into the
    inherited `serialize_history` template, and adds the HTTP request
    flow in `chat()`."""

    CACHE_MARKERS: CacheMarkers = OpenAICacheMarkers()

    async def chat(
        self,
        history: list[Message],
        tools: list[dict],
    ) -> AssistantMessage:
        wire = self.serialize_history(history)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": wire,
        }
        if tools:
            payload["tools"] = [tool_to_wire(t) for t in tools]
            # tool_choice "auto" is the OpenAI default; being explicit avoids
            # provider surprises.
            payload["tool_choice"] = "auto"

        try:
            r = await self._client.post("/chat/completions", json=payload)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            raise ProviderTransientError(f"transport: {e}") from e

        if r.status_code == 429 or r.status_code >= 500:
            log.warning("provider HTTP %s (transient): %s", r.status_code, r.text[:200])
            raise ProviderTransientError(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
            log.error("provider HTTP %s (permanent): %s", r.status_code, r.text[:500])
            raise ProviderPermanentError(f"HTTP {r.status_code}: {r.text[:500]}")

        return self._parse_response(r.json())

    async def list_models(self) -> list[dict]:
        """OpenAI-compatible providers all expose `GET /models`. Returns
        the `data` array verbatim — typically a list of dicts with `id`,
        `object`, `created`, `owned_by`."""
        try:
            r = await self._client.get("/models")
        except (httpx.TransportError, httpx.TimeoutException) as e:
            raise ProviderTransientError(f"transport: {e}") from e
        if r.status_code >= 400:
            raise ProviderPermanentError(f"HTTP {r.status_code}: {r.text[:300]}")
        body = r.json()
        return body.get("data") or []

    # ---------- serialize_history hooks (called by BaseProvider) ----------

    def _encode_message(self, msg: Message) -> dict | None:
        if isinstance(msg, SystemMessage):
            return {"role": "system", "content": msg.content}
        if isinstance(msg, UserMessage):
            return {"role": "user", "content": user_content_to_openai(msg.content)}
        if isinstance(msg, AssistantMessage):
            return assistant_to_wire(msg)
        if isinstance(msg, ToolResultMessage):
            return tool_result_to_wire(msg)
        # Exhaustive on the `Message` Union; runtime guard against future
        # subtypes added without updating the dispatch.
        assert_never(msg)

    # ---------- response parsing (vendor-overridable) ----------

    def _parse_response(self, raw: dict) -> AssistantMessage:
        """OpenAI chat completion → `AssistantMessage`. Drops provider-
        specific content fields (e.g. Qwen's `reasoning_content`) so
        they never leak into engine history; the raw dict is preserved
        on `.raw` for log-side inspection. Vendors with non-standard
        usage shapes override `_parse_usage`, not this method."""
        choice = raw.get("choices", [{}])[0]
        message = choice.get("message") or {}
        finish_raw = choice.get("finish_reason") or "stop"

        content = message.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            try:
                fn = tc.get("function") or {}
                args_str = fn.get("arguments") or "{}"
                try:
                    args = (
                        json.loads(args_str) if isinstance(args_str, str) else args_str
                    )
                    if not isinstance(args, dict):
                        args = {"_raw": args}
                except json.JSONDecodeError:
                    # Principle 4/5: don't silently drop; pass malformed
                    # args through so the validator flags it on dispatch.
                    args = {"_malformed_json": args_str}
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id") or f"auto_{uuid.uuid4().hex[:8]}",
                        name=fn.get("name") or "",
                        arguments=args,
                    )
                )
            except Exception:
                log.exception("failed to parse tool_call: %s", tc)

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls,
            finish_reason=_normalize_finish(finish_raw),
            usage=self._parse_usage(raw),
            raw=raw,
        )

    def _parse_usage(self, raw: dict) -> Usage:
        """OpenAI-standard `usage` block → normalized `Usage`. Cache
        stats live under `prompt_tokens_details.cached_tokens` and
        `prompt_tokens_details.cache_creation_input_tokens`; when the
        nested location is empty, `cached_tokens` is also probed at the
        top of the block — Moonshot K2 sometimes surfaces it there, and
        the probe is a no-op for vendors that never do (standard OpenAI
        shapes carry no top-level key). Vendors whose shape differs
        beyond that override this method on the vendor class."""
        u = (raw or {}).get("usage") or {}
        details = u.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens", 0) or 0)
        if not cached:
            cached = int(u.get("cached_tokens", 0) or 0)
        return Usage(
            prompt_tokens=int(u.get("prompt_tokens", 0) or 0),
            completion_tokens=int(u.get("completion_tokens", 0) or 0),
            cached_tokens=cached,
            cache_creation_tokens=int(
                details.get("cache_creation_input_tokens", 0) or 0
            ),
        )


# ---------- cache marker helper ----------


def _with_cache_marker(entry: dict) -> dict:
    """Shallow-copy `entry` and wrap its string content in a single text
    block carrying an ephemeral cache_control. Caller guarantees string
    content (system message + stubbed tool_result both qualify in the
    OpenAI wire shape)."""
    out = dict(entry)
    out["content"] = [
        {
            "type": "text",
            "text": entry["content"],
            "cache_control": EPHEMERAL_CACHE_CONTROL,
        }
    ]
    return out


# ---------- finish-reason normalization ----------


def _normalize_finish(r: str) -> FinishReason:
    # OpenAI surfaces: stop, length, tool_calls, content_filter, function_call.
    if r == "function_call":
        return FinishReason.TOOL_CALLS
    try:
        return FinishReason(r)
    except ValueError:
        log.warning("unknown finish_reason %r — treating as stop", r)
        return FinishReason.STOP
