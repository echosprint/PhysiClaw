"""Provider Protocol, errors, and the abstract base.

This file is the slim core: no wire-format, no HTTP request flow. Two
concrete bases sit on top of `BaseProvider`:
  - `OpenAICompatibleProvider` (in `openai_compat.py`) — vendors
    speaking the OpenAI `/chat/completions` shape (Qwen, OpenAI,
    Moonshot, DeepSeek, Google).
  - `AnthropicCompatibleProvider` (in `anthropic_compat.py`) —
    vendors speaking Anthropic's `/v1/messages` shape (Anthropic).

Vendor classes declare `PROVIDER_ID`, `BASE_URL`, and — when the
defaults don't fit — `API_KEY_ENV_VARS` (env-var aliases) or
`SYSTEM_PROMPT_FRAGMENT` (reasoning-convention addendum). `BASE_URL`
is overridable per-instance via `~/.physiclaw/config.toml`'s
`[providers.<id>] base_url = "..."` (e.g. Moonshot's .cn vs .ai split,
or pointing at a proxy).

Principle 2: normalize at the boundary — providers return
`AssistantMessage` regardless of their wire shape.
Principle 3: preserve the real `finish_reason` — never derive it.
"""

import logging
import os
from typing import Protocol

import httpx

from physiclaw.contract.dto import (
    AssistantMessage,
    CollapsePolicy,
    ImageBlock,
    Message,
    SystemMessage,
    ToolResultMessage,
)

log = logging.getLogger(__name__)


# ---------- errors / Protocol ----------


class ProviderError(Exception):
    """Base for provider failures."""


class ProviderTransientError(ProviderError):
    """Transport issue, timeout, 429, or 5xx — worth retrying."""


class ProviderPermanentError(ProviderError):
    """4xx (except 429) — retries will keep failing, fail fast."""


def describe(e: BaseException) -> str:
    """`ClassName: message`, or just `ClassName` when the message is
    empty — httpx timeouts commonly str() to '', which made wrapped
    errors log as the famously unhelpful `transport: `."""
    msg = str(e)
    return f"{type(e).__name__}: {msg}" if msg else type(e).__name__


class Provider(Protocol):
    model: str
    COLLAPSE: CollapsePolicy

    async def chat(
        self,
        history: list[Message],
        tools: list[dict],
    ) -> AssistantMessage: ...

    def serialize_history(self, history: list[Message]) -> list[dict]:
        """Convert engine DTOs into the wire-format messages this provider
        will send, with cache-control markers applied. Engine calls this
        for trace logging; `chat()` calls the same internally before
        POSTing — the two calls produce equivalent fresh wire so the log
        faithfully captures what hits the API."""
        ...

    async def aclose(self) -> None: ...


# ---------- shared constants ----------


# 5-minute TTL prefix-cache marker. DashScope, Moonshot K2, and
# Anthropic all accept `{type: "ephemeral"}` in their respective
# surrounding shapes. Treat as immutable — never mutate.
EPHEMERAL_CACHE_CONTROL = {"type": "ephemeral"}


# ---------- cache-marker factory ----------


class CacheMarkers:
    """Wire-shape cache-marker mechanics — *how* a `cache_control`
    attaches to a wire entry. The `serialize_history` template owns
    *where* markers go (system entry at index 0, latest superseded
    tool_result, and the two moving tail anchors around the live
    screen); each wire shape ships one `CacheMarkers` subclass saying
    how its entries carry the marker, and a provider picks its factory
    once via the `CACHE_MARKERS` class attribute.

    All hooks return a marked shallow copy; caller-held dicts are
    never mutated. Defaults are identity, which makes the base the
    null object — a provider that must not mark declares
    `NO_CACHE_MARKERS` (e.g. Google, whose shim ignores the field and
    whose implicit cache the on/off wrapping would perturb) — and lets
    a shape override only the placements its cache semantics actually
    reward (Anthropic marks system on the top-level `system` payload
    field in `chat()`, not on a messages entry, and is the only shape
    that needs `mark_tail` — its cache stops at explicit breakpoints,
    while the OpenAI-shape vendors extend over stable prefixes on
    their own).
    """

    def mark_system(self, entry: dict) -> dict:
        return entry

    def mark_stub(self, entry: dict) -> dict:
        return entry

    def mark_tail(self, entry: dict) -> dict:
        return entry


# The null object — identity on all hooks. Declaring this (rather
# than a `None` sentinel) keeps `serialize_history` branch-free.
NO_CACHE_MARKERS = CacheMarkers()


def _has_image(content) -> bool:
    """True iff a tool result's DTO content carries an `ImageBlock` —
    i.e. it is (or was, until superseded) a live screen capture."""
    return isinstance(content, list) and any(isinstance(b, ImageBlock) for b in content)


def _tail_anchor_indexes(results: list[tuple[int, bool]]) -> list[int]:
    """The moving tail anchors for one request, from `serialize_history`'s
    (out-index, carries-image) record of every tool result: the last
    result BEFORE the live screen (survives the next supersede rewrite)
    and the last result overall (advances every turn, dies on the next
    supersede). Deduped; one entry when they coincide or when no live
    screen exists."""
    anchors: list[int] = []
    images = [i for i, has in results if has]
    if images:
        before = [i for i, _ in results if i < images[-1]]
        if before:
            anchors.append(before[-1])
    if results:
        anchors.append(results[-1][0])
    return list(dict.fromkeys(anchors))


# ---------- BaseProvider ----------


class BaseProvider:
    """Auth resolution + HTTP-client construction.

    Subclasses (`OpenAICompatibleProvider`, `AnthropicCompatibleProvider`)
    layer wire-format and request/response flow on top.

    Subclass MUST set:
      - `PROVIDER_ID` — short id, also the registry key.
      - `BASE_URL` — endpoint, e.g. `"https://api.openai.com/v1"`.

    Subclass MAY set:
      - `API_KEY_ENV_VARS` — env vars to consult (in order, first hit
        wins). Defaults to `("<PROVIDER_ID>_API_KEY",)`. Override only
        when a vendor accepts more than one (e.g. Qwen takes both
        `QWEN_API_KEY` and `DASHSCOPE_API_KEY`).
      - `SYSTEM_PROMPT_FRAGMENT` — per-vendor reasoning-convention
        addendum for the system prompt (e.g. Qwen's `<think>...</think>`
        wrapper).

    Subclass MAY override:
      - `_build_client()` if the wire client isn't a stock
        `httpx.AsyncClient` (e.g. Anthropic's `AsyncAnthropic` SDK)
      - `_api_key()` if auth doesn't fit the env-var/config pattern
      - `_missing_key_message()` for a richer error string
      - `_model_env_var()` if the env override isn't `<ID>_MODEL`
      - `chat()` and `serialize_history()` — provided by the wire-shape
        intermediate base; vendors normally don't touch them

    `BASE_URL` is overridable per-instance via `~/.physiclaw/config.toml`'s
    `[providers.<id>] base_url = "..."` so users can point at proxies
    or alt endpoints without code changes.
    """

    PROVIDER_ID: str = ""
    BASE_URL: str = ""
    API_KEY_ENV_VARS: tuple[str, ...] = ()

    # Per-vendor system-prompt addendum; see `system_prompt_fragment()`.
    SYSTEM_PROMPT_FRAGMENT: str = ""

    # Whether the SYSTEM prompt carries the inline `## Tooling` index card.
    # Open-weight models (Qwen, Moonshot) routinely miss tools that appear
    # only in the native `tools=` schema, so the card is a redundant anchor
    # for them — default on. Vendors whose models reliably read the schema
    # set False and save the duplication (~700 tokens per request).
    INLINE_TOOL_INDEX: bool = True

    # Cache-marker factory for this provider's wire shape. Each
    # wire-shape base sets its own (`OpenAICacheMarkers` /
    # `AnthropicCacheMarkers`); a vendor whose endpoint ignores markers
    # resets to `NO_CACHE_MARKERS` — see `GoogleProvider`.
    CACHE_MARKERS: CacheMarkers = NO_CACHE_MARKERS

    # Turn-age summary collapse — see `compact.collapse_old_turns`.
    # The policy lives here so vendor-specific tuning (cache mechanics
    # differ per provider) can override it in one place; a vendor
    # declares its own `COLLAPSE = CollapsePolicy(...)`, which validates
    # the triple at import time. The defaults (F=30, K=10, I=20) are an
    # EOQ optimum for vendors with anchored caches (Anthropic/Qwen) and
    # match every shipping provider; Moonshot carries the highest
    # cost-per-collapse (whole-prefix invalidation vs anchored caches)
    # but accepts the tax in exchange for tighter prompts on long
    # sessions — see `MoonshotProvider` for the trade-off rationale.
    COLLAPSE: CollapsePolicy = CollapsePolicy(first_at=30, keep=10, interval=20)

    def __init__(
        self,
        model: str | None = None,
        timeout: float = 120.0,
        base_url: str | None = None,
    ):
        if not (self.PROVIDER_ID and self.BASE_URL):
            raise RuntimeError(
                f"{type(self).__name__}: PROVIDER_ID and BASE_URL must "
                "be set on the subclass"
            )
        key = self._api_key()
        if not key:
            raise RuntimeError(self._missing_key_message())
        # Model resolution: explicit arg → env override → empty (chat()
        # will surface a clear API error if invoked without a model).
        self.model = model or os.environ.get(self._model_env_var()) or ""
        self._client = self._build_client(key, timeout=timeout, base_url=base_url)

    # ---------- HTTP client ----------

    def _build_client(self, key: str, *, timeout: float, base_url: str | None):
        """Construct the underlying HTTP client. Default is a stock
        `httpx.AsyncClient` with Bearer auth (works for every OpenAI-
        compatible vendor). Override when a vendor uses an SDK or a
        non-Bearer auth scheme."""
        return httpx.AsyncClient(
            base_url=base_url or self._resolved_base_url(),
            timeout=timeout,
            headers={"Content-Type": "application/json", **self._auth_headers(key)},
            # The provider is an external endpoint, so honour the system proxy
            # (HTTP(S)_PROXY) — behind a corporate/VPN/region proxy the runtime
            # may only reach the LLM API through it. This is the deliberate
            # inverse of the localhost clients (MCP transport, status poll,
            # phone-watch hook), which stay off the proxy for 127.0.0.1 via
            # trust_env=platform.TRUST_PROXY_ENV plus the NO_PROXY loopback
            # pin from `normalize_proxy_env()`. Don't unify the two.
            trust_env=True,
        )

    @classmethod
    def _resolved_base_url(cls) -> str:
        """Class `BASE_URL` unless the user has set
        `[providers.<id>] base_url = "..."` in `~/.physiclaw/config.toml`."""
        from physiclaw.common.config import provider_base_url_override

        return provider_base_url_override(cls.PROVIDER_ID) or cls.BASE_URL

    def _auth_headers(self, key: str) -> dict[str, str]:
        """HTTP headers for authentication. Default is OpenAI's
        `Authorization: Bearer <key>`. Used by the default
        `_build_client`; vendors that override `_build_client` (e.g.
        Anthropic's SDK) typically don't need this hook."""
        return {"Authorization": f"Bearer {key}"}

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---------- auth lookup ----------

    def _api_key(self) -> str | None:
        """Default lookup: env vars (in `API_KEY_ENV_VARS` order, defaulting
        to `<ID>_API_KEY` by convention) → config.toml `[provider]
        <PROVIDER_ID>_api_key`. Override for non-standard auth (OAuth,
        sigv4, etc.)."""
        from physiclaw.common.config import resolve_provider_key

        return resolve_provider_key(self._env_vars(), self._config_key())[0]

    def _missing_key_message(self) -> str:
        envs = " / ".join(self._env_vars())
        return (
            f"{self.PROVIDER_ID} credential not found. Set {envs} env var "
            f"or [provider] {self._config_key()} in ~/.physiclaw/config.toml."
        )

    @classmethod
    def _env_vars(cls) -> tuple[str, ...]:
        return cls.API_KEY_ENV_VARS or (f"{cls.PROVIDER_ID.upper()}_API_KEY",)

    @classmethod
    def _config_key(cls) -> str:
        return f"{cls.PROVIDER_ID}_api_key"

    @classmethod
    def _model_env_var(cls) -> str:
        return f"{cls.PROVIDER_ID.upper()}_MODEL"

    @classmethod
    def system_prompt_fragment(cls) -> str:
        """Per-vendor system-prompt addendum (e.g. Qwen's `<think>...</think>`
        wrapper instructions). Default: empty. Declare
        `SYSTEM_PROMPT_FRAGMENT` on the vendor class when the model
        needs an explicit reasoning convention."""
        return cls.SYSTEM_PROMPT_FRAGMENT

    # ---------- DTO → wire (template method, shared across wire shapes) ----------

    def serialize_history(self, history: list[Message]) -> list[dict]:
        """Single-pass DTO history → provider wire-format messages, with
        cache markers attached (via the `CACHE_MARKERS` factory; the
        default `NO_CACHE_MARKERS` is identity) to:
          - the `SystemMessage` at index 0,
          - the latest `ToolResultMessage` flagged `is_superseded`, and
          - two moving tail anchors (both via `mark_tail`, deduped, so
            at most 4 breakpoints total): the last `ToolResultMessage`
            BEFORE the live screen (the deepest point that survives the
            next `drop_stale_screens` rewrite — an entry covering the
            live view dies unread the turn a new screen supersedes it),
            and the last `ToolResultMessage` overall (advances every
            turn, so note-only stretches read for pennies; it dies on
            the next supersede, which the pre-view anchor then backs
            up). Models order a turn's results differently ([note,
            view] vs [view, note]), so "before the live screen" is
            positional, not type-based. The volatile plan/scratchpad
            tails are UserMessages appended after all anchors, outside
            the cached prefix.

        Subclasses implement `_encode_message` (DTO → wire dict, list of
        wire dicts, or `None` to skip). Most encodings are 1:1; the list
        form is for vendor-specific splits — e.g. Google's shim rejects
        `image_url` parts in `role: tool`, so `GoogleProvider` splits a
        ToolResultMessage with images into [text-only tool, synthetic
        user with images]. Anthropic's `SystemMessage` returns `None`
        (system rides outside the messages array). Cache markers attach
        to the first entry of any split; superseded results don't carry
        images, so their list is always single-element."""
        markers = self.CACHE_MARKERS
        out: list[dict] = []
        last_stub_idx: int | None = None
        results: list[tuple[int, bool]] = []  # (out index, carries image)
        for i, msg in enumerate(history):
            entries = self._encode_message(msg)
            if entries is None:
                continue
            if isinstance(entries, dict):
                entries = [entries]
            if not entries:
                continue
            if i == 0 and isinstance(msg, SystemMessage):
                entries[0] = markers.mark_system(entries[0])
            elif isinstance(msg, ToolResultMessage):
                results.append((len(out), _has_image(msg.content)))
                if msg.is_superseded:
                    last_stub_idx = len(out)
            out.extend(entries)
        if last_stub_idx is not None:
            out[last_stub_idx] = markers.mark_stub(out[last_stub_idx])
        for idx in _tail_anchor_indexes(results):
            if idx != last_stub_idx:
                out[idx] = markers.mark_tail(out[idx])
        return out

    def _encode_message(self, msg: Message) -> dict | list[dict] | None:
        """Encode one DTO into a provider wire dict (or list of dicts
        for vendor splits, or `None` to skip). MUST be implemented by
        wire-shape subclasses."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _encode_message"
        )

    # ---------- request flow: provided by wire-shape subclasses ----------

    async def chat(
        self,
        history: list[Message],
        tools: list[dict],
    ) -> AssistantMessage:
        raise NotImplementedError(
            f"{type(self).__name__} must inherit from a wire-shape base "
            "(OpenAICompatibleProvider or AnthropicCompatibleProvider)"
        )

    async def list_models(self) -> list[dict]:
        """Live model list from the provider's `/v1/models` endpoint.

        Each entry is a dict with at least an `id` field; vendors may
        include `display_name`, `created_at`, `owned_by`, etc. — caller
        normalizes for display. Implemented by the wire-shape bases;
        BaseProvider raises so a missing implementation surfaces clearly."""
        raise NotImplementedError(
            f"{type(self).__name__} must inherit from a wire-shape base"
        )
