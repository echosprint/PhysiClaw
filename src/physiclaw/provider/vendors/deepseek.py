"""DeepSeek — OpenAI-compatible endpoint. Declaration + one knob.

Only `deepseek-v4-flash-vision-exp` (2026-08-21) takes images —
`image_url` base64 data URLs, the shape `wire.py` already emits —
so it's the one model that can drive the loop; a text-only pick
(`deepseek-v4-flash`, `deepseek-v4-pro`) 400s on the first peek.
Images are resized to ~800×800 and billed at ≤384 tokens: fine for
phone screens, lossy on dense text.

Cache markers OFF (live-probed 2026-08-31): caching is automatic
prefix caching (64-token blocks, no opt-in field) — markers buy
nothing, and the docs specify string-only system content, which
`mark_system`'s rewrap would violate (tolerated today, still
pointless risk). Image tokens earned no hits — screens recompute
every turn — so the supersede-stub rewrite costs no cache here.

Usage: live responses duplicate the cache-hit count into the standard
nested `cached_tokens`, and DeepSeek's documented top-level
`prompt_cache_hit_tokens` spelling is covered by the base
`_parse_usage` fallback (same treatment as Moonshot K2). Misses are
ordinary-priced input (auto-stored), not cache writes.

Auth: `DEEPSEEK_API_KEY` env, or `[provider] deepseek_api_key` in
`~/.physiclaw/config.toml`.
"""

from physiclaw.provider.openai_compat import OpenAICompatibleProvider
from physiclaw.provider.provider_base import NO_CACHE_MARKERS


class DeepSeekProvider(OpenAICompatibleProvider):
    PROVIDER_ID = "deepseek"
    BASE_URL = "https://api.deepseek.com/v1"
    CACHE_MARKERS = NO_CACHE_MARKERS
