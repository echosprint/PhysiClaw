"""Moonshot — OpenAI-compatible endpoint. Pure declaration. Provider id
is the API surface (`moonshot`), not the brand (`kimi`).

Auth: `MOONSHOT_API_KEY` env, or `[provider] moonshot_api_key` in
`~/.physiclaw/config.toml`. `BASE_URL` defaults to the China endpoint;
override via `[providers.moonshot] base_url = "https://api.moonshot.ai/v1"`
in user config (a key minted for one domain returns 401 on the other).

Caching: on K2, markers looked load-bearing (A/B: user-message change
hit 0% without markers, 100% with). On K3 a three-arm session replay
(logged bytes / byte-stable stub shapes / markers stripped) returned
token-identical cached counts — the cache is content-based and ignores
both `cache_control` and text-block vs bare-string shape, so the
inherited `OpenAICacheMarkers` are harmless uniformity, not
load-bearing. K3's hit collapses (~95% → ~75% on screen turns) are
`compact.drop_stale_screens` rewriting the previous screen to its stub
— the compaction trade-off, not a serialization bug. Entries take
~60–120s to materialize (first 3–4 turns of a session read 0%); TTL is
5–30 min.

Collapse cadence: inherits the base default `COLLAPSE` policy
(interval = 20) — same cadence as Anthropic/Qwen. The whole-prefix invalidation
on K2.x makes each collapse more expensive than for anchored caches,
but a longer interval (was 30) noticeably hurt context-length quality
on long sessions; 20 keeps the prompt tighter at the cost of one extra
cache write per ~10 turns.

K2's occasional top-level `usage.cached_tokens` placement is handled
by the base `_parse_usage` fallback in `openai_compat.py`.
"""

from physiclaw.provider.openai_compat import OpenAICompatibleProvider


class MoonshotProvider(OpenAICompatibleProvider):
    PROVIDER_ID = "moonshot"
    BASE_URL = "https://api.moonshot.cn/v1"
