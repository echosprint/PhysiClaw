"""Moonshot — OpenAI-compatible endpoint. Pure declaration. Provider id
is the API surface (`moonshot`), not the brand (`kimi`).

Auth: `MOONSHOT_API_KEY` env, or `[provider] moonshot_api_key` in
`~/.physiclaw/config.toml`. `BASE_URL` defaults to the China endpoint;
override via `[providers.moonshot] base_url = "https://api.moonshot.ai/v1"`
in user config (a key minted for one domain returns 401 on the other).

Caching: Moonshot honors `cache_control: {type: ephemeral}` markers on
text blocks — same shape as DashScope, so the inherited
`OpenAICacheMarkers` are load-bearing for cross-wake hits. Empirically
(A/B against the API):

  - Markers OFF + identical request          : 100% hit (full byte match)
  - Markers OFF + user-message change        :   0% hit (cache busted)
  - Markers ON  + user-message change        : 100% hit (anchors honored)

PhysiClaw rewrites the wake-volatile content (cron stamps, trigger
text) into the user message right after the system. Without markers,
every wake starts cold even when the system is byte-stable. With
markers, every wake hits cache up to the marked anchors. The earlier
"K2 is purely auto-prefix, markers redundant" reading turned out to
match only the byte-identical case.

Collapse cadence: inherits the base defaults (COLLAPSE_INTERVAL_TURNS
= 20) — same cadence as Anthropic/Qwen. The whole-prefix invalidation
on K2.x makes each collapse more expensive than for anchored caches,
but a longer interval (was 30) noticeably hurt context-length quality
on long sessions; 20 keeps the prompt tighter at the cost of one extra
cache write per ~10 turns.

K2's occasional top-level `usage.cached_tokens` placement is handled
by the base `_parse_usage` fallback in `openai_compat.py`.
"""
from physiclaw.agent.provider.openai_compat import OpenAICompatibleProvider


class MoonshotProvider(OpenAICompatibleProvider):
    PROVIDER_ID = "moonshot"
    BASE_URL = "https://api.moonshot.cn/v1"
