"""Anthropic — Claude models via the official `anthropic` SDK.

Distinct from `claude-code` (the subprocess engine that shells out to
the `claude` CLI). This provider calls Anthropic's `/v1/messages`
endpoint directly through `AsyncAnthropic` — wire-format flow lives in
`provider/anthropic_compat.py`.

Auth: `ANTHROPIC_API_KEY` env, or `[provider] anthropic_api_key` in
`~/.physiclaw/config.toml`.
"""

from physiclaw.provider.anthropic_compat import AnthropicCompatibleProvider


class AnthropicProvider(AnthropicCompatibleProvider):
    PROVIDER_ID = "anthropic"
    # Informational only — `AsyncAnthropic` manages the URL itself.
    BASE_URL = "https://api.anthropic.com/v1"
    # INLINE_TOOL_INDEX stays default-True: Claude reads the native
    # `tools=` schema fine, but the ~800-token card is kept as cheap
    # insurance that every tool stays visible in SYSTEM.
