"""OpenAI — native OpenAI endpoint. Pure declaration.

Provider id is `openai` (matches OpenClaw / Anthropic-SDK convention —
the API surface, not the brand `chatgpt`). Reasoning models (gpt-5,
o4-mini) surface chain-of-thought via API field, not a prompt wrapper —
so no `SYSTEM_PROMPT_FRAGMENT`.

Auth: `OPENAI_API_KEY` env, or `[provider] openai_api_key` in
`~/.physiclaw/config.toml`.
"""
from physiclaw.agent.provider.openai_compat import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    PROVIDER_ID = "openai"
    BASE_URL = "https://api.openai.com/v1"
