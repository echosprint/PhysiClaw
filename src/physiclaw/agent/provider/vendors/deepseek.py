"""DeepSeek — OpenAI-compatible endpoint. Stub: vision not supported.

DeepSeek's chat-completions API accepts text-only `content` (no
`image_url` part per the official API reference). PhysiClaw requires
vision on every `peek`, so the provider can't drive the loop. The
`__init__` override raises immediately so misconfiguration is loud.

Auth: `DEEPSEEK_API_KEY` env, or `[provider] deepseek_api_key` in
`~/.physiclaw/config.toml`.
"""

from physiclaw.agent.provider.openai_compat import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    PROVIDER_ID = "deepseek"
    BASE_URL = "https://api.deepseek.com/v1"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "deepseek provider is not implemented — chat endpoint is "
            "text-only and PhysiClaw requires vision for every peek."
        )
