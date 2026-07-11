"""Provider package — one file per vendor, two wire-shape bases.

Provider ids follow OpenClaw's convention: the API surface (vendor /
company), not the brand. So `openai` (not `chatgpt`), `moonshot` (not
`kimi`), `google` (not `gemini`), `qwen` (Alibaba's Qwen API),
`anthropic` (direct API). The existing `claude-code` is a separate
concept (subprocess engine, not direct Anthropic API).

Layout:
  - `provider_base.py`     — `Provider` Protocol, errors,
                             `CacheMarkers` factory contract,
                             `BaseProvider` (auth + HTTP-client +
                             `SYSTEM_PROMPT_FRAGMENT` hook +
                             `serialize_history` template)
  - `openai_compat.py`     — `OpenAICompatibleProvider`:
                             `/chat/completions` wire flow +
                             `OpenAICacheMarkers` + response parser
  - `anthropic_compat.py`  — `AnthropicCompatibleProvider`:
                             `/v1/messages` wire flow (Anthropic SDK) +
                             `AnthropicCacheMarkers` (per-block)
  - `wire.py`              — DTO ↔ OpenAI wire-format adapters used by
                             `openai_compat.py`, plus the shared
                             `encode_content` block dispatch both wire
                             shapes route through
  - `registry.py`          — id → class map, lookup helpers
                             (`make_provider`, `provider_class`,
                             `provider_key_status`, `is_known`,
                             `in_process_provider_ids`, `CLAUDE_CODE_ID`)
  - `vendors/`             — one file per concrete vendor:
      - `qwen.py`      — `QwenProvider` (DashScope)   — OpenAI-compat
      - `moonshot.py`  — `MoonshotProvider` (Kimi)    — OpenAI-compat
      - `openai.py`    — `OpenAIProvider` (GPT-5)     — OpenAI-compat
      - `deepseek.py`  — `DeepSeekProvider` (stub)    — OpenAI-compat
      - `google.py`    — `GoogleProvider` (Gemini)    — OpenAI-compat
                         with shim-quirk overrides
      - `anthropic.py` — `AnthropicProvider` (Claude) — Anthropic-compat

This `__init__.py` is a thin re-export surface — import from here
externally; logic lives in the modules above.

Selection: the user picks a `provider/model` ref (e.g. `qwen/qwen3.6-plus`)
in `~/.physiclaw/config.toml [agent] model`. The launcher parses the ref
and routes on the provider id — `claude-code` goes to the subprocess
engine, anything else here goes to the in-process engine via
`make_provider(provider_id, model_id)`.
"""
from physiclaw.agent.provider.anthropic_compat import AnthropicCompatibleProvider
from physiclaw.agent.provider.openai_compat import OpenAICompatibleProvider
from physiclaw.agent.provider.provider_base import (
    BaseProvider,
    Provider,
    ProviderError,
    ProviderPermanentError,
    ProviderTransientError,
)
from physiclaw.agent.provider.registry import (
    CLAUDE_CODE_ID,
    in_process_provider_ids,
    is_known,
    make_provider,
    provider_class,
    provider_key_status,
)
from physiclaw.agent.provider.vendors.anthropic import AnthropicProvider
from physiclaw.agent.provider.vendors.deepseek import DeepSeekProvider
from physiclaw.agent.provider.vendors.google import GoogleProvider
from physiclaw.agent.provider.vendors.moonshot import MoonshotProvider
from physiclaw.agent.provider.vendors.openai import OpenAIProvider
from physiclaw.agent.provider.vendors.qwen import QwenProvider
from physiclaw.agent.provider.wire import (
    assistant_to_wire,
    mcp_blocks_to_content_blocks,
    tool_result_to_wire,
    tool_to_wire,
    user_content_to_openai,
)

__all__ = [
    "CLAUDE_CODE_ID",
    "AnthropicCompatibleProvider",
    "AnthropicProvider",
    "BaseProvider",
    "DeepSeekProvider",
    "GoogleProvider",
    "MoonshotProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "Provider",
    "ProviderError",
    "ProviderPermanentError",
    "ProviderTransientError",
    "QwenProvider",
    "assistant_to_wire",
    "in_process_provider_ids",
    "is_known",
    "make_provider",
    "mcp_blocks_to_content_blocks",
    "provider_class",
    "provider_key_status",
    "tool_result_to_wire",
    "tool_to_wire",
    "user_content_to_openai",
]
