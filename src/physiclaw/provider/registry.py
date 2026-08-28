"""Provider registry — id → class mapping + lookup helpers.

Each concrete vendor declares its own `PROVIDER_ID` / `BASE_URL`;
this module just wires them into a dict and exposes the lookups
(`provider_class`, `provider_key_status`, `make_provider`,
`is_known`).
"""

from physiclaw.provider.provider_base import BaseProvider, Provider
from physiclaw.provider.vendors.anthropic import AnthropicProvider
from physiclaw.provider.vendors.deepseek import DeepSeekProvider
from physiclaw.provider.vendors.google import GoogleProvider
from physiclaw.provider.vendors.moonshot import MoonshotProvider
from physiclaw.provider.vendors.openai import OpenAIProvider
from physiclaw.provider.vendors.qwen import QwenProvider

# Subprocess engine; routed by launcher, not via _PROVIDER_CLASSES.
CLAUDE_CODE_ID = "claude-code"


_PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {
    QwenProvider.PROVIDER_ID: QwenProvider,
    MoonshotProvider.PROVIDER_ID: MoonshotProvider,
    OpenAIProvider.PROVIDER_ID: OpenAIProvider,
    AnthropicProvider.PROVIDER_ID: AnthropicProvider,
    GoogleProvider.PROVIDER_ID: GoogleProvider,
    DeepSeekProvider.PROVIDER_ID: DeepSeekProvider,
}


def in_process_provider_ids() -> tuple[str, ...]:
    """Provider ids handled by the in-process engine. Stubs are
    included — they fail at instantiation with a clear message rather
    than silently disappearing from the menu."""
    return tuple(_PROVIDER_CLASSES)


def is_known(provider_id: str) -> bool:
    return provider_id in _PROVIDER_CLASSES


def provider_class(provider_id: str) -> type[BaseProvider] | None:
    """Class for an in-process provider id, or None if `provider_id`
    isn't one of ours (e.g. `claude-code`)."""
    return _PROVIDER_CLASSES.get(provider_id)


def provider_key_status(provider_id: str) -> tuple[str | None, str | None]:
    """Return `(masked_value_or_None, source_or_None)` for a provider's
    API key, mirroring `BaseProvider._api_key()` resolution exactly so
    CLI displays match what the runtime will pick up.

    Both fields are `None` if the key is unset OR `provider_id` isn't
    a known in-process provider."""
    from physiclaw.common.config import resolve_provider_key

    cls = _PROVIDER_CLASSES.get(provider_id)
    if cls is None:
        return None, None
    val, source = resolve_provider_key(cls._env_vars(), cls._config_key())
    return ("********" if val else None), source


def make_provider(provider_id: str, model_id: str) -> Provider:
    """`(provider_id, model_id)` → `Provider` instance.

    `model_id` is passed through verbatim — the provider's API rejects
    unknown ids on the first chat. Credentials come from the provider's
    own `_api_key()` lookup in `__init__`; stubs raise
    `NotImplementedError` from there with a setup pointer."""
    cls = _PROVIDER_CLASSES.get(provider_id)
    if cls is None:
        raise ValueError(
            f"unknown provider {provider_id!r} "
            f"(known in-process: {', '.join(_PROVIDER_CLASSES)}; "
            f"or use {CLAUDE_CODE_ID!r} for the subprocess engine)"
        )
    return cls(model=model_id)
