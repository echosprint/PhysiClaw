"""Model + provider selection — the agent-side half of configuration.

Split from ``common.config`` (the TOML loader) so agent-facing changes
(model refs, credential resolution, provider endpoint overrides) stop
colliding with loader changes in one file. ``config`` re-exports
everything here, so existing
``from physiclaw.common.config import model_ref`` call sites keep
working unchanged.

Order: env var > config.toml > raise (or ``None`` for credentials).
There is no implicit default model — the user must configure one. Refs
use ``provider/model`` shape.
"""

import os
import tomllib

from physiclaw.common.text import read_text

# CONFIG access is deferred into the function bodies, and always reads
# through the module (`config.CONFIG`), never `from config import CONFIG`:
# `set_dotted` / `unset_dotted` REBIND `config.CONFIG` after a write, and
# these resolvers must see the fresh object. Deferring also keeps the
# import cycle with `config` (which re-exports this module) harmless in
# either import order.

MODEL_ENV_VAR = "PHYSICLAW_MODEL"

_NO_MODEL_MSG = (
    "no model configured.\n"
    "  Quick start:\n"
    "    physiclaw models key <provider>     # e.g. anthropic, openai, qwen\n"
    "    physiclaw models use <provider/model>\n"
    f"  Or set {MODEL_ENV_VAR}=<provider>/<model> in your shell."
)


def model_ref() -> str:
    """Resolve effective model ref: PHYSICLAW_MODEL > [agent] model > raise.

    Returns a `provider/model` string like `"qwen/qwen3.6-plus"`. Use
    `parse_model_ref` to split into the two parts. Display callers
    that want the source label too should call `model_ref_with_source`.
    """
    return model_ref_with_source()[0]


def model_ref_with_source() -> tuple[str, str]:
    """`(ref, source)` for the active model — same env > config order as
    `model_ref`. Raises `RuntimeError` when nothing is configured.
    `source` is a human-readable string for log / diagnostic output
    (`"PHYSICLAW_MODEL env"` or `"config.toml [agent] model"`).
    """
    from physiclaw.common import config

    if ref := os.environ.get(MODEL_ENV_VAR):
        return ref, f"{MODEL_ENV_VAR} env"
    if config.CONFIG.agent.model:
        return config.CONFIG.agent.model, "config.toml [agent] model"
    raise RuntimeError(_NO_MODEL_MSG)


def parse_model_ref(ref: str) -> tuple[str, str]:
    """Split `"provider/model-id"` on the FIRST slash.

    `"qwen/qwen3.6-plus"`  →  `("qwen", "qwen3.6-plus")`.
    `"openrouter/openai/gpt-5"`  →  `("openrouter", "openai/gpt-5")`.
    """
    if "/" not in ref:
        raise ValueError(
            f"model ref {ref!r} must be 'provider/model' (e.g. 'openai/gpt-5')"
        )
    provider_id, model_id = ref.split("/", 1)
    if not (provider_id and model_id):
        raise ValueError(f"model ref {ref!r} has empty provider or model segment")
    return provider_id, model_id


def resolve_provider_key(
    env_vars: tuple[str, ...],
    config_key: str,
) -> tuple[str | None, str | None]:
    """Generic credential resolver. Returns ``(key, source)``; both
    ``None`` if not set anywhere.

    ``env_vars`` are checked in order (first hit wins). If none match,
    falls through to ``CONFIG.provider.<config_key>``. Empty strings in
    config count as "unset" and fall through. ``source`` is a
    human-readable string for diagnostic output (``"OPENAI_API_KEY env"``
    or ``"config.toml [provider] openai_api_key"``).
    """
    from physiclaw.common import config

    for var in env_vars:
        val = os.environ.get(var)
        if val:
            return val, f"{var} env"
    val = getattr(config.CONFIG.provider, config_key, "")
    if val:
        return val, f"config.toml [provider] {config_key}"
    return None, None


def provider_base_url_override(provider_id: str) -> str | None:
    """Read `[providers.<provider_id>] base_url` from user config.toml.
    Lets users point a builtin provider at a proxy / alt endpoint
    (e.g. Moonshot's .ai vs .cn). Returns None when unset or the file
    is absent. Called once at provider construction — no caching."""
    from physiclaw.common import config

    try:
        raw = read_text(config.config_path())
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        # Encoding errors get the same fail-soft as a missing file —
        # this code path is best-effort (returns None when no override).
        # The friendly recovery hint comes from `load()` when the user
        # actually runs a command that needs the config.
        return None
    try:
        doc = tomllib.loads(raw)
    except tomllib.TOMLDecodeError:
        return None
    # Shape-guard every level: `config.load()` rejects a malformed
    # `[providers]` loudly (`_validate_providers`), but this reader
    # parses the raw file independently and can still see it — e.g. when
    # the import-time load failed soft to defaults. Fail soft per this
    # function's contract, don't AttributeError.
    providers = doc.get("providers")
    if not isinstance(providers, dict):
        return None
    entry = providers.get(provider_id)
    if not isinstance(entry, dict):
        return None
    val = entry.get("base_url")
    return val if isinstance(val, str) else None
