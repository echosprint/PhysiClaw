"""``physiclaw models`` — list, inspect, and switch the active model.

Sugar over the generic ``physiclaw config`` interface for the three
model-relevant TOML fields:

  - ``[agent] model``           → ``physiclaw models use <ref>``
  - ``[provider] *_api_key``    → ``physiclaw models key <provider> [val]``
                                  / ``physiclaw models keys``

Models come from each vendor's live ``/v1/models`` endpoint, cached
locally by ``physiclaw models discover <provider>`` (and automatically
by ``models key`` right after writing the key). PhysiClaw doesn't ship
a curated model list — discovery is the source of truth.

Subcommands:

  - ``physiclaw models``                — bare: active ref + source +
                                          key status for active provider
  - ``physiclaw models list [provider]``— per-provider discovery cache
  - ``physiclaw models use <provider/model>``
                                        — switch active model. Exact
                                          ref required (e.g.
                                          ``openai/gpt-5.4``). Run
                                          ``models list`` to see
                                          candidates. (alias: ``set``)
  - ``physiclaw models key <provider> [<value>]``
                                        — set provider API key;
                                          prompts with hidden input if
                                          ``<value>`` is omitted
  - ``physiclaw models keys``           — list every provider's key
                                          status (env / config / unset),
                                          values masked
  - ``physiclaw models discover <provider>``
                                        — re-fetch the live model list
"""

from typing import Annotated

import typer

from physiclaw.common import config as _config
from physiclaw.cli._format import exit_error, next_hint, ok, section, warn

# Provider package imports happen inside command bodies — pulling
# `agent.provider` at module load drags httpx (~80ms) into every
# `physiclaw --help` invocation.

# Provider-id aliases for discovery cache + API key sourcing.
# Read as: <alias> reuses <target>'s catalog and key.
# claude-code routes through the `claude` CLI, which talks to the
# Anthropic API — its model ids and key live under anthropic.
# Hardcoded (rather than imported as `CLAUDE_CODE_ID`) to keep this
# module's load cost independent of `agent.provider`.
_PROVIDER_ALIAS: dict[str, str] = {
    "claude-code": "anthropic",
}


def _discovery_source(provider_id: str) -> str:
    """Provider id whose discovery cache (and API key) backs this one.
    Returns the input unchanged unless `provider_id` is an alias."""
    return _PROVIDER_ALIAS.get(provider_id, provider_id)


def _known_provider_ids() -> tuple[str, ...]:
    """All provider ids the CLI accepts — in-process + alias keys.
    Lazy-imports `agent.provider` so `physiclaw --help` stays fast."""
    from physiclaw.agent.provider import in_process_provider_ids

    return (*in_process_provider_ids(), *_PROVIDER_ALIAS.keys())


models_app = typer.Typer(
    help="List, inspect, and switch the active model.",
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
    add_completion=False,
)


def _key_config_path(provider_id: str) -> str:
    """Dotted config path for one provider's API key — single source of
    truth so callers don't reconstruct `f"provider.{x}_api_key"` ad-hoc."""
    return f"provider.{provider_id}_api_key"


def _format_key_row(provider_id: str, *, indent: int = 2) -> str:
    """One-line key status for a provider, masked. Resolution lives in
    `provider.provider_key_status` — same source the runtime uses."""
    from physiclaw.agent.provider import provider_key_status

    pad = " " * indent
    masked, source = provider_key_status(provider_id)
    if masked is None:
        return f"{pad}{provider_id} api key: (unset)"
    return f"{pad}{provider_id} api key: {masked}  [{source}]"


@models_app.callback()
def _root(ctx: typer.Context) -> None:
    """Bare `physiclaw models` shows status; subcommands handle the rest."""
    if ctx.invoked_subcommand is not None:
        return
    typer.echo(section("Active model"))
    try:
        ref, source = _config.model_ref_with_source()
    except RuntimeError:
        typer.echo(
            warn(
                "none — run `physiclaw models key <provider>`, "
                "then `physiclaw models use <provider/model>`"
            )
        )
        typer.echo()
        typer.echo(next_hint("physiclaw models list"))
        return
    try:
        provider_id, model_id = _config.parse_model_ref(ref)
    except ValueError as e:
        typer.echo(warn(f"{ref} (from {source}) — invalid ref: {e}"))
        return
    typer.echo(ok(f"{ref}  (from {source})"))
    typer.echo(f"  {provider_id}/{model_id}")
    typer.echo(_format_key_row(provider_id, indent=2))


@models_app.command("list")
def _list(
    provider: Annotated[
        str | None,
        typer.Argument(
            help="Filter to one provider (e.g. `qwen`). Omit for all.",
        ),
    ] = None,
) -> None:
    """List discovered models per provider.

    Reads each provider's discovery cache. If a provider has never been
    discovered, prints a hint to run `physiclaw models discover`.
    `claude-code` reuses anthropic's catalog — the `claude` CLI talks to
    the Anthropic API, so its accepted model ids are anthropic's.
    """
    from physiclaw.agent.provider import discovered

    known_all = _known_provider_ids()
    if provider is not None and provider not in known_all:
        typer.echo(
            warn(f"unknown provider {provider!r} (known: {', '.join(known_all)})")
        )
        raise typer.Exit(code=1)

    targets = (provider,) if provider else known_all
    for pid in targets:
        source = _discovery_source(pid)
        typer.echo(section(pid))
        ids = sorted(discovered.model_ids(source))
        if not ids:
            typer.echo(
                f"  (no discovery cache — run `physiclaw models discover {pid}`)"
            )
        else:
            for mid in ids:
                typer.echo(f"  {pid}/{mid}")
        typer.echo()


def _use_impl(ref: str) -> None:
    """Switch the active model — validates the `provider/model` ref
    against the discovery cache and writes to `[agent] model`. Run
    `physiclaw models discover <provider>` first if the cache is empty."""
    from physiclaw.agent.provider import discovered

    if "/" not in ref:
        exit_error(
            f"{ref!r} is not a `provider/model` ref. "
            "Pass an exact ref like `openai/gpt-5.4` "
            "(run `physiclaw models list` to see candidates)."
        )

    try:
        provider_id, model_id = _config.parse_model_ref(ref)
    except ValueError as e:
        exit_error(str(e))

    known_all = _known_provider_ids()
    if provider_id not in known_all:
        exit_error(f"unknown provider {provider_id!r} (known: {', '.join(known_all)})")

    source = _discovery_source(provider_id)
    if not discovered.is_cached(source, model_id):
        # claude-code with an empty anthropic cache: a subscription-only
        # user has no API key to run discovery with, and the `claude`
        # CLI validates model ids itself — let the ref through.
        skip = provider_id == "claude-code" and not discovered.model_ids(source)
        if not skip:
            exit_error(
                f"model {model_id!r} not in {provider_id} discovery cache.\n"
                f"  hint: run `physiclaw models discover {provider_id}` "
                "to refresh the live list, then retry."
            )

    try:
        _config.set_dotted("agent.model", ref)
    except _config.ConfigError as e:
        exit_error(str(e))
    typer.echo(ok(f"agent.model = {ref}"))
    typer.echo(f"  {provider_id}/{model_id}")
    typer.echo("Restart `physiclaw server` to apply.")


@models_app.command("use")
def _use(
    ref: Annotated[
        str,
        typer.Argument(
            help="Exact `provider/model` ref, e.g. `openai/gpt-5.4` "
            "or `claude-code/claude-sonnet-4-6`.",
        ),
    ],
) -> None:
    """Switch the active model. Pass an exact `provider/model` ref. Run
    `physiclaw models list` to see candidates."""
    _use_impl(ref)


models_app.command("set", hidden=True)(_use)


@models_app.command("key")
def _key(
    provider: Annotated[
        str,
        typer.Argument(help="Provider id, e.g. `moonshot`, `qwen`, `anthropic`."),
    ],
    value: Annotated[
        str | None,
        typer.Argument(
            help="Key value. Omit for an interactive hidden prompt "
            "(keeps the secret out of shell history).",
        ),
    ] = None,
) -> None:
    """Set the API key for one provider — writes
    `[provider] <id>_api_key`. Prompts with hidden input if `value` is
    omitted; that's the safer default since shell history retains
    args. After writing, fetches the live model list from the
    provider's API so you can pick one with `models use` immediately."""
    if provider in _PROVIDER_ALIAS:
        target = _PROVIDER_ALIAS[provider]
        exit_error(
            f"{provider} reuses {target}'s key — set it there.\n"
            f"  hint: run `physiclaw models key {target}`."
        )
    known_all = _known_provider_ids()
    if provider not in known_all:
        exit_error(f"unknown provider {provider!r} (known: {', '.join(known_all)})")
    if value is None:
        value = typer.prompt(f"{provider} api key", hide_input=True)
    path = _key_config_path(provider)
    try:
        _config.set_dotted(path, value)
    except _config.ConfigError as e:
        exit_error(str(e))
    typer.echo(ok(f"{path} set"))
    typer.echo()
    try:
        models = _fetch_live_models(provider)
    except Exception as e:
        typer.echo(warn(f"couldn't fetch live models — {type(e).__name__}: {e}"))
        typer.echo(next_hint(f"physiclaw models discover {provider}"))
        return
    _print_live_models_table(provider, models)


def _fetch_live_models(provider: str) -> list[dict]:
    """Build a provider client and fetch its `/v1/models` list. Raises
    on any failure (caller decides whether fatal)."""
    import asyncio

    from physiclaw.agent.provider import provider_class

    cls = provider_class(provider)
    p = cls(model="")

    async def _run():
        try:
            return await p.list_models()
        finally:
            await p.aclose()

    return asyncio.run(_run())


def _print_live_models_table(
    provider: str,
    models: list[dict],
    *,
    display: str | None = None,
) -> None:
    """Save the live list to the discovery cache and print it.

    `provider` is the cache key (where the list is saved).
    `display` is what to show in the header/hints; defaults to `provider`.
    Diverges only for aliases like `claude-code → anthropic`, where the
    user typed `claude-code` and should see that name echoed back.
    """
    from physiclaw.agent.provider import discovered

    discovered.save(provider, models)
    label = display or provider

    typer.echo(section(f"{label} — {len(models)} model(s) live"))
    for m in models:
        typer.echo(f"    {m.get('id', '')}")
    typer.echo()
    typer.echo(next_hint(f"physiclaw models use {label}/<id>           # pick one"))
    typer.echo(
        next_hint(
            f"physiclaw models discover {label}     # re-fetch later for new releases"
        )
    )


@models_app.command("keys")
def _keys() -> None:
    """List every provider's API-key status (masked, with source —
    env var or config file). Mirrors `BaseProvider._api_key()`
    resolution so what's shown is what the server will use."""
    from physiclaw.agent.provider import in_process_provider_ids

    typer.echo(section("Provider API keys"))
    for pid in in_process_provider_ids():
        typer.echo(_format_key_row(pid, indent=2))


@models_app.command("discover")
def _discover(
    provider: Annotated[
        str,
        typer.Argument(
            help="Provider id, e.g. `openai`, `qwen`, `anthropic`. "
            "`claude-code` aliases `anthropic` (shared catalog)."
        ),
    ],
) -> None:
    """Fetch the live model list from <provider>'s API.

    `models key` runs this automatically after writing a key, so this is
    mostly for re-checking later when new releases land. `claude-code`
    routes to anthropic — the `claude` CLI talks to the Anthropic API,
    so its model catalog is anthropic's.
    """
    known_all = _known_provider_ids()
    if provider not in known_all:
        exit_error(f"unknown provider {provider!r} (known: {', '.join(known_all)})")

    source = _discovery_source(provider)
    try:
        models = _fetch_live_models(source)
    except Exception as e:
        exit_error(f"discover failed — {type(e).__name__}: {e}")
    _print_live_models_table(source, models, display=provider)


__all__ = ["models_app"]
