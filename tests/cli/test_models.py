"""Tests for `physiclaw.cli.models` — model selection / discovery CLI."""

from __future__ import annotations

import importlib

from typer.testing import CliRunner

from physiclaw.common import config as _config

models_mod = importlib.import_module("physiclaw.cli.models")
models_app = models_mod.models_app

runner = CliRunner()


# ---------- _discovery_source / _key_config_path ----------


def test_discovery_source_passes_through_unknown() -> None:
    assert models_mod._discovery_source("openai") == "openai"


def test_discovery_source_resolves_alias() -> None:
    assert models_mod._discovery_source("claude-code") == "anthropic"


def test_key_config_path_format() -> None:
    assert models_mod._key_config_path("openai") == "provider.openai_api_key"


def test_known_provider_ids_includes_aliases(mocker) -> None:
    mocker.patch(
        "physiclaw.agent.provider.in_process_provider_ids",
        return_value=("openai", "anthropic"),
    )

    out = models_mod._known_provider_ids()

    assert "openai" in out
    assert "anthropic" in out
    # Aliases tacked on at the end.
    assert "claude-code" in out


# ---------- bare `physiclaw models` ----------


def test_bare_models_shows_active_when_set(mocker) -> None:
    mocker.patch.object(
        models_mod._config,
        "model_ref_with_source",
        return_value=("openai/gpt-5", "config"),
    )
    mocker.patch.object(
        models_mod._config,
        "parse_model_ref",
        return_value=("openai", "gpt-5"),
    )
    mocker.patch.object(
        models_mod,
        "_format_key_row",
        return_value="  openai api key: <set>",
    )

    result = runner.invoke(models_app, [])

    assert result.exit_code == 0
    assert "openai/gpt-5" in result.stdout
    assert "(from config)" in result.stdout


def test_bare_models_warns_when_unset(mocker) -> None:
    mocker.patch.object(
        models_mod._config,
        "model_ref_with_source",
        side_effect=RuntimeError("unset"),
    )

    result = runner.invoke(models_app, [])

    assert result.exit_code == 0
    assert "physiclaw models use" in result.stdout


def test_bare_models_warns_on_invalid_ref(mocker) -> None:
    mocker.patch.object(
        models_mod._config,
        "model_ref_with_source",
        return_value=("garbage", "config"),
    )
    mocker.patch.object(
        models_mod._config,
        "parse_model_ref",
        side_effect=ValueError("bad ref"),
    )

    result = runner.invoke(models_app, [])

    assert result.exit_code == 0
    assert "invalid ref" in result.stdout


# ---------- list ----------


def test_list_unknown_provider_exits_1(mocker) -> None:
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("openai", "anthropic"),
    )

    result = runner.invoke(models_app, ["list", "nope"])

    assert result.exit_code == 1
    assert "unknown provider" in result.stdout


def test_list_specific_provider_shows_models(mocker) -> None:
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("openai",),
    )
    fake_discovered = mocker.patch(
        "physiclaw.agent.provider.discovered",
    )
    fake_discovered.model_ids.return_value = ["gpt-5", "gpt-5-mini"]

    result = runner.invoke(models_app, ["list", "openai"])

    assert result.exit_code == 0
    assert "openai/gpt-5" in result.stdout


def test_list_empty_cache_prints_hint(mocker) -> None:
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("openai",),
    )
    fake_discovered = mocker.patch(
        "physiclaw.agent.provider.discovered",
    )
    fake_discovered.model_ids.return_value = []

    result = runner.invoke(models_app, ["list", "openai"])

    assert "no discovery cache" in result.stdout


def test_list_all_iterates_known_providers(mocker) -> None:
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("openai", "anthropic"),
    )
    fake_discovered = mocker.patch(
        "physiclaw.agent.provider.discovered",
    )
    fake_discovered.model_ids.return_value = ["m1"]

    result = runner.invoke(models_app, ["list"])

    assert "openai" in result.stdout
    assert "anthropic" in result.stdout


# ---------- use ----------


def test_use_rejects_ref_without_slash() -> None:
    result = runner.invoke(models_app, ["use", "gpt-5"])

    assert result.exit_code == 1
    assert "is not a `provider/model`" in result.output


def test_use_exits_1_on_parse_error(mocker) -> None:
    mocker.patch.object(
        models_mod._config,
        "parse_model_ref",
        side_effect=ValueError("bad form"),
    )

    result = runner.invoke(models_app, ["use", "x/y"])

    assert result.exit_code == 1


def test_use_unknown_provider_exits_1(mocker) -> None:
    mocker.patch.object(
        models_mod._config,
        "parse_model_ref",
        return_value=("nopeprovider", "model"),
    )
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("openai",),
    )

    result = runner.invoke(models_app, ["use", "nopeprovider/m"])

    assert result.exit_code == 1
    assert "unknown provider" in result.output


def test_use_model_not_in_cache_exits_1(mocker) -> None:
    mocker.patch.object(
        models_mod._config,
        "parse_model_ref",
        return_value=("openai", "ghost"),
    )
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("openai",),
    )
    fake_disc = mocker.patch("physiclaw.agent.provider.discovered")
    fake_disc.is_cached.return_value = False

    result = runner.invoke(models_app, ["use", "openai/ghost"])

    assert result.exit_code == 1
    assert "not in openai discovery cache" in result.output


def test_use_claude_code_with_empty_cache_writes_config(mocker) -> None:
    """Subscription-only setup: no anthropic key, so the aliased cache is
    empty — claude-code refs skip cache validation (the `claude` CLI
    validates model ids itself)."""
    mocker.patch.object(
        models_mod._config,
        "parse_model_ref",
        return_value=("claude-code", "claude-sonnet-4-6"),
    )
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("claude-code",),
    )
    fake_disc = mocker.patch("physiclaw.agent.provider.discovered")
    fake_disc.is_cached.return_value = False
    fake_disc.model_ids.return_value = set()
    spy = mocker.patch.object(models_mod._config, "set_dotted")

    result = runner.invoke(models_app, ["use", "claude-code/claude-sonnet-4-6"])

    assert result.exit_code == 0
    spy.assert_called_once_with("agent.model", "claude-code/claude-sonnet-4-6")


def test_use_claude_code_with_populated_cache_still_validates(mocker) -> None:
    mocker.patch.object(
        models_mod._config,
        "parse_model_ref",
        return_value=("claude-code", "ghost"),
    )
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("claude-code",),
    )
    fake_disc = mocker.patch("physiclaw.agent.provider.discovered")
    fake_disc.is_cached.return_value = False
    fake_disc.model_ids.return_value = {"claude-sonnet-4-6"}

    result = runner.invoke(models_app, ["use", "claude-code/ghost"])

    assert result.exit_code == 1
    assert "not in claude-code discovery cache" in result.output


def test_use_happy_path_writes_config(mocker) -> None:
    mocker.patch.object(
        models_mod._config,
        "parse_model_ref",
        return_value=("openai", "gpt-5"),
    )
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("openai",),
    )
    fake_disc = mocker.patch("physiclaw.agent.provider.discovered")
    fake_disc.is_cached.return_value = True
    spy = mocker.patch.object(models_mod._config, "set_dotted")

    result = runner.invoke(models_app, ["use", "openai/gpt-5"])

    assert result.exit_code == 0
    spy.assert_called_once_with("agent.model", "openai/gpt-5")
    assert "Restart" in result.stdout


def test_use_alias_set_works_too(mocker) -> None:
    """`models set` is a hidden alias for `use`."""
    mocker.patch.object(
        models_mod._config,
        "parse_model_ref",
        return_value=("openai", "gpt-5"),
    )
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("openai",),
    )
    fake_disc = mocker.patch("physiclaw.agent.provider.discovered")
    fake_disc.is_cached.return_value = True
    mocker.patch.object(models_mod._config, "set_dotted")

    result = runner.invoke(models_app, ["set", "openai/gpt-5"])

    assert result.exit_code == 0


def test_use_config_error_exits_1(mocker) -> None:
    mocker.patch.object(
        models_mod._config,
        "parse_model_ref",
        return_value=("openai", "gpt-5"),
    )
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("openai",),
    )
    fake_disc = mocker.patch("physiclaw.agent.provider.discovered")
    fake_disc.is_cached.return_value = True
    mocker.patch.object(
        models_mod._config,
        "set_dotted",
        side_effect=_config.ConfigError("write failed"),
    )

    result = runner.invoke(models_app, ["use", "openai/gpt-5"])

    assert result.exit_code == 1


# ---------- key ----------


def test_key_alias_redirects_to_target(mocker) -> None:
    result = runner.invoke(models_app, ["key", "claude-code", "secret"])

    assert result.exit_code == 1
    assert "reuses anthropic" in result.output


def test_key_unknown_provider_exits_1(mocker) -> None:
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("openai",),
    )

    result = runner.invoke(models_app, ["key", "nope", "k"])

    assert result.exit_code == 1
    assert "unknown provider" in result.output


def test_key_with_value_writes_and_fetches(mocker) -> None:
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("openai",),
    )
    set_spy = mocker.patch.object(models_mod._config, "set_dotted")
    mocker.patch.object(
        models_mod,
        "_fetch_live_models",
        return_value=[{"id": "gpt-5"}, {"id": "gpt-4o"}],
    )
    print_spy = mocker.patch.object(models_mod, "_print_live_models_table")

    result = runner.invoke(models_app, ["key", "openai", "sk-xxx"])

    assert result.exit_code == 0
    set_spy.assert_called_once_with("provider.openai_api_key", "sk-xxx")
    print_spy.assert_called_once()


def test_key_warns_when_fetch_fails(mocker) -> None:
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("openai",),
    )
    mocker.patch.object(models_mod._config, "set_dotted")
    mocker.patch.object(
        models_mod,
        "_fetch_live_models",
        side_effect=RuntimeError("network down"),
    )

    result = runner.invoke(models_app, ["key", "openai", "sk-xxx"])

    assert result.exit_code == 0
    assert "couldn't fetch live models" in result.stdout
    assert "physiclaw models discover openai" in result.stdout


def test_key_set_dotted_error_exits_1(mocker) -> None:
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("openai",),
    )
    mocker.patch.object(
        models_mod._config,
        "set_dotted",
        side_effect=_config.ConfigError("readonly"),
    )

    result = runner.invoke(models_app, ["key", "openai", "k"])

    assert result.exit_code == 1


def test_key_prompts_when_value_omitted(mocker) -> None:
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("openai",),
    )
    mocker.patch.object(models_mod._config, "set_dotted")
    mocker.patch.object(
        models_mod.typer,
        "prompt",
        return_value="prompted-key",
    )
    mocker.patch.object(
        models_mod,
        "_fetch_live_models",
        side_effect=RuntimeError("skip"),
    )

    result = runner.invoke(models_app, ["key", "openai"])

    assert result.exit_code == 0


# ---------- keys ----------


def test_keys_lists_all_providers(mocker) -> None:
    mocker.patch(
        "physiclaw.agent.provider.in_process_provider_ids",
        return_value=("openai", "anthropic"),
    )
    mocker.patch.object(
        models_mod,
        "_format_key_row",
        side_effect=lambda pid, indent=2: f"{' ' * indent}{pid}: <row>",
    )

    result = runner.invoke(models_app, ["keys"])

    assert result.exit_code == 0
    assert "openai: <row>" in result.stdout
    assert "anthropic: <row>" in result.stdout


# ---------- discover ----------


def test_discover_unknown_provider_exits_1(mocker) -> None:
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("openai",),
    )

    result = runner.invoke(models_app, ["discover", "nope"])

    assert result.exit_code == 1


def test_discover_fetch_failure_exits_1(mocker) -> None:
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("openai",),
    )
    mocker.patch.object(
        models_mod,
        "_fetch_live_models",
        side_effect=RuntimeError("network"),
    )

    result = runner.invoke(models_app, ["discover", "openai"])

    assert result.exit_code == 1
    assert "discover failed" in result.output


def test_discover_happy_path(mocker) -> None:
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("openai", "claude-code"),
    )
    mocker.patch.object(
        models_mod,
        "_fetch_live_models",
        return_value=[{"id": "gpt-5"}],
    )
    print_spy = mocker.patch.object(models_mod, "_print_live_models_table")

    result = runner.invoke(models_app, ["discover", "openai"])

    assert result.exit_code == 0
    print_spy.assert_called_once()
    # `display` defaults to provider arg.
    assert print_spy.call_args.kwargs["display"] == "openai"


def test_discover_alias_uses_target_cache(mocker) -> None:
    mocker.patch.object(
        models_mod,
        "_known_provider_ids",
        return_value=("anthropic", "claude-code"),
    )
    fetch_spy = mocker.patch.object(
        models_mod,
        "_fetch_live_models",
        return_value=[],
    )
    print_spy = mocker.patch.object(models_mod, "_print_live_models_table")

    runner.invoke(models_app, ["discover", "claude-code"])

    # Fetch happens against `anthropic` (the resolved source).
    fetch_spy.assert_called_once_with("anthropic")
    # But display label echoes what user typed.
    assert print_spy.call_args.kwargs["display"] == "claude-code"


# ---------- _format_key_row ----------


def test_format_key_row_unset(mocker) -> None:
    mocker.patch(
        "physiclaw.agent.provider.provider_key_status",
        return_value=(None, "config"),
    )

    out = models_mod._format_key_row("openai")

    assert "(unset)" in out


def test_format_key_row_set_with_source(mocker) -> None:
    mocker.patch(
        "physiclaw.agent.provider.provider_key_status",
        return_value=("sk-***x", "env"),
    )

    out = models_mod._format_key_row("openai")

    assert "sk-***x" in out
    assert "[env]" in out


# ---------- _fetch_live_models ----------


def test_fetch_live_models_calls_provider_class(mocker) -> None:
    fake_provider = mocker.MagicMock()

    async def _list_models():
        return [{"id": "m1"}]

    async def _aclose():
        pass

    fake_provider.list_models = _list_models
    fake_provider.aclose = _aclose

    fake_cls = mocker.MagicMock(return_value=fake_provider)
    mocker.patch(
        "physiclaw.agent.provider.provider_class",
        return_value=fake_cls,
    )

    out = models_mod._fetch_live_models("openai")

    assert out == [{"id": "m1"}]


# ---------- _print_live_models_table ----------


def test_print_live_models_table_saves_and_prints(mocker, capsys) -> None:
    fake_disc = mocker.patch("physiclaw.agent.provider.discovered")

    models_mod._print_live_models_table("openai", [{"id": "gpt-5"}])
    out = capsys.readouterr().out

    fake_disc.save.assert_called_once_with("openai", [{"id": "gpt-5"}])
    assert "openai" in out
    assert "gpt-5" in out


def test_print_live_models_table_uses_display_label(mocker, capsys) -> None:
    mocker.patch("physiclaw.agent.provider.discovered")

    models_mod._print_live_models_table(
        "anthropic",
        [{"id": "x"}],
        display="claude-code",
    )
    out = capsys.readouterr().out

    # Header uses display.
    assert "claude-code" in out
    # `models use claude-code/<id>` hint, not anthropic/.
    assert "physiclaw models use claude-code/" in out
