"""Tests for the engine's turn-plugin loader — the blind end of the seam.

The loader knows a config string and the `TurnPlugin` protocol, nothing
else; the built-in default set is its data, not the config layer's.
Every failure mode is fail-open: the session runs with whatever loaded.
The other half of the rollback story — that an EngineRun with
`plugins=()` is a plain provider session — is pinned across the whole
loop suite, whose `_mk_run` builds exactly that.
"""

from __future__ import annotations

from physiclaw.agent.engine import plugins as plugins_mod
from physiclaw.common.config import CONFIG
from physiclaw.contract.plugin import TurnPlugin


def test_none_is_the_off_switch(monkeypatch, caplog) -> None:
    # The rollback switch: `[agent] plugins = "none"` → every turn is a
    # plain provider call, exactly the pre-plugin engine — and the log
    # says so, so a plugin-less session reads as a choice in forensics.
    monkeypatch.setattr(CONFIG.agent, "plugins", "none")
    with caplog.at_level("INFO"):
        assert plugins_mod.load_plugins() == ()
    assert "disabled by config" in caplog.text
    monkeypatch.setattr(CONFIG.agent, "plugins", " None ")
    assert plugins_mod.load_plugins() == ()


def test_empty_config_resolves_to_the_builtin_default(caplog) -> None:
    # The config layer defaults to "" and names no plugin; the loader
    # owns the default set. The shipped default is the conductor — this
    # test rides the real dotted path on purpose: if it rots, the
    # engine silently loses its conductor. The resolved set is logged
    # per session ("was the conductor even in play?").
    assert CONFIG.agent.plugins == ""
    with caplog.at_level("INFO"):
        plugins = plugins_mod.load_plugins()
    assert len(plugins) == 1
    assert isinstance(plugins[0], TurnPlugin)
    assert plugins_mod.DEFAULT_PLUGINS in caplog.text


def test_unimportable_path_is_skipped(monkeypatch, caplog) -> None:
    monkeypatch.setattr(CONFIG.agent, "plugins", "no.such.module:build")
    assert plugins_mod.load_plugins() == ()
    assert "failed to load" in caplog.text


def test_factory_crash_is_skipped(monkeypatch, caplog) -> None:
    # `builtins:iter` with no argument raises inside the factory call.
    monkeypatch.setattr(CONFIG.agent, "plugins", "builtins:iter")
    assert plugins_mod.load_plugins() == ()
    assert "failed to load" in caplog.text


def test_wrong_shaped_object_is_skipped(monkeypatch, caplog) -> None:
    # `builtins:object` builds fine but satisfies nothing.
    monkeypatch.setattr(CONFIG.agent, "plugins", "builtins:object")
    assert plugins_mod.load_plugins() == ()
    assert "does not satisfy TurnPlugin" in caplog.text


def test_bad_paths_do_not_sink_good_ones(monkeypatch) -> None:
    monkeypatch.setattr(
        CONFIG.agent,
        "plugins",
        f"no.such.module:build, {plugins_mod.DEFAULT_PLUGINS}",
    )
    assert len(plugins_mod.load_plugins()) == 1
