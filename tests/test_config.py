"""Tests for `physiclaw.config` — TOML loader, write, set/unset, model
ref resolution, provider credential resolution.

Module-level `CONFIG` is loaded once at import and refreshed by
`set_dotted` / `unset_dotted`. The autouse `_isolated_config` fixture
snapshots/restores it so mutating tests don't bleed.

`config_path()` returns `paths.HOME / "config.toml"`. `paths.HOME` is
pinned per-test by the global conftest fixture, but tests that exercise
default-path behavior pass `tmp_path / "config.toml"` explicitly to
avoid relying on the conftest interaction.

Accepted equivalent mutmut survivor: in `get()`'s "unknown key" branch,
`walked = ".".join(parts[:i])` mutated to `"XX.XX".join(parts[:i])` is
unreachable as a discriminator — when this line fires, parts[:i] has
at most one element (Config has only one nesting level: Config →
SectionConfig → leaf), so the join separator is never applied to two
strings. Joining an empty or 1-element list ignores the separator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from physiclaw import config


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshot the module-level CONFIG so tests that call set_dotted /
    unset_dotted (which mutate it) don't leak state."""
    monkeypatch.setattr(config, "CONFIG", config.Config())


@pytest.fixture(autouse=True)
def _no_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop PHYSICLAW_MODEL from the env so model-ref tests aren't
    contaminated by the host shell."""
    monkeypatch.delenv("PHYSICLAW_MODEL", raising=False)


# ---------- ConfigError ----------


def test_config_error_is_value_error_subclass() -> None:
    assert issubclass(config.ConfigError, ValueError)


# ---------- Config dataclass defaults ----------


@pytest.mark.parametrize(
    "section, field, expected",
    [
        # ServerConfig
        ("server", "port", 8048),
        # Loopback: the control plane must not be LAN-reachable — only the
        # bridge listener (port+1) binds 0.0.0.0 (see core/server/planes.py).
        ("server", "host", "127.0.0.1"),
        ("server", "save_tool_calls", False),
        ("server", "save_snapshots", False),
        ("server", "save_screenshots", False),
        ("server", "save_raw_camera", False),
        # WarmStartConfig
        ("warm_start", "bridge_wait_timeout_seconds", 120),
        ("warm_start", "bridge_settle_seconds", 2.0),
        ("warm_start", "port_wait_timeout_seconds", 10.0),
        ("warm_start", "port_wait_connect_timeout_seconds", 0.2),
        ("warm_start", "port_wait_interval_seconds", 0.1),
        # AutoPickConfig
        ("auto_pick", "bridge_wait_timeout_seconds", 25),
        ("auto_pick", "bridge_settle_seconds", 1.5),
        # EngineConfig
        ("engine", "max_turns", 300),
        ("engine", "max_attempts", 3),
        ("engine", "retry_backoff_seconds", 5.0),
        ("engine", "wait_default_minutes", 15),
        ("engine", "react_cooldown_seconds", 6.0),
        ("engine", "stale_tick_threshold", 8),
        ("engine", "state_decay_turns", 4),
        ("engine", "same_target_warn", 3),
        ("engine", "same_target_block", 5),
        ("engine", "step_stuck_warn", 12),
        ("engine", "step_stuck_urgent", 18),
        ("engine", "plan_required_after", 8),
        # AgentConfig
        ("agent", "model", ""),
        # ProviderConfig
        ("provider", "qwen_api_key", ""),
        ("provider", "moonshot_api_key", ""),
        ("provider", "openai_api_key", ""),
        ("provider", "anthropic_api_key", ""),
        ("provider", "google_api_key", ""),
        ("provider", "deepseek_api_key", ""),
        # CompactConfig
        ("compact", "max_image_edge_px", 1566),
        ("compact", "jpeg_quality", 85),
        # MemoryConfig
        ("memory", "default_log_entries", 20),
        ("memory", "bootstrap_log_entries", 10),
        # ClaudeConfig
        ("claude", "timeout_seconds", 180),
        ("claude", "stream_buffer_mb", 10),
        ("claude", "max_attempts", 3),
        ("claude", "retry_backoff_seconds", 5.0),
        # RetentionConfig
        ("retention", "trace_days", 7),
        # SkillsConfig
        ("skills", "default_source", ""),
        (
            "skills",
            "official_base_url",
            "https://physiclaw.ai/downloads/official-skills",
        ),
        ("skills", "sync_auto", True),
    ],
)
def test_default_config_field_value_pinned(
    section: str, field: str, expected: object
) -> None:
    cfg = config.Config()

    assert getattr(getattr(cfg, section), field) == expected


def test_default_config_factories_not_shared_between_instances() -> None:
    a = config.Config()
    b = config.Config()

    a.server.port = 9999

    assert b.server.port == 8048


# ---------- config_path ----------


def test_config_path_lives_under_paths_HOME(physiclaw_home: Path) -> None:
    assert config.config_path() == physiclaw_home / "config.toml"


# ---------- load ----------


def test_load_returns_default_config_when_file_missing(tmp_path: Path) -> None:
    p = tmp_path / "nonexistent.toml"

    cfg = config.load(p)

    assert cfg == config.Config()


def test_load_raises_friendly_ConfigError_when_file_isnt_utf8(
    tmp_path: Path,
) -> None:
    """Regression: on Chinese Windows (cp936), Path.write_text without
    encoding= silently encoded the template's em-dashes / ellipsis as GBK,
    then tomllib.load (which mandates UTF-8) raised an opaque
    UnicodeDecodeError that propagated all the way out and crashed the
    CLI at every subsequent invocation. The fix pins UTF-8 on every write
    AND converts a non-UTF-8 file into a ConfigError with a recovery hint.
    """
    p = tmp_path / "corrupted.toml"
    # Mimic the user's bug: "—" in a comment encoded as GBK (0xa1 0xad).
    p.write_bytes(b"# header with em-dash \xa1\xad and a key\nfoo = 1\n")

    with pytest.raises(config.ConfigError) as exc_info:
        config.load(p)

    msg = str(exc_info.value)
    assert "not valid UTF-8" in msg
    assert "0xa1" in msg  # the offending byte the user actually saw
    assert "delete the file" in msg  # recovery hint
    assert str(p) in msg  # path so the user knows where to delete


def test_load_parses_valid_toml(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text("[server]\nport = 9999\n\n[engine]\nmax_turns = 500\n")

    cfg = config.load(p)

    assert cfg.server.port == 9999
    assert cfg.engine.max_turns == 500
    # Untouched sections retain defaults.
    assert cfg.compact.jpeg_quality == 85


def test_load_raises_ConfigError_on_unknown_section(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text("[mystery]\nx = 1\n")

    # End-anchored to catch XX-wrap on the trailing "valid sections: …"
    # fragment.
    with pytest.raises(
        config.ConfigError,
        match=(
            r"^unknown section\(s\) in .+: \['mystery'\] — "
            r"valid sections: \[.+\]$"
        ),
    ):
        config.load(p)


def test_load_raises_ConfigError_on_unknown_key_in_known_section(
    tmp_path: Path,
) -> None:
    p = tmp_path / "config.toml"
    p.write_text("[server]\nbogus = 1\n")

    with pytest.raises(
        config.ConfigError, match=r"^unknown key\(s\) in \[server\]: \['bogus'\]"
    ):
        config.load(p)


def test_load_raises_ConfigError_when_section_is_not_a_table(
    tmp_path: Path,
) -> None:
    p = tmp_path / "config.toml"
    p.write_text('server = "string-not-a-table"\n')

    with pytest.raises(
        config.ConfigError, match=r"^\[server\] must be a table, got str$"
    ):
        config.load(p)


def test_load_raises_ConfigError_on_malformed_toml(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text("[server\nport = 9999\n")  # missing closing bracket

    with pytest.raises(config.ConfigError, match=r"^failed to read"):
        config.load(p)


def test_load_accepts_freeform_providers_section_without_validating(
    tmp_path: Path,
) -> None:
    # `[providers.<id>]` is in _FREEFORM_SECTIONS — read directly elsewhere.
    p = tmp_path / "config.toml"
    p.write_text('[providers.openai]\nbase_url = "https://proxy.example/v1"\n')

    cfg = config.load(p)

    assert cfg == config.Config()


# ---------- to_toml ----------


def test_to_toml_round_trips_through_load(tmp_path: Path) -> None:
    cfg = config.Config()
    cfg.server.port = 1234
    cfg.engine.max_turns = 42
    cfg.agent.model = "qwen/qwen3-plus"

    p = tmp_path / "config.toml"
    p.write_text(config.to_toml(cfg))
    reloaded = config.load(p)

    assert reloaded == cfg


def test_to_toml_emits_header_and_section_comments_when_with_comments() -> None:
    out = config.to_toml(config.Config(), with_comments=True)

    assert out.startswith("# PhysiClaw config.")
    assert "# Timeouts for `physiclaw server --warm-start`" in out
    assert "# Engine + model selection." in out


@pytest.mark.parametrize(
    "section_key, expected_comment_fragment",
    [
        (
            "warm_start",
            "Timeouts for `physiclaw server --warm-start` hardware reconnect.",
        ),
        (
            "auto_pick",
            "Timeouts for the camera auto-pick step in `physiclaw setup hardware`.",
        ),
        (
            "engine",
            "Agent tool-call loop: runaway safeguards (turn cap, stuck guard, "
            "plan gate) + retry + pacing.",
        ),
        ("compact", "Screenshot compression before sending to the LLM."),
        (
            "memory",
            "Daily-log loading: bootstrap preload + on-demand `read_logs` defaults.",
        ),
        (
            "claude",
            "Applied when [agent] model = 'claude-code/...' (external CLI subprocess).",
        ),
        ("retention", "Purge window for on-disk engine trace logs + cron job history."),
    ],
)
def test_section_comment_pinned(
    section_key: str, expected_comment_fragment: str
) -> None:
    assert config._SECTION_COMMENTS[section_key] == expected_comment_fragment


def test_agent_section_comment_includes_provider_model_examples() -> None:
    # The agent comment is built via parens + concatenation; assert exact.
    assert config._SECTION_COMMENTS["agent"] == (
        "Engine + model selection. `model` is a `provider/model` ref, e.g. "
        "`qwen/qwen3.6-plus` or `claude-code/claude-sonnet-4-6`. "
        "`PHYSICLAW_MODEL` env var overrides."
    )


def test_provider_section_comment_pinned() -> None:
    assert config._SECTION_COMMENTS["provider"] == (
        "Per-provider API keys. Field names match the provider id "
        "(qwen/moonshot/openai/anthropic). Env vars (QWEN_API_KEY, "
        "MOONSHOT_API_KEY, OPENAI_API_KEY, …) override these. Treat "
        "keys here like ssh keys."
    )


def test_skills_section_comment_pinned() -> None:
    assert config._SECTION_COMMENTS["skills"] == (
        "`physiclaw skills`. `default_source`: repo for `install` (empty = "
        "require `--from`; `owner/repo` or a git URL). `official_base_url` + "
        "`sync_auto`: the `sync official` pack source + auto-sync at startup."
    )


@pytest.mark.parametrize(
    "section, field, expected_inline",
    [
        ("server", "save_tool_calls", "dump every peek/screenshot output"),
        (
            "server",
            "save_snapshots",
            "dump each snapshot frame (rotated, with bbox overlay)",
        ),
        ("server", "save_screenshots", "dump every raw phone-own screenshot"),
        ("server", "save_raw_camera", "dump every raw camera frame at capture"),
        (
            "memory",
            "default_log_entries",
            "on-demand `read_logs` default size (max 200)",
        ),
        (
            "memory",
            "bootstrap_log_entries",
            "auto-preloaded into the memory slot at every wake",
        ),
    ],
)
def test_field_inline_comment_pinned(
    section: str, field: str, expected_inline: str
) -> None:
    assert config._FIELD_COMMENTS[(section, field)] == expected_inline


def test_to_toml_omits_comments_by_default() -> None:
    out = config.to_toml(config.Config())

    assert "PhysiClaw config." not in out
    assert "[server]" in out


def test_to_toml_emits_inline_comments_for_known_fields() -> None:
    out = config.to_toml(config.Config(), with_comments=True)

    assert "# dump every peek/screenshot output" in out


def test_to_toml_serializes_each_scalar_type_correctly() -> None:
    cfg = config.Config()
    cfg.server.port = 42
    cfg.server.save_tool_calls = True
    cfg.warm_start.bridge_settle_seconds = 1.5
    cfg.agent.model = "x/y"

    out = config.to_toml(cfg)

    assert "port = 42" in out
    assert "save_tool_calls = true" in out
    assert "bridge_settle_seconds = 1.5" in out
    assert 'model = "x/y"' in out


def test_to_toml_raises_on_non_finite_float() -> None:
    cfg = config.Config()
    cfg.warm_start.bridge_settle_seconds = float("nan")

    with pytest.raises(
        config.ConfigError, match=r"^non-finite float not representable in TOML"
    ):
        config.to_toml(cfg)


def test_to_toml_raises_on_unsupported_value_type() -> None:
    cfg = config.Config()
    cfg.server.port = [1, 2, 3]  # type: ignore[assignment]

    with pytest.raises(config.ConfigError, match=r"^cannot serialize"):
        config.to_toml(cfg)


# ---------- write_default ----------


def test_write_default_creates_file_with_commented_template(tmp_path: Path) -> None:
    p = tmp_path / "nested" / "config.toml"

    config.write_default(p)

    text = p.read_text()
    assert text.startswith("# PhysiClaw config.")
    assert "[server]" in text


def test_write_default_creates_intermediate_parent_directories(
    tmp_path: Path,
) -> None:
    # `mkdir(parents=True)` — required because the deep parent chain
    # doesn't exist yet.
    p = tmp_path / "a" / "b" / "c" / "config.toml"

    config.write_default(p)

    assert p.is_file()


def test_write_default_is_noop_when_file_already_exists(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text("# user-written content\n[server]\nport = 1\n")

    config.write_default(p)

    assert p.read_text() == "# user-written content\n[server]\nport = 1\n"


def test_write_default_returns_path_argument(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"

    assert config.write_default(p) == p


# ---------- get ----------


def test_get_returns_section_dataclass_for_section_name() -> None:
    cfg = config.Config()

    section = config.get(cfg, "server")

    assert section is cfg.server


def test_get_returns_field_value_for_dotted_path() -> None:
    cfg = config.Config()

    assert config.get(cfg, "engine.max_turns") == 300


def test_get_raises_on_empty_path() -> None:
    with pytest.raises(config.ConfigError, match=r"^empty path: ''$"):
        config.get(config.Config(), "")


def test_get_raises_when_path_contains_empty_segment() -> None:
    with pytest.raises(
        config.ConfigError, match=r"^empty path: 'engine\.\.max_turns'$"
    ):
        config.get(config.Config(), "engine..max_turns")


def test_get_walked_path_uses_dot_separator_in_error_message() -> None:
    # When the failing segment is at depth 2, the error message reports
    # the walked path as "section.field". Mutating the join sep would
    # render it as "sectionXX.XXfield".
    with pytest.raises(
        config.ConfigError,
        match=r"^'engine\.max_turns' is a leaf value, not a section$",
    ):
        config.get(config.Config(), "engine.max_turns.deeper")


def test_get_raises_when_descending_into_a_leaf_value() -> None:
    with pytest.raises(
        config.ConfigError, match=r"^'engine.max_turns' is a leaf value"
    ):
        config.get(config.Config(), "engine.max_turns.deeper")


def test_get_raises_when_segment_unknown_at_level() -> None:
    with pytest.raises(config.ConfigError, match=r"^unknown key 'mystery' at <root>"):
        config.get(config.Config(), "mystery")


# ---------- _coerce (private; exercised via set_dotted) ----------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("TRUE", True),
        ("Yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ],
)
def test_coerce_bool_accepts_known_variants(raw: str, expected: bool) -> None:
    assert config._coerce(raw, True) is expected


def test_coerce_bool_rejects_unknown_string() -> None:
    with pytest.raises(config.ConfigError, match=r"^can't parse 'maybe' as bool"):
        config._coerce("maybe", True)


def test_coerce_int_parses_decimal_string() -> None:
    assert config._coerce("42", 0) == 42


def test_coerce_int_rejects_non_numeric() -> None:
    with pytest.raises(config.ConfigError, match=r"^can't parse 'abc' as int"):
        config._coerce("abc", 0)


def test_coerce_float_parses_decimal_string() -> None:
    assert config._coerce("3.14", 0.0) == 3.14


def test_coerce_float_rejects_non_numeric() -> None:
    with pytest.raises(config.ConfigError, match=r"^can't parse 'pi' as float"):
        config._coerce("pi", 0.0)


def test_coerce_str_passes_through_unchanged() -> None:
    assert config._coerce("hello world", "default") == "hello world"


# ---------- _validate_dotted (private; exercised via set/unset_dotted) ----------


def test_validate_dotted_returns_section_and_field() -> None:
    assert config._validate_dotted("engine.max_turns") == ("engine", "max_turns")


@pytest.mark.parametrize(
    "bad",
    ["", "single", "a.b.c", ".trailing", "leading.", "."],
)
def test_validate_dotted_rejects_wrong_shape(bad: str) -> None:
    with pytest.raises(config.ConfigError, match=r"^key must be section\.field"):
        config._validate_dotted(bad)


def test_validate_dotted_rejects_unknown_section() -> None:
    with pytest.raises(config.ConfigError, match=r"^unknown section 'mystery'"):
        config._validate_dotted("mystery.field")


def test_validate_dotted_rejects_unknown_field_within_known_section() -> None:
    with pytest.raises(config.ConfigError, match=r"^unknown key 'mystery' at server"):
        config._validate_dotted("server.mystery")


# ---------- set_dotted ----------


def test_set_dotted_writes_value_to_existing_section(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text("[engine]\nmax_turns = 300\n")

    config.set_dotted("engine.max_turns", "500", path=p)

    text = p.read_text()
    assert "max_turns = 500" in text


def test_set_dotted_creates_section_when_absent(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"

    config.set_dotted("engine.max_turns", "200", path=p)

    assert p.is_file()
    assert "max_turns = 200" in p.read_text()


def test_set_dotted_coerces_to_field_type(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"

    config.set_dotted("server.save_tool_calls", "yes", path=p)

    # After write, reload sees True (not the string "yes").
    cfg = config.load(p)
    assert cfg.server.save_tool_calls is True


def test_set_dotted_refreshes_module_level_CONFIG(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"

    config.set_dotted("engine.max_turns", "777", path=p)

    assert config.CONFIG.engine.max_turns == 777


def test_set_dotted_adds_section_to_existing_file_that_lacks_it(
    tmp_path: Path,
) -> None:
    # File exists with one section; setting a key in a different section
    # must add the new [section] block via tomlkit, not error out.
    p = tmp_path / "config.toml"
    p.write_text("[server]\nport = 8048\n")

    config.set_dotted("engine.max_turns", "200", path=p)

    text = p.read_text()
    assert "[engine]" in text
    assert "max_turns = 200" in text


def test_set_dotted_raises_on_unknown_dotted(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"

    with pytest.raises(config.ConfigError, match=r"^unknown section"):
        config.set_dotted("mystery.field", "x", path=p)


# ---------- unset_dotted ----------


def test_unset_dotted_removes_existing_key(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text("[engine]\nmax_turns = 999\n")

    removed = config.unset_dotted("engine.max_turns", path=p)

    assert removed is True
    assert "max_turns" not in p.read_text()


def test_unset_dotted_returns_false_when_file_missing(tmp_path: Path) -> None:
    p = tmp_path / "missing.toml"

    assert config.unset_dotted("engine.max_turns", path=p) is False


def test_unset_dotted_returns_false_when_key_absent(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text("[engine]\nmax_turns = 999\n")

    assert config.unset_dotted("engine.max_attempts", path=p) is False


def test_unset_dotted_returns_false_when_section_absent(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text("[server]\nport = 8048\n")

    assert config.unset_dotted("engine.max_turns", path=p) is False


def test_unset_dotted_refreshes_CONFIG_to_default(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text("[engine]\nmax_turns = 999\n")

    config.unset_dotted("engine.max_turns", path=p)

    assert config.CONFIG.engine.max_turns == 300  # back to default


# ---------- provider_base_url_override ----------


def test_provider_base_url_override_returns_none_when_file_missing(
    physiclaw_home: Path,
) -> None:
    assert config.provider_base_url_override("openai") is None


def test_provider_base_url_override_returns_none_on_malformed_toml(
    physiclaw_home: Path,
) -> None:
    p = config.config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("[providers.openai\nbase_url = ...\n")  # malformed

    assert config.provider_base_url_override("openai") is None


def test_provider_base_url_override_returns_none_when_provider_absent(
    physiclaw_home: Path,
) -> None:
    p = config.config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('[providers.qwen]\nbase_url = "https://x.com"\n')

    assert config.provider_base_url_override("openai") is None


def test_provider_base_url_override_returns_string_when_present(
    physiclaw_home: Path,
) -> None:
    p = config.config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('[providers.openai]\nbase_url = "https://proxy.example/v1"\n')

    assert config.provider_base_url_override("openai") == "https://proxy.example/v1"


def test_provider_base_url_override_returns_none_for_non_string_value(
    physiclaw_home: Path,
) -> None:
    p = config.config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("[providers.openai]\nbase_url = 42\n")

    assert config.provider_base_url_override("openai") is None


# ---------- model_ref + model_ref_with_source ----------


def test_model_ref_raises_when_neither_env_nor_config_set() -> None:
    # Exact-equality on the full message so any prose drift fails the test.
    expected = (
        "no model configured.\n"
        "  Quick start:\n"
        "    physiclaw models key <provider>     # e.g. anthropic, openai, qwen\n"
        "    physiclaw models use <provider/model>\n"
        "  Or set PHYSICLAW_MODEL=<provider>/<model> in your shell."
    )
    with pytest.raises(RuntimeError) as exc_info:
        config.model_ref()
    assert str(exc_info.value) == expected


def test_model_ref_uses_env_var_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.MODEL_ENV_VAR, "qwen/qwen3-plus")

    assert config.model_ref() == "qwen/qwen3-plus"


def test_model_ref_falls_back_to_config_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config.CONFIG.agent.model = "openai/gpt-5"

    assert config.model_ref() == "openai/gpt-5"


def test_model_ref_with_source_reports_env_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(config.MODEL_ENV_VAR, "x/y")

    ref, source = config.model_ref_with_source()

    assert ref == "x/y"
    assert source == "PHYSICLAW_MODEL env"


def test_model_ref_with_source_reports_config_origin() -> None:
    config.CONFIG.agent.model = "anthropic/claude-opus-4-7"

    ref, source = config.model_ref_with_source()

    assert ref == "anthropic/claude-opus-4-7"
    assert source == "config.toml [agent] model"


def test_model_env_var_constant_pinned() -> None:
    assert config.MODEL_ENV_VAR == "PHYSICLAW_MODEL"


# ---------- parse_model_ref ----------


def test_parse_model_ref_splits_on_first_slash() -> None:
    assert config.parse_model_ref("qwen/qwen3-plus") == ("qwen", "qwen3-plus")


def test_parse_model_ref_keeps_extra_slashes_in_model_segment() -> None:
    assert config.parse_model_ref("openrouter/openai/gpt-5") == (
        "openrouter",
        "openai/gpt-5",
    )


def test_parse_model_ref_raises_on_no_slash() -> None:
    expected = (
        "model ref 'qwen-only' must be 'provider/model' (e.g. 'qwen/qwen3.6-plus')"
    )
    with pytest.raises(ValueError) as exc_info:
        config.parse_model_ref("qwen-only")
    assert str(exc_info.value) == expected


@pytest.mark.parametrize("bad", ["/model", "provider/", "/"])
def test_parse_model_ref_raises_on_empty_segment(bad: str) -> None:
    with pytest.raises(ValueError, match=r"empty provider or model segment"):
        config.parse_model_ref(bad)


# ---------- resolve_provider_key ----------


def test_resolve_provider_key_uses_first_matching_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALT_KEY", "from-alt")
    monkeypatch.setenv("MAIN_KEY", "from-main")

    key, source = config.resolve_provider_key(
        ("MAIN_KEY", "ALT_KEY"), config_key="qwen_api_key"
    )

    assert key == "from-main"
    assert source == "MAIN_KEY env"


def test_resolve_provider_key_skips_empty_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAIN_KEY", "")  # empty — falsy
    monkeypatch.setenv("ALT_KEY", "from-alt")

    key, source = config.resolve_provider_key(
        ("MAIN_KEY", "ALT_KEY"), config_key="qwen_api_key"
    )

    assert key == "from-alt"
    assert source == "ALT_KEY env"


def test_resolve_provider_key_falls_back_to_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config.CONFIG.provider.qwen_api_key = "from-config"

    key, source = config.resolve_provider_key(("UNSET_VAR",), config_key="qwen_api_key")

    assert key == "from-config"
    assert source == "config.toml [provider] qwen_api_key"


def test_resolve_provider_key_returns_none_when_nothing_set() -> None:
    key, source = config.resolve_provider_key(("UNSET_VAR",), config_key="qwen_api_key")

    assert key is None
    assert source is None


def test_resolve_provider_key_returns_none_for_unknown_config_key() -> None:
    # getattr(provider, unknown, "") returns "" (the default), so source
    # is None.
    key, source = config.resolve_provider_key(
        ("UNSET_VAR",), config_key="not_a_real_key"
    )

    assert key is None
    assert source is None


def test_camera_exposure_keys_default_and_render() -> None:
    cfg = config.Config()

    assert cfg.camera.auto_exposure is True
    assert cfg.camera.exposure == -6

    out = config.to_toml(cfg, with_comments=True)
    assert "auto_exposure = true" in out
    assert "exposure = -6" in out
