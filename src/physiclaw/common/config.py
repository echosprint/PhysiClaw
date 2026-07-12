"""User-tunable configuration loaded from ``~/.physiclaw/config.toml``.

Defaults live on the ``@dataclass`` sections — ``to_toml()`` + ``write_default()``
render the commented template from those same defaults, so bumping a default
doesn't need two edits.

Unknown top-level sections and unknown keys inside a section raise
``ConfigError`` on load. The CLI catches this and points users at
``physiclaw config edit``.

Layering (first match wins):
    CLI flag  >  env var  >  config.toml  >  built-in default
"""

import dataclasses
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from physiclaw.common import paths
from physiclaw.common.text import read_text, write_text


class ConfigError(ValueError):
    """Malformed ``config.toml`` — unknown section/key, or type mismatch."""


@dataclass
class ServerConfig:
    """``port``/``host`` bind the CONTROL plane (MCP + setup + calibrate) —
    loopback by default so the arm-driving surface is unreachable from the
    LAN. The phone bridge always binds ``0.0.0.0`` on ``port``+1."""

    port: int = 8048
    host: str = "127.0.0.1"
    save_tool_calls: bool = False
    save_snapshots: bool = False
    save_screenshots: bool = False
    save_raw_camera: bool = False


@dataclass
class UpdateConfig:
    """Update check. ``check = true`` makes ``physiclaw server`` check PyPI in
    the background and, at the next start, print a notice when a newer release
    is ready — it never installs on its own (reinstalling the venv under a live
    server can corrupt it on Windows). Apply with ``uv tool upgrade physiclaw``
    once the server is stopped. ``PHYSICLAW_DISABLE_UPDATE_CHECK=1`` env
    overrides to off."""

    check: bool = True


@dataclass
class WarmStartConfig:
    bridge_wait_timeout_seconds: int = 120
    bridge_settle_seconds: float = 2.0
    port_wait_timeout_seconds: float = 10.0
    port_wait_connect_timeout_seconds: float = 0.2
    port_wait_interval_seconds: float = 0.1


@dataclass
class AutoPickConfig:
    """Timeouts for the camera auto-pick step in `physiclaw setup hardware`.
    Values are tighter than warm-start: interactive setup wants snappier
    feedback when the phone /bridge page isn't responding. Cap stays
    below the CLI's HTTP timeout (60s) so the auto-pick loop has time
    to iterate camera indices after the bridge comes online."""

    bridge_wait_timeout_seconds: int = 25
    bridge_settle_seconds: float = 1.5


@dataclass
class CameraConfig:
    """Resolution + pixel format the server requests from cv2.VideoCapture.

    On Windows MSMF the default negotiation picks an uncompressed YUY2
    mode that fits USB bandwidth, which usually caps at 640×480 even
    on 4K cameras. Switching to MJPG-compressed lets us actually hit
    1920×1080 over a single USB cable. cv2.set() is best-effort —
    drivers snap to their nearest supported mode; the actual size is
    logged by Camera._warmup after the first frame."""

    width: int = 1920
    height: int = 1080
    fourcc: str = "MJPG"
    # Exposure control (Windows/Linux; macOS AVFoundation exposes no
    # exposure props through OpenCV, so both keys are ignored there).
    # The startup tune verifies with measured brightness and falls back
    # to manual stepping when firmware auto-exposure misbehaves.
    auto_exposure: bool = True
    exposure: int = -6


@dataclass
class EngineConfig:
    max_turns: int = 300
    # Session-level STUCK retries (engine.run): fresh session attempts after
    # a session ends STUCK. Distinct from `provider_retry_attempts` below.
    max_attempts: int = 3
    # Per-call provider retries on transient errors (rate limit, 5xx) within
    # one turn, each spaced by `retry_backoff_seconds`.
    provider_retry_attempts: int = 3
    retry_backoff_seconds: float = 5.0
    wait_default_minutes: int = 15
    react_cooldown_seconds: float = 6.0
    stale_tick_threshold: int = 8
    # Draft-the-plan reminder. Reading the IM takes ~3 turns of navigation
    # (wake peek → open app → open thread) before drafting is possible, so 4
    # fires exactly at the natural draft turn — silent during legitimate
    # navigation, per the tip philosophy: always-appearing tips get ignored.
    state_decay_turns: int = 4
    # Stuck guard (agent.engine.stuck) warn/block tiers — shared by all its
    # detectors, named for the flagship case: the warn-th camera-verified
    # no-change press on one target draws a ⚠; the block-th is refused
    # pre-dispatch.
    same_target_warn: int = 3
    same_target_block: int = 5
    # Plan step watchdog (agent.engine.plan): one in_progress step running
    # this many turns raises the stuck tip (warn), then the report-to-user
    # escalation (urgent).
    step_stuck_warn: int = 12
    step_stuck_urgent: int = 18
    # Plan gate (agent.engine.engine._dispatch): after this many turns a
    # session with no drafted plan has every tool blocked except
    # note / update_progress / end_session. Must exceed a legitimate
    # plan-less idle wake (peek IM → nothing → close) and the reminder
    # threshold `state_decay_turns` above.
    plan_required_after: int = 8
    # Scratchpad hard cap (chars) — ships every turn, so a working-memory ceiling,
    # not a dumping ground. Over-cap writes rejected with "summarize first".
    scratchpad_max_chars: int = 8192
    # Session trajectory: plan + scratchpad snapshots (per explicit write) fed to
    # the pre-close pitfalls corrective. Both caps apply from the newest entry
    # backward, keeping whole entries until either hits. `trajectory_max_snapshots`
    # = entry count (+ per-log memory retention); `trajectory_budget` = rendered
    # chars (practically large so a whole run is fed).
    trajectory_max_snapshots: int = 300
    trajectory_budget: int = 250 * 1024
    # Pre-close memory-cue gate: scan each turn's note/scratchpad/plan for
    # "remember this"/"记住" signals; at close, an unaddressed cue forces one
    # `save_memory` nudge (fail-open). Disable to skip the scan + gate.
    memory_cue_enabled: bool = True


@dataclass
class AgentConfig:
    """Agent runtime selection.

    ``model`` is a ``provider/model`` ref, e.g. ``"qwen/qwen3.6-plus"`` or
    ``"claude-code/claude-sonnet-4-6"``. The first segment selects the
    engine + provider; the second is the model id passed to that
    provider verbatim. Empty string means "use ``PHYSICLAW_MODEL`` env
    var, then fail loudly" — there is no universal default.
    """

    model: str = ""


@dataclass
class ProviderConfig:
    """Per-provider credentials (only).

    Empty strings mean "fall back to env" — see ``resolve_provider_key``
    for the env-var → config-key precedence each provider applies.
    Provider/model selection lives under ``[agent] model``.

    Field names match the provider id (qwen/moonshot/openai/anthropic/
    google/deepseek) — same convention OpenClaw uses.
    """

    qwen_api_key: str = ""
    moonshot_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    deepseek_api_key: str = ""


@dataclass
class CompactConfig:
    max_image_edge_px: int = 1566
    jpeg_quality: int = 85


@dataclass
class MemoryConfig:
    default_log_entries: int = 20
    bootstrap_log_entries: int = 10
    # Soft size budget for memory.md (chars) — rides every SYSTEM prompt. Over it,
    # `save_memory` nudges consolidation via `update_memory` (soft, not enforced).
    soft_cap_chars: int = 2048


@dataclass
class ClaudeConfig:
    timeout_seconds: int = 180
    stream_buffer_mb: int = 10
    max_attempts: int = 3
    retry_backoff_seconds: float = 5.0


@dataclass
class RetentionConfig:
    trace_days: int = 7
    # Daily human-readable log files (engine-/runtime-/claude-*.log) are
    # tiny next to session artifacts, so they keep a longer window.
    log_days: int = 30
    # A cron job stuck in `fired` (no session ever closed it) is
    # auto-closed after this long: one-time → fail, periodic → re-armed.
    fired_expire_hours: int = 24


@dataclass
class PitfallsConfig:
    """Agent-flagged turn-wasting traps (`add_pitfall`, 0–3 per long DONE
    session, append-only, newest on top). Always injected after user skills.
    `max_items` caps the list (curator consolidates toward it; a hard cut from
    the bottom/oldest enforces it); `max_item_chars` clamps each line.
    `capture_turn_floor` = the turn count above which a DONE close forces the
    capture nudge (a long run almost always hit a trap worth banking; the agent
    may still add 0). `capture_enabled` / `curate_enabled` gate the two passes."""

    max_items: int = 20
    max_item_chars: int = 120
    capture_turn_floor: int = 50
    capture_enabled: bool = True
    curate_enabled: bool = True


@dataclass
class SkillsConfig:
    """Source repo for ``physiclaw skills install``. Empty = no default;
    users must pass ``--from`` or set this key. Convention: the source
    repo must contain a top-level ``skills/<name>/SKILL.md`` layout.

    ``official_base_url`` is the site namespace ``skills sync official``
    pulls from — ``latest.json``, the ``.zip`` pack, and its ``.sha256``
    all live directly under it. Override only to point at a mirror/staging.

    ``sync_auto`` runs that sync at ``physiclaw server`` startup in a background
    thread (never blocks startup; fail-soft; idempotent — a no-op when the commit
    is unchanged), so official skills stay current without a manual command. The
    next agent session picks up a change. Set false to pin what's mounted."""

    default_source: str = ""
    official_base_url: str = "https://physiclaw.ai/downloads/official-skills"
    sync_auto: bool = True


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)
    warm_start: WarmStartConfig = field(default_factory=WarmStartConfig)
    auto_pick: AutoPickConfig = field(default_factory=AutoPickConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    compact: CompactConfig = field(default_factory=CompactConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    pitfalls: PitfallsConfig = field(default_factory=PitfallsConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)


_SECTION_TYPES: dict[str, type] = {
    "server": ServerConfig,
    "update": UpdateConfig,
    "warm_start": WarmStartConfig,
    "auto_pick": AutoPickConfig,
    "camera": CameraConfig,
    "engine": EngineConfig,
    "agent": AgentConfig,
    "provider": ProviderConfig,
    "compact": CompactConfig,
    "memory": MemoryConfig,
    "claude": ClaudeConfig,
    "retention": RetentionConfig,
    "pitfalls": PitfallsConfig,
    "skills": SkillsConfig,
}


# Top-level sections parsed elsewhere — accepted by the loader but
# skipped by the dataclass validator. `providers` holds per-provider
# overrides like `[providers.<id>] base_url = "..."` (read directly via
# `provider_base_url_override`).
_FREEFORM_SECTIONS: frozenset[str] = frozenset({"providers"})


_FILE_HEADER = """\
# PhysiClaw config. Edit with `physiclaw config edit`. Changes apply on
# next `physiclaw server` start. Delete a key to revert to the built-in
# default. Unknown keys / sections fail loudly on load.
"""

_SECTION_COMMENTS: dict[str, str] = {
    "update": (
        "Update check. `check = true` checks PyPI in the background while "
        "`physiclaw server` runs and notifies at the next start when a newer "
        "release is ready; it never installs on its own — apply it with "
        "`uv tool upgrade physiclaw` once the server is stopped. "
        "PHYSICLAW_DISABLE_UPDATE_CHECK=1 env overrides to off."
    ),
    "warm_start": "Timeouts for `physiclaw server --warm-start` hardware reconnect.",
    "auto_pick": "Timeouts for the camera auto-pick step in `physiclaw setup hardware`.",
    "camera": (
        "Resolution + pixel format requested from cv2.VideoCapture. MJPG "
        "is needed on Windows to hit 1080p — the YUY2 default snaps to "
        "640×480 over USB. Drivers may round to the nearest supported mode."
    ),
    "engine": (
        "Agent tool-call loop: runaway safeguards (turn cap, stuck guard, "
        "plan gate) + retry + pacing."
    ),
    "agent": (
        "Engine + model selection. `model` is a `provider/model` ref, e.g. "
        "`qwen/qwen3.6-plus` or `claude-code/claude-sonnet-4-6`. "
        "`PHYSICLAW_MODEL` env var overrides."
    ),
    "provider": (
        "Per-provider API keys. Field names match the provider id "
        "(qwen/moonshot/openai/anthropic). Env vars (QWEN_API_KEY, "
        "MOONSHOT_API_KEY, OPENAI_API_KEY, …) override these. Treat "
        "keys here like ssh keys."
    ),
    "compact": "Screenshot compression before sending to the LLM.",
    "memory": "Daily-log loading: bootstrap preload + on-demand `read_logs` defaults.",
    "claude": "Applied when [agent] model = 'claude-code/...' (external CLI subprocess).",
    "retention": "Purge window for on-disk engine trace logs + cron job history.",
    "pitfalls": (
        "Agent-flagged turn-wasting traps (~/.physiclaw/learned/pitfalls/), "
        "0–3 appended per long DONE session, always shown after user skills. "
        "`max_items` / `max_item_chars` bound the list; a curator consolidates."
    ),
    "skills": (
        "`physiclaw skills`. `default_source`: repo for `install` (empty = "
        "require `--from`; `owner/repo` or a git URL). `official_base_url` + "
        "`sync_auto`: the `sync official` pack source + auto-sync at startup."
    ),
}

_FIELD_COMMENTS: dict[tuple[str, str], str] = {
    ("server", "save_tool_calls"): "dump every peek/screenshot output",
    (
        "server",
        "save_snapshots",
    ): "dump each snapshot frame (rotated, with bbox overlay)",
    ("server", "save_screenshots"): "dump every raw phone-own screenshot",
    ("server", "save_raw_camera"): "dump every raw camera frame at capture",
    ("memory", "default_log_entries"): "on-demand `read_logs` default size (max 200)",
    (
        "memory",
        "bootstrap_log_entries",
    ): "auto-preloaded into the memory slot at every wake",
    (
        "camera",
        "auto_exposure",
    ): "false = hold `exposure` manually (Windows/Linux; macOS ignores)",
    (
        "camera",
        "exposure",
    ): "log2 seconds (-6 = 1/64s, indoors -4..-8); manual-fallback start",
}


def config_path() -> Path:
    return paths.HOME / "config.toml"


def _build_section(name: str, cls: type, overrides: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    extra = set(overrides) - known
    if extra:
        raise ConfigError(
            f"unknown key(s) in [{name}]: {sorted(extra)} — valid keys: {sorted(known)}"
        )
    return cls(**overrides)


def load(path: Path | None = None) -> Config:
    path = path or config_path()
    if not path.exists():
        return Config()
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except UnicodeDecodeError as e:
        # TOML mandates UTF-8. On non-UTF-8 Windows codepages (cp936, cp1252)
        # an editor or an old physiclaw build that called write_text without
        # an explicit encoding could leave the file in the system codepage.
        # Surface a friendly recovery hint instead of a stack trace.
        raise ConfigError(
            f"{path} is not valid UTF-8 (got byte 0x{e.object[e.start]:02x} at "
            f"position {e.start}). TOML requires UTF-8.\n"
            f"  Recover: delete the file and re-run.\n"
            f'      Windows: Remove-Item "{path}"\n'
            f'      macOS/Linux: rm "{path}"'
        ) from e
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"failed to read {path}: {e}") from e

    unknown_sections = set(raw) - set(_SECTION_TYPES) - _FREEFORM_SECTIONS
    if unknown_sections:
        raise ConfigError(
            f"unknown section(s) in {path}: {sorted(unknown_sections)} — "
            f"valid sections: {sorted(_SECTION_TYPES)}"
        )

    built: dict[str, Any] = {}
    for key, cls in _SECTION_TYPES.items():
        overrides = raw.get(key, {})
        if not isinstance(overrides, dict):
            raise ConfigError(
                f"[{key}] must be a table, got {type(overrides).__name__}"
            )
        built[key] = _build_section(key, cls, overrides)

    return Config(**built)


def _toml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return repr(v)
    if isinstance(v, float):
        import math

        if math.isnan(v) or math.isinf(v):
            raise ConfigError(f"non-finite float not representable in TOML: {v!r}")
        return repr(v)
    if isinstance(v, str):
        return f'"{v}"'
    raise ConfigError(f"cannot serialize {v!r} ({type(v).__name__}) to TOML")


def to_toml(cfg: Config, *, with_comments: bool = False) -> str:
    parts: list[str] = []
    if with_comments:
        parts.append(_FILE_HEADER.rstrip() + "\n")
    for name in _SECTION_TYPES:
        header_comment = _SECTION_COMMENTS.get(name) if with_comments else None
        if header_comment:
            parts.append(f"# {header_comment}")
        parts.append(f"[{name}]")
        section = getattr(cfg, name)
        for f in fields(section):
            val = getattr(section, f.name)
            line = f"{f.name} = {_toml_scalar(val)}"
            inline = _FIELD_COMMENTS.get((name, f.name)) if with_comments else None
            if inline:
                line = f"{line:<32} # {inline}"
            parts.append(line)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def write_default(path: Path | None = None) -> Path:
    """Write a commented default ``config.toml`` if absent. No-op if present."""
    path = path or config_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text(path, to_toml(Config(), with_comments=True))
    return path


def get(cfg: Config, dotted: str) -> Any:
    """Walk a dotted path like ``engine.max_turns`` against ``cfg``.

    Raises ``ConfigError`` with the list of siblings at the failing level so
    the CLI can print an actionable message.
    """
    parts = dotted.split(".")
    if not parts or not all(parts):
        raise ConfigError(f"empty path: {dotted!r}")
    cursor: Any = cfg
    for i, part in enumerate(parts):
        if not dataclasses.is_dataclass(cursor):
            walked = ".".join(parts[:i])
            raise ConfigError(f"{walked!r} is a leaf value, not a section")
        siblings = [f.name for f in fields(cursor)]
        if part not in siblings:
            walked = ".".join(parts[:i]) or "<root>"
            raise ConfigError(
                f"unknown key {part!r} at {walked}; valid: {sorted(siblings)}"
            )
        cursor = getattr(cursor, part)
    return cursor


def _coerce(raw: str, current: Any) -> Any:
    """Cast a CLI string argument to the type of ``current`` (the field default)."""
    if isinstance(current, bool):  # bool before int — bool is an int subclass
        if raw.lower() in ("true", "1", "yes", "on"):
            return True
        if raw.lower() in ("false", "0", "no", "off"):
            return False
        raise ConfigError(f"can't parse {raw!r} as bool (use true/false)")
    if isinstance(current, int):
        try:
            return int(raw)
        except ValueError as e:
            raise ConfigError(f"can't parse {raw!r} as int") from e
    if isinstance(current, float):
        try:
            return float(raw)
        except ValueError as e:
            raise ConfigError(f"can't parse {raw!r} as float") from e
    return raw  # str


def _validate_dotted(dotted: str) -> tuple[str, str]:
    """Confirm ``dotted`` is a known ``section.field`` and return the parts."""
    parts = dotted.split(".")
    if len(parts) != 2 or not all(parts):
        raise ConfigError(f"key must be section.field (got {dotted!r})")
    section, field_name = parts
    if section not in _SECTION_TYPES:
        raise ConfigError(
            f"unknown section {section!r}; valid: {sorted(_SECTION_TYPES)}"
        )
    field_names = {f.name for f in fields(_SECTION_TYPES[section])}
    if field_name not in field_names:
        raise ConfigError(
            f"unknown key {field_name!r} at {section}; valid: {sorted(field_names)}"
        )
    return section, field_name


def set_dotted(dotted: str, raw_value: str, path: Path | None = None) -> None:
    """Update one ``section.field`` in ``config.toml`` in place.

    Preserves comments + ordering via tomlkit. The new value is coerced to
    the field's default type. Re-validates by reloading via the strict
    ``load()`` so the file is never left unparseable.
    """
    import tomlkit

    path = path or config_path()
    section, field_name = _validate_dotted(dotted)
    current = getattr(getattr(load(path), section), field_name)
    coerced = _coerce(raw_value, current)

    if not path.exists():
        write_default(path)
    doc = tomlkit.parse(read_text(path))
    if section not in doc:
        doc.add(section, tomlkit.table())
    doc[section][field_name] = coerced
    write_text(path, tomlkit.dumps(doc))
    # Refresh module-level CONFIG so same-process callers see the write
    # (re-import wouldn't trigger between CLI commands and immediate use).
    global CONFIG
    CONFIG = load(path)


def provider_base_url_override(provider_id: str) -> str | None:
    """Read `[providers.<provider_id>] base_url` from user config.toml.
    Lets users point a builtin provider at a proxy / alt endpoint
    (e.g. Moonshot's .ai vs .cn). Returns None when unset or the file
    is absent. Called once at provider construction — no caching."""
    path = config_path()
    try:
        raw = read_text(path)
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
    val = doc.get("providers", {}).get(provider_id, {}).get("base_url")
    return val if isinstance(val, str) else None


def unset_dotted(dotted: str, path: Path | None = None) -> bool:
    """Remove one ``section.field`` from ``config.toml`` so the built-in
    default applies. Returns True if a key was actually removed.
    """
    import tomlkit

    path = path or config_path()
    section, field_name = _validate_dotted(dotted)
    if not path.exists():
        return False
    doc = tomlkit.parse(read_text(path))
    if section not in doc or field_name not in doc[section]:
        return False
    del doc[section][field_name]
    write_text(path, tomlkit.dumps(doc))
    global CONFIG
    CONFIG = load(path)
    return True


# Module-level singleton, evaluated once at import. See CONFIG usage in the
# migrated consumers (engine, claude, runtime, plan, compact, memory, trace,
# warm_start, job_store).
#
# Catch ConfigError at import-time so a corrupted ~/.physiclaw/config.toml
# doesn't brick the entire CLI — without this, ``physiclaw uninstall`` and
# even ``physiclaw config edit`` would crash with a stack trace before they
# ever start, leaving the user no in-band recovery path. We fall back to
# defaults and emit ONE clear stderr line; the user still sees the error
# the moment they run a command that actually depends on their config.
try:
    CONFIG: Config = load()
except ConfigError as _e:
    import sys as _sys

    print(f"physiclaw: {_e}", file=_sys.stderr)
    print(
        "physiclaw: continuing with default config; fix the file or delete it.",
        file=_sys.stderr,
    )
    CONFIG = Config()


# --- Model + provider selection ----------------------------------------------
# Order: env var > config.toml > raise. There is no implicit default —
# the user must configure a model. Refs use `provider/model` shape.


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
    if os.environ.get(MODEL_ENV_VAR):
        return os.environ[MODEL_ENV_VAR], f"{MODEL_ENV_VAR} env"
    if CONFIG.agent.model:
        return CONFIG.agent.model, "config.toml [agent] model"
    raise RuntimeError(_NO_MODEL_MSG)


def parse_model_ref(ref: str) -> tuple[str, str]:
    """Split `"provider/model-id"` on the FIRST slash.

    `"qwen/qwen3.6-plus"`  →  `("qwen", "qwen3.6-plus")`.
    `"openrouter/openai/gpt-5"`  →  `("openrouter", "openai/gpt-5")`.
    """
    if "/" not in ref:
        raise ValueError(
            f"model ref {ref!r} must be 'provider/model' (e.g. 'qwen/qwen3.6-plus')"
        )
    provider_id, model_id = ref.split("/", 1)
    if not (provider_id and model_id):
        raise ValueError(f"model ref {ref!r} has empty provider or model segment")
    return provider_id, model_id


# --- Provider credential resolution. -----------------------------------------
# Order: env var(s) in declaration order > config.toml > None. Empty
# strings in config count as "unset" and fall through to the next layer.


def resolve_provider_key(
    env_vars: tuple[str, ...],
    config_key: str,
) -> tuple[str | None, str | None]:
    """Generic credential resolver. Returns ``(key, source)``; both
    ``None`` if not set anywhere.

    ``env_vars`` are checked in order (first hit wins). If none match,
    falls through to ``CONFIG.provider.<config_key>``. ``source`` is a
    human-readable string for diagnostic output (``"OPENAI_API_KEY env"``
    or ``"config.toml [provider] openai_api_key"``).
    """
    for var in env_vars:
        val = os.environ.get(var)
        if val:
            return val, f"{var} env"
    val = getattr(CONFIG.provider, config_key, "")
    if val:
        return val, f"config.toml [provider] {config_key}"
    return None, None


__all__ = [
    "CONFIG",
    "Config",
    "ConfigError",
    "MODEL_ENV_VAR",
    "config_path",
    "get",
    "load",
    "model_ref",
    "model_ref_with_source",
    "parse_model_ref",
    "provider_base_url_override",
    "resolve_provider_key",
    "set_dotted",
    "to_toml",
    "unset_dotted",
    "write_default",
]
