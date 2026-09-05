"""User-tunable configuration loaded from ``~/.physiclaw/config.toml``.

Defaults live on the ``@dataclass`` sections — ``to_toml()`` + ``write_default()``
render the commented template from those same defaults, so bumping a default
doesn't need two edits.

Unknown top-level sections, unknown keys inside a section, and
wrong-typed values (a quoted number, a bool for an int) raise
``ConfigError`` on load. The CLI catches this and points users at
``physiclaw config edit``.

Layering (first match wins):
    CLI flag  >  env var  >  config.toml  >  built-in default
"""

import dataclasses
import json
import os
import tomllib
from collections.abc import MutableMapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_type_hints

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

    cv2.set() is best-effort — drivers snap to their nearest supported
    mode, so the 4K default serves 4K/2K/1080p cameras alike;
    Camera._warmup logs the negotiated size and steps down
    RESOLUTION_FALLBACKS when a driver mishandles the over-ask (Windows
    MSMF). Extra capture pixels never reach the LLM (CompactConfig caps
    every view) — they survive as sharpness in the downscale. MJPG
    matters on Windows: the YUY2 default caps at 640×480 over USB even
    on 4K cameras."""

    width: int = 3840
    height: int = 2160
    fourcc: str = "MJPG"
    # Exposure control. Windows/Linux go through OpenCV capture props;
    # macOS AVFoundation exposes none, so there the keys apply only when
    # the UVC/IOKit channel controls the camera (see platform.uvc) and
    # are ignored otherwise. Default is manual (`auto_exposure = False`): a
    # held value keeps one exposure regime for the screen-filming rig,
    # where firmware AE drifts and thrashes as the agent flips between dark
    # and white screens. `exposure` is then the held value (calibrate phone
    # brightness so white renders just below clipping); flip to True to let
    # firmware AE drive, with the tune stepping from `exposure` if it fails.
    # Log2-seconds scale within exposure.py's -8..-4 band (-5 ceiling on
    # rigs whose lens isn't pinned; see UNPINNED_MAX_EXPOSURE).
    auto_exposure: bool = False
    exposure: int = -6


@dataclass
class VisionConfig:
    """Field-tunable detection thresholds.

    Defaults are the values calibrated on the reference rigs (2026-07,
    real session frames — see the docstrings in ``core/vision``). A
    different camera, distance, or lighting can shift what "blurry",
    "washed out", or "a badge appeared" look like; these keys let a rig
    be re-tuned with a config edit instead of a source edit. The vision
    modules log their measured values, so tune against a live session.
    """

    # quality.py — is a view usable?
    blur_threshold: float = 80.0  # Laplacian variance floor (sharp ≥ 300)
    blown_clip_pct: float = 0.12  # clipped-pixel fraction for the blown rule
    blown_median_luma: float = 200.0  # ...only when the median is below this
    blown_blob_count: int = 6  # icon-grid white-out blob count
    washed_highlight_pct: float = 0.5  # highlight (≥225) fraction for washed-white
    # watchdog.py — did the idle screen wake?
    wake_std_increase: float = 4.0
    wake_mean_increase: float = 4.0
    badge_min_area: int = 30  # warm pixels for a badge wake
    # change.py — did a gesture change the screen?
    change_ratio_threshold: float = 0.003


@dataclass
class EngineConfig:
    max_turns: int = 300
    # Wall-clock backstop for one engine session (seconds; 0 disables). The
    # turn cap above can't bound time — one hung MCP call holds a turn for
    # minutes — so the loop also watches the clock. Exhaustion closes STUCK
    # and is not retried (a slow environment would burn every retry too).
    max_session_seconds: int = 3600
    # Session-level STUCK retries (engine.run): fresh session attempts after
    # a session ends STUCK. Distinct from `provider_retry_attempts` below.
    max_attempts: int = 3
    # Per-call provider retries on transient errors (rate limit, 5xx) within
    # one turn, each spaced by `retry_backoff_seconds`.
    provider_retry_attempts: int = 3
    retry_backoff_seconds: float = 5.0
    # HTTP read timeout for one provider call (seconds). Reasoning models
    # spend 80–130s on a decision over a full screen listing (field-
    # measured on kimi-k2.6), so a 120s ceiling turned a slow answer into
    # a transient error, a retry, and a conductor handover.
    provider_timeout_seconds: float = 300.0
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
    # this many turns raises the stuck tip (warn), then the forced re-plan
    # rejection (urgent — policy.StuckReflection). A real 11-turn flail sat
    # under the old warn=12; steps are sized at ~5+ calls (CONVENTION) and
    # the counter resets on every tick, so 8 stays quiet on healthy steps.
    step_stuck_warn: int = 8
    step_stuck_urgent: int = 12
    # Plan gate (agent.engine.dispatch, via policy.PlanGate): after this many turns a
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

    ``plugins`` lists the engine's turn plugins as comma-separated
    ``module:factory`` paths (`contract.plugin.TurnPlugin`), asked in
    order each turn before the LLM. Empty string (the default) means
    the engine's built-in plugin set (`engine/plugins.py` owns that
    default — this neutral layer names no plugin); ``"none"`` disables
    them all, making every turn a plain provider call. The engine loads
    these blindly and fail-open: it knows the protocol, not the names.
    """

    model: str = ""
    plugins: str = ""


@dataclass
class ProviderConfig:
    """Per-provider credentials (only).

    Empty strings mean "fall back to env" — see
    ``common.model_ref.resolve_provider_key`` for the env-var →
    config-key precedence each provider applies. Provider/model
    selection lives under ``[agent] model``.

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
    """Size/quality of every image the LLM sees — applied server-side at
    encode (`vision.util.encode_view_jpeg`) and again at the agent's
    ingress (`common.image.scale_image_bytes`) as a safety net. 1566 keeps a
    portrait phone view at ~1.15 megapixels: Claude's hard-resize sweet
    spot (1568), well inside Kimi/Qwen budgets — larger only adds
    tokens + latency, smaller blurs fine print."""

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
class ConductorConfig:
    """Micro-call knobs for the conductor's DECIDE nodes.

    ``micro_model`` is a ``provider/model`` ref for the decision calls
    (the plan's cheap tier); empty = the session's own model. Fail-open:
    an unbuildable override falls back to the session provider.
    ``micro_confidence`` is the escalation floor — a validated answer
    reporting less hands the playbook over to the model instead of
    guessing. Tune by replaying recorded listings
    (`physiclaw playbooks replay`)."""

    micro_model: str = ""
    micro_confidence: float = 0.6


@dataclass
class ClaudeConfig:
    timeout_seconds: int = 180
    stream_buffer_mb: int = 10
    max_attempts: int = 3
    retry_backoff_seconds: float = 5.0


@dataclass
class RetentionConfig:
    # Per-session artifact dirs (engine + claude sessions, macro run
    # dirs) and terminal cron jobs: a month covers a post-mortem on a
    # run the user only notices weeks later.
    trace_days: int = 30
    # Daily human-readable log files (engine-/runtime-/claude-*.log),
    # tiny next to session artifacts.
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
class MacrosConfig:
    """Global gate for user-authored gesture macros. ``enabled = false``
    hides ALL macros from the agent regardless of each file's own
    ``enabled`` flag — the prompt then carries zero macro content (no
    section, no run_macro tool, no MACRO.md doctrine). CLI authoring
    commands (init/check/run/stats) ignore this gate so a macro can be
    rehearsed while the fleet-wide switch is off."""

    enabled: bool = True


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)
    warm_start: WarmStartConfig = field(default_factory=WarmStartConfig)
    auto_pick: AutoPickConfig = field(default_factory=AutoPickConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    compact: CompactConfig = field(default_factory=CompactConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    conductor: ConductorConfig = field(default_factory=ConductorConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    pitfalls: PitfallsConfig = field(default_factory=PitfallsConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    macros: MacrosConfig = field(default_factory=MacrosConfig)


def _section_types() -> dict[str, type]:
    """The section-name → dataclass map, derived from ``Config``'s own
    fields — the sections ARE the dataclass, so a new section is one
    declaration, not a hand-kept mirror edit. Field order is declaration
    order, which ``to_toml`` relies on for section ordering."""
    hints = get_type_hints(Config)
    return {f.name: hints[f.name] for f in fields(Config)}


_SECTION_TYPES: dict[str, type] = _section_types()


# Top-level sections parsed elsewhere — skipped by the dataclass
# validator (their KEYS are free-form), but shape-checked on load.
# `providers` holds per-provider overrides like
# `[providers.<id>] base_url = "..."` (read directly via
# `model_ref.provider_base_url_override`).
_FREEFORM_SECTIONS: frozenset[str] = frozenset({"providers"})


def _validate_providers(providers: Any) -> None:
    """Shape-check the free-form ``[providers]`` section: unknown
    provider ids and extra keys are fine, but each entry must be a
    sub-table (``[providers.<id>]``) and ``base_url`` a string. The flat
    spelling (``moonshot = "url"`` under ``[providers]``) used to load
    silently and AttributeError at provider construction. The runtime
    reader stays fail-soft (``model_ref.provider_base_url_override``)."""
    if not isinstance(providers, dict):
        raise ConfigError(
            f"[providers] must be a table, got {type(providers).__name__}"
        )
    for pid, entry in providers.items():
        if not isinstance(entry, dict):
            raise ConfigError(
                f"[providers] {pid}: expected a sub-table — write\n"
                f"    [providers.{pid}]\n"
                f'    base_url = "https://..."\n'
                f"  not: {pid} = {entry!r}"
            )
        if "base_url" in entry:
            _check_scalar_type(f"providers.{pid}", "base_url", str, entry["base_url"])


_FILE_HEADER = """\
# PhysiClaw config. Edit with `physiclaw config edit`. Changes apply on
# next `physiclaw server` start. Delete a key to revert to the built-in
# default. Unknown keys / sections / wrong-typed values fail loudly on load.
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
        "Resolution + pixel format requested from cv2.VideoCapture. The "
        "default asks for 4K; drivers snap to the camera's nearest "
        "supported mode (2K → 2560×1440, 1080p → 1920×1080), so one "
        "default fits any camera. MJPG is needed on Windows for high "
        "resolutions — the YUY2 default snaps to 640×480 over USB."
    ),
    "vision": (
        "Detection thresholds (blur / washed-out / wake / screen-change), "
        "calibrated on the reference rigs. Re-tune here when a different "
        "camera, distance, or lighting mis-flags views — the vision "
        "modules log their measured values."
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
    "compact": (
        "Size/quality of every image the LLM sees. `max_image_edge_px` caps "
        "the long edge of camera views and phone screenshots (applied "
        "server-side at capture and again at the agent's ingress); 1566 ≈ "
        "the ~1.15-megapixel vision sweet spot — larger wastes tokens, "
        "smaller blurs fine print."
    ),
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
    "macros": (
        "`physiclaw macros`. `enabled = false` hides ALL macros from the "
        "agent (per-file `enabled` flags don't matter); the CLI authoring "
        "commands keep working so macros can still be rehearsed."
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
    (
        "agent",
        "plugins",
    ): "turn plugins, comma-separated module:factory; '' = built-in set, 'none' = off",
    ("memory", "default_log_entries"): "on-demand `read_logs` default size (max 200)",
    (
        "memory",
        "bootstrap_log_entries",
    ): "auto-preloaded into the memory slot at every wake",
    (
        "camera",
        "auto_exposure",
    ): "false (default) holds `exposure` manually; true = firmware AE (Windows/Linux; macOS ignores)",
    (
        "camera",
        "exposure",
    ): "log2 seconds (-6 = 1/64s, indoors -4..-8); the held manual value (tune start when auto)",
}


def config_path() -> Path:
    return paths.HOME / "config.toml"


def _check_scalar_type(section: str, key: str, declared: Any, value: Any) -> Any:
    """Validate one TOML override against the field's declared type — the
    'type mismatch' half of ConfigError's contract. Without it a quoted
    number (`blur_threshold = "80"`) loads silently and crashes mid-session
    at the first comparison instead of erroring at startup.

    tomllib already yields real Python types, so this compares rather than
    parses. Two scalar subtleties: TOML `true` IS a Python bool and bool
    subclasses int, so int fields must reject bools explicitly; and a TOML
    integer for a float field (`exposure = -4`) is natural — coerced. Every
    config field today is a plain int/float/str/bool; anything else passes
    through untouched so a future richer field isn't silently rejected.
    """
    if declared not in (int, float, str, bool):
        return value
    if declared is bool:
        if isinstance(value, bool):
            return value
    elif declared is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif declared is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif isinstance(value, str):
        return value
    raise ConfigError(
        f"[{section}] {key}: expected {declared.__name__}, "
        f"got {type(value).__name__} ({value!r})"
    )


def _build_section(name: str, cls: type, overrides: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    extra = set(overrides) - known
    if extra:
        raise ConfigError(
            f"unknown key(s) in [{name}]: {sorted(extra)} — valid keys: {sorted(known)}"
        )
    hints = get_type_hints(cls)
    checked = {
        key: _check_scalar_type(name, key, hints.get(key), value)
        for key, value in overrides.items()
    }
    return cls(**checked)


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
    if "providers" in raw:
        _validate_providers(raw["providers"])

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
        # json.dumps escaping (\" \\ \uXXXX, control chars) is valid TOML
        # basic-string syntax — a bare f'"{v}"' emitted invalid TOML for
        # values containing quotes or backslashes (Windows paths).
        return json.dumps(v)
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
    section_tbl = doc[section]
    if not isinstance(section_tbl, MutableMapping):
        # Unreachable: the strict load() above rejects non-table sections.
        raise ConfigError(f"[{section}] is not a table in {path}")
    section_tbl[field_name] = coerced
    write_text(path, tomlkit.dumps(doc))
    # Refresh module-level CONFIG so same-process callers see the write
    # (re-import wouldn't trigger between CLI commands and immediate use).
    global CONFIG
    CONFIG = load(path)


SERVER_ENV_VAR = "PHYSICLAW_SERVER"

# The debug-harness switches — deliberately env-only, never config
# fields: `physiclaw debug` exports them in the server process it
# starts, the runtime subprocess inherits them, and they die with the
# run — debug mode can't be left armed in a deployment. DEBUG
# virtualizes the user's side of the IM thread (`--task --reply`
# scripts it); MACRO_FAILURE halts on the first macro abort for
# inspection (the debug default; `--no-macro-failure` opts out).
DEBUG_ENV_VAR = "PHYSICLAW_DEBUG"
MACRO_FAILURE_ENV_VAR = "PHYSICLAW_MACRO_FAILURE"


# The wildcard bind addresses. One contract, several consumers: a server
# can listen on these but a client can't dial them (URL builders map them
# to loopback), and the control plane's Host gate treats them as the
# user's LAN-exposure opt-in. Shared so the halves can't drift.
WILDCARD_HOSTS = frozenset({"0.0.0.0", "::"})


def url_host(host: str) -> str:
    """Bind address → the host part of a dialable URL: wildcards map to
    loopback (a server can listen on them, a client can't dial them) and
    a bare IPv6 literal gets bracketed (`http://::1:8048` is unparseable).
    URL-scoped on purpose — sockets need the unbracketed form."""
    if host in WILDCARD_HOSTS:
        return "127.0.0.1"
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def server_url() -> str:
    """The control-plane base URL agent-side clients dial.

    `$PHYSICLAW_SERVER` wins (the launcher pins it for its subprocess
    tree); otherwise built from ``[server] host``/``port``, with the
    wildcard binds normalized to loopback — a server can listen on
    0.0.0.0/:: but a client can't dial them. Every hop that talks to
    the control plane resolves through here; the phone-bridge and QR
    URLs are built from the LIVE bound port instead and deliberately
    don't use this.
    """
    env = os.environ.get(SERVER_ENV_VAR)
    if env:
        return env
    return f"http://{url_host(CONFIG.server.host)}:{CONFIG.server.port}"


def unset_dotted(dotted: str, path: Path | None = None) -> bool:
    """Remove one ``section.field`` from ``config.toml`` so the built-in
    default applies. Returns True if a key was actually removed.
    """
    import tomlkit

    path = path or config_path()
    section, field_name = _validate_dotted(dotted)
    if not path.exists():
        return False
    # Same strict-load guard as set_dotted: a malformed file (e.g. a
    # non-table `camera = 3`) surfaces the schema ConfigError instead of
    # a raw TypeError from indexing the tomlkit doc below.
    load(path)
    doc = tomlkit.parse(read_text(path))
    if section not in doc:
        return False
    section_tbl = doc[section]
    if not isinstance(section_tbl, MutableMapping):
        # Unreachable: the strict load() above rejects non-table sections.
        raise ConfigError(f"[{section}] is not a table in {path}")
    if field_name not in section_tbl:
        return False
    del section_tbl[field_name]
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


# --- Model + provider selection lives in `common.model_ref` (see its
# docstring for the split rationale and the CONFIG-rebind subtlety).
# Re-exported for existing call sites — this list is frozen compat; new
# call sites import from `physiclaw.common.model_ref`. Position at the
# bottom is cosmetic: model_ref defers all CONFIG access into function
# bodies, so import order doesn't matter.
from physiclaw.common.model_ref import (  # noqa: E402
    MODEL_ENV_VAR,
    model_ref,
    model_ref_with_source,
    parse_model_ref,
    provider_base_url_override,
    resolve_provider_key,
)

__all__ = [
    "CONFIG",
    "Config",
    "ConfigError",
    "DEBUG_ENV_VAR",
    "MACRO_FAILURE_ENV_VAR",
    "MODEL_ENV_VAR",
    "SERVER_ENV_VAR",
    "config_path",
    "get",
    "load",
    "model_ref",
    "model_ref_with_source",
    "parse_model_ref",
    "provider_base_url_override",
    "resolve_provider_key",
    "server_url",
    "url_host",
    "set_dotted",
    "to_toml",
    "unset_dotted",
    "write_default",
]
