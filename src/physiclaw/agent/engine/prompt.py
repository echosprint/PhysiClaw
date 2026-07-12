"""SYSTEM prompt composition + prefix-cache verification.

Each section is a `list[str]` builder; the final prompt is `"\\n".join(lines)`.
Disabled sections return `[]` so their absence costs nothing.

Cache layout: the system message is entirely session-stable, so it
caches once and hits on every subsequent turn — and on the first turn
of every later wake within the 5-min TTL. Wake-volatile content (the
cron-fired jobs block, the trigger stamps) lives in the wake-trigger
user message that follows, keeping the system bytes identical across
wakes. DashScope caches at message granularity, so this is the only
structure that gets cross-session hits.

The inline `## Tooling` index card duplicates the schema sent via the
provider's `tools=` API — open-weight models often miss tools that appear
only in the schema, so the card is a redundant anchor. Providers whose
models read the schema reliably opt out via `INLINE_TOOL_INDEX = False`
(see `BaseProvider`) and skip the card entirely.
"""

import hashlib
import logging
import re
from pathlib import Path

from physiclaw.agent.engine import memory, mcp_inventory, screen_layout
from physiclaw.common.config import CONFIG
from physiclaw.common.text import read_text

log = logging.getLogger(__name__)

CONTEXT_DIR = Path(__file__).resolve().parent.parent / "context"

# Engine-enforced thresholds quoted by the doctrine files. Doctrine uses
# `{{token}}` placeholders so the prompt can never drift from the config
# the engine actually enforces; substituted on every render (config is
# process-constant, so the rendered bytes stay cache-stable).
_DOCTRINE_TOKENS = {
    "{{same_target_warn}}": str(CONFIG.engine.same_target_warn),
    "{{same_target_block}}": str(CONFIG.engine.same_target_block),
    "{{step_stuck_warn}}": str(CONFIG.engine.step_stuck_warn),
    "{{step_stuck_urgent}}": str(CONFIG.engine.step_stuck_urgent),
    "{{plan_required_after}}": str(CONFIG.engine.plan_required_after),
}


def _fill_tokens(name: str, body: str) -> str:
    """Substitute `{{token}}` placeholders; warn on leftovers (a typo'd
    token would otherwise reach the model as literal braces)."""
    for token, value in _DOCTRINE_TOKENS.items():
        body = body.replace(token, value)
    leftover = re.search(r"\{\{[^}\n]*\}\}|\{\{", body)
    if leftover:
        log.warning(
            "doctrine %s: unresolved placeholder %r",
            name,
            leftover.group(),
        )
    return body


# OpenClaw-style modular doctrine: each named file is a slot with a defined
# role. Files are rendered in this fixed order; any missing file contributes
# nothing. Drop a new file in src/physiclaw/agent/context/ to opt into a slot — picked up
# on next session start without touching code.
DOCTRINE_FILE_ORDER = (
    "IDENTITY.md",  # who PhysiClaw is — short, one-paragraph card
    "USER.md",  # who PhysiClaw serves — read from memory/USER.md
    "SOUL.md",  # personality / tone / voice
    "AGENT.md",  # operational rules: Loop / Boundaries / Rules / Continuity
    "PHYSICLAW.md",  # tool-surface mechanics — also shipped via MCP initialize
    "TOOLS.md",  # extra user-authored tool guidance
    "PERSISTENCE.md",  # memory.md vs YYYY-MM-DD.md + read/write tools
    "JOBS.md",  # jobs.md + create_job/get_job/list_jobs/finish_job (immutable, append-only)
    "CONVENTION.md",  # engine turn rules — last so it sits next to mechanics
)


def render_system_prompts(
    *,
    local_tool_schemas: list[dict] | None = None,
    memory_ctx: str = "",
    builtin_skills_ctx: str = "",
    user_skills_ctx: str = "",
    pitfalls_ctx: str = "",
    provider_id: str = "",
) -> str:
    """Compose the full SYSTEM for one session — entirely session-stable.

    Order (above → below):
      # Doctrine           file-loop over DOCTRINE_FILE_ORDER. Slots:
                           IDENTITY, USER, SOUL, AGENT, PHYSICLAW, TOOLS,
                           PERSISTENCE, JOBS, CONVENTION — each rendered as a
                           `## <name>` block; missing = skipped. USER reads
                           from memory/.
      ## Tooling           inline tool index card — a redundant anchor for
                           models that miss schema-only tools; skipped when
                           the provider sets INLINE_TOOL_INDEX = False
      ## Built-in Skills   built-in skills, FULL bodies (always active)
      ## Screen layout     learned input/keyboard bboxes the skills above use
      ## Available skills  official + user skills (### subsections),
                           name+description index (load on demand via Skill)
      ## Learned pitfalls  agent-flagged traps, always-on (add_pitfall appends)
      ## Examples          ❌/✅ for the most common per-turn failures
      ## Reasoning Format  provider-specific reasoning wrapper (e.g. Qwen
                           `<think>...</think>`); pulled from
                           `BaseProvider.system_prompt_fragment()`
      ## memory.md         session-stable persistent facts — live file
                           dump (the spec lives in the PERSISTENCE.md slot)

    Built-in and user skills render into separate sections (both already
    carry their own `##` header + framing) so they never collide. Wake-
    volatile content (fired-jobs block, trigger stamps) lives in the user
    message the engine appends right after — keeping the system message
    byte-stable across wakes for cross-session cache hits.
    """
    lines: list[str] = [
        *_render_doctrine(),
        *_render_tooling(local_tool_schemas or [], provider_id),
        *_render_section(builtin_skills_ctx),
        *_render_screen_layout(),
        *_render_section(user_skills_ctx),
        *_render_section(pitfalls_ctx),
        *_render_examples(),
        *_render_reasoning_format(provider_id),
        *_render_memory(memory_ctx),
    ]
    # No trailing newline: sections end with a separator blank, and the
    # CLI dump asserts byte-equality with this exact string.
    return "\n".join(lines).rstrip("\n")


# ---------- section builders ----------


def _render_doctrine() -> list[str]:
    """Emit one `## <FileName>` block per file in DOCTRINE_FILE_ORDER,
    bodies injected raw under a `# Doctrine` wrapper. Re-read every call
    so file edits take effect on the next wake without a restart."""
    files = _load_doctrine_files()
    if not files:
        return []
    out: list[str] = ["# Doctrine", ""]
    for name, body in files:
        out.append(f"## {name}")
        out.append("")
        out.append(body)
        out.append("")
    return out


def _load_doctrine_files() -> list[tuple[str, str]]:
    """Return [(name, body)] for files in DOCTRINE_FILE_ORDER that exist
    and are non-empty, in order. Missing or empty files are skipped — the
    user opts in to a slot by creating the file.

    USER.md routes through `memory.load_user()` so the gitignored
    user-private file (in `memory/`, not `src/physiclaw/agent/context/`) renders
    inside `# Doctrine` from a single source of truth."""
    out: list[tuple[str, str]] = []
    for name in DOCTRINE_FILE_ORDER:
        if name == "USER.md":
            body = memory.load_user()
        else:
            path = CONTEXT_DIR / name
            body = read_text(path).rstrip() if path.exists() else ""
        if not body:
            continue
        out.append((name, _fill_tokens(name, body)))
    return out


def _render_tooling(local_tool_schemas: list[dict], provider_id: str = "") -> list[str]:
    """Inline tool index card. Each tool gets one line: `- **name** — first
    sentence`.

    MCP tools come from `mcp_inventory.discover_mcp_tools()` (AST parse of
    `src/physiclaw/core/server/tools.py`) — works offline, no MCP roundtrip. Local
    engine tools come from the caller. The two sources don't overlap by
    name, so the merge is a simple concat.

    Why the inline card at all: open-weight models (Qwen, Moonshot)
    routinely forget tools that appear only in the native `tools=` payload.
    Providers that don't need the anchor (`INLINE_TOOL_INDEX = False`, e.g.
    Anthropic) skip the whole card; unknown/empty provider keeps it — the
    safe default for dumps and new vendors."""
    if not _wants_tool_index(provider_id):
        return []
    all_tools = mcp_inventory.discover_mcp_tools() + (local_tool_schemas or [])
    if not all_tools:
        return []
    out: list[str] = [
        "## Tooling",
        "",
        "All available via the native tool-call API; names are "
        "case-sensitive — call them exactly as listed.",
        "",
    ]
    for s in all_tools:
        name = s.get("name") or ""
        if not name:
            continue
        desc = _first_sentence(s.get("description") or "")
        out.append(f"- **{name}** — {desc}" if desc else f"- **{name}**")
    out.append("")
    return out


def _wants_tool_index(provider_id: str) -> bool:
    """Whether the active provider needs the inline `## Tooling` card.
    Unknown or unset provider → True (dumps and new vendors keep the
    redundant anchor; only vendors that opt out lose it)."""
    from physiclaw.agent.provider import provider_class

    cls = provider_class(provider_id)
    return cls is None or cls.INLINE_TOOL_INDEX


def _render_section(ctx: str) -> list[str]:
    """Include a pre-rendered skill section (built-in bodies or user index).
    Each already carries its own `##` header + framing from the skill
    module, so this just drops it in with a trailing blank line."""
    return [ctx, ""] if ctx else []


def _render_screen_layout() -> list[str]:
    """Inject the learned screen layout (Spotlight + chat input boxes +
    keyboard) once ALL three pages are captured. Read from
    ``~/.physiclaw/screen-layout/`` at session start — cache-stable.

    Nothing here until setup is complete: while pages are still missing the
    turn-volatile tail reminder (`screen_layout.inject_tail`) lists the fields
    left to capture, and each `report_screen_layout` result confirms the save
    and how many remain. Once the last box lands, this loads the full layout on
    the next wake.
    """
    if not screen_layout.is_learned():
        return []
    md = screen_layout.load_layout_md() or "not set yet"
    return ["## Screen layout", "", md, ""]


def _render_examples() -> list[str]:
    """Concrete ❌ Wrong / ✅ Right patterns for the most common per-turn
    failure modes. Each pairs a doctrine rule with a worked example; rules
    the engine already corrects mid-run (turn shape aside) aren't repeated
    here — AGENT.md / CONVENTION.md carry them once."""
    return [
        "## Examples",
        "",
        "**Bbox transcription** — verbatim from the listing's bracket contents.",
        '❌ Row `45 [icon] "" [0.520,0.662,0.717,0.775] 0.56` → `tap(bbox=[0.518, …])` — the regenerated `0.518` lands on the neighbor.',
        "✅ All four numbers exactly — `0.520` stays `0.520`. Not in any grounded source → `screenshot`, never fabricate.",
        "",
        "**Screenshot escalation** — re-peeking a missed target returns the same gap.",
        '❌ `peek` home, no JD icon → `peek` again → tap a Safari link labeled "JD" → wrong page, STUCK.',
        "✅ `screenshot` next turn → scratchpad the icon's bbox → `peek` → tap the copy. The icon was always there; the camera couldn't resolve it.",
        "",
        "**Unchanged screen** — read the value: a small change (top-right cart badge 2 → 3, page stays put) means landed; a value unmoved through two presses is the app saying no (the toast already vanished).",
        "❌ Cart badge unread → tap again → duplicates. Or qty stays 2 → 20 tap/double_tap/nudged-bbox retries across cart and checkout.",
        '✅ tap `+` → qty still 2 in the attached view → retry once → still 2 = refusal. Find the evidence ("limit 2 per order", greyed `+`), scratchpad `TRIED: + stepper ×2, qty stuck at 2`, ask the user: "Store caps this at 2 — take 2, or switch brands?" and WAIT. Badge unreadable → open the cart and check.',
        "",
        "**Scratchpad accumulates** — plan-relevant payloads go in as you see them; navigation doesn't.",
        "❌ Scroll 2 orders across 4 pages, scratchpad nothing → views stubbed by compaction, items lost, re-scrape.",
        '✅ `note(..., scratchpad="order 1: cable ¥45, clips ¥17")` → scroll → reissue with `tape ¥22` appended → order 2 → append its lines. Reaching WeChat, the scratchpad is the full reply.',
        "",
    ]


def _render_reasoning_format(provider_id: str) -> list[str]:
    """Provider-specific reasoning fragment (e.g. Qwen's `<think>` wrapper),
    pulled from the vendor class's `system_prompt_fragment()` classmethod.
    Empty when the provider doesn't override it."""
    from physiclaw.agent.provider import provider_class

    cls = provider_class(provider_id)
    if cls is None:
        return []
    fragment = cls.system_prompt_fragment()
    if not fragment:
        return []
    return ["## Reasoning Format", fragment, ""]


def _render_memory(memory_ctx: str) -> list[str]:
    """Memory snapshot for this session. Sits ABOVE the cache boundary
    because it is byte-stable for the session's lifetime — re-emitted
    identically on every turn within one wake."""
    if not memory_ctx:
        return []
    return ["## memory.md", "", memory_ctx, ""]


# ---------- helpers ----------


def _first_sentence(text: str) -> str:
    """First sentence (or first line) of a tool description. Used to
    keep the inline tooling card to one line per tool."""
    line = (text or "").strip().split("\n", 1)[0].strip()
    if not line:
        return ""
    # Crude but predictable: cut at the first period followed by a space
    # (so "e.g. foo" inside a sentence doesn't get truncated).
    for sep in (". ", "; "):
        idx = line.find(sep)
        if 0 < idx < 200:
            return line[: idx + 1].rstrip()
    return line[:200].rstrip()


# Cache-marker placement lives in the provider classes — see
# `BaseProvider.serialize_history` (template) and the per-shape
# `CacheMarkers` factories. It's a wire-format concern that
# belongs with the request builder, not the prompt assembly.


def prefix_hash(system_prompt: str) -> str:
    """sha256 of the SYSTEM prompt content. Logged at session start so
    cache-hit rates are verifiable across wakes — the hash stays
    identical while doctrine/memory/tools don't change."""
    if not isinstance(system_prompt, str):
        raise ValueError(
            f"prefix_hash: system_prompt must be str, got {type(system_prompt).__name__}"
        )
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


# ---------- offline dump ----------


def dump(
    *,
    local_tool_schemas: list[dict] | None = None,
    memory_ctx: str = "",
    builtin_skills_ctx: str = "",
    user_skills_ctx: str = "",
    pitfalls_ctx: str = "",
    provider_id: str = "",
) -> str:
    """Render the SYSTEM prompt the same way the engine does, with all
    inputs optional so callers can dump the static skeleton without
    spinning up MCP or loading memory."""
    return render_system_prompts(
        local_tool_schemas=local_tool_schemas,
        memory_ctx=memory_ctx,
        builtin_skills_ctx=builtin_skills_ctx,
        user_skills_ctx=user_skills_ctx,
        pitfalls_ctx=pitfalls_ctx,
        provider_id=provider_id,
    )
