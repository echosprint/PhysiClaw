"""Macro discovery and the `## Available Macros` prompt section.

One directory per macro under ``~/.physiclaw/macros/`` holding a
``MACRO.yml`` (the skill-folder shape, filename capitalized like
``SKILL.md``: directory name IS the identity; the file's ``name`` field
must match). Only ``*/MACRO.yml`` is scanned,
so the machine-written ``stats.json`` at the root can never be read as a
macro, and a macro dir may carry the user's rehearsal notes untouched.

Discovery is realpath-scoped like `engine.skill`: a symlinked dir that
escapes the root is rejected. A file failing validation is excluded whole
with its reason kept on the ScanEntry — `physiclaw macros check` prints
it; the engine only ever sees valid AND enabled specs.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from physiclaw.agent.macros import scaffold
from physiclaw.agent.macros.model import (
    MACRO_FILENAME,
    Macro,
    MacroError,
    check_name,
)
from physiclaw.agent.macros.parse import parse_macro
from physiclaw.common import paths
from physiclaw.common.config import CONFIG
from physiclaw.common.logger import ensure_readme
from physiclaw.common.text import read_text, write_text

log = logging.getLogger(__name__)


def init_macro(name: str) -> Path:
    """Scaffold ``macros/<name>/MACRO.yml`` from the commented template
    and return its path. The scaffold parses clean (disabled) so `check`
    passes before any editing. Raises MacroError on a bad name or an
    existing directory."""
    check_name(name)
    d = paths.macros_dir() / name
    if d.exists():
        raise MacroError(f"macro directory already exists: {d}")
    d.mkdir(parents=True)
    md = d / MACRO_FILENAME
    write_text(md, scaffold.render_init(name))
    ensure_format_readme()
    return md


def ensure_format_readme() -> None:
    """Keep the format doc at ``macros/README.md`` current — the
    `ensure_readme` pattern (sessions dir precedent): rewritten whenever
    the shipped constant changed, fail-open, cheap. Called from the CLI's
    user-facing moments (init/list/check), not from the engine's scan."""
    ensure_readme(paths.macros_dir(), scaffold.README_CONTENT)


@dataclass(frozen=True)
class ScanEntry:
    """One macro directory as found on disk — either a parsed spec or the
    reason it was excluded. `dir_name` is always set (it's the identity)."""

    dir_name: str
    spec: Macro | None = None
    error: str | None = None


def list_dir_names() -> set[str]:
    """The macro directories on disk (a MACRO.yml present) — identity only,
    no read, no parse. The stats prune set: `run_and_record` needs just
    the names, so it must not pay `scan()`'s full-parse of every file."""
    root = paths.macros_dir()
    if not root.exists():
        return set()
    return {
        d.name
        for d in root.iterdir()
        if d.is_dir()
        and not d.name.startswith(("_", "."))
        and (d / MACRO_FILENAME).exists()
    }


def scan() -> list[ScanEntry]:
    """Every macro directory under ``macros_dir()``, sorted by name —
    valid or not. The CLI's view; the engine uses `discover_enabled`."""
    root = paths.macros_dir()
    if not root.exists():
        return []
    root_real = root.resolve()
    out: list[ScanEntry] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith(("_", ".")):
            continue
        real = d.resolve()
        if not real.is_relative_to(root_real):
            log.warning(
                "macro %s resolves outside %s; skipping (path traversal guard)",
                d.name,
                root,
            )
            out.append(ScanEntry(d.name, error="resolves outside the macros dir"))
            continue
        md = real / MACRO_FILENAME
        if not md.exists():
            out.append(ScanEntry(d.name, error=f"no {MACRO_FILENAME}"))
            continue
        try:
            out.append(ScanEntry(d.name, spec=parse_macro(read_text(md), d.name)))
        except Exception as e:
            # Deliberately broad: a malformed file must be excluded WHOLE, per
            # this module's contract, and the YAML loader does not confine
            # itself to YAMLError — deep nesting surfaces as RecursionError,
            # which used to escape all the way to `build_prompt_bundle` and
            # STUCK every wake. `BaseException` still propagates, so
            # KeyboardInterrupt and CancelledError are untouched.
            out.append(ScanEntry(d.name, error=str(e) or type(e).__name__))
    return out


def discover_enabled() -> dict[str, Macro]:
    """The macros the agent may run: the ``[macros] enabled`` config gate
    is on AND the file is valid AND not ``enabled: false``. This is the
    ONE agent-facing view — the prompt section, the run_macro tool, and
    the MACRO.md doctrine all key on this dict being non-empty, so an
    empty result means zero macro bytes in SYSTEM. A broken or
    still-in-authoring file skips silently (reason surfaced by
    `physiclaw macros check`, logged here) rather than taking down the
    session — same posture as skill discovery."""
    if not CONFIG.macros.enabled:
        return {}
    out: dict[str, Macro] = {}
    for entry in scan():
        if entry.error is not None:
            log.warning("macro %s excluded: %s", entry.dir_name, entry.error)
        elif entry.spec is not None and entry.spec.enabled:
            out[entry.spec.name] = entry.spec
    return out


def render_section(macros: dict[str, Macro]) -> str:
    """`## Available Macros` — name, description, and per-input lines so the
    model can fill inputs without loading anything. Byte-stable across wakes
    (files rarely change), so it sits in the cached SYSTEM prefix; run stats
    deliberately do NOT render here (they change every run and would churn
    the prefix — they're for the user, via `physiclaw macros stats`).
    Empty string when no macros are enabled."""
    if not macros:
        return ""
    lines = [
        "## Available Macros",
        "",
        # One line, not the semantics: MACRO.md always co-renders with this
        # section (prompt.py keys both on the same condition), so usage and
        # abort rules live there once — duplicating them here billed ~80
        # cached-prefix tokens per wake for nothing.
        "Rehearsed gesture macros — `run_macro(name, inputs)` replays one "
        "as a single step. Usage, `start_at`, and abort rules: MACRO "
        "doctrine above.",
        "",
    ]
    for spec in macros.values():
        lines.append(f"- **{spec.name}** — {spec.description}")
        for inp in spec.inputs:
            hint = "required" if inp.required else f"default: `{inp.default}`"
            line = f"  - `{inp.name}` ({hint}): {inp.description}"
            if inp.example:
                # Parenthesized, not appended prose: the description is
                # user-authored and may not end with punctuation, and
                # "…the chat Example: …" reads as one run-on sentence.
                line += f" (example: `{inp.example}`)"
            lines.append(line)
        # Every step is named and unique (parse enforces it), so this line
        # is both the step count and the set of valid `start_at` values.
        lines.append(f"  - steps: {', '.join(s.name for s in spec.steps)}")
    return "\n".join(lines)
