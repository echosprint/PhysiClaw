"""Macro discovery and the `## Available Macros` prompt section.

One file per macro under ``~/.physiclaw/macros/``: ``<name>.yml``, the
stem IS the identity and the file's ``name`` field must match (a macro
owns no other files, so it is a leaf, not a folder). Only ``*.yml`` is
scanned, so the machine-written ``stats.json`` and the format
``README.md`` at the root can never be read as a macro.

Discovery is realpath-scoped like `engine.skill`: a symlinked file that
escapes the root is rejected. A file failing validation is excluded whole
with its reason kept on the ScanEntry — `physiclaw macros check` prints
it; the engine only ever sees valid AND enabled specs.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from physiclaw.common import paths
from physiclaw.common.config import CONFIG
from physiclaw.common.logger import ensure_readme
from physiclaw.common.text import read_text, write_text
from physiclaw.macros import scaffold
from physiclaw.macros.model import (
    MACRO_SUFFIX,
    Macro,
    MacroError,
    check_name,
)
from physiclaw.macros.parse import parse_macro

log = logging.getLogger(__name__)


def macro_path(root: Path, name: str) -> Path:
    """Where macro ``name`` lives under ``root`` — the one spelling of
    the leaf-file rule."""
    return root / f"{name}{MACRO_SUFFIX}"


def init_macro(name: str, root: Path | None = None) -> Path:
    """Scaffold ``macros/<name>.yml`` from the commented template and
    return its path — under the home macros dir, or under ``root`` (a
    pack's or a playbook's ``macros/``). The scaffold parses clean
    (disabled) so `check` passes before any editing. Raises MacroError
    on a bad name or an existing file."""
    check_name(name)
    md = macro_path(paths.macros_dir() if root is None else root, name)
    if md.exists():
        raise MacroError(f"macro file already exists: {md}")
    md.parent.mkdir(parents=True, exist_ok=True)
    write_text(md, scaffold.render_init(name))
    if root is None:
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
    """One macro file as found on disk — either a parsed spec or the
    reason it was excluded. `name` (the file stem) is always set."""

    name: str
    spec: Macro | None = None
    error: str | None = None


def macro_files(root: Path) -> list[Path]:
    """The macro files under ``root``, sorted — `paths.leaf_files` with
    the macro suffix."""
    return paths.leaf_files(root, MACRO_SUFFIX)


def list_names() -> set[str]:
    """The macro names on disk across the search path — identity only,
    no read, no parse. The stats prune set: `run_and_record` needs just
    the names, so it must not pay `scan()`'s full parse of every file."""
    return {p.stem for root in paths.macros_dirs() for p in macro_files(root)}


def scan(root: Path | None = None) -> list[ScanEntry]:
    """Every macro file, sorted by name — valid or not. With no ``root``,
    unions the search path (first dir wins per name — the
    `paths.playbooks_dirs` layering rule). The CLI's view; the engine
    uses `discover_enabled`. Conductor packs point this at their private
    ``macros/`` roots — one scanner, one traversal guard, one
    broad-except lesson."""
    if root is None:
        seen: set[str] = set()
        merged: list[ScanEntry] = []
        for r in paths.macros_dirs():
            for e in scan(r):
                if e.name not in seen:
                    seen.add(e.name)
                    merged.append(e)
        return sorted(merged, key=lambda e: e.name)
    files = macro_files(root)
    if not files:
        return []
    root_real = root.resolve()
    out: list[ScanEntry] = []
    for md in files:
        real = md.resolve()
        if not real.is_relative_to(root_real):
            log.warning(
                "macro %s resolves outside %s; skipping (path traversal guard)",
                md.stem,
                root,
            )
            out.append(ScanEntry(md.stem, error="resolves outside the macros dir"))
            continue
        try:
            out.append(ScanEntry(md.stem, spec=parse_macro(read_text(real), md.stem)))
        except Exception as e:
            # Deliberately broad: a malformed file must be excluded WHOLE, per
            # this module's contract, and the YAML loader does not confine
            # itself to YAMLError — deep nesting surfaces as RecursionError,
            # which used to escape all the way to `build_prompt_bundle` and
            # STUCK every wake. `BaseException` still propagates, so
            # KeyboardInterrupt and CancelledError are untouched.
            out.append(ScanEntry(md.stem, error=str(e) or type(e).__name__))
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
            log.warning("macro %s excluded: %s", entry.name, entry.error)
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
        # Every step's handle: the count, what each step does, and the
        # exact `start_at` values — one list, three answers.
        lines.append(f"  - steps: {', '.join(s.name for s in spec.steps)}")
    return "\n".join(lines)
