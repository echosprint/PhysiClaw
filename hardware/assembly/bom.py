"""BOM (bill of materials) for assembly steps — library.

Each ``BasePart`` instance pushes one entry (``bom_key``, ``qty``,
``category``) into a process-wide ``BOM_REGISTRY`` (in
``hardware.parts.base``) the first time it builds — memo-guarded, so
re-builds of the same instance don't double-push. To get the BOM for a
procedure step, ``collect()`` clears the registry, drives the step's
build (which transitively builds every prior step in the chain), then
aggregates the registry by ``bom_key``.

Predecessor for a delta is found by convention: same family prefix
(``belt_`` / ``frame_`` / …), next-lower ``NN`` index in the filename.

``write_bom`` is the public entry point: ``hardware.assembly.build_procedures``
calls it during a render pass so one build also emits the Markdown BOM to
``output/bom/`` (cheap there because the leaf-part cache is already warm).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from hardware.assembly.base import BaseAssembly
from hardware.assembly.dispatch import load_step
from hardware.parts.base import BOM_REGISTRY, BomCategory
from hardware.scheme import BOM_DIR, PROCEDURES_DIR, STEM_RE


@dataclass(frozen=True, order=True)
class BomEntry:
    category: BomCategory
    key: tuple
    qty: int
    custom_display: str | None = None

    @property
    def display(self) -> str:
        """e.g. ('Screw', 'SHCS', 'M6', 16) -> 'Screw SHCS M6×16'. If the
        part class provided a ``bom_display`` override, use that instead."""
        if self.custom_display is not None:
            return self.custom_display
        head, *rest = self.key
        out = head
        for item in rest:
            out += f"×{item:g}" if isinstance(item, (int, float)) else f" {item}"
        return out


# ── Collection ────────────────────────────────────────────────────────────────


def collect(step: BaseAssembly) -> list[BomEntry]:
    """Build the step and aggregate its BOM. Returns a sorted list.

    Clears the registry up front (and the step's memo) so a re-collect
    rebuilds cleanly. Every ``BasePart.build()`` in the chain pushes one
    entry; we just aggregate by ``bom_key``."""
    BOM_REGISTRY.clear()
    if hasattr(step, "_built"):
        delattr(step, "_built")
    step.build()

    sums: dict[tuple, int] = defaultdict(int)
    cats: dict[tuple, str] = {}
    displays: dict[tuple, str | None] = {}
    for key, qty, category, display in BOM_REGISTRY:
        sums[key] += qty
        cats[key] = category
        displays[key] = display

    entries = [BomEntry(cats[k], k, q, displays[k]) for k, q in sums.items()]
    entries.sort()
    return entries


# ── Delta against a predecessor step ──────────────────────────────────────────


def delta(current: list[BomEntry], previous: list[BomEntry]) -> list[BomEntry]:
    """Entries added at the current step that weren't at the previous
    step. Quantity is the cumulative delta (current − previous, omitting
    zero/negative deltas). Categories follow the current step."""
    prev_qty = {e.key: e.qty for e in previous}
    out: list[BomEntry] = []
    for e in current:
        diff = e.qty - prev_qty.get(e.key, 0)
        if diff > 0:
            out.append(BomEntry(e.category, e.key, diff, e.custom_display))
    out.sort()
    return out


# ── Predecessor discovery ────────────────────────────────────────────────────


def predecessor_module(module_name: str) -> str | None:
    """Return the predecessor procedure's module name (e.g.
    'hardware.assembly.procedures.belt_10_motor_a') or None if this is
    the first step in its family.

    Convention: predecessor = same family prefix, largest NN strictly
    less than this file's NN. Cross-family chains aren't auto-resolved."""
    stem = module_name.rsplit(".", 1)[-1]
    m = STEM_RE.match(stem)
    if not m:
        return None
    family, nn = m["family"], int(m["nn"])
    candidates: list[tuple[int, str]] = []
    for path in PROCEDURES_DIR.glob(f"{family}_*.py"):
        cm = STEM_RE.match(path.stem)
        if not cm:
            continue
        cnn = int(cm["nn"])
        if cnn < nn:
            candidates.append((cnn, path.stem))
    if not candidates:
        return None
    candidates.sort()
    prior_stem = candidates[-1][1]
    return f"hardware.assembly.procedures.{prior_stem}"


# ── Markdown writer ───────────────────────────────────────────────────────────


def to_markdown(entries: list[BomEntry], *, title: str) -> str:
    standard = [e for e in entries if e.category == "standard"]
    custom = [e for e in entries if e.category == "custom"]
    lines = [f"# {title}", ""]
    lines += _section("Standard parts", standard)
    lines += _section("Custom parts", custom)
    return "\n".join(lines).rstrip() + "\n"


def _section(heading: str, entries: list[BomEntry]) -> list[str]:
    if not entries:
        return []
    out = [f"## {heading}", "", "| Qty | Part |", "|----:|------|"]
    for e in entries:
        out.append(f"| {e.qty} | {e.display} |")
    out.append("")
    return out


# ── Public entry point ────────────────────────────────────────────────────────


def write_bom(
    step_name: str, *, cumulative: bool = True, want_delta: bool = False
) -> None:
    """Build ``step_name`` and write its BOM Markdown to ``BOM_DIR`` —
    cumulative and/or delta-vs-predecessor.

    Reused by ``hardware.assembly.build_procedures`` so one render pass can
    also emit BOMs; the build is cheap there because the leaf-part cache is
    already warm. Delta is skipped silently for the first step in a family
    (no predecessor)."""
    current = collect(load_step(step_name))
    BOM_DIR.mkdir(parents=True, exist_ok=True)
    if cumulative:
        text = to_markdown(current, title=f"{step_name} — cumulative BOM")
        (BOM_DIR / f"{step_name}.md").write_text(text)
    if want_delta:
        prev_name = predecessor_module(step_name)
        if prev_name is not None:
            entries = delta(current, collect(load_step(prev_name)))
            title = f"{step_name} — delta vs. {prev_name.rsplit('.', 1)[-1]}"
            (BOM_DIR / f"{step_name}_delta.md").write_text(
                to_markdown(entries, title=title)
            )
