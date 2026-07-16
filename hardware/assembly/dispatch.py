"""Procedure-build orchestration shared by the two drivers — the render
pipeline (``build_procedures``) and the BOM generator (``bom``).

Owns the bits that are about *the set of procedures and how to run them*,
not about rendering or bills of materials: discovering the procedure
modules, loading their assemblies, ordering/batching them by family, and
re-running crashed batches. Kept here (rather than in either driver) so the
two stay independent of each other; it depends only on the assembly/parts
base, so there's no import cycle.

A worker can crash (SIGSEGV) mid-batch: an intermittent OCCT hazard
(geometry kernel, or exact HLR in the render pipeline), not an OOM. It's
nondeterministic, so re-running a crashed stem in a fresh process
eventually succeeds — but the worst stems fail most attempts, hence
``retry_stems`` and the deep ``MAX_STEM_RETRIES`` budget.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from itertools import groupby

from hardware.assembly.base import BaseAssembly
from hardware.scheme import (
    FAMILY_PRIORITY,
    PROCEDURES_DIR,
    STEM_CONVENTION,
    STEM_RE,
    family_of,
)

DEFAULT_BATCH_SIZE = 5
# Exact-HLR crashes hit up to ~80% per attempt on the worst stem
# (board_30_pcb assembled). A failed retry costs ~15s — only the missing
# outputs re-run — so a deep budget is cheap; 12 tries keeps the residual
# per-stem failure rate near 1%.
MAX_STEM_RETRIES = 12


# ── Discovery & loading ───────────────────────────────────────────────────────


def list_procedures() -> list[str]:
    """All procedure module stems, in build order: family dependency
    order (``FAMILIES``), then NN within the family.

    Fails fast on a ``.py`` that doesn't match the stem convention or
    names an unknown family — either would otherwise be silently skipped
    (a typo'd stem never builds) or silently ordered last (an unlisted
    family). Underscore-prefixed files are non-procedures by convention
    and stay exempt."""
    keyed = []
    bad: list[str] = []
    for path in PROCEDURES_DIR.glob("*.py"):
        if path.stem.startswith("_"):
            continue
        m = STEM_RE.match(path.stem)
        if not m or m["family"] not in FAMILY_PRIORITY:
            bad.append(path.name)
            continue
        keyed.append((FAMILY_PRIORITY[m["family"]], int(m["nn"]), path.stem))
    if bad:
        raise ValueError(
            f"procedure file(s) {sorted(bad)} don't match "
            f"{STEM_CONVENTION}.py with a family from "
            f"hardware.scheme.FAMILIES; rename them (or prefix with '_' "
            f"to exclude)"
        )
    keyed.sort()
    return [stem for *_, stem in keyed]


def load_class(module_name: str) -> type[BaseAssembly]:
    """Import a procedure module and return its BaseAssembly subclass.
    The procedure files each define exactly one such class."""
    if "." not in module_name:
        module_name = f"hardware.assembly.procedures.{module_name}"
    mod = importlib.import_module(module_name)
    for obj in vars(mod).values():
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseAssembly)
            and obj is not BaseAssembly
            and obj.__module__ == mod.__name__
        ):
            return obj
    raise LookupError(f"No BaseAssembly subclass in {module_name}")


def load_step(module_name: str) -> BaseAssembly:
    """Instantiate a procedure's BaseAssembly subclass (assembled variant)."""
    return load_class(module_name)(exploded=False)


# ── Batching ──────────────────────────────────────────────────────────────────


def _batches(batch_size: int) -> list[list[str]]:
    """Family-clustered chunks of up to ``batch_size``, dependency order."""
    out: list[list[str]] = []
    for _, group in groupby(list_procedures(), key=family_of):
        stems = list(group)
        for i in range(0, len(stems), batch_size):
            out.append(stems[i : i + batch_size])
    return out


# ── Crash-retry ────────────────────────────────────────────────────────────────


def retry_stems(
    stems: list[str],
    *,
    run: Callable[[str], object],
    done: Callable[[str], bool],
    log: Callable[[str, int], object],
) -> None:
    """Re-run each stem in a fresh process (``run(stem)``) until it
    completes (``done(stem)`` is true) or ``MAX_STEM_RETRIES`` is exhausted,
    calling ``log(stem, attempt)`` before each attempt."""
    for stem in stems:
        for attempt in range(1, MAX_STEM_RETRIES + 1):
            log(stem, attempt)
            run(stem)
            if done(stem):
                break
