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
import re
from collections.abc import Callable
from itertools import groupby

from hardware.assembly.base import BaseAssembly
from hardware.parts.base import REPO_ROOT

PROCEDURES_DIR = REPO_ROOT / "hardware" / "assembly" / "procedures"

# Filename convention: <family>_<NN>_<descriptor>.py (e.g. belt_20_clamp).
_STEM_RE = re.compile(r"^(?P<family>[a-z]+)_(?P<nn>\d+)_(?P<descriptor>.+)$")

# Dependency-order families — lower index built first. Clustering each batch
# within a family lets the in-batch geometry cache reuse the shared chain.
_FAMILIES = (
    "fastener",
    "frame",
    "idler",
    "motor",
    "linear",
    "belt",
    "tapz",
    "phone",
    "board",
    "camera",
    "wire",
)
_FAMILY_PRIORITY = {name: i for i, name in enumerate(_FAMILIES)}
DEFAULT_BATCH_SIZE = 5
# Exact-HLR crashes hit up to ~80% per attempt on the worst stem
# (board_30_pcb assembled). A failed retry costs ~15s — only the missing
# outputs re-run — so a deep budget is cheap; 12 tries keeps the residual
# per-stem failure rate near 1%.
MAX_STEM_RETRIES = 12


# ── Discovery & loading ───────────────────────────────────────────────────────


def list_procedures() -> list[str]:
    """All procedure module stems, sorted by family then NN."""
    keyed = []
    for path in PROCEDURES_DIR.glob("*.py"):
        m = _STEM_RE.match(path.stem)
        if not m or path.stem.startswith("_"):
            continue
        keyed.append((m["family"], int(m["nn"]), path.stem))
    keyed.sort()
    return [stem for _, _, stem in keyed]


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


# ── Ordering & batching ───────────────────────────────────────────────────────


def _family_of(stem: str) -> str:
    return stem.split("_", 1)[0]


def _ordered_stems() -> list[str]:
    """Procedure stems in dependency-order family, then NN. list_procedures
    already sorts by (family, NN), but alphabetical family order isn't the
    dependency/build order — so re-sort by _FAMILY_PRIORITY."""
    fallback = len(_FAMILY_PRIORITY)
    return sorted(
        list_procedures(),
        key=lambda s: (_FAMILY_PRIORITY.get(_family_of(s), fallback), s),
    )


def _batches(batch_size: int) -> list[list[str]]:
    """Family-clustered chunks of up to ``batch_size``, dependency order."""
    out: list[list[str]] = []
    for _, group in groupby(_ordered_stems(), key=_family_of):
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
