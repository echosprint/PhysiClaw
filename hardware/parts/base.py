import copy
import sys
from pathlib import Path
from typing import Literal

from build123d import export_step

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STEP_DIR = REPO_ROOT / "hardware" / "output" / "step"

BomCategory = Literal["standard", "custom"]

# Append-only registry; collect() clears + reaggregates. A list rather
# than a dict keyed by id(shape) because shape transforms (.move /
# .rotate) return new objects — id-lookups would silently drop any
# transformed part.
BOM_REGISTRY: list[tuple[tuple, int, BomCategory, str | None]] = []

# Geometry cache: geom_key → (pristine _build() result, post-build
# state dict). `build()` returns copy.copy() of the shape (shares OCCT
# TShape; fresh Python wrapper so anytree re-parenting works). State
# carries post-build instance attrs downstream consumers read. Unbounded;
# subprocess batching in build_procedures bounds total memory.
_BUILD_CACHE: dict[tuple, tuple[object, dict]] = {}


class BasePart:
    """Base class for parts. Subclasses implement `_build()` to return the
    build123d shape; calling `build()` returns the labeled shape, `export()`
    writes a STEP file to `output_path()`.

    Default filename: `<repo>/hardware/output/step/<module_name>_x<qty>.step`,
    derived from the defining file's basename — so `python -m` invocation
    produces the right name instead of `__main__.step`. Subclasses with extra
    naming dimensions (e.g. length) override `name_suffix()`.

    BOM identity: ``bom_key()`` returns a hashable spec tuple (default the
    class name); ``bom_category`` is "standard" (purchasable) or "custom"
    (manufactured). Override ``bom_key`` on subclasses with parameters
    (Screw, TNut, …) and use ``BaseCustomPart`` for manufactured parts
    instead of setting ``bom_category`` by hand. Assemblies opt out by
    returning None from ``bom_key``."""

    bom_category: BomCategory = "standard"

    def __init__(self, qty: int = 1):
        self.qty = qty

    @classmethod
    def _module_stem(cls) -> str:
        return Path(sys.modules[cls.__module__].__file__).stem

    def name_suffix(self) -> str:
        return f"_x{self.qty}"

    def output_path(self) -> Path:
        return STEP_DIR / f"{self._module_stem()}{self.name_suffix()}.step"

    def bom_key(self):
        """Hashable identity used to aggregate this part in the BOM.

        Default uses the class name alone — fine for zero-arg parts
        (Bumper, Tip). Override to include parameters that distinguish
        instances (size, length, kind). Return None to exclude this part
        from the BOM (assemblies do this)."""
        return (type(self).__name__,)

    def bom_display(self) -> str | None:
        """Optional human-readable label for the BOM table. Return None
        to use the default algorithm in ``hardware.assembly.bom.BomEntry``
        (head + space-joined strings + ×-prefixed numbers). Override
        when that default doesn't read naturally for a specific part."""
        return None

    def geom_key(self):
        """Identity for the SHAPE (geometry-cache key). Default returns
        ``None`` — no caching, no implicit aliasing of ``bom_key``.

        Standard / custom parts opt in to caching by inheriting from
        ``BaseStandardPart`` / ``BaseCustomPart``, which alias
        ``geom_key`` to ``bom_key`` (correct when every BOM-distinct
        instance also produces a unique shape — the common case).
        Parts whose geometry varies beyond what ``bom_key`` captures
        (e.g. MGN9H ``slider_position``, Belt ``path``) override
        ``geom_key`` directly with their own tuple."""
        return None

    def _build(self):
        raise NotImplementedError

    def _snapshot_state(self):
        """Capture instance state before `_build()` runs. Override to
        opt into post-build state caching (assemblies do this).
        Default: no state."""
        return None

    def _diff_state(self, snapshot):
        """Return the state dict to cache. Receives whatever
        `_snapshot_state` returned."""
        return {}

    def _restore_state(self, state):
        """Apply cached state to `self` on a cache hit."""
        pass

    def build(self):
        """Return the build123d shape (labeled, no STEP export).

        Memoized on the instance via ``self._built``. Leaf parts
        (``BaseStandardPart`` / ``BaseCustomPart``) also reuse a
        process-wide ``_BUILD_CACHE`` keyed by ``geom_key()``: the first
        instance of each unique geometry runs ``_build()``; subsequent
        instances return ``copy.copy()`` of the cached pristine shape.
        That copy is cheap relative to rebuilding an expensive leaf, but
        it is a full geometry deepcopy (build123d's ``copy.copy`` ⇒
        ``copy.deepcopy``), so assemblies opt OUT (``geom_key`` → None)
        and are recomposed from their cached leaves instead of cached and
        deep-copied. The ``_snapshot_state`` / ``_diff_state`` /
        ``_restore_state`` hooks below let a caching subclass carry
        post-build instance attrs across a cache hit; no subclass
        currently needs them.
        """
        if (cached := getattr(self, "_built", None)) is not None:
            return cached
        gkey = self.geom_key()
        if gkey is None:
            shape = self._build()
        elif (entry := _BUILD_CACHE.get(gkey)) is not None:
            cached_shape, cached_state = entry
            shape = copy.copy(cached_shape)
            self._restore_state(cached_state)
        else:
            snap = self._snapshot_state()
            shape = self._build()
            _BUILD_CACHE[gkey] = (shape, self._diff_state(snap))
            shape = copy.copy(shape)
        if not getattr(shape, "label", None):
            shape.label = type(self).__name__
        self._built = shape
        key = self.bom_key()
        if key is not None:
            BOM_REGISTRY.append((key, self.qty, self.bom_category, self.bom_display()))
        return shape

    def export(self):
        shape = self.build()
        path = self.output_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write via a temp name and rename into place so the final path
        # only ever holds a complete STEP — the build pipeline's crash
        # retries trust an existing file as done.
        tmp = path.with_suffix(".tmp")
        export_step(shape, str(tmp))
        tmp.replace(path)
        return shape


class BaseStandardPart(BasePart):
    """Off-the-shelf purchasable parts. Opts into geometry caching by
    aliasing ``geom_key`` to ``bom_key`` — appropriate when every
    BOM-distinct instance also produces a unique shape (the common
    case). Parts that need a different cache key override ``geom_key``
    in the subclass."""

    def geom_key(self):
        return self.bom_key()


class BaseCustomPart(BasePart):
    """Parts manufactured for this build (printed / machined). BOM
    aggregation tags them as ``custom``; geometry caching opted in the
    same way as ``BaseStandardPart``."""

    bom_category: BomCategory = "custom"

    def geom_key(self):
        return self.bom_key()


def export_all(parts):
    for part in parts:
        part.export()
        print(f"  exported {type(part).__name__:20s} → {part.output_path()}")
