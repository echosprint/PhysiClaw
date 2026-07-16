"""Persistent per-step build cache for the procedure pipeline.

Three output kinds, two invalidation layers (their "subtly different logic"):

  * ``.step`` (geometry) and the raw ``.svg`` (render) share one **source
    key** — a content hash of the procedure's transitive ``hardware.*``
    import closure (its source + every part / predecessor it pulls in) plus
    the build123d version. The cumulative / delta BOM ``.md`` rides along:
    it is a pure function of the same parts. Change a part → source key
    changes → rebuild geometry, re-render, re-aggregate BOM.
  * the patch-snapshot ``.svg`` (the raw render with the mark-tool
    annotations composited in) carries an extra **patch key** on top: the
    stem's patch JSON(s) plus the replay engine's own source. Edit only a
    patch and the source key still matches — so we keep the cached ``.step``
    and raw ``.svg`` and just re-run the replay — no geometry rebuild, though
    the replay itself still imports build123d (via ``assembly.base``).

No stale cache, three ways:
  * stale HIT — the keys fold in every input file (import closure, kernel
    version, patch JSON, replay engine), so any change that could alter an
    output changes that output's key and forces a rebuild;
  * stale FILE — each stem keeps exactly one entry; ``store_source`` /
    ``store_snapshots`` replace their layer's files wholesale, and
    ``prune()`` drops entries for stems that no longer exist;
  * damaged ENTRY — each layer's key file is a manifest (key + each stored
    file's sha256), written last; a hit requires the key to match AND every
    listed file to be present with its recorded hash, so a partial entry
    (store killed mid-copy), a deleted file, or a corrupted/edited one all
    fall back to a rebuild instead of restoring a bad set. (Output files are
    reconstructed from the entry on every hit, so tampering there self-heals
    on the next build.)

The cache lives at ``hardware/output/.cache`` (git-ignored with the rest of
``output/``) and needs no build123d import — only the stdlib — so the
dispatcher can decide hit/miss before spawning a worker.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import re
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from hardware.scheme import (
    BOM_DIR,
    CACHE_DIR,
    MARK_DIR,
    PATCH_DIR,
    PROCEDURES_DIR,
    REPO_ROOT,
    STEP_DIR,
    SVG_DIR,
    VARIANTS,
    raw_svg_re,
    snapshot_svg_re,
)

_SOURCE_KEY = "source.key"
_PATCH_KEY = "patch.key"


# ── content hashing ──────────────────────────────────────────────────────────


def _module_to_path(mod: str) -> Path | None:
    """``hardware.a.b.c`` → ``<repo>/hardware/a/b/c.py`` (None if not a file)."""
    if not (mod == "hardware" or mod.startswith("hardware.")):
        return None
    p = REPO_ROOT / Path(*mod.split(".")).with_suffix(".py")
    return p if p.is_file() else None


def _imports(path: Path) -> set[str]:
    """Absolute ``hardware.*`` modules imported by ``path`` (AST, no exec)."""
    mods: set[str] = set()
    for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.startswith("hardware"):
                mods.add(node.module)
                # `from hardware.parts import custom` names the SUBMODULE
                # hardware/parts/custom.py — record the package-qualified
                # name per alias too. Non-module names (functions, classes)
                # resolve to no file in _module_to_path and drop out.
                mods.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            mods.update(a.name for a in node.names if a.name.startswith("hardware"))
    return mods


@functools.cache
def _closure(path: Path) -> frozenset[Path]:
    """All hardware source files reachable from ``path`` through imports."""
    seen: set[Path] = set()
    stack = [path]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        for mod in _imports(p):
            dep = _module_to_path(mod)
            if dep is not None and dep not in seen:
                stack.append(dep)
    return frozenset(seen)


@functools.cache
def _kernel_version() -> str:
    try:
        return version("build123d")
    except PackageNotFoundError:
        return "unknown"


def _hash_files(h: "hashlib._Hash", files) -> None:
    for f in sorted(files, key=lambda p: str(p)):
        h.update(str(f.relative_to(REPO_ROOT)).encode())
        h.update(b"\0")
        h.update(f.read_bytes() if f.is_file() else b"")
        h.update(b"\0")


def source_key(stem: str) -> str:
    """Content hash of ``stem``'s import closure + kernel version.
    Governs the ``.step``, the raw ``.svg``, and the BOM ``.md``."""
    h = hashlib.sha256()
    h.update(b"build123d=" + _kernel_version().encode() + b"\n")
    _hash_files(h, _closure(PROCEDURES_DIR / f"{stem}.py"))
    return h.hexdigest()


def _patch_files(stem: str) -> list[Path]:
    out: list[Path] = []
    for v in VARIANTS:
        out += PATCH_DIR.glob(f"{stem}_{v}_cam*.json")
    return out


def _replay_engine_files() -> list[Path]:
    """Replay-engine source — the ``mark/`` files reachable from ``replay.py``.
    Taken from the import closure (not a hardcoded list) so a new ``mark/``
    module the replay path grows is covered automatically; shared
    ``assembly/`` deps it pulls (base, svg_utils) already ride in every stem's
    ``source_key`` via ``base``, so only ``mark/`` files are kept here."""
    return [f for f in _closure(MARK_DIR / "replay.py") if MARK_DIR in f.parents]


def patch_key(stem: str) -> str:
    """Content hash of ``stem``'s patch JSON(s) + replay-engine source.
    Combined with ``source_key`` it governs the patch-snapshot ``.svg``s."""
    h = hashlib.sha256()
    _hash_files(h, _replay_engine_files())
    _hash_files(h, _patch_files(stem))
    return h.hexdigest()


# ── store / restore / prune ──────────────────────────────────────────────────


def _entry(stem: str) -> Path:
    return CACHE_DIR / stem


def _dest_dir(name: str) -> Path:
    return {".step": STEP_DIR, ".svg": SVG_DIR, ".md": BOM_DIR}[Path(name).suffix]


# Layer membership — one predicate per layer, shared by store (which old
# files to drop), restore (which to copy back), and the completeness checks.
def _is_source(stem: str, name: str, *, want_bom: bool = True) -> bool:
    """``name`` belongs to ``stem``'s source layer (.step / raw .svg / BOM)."""
    return (
        name.endswith(".step")
        or bool(raw_svg_re(stem).match(name))
        or (want_bom and name.endswith(".md"))
    )


def _is_snapshot(stem: str, name: str) -> bool:
    """``name`` belongs to ``stem``'s patch-snapshot layer."""
    return bool(snapshot_svg_re(stem).match(name))


def _svgs(stem: str, pat: re.Pattern) -> list[Path]:
    return [p for p in SVG_DIR.glob(f"{stem}_*_cam*.svg") if pat.match(p.name)]


def _source_outputs(stem: str) -> list[Path]:
    """``.step`` + raw ``.svg`` + BOM ``.md`` for ``stem`` currently on disk."""
    files = [p for v in VARIANTS for p in STEP_DIR.glob(f"{stem}_{v}.step")]
    files += _svgs(stem, raw_svg_re(stem))
    files += [
        p for p in (BOM_DIR / f"{stem}.md", BOM_DIR / f"{stem}_delta.md") if p.exists()
    ]
    return files


def _snapshot_outputs(stem: str) -> list[Path]:
    return _svgs(stem, snapshot_svg_re(stem))


def clear_outputs(stem: str) -> None:
    """Remove ``stem``'s output files (.step / .svg / .md), so a rebuild that
    now emits fewer cameras/variants doesn't leave a stale file behind.
    Sweeps the stem's in-progress .tmp files too (a SIGSEGV'd render or
    export can leave one)."""
    tmps = list(SVG_DIR.glob(f"{stem}_*_cam*.tmp"))
    tmps += STEP_DIR.glob(f"{stem}_*.tmp")
    for f in _source_outputs(stem) + _snapshot_outputs(stem) + tmps:
        f.unlink()


def _restore(files) -> None:
    for f in files:
        dest = _dest_dir(f.name)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest / f.name)


def _sha256_file(p: Path) -> str:
    """sha256 of a file's bytes, streamed (the .step files run to ~18 MB)."""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _store_layer(stem: str, *, key_file: str, key_value: str, belongs, outputs) -> None:
    """Replace one layer of ``stem``'s entry: drop its old files (the layer's
    key + everything it ``belongs`` to), copy in the current ``outputs``, then
    write the key file LAST as a manifest ``{"key", "files": {name: sha256}}``.
    The other layer is untouched.

    Writing the manifest last makes the layer self-describing and crash-safe;
    the per-file digests let a reader verify integrity. A hit demands the key
    match AND every listed file be present with its recorded hash — so a partial
    entry (store killed mid-copy), a deleted file, or a corrupted/edited one all
    read as a miss, never a hit that restores a bad set."""
    entry = _entry(stem)
    entry.mkdir(parents=True, exist_ok=True)
    for f in entry.iterdir():
        if f.name == key_file or belongs(f.name):
            f.unlink()
    files: dict[str, str] = {}
    for f in outputs:
        dest = entry / f.name
        shutil.copy2(f, dest)
        files[f.name] = _sha256_file(dest)
    (entry / key_file).write_text(json.dumps({"key": key_value, "files": files}))


def _read_layer(stem: str, key_file: str) -> tuple[str | None, dict[str, str]]:
    """``(stored key, {filename: sha256})`` for a layer, or ``(None, {})`` when
    the key file is missing or unreadable — a pre-manifest entry, or one a
    killed store left half-written. Both read as a miss."""
    p = _entry(stem) / key_file
    if not p.is_file():
        return None, {}
    try:
        m = json.loads(p.read_text())
        return m["key"], m["files"]
    except (ValueError, KeyError, TypeError):
        return None, {}


def _entry_intact(stem: str, files: dict[str, str]) -> bool:
    """Every manifested file is present in the entry with its recorded hash.
    The expensive check (it reads each stored file), so callers run the cheap
    key/manifest checks first and only reach here on a key match."""
    entry = _entry(stem)
    for name, sha in files.items():
        p = entry / name
        if not p.is_file() or _sha256_file(p) != sha:
            return False
    return True


def store_source(stem: str) -> None:
    """Replace ``stem``'s source layer (.step / raw .svg / .md) + source key."""
    _store_layer(
        stem,
        key_file=_SOURCE_KEY,
        key_value=source_key(stem),
        belongs=lambda n: _is_source(stem, n),
        outputs=_source_outputs(stem),
    )


def store_snapshots(stem: str) -> None:
    """Replace ``stem``'s snapshot layer (.svg op snapshots) + patch key."""
    _store_layer(
        stem,
        key_file=_PATCH_KEY,
        key_value=patch_key(stem),
        belongs=lambda n: _is_snapshot(stem, n),
        outputs=_snapshot_outputs(stem),
    )


def source_cached(stem: str, *, want_bom: bool) -> bool:
    """True only if the source key matches AND every recorded file is present
    with its recorded hash — a changed, damaged, incomplete, or corrupted entry
    all fall back to a rebuild. Cheap checks first; the integrity hash runs only
    on a key match (a rebuild from a source edit pays no hashing)."""
    key, files = _read_layer(stem, _SOURCE_KEY)
    if key != source_key(stem):
        return False
    has_step = any(n.endswith(".step") for n in files)
    has_raw = any(raw_svg_re(stem).match(n) for n in files)
    if not (has_step and has_raw):
        return False
    if want_bom and not any(n.endswith(".md") for n in files):
        return False
    return _entry_intact(stem, files)


def snapshots_cached(stem: str) -> bool:
    """True only if the patch key matches AND every recorded snapshot is present
    with its hash (a stem with no patches records an empty set — trivially true)."""
    key, files = _read_layer(stem, _PATCH_KEY)
    if key != patch_key(stem):
        return False
    return _entry_intact(stem, files)


def restore_source(stem: str, *, want_bom: bool = True) -> None:
    _restore(
        f for f in _entry(stem).iterdir() if _is_source(stem, f.name, want_bom=want_bom)
    )


def restore_snapshots(stem: str) -> None:
    _restore(f for f in _entry(stem).iterdir() if _is_snapshot(stem, f.name))


def prune(valid_stems) -> int:
    """Drop cache entries for stems not in ``valid_stems``. Returns count."""
    if not CACHE_DIR.is_dir():
        return 0
    valid = set(valid_stems)
    removed = 0
    for entry in CACHE_DIR.iterdir():
        if entry.is_dir() and entry.name not in valid:
            shutil.rmtree(entry)
            removed += 1
    return removed
