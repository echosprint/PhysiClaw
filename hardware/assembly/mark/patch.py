"""Op-id naming and the per-source patch JSON.

Each save is one *operation*. An op has:

* ``id``       — 4 random lowercase letters, unique within the source.
                  Never the literal ``"orig"`` (reserved).
* ``preop``    — the op this one is applied **on top of**: either the
                  literal ``"orig"`` (the source SVG) or another op's
                  4-letter id. Lets a replay reconstruct intermediate
                  snapshots by chaining ops.
* ``shapes``   — list of ``{type, geom, color, outlined}`` entries
                  (polygon / rect / circle / ellipse / line / arrow);
                  ``geom`` carries the type-specific coordinates in
                  source-SVG units.
* ``viewBox``  — ``"x y w h"`` if this op crops, else ``null``.

The output snapshot is ``<src stem>_<id>.svg`` next to the source; all
ops for one source share a single JSON accumulator at
``hardware/assembly/patch/<src stem>.json`` (top-level array of entries
in save order). The patch dir lives inside ``assembly/`` (not
``output/``) so it stays version-controlled."""

from __future__ import annotations

import json
import random
import re
import string
from pathlib import Path
from typing import Iterable

from hardware.assembly.base import SVG_DIR
from hardware.parts.base import REPO_ROOT

PATCH_DIR = REPO_ROOT / "hardware" / "assembly" / "patch"
ID_ALPHABET = string.ascii_lowercase
ID_LEN = 4
ORIG_SENTINEL = "orig"
ID_RE = re.compile(rf"^[{ID_ALPHABET}]{{{ID_LEN}}}$")


def snapshot_path(src: Path, op_id: str) -> Path:
    """``<src dir>/<src stem>_<op_id>.svg`` — the snapshot for one op.
    Uses ``_`` (not ``.``) before the op id so the filename has a
    single suffix; tools that key on ``Path.stem`` get the full
    ``<src stem>_<op_id>`` instead of stripping ``.<op_id>`` away."""
    return src.parent / f"{src.stem}_{op_id}.svg"


def new_id(src: Path, taken: set[str] | None = None) -> str:
    """Random 4-letter op id, never ``"orig"``, that doesn't collide
    with an existing snapshot or a caller-supplied set of taken ids
    (typically the ids already in the patch). With ``26**4 = 456 976``
    options the loop draws again on collision."""
    taken = taken or set()
    while True:
        op_id = "".join(random.choices(ID_ALPHABET, k=ID_LEN))
        if op_id == ORIG_SENTINEL or op_id in taken:
            continue
        if not snapshot_path(src, op_id).exists():
            return op_id


def validate_preop(raw) -> str:
    """Accept the literal ``"orig"`` or any 4 lowercase letters."""
    if raw == ORIG_SENTINEL or (isinstance(raw, str) and ID_RE.match(raw)):
        return raw
    raise ValueError(
        f"preop must be {ORIG_SENTINEL!r} or four lowercase letters; got {raw!r}"
    )


def patch_path(source_svg: Path) -> Path:
    """``hardware/assembly/patch/<src stem>.json`` — one accumulator per
    source SVG. A replay script walks this directory to find every
    edited source and rebuilds each op by chaining on its ``preop``."""
    return PATCH_DIR / f"{source_svg.stem}.json"


def source_for_patch(patch_file: Path) -> Path:
    """The source SVG a patch JSON annotates — the inverse of
    ``patch_path``: ``output/svg/<stem>.svg``."""
    return SVG_DIR / f"{patch_file.stem}.svg"


def load_patch(source_svg: Path) -> list[dict]:
    """Return the current patch entries (``[]`` if no file). Raises if
    the file exists but isn't a JSON array — refuses to silently
    discard a user's edit history."""
    path = patch_path(source_svg)
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text() or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(f"existing patch file isn't valid JSON: {exc}") from exc
    if not isinstance(loaded, list):
        raise ValueError(f"existing patch file is not a JSON array: {path}")
    return loaded


def write_patch(source_svg: Path, entries: list[dict]) -> Path:
    """Overwrite the patch file with ``entries``. The caller owns
    validation — this just persists what it's handed."""
    path = patch_path(source_svg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2))
    return path


def make_entry(
    op_id: str,
    preop: str,
    shapes: Iterable[dict],
    viewbox: str | None,
) -> dict:
    """Shape the in-memory op entry that gets persisted by
    ``write_patch``. ``shapes`` is ``[{type, geom, color, outlined},
    ...]`` — each shape carries its own colour + style (locked at
    draw time). Polygon vertex tuples are coerced to lists so the
    JSON round-trip is stable."""
    out = []
    for s in shapes:
        item = dict(s)
        if s["type"] == "polygon":
            # Deep-copy geom so the points list isn't aliased.
            item["geom"] = {
                "points": [list(pt) for pt in s["geom"]["points"]],
            }
        out.append(item)
    return {
        "id": op_id,
        "preop": preop,
        "shapes": out,
        "viewBox": viewbox,
    }


def upsert_entry(src, entries, edit_id, preop, shapes, viewbox):
    """Append a new op (``edit_id is None``) or replace the op with id
    ``edit_id`` in place. Returns ``(op_id, updated_entries)``. The caller
    validates ``edit_id`` (format / existence) first; this just mutates."""
    if edit_id is not None:
        entry = make_entry(edit_id, preop, shapes, viewbox)
        return edit_id, [entry if e["id"] == edit_id else e for e in entries]
    op_id = new_id(src, taken={e["id"] for e in entries})
    return op_id, [*entries, make_entry(op_id, preop, shapes, viewbox)]
