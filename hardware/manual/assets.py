"""Image-asset strategies for the manual build.

Resolve an SVG name to an ``<img src>`` and emit whatever files the chosen
build mode needs. Images stay ``<img>`` (not inline ``<svg>``) so the figure
framing can rely on object-fit / object-position.
"""

from __future__ import annotations

import base64
import shutil
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Protocol

from hardware.manual import BuildError
from hardware.manual.icon_svg import CRAB_SVG, WIRE_SPLICE_SVG
from hardware.scheme import SVG_DIR, SVG_NAME_RE

ASSETS_SUBDIR = "assets"  # where external mode writes images, relative to the HTML.

# Hand-authored figures kept as constants in icon_svg.py (so they are tracked,
# unlike the generated output/svg renders). build() writes them into the SVG
# pool before rendering, so the figure pipeline treats them like any render and
# the file stays a single-source copy of the constant.
HAND_FIGURES = {"wire_splice.svg": WIRE_SPLICE_SVG}


def _render_stem(basename: str) -> str:
    """The procedure stem behind a render filename, e.g.
    ``camera_40_frame_assembled_cam0_ljek.svg`` -> ``camera_40_frame``.
    A name that isn't a render (e.g. a hand figure) passes through whole."""
    m = SVG_NAME_RE.match(basename)
    return m["stem"] if m else basename


def _require_renders(names) -> None:
    """Raise a clear ``BuildError`` if any referenced SVG render is absent from
    output/svg. Those files come from the assembly render pipeline (not this
    package), so the message names exactly what's missing and the command that
    regenerates precisely those stems."""
    missing = sorted(n for n in names if not (SVG_DIR / n).exists())
    if not missing:
        return
    stems = sorted({_render_stem(n) for n in missing})
    listed = "\n".join(f"    {n}" for n in missing)
    raise BuildError(
        f"{len(missing)} referenced SVG render(s) missing from output/svg:\n"
        f"{listed}\n\n"
        "These are produced by the assembly render pipeline, not build_manual.\n"
        "Regenerate them with:\n"
        f"    uv run --group cad python -m hardware.assembly.build_procedures "
        f"--stems {' '.join(stems)}"
    )


@cache
def _read_svg(basename: str) -> str:
    """Read one SVG render from output/svg (cached — several are reused)."""
    _require_renders([basename])
    return (SVG_DIR / basename).read_text(encoding="utf-8")


def _data_uri(svg_text: str) -> str:
    """Encode SVG markup as a base64 ``data:image/svg+xml`` URI."""
    b64 = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


class Assets(Protocol):
    """Maps image references to ``<img src>`` values for a build mode."""

    def figure(self, basename: str) -> str: ...
    @property
    def crab(self) -> str: ...
    def preload(self, src: str) -> str: ...  # optional <head> preload for src
    def emit(self, out_dir: Path) -> None: ...  # write any external files


@dataclass
class InlineAssets:
    """Embed every image directly in the HTML as a ``data:`` URI."""

    def figure(self, basename: str) -> str:
        return _data_uri(_read_svg(basename))

    @property
    def crab(self) -> str:
        return _data_uri(CRAB_SVG)

    def preload(self, src: str) -> str:
        return ""  # nothing to preload — the bytes are already in the document.

    def emit(self, out_dir: Path) -> None:
        pass  # self-contained: no sidecar files.


@dataclass
class ExternalAssets:
    """Write images as sidecar files and reference them with relative URLs."""

    referenced: set[str] = field(default_factory=set)

    def figure(self, basename: str) -> str:
        self.referenced.add(basename)
        return f"{ASSETS_SUBDIR}/{basename}"

    @property
    def crab(self) -> str:
        return f"{ASSETS_SUBDIR}/crab.svg"

    def preload(self, src: str) -> str:
        return f'<link rel="preload" as="image" href="{src}">'

    def emit(self, out_dir: Path) -> None:
        assets_dir = out_dir / ASSETS_SUBDIR
        assets_dir.mkdir(parents=True, exist_ok=True)
        (assets_dir / "crab.svg").write_text(CRAB_SVG, encoding="utf-8")
        _require_renders(self.referenced)  # all missing at once, with a fix hint
        for name in sorted(self.referenced):
            shutil.copyfile(SVG_DIR / name, assets_dir / name)
