#!/usr/bin/env python3
"""Render the bilingual PhysiClaw assembly manual from JSON content files.

The manual's content lives as a sequence of ``content/*.json`` files, each a
list of *page* dicts in visual (DOM) order. This script turns those pages into
one HTML document per language, laid out by ``styles.css`` (always inlined).

Image assets (the SVG renders + the crab logo) are handled by one of two
strategies, chosen with ``--assets``:

- ``external`` (default) — best for the web. SVGs are written as separate files
  under ``assets/`` and referenced with ``loading="lazy" decoding="async"`` so
  the browser fetches only what scrolls into view, in parallel, and caches each
  file. The above-the-fold cover render loads eagerly and is preloaded for a
  fast LCP. The HTML itself stays tiny (~100 KB).
- ``inline`` — a single self-contained file. Every image is embedded as a
  base64 ``data:`` URI (no external requests, no lazy loading — best for
  offline use / emailing one file around, at the cost of a large document).

Run under ``uv`` from the repo root (standard library only, Python 3.12+);
all paths resolve relative to this file, so the cwd does not matter::

    uv run hardware/manual/build_manual.py                 # en + zh, external assets, HTML only
    uv run hardware/manual/build_manual.py --pdf            # also render a PDF per language
    uv run hardware/manual/build_manual.py --lang en        # English only -> physiclaw_manual.html
    uv run hardware/manual/build_manual.py --assets inline  # single self-contained file
    uv run hardware/manual/build_manual.py --out /tmp/out   # custom output directory

PDF output uses an already-installed Chromium-family browser (Chrome / Chromium
/ Edge); if none is found the HTML still builds and PDF is skipped with a note.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Callable, Protocol

from icon_svg import (
    BACK_CORNER_SVG,
    COVER_STRIPES_SVG,
    CRAB_SVG,
    GITHUB_OCTICON_SVG,
    INFO_ICON_SVG,
    WIRE_SPLICE_SVG,
)

# Note "icon" keys -> inline SVG markup, for the leading icon of a flex callout.
NOTE_ICONS = {"info": INFO_ICON_SVG}

# Hand-authored figures kept as constants in icon_svg.py (so they are tracked,
# unlike the generated output/svg renders). build() writes them into the SVG
# pool before rendering, so the figure pipeline treats them like any render and
# the file stays a single-source copy of the constant.
HAND_FIGURES = {"wire_splice.svg": WIRE_SPLICE_SVG}

# --------------------------------------------------------------------------- #
# Paths — everything is resolved relative to this file so cwd does not matter.
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
CONTENT_DIR = SCRIPT_DIR / "content"
STYLES_CSS = SCRIPT_DIR / "styles.css"
SVG_DIR = SCRIPT_DIR / ".." / "output" / "svg"  # source SVG renders.
OUTPUT_DIR = SCRIPT_DIR / ".." / "output" / "manual"
ASSETS_SUBDIR = "assets"  # where external mode writes images, relative to the HTML.

# Output filename per language, plus the <html lang> attribute value.
LANG_FILENAME = {"en": "physiclaw_manual.html", "zh": "physiclaw装配手册.html"}
HTML_LANG = {"en": "en", "zh": "zh-Hans"}

# Inline-SVG chrome (cover stripes, GitHub octicon, back corner, crab logo)
# lives in icon_svg.py — imported above.
URL_MARK = "PhysiClaw.ai"  # masthead mark, never localized.


# --------------------------------------------------------------------------- #
# Assets — resolve an SVG name to an <img src> and emit whatever files the
# chosen strategy needs. Images stay <img> (not inline <svg>) so the figure
# framing can rely on object-fit / object-position.
# --------------------------------------------------------------------------- #
class BuildError(Exception):
    """A user-facing build failure: ``main()`` prints the message on its own,
    without a Python traceback."""


def _render_stem(basename: str) -> str:
    """The procedure stem behind a render filename, e.g.
    ``camera_40_frame_assembled_cam0_ljek.svg`` -> ``camera_40_frame``."""
    return re.sub(r"_(?:exploded|assembled)_cam\d+(?:_[a-z0-9]+)?\.svg$", "", basename)


def _require_renders(names) -> None:
    """Raise a clear ``BuildError`` if any referenced SVG render is absent from
    output/svg. Those files come from the assembly render pipeline (not this
    script), so the message names exactly what's missing and the command that
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


@dataclass(frozen=True)
class Ctx:
    """Per-render context: the active language and the asset strategy."""

    lang: str
    assets: Assets


# --------------------------------------------------------------------------- #
# Localization
# --------------------------------------------------------------------------- #
def loc(value: Any, lang: str) -> str:
    """Resolve a localized value.

    Localized text is ``{"en": ..., "zh": ...}``; this returns the requested
    language, falling back to English when the translation is empty. Plain
    strings (specs, ids, URLs) pass straight through. Localized strings are
    trusted HTML and are emitted raw — the source embeds inline ``<span>``/``<a>``.
    """
    if isinstance(value, dict):
        return value.get(lang) or value.get("en", "")
    return value


# --------------------------------------------------------------------------- #
# Shared sub-renderers
# --------------------------------------------------------------------------- #
def render_note(note: dict, ctx: Ctx) -> str:
    """Render a `.note` block (plain, or an absolute overlay card).

    ``classes`` and the optional ``style`` are emitted verbatim; the body may
    contain its own inline markup (``<p>`` wrappers, ``<a>``, ``<span>``).

    A note with an ``icon`` is a flex callout: a leading SVG icon sits beside a
    ``<div>`` wrapping the heading + body (so the title and text stack, rather
    than becoming separate flex items).
    """
    style = f' style="{note["style"]}"' if note.get("style") else ""
    h3_style = f' style="{note["h3Style"]}"' if note.get("h3Style") else ""
    h3_class = f' class="{note["h3Class"]}"' if note.get("h3Class") else ""
    body = loc(note["body"], ctx.lang)
    # A body already wrapped in block tags (e.g. <p>…</p>) is inserted as-is;
    # a bare run of text gets a single <p> wrapper to match the source markup.
    body_html = body if body.lstrip().startswith("<") else f"<p>{body}</p>"
    heading = f"<h3{h3_class}{h3_style}>{loc(note['h3'], ctx.lang)}</h3>{body_html}"
    inner = (
        f"{NOTE_ICONS[note['icon']]}<div>{heading}</div>"
        if note.get("icon")
        else heading
    )
    return f'<div class="{note["classes"]}"{style}>{inner}</div>'


def render_bom(bom: dict, ctx: Ctx) -> str:
    """Render the bill-of-materials overlay table."""
    rows = "".join(
        f"<tr><td>{loc(r['component'], ctx.lang)}</td>"
        f'<td class="spec">{loc(r["spec"], ctx.lang)}</td>'
        f'<td class="qty">{r["qty"]}</td></tr>'
        for r in bom["rows"]
    )
    headers = [
        {"en": "Component", "zh": "组件"},
        {"en": "Spec", "zh": "规格"},
        {"en": "Qty", "zh": "数量"},
    ]
    head_cells = "".join(f"<th>{loc(h, ctx.lang)}</th>" for h in headers)
    style = f' style="{bom["style"]}"' if bom.get("style") else ""
    return (
        f'<div class="bom"{style}>'
        f'<span class="label">{loc(bom["label"], ctx.lang)}</span>'
        f"<table><thead><tr>{head_cells}</tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def figure_alt(fig: dict) -> str:
    """Accessible alt text for a figure render.

    Authors may pin an exact ``alt`` per figure; otherwise it is derived from
    the SVG ``src``: drop the ``.svg`` suffix and the trailing ``_camN`` (plus
    any render-hash token), then surface the final ``_word`` (e.g. ``exploded``/
    ``assembled``) as a separate word — ``frame_10_extrusion_tnut_exploded_cam0_rbun.svg``
    becomes ``frame_10_extrusion_tnut exploded``.
    """
    if fig.get("alt") is not None:
        return fig["alt"]
    base = re.sub(r"\.svg$", "", fig["src"])
    base = re.sub(r"_cam\d+(?:_[a-z0-9]+)?$", "", base)
    return re.sub(r"_([a-z]+)$", r" \1", base)


def render_figure(
    fig: dict, ctx: Ctx, extra_class: str = "", style_on: str = "img"
) -> str:
    """Render a `.fig` cell wrapping a lazily-loaded `<img>` for one SVG render.

    ``style_on`` selects where the figure's inline style lands: on the ``<img>``
    (default — e.g. ``object-position`` framing) or on the ``.fig`` wrapper
    (``"fig"`` — e.g. the ``transform`` shifts on full-page main figures, which
    must move the clipped box, not the image inside it).
    """
    cls = f"fig {extra_class}".strip()
    style = fig.get("style") or ""
    attr = f' style="{style}"' if style else ""
    fig_attr = attr if style_on == "fig" else ""
    img_attr = attr if style_on == "img" else ""
    src = ctx.assets.figure(fig["src"])
    return (
        f'<div class="{cls}"{fig_attr}><img src="{src}" alt="{figure_alt(fig)}" '
        f'loading="lazy"{img_attr}></div>'
    )


def render_label(label: dict, ctx: Ctx) -> str:
    """Render a free-floating absolutely-positioned text label over the figure."""
    return f'<div style="{label["style"]}">{loc(label["text"], ctx.lang)}</div>'


def render_notes_and_bom(page: dict, ctx: Ctx) -> str:
    """Render the page overlays: free labels, then notes, then the optional BOM."""
    out = [render_label(lbl, ctx) for lbl in page.get("labels", [])]
    out += [render_note(n, ctx) for n in page.get("notes", [])]
    if "bom" in page:
        out.append(render_bom(page["bom"], ctx))
    return "".join(out)


# --------------------------------------------------------------------------- #
# Shared page chrome
# --------------------------------------------------------------------------- #
def render_head(head: dict | None, ctx: Ctx) -> str:
    """Render the masthead: a (possibly empty) title + the PhysiClaw.ai mark.

    Openers omit `head` entirely (their big title lives in the body), so the
    title span renders empty — matching the original markup.
    """
    title_html = ""
    if head and "title" in head:
        title_html = loc(head["title"], ctx.lang)
        if head.get("small"):
            title_html += f"<small>{head['small']}</small>"
    return (
        '<div class="head">'
        f'<span class="title">{title_html}</span>'
        f'<span class="url">{URL_MARK}</span></div>'
    )


def render_foot(page: dict) -> str:
    """Render the footer: page number, a rule, then the page number again.

    The number is assigned by position in ``assign_page_numbers`` — never
    hardcoded in the content JSON."""
    num = page["_pageno"]
    return f'<div class="foot"><span>{num}</span><span class="rule"></span><span>{num}</span></div>'


def page_shell(page: dict, ctx: Ctx, body: str, section_class: str = "") -> str:
    """Wrap a page body in the standard `.page > .page-inner` chrome.

    This is the single place the header/footer boilerplate is written. Every
    interior page carries the masthead (openers get an empty title), while the
    cover and back render their own bespoke chrome and so opt out via
    ``section_class``; those two also omit ``page`` and get no `.foot`.
    """
    cls = f"page {section_class}".strip()
    # Each page's semantic ``page`` id doubles as its HTML anchor (so TOC and
    # any cross-links can target it); the printed number comes from position.
    # ``assign_page_numbers`` guarantees every page carries a ``page`` id.
    anchor = f' id="{page["page"]}"'
    bespoke = section_class in ("cover", "back")
    head = "" if bespoke else render_head(page.get("head"), ctx)
    # Every interior page carries a numbered footer; the cover and back don't.
    foot = "" if bespoke else render_foot(page)
    return (
        f'<section class="{cls}"{anchor}>'
        f'<div class="page-inner">{head}{body}{foot}</div></section>'
    )


# --------------------------------------------------------------------------- #
# Per-type page renderers
# --------------------------------------------------------------------------- #
def render_cover(page: dict, ctx: Ctx) -> str:
    # The cover render is the largest above-the-fold image; it is preloaded in
    # <head> for a fast LCP.
    render_src = ctx.assets.figure(page["render"]["src"])
    body = (
        f'<div class="stripes">{COVER_STRIPES_SVG}</div>'
        '<div class="brand"><div class="mark">'
        f'<img src="{ctx.assets.crab}" alt="PhysiClaw logo">'
        '<span class="word">PhysiClaw<span class="tld">.ai</span></span>'
        "</div></div>"
        f'<div class="render"><img src="{render_src}" '
        'alt="PhysiClaw.ai, fully assembled"></div>'
        '<div class="title-block">'
        f"<h1>{loc(page['title'], ctx.lang)}</h1>"
        f'<p class="tag"> {loc(page["tag"], ctx.lang)} </p>'
        '<div class="red-rule"></div></div>'
        f'<div class="ver">VERSION {page["version"]}</div>'
    )
    return page_shell(page, ctx, body, "cover")


def render_toc(page: dict, ctx: Ctx) -> str:
    # Each row references a page by its semantic ``page`` id; the number is
    # resolved from that page's position (see assign_page_numbers), so the TOC
    # never hardcodes page numbers either.
    rows = "".join(
        f'<a class="toc-row" href="#{r.get("page", "")}">'
        f"<span>{loc(r['label'], ctx.lang)}</span>"
        f'<span class="pg">{r["_pgno"]:02d}</span></a>'
        for r in page["rows"]
    )
    return page_shell(page, ctx, f'<div class="toc-grid">{rows}</div>')


def render_intro(page: dict, ctx: Ctx) -> str:
    notes = "".join(render_note(n, ctx) for n in page["notes"])
    link_cells = []
    for link in page["links"]:
        logo = (
            GITHUB_OCTICON_SVG
            if link["logo"] == "github"
            else f'<img src="{ctx.assets.crab}" alt="">'
        )
        link_cells.append(
            f'<div class="intro-logo">{logo}'
            f'<span class="word">{loc(link["word"], ctx.lang)}</span></div>'
            f'<a class="intro-url" href="{link["url"]}">{link["url"]}</a>'
        )
    body = (
        f'<div class="intro-body">{notes}'
        f'<div class="intro-links">{"".join(link_cells)}</div></div>'
    )
    return page_shell(page, ctx, body)


def render_hardware_ref(page: dict, ctx: Ctx) -> str:
    entries = []
    for e in page["entries"]:
        ref = f'<p class="ref">{loc(e["ref"], ctx.lang)}</p>' if e.get("ref") else ""
        # A rendered SVG (``src``) replaces the dashed-frame placeholder once
        # the part has an icon; entries still carrying only ``iconLabel`` fall
        # back to the placeholder so a half-populated page still builds.
        if e.get("src"):
            icon = (
                f'<img src="{ctx.assets.figure(e["src"])}" '
                f'alt="{loc(e["h3"], ctx.lang)}" loading="lazy" decoding="async">'
            )
        else:
            icon = (
                '<svg class="ph" viewBox="0 0 200 140" xmlns="http://www.w3.org/2000/svg">'
                '<rect class="frame" x="20" y="20" width="160" height="100"/>'
                f'<text x="100" y="75" text-anchor="middle">{e["iconLabel"]}</text></svg>'
            )
        entries.append(
            f'<div class="hw-entry"><div class="icon">{icon}</div>'
            '<div class="note">'
            f"<h3>{loc(e['h3'], ctx.lang)}</h3><p>{loc(e['body'], ctx.lang)}</p>{ref}"
            "</div></div>"
        )
    # Row count is a pure layout concern — `.hw-grid` uses `grid-auto-rows`
    # so the rows fill the fixed height evenly whatever the entry count is.
    return page_shell(page, ctx, f'<div class="hw-grid">{"".join(entries)}</div>')


def render_printed_parts(page: dict, ctx: Ctx) -> str:
    before = "".join(render_note(n, ctx) for n in page["notesBefore"])
    specs = "".join(
        f'<div class="spec-item"><p class="k">{loc(s["k"], ctx.lang)}</p>'
        f'<p class="v">{loc(s["v"], ctx.lang)}</p></div>'
        for s in page["specs"]
    )
    after = "".join(render_note(n, ctx) for n in page["notesAfter"])
    body = (
        f'<div class="print-page">{before}'
        f'<div class="spec-grid">{specs}</div>{after}</div>'
    )
    return page_shell(page, ctx, body)


def render_opener(page: dict, ctx: Ctx) -> str:
    body = (
        '<div class="opener">'
        f'<div class="secnum">{loc(page["secnum"], ctx.lang)}</div>'
        f"<h1>{loc(page['h1'], ctx.lang)}</h1>"
        f'<p class="lede"> {loc(page["lede"], ctx.lang)} </p>'
        '<div class="bar"></div></div>'
    )
    return page_shell(page, ctx, body)


def render_solo(page: dict, ctx: Ctx) -> str:
    figs = "".join(render_figure(f, ctx) for f in page["figures"])
    body = f'<div class="solo">{figs}</div>{render_notes_and_bom(page, ctx)}'
    return page_shell(page, ctx, body)


def render_tall_left(page: dict, ctx: Ctx) -> str:
    figures = page["figures"]
    # figures[0] is the tall LEFT cell; the rest stack on the right in order.
    cells = [render_figure(figures[0], ctx, "tall")]
    cells += [render_figure(f, ctx) for f in figures[1:]]
    cls = "tall-left hero" if page.get("variant") == "hero" else "tall-left"
    grid_style = f' style="{page["gridStyle"]}"' if page.get("gridStyle") else ""
    body = (
        f'<div class="{cls}"{grid_style}>{"".join(cells)}</div>'
        f"{render_notes_and_bom(page, ctx)}"
    )
    return page_shell(page, ctx, body)


def render_wide_top(page: dict, ctx: Ctx) -> str:
    figures = page["figures"]
    # figures[0] is the wide TOP cell; the rest sit below in order.
    cells = [render_figure(figures[0], ctx, "wide")]
    cells += [render_figure(f, ctx) for f in figures[1:]]
    body = (
        f'<div class="wide-top">{"".join(cells)}</div>{render_notes_and_bom(page, ctx)}'
    )
    return page_shell(page, ctx, body)


def render_main_inset_br(page: dict, ctx: Ctx) -> str:
    # The main (and optional inset) figure always live inside .main-inset-br.
    # The overlay notes/BOM normally sit *outside* it (siblings under
    # .page-inner) so they position against the full page; pages that set
    # "notesInside" instead keep them within the figure box (its positioning
    # context) — the original manual does both, so this is per-page.
    figs = [render_figure(page["main"], ctx, "main", style_on="fig")]
    if page.get("inset"):
        figs.append(render_figure(page["inset"], ctx, "inset", style_on="fig"))
    overlays = render_notes_and_bom(page, ctx)
    if page.get("notesInside"):
        body = f'<div class="main-inset-br">{"".join(figs)}{overlays}</div>'
    else:
        body = f'<div class="main-inset-br">{"".join(figs)}</div>{overlays}'
    return page_shell(page, ctx, body)


def render_back(page: dict, ctx: Ctx) -> str:
    sub = f"<div>{loc(page['sub'], ctx.lang)}</div>" if page.get("sub") else ""
    body = (
        f'<div class="corner">{BACK_CORNER_SVG}</div>'
        '<div class="top">'
        f'<div class="stamp">{loc(page["stamp"], ctx.lang)}</div>'
        f"<h2>{loc(page['h2'], ctx.lang)}</h2>"
        f'<p class="quote"> {loc(page["quote"], ctx.lang)} </p></div>'
        '<div class="bottom"><div class="brand-line">'
        f'<img class="footmark" src="{ctx.assets.crab}" alt="PhysiClaw logo">'
        f"<div><div><strong>{loc(page['brand'], ctx.lang)}</strong></div>{sub}</div>"
        "</div></div>"
    )
    return page_shell(page, ctx, body, "back")


def _rowspans(rows: list[dict], key: Callable[[dict], Any]) -> list[int]:
    """For consecutive rows sharing ``key``, return the group size at each
    group's first row and 0 at the rest — i.e. the rowspan to emit (or skip)."""
    spans = [0] * len(rows)
    i = 0
    while i < len(rows):
        j = i + 1
        while j < len(rows) and key(rows[j]) == key(rows[i]):
            j += 1
        spans[i] = j - i
        i = j
    return spans


def render_bom_page(page: dict, ctx: Ctx) -> str:
    """Full-page consolidated bill of materials — a five-column table
    (Class / Component / Spec / Qty / Application). Rows are grouped so the Class
    cell merges over all its components and each Component cell merges over its
    specs (two-level rowspan). Spans are computed per page, so a class split
    across a page break simply repeats its label on the continuation page. Rows
    are authored in ``content/11_bom.json``; ``paginate_bom_pages`` splits an
    over-long list across continuation pages."""
    rows = page.get("rows", [])
    cls_span = _rowspans(rows, lambda r: r["cls"]["en"])
    comp_span = _rowspans(rows, lambda r: (r["cls"]["en"], r["component"]["en"]))

    body_parts: list[str] = []
    for idx, r in enumerate(rows):
        cells = ""
        if cls_span[idx]:
            cells += f'<td class="cls" rowspan="{cls_span[idx]}">{loc(r["cls"], ctx.lang)}</td>'
        if comp_span[idx]:
            cells += (
                f'<td class="comp" rowspan="{comp_span[idx]}">'
                f"{loc(r['component'], ctx.lang)}</td>"
            )
        # Group separators: "grp" starts a new class, "sub" a new component.
        row_name = "grp" if cls_span[idx] else "sub" if comp_span[idx] else ""
        row_cls = f' class="{row_name}"' if row_name else ""
        body_parts.append(
            f"<tr{row_cls}>{cells}"
            f'<td class="spec">{loc(r["spec"], ctx.lang)}</td>'
            f'<td class="qty">{r["qty"]}</td>'
            f'<td class="desc">{loc(r["desc"], ctx.lang)}</td></tr>'
        )

    headers = [
        {"en": "Class", "zh": "类别"},
        {"en": "Component", "zh": "组件"},
        {"en": "Spec", "zh": "规格"},
        {"en": "Qty", "zh": "数量"},
        {"en": "Application", "zh": "用途"},
    ]
    head_cells = "".join(f"<th>{loc(h, ctx.lang)}</th>" for h in headers)
    table = (
        '<table class="bom-table"><thead><tr>'
        f"{head_cells}"
        f"</tr></thead><tbody>{''.join(body_parts)}</tbody></table>"
    )
    # The masthead title already reads "Bill of materials" (with the "(cont.)"
    # marker on overflow pages), so no small in-body label is rendered — the
    # table gets that vertical room instead.
    body = f'<div class="bom-page">{table}</div>'
    return page_shell(page, ctx, body)


# Dispatch table: page "type" -> renderer.
RENDERERS: dict[str, Callable[[dict, Ctx], str]] = {
    "cover": render_cover,
    "toc": render_toc,
    "intro": render_intro,
    "hardware-ref": render_hardware_ref,
    "printed-parts": render_printed_parts,
    "opener": render_opener,
    "solo": render_solo,
    "tall-left": render_tall_left,
    "wide-top": render_wide_top,
    "main-inset-br": render_main_inset_br,
    "bom": render_bom_page,
    "back": render_back,
}


# --------------------------------------------------------------------------- #
# Document assembly
# --------------------------------------------------------------------------- #
def load_pages() -> list[dict]:
    """Load every content/*.json (sorted by filename = page order) into pages."""
    pages: list[dict] = []
    for path in sorted(CONTENT_DIR.glob("*.json")):
        pages.extend(json.loads(path.read_text(encoding="utf-8")))
    return pages


def assign_page_numbers(pages: list[dict]) -> None:
    """Number pages by position so the content JSON never hardcodes a page
    number — inserting or reordering a page needs no number edits anywhere.

    Each page is identified by a unique semantic ``page`` id (e.g. "frame_20",
    "hardware-reference"); the printed number is its 1-based position, stored
    in ``_pageno`` (cover and back render no footer). A TOC row references a
    page by that same ``page`` id, and the number it prints is resolved here
    from the referenced page's position into ``_pgno``."""
    # Build position map, validating the two invariants a new page can break:
    # every page carries a ``page`` id and those ids are unique (they double as
    # HTML anchors, so a collision would silently misroute links).
    id_to_no: dict[str, int] = {}
    toc_pages: list[dict] = []
    for i, page in enumerate(pages, start=1):
        page["_pageno"] = i
        if page["type"] == "toc":
            toc_pages.append(page)
        pid = page.get("page")
        if pid is None:
            raise ValueError(
                f"page #{i} (type {page.get('type')!r}) is missing a 'page' id"
            )
        if pid in id_to_no:
            raise ValueError(
                f"duplicate page id {pid!r} (pages #{id_to_no[pid]} and #{i})"
            )
        id_to_no[pid] = i
    # Resolve each TOC row's printed number now that the full position map is
    # built — a row may reference a page that follows the TOC itself.
    for page in toc_pages:
        for row in page.get("rows", []):
            ref = row.get("page")
            if ref not in id_to_no:
                raise ValueError(f"TOC row references unknown page id {ref!r}")
            row["_pgno"] = id_to_no[ref]


# Max table rows per BOM page. Tuned to the landscape page height for airy
# spacing (leaving margin below the last row); if the parts list grows, more
# pages are added automatically.
BOM_ROWS_PER_PAGE = 16

# Per-language "continued" marker appended to a BOM page's label on overflow
# pages. Falls back to the English form for any unlisted language.
BOM_CONT_SUFFIX = {"en": " (cont.)", "zh": "（续）"}


def _balanced_split(rows: list[dict], per_page: int) -> list[list[dict]]:
    """Partition rows into balanced pages without splitting any component's
    spec-rows (its rowspan must stay on one page). Uses ``ceil(total/per_page)``
    pages and places each evenly-spaced cut at the component boundary nearest its
    ideal position, so pages come out balanced and the last one is never left
    sparse. A class spanning a break repeats its label (spans recomputed per
    page)."""
    if not rows:
        return [[]]
    total = len(rows)
    n_pages = max(1, -(-total // per_page))  # ceil
    # Component boundaries — the only row indices a page may break at.
    free = [
        i
        for i in range(1, total)
        if (rows[i]["cls"]["en"], rows[i]["component"]["en"])
        != (rows[i - 1]["cls"]["en"], rows[i - 1]["component"]["en"])
    ]
    # Snap each evenly-spaced cut to the nearest unused boundary (n_pages == 1
    # makes this loop a no-op, leaving a single full-page chunk).
    cuts: list[int] = []
    for k in range(1, n_pages):
        if not free:
            break
        ideal = k * total / n_pages
        best = min(free, key=lambda s: abs(s - ideal))
        free.remove(best)
        cuts.append(best)
    points = [0, *sorted(cuts), total]
    return [rows[a:b] for a, b in zip(points, points[1:])]


def _paginate_bom(rows: list[dict], per_page: int) -> list[list[dict]]:
    """Partition the consolidated rows into pages. A row carrying
    ``"break_before": true`` forces a hard page break (it always starts a new
    page); the spans between forced breaks are then balanced independently by
    ``_balanced_split`` so each one still respects ``per_page``. Empty input
    falls through as a single empty span to ``_balanced_split``'s own ``[[]]``."""
    forced = [
        0,
        *(i for i in range(1, len(rows)) if rows[i].get("break_before")),
        len(rows),
    ]
    pages: list[list[dict]] = []
    for a, b in zip(forced, forced[1:]):
        pages.extend(_balanced_split(rows[a:b], per_page))
    return pages


def paginate_bom_pages(pages: list[dict]) -> None:
    """Split the ``bom`` page's authored rows (the hand-maintained parts list in
    ``content/11_bom.json``) across as many pages as fit, cloning the page shell
    for any overflow. This is layout only — it does NOT derive the list from the
    per-section BOMs; the consolidated BOM is kept consistent by hand. Must run
    BEFORE ``assign_page_numbers`` so any added pages get numbered."""
    idx = next((i for i, p in enumerate(pages) if p["type"] == "bom"), None)
    if idx is None:
        return
    template = pages[idx]
    chunks = _paginate_bom(template.get("rows", []), BOM_ROWS_PER_PAGE)
    template["rows"] = chunks[0]

    # Build continuation pages for any overflow chunks and splice them in.
    extras: list[dict] = []
    for n, chunk in enumerate(chunks[1:], start=2):
        cont = copy.deepcopy(template)
        cont["page"] = f"{template['page']}-{n}"
        cont["rows"] = chunk
        # Tag the continuation in the masthead title (the in-body label was
        # dropped to give the table more room), per translation.
        title = cont.get("head", {}).get("title")
        if title:
            cont["head"]["title"] = {
                lang: text + BOM_CONT_SUFFIX.get(lang, BOM_CONT_SUFFIX["en"])
                for lang, text in title.items()
            }
        extras.append(cont)
    if extras:
        pages[idx + 1 : idx + 1] = extras


def cover_render_src(pages: list[dict], ctx: Ctx) -> str | None:
    """The cover render's resolved src, for preloading (None if no cover)."""
    cover = next((p for p in pages if p["type"] == "cover"), None)
    return ctx.assets.figure(cover["render"]["src"]) if cover else None


def cover_title(pages: list[dict], ctx: Ctx) -> str:
    """The cover's localized title, reused as the browser-tab <title> so the
    two never drift (and the ZH manual gets a ZH tab title)."""
    cover = next((p for p in pages if p["type"] == "cover"), None)
    return (
        loc(cover["title"], ctx.lang)
        if cover and "title" in cover
        else "PhysiClaw Assembly Manual"
    )


def render_document(pages: list[dict], css: str, ctx: Ctx) -> str:
    """Assemble the full HTML document for one language."""
    sections = "\n".join(RENDERERS[p["type"]](p, ctx) for p in pages)
    src = cover_render_src(pages, ctx)
    preload = ctx.assets.preload(src) if src else ""
    return (
        f'<!DOCTYPE html>\n<html lang="{HTML_LANG[ctx.lang]}">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{cover_title(pages, ctx)}</title>\n"
        f"{preload}\n<style>\n{css}</style>\n</head>\n<body>\n"
        f"{sections}\n</body>\n</html>\n"
    )


# --------------------------------------------------------------------------- #
# PDF (headless Chrome) — the manual's CSS is print-ready (@page A4 landscape),
# so we render it with the browser engine it's designed for rather than a
# separate PDF library. Chrome is used only if already installed; absent it,
# the HTML build still succeeds and PDF is skipped with a note.
# --------------------------------------------------------------------------- #
CHROME_APP_PATHS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)


def find_chrome() -> str | None:
    """Path to a Chromium-family browser, or None if none is installed."""
    for name in (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "chrome",
    ):
        if found := shutil.which(name):
            return found
    return next((p for p in CHROME_APP_PATHS if Path(p).exists()), None)


PDF_TIMEOUT = 90  # s per attempt — a clean print is ~40s; this is generous
PDF_ATTEMPTS = 2  # a fresh process retry covers the rare launch that never
# writes the file (cf. the OCCT crash-retry in dispatch.py).


def _chrome_print(chrome: str, src_uri: str, profile: str, pdf_path: Path) -> bool:
    """One headless-Chrome print. Returns True once the PDF is written.

    ``--headless=new --print-to-pdf`` reliably *writes* the PDF but then often
    fails to *exit*, so we watch the output file rather than the process exit
    code: once it appears and its size stops growing, the print is done and we
    kill Chrome. Chrome runs in its own process group so a stuck launch (and
    all its renderer/helper children) dies as a unit — a survivor would
    contend with the next attempt."""
    pdf_path.unlink(missing_ok=True)  # so a stale file isn't read as success

    def written() -> bool:
        return pdf_path.exists() and pdf_path.stat().st_size > 0

    proc = subprocess.Popen(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-background-networking",
            f"--user-data-dir={profile}",
            "--no-pdf-header-footer",
            "--virtual-time-budget=30000",
            f"--print-to-pdf={pdf_path}",
            src_uri,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + PDF_TIMEOUT
        last_size, stable = -1, 0
        while time.monotonic() < deadline:
            size = pdf_path.stat().st_size if pdf_path.exists() else 0
            stable = stable + 1 if size > 0 and size == last_size else 0
            if stable >= 3:  # size held ~2.4s → fully written
                return True
            last_size = size
            if proc.poll() is not None:  # exited on its own
                return written()
            time.sleep(0.8)
        return written()
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()


def render_pdf(html: str, pdf_path: Path, chrome: str) -> bool:
    """Render ``html`` to ``pdf_path`` via headless Chrome, retrying in a
    fresh process if a launch hangs.

    The HTML is written to a temp file beside the PDF (so its relative
    ``assets/`` figure references resolve against the emitted asset dir) and
    printed with the new headless mode, which honours the document's CSS
    @page size/margins (A4 landscape, zero margin). ``loading="lazy"`` is
    stripped so every figure is painted in the single print pass. Returns
    True once a non-empty PDF is produced, else False."""
    html = html.replace(' loading="lazy"', "")
    for attempt in range(1, PDF_ATTEMPTS + 1):
        with (
            tempfile.TemporaryDirectory() as profile,
            tempfile.NamedTemporaryFile(
                "w", suffix=".html", dir=pdf_path.parent, encoding="utf-8", delete=False
            ) as tmp,
        ):
            tmp_path = Path(tmp.name)
            tmp.write(html)
            tmp.flush()
            try:
                result = _chrome_print(chrome, tmp_path.as_uri(), profile, pdf_path)
            finally:
                tmp_path.unlink(missing_ok=True)
        if result:  # _chrome_print returns True only after the PDF is written
            return True
        pdf_path.unlink(missing_ok=True)
        tail = " — retrying" if attempt < PDF_ATTEMPTS else " — skipped"
        print(
            f"warning: PDF render failed for {pdf_path.name} (attempt {attempt}){tail}"
        )
    return False


def _clear_output_dir(out_dir: Path) -> None:
    """Wipe stale manual artifacts so each run starts clean — a figure no
    longer referenced, the other language's HTML after a ``--lang`` build, or
    the sidecar assets after switching to ``--assets inline``. Mirrors
    ``build_procedures._clear_outputs``: only the extensions this script
    generates are touched, so user-placed files are left alone."""
    targets = [
        (out_dir, "*.html"),
        (out_dir, "*.pdf"),
        (out_dir / ASSETS_SUBDIR, "*.svg"),
    ]
    cleared = 0
    for d, pattern in targets:
        if not d.exists():
            continue
        for f in d.glob(pattern):
            f.unlink()
            cleared += 1
    print(f"  cleared {cleared} stale file(s)")


_STEP_LABEL_W = 22  # pad labels to a column so every duration lines up


@contextlib.contextmanager
def _step(label: str) -> Iterator[None]:
    """Print a build step (padded with dot leaders to a fixed column) and, once
    it finishes, how long it took — so a slow phase shows what's running and the
    times read down a clean right-aligned column."""
    print(f"  {label:.<{_STEP_LABEL_W}} ", end="", flush=True)
    t0 = time.monotonic()
    try:
        yield
    except BaseException:
        print("FAILED", flush=True)
        raise
    print(f"{time.monotonic() - t0:>5.1f}s")


def build(
    langs: list[str], out_dir: Path, inline: bool, pdf: bool = False
) -> list[Path]:
    """Render the requested languages into ``out_dir`` and return written files.

    Always writes the HTML; also writes a PDF per language when ``pdf`` is set
    and a Chromium-family browser is available. The PDF is printed from the
    just-written HTML — in external mode (default) Chrome loads the figures
    from the emitted ``assets/`` dir and embeds them into the PDF; in inline
    mode they are already data URIs."""
    _clear_output_dir(out_dir)  # start from a clean output dir (no stale files)
    out_dir.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    for name, svg in HAND_FIGURES.items():  # regenerate tracked hand figures
        (SVG_DIR / name).write_text(svg, encoding="utf-8")
    with _step("load + number pages"):
        css = STYLES_CSS.read_text(encoding="utf-8")
        pages = load_pages()
        paginate_bom_pages(
            pages
        )  # split the authored BOM rows across pages (may add pages)
        assign_page_numbers(
            pages
        )  # position-derived footer + TOC numbers (after BOM split)
    assets: Assets = InlineAssets() if inline else ExternalAssets()

    chrome = find_chrome() if pdf else None
    if pdf and chrome is None:
        print("note: no Chrome/Chromium found — skipping PDF output")

    written: list[Path] = []
    docs: dict[str, tuple[Path, str]] = {}
    for lang in langs:
        path = out_dir / LANG_FILENAME[lang]
        with _step(f"render html [{lang}]"):
            html = render_document(pages, css, Ctx(lang, assets))
            path.write_text(html, encoding="utf-8")
        written.append(path)
        docs[lang] = (path, html)
    with _step("emit assets"):
        assets.emit(out_dir)  # external mode writes the shared assets/ dir once.

    # PDFs print from the just-written HTML against the on-disk assets/, so
    # Chrome parses a light ~100 KB document (not a multi-MB inline one) and
    # loads each figure as its own file; the rendered PDF still embeds them.
    if chrome is not None:
        for lang in langs:
            path, html = docs[lang]
            pdf_path = path.with_suffix(".pdf")
            with _step(f"render pdf  [{lang}]"):
                ok = render_pdf(html, pdf_path, chrome)
            if ok:
                written.append(pdf_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lang",
        choices=("en", "zh", "all"),
        default="all",
        help="language(s) to render (default: all)",
    )
    parser.add_argument(
        "--assets",
        choices=("external", "inline"),
        default="external",
        help="external: lazy-loaded sidecar files (default); inline: one self-contained file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR,
        help="output directory (default: ../output/manual)",
    )
    parser.add_argument(
        "--pdf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also render a PDF per language via headless Chrome (default: off — HTML only)",
    )
    args = parser.parse_args()

    langs = ["en", "zh"] if args.lang == "all" else [args.lang]
    out = args.out.resolve()
    # Show the output dir relative to the cwd when possible — an absolute path
    # is usually too long to read at a glance.
    shown = out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out
    print(f"building manual [{', '.join(langs)}] -> {shown}")
    try:
        written = build(langs, out, inline=args.assets == "inline", pdf=args.pdf)
    except BuildError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"\ndone — wrote {len(written)} file(s):")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
