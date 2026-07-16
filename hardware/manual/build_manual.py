#!/usr/bin/env python3
"""Render the bilingual PhysiClaw assembly manual from JSON content files.

The manual's content lives as a sequence of ``content/*.json`` files, each a
list of *page* dicts in visual (DOM) order. This script turns those pages into
one HTML document per language, laid out by ``styles.css`` (always inlined).

Image assets (the SVG renders + the crab logo) are handled by one of two
strategies, chosen with ``--assets`` (see ``hardware.manual.assets``):

- ``external`` (default) — best for the web. SVGs are written as separate files
  under ``assets/`` and referenced with ``loading="lazy" decoding="async"`` so
  the browser fetches only what scrolls into view, in parallel, and caches each
  file. The above-the-fold cover render loads eagerly and is preloaded for a
  fast LCP. The HTML itself stays tiny (~100 KB).
- ``inline`` — a single self-contained file. Every image is embedded as a
  base64 ``data:`` URI (no external requests, no lazy loading — best for
  offline use / emailing one file around, at the cost of a large document).

Run under ``uv`` from the repo root (standard library only, Python 3.12+)::

    uv run python -m hardware.manual.build_manual                  # en + zh, external assets, HTML only
    uv run python -m hardware.manual.build_manual --pdf            # also render a PDF per language
    uv run python -m hardware.manual.build_manual --lang en        # English only -> physiclaw_manual.html
    uv run python -m hardware.manual.build_manual --assets inline  # single self-contained file
    uv run python -m hardware.manual.build_manual --out /tmp/out   # custom output directory

PDF output uses an already-installed Chromium-family browser (Chrome / Chromium
/ Edge); if none is found the HTML still builds and PDF is skipped with a note
(see ``hardware.manual.pdf``).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from hardware.manual import BuildError
from hardware.manual.assets import (
    ASSETS_SUBDIR,
    HAND_FIGURES,
    Assets,
    ExternalAssets,
    InlineAssets,
)
from hardware.manual.common import (
    HTML_LANG,
    URL_MARK,
    _rowspans,
    _step,
    load_pages,
    loc,
)
from hardware.manual.icon_svg import (
    BACK_CORNER_SVG,
    COVER_STRIPES_SVG,
    GITHUB_OCTICON_SVG,
    INFO_ICON_SVG,
)
from hardware.manual.paginate import assign_page_numbers, paginate_bom_pages
from hardware.manual.pdf import find_chrome, render_pdf
from hardware.scheme import OUTPUT_DIR as _OUTPUT_ROOT
from hardware.scheme import SVG_DIR

# Note "icon" keys -> inline SVG markup, for the leading icon of a flex callout.
NOTE_ICONS = {"info": INFO_ICON_SVG}

# --------------------------------------------------------------------------- #
# Paths — package files are resolved relative to this file; output paths come
# from hardware.scheme, so the cwd does not matter.
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
STYLES_CSS = SCRIPT_DIR / "styles.css"
OUTPUT_DIR = _OUTPUT_ROOT / "manual"

# Output filename per language.
LANG_FILENAME = {"en": "physiclaw_manual.html", "zh": "physiclaw装配手册.html"}


@dataclass(frozen=True)
class Ctx:
    """Per-render context: the active language and the asset strategy."""

    lang: str
    assets: Assets


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
        help="output directory (default: hardware/output/manual)",
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
