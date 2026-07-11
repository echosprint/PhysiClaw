#!/usr/bin/env python3
"""Render the bilingual PhysiClaw sourcing guide from the manual BOM + supplier data.

The parts list is read straight from the manual's bill-of-materials content
(the ``"bom"``-type page in ``content/``), so the sourcing guide can never
drift from the assembly manual — class, component, spec, qty and application
render verbatim from those rows, each carrying a short stable ``part_id``
(e.g. ``rail-y``).

``sourcing_vendors.json`` is an array with one entry per BOM row, in BOM
order, keyed by ``part_id``. Each entry adds the sourcing columns of the
table — all optional:

- ``ref`` — 参考价: a rough expected cost for the needed quantity, so the
  buyer can judge whether a shop's price is reasonable (missing -> em dash);
- ``inquiry`` — a ready-to-paste message for the shop (one fact per line).
  It has no cell of its own: the part's ``note`` embeds it via the
  ``{inquiry}`` placeholder as a click-to-copy "询价说明" control (hover
  previews the message). Only custom / cut-to-order parts carry one;
  standard parts are bought off the shelf by spec;
- ``suppliers`` — 供应商 1..3: up to three shops, each ``{name, link?,
  product?}``: the name (linked to the shop's ``link`` URL when filled in)
  over the shop's own ``product`` spec. Missing slots render as pending;
  a slot whose name is ``—`` renders as a muted dash (nothing to buy —
  e.g. the part comes bundled with another purchase).
  Localizable values take ``{"en":…,"zh":…}``; plain strings show in both
  languages;
- ``note`` — 备注: a free remark column at the table's end (missing -> em
  dash). Notes are prose and emitted as trusted HTML, the manual's
  convention for content strings (inline ``<a>`` is fine); data fields
  (names, products, refs) are escaped instead. The placeholder ``{inquiry}``
  renders as the same click-to-copy 询价说明 control as in the supplier
  cells.

The special value ``"Ditto"`` makes a field merge with the row above — the
table cell spans both rows, exactly like a ditto mark in a paper ledger. Use
it when consecutive parts are sourced together: the frame extrusions carry
one ref / inquiry / suppliers set on their first row and ``"Ditto"`` on the
rest, so the cutting order reads as one spanned block. ``"Ditto"`` works per
field (``ref`` / ``inquiry`` / ``suppliers`` / ``note``), or per supplier
slot inside the ``suppliers`` array; on the first row it is an error.

Run under ``uv`` from the repo root (standard library only, Python 3.12+);
all paths resolve relative to this file, so the cwd does not matter::

    uv run hardware/manual/build_sourcing_guide.py              # en + zh
    uv run hardware/manual/build_sourcing_guide.py --lang en    # English only -> sourcing_guide.html
    uv run hardware/manual/build_sourcing_guide.py --scaffold   # sync sourcing_vendors.json with the BOM, then build

``--scaffold`` inserts a bare ``{"part_id": …}`` entry for every BOM row
missing from the file, reorders entries to BOM order, and reports stale ids;
it never expands entries with defaults.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

# Shared with the manual build: localization, BOM row grouping, step timing,
# content loading and the html-lang / masthead-mark conventions.
from build_manual import (
    CONTENT_DIR,
    HTML_LANG,
    URL_MARK,
    BuildError,
    _rowspans,
    _step,
    load_pages,
    loc,
)

# --------------------------------------------------------------------------- #
# Paths — everything is resolved relative to this file so cwd does not matter.
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR_FILE = SCRIPT_DIR / "sourcing_vendors.json"
STYLES_CSS = SCRIPT_DIR / "sourcing_styles.css"
# Own output dir — build_manual._clear_output_dir wipes *.html in output/manual,
# so the sourcing guide must not live there.
OUTPUT_DIR = SCRIPT_DIR / ".." / "output" / "sourcing"

LANG_FILENAME = {"en": "sourcing_guide.html", "zh": "physiclaw采购指南.html"}

SUPPLIERS_PER_ROW = 3  # supplier slots per part
DITTO = "Ditto"  # field value: merge this cell with the row above

# UI chrome strings (content strings come localized from the JSON sources).
UI = {
    "doc_title": {"en": "PhysiClaw.ai Sourcing Guide", "zh": "PhysiClaw.ai 采购指南"},
    "h1": {"en": "Sourcing guide", "zh": "采购指南"},
    "lede": {
        "en": "This guide covers every part in the assembly manual's bill "
        "of materials, with three reference suppliers per part.<br>"
        "Reference prices are subtotals for the required quantity "
        "(not unit prices; shipping excluded) and do not represent "
        "actual transaction prices — vendors may adjust pricing at "
        "any time; use them only to gauge the general price level.",
        "zh": "本指南涵盖装配手册物料清单的全部零件，每项零件列出三家参考供应商。<br>"
        "参考价为所需数量的合计金额（非单价，不含运费），并非实际成交价格"
        "——厂商可能随时调价，仅供衡量大致价格水平。",
    },
    "pending": {"en": "to be found", "zh": "待补充"},
    "disclaimer": {
        "en": "<strong>Disclaimer:</strong> Supplier and product listings in "
        "this guide are provided for reference only. We are not "
        "affiliated with, and receive no compensation from, any of the "
        "vendors listed, and we make no representation or warranty as "
        "to the quality, pricing, or availability of their products. "
        "Please verify the specifications and purchase from a vendor "
        "of your choice. Should any listed product prove "
        "unsatisfactory, or should you have a reliable vendor to "
        "recommend, we welcome your feedback and will review and "
        "update the listings accordingly.",
        "zh": "<strong>免责声明：</strong>本指南所列供应商及商品链接仅供采购参考。"
        "我们与所列商家不存在任何关联或利益关系，"
        "对其商品的质量、价格及供货情况不作任何保证。"
        "请按规格自行甄选，从您信任的商家购买。"
        "如所列商品存在质量问题，或您有优质供应商推荐，"
        "欢迎向我们反馈，我们将及时核实并更新相关条目。",
    },
    "inquiry_label": {"en": "Inquiry message", "zh": "询价说明"},
    "supplier_n": {"en": "Supplier", "zh": "供应商"},
    "th_cls": {"en": "Class", "zh": "类别"},
    "th_component": {"en": "Component", "zh": "组件"},
    "th_spec": {"en": "Spec", "zh": "规格"},
    "th_qty": {"en": "Qty", "zh": "数量"},
    "th_desc": {"en": "Application", "zh": "用途"},
    "th_ref": {"en": "Ref. price", "zh": "参考价"},
    "th_note": {"en": "Notes", "zh": "备注"},
}


def ui(key: str, lang: str) -> str:
    return loc(UI[key], lang)


# Snippet-style copy button icons (Feather "copy" / "check"), swapped by CSS
# when the button briefly carries the `ok` class after a successful copy.
COPY_ICON_SVG = (
    '<svg class="ic-copy" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>'
    '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>'
)
CHECK_ICON_SVG = (
    '<svg class="ic-check" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M20 6 9 17l-5-5"/></svg>'
)


def clean_taobao_url(url: str) -> str:
    """Strip a Taobao/Tmall item URL down to only the identifying params.

    Item links pasted from the browser carry a long tracking tail (spm,
    utparam, xxc, mi_id, ...); only ``id`` — plus ``skuId`` when present —
    identifies the listing. The query is rebuilt from that whitelist, the
    scheme/host/path are preserved, and anything that isn't a Taobao-family
    URL with an ``id`` passes through unchanged (e.g. shop homepages), so
    this is safe to apply to every supplier ``url`` at render time.

    >>> clean_taobao_url("https://item.taobao.com/item.htm?abbucket=14"
    ...                  "&id=12345&mi_id=xyz&skuId=67890&spm=a21xtw.123&xxc=ad")
    'https://item.taobao.com/item.htm?id=12345&skuId=67890'
    >>> clean_taobao_url("https://item.taobao.com/item.htm?ns=1&id=98765")
    'https://item.taobao.com/item.htm?id=98765'
    >>> clean_taobao_url("https://shop123.taobao.com/")
    'https://shop123.taobao.com/'
    >>> clean_taobao_url("https://example.com/listing?id=1&color=red")
    'https://example.com/listing?id=1&color=red'
    """
    parts = urlparse(url)
    host = parts.netloc.lower()
    if not host.endswith(("taobao.com", "tmall.com")):
        return url
    params = parse_qs(parts.query)
    if "id" not in params:
        return url
    kept = {key: params[key] for key in ("id", "skuId") if key in params}
    return urlunparse(parts._replace(query=urlencode(kept, doseq=True)))


# --------------------------------------------------------------------------- #
# Data — BOM rows from the manual content, sourcing entries from the data file.
# --------------------------------------------------------------------------- #
def load_bom_rows() -> list[dict]:
    """The ``bom`` page rows from the manual content, in order — located by
    page type (not filename) via the same loader the manual build uses, so a
    renamed or split content file cannot desync the two builds. Every row
    must carry a unique ``part_id`` — that id is the sourcing data's key, so
    a missing or duplicated one would silently orphan entries."""
    rows = [
        row
        for page in load_pages()
        if page.get("type") == "bom"
        for row in page.get("rows", [])
    ]
    if not rows:
        raise BuildError(f"no 'bom' page rows found in {CONTENT_DIR}")
    missing = [
        f"{r['component']['en']} | {r['spec']['en']}"
        for r in rows
        if not r.get("part_id")
    ]
    if missing:
        raise BuildError(
            "BOM row(s) missing a part_id:\n" + "\n".join(f"    {k}" for k in missing)
        )
    ids = [r["part_id"] for r in rows]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise BuildError(
            "duplicate part_id(s) in the BOM:\n" + "\n".join(f"    {i}" for i in dupes)
        )
    return rows


def load_sourcing_data() -> list[dict]:
    """Read sourcing_vendors.json (an array, one entry per BOM row),
    bootstrapping an empty list on first run."""
    if not VENDOR_FILE.exists():
        return []
    data = json.loads(VENDOR_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise BuildError(
            f"{VENDOR_FILE.name} must be a JSON array "
            "(one entry per BOM row, keyed by part_id)"
        )
    return data


def sync_entries(
    rows: list[dict], entries: list[dict]
) -> tuple[list[dict], list[str], list[str]]:
    """Align the entry array with the BOM: one entry per row, in BOM order.
    Missing rows get a bare ``{"part_id": …}``; entries whose id no longer
    matches a row are dropped and reported. Authored entries pass through
    untouched. Returns ``(synced, added_ids, stale_ids)``."""
    ids = [e.get("part_id") for e in entries]
    dupes = sorted({i for i in ids if i and ids.count(i) > 1})
    if dupes:
        raise BuildError(
            f"duplicate part_id(s) in {VENDOR_FILE.name}:\n"
            + "\n".join(f"    {i}" for i in dupes)
        )
    by_id = {e.get("part_id"): e for e in entries}
    row_ids = {r["part_id"] for r in rows}
    synced: list[dict] = []
    added: list[str] = []
    for row in rows:
        pid = row["part_id"]
        if pid not in by_id:
            added.append(pid)
        synced.append(by_id.get(pid) or {"part_id": pid})
    stale = [i for i in ids if i not in row_ids]
    return synced, added, stale


def write_sourcing_file(entries: list[dict]) -> None:
    VENDOR_FILE.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Ditto resolution — per column, "Ditto" merges a cell with the row above.
# --------------------------------------------------------------------------- #
def ditto_walk(values: list, col: str, ids: list[str]) -> tuple[list[int], list]:
    """Resolve one column's ``"Ditto"`` marks in a single walk, returning
    ``(spans, resolved)``: a non-Ditto value opens a cell of span 1; each
    following ``"Ditto"`` extends that cell (span 0 — no cell emitted) and
    inherits the anchor's value into ``resolved``. ``"Ditto"`` on the first
    row has nothing to merge with."""
    spans = [0] * len(values)
    resolved: list = []
    anchor = -1
    for i, value in enumerate(values):
        if value == DITTO:
            if anchor < 0:
                raise BuildError(
                    f'"{DITTO}" in {col!r} of entry {ids[i]!r} '
                    "has no row above to merge with"
                )
            spans[anchor] += 1
            resolved.append(resolved[anchor])
        else:
            anchor = i
            spans[i] = 1
            resolved.append(value)
    return spans, resolved


def supplier_columns(entries: list[dict]) -> list[list]:
    """The three supplier columns as per-row value lists. An entry's
    ``suppliers`` may be ``"Ditto"`` (all three slots merge), an array of up
    to three slots (each a supplier dict or ``"Ditto"``), or missing."""
    columns: list[list] = [[] for _ in range(SUPPLIERS_PER_ROW)]
    for entry in entries:
        suppliers = entry.get("suppliers")
        if suppliers == DITTO:
            slots = [DITTO] * SUPPLIERS_PER_ROW
        elif suppliers is None:
            slots = [None] * SUPPLIERS_PER_ROW
        else:
            slots = list(suppliers)[:SUPPLIERS_PER_ROW]
            slots += [None] * (SUPPLIERS_PER_ROW - len(slots))
        for j in range(SUPPLIERS_PER_ROW):
            columns[j].append(slots[j])
    return columns


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _span_attr(span: int) -> str:
    return f' rowspan="{span}"' if span > 1 else ""


def _inquiry_button(message: str, lang: str, flip: bool = False) -> str:
    """The click-to-copy 询价说明 control: copy icon + label; hovering shows
    the message in a tooltip, clicking copies it (icon morphs to a check).
    ``flip`` opens the tooltip leftward — for buttons near the table's right
    edge, where it would otherwise be clipped."""
    # &#10; keeps the message's line breaks intact inside the attribute;
    # the CSS tooltip renders it via content:attr(data-q) + pre-line, so
    # the hover preview and the copied text are the same string.
    msg_attr = html.escape(message, quote=True).replace("\n", "&#10;")
    cls = "copy flip" if flip else "copy"
    return (
        f'<button class="{cls}" type="button" data-q="{msg_attr}">'
        f"{COPY_ICON_SVG}{CHECK_ICON_SVG}"
        f"<span>{ui('inquiry_label', lang)}</span></button>"
    )


def render_supplier_cell(supplier: dict | None, lang: str, span: int) -> str:
    """One supplier cell: the shop name (linked to its url) over the shop's
    own ``product`` spec, when set."""
    name = loc(supplier.get("name") or "", lang) if supplier else ""
    if not name:
        return f'<td class="offer pending"{_span_attr(span)}>{ui("pending", lang)}</td>'
    if name == "—":
        return f'<td class="offer na"{_span_attr(span)}>—</td>'
    name = html.escape(name)
    if supplier.get("url"):
        href = html.escape(clean_taobao_url(supplier["url"]), quote=True)
        name = f'<a href="{href}" target="_blank" rel="noopener">{name}</a>'
    product = loc(supplier.get("product") or "", lang)
    product_html = (
        f'<div class="v-prod">{html.escape(product)}</div>' if product else ""
    )
    return (
        f'<td class="offer"{_span_attr(span)}><div class="v-vendor">{name}</div>'
        f"{product_html}</td>"
    )


def render_table(rows: list[dict], entries: list[dict], lang: str) -> str:
    """The grouped parts table: BOM columns rendered verbatim from the manual
    content, then the sourcing columns (ref price / one column per supplier
    slot / notes). ``"Ditto"`` fields merge their cell with the row above."""
    ids = [r["part_id"] for r in rows]
    cls_span = _rowspans(rows, lambda r: r["cls"]["en"])
    comp_span = _rowspans(rows, lambda r: (r["cls"]["en"], r["component"]["en"]))

    ref_spans, refs = ditto_walk([e.get("ref") for e in entries], "ref", ids)
    note_spans, notes = ditto_walk([e.get("note") for e in entries], "note", ids)
    sup_walks = [
        ditto_walk(col, f"suppliers[{j}]", ids)
        for j, col in enumerate(supplier_columns(entries))
    ]
    # The inquiry has no cell of its own — it rides as a copy control inside
    # the part's note ({inquiry}), so only its resolved values are used.
    _, inquiries = ditto_walk([e.get("inquiry") for e in entries], "inquiry", ids)

    body: list[str] = []
    for idx, row in enumerate(rows):
        cells = ""
        if cls_span[idx]:
            cells += f'<td class="cls"{_span_attr(cls_span[idx])}>{loc(row["cls"], lang)}</td>'
        if comp_span[idx]:
            cells += (
                f'<td class="comp"{_span_attr(comp_span[idx])}>'
                f"{loc(row['component'], lang)}</td>"
            )
        cells += (
            f'<td class="spec">{loc(row["spec"], lang)}</td>'
            f'<td class="qty">{row["qty"]}</td>'
            f'<td class="desc">{loc(row["desc"], lang)}</td>'
        )

        message = loc(inquiries[idx] or "", lang)
        for spans, col in sup_walks:
            if spans[idx]:
                cells += render_supplier_cell(col[idx], lang, spans[idx])
        if ref_spans[idx]:
            ref = loc(refs[idx] or "", lang) or "—"
            cells += (
                f'<td class="ref"{_span_attr(ref_spans[idx])}>{html.escape(ref)}</td>'
            )
        if note_spans[idx]:
            # Notes are prose — emitted as trusted HTML, like the manual's
            # content strings.
            note = loc(notes[idx] or "", lang)
            if "{inquiry}" in note:
                if not message:
                    raise BuildError(
                        f"note of entry {ids[idx]!r} uses "
                        "{inquiry} but the part has no inquiry message"
                    )
                note = note.replace(
                    "{inquiry}", _inquiry_button(message, lang, flip=True)
                )
            cells += f'<td class="note"{_span_attr(note_spans[idx])}>{note or "—"}</td>'

        # Only class starts get a heavier separator; the full cell grid
        # already delineates components, so no `sub` class here.
        row_cls = ' class="grp"' if cls_span[idx] else ""
        body.append(f"<tr{row_cls}>{cells}</tr>")

    supplier_ths = "".join(
        f"<th>{ui('supplier_n', lang)} {j + 1}</th>" for j in range(SUPPLIERS_PER_ROW)
    )
    head = (
        f"<tr><th>{ui('th_cls', lang)}</th><th>{ui('th_component', lang)}</th>"
        f"<th>{ui('th_spec', lang)}</th><th>{ui('th_qty', lang)}</th>"
        f"<th>{ui('th_desc', lang)}</th>"
        f"{supplier_ths}<th>{ui('th_ref', lang)}</th>"
        f"<th>{ui('th_note', lang)}</th></tr>"
    )
    return (
        f'<div class="table-wrap"><table class="sourcing">'
        f"<thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div>"
    )


# Page script: copy-to-clipboard for the per-row inquiry message. Both file://
# and https:// are secure contexts, so navigator.clipboard is always there.
PAGE_JS = """\
document.querySelectorAll('button.copy').forEach(function (b) {
  b.addEventListener('click', function () {
    navigator.clipboard.writeText(b.dataset.q).then(function () {
      b.classList.add('ok');
      setTimeout(function () { b.classList.remove('ok'); }, 1200);
    });
  });
});
"""


def render_document(rows: list[dict], entries: list[dict], css: str, lang: str) -> str:
    body = (
        '<div class="wrap">'
        f'<header class="mast"><h1>{ui("h1", lang)}</h1>'
        f'<span class="url">{URL_MARK}</span></header>'
        f'<div class="preamble"><p class="lede">{ui("lede", lang)}</p>'
        f'<p class="disclaimer">{ui("disclaimer", lang)}</p></div>'
        f"{render_table(rows, entries, lang)}</div>"
    )
    return (
        f'<!DOCTYPE html>\n<html lang="{HTML_LANG[lang]}">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{ui('doc_title', lang)}</title>\n"
        f"<style>\n{css}</style>\n</head>\n<body>\n{body}\n"
        f"<script>\n{PAGE_JS}</script>\n</body>\n</html>\n"
    )


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(langs: list[str], out_dir: Path, scaffold: bool) -> list[Path]:
    with _step("load bom + sourcing"):
        rows = load_bom_rows()
        entries, added, stale = sync_entries(rows, load_sourcing_data())
    if scaffold or not VENDOR_FILE.exists():
        write_sourcing_file(entries)
        print(f"  synced {VENDOR_FILE.name}: +{len(added)} new empty entr(y/ies)")
    elif added:
        print(
            f"  note: {len(added)} BOM row(s) have no sourcing entry "
            f"(rendered with defaults) — run with --scaffold to add them"
        )
    if stale:
        print(
            f"  warning: {len(stale)} stale entr(y/ies) no longer match a BOM part_id:"
        )
        for i in stale:
            print(f"    {i}")

    css = STYLES_CSS.read_text(encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for lang in langs:
        path = out_dir / LANG_FILENAME[lang]
        with _step(f"render html [{lang}]"):
            path.write_text(render_document(rows, entries, css, lang), encoding="utf-8")
        written.append(path)
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
        "--out",
        type=Path,
        default=OUTPUT_DIR,
        help="output directory (default: ../output/sourcing)",
    )
    parser.add_argument(
        "--scaffold",
        action="store_true",
        help="sync sourcing_vendors.json with the BOM (add bare part_id entries) before building",
    )
    args = parser.parse_args()

    langs = ["en", "zh"] if args.lang == "all" else [args.lang]
    out = args.out.resolve()
    shown = out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out
    print(f"building sourcing guide [{', '.join(langs)}] -> {shown}")
    try:
        written = build(langs, out, scaffold=args.scaffold)
    except BuildError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"\ndone — wrote {len(written)} file(s):")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
