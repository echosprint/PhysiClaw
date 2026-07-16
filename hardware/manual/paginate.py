"""Page numbering and BOM pagination for the manual build.

Pure page-list transforms: number pages by position (so the content JSON
never hardcodes a number) and split the authored BOM rows across as many
pages as fit.
"""

from __future__ import annotations

import copy

# Max table rows per BOM page. Tuned to the landscape page height for airy
# spacing (leaving margin below the last row); if the parts list grows, more
# pages are added automatically.
BOM_ROWS_PER_PAGE = 16

# Per-language "continued" marker appended to a BOM page's label on overflow
# pages. Falls back to the English form for any unlisted language.
BOM_CONT_SUFFIX = {"en": " (cont.)", "zh": "（续）"}


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
