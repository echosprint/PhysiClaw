"""Card clustering — item candidates as CARDS, not OCR fragments.

A shopping result page renders one item as a stack of rows: title,
price, sales count, shop name. Feeding those rows to `choose_item` one
by one makes the price rows candidates themselves while the title
carries no price — the model chooses among fragments. This module
groups a screen's labeled rows into cards so one candidate carries the
whole story: the TITLE row is the key (verbatim — `picked.key` flows
into `{node.field}` refs, the ledger label, and cart-row matching, so
it must stay a raw screen label) and the tap target; every other row in
the card is prompt-only metadata.

Grouping is 2D on purpose: real result pages are two-column waterfalls,
so rows at the same height can belong to different items. Two rows join
a card only when their horizontal extents overlap AND the vertical gap
between them is within a card's line spacing; side-by-side rows (price
left, sales right) join through the title above them spanning both.
Everything degrades gracefully — a lone row is a one-row card, exactly
the pre-card candidate, so non-shopping screens lose nothing.
"""

from dataclasses import dataclass

from physiclaw.common.listing import Element
from physiclaw.common.text import clip

# The vertical edge gap that separates cards: within a card, text rows
# stack at line spacing (~0.02-0.03 of screen height); between cards the
# next item's IMAGE sits between the text blocks, so the gap in labeled
# rows is far larger. Sits above the spacing and below the image band.
CARD_GAP = 0.04

# Prompt-size guards on the metadata: fragments per card and characters
# per fragment. A card's story is title + a few short facts; more is a
# merged mega-cluster (a spanning banner bridged two columns) and extra
# tokens without extra signal.
MAX_META = 4
META_CLIP = 40


@dataclass(frozen=True)
class Card:
    """One clustered item: the title row (key + tap target) and the
    other labels of its band, in reading order, clipped."""

    title: Element
    meta: tuple[str, ...]


def group_cards(rows: tuple[Element, ...]) -> list[Card]:
    """Cluster labeled text rows into cards, top-to-bottom (then left to
    right). Title = the longest label of the cluster (product titles
    are long, price/sales rows short), topmost on a tie."""
    labeled = [r for r in rows if r.kind == "text" and r.label.strip()]
    clusters = _cluster(labeled)
    out: list[Card] = []
    for cluster in clusters:
        ordered = sorted(cluster, key=lambda r: (r.bbox[1], r.bbox[0]))
        title = max(ordered, key=lambda r: (len(r.label.strip()), -r.bbox[1]))
        meta = tuple(
            clip(r.label.strip(), META_CLIP) for r in ordered if r is not title
        )[:MAX_META]
        out.append(Card(title=title, meta=meta))
    out.sort(key=lambda c: (c.title.bbox[1], c.title.bbox[0]))
    return out


def _cluster(rows: list[Element]) -> list[list[Element]]:
    """Union-find over the joins-a-card relation: x-overlap AND a
    vertical edge gap within CARD_GAP. O(n²) pairs over ≤ ~60 rows."""
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if _same_card(rows[i], rows[j]):
                parent[find(i)] = find(j)

    by_root: dict[int, list[Element]] = {}
    for i, row in enumerate(rows):
        by_root.setdefault(find(i), []).append(row)
    return list(by_root.values())


def _same_card(a: Element, b: Element) -> bool:
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    if min(ax1, bx1) - max(ax0, bx0) <= 0:
        return False  # different columns — same height means nothing
    gap = max(by0 - ay1, ay0 - by1)  # negative when they overlap vertically
    return gap < CARD_GAP
