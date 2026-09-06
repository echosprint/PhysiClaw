"""Capture — mine learned page geometry from on-device observations.

Fingerprint geometry is never authored or shipped: given N observations
of a declared page on the actual device, this module mines anchor
positions/tolerances/variants and reports how the page's whole rule
(every anchor shows, no forbid term does) fares on its own genuine
observations and on same-app negatives — an inseparable page is
visible instead of silent. First-run capture (system pages) and
rehearsal-as-capture (pack pages) call `capture_app`, the one entry that
produces finished `LearnedPage`s; the conductor CLI feeds it recorded
sessions and live peeks.
"""

import statistics
from dataclasses import dataclass, replace

from physiclaw.common.bbox import center_of
from physiclaw.common.listing import Element, Screen
from physiclaw.conductor.spec.match import candidate_rows, normalize, score_page
from physiclaw.conductor.spec.pages import (
    AnchorDecl,
    LearnedAnchor,
    LearnedPage,
    PageDecl,
    PagePrint,
)

# An anchor must appear in at least this fraction of observations to earn
# geometry (the web template-detection rule: static chrome repeats).
MIN_FREQ = 0.6
# Positional tolerance: k·stddev with a floor at the fleet-measured p99
# jitter (0.007) headroom and a cap that keeps "same spot" meaningful.
TOL_K = 2.0
TOL_FLOOR = 0.02
TOL_CAP = 0.08
MAX_VARIANTS = 5


@dataclass(frozen=True)
class CaptureReport:
    page: str
    observations: int
    anchors_learned: int
    genuine_pass: float  # fraction of genuine observations the whole rule reads
    impostor_pass: int | None  # negatives the rule also reads; None without negatives

    @property
    def separable(self) -> bool:
        return not self.impostor_pass


def capture_app(
    app: str,
    decls: dict[str, PageDecl],
    observations_by_page: dict[str, list[Screen]],
    negatives: list[Screen],
) -> tuple[dict[str, LearnedPage], list[CaptureReport], list[str]]:
    """Mine every declared page that has observations — the one producer
    of finished `LearnedPage`s. Pages without observations are skipped
    (reported in warnings). `negatives` are same-app hard negatives; each
    report says whether the page separates from them."""
    learned: dict[str, LearnedPage] = {}
    reports: list[CaptureReport] = []
    warnings: list[str] = []
    for name, decl in decls.items():
        obs = observations_by_page.get(name, [])
        if not obs:
            warnings.append(f"{app}.{name}: no observations — skipped")
            continue
        anchors, page_warnings = mine_anchors(decl, obs)
        warnings.extend(f"{app}.{name}: {w}" for w in page_warnings)
        lp = LearnedPage(anchors=anchors, observations=len(obs))
        learned[name] = lp
        reports.append(
            _report(PagePrint(app=app, decl=decl, learned=lp), obs, negatives)
        )
    return learned, reports, warnings


def mine_anchors(
    decl: PageDecl,
    observations: list[Screen],
) -> tuple[dict[str, LearnedAnchor], list[str]]:
    """Mine geometry for one page's declared anchors from ≥1 observations.
    Anchors below MIN_FREQ stay declaration-only (warned, not learned)."""
    warnings: list[str] = []
    anchors: dict[str, LearnedAnchor] = {}
    n = len(observations)
    for a in decl.anchors:
        xs, ys, labels = _readings(a, observations)
        freq = len(xs) / n if n else 0.0
        if freq < MIN_FREQ:
            warnings.append(
                f"anchor {a.text!r}: seen in {len(xs)}/{n} observations "
                f"(< {MIN_FREQ:.0%}) — kept declaration-only"
            )
            continue
        anchors[a.text] = LearnedAnchor(
            text=a.text,
            cx=round(statistics.median(xs), 4),
            cy=round(statistics.median(ys), 4),
            pos_tol=_tolerance(xs, ys),
            freq=round(freq, 3),
            variants=_variants(a.text, labels),
        )
    if not anchors:
        warnings.append("no anchor earned geometry")
    return anchors, warnings


def _report(
    pp: PagePrint, genuine: list[Screen], negatives: list[Screen]
) -> CaptureReport:
    """How the page's whole rule fares with its geometry learned: the
    share of genuine observations it reads, and how many same-app
    negatives it also reads — an inseparable page is visible instead of
    silent."""
    assert pp.learned is not None
    return CaptureReport(
        page=pp.page_id,
        observations=pp.learned.observations,
        anchors_learned=len(pp.learned.anchors),
        genuine_pass=round(
            statistics.fmean(score_page(pp, s).passes for s in genuine), 3
        ),
        impostor_pass=(
            sum(score_page(pp, s).passes for s in negatives) if negatives else None
        ),
    )


def propose_anchors(screen: Screen, *, limit: int = 14) -> list[str]:
    """Anchor-declaration candidates from one screen: short, letter-bearing
    labels — chrome-shaped, not content-shaped. A human prunes the list
    into the pack's `pages:`; this only narrows the haystack."""
    out: list[str] = []
    for row in screen.rows:
        label = row.label.strip()
        if not (2 <= len(label) <= 12) or label in out:
            continue
        if not any(ch.isalpha() for ch in label):
            continue  # bare numbers/prices/times are content, not chrome
        out.append(label)
        if len(out) >= limit:
            break
    return out


def _readings(
    anchor: AnchorDecl, observations: list[Screen]
) -> tuple[list[float], list[float], list[str]]:
    """Per observation containing the anchor (best row by confidence):
    parallel center-x, center-y, and raw-label lists.

    Two passes through ONE ladder. First the run-time rule
    (`candidate_rows`). Then, for observations it found nothing on, the
    same rule read `loose` and pinned to the spot the exact readings
    agree on — how a two-character anchor's OCR confusion (综合 read 综台)
    gets mined into `variants` even though the matcher never admits it
    unmined: here the observations are labeled genuine and the position
    vouches for the row."""
    picked: list[Element] = []
    misses: list[Screen] = []
    for screen in observations:
        rows = candidate_rows(anchor, screen.rows, ())
        if rows:
            picked.append(max(rows, key=lambda r: r.conf))
        else:
            misses.append(screen)
    centers = [c for r in picked if (c := center_of(r.bbox)) is not None]
    if centers and misses:
        cx = statistics.median(c[0] for c in centers)
        cy = statistics.median(c[1] for c in centers)
        spot = replace(
            anchor,
            within=(
                max(0.0, cx - TOL_CAP),
                max(0.0, cy - TOL_CAP),
                min(1.0, cx + TOL_CAP),
                min(1.0, cy + TOL_CAP),
            ),
        )
        for screen in misses:
            rows = candidate_rows(spot, screen.rows, (), loose=True)
            if rows:
                picked.append(max(rows, key=lambda r: r.conf))
    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    for r in picked:
        c = center_of(r.bbox)
        if c is None:
            continue
        xs.append(c[0])
        ys.append(c[1])
        labels.append(r.label)
    return xs, ys, labels


def _tolerance(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return TOL_FLOOR
    spread = max(statistics.stdev(xs), statistics.stdev(ys))
    return round(min(max(TOL_K * spread, TOL_FLOOR), TOL_CAP), 4)


def _variants(anchor_text: str, labels: list[str]) -> tuple[str, ...]:
    """Distinct normalized OCR readings that differ from the anchor's own
    normalized form — the mined confusion set exact-matched at runtime."""
    anchor_norm = normalize(anchor_text)
    seen: list[str] = []
    for label in labels:
        n = normalize(label)
        if n != anchor_norm and n not in seen:
            seen.append(n)
    return tuple(seen[:MAX_VARIANTS])
