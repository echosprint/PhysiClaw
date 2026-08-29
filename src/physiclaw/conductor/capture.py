"""Capture — mine learned page geometry from on-device observations.

Fingerprint geometry is never authored or shipped: given N observations
of a declared page on the actual device, this module mines anchor
positions/tolerances/variants, weights them, and calibrates the page's
accept threshold from its own genuine-score distribution (per-page
thresholds — no universal cutoff exists, per the near-duplicate GUI-state
literature). First-run capture (system pages) and rehearsal-as-capture
(pack pages) call `capture_app`, the one entry that produces finished
`LearnedPage`s; the conductor CLI feeds it recorded sessions and live peeks.
"""

import statistics
from dataclasses import dataclass

from physiclaw.common.bbox import center_of
from physiclaw.common.listing import Screen
from physiclaw.conductor.match import candidate_rows, normalize, score_page
from physiclaw.conductor.pages import (
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
# Threshold calibration: 90% of the weakest genuine score, clamped.
THRESHOLD_SAFETY = 0.9
THRESHOLD_LO = 0.3
THRESHOLD_HI = 0.9


@dataclass(frozen=True)
class CaptureReport:
    page: str
    observations: int
    anchors_learned: int
    threshold: float
    genuine_min: float
    impostor_max: float | None  # None when no negatives were given

    @property
    def separable(self) -> bool:
        return self.impostor_max is None or self.impostor_max < self.threshold


def capture_app(
    app: str,
    decls: dict[str, PageDecl],
    observations_by_page: dict[str, list[Screen]],
    negatives: list[Screen],
) -> tuple[dict[str, LearnedPage], list[CaptureReport], list[str]]:
    """Mine + calibrate every declared page that has observations — the
    one producer of finished `LearnedPage`s, so an uncalibrated threshold
    never escapes this module. Pages without observations are skipped
    (reported in warnings). `negatives` are same-app hard negatives; each
    report says whether the page separates from them."""
    df = app_document_frequency(decls)
    learned: dict[str, LearnedPage] = {}
    reports: list[CaptureReport] = []
    warnings: list[str] = []
    for name, decl in decls.items():
        obs = observations_by_page.get(name, [])
        if not obs:
            warnings.append(f"{app}.{name}: no observations — skipped")
            continue
        anchors, page_warnings = mine_anchors(decl, obs, app_df=df)
        warnings.extend(f"{app}.{name}: {w}" for w in page_warnings)
        lp, report = _calibrate(
            PagePrint(
                app=app,
                decl=decl,
                # Staging value for scoring the genuine observations; the
                # calibrated threshold replaces it before anything escapes.
                learned=LearnedPage(
                    anchors=anchors, threshold=THRESHOLD_HI, observations=len(obs)
                ),
            ),
            obs,
            negatives,
        )
        learned[name] = lp
        reports.append(report)
    return learned, reports, warnings


def mine_anchors(
    decl: PageDecl,
    observations: list[Screen],
    *,
    app_df: dict[str, int] | None = None,
) -> tuple[dict[str, LearnedAnchor], list[str]]:
    """Mine geometry for one page's declared anchors from ≥1 observations.

    `app_df` is the within-app document frequency of each normalized
    anchor text (how many of the app's pages declare it) — shared chrome
    like a tab bar discriminates poorly, so weight divides by it. Anchors
    below MIN_FREQ stay declaration-only (warned, not learned)."""
    warnings: list[str] = []
    anchors: dict[str, LearnedAnchor] = {}
    n = len(observations)
    for a in decl.anchors:
        xs, ys, labels, confs = _readings(a, observations)
        freq = len(xs) / n if n else 0.0
        if freq < MIN_FREQ:
            warnings.append(
                f"anchor {a.text!r}: seen in {len(xs)}/{n} observations "
                f"(< {MIN_FREQ:.0%}) — kept declaration-only"
            )
            continue
        df = (app_df or {}).get(normalize(a.text), 1)
        anchors[a.text] = LearnedAnchor(
            text=a.text,
            cx=round(statistics.median(xs), 4),
            cy=round(statistics.median(ys), 4),
            pos_tol=_tolerance(xs, ys),
            freq=round(freq, 3),
            weight=round(freq * statistics.fmean(confs) / max(df, 1), 4),
            variants=_variants(a.text, labels),
        )
    if not anchors:
        warnings.append("no anchor earned geometry")
    return anchors, warnings


def _calibrate(
    pp: PagePrint,
    genuine: list[Screen],
    negatives: list[Screen],
) -> tuple[LearnedPage, CaptureReport]:
    """Set the page's accept threshold from its own genuine-score
    distribution, and report the hardest same-app negative so an
    inseparable page is visible instead of silent."""
    assert pp.learned is not None
    genuine_scores = [score_page(pp, s).score for s in genuine]
    impostor_scores = [score_page(pp, s).score for s in negatives]
    g_min = min(genuine_scores) if genuine_scores else 0.0
    threshold = min(max(g_min * THRESHOLD_SAFETY, THRESHOLD_LO), THRESHOLD_HI)
    learned = LearnedPage(
        anchors=pp.learned.anchors,
        threshold=round(threshold, 3),
        observations=pp.learned.observations,
    )
    report = CaptureReport(
        page=pp.page_id,
        observations=pp.learned.observations,
        anchors_learned=len(pp.learned.anchors),
        threshold=learned.threshold,
        genuine_min=round(g_min, 3),
        impostor_max=round(max(impostor_scores), 3) if impostor_scores else None,
    )
    return learned, report


def app_document_frequency(decls: dict[str, PageDecl]) -> dict[str, int]:
    """How many of the app's pages declare each normalized anchor text —
    the IDF input that downweights shared chrome."""
    df: dict[str, int] = {}
    for d in decls.values():
        for text in {normalize(a.text) for a in d.anchors}:
            df[text] = df.get(text, 0) + 1
    return df


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
    anchor, observations: list[Screen]
) -> tuple[list[float], list[float], list[str], list[float]]:
    """Per observation containing the anchor (best row by confidence):
    parallel center-x, center-y, raw-label, and confidence lists."""
    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    confs: list[float] = []
    for screen in observations:
        rows = candidate_rows(anchor, screen.rows, ())
        if not rows:
            continue
        row = max(rows, key=lambda r: r.conf)
        c = center_of(row.bbox)
        if c is None:
            continue
        xs.append(c[0])
        ys.append(c[1])
        labels.append(row.label)
        confs.append(row.conf)
    return xs, ys, labels, confs


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
