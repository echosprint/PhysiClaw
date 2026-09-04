"""Page declarations + learned geometry — the two halves of a fingerprint.

A page fingerprint is split by audience:

  - DECLARATIONS (the pack file's ``pages:`` section, human-authored): the page's
    name, which label texts identify it, forbid terms, coarse region hints.
    Semantics only — portable across devices and app versions.
  - LEARNED (``learned/pages/<app>.json``, machine-written by capture):
    anchor positions and tolerances, OCR variants, calibrated per-page
    threshold. Geometry is ALWAYS captured on-device, never authored or
    shipped — iPhone models, iOS and app versions make shipped geometry
    infeasible (the same reason first-run layout learning exists).

``PagePrint`` merges the two at load; a declared page with no learned
geometry yet still matches, text-only, at a stricter default threshold —
which is what lets capture bootstrap from declarations alone.
"""

import json
import logging
import math
from dataclasses import dataclass
from typing import Any

from physiclaw.common import paths
from physiclaw.common.bbox import Bbox, parse_box, parse_within
from physiclaw.common.logger import write_json_atomic
from physiclaw.common.text import read_text
from physiclaw.conductor.spec import specfile
from physiclaw.conductor.spec.conventions import page_id
from physiclaw.conductor.spec.limits import (
    MAX_ANCHOR_LEN,
    MAX_ANCHORS,
    MAX_FORBID,
    MAX_LANDMARKS,
    MAX_PAGES,
)
from physiclaw.macros.model import MAX_LABEL_READINGS, checked_readings

log = logging.getLogger(__name__)

# Acceptable readings of ONE anchor, canonical included (see `AnchorDecl`).
# A handful covers the real cases — a bilingual label plus a known OCR
# confusion; more than that is usually two anchors wearing one coat.
# The VALUE is the macro layer's one alts-per-target cap (`specfile`
# doctrine): anchors and gesture labels follow the same convention, so
# the two caps can never drift.
MAX_ANCHOR_READINGS = MAX_LABEL_READINGS

# Matching defaults. Learned pages carry their own calibrated threshold;
# declaration-only pages need most anchors present because text is all
# there is to go on. The best-vs-runner-up margin is one constant until
# calibration ever derives a per-page value.
DECL_ONLY_THRESHOLD = 0.75
DEFAULT_MARGIN = 0.15
# How many equal-weight anchors a declaration-only page needs to still
# clear its threshold with ONE missing: (n-1)/n ≥ threshold. Derived,
# never hand-set — the readiness advisory reads it.
MIN_ROBUST_ANCHORS = math.ceil(1 / (1 - DECL_ONLY_THRESHOLD))


class PagesError(specfile.SpecError):
    """A `pages:` section is invalid. Message is user-facing: the
    conductor CLI prints it verbatim. All-or-nothing — a pack failing
    any check is excluded whole, never partially loaded."""


# Shared spec substrate (`specfile`): the macro naming/prose rules bound
# to this spec's error class.
_require_str, _prose, _opt_prose, _check_name = specfile.bind(PagesError)


@dataclass(frozen=True)
class AnchorDecl:
    """One declared identity anchor: the label text that should be on the
    page, optionally pinned to where it must sit (`within`).

    `alts` are further acceptable READINGS of that SAME anchor — any one
    satisfies it, and it still counts ONCE toward the page score. They are
    the authored counterpart of `LearnedAnchor.variants` (the readings
    capture mines on-device), and carry that same field shape on purpose:
    a mixed-locale phone (English system UI, Chinese apps) or a known OCR
    confusion is declared up front instead of waiting for capture to find
    it. The two converge — an alternate that does show up on this device
    gets mined into `variants` as well.

    Alternates must never be written as separate anchors: scoring is a
    weighted fraction over ALL declared anchors, so two spellings of one
    label would halve the page's score on every device — below the
    declaration-only threshold, on a page that is in fact right there.

    `text` stays the canonical reading: learned geometry keys off it, and
    it is the name hits/missing report.
    """

    text: str
    alts: tuple[str, ...] = ()
    # Where the anchor must sit (a band resolved to its box at parse), or
    # None = anywhere. A pinned anchor is chrome — it does not scroll.
    within: "Bbox | None" = None

    @property
    def readings(self) -> tuple[str, ...]:
        """Canonical first, then the alternates — what the matcher tries."""
        return (self.text, *self.alts)


@dataclass(frozen=True)
class PageDecl:
    name: str
    anchors: tuple[AnchorDecl, ...]
    forbid: tuple[str, ...] = ()
    scrollable: bool = False


# The pack-level fixed spots: `landmarks:`, an OPEN vocabulary of named
# spots the author knows — recover hands tap them, agent episodes are
# granted them by name. No name is reserved: what a landmark is for is
# said where it is used. (`MAX_LANDMARKS` caps them, in `limits.py`.)


@dataclass(frozen=True)
class Landmark:
    """One declared landmark — the gesture-target shape ({label, bbox})
    as PACK knowledge: prior the author holds about the app's fixed
    chrome, consumed by recover hands and agent-episode grants (never by
    money paths). `label` is the readings tuple; on-screen text lets the
    tap be located live, a description documents the coordinates.
    `page` (optional) scopes it: an episode is granted the landmark
    only while that page is the verified reading."""

    label: tuple[str, ...]
    bbox: Bbox
    page: str | None = None


@dataclass(frozen=True)
class LearnedAnchor:
    """Captured geometry for one declared anchor on this device."""

    text: str
    cx: float
    cy: float
    pos_tol: float
    freq: float  # fraction of capture observations that contained it
    weight: float  # freq × mean OCR conf ÷ within-app document frequency
    variants: tuple[str, ...] = ()  # OCR misreads observed for this label


@dataclass(frozen=True)
class LearnedPage:
    anchors: dict[str, LearnedAnchor]  # keyed by declared anchor text
    threshold: float
    observations: int


@dataclass(frozen=True)
class PagePrint:
    """One matchable page: declaration merged with whatever geometry has
    been captured (None until the first capture)."""

    app: str
    decl: PageDecl
    learned: LearnedPage | None = None

    @property
    def page_id(self) -> str:
        return page_id(self.app, self.decl.name)

    @property
    def threshold(self) -> float:
        if self.learned is not None:
            return self.learned.threshold
        return DECL_ONLY_THRESHOLD


# ---------- `pages:` parsing ----------


def parse_pages(text: str, app: str) -> dict[str, PageDecl]:
    """Parse + validate one `pages:` section given as YAML text — the
    text-shaped door tests and tooling use; `scan_app_decls` reads the
    live section out of the pack file. Raises PagesError naming the
    offending field; never returns a partially-valid set."""
    data = specfile.load_yaml(text, PagesError)
    return parse_pages_data(data, app)


# The page-declaration field vocabulary, spelled ONCE: `_parse_page`
# validates exactly these keys, `route_decl` decides whether a route
# waypoint declares (vs merely references), and pack.py derives its
# waypoint key set from it — a new page field lands here and reaches
# every door.
PAGE_DECL_FIELDS = ("anchors", "forbid", "scrollable")
# A manifest page's one non-declaration key: the recover hand every
# route of the pack inherits for it (a route may declare its own).
PAGE_RECOVERY_FIELDS = ("recover", "tries")


def recovery_fields(spec: dict) -> dict:
    """The recovery half of a page mapping — its PAGE_RECOVERY_FIELDS
    subset ({} when it declares none). `route_decl`'s mirror: the pack
    door (`collect_page_recovers`) and the route compiler read the
    same slice, so what "declares recovery" means has one home."""
    return {k: spec[k] for k in PAGE_RECOVERY_FIELDS if k in spec}


def collect_page_recovers(doc: dict) -> dict[str, dict]:
    """The RAW recovery the manifest's `pages:` declare, by page name —
    `{recover: hand(s), tries: n}`, whichever keys the page carries —
    resolved by the route compiler (`route._inherited_hands`) against
    the pack's macros and landmarks, so the grammar has one home. Shape
    errors surface at that parse; this only collects."""
    appendix = doc.get("pages")
    if not isinstance(appendix, dict):
        return {}
    out: dict[str, dict] = {}
    for name, spec in appendix.items():
        if not isinstance(spec, dict):
            continue
        picked = recovery_fields(spec)
        if picked:
            out[str(name)] = picked
    return out


def route_decl(entry: dict) -> "dict | None":
    """The declaration half of one route waypoint — its PAGE_DECL_FIELDS
    subset, or None for a bare reference. The ONE predicate for "does
    this waypoint declare": `collect_page_decls` (the pack door) and the
    playbook parser's prepass (the text door) must never disagree on
    it."""
    decl = {k: entry[k] for k in PAGE_DECL_FIELDS if k in entry}
    return decl or None


def collect_page_decls(doc: dict, playbook_docs: dict | None = None) -> dict:
    """The pack's RAW page declarations, wherever they were written: the
    manifest's `pages:` appendix plus every route waypoint carrying
    declaration fields beside its `page:` key, across every playbook
    file. Data-level on purpose — this runs at the pack door
    (`scan_app_decls`, `load_pack`) before any playbook parses, so the
    matcher sees route-declared pages through every door and
    pack.py never re-owns the page grammar. A page is DECLARED
    exactly once per pack; a second site raises with both named — the
    same page declared in two files is a pack error, never a silent
    merge. Malformed playbook shapes are skipped here — each playbook
    excludes itself at its own parse, never the pack."""
    out: dict[str, Any] = {}
    sites: dict[str, str] = {}
    appendix = doc.get("pages")
    if appendix is not None:
        if not isinstance(appendix, dict):
            raise PagesError("`pages` must be a YAML mapping of page name → spec")
        for name, spec in appendix.items():
            # The declaration half only: a manifest page may also carry
            # `recover:`, the pack-level hand (`collect_page_recovers`).
            out[str(name)] = (
                {k: v for k, v in spec.items() if k not in PAGE_RECOVERY_FIELDS}
                if isinstance(spec, dict)
                else spec
            )
            sites[str(name)] = "the manifest's `pages:` section"
    for pb_name, pb in (playbook_docs or {}).items():
        route = pb.get("route") if isinstance(pb, dict) else None
        if not isinstance(route, list):
            continue
        for entry in route:
            if not isinstance(entry, dict) or "page" not in entry:
                continue
            decl = route_decl(entry)
            name = str(entry["page"])
            if decl is None or "." in name:
                # A bare waypoint is a reference, not a declaration; a
                # dotted one names a reserved built-in — its own route's
                # parse refuses the declaration with the exact reason,
                # never the whole pack.
                continue
            site = f"{pb_name}.yml's route"
            if name in out:
                raise PagesError(
                    f"page {name!r} declared twice — in {sites[name]} and on "
                    f"{site}; declare once, reference it bare everywhere else"
                )
            out[name] = decl
            sites[name] = site
    return out


_LANDMARK_KEYS = frozenset({"label", "at", "page"})


def parse_landmarks(data: Any, pages: "set[str] | None" = None) -> dict[str, Landmark]:
    """The `landmarks:` section → validated Landmarks. Each entry is the
    gesture-target shape (`{label, at}` — a macro step's pairing rule,
    at pack level) plus an optional `page:` scope, under any valid
    name; its consumers are the pack's own recover hands and agent
    grants. `pages` (the pack's declared page names) validates the
    scope when given."""
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise PagesError("`landmarks` must be a mapping of name → target")
    if len(data) > MAX_LANDMARKS:
        raise PagesError(f"`landmarks`: {len(data)} entries > max {MAX_LANDMARKS}")
    out: dict[str, Landmark] = {}
    for name, spec in data.items():
        where = f"landmark {name!r}"
        _check_name(name, where)
        if (
            not isinstance(spec, dict)
            or not {"label", "at"} <= set(spec.keys())
            or not set(spec.keys()) <= _LANDMARK_KEYS
        ):
            raise PagesError(f"{where} must be a {{label, at, [page]}} mapping")
        label = checked_readings(spec, where, _require_str, PagesError)
        try:
            left, top, right, bottom = parse_box(spec["at"])
        except (ValueError, TypeError) as e:
            raise PagesError(f"{where}: {e}") from e
        page = spec.get("page")
        if page is not None:
            _check_name(page, f"{where}: `page`")
            if pages is not None and page not in pages:
                known = ", ".join(sorted(pages)) or "(none)"
                raise PagesError(
                    f"{where}: `page` {page!r} is not a declared page of this "
                    f"pack. Declared: {known}"
                )
        out[name] = Landmark(label=label, bbox=(left, top, right, bottom), page=page)
    return out


def pack_landmarks(doc: dict) -> dict[str, Landmark]:
    """The pack document's `landmarks:` section, parsed (empty when
    absent) — with every `page:` scope checked against the pages the
    same document declares, so no door can accept a scope another
    refuses."""
    return parse_landmarks(doc.get("landmarks"), set(collect_page_decls(doc)))


def parse_pages_data(data: Any, app: str) -> dict[str, PageDecl]:
    """The `pages:` section of a pack file → validated declarations."""
    _check_name(app, "app name")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise PagesError("`pages` must be a YAML mapping of page name → spec")
    if len(data) > MAX_PAGES:
        raise PagesError(f"{len(data)} pages > max {MAX_PAGES}")

    out: dict[str, PageDecl] = {}
    for name, spec in data.items():
        _check_name(name, "page name")
        out[name] = _parse_page(name, spec)
    return out


def _parse_page(name: str, spec: Any) -> PageDecl:
    where = f"page `{name}`"
    if not isinstance(spec, dict):
        raise PagesError(f"{where}: spec must be a mapping")
    unknown = sorted(set(spec.keys()) - set(PAGE_DECL_FIELDS))
    if unknown:
        raise PagesError(f"{where}: unknown key(s): {', '.join(map(str, unknown))}")

    anchors = _parse_anchors(spec.get("anchors"), where)

    raw_forbid = spec.get("forbid", [])
    if not isinstance(raw_forbid, list):
        raise PagesError(f"{where}: `forbid` must be a list of strings")
    if len(raw_forbid) > MAX_FORBID:
        raise PagesError(f"{where}: {len(raw_forbid)} forbid terms > max {MAX_FORBID}")
    forbid = tuple(_anchor_text(t, f"{where} forbid") for t in raw_forbid)

    scrollable = spec.get("scrollable", False)
    if not isinstance(scrollable, bool):
        raise PagesError(f"{where}: `scrollable` must be true or false")

    return PageDecl(name=name, anchors=anchors, forbid=forbid, scrollable=scrollable)


def _parse_anchors(raw: Any, where: str) -> tuple[AnchorDecl, ...]:
    """`anchors:` — a LIST of anchors, all of which the page should show
    (scored fractionally, not a hard boolean — the threshold decides).
    Each anchor is a text, a list of alternate readings of ONE text, or
    `{text, within}` with `within` a band (`top`, `bottom`, `left`,
    `right`) or a box — the check shape a macro step's `require:` uses.
    Alternates go INSIDE an anchor, never as separate anchors: each
    declared anchor is a share of the page's score."""
    if isinstance(raw, str):
        raise PagesError(f"{where}: `anchors` is a list — write `anchors: [{raw!r}]`")
    if not isinstance(raw, list):
        raise PagesError(f"{where}: `anchors` must be a list of anchors")
    if not raw:
        raise PagesError(f"{where}: `anchors` must be non-empty")
    if len(raw) > MAX_ANCHORS:
        raise PagesError(f"{where}: {len(raw)} anchors > max {MAX_ANCHORS}")
    return tuple(_parse_anchor(a, where) for a in raw)


def _parse_anchor(raw: Any, where: str) -> AnchorDecl:
    raw_text: Any  # validated (and narrowed to str) by _anchor_text below
    within: Bbox | None = None
    if isinstance(raw, (str, list)):
        raw_text = raw
    elif isinstance(raw, dict):
        unknown = sorted(set(raw.keys()) - {"text", "within"})
        if unknown:
            raise PagesError(
                f"{where}: anchor has unknown key(s): {', '.join(map(str, unknown))}"
            )
        if "text" not in raw:
            raise PagesError(f"{where}: an anchor mapping needs `text`")
        raw_text = raw["text"]
        within = _parse_within(raw.get("within"), f"{where}: anchor `within`")
    else:
        raise PagesError(
            f"{where}: each anchor must be a text, a list of readings, or a "
            "{text, within} mapping"
        )
    # `text:` takes one reading, or a list of alternate readings of the SAME
    # anchor — any one satisfies it, and it counts once (see `AnchorDecl`).
    readings = list(raw_text) if isinstance(raw_text, list) else [raw_text]
    if not readings:
        raise PagesError(f"{where}: anchor `text` list is empty")
    if len(readings) > MAX_ANCHOR_READINGS:
        raise PagesError(
            f"{where}: anchor has {len(readings)} readings > max {MAX_ANCHOR_READINGS}"
        )
    texts: list[str] = []
    for one in readings:
        text = _anchor_text(one, f"{where} anchor")
        # A single character as a whole-screen anchor would match inside almost
        # any label — the macro grammar's rule, for the same reason. Checked
        # per reading: one loose alternate opens the same door as one loose
        # anchor.
        if len(text) == 1 and within is None:
            raise PagesError(
                f"{where}: single-character anchor {text!r} needs a `within`"
            )
        if text in texts:
            raise PagesError(f"{where}: anchor repeats the reading {text!r}")
        texts.append(text)
    # First reading is canonical — the learned-geometry key (see AnchorDecl).
    return AnchorDecl(text=texts[0], alts=tuple(texts[1:]), within=within)


def _parse_within(raw: Any, where: str) -> "Bbox | None":
    """A check's `within:` — a band name or a box, read by the one shared
    parser (`common.bbox.parse_within`); None when absent."""
    if raw is None:
        return None
    try:
        return parse_within(raw)
    except (ValueError, TypeError) as e:
        raise PagesError(f"{where}: {e}") from e


def _anchor_text(value: Any, where: str) -> str:
    text = _require_str(value, f"{where}: text")
    if len(text) > MAX_ANCHOR_LEN:
        raise PagesError(f"{where}: text {len(text)} chars > max {MAX_ANCHOR_LEN}")
    if "".join(text.splitlines()) != text:
        raise PagesError(f"{where}: text must be single-line: {text!r}")
    return text


# ---------- discovery ----------


def scan_app_decls(app: str) -> dict[str, PageDecl]:
    """The declared pages of one app pack — the `pages:` appendix PLUS
    every route-declared waypoint (`collect_page_decls`); {} when the
    pack doesn't exist or declares none. Raises PagesError on a
    malformed file (the CLI surfaces it; runtime callers catch and
    treat the app as undeclared). The name is validated BEFORE any path
    is built from it. Every pack reads the same way, `ios` included —
    declarations live on disk, under the user's hand, never in the
    wheel."""
    _check_name(app, "app name")
    doc = specfile.load_pack_doc(app, PagesError)
    if doc is None:
        return {}
    docs, _errors = specfile.load_playbook_docs(app, PagesError)
    return parse_pages_data(collect_page_decls(doc, docs), app)


# ---------- learned store ----------

_LEARNED_SCHEMA = 1


def load_learned(app: str) -> dict[str, LearnedPage]:
    """The captured geometry for one app — {} on missing/unreadable/stale
    file (fail-open: a bad learned file degrades to declaration-only
    matching, never takes a session down)."""
    p = paths.learned_pages_dir() / f"{app}.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(read_text(p))
        if data.get("schema") != _LEARNED_SCHEMA:
            log.warning("learned pages %s: unknown schema; ignoring", p)
            return {}
        out: dict[str, LearnedPage] = {}
        for name, lp in data["pages"].items():
            anchors = {
                a["text"]: LearnedAnchor(
                    text=a["text"],
                    cx=float(a["cx"]),
                    cy=float(a["cy"]),
                    pos_tol=float(a["pos_tol"]),
                    freq=float(a["freq"]),
                    weight=float(a["weight"]),
                    variants=tuple(a.get("variants", ())),
                )
                for a in lp["anchors"]
            }
            out[name] = LearnedPage(
                anchors=anchors,
                threshold=float(lp["threshold"]),
                observations=int(lp["observations"]),
            )
        return out
    except Exception:
        log.warning("learned pages %s unreadable; ignoring", p, exc_info=True)
        return {}


def save_learned(app: str, pages: dict[str, LearnedPage]) -> None:
    paths.learned_pages_dir().mkdir(parents=True, exist_ok=True)
    obj = {
        "schema": _LEARNED_SCHEMA,
        "app": app,
        "pages": {
            name: {
                "threshold": lp.threshold,
                "observations": lp.observations,
                "anchors": [
                    {
                        "text": a.text,
                        "cx": a.cx,
                        "cy": a.cy,
                        "pos_tol": a.pos_tol,
                        "freq": a.freq,
                        "weight": a.weight,
                        "variants": list(a.variants),
                    }
                    for a in lp.anchors.values()
                ],
            }
            for name, lp in pages.items()
        },
    }
    write_json_atomic(paths.learned_pages_dir() / f"{app}.json", obj)


def prints_for_app(
    app: str, decls: dict[str, PageDecl] | None = None
) -> list[PagePrint]:
    """Declarations merged with learned geometry — the matcher's candidate
    set for one app. Callers already holding the Pack pass
    `decls=pack.pages` so the spec file is not re-read (the
    `scan_playbooks` rule)."""
    if decls is None:
        decls = scan_app_decls(app)
    learned = load_learned(app)
    return [
        PagePrint(app=app, decl=d, learned=learned.get(name))
        for name, d in decls.items()
    ]
