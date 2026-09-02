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
from dataclasses import dataclass
from typing import Any

from physiclaw.common import paths
from physiclaw.common.bbox import Bbox, validate_bbox
from physiclaw.common.logger import write_json_atomic
from physiclaw.common.text import read_text
from physiclaw.conductor import _spec
from physiclaw.macros.model import AND, MAX_LABEL_READINGS, OR, checked_readings

log = logging.getLogger(__name__)

MAX_PAGES = 30
MAX_ANCHORS = 12
MAX_FORBID = 8
MAX_ANCHOR_LEN = 80
# Acceptable readings of ONE anchor, canonical included (see `AnchorDecl`).
# A handful covers the real cases — a bilingual label plus a known OCR
# confusion; more than that is usually two anchors wearing one coat.
# The VALUE is the macro layer's one alts-per-target cap (`_spec`
# doctrine): anchors and gesture labels follow the same convention, so
# the two caps can never drift.
MAX_ANCHOR_READINGS = MAX_LABEL_READINGS


# Reserved app namespaces a pack's page refs may cross into. Neither is
# a task pack, and both are scaffolded into playbooks/<app>/ like any
# other: `channel` is the user-channel IM pages + send macros, `ios` is
# OS-level state. Playbooks never name either — the conductor reaches
# them through node types, which is what "reserved" buys.
CHANNEL_APP = "channel"

# The OS-state pack. Same shape as any other pack — a scaffolded
# `playbooks/ios/` the user owns and edits — but the conductor knows its
# name, because telling "locked" from "a screen I don't recognize" is
# its own job, not a playbook's. `scaffold.IOS_PAGES_STUB` is the
# starting text; absent (never scaffolded) simply means the conductor
# cannot name those states, and every one of them reads as unknown.
IOS_APP = "ios"

# Derived, never re-spelled: renaming either constant must not leave this
# set pointing at the old string.
RESERVED_APPS = frozenset({CHANNEL_APP, IOS_APP})

# The channel pack's conventions — the three names the conductor knows.
# Declared HERE (beside CHANNEL_APP) because both channel.py and
# scaffold.py need them and scaffold must not import channel: the stubs
# interpolate these, never hand-copy them.
THREAD_PAGE = "thread"  # the channel pack's `pages:` must declare this
SEND_MACRO = "send"  # nav to the user's thread + paste + send {message}
# `open` serves twice, one spelling: the channel's (nav to the thread —
# resume reads) and every TASK pack's (launch ITS app to a home state —
# the rescue ladder's reset rung re-enters through it after force_quit;
# a pack without one simply skips that rung).
OPEN_MACRO = "open"

# The ios pack's one convention, here for the same reason: the boot
# matches this page and `scaffold.IOS_PAGES_STUB` declares it, so the two
# interpolate one constant instead of both spelling "locked".
LOCKED_PAGE = "locked"


def page_id(app: str, page: str) -> str:
    """The `app.page` spelling — the ONE format matcher verdicts carry
    and every expectation is compared against."""
    return f"{app}.{page}"


# The full ids, precomputed: what the gate's peeks and the boot match
# against.
THREAD_ID = page_id(CHANNEL_APP, THREAD_PAGE)
LOCKED_ID = page_id(IOS_APP, LOCKED_PAGE)


# Coarse region hints — bands, not bboxes: exact geometry is learned, the
# hint only disambiguates before capture and pins chrome that must not
# drift (a tab bar found mid-screen is not the tab bar).
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "top": (0.0, 0.0, 1.0, 0.25),
    "bottom": (0.0, 0.75, 1.0, 1.0),
    "left": (0.0, 0.0, 0.3, 1.0),
    "right": (0.7, 0.0, 1.0, 1.0),
}

# Matching defaults. Learned pages carry their own calibrated threshold;
# declaration-only pages need most anchors present because text is all
# there is to go on. The best-vs-runner-up margin is one constant until
# calibration ever derives a per-page value.
DECL_ONLY_THRESHOLD = 0.75
DEFAULT_MARGIN = 0.15


class PagesError(ValueError):
    """A `pages:` section is invalid. Message is user-facing: the
    conductor CLI prints it verbatim. All-or-nothing — a pack failing
    any check is excluded whole, never partially loaded."""


# Shared spec substrate (`_spec`): the macro naming/prose rules bound
# to this spec's error class.
_require_str, _prose, _opt_prose, _check_name = _spec.bind(PagesError)


@dataclass(frozen=True)
class AnchorDecl:
    """One declared identity anchor: the label text that should be on the
    page, optionally pinned to a coarse region band.

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
    region: str | None = None  # key into REGIONS

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
# granted them by name, and the two names below are the ones the rescue
# ladder itself consults when a pack declares them.
LANDMARK_BACK = "back"  # the app's own back affordance (iOS: top-left chevron)
LANDMARK_DISMISS = "dismiss"  # empty scrim area a modal/sheet dismisses from
MAX_LANDMARKS = 12


@dataclass(frozen=True)
class Landmark:
    """One declared landmark — the gesture-target shape ({label, bbox})
    as PACK knowledge: prior the author holds about the app's fixed
    chrome, consumed by recover hands and agent-episode grants (never by
    money paths). `label` is the readings tuple; on-screen text lets the
    tap be located live, a description documents the coordinates."""

    label: tuple[str, ...]
    bbox: Bbox


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
    data = _spec.load_yaml(text, PagesError)
    return parse_pages_data(data, app)


# The page-declaration field vocabulary, spelled ONCE: `_parse_page`
# validates exactly these keys, `route_decl` decides whether a route
# waypoint declares (vs merely references), and playbook.py derives its
# waypoint key set from it — a new page field lands here and reaches
# every door.
PAGE_DECL_FIELDS = ("anchors", "forbid", "scrollable")


def route_decl(entry: dict) -> "dict | None":
    """The declaration half of one route waypoint — its PAGE_DECL_FIELDS
    subset, or None for a bare reference. The ONE predicate for "does
    this waypoint declare": `collect_page_decls` (the pack door) and the
    playbook parser's prepass (the text door) must never disagree on
    it."""
    decl = {k: entry[k] for k in PAGE_DECL_FIELDS if k in entry}
    return decl or None


def collect_page_decls(doc: dict) -> dict:
    """The pack's RAW page declarations, wherever they were written: the
    `pages:` appendix plus every playbook-route waypoint carrying
    declaration fields beside its `page:` key. Data-level on purpose —
    this runs at the pack door (`scan_app_decls`, `load_pack`) before
    any playbook parses, so the matcher sees route-declared pages
    through every door and playbook.py never re-owns the page grammar.
    A page is DECLARED exactly once per pack; a second site raises with
    both named. Malformed playbook shapes are skipped here — each
    playbook excludes itself at its own parse, never the pack."""
    out: dict[str, Any] = {}
    sites: dict[str, str] = {}
    appendix = doc.get("pages")
    if appendix is not None:
        if not isinstance(appendix, dict):
            raise PagesError("`pages` must be a YAML mapping of page name → spec")
        for name, spec in appendix.items():
            out[str(name)] = spec
            sites[str(name)] = "the `pages:` section"
    raw_playbooks = doc.get("playbooks")
    if not isinstance(raw_playbooks, dict):
        return out
    for pb_name, pb in raw_playbooks.items():
        route = pb.get("route") if isinstance(pb, dict) else None
        if not isinstance(route, list):
            continue
        for entry in route:
            if not isinstance(entry, dict) or "page" not in entry:
                continue
            decl = route_decl(entry)
            if decl is None:
                continue  # bare waypoint — a reference, not a declaration
            name = str(entry["page"])
            site = f"playbook {pb_name!r}'s route"
            if name in out:
                raise PagesError(
                    f"page {name!r} declared twice — in {sites[name]} and on "
                    f"{site}; declare once, reference it bare everywhere else"
                )
            out[name] = decl
            sites[name] = site
    return out


def parse_landmarks(data: Any) -> dict[str, Landmark]:
    """The `landmarks:` section → validated Landmarks. Each entry is the
    gesture-target shape (`{label, bbox}` — the macro grammar's pairing
    rule, at pack level) under any valid name; its consumers are the
    pack's own recover hands and agent grants."""
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
        if not isinstance(spec, dict) or set(spec.keys()) != {"label", "bbox"}:
            raise PagesError(f"{where} must be a {{label, bbox}} mapping")
        label = checked_readings(spec, where, _require_str, PagesError)
        try:
            left, top, right, bottom = map(float, validate_bbox(spec["bbox"]))
        except ValueError as e:
            raise PagesError(f"{where}: {e}") from e
        out[name] = Landmark(label=label, bbox=(left, top, right, bottom))
    return out


def pack_landmarks(doc: dict) -> dict[str, Landmark]:
    """The pack document's `landmarks:` section, parsed (empty when absent)."""
    return parse_landmarks(doc.get("landmarks"))


def parse_pages_data(data, app: str) -> dict[str, PageDecl]:
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

    anchors = _parse_anchor_clause(spec.get("anchors"), where)

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


def _parse_anchor_clause(raw: Any, where: str) -> tuple[AnchorDecl, ...]:
    """`anchors:` — ONE clause, the macro guard grammar's shapes: a bare
    string or one `{text|or, region}` dict is a single anchor; `{and:
    [clause, ...]}` is the multi-anchor set (scored fractionally, not a
    hard boolean — the threshold decides); `or:` inside a clause lists
    alternate READINGS of one anchor (any hit scores it once). The
    legacy spelling — a top-level list of anchor entries — still parses
    as the implicit `and`."""
    if isinstance(raw, list):
        items = raw  # legacy: a list IS the and-set
    elif isinstance(raw, dict) and AND in raw:
        unknown = sorted(set(raw.keys()) - {AND})
        if unknown:
            raise PagesError(
                f"{where}: `anchors.{AND}` takes no sibling key(s): "
                f"{', '.join(map(str, unknown))}"
            )
        items = raw[AND]
        if not isinstance(items, list):
            raise PagesError(f"{where}: `anchors.{AND}` must be a list of anchors")
    elif isinstance(raw, (str, dict)):
        items = [raw]  # a single clause is the whole set
    else:
        raise PagesError(
            f"{where}: `anchors` must be a clause (a string, a "
            "{text|or, region} mapping, or {and: [...]}) or a list"
        )
    if not items:
        raise PagesError(f"{where}: `anchors` must be non-empty")
    if len(items) > MAX_ANCHORS:
        raise PagesError(f"{where}: {len(items)} anchors > max {MAX_ANCHORS}")
    return tuple(_parse_anchor(a, where) for a in items)


def _parse_anchor(raw: Any, where: str) -> AnchorDecl:
    raw_text: Any  # validated (and narrowed to str) by _anchor_text below
    if isinstance(raw, str):
        raw_text, region = raw, None
    elif isinstance(raw, dict):
        if AND in raw:
            raise PagesError(f"{where}: `{AND}` does not nest inside an anchor")
        unknown = sorted(set(raw.keys()) - {"text", OR, "region"})
        if unknown:
            raise PagesError(
                f"{where}: anchor has unknown key(s): {', '.join(map(str, unknown))}"
            )
        if "text" in raw and OR in raw:
            raise PagesError(
                f"{where}: an anchor takes `text` or `{OR}`, not both — "
                f"`{OR}` IS the readings list"
            )
        # `or:` is the clause grammar's spelling of alternate readings —
        # the same meaning the legacy list-under-`text:` carried.
        raw_text = raw[OR] if OR in raw else raw.get("text")
        if OR in raw and not isinstance(raw_text, list):
            raise PagesError(f"{where}: anchor `{OR}` must be a list of readings")
        region = raw.get("region")
        if region is not None and region not in REGIONS:
            raise PagesError(
                f"{where}: anchor region {region!r} must be one of "
                f"{', '.join(sorted(REGIONS))}"
            )
    else:
        raise PagesError(
            f"{where}: each anchor must be a string or {{text, region}} mapping"
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
        if len(text) == 1 and region is None:
            raise PagesError(
                f"{where}: single-character anchor {text!r} needs a `region`"
            )
        if text in texts:
            raise PagesError(f"{where}: anchor repeats the reading {text!r}")
        texts.append(text)
    # First reading is canonical — the learned-geometry key (see AnchorDecl).
    return AnchorDecl(text=texts[0], alts=tuple(texts[1:]), region=region)


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
    doc = _spec.load_pack_doc(app, PagesError)
    if doc is None:
        return {}
    return parse_pages_data(collect_page_decls(doc), app)


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
