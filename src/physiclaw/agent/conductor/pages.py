"""Page declarations + learned geometry — the two halves of a fingerprint.

A page fingerprint is split by audience:

  - DECLARATIONS (``playbooks/<app>/pages.yml``, human-authored): the page's
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

import io
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from ruamel.yaml import YAML

from physiclaw.common import paths
from physiclaw.common.logger import write_json_atomic
from physiclaw.common.text import read_text

log = logging.getLogger(__name__)

PAGES_FILENAME = "pages.yml"

MAX_PAGES = 30
MAX_ANCHORS = 12
MAX_FORBID = 8
MAX_ANCHOR_LEN = 80

# Page and app names follow the macro/skill folder convention (the same
# rule as `agent/macros/model.py check_name` — kept in step by the shared
# spelling and the 64-char cap; a cross-package call would invert the
# dependency direction the conductor just shed).
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_NAME_LEN = 64

# Reserved app namespaces: system pages (OS states, the user-channel IM
# pages) resolve against built-ins shipped in-tree — a pack must never
# redefine or shadow them.
RESERVED_APPS = frozenset({"ios", "channel"})

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

_yaml = YAML(typ="safe", pure=True)


class PagesError(ValueError):
    """A pages.yml is invalid. Message is user-facing: the conductor CLI
    prints it verbatim. All-or-nothing — a file failing any check is
    excluded whole, never partially loaded."""


@dataclass(frozen=True)
class AnchorDecl:
    """One declared identity anchor: a label text that should be on the
    page, optionally pinned to a coarse region band."""

    text: str
    region: str | None = None  # key into REGIONS


@dataclass(frozen=True)
class PageDecl:
    name: str
    anchors: tuple[AnchorDecl, ...]
    forbid: tuple[str, ...] = ()
    scrollable: bool = False


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
        return f"{self.app}.{self.decl.name}"

    @property
    def threshold(self) -> float:
        if self.learned is not None:
            return self.learned.threshold
        return DECL_ONLY_THRESHOLD


# ---------- pages.yml parsing ----------


def parse_pages(text: str, app: str) -> dict[str, PageDecl]:
    """Parse + validate one pages.yml. Raises PagesError naming the
    offending field; never returns a partially-valid set."""
    _check_name(app, "app name")
    try:
        data = _yaml.load(io.StringIO(text))
    except Exception as e:
        # Deliberately broad, like macro discovery: the YAML loader does
        # not confine itself to YAMLError (deep nesting surfaces as
        # RecursionError), and this module's contract is PagesError-only.
        # BaseException still propagates.
        raise PagesError(f"invalid YAML: {e or type(e).__name__}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise PagesError("pages.yml must be a YAML mapping of page name → spec")
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
    unknown = sorted(set(spec.keys()) - {"anchors", "forbid", "scrollable"})
    if unknown:
        raise PagesError(f"{where}: unknown key(s): {', '.join(map(str, unknown))}")

    raw_anchors = spec.get("anchors")
    if not isinstance(raw_anchors, list) or not raw_anchors:
        raise PagesError(f"{where}: `anchors` must be a non-empty list")
    if len(raw_anchors) > MAX_ANCHORS:
        raise PagesError(f"{where}: {len(raw_anchors)} anchors > max {MAX_ANCHORS}")
    anchors = tuple(_parse_anchor(a, where) for a in raw_anchors)

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


def _parse_anchor(raw: Any, where: str) -> AnchorDecl:
    if isinstance(raw, str):
        text, region = raw, None
    elif isinstance(raw, dict):
        unknown = sorted(set(raw.keys()) - {"text", "region"})
        if unknown:
            raise PagesError(
                f"{where}: anchor has unknown key(s): {', '.join(map(str, unknown))}"
            )
        text = raw.get("text")
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
    text = _anchor_text(text, f"{where} anchor")
    # A single character as a whole-screen anchor would match inside almost
    # any label — the macro grammar's rule, for the same reason.
    if len(text) == 1 and region is None:
        raise PagesError(f"{where}: single-character anchor {text!r} needs a `region`")
    return AnchorDecl(text=text, region=region)


def _anchor_text(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise PagesError(
            f"{where}: text must be a string, got {type(value).__name__} "
            f"({value!r}) — quote it"
        )
    text = value.strip()
    if not text:
        raise PagesError(f"{where}: text must not be blank")
    if len(text) > MAX_ANCHOR_LEN:
        raise PagesError(f"{where}: text {len(text)} chars > max {MAX_ANCHOR_LEN}")
    if "".join(text.splitlines()) != text:
        raise PagesError(f"{where}: text must be single-line: {text!r}")
    return text


def _check_name(name, where: str) -> None:
    if not isinstance(name, str) or len(name) > MAX_NAME_LEN or not NAME_RE.match(name):
        raise PagesError(
            f"{where} {name!r} must be lowercase letters/digits/hyphens, "
            f"≤{MAX_NAME_LEN} chars"
        )


# ---------- discovery ----------


def scan_app_decls(app: str) -> dict[str, PageDecl]:
    """The declared pages of one app pack — {} when the pack or its
    pages.yml doesn't exist. Raises PagesError on a malformed file (the
    CLI surfaces it; runtime callers catch and treat the app as
    undeclared). The name is validated BEFORE any path is built from it."""
    _check_name(app, "app name")
    if app in RESERVED_APPS:
        raise PagesError(f"{app!r} is a reserved namespace (built-in pages)")
    p = paths.playbooks_dir() / app / PAGES_FILENAME
    if not p.exists():
        return {}
    return parse_pages(read_text(p), app)


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


def prints_for_app(app: str) -> list[PagePrint]:
    """Declarations merged with learned geometry — the matcher's candidate
    set for one app."""
    decls = scan_app_decls(app)
    learned = load_learned(app)
    return [
        PagePrint(app=app, decl=d, learned=learned.get(name))
        for name, d in decls.items()
    ]
