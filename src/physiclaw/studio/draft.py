"""The studio's per-app authoring draft — declarations-in-progress,
captured shot listings, and landmark picks, under `paths.studio_dir()`.

Layout: `<studio>/<app>/draft.json` plus `shots/<id>.jpg`. Every
mutation validates through the REAL pack doors (`parse_pages_data`,
`parse_landmarks`) so a draft that saves is a draft that will commit —
their verbatim errors are the rail's inline feedback. Unlike the
learned store this is NOT fail-open: a draft is user work, and an
unreadable file must be a loud error, never a silent fresh start.
"""

import base64
import json
import shutil
from pathlib import Path

from physiclaw.common import paths
from physiclaw.common.logger import write_json_atomic
from physiclaw.common.text import read_text
from physiclaw.conductor.pages import parse_landmarks, parse_pages_data

_DRAFT_SCHEMA = 1


class DraftError(ValueError):
    """A refused draft mutation or an unreadable draft file — the
    message is user-facing (the rail shows it verbatim)."""


def draft_dir(app: str) -> Path:
    return paths.studio_dir() / app


def _fresh(app: str) -> dict:
    return {
        "schema": _DRAFT_SCHEMA,
        "app": app,
        "pages": {},
        "shots": {},
        "landmarks": {},
        "next_shot": 1,
        # Later-added fields ride the same schema (the alpha rule:
        # additive fields, older drafts read them via setdefault/.get).
        "macros": {},
        "recording": None,
        "playbooks": {},
    }


def load_draft(app: str) -> dict:
    p = draft_dir(app) / "draft.json"
    if not p.exists():
        return _fresh(app)
    try:
        data = json.loads(read_text(p))
    except (OSError, json.JSONDecodeError) as e:
        raise DraftError(f"draft {p} unreadable: {e}") from e
    if data.get("schema") != _DRAFT_SCHEMA:
        raise DraftError(f"draft {p}: unknown schema {data.get('schema')!r}")
    return data


def save_draft(app: str, draft: dict) -> None:
    d = draft_dir(app)
    d.mkdir(parents=True, exist_ok=True)
    write_json_atomic(d / "draft.json", draft)


# ---------- mutations ----------
# Each takes the loaded draft, validates, mutates in place; the caller
# saves. Page-shaped fields are stored exactly as PLAYBOOK.yml spells
# them (a string or {text, region} per anchor), so validation IS the
# pack parser and emission is verbatim.


def decl_data(draft: dict) -> dict:
    """The draft's pages as `pages:`-shaped data (decl fields only)."""
    return {
        name: {
            k: v
            for k, v in page.items()
            if k in ("anchors", "forbid", "scrollable") and v not in ([], False)
        }
        for name, page in draft["pages"].items()
    }


def _validate_pages(draft: dict) -> None:
    parse_pages_data(decl_data(draft), draft["app"])


def add_page(draft: dict, name: str) -> None:
    if name in draft["pages"]:
        raise DraftError(f"page {name!r} already drafted")
    # An empty page is mid-authoring, not an error — anchors are checked
    # when they arrive and again at commit; only the name is checked now,
    # BEFORE the insert (no rollback to get wrong).
    parse_pages_data({name: {"anchors": ["placeholder"]}}, draft["app"])
    draft["pages"][name] = {
        "anchors": [],
        "forbid": [],
        "scrollable": False,
        "shots": [],
    }


def _page(draft: dict, name: str) -> dict:
    page = draft["pages"].get(name)
    if page is None:
        raise DraftError(f"no drafted page {name!r}")
    return page


def update_page(
    draft: dict,
    name: str,
    *,
    anchors: list | None = None,
    forbid: list | None = None,
    scrollable: bool | None = None,
) -> None:
    page = _page(draft, name)
    before = {k: page[k] for k in ("anchors", "forbid", "scrollable")}
    if anchors is not None:
        page["anchors"] = anchors
    if forbid is not None:
        page["forbid"] = forbid
    if scrollable is not None:
        page["scrollable"] = scrollable
    if not page["anchors"]:
        return  # still mid-authoring; commit enforces non-empty
    try:
        _validate_pages(draft)
    except Exception:
        page.update(before)
        raise


def delete_page(draft: dict, name: str) -> None:
    page = _page(draft, name)
    for shot_id in list(page["shots"]):
        delete_snap(draft, shot_id)
    del draft["pages"][name]


def add_shot(draft: dict, page_name: str, listing: str, jpeg_b64: str) -> str:
    """Attach one observation (listing + JPEG) to a drafted page."""
    page = _page(draft, page_name)
    if not listing.strip():
        raise DraftError("shot has an empty listing — the camera read failed")
    shot_id = save_snap(draft, jpeg_b64, listing, page=page_name)
    page["shots"].append(shot_id)
    return shot_id


def save_snap(
    draft: dict, jpeg_b64: str, listing: str, page: "str | None" = None
) -> str:
    """One snapshot into the ONE registry (`draft['shots']`): JPEG on
    disk, listing beside the owner ref. Page observations carry their
    page; macro-step snapshots carry `page: None` — same pool, same
    serving routes, same cleanup (`delete_snap`)."""
    shot_id = f"s{draft['next_shot']}"
    draft["next_shot"] += 1
    shots = draft_dir(draft["app"]) / "shots"
    shots.mkdir(parents=True, exist_ok=True)
    (shots / f"{shot_id}.jpg").write_bytes(base64.b64decode(jpeg_b64))
    draft["shots"][shot_id] = {"page": page, "listing": listing}
    return shot_id


def delete_snap(draft: dict, shot_id: "str | None") -> None:
    """The one snapshot-removal spelling: registry entry + file.
    None (a step with no snap) is a no-op."""
    if not shot_id:
        return
    draft["shots"].pop(shot_id, None)
    (draft_dir(draft["app"]) / "shots" / f"{shot_id}.jpg").unlink(missing_ok=True)


def delete_shot(draft: dict, shot_id: str) -> None:
    shot = draft["shots"].get(shot_id)
    if shot is None or shot["page"] is None:
        raise DraftError(f"no page shot {shot_id!r}")
    _page(draft, shot["page"])["shots"].remove(shot_id)
    delete_snap(draft, shot_id)


def shot_jpeg(app: str, shot_id: str) -> Path:
    p = draft_dir(app) / "shots" / f"{shot_id}.jpg"
    if not p.exists():
        raise DraftError(f"no shot image {shot_id!r}")
    return p


def set_landmark(draft: dict, name: str, label, bbox: list) -> None:
    """Draft one fixed spot, validated through the pack's own open
    `landmarks:` door (any valid name; the commit writes that section)."""
    before = dict(draft["landmarks"])
    draft["landmarks"][name] = {"label": label, "bbox": bbox}
    try:
        parse_landmarks(draft["landmarks"])
    except Exception:
        draft["landmarks"] = before
        raise


def clear_landmark(draft: dict, name: str) -> None:
    if draft["landmarks"].pop(name, None) is None:
        raise DraftError(f"no landmark {name!r} drafted")


def discard(app: str) -> None:
    """Abandon the whole draft (directory and shots)."""
    shutil.rmtree(draft_dir(app), ignore_errors=True)
