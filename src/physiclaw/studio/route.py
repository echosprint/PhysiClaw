"""Route assembly brains — playbook drafts, live validation through the
real route compiler, commit as ONE key under `playbooks:`, rehearse
arming.

Unlike page and macro drafting, route mutations are UNCHECKED: a route
mid-assembly is usually invalid (it ends without its landing page, an
ask has no message yet), so refusing would fight the editor. Instead
every draft reply carries `validation` — the verbatim `parse_playbook`
error per playbook, or None when it compiles — and commit refuses
while any remains. Entries are stored exactly as PLAYBOOK.yml spells
them, so emission is a dump and validation is the real door.
"""

import io

from ruamel.yaml import YAML

from physiclaw.common import paths
from physiclaw.common.paths import PACK_FILENAME
from physiclaw.common.text import read_text, write_text
from physiclaw.conductor import _spec
from physiclaw.conductor.playbook import (
    PlaybookError,
    check_name,
    load_pack,
    parse_playbook,
    scan_playbooks,
)
from physiclaw.studio.curate import splice_playbook
from physiclaw.studio.draft import DraftError
from physiclaw.studio.record import macro_data

# Round-trip dumper for emission: block style, 2-space indent — plain
# dicts come out as readable YAML, and strings that look like syntax
# (`{inputs.q}`) get quoted exactly as needed to load back verbatim.
_yaml = YAML()
_yaml.default_flow_style = False


def _playbook(draft: dict, name: str) -> dict:
    p = draft.setdefault("playbooks", {}).get(name)
    if p is None:
        raise DraftError(f"no drafted playbook {name!r}")
    return p


def add_playbook(draft: dict, name: str) -> None:
    books = draft.setdefault("playbooks", {})
    if name in books:
        raise DraftError(f"playbook {name!r} already drafted")
    try:
        check_name(name, "playbook name")
    except PlaybookError as e:
        raise DraftError(str(e)) from e
    books[name] = {
        "description": "",
        "inputs": {},
        "route": [],
    }


def delete_playbook(draft: dict, name: str) -> None:
    _playbook(draft, name)
    del draft["playbooks"][name]


def update_playbook(draft: dict, name: str, fields: dict) -> None:
    """Merge top-level fields (description/inputs) —
    unchecked; live validation is the feedback."""
    p = _playbook(draft, name)
    unknown = sorted(set(fields) - {"description", "inputs"})
    if unknown:
        raise DraftError(f"playbook field(s) not editable: {', '.join(unknown)}")
    p.update(fields)


def entry_insert(draft: dict, name: str, index: "int | None", entry: dict) -> None:
    p = _playbook(draft, name)
    if not isinstance(entry, dict):
        raise DraftError("a route entry must be a mapping")
    if index is None:
        index = len(p["route"])
    if not (0 <= index <= len(p["route"])):
        raise DraftError(f"cannot insert at {index}")
    p["route"].insert(index, entry)


def entry_update(draft: dict, name: str, index: int, entry: dict) -> None:
    p = _playbook(draft, name)
    if not isinstance(entry, dict):
        raise DraftError("a route entry must be a mapping")
    if not (0 <= index < len(p["route"])):
        raise DraftError(f"no route entry {index}")
    p["route"][index] = entry


def entry_delete(draft: dict, name: str, index: int) -> None:
    p = _playbook(draft, name)
    if not (0 <= index < len(p["route"])):
        raise DraftError(f"no route entry {index}")
    del p["route"][index]


def entry_move(draft: dict, name: str, index: int, delta: int) -> None:
    p = _playbook(draft, name)
    route = p["route"]
    j = index + delta
    if not (0 <= index < len(route)) or not (0 <= j < len(route)):
        raise DraftError(f"cannot move entry {index} by {delta}")
    route[index], route[j] = route[j], route[index]


def entry_from_macro(draft: dict, macro_name: str) -> dict:
    """A `do` entry embedding one drafted macro inline (the default —
    the walk reads top-down without digging into other files), its
    `with:` prefilled to same-named playbook inputs."""
    m = draft.get("macros", {}).get(macro_name)
    if m is None:
        raise DraftError(f"no drafted macro {macro_name!r}")
    entry: dict = {"do": macro_name, "macro": macro_data(m)}
    if m["inputs"]:
        entry["with"] = {i: f"{{inputs.{i}}}" for i in m["inputs"]}
    return entry


# ---------- emission / validation ----------


def playbook_data(p: dict) -> dict:
    """The draft as one `playbooks:` map entry, keys in template order.
    Always `enabled: false` — rehearse-then-enable is a human act."""
    out: dict = {"description": p["description"] or "drafted in the studio"}
    out["enabled"] = False
    if p["inputs"]:
        out["inputs"] = p["inputs"]
    out["route"] = p["route"]
    return out


def _dump(data) -> str:
    buf = io.StringIO()
    _yaml.dump(data, buf)
    return buf.getvalue()


def emit_playbook_yaml(p: dict) -> str:
    """The playbook body alone — what `parse_playbook` validates."""
    return _dump(playbook_data(p))


def emit_playbook_block(name: str, p: dict) -> str:
    """The same body as a `playbooks:` map entry, ready to splice."""
    return _dump({name: playbook_data(p)})


def editor_feedback(draft: dict) -> dict:
    """The route editor's whole feedback from ONE pack load: per
    drafted playbook the verbatim compiler error (None = compiles
    against the COMMITTED pack plus anything declared in place), and
    the pack's page/macro names for the pickers."""
    books = draft.get("playbooks", {})
    try:
        pack = load_pack(draft["app"])
    except PlaybookError as e:
        msg = f"pack not loadable yet — commit pages first ({e})"
        return {
            "validation": {name: msg for name in books},
            "pack": {"pages": [], "macros": [], "error": str(e)},
        }
    validation: dict[str, str | None] = {}
    for name, p in books.items():
        try:
            parse_playbook(emit_playbook_yaml(p), name, pack)
            validation[name] = None
        except PlaybookError as e:
            validation[name] = str(e)
    return {
        "validation": validation,
        "pack": {
            "pages": sorted(pack.pages),
            "macros": sorted(pack.macros),
            "error": None,
        },
    }


def validation(draft: dict) -> dict:
    """The compiler-error half alone (tests and tooling)."""
    return editor_feedback(draft)["validation"]


def commit_playbook(app: str, name: str, p: dict) -> dict:
    """Write the drafted playbook as one key under `playbooks:` — only
    when it compiles (the live-validation error is the refusal), and
    post-checked through `scan_playbooks` like `playbooks check`."""
    try:
        pack = load_pack(app)
    except PlaybookError as e:
        raise DraftError(f"pack not loadable — commit pages first ({e})") from e
    try:
        parse_playbook(emit_playbook_yaml(p), name, pack)
    except PlaybookError as e:
        raise DraftError(f"commit refused — {e}") from e

    pack_file = paths.pack_root(app) / PACK_FILENAME
    text = splice_playbook(read_text(pack_file), name, emit_playbook_block(name, p))
    # The whole file must still parse BEFORE a byte lands — a splice
    # surprise must refuse, never write a broken pack.
    doc = _spec.load_yaml(text, PlaybookError)
    if not isinstance(doc.get("playbooks"), dict) or name not in doc["playbooks"]:
        raise DraftError("commit refused — the spliced `playbooks:` lost the entry")
    write_text(pack_file, text)

    fresh = load_pack(app)
    entry = next((e for e in scan_playbooks(app, fresh) if e.name == name), None)
    check = (
        "ok"
        if entry is not None and entry.spec is not None
        else (entry.error if entry is not None else "playbook missing after write")
    )
    return {"pack_file": str(pack_file), "playbook": name, "check": check}
