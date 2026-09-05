"""The pack door — `playbooks/<app>/` on disk → validated `Playbook`s.

A pack is a folder: `PLAYBOOK.yml`, the MANIFEST (what the app is and
what its routes share — meta, placeholders, landmarks, pages; every
section optional, the file may be empty), one `<name>.yml` per
playbook beside it (the stem is the name, referenced as `<app>/<name>`,
and `name:` must agree with it), and `macros/<name>/MACRO.yml` for the
recorded hands routes share. The manifest never carries a route.

This module loads and scans packs, spells the qualified `app/name`
dispatch key every macro site shares, and holds the live rule a wake
requires of a playbook. The model is `model.py`; the compiler is
`route.py`.
"""

from typing import Any

from physiclaw.common import paths
from physiclaw.common.paths import PACK_FILENAME
from physiclaw.conductor.spec import scaffold, specfile
from physiclaw.conductor.spec.conventions import CHANNEL_APP
from physiclaw.conductor.spec.model import (
    PACK_MACROS_DIRNAME,
    AgentNode,
    AskNode,
    DoNode,
    Pack,
    Playbook,
    PlaybookEntry,
    PlaybookError,
    PlaybookInput,
    check_name,
    prose,
    require_str,
)
from physiclaw.conductor.spec.pages import (
    PagesError,
    collect_page_decls,
    collect_page_recovers,
    pack_landmarks,
    parse_pages_data,
)
from physiclaw.conductor.spec.route import compile_route
from physiclaw.macros import parse as macro_parse
from physiclaw.macros import store as macro_store
from physiclaw.macros.model import Macro, MacroError


def qualified_macro(app: str, name: str) -> str:
    """The qualified `app/name` dispatch key — the ONE spelling of the
    convention the run_macro handler resolves (user macro names can
    never contain "/", so no collision). Lives beside `Pack`, the owner
    of macro dicts — every pack site (channel included) consumes it."""
    return f"{app}/{name}"


def split_ref(ref: str) -> tuple[str, str]:
    """`<app>/<playbook>` → (app, playbook) — the one parse of the ref
    every skin takes (the CLI exits on the error, the studio answers
    400). Raises PlaybookError."""
    app, sep, name = ref.partition("/")
    if not sep or not app or not name or "/" in name:
        raise PlaybookError(f"{ref!r} is not <app>/<playbook>")
    return app, name


def macro_app(name: str) -> str:
    """The app half of a qualified dispatch key — `qualified_macro`'s
    inverse, kept beside it so the "/" convention has one spelling.
    "" for an unqualified name (user macros never carry an app)."""
    app, sep, _ = name.partition("/")
    return app if sep else ""


def qualified_pack(app: str, pack: Pack) -> dict[str, Macro]:
    """A pack's macros under their qualified dispatch keys."""
    return {qualified_macro(app, n): m for n, m in pack.macros.items()}


def qualified_inline(app: str, spec: Playbook) -> dict[str, Macro]:
    """A playbook's inline macros under their qualified dispatch keys —
    `qualified_pack`'s sibling for the hands that live in the playbook
    itself. Every registry a walk can dispatch through takes both."""
    return {qualified_macro(app, n): m for n, m in spec.inline_macros.items()}


def qualified_all(app: str, pack: Pack) -> dict[str, Macro]:
    """Everything a walk of this pack can dispatch: the directory macros
    plus every playbook's inline bodies — disabled playbooks included
    (gating is the caller's filter, never the dispatch table)."""
    macros = qualified_pack(app, pack)
    for entry in scan_playbooks(app, pack):
        if entry.spec is not None:
            macros.update(qualified_inline(app, entry.spec))
    return macros


def load_pack(app: str) -> Pack:
    """The app pack, whole: the manifest (`PLAYBOOK.yml` — what the app
    is and what its routes share: meta, placeholders, landmarks,
    pages), the playbook files beside it (raw, parsed per entry by
    `scan_playbooks`), and the recorded macros. A broken pack macro is
    carried as its error string so the playbook referencing it fails
    with the cause; a broken playbook file rides the same way."""
    doc = specfile.load_pack_doc(app, PlaybookError)
    if doc is None:
        raise PlaybookError(f"no pack {app!r} on disk (missing {PACK_FILENAME})")
    _check_pack_meta(doc, app)
    root = paths.pack_root(app)
    if app == CHANNEL_APP:
        # The boot file is a template the user owns, materialized beside
        # an existing channel pack on first look (the ios pack's
        # pattern): a channel recorded before the boot was a file keeps
        # its wake, and every door — wake, step, run, check — sees the
        # same pack.
        scaffold.ensure_channel_boot(root)
    docs, pb_errors = specfile.load_playbook_docs(app, PlaybookError, root)
    try:
        # Appendix + route-declared waypoints across every playbook
        # file, one namespace — the matcher and every playbook validate
        # against the same set.
        pages = parse_pages_data(collect_page_decls(doc, docs), app)
    except PagesError as e:
        raise PlaybookError(f"{app}/{PACK_FILENAME} pages: {e}") from e
    try:
        landmarks = pack_landmarks(doc)
    except PagesError as e:
        raise PlaybookError(f"{app}/{PACK_FILENAME} landmarks: {e}") from e
    macros: dict[str, Macro] = {}
    errors: dict[str, str] = {}
    macros_root = root / PACK_MACROS_DIRNAME
    if macros_root.is_dir():
        # One scanner for both macro roots: traversal guard, dot-dir
        # convention, and the broad-except lesson live in `store.scan`.
        for entry in macro_store.scan(macros_root):
            if entry.spec is not None:
                macros[entry.dir_name] = entry.spec
            else:
                errors[entry.dir_name] = entry.error or "invalid"
    return Pack(
        app=app,
        pages=pages,
        macros=macros,
        macro_errors=errors,
        playbook_docs=docs,
        playbook_errors=pb_errors,
        landmarks=landmarks,
        page_recovers=collect_page_recovers(doc),
    )


def _check_pack_meta(doc: dict, app: str) -> None:
    """The manifest's meta, every field optional: `app` (which app this
    pack automates) must equal the directory when present — the folder
    IS the app, the field catches a pack copied under the wrong name;
    `description` is real prose when present (`install` prints it);
    `placeholders` (install-time constants, validated here so `check`
    catches a malformed map before install prompts read it) is
    name → {description, [example]}."""
    if "app" in doc:
        declared = require_str(doc.get("app"), "`app`")
        if declared != app:
            raise PlaybookError(
                f"app {declared!r} must equal the pack directory {app!r}"
            )
    if app == "pages":
        raise PlaybookError(
            "a pack cannot be named 'pages' — it is the page-reference root"
        )
    if "description" in doc:
        prose(doc.get("description"), "`description`")
    ph = doc.get("placeholders")
    if ph is None:
        return
    if not isinstance(ph, dict):
        raise PlaybookError("`placeholders` must be a mapping of TOKEN → spec")
    for key, spec in ph.items():
        where = f"placeholder {key!r}"
        if not isinstance(spec, dict) or set(spec) - {"description", "example"}:
            raise PlaybookError(f"{where} must be a {{description, example}} mapping")
        prose(spec.get("description"), f"{where}: `description`")


def scan_playbooks(app: str, pack: Pack | None = None) -> list[PlaybookEntry]:
    """Every playbook file of the pack, parsed against it — plus the
    files that would not load, as invalid entries. Callers that already
    hold the Pack thread it through so the spec file and macros are
    not re-read."""
    if pack is None:
        if not (paths.pack_root(app) / PACK_FILENAME).exists():
            return []
        pack = load_pack(app)
    out: list[PlaybookEntry] = [
        PlaybookEntry(app=app, name=n, error=e)
        for n, e in sorted(pack.playbook_errors.items())
    ]
    for name, data in pack.playbook_docs.items():
        name = str(name)
        try:
            spec = _parse_playbook_data(data, name, pack)
            out.append(PlaybookEntry(app=app, name=name, spec=spec))
        except Exception as e:  # broad: exclude whole, never take a session down
            out.append(
                PlaybookEntry(app=app, name=name, error=str(e) or type(e).__name__)
            )
    return out


def stray_dirs() -> list[str]:
    """Folders under a playbooks root holding YAML beside no manifest —
    an author who wrote a route and forgot the marker. Skips the `_`
    and `.` prefixes every lister does, and a name a home pack already
    claims (the home layer shadows the tree's)."""
    packs = set(list_apps())
    out: list[str] = []
    for root in paths.playbooks_dirs():
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if (
                d.is_dir()
                and not d.name.startswith(("_", "."))
                and d.name not in packs
                and any(d.glob("*.yml"))
            ):
                out.append(f"{root.name}/{d.name}")
    return out


def list_apps() -> list[str]:
    """Packs across the search path (the `paths.playbooks_dirs` layering),
    sorted — a PLAYBOOK.yml marks a pack."""
    return sorted(paths.marked_subdirs(paths.playbooks_dirs(), PACK_FILENAME))


def parse_playbook(text: str, name: str, pack: Pack) -> Playbook:
    """One playbook given as YAML text — the text-shaped door tests and
    tooling use; the live path is `scan_playbooks` over the pack's
    playbook files. Raises PlaybookError naming the offending field;
    never a partial spec."""
    data = specfile.load_yaml(text, PlaybookError)
    return _parse_playbook_data(data, name, pack)


_PLAY_KEYS = {"name", "description", "enabled", "inputs", "route"}


def _parse_playbook_data(data: Any, name: str, pack: Pack) -> Playbook:
    """One playbook file's document → a validated Playbook. The file
    stem IS the name; the `name:` inside must agree with it."""
    if not isinstance(data, dict):
        raise PlaybookError("a playbook must be a YAML mapping (key: value pairs)")
    unknown = sorted(set(map(str, data.keys())) - _PLAY_KEYS)
    if unknown:
        raise PlaybookError(f"unknown key(s): {', '.join(unknown)}")
    # A playbook names itself, like a macro and a skill do — and the
    # name must be the file's, so a copied file cannot lie about what
    # it is (the same rule `app` keeps with the pack folder).
    if "name" not in data:
        raise PlaybookError(
            f"every .yml beside the manifest is a playbook, and this one has "
            f"no `name:` — a playbook starts `name: {name}`"
        )
    check_name(name, "playbook name")
    declared = require_str(data.get("name"), "`name`")
    if declared != name:
        raise PlaybookError(
            f"name {declared!r} must equal the file name {name!r} ({name}.yml)"
        )
    description = prose(data.get("description"), "`description`")
    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise PlaybookError("`enabled` must be true or false")
    inputs = _parse_inputs(data.get("inputs", {}))
    input_names = {i.name for i in inputs}
    route = compile_route(
        data.get("route"), playbook=name, input_names=input_names, pack=pack
    )
    return Playbook(
        app=pack.app,
        name=name,
        description=description,
        enabled=enabled,
        inputs=inputs,
        nodes=tuple(route.nodes),
        start=route.start,
        inline_macros=route.inline,
        recovers=route.recovers,
    )


def _parse_inputs(raw: Any) -> tuple[PlaybookInput, ...]:
    """`inputs:` through the macro grammar's parser, the error class
    translated at this one seam."""
    try:
        return macro_parse.parse_inputs(raw)
    except MacroError as e:
        raise PlaybookError(str(e)) from e


def require_live(spec: Playbook, pack: Pack) -> None:
    """The live rule, spelled once: what a real wake needs of a playbook
    — enabled, with every referenced pack macro enabled. A resuming
    suspension and the boot must satisfy it; a rehearsal deliberately
    need not (you rehearse BEFORE you enable). Raises PlaybookError
    naming the gap."""
    gap = live_gap(spec, pack)
    if gap is not None:
        raise PlaybookError(f"{spec.app}/{spec.name}: {gap} — rehearse, then enable")


def live_gap(spec: Playbook, pack: Pack) -> str | None:
    """The one thing that keeps a valid playbook from a wake, in a word
    or two — None when it is live. `require_live` raises off it; the
    wake roster prints it; both read one rule."""
    if not spec.enabled:
        return "disabled"
    disabled = disabled_macros(spec, pack)
    if disabled:
        return (
            f"disabled macro{'s' if len(disabled) > 1 else ''}: {', '.join(disabled)}"
        )
    return None


def disabled_macros(spec: Playbook, pack: Pack) -> list[str]:
    """Referenced pack macros still disabled — the live-readiness rule:
    `playbooks check` warns about it and the boot will not offer such
    a playbook at all. Covers every dispatching role: do moves, an ask's
    `resume:`, a page's `recover:` hands, and an agent's granted macros.
    Safe unguarded access: parse
    validated every directory name against `pack.macros`."""
    named: set[str] = set()
    for recovery in spec.recovers.values():
        named.update(h.macro for h in recovery.hands if h.macro is not None)
    for n in spec.nodes:
        if isinstance(n, DoNode):
            named.add(n.macro)
        elif isinstance(n, AskNode) and n.resume is not None:
            named.add(n.resume)
        elif isinstance(n, AgentNode):
            named.update(n.macros)
    # One rule, no inline special case: each name resolves through the
    # merged view, and an inline macro is enabled by construction — its
    # gate is the playbook's own `enabled:` — so only directory names
    # can report.
    return sorted(
        m for m in named if not (spec.inline_macros.get(m) or pack.macros[m]).enabled
    )
