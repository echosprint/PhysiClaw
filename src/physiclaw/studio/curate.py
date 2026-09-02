"""Curation brains over a draft — evidence badges, the dry-run
confusion matrix, and commit.

Everything scores through the conductor's own cores (`capture_app`,
`score_page`, the pack parsers): the studio adds no second matcher, so
what the matrix predicts is what the walk will see. Nothing here
touches hardware; a saved draft curates offline.
"""

import json

from physiclaw.common import paths
from physiclaw.common.listing import Screen
from physiclaw.common.paths import PACK_FILENAME
from physiclaw.common.text import read_text, write_text
from physiclaw.conductor import _spec
from physiclaw.conductor.capture import capture_app
from physiclaw.conductor.match import (
    candidate_rows,
    match_screen,
    normalize,
    score_page,
)
from physiclaw.conductor.pages import (
    AnchorDecl,
    PageDecl,
    PagePrint,
    PagesError,
    collect_page_decls,
    load_learned,
    pack_landmarks,
    parse_landmarks,
    parse_pages_data,
    save_learned,
)
from physiclaw.studio.draft import DraftError, decl_data

# A cross-page shot scoring at or above this fraction of the page's own
# threshold reads as a lookalike worth a forbid suggestion, even when it
# does not (yet) misidentify outright.
LOOKALIKE_FRACTION = 0.75
MAX_FORBID_SUGGESTIONS = 5


def _screens(draft: dict) -> dict[str, list[Screen]]:
    return {
        name: [Screen.read(draft["shots"][sid]["listing"]) for sid in page["shots"]]
        for name, page in draft["pages"].items()
    }


def _decls(draft: dict) -> dict[str, PageDecl]:
    """Parsed decls for the draft's COMPLETE pages (those with anchors).
    A page still waiting for its first anchor is mid-authoring, not an
    error — it is simply absent from evidence and the matrix."""
    data = {k: v for k, v in decl_data(draft).items() if v.get("anchors")}
    try:
        return parse_pages_data(data, draft["app"])
    except PagesError as e:
        raise DraftError(str(e)) from e


# ---------- evidence badges ----------


def evidence(draft: dict) -> dict:
    """Per declared anchor: how often the page's own shots contain it
    (frequency — geometry needs ≥60%) and how many OTHER drafted pages'
    shots also show it (shared chrome discriminates poorly). The same
    two axes `mine_anchors` weights by, surfaced before mining."""
    decls = _decls(draft)
    screens = _screens(draft)
    out: dict[str, dict] = {}
    for name, decl in decls.items():
        own = screens.get(name, [])
        others = {p: s for p, s in screens.items() if p != name and s}
        badges = {}
        for a in decl.anchors:
            badges[a.text] = {
                "seen": _seen_in(a, own),
                "shots": len(own),
                "other_pages": sum(
                    1 for shots in others.values() if any(_hits(a, s) for s in shots)
                ),
            }
        out[name] = badges
    return out


def _hits(a: AnchorDecl, screen: Screen) -> bool:
    return bool(candidate_rows(a, screen.rows, ()))


def _seen_in(a: AnchorDecl, screens: list[Screen]) -> int:
    return sum(1 for s in screens if _hits(a, s))


def page_guess(draft: dict, listing: str) -> "str | None":
    """Which drafted page one listing reads as — `match_screen`, the
    decision the real walk makes (margin over the runner-up, forbid
    veto, decl-only thresholds), over declaration-only prints. None on
    anything but a confident match. Feeds the recorder's page-boundary
    split hint."""
    prints = [PagePrint(app=draft["app"], decl=d) for d in _decls(draft).values()]
    v = match_screen(Screen.read(listing), prints)
    if v.kind != "match" or v.page_id is None:
        return None
    return v.page_id.partition(".")[2]


# ---------- dry run: mine + confusion matrix ----------


def dry_run(draft: dict) -> dict:
    """Mine learned geometry from the draft's shots and score EVERY
    drafted page against EVERY shot — without writing a byte. The
    matrix is the separability check: a page whose threshold is met by
    another page's shot will misidentify in the field."""
    return _mined_dry_run(draft)[1]


def _mined_dry_run(draft: dict) -> "tuple[dict, dict]":
    """(learned, dry-run result) from ONE mining pass — `commit` needs
    both halves and must never pay for (or drift from) a second mine."""
    app = draft["app"]
    decls = _decls(draft)
    screens = _screens(draft)
    learned, reports, warnings = capture_app(app, decls, screens, negatives=[])
    prints = {
        name: PagePrint(app=app, decl=d, learned=learned.get(name))
        for name, d in decls.items()
    }

    matrix = []
    for owner, page in draft["pages"].items():
        for sid, screen in zip(page["shots"], screens[owner]):
            matrix.append(
                {
                    "shot": sid,
                    "page": owner,
                    "scores": {
                        name: round(score_page(pp, screen).score, 3)
                        for name, pp in prints.items()
                    },
                }
            )

    pages = {}
    by_page = {r.page.split(".", 1)[1]: r for r in reports}
    for name, pp in prints.items():
        impostors = [
            (cell["scores"][name], cell["shot"], cell["page"])
            for cell in matrix
            if cell["page"] != name
        ]
        worst = max(impostors, default=None)
        report = by_page.get(name)
        pages[name] = {
            "threshold": round(pp.threshold, 3),
            "observations": report.observations if report else 0,
            "anchors_learned": report.anchors_learned if report else 0,
            "impostor_max": round(worst[0], 3) if worst else None,
            "impostor_shot": worst[1] if worst else None,
            "separable": worst is None or worst[0] < pp.threshold,
        }
    return learned, {
        "pages": pages,
        "matrix": matrix,
        "forbid_suggestions": _forbid_suggestions(draft, prints, screens, matrix),
        "warnings": warnings,
    }


def _forbid_suggestions(
    draft: dict,
    prints: dict[str, PagePrint],
    screens: dict[str, list[Screen]],
    matrix: list[dict],
) -> dict[str, list[str]]:
    """For each page, labels its lookalike near-misses carry that its
    own shots never do — one forbid term flips such an impostor to 0."""
    own_norms = {
        name: {normalize(r.label) for s in shots for r in s.rows if r.label}
        for name, shots in screens.items()
    }
    shot_screens = {
        sid: screen
        for name, page in draft["pages"].items()
        for sid, screen in zip(page["shots"], screens[name])
    }
    out: dict[str, list[str]] = {}
    for name, pp in prints.items():
        lookalikes = [
            cell
            for cell in matrix
            if cell["page"] != name
            and cell["scores"][name] >= pp.threshold * LOOKALIKE_FRACTION
        ]
        seen: list[str] = []
        for cell in lookalikes:
            for row in shot_screens[cell["shot"]].rows:
                label = row.label.strip()
                if len(label) < 2 or label in seen:
                    continue
                if normalize(row.label) in own_norms.get(name, set()):
                    continue
                seen.append(label)
        if seen:
            out[name] = seen[:MAX_FORBID_SUGGESTIONS]
    return out


# ---------- commit ----------


def commit(draft: dict) -> dict:
    """Write the draft into the real files: the `pages:` and `landmarks:`
    sections of PLAYBOOK.yml (pack scaffolded if absent) and the mined
    geometry merged per-page into `learned/pages/<app>.json`. The pages
    section is the studio's whole; landmarks MERGE — the draft refreshes
    the spots it marked and the pack's hand-authored ones survive. The new
    file text is validated through the pack doors BEFORE anything is
    written — including against route-declared pages (a page is
    declared ONCE)."""
    app = draft["app"]
    pages_data = decl_data(draft)
    if not pages_data:
        raise DraftError("nothing to commit — draft no pages first")
    for name, spec in pages_data.items():
        if not spec.get("anchors"):
            raise DraftError(f"page {name!r} has no anchors yet")
    learned, dry = _mined_dry_run(draft)

    pack_file = _pack_file(app)
    text = read_text(pack_file)
    if draft["landmarks"]:
        merged = _merged_landmarks(text, draft["landmarks"])
        text = _splice_section(text, "landmarks", _emit_landmarks(merged))
    text = _splice_section(text, "pages", _emit_pages(pages_data))
    _validate_pack_text(text, app)
    write_text(pack_file, text)

    merged = {**load_learned(app), **learned}
    save_learned(app, merged)

    # The `playbooks check` core over the file just written — pages and
    # landmarks were pre-validated, this also re-parses the untouched
    # walks against the new page set.
    from physiclaw.conductor import playbook as pb

    try:
        pb.load_pack(app)
        check = "ok"
    except pb.PlaybookError as e:
        check = str(e)
    return {
        "pack_file": str(pack_file),
        "pages_written": sorted(pages_data),
        "landmarks_written": sorted(draft["landmarks"]),
        "learned_pages": sorted(learned),
        "check": check,
        "dry_run": dry,
    }


def _pack_file(app: str):
    from physiclaw.conductor.scaffold import init_pack

    f = paths.pack_root(app) / PACK_FILENAME
    if not f.exists():
        return init_pack(app) / PACK_FILENAME
    return f


def _validate_pack_text(text: str, app: str) -> None:
    try:
        doc = _spec.load_yaml(text, PagesError)
        parse_pages_data(collect_page_decls(doc), app)
        pack_landmarks(doc)
    except PagesError as e:
        raise DraftError(
            f"commit refused — the new {PACK_FILENAME} would not parse: {e}"
        ) from e


def _merged_landmarks(text: str, drafted: dict) -> dict:
    """The pack file's own landmarks under the draft's, in the draft's
    {label, bbox, [page]} shape — so a commit never drops a
    hand-authored spot the walks or recover hands name. Read section-
    level (scopes unchecked): the commit REPLACES the pages section, so
    a scope is judged against the new pages by `_validate_pack_text`."""
    try:
        existing = parse_landmarks(_spec.load_yaml(text, PagesError).get("landmarks"))
    except PagesError as e:
        raise DraftError(
            f"commit refused — the current {PACK_FILENAME} would not parse: {e}"
        ) from e
    kept = {
        name: {
            "label": c.label[0] if len(c.label) == 1 else list(c.label),
            "bbox": list(c.bbox),
            **({"page": c.page} if c.page else {}),
        }
        for name, c in existing.items()
    }
    return {**kept, **drafted}


# ---------- YAML emission ----------
# House-style text, not a dumper: strings quoted via JSON (a JSON string
# is a valid YAML string, and CJK stays readable with ensure_ascii off).


def quote_yaml(value) -> str:
    """A value as a YAML scalar via JSON (a JSON string IS a YAML
    string; CJK stays readable with ensure_ascii off) — the ONE
    quoting spelling for studio-emitted pack/macro text."""
    return json.dumps(value, ensure_ascii=False)


def _emit_pages(pages_data: dict) -> str:
    lines = ["pages:"]
    for name, spec in pages_data.items():
        lines.append(f"  {name}:")
        lines.append("    anchors:")
        for a in spec["anchors"]:
            if isinstance(a, str):
                lines.append(f"      - {quote_yaml(a)}")
            else:
                fields = [f"text: {quote_yaml(a['text'])}"]
                if a.get("region"):
                    fields.append(f"region: {a['region']}")
                lines.append(f"      - {{{', '.join(fields)}}}")
        if spec.get("forbid"):
            terms = ", ".join(quote_yaml(t) for t in spec["forbid"])
            lines.append(f"    forbid: [{terms}]")
        if spec.get("scrollable"):
            lines.append("    scrollable: true")
    return "\n".join(lines) + "\n"


def _emit_landmarks(landmarks: dict) -> str:
    lines = ["landmarks:"]
    for name, spec in landmarks.items():
        lines.append(f"  {name}:")
        lines.append(f"    label: {quote_yaml(spec['label'])}")
        coords = ", ".join(f"{float(v):g}" for v in spec["bbox"])
        lines.append(f"    bbox: [{coords}]")
        if spec.get("page"):
            lines.append(f"    page: {spec['page']}")
    return "\n".join(lines) + "\n"


def splice_playbook(text: str, name: str, block: str) -> str:
    """Replace ONE key under the `playbooks:` section (or add it),
    leaving hand-authored sibling playbooks byte-for-byte. The studio
    owns its drafted playbook, never the section. `block` is the
    `{name: …}` mapping at column 0; it lands indented under the
    section head. Sub-entries run to the next indent-2 content
    (comments at that indent are boundaries, like `_splice_section`)."""
    indented = "\n".join(
        ("  " + ln if ln.strip() else ln) for ln in block.rstrip("\n").splitlines()
    )
    lines = text.splitlines()
    start = _find_top_key(lines, "playbooks")
    if start is None:
        return _appended(text, "playbooks:\n" + indented + "\n")
    trailing = lines[start].partition(":")[2].strip()
    if trailing == "{}":
        # The empty-stub spelling — reopen it as a block mapping.
        lines[start] = "playbooks:"
    elif trailing:
        raise DraftError(
            "`playbooks:` is written flow-style — the studio can only "
            "splice a block-style section; reshape it by hand first"
        )
    end = _section_end(lines, start)
    sub = next(
        (
            i
            for i in range(start + 1, end)
            if lines[i] == f"  {name}:" or lines[i].startswith(f"  {name}: ")
        ),
        None,
    )
    if sub is None:
        replaced = lines[:end] + indented.splitlines() + lines[end:]
        return "\n".join(replaced) + "\n"
    sub_end = sub + 1
    while sub_end < end and (
        not lines[sub_end].strip() or lines[sub_end].startswith("    ")
    ):
        sub_end = sub_end + 1
    replaced = lines[:sub] + indented.splitlines() + lines[sub_end:]
    return "\n".join(replaced) + "\n"


def _splice_section(text: str, key: str, block: str) -> str:
    """Replace one top-level section of the pack file's RAW text (or
    append it), leaving every other byte — comments, placeholders, the
    walks — untouched. A section runs from `^key:` to the next
    column-zero content; column-zero comments are boundaries too, so a
    comment block above the NEXT section stays with it."""
    lines = text.splitlines()
    start = _find_top_key(lines, key)
    if start is None:
        return _appended(text, block)
    end = _section_end(lines, start)
    replaced = lines[:start] + block.rstrip("\n").splitlines() + lines[end:]
    return "\n".join(replaced) + "\n"


def _find_top_key(lines: list, key: str) -> "int | None":
    """The column-zero `key:` line, or None."""
    return next(
        (
            i
            for i, ln in enumerate(lines)
            if ln == f"{key}:" or ln.startswith(f"{key}: ")
        ),
        None,
    )


def _section_end(lines: list, start: int) -> int:
    """Where the section starting at `start` ends: the next column-zero
    content. Column-zero comments are boundaries too, so a comment
    block above the NEXT section stays with it."""
    end = start + 1
    while end < len(lines) and lines[end][:1] in ("", " ", "\t"):
        end = end + 1
    return end


def _appended(text: str, block: str) -> str:
    """`block` appended as a new section at the end of the file."""
    joined = text.rstrip("\n")
    return (joined + "\n\n" if joined else "") + block
