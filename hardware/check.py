"""Static consistency check for the hardware pipeline — catches
cross-artifact drift in milliseconds instead of mid-way through a full
rebuild or a manual build.

What a rename or typo breaks quietly, this catches loudly:

  procedures/  stem naming + known family; exactly one assembly class per
               module; a well-formed class-level ``views`` declaration.
  patch/       every patch JSON names a procedure stem and a declared
               (variant, camera) view; its ops have unique ids and every
               ``preop`` chain terminates at ``orig``.
  manual       every figure ``src``/``render`` in ``manual/content/*.json``
               resolves to a declared view's raw render, one of its patch
               leaves' snapshots, or a hand figure the manual builder
               generates itself (``HAND_FIGURES``).
  sourcing     ``sourcing_vendors.json`` part_ids match the BOM content
               rows one-to-one (no missing, stale, or duplicate ids).

Static only: everything is read via AST / JSON — no build123d import, no
``--group cad`` — so it runs anywhere (CI included) in under a second. The
one dynamic rule it cannot see is a camera index out of range for a
variant's camera *list* (``self.camera`` can be set per variant at runtime);
``BaseAssembly.view_indices`` still validates that at build time.

Run from the repo root::

    uv run python -m hardware.check      # or: python -m hardware check

Exit 0 when clean; exit 1 with one line per finding.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter

from hardware.assembly.mark.patch import ID_RE, ORIG_SENTINEL
from hardware.assembly.mark.replay import find_leaves
from hardware.manual.assets import HAND_FIGURES
from hardware.manual.common import CONTENT_DIR, VENDOR_FILE
from hardware.scheme import (
    FAMILY_PRIORITY,
    PATCH_DIR,
    PATCH_NAME_RE,
    PROCEDURES_DIR,
    STEM_CONVENTION,
    STEM_RE,
    SVG_NAME_RE,
    VARIANTS,
)

# A view tag name as written in a ``views`` declaration (EXPLODED_CAM1, …).
_TAG_RE = re.compile(rf"^({'|'.join(v.upper() for v in VARIANTS)})_CAM(\d+)$")

# Sentinel for "views not declared on this class" (inherit / render-all).
_UNDECLARED = object()


# ── procedures ────────────────────────────────────────────────────────────────


def _parse_views(node: ast.expr, where: str, findings: list[str]):
    """A ``views`` assignment value → set of (variant, cam) pairs, None for
    an explicit ``views = None`` (render every camera), or _UNDECLARED when
    the expression isn't statically readable (reported as a finding)."""
    if isinstance(node, ast.Constant) and node.value is None:
        return None
    if not isinstance(node, ast.List):
        findings.append(f"{where}: views is not a plain list literal")
        return _UNDECLARED
    views: list[tuple[str, int]] = []
    for elt in node.elts:
        if isinstance(elt, ast.Name) and (m := _TAG_RE.match(elt.id)):
            views.append((m[1].lower(), int(m[2])))
        elif isinstance(elt, ast.Tuple):
            try:
                variant, cam = ast.literal_eval(elt)
            except (ValueError, SyntaxError):
                findings.append(f"{where}: unreadable views entry")
                continue
            views.append((variant, cam))
        else:
            findings.append(f"{where}: unreadable views entry")
    for variant, cam in views:
        if variant not in VARIANTS or not isinstance(cam, int):
            findings.append(f"{where}: unknown view tag ({variant!r}, {cam!r})")
    if len(set(views)) != len(views):
        findings.append(f"{where}: duplicate view tags")
    return set(views)


def check_procedures() -> tuple[list[str], dict[str, set | None]]:
    """Validate procedures/ and return (findings, declared views by stem).
    A stem mapping to ``None`` renders every camera (permissive downstream)."""
    findings: list[str] = []
    # Per stem: the module's classes as {name: (bases, views)}.
    modules: dict[str, dict[str, tuple[list[str], object]]] = {}

    for path in sorted(PROCEDURES_DIR.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        m = STEM_RE.match(path.stem)
        if not m or m["family"] not in FAMILY_PRIORITY:
            findings.append(
                f"procedures: {path.name} doesn't match "
                f"{STEM_CONVENTION}.py with a known family"
            )
            continue
        classes: dict[str, tuple[list[str], object]] = {}
        for node in ast.parse(path.read_bytes(), filename=str(path)).body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            views: object = _UNDECLARED
            for stmt in node.body:
                targets = (
                    stmt.targets
                    if isinstance(stmt, ast.Assign)
                    else [stmt.target]
                    if isinstance(stmt, ast.AnnAssign)
                    else []
                )
                if any(isinstance(t, ast.Name) and t.id == "views" for t in targets):
                    views = _parse_views(
                        stmt.value, f"procedures: {path.name} {node.name}", findings
                    )
            classes[node.name] = (bases, views)
        modules[path.stem] = classes

    # Which classes are assemblies? Fixpoint over base names, since a
    # procedure may subclass another procedure's class rather than
    # BaseAssembly directly (e.g. ID20Ru(ID10Lu)).
    class_stem = {name: stem for stem, cls in modules.items() for name in cls}
    assembly = {"BaseAssembly"}
    changed = True
    while changed:
        changed = False
        for classes in modules.values():
            for name, (bases, _) in classes.items():
                if name not in assembly and assembly.intersection(bases):
                    assembly.add(name)
                    changed = True

    views_by_stem: dict[str, set | None] = {}
    for stem, classes in modules.items():
        asm = [n for n in classes if n in assembly]
        if len(asm) != 1:
            findings.append(
                f"procedures: {stem}.py defines {len(asm)} assembly classes "
                f"(need exactly 1)"
            )
            continue
        # Resolve an undeclared ``views`` up the inheritance chain: a
        # procedure subclassing another procedure's class (ID20Ru(ID10Lu))
        # inherits its declaration unless it overrides.
        name = asm[0]
        views = classes[name][1]
        seen = {name}
        while views is _UNDECLARED:
            bases, _ = modules[class_stem[name]][name]
            parent = next((b for b in bases if b in class_stem and b not in seen), None)
            if parent is None:
                views = None  # reached BaseAssembly: default renders all
                break
            seen.add(parent)
            name = parent
            views = modules[class_stem[name]][name][1]
        views_by_stem[stem] = views
    return findings, views_by_stem


def _view_declared(views: set | None, variant: str, cam: int) -> bool:
    return views is None or (variant, cam) in views


# ── patches ───────────────────────────────────────────────────────────────────


def check_patches(
    views_by_stem: dict[str, set | None],
) -> tuple[list[str], dict[str, set[str]]]:
    """Validate patch/*.json and return (findings, leaf op-ids per SVG name)."""
    findings: list[str] = []
    leaves_by_svg: dict[str, set[str]] = {}
    for path in sorted(PATCH_DIR.glob("*.json")):
        m = PATCH_NAME_RE.match(path.stem)
        if not m:
            findings.append(
                f"patch: {path.name} doesn't match <stem>_<variant>_cam<i>.json"
            )
            continue
        stem, variant, cam = m["stem"], m["variant"], int(m["cam"])
        if stem not in views_by_stem:
            findings.append(f"patch: {path.name} names unknown procedure {stem!r}")
            continue
        if not _view_declared(views_by_stem[stem], variant, cam):
            findings.append(
                f"patch: {path.name} targets {variant}_cam{cam}, "
                f"which {stem} doesn't declare in views"
            )
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            findings.append(f"patch: {path.name} is not valid JSON ({exc})")
            continue
        if not isinstance(entries, list):
            findings.append(f"patch: {path.name} must be a JSON array of ops")
            continue
        ids = [e.get("id") for e in entries if isinstance(e, dict)]
        bad_ids = [i for i in ids if not (isinstance(i, str) and ID_RE.match(i))]
        if len(ids) != len(entries) or bad_ids:
            findings.append(f"patch: {path.name} has op(s) without a valid id")
        if len(set(ids)) != len(ids):
            findings.append(f"patch: {path.name} has duplicate op ids")
        known = set(ids)
        preop = {
            e["id"]: e.get("preop")
            for e in entries
            if isinstance(e, dict) and "id" in e
        }
        for op_id in preop:
            seen: set[str] = set()
            cur = op_id
            while cur not in seen:
                seen.add(cur)
                nxt = preop.get(cur)
                if nxt == ORIG_SENTINEL:
                    break
                if nxt not in known:
                    findings.append(
                        f"patch: {path.name} op {op_id!r} chains to "
                        f"unknown preop {nxt!r}"
                    )
                    break
                cur = nxt
            else:
                findings.append(f"patch: {path.name} op {op_id!r} preop chain loops")
        # Feed find_leaves only well-formed ops: an op missing "id" or
        # "preop" was already reported above, and letting it through would
        # KeyError — crashing the gate and masking every other finding.
        well_formed = [
            e for e in entries if isinstance(e, dict) and "id" in e and "preop" in e
        ]
        leaves_by_svg[f"{path.stem}.svg"] = {e["id"] for e in find_leaves(well_formed)}
    return findings, leaves_by_svg


# ── manual content ────────────────────────────────────────────────────────────


def load_content() -> tuple[list[str], dict[str, list]]:
    """Parse every ``content/*.json`` once for both content checks. Returns
    (findings for unparseable files, pages keyed by filename)."""
    findings: list[str] = []
    pages_by_file: dict[str, list] = {}
    for path in sorted(CONTENT_DIR.glob("*.json")):
        try:
            pages_by_file[path.name] = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            findings.append(f"manual: {path.name} is not valid JSON ({exc})")
    return findings, pages_by_file


def _figure_refs(pages: object) -> set[str]:
    """Every ``src``/``render`` string ending in .svg, recursively."""
    refs: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("src", "render") and isinstance(v, str) and v.endswith(".svg"):
                    refs.add(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(pages)
    return refs


def check_manual(
    views_by_stem: dict[str, set | None],
    leaves_by_svg: dict[str, set[str]],
    pages_by_file: dict[str, list],
) -> list[str]:
    findings: list[str] = []
    hand = set(HAND_FIGURES)  # SVGs the manual builder writes itself
    for filename, pages in pages_by_file.items():
        for ref in sorted(_figure_refs(pages)):
            if ref in hand:
                continue
            m = SVG_NAME_RE.match(ref)
            if not m or m["stem"] not in views_by_stem:
                findings.append(
                    f"manual: {filename} references {ref!r}, which is neither "
                    f"a procedure render nor a hand figure"
                )
                continue
            stem, variant, cam, op = m["stem"], m["variant"], int(m["cam"]), m["op"]
            if not _view_declared(views_by_stem[stem], variant, cam):
                findings.append(
                    f"manual: {filename} references {ref!r}, but {stem} "
                    f"doesn't declare {variant}_cam{cam} in views"
                )
                continue
            if op is None:
                continue
            base = f"{stem}_{variant}_cam{cam}.svg"
            if base not in leaves_by_svg:
                findings.append(
                    f"manual: {filename} references snapshot {ref!r}, "
                    f"but {base} has no patch JSON"
                )
            elif op not in leaves_by_svg[base]:
                findings.append(
                    f"manual: {filename} references snapshot {ref!r}, but "
                    f"{op!r} is not a leaf op in {base}'s patch"
                )
    return findings


# ── sourcing ──────────────────────────────────────────────────────────────────


def _duplicates(ids: list) -> list[str]:
    """Non-empty ids appearing more than once, sorted."""
    return sorted(i for i, n in Counter(filter(None, ids)).items() if n > 1)


def check_sourcing(pages_by_file: dict[str, list]) -> list[str]:
    """The BOM content rows and sourcing_vendors.json must join one-to-one
    on part_id — the same invariants build_sourcing_guide enforces at build
    time (missing/duplicate row ids) plus the sync drift it only reports
    (stale / missing vendor entries)."""
    findings: list[str] = []
    rows: list[dict] = [
        row
        for pages in pages_by_file.values()
        if isinstance(pages, list)
        for page in pages
        if isinstance(page, dict) and page.get("type") == "bom"
        for row in page.get("rows", [])
    ]
    row_ids = [r.get("part_id") for r in rows]
    if missing := sum(1 for i in row_ids if not i):
        findings.append(f"sourcing: {missing} BOM row(s) missing a part_id")
    if dupes := _duplicates(row_ids):
        findings.append(f"sourcing: duplicate BOM part_id(s): {', '.join(dupes)}")

    if not VENDOR_FILE.exists():
        findings.append(f"sourcing: {VENDOR_FILE.name} not found")
        return findings
    try:
        entries = json.loads(VENDOR_FILE.read_text(encoding="utf-8"))
    except ValueError as exc:
        findings.append(f"sourcing: {VENDOR_FILE.name} is not valid JSON ({exc})")
        return findings
    vendor_ids = [e.get("part_id") for e in entries if isinstance(e, dict)]
    if dupes := _duplicates(vendor_ids):
        findings.append(
            f"sourcing: duplicate part_id(s) in {VENDOR_FILE.name}: {', '.join(dupes)}"
        )
    row_set = {i for i in row_ids if i}
    if stale := sorted(set(filter(None, vendor_ids)) - row_set):
        findings.append(
            f"sourcing: {VENDOR_FILE.name} has stale part_id(s) with no BOM "
            f"row: {', '.join(stale)}"
        )
    if absent := sorted(row_set - set(vendor_ids)):
        findings.append(
            f"sourcing: BOM part_id(s) missing from {VENDOR_FILE.name}: "
            f"{', '.join(absent)} (run `python -m hardware sourcing --scaffold`)"
        )
    return findings


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    del argv  # no flags; kept for the __main__ delegation convention
    proc_findings, views_by_stem = check_procedures()
    patch_findings, leaves_by_svg = check_patches(views_by_stem)
    content_findings, pages_by_file = load_content()
    findings = (
        proc_findings
        + patch_findings
        + content_findings
        + check_manual(views_by_stem, leaves_by_svg, pages_by_file)
        + check_sourcing(pages_by_file)
    )
    if findings:
        for f in findings:
            print(f, file=sys.stderr)
        print(f"\n{len(findings)} consistency finding(s)", file=sys.stderr)
        return 1
    n_patches = len(list(PATCH_DIR.glob("*.json")))
    print(
        f"hardware check OK — {len(views_by_stem)} procedures, "
        f"{n_patches} patches, manual + sourcing consistent"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
