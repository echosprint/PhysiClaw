"""Architectural boundary guards for the agent ↔ core split.

The contract: the ``physiclaw.agent`` package (LLM tool-call loop,
runtime, providers) and the ``physiclaw.core`` package (hardware,
vision, MCP server) are decoupled. They talk to each other **only**
over the MCP wire protocol — the agent opens an HTTP MCP session to the
core server (see ``physiclaw.agent.engine.mcp_tool``) and discovers /
invokes tools dynamically via ``list_tools`` / ``call_tool``. Neither
package imports the other's Python modules.

Shared, domain-agnostic infrastructure lives in ``physiclaw.common``
(``platform``, ``logger``, ``paths``, ``config``, ``text``, …) so both
layers can depend on it *downward* without depending on *each other*.
``common`` is a leaf: it imports neither ``core`` nor ``agent``, which
keeps the dependency graph acyclic. ``physiclaw.cli`` is the
composition root and is allowed to import everything — it is not
covered by these guards.

These tests parse the AST of every source file (so docstring / comment
mentions of the other package don't count — only real ``import`` and
``from … import`` statements do) and fail if any edge crosses the
boundary. They exist to keep the decoupling from silently regressing:
if you find yourself needing to import across it, the dependency
belongs either behind an MCP tool or in the shared infra layer.
"""

from __future__ import annotations

import ast
from pathlib import Path

import physiclaw

PKG_ROOT = Path(physiclaw.__file__).resolve().parent
SRC_ROOT = PKG_ROOT.parent  # the `src/` dir; `physiclaw.<...>` is relative to it


def _module_name(path: Path) -> str:
    """Dotted module name for a source file, e.g. ``physiclaw.agent.engine``."""
    rel = path.relative_to(SRC_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _strip_type_checking(tree: ast.Module) -> ast.Module:
    """Drop ``if TYPE_CHECKING:`` blocks — type-only edges create no runtime
    dependency (and no import cycle), so the layering guards ignore them."""

    class _Strip(ast.NodeTransformer):
        def visit_If(self, node: ast.If):
            self.generic_visit(node)
            test = node.test
            name = (
                test.id
                if isinstance(test, ast.Name)
                else test.attr
                if isinstance(test, ast.Attribute)
                else None
            )
            if name == "TYPE_CHECKING":
                return ast.Module(body=node.orelse, type_ignores=[])
            return node

    return _Strip().visit(tree)


def _imported_modules(path: Path) -> set[str]:
    """Absolute dotted module names imported by a file (relative imports
    resolved; ``TYPE_CHECKING``-only imports excluded)."""
    tree = _strip_type_checking(
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    )
    module_name = _module_name(path)
    is_pkg = path.name == "__init__.py"
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    out.add(node.module)
                continue
            # Resolve a relative import to its absolute target.
            base = module_name.split(".")
            if not is_pkg:
                base = base[:-1]  # a module resolves relative to its package
            if node.level > 1:
                base = base[: -(node.level - 1)]
            out.add(".".join([*base, node.module]) if node.module else ".".join(base))
    return out


def _files(pkg: str) -> list[Path]:
    return sorted((PKG_ROOT / pkg).rglob("*.py"))


def _violations(from_pkg: str, forbidden_prefix: str) -> list[str]:
    bad: list[str] = []
    for f in _files(from_pkg):
        for mod in _imported_modules(f):
            if mod == forbidden_prefix or mod.startswith(forbidden_prefix + "."):
                bad.append(f"{f.relative_to(SRC_ROOT)} -> {mod}")
    return bad


def test_agent_does_not_import_core() -> None:
    """The agent reaches core only through the MCP server interface."""
    bad = _violations("agent", "physiclaw.core")
    assert not bad, "agent must not import core directly:\n" + "\n".join(bad)


def test_core_does_not_import_agent() -> None:
    """Core is the lower layer; it never depends on the agent."""
    bad = _violations("core", "physiclaw.agent")
    assert not bad, "core must not import agent:\n" + "\n".join(bad)


def test_common_is_a_leaf() -> None:
    """Shared infra depends on neither domain layer — keeps the graph acyclic."""
    bad = _violations("common", "physiclaw.core") + _violations(
        "common", "physiclaw.agent"
    )
    assert not bad, "common must not import core or agent:\n" + "\n".join(bad)


# The agent packages that ARE the engine or drive it directly — they may
# import engine internals freely. Everything else under `agent/`
# (provider, trace, layout, and whatever lands next) is shared across
# runtimes: the type vocabulary they used to take from `engine.dto` now
# lives in `physiclaw.contract`, so they have NO legal engine edge at
# all. Stated as the exception set so a new shared package is guarded
# from birth, with no test edit to remember.
_ENGINE_RUNTIMES = frozenset({"engine", "claude", "hooks", "runtime"})


def test_shared_agent_packages_do_not_import_the_engine() -> None:
    """Anything under `agent/` that isn't an engine runtime imports
    nothing from the engine — the message shapes live in
    `physiclaw.contract.dto`, engine behavior stays in the engine, and
    shared behavior belongs in `common` (e.g. `common.image`)."""
    bad = []
    for pkg_dir in sorted((PKG_ROOT / "agent").iterdir()):
        if not pkg_dir.is_dir() or pkg_dir.name in _ENGINE_RUNTIMES | {"__pycache__"}:
            continue
        bad += _violations(f"agent/{pkg_dir.name}", "physiclaw.agent.engine")
    assert not bad, (
        "shared agent packages must not import the engine "
        "(message shapes are physiclaw.contract.dto):\n" + "\n".join(bad)
    )


# ─── The plugin seam: conductor / contract / macros / provider ───────
#
# The conductor is a peer of the agent behind the turn-plugin seam
# (`contract.plugin`): the engine loads it from a config-listed dotted
# path and knows the protocol, never the name. These allowlists pin
# each package's TOTAL first-party import surface so no edge grows back
# silently; the two directional guards below them are the seam itself.

_EDGE_ALLOWLISTS = {
    # package dir → allowed physiclaw import prefixes (self always allowed)
    # contract may name macro types (SessionSetup.gated_macros) — data
    # shapes only, never agent or conductor (both import contract).
    "contract": ("physiclaw.common", "physiclaw.macros.model"),
    "macros": ("physiclaw.common",),
    "provider": (
        "physiclaw.common",
        "physiclaw.contract",
    ),
    "conductor": (
        "physiclaw.common",
        "physiclaw.contract",
        "physiclaw.macros",
        "physiclaw.provider",
    ),
    # The e2e debug harness: renders the virtual thread against the
    # channel pack's fingerprint (conductor) and speaks contract shapes.
    # The engine reaches it only via debug mode's dotted-path
    # load (`agent.engine.plugins.load_debug_intercept`) — never an
    # import, same blindness as the conductor plugin.
    # The stepping driver (`debug/stepping.py`) also steps macros one
    # gesture at a time through the macro runner — the `macros run`
    # core, wrapped once more for the studio's Macro tab.
    "debug": (
        "physiclaw.common",
        "physiclaw.contract",
        "physiclaw.conductor",
        "physiclaw.macros",
    ),
}


def test_package_edge_allowlists() -> None:
    """Each seam package imports only its pinned prefixes."""
    bad: list[str] = []
    for pkg, allowed in _EDGE_ALLOWLISTS.items():
        prefixes = (*allowed, "physiclaw." + pkg)
        for f in _files(pkg):
            for mod in _imported_modules(f):
                if mod.startswith("physiclaw") and not any(
                    mod == p or mod.startswith(p + ".") for p in prefixes
                ):
                    bad.append(f"{f.relative_to(SRC_ROOT)} -> {mod}")
    assert not bad, "migration allowlist violated:\n" + "\n".join(bad)


def test_agent_does_not_import_the_conductor() -> None:
    """The seam's own rule: the agent reaches the conductor ONLY through
    the plugin loader's dotted-path import (`[agent] plugins` — a config
    string, invisible to this AST scan by design). The debug harness
    rides the same seam (debug mode → dotted path), so it
    gets the same guard. A static import anywhere under `agent/` would
    re-couple what the seams decoupled. The reverse direction is covered
    by each package's allowlist above; the CLI is the composition root
    and may import both."""
    bad = _violations("agent", "physiclaw.conductor") + _violations(
        "agent", "physiclaw.debug"
    )
    assert not bad, (
        "agent must not import the conductor or the debug harness:\n" + "\n".join(bad)
    )


def test_agent_does_not_touch_conductor_owned_storage() -> None:
    """Data ownership across the seam: `playbooks/` (packs, playbooks,
    suspended.json) belongs to the conductor; the agent must reach that
    state only through the plugin protocol, never the disk. Scanned as
    identifiers (`paths.playbooks_dir` / `suspended` attribute or name
    references), so comments don't count but any code path does."""
    owned = {"playbooks_dir", "suspended_path", "clear_suspended", "suspended_ref"}
    bad: list[str] = []
    for f in _files("agent"):
        tree = _strip_type_checking(
            ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        )
        for node in ast.walk(tree):
            name = (
                node.attr
                if isinstance(node, ast.Attribute)
                else node.id
                if isinstance(node, ast.Name)
                else None
            )
            if name in owned:
                bad.append(f"{f.relative_to(SRC_ROOT)} -> {name}")
    assert not bad, "agent must not touch conductor-owned storage:\n" + "\n".join(bad)


# ─── Intra-core layering ─────────────────────────────────────
#
# Inside ``physiclaw.core`` the layers are:
#   geometry, vision                            — leaves
#   hardware                                    — drivers; may consume vision's
#                                                 pure metrics (exposure/focus
#                                                 policy meters quality) and
#                                                 geometry
#   bridge                                      — page state + LAN routes
#   calibration                                 — steps that drive the page
#   orchestration, server                       — composition (import anything)
#
# The shared render↔measure contract lives in ``core.geometry`` so
# bridge and calibration never need each other's constants — that is what
# keeps the graph below acyclic. TYPE_CHECKING-only imports are exempt
# (see ``_strip_type_checking``).

CORE_LAYERING = {
    # from-package (under core/) → packages it must never import (under core/)
    "vision": ("hardware", "bridge", "calibration", "orchestration", "server"),
    "hardware": ("bridge", "calibration", "orchestration", "server"),
    "bridge": ("hardware", "calibration", "orchestration", "server"),
    "calibration": ("orchestration", "server"),
}


def test_core_internal_layering() -> None:
    bad: list[str] = []
    for pkg, forbidden in CORE_LAYERING.items():
        for target in forbidden:
            bad += _violations(f"core/{pkg}", f"physiclaw.core.{target}")
    assert not bad, "core-internal layering violated:\n" + "\n".join(bad)


def test_geometry_is_a_leaf() -> None:
    """The geometry package imports nothing from physiclaw outside itself."""
    bad = [
        f"{f.relative_to(SRC_ROOT)} -> {m}"
        for f in _files("core/geometry")
        for m in _imported_modules(f)
        if m.startswith("physiclaw") and not m.startswith("physiclaw.core.geometry")
    ]
    assert not bad, "core.geometry must stay a leaf:\n" + "\n".join(bad)
