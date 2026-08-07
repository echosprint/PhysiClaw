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


def test_provider_imports_only_dto_from_engine() -> None:
    """The provider layer's one engine edge is the shared type vocabulary,
    `engine.dto` — engine behavior modules stay out of providers (shared
    behavior belongs in `common`, e.g. `common.image`). Bare
    `from physiclaw.agent.engine import X` resolves to the package here
    and fails too: import dto by its full path."""
    bad = [
        v
        for v in _violations("agent/provider", "physiclaw.agent.engine")
        if not v.endswith("-> physiclaw.agent.engine.dto")
    ]
    assert not bad, (
        "provider may import only physiclaw.agent.engine.dto from the engine:\n"
        + "\n".join(bad)
    )


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
