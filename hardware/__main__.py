"""Unified CLI for the PhysiClaw hardware pipeline — one front door for
every generator.

The legacy invocation is unchanged: ``uv run --group cad python -m hardware
[--custom] [--standard]`` still exports part STEPs exactly as before. Each stage now
also has a named subcommand, and every subcommand forwards its trailing
arguments untouched to the underlying script, so all the original flags
still apply:

    uv run --group cad python -m hardware parts     [--custom] [--standard]   export part STEPs (default)
    uv run --group cad python -m hardware build     [--bom] [--bom-delta]     build assembly steps (STEP + SVG)
    uv run --group cad python -m hardware step      <stem>                    build one step via `build --bom --stems`
    uv run --group cad python -m hardware print                               3D-print package (zip)
    uv run            python -m hardware check                                static consistency check (CI-friendly)
    uv run            python -m hardware manual     [--pdf] [--lang …] …      bilingual build manual
    uv run            python -m hardware sourcing   [--lang …] [--scaffold]   sourcing guide
    uv run --group cad python -m hardware mark      <svg|json>                annotate a step drawing
    uv run --group cad python -m hardware replay    [file]                    replay annotation patches
    uv run --group cad python -m hardware camera    "<freecad-view>"          FreeCAD view → Camera() literal

Geometry commands need ``--group cad`` (build123d); ``check`` and the manual
and sourcing builders are standard-library only. Run all commands from the
repo root.

Each underlying module also stays runnable on its own (e.g.
``uv run --group cad python -m hardware.assembly.build_procedures``) — the
subcommands are a convenience wrapper, not a replacement.
"""

import argparse
import subprocess
import sys

# Subcommand → argv it delegates to (after ``sys.executable``), every one a
# ``-m`` module run.
_DELEGATED: dict[str, list[str]] = {
    "build": ["-m", "hardware.assembly.build_procedures"],
    "check": ["-m", "hardware.check"],
    "print": ["-m", "hardware.parts.build_custom_parts"],
    "mark": ["-m", "hardware.assembly.mark"],
    "replay": ["-m", "hardware.assembly.mark.replay"],
    "camera": ["-m", "hardware.assembly.projection"],
    "manual": ["-m", "hardware.manual.build_manual"],
    "sourcing": ["-m", "hardware.manual.build_sourcing_guide"],
}


def _parts(argv: list[str]) -> int:
    """Export part STEPs — the legacy default behaviour. Imported lazily so
    non-geometry subcommands (manual, sourcing) don't need ``--group cad``."""
    import shutil

    from hardware.parts.base import export_all
    from hardware.parts.export_custom import ALL_PARTS as CUSTOM_PARTS
    from hardware.parts.export_standard import ALL_PARTS as STANDARD_PARTS
    from hardware.scheme import STEP_DIR

    parser = argparse.ArgumentParser(prog="python -m hardware")
    parser.add_argument(
        "--custom", action="store_true", help="export custom parts (default)"
    )
    parser.add_argument("--standard", action="store_true", help="export standard parts")
    args = parser.parse_args(argv)
    export_custom = args.custom or not args.standard
    shutil.rmtree(STEP_DIR, ignore_errors=True)
    if args.standard:
        export_all(STANDARD_PARTS)
    if export_custom:
        export_all(CUSTOM_PARTS)
    return 0


def _delegate(prefix: list[str], argv: list[str]) -> int:
    """Run a sibling entry point in its own process, forwarding ``argv``.
    A subprocess (rather than an in-process import) keeps each script's
    behaviour, sys.path, and exit code identical to its standalone form."""
    return subprocess.call([sys.executable, *prefix, *argv])


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0

    # Pick the subcommand. A bare call or a leading option (--custom /
    # --standard) keeps the legacy default of exporting part STEPs.
    if argv and not argv[0].startswith("-"):
        cmd, rest = argv[0], argv[1:]
    else:
        cmd, rest = "parts", argv

    if cmd == "parts":
        return _parts(rest)
    if cmd == "step":
        if not rest:
            print("usage: python -m hardware step <procedure_stem>", file=sys.stderr)
            return 2
        stem, *step_args = rest
        # One stem through the full `build` pipeline (STEP + SVG + patch replay
        # + its BOM) — i.e. `build --bom --stems <stem>`, reusing build's own
        # stem resolution rather than hand-building a procedures.<stem> path.
        return _delegate(
            ["-m", "hardware.assembly.build_procedures", "--bom", "--stems", stem],
            step_args,
        )
    if cmd in _DELEGATED:
        return _delegate(_DELEGATED[cmd], rest)

    print(f"python -m hardware: unknown command {cmd!r}\n", file=sys.stderr)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
