"""Build & render every procedure in hardware/assembly/procedures.

Procedures are grouped by family in dependency order and split into
``DEFAULT_BATCH_SIZE`` batches, each run in its own subprocess so the
OS reclaims all OCCT memory on worker exit. OCCT's allocator here is
native malloc, which never returns freed memory to the OS, so a single
all-in-one process balloons across all the procedures and gets
OOM-killed; per-batch subprocesses bound peak RSS. A worker's peak is
set by its single heaviest variant (~1.2 GB on the board_* stems), not
by accumulation across the batch — a per-variant ``gc.collect()``
measured no reduction, so there isn't one. Batches run one at a time,
by choice: parallel dispatch was tried and reverted as not worth the
complexity. Don't run two dispatchers at once — they race on the
shared output dirs.

Exact HLR rendering (``project_to_viewport`` → ``HLRBRep_Algo.Hide``)
intermittently SIGSEGVs — a nondeterministic OCCT hazard, not an OOM;
on the heaviest stems (board_*/camera_*) the crash rate reaches ~80%
per attempt. The dispatcher retries each incomplete stem solo
(``MAX_STEM_RETRIES``) with ``--only-missing``: a retry re-renders only
the cameras whose SVG is absent, so completed HLR passes are never
re-exposed to the crash and the retries converge instead of gambling
the whole stem each attempt. Workers line-buffer stdout (completed rows
survive a crash) and write faulthandler stacks to ``output/crash.log``
rather than stderr, so the console stays a clean progress table.

Each procedure's ``views`` declaration (see ``BaseAssembly``) trims the
render set to the (variant, camera) drawings the manual actually uses —
roughly half of all camera × variant combinations — so unused views
never pay an HLR pass or roll its crash dice. STEPs still export for
both variants regardless.

After each variant renders, any patch JSON in
``hardware/assembly/patch/`` whose stem matches a rendered SVG is
replayed against it, so marker-tool snapshots stay in sync with the
underlying drawing.

The dispatcher is incremental: each stem's outputs are cached under
``output/.cache`` keyed by a content hash of its import closure (and its
patch JSON for the snapshots), so an unchanged stem is restored instead of
rebuilt+re-rendered and a patch-only edit just re-runs the replay. See
``assembly/cache.py``; pass ``--no-cache`` to force a full rebuild.

Pass ``--bom`` / ``--bom-delta`` to also emit each procedure's BOM to
output/bom/ in the same pass — the build is already done, so collecting
the BOM is cheap off the warm leaf cache (one run yields STEP + SVG + BOM).

Run from the repo root:

    uv run --group cad python -m hardware.assembly.build_procedures
    uv run --group cad python -m hardware.assembly.build_procedures --bom
    uv run --group cad python -m hardware.assembly.build_procedures --bom --bom-delta

Worker mode (invoked internally by the dispatcher; crash retries add
``--only-missing`` to skip outputs already on disk):

    uv run --group cad python -m hardware.assembly.build_procedures --stems frame_10_extrusion_tnut frame_20_SHCS ...
"""

from __future__ import annotations

import argparse
import faulthandler
import subprocess
import sys
import time
import traceback
from functools import cache

from hardware.assembly.base import (
    SVG_DIR,
    BaseAssembly,
    svg_path_for,
    variant_suffix,
)
from hardware.assembly.mark.patch import patch_path
from hardware.assembly.mark.replay import replay_one

# Procedure ordering & batching live in dispatch.py so the BOM and render
# pipelines share one source of truth.
from hardware.assembly.dispatch import (
    DEFAULT_BATCH_SIZE,
    MAX_STEM_RETRIES,
    _batches,
    _family_of,
    load_class,
    retry_stems,
)
from hardware.assembly.bom import (
    BOM_DIR,
    write_bom,
)  # only for the optional --bom feature
from hardware.assembly import cache as stepcache
from hardware.parts.base import REPO_ROOT, STEP_DIR

# Worker faulthandler target. The OCCT HLR SIGSEGV is routine, and its full
# dump (thread stack + extension-module list) would bury the progress table
# on every crash — so workers write it here instead of stderr. Fresh per
# dispatch run (the dispatcher unlinks it up front).
CRASH_LOG = REPO_ROOT / "hardware" / "output" / "crash.log"


def _clear_outputs(clear_bom: bool = False) -> None:
    """Wipe stale .step / .svg artifacts (and .md BOMs when ``clear_bom``)
    so deleted procedures and renamed variants don't survive across runs.
    Also sweeps in-progress .tmp files a SIGSEGV'd render or export left
    behind.
    Only touches the file extensions we generate — user-placed files in the
    output dirs are left alone."""
    targets = [
        (STEP_DIR, "*.step"),
        (STEP_DIR, "*.tmp"),
        (SVG_DIR, "*.svg"),
        (SVG_DIR, "*.tmp"),
    ]
    if clear_bom:
        targets.append((BOM_DIR, "*.md"))
    cleared = 0
    for d, pattern in targets:
        if not d.exists():
            continue
        for f in d.glob(pattern):
            f.unlink()
            cleared += 1
    kinds = ".step / .svg / .tmp" + (" / .md" if clear_bom else "")
    print(f"Cleared {cleared} stale {kinds} file(s)\n")


def _run_one(
    cls: type[BaseAssembly], exploded: bool, *, only_missing: bool = False
) -> tuple[float, float, float] | None:
    """Build + export + render one variant. With ``only_missing`` (the
    crash-retry path), work already on disk is not repeated: a variant
    whose STEP and every declared-view SVG are present is skipped
    outright (returns None), an existing STEP is not re-exported, and
    ``render`` skips the cameras whose SVG landed before the crash.
    Both writers are atomic (temp + rename), so file presence means
    complete. Each exact-HLR pass is an independent shot at the
    intermittent OCCT SIGSEGV — never re-rendering finished cameras is
    what makes the retries converge."""
    asm = cls(exploded=exploded)
    svgs = [asm.svg_path(index=i) for i in asm.view_indices]
    step_done = only_missing and asm.output_path().exists()
    if step_done and all(p.exists() for p in svgs):
        return None
    t0 = time.monotonic()
    asm.build()
    t_build = time.monotonic() - t0
    t_export = 0.0
    if not step_done:
        t0 = time.monotonic()
        asm.export()
        t_export = time.monotonic() - t0
    t0 = time.monotonic()
    asm.render(only_missing=only_missing)
    t_render = time.monotonic() - t0
    return t_build, t_export, t_render


def _missing_boms(stem: str, *, bom: bool, bom_delta: bool) -> bool:
    """Any requested BOM .md for ``stem`` absent from output/bom/? Used by
    the ``--only-missing`` retry path: BOM presence is trustworthy there
    because the dispatcher cleared the stem's outputs (BOMs included)
    before its first attempt, so a file on disk is from this build cycle."""
    wanted = [BOM_DIR / f"{stem}.md"] if bom else []
    if bom_delta:
        wanted.append(BOM_DIR / f"{stem}_delta.md")
    return any(not p.exists() for p in wanted)


def _replay_patches_for(stem: str, exploded: bool) -> int:
    """For every SVG just rendered for this procedure variant, replay
    any matching patch JSON in ``hardware/assembly/patch/``. Returns
    the total number of patch-leaf snapshots written (0 if none of the
    rendered SVGs has a patch). Per-source failures are logged but
    don't propagate — patches are user annotations, not build outputs."""
    variant = "exploded" if exploded else "assembled"
    written = 0
    for svg in sorted(SVG_DIR.glob(f"{stem}_{variant}_cam*.svg")):
        if not patch_path(svg).exists():
            continue
        try:
            written += len(replay_one(svg))
        except Exception as exc:  # malformed patch / I/O — keep building
            print(
                f"    WARN: patch replay failed for {svg.name}: "
                f"{type(exc).__name__}: {exc}"
            )
    return written


def _build_stems(
    stems: list[str],
    *,
    bom: bool = False,
    bom_delta: bool = False,
    only_missing: bool = False,
) -> int:
    classes = [(stem, load_class(stem)) for stem in stems]

    name_w = max(len(name) for name, _ in classes)
    runs = len(classes) * 2
    failures: list[tuple[str, str, BaseException]] = []
    sums = {"build": 0.0, "export": 0.0, "render": 0.0}
    t_wall0 = time.monotonic()

    print(f"Building {len(classes)} procedures × 2 variants = {runs} runs\n")
    print(
        f"  {'procedure':<{name_w}}  {'variant':<9}  "
        f"{'build':>7}  {'export':>7}  {'render':>7}  {'total':>7}  status"
    )
    print(
        f"  {'-' * name_w}  {'-' * 9}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}  ------"
    )

    for short, cls in classes:
        for exploded in (True, False):
            variant = "exploded" if exploded else "assembled"
            try:
                timings = _run_one(cls, exploded, only_missing=only_missing)
                if timings is None:
                    # Still replay: the crashed attempt may have died between
                    # the last camera and its patch replay. Idempotent and
                    # geometry-free, so re-running it costs almost nothing.
                    n_patches = _replay_patches_for(short, exploded)
                    patch_tag = f"  +{n_patches} patch snap" if n_patches else ""
                    print(
                        f"  {short:<{name_w}}  {variant:<9}  "
                        f"{'-':>7}  {'-':>7}  {'-':>7}  {'-':>7}  skip (on disk)"
                        f"{patch_tag}"
                    )
                    continue
                tb, te, tr = timings
                sums["build"] += tb
                sums["export"] += te
                sums["render"] += tr
                n_patches = _replay_patches_for(short, exploded)
                patch_tag = f"  +{n_patches} patch snap" if n_patches else ""
                print(
                    f"  {short:<{name_w}}  {variant:<9}  "
                    f"{tb:6.2f}s  {te:6.2f}s  {tr:6.2f}s  {tb + te + tr:6.2f}s  ok"
                    f"{patch_tag}"
                )
            except Exception as exc:
                failures.append((short, variant, exc))
                msg = f"FAIL {type(exc).__name__}: {exc}"
                print(
                    f"  {short:<{name_w}}  {variant:<9}  "
                    f"{'-':>7}  {'-':>7}  {'-':>7}  {'-':>7}  {msg}"
                )

        # BOM (cumulative / delta) from the same warm build — nearly free.
        if bom or bom_delta:
            if only_missing and not _missing_boms(short, bom=bom, bom_delta=bom_delta):
                print(f"  {short:<{name_w}}  bom        skip (on disk)")
                continue
            try:
                write_bom(short, cumulative=bom, want_delta=bom_delta)
                kinds = " + ".join(
                    k for k, on in (("cumulative", bom), ("delta", bom_delta)) if on
                )
                print(f"  {short:<{name_w}}  bom        wrote {kinds}")
            except Exception as exc:
                print(
                    f"  {short:<{name_w}}  bom        FAIL {type(exc).__name__}: {exc}"
                )

    wall = time.monotonic() - t_wall0
    cpu = sum(sums.values())
    print(
        f"\n{runs - len(failures)}/{runs} OK   "
        f"wall {wall:.1f}s   "
        f"build {sums['build']:.1f}s  export {sums['export']:.1f}s  render {sums['render']:.1f}s   "
        f"(cpu/wall = {cpu / wall:.2f})"
    )

    if failures:
        print("\nTracebacks:")
        for short, variant, exc in failures:
            print(f"\n--- {short} ({variant}) ---")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        return 1
    return 0


def _run_subprocess(
    stems: list[str],
    bom_flags: tuple[str, ...] = (),
    *,
    only_missing: bool = False,
) -> int:
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "hardware.assembly.build_procedures",
            "--stems",
            *stems,
            *bom_flags,
            *(("--only-missing",) if only_missing else ()),
        ]
    )


_VARIANTS = (("exploded", True), ("assembled", False))


@cache
def _view_indices(stem: str, exploded: bool) -> tuple[int, ...]:
    """Camera indices this stem renders for this variant — i.e. which
    ``_cam<i>`` SVGs the variant should produce. Read per variant because
    a procedure may set ``self.camera`` differently for exploded vs
    assembled (e.g. frame_10_extrusion_tnut) and ``views`` declares each
    variant separately. Reads the same ``BaseAssembly.view_indices`` that
    ``render()`` iterates, so producer and verifier stay in lockstep."""
    return tuple(load_class(stem)(exploded=exploded).view_indices)


def _missing_variants(stem: str) -> list[str]:
    """Variant names ("exploded" / "assembled") with the STEP or any
    declared-view SVG missing on disk for this stem. A variant is complete
    only when ALL its outputs exist: checking just ``_cam0`` would miss a
    crash *between* cameras — exactly the OCCT HLR SIGSEGV the retry
    exists to recover from — and a variant that declares no views still
    owes its STEP. Both writers are atomic, so presence means complete."""
    return [
        name
        for name, exploded in _VARIANTS
        if not (STEP_DIR / f"{stem}{variant_suffix(exploded)}.step").exists()
        or any(
            not svg_path_for(stem, exploded, index=i).exists()
            for i in _view_indices(stem, exploded)
        )
    ]


def _dispatch(
    batch_size: int, bom_flags: tuple[str, ...] = (), use_cache: bool = True
) -> int:
    """Build every batch, skipping steps the cache already covers.

    Per stem (see ``assembly/cache.py``): if the *source* layer is fresh
    (.step / raw .svg / BOM unchanged) restore it instead of rebuilding,
    then either restore the patch snapshots (patch unchanged) or just
    re-run the cheap annotation replay (patch edited — geometry stays
    cached). Stems with no cache hit go to a worker subprocess in family
    batches; crashed stems solo-retry. ``bom_flags`` (--bom / --bom-delta)
    pass through so workers also emit BOMs."""
    want_bom = bool(bom_flags)
    CRASH_LOG.unlink(missing_ok=True)  # this run's workers append fresh
    # With the cache on, don't wipe everything up front (it would just be
    # restored again) — the cache owns each stem's freshness, and only the
    # miss stems are cleared, right before they rebuild. A --no-cache run
    # still wipes all stale outputs so deleted/renamed steps don't survive.
    if not use_cache:
        _clear_outputs(clear_bom=want_bom)

    batches = _batches(batch_size)
    position = {s: i + 1 for i, s in enumerate(s for b in batches for s in b)}
    total = len(position)

    # Classify each stem once (source_cached verifies the entry's hashes — not
    # something to run twice); the loop below reuses this map.
    cached: dict[str, bool] = {}
    if use_cache:
        pruned = stepcache.prune(position)
        cached = {s: stepcache.source_cached(s, want_bom=want_bom) for s in position}
        n_cached = sum(cached.values())
        print(
            f"cache: {stepcache.CACHE_DIR.relative_to(REPO_ROOT)} — "
            f"{n_cached}/{total} cached, {total - n_cached} to build"
            + (f"   (pruned {pruned} stale)" if pruned else "")
        )
    t_wall0 = time.monotonic()

    def header(stems: list[str]) -> str:
        start, end = position[stems[0]], position[stems[-1]]
        progress = f"{start}/{total}" if start == end else f"{start}-{end}/{total}"
        tail = stems[0] if start == end else f"{stems[0]} … {stems[-1]}"
        return f"\n=== [{progress}] {_family_of(stems[0])}: {tail} ==="

    n_fresh = n_replay = 0
    for batch in batches:
        misses = []
        for s in batch:
            if not cached.get(s):
                misses.append(s)
                continue
            stepcache.restore_source(s, want_bom=want_bom)
            if stepcache.snapshots_cached(s):
                stepcache.restore_snapshots(s)
                n_fresh += 1
            else:
                # Patch edited but geometry/render unchanged: keep the cached
                # .step + raw .svg, just re-apply the (build123d-free) replay.
                print(f"  [patch] {s}: geometry cached, re-replaying annotations")
                for exploded in (True, False):
                    _replay_patches_for(s, exploded)
                stepcache.store_snapshots(s)
                n_replay += 1

        if not misses:
            continue
        if use_cache:
            # Clear each miss stem's own outputs so a rebuild that now renders
            # fewer cameras/variants leaves no stale file behind.
            for s in misses:
                stepcache.clear_outputs(s)
        print(header(misses))
        rc = _run_subprocess(misses, bom_flags)
        if rc != 0:
            # Nondeterministic HLR SIGSEGV: re-run each incomplete stem in a
            # fresh process (new heap layout) until its outputs land or we
            # exhaust the retries.
            incomplete = [s for s in misses if _missing_variants(s)]
            print(
                f"\n--- batch exit {rc}; retrying {len(incomplete)}/{len(misses)} "
                f"incomplete stem(s)  [stacks → {CRASH_LOG.relative_to(REPO_ROOT)}] ---"
            )
            retry_stems(
                incomplete,
                # only_missing: a retry re-renders just the cameras whose
                # SVG is absent — finished work is never re-exposed to the
                # HLR crash, so each attempt only has to survive the small
                # remainder and the retries converge.
                run=lambda s: _run_subprocess([s], bom_flags, only_missing=True),
                done=lambda s: not _missing_variants(s),
                log=lambda s, a: print(
                    header([s]) + f"  (retry {a}/{MAX_STEM_RETRIES})"
                ),
            )
        # Store every stem that completed — also on --no-cache runs: the
        # flag bypasses cache READS, but the fresh rebuild is exactly what
        # is worth storing; skipping the store would make the next cached
        # run rebuild all steps a second time.
        for s in misses:
            if not _missing_variants(s):
                stepcache.store_source(s)
                stepcache.store_snapshots(s)

    wall = time.monotonic() - t_wall0
    failed_variants = [(s, v) for s in position for v in _missing_variants(s)]
    ok_assemblies = sum(1 for s in position if not _missing_variants(s))
    n_step = len(list(STEP_DIR.glob("*.step")))
    n_svg = len(list(SVG_DIR.glob("*.svg")))
    cache_note = (
        f"   cache: {n_fresh} hit + {n_replay} patch-replay" if use_cache else ""
    )
    tally = (
        f"{ok_assemblies}/{total} assemblies   "
        f"wrote {n_step} .step / {n_svg} .svg{cache_note}   "
        f"total wall {wall:.1f}s"
    )
    if failed_variants:
        print(f"\nFAILED variants ({len(failed_variants)}):")
        for stem, variant in failed_variants:
            print(f"  {stem} {variant}")
        print(tally)
        return 1
    print(f"\nAll OK   {tally}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stems",
        nargs="+",
        help="procedure stems to build (worker mode); omit to dispatch all batches as subprocesses",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"max procedures per subprocess in dispatcher mode (default {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--bom",
        action="store_true",
        help="also write each procedure's cumulative BOM to output/bom/",
    )
    parser.add_argument(
        "--bom-delta",
        action="store_true",
        help="also write each procedure's delta-vs-predecessor BOM to output/bom/",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="don't read the persistent step cache — rebuild & re-render every "
        "step (the rebuilt outputs still refresh the cache)",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="skip variants/cameras whose outputs are already on disk "
        "(worker mode; the dispatcher's crash retries pass this)",
    )
    args = parser.parse_args(argv)

    # Line-buffer stdout: under a pipe (make … | tee, CI) the default is
    # block buffering, which holds the dispatcher's progress until exit and
    # loses a crashed worker's entire report with the SIGSEGV.
    sys.stdout.reconfigure(line_buffering=True)

    if args.stems:
        # Route crash forensics to CRASH_LOG instead of stderr; the header
        # line ties each dumped stack to the worker's stems. Deliberately
        # not a ``with`` block: faulthandler writes to this fd at signal
        # time, so it must stay open for the life of the process.
        CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
        crash_log = CRASH_LOG.open("a")
        crash_log.write(
            f"=== {time.strftime('%H:%M:%S')} worker --stems {' '.join(args.stems)}\n"
        )
        crash_log.flush()
        faulthandler.enable(file=crash_log)
        return _build_stems(
            args.stems,
            bom=args.bom,
            bom_delta=args.bom_delta,
            only_missing=args.only_missing,
        )

    bom_flags = tuple(
        f for f, on in (("--bom", args.bom), ("--bom-delta", args.bom_delta)) if on
    )
    return _dispatch(args.batch_size, bom_flags, use_cache=not args.no_cache)


if __name__ == "__main__":
    sys.exit(main())
