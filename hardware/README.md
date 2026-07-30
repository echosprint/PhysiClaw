# PhysiClaw — Hardware

**The `hardware` module delivers two documents — all you need to make your own
PhysiClaw hardware:** the **Assembly Manual** shows how to put the machine
together, step by step, and the **Sourcing Guide** lists every part — its spec,
where to buy it, and roughly what it costs. Everything else here — the parts,
the assembly steps, the renderer, the build cache — exists only to generate
those two.

It's **CAD-as-code**: every part, every assembly step, and both documents are
generated from Python — there is no GUI CAD file to hand-edit, and re-running
the scripts reproduces every artifact from source.

The geometry kernel is [build123d](https://build123d.readthedocs.io)
(OpenCASCADE under the hood); the manual and sourcing builders are plain
standard-library Python.

---

## Pipeline at a glance

```text
parts/      standard + custom parts        →  STEP solids
   │
   ▼
assembly/   compose parts into ~70 steps   →  STEP + SVG line-art  (+ per-step Markdown BOM)
   │        each step's placement derives from the upstream chain
   ▼
mark/       annotate the step SVGs         →  snapshot SVGs (raw if unmarked)
   │
   ▼
manual/     content/*.json + step SVGs     →  bilingual HTML / PDF
   └─ content/11_bom.json (the parts list) feeds Section 11 AND ─┐
                                                                 ▼
sourcing/   that same BOM + vendor data    →  bilingual HTML
```

All generated files land under `output/` and are not committed.

---

## Directory layout

```text
hardware/
├── __main__.py            Unified CLI front door (python -m hardware <subcommand>)
├── scheme.py              Shared naming scheme: output dirs, stem convention,
│                          variant/filename builders + their matching regexes
├── check.py               Static cross-artifact consistency check (CI gate)
│
├── parts/                 Parametric part definitions
│   ├── base.py            BasePart: build/export, geometry cache, BOM registry
│   ├── _fits.py           Shared tolerances (clearance holes, nut dims, pitches)
│   ├── standard/          Off-the-shelf parts (screws, nuts, rail, motor, …)
│   ├── custom/            3D-printed / machined parts (clamps, joints, mounts)
│   ├── export_standard.py Export all standard parts → output/step/
│   ├── export_custom.py   Export all custom parts → output/step/
│   └── build_custom_parts.py  Bundle custom STEPs + manifests → print_3d/*.zip
│
├── assembly/              Assembly steps and rendering
│   ├── base.py            BaseAssembly: STEP export + two-variant SVG render
│   ├── projection.py      Camera model + FreeCAD-view → Camera() helper
│   ├── dispatch.py        Procedure discovery, family ordering, batching, retry
│   ├── build_procedures.py  Build & render every step (the main driver)
│   ├── bom.py             BOM library (collect / delta / write_bom)
│   ├── procedures/        ~70 assembly-step modules (<family>_<NN>_<name>.py)
│   ├── patch/             Saved annotation ops, one JSON per drawing
│   └── mark/              Browser tool to annotate step SVGs
│
├── manual/                Bilingual (EN/ZH) assembly manual + sourcing guide
│   ├── build_manual.py        content/*.json + SVGs → HTML / PDF
│   ├── build_sourcing_guide.py  manual BOM + vendor data → HTML
│   ├── assets.py / common.py / paginate.py / pdf.py   Support modules (asset
│   │                          strategies, shared helpers, page numbering +
│   │                          BOM splits, headless-Chrome PDF)
│   ├── content/           13 ordered JSON sections (front + 11 chapters + back)
│   ├── MANUAL_VERSION     The cover's version stamp (single source of truth)
│   └── sourcing_vendors.json   Supplier data, keyed to BOM rows
│
└── output/                Generated artifacts (git-ignored)
    ├── step/  svg/  bom/  manual/  sourcing/  print_3d/  render/
```

---

## Design

**Parts register themselves.** Every `BasePart.build()` does two things beyond
returning geometry: it pushes one row (`bom_key`, `qty`, `category`) into a
process-wide BOM registry, and — for leaf parts — it caches the built solid by
`geom_key` so repeated instances are a cheap copy instead of a rebuild. Parts
are tagged `standard` (purchasable) or `custom` (manufactured for this build).

**Assemblies derive their placement.** Each step is a `BaseAssembly` that
embeds its predecessor and positions new parts **relative to the upstream
chain** rather than from hardcoded coordinates — a clamp's position is computed
by walking the rail → carriage → joint math. This keeps the model
self-consistent: change a part dimension and every downstream step follows.
Assemblies deliberately opt **out** of the geometry cache (caching a whole
compound deep-copies the entire tree); they are recomposed from cached leaves
instead.

**Two variants per step.** Every step builds both `_exploded` (install motion,
with ghost layers) and `_assembled` (finished state) and exports a STEP for
each; SVGs are named `<step>_<variant>_cam<i>.svg`. A class-level `views`
declaration marks which (variant, camera) drawings the manual actually uses —
only those render (each skipped view saves an exact-HLR pass, the dominant
cost and the crash surface). No `views` means render every camera for both
variants. Camera indices stay stable, so trimming never renames files.

**Built in subprocesses.** OpenCASCADE never returns freed memory to the OS, so
building all steps in one process is OOM-killed. `build_procedures` groups steps
by family in dependency order, runs each batch in its own subprocess, and
**retries crashed steps solo** (the hidden-line renderer intermittently
segfaults — up to ~80% per attempt on the heaviest steps). A retry passes
`--only-missing`, re-rendering just the SVGs that are absent, so finished
work is never re-exposed to the crash and the retries converge.

**Naming convention.** Procedure and part files follow
`<family>_<NN>_<descriptor>.py`, where `NN` orders steps within a family in
gaps of 10. Families build in dependency order: `fastener → frame → idler →
motor → linear → belt → tapz → phone → board → camera → wire`.

---

## Prerequisites

- **[uv](https://docs.astral.sh/uv/)** — runs everything; resolves
  dependencies on demand.
- **`--group cad`** — pulls in build123d. Required for any script that touches
  geometry (parts, assemblies, BOM). The manual and sourcing builders are
  standard-library only and need no group.
- **Optional:** a Chromium-family browser for manual PDF export.

Run all commands **from the repo root**.

---

## Script usage

One entry point drives every stage. List the subcommands and their flags
with `--help`:

```bash
uv run --group cad python -m hardware --help
```

The subcommands — each forwarding its flags to the stage it wraps:

| subcommand | stage |
|---|---|
| `parts` | export part STEPs → `output/step/` |
| `build` | build assembly steps (STEP + SVG; `--bom` adds the BOM) |
| `check` | static consistency check — stems, patches, manual figures, sourcing ids |
| `step <stem>` | build one step via `build --bom --stems` (both variants) |
| `print` | 3D-print package → `output/print_3d/*.zip` |
| `manual` | bilingual HTML / PDF manual → `output/manual/` |
| `sourcing` | sourcing guide → `output/sourcing/` |
| `mark` / `replay` | annotate step SVGs / replay saved patches |
| `camera` | FreeCAD camera view → `Camera()` literal |

Geometry subcommands need `--group cad`; `check`, `manual`, and `sourcing`
are standard-library only. Each stage module is also runnable on its own (e.g.
`uv run --group cad python -m hardware.parts.custom.solenoid_mount`).

`build` is incremental: each step's outputs are cached under
`output/.cache/` keyed by a content hash of its import closure (source +
parts + predecessors), the build123d version, and its patch JSON. Unchanged
steps are restored instead of rebuilt+re-rendered — and a patch-only edit
just re-runs the cheap annotation replay. Pass `--no-cache` to force a full
rebuild — the rebuilt outputs still refresh the cache. (`step` always
rebuilds its one stem.)

For shorter typing, the repo `Makefile` wraps each subcommand as a `hw-*`
target (flags via `ARGS`):

```bash
make hw-parts                     # export part STEPs
make hw-build ARGS="--bom"        # build steps + cumulative BOM
make hw-check                     # static consistency check (also run in CI)
make hw-step ARGS=belt_20_clamp   # build one step (= build --bom --stems)
make hw-print                     # 3D-print package (zip)
make hw-manual                    # build the assembly manual (HTML)
make hw-manual-pdf                # build the assembly manual, also as PDF
make hw-sourcing                  # build sourcing guide
make hw-mark ARGS=<svg|json>      # annotate a step drawing
pbpaste | make hw-camera          # FreeCAD view → Camera() literal
make hw-rebuild                   # full rebuild, all stages
make hw-help                      # list every subcommand
```

> **Photoreal render — WIP.** A separate Blender render of the full machine
> (`camera_40_frame`) is being reworked; its scripts were cleared and are not
> currently in the tree. The line-art SVG pipeline is unaffected.

---

## Typical full rebuild

```bash
uv run --group cad python -m hardware parts --custom --standard  # part STEPs
uv run --group cad python -m hardware build --bom                # steps + BOM
uv run --group cad python -m hardware print                      # 3D-print package
uv run            python -m hardware manual                      # the manual
uv run            python -m hardware sourcing                    # the sourcing guide
```

---

## Cutting a release

Each `physiclaw-hardware-vX.Y` release bundles four zips packaged from a
freshly regenerated `output/`:

- **Assembly manual** — the whole `manual/` folder (HTML + PDF in English
  and 中文, plus the step figures).
- **Sourcing guide** — the `sourcing/` folder (HTML in English and 中文).
- **Camera frame** — just the assembled camera-frame STEP file, on its own.
- **Custom parts** — the print package from `output/print_3d/`.

The version stamp on the manual's cover comes from `manual/MANUAL_VERSION` —
bump that file when the manual content changes.

One key does it all:

```bash
make hw-deploy HW_VERSION=X.Y
```

This preflights (version, `gh`, tag free, tree clean and pushed — the tag
points at GitHub's `main`), rebuilds what's stale (the step build is
incremental via `output/.cache`; manual with PDFs and sourcing always
re-render), runs `hw-check`, then packages the four zips and publishes the
release with tag `physiclaw-hardware-vX.Y`. To publish `output/` as-is
without building, use `make hw-release HW_VERSION=X.Y` — it guards that the
artifacts (including the manual PDFs) exist first.

After publishing, redeploy the docs site (docs-site) — it serves the
assembly manual and its PDF from the release assets, so the live pages stay
on the old manual until the site rebuilds.
