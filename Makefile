.PHONY: test test-cov test-fast test-slow test-integration test-all mutate lint fmt typecheck help bump build publish release \
        hw-help hw-parts hw-build hw-check hw-step hw-print hw-manual hw-manual-pdf hw-sourcing hw-mark hw-replay hw-camera hw-rebuild hw-preflight hw-deploy hw-release _hw-package

PY ?= uv run

# Hardware CAD pipeline (see hardware/README.md). Geometry stages need the
# `cad` group; the manual / sourcing builders are standard-library only.
# Pass flags through ARGS, e.g.  make hw-build ARGS="--bom --bom-delta"
HW     = $(PY) --group cad python -m hardware
HW_DOC = $(PY) python -m hardware

# Hardware GitHub release — assets are packaged from hardware/output/.
# Versioned separately from the PyPI package: set HW_VERSION per release,
# e.g.  make hw-deploy HW_VERSION=0.3  (build + publish in one key)
HW_OUT     = hardware/output
HW_REL_DIR = $(HW_OUT)/release
HW_REL_TAG = physiclaw-hardware-v$(HW_VERSION)

# Guard for hw-* targets that need a positional: fail with a usage hint when
# ARGS is empty. $(1) describes the expected value.
need-args = @if [ -z "$(ARGS)" ]; then echo 'usage: make $@ ARGS=$(1)'; exit 2; fi

# Version currently declared in pyproject.toml — used by `build` and `publish`
# so the user only types it once (in `make bump`).
PKG_VERSION = $(shell sed -n 's/^version = "\(.*\)"$$/\1/p' pyproject.toml)

help:
	@echo "Targets:"
	@echo "  test                  — fast unit suite (default; excludes slow + integration)"
	@echo "  test-cov              — fast suite with coverage report (term + html)"
	@echo "  test-fast             — alias for test"
	@echo "  test-slow             — only @pytest.mark.slow"
	@echo "  test-integration      — only @pytest.mark.integration"
	@echo "  test-all              — every test, including slow and integration"
	@echo "  mutate MOD=path       — mutmut on a path (e.g. MOD=src/physiclaw/agent/engine/validator.py)"
	@echo "  lint                  — ruff check + format check"
	@echo "  fmt                   — ruff format (src, tests, hardware, scripts)"
	@echo "  typecheck             — mypy on src/physiclaw (scope in [tool.mypy])"
	@echo "  bump [VERSION=X.Y.Z]  — bump version. Defaults to incrementing the last"
	@echo "                          component split by '.' (0.0.7 → 0.0.8). Override with"
	@echo "                          VERSION for major/minor jumps. Commits LOCALLY."
	@echo "  build                 — uv build wheel + sdist for the version in pyproject.toml"
	@echo "  publish               — upload dist/* to PyPI, tag vX.Y.Z, push. Irreversible."
	@echo "                          Reads version from pyproject.toml. Needs UV_PUBLISH_TOKEN."
	@echo "  release [VERSION=X.Y.Z]"
	@echo "                        — full release: bump + build + publish in one shot."
	@echo ""
	@echo "Hardware (CAD-as-code) — pass flags via ARGS=\"...\":"
	@echo "  hw-help                 — list the hardware CLI subcommands"
	@echo "  hw-parts                — export part STEPs"
	@echo "  hw-build [ARGS=--bom]   — build assembly steps (STEP + SVG)"
	@echo "  hw-check                — static consistency check (stems/patches/manual/sourcing)"
	@echo "  hw-step ARGS=<stem>     — build one step (= build --bom --stems)"
	@echo "  hw-print                — 3D-print package (zip)"
	@echo "  hw-manual [ARGS=--pdf]  — bilingual build manual"
	@echo "  hw-sourcing             — sourcing guide"
	@echo "  hw-mark ARGS=<svg|json> — annotate a step drawing"
	@echo "  hw-replay [ARGS=file]   — replay annotation patches"
	@echo "  hw-camera ARGS=\"...\"    — FreeCAD camera view → Camera() literal"
	@echo "                            (or pipe: pbpaste | make hw-camera)"
	@echo "  hw-rebuild              — full rebuild: parts → build → print → manual+pdf → sourcing"
	@echo "  hw-deploy HW_VERSION=X.Y  — one-key release: cached build → docs → check → publish"
	@echo "  hw-release HW_VERSION=X.Y — package output/ as-is into the 4 zips & publish (gh)"

test:
	$(PY) pytest

test-cov:
	$(PY) pytest --cov=src/physiclaw --cov-report=term-missing --cov-report=html --cov-branch

test-fast: test

test-slow:
	$(PY) pytest -m slow

test-integration:
	$(PY) pytest -m integration

test-all:
	$(PY) pytest -m ""

mutate:
	@if [ -z "$(MOD)" ]; then echo "usage: make mutate MOD=src/physiclaw/<path>"; exit 2; fi
	$(PY) mutmut run --paths-to-mutate $(MOD)

lint:
	$(PY) ruff check src/ tests/
	$(PY) ruff format --check src/ tests/ hardware/ scripts/

fmt:
	$(PY) ruff format src/ tests/ hardware/ scripts/

# Scope (src/physiclaw) and settings live in [tool.mypy] in pyproject.toml.
typecheck:
	$(PY) mypy

# --- Release workflow ---------------------------------------------------------
#
# Three atoms; only the last one is irreversible:
#
#   make bump VERSION=0.0.8   # edits pyproject + uv.lock, commits LOCALLY
#   make build                # uv build (reads version from pyproject)
#   make publish              # uv publish, git tag vX.Y.Z, git push. Irreversible.
#
# Up to (and including) `build`, everything is reversible:
#   git reset --hard HEAD~        # undo the bump commit
#   rm dist/physiclaw-0.0.8*      # discard the build
#
# `publish` reads the version from pyproject.toml so you only type it once
# (in `bump`). Tag creation happens in `publish` — that way a failed build /
# abandoned attempt doesn't leave a stale local tag for a version that was
# never released.

bump:
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "✗ working tree dirty — commit or stash first"; exit 1; fi
	@if [ "$$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then \
		echo "✗ not on main branch"; exit 1; fi
	@set -e; \
	if [ -n "$(VERSION)" ]; then \
		NEW="$(VERSION)"; \
	else \
		LAST="$$(echo $(PKG_VERSION) | awk -F. '{print $$NF}')"; \
		case "$$LAST" in *[!0-9]*|"") \
			echo "✗ can't auto-bump non-numeric last component '$$LAST' of $(PKG_VERSION)"; \
			echo "  use 'make bump VERSION=X.Y.Z' explicitly"; exit 1;; \
		esac; \
		NEW="$$(echo $(PKG_VERSION) | awk 'BEGIN{FS=OFS="."} {$$NF = $$NF + 1; print}')"; \
		echo "Auto-incrementing $(PKG_VERSION) → $$NEW"; \
	fi; \
	if git rev-parse "v$$NEW" >/dev/null 2>&1; then \
		echo "✗ tag v$$NEW already exists"; exit 1; fi; \
	sed -i.bak "s/^version = \"[^\"]*\"$$/version = \"$$NEW\"/" pyproject.toml; \
	rm pyproject.toml.bak; \
	uv lock; \
	git add pyproject.toml uv.lock; \
	git commit -m "chore: bump version to $$NEW"; \
	printf '\n\033[32m✓\033[0m Bumped to %s.\n' "$$NEW"; \
	printf 'Next:  make build  &&  make publish\n\n'

build:
	@if [ -z "$(PKG_VERSION)" ]; then \
		echo "✗ couldn't read version from pyproject.toml"; exit 1; fi
	@rm -f dist/physiclaw-$(PKG_VERSION)*
	uv build
	@printf '\n\033[32m✓\033[0m Built $(PKG_VERSION):\n'
	@echo "  dist/physiclaw-$(PKG_VERSION)-py3-none-any.whl"
	@echo "  dist/physiclaw-$(PKG_VERSION).tar.gz"

publish:
	@if [ -z "$(PKG_VERSION)" ]; then \
		echo "✗ couldn't read version from pyproject.toml"; exit 1; fi
	@if [ -z "$$UV_PUBLISH_TOKEN" ]; then \
		echo "✗ UV_PUBLISH_TOKEN not set"; \
		echo "  Create a PyPI API token: https://pypi.org/manage/account/token/"; \
		echo "  Then: export UV_PUBLISH_TOKEN=pypi-AgEI..."; \
		exit 1; fi
	@if [ ! -f "dist/physiclaw-$(PKG_VERSION)-py3-none-any.whl" ]; then \
		echo "✗ dist/physiclaw-$(PKG_VERSION)-py3-none-any.whl not found"; \
		echo "  Run 'make build' first"; exit 1; fi
	@if git rev-parse "v$(PKG_VERSION)" >/dev/null 2>&1; then \
		echo "✗ tag v$(PKG_VERSION) already exists — was this version already published?"; \
		exit 1; fi
	@printf 'Uploading to PyPI:\n'
	@printf '  dist/physiclaw-$(PKG_VERSION)-py3-none-any.whl\n'
	@printf '  dist/physiclaw-$(PKG_VERSION).tar.gz\n'
	@COUNT=$$(ls dist/physiclaw-*.whl 2>/dev/null | grep -vc "physiclaw-$(PKG_VERSION)-py3" || true); \
	 if [ "$$COUNT" -gt 0 ]; then \
		printf '  (%s older wheels in dist/ ignored)\n' "$$COUNT"; \
	 fi
	@printf '\n'
	uv publish dist/physiclaw-$(PKG_VERSION)-py3-none-any.whl dist/physiclaw-$(PKG_VERSION).tar.gz
	git tag -a v$(PKG_VERSION) -m "physiclaw v$(PKG_VERSION)"
	git push origin main
	git push origin v$(PKG_VERSION)
	@printf '\n\033[32m✓\033[0m Published $(PKG_VERSION) to PyPI and pushed to GitHub.\n'
	@echo "  https://pypi.org/project/physiclaw/$(PKG_VERSION)/"

# `make release [VERSION=X.Y.Z]` — bump + build + publish in one shot.
#
# Make reserves `-p` (print database), so a `make bump -p` style custom
# flag isn't possible — this meta-target is the equivalent.
#
# Make runs prerequisites left-to-right (no -j); the deferred-eval
# PKG_VERSION re-reads pyproject.toml at recipe time, so build and
# publish pick up the freshly-bumped version that bump just wrote.
release: bump build publish
	@printf '\n\033[32m✓\033[0m Release flow complete for $(PKG_VERSION).\n'

# --- Hardware CAD pipeline ----------------------------------------------------
# Thin wrappers over `python -m hardware <subcommand>` (see `make hw-help` and
# hardware/README.md). hw-prefixed so `build` stays the wheel build above.
# Flags / positionals go through ARGS, e.g.  make hw-build ARGS="--bom".

hw-help:
	$(HW) --help

hw-parts:
	$(HW) parts $(ARGS)

hw-build:
	$(HW) build $(ARGS)

# Static cross-artifact consistency check (stems / patches / manual /
# sourcing) — stdlib-only, no cad group, also run in CI.
hw-check:
	$(HW_DOC) check

hw-step:
	$(call need-args,<procedure_stem>)
	$(HW) step $(ARGS)

hw-print:
	$(HW) print $(ARGS)

hw-manual:
	$(HW_DOC) manual $(ARGS)

# Convenience alias for `make hw-manual ARGS="--pdf"`; extra flags still go
# through ARGS, e.g.  make hw-manual-pdf ARGS="--lang en".
hw-manual-pdf:
	$(HW_DOC) manual --pdf $(ARGS)

hw-sourcing:
	$(HW_DOC) sourcing $(ARGS)

hw-mark:
	$(call need-args,<svg|json>)
	$(HW) mark $(ARGS)

hw-replay:
	$(HW) replay $(ARGS)

# No need-args guard: `camera` also reads the view from stdin when ARGS is
# empty (projection.py), so `pbpaste | make hw-camera` works too.
hw-camera:
	$(HW) camera $(ARGS)

# Full rebuild — STEPs → steps+BOM → print package → manual (HTML+PDF) →
# sourcing. Includes --pdf so the result always satisfies hw-release's
# artifact guard: a "full" rebuild is a releasable one.
hw-rebuild:
	$(HW) parts --custom --standard
	$(HW) build --bom
	$(HW) print
	$(HW_DOC) manual --pdf
	$(HW_DOC) sourcing

# Shared release preflight — every check that can fail does so here, before
# any long build. Beyond the version/gh/tag guards: `gh release create
# --target main` tags GitHub's main, not the checkout — a dirty tree or
# unpushed commits would silently ship a release without them.
hw-preflight:
	@if [ -z "$(HW_VERSION)" ]; then echo 'usage: make hw-deploy|hw-release HW_VERSION=X.Y'; exit 2; fi
	@command -v gh >/dev/null 2>&1 || { echo "✗ gh (GitHub CLI) not found — https://cli.github.com/"; exit 1; }
	@if gh release view "$(HW_REL_TAG)" >/dev/null 2>&1; then echo "✗ release $(HW_REL_TAG) already exists"; exit 1; fi
	@if [ -n "$$(git status --porcelain)" ]; then echo "✗ working tree not clean — commit (and push) first"; exit 1; fi
	@if [ "$$(git rev-parse main)" != "$$(git rev-parse origin/main)" ]; then echo "✗ local main != origin/main — push first"; exit 1; fi

# One-key deploy — build what's stale and publish the release. `build` is
# incremental (content-hash cache under output/.cache), so unchanged stems
# restore instead of rebuilding; the doc builders are fast and always re-run,
# so manual/sourcing can never ship stale. Preflight runs first so a bad
# version/tag/tree fails in seconds, not after the build; the build writes
# only to gitignored output/, so the checks can't go stale before _hw-package.
#   make hw-deploy HW_VERSION=0.3
hw-deploy: hw-preflight
	$(HW) build --bom
	$(HW) print
	$(HW_DOC) manual --pdf
	$(HW_DOC) sourcing
	$(HW_DOC) check
	$(MAKE) _hw-package HW_VERSION=$(HW_VERSION)

# Package what's already in hardware/output into the four release zips and
# publish them as a GitHub release. Prefer `hw-deploy`, which builds first;
# this exists for re-publishing an output/ that is already built and checked.
#   make hw-release HW_VERSION=0.3
hw-release: hw-preflight _hw-package

# Internal packaging body — no guards of its own beyond the artifact check;
# reached only through hw-deploy / hw-release, which both run hw-preflight
# exactly once. The artifact guard catches a missing or HTML-only build
# (zip -r would happily package a manual without its PDFs); the filenames
# mirror their owners — LANG_FILENAME in build_manual.py /
# build_sourcing_guide.py, ZIP_PATH in build_custom_parts.py, scheme.py's
# step naming — keep them in sync. set -e aborts before the release is cut,
# so no half-published release.
_hw-package:
	@set -e; \
	rm -rf "$(HW_REL_DIR)"; mkdir -p "$(HW_REL_DIR)"; \
	REL="$$(cd "$(HW_REL_DIR)" && pwd)"; cd "$(HW_OUT)"; \
	for f in print_3d/physiclaw_custom_parts.zip \
	         step/camera_40_frame_assembled.step \
	         manual/physiclaw_manual.html manual/physiclaw_manual.pdf \
	         manual/physiclaw装配手册.html manual/physiclaw装配手册.pdf \
	         sourcing/sourcing_guide.html sourcing/physiclaw采购指南.html; do \
		[ -f "$$f" ] || { echo "✗ missing $(HW_OUT)/$$f — run: make hw-deploy HW_VERSION=$(HW_VERSION)"; exit 1; }; \
	done; \
	cp print_3d/physiclaw_custom_parts.zip "$$REL"/; \
	zip -jq "$$REL/physiclaw_camera_frame_assembled.zip" step/camera_40_frame_assembled.step; \
	zip -rq "$$REL/physiclaw-assembly-manual.zip" manual   -x '*.DS_Store'; \
	zip -rq "$$REL/physiclaw-sourcing-guide.zip"  sourcing -x '*.DS_Store'; \
	printf '%s\n' \
		'Build artifacts for assembling a PhysiClaw rig (English + 中文).' '' \
		'- **physiclaw-assembly-manual.zip** — full assembly manual: HTML + PDF in English and 中文, with all exploded/step SVG figures.' \
		'- **physiclaw-sourcing-guide.zip** — sourcing guide: HTML in English and 中文.' \
		'- **physiclaw_custom_parts.zip** — the 9 custom 3D-printed parts as STEP files (print in black PA12 via SLS/MJF) plus a bilingual print guide.' \
		> "$$REL/notes.md"; \
	gh release create "$(HW_REL_TAG)" "$$REL"/*.zip \
		--target main --latest \
		--title "PhysiClaw hardware v$(HW_VERSION) — assembly manual, sourcing guide & printed parts (STEP)" \
		--notes-file "$$REL/notes.md"; \
	printf '\n\033[32m✓\033[0m Released $(HW_REL_TAG).\n'; \
	printf '\033[33m!\033[0m The docs site serves the manual from this release — redeploy docs-site to pick it up.\n'
