# PhysiClaw Whole-Repo Refinement Plan

Produced 2026-07-13 by a multi-agent audit (14 module/architecture reviewers, 47
adversarial verifiers, synthesis + completeness critic). Of the high/medium
findings, 38 were confirmed under adversarial verification, 9 refuted, and 47
low-severity items carried as unverified polish. The completeness critic's five
corrections are merged below and marked *(critic)*.

## Architecture verdict

PhysiClaw is in genuinely good shape for a pre-1.0 single-maintainer project:
the agent/core MCP-boundary doctrine holds at the import level, the engine's
mechanics/judgment split, the provider template-method design, the vision
layer's threshold provenance, and the hardware layer's safety invariants
(`_energized()`, exposure DI) are all above the bar most teams ship at, and
rationale-dense docstrings make the system auditable from source. The debt is
concentrated, not diffuse, in four buckets:

1. **The quality gate is currently theater** — 277 hermetic tests are
   quarantined behind a mislabeled `integration` marker, CI runs bare `pytest`
   with no coverage, and one quarantined test has already rotted to failure on
   main (verified: full combined run = 3131 passed / 1 failed, 25.8s, 95.86%
   branch coverage).
2. **Real failure-path bugs** exactly where a phone-operating robot can't
   afford them: an orphanable live `claude` subprocess, a camera `close()`
   deadlock, a camera auto-pick crash, a sentinel regex that parses
   `>> DONEDEAL` as DONE.
3. **Cross-boundary vocabulary duplicated as string literals** held together by
   "keep in sync" comments (gesture tool names, bbox validation, the server
   URL in 7 copies), invisible to the architecture guard.
4. **Four god modules** — `calibrate.py` (1092 lines), `spawn.py` (807),
   `setup/hardware.py`'s 254-line `run()`, calibration `handler.py`'s six-fold
   lock boilerplate — whose size demonstrably blocks the repo's own coverage
   standard.

Fix the gate first — everything after is safer with it on.

## Phase 1 — Make the test gate real again

**Status: completed 2026-07-13** — 3116 passed / 1 skipped, 95.84% branch
coverage in 27.6s; lint clean. `pytest -m integration` now collects zero
tests by design: the marker is reserved until Phase 5 adds the
`tests/hardware/` replay tests.

**Leverage: maximal. Risk: near zero. Effort: ~1 person-day.**

**Goal:** every hermetic test runs in CI, coverage actually gates, and the one
rotted test is fixed — so all later phases land against a live safety net.

Resolves:

- `tests/core/calibration/test_calibrate.py:36` (+11 other fake-backed files) —
  mis-marked `integration` (HIGH)
- `pyproject.toml:82` / `.github/workflows/ci.yml:48` — unenforced 90% gate (HIGH)
- `tests/agent/engine/test_engine_loop.py:48` — suite file-marked `slow`
  though all 65 tests run in 1.28s total (MEDIUM)
- `src/physiclaw/core/vision/screen_match.py` — confirmed dead module
- `Makefile:85` / pyproject — no `[tool.ruff.lint]`; import-sort drift

Steps:

1. Fix the rotted `test_measure_viewport_shift_png_cache_extension` (stub
   `take_pending_screenshot`, not `wait_screenshot`).
2. Strip `integration` marks from the fake-backed files (test_arm, test_camera,
   test_calibrate, test_tools, test_bridge, test_watch, test_server_hardware,
   test_warm_start's phase-5 decorators + dead `pytestmark_phase5`,
   test_calibration, test_spawn_claude, test_runtime, test_launcher); reserve
   the marker for tests that truly open a port/camera. Remove the file-level
   `slow` mark from test_engine_loop.py.
3. Change ci.yml's test step to `uv run pytest --cov=src/physiclaw
   --cov-branch` — the combined run was measured at 95.86% branch coverage in
   25.8s, comfortably over the 90% gate.
4. Delete `screen_match.py` + `tests/core/vision/test_screen_match.py`, drop it
   from the `vision/__init__.py` docstring (zero production imports).
5. Add `[tool.ruff.lint]` with `select = ["E4","E7","E9","F","I"]` and run
   `ruff check --fix --select I` (112 auto-fixable). Do **not** enable
   ARG/BLE/C901 in this pass (~435 findings, mostly deliberate patterns and
   pytest noise).
6. Delete empty `tests/dashboard` and `tests/ops`.
7. *(critic)* The strip list in step 2 is exhaustive — every `integration` mark
   in the tree — so afterward `pytest -m integration` would collect zero tests
   (exit code 5). Land the `hardware/assembly/mark/replay.py` integration
   marking (from Phase 5) together with this phase so the reserved marker is
   never momentarily meaningless.

**Proof:** CI goes red-then-green on the coverage gate itself; `pytest -m
integration` collects only genuinely hardware-bound tests; full suite (3100+
tests) passes in <30s.

## Phase 2 — Failure-path correctness on the live-hardware paths

**Status: completed 2026-07-13** — all 13 fixes landed; every src/ fix
has pinned regression tests (23 new tests). The `cache.py` `_imports()`
fix is verified by direct execution only — its test lands with Phase 5's
`tests/hardware/`. 3139 passed, 95.86% branch coverage, `ruff check`
clean; touched files are format-clean (pre-existing `ruff format` drift
remains in screen_layout.py, calibrate.py, test_spawn.py — out of scope
here). Notes: camera close() got tunable CLOSE_JOIN/LOCK timeout
constants; spawn.py got EXIT_WAIT_SECONDS for the post-EOF bound; the
wizard's 'q' now aborts inside ask() itself (every ask gates a mandatory
step); sync_official_skills exports SyncError.

**High leverage, small independent diffs. Effort: ~3 person-days.**

**Goal:** eliminate the confirmed bugs where a one-off glitch orphans a
process, wedges a thread, or silently disarms a safety check. All S-effort,
independently landable, guarded by the Phase 1 gate.

- `src/physiclaw/agent/claude/spawn.py:766` — un-exception-safe subprocess
  lifetime (HIGH): construct `_SessionLog` before spawn (or fail-open its
  `__init__`), generalize the kill to a try/finally covering all exceptions
  (init `proc = None` first), bound the post-EOF `proc.wait()` with
  `asyncio.wait_for`, catch `ValueError` from oversized readline.
- `src/physiclaw/core/hardware/handler.py:116` — auto-pick crashes on first
  missing camera index (HIGH): move `Camera(idx)` inside the try, None-guard
  the finally, add constructor-raise test.
- `src/physiclaw/core/hardware/camera.py:471` — `close()` deadlock: use
  timeout-guarded `_cap_lock.acquire(timeout=2.0)`; **never** release the cap
  without the lock (native race can segfault the whole server); log-and-leak
  on failure.
- `src/physiclaw/agent/runtime/sentinel.py:17` — add `\b` after the status
  alternation; add `>> DONEDEAL` / `> waiting on user` negative tests.
- `src/physiclaw/agent/engine/policy.py:450` — KeyboardBelief demoted by failed
  *local* tools: guard the `state = "unknown"` degrade by gesture family only
  (do **not** skip result observers wholesale — StuckRecorder needs failed-call
  evidence).
- `src/physiclaw/agent/provider/openai_compat.py:120` — wrap `r.json()` in
  try/except ValueError → `ProviderTransientError`; use
  `(raw.get("choices") or [{}])[0]` (matching google.py); add 200-non-JSON and
  empty-choices tests. *(critic)* Also fold in `:132` — `list_models()` has the
  identical unguarded `body = r.json()`, which otherwise escapes as a raw
  `JSONDecodeError` into `cli/models.py`.
- *(critic)* `src/physiclaw/agent/hooks/poll.py:48` — `data = r.json()` sits
  outside the try/except implementing the deliberate warn-once `_in_blip`
  pattern, so a 200 response with a non-JSON body (server mid-restart, proxy
  interception) emits a full `log.exception` traceback every ~2s tick and
  bypasses the blip logic. Move the parse inside the try.
- `src/physiclaw/core/bridge/handler.py:206` — LAN routes 500 on malformed
  input: `_json_or_400` helper; 400 on bad/missing fields, 409 when
  viewport_shift is None.
- `src/physiclaw/common/dumps.py:31` — the `_ENSURED` mkdir cache breaks
  `physiclaw clear`'s documented "dirs recreated on next save" contract for a
  live server: drop the cache (mkdir per dump is idempotent) or fail-open the
  save bodies; update `test_mkdir_caches_ensured_dirs`.
- `src/physiclaw/cli/doctor.py:406` — call `_cfg.load()` in try/except
  ConfigError instead of trusting the stale import invariant; also patch
  `_cfg.load` in the test helper.
- `src/physiclaw/cli/setup/hardware.py:174` — make 'q' actually quit (or move
  `_done` inside the `if ask(...)` and abort on declined mandatory steps);
  align docstring.
- `src/physiclaw/cli/setup/hardware.py:150` — fix the port parse with
  `urlsplit(BASE).port or 8048`; **keep the module-global BASE pattern** — it
  is documented and test-pinned.
- `src/physiclaw/cli/sync_official_skills.py:419` — introduce `SyncError`
  raised by helpers; CLI command converts to `exit_error`, `_run_sync_quiet`
  logs via `log.warning` so failures reach the daily log.
- `hardware/assembly/cache.py:93` — `_imports()` drops package-form submodule
  imports from the dependency closure (verified by direct execution → silently
  stale CAD cache): also resolve `f"{node.module}.{alias.name}"` per alias
  (~4 lines).

**Proof:** each fix ships with its pinned regression test in the 1:1 mirror;
existing pinned tests (`test_spawn_claude_kills_on_timeout`,
`test_close_stops_thread_and_releases`, provider error-prefix pins) pass
unchanged.

## Phase 3 — Single-source the cross-boundary vocabulary

**Kills sync-by-comment drift. Effort: ~3.5 person-days.**

**Goal:** move the duplicated contracts into the sanctioned `common` leaf (and
one intra-engine leaf), with contract tests, so the next rename can't erode
silently. Depends on Phase 1's gate; makes Phase 4's file moves mechanical.

- `src/physiclaw/core/vision/util.py:145` ↔ `agent/engine/validator.py:76` —
  the byte-identical 4-check bbox validator ("keep the two in sync" comments)
  → `common/bbox.py`: common raises ValueError; core keeps `validate_bbox` as
  thin re-export (return-identity test holds); engine wraps into
  `ValidationError`. Messages stay byte-identical; add mirrored
  `tests/common/test_bbox.py`; delete the sync comments.
- `src/physiclaw/agent/engine/stuck.py:57` ↔ `screen_layout.py:158` (+ core's
  step shapes) — a common vocabulary module holding PRESS_TOOLS, NAV_TOOLS,
  sequence step keys (`"actions"`, `"tool_name"`, `"arg"`), on the verdict.py
  precedent. Core consumes only where strings are compared (orchestration
  gesture dispatch/step keys), **not** tools.py registration (FastMCP function
  identifiers). Contract test: registered tool names ⊇ vocabulary. Keep an
  engine-side leaf for the geometry helpers + both tolerance constants
  (MATCH_TOLERANCE and LINT_MARGIN separately named); repoint policy.py's
  derivation. *(critic)* Name the module `common/gesture_vocab.py` (not
  `gestures.py`): tests/ has no `__init__.py` files and pytest runs in prepend
  import mode, so `tests/common/test_gestures.py` would collide with the
  existing `tests/core/orchestration/test_gestures.py` basename and abort
  collection. (Alternative: adopt `--import-mode=importlib` first.)
- `src/physiclaw/cli/setup/hardware.py:25` + 5 agent sites (one already drifted
  to `localhost`) — `server_url()` in `common/config`: env override, else
  `http://{host}:{port}` with 0.0.0.0/:: normalized to 127.0.0.1; update the
  pinned localhost assertion and `--server-url` help; leave the two live-port
  builders alone.
- `src/physiclaw/common/paths.py:153` — `builtin_skills_dir()` imports
  `physiclaw.agent` via `importlib.resources`, a hidden edge the AST leaf test
  can't see: move it into `agent/engine/skill.py` (sole consumer; tests
  unaffected).
- `src/physiclaw/agent/engine/mcp_inventory.py:19` — AST-parses core's
  tools.py by hardcoded path and returns `[]` silently on drift: add the
  contract test asserting `discover_mcp_tools()` names equal the `@mcp.tool`
  set actually registered (tests may import both layers; FakeMcp-recorder
  pattern exists).
- `src/physiclaw/agent/hooks/cron.py:198` — make done/fail/cancel a thin
  wrapper over `engine.jobs.finish_job` (keep CLI argv/output; catch
  ValueError → error+1); replace the concrete cron-CLI close syntax in the
  shared trigger description with an engine-neutral instruction; update the
  pinned trigger-description test and the stale test comment.
- `src/physiclaw/core/calibration/calibrate.py:853` — replace the inline
  move/flush/tap/retry block with `_tap_and_read` (verified: zero test edits
  needed).
- `src/physiclaw/core/static/bridge.html:226` — single-source the
  screenshot_cal square via constants in `core/bridge/calib.py` or
  `transforms.py` (**not** calibrate.py — circular import), serve a `square`
  block through `get_state`, correct or guard the stale `||` fallbacks rather
  than deleting (get_state legitimately omits at/nonce).

**Proof:** new mirrored tests for `common/bbox.py` and
`common/gesture_vocab.py`; the mcp_inventory and gesture-vocabulary contract
tests are the deliverable — they turn silent fail-open drift into red CI. All
byte-pinned message/hint tests pass unchanged by construction.

## Phase 4 — Decompose the four god modules along tested seams

**Highest churn — after the safety net, before annotation sweeps. Effort: ~6
person-days.**

- `src/physiclaw/core/calibration/calibrate.py` (1092 lines, L): split into
  viewport.py, camera_frame.py, arm_cal.py, camera_map.py, validate.py,
  at_verify.py + `_common.py` (grid_positions, `_tap_once`,
  CAL_STRIKE_DURATION), covering all seven steps; update the three import
  sites (handler.py:19, handler.py:286 lazy, warm_start.py:47 lazy) or keep a
  re-export facade; split the 1282-line test file 1:1 along the same seams,
  fixing the 8 `mocker.patch` paths; extract `_run_validation_test(i)` and a
  no-arg `_failed()`.
- `src/physiclaw/agent/claude/spawn.py:257` (M): extract `_redact_images`,
  README constant, `_ClaudeSummary`, `_SessionLog` into
  `agent/claude/session_log.py`; create `tests/agent/claude/test_session_log.py`
  by moving the ~25 relevant tests out of test_spawn.py; repoint the `LOG_DIR`
  monkeypatch; move the `_last_text` capture out of `_summarize` into
  `event()` (last-non-empty-wins preserved); drop the stale `preview`
  docstring reference.
- `src/physiclaw/core/calibration/handler.py:177` (M): one
  `_run_locked_step(physiclaw, do, *, precheck=None)` helper running precheck
  **before** the non-blocking acquire (preserving today's precondition-vs-busy
  error ordering); no needs_arm/needs_cam flags, no reuse of
  `Orchestrator.locked()`; existing precheck-message and release-on-failure
  assertions pass unchanged.
- `src/physiclaw/cli/setup/hardware.py:124` (M): extract one `_step_*`
  function per numbered wizard step, `run()` stays the ordered driver
  mirroring the browser wizard's step wording; camera-fallback block gets
  focused tests (module measured at 81% branch coverage today, below the gate,
  with misses exactly in the inline branches).
- `src/physiclaw/core/orchestration/orchestrator.py:349` (M): type `arm`/`cam`
  honestly (`| None`) or add `require_arm()`/`require_cam()`; convert
  calibration handler.py's 16 private accesses to the public API; switch
  calibrate.py's `cam._fresh_frame()` to the existing public `raw_frame()`;
  **keep** `_fast_move` underscored (deliberately outside the agent-facing
  public section).

**Proof:** the Phase-1 coverage gate is the regression harness — every
extraction holds ≥90% branch; the 1372-line orchestrator suite, calibration
handler assertions, and test_spawn.py pins run unchanged apart from mechanical
patch-path/file moves; setup wizard step wording stays identical to
`setup-hardware.html`.

## Phase 5 — Coverage completion and consistency backfill

**Mechanical passes. Effort: ~5 person-days.**

- `src/physiclaw/cli/flash.py:137` / `clear.py` — add the missing mirror tests
  (mock http_get/stream with in-memory zip; patch `candidate_ports` at
  `physiclaw.core.hardware.grbl` since the import is lazy; dry-run, no-port,
  esptool failure, `--erase` ordering; clear: tmp dirs, `--yes` vs declined,
  empty).
- `src/physiclaw/core/hardware/arm.py:98` etc. — annotate core
  hardware/calibration/server to the agent layer's standard (verified cliff:
  agent ~93% annotated vs core ~57%; 267 ANN001 + 190 missing-return,
  overwhelmingly core; start with public StylusArm/Camera/PhysiClaw/
  calibration-handler methods). Runtime-inert, mechanical.
- `tests/core/orchestration/test_orchestrator.py:41` — two-tier: immediate,
  enforce spec'd doubles (`create_autospec(BridgeState)`,
  `MagicMock(spec=StylusArm)`) so nonexistent-method drift fails loudly (this,
  not injection, is what would have caught the frombuffer rot); structural
  (optional), factory injection on PhysiClaw behind small Protocols with
  behavior-faithful FakeArm/FakeCamera (don't rewire FakeSerial/
  FakeVideoCapture — wrong layer).
- `hardware/assembly/procedures/idler_20_ru.py:86` — collapse the two cloned
  idler pairs (~70 verbatim-duplicated lines vs idler_10_lu.py) onto the
  repo's own subclass idiom (columns()/stack() hooks + class attrs), exactly
  the ID11Lu→ID21Ru precedent; one file per procedure and NN naming untouched.
- `hardware/assembly/cache.py:22` — add `tests/hardware/` scoped to genuinely
  build123d-free modules — cache.py key/miss/prune invariants, mark/svg.py
  geometry, build_manual pagination, ditto_walk (hardware/manual on sys.path
  via conftest; add "." to pytest pythonpath). **Exclude replay.py** from the
  default gate (it imports build123d via assembly.base; CI installs only
  --group dev) — mark it `integration`/cad or first extract
  chain_to/find_leaves into a base-free module. *(critic)* When doing so, fix
  `cache.py:17`'s docstring, which calls the replay "cheap, build123d-free" —
  disproven by this same analysis.

**Proof:** coverage report shows flash/clear join the mirror; `ruff --select
ANN` count in core trends to agent parity; the new hardware tests run in the
normal gate without the cad group; spec'd-mock conversion proven by
intentionally reverting the Phase-1 rotted-test fix and watching it fail
loudly.

## Do not touch

Confirmed healthy or intentional — a naive cleanup would damage these:

- **The agent↔core MCP layering and `tests/test_architecture.py`** — doctrine,
  not duplication. All Phase-3 extractions go to `common` (the sanctioned
  leaf) or stay intra-layer precisely to preserve it.
- **`agent/provider/vendors/` one-file-per-vendor** and their per-unit test
  files — never merge; declarative refactors only.
- **`INLINE_TOOL_INDEX = True` for all providers** including Anthropic.
- **Trailing hint strings in `core/server/tools.py` replies** — byte-pinned
  prompt contract; Phase 3's vocabulary work deliberately avoids the
  registration names and reply texts.
- **`compact.scale_image_bytes` ↔ `vision/util.encode_view_jpeg` duplication**
  — healthy doctrine-forced decoupling driven by one config knob
  (`CONFIG.compact.max_image_edge_px`), cross-documented; do not unify by
  import.
- **The local, domain-shaped retry loops** (calibrate tap re-fire, camera
  re-shoot, provider backoff, spawn re-attempt) — a generic retry abstraction
  would not pay for itself.
- **The module-global `BASE` in `cli/setup/hardware.py`** — documented,
  test-pinned, mutated once per process before any read; fix only the port
  parse (Phase 2).
- **`hardware/assembly/procedures/` `<subassembly>_<NN>_` naming (gaps of 10)**
  and sub-mm slip-fit dimension pull-ins in custom parts.
- **Fail-open discipline** at boundary parsers, trace/retention writes, and
  the always-on runtime loop — pair with contract tests (Phase 3), never
  convert to fail-closed.
- **The Solenoid `_energized()` invariant, exposure.py's injected-callable
  purity, assemble.py's wire-identical prompt dump, `common/verdict.py`** —
  the patterns the plan replicates, not refactors.
- **`_http.py`'s two-policy split including the documented raw-urlopen
  exception**, and cli lazy-import discipline.
- **Camera `_cap_lock` semantics** — never call `cap.release()` without
  holding it (Phase 2's deadlock fix leaks the handle instead, deliberately).

## Optional polish (low-severity backlog, unverified)

- `agent/engine/engine.py` — move turn-shape judgment + corrective copywriting
  next to the other judgment; keep protocol mechanics in engine.py.
- `agent/engine/dto.py` / `compact.py` — delete the ToolResult alias and
  relocate `_ROW_RE` into its test.
- `agent/engine/trajectory.py` — fix the stale `reflect.py` docstring anchor;
  add TYPE_CHECKING Session annotations.
- `agent/provider/anthropic_compat.py` — `assert_never` in `_encode_message`;
  wrap `list_models()` SDK errors into ProviderError (shared `_map_sdk_error`).
- `agent/provider/wire.py` — hoist `auto_call_id()`/`tool_input_schema()`
  shared by both compat bases.
- `agent/claude/spawn.py` — import trace's session-artifact helpers via public
  names with a cross-engine-contract docstring.
- `agent/hooks/poll.py` + `runtime.py` — shared lazy-httpx + warn-once
  BlipLogger helper.
- `agent/runtime/hook.py` — make `clear()` honest about non-re-discoverable
  built-in hooks.
- `core/hardware/camera.py` — release the cv2 handle on `__init__` failure
  paths.
- `core/hardware/iphone.py` — extract `_gesture()` for the
  tap/double_tap/long_press triplication; name the 5s screenshot-save sleep.
- `core/hardware/handler.py` — fix the "0..3" docstrings vs `range(8)`.
- `core/vision/util.py` — split numpad domain logic and the rig diagnostic out
  of the codec grab-bag; fix the `__init__` docstring.
- `core/vision/ocr.py` — stop `logging.disable` mutation in library code;
  silence RapidOCR via its own logger.
- `core/vision/watchdog.py` — name the ZONES at the data level.
- `core/vision/` — extract one `draw_labeled_box()` shared by
  ocr/icon_detect/render annotate paths.
- `core/orchestration/orchestrator.py` — move gesture shape/range validation
  into the frozen dataclasses' `__post_init__` (keep byte-pinned messages);
  log the swallowed park failure in `locked()`.
- `core/calibration/calibrate.py` — reuse `decode_image`, import
  ROTATION_NAMES, add a `_rotated()` helper (fold into Phase 4 split).
- `core/calibration/transforms.py` — drop the hand-written `__init__` on the
  dataclass (broken generated `__eq__`).
- `core/bridge/state.py` — set the clipboard-copied event inside the lock in
  `fetch_text`.
- `core/server/warm_start.py` — use public `wait_for_connection`/promote the
  app.py singletons; emit the banner via `log.info`.
- `core/server/app.py` — PEP 562 `__getattr__` re-exports so leaf helpers
  import without triggering singleton assembly.
- `core/server/watch.py` + `core/calibration/handler.py` — replace deprecated
  `get_event_loop().run_in_executor` with `asyncio.to_thread` (8 sites; folds
  into Phase 4's `_run_locked_step`).
- `core/orchestration` — narrow typed exceptions (HardwareBusy, NotCalibrated)
  subclassing RuntimeError with identical messages; type the cross-module
  bare-dict payloads (ScreenDimension etc.).
- `cli/setup/hardware.py` — delete `lan_ip()` in favor of
  `core.bridge.lan.get_lan_ip`.
- `cli/update.py` — promote the four underscore imports from `_update_check`
  to public; shared `is_ci()`.
- `common/config.py` — make `set_dotted`'s CONFIG refresh mutate in place;
  escape strings in `_toml_scalar` via `json.dumps`; longer-term split into a
  config/ package (schema / toml_io / resolve) with re-exports.
- `common/logger/logger.py` — `_TAG_COLORS.get(tag, default)` so unknown tags
  don't KeyError on a color TTY.
- `hardware/assembly/build_procedures.py` — clear stale snapshot outputs
  before patch replay.
- `hardware/parts/build_custom_parts.py` — collapse the two parallel part/qty
  tables into one registry.
- `hardware/manual/` — make it a package with absolute imports and public
  shared helpers.
- `hardware/assembly/procedures/` — one `render_both(cls)` helper replacing
  the 71 copied `__main__` blocks.
- `orchestrator.py` / common — hoist the `"layout_learned"` marker key into
  common per the verdict.py pattern.
- tests — Hypothesis suites for the pure-math vision/calibration surfaces;
  reconcile TEST.md drift (exclude_also sentence, dead `pytestmark_phase5`,
  bless-or-ban bare does-not-raise tests).

## Sequencing summary

| Phase | Theme | Effort | Risk |
|-------|-------|--------|------|
| 1 | Make the test gate real | ~1 day | near zero |
| 2 | Failure-path correctness | ~3 days | low (small diffs) |
| 3 | Single-source cross-boundary vocabulary | ~3.5 days | low-medium |
| 4 | Decompose the four god modules | ~6 days | medium (churn) |
| 5 | Coverage + consistency backfill | ~5 days | low |

Total ~18.5 person-days. Phases 1+2 alone capture most of the risk reduction.
Each phase lands independently; do not start a phase until the previous one is
green.
