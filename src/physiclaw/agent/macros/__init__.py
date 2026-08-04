"""User-authored gesture macros — rehearsed MCP-gesture sequences.

A macro is a named, linear, parameterized batch of gesture calls the user
authors by hand as ``~/.physiclaw/macros/<name>/MACRO.yml`` and rehearses
via ``physiclaw macros run`` (a valid macro is enabled by default; the
scaffold starts ``enabled: false`` until rehearsed). The agent then
executes it as ONE tool call (``run_macro``) instead of one LLM round trip
per gesture — an Excel macro, not a workflow: no general conditionals, no loops, no
macro-calling-macro (a `wait` step may settle, and `skip_when` may skip a
step whose postcondition already holds — neither is a branch in the flow). Anything needing judgment stays with the agent; a macro
only replays a rehearsed, fixed path and aborts the moment the screen stops
matching it (per-step ``guard`` blocks).

Module split — depends on ``common`` only (no ``engine`` imports; the engine
imports us). Listed leaf-first, the DSL pipeline top to bottom:

    template.py the `{name}` placeholder engine — the true leaf, pure strings
    model.py    the shapes: constants, MacroError, clause algebra, Screen
    steps.py    the executable step hierarchy (GestureStep / WaitStep)
    parse.py    MACRO.yml → a validated Macro (YAML 1.2 + strict types)
    inputs.py   input resolution + `{name}` substitution at replay time
    scaffold.py the authoring texts: `init` template + the macros README
    store.py    discovery under ~/.physiclaw/macros/, prompt-section render
    runner.py   replay a spec over an MCP caller, compose the one tool result
    runlog.py   per-step forensic log, one `macro-run-<hex6>` id per run
    stats.py    per-macro run counters in macros/stats.json

Nothing is re-exported here on purpose: every consumer imports the
submodule it needs, and a package-level re-export drags `parse`
(ruamel) and `runner` (asyncio) in behind any one of them.
"""
