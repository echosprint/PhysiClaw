# Macros

`## Available Macros` lists gesture sequences your user authored and rehearsed on this rig. When one matches the next stretch of work, call `run_macro(name, inputs)` instead of gesturing turn by turn — one call replays the whole sequence and returns the step log plus the current screen. Prefer a matching macro: it is faster and cheaper than manual gestures, and its path is user-tested.

- Fill every required input; values are strings. Omit optional inputs to use their defaults.
- Already did the first steps by hand? Pass `start_at` with a step name from the macro's `steps:` list to begin there. Those earlier steps are NOT executed, so only skip what you have actually already done.
- The result's first line is the outcome. On `ABORTED`, the completed steps already happened on the phone. That macro is then blocked for the rest of the session — a re-run and a `start_at` resume are both refused, so do not spend a turn trying. Read the returned view and finish that stretch with individual gestures.
- Macros never pause for judgment. Confirmations, payments, and choices stay yours — handle them before or after the macro, never assume the macro did.
- None fits the situation: work normally with gestures. Never guess a macro name that is not in the list.
