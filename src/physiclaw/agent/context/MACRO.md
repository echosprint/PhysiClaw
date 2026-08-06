# Macros

`## Available Macros` lists gesture sequences your user rehearsed on this rig. When one covers the next stretch of work, one `run_macro(name, inputs)` call replaces every turn that stretch would cost: macro > `sequence` (a skill's bundled fast path is one) > single gesture.

- **Plan with the list.** A covered stretch is ONE plan step (`Run send-report macro`), not a step per gesture.
- **Steps already done by hand don't disqualify a macro** — pass `start_at` with the step you'd continue from.
- On `ABORTED`, the completed steps already happened — finish the stretch with gestures; that macro is blocked for the session.
- Macros never pause for judgment. Confirmations, payments, and choices stay yours — before or after, never assumed done.
- None fits → gesture normally. Never guess a name not in the list.
