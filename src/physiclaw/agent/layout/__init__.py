"""First-run screen layout: validate the boxes the agent measured, accumulate.

The full capture story lives on `store` (schema, validation,
persistence, rendering, the `record` tool entry point, and the
first-run tail reminder). Beside it, one submodule per consumer-facing
concern: `keyboard` (the cross-call `KeyboardTracker` belief), `lint`
(pre-dispatch gesture lint against the learned boxes), `skills`
(pruning the capture skill and filling box placeholders into built-in
skill templates once learned).
"""

from physiclaw.agent.layout.keyboard import KeyboardTracker
from physiclaw.agent.layout.lint import lint_gesture, lint_sequence
from physiclaw.agent.layout.skills import (
    SKILL_NAME,
    fill_builtin_boxes,
    prune_builtin_skills,
)
from physiclaw.agent.layout.store import (
    ALL_FIELDS,
    PAGES,
    inject_tail,
    is_learned,
    load_layout_md,
    missing_pages,
    record,
    repeatable_key_boxes,
    tail_reminder,
)

__all__ = [
    "ALL_FIELDS",
    "PAGES",
    "SKILL_NAME",
    "KeyboardTracker",
    "fill_builtin_boxes",
    "inject_tail",
    "is_learned",
    "lint_gesture",
    "lint_sequence",
    "load_layout_md",
    "missing_pages",
    "prune_builtin_skills",
    "record",
    "repeatable_key_boxes",
    "tail_reminder",
]
