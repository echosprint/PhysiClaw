"""The conductor's prompt texts — what its three micro-calls say.

Text only: the mechanism (`micro.py`) assembles the skeleton — role
sentence → reply contract → answer legend — and the rows there name
these. Kept apart so a prompt can be read and edited as prose, and the
agent rows stay what they are: the author's prompt is the whole brief
and the conductor adds only the output contract.
"""

# parse_task — the boot's one question over the user's thread.
PARSE_TASK_ROLE = (
    "You read an instant-message thread, oldest line first and "
    "newest last, and decide whether the user has a request still "
    "OUTSTANDING that one of the available playbooks performs."
)
PARSE_TASK_LEGEND = (
    '"answer" is one playbook EXACTLY as listed; or "not_a_task" '
    "for greetings, chat, questions, or anything no playbook "
    'covers (when unsure, "not_a_task"); or "scroll_up" when the '
    "newest message refers to an earlier request that is NOT "
    "visible in this thread — older messages sit above the fold "
    "and must be read before deciding. "
    # A wake is usually the user's SECOND prod, not their first:
    # the run they asked for was cut short, so the newest line is
    # a nudge and the request it refers to sits above it. Reading
    # only the newest line answered `not_a_task` to a user who
    # was asking for exactly the playbook on the menu.
    "WHICH request: normally the user's newest one — but when "
    "their newest message is only a nudge to carry on (a bare "
    '"go on" / "继续" / "any update?"), it refers to their most '
    "recent request above it that is still outstanding. "
    # The completion reply is what makes this safe to widen: the
    # assistant reports every finished task into this same
    # thread, so "already done" is visible rather than inferred.
    "A request is FINISHED — never answer it with a playbook — "
    "once the assistant has reported it done or the user "
    "cancelled it; only a request with no such reply after it is "
    "still outstanding. Judge that from THIS thread: the report or "
    "cancellation must sit after the request here. The Context "
    "block is background only — it records earlier, separate "
    "tasks (even for the same product on another day) and never "
    "finishes a request this thread shows unanswered. Re-running "
    "a finished task can spend money twice, so on any doubt about "
    'which it is, answer "not_a_task". '
    "When you answer with a "
    'playbook, ALSO add a fourth field "inputs": an object filling '
    "that playbook's declared inputs from the words of THAT "
    "request. "
    "OMIT any input the message does not specify — leave the key "
    "out entirely rather than filling it with null or an empty "
    "string; each omitted input falls back to its own default. "
    "Each value contains ONLY what that input's description asks "
    "for — a search-keyword input takes the bare product/search "
    "term (follow its e.g. example when shown), never quantity or "
    "count words; those belong only in an input that asks for "
    "them."
)

# agent_fields — the pure-text call: the author's prompt in, fields out.
AGENT_FIELDS_LEGEND = (
    '"answer" is "done" (then ALSO add one field per return field '
    'listed, each a plain string) or "escalate" when the brief '
    "cannot be fulfilled from what it gives you."
)

# agent_act — the episode legend's fixed options (`micro._act_legend`
# adds the granted tools' lines from `calls.py`).
GRANTED_LANDMARKS_OPTION = (
    "a granted landmark name, but ONLY one the NEWEST screen block "
    "lists under 'Granted landmarks' ({give})"
)
GRANTED_MACROS_OPTION = (
    "a granted macro name, copied exactly — it runs that recorded "
    "gesture sequence ({macros})"
)
DONE_OPTION = (
    '"done" ONLY when the goal is fully met (then ALSO add one field per '
    "return field listed, each a plain string)"
)
ESCALATE_OPTION = (
    '"escalate" when you are stuck, the screen is unexpected, or the goal '
    "needs an action you were not given"
)
RETURN_FIELDS_HEADER = 'Return fields (each a plain string beside "answer": "done"):'
