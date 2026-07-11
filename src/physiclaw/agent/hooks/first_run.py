"""First-run hook — wake the agent to learn the screen layout.

After hardware calibration the runtime otherwise sits idle until the phone
screen changes. But first-run screen-layout learning (the input-box / keyboard
/ Paste bboxes) needs the agent to act, and a freshly-calibrated phone may sit
on a static screen that never trips the camera watchdog — so the layout would
never get learned. On the first ready tick with no learned layout, fire one
wake to kick off learning.

Fires at most once per process: a single wake runs the whole `screen-layout`
capture, then the session ends — this wake carries no request to resume, so the
engine does NOT restart it; the captured layout loads on the next wake. If the
wake fails, the phone watchdog plus the SYSTEM first-run notice retry it on the
next screen change. Auto-discovered by `physiclaw.agent.runtime.hook.load_hooks()`.
"""

import logging

from physiclaw.agent.engine import screen_layout
from physiclaw.agent.runtime.hook import Trigger, register

log = logging.getLogger(__name__)

_fired = False


@register
def first_run_layout() -> Trigger | None:
    global _fired
    if _fired or screen_layout.is_learned():
        return None
    _fired = True
    log.info("screen layout not learned — waking agent for first-run setup")
    return Trigger(
        description=(
            "First run: learn the screen layout (input boxes, keyboard, Paste) "
            "before handling any request — follow the `screen-layout` skill "
            "(in `## Built-in Skills`; already in your system prompt, no "
            "Skill() load needed)."
        ),
        source="first-run",
    )
