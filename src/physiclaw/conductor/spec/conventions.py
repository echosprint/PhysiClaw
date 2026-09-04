"""The conductor's fixed names — the packs and pages it knows by name.

Two reserved app namespaces (`channel`, `ios`), the three channel
conventions (the thread page, the send and open macros), the boot
playbook, the OS lock page, and the `app.page` id spelling every
matcher verdict carries, and the one spelling of a currency amount.
A leaf: nothing here imports the conductor.
"""

import re

# Reserved app namespaces a pack's page refs may cross into. Neither is
# a task pack, and both are scaffolded into playbooks/<app>/ like any
# other: `channel` is the user-channel IM pages + send macros, `ios` is
# OS-level state. Playbooks never name either — the conductor reaches
# them through node types, which is what "reserved" buys.
CHANNEL_APP = "channel"

# The OS-state pack. Same shape as any other pack — a scaffolded
# `playbooks/ios/` the user owns and edits — but the conductor knows its
# name, because telling "locked" from "a screen I don't recognize" is
# its own job, not a playbook's. `scaffold.IOS_PAGES_STUB` is the
# starting text; absent (never scaffolded) simply means the conductor
# cannot name those states, and every one of them reads as unknown.
IOS_APP = "ios"

# Derived, never re-spelled: renaming either constant must not leave this
# set pointing at the old string.
RESERVED_APPS = frozenset({CHANNEL_APP, IOS_APP})

# The channel pack's conventions — the three names the conductor knows.
# Declared HERE (beside CHANNEL_APP) because both channel.py and
# scaffold.py need them and scaffold must not import channel: the stubs
# interpolate these, never hand-copy them.
THREAD_PAGE = "thread"  # the channel pack's `pages:` must declare this
SEND_MACRO = "send"  # nav to the user's thread + paste + send {message}
# `open` is the channel's hand: nav to the user's thread — the boot's
# route recovers through it and a resumed ask reads through it.
OPEN_MACRO = "open"
# The channel pack's one playbook: the boot — the walk every wake plays
# before any app playbook (reach the thread, read the request there).
# `route.py` admits its `activate` step in this one file only.
BOOT_PLAYBOOK = "boot"

# The ios pack's one convention, here for the same reason: the boot
# matches this page and `scaffold.IOS_PAGES_STUB` declares it, so the two
# interpolate one constant instead of both spelling "locked".
LOCKED_PAGE = "locked"


def page_id(app: str, page: str) -> str:
    """The `app.page` spelling — the ONE format matcher verdicts carry
    and every expectation is compared against."""
    return f"{app}.{page}"


def page_name(pid: str) -> str:
    """The page half of a full id — `page_id`'s inverse."""
    return pid.partition(".")[2]


def owned_by(pid: str, app: str) -> bool:
    """Whether a full id names one of `app`'s own pages (the reserved
    namespaces and the channel are somebody else's)."""
    return pid.startswith(f"{app}.")


# The full ids, precomputed: what the gate's peeks and the boot's route
# check against.
THREAD_ID = page_id(CHANNEL_APP, THREAD_PAGE)
LOCKED_ID = page_id(IOS_APP, LOCKED_PAGE)

# The one spelling of "a ¥/￥ amount": `match.normalize` class-tokenizes
# with it, `money.amounts` reads its group — thousands separators
# included ("¥1,234.56"), which `money.amount` strips. At most two
# decimals: a price has no third, and OCR glues the next word's leading
# digit onto one ("实付￥24.751优惠前" once quoted ¥24.751 to the user).
PRICE_RE = re.compile(r"[¥￥]\s*((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)")
