"""The user channel — `playbooks/channel/`, the ONE infrastructure pack.

The IM thread's page fingerprint plus rehearsed send/open macros,
recorded on-device like any pack — but playbooks never name it: asks
(CONFIRM / HUMAN_GATE) and activation reach it through the conductor's
node types. The three convention names live in pages.py (the scaffold
interpolates them).
"""

import logging
from dataclasses import dataclass

from physiclaw.conductor.pages import (
    CHANNEL_APP,
    OPEN_MACRO,
    SEND_MACRO,
    THREAD_PAGE,
    PagePrint,
    prints_for_app,
)
from physiclaw.conductor.playbook import (
    load_pack,
    qualified_macro,
    qualified_pack,
)
from physiclaw.macros.model import Macro

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Channel:
    """The loaded user-channel infrastructure: thread fingerprints plus
    the qualified macros. `send`/`open` resolve only when the macro
    exists AND is enabled — unavailable members degrade to hand-over at
    the moment they are needed, never earlier."""

    prints: list[PagePrint]
    macros: dict[str, Macro]  # qualified channel/<name>, enabled or not

    def _live(self, name: str) -> str | None:
        key = qualified_macro(CHANNEL_APP, name)
        m = self.macros.get(key)
        return key if m is not None and m.enabled else None

    @property
    def send(self) -> str | None:
        return self._live(SEND_MACRO)

    @property
    def open(self) -> str | None:
        return self._live(OPEN_MACRO)


def load_channel() -> Channel | None:
    """The channel pack, fail-open: absent or broken → None (asks and
    activation degrade; legs run unaffected)."""
    try:
        pack = load_pack(CHANNEL_APP)
        prints = prints_for_app(CHANNEL_APP, decls=pack.pages)
    except Exception as e:
        log.warning("channel pack unusable (%s) — asks will hand over", e)
        return None
    if not any(p.decl.name == THREAD_PAGE for p in prints):
        return None
    return Channel(prints=prints, macros=qualified_pack(CHANNEL_APP, pack))
