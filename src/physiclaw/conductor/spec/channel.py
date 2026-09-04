"""The user channel — `playbooks/channel/`, the ONE infrastructure pack.

The IM thread's page fingerprint, the rehearsed send/open macros, and
the boot playbook (`boot.yml` — the walk every wake plays before any
app playbook), recorded and declared on-device like any pack — but
app playbooks never name it: asks reach it through the conductor's
node types, and the boot is the conductor's own to run. The
convention names live in `conventions.py`.
"""

import logging
from dataclasses import dataclass

from physiclaw.conductor.spec.conventions import (
    BOOT_PLAYBOOK,
    CHANNEL_APP,
    OPEN_MACRO,
    SEND_MACRO,
    THREAD_PAGE,
)
from physiclaw.conductor.spec.model import Pack, Playbook, PlaybookError
from physiclaw.conductor.spec.pack import (
    load_pack,
    qualified_macro,
    qualified_pack,
    require_live,
    scan_playbooks,
)
from physiclaw.conductor.spec.pages import PagePrint, prints_for_app
from physiclaw.macros.model import Macro

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Channel:
    """The loaded user-channel infrastructure: thread fingerprints plus
    the qualified macros. `send`/`open` resolve only when the macro
    exists AND is enabled — a missing `send` degrades to hand-over at
    the ask that needs it. `boot` is the boot playbook when it is live
    (on disk, valid, enabled, every hand it names enabled) — else None,
    the reason logged, and the wake is a plain model session; `pack`
    is what the boot builds against."""

    prints: list[PagePrint]
    macros: dict[str, Macro]  # qualified channel/<name>, enabled or not
    pack: Pack
    boot: "Playbook | None" = None

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
    activation degrade; moves run unaffected)."""
    try:
        pack = load_pack(CHANNEL_APP)
        prints = prints_for_app(CHANNEL_APP, decls=pack.pages)
    except Exception as e:
        log.warning("channel pack unusable (%s) — asks will hand over", e)
        return None
    if not any(p.decl.name == THREAD_PAGE for p in prints):
        return None
    return Channel(
        prints=prints,
        macros=qualified_pack(CHANNEL_APP, pack),
        pack=pack,
        boot=_live_boot(pack),
    )


def _live_boot(pack: Pack) -> "Playbook | None":
    """The boot playbook, held to what a wake needs (`require_live`:
    enabled, every referenced macro enabled) — or None with the reason
    logged. Fail-open: no boot means the model drives the wake itself."""
    entry = next(
        (e for e in scan_playbooks(CHANNEL_APP, pack) if e.name == BOOT_PLAYBOOK),
        None,
    )
    if entry is None:
        log.info("conductor: channel has no %s.yml — no boot at wake", BOOT_PLAYBOOK)
        return None
    if entry.spec is None:
        log.warning("conductor: channel boot is invalid (%s) — no boot", entry.error)
        return None
    try:
        require_live(entry.spec, pack)
    except PlaybookError as e:
        log.info("conductor: %s — no boot at wake", e)
        return None
    return entry.spec
