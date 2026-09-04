"""The ask-and-hold state — everything between "ask sent" and "consent
spent", in one object with one suspension projection.

The walk (`program.py`) owns the cursor; the gate owns the ask text,
the thread snapshot the reply is diffed against, the money numbers a
payment ask quotes and a `yes:` binds, the silence counter, and the
reply words the last send declared. Consent lives here rather than on
the ask step because the payment move AFTER the ask spends it, and a
suspension in between must carry it across wakes.
"""

from dataclasses import dataclass, field


@dataclass
class Gate:
    ask: str = ""
    baseline: set[str] = field(default_factory=set)
    quoted: float | None = None
    consented: float | None = None
    # Every amount on the sheet when the ask quoted it — the fire-time
    # bound's reference for "what the user saw".
    seen: tuple[float, ...] = ()
    silence: int = 0
    awaiting: bool = False  # ask sent, polling for the reply
    # The reply words the last LANDED send declared (already in
    # `reply.normalize` space) — what every read of the thread after it
    # matches: this ask's check, a later send's landing. `next_words`
    # holds the words of a send still in flight.
    yes: tuple[str, ...] = ()
    no: tuple[str, ...] = ()
    next_words: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    tried_open: bool = False

    def abandon_ask(self) -> "Gate":
        """Leave the ask the cursor was holding (a stepping jump): the
        ask text, its reply words, the thread snapshot they were read
        against, and the hold — a later send's landing must not read a
        reply staged for THAT ask as its own. Consent is kept: a jump
        onto the payment move after a confirmed ask is how it is
        stepped. Returns self."""
        self.ask = ""
        self.baseline = set()
        self.awaiting = False
        self.yes = self.no = ()
        return self

    def spend(self) -> float | None:
        """Consent is CONSUMED by firing: a later payment needs its own
        gate's fresh confirm, never this one's leftovers. Returns the
        amount that fired."""
        amount, self.consented, self.quoted = self.consented, None, None
        self.seen = ()
        return amount

    def to_suspended(self) -> dict:
        """The persisted projection — the one field list, beside the
        fields. Counters and the in-flight handshake deliberately reset
        on resume; `consented` persists so a post-consent suspension can
        never resume into a refused payment."""
        return {
            "ask_text": self.ask,
            "baseline": sorted(self.baseline),
            "quoted": self.quoted,
            "consented": self.consented,
            "seen": list(self.seen),
            "awaiting": self.awaiting,
            "yes": list(self.yes),
            "no": list(self.no),
        }

    @classmethod
    def from_suspended(cls, data: dict) -> "Gate":
        return cls(
            ask=str(data.get("ask_text") or ""),
            baseline=set(data.get("baseline") or []),
            quoted=data.get("quoted"),
            consented=data.get("consented"),
            seen=tuple(float(a) for a in (data.get("seen") or [])),
            awaiting=bool(data.get("awaiting")),
            yes=tuple(str(w) for w in (data.get("yes") or [])),
            no=tuple(str(w) for w in (data.get("no") or [])),
        )
