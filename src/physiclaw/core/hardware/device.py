"""Device contract — the shared error taxonomy and lifecycle protocol.

Every driver raises out of one typed hierarchy so callers can catch
selectively (an unplugged device vs a wedged one vs a firmware alarm)
instead of blanket ``except Exception``. The hierarchy subclasses
``RuntimeError`` deliberately: the pre-existing ``except RuntimeError``
handlers up the stack (HTTP handlers, the watch route's not-calibrated
fallback) keep catching driver failures exactly as before.
"""

from typing import Protocol, runtime_checkable


class HardwareError(RuntimeError):
    """Base for every driver-raised failure."""


class DeviceNotFound(HardwareError):
    """No matching device on the bus (detection/open failed)."""


class DeviceTimeout(HardwareError):
    """The device stopped answering (serial silence, frame drought)."""


class ProtocolError(HardwareError):
    """The device answered with an error (GRBL error:N, bad reply)."""


class DeviceAlarm(HardwareError):
    """The device locked itself out (GRBL ALARM state)."""


@runtime_checkable
class Device(Protocol):
    """Minimal lifecycle every rig-owned device implements."""

    def close(self) -> None:
        """Release the underlying handle; safe to call more than once."""
        ...

    def health(self) -> bool:
        """True when the device is answering right now — a live check,
        not a cached flag. Lets callers detect a cable pulled mid-session
        instead of inferring liveness from 'was connected once'."""
        ...
