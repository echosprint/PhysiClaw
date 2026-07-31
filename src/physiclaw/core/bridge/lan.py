"""Network helpers for the LAN bridge."""

import socket
import threading

from physiclaw.common import platform


def get_lan_ip() -> str:
    """Detect this machine's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_mdns_host() -> str | None:
    """Return the host's mDNS name (e.g. 'physiclaw-mac.local'), or None.

    Survives DHCP-driven IP shifts: iOS resolves *.local via Bonjour on the
    same Wi-Fi. Returns None if the name can't be determined or isn't
    resolvable on this network (e.g. mDNS blocked, Bonjour for Windows
    not installed).
    """
    name = platform.local_hostname()
    if not name:
        return None

    host = f"{name.lower()}.local"
    # Bound the lookup with a daemon thread + join: setdefaulttimeout
    # does NOT apply to gethostbyname (no socket object involved), so
    # the old 1s "timeout" was inert — on mDNS-blocked networks the
    # resolver blocked for its own OS timeout, synchronously inside the
    # async QR/setup handlers — and it mutated a process-global default
    # other threads could see.
    ok = threading.Event()

    def _lookup() -> None:
        try:
            socket.gethostbyname(host)
            ok.set()
        except OSError:
            pass

    t = threading.Thread(target=_lookup, daemon=True, name="mdns-probe")
    t.start()
    t.join(timeout=1.0)
    return host if ok.is_set() else None


def bridge_port(control_port: int) -> int:
    """The LAN bridge listens next to the control port. One source of truth
    for the +1 convention (`cli/server.py` bind, URL templating, doctor)."""
    return control_port + 1


def bridge_base_urls(port: int = 8048) -> tuple[str, str]:
    """Return (primary, fallback) base URLs for the LAN bridge.

    ``port`` is the CONTROL port; the returned URLs point at the bridge
    listener (`bridge_port`). Primary is `http://<host>.local:<port>` when
    mDNS resolves, else equal to fallback. Fallback is
    `http://<lan-ip>:<port>`. No trailing slash. One source of truth for
    the startup banner, the QR page, and the setup wizard.
    """
    ip = get_lan_ip()
    mdns = get_mdns_host()
    lan_port = bridge_port(port)
    fallback = f"http://{ip}:{lan_port}"
    primary = f"http://{mdns}:{lan_port}" if mdns else fallback
    return primary, fallback
