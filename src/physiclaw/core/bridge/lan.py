"""Network helpers for the LAN bridge."""

import socket

from physiclaw.core import platform


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
    prev_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(1)
        socket.gethostbyname(host)
    except (socket.gaierror, socket.timeout, OSError):
        return None
    finally:
        socket.setdefaulttimeout(prev_timeout)

    return host


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
