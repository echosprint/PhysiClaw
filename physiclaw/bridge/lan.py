"""Network helpers for the LAN bridge."""

import socket
import subprocess


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
    """Return the Mac's mDNS hostname (e.g. 'physiclaw-mac.local'), or None.

    Survives DHCP-driven IP shifts: iOS resolves *.local via Bonjour on the
    same Wi-Fi. Returns None if the name can't be determined or isn't
    resolvable on this network (e.g. mDNS blocked).
    """
    name: str | None = None
    try:
        result = subprocess.run(
            ["scutil", "--get", "LocalHostName"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        if result.returncode == 0:
            name = result.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if not name:
        try:
            name = socket.gethostname().split(".")[0] or None
        except Exception:
            return None

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


def bridge_base_urls(port: int = 8048) -> tuple[str, str]:
    """Return (primary, fallback) base URLs for the LAN bridge.

    Primary is `http://<host>.local:<port>` when mDNS resolves, else equal
    to fallback. Fallback is `http://<lan-ip>:<port>`. No trailing slash.
    One source of truth for the startup banner, the QR page, and the
    repair page.
    """
    ip = get_lan_ip()
    mdns = get_mdns_host()
    fallback = f"http://{ip}:{port}"
    primary = f"http://{mdns}:{port}" if mdns else fallback
    return primary, fallback
