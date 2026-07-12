"""Server networking helpers.

``wait_for_port`` lives here — not in ``warm_start`` — so both the warm-start
thread and the ``physiclaw auto`` setup worker can share it. The module body
is light (stdlib + config only), but note it sits under the ``core.server``
package: importing it runs ``core.server.__init__`` (which builds the app
singletons), so callers outside the server process (e.g. the setup CLI)
import it lazily.
"""

import socket
import time

from physiclaw.common.config import CONFIG

# How long to wait for uvicorn's listening socket to accept connections, in
# seconds. IPv4 only — if --host is IPv6 this times out; today all callers
# use v4.
PORT_WAIT_TIMEOUT = CONFIG.warm_start.port_wait_timeout_seconds
PORT_WAIT_CONNECT_TIMEOUT = CONFIG.warm_start.port_wait_connect_timeout_seconds
PORT_WAIT_INTERVAL = CONFIG.warm_start.port_wait_interval_seconds


def wait_for_port(host: str, port: int, timeout: float = PORT_WAIT_TIMEOUT) -> bool:
    """Block until something is accepting TCP connections on (host, port).

    Returns True on first successful connect, False if ``timeout`` elapses.
    Used to synchronize with uvicorn startup — signalling SIGINT mid-startup
    leaks CancelledError tracebacks through the lifespan machinery.
    """
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(PORT_WAIT_CONNECT_TIMEOUT)
            try:
                s.connect((probe_host, port))
                return True
            except OSError:
                pass
        time.sleep(PORT_WAIT_INTERVAL)
    return False
