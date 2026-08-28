"""Turn-plugin loading — the engine's blind end of the plugin seam.

The engine knows `contract.plugin.TurnPlugin` and a config string; it
never imports a plugin's package. `[agent] plugins` holds
comma-separated ``module:factory`` paths; each factory returns one
plugin. Config registration because we own every plugin (see
`contract.plugin` for the entry-point upgrade trigger). Fail-open at
every step: a path that won't import, a factory that raises, a
wrong-shaped object — logged and skipped, and the session runs with
whatever loaded (possibly nothing, which is a plain LLM session).

The built-in default lives HERE, not in config: the config layer stays
neutral (it names no plugin) and the rendered template does not
advertise internals. An empty config value means this default set;
``"none"`` is the explicit off switch — every turn a plain provider
call. The name below is the one place agent-side code spells a plugin's
module, and it is data to `importlib`, never an import — the AST guards
in `tests/test_architecture.py` stay honest.
"""

import importlib
import logging

from physiclaw.common.config import CONFIG
from physiclaw.contract.plugin import TurnPlugin

log = logging.getLogger(__name__)

# What an empty `[agent] plugins` resolves to.
DEFAULT_PLUGINS = "physiclaw.conductor.plugin:build"

# The explicit off switch — distinct from "absent", which means default.
DISABLED = "none"


def load_plugins() -> tuple[TurnPlugin, ...]:
    raw = CONFIG.agent.plugins.strip()
    if raw.lower() == DISABLED:
        # Logged so a forensic read of a plugin-less session shows a
        # choice, not a silent failure.
        log.info("turn plugins: disabled by config")
        return ()
    out: list[TurnPlugin] = []
    loaded: list[str] = []
    for path in _paths(raw or DEFAULT_PLUGINS):
        try:
            mod_name, _, attr = path.partition(":")
            factory = getattr(importlib.import_module(mod_name), attr)
            plugin = factory()
        except Exception:
            log.warning("turn plugin %r failed to load — skipped", path, exc_info=True)
            continue
        if not isinstance(plugin, TurnPlugin):
            log.warning("turn plugin %r does not satisfy TurnPlugin — skipped", path)
            continue
        out.append(plugin)
        loaded.append(path)
    # The resolved set, once per session — the session log's answer to
    # "was the conductor even in play?" (a rotted default otherwise
    # reads exactly like a healthy plain-LLM session).
    log.info("turn plugins: %s", ", ".join(loaded) or "(none loaded)")
    return tuple(out)


def _paths(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]
