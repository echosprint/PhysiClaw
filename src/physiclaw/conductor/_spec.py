"""Shared scalar layer for the conductor's YAML spec files.

A pack's spec file (PLAYBOOK.yml) follows the MACRO.yml conventions — same
naming rule, same prose caps, same strict-string coercion hints, same
YAML 1.2 pure loader. The rules' one true home is the macro layer
(`macros/model.py` constants, `check_name`'s wording); this module
re-exports them for the conductor and binds the scalar validators to
each spec's own error class, so `pages.py` and `playbook.py` keep their
distinct catch surfaces without re-spelling the validation:

    _require_str, _prose, _opt_prose, _check_name = bind(PagesError)
"""

from typing import Any, Callable

from ruamel.yaml import YAML

from physiclaw.common import paths
from physiclaw.common.paths import PACK_FILENAME
from physiclaw.common.placeholders import resolve_placeholders
from physiclaw.common.text import read_text
from physiclaw.macros.model import (
    INPUT_NAME_RE as INPUT_NAME_RE,
)
from physiclaw.macros.model import (
    MAX_NAME_LEN as MAX_NAME_LEN,
)
from physiclaw.macros.model import (
    MAX_PROSE_LEN as MAX_PROSE_LEN,
)
from physiclaw.macros.model import (
    NAME_RE as NAME_RE,
)
from physiclaw.macros.model import MacroError
from physiclaw.macros.model import check_name as _model_check_name
from physiclaw.macros.parse import reject_aliases

# YAML 1.2 safe loader, pure-python, load-only — the same construction as
# `macros/parse.py`. `load_yaml` below is the resolved-spec door; the
# install flow's raw template read (`scaffold.read_template_manifest`)
# shares the instance.
yaml_loader = YAML(typ="safe", pure=True)


_PACK_TOP_KEYS = frozenset({"app", "description", "placeholders", "pages", "playbooks"})


def load_yaml(text: str, error_cls: type[Exception], where: str = "") -> Any:
    """The one load ritual every spec door shares: fill `<<TOKEN>>`s
    from the local values file (rejecting survivors), YAML 1.2 load,
    reject anchors/aliases document-wide, wrap errors in the door's own
    error class. The alias guard is the macro parser's: inline macros
    put clause parsing — which materializes a Clause per path, the
    alias-bomb ride — behind these doors, and running it HERE keeps the
    grammar door-independent (a text that parses at the test/tooling
    door must not fail only at the live pack file)."""
    text = resolve_placeholders(text, error_cls)
    prefix = f"{where}: " if where else ""
    try:
        data = yaml_loader.load(text)
    except Exception as e:  # loader errors are not confined to YAMLError
        raise error_cls(f"{prefix}invalid YAML: {e or type(e).__name__}") from e
    try:
        reject_aliases(data)
    except MacroError as e:
        raise error_cls(f"{prefix}{e}") from e
    return data


def load_pack_doc(app: str, error_cls: type[Exception]) -> dict | None:
    """`playbooks/<app>/PLAYBOOK.yml`, loaded and top-checked: a mapping
    with only the known top-level sections, no unpopulated template
    placeholders. None when the file doesn't exist (not a pack).
    Section contents are validated by their owners (`pages.py`,
    `playbook.py`) — this is just the shared front door."""
    path = paths.pack_root(app) / PACK_FILENAME
    if not path.exists():
        return None
    data = load_yaml(read_text(path), error_cls, where=f"{app}/{PACK_FILENAME}")
    if not isinstance(data, dict):
        raise error_cls(f"{app}/{PACK_FILENAME} must be a YAML mapping")
    unknown = sorted(set(map(str, data.keys())) - _PACK_TOP_KEYS)
    if unknown:
        raise error_cls(
            f"{app}/{PACK_FILENAME}: unknown key(s): {', '.join(unknown)} "
            f"(sections: {', '.join(sorted(_PACK_TOP_KEYS))})"
        )
    return data


def bind(
    err: type[ValueError],
) -> tuple[
    Callable[[Any, str], str],
    Callable[[Any, str], str],
    Callable[[Any, str], str | None],
    Callable[[Any, str], None],
]:
    """The four scalar terminals, raising `err`. Wording matches the macro
    parser's — one user-facing voice for one rule set."""

    def require_str(value: Any, where: str) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is None or isinstance(value, str):
            raise err(f"{where} is required and must be a non-empty string")
        raise err(
            f"{where} must be a string but YAML parsed {value!r} "
            f'({type(value).__name__}) — quote it: "{value}"'
        )

    def prose(value: Any, where: str) -> str:
        text = require_str(value, where)
        if "\n" in text or "\r" in text:
            raise err(f"{where} must be a single line")
        if len(text) > MAX_PROSE_LEN:
            raise err(f"{where} is {len(text)} characters (max {MAX_PROSE_LEN})")
        return text

    def opt_prose(value: Any, where: str) -> str | None:
        return None if value is None else prose(value, where)

    def name_check(name: Any, where: str) -> None:
        if not isinstance(name, str):
            raise err(f"{where} must be a string (got {name!r})")
        try:
            # The macro rule verbatim (wording included), translated to
            # this spec's error class at the one seam.
            _model_check_name(name, where)
        except MacroError as e:
            raise err(str(e)) from e

    return require_str, prose, opt_prose, name_check
