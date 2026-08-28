"""Shared scalar layer for the conductor's YAML spec files.

`pages.yml` and playbook files follow the MACRO.yml conventions — same
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

# YAML 1.2 safe loader, pure-python, load-only — the same construction as
# `macros/parse.py`. Each spec module aliases this as its own `_yaml` so
# tests can keep patching per-module.
yaml_loader = YAML(typ="safe", pure=True)


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
