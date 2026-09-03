"""Template-pack placeholders — the `<<TOKEN>>` grammar and its local
values file, in one module below both parsers (`macros`, `conductor`).

The sigil is deliberately NOT the macro grammar's `{input}` braces: a
placeholder is a per-INSTALLATION constant, while inputs are per-run
values. Pack files keep their tokens verbatim on disk (shareable,
diffable against the template); values live ONLY in the local
`playbooks/placeholders.yml` and are filled at load. A token with no
value fails loudly instead of, say, searching WeChat for the literal
string "<<CONTACT>>".
"""

import io
import re

from ruamel.yaml import YAML

from physiclaw.common import paths
from physiclaw.common.text import read_text, write_text

PLACEHOLDER_RE = re.compile(r"<<([A-Z][A-Z0-9_]*)>>")

# ~/.physiclaw/playbooks/placeholders.yml — the flat {TOKEN: value}
# map `resolve_placeholders` fills from. Written by `physiclaw
# playbooks install`, hand-editable.
PLACEHOLDER_VALUES_FILENAME = "placeholders.yml"

# YAML 1.2 safe handler for the values file — this module's one
# instance (the `specfile.yaml_loader` pattern, one layer down).
_yaml = YAML(typ="safe", pure=True)


def find_placeholders(text: str) -> list[str]:
    """Unpopulated template tokens in `text`, first-seen order, deduped."""
    return list(dict.fromkeys(PLACEHOLDER_RE.findall(text)))


def fill_placeholders(text: str, values: dict[str, str]) -> str:
    """Substitute `<<TOKEN>>` occurrences from `values`, leaving unknown
    tokens in place (`resolve_placeholders` rejects survivors) — the
    sigil's one writer, beside its one reader above."""
    return PLACEHOLDER_RE.sub(lambda m: values.get(m.group(1), m.group(0)), text)


def placeholder_values() -> dict[str, str]:
    """The local `playbooks/placeholders.yml` as a flat str→str map;
    {} when absent. Raises ValueError on a malformed file — silence
    here would resurface as a baffling "unpopulated placeholder" at
    every pack parse."""
    path = paths.playbooks_dir() / PLACEHOLDER_VALUES_FILENAME
    if not path.exists():
        return {}
    try:
        data = _yaml.load(read_text(path))
    except Exception as e:
        raise ValueError(
            f"{PLACEHOLDER_VALUES_FILENAME}: invalid YAML: {e or type(e).__name__}"
        ) from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{PLACEHOLDER_VALUES_FILENAME} must be a flat TOKEN: value mapping"
        )
    return {str(k): str(v) for k, v in data.items()}


def write_placeholder_values(values: dict[str, str]) -> None:
    """Write the full `playbooks/placeholders.yml` map — the reader
    above's one counterpart, so the file shape never drifts."""
    buf = io.StringIO()
    _yaml.dump(dict(sorted(values.items())), buf)
    path = paths.playbooks_dir() / PLACEHOLDER_VALUES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text(
        path,
        "# Per-installation constants — filled into pack files' <<TOKEN>>s\n"
        "# at load. Written by `physiclaw playbooks install`; hand-editable.\n"
        + buf.getvalue(),
    )


def resolve_placeholders(
    text: str, error_cls: type[Exception], values: dict[str, str] | None = None
) -> str:
    """`text` with its `<<TOKEN>>`s filled from the local values file —
    the one wrapper every pack-file parser (macros, pages, playbooks)
    calls. A surviving token (or a broken values file) raises
    `error_cls`, so the message cannot drift between grammars."""
    if values is None:
        # A caller loading many files reads the values file once and
        # threads them; a single load reads it here.
        try:
            values = placeholder_values()
        except ValueError as e:
            raise error_cls(str(e)) from e
    if values:
        text = fill_placeholders(text, values)
    tokens = find_placeholders(text)
    if tokens:
        raise error_cls(
            f"unpopulated template placeholder(s) {', '.join(tokens)} — "
            f"add values to playbooks/{PLACEHOLDER_VALUES_FILENAME} "
            "(`physiclaw playbooks install` prompts for them)"
        )
    return text
