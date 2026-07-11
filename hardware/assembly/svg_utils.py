"""Root-tag attribute rewriting and style injection for build123d-emitted SVGs.

Operations shared between the build pipeline (post-render stripping of
``width`` / ``height`` + a non-scaling-stroke style block) and the
marker tool (setting ``viewBox`` to a user-cropped rect). The
attribute rewrites touch only the root ``<svg ...>`` opening tag —
nested elements with their own ``width`` / ``height`` (e.g.
``<rect>``) are untouched."""

from __future__ import annotations

import re

# Anchored on ``<svg\b`` so element names like ``<svgImage>`` aren't
# matched, and used with ``count=1`` so only the root opening tag is
# rewritten even if a file contains nested ``<svg>`` elements.
ROOT_TAG_RE = re.compile(r"<svg\b[^>]*>")

_DIM_ATTRS_RE = re.compile(
    r"""\s+(?:width|height)\s*=\s*(?:"[^"]*"|'[^']*')""",
)
_VIEWBOX_ATTR_RE = re.compile(
    r"""viewBox\s*=\s*(?:"[^"]*"|'[^']*')""",
)
_VIEWBOX_VAL_CAPTURE_RE = re.compile(
    r"""\bviewBox\s*=\s*(?:"([^"]*)"|'([^']*)')""",
)

# Validation for user-supplied viewBox values: exactly four numbers
# separated by whitespace, decimals + scientific notation allowed.
VIEWBOX_VAL_RE = re.compile(
    r"^\s*-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"(?:\s+-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?){3}\s*$"
)


def strip_root_dims(text: str) -> str:
    """Drop ``width=`` / ``height=`` from the root ``<svg>`` opening tag.

    The ``viewBox`` and every other attribute are preserved so the SVG
    scales responsively in browsers instead of rendering at the fixed
    physical size baked into ``width`` / ``height``."""
    return ROOT_TAG_RE.sub(
        lambda m: _DIM_ATTRS_RE.sub("", m.group(0)),
        text,
        count=1,
    )


_NON_SCALING_STROKE_STYLE = (
    "<style>svg * { vector-effect: non-scaling-stroke; }</style>"
)


def inject_non_scaling_strokes(text: str) -> str:
    """Insert a ``<style>`` rule right after the root ``<svg>`` tag so
    stroke widths render at a fixed pixel size regardless of how the
    SVG is scaled in the consumer (browser, PDF, …). Without this, an
    SVG shown at 1/4 size has 1/4-thick strokes and lines disappear."""
    return ROOT_TAG_RE.sub(
        lambda m: m.group(0) + _NON_SCALING_STROKE_STYLE,
        text,
        count=1,
    )


def get_root_viewbox(text: str) -> str | None:
    """Return the viewBox value (``"x y w h"``) from the root ``<svg>``
    opening tag, or ``None`` if the tag has no viewBox attribute."""
    root = ROOT_TAG_RE.search(text)
    if root is None:
        return None
    m = _VIEWBOX_VAL_CAPTURE_RE.search(root.group(0))
    if m is None:
        return None
    return m.group(1) if m.group(1) is not None else m.group(2)


def set_root_viewbox(text: str, viewbox: str) -> str:
    """Replace (or insert) the ``viewBox`` attribute on the root
    ``<svg>`` opening tag. Other attributes are preserved verbatim."""

    def rewrite(m: re.Match) -> str:
        tag = m.group(0)
        if _VIEWBOX_ATTR_RE.search(tag):
            return _VIEWBOX_ATTR_RE.sub(f'viewBox="{viewbox}"', tag, count=1)
        # Insert just before the closing ">" (or "/>") of the opening tag.
        end = -2 if tag.endswith("/>") else -1
        return f'{tag[:end]} viewBox="{viewbox}"{tag[end:]}'

    return ROOT_TAG_RE.sub(rewrite, text, count=1)


def validate_viewbox(raw) -> str | None:
    """Return the trimmed viewBox string, or ``None`` if ``raw`` is None.

    Raises ``ValueError`` if ``raw`` isn't exactly four whitespace-separated
    numbers (``x y width height``)."""
    if raw is None:
        return None
    if not isinstance(raw, str) or not VIEWBOX_VAL_RE.match(raw):
        raise ValueError("viewBox must be 'x y width height' (four numbers)")
    return raw.strip()
