"""Bilingual document builders — the assembly manual and the sourcing guide.

Standard-library only (no build123d): the builders consume the SVG renders
the assembly pipeline already produced. ``build_manual`` and
``build_sourcing_guide`` are the two entry points; ``assets`` / ``common``
/ ``paginate`` / ``pdf`` are their support modules.
"""


class BuildError(Exception):
    """A user-facing build failure: ``main()`` prints the message on its own,
    without a Python traceback."""
