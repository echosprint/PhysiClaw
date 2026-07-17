"""Cross-repo guards for the CLI ↔ PhysiClaw-site download contracts.

The CLI downloads three artifacts from physiclaw.ai, and the site's build
(``PhysiClaw-site/scripts/*.mjs``) decides what is actually served there:

- firmware bundle: ``physiclaw.cli.flash.FIRMWARE_URL`` ↔ the
  ``fetch-downloads.mjs`` entry mirroring the ``fluidnc.zip`` release
  asset built by ``scripts/firmware/build_bundle.py``,
- vision model: ``physiclaw.cli.setup.vision`` base64 parts ↔ the
  ``fetch-downloads.mjs`` entry that splits it (``parts``),
- official skills: ``SkillsConfig.official_base_url`` ↔ the namespace
  ``build-official-skills.mjs`` writes under ``public/downloads/``.

These only run when a PhysiClaw-site checkout sits beside this repo —
skipped otherwise (e.g. CI without the sibling), so local development,
where both repos are edited together, is where drift gets caught.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from urllib.parse import urlparse

import pytest

from physiclaw.cli.flash import FIRMWARE_URL
from physiclaw.cli.setup.vision import _PREBUILT_PARTS, _PREBUILT_PARTS_URL
from physiclaw.common.config import SkillsConfig

REPO = Path(__file__).resolve().parents[1]
SITE = REPO.parent / "PhysiClaw-site"

pytestmark = pytest.mark.skipif(
    not SITE.is_dir(), reason="PhysiClaw-site checkout not found beside this repo"
)


def _load_build_bundle():
    """Import the standalone script (stdlib-only at module level)."""
    path = REPO / "scripts" / "firmware" / "build_bundle.py"
    spec = importlib.util.spec_from_file_location("build_bundle", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _site_downloads() -> dict[str, tuple[str, int | None]]:
    """fetch-downloads.mjs entries as {dest: (release asset path, parts)}."""
    text = (SITE / "scripts" / "fetch-downloads.mjs").read_text()
    entries = re.findall(
        r"src: `\$\{RELEASES\}/([^`]+)`,\s*"
        r"dest: '([^']+)',\s*"
        r"(?:parts: (\d+),\s*)?",
        text,
    )
    assert entries, "no DOWNLOADS entries parsed from fetch-downloads.mjs"
    return {dest: (src, int(parts) if parts else None) for src, dest, parts in entries}


def _url_path(url: str) -> str:
    return urlparse(url).path.lstrip("/")


def test_firmware_url_served_by_site():
    src, parts = _site_downloads()[_url_path(FIRMWARE_URL)]
    assert parts is None, "firmware bundle must be served whole, not split"
    # The site mirrors the same version-free asset name the build script
    # uploads to the firmware_fluidNC release.
    assert src == f"firmware_fluidNC/{_load_build_bundle().BUNDLE_NAME}"


def test_vision_model_parts_match_site_split():
    base, suffix = _PREBUILT_PARTS_URL.rsplit(".b64.", 1)
    assert suffix == "{i:02d}", "part naming must match the site's .b64.NN scheme"
    _, parts = _site_downloads()[_url_path(base)]
    assert parts == _PREBUILT_PARTS


def test_official_skills_namespace_built_by_site():
    namespace = _url_path(SkillsConfig.official_base_url)
    builder = (SITE / "scripts" / "build-official-skills.mjs").read_text()
    assert namespace in builder
