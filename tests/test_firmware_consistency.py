"""Consistency guards for facts the firmware bundle duplicates.

Two sets of facts about the flash bundle live in more than one place:

- the FluidNC machine config: ``scripts/firmware/config.yaml`` (baked
  into the bundle's littlefs image by ``build_bundle.py``) and the
  fenced ``yaml`` block in the EN/ZH firmware docs (pasted by users
  following the manual browser flash flow),
- the flash layout: ``physiclaw.cli.flash.FLASH_LAYOUT`` (what the CLI
  writes to the board) and the constants in
  ``scripts/firmware/build_bundle.py``, a standalone PEP 723 script
  that cannot import the CLI's copies.

The docs copies deliberately add tutorial comments, and FluidNC's YAML
subset allows comments only on their own lines, so the config guard
compares functional lines (comments and blanks stripped) exactly.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from physiclaw.cli.flash import FLASH_LAYOUT

REPO = Path(__file__).resolve().parents[1]
BUNDLE_CONFIG = REPO / "scripts" / "firmware" / "config.yaml"
DOCS_EN = REPO / "docs" / "set-up" / "firmware.mdx"
DOCS_ZH = REPO / "docs" / "set-up" / "firmware.zh.mdx"


def _load_build_bundle():
    """Import the standalone script (stdlib-only at module level)."""
    path = REPO / "scripts" / "firmware" / "build_bundle.py"
    spec = importlib.util.spec_from_file_location("build_bundle", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_bundle = _load_build_bundle()


def _yaml_block(mdx: Path) -> str:
    blocks = re.findall(r"^```yaml\n(.*?)^```", mdx.read_text(), re.S | re.M)
    assert len(blocks) == 1, f"expected exactly one yaml block in {mdx.name}"
    return blocks[0]


def _functional_lines(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_docs_yaml_blocks_en_zh_identical():
    assert _yaml_block(DOCS_EN) == _yaml_block(DOCS_ZH)


def test_docs_config_matches_bundle_source():
    # EN only — the identity test above makes the ZH comparison redundant.
    assert _functional_lines(_yaml_block(DOCS_EN)) == _functional_lines(
        BUNDLE_CONFIG.read_text()
    )


def test_build_script_layout_matches_flash_command():
    offsets = {name: int(offset, 16) for offset, name in FLASH_LAYOUT}
    assert build_bundle.LITTLEFS_OFFSET == offsets["littlefs.bin"]
    flashed = [
        name for name in build_bundle.ORDER if name not in build_bundle.REFERENCE_FILES
    ]
    assert flashed == [name for _, name in FLASH_LAYOUT]
