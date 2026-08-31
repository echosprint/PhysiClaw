"""Tests for `physiclaw.studio.curate` — evidence badges, the dry-run
confusion matrix, and commit (all against synthetic shot listings)."""

from __future__ import annotations

import base64

import pytest

from physiclaw.common import paths
from physiclaw.common.paths import PACK_FILENAME
from physiclaw.common.text import read_text, write_text
from physiclaw.conductor.pages import load_learned
from physiclaw.studio import curate
from physiclaw.studio import draft as ds
from physiclaw.studio.draft import DraftError

JPEG_B64 = base64.b64encode(b"\xff\xd8fake").decode()


def _listing(*rows: tuple[str, float, float]) -> str:
    lines = []
    for i, (label, cx, cy) in enumerate(rows, 1):
        box = f"{cx - 0.05:.3f},{cy - 0.02:.3f},{cx + 0.05:.3f},{cy + 0.02:.3f}"
        lines.append(f'{i} [text] "{label}" [{box}] 0.95')
    return "\n".join(lines)


HOME = [("首页", 0.2, 0.95), ("推荐", 0.5, 0.1)]
RESULTS = [("综合", 0.2, 0.1), ("销量", 0.5, 0.1), ("筛选", 0.8, 0.1)]


def _draft(app: str = "shopdemo") -> dict:
    d = ds.load_draft(app)
    ds.add_page(d, "home")
    ds.update_page(d, "home", anchors=["首页", "推荐"])
    ds.add_page(d, "results")
    ds.update_page(d, "results", anchors=["综合", "销量"])
    for _ in range(2):
        ds.add_shot(d, "home", _listing(*HOME), JPEG_B64)
        ds.add_shot(d, "results", _listing(*RESULTS), JPEG_B64)
    return d


def test_evidence_counts_own_shots_and_shared_chrome() -> None:
    d = _draft()
    # "首页" also shows up in one results shot — shared chrome.
    ds.add_shot(d, "results", _listing(*RESULTS, ("首页", 0.2, 0.95)), JPEG_B64)

    ev = curate.evidence(d)

    assert ev["home"]["首页"] == {"seen": 2, "shots": 2, "other_pages": 1}
    assert ev["results"]["综合"] == {"seen": 3, "shots": 3, "other_pages": 0}


def test_dry_run_separates_distinct_pages() -> None:
    dry = curate.dry_run(_draft())

    assert dry["pages"]["home"]["separable"] is True
    assert dry["pages"]["results"]["separable"] is True
    own = [c for c in dry["matrix"] if c["page"] == "home"]
    assert all(c["scores"]["home"] == 1.0 for c in own)
    assert all(c["scores"]["results"] == 0.0 for c in own)


def test_dry_run_flags_a_lookalike_and_suggests_forbid() -> None:
    d = _draft()
    ds.add_page(d, "fake-home")
    ds.update_page(d, "fake-home", anchors=["首页", "推荐"])  # same fingerprint
    ds.add_shot(d, "fake-home", _listing(*HOME, ("直播中", 0.5, 0.5)), JPEG_B64)

    dry = curate.dry_run(d)

    assert dry["pages"]["home"]["separable"] is False
    # The impostor's unique label is the suggested veto term.
    assert "直播中" in dry["forbid_suggestions"]["home"]


def test_dry_run_skips_pages_still_without_anchors() -> None:
    d = _draft()
    ds.add_page(d, "later")

    dry = curate.dry_run(d)

    assert "later" not in dry["pages"]


def test_commit_writes_sections_learned_and_checks(tmp_path) -> None:
    pack_dir = paths.playbooks_dir() / "shopdemo"
    pack_dir.mkdir(parents=True)
    write_text(
        pack_dir / PACK_FILENAME,
        "app: shopdemo\n"
        "description: test pack\n"
        "\n"
        "pages:\n"
        "  stale:\n"
        '    anchors: ["旧"]\n'
        "\n"
        "# the walks stay untouched\n"
        "playbooks: {}\n",
    )
    d = _draft()
    ds.set_control(d, "back", "back chevron", [0.02, 0.05, 0.1, 0.1])

    result = curate.commit(d)

    text = read_text(pack_dir / PACK_FILENAME)
    assert "stale" not in text  # the pages section was replaced whole
    assert '- "首页"' in text and '- "综合"' in text
    assert 'label: "back chevron"' in text
    assert "# the walks stay untouched\nplaybooks: {}" in text
    assert result["check"] == "ok"
    assert sorted(load_learned("shopdemo")) == ["home", "results"]


def test_commit_scaffolds_a_missing_pack() -> None:
    # Distinct names on purpose: the scaffold stub's example walk
    # declares `home` in-route, and a draft page of the same name is a
    # loud duplicate (covered below) — authors rename or edit the stub.
    d = ds.load_draft("shopdemo")
    ds.add_page(d, "front")
    ds.update_page(d, "front", anchors=["首页"])
    ds.add_shot(d, "front", _listing(*HOME), JPEG_B64)

    result = curate.commit(d)

    text = read_text(paths.playbooks_dir() / "shopdemo" / PACK_FILENAME)
    assert "pages:\n  front:" in text
    assert result["pages_written"] == ["front"]


def test_commit_refuses_a_page_without_anchors() -> None:
    d = _draft()
    ds.add_page(d, "empty")

    with pytest.raises(DraftError, match="no anchors"):
        curate.commit(d)


def test_commit_refuses_a_clash_with_route_declared_pages() -> None:
    pack_dir = paths.playbooks_dir() / "shopdemo"
    pack_dir.mkdir(parents=True)
    write_text(
        pack_dir / PACK_FILENAME,
        "app: shopdemo\n"
        "description: test pack\n"
        "playbooks:\n"
        "  demo:\n"
        "    description: walk\n"
        "    enabled: false\n"
        "    route:\n"
        "      - page: home\n"
        '        anchors: ["首页"]\n'
        "      - tell: done\n"
        '        message: "done"\n',
    )

    with pytest.raises(DraftError, match="commit refused"):
        curate.commit(_draft())

    # Nothing was written: the route's own declaration survives alone.
    assert read_text(pack_dir / PACK_FILENAME).count("首页") == 1


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        # replace in the middle, later sections and their comments kept
        (
            "app: a\npages:\n  x:\n    anchors: [q]\n# doc\nplaybooks: {}\n",
            "app: a\nNEW\n# doc\nplaybooks: {}\n",
        ),
        # absent → appended
        ("app: a\n", "app: a\n\nNEW\n"),
        # replace at the end
        ("app: a\npages:\n  x:\n    anchors: [q]\n", "app: a\nNEW\n"),
    ],
)
def test_splice_section(original: str, expected: str) -> None:
    assert curate._splice_section(original, "pages", "NEW\n") == expected
