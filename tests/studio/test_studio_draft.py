"""Tests for `physiclaw.studio.draft` — the per-app authoring draft
store (mutations validate through the real pack doors)."""

from __future__ import annotations

import base64

import pytest

from physiclaw.conductor.pages import PagesError
from physiclaw.studio import draft as ds
from physiclaw.studio.draft import DraftError

JPEG_B64 = base64.b64encode(b"\xff\xd8fakejpeg").decode()
LISTING = '1 [text] "首页" [0.100,0.900,0.200,0.950] 0.95'


def _drafted(app: str = "shopdemo") -> dict:
    d = ds.load_draft(app)
    ds.add_page(d, "home")
    return d


def test_fresh_draft_and_round_trip() -> None:
    d = _drafted()
    ds.add_shot(d, "home", LISTING, JPEG_B64)
    ds.save_draft("shopdemo", d)

    again = ds.load_draft("shopdemo")

    assert again == d
    assert ds.shot_jpeg("shopdemo", "s1").read_bytes() == b"\xff\xd8fakejpeg"


def test_add_page_refuses_duplicates_and_bad_names() -> None:
    d = _drafted()

    with pytest.raises(DraftError, match="already drafted"):
        ds.add_page(d, "home")
    with pytest.raises(PagesError):
        ds.add_page(d, "Not A Name!")
    assert "Not A Name!" not in d["pages"]


def test_update_page_validates_and_rolls_back() -> None:
    d = _drafted()
    ds.update_page(d, "home", anchors=["首页", {"text": "搜索", "region": "top"}])

    with pytest.raises(PagesError, match="region"):
        ds.update_page(d, "home", anchors=[{"text": "x", "region": "middle"}])

    # The failed mutation left the page exactly as it was.
    assert d["pages"]["home"]["anchors"] == [
        "首页",
        {"text": "搜索", "region": "top"},
    ]


def test_empty_anchor_list_is_mid_authoring_not_an_error() -> None:
    d = _drafted()
    ds.update_page(d, "home", anchors=["首页"])

    ds.update_page(d, "home", anchors=[])

    assert d["pages"]["home"]["anchors"] == []


def test_shots_delete_with_their_page() -> None:
    d = _drafted()
    sid = ds.add_shot(d, "home", LISTING, JPEG_B64)
    path = ds.shot_jpeg("shopdemo", sid)

    ds.delete_page(d, "home")

    assert d["pages"] == {} and d["shots"] == {}
    assert not path.exists()


def test_delete_shot_removes_file_and_reference() -> None:
    d = _drafted()
    sid = ds.add_shot(d, "home", LISTING, JPEG_B64)

    ds.delete_shot(d, sid)

    assert d["shots"] == {} and d["pages"]["home"]["shots"] == []
    with pytest.raises(DraftError, match="no shot"):
        ds.shot_jpeg("shopdemo", sid)


def test_add_shot_refuses_an_empty_listing() -> None:
    d = _drafted()

    with pytest.raises(DraftError, match="camera read failed"):
        ds.add_shot(d, "home", "   ", JPEG_B64)


def test_controls_validate_through_the_pack_door() -> None:
    d = ds.load_draft("shopdemo")
    ds.set_control(d, "back", "back chevron", [0.02, 0.05, 0.1, 0.1])

    with pytest.raises(PagesError, match="closed"):
        ds.set_control(d, "cart", "cart", [0.8, 0.0, 0.9, 0.1])
    with pytest.raises(PagesError, match="left < right"):
        ds.set_control(d, "dismiss", "scrim", [0.9, 0.1, 0.2, 0.2])

    # Failures rolled back; the good control survived both.
    assert set(d["controls"]) == {"back"}
    ds.clear_control(d, "back")
    assert d["controls"] == {}


def test_discard_removes_the_draft_directory() -> None:
    d = _drafted()
    ds.add_shot(d, "home", LISTING, JPEG_B64)
    ds.save_draft("shopdemo", d)

    ds.discard("shopdemo")

    assert not ds.draft_dir("shopdemo").exists()
    assert ds.load_draft("shopdemo")["pages"] == {}
