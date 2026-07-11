"""Tests for `physiclaw.core.hardware.exposure` — the convergence loop."""

from __future__ import annotations

from physiclaw.core.hardware import exposure
from physiclaw.core.hardware.exposure import converge
from physiclaw.core.vision.quality import QualityReport


def _r(median: float, clip: float = 0.0, sharp: float = 500.0) -> QualityReport:
    return QualityReport(sharpness=sharp, clip_pct=clip, median_luma=median)


GOOD = _r(120.0)
BLOWN = _r(150.0, clip=0.3)  # clip > 12% on a non-white screen


class Rig:
    """Fake camera: metered luma is a function of the current exposure
    state, so the loop's steps show up as measured changes (or don't,
    for the driver-ignores case)."""

    def __init__(self, auto_report: QualityReport, luma_of_exp: dict):
        self._auto = auto_report
        self._luma = luma_of_exp
        self.mode = "auto"
        self.exposure: int | None = None
        self.auto_calls = 0
        self.manual_calls: list[int] = []

    def set_auto(self) -> None:
        self.mode = "auto"
        self.auto_calls += 1

    def set_manual(self, value: int) -> None:
        self.mode = "manual"
        self.exposure = value
        self.manual_calls.append(value)

    def meter(self) -> QualityReport | None:
        if self.mode == "auto":
            return self._auto
        return self._luma[self.exposure]


def test_in_band_as_is_touches_nothing() -> None:
    rig = Rig(GOOD, {})

    res = converge(rig.meter, rig.set_auto, rig.set_manual, start=-6)

    assert res.ok and res.mode == "auto"
    assert rig.auto_calls == 0 and rig.manual_calls == []


def test_ae_reassert_recovers_without_manual() -> None:
    reports = iter([BLOWN, GOOD])  # bad as-is, good after re-assert
    rig = Rig(GOOD, {})

    res = converge(
        lambda: next(reports),
        rig.set_auto,
        rig.set_manual,
        start=-6,
    )

    assert res.ok and res.mode == "auto"
    assert rig.auto_calls == 1 and rig.manual_calls == []
    assert "re-assert" in res.detail


def test_converges_manually_stepping_darker() -> None:
    # AE keeps blowing the frame; -6 is still blown, -7 lands in band.
    rig = Rig(BLOWN, {-6: _r(180.0, clip=0.2), -7: _r(120.0)})

    res = converge(rig.meter, rig.set_auto, rig.set_manual, start=-6)

    assert res.ok and res.mode == "manual" and res.exposure == -7
    assert rig.manual_calls == [-6, -7]


def test_driver_ignoring_sets_reverts_to_auto() -> None:
    # Manual set produces the exact same luma as auto: the classic
    # ignored-write case (opencv#9738) — stall check must catch it.
    rig = Rig(BLOWN, {-6: BLOWN})

    res = converge(rig.meter, rig.set_auto, rig.set_manual, start=-6)

    assert not res.ok and res.mode == "auto"
    assert rig.manual_calls == [-6]
    assert rig.auto_calls == 2  # phase-2 re-assert + final revert
    assert "ignores" in res.detail


def test_oscillation_holds_darkest_non_blown_step() -> None:
    # Band lies between -7 (crushed dark) and -6 (still blown): after two
    # direction flips the loop stops and holds the darkest non-blown try.
    rig = Rig(BLOWN, {-6: _r(180.0, clip=0.2), -7: _r(15.0)})

    res = converge(rig.meter, rig.set_auto, rig.set_manual, start=-6)

    assert not res.ok and res.mode == "manual" and res.exposure == -7
    assert rig.manual_calls[-1] == -7  # re-applied as the held value
    assert "between steps" in res.detail


def test_meter_losing_frames_mid_stepping_fails_open() -> None:
    reports = iter([BLOWN, BLOWN, None])  # as-is, post-re-assert, step 1
    rig = Rig(BLOWN, {})

    res = converge(
        lambda: next(reports),
        rig.set_auto,
        rig.set_manual,
        start=-6,
    )

    assert not res.ok and res.mode == "auto"
    assert rig.auto_calls == 2  # re-assert + final revert


def test_start_is_clamped_into_range() -> None:
    rig = Rig(BLOWN, {exposure.MIN_EXPOSURE: _r(230.0, clip=0.2)})

    converge(rig.meter, rig.set_auto, rig.set_manual, start=-99)

    assert rig.manual_calls[0] == exposure.MIN_EXPOSURE


def test_range_exhaustion_reverts_to_auto() -> None:
    # Every step measures darker but stays blown; from MIN_EXPOSURE the
    # next darker step clamps to itself → range exhausted → auto.
    luma = {e: _r(150.0 + 5 * e, clip=0.3) for e in range(exposure.MIN_EXPOSURE, -5)}
    rig = Rig(BLOWN, luma)

    res = converge(
        rig.meter,
        rig.set_auto,
        rig.set_manual,
        start=-6,
        max_steps=20,
    )

    assert not res.ok and res.mode == "auto"
    assert rig.manual_calls[-1] == exposure.MIN_EXPOSURE
    assert "exhausted" in res.detail


def test_prefer_auto_false_skips_reassert_and_keeps_best_manual() -> None:
    # User pinned manual in config: no AE re-assert, and on failure the
    # best (darkest non-blown) step is held instead of reverting.
    rig = Rig(BLOWN, {-6: _r(190.0, clip=0.2), -7: _r(185.0, clip=0.2), -8: _r(15.0)})

    res = converge(
        rig.meter,
        rig.set_auto,
        rig.set_manual,
        start=-6,
        max_steps=3,
        prefer_auto=False,
    )

    assert not res.ok and res.mode == "manual" and res.exposure == -8
    assert rig.auto_calls == 0
    assert "auto disabled" in res.detail


def test_max_steps_bounds_the_search() -> None:
    rig = Rig(
        BLOWN,
        {-6: _r(190.0, clip=0.2), -7: _r(185.0, clip=0.2), -8: _r(180.0, clip=0.2)},
    )

    res = converge(
        rig.meter,
        rig.set_auto,
        rig.set_manual,
        start=-6,
        max_steps=3,
    )

    assert len(rig.manual_calls) == 3
    assert res.mode == "auto" and not res.ok
