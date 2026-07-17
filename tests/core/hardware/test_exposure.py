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
    # Bad as-is; good after re-assert (the settled meter reads twice to
    # confirm the firmware AE loop has stopped moving).
    reports = iter([BLOWN, GOOD, GOOD])
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


def test_ae_reassert_meters_until_luma_settles() -> None:
    # The firmware AE re-converges over frames: consecutive reads that
    # still disagree are re-metered (bounded), and the settled value is
    # the one judged.
    reports = iter([BLOWN, _r(40.0), _r(110.0), _r(112.0)])
    rig = Rig(GOOD, {})

    res = converge(
        lambda: next(reports),
        rig.set_auto,
        rig.set_manual,
        start=-6,
    )

    assert res.ok and res.mode == "auto"
    assert "re-assert" in res.detail


def test_converges_manually_stepping_darker() -> None:
    # AE keeps blowing the frame; -6 is still blown, -7 lands in band.
    rig = Rig(BLOWN, {-6: _r(180.0, clip=0.2), -7: _r(120.0)})

    res = converge(rig.meter, rig.set_auto, rig.set_manual, start=-6)

    assert res.ok and res.mode == "manual" and res.exposure == -7
    assert rig.manual_calls == [-6, -7]


def test_driver_ignoring_sets_reverts_to_auto() -> None:
    # Manual sets produce the exact same luma step after step: the
    # classic ignored-write case (opencv#9738) — stall check must catch
    # it. Detection takes two manual reads: step 1 compares against the
    # auto baseline, which a working driver can legitimately match.
    rig = Rig(BLOWN, {-6: BLOWN, -7: BLOWN})

    res = converge(rig.meter, rig.set_auto, rig.set_manual, start=-6)

    assert not res.ok and res.mode == "auto"
    assert rig.manual_calls == [-6, -7]
    assert rig.auto_calls == 2  # phase-2 re-assert + final revert
    assert "ignores" in res.detail


def test_first_step_matching_auto_brightness_is_not_a_stall() -> None:
    # A manual start whose effective brightness matches what AE had
    # converged to is normal, not a driver ignoring writes: the search
    # must continue stepping — the next darker step lands in band. The
    # old first-step stall check aborted here and looped futile re-tunes.
    hot = _r(150.0, clip=0.05)  # over TUNE_CLIP_PCT, under the blown rule
    rig = Rig(hot, {-6: hot, -7: _r(120.0)})

    res = converge(rig.meter, rig.set_auto, rig.set_manual, start=-6)

    assert res.ok and res.mode == "manual" and res.exposure == -7
    assert rig.manual_calls == [-6, -7]


def test_in_band_step_is_accepted_however_little_the_luma_moved() -> None:
    # A step that lands in band is a success even when the median barely
    # moved from the previous read — the in-band accept must run before
    # the stall judgment, or a marginal AE state one step from the band
    # gets rejected as "driver ignores writes".
    rig = Rig(_r(130.0, clip=0.03), {-6: _r(129.0, clip=0.01)})

    res = converge(rig.meter, rig.set_auto, rig.set_manual, start=-6)

    assert res.ok and res.mode == "manual" and res.exposure == -6


def test_oscillation_holds_darkest_usable_step() -> None:
    # Band lies between -7 (crushed dark) and -6 (some clip, but under
    # the warning threshold and bright enough): after two direction
    # flips the loop stops and holds the darkest usable try.
    rig = Rig(BLOWN, {-6: _r(160.0, clip=0.08), -7: _r(18.0)})

    res = converge(rig.meter, rig.set_auto, rig.set_manual, start=-6)

    assert not res.ok and res.mode == "manual" and res.exposure == -6
    assert rig.manual_calls[-1] == -6  # re-applied as the held value
    assert "between steps" in res.detail


def test_oscillation_with_no_usable_step_reverts_to_auto() -> None:
    # -6 is heavily clipped and -7 is crushed to black: neither may be
    # held (a black frame passes "not blown" but is useless — the old
    # rule froze exactly such frames). With nothing usable, auto it is.
    rig = Rig(BLOWN, {-6: _r(180.0, clip=0.2), -7: _r(15.0)})

    res = converge(rig.meter, rig.set_auto, rig.set_manual, start=-6)

    assert not res.ok and res.mode == "auto"
    assert "no usable step" in res.detail


def test_meter_losing_frames_mid_stepping_fails_open() -> None:
    # as-is, settled post-re-assert pair, then step 1 loses the frame
    reports = iter([BLOWN, BLOWN, BLOWN, None])
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
    # best (darkest usable) step is held instead of reverting.
    rig = Rig(
        BLOWN,
        {-6: _r(190.0, clip=0.2), -7: _r(185.0, clip=0.2), -8: _r(40.0, clip=0.05)},
    )

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


def test_prefer_auto_false_holds_manual_start_when_no_step_is_usable() -> None:
    # Pinned manual + no usable step at all (best stays None): must still
    # NOT flip to firmware AE the user disabled — hold the start value.
    rig = Rig(BLOWN, {-6: _r(180.0, clip=0.2), -7: _r(15.0)})

    res = converge(
        rig.meter,
        rig.set_auto,
        rig.set_manual,
        start=-6,
        prefer_auto=False,
    )

    assert not res.ok and res.mode == "manual" and res.exposure == -6
    assert rig.auto_calls == 0
    assert rig.manual_calls[-1] == -6


def test_prefer_auto_false_clamps_held_start_into_range() -> None:
    # Pinned manual + no usable step, with a config start ABOVE the
    # range: the held fallback must be the clamped start (the value the
    # stepping loop actually ran), never the raw out-of-range config one.
    rig = Rig(BLOWN, {-5: _r(180.0, clip=0.2), -6: _r(15.0)})

    res = converge(
        rig.meter,
        rig.set_auto,
        rig.set_manual,
        start=-4,  # above MAX_EXPOSURE
        prefer_auto=False,
    )

    assert not res.ok and res.mode == "manual"
    assert res.exposure == exposure.MAX_EXPOSURE
    assert rig.manual_calls[-1] == exposure.MAX_EXPOSURE


def test_blob_blown_frame_steps_darker_to_a_clean_hold() -> None:
    # Low global clip but a burned icon grid: not acceptable as-is (the
    # QualityMonitor would flag it — a no-op re-tune loop otherwise),
    # and the step direction must be DARKER — the icons are clipped even
    # though the clip fraction alone reads "fine, go brighter".
    blob_blown = QualityReport(
        sharpness=500.0, clip_pct=0.015, median_luma=210.0, white_blobs=14
    )
    rig = Rig(
        blob_blown,
        {
            -6: QualityReport(
                sharpness=500.0, clip_pct=0.01, median_luma=180.0, white_blobs=14
            ),
            -7: _r(120.0),
        },
    )

    res = converge(rig.meter, rig.set_auto, rig.set_manual, start=-6)

    assert res.ok and res.mode == "manual" and res.exposure == -7
    assert rig.manual_calls == [-6, -7]


def test_dark_screen_defers_instead_of_converging() -> None:
    # The screen is dark (asleep / resting lock screen) even under AE:
    # there is no white level to expose for. Converging here is how a
    # frozen value blows out the screen once it lights up — defer, leave
    # firmware AE in charge, and let the caller retry on a bright view.
    rig = Rig(_r(5.0), {})

    res = converge(rig.meter, rig.set_auto, rig.set_manual, start=-6)

    assert res.deferred and not res.ok and res.mode == "auto"
    assert rig.manual_calls == []
    assert rig.auto_calls == 1  # the phase-2 re-assert, nothing else
    assert "deferred" in res.detail


def test_dark_screen_defers_without_touching_pinned_manual() -> None:
    # User pinned manual in config: the dark-screen deferral must not
    # flip the camera to auto behind their back.
    rig = Rig(_r(5.0), {})

    res = converge(
        rig.meter,
        rig.set_auto,
        rig.set_manual,
        start=-6,
        prefer_auto=False,
    )

    assert res.deferred and not res.ok
    assert rig.auto_calls == 0 and rig.manual_calls == []


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
