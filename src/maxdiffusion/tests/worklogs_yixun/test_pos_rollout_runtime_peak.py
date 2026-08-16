"""exp_06 F9 `runtime-peaks`: the probe measures a RUNTIME peak, or it says it did not.

**The production result this round exists to fix.** M1-6 completed the whole fit-probe ladder on
v6e-8 and the authorization refused **all twelve cells on ``peak_source``**: every measurement
recorded ``"compiled memory analysis"``, and the plan (v2.8 §4-P1) plus the authorization floor
require runtime-derived evidence. The floor did exactly what it was reviewed to do. What it was
refusing was a probe that never took a runtime reading it could attribute to a cell.

**Why the existing code could not produce one, deterministically.** ``_measure_under_mesh`` opened
its attribution window *after* the compile step and ``WARMUP_STEPS`` warm-up steps of the very same
program. The allocator's ``peak_bytes_in_use`` is a monotone LIFETIME high-water mark, so the cell's
own warm-up had already set it; the timed steps then re-ran an identical program and could not raise
it. With no rise and no reset facility -- ``jaxlib._jax.Device`` in jax 0.10.2 exposes ``memory_stats``
and nothing that clears it, so ``reset_peak()`` is ``False`` on every backend this stack has -- the
only surviving source was the compiled analysis. Twelve cells, twelve analyses, no measurement.

**The two structural facts that shape the fix** (argued in full in the F9 worklog entry):

1. The window has to open at the top of the CELL -- before the per-cell backbone load and the
   compile -- because a window that opens after the cell's own warm-up can never be raised by the
   cell.
2. Raise-only attribution is not merely order-sensitive, it is *structurally* incapable of
   authorizing anything: ``TRIALS_PER_CELL == 2``, and trial 2 of a cell cannot raise the mark that
   trial 1 just set. So the standing mark has to be admissible -- and it is admissible *soundly*,
   because a monotone lifetime mark read after this cell's window is an UPPER bound on what this
   cell needed, whether or not this cell set it (the theorem is stated on
   :func:`~maxdiffusion.pos_rollout_fit_probe.classify_peak`). It is admitted only when it dominates
   this cell's own compiled analysis, so a "ceiling" below the cell's own floor is never reported as
   a ceiling.

Everything a device knows is a seam: these tests inject fake devices whose ``memory_stats`` is
absent / rising / non-rising / present-but-useless and pin the classification, the never-fake rule,
the floor's consumption of the result, and the fact that none of it moves the recipe fingerprint.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import json
import pathlib
import tempfile

import pytest
import yaml

from maxdiffusion import pos_rollout_fit_probe as probe

_GIB = 1024**3
_CAPACITY = 32 * _GIB
_CONFIG_PATH = pathlib.Path(probe.__file__).parent / "configs" / "base_wan_5b_pos_rollout.yml"


# =================================================================================================
# Fake devices: the four shapes a backend's memory statistics come in.
# =================================================================================================


class _Blind:
    """A CPU-shaped device: the attribute exists and answers ``None`` (jax 0.10.2 CPU does this)."""

    device_kind = "cpu"

    def memory_stats(self):
        return None


class _Absent:
    """A device with no statistics facility at all -- the ``AttributeError`` branch."""

    device_kind = "cpu"


class _Reporting:
    """A TPU-shaped device whose watermark the test drives."""

    device_kind = "v6e"

    def __init__(self, peak, *, capacity=_CAPACITY, keys=("peak_bytes_in_use", "bytes_limit")):
        self.peak = int(peak)
        self.capacity = int(capacity)
        self.keys = tuple(keys)

    def memory_stats(self):
        full = {"peak_bytes_in_use": int(self.peak), "bytes_limit": int(self.capacity), "bytes_in_use": 1}
        return {key: value for key, value in full.items() if key in self.keys or key == "bytes_in_use"}


def _telemetry(devices, *, resets=False):
    telemetry = probe.DeviceTelemetry()
    telemetry.devices = lambda: list(devices)
    if resets:
        telemetry.reset_peak = lambda: True
    else:
        telemetry.reset_peak = lambda: False
    return telemetry


def _classified(**over):
    values = {
        "watermark": 20 * _GIB,
        "capacity": _CAPACITY,
        "reset": False,
        "watermark_before": 20 * _GIB,
        "cell_watermark": 20 * _GIB,
        "analysis_bytes": 15 * _GIB,
    }
    values.update(over)
    return probe.classify_peak(**values)


# =================================================================================================
# 1. The vocabulary, and the never-fake rule.
# =================================================================================================


def test_the_attribution_vocabulary_is_exactly_the_four_states():
    """`reset` / `raised` / `standing` / `none` -- and the first three are the ones a runtime source
    may carry. A fifth state would be a provenance claim with no reviewed meaning."""
    assert probe.PEAK_ATTRIBUTIONS == (
        probe.PEAK_ATTRIBUTION_RESET,
        probe.PEAK_ATTRIBUTION_RAISED,
        probe.PEAK_ATTRIBUTION_STANDING,
        probe.PEAK_ATTRIBUTION_NONE,
    )
    assert probe.PEAK_ATTRIBUTION_NONE == "none"
    # The two RUNTIME sources are still exactly the reset one and the raised one -- that vocabulary
    # is about where a reading came from and F10 did not touch it. What F10 moved is the
    # AUTHORIZING set: the analysis is the bound now, so it is authorization-eligible, and the only
    # source that can never authorize is the one that reports a capacity a refusal hit.
    assert probe.RUNTIME_PEAK_SOURCES == (probe.PEAK_SOURCE_RUNTIME_RESET, probe.PEAK_SOURCE_RUNTIME_RAISED)
    assert probe.AUTHORIZING_PEAK_SOURCES == probe.RUNTIME_PEAK_SOURCES + (probe.PEAK_SOURCE_ANALYSIS,)
    assert probe.PEAK_SOURCE_REFUSED not in probe.AUTHORIZING_PEAK_SOURCES
    assert probe.PEAK_SOURCES == probe.AUTHORIZING_PEAK_SOURCES + (probe.PEAK_SOURCE_REFUSED,)


def test_an_unreadable_watermark_is_recorded_as_an_analysis_and_never_as_a_runtime_source():
    """THE never-fake rule, unchanged: a backend that reports no usable peak produces
    ``"compiled memory analysis"`` and says so.

    **F10b, review MAJOR 1 — and the F10 version of this test was the defect.** It asserted that a
    cell with a bound and NO watermark authorizes, on the reasoning that nothing falsified the
    bound. That pinned the implementation rather than the contract: Yixun's Option A RETAINS the
    watermark as a cross-check, and a cross-check that is skipped whenever it is absent is optional,
    not retained. Nothing had cross-checked that cell. The classification below is unchanged --
    provenance is still never invented -- and the verdict now refuses on ``watermark_missing``.
    """
    evidence = _classified(watermark=None, analysis_bytes=9 * _GIB)
    assert evidence.peak_source == probe.PEAK_SOURCE_ANALYSIS
    assert evidence.peak_attribution == probe.PEAK_ATTRIBUTION_NONE
    assert evidence.peak_bytes == 9 * _GIB and evidence.watermark_bytes is None
    verdict = probe.cell_verdict(_measurement(**_from(evidence)))
    assert not verdict.fits and verdict.reasons == ("watermark_missing",)
    assert verdict.numbers["authorized_bytes"] == 9 * _GIB, "the bound is still recorded, and still refused"
    assert verdict.numbers["watermark_bytes"] is None, "no reading is recorded as none, never as zero"
    # ...and the same cell with its analysis over the floor is refused on BOTH counts.
    big = _classified(watermark=None, analysis_bytes=int(_CAPACITY * 0.95))
    assert probe.cell_verdict(_measurement(**_from(big))).reasons == ("watermark_missing", "headroom")

    # ...and with nothing at all it fails closed rather than inventing either one.
    with pytest.raises(ValueError, match="no per-cell steady-state peak could be obtained"):
        _classified(watermark=None, analysis_bytes=None)


def test_a_backend_that_reports_a_capacity_but_no_MARK_authorizes_nothing():
    """F10b: the missing-watermark refusal is REACHABLE, not hypothetical.

    ``DeviceTelemetry.watermark_and_capacity`` returns ``(None, capacity)`` for a backend that
    reports ``bytes_limit`` without ``peak_bytes_in_use`` -- the "present but useless" shape this
    file has always tested at the telemetry level. Followed through to the verdict, that cell has a
    bound and no cross-check, and it must not authorize.
    """
    telemetry = _telemetry([_Reporting(30 * _GIB, keys=("bytes_limit",))])
    before = telemetry.begin_steady_state(cell_watermark=telemetry.begin_cell())
    evidence = telemetry.steady_state_evidence(before, telemetry.close_steady_state(before), program_bytes=11 * _GIB)
    assert evidence.peak_bytes == 11 * _GIB and evidence.watermark_bytes is None
    verdict = probe.cell_verdict(_measurement(**_from(evidence)))
    assert not verdict.fits and "watermark_missing" in verdict.reasons


def test_no_code_path_can_mint_a_runtime_source_without_a_watermark():
    """A structural read of the one function that assigns a source: every branch that names a
    runtime source is guarded by a watermark that was actually read."""
    body = inspect.getsource(probe.classify_peak)
    assert "watermark is None" in body, "the unreadable case is handled explicitly, not by falling through"
    for source in probe.RUNTIME_PEAK_SOURCES:
        assert body.count(repr(source)) == 0, "sources are referenced by constant, never spelled inline"


# =================================================================================================
# 2. The four fake-device shapes, end to end through DeviceTelemetry.
# =================================================================================================


def test_a_device_with_no_statistics_fails_closed_rather_than_measuring_nothing():
    for device in (_Blind(), _Absent()):
        telemetry = _telemetry([device])
        with pytest.raises(ValueError, match="reports no memory statistics"):
            telemetry.begin_cell()


def test_a_rising_watermark_is_this_cells_own_high_water_mark():
    """The cell's window opens BEFORE its backbone load and compile, so the rise is the cell's."""
    device = _Reporting(4 * _GIB)
    telemetry = _telemetry([device])
    cell_watermark = telemetry.begin_cell()
    assert cell_watermark == 4 * _GIB

    device.peak = 18 * _GIB  # the load + compile + warm-up raised it
    before = telemetry.begin_steady_state(cell_watermark=cell_watermark)
    device.peak = 21 * _GIB  # ...and the timed steps raised it further
    after = telemetry.close_steady_state(before)
    evidence = telemetry.steady_state_evidence(before, after, program_bytes=15 * _GIB)

    assert evidence.peak_source == probe.PEAK_SOURCE_RUNTIME_RAISED
    assert evidence.peak_attribution == probe.PEAK_ATTRIBUTION_RAISED
    assert evidence.peak_bytes == 21 * _GIB
    assert evidence.watermark_bytes == 21 * _GIB and evidence.watermark_before_bytes == 4 * _GIB
    assert evidence.analysis_bytes == 15 * _GIB


def test_a_rise_inside_the_load_and_compile_still_belongs_to_this_cell():
    """The regression M1-6 shipped: the steady window alone can never be raised, because the cell's
    own warm-up already set the mark. Attribution is against the CELL window."""
    device = _Reporting(4 * _GIB)
    telemetry = _telemetry([device])
    cell_watermark = telemetry.begin_cell()
    device.peak = 19 * _GIB  # raised by the load/compile/warm-up, and never again
    before = telemetry.begin_steady_state(cell_watermark=cell_watermark)
    evidence = telemetry.steady_state_evidence(before, telemetry.close_steady_state(before), program_bytes=15 * _GIB)
    assert evidence.peak_source == probe.PEAK_SOURCE_RUNTIME_RAISED
    assert evidence.peak_attribution == probe.PEAK_ATTRIBUTION_RAISED
    assert evidence.peak_bytes == 19 * _GIB


def test_a_non_rising_watermark_is_a_STANDING_ceiling_admitted_only_over_this_cells_analysis():
    """Trial 2 of every cell lands here, and so does every cell after the ladder's largest.

    The standing mark bounds this cell (the theorem), so it is admitted -- but only when it dominates
    the cell's own compiled analysis, because a ceiling below this cell's own floor is not a ceiling.
    """
    device = _Reporting(30 * _GIB)
    telemetry = _telemetry([device])
    before = telemetry.begin_steady_state(cell_watermark=telemetry.begin_cell())
    evidence = telemetry.steady_state_evidence(before, telemetry.close_steady_state(before), program_bytes=17 * _GIB)
    assert evidence.peak_source == probe.PEAK_SOURCE_RUNTIME_RAISED
    assert evidence.peak_attribution == probe.PEAK_ATTRIBUTION_STANDING
    assert evidence.peak_bytes == 30 * _GIB, "the ceiling is reported, not the floor"

    # The standing mark BELOW this cell's own analysis is discarded: the two disagree, so the runtime
    # number is not a demonstrated ceiling over this program and the analysis is what gets reported.
    inconsistent = _classified(
        watermark=12 * _GIB, watermark_before=12 * _GIB, cell_watermark=12 * _GIB, analysis_bytes=17 * _GIB
    )
    assert inconsistent.peak_source == probe.PEAK_SOURCE_ANALYSIS
    assert inconsistent.peak_attribution == probe.PEAK_ATTRIBUTION_NONE
    assert inconsistent.peak_bytes == 17 * _GIB


def test_a_standing_mark_with_no_analysis_at_all_still_fails_closed():
    """F1b's rule, preserved verbatim: with nothing cell-local to check the mark against, there is no
    per-cell number and the measurement refuses. (The reviewer's `attack_f1b_inherited_peak` is the
    battery's guard on exactly this.)"""
    telemetry = _telemetry([_Reporting(30 * _GIB)])
    before = telemetry.begin_steady_state()
    with pytest.raises(ValueError, match="no per-cell steady-state peak could be obtained"):
        telemetry.end_steady_state(before, program_bytes=None)


def test_statistics_that_report_no_peak_key_still_yield_a_capacity_and_an_analysis():
    """'present but useless' -- the fourth shape. The headroom rule needs the capacity, so losing the
    peak key must not lose the capacity too; and the source says analysis, because it is one."""
    telemetry = _telemetry([_Reporting(30 * _GIB, keys=("bytes_limit",))])
    assert telemetry.begin_cell() is None
    before = telemetry.begin_steady_state()
    peak, capacity, source = telemetry.end_steady_state(before, program_bytes=11 * _GIB)
    assert (peak, capacity, source) == (11 * _GIB, _CAPACITY, probe.PEAK_SOURCE_ANALYSIS)


def test_a_refused_allocation_is_still_recorded_when_only_the_peak_key_is_missing():
    """The capacity is what decides a miss, and a miss is a measured RESULT about the cell. Losing it
    over a statistic the refusal does not use would turn a fact into a crash."""
    telemetry = _telemetry([_Reporting(30 * _GIB, keys=("bytes_limit",))])
    assert probe._capacity_after_refusal(telemetry) == _CAPACITY

    blind = _telemetry([_Blind()])
    with pytest.raises(ValueError, match="exhausted the device AND the backend reports no memory statistics"):
        probe._capacity_after_refusal(blind)


def test_a_backend_that_can_reset_reports_the_reset_source():
    telemetry = _telemetry([_Reporting(5 * _GIB)], resets=True)
    before = telemetry.begin_steady_state(cell_watermark=99 * _GIB)
    evidence = telemetry.steady_state_evidence(before, telemetry.close_steady_state(before), program_bytes=None)
    assert evidence.peak_source == probe.PEAK_SOURCE_RUNTIME_RESET
    assert evidence.peak_attribution == probe.PEAK_ATTRIBUTION_RESET
    assert evidence.peak_bytes == 5 * _GIB, "a reset makes the mark the cell's own, whatever stood before"


def test_this_jax_offers_no_reset_so_the_reset_path_is_documented_not_assumed():
    """`runtime-reset` is kept because a backend may grow the facility; it is not what this stack
    does. Faking one would be inventing provenance -- the thing this whole round is about."""
    import jax

    device = jax.devices()[0]
    assert not hasattr(device, "clear_memory_stats") and not hasattr(device, "reset_memory_stats")
    assert probe.DeviceTelemetry().reset_peak() is False


# =================================================================================================
# 3. max(runtime, analysis), and the source that names the winner.
# =================================================================================================


def test_the_reported_peak_is_the_larger_and_the_source_names_where_it_came_from():
    """`classify_peak` names the origin of the number it reports exactly, so a cell whose analysis
    wins is refused on provenance and a cell whose runtime mark wins is judged on a ceiling.

    The invariant that survives the whole pipeline is the ONE-SIDED version -- the source never
    OVERSTATES -- and it has its own test below; see `classify_peak` (review F9c, MINOR)."""
    runtime_wins = _classified(watermark=25 * _GIB, cell_watermark=1 * _GIB, analysis_bytes=15 * _GIB)
    assert runtime_wins.peak_bytes == 25 * _GIB and runtime_wins.peak_source == probe.PEAK_SOURCE_RUNTIME_RAISED

    analysis_wins = _classified(watermark=25 * _GIB, cell_watermark=1 * _GIB, analysis_bytes=27 * _GIB)
    assert analysis_wins.peak_bytes == 27 * _GIB and analysis_wins.peak_source == probe.PEAK_SOURCE_ANALYSIS
    assert analysis_wins.watermark_bytes == 25 * _GIB, "the runtime reading is kept for audit either way"


def test_the_analysis_is_kept_as_its_own_field_whichever_number_wins():
    for analysis in (15 * _GIB, 27 * _GIB):
        evidence = _classified(watermark=25 * _GIB, cell_watermark=1 * _GIB, analysis_bytes=analysis)
        assert evidence.analysis_bytes == analysis


# =================================================================================================
# 4. The authorization floor's consumption -- the round's red anchor.
# =================================================================================================


def _from(evidence):
    return {
        "peak_bytes": evidence.peak_bytes,
        "capacity_bytes": evidence.capacity_bytes,
        "peak_source": evidence.peak_source,
        "peak_attribution": evidence.peak_attribution,
        "analysis_bytes": evidence.analysis_bytes,
        "watermark_bytes": evidence.watermark_bytes,
        "watermark_before_bytes": evidence.watermark_before_bytes,
    }


@functools.lru_cache(maxsize=1)
def _local_model_dir() -> str:
    """Provenance is CONTENT-bound, so a test config has to name a directory that exists."""
    directory = pathlib.Path(tempfile.mkdtemp(prefix="exp06_f9_model_")) / "snapshot"
    (directory / "transformer").mkdir(parents=True)
    (directory / "transformer" / "weights.safetensors").write_bytes(b"w" * 512)
    (directory / "model_index.json").write_text('{"_class_name": "test"}')
    return str(directory)


class _LadderDevice:
    """The device kind the context binds to. Eight of them is the v6e-8 M1 measures on."""

    device_kind = "v6e"


def _tiny_config(**overrides):
    """The production YAML with a resolvable model directory — the config M1 derives its context
    from, so a measurement built against `_context()` binds to a probe run given this config."""
    values = yaml.safe_load(_CONFIG_PATH.read_text())
    values["pretrained_model_name_or_path"] = _local_model_dir()
    values.update(overrides)

    class _Config:
        def __init__(self, mapping):
            self.__dict__.update(mapping)

        def get_keys(self):
            return dict(self.__dict__)

    return _Config(values)


def _context(config=None):
    return probe.derive_probe_context(
        config or _tiny_config(), devices=[_LadderDevice() for _ in range(8)], environ={}
    )


def _measurement(context=None, *, arm="rollout", microbatch=32, k_b=2, **over):
    context = context or _context()
    values = {
        "cell": probe.FitCell(arm=arm, microbatch=microbatch, k_b=k_b),
        "context_digest": context.binding_digest(),
        "compile_seconds": 480.0,
        "step_seconds": 3.5,
        "eval_seconds": 600.0,
        "checkpoint_seconds": 90.0,
        "peak_bytes": 20 * _GIB,
        "capacity_bytes": _CAPACITY,
        "reservation_failures": 0,
        # **F10 moved this default, deliberately.** Until F10 it was a runtime-WON cell (peak 20 GiB
        # from a watermark, over a 15 GiB analysis) because only a runtime source could authorize.
        # Under Yixun's Option A that exact shape is the INCONSISTENT one -- a watermark above the
        # claimed upper bound -- and it is now the fixture the refusal cases build explicitly. The
        # default is M1-9's measured shape instead: the analysis is the reported peak and the bound,
        # and the runtime mark sits below it (the PJRT watermark never sees XLA's temp arena).
        "peak_source": probe.PEAK_SOURCE_ANALYSIS,
        "peak_attribution": probe.PEAK_ATTRIBUTION_NONE,
        "analysis_bytes": 20 * _GIB,
        "watermark_bytes": 15 * _GIB,
        "watermark_before_bytes": 4 * _GIB,
    }
    values.update(over)
    return probe.CellMeasurement(**values)


def test_F10_authorizes_on_the_ANALYSIS_with_the_watermark_only_as_a_cross_check():
    """**The F10 anchor** (Yixun's Option A, plan v2.9 §4-P1), and the inversion of F9's.

    F9's version of this test asserted the opposite: a runtime-sourced peak authorizes and an
    analysis-sourced one is refused on ``peak_source``. M1-9 ran that rule on v6e and it authorized
    NOTHING -- twelve cells, watermarks 4.2-4.9 GiB against analyses of 10-30 GiB, so the analysis
    won every ``max`` and every cell was refused on its own conservative evidence. The bound is now
    the analysis; the watermark is recorded beside it and can only refuse.
    """
    # (a) The measured shape: analysis under the floor, watermark below the analysis. AUTHORIZED.
    verdict = probe.cell_verdict(_measurement())
    assert verdict.fits and verdict.reasons == ()
    assert verdict.numbers["peak_source"] == probe.PEAK_SOURCE_ANALYSIS

    # ...and a runtime label authorizes on exactly the same evidence: the LABEL stopped deciding.
    for source, attribution in (
        (probe.PEAK_SOURCE_RUNTIME_RESET, probe.PEAK_ATTRIBUTION_RESET),
        (probe.PEAK_SOURCE_RUNTIME_RAISED, probe.PEAK_ATTRIBUTION_RAISED),
        (probe.PEAK_SOURCE_RUNTIME_RAISED, probe.PEAK_ATTRIBUTION_STANDING),
    ):
        runtime = probe.cell_verdict(
            _measurement(peak_source=source, peak_attribution=attribution, watermark_bytes=20 * _GIB)
        )
        assert runtime.fits and runtime.reasons == (), f"{source}/{attribution} carries the same bound"

    # (b) Headroom, now read off the ANALYSIS: 30 of 32 GiB is 93.75%.
    over = probe.cell_verdict(_measurement(peak_bytes=30 * _GIB, analysis_bytes=30 * _GIB))
    assert not over.fits and over.reasons == ("headroom",)
    # ...and it is the analysis that decides it, not the reported peak: a 30 GiB analysis refuses
    # even when the reported peak is the same 30 GiB, and a 20 GiB analysis authorizes.
    assert probe.cell_verdict(_measurement(peak_bytes=20 * _GIB, analysis_bytes=20 * _GIB)).fits

    # (c) THE NEW REFUSAL: a watermark ABOVE the analysis falsifies the upper-bound premise.
    inconsistent = probe.cell_verdict(
        _measurement(
            peak_bytes=25 * _GIB,
            analysis_bytes=15 * _GIB,
            watermark_bytes=25 * _GIB,
            peak_source=probe.PEAK_SOURCE_RUNTIME_RAISED,
            peak_attribution=probe.PEAK_ATTRIBUTION_RAISED,
        )
    )
    assert not inconsistent.fits and inconsistent.reasons == ("watermark_exceeds_analysis",)

    # (d) No analysis at all is NO BOUND, and no bound authorizes nothing however good the reading.
    unbounded = probe.cell_verdict(
        _measurement(
            analysis_bytes=None,
            peak_bytes=20 * _GIB,
            watermark_bytes=20 * _GIB,
            peak_source=probe.PEAK_SOURCE_RUNTIME_RESET,
            peak_attribution=probe.PEAK_ATTRIBUTION_RESET,
        )
    )
    assert not unbounded.fits and unbounded.reasons == ("analysis_missing",)
    assert not probe.cell_verdict(_measurement(analysis_bytes=0)).fits, "a zero bound is not a bound"

    # (g) THE BOUNDARY: equality is not an excess. A watermark exactly at the analysis authorizes.
    boundary = probe.cell_verdict(
        _measurement(
            peak_bytes=20 * _GIB,
            analysis_bytes=20 * _GIB,
            watermark_bytes=20 * _GIB,
            peak_source=probe.PEAK_SOURCE_RUNTIME_RAISED,
            peak_attribution=probe.PEAK_ATTRIBUTION_RAISED,
        )
    )
    assert boundary.fits and boundary.reasons == ()
    # ...and one byte over is not.
    assert probe.cell_verdict(
        _measurement(
            peak_bytes=20 * _GIB + 1,
            analysis_bytes=20 * _GIB,
            watermark_bytes=20 * _GIB + 1,
            peak_source=probe.PEAK_SOURCE_RUNTIME_RAISED,
            peak_attribution=probe.PEAK_ATTRIBUTION_RAISED,
        )
    ).reasons == ("watermark_exceeds_analysis",)


def test_the_headroom_boundary_is_decided_in_EXACT_INTEGER_ARITHMETIC():
    """**F10b, review MODERATE.** The rule and its boundary are unchanged -- exactly 90% authorizes,
    one byte above refuses -- but the comparison is no longer a float division.

    A pinned gate should not depend on where the mantissa runs out. At HBM magnitudes the two forms
    agree; the pair below is chosen so they do NOT (``a/c`` evaluates to exactly ``0.9`` while
    ``10a > 9c``), which is the whole point: the predicate is the rule, and floating point is an
    implementation detail it must not inherit.
    """
    assert probe.HEADROOM_NUMERATOR / probe.HEADROOM_DENOMINATOR == probe.HEADROOM_FRACTION
    capacity = 40 * _GIB
    exactly_ninety = capacity * probe.HEADROOM_NUMERATOR // probe.HEADROOM_DENOMINATOR
    assert exactly_ninety * probe.HEADROOM_DENOMINATOR == capacity * probe.HEADROOM_NUMERATOR, "an exact 90%"
    at_the_line = _measurement(
        peak_bytes=exactly_ninety, analysis_bytes=exactly_ninety, watermark_bytes=1, capacity_bytes=capacity
    )
    assert probe.cell_verdict(at_the_line).fits, "exactly 90% of capacity authorizes"
    one_byte_over = dataclasses.replace(at_the_line, analysis_bytes=exactly_ninety + 1, peak_bytes=exactly_ninety + 1)
    assert probe.cell_verdict(one_byte_over).reasons == ("headroom",), "one byte above it does not"

    # The pair float division cannot separate: 10*analysis > 9*capacity, and analysis/capacity == 0.9.
    analysis, capacity = 8106479329266895, 9007199254740994
    assert analysis / capacity == probe.HEADROOM_FRACTION and 10 * analysis > 9 * capacity
    rounded = _measurement(peak_bytes=analysis, analysis_bytes=analysis, watermark_bytes=1, capacity_bytes=capacity)
    assert probe.cell_verdict(rounded).reasons == ("headroom",), "the float form authorized this cell"


def test_a_fractional_or_boolean_byte_count_is_a_MALFORMED_RECORD_not_a_measurement():
    """**F10b, review MODERATE.** ``int()`` rounds toward zero, so ``analysis = peak = watermark =
    9.9`` against a capacity of 10 used to become ``9/10`` and authorize, and ``True`` used to parse
    as one byte. A byte count is an integer; anything else was not measured by this probe, and it is
    refused the way every other structural defect is -- as a record that does not parse.
    """
    for override in (
        {"analysis_bytes": 9.9},
        {"watermark_bytes": 0.5},
        {"peak_bytes": 20 * _GIB + 0.5},
        {"capacity_bytes": float(_CAPACITY) + 0.5},
        {"analysis_bytes": True},
        {"peak_bytes": True},
        {"reservation_failures": True},
    ):
        with pytest.raises(ValueError):
            probe.cell_verdict(_measurement(**override))
    # The reviewer's exact construction, in the shape it would have arrived in: 9.9 of a 10-byte
    # device is 99% and authorized as 9/10.
    with pytest.raises(ValueError):
        probe.cell_verdict(_measurement(peak_bytes=9.9, analysis_bytes=9.9, watermark_bytes=9.9, capacity_bytes=10))
    # ...while an integral float is a whole number of bytes written another way, and still parses.
    assert probe.cell_verdict(_measurement(analysis_bytes=float(20 * _GIB))).fits


def test_the_integrality_test_is_against_the_ORIGINAL_value_and_not_a_float_of_it():
    """**F10c, review MODERATE (a).** F10b tested ``float(value).is_integer()``, and ``float`` is
    where the precision goes: a genuinely fractional ``Fraction``/``Decimal`` at this magnitude
    rounds to an integral float, truncates, and the truncated number sits exactly ON the 90% line
    while the true one is over it. The comparison is now ``value == int(value)``, which is exact for
    all three types. Digit STRINGS are rejected outright for the same reason: ``int("9")`` parses,
    and a record written as text was not written by this probe.
    """
    import decimal
    import fractions

    capacity = 9007199254740990  # divisible by 10, so its 90% point is exact
    exact_ninety = 9 * capacity // 10  # 8106479329266891 -- exactly on the line, and authorizing
    fractional = fractions.Fraction(2 * exact_ninety + 1, 2)  # ...891.5: OVER the line
    assert float(fractional).is_integer(), "the F10b test would have accepted this as a whole number"
    assert 10 * fractional > 9 * capacity, "the true value violates the headroom rule"
    assert 10 * int(fractional) == 9 * capacity, "...and its truncation sits exactly on it, authorizing"
    for value in (fractional, decimal.Decimal("8106479329266891.5"), "8106479329266891", b"9"):
        with pytest.raises(ValueError):
            probe.cell_verdict(
                _measurement(peak_bytes=value, analysis_bytes=value, watermark_bytes=1, capacity_bytes=capacity)
            )
    # The legitimate spellings of a whole number still parse: int, integral Fraction/Decimal, float.
    for value in (exact_ninety, fractions.Fraction(exact_ninety, 1), decimal.Decimal(exact_ninety)):
        assert probe.cell_verdict(
            _measurement(peak_bytes=value, analysis_bytes=value, watermark_bytes=1, capacity_bytes=capacity)
        ).fits


def test_a_recorded_payload_is_parsed_EXACTLY_rather_than_coerced():
    """**F10c, review MODERATE (b), at the deserialization boundary itself.**

    ``CellMeasurement.from_payload`` applied bare ``int()`` to the three counts, which runs BEFORE
    ``_checked`` ever sees the record — so a stored payload carrying ``peak_bytes: 9.9`` or ``true``
    became a well-formed measurement on the way in. The cell-artifact round trip (publish a
    malformed bank, attempt adoption, require re-measurement) is in the publication test file; this
    is the boundary in isolation, including the cell IDENTITY, where a truncation would authorize a
    DIFFERENT cell than the one measured.
    """
    payload = _measurement().as_payload()
    for field, value in (
        ("peak_bytes", 9.9),
        ("peak_bytes", True),
        ("capacity_bytes", "34359738368"),
        ("reservation_failures", 0.9),
        ("analysis_bytes", 9.9),
        ("watermark_bytes", True),
    ):
        with pytest.raises(ValueError, match="not a recorded cell measurement"):
            probe.CellMeasurement.from_payload({**payload, field: value})
    for field in ("microbatch", "k_b"):
        with pytest.raises(ValueError):
            probe.FitCell.from_payload({**payload["cell"], field: 8.5})
    # The honest payload still round-trips, unchanged.
    assert probe.CellMeasurement.from_payload(payload).as_payload() == payload


def test_a_reported_peak_above_the_bound_that_no_watermark_explains_is_refused_too():
    """The residue of the same fault. ``classify_peak`` reports ``max(runtime, analysis)``, so a
    peak above the analysis always has a watermark behind it -- unless the record was assembled
    somewhere else. Then the bound does not bound the number printed beside it, and the cell is
    refused rather than authorized on a bound its own row contradicts."""
    forged = probe.cell_verdict(
        _measurement(peak_bytes=int(_CAPACITY * 0.99), analysis_bytes=8 * _GIB, watermark_bytes=4 * _GIB)
    )
    assert not forged.fits and forged.reasons == ("peak_exceeds_analysis",)


def test_the_verdicts_numbers_carry_the_attribution_the_analysis_and_the_watermark_for_audit():
    """F10 adds the two numbers that say WHAT WAS DECIDED, beside the ones that say what was read."""
    numbers = probe.cell_verdict(_measurement()).numbers
    assert numbers["peak_attribution"] == probe.PEAK_ATTRIBUTION_NONE
    assert numbers["analysis_bytes"] == 20 * _GIB
    assert numbers["watermark_bytes"] == 15 * _GIB
    assert numbers["authorized_bytes"] == 20 * _GIB, "the bound the headroom rule was decided on"
    assert numbers["authorized_fraction"] == pytest.approx(20 / 32)
    assert numbers["headroom_fraction"] == probe.HEADROOM_FRACTION
    # A cell with no bound says so rather than reporting a fraction of nothing.
    unbounded = probe.cell_verdict(_measurement(analysis_bytes=None)).numbers
    assert unbounded["authorized_bytes"] is None and unbounded["authorized_fraction"] is None


def test_a_runtime_source_may_not_claim_no_attribution():
    with pytest.raises(ValueError, match="attribution"):
        probe.cell_verdict(
            _measurement(peak_source=probe.PEAK_SOURCE_RUNTIME_RAISED, peak_attribution=probe.PEAK_ATTRIBUTION_NONE)
        )


def test_an_unrecognised_attribution_is_refused():
    with pytest.raises(ValueError, match="attribution"):
        probe.cell_verdict(_measurement(peak_attribution="it looked fine"))


def test_a_runtime_peak_below_this_cells_own_analysis_is_refused_on_load():
    """The consistency guard, enforced at the gate and not only at capture time: a forged artifact
    cannot claim a runtime ceiling smaller than the same cell's static floor."""
    with pytest.raises(ValueError, match="below"):
        probe.cell_verdict(
            _measurement(
                peak_bytes=10 * _GIB,
                analysis_bytes=15 * _GIB,
                peak_source=probe.PEAK_SOURCE_RUNTIME_RAISED,
                peak_attribution=probe.PEAK_ATTRIBUTION_RAISED,
            )
        )


# =================================================================================================
# 5. Aggregation, publication, protocol.
# =================================================================================================


def test_trials_aggregate_to_the_WEAKEST_attribution():
    """Trial 1 raises the mark and trial 2 cannot; the cell is only as attributable as its worst
    trial, exactly as its peak is only as good as its worst trial."""
    context = _context()
    runtime = {
        "peak_source": probe.PEAK_SOURCE_RUNTIME_RAISED,
        "peak_bytes": 22 * _GIB,
        "analysis_bytes": 22 * _GIB,
    }
    aggregated = probe.aggregate_trials(
        [
            _measurement(
                context, peak_attribution=probe.PEAK_ATTRIBUTION_RAISED, watermark_bytes=20 * _GIB, **runtime
            ),
            _measurement(
                context, peak_attribution=probe.PEAK_ATTRIBUTION_STANDING, watermark_bytes=22 * _GIB, **runtime
            ),
        ]
    )
    assert aggregated[0].peak_attribution == probe.PEAK_ATTRIBUTION_STANDING
    assert aggregated[0].peak_source == probe.PEAK_SOURCE_RUNTIME_RAISED, "both trials are still runtime-derived"
    assert aggregated[0].watermark_bytes == 22 * _GIB and aggregated[0].analysis_bytes == 22 * _GIB
    assert aggregated[0].watermark_before_bytes == 4 * _GIB


def test_the_WORST_watermark_of_the_trials_is_what_the_F10_cross_check_reads():
    """F10's cross-check inherits F9's conservative aggregation for free: the cell's watermark is the
    MAX over its trials, so one trial whose mark went over the analysis refuses the whole cell."""
    context = _context()
    (aggregated,) = probe.aggregate_trials(
        [
            _measurement(context, watermark_bytes=4 * _GIB),
            _measurement(context, watermark_bytes=25 * _GIB),  # this trial's mark cleared the bound
        ]
    )
    assert aggregated.watermark_bytes == 25 * _GIB and aggregated.analysis_bytes == 20 * _GIB
    assert probe.cell_verdict(aggregated).reasons == ("watermark_exceeds_analysis",)


def _gated(tmp_path, name, trials, context):
    """Publish these TRIALS, reload the table, and try to gate a launch with the cell they describe.

    The reviewer's three exploits were executed end to end -- publication, reload,
    ``assert_cell_authorized`` -- because an aggregation defect that stops at the aggregate is a
    curiosity and one that reaches the gate is a 64-chip reservation. Returns ``(table, refusal)``;
    ``refusal is None`` means the cell reached a training run.
    """
    evidence = probe.build_evidence(context, trials, max_train_steps=10_000, eval_every=1_000, checkpoint_every=1_000)
    probe.publish_authorization(str(tmp_path / name), evidence)
    table = probe.load_authorization(str(tmp_path / name))
    try:
        probe.assert_cell_authorized(table, trials[0].cell, context=context)
    except ValueError as error:
        return table, str(error)
    return table, None


def test_a_TRIAL_LOCAL_contradiction_survives_aggregation_and_reaches_the_gate_as_a_refusal(tmp_path):
    """**F10b, review MAJOR 2 — the two exploits that agreed on the analysis.**

    The published table carries ONE aggregated record per cell, so a fault that lives in one trial
    has to survive into that record or nothing downstream can ever see it: the loader re-decides the
    aggregate, and the launch gate reads the loader's answer. With the analyses equal, the maxima do
    that -- but it is the END-TO-END path that has to demonstrate it, which is what the reviewer's
    executed exploits did and what this test does.
    """
    context = _context()
    # A mark that cleared the bound in the SECOND trial only.
    table, refusal = _gated(
        tmp_path,
        "trial_mark.json",
        [_measurement(context, watermark_bytes=4 * _GIB), _measurement(context, watermark_bytes=25 * _GIB)],
        context,
    )
    assert refusal is not None and table["authorized_cells"] == []
    assert table["refused_cells"][0]["reasons"] == ["watermark_exceeds_analysis"]

    # A peak that cleared the bound in the FIRST trial only.
    table, refusal = _gated(
        tmp_path,
        "trial_peak.json",
        [
            _measurement(context, peak_bytes=31 * _GIB, watermark_bytes=1 * _GIB),
            _measurement(context, watermark_bytes=1 * _GIB),
        ],
        context,
    )
    assert refusal is not None and table["authorized_cells"] == []
    assert table["refused_cells"][0]["reasons"] == ["peak_exceeds_analysis"]


def test_ONE_trial_without_the_evidence_makes_the_CELL_without_it(tmp_path):
    """**F10b, review MAJOR 2's third exploit.** ``max`` over "the trials that recorded it" answered
    with the friendly trial's number, so one trial with no analysis beside one with a 20 GiB bound
    authorized on a bound only half the cell had. An absent number now makes the cell absent."""
    context = _context()
    table, refusal = _gated(
        tmp_path,
        "trial_no_bound.json",
        [_measurement(context, analysis_bytes=None, peak_source=probe.PEAK_SOURCE_ANALYSIS), _measurement(context)],
        context,
    )
    assert refusal is not None and table["authorized_cells"] == []
    assert table["refused_cells"][0]["reasons"] == ["analysis_missing"]
    assert table["measurements"][0]["analysis_bytes"] is None, "the aggregate carries the absence, not the number"

    # ...and the same for the cross-check reading.
    table, refusal = _gated(
        tmp_path,
        "trial_no_mark.json",
        [_measurement(context, watermark_bytes=None), _measurement(context)],
        context,
    )
    assert refusal is not None and table["refused_cells"][0]["reasons"] == ["watermark_missing"]
    assert table["measurements"][0]["watermark_bytes"] is None


def test_TRIALS_THAT_DISAGREE_ABOUT_THE_ANALYSIS_ARE_REFUSED_OUTRIGHT():
    """**F10b, review MAJOR 2 — the exploits that needed two different bounds.**

    ``analysis=10, watermark=11`` beside ``analysis=20, watermark=5`` aggregated to
    ``analysis=20, watermark=11`` and authorized: the larger bound covered the smaller trial's
    contradicted mark. No choice of per-field aggregate fixes that, because the two trials disagree
    about what this cell IS. It is one compiled executable measured twice, so its own account of its
    footprint is one number -- M1-9's banked artifacts agree exactly, trial for trial, on all twelve
    cells -- and two numbers are refused the way two contexts already were: not collapsed.
    """
    context = _context()
    for left, right in (
        ({"analysis_bytes": 10 * _GIB, "watermark_bytes": 11 * _GIB}, {"analysis_bytes": 20 * _GIB}),
        (
            {"analysis_bytes": 8 * _GIB, "peak_bytes": 31 * _GIB, "watermark_bytes": 1 * _GIB},
            {"analysis_bytes": 32 * _GIB, "peak_bytes": 32 * _GIB},
        ),
    ):
        with pytest.raises(ValueError, match="analysis_disagreement"):
            probe.aggregate_trials([_measurement(context, **left), _measurement(context, **right)])
    # The honest case is untouched: two trials of one executable report one number, and it aggregates.
    (agreed,) = probe.aggregate_trials([_measurement(context), _measurement(context)])
    assert agreed.analysis_bytes == 20 * _GIB and probe.cell_verdict(agreed).fits


def test_the_audit_fields_survive_a_publication_round_trip(tmp_path):
    context = _context()
    evidence = probe.build_evidence(
        context, [_measurement(context)], max_train_steps=10_000, eval_every=1_000, checkpoint_every=1_000
    )
    published = probe.publish_authorization(str(tmp_path / "auth.json"), evidence)
    assert published["protocol"] == "exp06.fit_authorization.v7"
    recorded = published["measurements"][0]
    assert recorded["peak_attribution"] == probe.PEAK_ATTRIBUTION_NONE
    assert recorded["analysis_bytes"] == 20 * _GIB
    assert recorded["watermark_bytes"] == 15 * _GIB and recorded["watermark_before_bytes"] == 4 * _GIB
    reloaded = probe.load_authorization(str(tmp_path / "auth.json"))
    assert reloaded["measurements"][0]["peak_attribution"] == probe.PEAK_ATTRIBUTION_NONE
    assert reloaded["authorized_cells"], "an analysis-bounded cell authorizes once the headroom rule is satisfied"
    # F10: the published PROJECTION carries the bound the cell was decided on beside the watermark
    # that cross-checked it, so a reader re-derives the verdict from the table rather than trusting it.
    projection = reloaded["projections"][0]
    assert projection["authorized_bytes"] == 20 * _GIB and projection["watermark_bytes"] == 15 * _GIB
    assert projection["authorized_fraction"] == pytest.approx(20 / 32)


def test_the_protocols_name_the_shape_that_carries_the_runtime_evidence():
    """A v5 table / v1 cell cannot express an attribution, so a reader of one would take a
    ``peak_bytes`` whose provenance it cannot see. Fail closed on the version, as at every bump.

    **v7 (F10)** is the same discipline one step further: the FIELDS did not move, the DERIVATION
    did (analysis as the bound, watermark as the cross-check), so a v6 reader deciding a v7 table
    would be applying a rule the table was not decided under. The cell protocol is deliberately NOT
    bumped: what a trial MEASURES is unchanged, and bumping it would strand every banked M1-9 cell
    for a reason that has nothing to do with its numbers.
    """
    assert probe.AUTHORIZATION_PROTOCOL == "exp06.fit_authorization.v7"
    assert probe.CELL_PROTOCOL == "exp06.fit_cell.v2"


def test_a_v6_table_does_not_load_as_a_v7_one(tmp_path):
    """The bump has to BITE: a table decided under the old rule must not be read under the new one."""
    context = _context()
    evidence = probe.build_evidence(
        context, [_measurement(context)], max_train_steps=10_000, eval_every=1_000, checkpoint_every=1_000
    )
    path = str(tmp_path / "v6.json")
    probe.publish_authorization(path, evidence)
    stored = json.loads(pathlib.Path(path).read_text())
    stored["payload"]["protocol"] = "exp06.fit_authorization.v6"
    stored["sha256"] = hashlib.sha256(json.dumps(stored["payload"], sort_keys=True).encode("utf-8")).hexdigest()
    pathlib.Path(path).write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="is not exp06.fit_authorization.v7"):
        probe.load_authorization(path)
    # ...and the same table cannot gate a training run either, re-labelled or not.
    with pytest.raises(ValueError, match="needs the authorization the fit probe published"):
        probe.assert_cell_authorized(stored["payload"], probe.FitCell("rollout", 32, 2), context=context)


def test_a_DECLARED_exclusion_is_untouched_by_the_new_authorization_rule(tmp_path):
    """F6's exclusions are about cells this run never BUILT; F10 is about evidence for cells it did.

    The two must not bleed into each other: an excluded cell keeps its declared reason, stays out of
    the measured/authorized/refused lists, and is still refused AS AN EXCLUSION at the gate — while
    a measured cell beside it is decided under the new rule. Pinned because the F10 vocabulary is
    where a cell with no measurement could plausibly have acquired an ``analysis_missing`` refusal
    and quietly stopped being reported as a declaration.
    """
    context = _context()
    excluded = probe.FitCell("one_step", 32, 2)
    evidence = probe.build_evidence(
        context,
        [_measurement(context), _measurement(context, microbatch=16, analysis_bytes=int(_CAPACITY * 0.95))],
        max_train_steps=10_000,
        eval_every=1_000,
        checkpoint_every=1_000,
        exclusions=[(excluded, "bad_smem_address: deterministic XLA codegen fault (issue #18)")],
    )
    published = probe.publish_authorization(str(tmp_path / "excl.json"), evidence)
    reloaded = probe.load_authorization(str(tmp_path / "excl.json"))
    assert reloaded == published
    assert published["excluded_cells"] == [{**excluded.as_payload(), "reason": published["exclusion_reason"]}]
    assert "bad_smem_address" in published["exclusion_reason"]
    assert excluded.as_payload() not in published["measured_cells"]
    assert [entry["reasons"] for entry in published["refused_cells"]] == [["headroom"]]
    assert len(published["authorized_cells"]) == 1
    with pytest.raises(ValueError, match="DECLARED that cell EXCLUDED"):
        probe.assert_cell_authorized(published, excluded, context=context)


def test_a_previous_protocols_cell_artifact_is_refused(tmp_path):
    context = _context()
    artifact = probe.CellArtifact(
        cell=probe.FitCell("rollout", 32, 2), context=context, job_identity="j", trials=(_measurement(context),)
    )
    payload = artifact.as_payload()
    payload["protocol"] = "exp06.fit_cell.v1"
    with pytest.raises(ValueError, match="is not exp06.fit_cell.v2"):
        probe.CellArtifact.from_payload(payload)


# =================================================================================================
# 6. BINDING: peak capture is measurement mechanics, not recipe.
# =================================================================================================


def test_the_recipe_fingerprint_is_untouched_by_this_round():
    """Peak capture changes HOW a cell is measured, never WHAT it compiles to. If this digest moves,
    every banked cell in the campaign is invalidated for a reason that is not a recipe change."""
    values = yaml.safe_load(_CONFIG_PATH.read_text())
    values["pretrained_model_name_or_path"] = "/pinned/model"

    class _Config:
        def __init__(self, mapping):
            self.__dict__.update(mapping)

        def get_keys(self):
            return dict(self.__dict__)

    config = _Config(values)
    assert probe.recipe_fingerprint(config) == "42c5c870ba6eca7e0792c83909e378d9ae500e3e9f5d9743c91d0762a947fc55"
    assert len(probe.config_recipe(config)) == 177
    assert set(probe.FINGERPRINT_EXCLUSIONS.values()) <= set(probe.FINGERPRINT_EXCLUSION_REASONS)


def test_no_peak_capture_key_entered_the_fingerprint_exclusions():
    """The denylist is the fingerprint's contract; this round adds nothing to it."""
    for key in probe.FINGERPRINT_EXCLUSIONS:
        assert "peak" not in key and "watermark" not in key


# =================================================================================================
# 7. The measurement path opens the window where the fix requires.
# =================================================================================================


def test_the_cell_window_opens_before_the_backbone_load_and_the_compile():
    """The whole M1-6 defect in one assertion: ``begin_cell`` is called before
    ``build_probe_program``, so the cell's own load/compile/warm-up can raise a mark it is then
    credited with. A window opened after them can never be raised by the cell that opened it."""
    body = inspect.getsource(probe._measure_under_mesh)
    assert "begin_cell" in body, "the cell's attribution window has to be opened"
    assert body.index("begin_cell") < body.index(
        "build_probe_program"
    ), "the window must open BEFORE the backbone load and the compile, or the warm-up sets the mark first"
    assert "cell_watermark=" in body, "and the steady window has to be told where the cell's window opened"


def test_the_analysis_is_computed_after_the_watermark_is_read():
    """``_program_bytes`` lowers and compiles, which allocates. Reading the watermark first keeps a
    re-compile out of the number the cell is judged on."""
    body = inspect.getsource(probe._measure_under_mesh)
    assert body.index("close_steady_state") < body.index("_program_bytes")


def test_the_refused_allocation_path_records_no_runtime_claim():
    """A cell that exhausted the device reports the capacity it hit, with no attribution to invent."""
    source = inspect.getsource(probe.measure_cell_on_device)
    assert "PEAK_SOURCE_REFUSED" in source
    assert "PEAK_ATTRIBUTION_NONE" in source


def test_the_evidence_record_is_frozen_and_complete():
    fields = {field.name for field in dataclasses.fields(probe.PeakEvidence)}
    assert fields == {
        "peak_bytes",
        "capacity_bytes",
        "peak_source",
        "peak_attribution",
        "analysis_bytes",
        "watermark_bytes",
        "watermark_before_bytes",
    }


# =================================================================================================
# 8. F9b -- the ladder's EXECUTION ORDER, which exists because of everything above.
#
# The standing bound is sound but not tight, so the cell it belongs to matters: a cell above the
# headroom floor poisons every cell measured after it, and a cell below the floor cannot. Exactly one
# cell of this ladder is above the floor, so the fix is to run it LAST. These tests pin the declared
# order and then demonstrate, on a fake monotone device, the table the two orders produce.
# =================================================================================================

#: M1-6's measured analyses, per (arm, microbatch) -- the footprints the simulation replays. k barely
#: moves the peak (30.180 vs 30.182 GiB at mb=8), so the table is keyed on the pair.
_M1_6_FOOTPRINTS = {
    ("rollout", 8): 30.180,
    ("rollout", 16): 17.152,
    ("rollout", 32): 12.045,
    ("rollout", 64): 18.058,
    ("one_step", 8): 14.894,
    ("one_step", 16): 10.016,
}
#: The v6e-8 per-device capacity M1-6 reported, so the 90% floor is the production one.
_V6E_CAPACITY = int(31.246 * _GIB)


def test_the_declared_order_runs_the_only_above_floor_cell_LAST():
    """The ruling, pinned. `rollout` mb=8 is 96.6% of capacity and is refused on its own account
    whatever the order; what the order decides is whether it refuses everything measured after it."""
    assert probe.LADDER_ORDER[-1] == ("rollout", 8)
    cells = probe.ladder()
    assert cells[-2:] == (probe.FitCell("rollout", 8, 2), probe.FitCell("rollout", 8, 4))
    assert cells[0].arm == "one_step", "the one_step cells lead, ascending in microbatch"
    assert [cell.microbatch for cell in cells if cell.arm == "one_step"] == [8, 8, 16, 16, 32, 32, 64, 64]
    assert [cell.microbatch for cell in cells if cell.arm == "rollout"] == [32, 32, 16, 16, 64, 64, 8, 8]


def test_the_declared_order_covers_the_whole_ladder_exactly():
    """A pair the declaration forgets would run after the poisoning cell. It must name all of them."""
    assert set(probe.LADDER_ORDER) == {(arm, mb) for arm in probe.LADDER_ARMS for mb in probe.LADDER_MICROBATCH}
    assert len(probe.LADDER_ORDER) == len(set(probe.LADDER_ORDER)) == 8


def test_the_order_changes_the_SEQUENCE_and_nothing_else():
    """Orchestration only: same cells, same identities, same recipe fingerprint, same exclusions."""
    cells = probe.ladder()
    assert len(cells) == 16 and len(set(cells)) == 16
    assert set(cells) == {
        probe.FitCell(arm, mb, k)
        for arm in probe.LADDER_ARMS
        for mb in probe.LADDER_MICROBATCH
        for k in probe.LADDER_K
    }
    # The BINDING pin again, from this round's side: re-ordering the walk moves no fingerprint.
    values = yaml.safe_load(_CONFIG_PATH.read_text())
    values["pretrained_model_name_or_path"] = "/pinned/model"

    class _Config:
        def __init__(self, mapping):
            self.__dict__.update(mapping)

        def get_keys(self):
            return dict(self.__dict__)

    assert (
        probe.recipe_fingerprint(_Config(values)) == "42c5c870ba6eca7e0792c83909e378d9ae500e3e9f5d9743c91d0762a947fc55"
    )


def test_a_custom_ladder_keeps_the_declared_order_where_the_declaration_speaks():
    ordered = probe.ladder(arms=("rollout",), microbatches=(8, 64), horizons=(2,))
    assert ordered == (probe.FitCell("rollout", 64, 2), probe.FitCell("rollout", 8, 2))


# --- the poisoning simulation: the same cells, the two orders, the two tables --------------------


class _MonotoneProcess:
    """One process walking a ladder: a lifetime watermark that never falls, and no reset.

    This is the mechanism under test, not a mock of it -- the classification is production's
    :func:`classify_peak`; only the device's numbers are replayed.

    ``visible`` is the fraction of a cell's footprint the ALLOCATOR's mark can see. ``1.0`` is the
    idealised device F9b simulated. **M1-9 measured the real one at 0.15-0.45**: the PJRT watermark
    does not see XLA's temp-buffer arena, so the ladder read 4.2-4.9 GiB against 10-30 GiB analyses.
    F10's cross-check is decided by exactly this number, so it is a parameter and not a constant.
    """

    def __init__(self, footprints=None, capacity=_V6E_CAPACITY, visible=1.0):
        self.mark = 0
        self.capacity = int(capacity)
        self.footprints = footprints or _M1_6_FOOTPRINTS
        self.visible = float(visible)

    def footprint(self, cell) -> int:
        return int(self.footprints[(cell.arm, cell.microbatch)] * _GIB)

    def measure(self, cell) -> probe.PeakEvidence:
        """One trial: open the cell window, run the cell (which may raise the mark), classify."""
        opened = self.mark
        self.mark = max(self.mark, int(self.visible * self.footprint(cell)))
        return probe.classify_peak(
            watermark=self.mark,
            capacity=self.capacity,
            reset=False,
            watermark_before=opened,
            cell_watermark=opened,
            analysis_bytes=self.footprint(cell),
        )


def _walk(order, *, trials=probe.TRIALS_PER_CELL, process=None):
    """Walk `order` on a fresh process, TRIALS_PER_CELL trials per cell, and decide every cell."""
    context = _context()
    process = process or _MonotoneProcess()
    verdicts = {}
    for cell in order:
        taken = [
            _measurement(context, arm=cell.arm, microbatch=cell.microbatch, k_b=cell.k_b, **_from(evidence))
            for evidence in (process.measure(cell) for _ in range(trials))
        ]
        (aggregated,) = probe.aggregate_trials(taken)
        verdicts[cell] = probe.cell_verdict(aggregated)
    return verdicts


_RUNNABLE = (  # M1-6's measured set: the four one_step mb=32/64 cells were DECLARED unreachable
    probe.FitCell("one_step", 8, 2),
    probe.FitCell("one_step", 16, 2),
    probe.FitCell("rollout", 32, 2),
    probe.FitCell("rollout", 16, 2),
    probe.FitCell("rollout", 64, 2),
    probe.FitCell("rollout", 8, 2),
)


def test_the_OLD_order_refuses_the_small_cells_on_the_big_cells_MARK():
    """RED, kept as a test, and **F10 renames what it refuses on rather than repealing it.**

    `rollout` mb=8 first pushes the lifetime mark to 30.18 GiB. Under F9 that mark WAS the later
    cells' peak, so five cells that fit were refused on ``headroom``. Under F10 the later cells are
    bounded by their own analyses (10-18 GiB, comfortably under the floor) — and the standing mark
    now sits ABOVE those analyses, which is the cross-check's refusal: a watermark over the claimed
    upper bound. Same five cells, same cause, a reason that says what actually happened.
    """
    big_first = (_RUNNABLE[-1],) + _RUNNABLE[:-1]
    verdicts = _walk(big_first)
    assert verdicts[probe.FitCell("rollout", 8, 2)].reasons == ("headroom",)
    for cell in _RUNNABLE[:-1]:
        assert not verdicts[cell].fits, f"{cell} inherited the big cell's mark"
        assert verdicts[cell].reasons == ("watermark_exceeds_analysis",), "refused on a mark that is not its own"
        assert verdicts[cell].numbers["peak_attribution"] == probe.PEAK_ATTRIBUTION_STANDING
        assert verdicts[cell].numbers["authorized_bytes"] < verdicts[cell].numbers["watermark_bytes"]
    assert not any(verdict.fits for verdict in verdicts.values()), "the M1-6 outcome, for a new reason"


def test_the_DECLARED_order_on_a_TRUTH_TRACKING_device_still_refuses_what_the_mark_towers_over():
    """**The honest F10 reading of F9b's simulation, and it is not the F9 outcome.**

    This device's mark tracks the true footprint (``visible=1.0``), and the declaration orders by
    EXPECTED footprint, which M1-6's measured numbers do not exactly follow: `one_step` mb=8 came in
    at 14.9 GiB, above `one_step` mb=16 (10.0) and `rollout` mb=32 (12.0). Under F9 those two were
    authorized on a standing bound that was sound-but-loose and still under the floor. Under F10
    they are refused, because a 14.9 GiB mark is above their 10.0/12.0 GiB analyses and the contract
    refuses that conflict rather than adjudicating it.

    That is a REAL consequence of the new rule on a device of this shape, recorded here rather than
    smoothed over. The next test measures the shape M1-9 actually found, where it does not arise.
    """
    verdicts = _walk(_RUNNABLE)
    over_floor = probe.FitCell("rollout", 8, 2)
    assert verdicts[over_floor].reasons == ("headroom",), "refused on ITS OWN analysis, as it must be"
    assert verdicts[over_floor].numbers["authorized_bytes"] == int(30.180 * _GIB)
    assert verdicts[over_floor].numbers["peak_attribution"] == probe.PEAK_ATTRIBUTION_STANDING
    fits = {cell for cell, verdict in verdicts.items() if verdict.fits}
    assert fits == {
        probe.FitCell("one_step", 8, 2),  # the largest so far when measured: mark == its own analysis
        probe.FitCell("rollout", 16, 2),
        probe.FitCell("rollout", 64, 2),
    }
    for cell in (probe.FitCell("one_step", 16, 2), probe.FitCell("rollout", 32, 2)):
        assert verdicts[cell].reasons == ("watermark_exceeds_analysis",), f"{cell} runs under a taller mark"


def test_on_the_HARDWARE_M1_9_MEASURED_the_declared_order_authorizes_every_cell_that_fits():
    """**GREEN, and the case that decides M1-10.** The v6e PJRT watermark cannot see XLA's
    temp-buffer arena: M1-9 read 4.2-4.9 GiB while the analyses ran 10-30 GiB. Replay the same
    ladder with a mark that sees 30% of each footprint and the cross-check never fires — every cell
    under the floor authorizes on its analysis, and only the 96.6% cell is refused, on its own
    number."""
    verdicts = _walk(_RUNNABLE, process=_MonotoneProcess(visible=0.30))
    over_floor = probe.FitCell("rollout", 8, 2)
    assert verdicts[over_floor].reasons == ("headroom",)
    for cell in _RUNNABLE[:-1]:
        verdict = verdicts[cell]
        assert verdict.fits, f"{cell} is under the floor and nothing contradicts its bound"
        assert verdict.reasons == ()
        assert verdict.numbers["peak_source"] == probe.PEAK_SOURCE_ANALYSIS
        assert verdict.numbers["peak_source"] in probe.AUTHORIZING_PEAK_SOURCES
        assert verdict.numbers["authorized_fraction"] <= probe.HEADROOM_FRACTION
        assert verdict.numbers["watermark_bytes"] < verdict.numbers["authorized_bytes"]
    assert sum(1 for verdict in verdicts.values() if verdict.fits) == 5


def test_the_second_trial_of_every_cell_is_a_STANDING_bound_and_the_cell_still_authorizes():
    """Why the standing case had to be admitted at all: trial 2 cannot raise trial 1's mark, so a
    raise-only rule would have degraded every two-trial cell to its analysis — which F10 authorizes
    on, but F10 still needs the reading, because a mark it cannot attribute is a mark it cannot
    cross-check."""
    verdicts = _walk(_RUNNABLE)
    standing = probe.FitCell("rollout", 16, 2)  # the largest cell so far when it is measured
    assert verdicts[standing].numbers["peak_attribution"] == probe.PEAK_ATTRIBUTION_STANDING
    assert verdicts[standing].fits
    # ...and with one trial the same cell is `raised` when it is the largest so far.
    assert (
        _walk((probe.FitCell("one_step", 8, 2),), trials=1)[probe.FitCell("one_step", 8, 2)].numbers[
            "peak_attribution"
        ]
        == probe.PEAK_ATTRIBUTION_RAISED
    )


def test_a_partially_banked_restart_still_ends_with_the_big_cell_last():
    """Adoption SKIPS execution, so it cannot perturb the order: the cells actually executed are the
    declared order restricted, and `rollout` mb=8 is last among whatever survives."""
    declared = probe.ladder()
    for adopted in (
        set(),
        {probe.FitCell("one_step", 8, 2), probe.FitCell("one_step", 8, 4)},
        {cell for cell in declared if cell.arm == "one_step"},
        {cell for cell in declared if cell.microbatch in (16, 32)},
    ):
        executed = [cell for cell in declared if cell not in adopted]
        assert executed[-1] == probe.FitCell("rollout", 8, 4)
        assert executed[-2] == probe.FitCell("rollout", 8, 2)

    # ...and an EXCLUDED cell is absent from the order rather than reordering it.
    excluded = {probe.FitCell("one_step", 32, k) for k in probe.LADDER_K} | {
        probe.FitCell("one_step", 64, k) for k in probe.LADDER_K
    }
    survivors = [cell for cell in declared if cell not in excluded]
    assert not any(cell in excluded for cell in survivors)
    assert survivors[-2:] == [probe.FitCell("rollout", 8, 2), probe.FitCell("rollout", 8, 4)]
    assert [cell.arm for cell in survivors[:4]] == ["one_step"] * 4


# =================================================================================================
# 9. F9c -- the combined review's two production MAJORs and its MINOR.
# =================================================================================================


def test_a_peak_reached_only_during_the_EVALUATION_is_in_the_cells_evidence(monkeypatch):
    """Review MAJOR 1, and the reviewer's exact scenario.

    The window used to close straight after the timed steps, while this same function goes on to
    score a DEV pass and write a checkpoint -- and records the seconds of both in the measurement the
    projection uses. So a cell whose steps peaked under the floor and whose EVALUATION touched 95% of
    capacity was authorized on the pre-evaluation mark, and the 95% was never seen by anything.

    Here the steps take the mark to 87.5% (authorizing) and the eval takes it to 95.3% (refusing).
    Against the pre-F9c code this test fails: the cell comes back at 28 GiB and AUTHORIZED.

    **F10 keeps the scenario and changes which rule catches it.** The bound is the compiled analysis
    (28 GiB here, 87.5% — under the floor), so the headroom rule alone would authorize this cell;
    what refuses it is the cross-check, because the evaluation's allocations took the recorded
    watermark to 30.5 GiB, ABOVE the bound the cell claims. The phase bracketing is what makes that
    reading exist at all: with the window closed after the timed steps the mark would have read
    28 GiB, agreed with the analysis, and the cell would have been authorized.
    """
    marks = {"peak": 0}
    capacity = 32 * _GIB

    def _step(params, opt_state, batch, draws):
        marks["peak"] = max(marks["peak"], 28 * _GIB)  # 87.5% -- under the 90% floor
        return params, opt_state, 0.0

    def _score(params, batch, draws):
        marks["peak"] = max(marks["peak"], (61 * _GIB) // 2)  # 30.5 GiB = 95.3% -- over it
        return 0.0

    monkeypatch.setattr(
        probe,
        "build_probe_program",
        lambda config, cell, *, model_source=None: probe.ProbeProgram(
            step=_step, score=_score, params={}, opt_state={}, batch={}, draws=()
        ),
    )
    monkeypatch.setattr(probe, "_program_bytes", lambda program, params, opt_state: 28 * _GIB)
    monkeypatch.setattr(probe, "_time_one_checkpoint", lambda config, cell, params, opt_state: 0.01)

    class _Telemetry(probe.DeviceTelemetry):
        def reset_peak(self):
            return False

        def peak_and_capacity(self):
            return marks["peak"], capacity

    measurement = probe.measure_cell_on_device(
        cell=probe.FitCell("rollout", 32, 2),
        context=_context(),
        config=_tiny_config(),
        telemetry=_Telemetry(),
        model_source=object(),
    )
    assert measurement.peak_bytes == (61 * _GIB) // 2, "the evaluation's peak is the cell's peak"
    assert measurement.watermark_bytes == (61 * _GIB) // 2
    assert measurement.peak_attribution == probe.PEAK_ATTRIBUTION_RAISED
    verdict = probe.cell_verdict(measurement)
    assert verdict.numbers["authorized_bytes"] == 28 * _GIB
    assert verdict.numbers["authorized_fraction"] == pytest.approx(0.875), "the BOUND alone would authorize"
    assert not verdict.fits and verdict.reasons == ("watermark_exceeds_analysis",), "the eval's mark refuses it"
    # ...and the seconds of the phases the window now spans are still recorded.
    assert measurement.eval_seconds >= 0.0 and measurement.checkpoint_seconds == 0.01


def test_a_cell_whose_program_reports_no_analysis_has_no_bound_and_is_refused(monkeypatch):
    """F10's fail-closed corner, on the measurement path rather than on a hand-built record. The
    compiled analysis is the ONLY bound the authorization rule reads, so a program that cannot
    produce one produces no authorization — however clean its runtime reading is."""

    def _step(params, opt_state, batch, draws):
        return params, opt_state, 0.0

    monkeypatch.setattr(
        probe,
        "build_probe_program",
        lambda config, cell, *, model_source=None: probe.ProbeProgram(
            step=_step, score=lambda params, batch, draws: 0.0, params={}, opt_state={}, batch={}, draws=()
        ),
    )
    monkeypatch.setattr(probe, "_program_bytes", lambda program, params, opt_state: None)
    monkeypatch.setattr(probe, "_time_one_checkpoint", lambda config, cell, params, opt_state: 0.01)

    class _Telemetry(probe.DeviceTelemetry):
        """A mark that RISES inside the cell's window: the best runtime evidence this stack offers."""

        def __init__(self):
            self.calls = 0

        def reset_peak(self):
            return False

        def peak_and_capacity(self):
            self.calls += 1
            return (4 if self.calls > 1 else 1) * _GIB, 32 * _GIB

    measurement = probe.measure_cell_on_device(
        cell=probe.FitCell("rollout", 32, 2),
        context=_context(),
        config=_tiny_config(),
        telemetry=_Telemetry(),
        model_source=object(),
    )
    assert measurement.peak_attribution == probe.PEAK_ATTRIBUTION_RAISED, "the reading is as good as it gets"
    assert measurement.analysis_bytes is None
    assert probe.cell_verdict(measurement).reasons == ("analysis_missing",)


def test_the_window_closes_after_every_phase_the_cell_reports():
    """The structural companion: the close follows the eval and the checkpoint in the source, and the
    docstring that describes the order is now describing what happens."""
    body = inspect.getsource(probe._measure_under_mesh)
    assert body.index("program.score") < body.index("close_steady_state")
    assert body.index("_time_one_checkpoint") < body.index("close_steady_state")
    assert body.index("close_steady_state") < body.index("_program_bytes")


# --- MAJOR 3: the declared order governs the public seam ----------------------------------------


def test_an_explicit_cells_sequence_is_re_ordered_into_the_declared_order():
    """Review MAJOR 3, the reviewer's construction. Two legitimate cells in an order that pushes the
    watermark over the floor before the small one is measured."""
    poisoning = [probe.FitCell("rollout", 8, 2), probe.FitCell("one_step", 8, 2)]
    assert probe.order_cells(poisoning) == (probe.FitCell("one_step", 8, 2), probe.FitCell("rollout", 8, 2))

    reversed_ladder = tuple(reversed(probe.ladder()))
    assert probe.order_cells(reversed_ladder) == probe.ladder(), "any permutation lands in declared order"
    assert probe.order_cells(probe.ladder()) == probe.ladder(), "and the default path is unchanged"


def test_a_cell_the_declaration_does_not_name_is_refused_rather_than_appended():
    """Appending it would put an unknown footprint AFTER the cell the ordering exists to run last."""
    with pytest.raises(ValueError, match="not named in LADDER_ORDER"):
        probe.order_cells([probe.FitCell("rollout", 2, 2)])


def test_the_probe_walks_an_explicit_sequence_in_the_declared_order(tmp_path):
    """End to end through `run_fit_probe`: the seam the review found is closed at the seam."""
    calls = []

    def _measurer(*, cell, context, config):
        calls.append(cell)
        return _measurement(context, arm=cell.arm, microbatch=cell.microbatch, k_b=cell.k_b)

    probe.run_fit_probe(
        _tiny_config(pos_fit_authorization=str(tmp_path / "m1.json")),
        measurer=_measurer,
        cells=[probe.FitCell("rollout", 8, 2), probe.FitCell("one_step", 8, 2)],
        trials=1,
        devices=[_LadderDevice() for _ in range(8)],
    )
    assert calls == [probe.FitCell("one_step", 8, 2), probe.FitCell("rollout", 8, 2)]


def test_an_adopted_big_cell_does_not_let_an_unbanked_small_cell_run_after_it():
    """The specific hole the review named. Adoption SKIPS execution, so once the requested sequence
    is sorted, the cells actually executed are the declared order restricted — and `rollout` mb=8
    cannot execute before a smaller unbanked cell however the caller ordered its list."""
    asked = [probe.FitCell("rollout", 8, 2), probe.FitCell("one_step", 8, 2), probe.FitCell("rollout", 32, 2)]
    ordered = probe.order_cells(asked)
    for banked in (
        set(),
        {probe.FitCell("rollout", 8, 2)},
        {probe.FitCell("rollout", 8, 2), probe.FitCell("one_step", 8, 2)},
    ):
        executed = [cell for cell in ordered if cell not in banked]
        above_floor = [index for index, cell in enumerate(executed) if (cell.arm, cell.microbatch) == ("rollout", 8)]
        assert all(
            index == len(executed) - 1 for index in above_floor
        ), f"with {sorted(map(str, banked))} banked, the above-floor cell must still be last"


# --- MINOR: the one-sided provenance claim ------------------------------------------------------


def test_the_source_never_OVERSTATES_and_since_F10_the_NUMBERS_decide_anyway():
    """Review MINOR, re-read under F10. The label still never overstates its evidence — but it no
    longer decides the verdict, so what has to hold is that the aggregated NUMBERS stay conservative:
    the worst peak, the worst watermark, the worst analysis. A mixed-provenance cell whose runtime
    trial cleared the bound is refused on the cross-check, where under F9 it was refused on the
    label."""
    context = _context()
    (mixed,) = probe.aggregate_trials(
        [
            _measurement(
                context,
                peak_bytes=25 * _GIB,
                watermark_bytes=25 * _GIB,
                peak_source=probe.PEAK_SOURCE_RUNTIME_RAISED,
                peak_attribution=probe.PEAK_ATTRIBUTION_RAISED,
            ),
            _measurement(context, peak_bytes=20 * _GIB),
        ]
    )
    # The NUMBER is the runtime trial's; the LABEL is the analysis — an understatement of the
    # evidence, which can only ever refuse. And the cell IS refused: its worst watermark, 25 GiB,
    # stands above the 20 GiB bound its own analysis claims.
    assert mixed.peak_bytes == 25 * _GIB and mixed.peak_source == probe.PEAK_SOURCE_ANALYSIS
    assert mixed.watermark_bytes == 25 * _GIB and mixed.analysis_bytes == 20 * _GIB
    assert probe.cell_verdict(mixed).reasons == ("watermark_exceeds_analysis",)
    # The direction that must NEVER happen: an authorization over evidence one of the trials
    # contradicted. Every authorizing aggregate is one where no trial's mark cleared the bound.
    (clean,) = probe.aggregate_trials([_measurement(context), _measurement(context)])
    assert clean.peak_source in probe.AUTHORIZING_PEAK_SOURCES and probe.cell_verdict(clean).fits
