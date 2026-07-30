"""Exhaustive CPU-only tests for the exp_02 success statistic (plan D11 / G4).

``src/maxdiffusion/overfit100_success_statistic.py`` is the PURE unit that turns the
evaluator's aggregation rows into the plan's predeclared verdict -- no IO, no jax, no tf, so
the exact rule the experiment is judged by is executable and testable rather than computed by
hand. The contracts pinned here are the plan's, verbatim:

  * ``m_corr(w, c) = median_{seed in {0,1,2}} SSIM(w, c, seed | context_mode=correct)`` --
    the MEDIAN over seeds, and CORRECT MODE ONLY: ``null`` / ``shuffled`` rows are reported
    context and must never enter the statistic (pinned by a test that floods the input with
    perfect-SSIM ablation rows and demands an unchanged verdict).
  * ``fraction(c) = frac{w : m_corr(w, c) >= threshold}`` over a DENOMINATOR FIXED AT BUILD
    (the 100 canonical windows). Collision-flagged windows are recorded and stay in the
    denominator -- there is no code path that drops them (G4).
  * Headline claim ("canonical-window memorization"): established iff
    ``max_{c in C3_100} fraction(c) >= 0.90`` at threshold 0.95; PARTIAL at threshold 0.90.
  * ``c*`` = argmax fraction, ties broken by higher mean ``m_corr``, then by the EARLIER step
    (deterministic -- pinned by three tie tests, including a total tie).
  * Two-tier claim: the stronger "full-set memorization" additionally requires
    ``frac{SSIM(w, c*, seed 0 | correct) >= 0.90} >= 0.90`` over ALL built windows.
  * Fail-closed coverage: a missing (window, seed) at a segment-final checkpoint raises by
    default (a median over 2 of 3 seeds is not the defined statistic); ``strict=False`` marks
    the window unmeasured and NOT passing, and records it.

Stdlib + numpy only.
"""

from __future__ import annotations

import json

import pytest

import maxdiffusion.overfit100_success_statistic as stat

SEEDS = (0, 1, 2)


# --------------------------------------------------------------------------------------
# Row / cohort builders.
# --------------------------------------------------------------------------------------


def _row(*, episode_id, window_start, checkpoint_step, seed, ssim, context_mode="correct", **extra):
    row = {
        "name": f"ep{episode_id}_v0_s{window_start:05d}",
        "episode_id": int(episode_id),
        "episode_index": int(episode_id) - 1000,
        "window_start": int(window_start),
        "checkpoint_step": int(checkpoint_step),
        "seed": int(seed),
        "context_mode": context_mode,
        "ssim": float(ssim),
        "latent_mse": 0.01,
        "pixel_mse": 0.001,
    }
    row.update(extra)
    return row


def _windows(n, *, start=48):
    """``n`` canonical window keys (episode_id 1000.., one window each)."""
    return [(1000 + i, start) for i in range(n)]


def _cohort_rows(windows, *, checkpoint, ssim_by_window, seeds=SEEDS, mode="correct"):
    """One row per (window, seed): ``ssim_by_window`` maps window key -> per-seed list/float."""
    rows = []
    for key in windows:
        value = ssim_by_window[key]
        values = list(value) if isinstance(value, (list, tuple)) else [value] * len(seeds)
        for seed, ssim in zip(seeds, values):
            rows.append(
                _row(
                    episode_id=key[0],
                    window_start=key[1],
                    checkpoint_step=checkpoint,
                    seed=seed,
                    ssim=ssim,
                    context_mode=mode,
                )
            )
    return rows


def _uniform_rows(windows, *, checkpoint, n_pass, pass_ssim=0.97, fail_ssim=0.50, seeds=SEEDS):
    mapping = {key: (pass_ssim if i < n_pass else fail_ssim) for i, key in enumerate(windows)}
    return _cohort_rows(windows, checkpoint=checkpoint, ssim_by_window=mapping, seeds=seeds)


# --------------------------------------------------------------------------------------
# (1) m_corr: MEDIAN over seeds, correct mode only.
# --------------------------------------------------------------------------------------


def test_m_corr_is_the_median_over_seeds_not_the_mean():
    windows = _windows(1)
    # 0.10 / 0.96 / 0.98 -> median 0.96 (mean would be 0.68, which fails the 0.95 threshold).
    rows = _cohort_rows(windows, checkpoint=2500, ssim_by_window={windows[0]: [0.10, 0.96, 0.98]})
    out = stat.m_corr(rows, checkpoint=2500, windows=windows, seeds=SEEDS)
    assert out[windows[0]] == pytest.approx(0.96)


def test_m_corr_ignores_seed_order_in_the_input_rows():
    windows = _windows(1)
    rows = _cohort_rows(windows, checkpoint=2500, ssim_by_window={windows[0]: [0.10, 0.96, 0.98]})
    assert stat.m_corr(list(reversed(rows)), checkpoint=2500, windows=windows, seeds=SEEDS) == stat.m_corr(
        rows, checkpoint=2500, windows=windows, seeds=SEEDS
    )


def test_m_corr_uses_only_correct_mode_rows():
    windows = _windows(1)
    rows = _cohort_rows(windows, checkpoint=2500, ssim_by_window={windows[0]: 0.40})
    # Ablation rows at a PERFECT ssim must not move the statistic at all.
    for mode in ("null", "shuffled"):
        rows += _cohort_rows(windows, checkpoint=2500, ssim_by_window={windows[0]: 1.0}, mode=mode)
    assert stat.m_corr(rows, checkpoint=2500, windows=windows, seeds=SEEDS)[windows[0]] == pytest.approx(0.40)


def test_m_corr_ignores_other_checkpoints():
    windows = _windows(1)
    rows = _cohort_rows(windows, checkpoint=2500, ssim_by_window={windows[0]: 0.40})
    rows += _cohort_rows(windows, checkpoint=5000, ssim_by_window={windows[0]: 0.99})
    assert stat.m_corr(rows, checkpoint=2500, windows=windows, seeds=SEEDS)[windows[0]] == pytest.approx(0.40)
    assert stat.m_corr(rows, checkpoint=5000, windows=windows, seeds=SEEDS)[windows[0]] == pytest.approx(0.99)


def test_m_corr_missing_seed_raises_naming_window_and_seed():
    windows = _windows(1)
    rows = _cohort_rows(windows, checkpoint=2500, ssim_by_window={windows[0]: 0.9}, seeds=(0, 1))
    with pytest.raises(ValueError) as ei:
        stat.m_corr(rows, checkpoint=2500, windows=windows, seeds=SEEDS)
    msg = str(ei.value)
    assert "2" in msg and "1000" in msg  # the missing seed and the offending episode


def test_m_corr_non_strict_marks_missing_as_unmeasured():
    windows = _windows(2)
    rows = _cohort_rows(windows[:1], checkpoint=2500, ssim_by_window={windows[0]: 0.9})
    out = stat.m_corr(rows, checkpoint=2500, windows=windows, seeds=SEEDS, strict=False)
    assert out[windows[0]] == pytest.approx(0.9)
    assert out[windows[1]] is None


def test_m_corr_duplicate_window_seed_row_raises():
    windows = _windows(1)
    rows = _cohort_rows(windows, checkpoint=2500, ssim_by_window={windows[0]: 0.9})
    rows.append(_row(episode_id=1000, window_start=48, checkpoint_step=2500, seed=1, ssim=0.2))
    with pytest.raises(ValueError) as ei:
        stat.m_corr(rows, checkpoint=2500, windows=windows, seeds=SEEDS)
    assert "duplicate" in str(ei.value).lower()


def test_m_corr_extra_seed_outside_the_declared_set_is_refused():
    windows = _windows(1)
    rows = _cohort_rows(windows, checkpoint=2500, ssim_by_window={windows[0]: 0.9})
    rows.append(_row(episode_id=1000, window_start=48, checkpoint_step=2500, seed=7, ssim=0.9))
    with pytest.raises(ValueError) as ei:
        stat.m_corr(rows, checkpoint=2500, windows=windows, seeds=SEEDS)
    assert "7" in str(ei.value)


def test_m_corr_row_missing_a_required_field_is_actionable():
    row = _row(episode_id=1000, window_start=48, checkpoint_step=2500, seed=0, ssim=0.9)
    del row["ssim"]
    with pytest.raises((KeyError, ValueError)) as ei:
        stat.m_corr([row], checkpoint=2500, windows=_windows(1), seeds=(0,))
    assert "ssim" in str(ei.value)


def test_non_finite_ssim_is_refused():
    windows = _windows(1)
    rows = _cohort_rows(windows, checkpoint=2500, ssim_by_window={windows[0]: [float("nan"), 0.9, 0.9]})
    with pytest.raises(ValueError) as ei:
        stat.m_corr(rows, checkpoint=2500, windows=windows, seeds=SEEDS)
    assert "finite" in str(ei.value).lower()


# --------------------------------------------------------------------------------------
# (2) fraction over a FIXED denominator.
# --------------------------------------------------------------------------------------


def test_fraction_uses_the_fixed_denominator_not_the_measured_count():
    windows = _windows(10)
    m = {key: (0.99 if i < 5 else None) for i, key in enumerate(windows)}
    # 5 of 10 pass; the 5 unmeasured windows stay in the denominator (never dropped).
    assert stat.fraction_at(m, 0.95, denominator=10) == pytest.approx(0.5)


def test_fraction_threshold_is_inclusive():
    windows = _windows(2)
    m = {windows[0]: 0.95, windows[1]: 0.9499999}
    assert stat.fraction_at(m, 0.95, denominator=2) == pytest.approx(0.5)


def test_fraction_rejects_a_non_positive_denominator():
    with pytest.raises(ValueError):
        stat.fraction_at({}, 0.95, denominator=0)


# --------------------------------------------------------------------------------------
# (3) The headline claim: established / partial / none boundaries.
# --------------------------------------------------------------------------------------


def _verdict(rows, windows, checkpoints, **kw):
    return stat.evaluate_success(
        rows,
        canonical_windows=windows,
        segment_final_checkpoints=checkpoints,
        **kw,
    )


def test_headline_established_at_exactly_90_percent_of_windows_at_0_95():
    windows = _windows(100)
    rows = _uniform_rows(windows, checkpoint=2500, n_pass=90)
    out = _verdict(rows, windows, [2500])
    assert out["verdict"] == "established"
    assert out["headline"]["fraction"] == pytest.approx(0.90)
    assert out["headline"]["threshold"] == pytest.approx(0.95)
    assert out["denominator"] == 100


def test_headline_falls_to_partial_just_below_the_claim_fraction():
    windows = _windows(100)
    # 89 windows at 0.97 (>= 0.95) and 11 at 0.92 -> 0.89 at 0.95 but 1.00 at 0.90.
    mapping = {key: (0.97 if i < 89 else 0.92) for i, key in enumerate(windows)}
    rows = _cohort_rows(windows, checkpoint=2500, ssim_by_window=mapping)
    out = _verdict(rows, windows, [2500])
    assert out["verdict"] == "partial"
    assert out["headline"]["fraction"] == pytest.approx(0.89)
    assert out["partial"]["fraction"] == pytest.approx(1.0)
    assert out["partial"]["threshold"] == pytest.approx(0.90)


def test_verdict_none_when_both_tiers_miss():
    windows = _windows(100)
    rows = _uniform_rows(windows, checkpoint=2500, n_pass=50)
    out = _verdict(rows, windows, [2500])
    assert out["verdict"] == "none"
    assert out["headline"]["fraction"] == pytest.approx(0.50)
    assert out["partial"]["fraction"] == pytest.approx(0.50)


def test_headline_takes_the_MAX_over_segment_final_checkpoints():
    windows = _windows(100)
    rows = _uniform_rows(windows, checkpoint=2500, n_pass=40)
    rows += _uniform_rows(windows, checkpoint=5000, n_pass=95)
    rows += _uniform_rows(windows, checkpoint=7500, n_pass=60)
    out = _verdict(rows, windows, [2500, 5000, 7500])
    assert out["verdict"] == "established"
    assert out["headline"]["fraction"] == pytest.approx(0.95)
    assert out["c_star"] == 5000
    assert [c["checkpoint_step"] for c in out["per_checkpoint"]] == [2500, 5000, 7500]


def test_s2_checkpoints_are_not_part_of_the_statistic():
    # C3_100 is the argument: rows from other (S2) checkpoints are ignored entirely.
    windows = _windows(100)
    rows = _uniform_rows(windows, checkpoint=250, n_pass=100)  # a perfect S2 checkpoint
    rows += _uniform_rows(windows, checkpoint=2500, n_pass=10)
    out = _verdict(rows, windows, [2500])
    assert out["verdict"] == "none"
    assert [c["checkpoint_step"] for c in out["per_checkpoint"]] == [2500]


def test_ablation_rows_never_change_the_verdict():
    windows = _windows(100)
    rows = _uniform_rows(windows, checkpoint=2500, n_pass=50)
    baseline = _verdict(rows, windows, [2500])
    flooded = list(rows)
    for mode in ("null", "shuffled"):
        flooded += _cohort_rows(windows, checkpoint=2500, ssim_by_window=dict.fromkeys(windows, 1.0), mode=mode)
    after = _verdict(flooded, windows, [2500])
    assert after["verdict"] == baseline["verdict"] == "none"
    assert after["headline"]["fraction"] == pytest.approx(baseline["headline"]["fraction"])
    # The ablation gaps ARE reported (context), just never inputs to the rule.
    assert after["ablation_summary"]["null"]["mean_ssim"] == pytest.approx(1.0)
    assert after["ablation_summary"]["shuffled"]["n_rows"] == 300


# --------------------------------------------------------------------------------------
# (4) c* tie-break determinism: fraction -> mean m_corr -> EARLIER step.
# --------------------------------------------------------------------------------------


def test_c_star_prefers_the_higher_fraction():
    windows = _windows(10)
    rows = _uniform_rows(windows, checkpoint=2500, n_pass=5)
    rows += _uniform_rows(windows, checkpoint=5000, n_pass=6)
    out = _verdict(rows, windows, [2500, 5000])
    assert out["c_star"] == 5000
    assert out["c_star_tie_break"] == "fraction"


def test_c_star_tie_on_fraction_breaks_on_higher_mean_m_corr():
    windows = _windows(4)
    # Both checkpoints pass 2/4 windows, but 5000's mean m_corr is higher.
    rows = _cohort_rows(
        windows,
        checkpoint=2500,
        ssim_by_window={windows[0]: 0.96, windows[1]: 0.96, windows[2]: 0.10, windows[3]: 0.10},
    )
    rows += _cohort_rows(
        windows,
        checkpoint=5000,
        ssim_by_window={windows[0]: 0.96, windows[1]: 0.96, windows[2]: 0.80, windows[3]: 0.80},
    )
    out = _verdict(rows, windows, [2500, 5000])
    assert out["c_star"] == 5000
    assert out["c_star_tie_break"] == "mean_m_corr"


def test_c_star_total_tie_breaks_on_the_earlier_step():
    windows = _windows(4)
    mapping = {windows[0]: 0.96, windows[1]: 0.96, windows[2]: 0.10, windows[3]: 0.10}
    rows = _cohort_rows(windows, checkpoint=2500, ssim_by_window=mapping)
    rows += _cohort_rows(windows, checkpoint=5000, ssim_by_window=mapping)
    rows += _cohort_rows(windows, checkpoint=7500, ssim_by_window=mapping)
    out = _verdict(rows, windows, [7500, 2500, 5000])  # deliberately unsorted input
    assert out["c_star"] == 2500
    assert out["c_star_tie_break"] == "earlier_step"


def test_c_star_is_invariant_to_row_and_checkpoint_input_order():
    windows = _windows(6)
    rows = _uniform_rows(windows, checkpoint=2500, n_pass=3)
    rows += _uniform_rows(windows, checkpoint=5000, n_pass=3)
    a = _verdict(rows, windows, [2500, 5000])
    b = _verdict(list(reversed(rows)), windows, [5000, 2500])
    assert a["c_star"] == b["c_star"] == 2500
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --------------------------------------------------------------------------------------
# (5) The two-tier full-set gate.
# --------------------------------------------------------------------------------------


def _full_set_rows(all_windows, *, checkpoint, n_pass, pass_ssim=0.93, fail_ssim=0.10):
    """The ``s3_full_set`` pass's OWN rows: seed 0, correct mode, EVERY built window."""
    return [
        _row(
            episode_id=key[0],
            window_start=key[1],
            checkpoint_step=checkpoint,
            seed=0,
            ssim=pass_ssim if i < n_pass else fail_ssim,
        )
        for i, key in enumerate(all_windows)
    ]


def test_full_set_claim_requires_the_seed0_fraction_over_all_built_windows():
    # D3: the stronger tier is scored from the ``s3_full_set`` pass's OWN rows -- a separate,
    # role-validated artifact covering EVERY built window at seed 0 / correct mode.
    canonical = _windows(100)
    all_windows = canonical + [(2000 + i, 4 * i) for i in range(100)]
    rows = _uniform_rows(canonical, checkpoint=2500, n_pass=95)
    full_set = {"windows": all_windows, "rows": _full_set_rows(all_windows, checkpoint=2500, n_pass=190)}
    out = _verdict(rows, canonical, [2500], full_set=full_set)
    assert out["verdict"] == "established"
    assert out["full_set_claim"]["evaluable"] is True
    assert out["full_set_claim"]["fraction"] == pytest.approx(0.95)  # 190 of 200
    assert out["full_set_claim"]["established"] is True
    assert out["full_set_claim"]["n_windows"] == 200


def test_full_set_claim_fails_when_the_gate_fraction_misses():
    canonical = _windows(100)
    all_windows = canonical + [(2000 + i, 4 * i) for i in range(100)]
    rows = _uniform_rows(canonical, checkpoint=2500, n_pass=95)
    full_set = {"windows": all_windows, "rows": _full_set_rows(all_windows, checkpoint=2500, n_pass=95)}
    out = _verdict(rows, canonical, [2500], full_set=full_set)
    assert out["verdict"] == "established"  # the canonical claim still stands
    assert out["full_set_claim"]["established"] is False
    assert out["full_set_claim"]["fraction"] == pytest.approx(0.475)  # 95 of 200


def test_full_set_claim_refuses_incomplete_coverage_instead_of_scoring_a_subset():
    # D3: partial coverage is NOT a low fraction -- it is not evaluable at all.
    canonical = _windows(10)
    all_windows = canonical + [(2000 + i, 0) for i in range(10)]
    rows = _uniform_rows(canonical, checkpoint=2500, n_pass=10)
    full_set = {
        "windows": all_windows,
        "rows": _full_set_rows(all_windows[:-1], checkpoint=2500, n_pass=19),  # one window absent
    }
    out = _verdict(rows, canonical, [2500], full_set=full_set)
    assert out["full_set_claim"]["evaluable"] is False
    assert out["full_set_claim"]["established"] is False
    assert "coverage" in out["full_set_claim"]["reason"]
    assert out["full_set_claim"]["n_measured"] == len(all_windows) - 1


def test_identical_duplicate_measurements_are_collapsed_but_conflicts_are_refused():
    # Two artifacts reporting the SAME (window, checkpoint, seed, mode) measurement is normal
    # (the canonical windows appear in both passes); a DIFFERENT value for it is not.
    windows = _windows(1)
    rows = _cohort_rows(windows, checkpoint=2500, ssim_by_window={windows[0]: [0.10, 0.96, 0.98]})
    same_again = [dict(row) for row in rows]
    out = stat.m_corr(rows + same_again, checkpoint=2500, windows=windows, seeds=SEEDS)
    assert out[windows[0]] == pytest.approx(0.96)
    conflicting = [dict(rows[0], ssim=0.42)]
    with pytest.raises(ValueError) as ei:
        stat.m_corr(rows + conflicting, checkpoint=2500, windows=windows, seeds=SEEDS)
    assert "conflict" in str(ei.value).lower()


def test_full_set_claim_not_evaluable_without_the_all_window_pass():
    windows = _windows(100)
    rows = _uniform_rows(windows, checkpoint=2500, n_pass=95)
    out = _verdict(rows, windows, [2500])
    assert out["verdict"] == "established"
    assert out["full_set_claim"]["evaluable"] is False
    assert out["full_set_claim"]["established"] is False
    # The reason names the artifact role the tier requires (D3), not a vague "no data".
    assert "s3_full_set" in out["full_set_claim"]["reason"]


def test_full_set_gate_is_measured_at_c_star_only():
    canonical = _windows(10)
    extra = [(2000 + i, 0) for i in range(10)]
    all_windows = canonical + extra
    rows = _uniform_rows(canonical, checkpoint=2500, n_pass=4)
    rows += _uniform_rows(canonical, checkpoint=5000, n_pass=9)  # c* = 5000
    # Two full-set passes, one per checkpoint: PERFECT at 2500 and terrible at 5000, so the gate
    # must read c* = 5000 (0.0) and not the 2500 pass (1.0).
    full_rows = _full_set_rows(all_windows, checkpoint=2500, n_pass=len(all_windows), pass_ssim=0.99)
    full_rows += _full_set_rows(all_windows, checkpoint=5000, n_pass=0)
    out = _verdict(rows, canonical, [2500, 5000], full_set={"windows": all_windows, "rows": full_rows})
    assert out["c_star"] == 5000
    assert out["full_set_claim"]["fraction"] == pytest.approx(0.0)  # NOT 1.0 (the 2500 pass)
    assert out["full_set_claim"]["established"] is False


# --------------------------------------------------------------------------------------
# (6) Collision flagging: recorded, NEVER dropped from the denominator.
# --------------------------------------------------------------------------------------


def test_flagged_windows_stay_in_the_denominator_and_are_recorded():
    windows = _windows(10)
    rows = _uniform_rows(windows, checkpoint=2500, n_pass=5)
    flagged = [windows[0], windows[9]]
    out = _verdict(rows, windows, [2500], flagged_windows=flagged)
    assert out["denominator"] == 10  # unchanged by flagging
    assert out["flagged_windows"] == [list(key) for key in flagged]
    assert out["headline"]["fraction"] == pytest.approx(0.5)
    # And flagging a window that PASSES does not remove its contribution either.
    out2 = _verdict(rows, windows, [2500], flagged_windows=[windows[0]])
    assert out2["headline"]["fraction"] == pytest.approx(0.5)


def test_flagged_window_outside_the_denominator_is_refused():
    windows = _windows(3)
    rows = _uniform_rows(windows, checkpoint=2500, n_pass=3)
    with pytest.raises(ValueError) as ei:
        _verdict(rows, windows, [2500], flagged_windows=[(9999, 0)])
    assert "9999" in str(ei.value)


# --------------------------------------------------------------------------------------
# (7) Fail-closed coverage + input validation.
# --------------------------------------------------------------------------------------


def test_incomplete_seed_coverage_raises_by_default():
    windows = _windows(3)
    rows = _uniform_rows(windows, checkpoint=2500, n_pass=3, seeds=(0, 1))
    with pytest.raises(ValueError):
        _verdict(rows, windows, [2500])


def test_non_strict_records_unmeasured_windows_as_not_passing():
    windows = _windows(10)
    rows = _uniform_rows(windows[:5], checkpoint=2500, n_pass=5)
    out = _verdict(rows, windows, [2500], strict=False)
    entry = out["per_checkpoint"][0]
    assert entry["n_measured"] == 5
    assert entry["fraction"] == pytest.approx(0.5)  # 5 of the FIXED 10
    assert entry["unmeasured_windows"] == [list(key) for key in windows[5:]]
    assert out["coverage_complete"] is False


def test_duplicate_canonical_window_in_the_denominator_is_refused():
    windows = _windows(3) + [_windows(3)[0]]
    rows = _uniform_rows(_windows(3), checkpoint=2500, n_pass=3)
    with pytest.raises(ValueError) as ei:
        _verdict(rows, windows, [2500])
    assert "duplicate" in str(ei.value).lower()


def test_empty_canonical_window_set_is_refused():
    with pytest.raises(ValueError):
        _verdict([], [], [2500])


def test_empty_checkpoint_set_is_refused():
    windows = _windows(2)
    rows = _uniform_rows(windows, checkpoint=2500, n_pass=2)
    with pytest.raises(ValueError):
        _verdict(rows, windows, [])


# --------------------------------------------------------------------------------------
# (8) The verdict dict is JSON-serializable and self-describing.
# --------------------------------------------------------------------------------------


def test_verdict_is_json_round_trippable_and_carries_its_inputs():
    windows = _windows(10)
    rows = _uniform_rows(windows, checkpoint=2500, n_pass=9)
    rows += _uniform_rows(windows, checkpoint=5000, n_pass=10)
    out = _verdict(rows, windows, [2500, 5000], flagged_windows=[windows[3]])
    text = json.dumps(out, sort_keys=True)
    assert json.loads(text) == out
    assert out["schema"] == stat.VERDICT_SCHEMA
    assert out["seeds"] == [0, 1, 2]
    assert out["segment_final_checkpoints"] == [2500, 5000]
    assert out["claim_fraction"] == pytest.approx(0.90)
    assert out["context_mode"] == "correct"
    # Per-window m_corr is reported for every checkpoint (the audit trail behind the fraction).
    per_ckpt = {c["checkpoint_step"]: c for c in out["per_checkpoint"]}
    assert per_ckpt[5000]["m_corr"][str(list(windows[0]))] == pytest.approx(0.97)


# NOTE: the verdict CLI's own contracts (manifest-derived cohort, hash verification, denominator
# refusal, role-validated C3_100, derived full-set cohort) live in
# ``test_overfit100_eval_contracts.py`` -- they need a manifest fixture, which this pure-unit file
# deliberately does not build.


def test_rows_from_artifacts_concatenates_in_order():
    windows = _windows(2)
    a = {"schema": "x", "rows": _uniform_rows(windows, checkpoint=2500, n_pass=2)}
    b = {"schema": "x", "rows": _uniform_rows(windows, checkpoint=5000, n_pass=1)}
    rows = stat.rows_from_artifacts([a, b])
    assert len(rows) == len(a["rows"]) + len(b["rows"])
    assert rows[0] == a["rows"][0]
    with pytest.raises((KeyError, ValueError)):
        stat.rows_from_artifacts([{"no_rows": []}])
