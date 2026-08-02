"""exp_03 cycle A round 1 — the two PREDECLARED diagnostics (plan v3.1, Mechanism A and B).

Both are predeclared *before* any trial runs, which is the point: the reduction formula, the
bootstrap protocol, the trace definition and the acceptance direction are fixed while nobody knows
which arm wins.

* ``d1_per_frame_slopes`` (Mechanism A, temporal) — per-window OLS on frames 1→32, reduction
  ``1 - mean_slope_trial / mean_slope_control``, paired per-episode bootstrap (10,000 resamples,
  95% CI, seed 0), plus the exp_02 self-validation check (mean-over-frames vs the SSIM the eval
  recorded), retained because the frames come from lossy MP4s.
* ``sigma_trajectory_trace`` (Mechanism B, denoising-trajectory) — the fixed-ε latent error against
  the ideal interpolant at every sigma step, on the SAME extracted sampler step the eval runs.
  D1 measures temporal decay of decoded frames; only this measures trajectory divergence, which is
  what the objectives actually target.

The slope tests use synthetic curves with slopes known in closed form, so a wrong sign, a
wrong-axis fit or an off-by-one in the frame window is a failure rather than a plausible number.
The trace tests use a perfect velocity oracle, whose trace is zero by the interpolant algebra.
"""

from __future__ import annotations

import ast
import inspect
import json
import zlib
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import maxdiffusion.generate_wan_side_adapter as gen
import maxdiffusion.probe_overfit100_sampling_steps as probe
from maxdiffusion.diagnostics_exp03 import d1_per_frame_slopes as d1
from maxdiffusion.diagnostics_exp03 import sigma_trajectory_trace as trace

_REPO = Path(gen.__file__).parents[2]
_MANIFEST = _REPO / "docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json"


# =============================================================================================
# Mechanism A — per-frame slopes.
# =============================================================================================


def _curve(intercept: float, slope: float, n_frames: int = 33, frame0: float = 0.97) -> list[float]:
    """``frame0`` then a straight line over frames 1..n-1 with a KNOWN slope."""
    values = [frame0] + [intercept + slope * i for i in range(1, n_frames)]
    return values


def test_the_slope_is_the_ols_slope_over_frames_one_to_thirty_two():
    values = _curve(0.95, -0.004)
    assert len(values) == 33  # frame 0 (pinned) + frames 1..32, the plan's window
    assert d1.window_slope(values) == pytest.approx(-0.004, abs=1e-12)
    # Frame 0 is the FREE pinned frame and must not enter the fit: a wild value cannot move it.
    values[0] = 0.10
    assert d1.window_slope(values) == pytest.approx(-0.004, abs=1e-12)
    assert d1.D1_FIRST_FRAME == 1


def test_the_slope_uses_the_frame_index_as_the_regressor():
    # A fit against 0..n-1 instead of first_frame..n-1 gives the same slope, but a fit against the
    # VALUES' order with a different spacing does not; pin the axis explicitly.
    slope = -0.01
    values = _curve(0.9, slope, n_frames=40)
    assert d1.window_slope(values, expected_frames=40) == pytest.approx(slope, abs=1e-12)
    flat = [0.5] * 40
    assert d1.window_slope(flat, expected_frames=40) == pytest.approx(0.0, abs=1e-12)
    rising = _curve(0.5, +0.002, n_frames=40)
    assert d1.window_slope(rising, expected_frames=40) > 0.0


def test_a_window_shorter_than_the_fit_window_is_refused():
    with pytest.raises(ValueError):
        d1.window_slope([0.9, 0.8])  # one usable point cannot define a slope
    with pytest.raises(ValueError):
        d1.window_slope([0.9, 0.8], expected_frames=2)


def test_the_reduction_formula_is_one_minus_the_slope_ratio():
    control = [-0.004, -0.006, -0.005]
    halved = [value / 2 for value in control]
    assert d1.slope_reduction(halved, control) == pytest.approx(0.5, abs=1e-12)
    assert d1.slope_reduction(control, control) == pytest.approx(0.0, abs=1e-12)
    # A trial that decays FASTER is a negative reduction, not an absolute-value "improvement".
    assert d1.slope_reduction([value * 2 for value in control], control) == pytest.approx(-1.0, abs=1e-12)
    # A flat control cannot be divided by: the statistic is undefined, not infinite.
    with pytest.raises(ValueError):
        d1.slope_reduction(halved, [0.0, 0.0, 0.0])


def test_the_bootstrap_is_paired_per_episode_and_seeded():
    rng = np.random.default_rng(7)
    control = [-0.004 - 0.001 * abs(value) for value in rng.normal(size=100)]
    # Per-episode jitter on top of a halved decay: without it every resample would return exactly
    # 0.5 and the interval could not be told apart from a constant.
    trial = [0.5 * value + 0.0002 * jitter for value, jitter in zip(control, rng.normal(size=100))]
    pairs = [
        {"episode_id": 100 + i, "trial_slope": trial[i], "control_slope": control[i]} for i in range(len(control))
    ]
    result = d1.paired_bootstrap_reduction(pairs)
    assert result["resamples"] == 10000 and result["seed"] == 0 and result["ci"] == 0.95
    assert result["reduction"] == pytest.approx(0.5, abs=0.05)
    assert result["ci_high"] - result["ci_low"] > 0.0  # a real interval, not a point
    assert result["ci_low"] <= result["reduction"] <= result["ci_high"]
    assert result["n_episodes"] == 100
    # Deterministic: the predeclared seed makes the interval reproducible, not merely stable-ish.
    again = d1.paired_bootstrap_reduction(pairs)
    assert (again["ci_low"], again["ci_high"]) == (result["ci_low"], result["ci_high"])
    # ...and it really is a resample of EPISODES: a different seed moves the interval.
    other = d1.paired_bootstrap_reduction(pairs, seed=1)
    assert (other["ci_low"], other["ci_high"]) != (result["ci_low"], result["ci_high"])


def test_the_bootstrap_pairs_within_an_episode_rather_than_across_arms():
    # Trial and control are paired by episode: shuffling the control list must change the answer,
    # otherwise the pairing is not being used at all.
    pairs = [{"episode_id": i, "trial_slope": -0.001 * (i + 1), "control_slope": -0.002 * (i + 1)} for i in range(40)]
    kept = d1.paired_bootstrap_reduction(pairs)
    shuffled = d1.paired_bootstrap_reduction(
        [
            {"episode_id": p["episode_id"], "trial_slope": p["trial_slope"], "control_slope": q["control_slope"]}
            for p, q in zip(pairs, list(reversed(pairs)))
        ]
    )
    assert kept["ci_low"] != shuffled["ci_low"] or kept["ci_high"] != shuffled["ci_high"]


def test_a_degenerate_cohort_gives_a_point_interval():
    pairs = [{"episode_id": i, "trial_slope": -0.002, "control_slope": -0.004} for i in range(30)]
    result = d1.paired_bootstrap_reduction(pairs)
    assert result["reduction"] == pytest.approx(0.5, abs=1e-12)
    assert result["ci_low"] == pytest.approx(0.5, abs=1e-12)
    assert result["ci_high"] == pytest.approx(0.5, abs=1e-12)


def test_the_predeclared_threshold_is_recorded_in_the_module():
    assert d1.D1_REDUCTION_THRESHOLD == 0.25
    assert (d1.D1_LAST_FRAME, d1.D1_FRAME_COUNT, d1.D1_COHORT_SIZE) == (32, 33, 100)
    assert d1.D1_BOOTSTRAP_RESAMPLES == 10000
    assert d1.D1_BOOTSTRAP_SEED == 0
    assert d1.D1_CI == 0.95


def test_the_exp02_self_validation_check_is_retained():
    # The frames come from lossy MP4s, so the script must keep proving that mean-over-frames tracks
    # the SSIM the eval recorded for the same window.
    per_window = {"ep100_v0_s00000": _curve(0.9, -0.002), "ep101_v0_s00000": _curve(0.8, -0.003)}
    assert all(len(values) == d1.D1_FRAME_COUNT for values in per_window.values())
    recorded = {name: float(np.mean(values)) for name, values in per_window.items()}
    report = d1.self_validation(per_window, recorded)
    assert report["n"] == 2
    assert report["mean_abs_diff"] == pytest.approx(0.0, abs=1e-12)
    assert report["max_abs_diff"] == pytest.approx(0.0, abs=1e-12)
    drifted = dict(recorded)
    drifted["ep100_v0_s00000"] += 0.05
    assert d1.self_validation(per_window, drifted)["max_abs_diff"] == pytest.approx(0.05, abs=1e-9)


def test_the_ssim_kernel_is_the_exp02_one():
    # Ported verbatim so the absolute levels stay comparable with the exp_02 D1 reading.
    rng = np.random.default_rng(0)
    a = rng.random((32, 40, 3)).astype(np.float32)
    assert d1.ssim_frame(a, a) == pytest.approx(1.0, abs=1e-6)
    b = np.clip(a + rng.normal(scale=0.1, size=a.shape), 0.0, 1.0).astype(np.float32)
    assert 0.0 < d1.ssim_frame(a, b) < 1.0


# =============================================================================================
# Mechanism B — the sigma-trajectory trace.
# =============================================================================================

_STEPS, _SHIFT = 25, 5.0
_B, _C, _F, _H, _W = 1, 4, 3, 4, 6


def _trace_inputs():
    z_gt = jnp.asarray(np.random.default_rng(1).normal(size=(_B, _C, _F, _H, _W)), dtype=jnp.float32)
    z_i0 = z_gt[:, :, :1]
    eps = jnp.asarray(np.random.default_rng(2).normal(size=(_B, _C, _F, _H, _W)), dtype=jnp.float32)
    context = jnp.zeros((_B, 7, 8), dtype=jnp.float32)
    sigmas, timesteps = trace.trace_grid(num_inference_steps=_STEPS, flow_shift=_SHIFT)
    return z_gt, z_i0, eps, context, sigmas, timesteps


def test_a_perfect_velocity_oracle_traces_identically_zero():
    z_gt, z_i0, eps, context, sigmas, timesteps = _trace_inputs()

    def oracle(hidden_states, timestep, encoder_hidden_states):
        del hidden_states, timestep, encoder_hidden_states
        return eps - z_gt

    errors = trace.sigma_trace(
        oracle, z_gt=z_gt, z_i0=z_i0, eps=eps, context=context, sigmas=sigmas, timesteps=timesteps
    )
    assert len(errors) == _STEPS + 1
    # Step 0 is exact by construction: the rollout STARTS at the interpolant (sigma_0 = 1 => eps).
    assert errors[0] == 0.0
    # The rest is exact algebra evaluated in float32, so "zero" is zero to rounding.
    assert max(errors) < 1e-10, errors


def test_a_wrong_velocity_traces_strictly_positive_and_growing_error():
    z_gt, z_i0, eps, context, sigmas, timesteps = _trace_inputs()

    def wrong(hidden_states, timestep, encoder_hidden_states):
        del timestep, encoder_hidden_states
        return jnp.zeros_like(hidden_states)  # predicts no motion at all

    errors = trace.sigma_trace(
        wrong, z_gt=z_gt, z_i0=z_i0, eps=eps, context=context, sigmas=sigmas, timesteps=timesteps
    )
    assert errors[0] == 0.0
    assert all(value > 0.0 for value in errors[1:])
    assert errors[-1] > errors[1]  # a stuck sampler diverges further as sigma falls


def test_the_trace_is_taken_against_the_pinned_ideal_interpolant():
    z_gt, z_i0, eps, _, sigmas, _ = _trace_inputs()
    sigma = float(sigmas[3])
    reference = trace.ideal_interpolant(z_gt, eps, sigma, z_i0)
    assert np.allclose(np.asarray(reference[:, :, :1]), np.asarray(z_i0))  # frame 0 pinned, as in the rollout
    body = np.asarray(reference[:, :, 1:])
    expected = (1.0 - sigma) * np.asarray(z_gt[:, :, 1:]) + sigma * np.asarray(eps[:, :, 1:])
    assert np.allclose(body, expected, atol=1e-6)
    assert trace.interpolant_error(reference, z_gt, eps, sigma, z_i0) == 0.0


def test_the_trace_rows_carry_their_sigma_and_index():
    z_gt, z_i0, eps, context, sigmas, timesteps = _trace_inputs()

    def zero(hidden_states, timestep, encoder_hidden_states):
        del timestep, encoder_hidden_states
        return jnp.zeros_like(hidden_states)

    rows = trace.trace_rows(zero, z_gt=z_gt, z_i0=z_i0, eps=eps, context=context, sigmas=sigmas, timesteps=timesteps)
    assert [row["index"] for row in rows] == list(range(_STEPS + 1))
    assert rows[0]["sigma"] == pytest.approx(1.0)
    assert rows[-1]["sigma"] == 0.0
    # The finite-precision floor travels in the SAME entry as the error it bounds, so the two can
    # never be misaligned by an index.
    assert all(set(row) == {"index", "sigma", "error", "floor"} for row in rows)


def test_the_trace_uses_the_extracted_sampler_step():
    # The whole point of Mechanism B is that it traces the operator the eval runs.
    source = Path(trace.__file__).read_text()
    called = {
        node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
    }
    assert "overfit100_sampler_step" in called
    assert "overfit100_sampler_grid" in called
    assert "sigmas[i + 1]" not in source  # no private copy of the update


def test_the_epsilon_is_the_evals_own_per_window_key():
    import jax

    key = trace.trace_noise_key(episode_id=137, window_start=42)
    expected_key = gen.window_fold_key(0, 137, 42)
    assert np.array_equal(np.asarray(jax.random.key_data(key)), np.asarray(jax.random.key_data(expected_key)))
    assert trace.TRACE_SEED == 0
    noise = trace.trace_noise(episode_id=137, window_start=42, shape=(1, 2, 3, 4, 5), dtype=jnp.float32)
    assert noise.shape == (1, 2, 3, 4, 5)
    # ...and it is the same draw the rollout starts from, so the trace begins at zero error.
    assert np.array_equal(np.asarray(noise), np.asarray(jax.random.normal(expected_key, (1, 2, 3, 4, 5), jnp.float32)))
    # A different window is a different epsilon (the key really folds in the identity).
    other = trace.trace_noise(episode_id=138, window_start=42, shape=(1, 2, 3, 4, 5), dtype=jnp.float32)
    assert not np.array_equal(np.asarray(noise), np.asarray(other))


def test_the_cohort_is_the_exp02_probes_thirty_window_selection():
    cohort = trace.trace_cohort(str(_MANIFEST), seed=0, num_windows=30)
    reference = probe.probe_cohort(str(_MANIFEST), seed=0, num_windows=30)
    assert [w["name"] for w in cohort] == [w["name"] for w in reference]
    assert len(cohort) == 30
    assert trace.TRACE_NUM_WINDOWS == 30
    # Reuse, not re-implementation.
    assert "probe_cohort" in Path(trace.__file__).read_text()


def test_the_output_path_is_canonical_and_refuses_verdict_directories():
    config = SimpleNamespace(output_dir="gs://bucket/out", run_name="exp03-A")
    path = trace.trace_output_path(config, 12500)
    assert path == "gs://bucket/out/exp03-A/validation_probe_sampling/sigma_trace_ckpt12500.json"
    assert trace.TRACE_OUTPUT_DIR == probe.PROBE_OUTPUT_DIR


@pytest.mark.parametrize(
    "output_dir,run_name",
    [
        ("gs://bucket/out", "step_012500_s3_intermediate"),
        ("gs://bucket/out/step_012500_s3_intermediate", "exp03-A"),
        ("gs://bucket/out", "ok/../step_012500_s3_full_set"),
    ],
)
def test_a_hostile_path_cannot_steer_the_trace_into_the_evidence_tree(output_dir, run_name):
    with pytest.raises(ValueError) as excinfo:
        trace.trace_output_path(SimpleNamespace(output_dir=output_dir, run_name=run_name), 12500)
    assert "step_" in str(excinfo.value)


def test_the_artifact_is_diagnostic_and_written_immutably(tmp_path):
    payload = trace.trace_artifact(
        SimpleNamespace(
            output_dir=str(tmp_path),
            run_name="exp03-A",
            checkpoint_dir="gs://x/ck",
            eval_data_dir="gs://x/train100",
            model_manifest_path=str(_MANIFEST),
            seed=0,
        ),
        checkpoint_step=12500,
        cohort=[{"name": "ep100_v0_s00000", "episode_id": 100, "episode_index": 0, "window_start": 0}],
        rows=[
            {
                "name": "ep100_v0_s00000",
                "episode_id": 100,
                "episode_index": 0,
                "window_start": 0,
                "trace": [{"index": 0, "sigma": 1.0, "error": 0.0}, {"index": 1, "sigma": 0.9, "error": 0.1}],
            }
        ],
    )
    assert payload["schema"] == trace.TRACE_SCHEMA
    assert payload["kind"] == "diagnostic"
    for forbidden in ("eval_pass_role", "canonical_cohort", "role_validation", "run_signature"):
        assert forbidden not in payload
    assert payload["mean_trace"][0]["error"] == 0.0

    path = str(tmp_path / "sigma.json")
    trace.write_trace_artifact(path, payload)
    trace.write_trace_artifact(path, payload)  # identical rewrite tolerated
    with pytest.raises(ValueError):
        trace.write_trace_artifact(path, {**payload, "checkpoint_step": 999})
    assert json.loads(Path(path).read_text())["schema"] == trace.TRACE_SCHEMA


def test_the_trace_touches_no_verdict_machinery():
    source = Path(trace.__file__).read_text()
    for forbidden in (
        "overfit100_aggregation_artifact",
        "assert_pass_role_plan",
        "validate_artifact_role",
        "write_staged_row",
        "overfit100_publication_state",
        "eval_pass_role",
    ):
        assert forbidden not in source, forbidden


# ---------------------------------------------------------------------------------------------
# Round-1 strengthening: fail-closed D1 contracts, the trace's approval pin, the bf16 floor.
# ---------------------------------------------------------------------------------------------


def _write_video_dirs(root, names, *, missing=None):
    """Directory skeletons; ``decode`` is stubbed, so the files only need to exist."""
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        directory = root / name
        directory.mkdir()
        for filename in ("ground_truth.mp4", "sample.mp4"):
            if missing and name in missing and filename in missing[name]:
                continue
            (directory / filename).write_bytes(b"")


def _stub_decode(monkeypatch, *, frames=None):
    """Synthetic videos whose SSIM really decays: the sample drifts from the ground truth."""
    count = int(frames or d1.D1_FRAME_COUNT)

    def decode(path):
        path = Path(path)
        rng = np.random.default_rng(zlib.crc32(path.parent.name.encode()))
        base = rng.random((count, 12, 12, 3)).astype(np.float32)
        if path.name == "ground_truth.mp4":
            return base
        # The trial drifts half as fast as the control, so the reduction is a real 0.5-ish number.
        factor = 0.01 if "trial" in path.parent.parent.name else 0.02
        drift = (np.arange(count, dtype=np.float32) * factor)[:, None, None, None]
        return np.clip(base + drift * rng.normal(size=base.shape).astype(np.float32), 0.0, 1.0)

    monkeypatch.setattr(d1, "decode", decode)


def test_unequal_videos_are_refused_rather_than_truncated():
    left = np.zeros((33, 8, 8, 3), dtype=np.float32)
    right = np.zeros((30, 8, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError) as excinfo:
        d1.per_frame_ssim(left, right)
    assert "length" in str(excinfo.value)
    # ...and the exact 33-frame window is required even when the two agree.
    with pytest.raises(ValueError):
        d1.per_frame_ssim(right, right)


def test_a_window_missing_a_video_is_a_hard_failure(tmp_path, monkeypatch):
    _stub_decode(monkeypatch)
    root = tmp_path / "videos"
    _write_video_dirs(root, [f"ep{100 + i}_v0_s00000" for i in range(3)], missing={"ep101_v0_s00000": {"sample.mp4"}})
    with pytest.raises(ValueError) as excinfo:
        d1.per_window_slopes(root, expected_windows=3)
    assert "sample.mp4" in str(excinfo.value)


def test_a_short_cohort_is_a_hard_failure(tmp_path, monkeypatch):
    _stub_decode(monkeypatch)
    root = tmp_path / "videos"
    _write_video_dirs(root, [f"ep{100 + i}_v0_s00000" for i in range(3)])
    with pytest.raises(ValueError) as excinfo:
        d1.per_window_slopes(root)  # the production default is the 100-window cohort
    assert str(d1.D1_COHORT_SIZE) in str(excinfo.value)
    assert len(d1.per_window_slopes(root, expected_windows=3)) == 3


def _slope_table(names, *, slope=-0.004, episode_offset=0):
    return {
        name: {
            "per_frame": _curve(0.95, slope),
            "slope": slope,
            "episode_id": int(name[2:].split("_", 1)[0]) + episode_offset,
        }
        for name in names
    }


def test_the_two_arms_must_present_the_identical_cohort():
    names = [f"ep{100 + i}_v0_s00000" for i in range(d1.D1_COHORT_SIZE)]
    trial = _slope_table(names)
    assert d1.assert_paired_cohorts(trial, _slope_table(names)) == sorted(names)
    # A window only one arm rendered is a hard failure, NOT an intersection.
    short = _slope_table(names[:-1])
    with pytest.raises(ValueError) as excinfo:
        d1.assert_paired_cohorts(trial, short)
    assert "only in the trial" in str(excinfo.value)
    with pytest.raises(ValueError):
        d1.assert_paired_cohorts(short, trial)
    # Right size, wrong composition.
    swapped = _slope_table(names[:-1] + ["ep999_v0_s00000"])
    with pytest.raises(ValueError):
        d1.assert_paired_cohorts(trial, swapped)


def test_the_paired_cohort_must_be_the_full_hundred_with_unique_episodes():
    names = [f"ep{100 + i}_v0_s00000" for i in range(30)]
    with pytest.raises(ValueError) as excinfo:
        d1.assert_paired_cohorts(_slope_table(names), _slope_table(names))
    assert str(d1.D1_COHORT_SIZE) in str(excinfo.value)

    full = [f"ep{100 + i}_v0_s00000" for i in range(d1.D1_COHORT_SIZE)]
    duplicated = _slope_table(full)
    duplicated[full[1]]["episode_id"] = duplicated[full[0]]["episode_id"]
    with pytest.raises(ValueError) as excinfo:
        d1.assert_paired_cohorts(duplicated, _slope_table(full))
    assert "episode ids repeat" in str(excinfo.value)
    # ...and the arms must agree about which episode a window belongs to.
    with pytest.raises(ValueError):
        d1.assert_paired_cohorts(_slope_table(full), _slope_table(full, episode_offset=1))


def test_the_self_validation_inputs_are_part_of_the_run_contract():
    signature = inspect.signature(d1.compare)
    assert list(signature.parameters) == [
        "trial_root",
        "control_root",
        "trial_aggregation",
        "control_aggregation",
    ]
    assert all(parameter.default is inspect.Parameter.empty for parameter in signature.parameters.values())


def test_compare_runs_the_full_contract_end_to_end(tmp_path, monkeypatch):
    names = [f"ep{100 + i}_v0_s00000" for i in range(d1.D1_COHORT_SIZE)]  # the real 100-window cohort
    _stub_decode(monkeypatch)
    trial_root, control_root = tmp_path / "trial", tmp_path / "control"
    _write_video_dirs(trial_root, names)
    _write_video_dirs(control_root, names)

    def _aggregation(path, offset):
        payload = {
            "rows": [{"name": name, "seed": 0, "context_mode": "correct", "ssim": 0.75 + offset} for name in names]
        }
        path.write_text(json.dumps(payload))
        return path

    trial_json = _aggregation(tmp_path / "trial.json", 0.0)
    control_json = _aggregation(tmp_path / "control.json", 0.0)
    result = d1.compare(trial_root, control_root, trial_json, control_json)
    assert result["n_windows"] == len(names)
    assert set(result["self_validation"]) == {"trial", "control"}
    assert result["self_validation"]["trial"]["n"] == len(names)
    assert result["bootstrap"]["n_episodes"] == len(names)

    # A window with no seed-0 correct row in the aggregation is a failure, not a quiet skip.
    thin = json.loads(trial_json.read_text())
    thin["rows"] = thin["rows"][:-1]
    (tmp_path / "thin.json").write_text(json.dumps(thin))
    with pytest.raises(ValueError) as excinfo:
        d1.compare(trial_root, control_root, tmp_path / "thin.json", control_json)
    assert "self-validation" in str(excinfo.value)


def test_the_trace_design_is_approval_pinned():
    assert trace.TRACE_SAMPLING_STEPS == 25
    trace.assert_approved_trace_design(seed=0, num_windows=30, sampling_steps=25)


@pytest.mark.parametrize(
    "seed,num_windows,sampling_steps,needle",
    [
        (1, 30, 25, "seed"),
        (0, 15, 25, "window"),
        (0, 100, 25, "window"),
        (0, 30, 50, "step"),
        (0, 30, 25 - 1, "step"),
    ],
)
def test_a_hostile_trace_config_is_refused(seed, num_windows, sampling_steps, needle):
    with pytest.raises(ValueError) as excinfo:
        trace.assert_approved_trace_design(seed=seed, num_windows=num_windows, sampling_steps=sampling_steps)
    assert needle in str(excinfo.value).lower()


def test_the_driver_pins_the_design_before_the_model_loads(monkeypatch):
    # The refusal must happen BEFORE _restore_overfit100_validation_state: a ~5B load that then
    # throws has already burned the slot.
    loaded = []
    monkeypatch.setattr(
        gen,
        "_restore_overfit100_validation_state",
        lambda config: loaded.append(config) or (_ for _ in ()).throw(AssertionError("must not be reached")),
    )
    config = SimpleNamespace(
        model_manifest_path=str(_MANIFEST),
        seed=1,  # hostile
        probe_num_windows=30,
        side_adapter_sampling_steps=25,
    )
    with pytest.raises(ValueError):
        trace.run_trace(config)
    assert loaded == []


def test_the_oracle_floor_is_measured_at_the_production_dtype():
    # fp32: the floor is numerically zero, so "growth is model error and nothing else" holds.
    z_gt, z_i0, eps, context, sigmas, timesteps = _trace_inputs()
    fp32_floor = trace.oracle_floor(z_gt=z_gt, z_i0=z_i0, eps=eps, context=context, sigmas=sigmas, timesteps=timesteps)
    assert fp32_floor[0] == 0.0 and max(fp32_floor) < 1e-10

    # bfloat16 (the production weights_dtype): the incremental Euler accumulation does NOT
    # reproduce the interpolant, so the floor is real and must be reported, not assumed away.
    bf16 = jnp.bfloat16
    z_gt_b, z_i0_b, eps_b = z_gt.astype(bf16), z_i0.astype(bf16), eps.astype(bf16)
    bf16_floor = trace.oracle_floor(
        z_gt=z_gt_b, z_i0=z_i0_b, eps=eps_b, context=context.astype(bf16), sigmas=sigmas, timesteps=timesteps
    )
    assert bf16_floor[0] == 0.0  # the start is still exact
    assert max(bf16_floor) > max(fp32_floor)
    assert len(bf16_floor) == len(fp32_floor) == _STEPS + 1


def test_the_oracle_at_production_dtype_still_traces_far_below_any_signal():
    # The floor must be small enough that Mechanism B can read a model effect through it.
    bf16 = jnp.bfloat16
    z_gt, z_i0, eps, context, sigmas, timesteps = _trace_inputs()
    z_gt, z_i0, eps, context = (value.astype(bf16) for value in (z_gt, z_i0, eps, context))

    def oracle(hidden_states, timestep, encoder_hidden_states):
        del timestep, encoder_hidden_states
        return (eps - z_gt).astype(hidden_states.dtype)

    errors = trace.sigma_trace(
        oracle, z_gt=z_gt, z_i0=z_i0, eps=eps, context=context, sigmas=sigmas, timesteps=timesteps
    )
    stuck = trace.sigma_trace(
        lambda hidden_states, timestep, encoder_hidden_states: jnp.zeros_like(hidden_states),
        z_gt=z_gt,
        z_i0=z_i0,
        eps=eps,
        context=context,
        sigmas=sigmas,
        timesteps=timesteps,
    )
    assert max(errors) < 1e-3
    assert max(errors) < 0.01 * max(stuck)  # the floor is two orders below a wrong model's reading


def test_the_reported_metric_is_raw_error_with_the_floor_alongside():
    doc = trace.__doc__ or ""
    assert "RAW" in doc and "floor" in doc
    assert str(jnp.dtype(trace.TRACE_REFERENCE_DTYPE)) == "float32"
    payload = trace.trace_artifact(
        SimpleNamespace(
            output_dir="/tmp/x",
            run_name="r",
            checkpoint_dir="gs://x/ck",
            eval_data_dir="gs://x/train100",
            model_manifest_path=str(_MANIFEST),
            seed=0,
            weights_dtype="bfloat16",
        ),
        checkpoint_step=12500,
        cohort=[{"name": "ep100_v0_s00000"}],
        rows=[
            {
                "name": "ep100_v0_s00000",
                "trace": [
                    {"index": 0, "sigma": 1.0, "error": 0.0, "floor": 0.0},
                    {"index": 1, "sigma": 0.9, "error": 0.2, "floor": 0.001},
                ],
            }
        ],
    )
    assert payload["reference_dtype"] == "float32"
    assert payload["latent_dtype"] == "bfloat16"
    assert "RAW" in payload["reported_metric"]
    assert payload["sampling_steps"] == trace.TRACE_SAMPLING_STEPS
    assert payload["mean_trace"][1]["floor"] == pytest.approx(0.001)
