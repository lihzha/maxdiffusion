"""CPU-only tests for the exp_02 encoder-validation gates V1-V4 (cycle B, deliverable B1).

Plan v4 D4 fixes four gates with FINAL thresholds, and R1 makes any failure fatal ("stop,
reconcile VAE config; no training"). The gate *logic* is therefore as load-bearing as the
encode itself: a gate that silently passes a broken encode would let exp_02 train on a
corrupt dataset and blame the model.

Each gate is exercised on synthetic fixtures in both directions -- a passing case and the
specific failure mode the gate exists to catch (V1: a missed normalization showing up as a
16x scale error; V2: a collapsed/blown-up latent std; V3: a decode that no longer resembles
the source frames; V4: a frame 0 that depends on future frames, i.e. leakage into `z_i0`).
No accelerator, no weights, no network.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from maxdiffusion.data_preprocessing import build_overfit100_dataset as builder
from maxdiffusion.data_preprocessing import build_overfit100_manifest as manifest_builder
from maxdiffusion.data_preprocessing.build_overfit100_dataset import (
    CYCLE_B_IMPLEMENTATION_PATHS,
    V1_PEARSON_MIN,
    V1_REL_L2_MAX,
    V2_ABS_MEAN_MAX,
    V2_STD_MAX,
    V2_STD_MIN,
    V3_EPISODE_INDICES,
    V3_SSIM_MIN,
    V4_RTOL,
    BuildError,
    GateFailure,
    assert_manifest_matches_committed,
    check_v1,
    check_v2,
    check_v3,
    check_v4,
    frames_ssim,
    gate_failures,
    main,
    implementation_provenance_errors,
    pearson_r,
    preflight,
    raise_on_gate_failures,
    relative_l2,
    summarize_v2,
)

_VAE_PIN = {
    "hf_repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "revision": "b8fff7315c768468a5333511427288870b2e9635",
    "vae_config_sha256": "d996c340fe9a7df5d7371f76a7d8d6956f6c98256080074d8434fa5eeac11360",
}


def _latents(seed=0, shape=(48, 9, 12, 20), mean=0.0, std=0.65):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=shape) * std + mean).astype(np.float16)


# ----------------------------------------------------------------------------------
# 0. The thresholds themselves are part of the contract (plan D4: "all thresholds final").
# ----------------------------------------------------------------------------------


def test_gate_thresholds_are_the_plan_values():
    assert (V1_REL_L2_MAX, V1_PEARSON_MIN) == (0.25, 0.97)
    assert (V2_STD_MIN, V2_STD_MAX, V2_ABS_MEAN_MAX) == (0.35, 0.95, 0.15)
    assert V3_SSIM_MIN == 0.80
    assert V4_RTOL == 1e-3
    assert tuple(V3_EPISODE_INDICES) == (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)


# ----------------------------------------------------------------------------------
# 1. Metric primitives.
# ----------------------------------------------------------------------------------


def test_relative_l2_is_zero_for_an_exact_match():
    z = _latents()
    assert relative_l2(z, z) == pytest.approx(0.0, abs=1e-12)


def test_relative_l2_of_a_16x_scale_error_is_fifteen():
    z = _latents().astype(np.float32)
    assert relative_l2(z * 16.0, z) == pytest.approx(15.0, rel=1e-5)


def test_relative_l2_raises_on_a_zero_reference():
    with pytest.raises(BuildError):
        relative_l2(np.ones((4,), np.float32), np.zeros((4,), np.float32))


def test_pearson_r_extremes():
    a = np.linspace(-1.0, 1.0, 64).astype(np.float32)
    assert pearson_r(a, a) == pytest.approx(1.0, rel=1e-6)
    assert pearson_r(-a, a) == pytest.approx(-1.0, rel=1e-6)
    assert pearson_r(a * 16.0 + 3.0, a) == pytest.approx(1.0, rel=1e-6)  # invariant to affine scaling


def test_pearson_r_raises_on_a_constant_input():
    with pytest.raises(BuildError):
        pearson_r(np.ones((16,), np.float32), np.linspace(0, 1, 16, dtype=np.float32))


# ----------------------------------------------------------------------------------
# 2. V1 -- round-trip against the cached reference windows.
# ----------------------------------------------------------------------------------


def test_v1_passes_on_a_realistic_round_trip():
    cached = _latents(1).astype(np.float32)
    rng = np.random.default_rng(2)
    reencoded = cached + rng.normal(size=cached.shape).astype(np.float32) * 0.10  # ~15% rel-L2
    result = check_v1("ep0_v0_s00000", cached, reencoded)
    assert result["passed"] is True
    assert result["name"] == "ep0_v0_s00000"
    assert result["rel_l2"] <= V1_REL_L2_MAX and result["pearson"] >= V1_PEARSON_MIN


def test_v1_catches_a_missing_normalization_sixteen_x_scale_error():
    cached = _latents(3).astype(np.float32)
    result = check_v1("ep0_v0_s00004", cached * 16.0, cached)
    # Perfectly correlated, so ONLY the rel-L2 half of the gate can catch it.
    assert result["pearson"] == pytest.approx(1.0, rel=1e-5)
    assert result["rel_l2"] > V1_REL_L2_MAX
    assert result["passed"] is False


def test_v1_catches_a_decorrelated_reencode():
    cached = _latents(4).astype(np.float32)
    result = check_v1("ep0_v0_s00008", _latents(5).astype(np.float32), cached)
    assert result["pearson"] < V1_PEARSON_MIN
    assert result["passed"] is False


def test_v1_needs_both_halves_at_the_boundary():
    cached = _latents(6).astype(np.float32)
    assert check_v1("w", cached * (1.0 + V1_REL_L2_MAX), cached)["passed"] is True  # exactly at 0.25
    assert check_v1("w", cached * (1.0 + V1_REL_L2_MAX + 1e-3), cached)["passed"] is False


def test_v1_rejects_a_shape_mismatch():
    with pytest.raises(BuildError):
        check_v1("w", np.zeros((48, 9, 12, 20), np.float32), np.zeros((48, 5, 12, 20), np.float32))


# ----------------------------------------------------------------------------------
# 3. V2 -- per-window statistics envelope (runs on EVERY built window).
# ----------------------------------------------------------------------------------


def test_v2_passes_on_cache_like_statistics():
    result = check_v2("ep1_v0_s00000", _latents(7, std=0.65))
    assert result["passed"] is True
    assert result["std"] == pytest.approx(0.65, abs=0.02)
    assert abs(result["mean"]) < V2_ABS_MEAN_MAX


def test_v2_catches_a_collapsed_std():
    result = check_v2("w", _latents(8, std=0.2))
    assert result["passed"] is False and result["std"] < V2_STD_MIN


def test_v2_catches_a_blown_up_std():
    result = check_v2("w", _latents(9, std=1.4))
    assert result["passed"] is False and result["std"] > V2_STD_MAX


def test_v2_catches_a_biased_mean():
    result = check_v2("w", _latents(10, mean=0.4))
    assert result["passed"] is False and abs(result["mean"]) > V2_ABS_MEAN_MAX


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_v2_catches_non_finite_values(bad):
    z = _latents(11)
    z[0, 0, 0, 0] = bad
    result = check_v2("w", z)
    assert result["finite"] is False and result["passed"] is False


def test_summarize_v2_reports_the_envelope_and_every_failure():
    good = [check_v2(f"g{i}", _latents(20 + i)) for i in range(4)]
    bad = check_v2("bad", _latents(30, std=0.2))
    summary = summarize_v2(good + [bad])
    assert summary["n_windows"] == 5
    assert [f["name"] for f in summary["failures"]] == ["bad"]
    assert summary["std_min"] == pytest.approx(min(s["std"] for s in good + [bad]))
    assert summary["std_max"] == pytest.approx(max(s["std"] for s in good + [bad]))
    assert summary["abs_mean_max"] == pytest.approx(max(abs(s["mean"]) for s in good + [bad]))


# ----------------------------------------------------------------------------------
# 4. V3 -- SSIM(decode(z_video), source frames), which doubles as the VAE ceiling.
# ----------------------------------------------------------------------------------


def test_v3_pass_fail_and_boundary():
    assert check_v3("w", 0.85)["passed"] is True
    assert check_v3("w", V3_SSIM_MIN)["passed"] is True  # >= is inclusive
    assert check_v3("w", 0.79)["passed"] is False
    assert check_v3("w", float("nan"))["passed"] is False


def test_v3_records_the_value_for_the_vae_ceiling():
    assert check_v3("ep0_v0_s00000", 0.9123)["ssim"] == pytest.approx(0.9123)


def test_frames_ssim_averages_over_frames_with_the_eval_parity_kwargs():
    calls = []

    def fake_ssim(target, pred, **kwargs):
        calls.append((target, pred, kwargs))
        return 0.5 + 0.1 * len(calls)

    pred = np.zeros((3, 8, 9, 3), dtype=np.float32)
    target = np.ones((3, 8, 9, 3), dtype=np.float32)
    value = frames_ssim(pred, target, ssim_fn=fake_ssim)

    assert value == pytest.approx(np.mean([0.6, 0.7, 0.8]))
    assert len(calls) == 3
    for target_arg, pred_arg, kwargs in calls:
        # exp_01 eval parity (`generate_wan_side_adapter._frame_ssim`): reference first.
        assert kwargs["data_range"] == 1.0 and kwargs["channel_axis"] == -1 and kwargs["win_size"] == 7
        assert target_arg.shape == (8, 9, 3) and pred_arg.shape == (8, 9, 3)


def test_frames_ssim_rejects_a_shape_mismatch():
    with pytest.raises(BuildError):
        frames_ssim(
            np.zeros((3, 8, 9, 3), np.float32), np.zeros((2, 8, 9, 3), np.float32), ssim_fn=lambda *a, **k: 1.0
        )


def test_structural_similarity_loader_fails_loudly_instead_of_returning_nan(monkeypatch):
    # exp_01's helper returns NaN when scikit-image is missing; for a BUILD GATE that is
    # indistinguishable from a silent pass, so the loader must raise instead.
    monkeypatch.setitem(sys.modules, "skimage.metrics", None)
    with pytest.raises(BuildError):
        builder.structural_similarity_fn()


# ----------------------------------------------------------------------------------
# 5. V4 -- frame-0 future-invariance (no leakage into z_i0 through the causal VAE).
# ----------------------------------------------------------------------------------


def test_v4_passes_when_frame_zero_is_identical():
    frame0 = _latents(12, shape=(48, 1, 12, 20)).astype(np.float32)
    result = check_v4(frame0, frame0.copy())
    assert result["passed"] is True
    assert result["max_abs_diff"] == 0.0


def test_v4_tolerates_numerics_below_the_rtol():
    frame0 = _latents(13, shape=(48, 1, 12, 20)).astype(np.float32) + 1.0
    perturbed = frame0 * (1.0 + 0.1 * V4_RTOL)
    result = check_v4(frame0, perturbed)
    assert result["passed"] is True
    assert 0.0 < result["max_rel_diff"] < V4_RTOL


def test_v4_catches_an_injected_future_dependent_frame_zero():
    frame0 = _latents(14, shape=(48, 1, 12, 20)).astype(np.float32) + 2.0
    leaked = frame0.copy()
    leaked[5, 0, 3, 7] *= 1.05  # 5% of one element: a real leak, far above float noise
    result = check_v4(frame0, leaked)
    assert result["passed"] is False
    assert result["n_violations"] == 1
    assert result["max_rel_diff"] > V4_RTOL


def test_v4_rejects_a_shape_mismatch():
    with pytest.raises(BuildError):
        check_v4(np.zeros((48, 1, 12, 20), np.float32), np.zeros((48, 1, 12, 21), np.float32))


# ----------------------------------------------------------------------------------
# 6. Aggregation -- any failure aborts the build, and a gate that did not run is a failure.
# ----------------------------------------------------------------------------------


def _passing_report():
    return {
        "v1": [
            check_v1(name, _latents(40 + i).astype(np.float32), _latents(40 + i).astype(np.float32))
            for i, name in enumerate(("a", "b", "c"))
        ],
        "v2": summarize_v2([check_v2("w", _latents(50))]),
        "v3": [check_v3(f"w{i}", 0.9) for i in range(10)],
        "v4": check_v4(np.ones((2, 1, 2, 2), np.float32), np.ones((2, 1, 2, 2), np.float32)),
    }


def test_gate_failures_is_empty_on_a_clean_report():
    assert gate_failures(_passing_report()) == []
    raise_on_gate_failures(_passing_report())  # must not raise


def test_gate_failures_names_every_failing_gate():
    report = _passing_report()
    report["v1"][1]["passed"] = False
    report["v2"]["failures"].append(check_v2("badwin", _latents(51, std=0.2)))
    report["v3"][3]["passed"] = False
    report["v4"]["passed"] = False
    messages = gate_failures(report)
    assert len(messages) == 4
    assert any(m.startswith("V1") for m in messages)
    assert any(m.startswith("V2") and "badwin" in m for m in messages)
    assert any(m.startswith("V3") for m in messages)
    assert any(m.startswith("V4") for m in messages)


def test_gate_failures_treats_a_gate_that_did_not_run_as_a_failure():
    report = _passing_report()
    report["v3"] = []
    assert any("V3" in m for m in gate_failures(report))
    del report["v4"]
    assert any("V4" in m for m in gate_failures(report))


def test_raise_on_gate_failures_raises_gate_failure_listing_them():
    report = _passing_report()
    report["v4"]["passed"] = False
    with pytest.raises(GateFailure) as excinfo:
        raise_on_gate_failures(report)
    assert "V4" in str(excinfo.value)


def test_main_returns_nonzero_when_a_gate_fails(monkeypatch):
    def boom(_args):
        raise GateFailure("V2 ep1_v0_s00000: std 0.20 outside [0.35, 0.95]")

    monkeypatch.setattr(builder, "run", boom)
    assert main(["--manifest", "m.json", "--out-root", "gs://bucket/x"]) != 0


def test_main_returns_zero_when_the_build_succeeds(monkeypatch):
    monkeypatch.setattr(builder, "run", lambda _args: 0)
    assert main(["--manifest", "m.json", "--out-root", "gs://bucket/x"]) == 0


# ----------------------------------------------------------------------------------
# 7. Production provenance -- a dataset may only be built from committed code (cycle A A1).
# ----------------------------------------------------------------------------------


def test_cycle_b_implementation_paths_exist():
    root = Path(__file__).resolve().parents[4]
    missing = [path for path in CYCLE_B_IMPLEMENTATION_PATHS if not (root / path).exists()]
    assert missing == []


def test_cycle_b_implementation_paths_cover_builder_tests_and_launcher():
    assert "src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py" in CYCLE_B_IMPLEMENTATION_PATHS
    assert "bash_scripts/build_overfit100_dataset.sh" in CYCLE_B_IMPLEMENTATION_PATHS
    assert sum(path.endswith("_test.py") or "/tests/" in path for path in CYCLE_B_IMPLEMENTATION_PATHS) == 5


def test_provenance_is_clean_when_every_cycle_b_file_is_committed():
    paths = CYCLE_B_IMPLEMENTATION_PATHS
    assert implementation_provenance_errors(set(paths), [], paths=paths) == []


def test_provenance_rejects_an_uncommitted_builder():
    paths = CYCLE_B_IMPLEMENTATION_PATHS
    errors = implementation_provenance_errors(set(paths), [paths[0]], paths=paths)
    assert len(errors) == 1 and paths[0] in errors[0]


def test_assert_implementation_committed_checks_the_paths_it_is_given(monkeypatch):
    calls = []

    def fake_git(_root, *args):
        calls.append(args)
        if args[0] == "ls-tree":
            return "\n".join(CYCLE_B_IMPLEMENTATION_PATHS) + "\n"
        if args[0] == "status":
            return ""
        return "a" * 40 + "\n"

    monkeypatch.setattr(manifest_builder, "_git", fake_git)
    sha = manifest_builder.assert_implementation_committed(paths=CYCLE_B_IMPLEMENTATION_PATHS)
    assert sha == "a" * 40
    status_call = next(args for args in calls if args[0] == "status")
    assert set(CYCLE_B_IMPLEMENTATION_PATHS) <= set(status_call)


def test_assert_implementation_committed_refuses_a_dirty_cycle_b_tree(monkeypatch):
    def fake_git(_root, *args):
        if args[0] == "ls-tree":
            return "\n".join(CYCLE_B_IMPLEMENTATION_PATHS) + "\n"
        if args[0] == "status":
            return f" M {CYCLE_B_IMPLEMENTATION_PATHS[0]}\n"
        return "b" * 40 + "\n"

    monkeypatch.setattr(manifest_builder, "_git", fake_git)
    with pytest.raises(manifest_builder.DirtyImplementationError):
        manifest_builder.assert_implementation_committed(paths=CYCLE_B_IMPLEMENTATION_PATHS)


# ----------------------------------------------------------------------------------
# 8. B2 -- V4 must never pass vacuously on non-finite output.
#
# `difference > tolerance` is FALSE for NaN, so without an explicit finiteness check a
# short-graph encode that produced NaNs would report "frame 0 is invariant". That is the
# worst possible failure: a silent pass on the gate that exists to prove z_i0 carries no
# future information.
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("side", ["full", "short"])
def test_v4_fails_on_non_finite_values_in_either_tensor(bad, side):
    frame0 = np.ones((4, 1, 3, 2), dtype=np.float32)
    other = frame0.copy()
    if side == "full":
        frame0 = frame0.copy()
        frame0[0, 0, 0, 0] = bad
        result = check_v4(frame0, other)
    else:
        other[0, 0, 0, 0] = bad
        result = check_v4(frame0, other)
    assert result["finite"] is False
    assert result["passed"] is False


def test_v4_reports_finite_on_a_clean_pair():
    frame0 = np.ones((4, 1, 3, 2), dtype=np.float32)
    assert check_v4(frame0, frame0.copy())["finite"] is True


def test_v4_failure_message_states_the_no_auto_retune_policy():
    frame0 = np.ones((2, 1, 2, 2), dtype=np.float32)
    leaked = frame0.copy()
    leaked[0, 0, 0, 0] = 2.0
    report = {
        "v1": [check_v1("a", np.arange(4.0), np.arange(4.0))],
        "v2": summarize_v2([check_v2("w", _latents(60))]),
        "v3": [check_v3("w", 0.9)],
        "v4": check_v4(frame0, leaked),
    }
    with pytest.raises(GateFailure) as excinfo:
        raise_on_gate_failures(report)
    message = str(excinfo.value)
    assert "V4" in message
    # The policy is part of the failure, not a knob: a trip aborts and the rerun is reviewed.
    assert "reviewed" in message.lower() and "rerun" in message.lower()


def test_v4_diagnostics_carry_difference_quantiles():
    rng = np.random.default_rng(0)
    full = rng.normal(size=(4, 1, 3, 2)).astype(np.float32) + 3.0
    short = full + rng.normal(size=full.shape).astype(np.float32) * 0.01
    diagnostics = builder.v4_diagnostics(full, short)
    quantiles = diagnostics["abs_diff_quantiles"]
    assert sorted(quantiles) == ["p100", "p50", "p90", "p99", "p999"]
    assert quantiles["p50"] <= quantiles["p90"] <= quantiles["p99"] <= quantiles["p100"]
    assert diagnostics["n_elements"] == full.size


def test_v4_failure_persists_both_frame_zero_tensors(tmp_path):
    full = np.ones((3, 1, 2, 2), dtype=np.float32)
    short = full.copy()
    short[0, 0, 0, 0] = 1.5
    path = builder.persist_v4_diagnostics(str(tmp_path), full, short, tmp_dir=tmp_path)
    assert Path(path).exists()
    with np.load(path) as data:
        np.testing.assert_array_equal(data["frame0_full"], full)
        np.testing.assert_array_equal(data["frame0_short"], short)
    written = json.loads((tmp_path / "v4_diagnostics.json").read_text())
    assert written["n_violations"] == 1


# ----------------------------------------------------------------------------------
# 9. B1 -- preflight binds the build to the pinned VAE, or aborts. No warn path.
# ----------------------------------------------------------------------------------


def _pinned_manifest(pin=None):
    return {
        "vae_fingerprint": dict(pin or _VAE_PIN),
        "fixture": {"uri": "gs://bucket/fixture.npz", "md5": "m", "size_bytes": 3, "names": ["a"]},
        "totals": {"episodes": 100, "windows": 1629},
    }


def _preflight_stubs(monkeypatch, resolved_pin=None, snapshot="/snap/b8fff73"):
    monkeypatch.setattr(builder, "verify_manifest", lambda *a, **k: [])
    monkeypatch.setattr(builder, "fetch_pinned", lambda uri, fingerprint, destination: destination)
    monkeypatch.setattr(builder, "verify_fixture", lambda *a, **k: [])
    monkeypatch.setattr(
        builder,
        "resolve_vae_snapshot",
        lambda repo, revision=None, local_files_only=False: {
            "snapshot_path": snapshot,
            "pin": dict(resolved_pin or _VAE_PIN),
        },
    )


def test_preflight_returns_the_snapshot_bound_to_the_pin(monkeypatch, tmp_path):
    _preflight_stubs(monkeypatch)
    result = preflight(_pinned_manifest(), tmp_path, log=lambda *_: None)
    assert result["vae_snapshot_path"] == "/snap/b8fff73"
    assert result["vae_fingerprint"] == _VAE_PIN


def test_preflight_aborts_when_the_manifest_carries_no_pin(monkeypatch, tmp_path):
    _preflight_stubs(monkeypatch)
    manifest = _pinned_manifest()
    manifest.pop("vae_fingerprint")
    with pytest.raises(BuildError, match="vae_fingerprint"):
        preflight(manifest, tmp_path, log=lambda *_: None)


@pytest.mark.parametrize("field", ["revision", "vae_config_sha256", "hf_repo"])
def test_preflight_aborts_when_the_resolved_vae_differs_from_the_pin(monkeypatch, tmp_path, field):
    observed = {**_VAE_PIN, field: "d" * len(_VAE_PIN[field])}
    _preflight_stubs(monkeypatch, resolved_pin=observed)
    with pytest.raises(BuildError, match="VAE"):
        preflight(_pinned_manifest(), tmp_path, log=lambda *_: None)


def test_preflight_aborts_on_a_malformed_pin(monkeypatch, tmp_path):
    _preflight_stubs(monkeypatch)
    with pytest.raises(BuildError):
        preflight(_pinned_manifest({"hf_repo": "x", "revision": "main"}), tmp_path, log=lambda *_: None)


def test_preflight_aborts_on_manifest_drift_before_touching_the_vae(monkeypatch, tmp_path):
    _preflight_stubs(monkeypatch)
    monkeypatch.setattr(builder, "verify_manifest", lambda *a, **k: ["gs://x: md5 drift"])
    monkeypatch.setattr(
        builder, "resolve_vae_snapshot", lambda *a, **k: pytest.fail("VAE resolved despite manifest drift")
    )
    with pytest.raises(BuildError, match="drift"):
        preflight(_pinned_manifest(), tmp_path, log=lambda *_: None)


# ----------------------------------------------------------------------------------
# 10. B6 -- the guard covers every code input, and the consumed manifest is content-hashed.
# ----------------------------------------------------------------------------------


def test_guard_covers_the_shared_manifest_builder_and_the_prefetch_helper():
    assert "src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py" in CYCLE_B_IMPLEMENTATION_PATHS
    assert "bash_scripts/prefetch_hf_snapshot.sh" in CYCLE_B_IMPLEMENTATION_PATHS
    assert "bash_scripts/build_overfit100_dataset.sh" in CYCLE_B_IMPLEMENTATION_PATHS


def test_assert_manifest_matches_committed_returns_the_sha_of_the_committed_file():
    committed = Path(__file__).resolve().parents[4] / builder.DEFAULT_MANIFEST
    digest = assert_manifest_matches_committed(committed)
    assert digest == hashlib.sha256(committed.read_bytes()).hexdigest()


def test_assert_manifest_matches_committed_accepts_a_byte_identical_copy(tmp_path):
    committed = Path(__file__).resolve().parents[4] / builder.DEFAULT_MANIFEST
    copied = tmp_path / "copy.json"
    copied.write_bytes(committed.read_bytes())
    assert assert_manifest_matches_committed(copied) == hashlib.sha256(committed.read_bytes()).hexdigest()


def test_assert_manifest_matches_committed_rejects_an_edited_manifest(tmp_path):
    committed = Path(__file__).resolve().parents[4] / builder.DEFAULT_MANIFEST
    edited = tmp_path / "edited.json"
    payload = json.loads(committed.read_text())
    payload["totals"]["windows"] = 1
    edited.write_text(json.dumps(payload, indent=2) + "\n")
    with pytest.raises(BuildError, match="sha256"):
        assert_manifest_matches_committed(edited)


# ----------------------------------------------------------------------------------
# 11. Deployed-code mode for the manifest check (probe failure 20260729-062523).
#
# The worker runs from an uploaded tarball, not a checkout. `assert_manifest_matches_committed`
# compares against the committed artifact, which on the worker is simply the copy SHIPPED IN
# THE TARBALL -- and that tarball IS the launch-time tree whose cleanliness the launcher
# verified. So in deployed-code mode the shipped copy is still compared when present (a
# hand-edited manifest passed by path is still caught) and otherwise hashed and recorded,
# with the reasoning logged. It must never crash and never silently skip.
# ----------------------------------------------------------------------------------


def test_manifest_check_records_the_shipped_hash_when_no_reference_is_shipped(tmp_path):
    consumed = tmp_path / "overfit100_manifest.json"
    consumed.write_bytes(b'{"episodes": []}\n')
    logs = []
    digest = assert_manifest_matches_committed(consumed, repo_root=tmp_path, log=logs.append)
    assert digest == hashlib.sha256(consumed.read_bytes()).hexdigest()
    message = " ".join(logs)
    assert "deployed-code" in message and "tarball" in message


def test_manifest_check_still_compares_against_a_shipped_reference(tmp_path):
    reference = tmp_path / builder.DEFAULT_MANIFEST
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_bytes(b'{"episodes": [1]}\n')
    other = tmp_path / "elsewhere.json"
    other.write_bytes(b'{"episodes": [2]}\n')
    # Same bytes -> accepted; different bytes -> refused, tarball or not.
    assert assert_manifest_matches_committed(reference, repo_root=tmp_path, log=lambda _: None)
    with pytest.raises(BuildError, match="sha256"):
        assert_manifest_matches_committed(other, repo_root=tmp_path, log=lambda _: None)


def test_manifest_check_in_a_git_repo_requires_the_committed_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "is_git_worktree", lambda _root: True)
    consumed = tmp_path / "m.json"
    consumed.write_bytes(b"{}")
    with pytest.raises(BuildError, match="committed manifest"):
        assert_manifest_matches_committed(consumed, repo_root=tmp_path, log=lambda _: None)


def test_manifest_check_unchanged_for_the_real_repository():
    committed = Path(__file__).resolve().parents[4] / builder.DEFAULT_MANIFEST
    assert assert_manifest_matches_committed(committed) == hashlib.sha256(committed.read_bytes()).hexdigest()
