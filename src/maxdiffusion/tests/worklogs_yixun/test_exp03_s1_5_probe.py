"""exp_03 S1.5 — the no-update discriminator probe (plan v3.2 §4).

What has to be true before its numbers mean anything:

* **It updates nothing.** The parameters going in are bit-identical to the parameters coming out —
  checked at the bit level, because a tolerance would hide exactly the small update an accidental
  ``apply_gradients`` produces.
* **The ``p_ss=0`` identity holds exactly enough.** A with scheduled sampling off IS the plain
  objective; if that drifts, every A result is about a bug.
* **The label isolation measures the label.** Same states, same draws, only the supervision differs.
* **The variance decomposition is the law of total variance**, checked against synthetic gradients
  whose support and data variances are known in closed form — including the degenerate case where
  the M "draws" are identical, which must report a support term of zero.
* Output is canonical and refuses verdict directories; the launcher carries the pinned apparatus.
"""

from __future__ import annotations

import inspect
import math
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import maxdiffusion.probe_exp03_s1_5 as s1_5
import maxdiffusion.trainers.wan_ti2v_exp03_trainer as exp03
import maxdiffusion.trainers.wan_ti2v_overfit100_trainer as parent

_REPO = Path(parent.__file__).parents[3]
_LAUNCHER = _REPO / "bash_scripts" / "probe_exp03_s1_5.sh"
_PROBE_LAUNCHER = _REPO / "bash_scripts" / "probe_wan_overfit100_sampling.sh"
_CONFIG = _REPO / "src" / "maxdiffusion" / "configs" / "base_wan_5b_exp03.yml"


# =============================================================================================
# 1. The design pins and the no-update invariant.
# =============================================================================================


def test_the_design_is_approval_pinned():
    assert (s1_5.S1_5_NUM_BATCHES, s1_5.S1_5_SUPPORT_DRAWS) == (8, 4)
    assert s1_5.S1_5_STATES == ("checkpoint", "init")
    s1_5.assert_approved_s1_5_design(num_batches=8, support_draws=4, states=("checkpoint", "init"))


@pytest.mark.parametrize(
    "num_batches,support_draws,states,needle",
    [
        (4, 4, ("checkpoint", "init"), "K=8"),
        (16, 4, ("checkpoint", "init"), "K=8"),
        (8, 1, ("checkpoint", "init"), "M=4"),
        (8, 8, ("checkpoint", "init"), "M=4"),
        (8, 4, ("checkpoint",), "half the question"),
        (8, 4, ("init", "checkpoint"), "half the question"),
    ],
)
def test_a_hostile_design_override_is_refused(num_batches, support_draws, states, needle):
    with pytest.raises(ValueError) as excinfo:
        s1_5.assert_approved_s1_5_design(num_batches=num_batches, support_draws=support_draws, states=states)
    assert needle in str(excinfo.value)


def test_the_no_update_invariant_is_bit_level():
    # A sha256 digest per leaf, so the check is BIT identity: a one-ULP change is caught, and so is
    # a permutation -- which a byte SUM would have missed entirely.
    values = np.linspace(1.0, 2.0, 1024, dtype=np.float32)
    params = {"gain": jnp.asarray(values)}
    assert all(isinstance(digest, str) and len(digest) == 64 for digest in s1_5.params_fingerprint(params))
    before = s1_5.params_fingerprint(params)
    s1_5.assert_no_update(before, s1_5.params_fingerprint(params))  # unchanged: fine

    nudged_values = values.copy()
    nudged_values[7] = np.nextafter(nudged_values[7], np.float32(2.0))
    assert nudged_values[7] != values[7]
    after = s1_5.params_fingerprint({"gain": jnp.asarray(nudged_values)})
    assert after != before

    with pytest.raises(RuntimeError) as excinfo:
        s1_5.assert_no_update(before, after)
    assert "applies no updates" in str(excinfo.value)

    # A PERMUTATION: same bytes, different tensor. A sum-based fingerprint would call this unchanged.
    swapped = values.copy()
    swapped[0], swapped[1] = values[1], values[0]
    with pytest.raises(RuntimeError):
        s1_5.assert_no_update(before, s1_5.params_fingerprint({"gain": jnp.asarray(swapped)}))


def test_the_driver_asserts_no_update_around_the_measurement():
    source = inspect.getsource(s1_5._run_one_state)
    assert "before = params_fingerprint(state.params)" in source
    assert "assert_no_update(before, params_fingerprint(state.params))" in source
    assert source.index("before = params_fingerprint") < source.index("state_report(")
    assert source.index("state_report(") < source.index("assert_no_update(")
    # No optimizer CALL anywhere in the module -- checked structurally, so the docstring that
    # explains why there is none is not mistaken for one.
    import ast

    tree = ast.parse(Path(s1_5.__file__).read_text())
    called = {
        getattr(node.func, "attr", getattr(node.func, "id", ""))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    # Computing gradients is the probe's JOB (value_and_grad is expected, and since the Job 8c fix
    # it is the jitted form); APPLYING them is what must never happen.
    assert not called & {"apply_gradients"}, called
    module_text = Path(s1_5.__file__).read_text()
    assert "tx.update" not in module_text
    assert "state.replace(params=" not in module_text.split("def restore_exact_step")[0]


# =============================================================================================
# 2. The support-variance decomposition (D2).
# =============================================================================================


def _grad(vector):
    return {"w": jnp.asarray(vector, dtype=jnp.float32)}


def test_the_variance_decomposition_is_the_law_of_total_variance():
    # Synthetic gradients with known parts: batch means at -1 and +1 (data spread), draws at +-0.5
    # around each mean (support spread). Both are exact in closed form.
    gradients = [
        [_grad([-1.5]), _grad([-0.5])],  # batch 0: mean -1, deviations -+0.5
        [_grad([0.5]), _grad([1.5])],  # batch 1: mean +1, deviations -+0.5
    ]
    stats = s1_5.variance_decomposition(gradients)
    assert stats["num_batches"] == 2 and stats["support_draws"] == 2
    assert stats["support_variance"] == pytest.approx(0.25)  # mean of (0.5^2)
    assert stats["batch_shared_rng_variance"] == pytest.approx(1.0)  # mean of (1^2)
    assert stats["total_variance"] == pytest.approx(1.25)
    assert stats["support_fraction"] == pytest.approx(0.2)
    assert stats["mean_grad_sq_norm"] == pytest.approx(0.0, abs=1e-12)  # grand mean is 0 here


def test_identical_draws_report_no_support_variance():
    # The control draws no support, so its within-batch spread must be exactly 0 -- that contrast is
    # what makes a trial's number readable.
    gradients = [[_grad([1.0]), _grad([1.0])], [_grad([3.0]), _grad([3.0])]]
    stats = s1_5.variance_decomposition(gradients)
    assert stats["support_variance"] == 0.0
    assert stats["batch_shared_rng_variance"] == pytest.approx(1.0)
    assert stats["support_fraction"] == 0.0
    assert stats["gradient_noise_scale"] == pytest.approx(1.0 / 4.0)  # total 1.0, mean grad 2.0


def test_a_single_draw_cannot_report_a_support_term():
    stats = s1_5.variance_decomposition([[_grad([1.0])], [_grad([3.0])]])
    assert stats["support_draws"] == 1 and stats["support_variance"] == 0.0
    assert stats["batch_shared_rng_variance"] == pytest.approx(1.0)


def test_ragged_draw_counts_are_refused():
    with pytest.raises(ValueError):
        s1_5.variance_decomposition([[_grad([1.0]), _grad([2.0])], [_grad([3.0])]])
    with pytest.raises(ValueError):
        s1_5.variance_decomposition([])


def test_the_driver_varies_the_support_not_the_batch_across_draws():
    # THE thing that makes the within-batch term a SUPPORT term: the batch is held fixed while the
    # draw index moves the global step the aux keys fold on.
    source = inspect.getsource(s1_5.state_report)
    assert "for salt in support_salts" in source
    # The salt now travels through the jitted builder's static key rather than a captured view.
    assert "exp03_support_salt=salt" in source and "static_key=(objective, int(salt))" in source
    # The global step, the batch and the shared-stream key are FIXED across the M draws: only the
    # salt moves, so the within-batch spread is the sigma support and nothing else.
    inner = source[source.index("for salt in support_salts") : source.index("# A's LABEL ISOLATION")]
    assert "global_step = jnp.asarray(first_step + index" not in inner  # fixed before the loop
    assert "jax.random.fold_in(rng, index)" not in inner  # the shared key is fixed too

    # The salt re-draws the SUPPORT...
    supports = {
        tuple(int(v) for v in exp03.corrective_support(seed=0, global_step=7, num_steps=25, k_a_max=2, support_salt=s))
        for s in s1_5.S1_5_SUPPORT_SALTS
    }
    assert len(supports) > 1
    # ...and leaves the coin and the ramp exactly where they were.
    coin = jax.random.uniform(exp03.exp03_aux_key(seed=0, global_step=7, purpose="p_ss_coin"), ())
    for salt in s1_5.S1_5_SUPPORT_SALTS:
        salted = jax.random.uniform(exp03.exp03_aux_key(seed=0, global_step=7, purpose="p_ss_coin"), ())
        assert float(salted) == float(coin)
    # A zero salt is exactly the pre-salt draw, so no existing run's randomness moved.
    assert tuple(int(v) for v in exp03.corrective_support(seed=0, global_step=7, num_steps=25, k_a_max=2)) == tuple(
        int(v) for v in exp03.corrective_support(seed=0, global_step=7, num_steps=25, k_a_max=2, support_salt=0)
    )


# =============================================================================================
# 3. Label isolation and p_ss=0 parity.
# =============================================================================================


def test_the_two_labels_differ_only_in_supervision():
    z_gt = jnp.asarray([[1.0, 2.0]])
    eps = jnp.asarray([[0.5, -0.5]])
    sigma_lo = 0.4
    on_path = (1.0 - sigma_lo) * z_gt + sigma_lo * eps
    # ON path the corrective label reduces to the same-eps label: the isolated effect is zero there.
    assert np.allclose(
        np.asarray(s1_5.corrective_label(on_path, z_gt, sigma_lo)),
        np.asarray(s1_5.same_eps_label(on_path, z_gt, eps)),
        atol=1e-6,
    )
    # OFF path they differ, and by exactly the state's displacement over sigma_lo.
    off_path = on_path + jnp.asarray([[0.3, -0.2]])
    difference = np.asarray(s1_5.corrective_label(off_path, z_gt, sigma_lo)) - np.asarray(
        s1_5.same_eps_label(off_path, z_gt, eps)
    )
    assert np.allclose(difference, np.asarray([[0.3, -0.2]]) / sigma_lo, atol=1e-6)


def test_label_isolation_reports_the_isolated_difference():
    report = s1_5.label_isolation(
        corrective_loss=0.75,
        same_eps_loss=0.5,
        corrective_grad=_grad([3.0, 4.0]),
        same_eps_grad=_grad([3.0, 0.0]),
    )
    assert report["loss_delta"] == pytest.approx(0.25)
    assert report["grad_norm_corrective"] == pytest.approx(5.0)
    assert report["grad_norm_same_eps"] == pytest.approx(3.0)
    assert report["grad_cosine"] == pytest.approx(0.6)  # (9)/(5*3)
    assert report["grad_relative_delta"] == pytest.approx(4.0 / 3.0)
    # Identical gradients and losses => no isolated effect at all.
    null = s1_5.label_isolation(
        corrective_loss=0.5, same_eps_loss=0.5, corrective_grad=_grad([1.0]), same_eps_grad=_grad([1.0])
    )
    assert null["loss_delta"] == 0.0 and null["grad_relative_delta"] == pytest.approx(0.0)
    assert null["grad_cosine"] == pytest.approx(1.0)


def test_the_parity_report_is_tight():
    exact = s1_5.parity_report(
        trial_loss=1.0, plain_loss=1.0, trial_grad=_grad([1.0, 2.0]), plain_grad=_grad([1.0, 2.0])
    )
    assert exact["passes"] is True and exact["loss_relative_gap"] == 0.0 and exact["grad_relative_gap"] == 0.0
    assert exact["tolerance"] == s1_5.S1_5_PARITY_TOLERANCE == 1e-5

    # A 1e-3 relative gradient difference is NOT parity -- the identity is exact mathematics.
    loose = s1_5.parity_report(
        trial_loss=1.0, plain_loss=1.0, trial_grad=_grad([1.001, 2.0]), plain_grad=_grad([1.0, 2.0])
    )
    assert loose["passes"] is False
    # ...and a loss-only match does not pass either.
    loss_only = s1_5.parity_report(
        trial_loss=1.0, plain_loss=1.0, trial_grad=_grad([2.0, 2.0]), plain_grad=_grad([1.0, 2.0])
    )
    assert loss_only["passes"] is False


def test_the_parity_tolerance_is_not_loosened_in_the_module():
    module = Path(s1_5.__file__).read_text()
    assert "S1_5_PARITY_TOLERANCE = 1e-5" in module
    for loose in ("1e-2", "1e-3", "atol=0.1"):
        assert f"tolerance: float = {loose}" not in module


# =============================================================================================
# 4. Output canonicality and isolation.
# =============================================================================================


def test_the_output_path_is_canonical_per_state():
    config = SimpleNamespace(output_dir="gs://bucket/out", run_name="exp03-s1_5")
    assert (
        s1_5.s1_5_output_path(config, state_label="checkpoint", checkpoint_step=10000)
        == "gs://bucket/out/exp03-s1_5/validation_probe_sampling/s1_5_checkpoint_ckpt10000.json"
    )
    assert (
        s1_5.s1_5_output_path(config, state_label="init", checkpoint_step=0)
        == "gs://bucket/out/exp03-s1_5/validation_probe_sampling/s1_5_init_ckpt0.json"
    )
    with pytest.raises(ValueError):
        s1_5.s1_5_output_path(config, state_label="somewhere_else", checkpoint_step=0)


@pytest.mark.parametrize(
    "output_dir,run_name",
    [
        ("gs://bucket/out", "step_010000_s3_intermediate"),
        ("gs://bucket/out/step_010000_s3_full_set", "exp03"),
        ("gs://bucket/out", "ok/../step_010000_s3_segment_final"),
    ],
)
def test_a_hostile_path_cannot_steer_the_probe_into_the_evidence_tree(output_dir, run_name):
    with pytest.raises(ValueError) as excinfo:
        s1_5.s1_5_output_path(
            SimpleNamespace(output_dir=output_dir, run_name=run_name), state_label="init", checkpoint_step=0
        )
    assert "step_" in str(excinfo.value)


def test_the_artifact_is_diagnostic_and_written_immutably(tmp_path):
    payload = s1_5.s1_5_artifact(
        SimpleNamespace(
            output_dir=str(tmp_path),
            run_name="r",
            checkpoint_dir="gs://x/ck",
            train_data_dir="gs://x/train100",
            model_manifest_path="gs://x/manifest.json",
        ),
        state_label="init",
        checkpoint_step=0,
        report={"state": "init", "per_objective": {}, "support_variance": {}},
    )
    assert payload["schema"] == s1_5.S1_5_SCHEMA and payload["kind"] == "diagnostic"
    assert payload["num_batches"] == 8 and payload["support_draws"] == 4
    assert payload["objectives"] == list(s1_5.S1_5_OBJECTIVES)
    for forbidden in ("eval_pass_role", "canonical_cohort", "role_validation", "run_signature"):
        assert forbidden not in payload

    path = str(tmp_path / "s1_5.json")
    s1_5.write_s1_5_artifact(path, payload)
    s1_5.write_s1_5_artifact(path, payload)  # identical rewrite tolerated
    with pytest.raises(ValueError):
        s1_5.write_s1_5_artifact(path, {**payload, "checkpoint_step": 999})


def test_the_probe_touches_no_verdict_machinery():
    module = Path(s1_5.__file__).read_text()
    for forbidden in (
        "overfit100_aggregation_artifact",
        "assert_pass_role_plan",
        "validate_artifact_role",
        "write_staged_row",
        "overfit100_publication_state",
        "eval_pass_role",
    ):
        assert forbidden not in module, forbidden


def test_the_probe_reuses_the_trainers_replay_rather_than_duplicating_it():
    source = inspect.getsource(s1_5.state_report)
    assert "exp03.exp03_frozen_replay(" in source
    # The per-objective sweep delegates; the probe's OWN losses exist only for the two questions the
    # replay cannot answer (label isolation needs two labels on one state, conditional parity needs a
    # fixed-support comparator), and both are built from the trainer's helpers.
    module = Path(s1_5.__file__).read_text()
    assert "exp03._exp03_prologue(" in module and "exp03._forward_velocity(" in module
    assert "def _denoising_loss" not in module and "def _corrective_ss_loss" not in module


def test_the_extended_replay_reports_cosines_against_the_control():
    source = inspect.getsource(exp03.exp03_frozen_replay)
    assert '"control"' in source and "grad_cosine_{name}_vs_control" in source
    assert "include_control" in source


# =============================================================================================
# 5. Launcher.
# =============================================================================================


def test_the_launcher_exists_and_passes_bash_n():
    assert _LAUNCHER.exists()
    bash = shutil.which("bash") or "/bin/bash"
    proc = subprocess.run([bash, "-n", str(_LAUNCHER)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr


def test_the_launcher_forwards_the_probe_knobs():
    text = _LAUNCHER.read_text()
    assert "src/maxdiffusion/probe_exp03_s1_5.py" in text
    assert "src/maxdiffusion/configs/base_wan_5b_exp03.yml" in text
    for env, key in (
        ("CHECKPOINT_STEP", "checkpoint_step"),
        ("CHECKPOINT_DIR", "checkpoint_dir"),
        ("S1_5_NUM_BATCHES", "s1_5_num_batches"),
        ("S1_5_SUPPORT_DRAWS", "s1_5_support_draws"),
    ):
        assert f'{key}="${{{env}}}"' in text, key
        assert f'echo "{env}=' in text, env
    # The Tier-1 checkpoint is GENUINELY required: the :? check fires because nothing defaults it
    # first (a default assigned above would make the check unreachable).
    assert ': "${CHECKPOINT_DIR:?' in text
    assert 'CHECKPOINT_DIR="${CHECKPOINT_DIR:-' not in text
    # The ramp origin is PER STATE and decided in the probe (checkpoint 10000, init 0), so the
    # launcher must NOT carry one -- a single env would give both states the same ramp.
    assert "EXP03_RAMP_ORIGIN" not in text
    assert s1_5.S1_5_STATE_PLAN["checkpoint"]["ramp_origin"] == 10000
    assert s1_5.S1_5_STATE_PLAN["init"]["ramp_origin"] == 0


def test_the_launcher_keeps_the_pinned_apparatus_and_needs_no_ffmpeg():
    text = _LAUNCHER.read_text()
    for required in ("prefetch_hf_snapshot.sh", "local_files_only=True", "export COMMIT", "MODEL_REVISION"):
        assert required in text, required
    assert "# >>> ffmpeg ensure" not in text
    assert "ffmpeg" in text  # ...and it says why it is absent


def test_the_launcher_does_not_drift_from_the_probe_launcher_it_was_cloned_from():
    # Same apparatus, different payload: every uppercase default the exp_02 probe launcher sets must
    # still be set here, with the same value, outside an explicit allowlist.
    assignment = re.compile(r"^(?:export\s+)?([A-Z][A-Z0-9_]*)=(.*)$", re.DOTALL)

    def defaults(text):
        joined, buffer = [], ""
        for raw in text.splitlines():
            buffer = raw if not buffer else buffer + "\n" + raw
            if raw.endswith("\\"):
                continue
            joined.append(buffer)
            buffer = ""
        out = {}
        for line in joined:
            match = assignment.match(line.strip())
            if match:
                out[match.group(1)] = match.group(2)
        return out

    base = defaults(_PROBE_LAUNCHER.read_text())
    mine = defaults(_LAUNCHER.read_text())
    assert "LIBTPU_INIT_ARGS" in base and base["LIBTPU_INIT_ARGS"].count("--xla") > 10
    allowed = {
        "RUN_NAME",
        "OUTPUT_DIR",
        "CHECKPOINT_STEP",
        "CHECKPOINT_DIR",
        "S1_5_NUM_BATCHES",
        "S1_5_SUPPORT_DRAWS",
        "PROBE_NUM_WINDOWS",
        "PROBE_STEPS",  # the exp_02 probe's arms; S1.5 has no sampler arms
    }
    missing = sorted((set(base) - allowed) - set(mine))
    assert not missing, missing
    differing = {key: (base[key], mine[key]) for key in (set(base) & set(mine)) - allowed if base[key] != mine[key]}
    assert not differing, differing


def test_the_config_carries_the_s1_5_keys():
    text = _CONFIG.read_text()
    assert "s1_5_num_batches: 8" in text and "s1_5_support_draws: 4" in text


# =============================================================================================
# 6. The wiring the first version lacked: the diagnostics must REACH the artifact.
# =============================================================================================


def test_the_report_wires_label_isolation_and_parity_and_forced_diagnostics():
    # The first version defined both and called neither, so the logger's branches were unreachable
    # and no artifact could ever contain them. Pinned structurally at the seam that builds the report.
    source = inspect.getsource(s1_5.state_report)
    for required in (
        "label_isolation(",
        "parity_report(",
        "plain_fixed_support_loss(",
        "corrective_at_support(",
        '"forced_p_ss_one"',
        '"per_batch"',
    ):
        assert required in source, required
    returned = source[source.index("return {") :]
    for key in ('"label_isolation"', '"p_ss_zero_parity"', '"forced_p_ss_one"', '"per_batch"', '"support_salts"'):
        assert key in returned, key


def test_the_artifact_records_the_state_identity_and_rejects_non_finite_values(tmp_path):
    config = SimpleNamespace(
        output_dir=str(tmp_path),
        run_name="r",
        checkpoint_dir="gs://x/ck",
        train_data_dir="gs://x/train100",
        model_manifest_path="gs://x/manifest.json",
        seed=0,
    )
    payload = s1_5.s1_5_artifact(
        config, state_label="checkpoint", checkpoint_step=10000, report={"state": "checkpoint"}
    )
    assert payload["first_global_step"] == 10000 and payload["ramp_origin"] == 10000
    assert payload["required_checkpoint_step"] == 10000 and payload["restored_step"] == 10000
    assert payload["iterator_seed"] == 10000  # production continuation semantics: seed + start_step
    assert tuple(payload["support_salts"]) == tuple(s1_5.S1_5_SUPPORT_SALTS)
    init_payload = s1_5.s1_5_artifact(config, state_label="init", checkpoint_step=0, report={"state": "init"})
    assert init_payload["first_global_step"] == 0 and init_payload["ramp_origin"] == 0

    # A NaN anywhere is refused, with the failing key named.
    with pytest.raises(ValueError) as excinfo:
        s1_5.write_s1_5_artifact(str(tmp_path / "bad.json"), {**payload, "report": {"grad_norm_a": float("nan")}})
    assert "report.grad_norm_a" in str(excinfo.value)
    with pytest.raises(ValueError):
        s1_5.assert_finite_payload({"rows": [{"x": 1.0}, {"x": float("inf")}]})


def test_the_state_plan_is_the_canonical_per_state_mapping():
    assert s1_5.S1_5_STATE_PLAN["checkpoint"] == {
        "first_global_step": 10000,
        "ramp_origin": 10000,
        "required_checkpoint_step": 10000,
    }
    assert s1_5.S1_5_STATE_PLAN["init"] == {
        "first_global_step": 0,
        "ramp_origin": 0,
        "required_checkpoint_step": 0,
    }
    # The restore is PINNED to the required step, not to whatever is latest.
    source = inspect.getsource(s1_5.build_probe_state)
    assert 'required = plan["required_checkpoint_step"]' in source
    assert "restore_exact_step(manager, state, required=required)" in source
    assert "if int(start_step) != int(required):" in source  # a backstop, not the selector
    assert "manager.latest_step()" not in source
    # ...and init goes through the SAME checkpoint-manager path, on an empty directory.
    assert "_empty_checkpoint_dir(config)" in source
    restorer = inspect.getsource(s1_5.restore_exact_step)
    assert "manager.restore(" in restorer
    assert "manager.latest_step()" not in restorer  # the docstring may NAME it; the code must not call it
    assert "manager.all_steps()" in restorer


def test_the_iterator_seed_follows_production_continuation():
    source = inspect.getsource(s1_5._run_one_state)
    assert "seed=config.seed + checkpoint_step" in source


def test_the_sigma_trace_uses_the_state_already_in_memory():
    # Re-restoring would have traced the checkpoint twice (the init trace was silently a second
    # checkpoint trace) and would hold two live 5B models at once.
    module = Path(s1_5.__file__).read_text()
    assert "trace.run_trace(" not in module
    source = inspect.getsource(s1_5._run_one_state)
    assert "trace_in_memory_state(" in source
    # ...and the standalone per-state trace JSON is written under the trace module's own rules.
    assert "trace.write_trace_artifact(trace.trace_output_path(config, checkpoint_step)" in source
    tracer = inspect.getsource(s1_5.trace_in_memory_state)
    assert "nnx.merge(state.graphdef, state.params, state.rest_of_state)" in tracer
    assert "_restore" not in tracer and "run_trace" not in tracer


def test_the_reductions_never_flatten_a_gradient_on_the_host():
    # At 5B parameters a float64 flatten is ~40 GB per gradient; the first version retained 32 of
    # them. Everything is leafwise/on-device now, and the Welford accumulator keeps O(1) trees.
    for module_text in (Path(s1_5.__file__).read_text(), Path(exp03.__file__).read_text()):
        assert "np.float64" not in module_text and "dtype=jnp.float64" not in module_text
        assert "np.concatenate" not in module_text
        assert "_flat_gradient" not in module_text
    welford = s1_5._TreeWelford()
    for value in ([1.0], [3.0], [5.0]):
        welford.update({"w": jnp.asarray(value, dtype=jnp.float32)})
    assert welford.count == 3
    assert float(welford.mean["w"][0]) == pytest.approx(3.0)
    assert welford.population_variance() == pytest.approx(8.0 / 3.0)  # ((-2)^2+0+2^2)/3
    assert welford.sample_variance() == pytest.approx(4.0)
    # The accumulator holds ONE tree, whatever it has consumed.
    assert len(jax.tree_util.tree_leaves(welford.mean)) == 1


def test_the_frozen_replay_default_is_unchanged_from_the_parent_commit():
    # MAJOR 6: the default must not have grown an extra control forward/backward.
    signature = inspect.signature(exp03.exp03_frozen_replay)
    assert signature.parameters["include_control"].default is False
    assert signature.parameters["with_gradients"].default is True
    # ...and grad_cosine is still a float32 TREE reduction, not a host flatten -- now through the
    # JITTED dot and squared-norm helpers (jaxopt's eager tree_l2_norm was part of what exhausted
    # HBM in Job 8c, and is gone from this module).
    cosine_source = inspect.getsource(exp03.grad_cosine)
    assert "tree_dot(" in cosine_source and "tree_sq_norm(" in cosine_source
    # Structural: the module may NAME jaxopt in the comment recording why it is gone; it must not
    # import or call it.
    import ast

    tree = ast.parse(Path(exp03.__file__).read_text())
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name.startswith("jaxopt") for name in imported), imported


def test_no_vacuous_assertions_survive_in_this_file():
    # The `or True` that made a signature assertion unconditional is the exact failure mode this
    # whole SOP exists to prevent; a test that asserts nothing is worse than no test.
    text = Path(__file__).read_text()
    needle = " or " + "True"
    occurrences = [line for line in text.splitlines() if needle in line and "needle" not in line]
    assert not occurrences, occurrences
    assert "assert " + "True" not in text.replace("assert " + "True" + '"', "")


# =============================================================================================
# 7. Closing round: the control flow itself, executed.
# =============================================================================================


def _production_config_keys():
    """The real exp_03 config's key set, so a test config has the shape production has.

    Without this, a test config is a bare namespace whose ``vars()`` happens to work -- which is
    precisely why the empty-``vars()`` defect reached hardware. Here the e2e runs against every key
    a real run carries, with only the shapes shrunk.
    """
    from maxdiffusion import pyconfig

    saved = (getattr(pyconfig, "_config", None), getattr(pyconfig, "config", None))
    try:
        pyconfig.initialize([None, str(_CONFIG), "run_name=s1_5-e2e-test"], unittest=True)
        return dict(pyconfig.config.get_keys())
    finally:
        pyconfig._config, pyconfig.config = saved


def _proxy_config(keys: dict):
    """A config with pyconfig's SHAPE, not merely its keys.

    Mimics ``pyconfig.HyperParameters``: the instance ``__dict__`` stays EMPTY (so ``vars()`` is
    ``{}``, the property that broke Job 8b), the keys are served from a closure through
    ``__getattr__``, ``get_keys()`` returns them, a missing key raises ``ValueError`` exactly as
    pyconfig does, and assignment is refused.

    A ``SimpleNamespace`` carrying the same 225 keys is NOT a substitute: its ``vars()`` works, so an
    end-to-end test built on one passes even with the defect reinstated. That is precisely how this
    defect shipped twice.
    """
    store = dict(keys)

    class _ProxyConfig:
        def __getattr__(self, attr):
            if attr not in store:
                raise ValueError(f"Requested key {attr}, not in config")
            return store[attr]

        def __setattr__(self, attr, value):
            raise ValueError

        def get_keys(self):
            return store

    proxy = _ProxyConfig()
    assert vars(proxy) == {}, "the proxy must have an empty __dict__ or it is not the shape under test"
    return proxy


def _toy_state_and_batches(num_batches=2, config_shape="namespace"):
    """A tiny real Overfit100TrainState and real batches — enough to EXECUTE the report path."""
    import optax
    from flax import nnx

    class _Stub(nnx.Module):
        def __init__(self):
            # A MATRIX, not a scalar: the gauge counts buffers whose (shape, dtype) matches a
            # parameter leaf, and a scalar parameter would match every scalar in the process --
            # losses, sigmas, metrics -- which is what made the toy count meaningless.
            self.gain = nnx.Param(jnp.full((8, 8), 0.25, dtype=jnp.float32))

        def __call__(self, **kwargs):
            hidden = kwargs["hidden_states"].astype(jnp.float32)
            gain = jnp.mean(self.gain[...])
            return (gain * jnp.tanh(hidden) + 0.01 * jnp.mean(kwargs["timestep"].astype(jnp.float32))).astype(
                kwargs["hidden_states"].dtype
            )

    graphdef, params, rest = nnx.split(_Stub(), nnx.Param, ...)
    state = parent.Overfit100TrainState.create(
        apply_fn=graphdef.apply,
        params=params,
        tx=optax.sgd(0.1),
        graphdef=graphdef,
        rest_of_state=rest,
        context_table=jax.random.normal(jax.random.key(41), (4, 4, 8), dtype=jnp.float32),
    )
    batches = []
    for index in range(num_batches):
        key = jax.random.key(100 + index)
        k1, k2 = jax.random.split(key)
        batches.append(
            {
                "z_i0": jax.random.normal(k1, (3, 3, 1, 5, 6), dtype=jnp.float32),
                "z_video": jax.random.normal(k2, (3, 3, 4, 5, 6), dtype=jnp.float32),
                "episode_index": jnp.asarray([0, 1, 2], dtype=jnp.int32),
            }
        )
    # PRODUCTION-SHAPED: every key a real run has, with the tensor shapes and dtypes shrunk. If the
    # view builders were reverted to vars(), the pyconfig closure tests would catch it -- and this
    # one keeps the e2e honest by giving it the same key surface rather than a five-field stub.
    settings = dict(_production_config_keys())
    settings.update(
        weights_dtype="float32",
        activations_dtype="float32",
        global_batch_size_to_train_on=2,
        side_adapter_sampling_steps=25,
        flow_shift=5.0,
        side_adapter_t_sampling="uniform",
        side_adapter_noise_mode="fresh",
        seed=0,
        exp03_objective="combined",
        exp03_k_a=2,
        exp03_k_b=2,
        exp03_lambda=0.5,
        exp03_p_ss_max=0.5,
        exp03_p_ss_ramp_steps=10,
        exp03_ramp_origin=0,
        exp03_support_salt=0,
        run_name="s1_5-test",
        checkpoint_dir="gs://x/ck",
        train_data_dir="gs://x/train100",
        model_manifest_path="gs://x/manifest.json",
    )
    if config_shape == "proxy":
        # THE production shape: vars() is empty, so a view built from vars() would carry nothing.
        config = _proxy_config(settings)
        assert vars(config) == {}
        assert config.weights_dtype == "float32"  # ...but the attribute path works
    else:
        config = SimpleNamespace(**settings)
        assert s1_5.CONFIG_SENTINEL_KEY in vars(config)  # the helper's fallback branch
    from maxdiffusion.schedulers import FlaxFlowMatchScheduler

    scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32, shift=5.0, sigma_min=0.0, sigma_max=1.0)
    return state, batches, config, scheduler


@pytest.mark.parametrize("config_shape", ["proxy", "namespace"])
@pytest.mark.parametrize("state_label", ["checkpoint", "init"])
def test_both_states_run_the_full_report_and_log_path(state_label, config_shape, tmp_path, monkeypatch):
    # END TO END through the REAL control flow, for BOTH states -- the wiring bug class that has now
    # bitten twice (an unreachable branch, then a KeyError on a renamed key that killed the init
    # state) is only caught by executing it, never by inspecting it.
    state, batches, config, scheduler = _toy_state_and_batches(config_shape=config_shape)
    if config_shape == "proxy":
        config.get_keys()["output_dir"] = str(tmp_path)  # the proxy refuses assignment, like pyconfig
    else:
        config.output_dir = str(tmp_path)
    checkpoint_step = s1_5.S1_5_STATE_PLAN[state_label]["required_checkpoint_step"]
    lines: list[str] = []
    monkeypatch.setattr(s1_5.max_logging, "log", lambda line: lines.append(str(line)))

    report = s1_5.state_report(
        state,
        batches,
        jax.random.key(1),
        config,
        scheduler,
        state_label=state_label,
        checkpoint_step=checkpoint_step,
        support_salts=(1, 2),
    )
    for key in (
        "label_isolation",
        "p_ss_zero_parity",
        "forced_p_ss_one",
        "per_batch",
        "support_variance",
        "branch_outcomes",
    ):
        assert key in report, key
    assert report["first_global_step"] == s1_5.S1_5_STATE_PLAN[state_label]["first_global_step"]
    assert set(report["support_variance"]) == set(s1_5.S1_5_OBJECTIVES)
    assert "batch_shared_rng_variance" in report["support_variance"]["control"]
    assert report["branch_outcomes"]["self_generated"] + report["branch_outcomes"]["teacher_forced"] == len(batches)

    payload = s1_5.s1_5_artifact(config, state_label=state_label, checkpoint_step=checkpoint_step, report=report)
    s1_5.log_summary(payload)  # the line that used to raise KeyError after the first state
    assert any("state=" + state_label in line for line in lines)
    assert any("support_var=" in line for line in lines)
    assert any("label isolation" in line for line in lines)
    assert any("p_ss=0 parity" in line for line in lines)
    # ...and the artifact is writable (no non-finite values anywhere in a real report).
    s1_5.write_s1_5_artifact(
        s1_5.s1_5_output_path(config, state_label=state_label, checkpoint_step=checkpoint_step), payload
    )


def test_the_first_state_is_released_before_the_next_one_is_built():
    import weakref

    class _Holder:
        pass

    holder = _Holder()
    reference = weakref.ref(holder)
    s1_5.release(holder)
    del holder
    import gc

    gc.collect()
    assert reference() is None, "the released object is still reachable"

    # The driver releases the per-state locals, and does so in a frame that returns before the next
    # state's pipeline is built.
    driver = inspect.getsource(s1_5.run_s1_5)
    assert "_run_one_state(config, trainer, scheduler, state_label)" in driver
    assert "release()" in driver
    per_state = inspect.getsource(s1_5._run_one_state)
    assert "release(state, batches, report, pipeline, iterator)" in per_state
    assert per_state.index("trainer._load_wan_pipeline()") < per_state.index("release(state,")


def test_the_required_step_is_selected_from_a_directory_that_holds_later_ones():
    # The exp_02 run directory REALLY holds later checkpoints, so "latest, then reject" would grab
    # 12500 and only complain afterwards.
    requested = {}

    class _Manager:
        def all_steps(self):
            return [10000, 12500]

        def latest_step(self):
            return 12500

        def restore(self, step, args=None):
            requested["step"] = step
            return {"params": {"w": jnp.asarray([1.0])}, "opt_state": (), "step": {"step": step}}

    state = SimpleNamespace(
        params={"w": jnp.asarray([0.0])},
        opt_state=(),
        replace=lambda **kw: SimpleNamespace(params=kw["params"], opt_state=kw["opt_state"]),
    )
    restored, step = s1_5.restore_exact_step(_Manager(), state, required=10000)
    assert requested["step"] == 10000 and step == 10000
    assert float(restored.params["w"][0]) == 1.0

    class _Missing(_Manager):
        def all_steps(self):
            return [12500]

    with pytest.raises(ValueError) as excinfo:
        s1_5.restore_exact_step(_Missing(), state, required=10000)
    assert "SELECTED" in str(excinfo.value)

    # init: nothing to restore, and a non-empty directory is refused.
    class _Empty(_Manager):
        def all_steps(self):
            return []

    unchanged, zero = s1_5.restore_exact_step(_Empty(), state, required=0)
    assert zero == 0 and unchanged is state
    with pytest.raises(ValueError):
        s1_5.restore_exact_step(_Manager(), state, required=0)


def test_the_variance_never_retains_more_than_a_couple_of_trees():
    # PEAK retained count, not just the final reduction type: the generator is consumed one gradient
    # at a time, so the K x M = 32 trees the first version held never coexist.
    import weakref

    live = {"count": 0, "peak": 0}

    def make(value):
        # A plain dict pytree; the weakref goes on the LEAF, which is what actually occupies memory.
        leaf = jnp.asarray([float(value)], dtype=jnp.float32)
        live["count"] += 1
        live["peak"] = max(live["peak"], live["count"])
        weakref.finalize(leaf, lambda: live.__setitem__("count", live["count"] - 1))
        return {"w": leaf}

    stats = s1_5.variance_decomposition((make(k * 4 + m) for m in range(4)) for k in range(8))
    assert stats["num_batches"] == 8 and stats["support_draws"] == 4
    assert live["peak"] <= 3, live  # one in flight, not 32
    # ...and the CALLER really streams. Checked structurally, because a substring assertion is
    # satisfied by `[list(_draws(...)) for ...]` -- which materializes all 32 trees again while
    # still mentioning the generator function.
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(s1_5.state_report)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "variance_decomposition"
    ]
    assert len(calls) == 1, calls
    (argument,) = calls[0].args
    assert isinstance(argument, ast.GeneratorExp), ast.dump(argument)  # not a list comprehension
    element = argument.elt
    assert isinstance(element, ast.Call), ast.dump(element)  # each row is a call...
    assert not isinstance(element, (ast.ListComp, ast.List))
    assert "list(" not in ast.unparse(argument), ast.unparse(argument)  # ...not list(call)
    # ...and the row-producer is itself a generator function, so a row is lazy too.
    draws = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_draws")
    assert any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(draws))
    consumer = inspect.getsource(s1_5.variance_decomposition)
    assert "del grad" in consumer and "list(gradients)" not in consumer


def test_an_unknown_commit_is_refused(monkeypatch):
    monkeypatch.setattr(s1_5.gen, "_eval_code_commit", lambda: "unknown")
    with pytest.raises(ValueError) as excinfo:
        s1_5.assert_commit_is_pinned()
    assert "40-hex" in str(excinfo.value)
    monkeypatch.setattr(s1_5.gen, "_eval_code_commit", lambda: "0" * 39)
    with pytest.raises(ValueError):
        s1_5.assert_commit_is_pinned()
    monkeypatch.setattr(s1_5.gen, "_eval_code_commit", lambda: "a" * 40)
    s1_5.assert_commit_is_pinned()
    assert "assert_commit_is_pinned()" in inspect.getsource(s1_5.run_s1_5)


def test_the_launcher_drift_test_rejects_unexpected_additions():
    # Bidirectional: a key the exp_02 probe launcher does not have, and that is not on the
    # allowlist, is drift too.
    assignment = re.compile(r"^(?:export\s+)?([A-Z][A-Z0-9_]*)=(.*)$", re.DOTALL)

    def defaults(text):
        joined, buffer = [], ""
        for raw in text.splitlines():
            buffer = raw if not buffer else buffer + "\n" + raw
            if raw.endswith("\\"):
                continue
            joined.append(buffer)
            buffer = ""
        return {m.group(1): m.group(2) for m in (assignment.match(line.strip()) for line in joined) if m}

    base = defaults(_PROBE_LAUNCHER.read_text())
    mine = defaults(_LAUNCHER.read_text())
    allowed = {
        "RUN_NAME",
        "OUTPUT_DIR",
        "CHECKPOINT_STEP",
        "CHECKPOINT_DIR",
        "S1_5_NUM_BATCHES",
        "S1_5_SUPPORT_DRAWS",
        "PROBE_NUM_WINDOWS",
        "PROBE_STEPS",
    }
    assert not sorted((set(base) - allowed) - set(mine)), "a shared default went missing"
    assert not sorted((set(mine) - allowed) - set(base)), "an unexpected default was added"


# =============================================================================================
# 8. The hardware failure: a module attribute that does not exist.
#
# The first S1.5 run died at startup on ``gen.load_next_batch`` -- a name the eval module has never
# had. No test caught it because the batches were BUILT in the tests rather than PULLED through the
# production call. Two guards, one cheap and total, one specific.
# =============================================================================================


def _module_attribute_references(module):
    """Every ``alias.attr`` in a module's source, resolved back to the module the alias names."""
    import ast

    tree = ast.parse(Path(module.__file__).read_text())
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for entry in node.names:
                aliases[entry.asname or entry.name.split(".")[0]] = entry.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for entry in node.names:
                aliases[entry.asname or entry.name] = f"{node.module}.{entry.name}"
    references = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            references.add((node.value.id, aliases[node.value.id], node.attr))
    return references


@pytest.mark.parametrize("module", [s1_5, exp03, parent])
def test_every_module_attribute_reference_resolves(module):
    # THE class of bug that killed the first hardware run, caught at test time without hardware: if
    # a module reference names an attribute that does not exist, this fails here rather than 20
    # minutes into a TPU job.
    import importlib

    unresolved = []
    for alias, target, attribute in sorted(_module_attribute_references(module)):
        try:
            imported = importlib.import_module(target)
        except ImportError:
            continue  # a from-import of a name, not a module: covered by import time itself
        if not hasattr(imported, attribute):
            unresolved.append(f"{alias}({target}).{attribute}")
    assert not unresolved, unresolved


def test_the_probe_pulls_batches_the_way_the_training_loop_does():
    # Same function, same call shape, same import site as start_training -- checked against the
    # trainer's own source rather than against a memory of it.
    probe_source = inspect.getsource(s1_5._run_one_state)
    loop_source = inspect.getsource(parent.WanTI2VOverfit100Trainer.start_training)
    assert "load_next_batch(iterator, None, config)" in probe_source
    assert "load_next_batch(train_iter, None, config)" in loop_source  # the loop's first pull
    # Structural, so the comment recording what went wrong is not mistaken for the thing itself.
    assert ("gen", "maxdiffusion.generate_wan_side_adapter", "load_next_batch") not in _module_attribute_references(
        s1_5
    )
    # ...and it comes from the module that defines it.
    from maxdiffusion import train_utils

    assert s1_5.load_next_batch is train_utils.load_next_batch
    assert not hasattr(s1_5.gen, "load_next_batch")  # the name the first run reached for
    # The iterator is built exactly as the training loop builds it, at the production seed.
    assert "trainer._load_dataset(mesh, is_training=True, seed=config.seed + checkpoint_step)" in probe_source
    assert "self._load_dataset(mesh, is_training=True, seed=config.seed + start_step)" in loop_source


def test_the_real_batch_pull_runs_against_a_real_iterator():
    # EXECUTED, not simulated: the production function, against a real iterator, with the
    # reuse_example_batch semantics the config carries.
    from maxdiffusion import train_utils

    batches = [
        {"z_i0": jnp.zeros((2, 3, 1, 5, 6)), "z_video": jnp.ones((2, 3, 4, 5, 6)) * value} for value in (1.0, 2.0)
    ]
    iterator = iter(batches)
    config = SimpleNamespace(reuse_example_batch=False)
    first = train_utils.load_next_batch(iterator, None, config)
    second = train_utils.load_next_batch(iterator, first, config)
    assert float(first["z_video"][0, 0, 0, 0, 0]) == 1.0
    assert float(second["z_video"][0, 0, 0, 0, 0]) == 2.0  # it really advanced
    # ...and the probe asks for a FRESH batch each time (example_batch=None), so a config with
    # reuse_example_batch=True cannot silently hand it the same batch K times.
    reusing = SimpleNamespace(reuse_example_batch=True)
    assert train_utils.load_next_batch(iter(batches), None, reusing)["z_video"][0, 0, 0, 0, 0] == 1.0
    assert "load_next_batch(iterator, None, config)" in inspect.getsource(s1_5._run_one_state)


# =============================================================================================
# 9. The config-view shape: vars() on the pyconfig proxy is EMPTY.
#
# Job 8b got past the 5B restore and the dataset pin and then died in _denoising_loss on
# config.weights_dtype, because every view was built from vars(pyconfig.config) -- which is {}.
# The fifth instance of the inspect-vs-execute class, in a shape the AST attribute guard cannot
# see: it is config-KEY flow, not module attributes. So these tests initialize pyconfig FOR REAL.
# =============================================================================================


@pytest.fixture(scope="module")
def real_pyconfig():
    """A genuine ``pyconfig.initialize`` against the exp_03 YAML — no TPU, no model build.

    pyconfig keeps module-level globals, so they are snapshotted and restored: this fixture must not
    change what any other test sees.
    """
    from maxdiffusion import pyconfig

    saved = (getattr(pyconfig, "_config", None), getattr(pyconfig, "config", None))
    pyconfig.initialize(
        [None, str(_CONFIG), "run_name=s1_5-config-view-test"],
        unittest=True,
    )
    try:
        yield pyconfig.config
    finally:
        pyconfig._config, pyconfig.config = saved


def test_the_pyconfig_proxy_hides_its_keys_from_vars(real_pyconfig):
    # THE canary, and the reason _config_key_dict exists. If pyconfig ever changes shape so that
    # vars() works, this fails and the helper gets re-examined rather than quietly outliving its
    # reason.
    assert vars(real_pyconfig) == {}
    keys = real_pyconfig.get_keys()
    assert len(keys) > 100
    assert keys["weights_dtype"] == "bfloat16"
    assert real_pyconfig.weights_dtype == "bfloat16"  # the attribute path works; vars() does not


def test_the_views_carry_every_production_key(real_pyconfig):
    # Built from the REAL config, then hard-read exactly as the replay path reads them.
    checkpoint = s1_5.state_view(real_pyconfig, "checkpoint")
    assert checkpoint.weights_dtype == "bfloat16"  # the read that crashed on hardware
    assert checkpoint.seed == real_pyconfig.seed
    assert checkpoint.exp03_ramp_origin == 10000  # the per-state override
    init = s1_5.state_view(real_pyconfig, "init")
    assert init.exp03_ramp_origin == 0

    objective = s1_5._objective_config(real_pyconfig, "corrective_ss")
    assert objective.exp03_objective == "corrective_ss"
    assert objective.weights_dtype == "bfloat16"
    assert objective.activations_dtype == real_pyconfig.activations_dtype

    # Nothing is lost: the view's keys are a superset of the config's.
    production_keys = set(real_pyconfig.get_keys())
    for view in (checkpoint, init, objective):
        assert production_keys <= set(vars(view)), sorted(production_keys - set(vars(view)))[:5]

    # A view of a view stays lossless (vars() on a namespace IS correct, and must keep working).
    nested = s1_5._objective_config(checkpoint, "rollout_loss")
    assert nested.exp03_objective == "rollout_loss"
    assert nested.exp03_ramp_origin == 10000  # the state override survived the second wrap
    assert production_keys <= set(vars(nested))


def test_a_config_that_exposes_no_keys_is_refused():
    # An empty view is the failure Job 8b actually had; it must be impossible to construct.
    with pytest.raises(ValueError) as excinfo:
        s1_5._config_key_dict(SimpleNamespace())
    assert "exposes no keys" in str(excinfo.value)

    class _EmptyProxy:
        def get_keys(self):
            return {}

    with pytest.raises(ValueError):
        s1_5._config_key_dict(_EmptyProxy())


def test_a_config_without_the_sentinel_key_is_refused():
    # Not merely non-empty: the key the replay hard-reads must be there, or the failure moves back
    # into the middle of a 5B forward.
    bare = SimpleNamespace(seed=0, exp03_objective="control")
    with pytest.raises(ValueError) as excinfo:
        s1_5._config_key_dict(bare)
    assert s1_5.CONFIG_SENTINEL_KEY in str(excinfo.value)
    assert s1_5.CONFIG_SENTINEL_KEY == "weights_dtype"
    # ...and both view builders go through the guard.
    for builder in (lambda c: s1_5.state_view(c, "init"), lambda c: s1_5._objective_config(c, "control")):
        with pytest.raises(ValueError):
            builder(bare)


def test_the_helper_prefers_get_keys_over_vars():
    # The ordering that matters: a config offering BOTH must be read through get_keys(), because on
    # the production proxy vars() is the empty one.
    class _Proxy:
        """A config that offers BOTH: bookkeeping in ``__dict__``, the real keys behind get_keys()."""

        def __init__(self):
            self._bookkeeping = "not a config key"
            self.weights_dtype = "float32"  # a STALE shadow of the real key

        def get_keys(self):
            return {"weights_dtype": "bfloat16", "seed": 7}

    keys = s1_5._config_key_dict(_Proxy())
    # get_keys() wins outright -- not "vars() if non-empty", which would take the stale shadow and
    # the bookkeeping field along with it.
    assert keys == {"weights_dtype": "bfloat16", "seed": 7}
    assert "_bookkeeping" not in keys and keys["weights_dtype"] == "bfloat16"
    source = inspect.getsource(s1_5._config_key_dict)
    assert "config.get_keys()" in source and "callable(getter)" in source


# =============================================================================================
# 10. Job 8c: the replay's memory SHAPE.
#
# HBM OOM inside exp03_frozen_replay -- eager (unjitted) backward through a 5B forward, four grad
# trees resident, and whole-tree temporaries from jaxopt's tree_l2_norm, a tree_reduce with a full
# abs copy per leaf, and a per-leaf finiteness sweep. OOM cannot be reproduced in CI, so what is
# pinned here is the STRUCTURE: fused statistics equal their plain references, the resident-tree
# high-water mark is recorded and capped, and the replay's numbers are unchanged.
# =============================================================================================


def _reference_grad_stats(grad):
    """The statistics as four separate plain traversals — the shape the fused pass replaced."""
    leaves = jax.tree_util.tree_leaves(grad)
    sq = float(sum(float(np.sum(np.asarray(leaf, dtype=np.float64) ** 2)) for leaf in leaves))
    return {
        "sq_norm": sq,
        "l2_norm": float(np.sqrt(sq)),
        "max_abs": max(float(np.max(np.abs(np.asarray(leaf)))) for leaf in leaves),
        "finite_leaves": float(sum(1 for leaf in leaves if bool(np.all(np.isfinite(np.asarray(leaf)))))),
        "total_leaves": float(len(leaves)),
    }


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_the_fused_grad_stats_equal_a_plain_reference(dtype):
    rng = np.random.default_rng(0)
    grad = {
        "w": jnp.asarray(rng.normal(size=(4, 5)), dtype=dtype),
        "b": jnp.asarray(rng.normal(size=(7,)) * 10.0, dtype=dtype),
        "deep": {"k": jnp.asarray(rng.normal(size=(3, 2, 2)), dtype=dtype)},
    }
    got = exp03.grad_stats(grad)
    want = _reference_grad_stats(grad)
    for key in ("sq_norm", "l2_norm", "max_abs"):
        assert got[key] == pytest.approx(want[key], rel=1e-5), key
    assert got["finite_leaves"] == want["finite_leaves"] == 3.0
    assert got["total_leaves"] == want["total_leaves"] == 3.0

    # A non-finite leaf is counted, not hidden.
    poisoned = {**grad, "b": grad["b"].at[0].set(jnp.asarray(float("nan"), dtype=dtype))}
    assert exp03.grad_stats(poisoned)["finite_leaves"] == 2.0

    # The empty-tree edge does not raise.
    empty = exp03.grad_stats({})
    assert empty["total_leaves"] == 0.0 and empty["sq_norm"] == 0.0
    # DELIBERATE convention change from the old reducer's -1.0 seed: an empty gradient has no
    # magnitude, and -1 would read as a real measurement in the artifact.
    assert empty["max_abs"] == 0.0 and empty["l2_norm"] == 0.0
    assert empty["finite_leaves"] == 0.0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_the_fused_dot_and_cosine_equal_plain_references(dtype):
    rng = np.random.default_rng(1)
    left = {"w": jnp.asarray(rng.normal(size=(6, 3)), dtype=dtype)}
    right = {"w": jnp.asarray(rng.normal(size=(6, 3)), dtype=dtype)}
    reference_dot = float(np.sum(np.asarray(left["w"], np.float64) * np.asarray(right["w"], np.float64)))
    assert exp03.tree_dot(left, right) == pytest.approx(reference_dot, rel=1e-3 if dtype is jnp.bfloat16 else 1e-5)

    reference_cosine = reference_dot / (
        float(np.linalg.norm(np.asarray(left["w"], np.float64)))
        * float(np.linalg.norm(np.asarray(right["w"], np.float64)))
    )
    tolerance = 1e-2 if dtype is jnp.bfloat16 else 1e-5
    assert exp03.grad_cosine(left, right) == pytest.approx(reference_cosine, rel=tolerance)
    # Supplying the squared norms (the caller already has them) must not change the answer.
    stats_left, stats_right = exp03.grad_stats(left), exp03.grad_stats(right)
    assert exp03.grad_cosine(
        left, right, left_sq_norm=stats_left["sq_norm"], right_sq_norm=stats_right["sq_norm"]
    ) == pytest.approx(exp03.grad_cosine(left, right), rel=1e-6)
    # A degenerate operand gives nan rather than a division blow-up.
    assert math.isnan(exp03.grad_cosine(left, {"w": jnp.zeros_like(right["w"])}))


def test_the_replay_caps_resident_gradient_trees_and_reports_the_peak():
    state, data, config, scheduler = _toy_state_and_batches(config_shape="proxy")
    view = s1_5.state_view(config, "checkpoint")
    report = exp03.exp03_frozen_replay(
        state, data[0], jax.random.key(3), view, scheduler, global_step=jnp.asarray(7, jnp.int32), include_control=True
    )
    # THE contract, measured rather than asserted in a comment: control + A + B is the high-water
    # mark, and it lands in the artifact so a run's own JSON shows whether it held.
    assert report["grad_trees_peak_resident"] == 3.0
    for name in ("control", "a", "b", "c"):
        assert f"grad_norm_{name}" in report and f"grad_max_abs_{name}" in report
        assert f"grad_finite_leaves_{name}" in report
    for name in ("a", "b", "c"):
        assert f"grad_cosine_{name}_vs_control" in report
    assert "grad_cosine_ab" in report


def test_the_report_peak_reaches_the_probe_artifact():
    # The auditability claim: the cap is readable from the run's own JSON.
    state, batches, config, scheduler = _toy_state_and_batches(config_shape="proxy")
    report = s1_5.state_report(
        state,
        batches,
        jax.random.key(1),
        config,
        scheduler,
        state_label="checkpoint",
        checkpoint_step=10000,
        support_salts=(1, 2),
    )
    peaks = [row["grad_trees_peak_resident"] for row in report["per_batch"]]
    assert peaks and max(peaks) <= 3.0
    assert report["per_objective"]["grad_trees_peak_resident"]["max"] <= 3.0


def test_the_replay_values_are_unchanged_by_the_memory_fix():
    # A REGRESSION reference implemented inline (not imported from the old code): the same losses
    # and gradient statistics, computed the previous way -- separate eager grads, plain traversals.
    state, data, config, scheduler = _toy_state_and_batches(config_shape="namespace")
    batch = data[0]
    rng = jax.random.key(3)
    global_step = jnp.asarray(7, jnp.int32)
    view = s1_5.state_view(config, "checkpoint")
    report = exp03.exp03_frozen_replay(
        state, batch, rng, view, scheduler, global_step=global_step, include_control=True
    )

    references = {"control": parent._denoising_loss, **exp03.EXP03_LOSSES}
    name_of = {"control": "control", "corrective_ss": "a", "rollout_loss": "b", "combined": "c"}
    grads = {}
    for objective, loss_fn in references.items():
        kwargs = {} if objective == "control" else {"global_step": global_step}

        def _loss(params, fn=loss_fn, kw=kwargs):
            return fn(params, state, batch, rng, view, scheduler, **kw)[0]

        short = name_of[objective]
        assert report[f"loss_{short}"] == pytest.approx(float(_loss(state.params)), rel=1e-6)
        grads[short] = jax.grad(_loss)(state.params)
        want = _reference_grad_stats(grads[short])
        assert report[f"grad_norm_{short}"] == pytest.approx(want["l2_norm"], rel=1e-5)
        assert report[f"grad_max_abs_{short}"] == pytest.approx(want["max_abs"], rel=1e-5)
        assert report[f"grad_finite_leaves_{short}"] == want["finite_leaves"]

    def _reference_cosine(left, right):
        dot = sum(
            float(np.sum(np.asarray(x, np.float64) * np.asarray(y, np.float64)))
            for x, y in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right))
        )
        norms = np.sqrt(_reference_grad_stats(left)["sq_norm"]) * np.sqrt(_reference_grad_stats(right)["sq_norm"])
        return dot / norms

    for short in ("a", "b", "c"):
        assert report[f"grad_cosine_{short}_vs_control"] == pytest.approx(
            _reference_cosine(grads[short], grads["control"]), rel=1e-5
        )
    assert report["grad_cosine_ab"] == pytest.approx(_reference_cosine(grads["a"], grads["b"]), rel=1e-5)


def test_the_welford_update_is_compiled_and_matches_a_plain_reference():
    # Behavioural: the jitted carry update must equal a plain Welford over the same stream.
    rng = np.random.default_rng(4)
    stream = [{"w": jnp.asarray(rng.normal(size=(5,)), dtype=jnp.float32)} for _ in range(6)]

    accumulator = s1_5._TreeWelford()
    for tree in stream:
        accumulator.update(tree)

    vectors = np.stack([np.asarray(tree["w"], dtype=np.float64) for tree in stream])
    reference_mean = vectors.mean(axis=0)
    reference_m2 = float(np.sum((vectors - reference_mean) ** 2))
    assert np.allclose(np.asarray(accumulator.mean["w"], dtype=np.float64), reference_mean, atol=1e-5)
    assert accumulator.m2 == pytest.approx(reference_m2, rel=1e-4)
    assert accumulator.population_variance() == pytest.approx(reference_m2 / len(stream), rel=1e-4)
    # ...and it goes through the compiled update with a donated carry.
    import ast

    tree = ast.parse(Path(s1_5.__file__).read_text())
    welford = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_welford_update"
    )
    decorators = [ast.unparse(node) for node in welford.decorator_list]
    assert any("jax.jit" in text and "donate_argnums" in text for text in decorators), decorators
    assert "_welford_update(" in inspect.getsource(s1_5._TreeWelford.update)


def test_no_eager_whole_tree_gradient_remains_in_the_probe_or_replay():
    # Kept as a cheap static net; the EXECUTABLE evidence is the both-states test below.
    import ast

    for module in (s1_5, exp03):
        tree = ast.parse(Path(module.__file__).read_text())
        bare = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "grad"
            and getattr(node.func.value, "id", "") == "jax"
        ]
        assert not bare, (module.__name__, bare)
    source = inspect.getsource(exp03._loss_and_grad_fn)
    assert "jax.jit(jax.value_and_grad(_call, has_aux=True))" in source
    assert "_LOSS_AND_GRAD_CACHE" in source


# =============================================================================================
# 11. BOTH states, EXECUTED back to back: the cache-collision regression, the compilation count,
# and the measured live-buffer high-water. The sixth inspect-only instance was a test that
# AST-matched `jax.grad` spellings and ran nothing; this one runs the thing.
# =============================================================================================


def _run_both_states(support_salts=(1, 2)):
    """Run the checkpoint state then the init state through the real ``state_report``."""
    state, batches, config, scheduler = _toy_state_and_batches(config_shape="proxy")
    reports = {}
    for label in ("checkpoint", "init"):
        reports[label] = s1_5.state_report(
            state,
            batches,
            jax.random.key(1),
            config,
            scheduler,
            state_label=label,
            checkpoint_step=s1_5.S1_5_STATE_PLAN[label]["required_checkpoint_step"],
            support_salts=support_salts,
        )
    return reports


def test_the_two_states_get_their_own_ramp_and_not_each_others():
    # THE regression pin for the cache collision. The states differ ONLY in ramp origin, and at
    # these global steps that difference decides A's branch: the checkpoint state runs at
    # global_step 10000.. with origin 10000 (elapsed 0 -> p_ss 0 -> teacher-forced), while init runs
    # at 0.. with origin 0 (same elapsed, same p_ss) -- so the discriminating quantity is p_ss
    # itself, which must be computed from EACH state's origin. Under the old tag-only cache the init
    # state reused the checkpoint's compiled closures and would have been evaluated with elapsed
    # 0 - 10000, clamping p_ss and flipping A's branch.
    _, _, config, _ = _toy_state_and_batches(config_shape="proxy")
    ramp = int(config.exp03_p_ss_ramp_steps)
    p_max = float(config.exp03_p_ss_max)
    for label in ("checkpoint", "init"):
        plan = s1_5.S1_5_STATE_PLAN[label]
        other = s1_5.S1_5_STATE_PLAN["checkpoint" if label == "init" else "init"]
        # Mid-ramp is where the two origins visibly disagree: the first batch sits at elapsed 0 for
        # BOTH (p_ss 0), so a first-step comparison would prove nothing.
        step = plan["first_global_step"] + ramp // 2
        correct = SimpleNamespace(
            exp03_p_ss_max=p_max, exp03_p_ss_ramp_steps=ramp, exp03_ramp_origin=plan["ramp_origin"]
        )
        crossed = SimpleNamespace(
            exp03_p_ss_max=p_max, exp03_p_ss_ramp_steps=ramp, exp03_ramp_origin=other["ramp_origin"]
        )
        assert float(exp03.exp03_p_ss(correct, step)) == pytest.approx(p_max / 2, abs=1e-6)
        assert float(exp03.exp03_p_ss(crossed, step)) != pytest.approx(p_max / 2, abs=1e-6), label

    reports = _run_both_states()
    # The reports carry each state's own origin and step range -- not the other's.
    assert reports["checkpoint"]["ramp_origin"] == 10000 and reports["init"]["ramp_origin"] == 0
    assert reports["checkpoint"]["first_global_step"] == 10000
    assert reports["init"]["first_global_step"] == 0
    for label in ("checkpoint", "init"):
        for row in reports[label]["per_batch"]:
            elapsed = row["global_step"] - reports[label]["ramp_origin"]
            expected = min(p_max, p_max * max(0.0, elapsed) / ramp) if ramp > 0 else p_max
            # THE cache-served observable: computed from the traced origin the jitted probe
            # functions receive. The replay's own p_ss_a never had the collision, so asserting on it
            # proved nothing -- this is the field that moves when a state runs with the other
            # state's origin.
            assert row["p_ss_from_threaded_origin"] == pytest.approx(expected, abs=1e-6), (
                label,
                row["global_step"],
            )
            assert row["p_ss_a"] == pytest.approx(expected, abs=1e-6), (label, row["global_step"])

    # ...and the two states genuinely disagree somewhere in the batch range, or the pin is vacuous.
    checkpoint_values = [row["p_ss_from_threaded_origin"] for row in reports["checkpoint"]["per_batch"]]
    init_values = [row["p_ss_from_threaded_origin"] for row in reports["init"]["per_batch"]]
    crossed = [min(p_max, p_max * max(0.0, row["global_step"] - 10000) / ramp) for row in reports["init"]["per_batch"]]
    assert init_values != crossed, "init would look identical under the checkpoint's origin"
    assert checkpoint_values == pytest.approx(init_values), "both states ramp from their own origin"


def test_the_probe_compiles_each_function_once_across_both_states():
    # REAL specializations, read from each jitted wrapper's own ``_cache_size()`` -- not a count of
    # wrapper constructions, which is bookkeeping and stays right even when the function underneath
    # is eager. A collision regression shows up as cache_size 2; an eager fallback shows up as a
    # wrapper with no cache at all.
    s1_5._PROBE_GRAD_CACHE.clear()
    s1_5.PROBE_COMPILATIONS.clear()
    exp03._LOSS_AND_GRAD_CACHE.clear()
    _run_both_states(support_salts=(1, 2))

    def _cache_size(fn):
        target = getattr(fn, "jitted", fn)
        sizer = getattr(target, "_cache_size", None)
        assert callable(sizer), f"{fn} is not a jitted function -- an eager fallback would look like this"
        return sizer()

    probe_sizes = {key: _cache_size(fn) for key, fn in s1_5._PROBE_GRAD_CACHE.items()}
    assert probe_sizes, "no probe gradient was compiled at all"
    for key, size in probe_sizes.items():
        # ONE specialization each, serving BOTH states -- which is only correct because the per-state
        # knob is traced. Two would mean the states are specializing separately again.
        assert size == 1, (key, size)

    replay_sizes = {}
    for key, (grad_fn, value_fn) in exp03._LOSS_AND_GRAD_CACHE.items():
        replay_sizes[key] = (_cache_size(grad_fn), _cache_size(value_fn))
        assert replay_sizes[key][0] == 1, (key, replay_sizes[key])

    helpers = {
        "grad_stats": exp03._jit_grad_stats,
        "tree_vdot": exp03._jit_tree_vdot,
        "welford_update": s1_5._welford_update,
        "welford_first": s1_5._welford_first,
        "relative_gap": s1_5._jit_relative_gap,
        "observed_p_ss": s1_5._jit_observed_p_ss,
    }
    helper_sizes = {name: _cache_size(fn) for name, fn in helpers.items()}
    for name, size in helper_sizes.items():
        assert size >= 1, (name, size)

    total = sum(probe_sizes.values()) + sum(size for size, _ in replay_sizes.values()) + sum(helper_sizes.values())
    assert total > 0
    # The probe side: 7 tags compiled once each, plus one per (objective, salt) variance draw.
    assert len(probe_sizes) == 7 + 4 * 2, sorted(probe_sizes)


def test_the_measured_grad_tree_peak_is_recorded_and_capped():
    # MEASURED from jax.live_arrays(), counted in GRAD-SHAPED buffers rather than bytes -- so the
    # "at most three resident gradient trees" contract is assertable at toy scale exactly as it is
    # at 5B, where a byte threshold would be the only thing that worked.
    reports = _run_both_states()
    for label, report in reports.items():
        assert report["param_leaves"] >= 1.0, label
        assert report["gauge_samples"] >= 8, (label, report["gauge_samples"])
        # The sample points must include the ones INSIDE peak windows; sampling after a release
        # measures the trough.
        points = set(report["gauge_sample_points"])
        assert {"baseline", "parity_peak", "forced_peak", "after_release"} <= points, sorted(points)
        assert any(point.startswith("replay_") for point in points), sorted(points)
        assert any(point == "variance_draw" for point in points), sorted(points)
        # THE contract: at most three gradient trees co-resident anywhere in the probe.
        # EXACT, not a bound: at toy scale the probe's peak is the replay's three-gradient moment,
        # and a bound of "<= 3" would sit still while a stale loop variable added a fourth tree
        # somewhere else. The mutation that removes a `del` moves this number.
        assert report["grad_trees_peak_measured"] == pytest.approx(3.0), (
            label,
            report["grad_trees_peak_measured"],
            report["gauge_sample_points"],
        )
        # The baseline is recorded, not dropped.
        assert report["gauge_baseline"]["where"] == "baseline"
        # THE stale-loop-variable detector: at the top of each variance draw nothing from the
        # previous iteration may still be resident, so only Welford's running mean is grad-shaped.
        entries = [reading for reading in report["gauge_readings"] if reading["where"] == "variance_draw_entry"]
        assert entries, "the entry sample point is missing"
        # The floor here is the accumulator's own honest footprint -- Welford's running mean plus
        # the buffer the donated update just produced -- measured at 2 trees. What must NOT happen
        # is growth beyond it: a gradient the previous iteration failed to release adds a third.
        assert max(reading["grad_trees"] for reading in entries) <= 2.0, [
            reading for reading in entries if reading["grad_trees"] > 2.0
        ][:3]
        # The replay's own counter agrees with the measurement.
        assert report["per_objective"]["grad_trees_peak_resident"]["max"] <= 3.0


def test_the_welford_update_is_transactional_and_correct():
    # The alias theory is WITHDRAWN: a Python alias does not inhibit JAX donation -- donation is
    # declared at the jit boundary and a surviving reference raises on later use rather than
    # silently declining. What the earlier dance did leave was a real bug: count was incremented
    # before the call, so a raising update left the accumulator claiming an observation it never
    # absorbed. State now advances only after the call returns.
    source = inspect.getsource(s1_5._TreeWelford.update)
    assert "mean, self.mean = self.mean, None" not in source  # the dance is gone
    assert source.index("_welford_update(") < source.index("self.count = next_count")

    accumulator = s1_5._TreeWelford()
    accumulator.update({"w": jnp.asarray([1.0], dtype=jnp.float32)})
    before = (accumulator.count, accumulator.m2)
    with pytest.raises(Exception):
        accumulator.update({"v": jnp.asarray([1.0], dtype=jnp.float32)})  # different tree structure
    assert (accumulator.count, accumulator.m2) == before  # nothing advanced
    assert accumulator.mean is not None  # ...and the mean was not left as None

    rng = np.random.default_rng(11)
    stream = [{"w": jnp.asarray(rng.normal(size=(4,)), dtype=jnp.float32)} for _ in range(5)]
    accumulator = s1_5._TreeWelford()
    for tree in stream:
        accumulator.update(tree)
    vectors = np.stack([np.asarray(tree["w"], np.float64) for tree in stream])
    assert np.allclose(np.asarray(accumulator.mean["w"], np.float64), vectors.mean(axis=0), atol=1e-5)
    assert accumulator.m2 == pytest.approx(float(np.sum((vectors - vectors.mean(axis=0)) ** 2)), rel=1e-4)


def test_the_gauge_counts_grad_shaped_buffers_not_the_params():
    params = {"big": jnp.zeros((16, 16), dtype=jnp.float32), "small": jnp.zeros((3,), dtype=jnp.float32)}
    gauge = s1_5.LiveBufferGauge(params)
    assert gauge.num_param_leaves == 2
    baseline = gauge.sample("baseline")
    # The parameters themselves are excluded, so a fresh gauge sees no gradient trees.
    assert baseline["grad_shaped_buffers"] == 0 and baseline["grad_trees"] == 0.0

    # One full "gradient tree": one buffer of each parameter leaf's shape and dtype.
    grad = {"big": jnp.ones((16, 16), dtype=jnp.float32), "small": jnp.ones((3,), dtype=jnp.float32)}
    reading = gauge.sample("one_tree")
    assert reading["grad_shaped_buffers"] == 2 and reading["grad_trees"] == pytest.approx(1.0)
    second = {"big": jnp.full((16, 16), 2.0, dtype=jnp.float32), "small": jnp.full((3,), 2.0, dtype=jnp.float32)}
    assert gauge.sample("two_trees")["grad_trees"] == pytest.approx(2.0)
    del grad, second
    import gc

    gc.collect()
    assert gauge.sample("released")["grad_trees"] == pytest.approx(0.0)
    # ...and the high-water mark survives the release, which is the point of a gauge.
    assert gauge.report()["grad_trees_peak_measured"] == pytest.approx(2.0)
    assert gauge.report()["gauge_baseline"]["where"] == "baseline"
