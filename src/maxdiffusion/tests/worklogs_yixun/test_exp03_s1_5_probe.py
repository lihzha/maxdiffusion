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
import json
import math
import re
import shutil
import subprocess
import sys
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
    assert "trace_forward_for(state, replicas=replicas, state_label=state_label)" in tracer
    assert "_restore" not in tracer and "run_trace" not in tracer
    # The merge moved INTO the shared compiled forward, which is the point: there the parameters are
    # a traced ARGUMENT, where closing over the merged module would have baked ~5B weights into the
    # executable as literals. It is still the state already in memory -- no second restore, no
    # second load -- it is simply crossing a jit boundary now.
    builder = inspect.getsource(s1_5.trace.jitted_tiled_forward)
    assert "gen.nnx.merge(graphdef, params, st.rest_of_state)" in builder
    assert "def _forward(params, st, hidden_states, timestep, encoder_hidden_states):" in builder


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
    assert "release(state, batches, report, iterator)" in per_state
    assert per_state.index("trainer._load_wan_pipeline()") < per_state.index("release(state,")
    # ...and the pipeline itself is dropped BEFORE the heavy phase, not merely at the end of it.
    assert "del pipeline" in per_state
    assert per_state.index("del pipeline") < per_state.index("state_report(")
    assert "trace_in_memory_state(\n" in per_state and "pipeline," not in per_state.split("del pipeline")[1]


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
    # ...and it goes through the compiled update with a donated carry AND a pinned carry sharding.
    # The decorator moved into ``welford_fns`` when the accumulators had to be pinned: an unpinned
    # float32 parameter-shaped carry is ~20 GB per chip at 5B if XLA replicates it, which is what it
    # did to the gradients in Job 8e.
    builder = inspect.getsource(s1_5.welford_fns)
    assert "donate_argnums=(0,)" in builder, builder
    assert "out_shardings=(shardings, None)" in builder, builder
    assert "out_shardings=shardings" in builder, builder
    assert "_welford_update(" in inspect.getsource(s1_5._TreeWelford.update)
    # The compiled pair really is jitted, and really is keyed by the sharding tree.
    first, update = s1_5.welford_fns(s1_5.tree_shardings({"w": jnp.zeros((2,), jnp.float32)}))
    assert s1_5.jit_cache_size(first) >= 0 and s1_5.jit_cache_size(update) >= 0
    again = s1_5.welford_fns(s1_5.tree_shardings({"w": jnp.zeros((2,), jnp.float32)}))
    assert again[0] is first and again[1] is update  # cached per sharding tree, not per accumulator


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
    assert "jax.value_and_grad(_call, has_aux=True)" in source
    assert "jax.jit(" in source
    # ...and the gradient's LAYOUT is pinned, not left to XLA (Job 8e).
    assert "out_shardings=None if grad_shardings is None else ((None, None), grad_shardings)" in source
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
        origin = reports[label]["ramp_origin"]
        for row in reports[label]["per_batch"]:
            elapsed = row["global_step"] - origin
            expected = min(p_max, p_max * max(0.0, elapsed) / ramp) if ramp > 0 else p_max
            # THE pin, and it is EMITTED rather than recomputed: these two fields come out of the
            # cache-served gradient function's own aux, computed inside it from the origin it was
            # traced with. The previous version of this assertion read a parallel jitted
            # recomputation standing beside the cache, which is why the review's mutation -- force
            # the origin only inside cache-served calls -- left it green.
            assert row["p_ss_cache_served"] == pytest.approx(expected, abs=1e-6), (label, row["global_step"])
            assert row["ramp_elapsed_cache_served"] == pytest.approx(elapsed, abs=1e-6), (label, row["global_step"])
            # The replay's own p_ss, kept as a SECONDARY cross-check: its cache is keyed per state
            # view, so it never had the collision and can never be the pin.
            assert row["p_ss_a"] == pytest.approx(expected, abs=1e-6), (label, row["global_step"])
        # ...and EVERY cache-served function, not just A's: each one emitted the elapsed ramp it
        # ran with, and each must be this state's.
        emitted = reports[label]["cache_served_ramp"]
        assert emitted, label
        tags = {row["tag"] for row in emitted}
        assert tags >= {
            "parity_trial",
            "parity_comparator",
            "parity_production_control",
            "isolation_corrective",
            "isolation_same_eps",
            "forced_corrective_ss",
            "forced_combined",
        }, sorted(tags)
        # ...including every (objective, salt) variance compilation, each of which is its own tag.
        for objective in s1_5.S1_5_OBJECTIVES:
            for salt in (1, 2):
                assert s1_5.variance_tag(objective, salt) in tags, (objective, salt, sorted(tags))
        for row in emitted:
            assert row["ramp_elapsed"] == pytest.approx(row["global_step"] - origin, abs=1e-6), (label, row)

    # ...and the two states genuinely disagree somewhere in the batch range, or the pin is vacuous.
    checkpoint_values = [row["p_ss_cache_served"] for row in reports["checkpoint"]["per_batch"]]
    init_values = [row["p_ss_cache_served"] for row in reports["init"]["per_batch"]]
    crossed = [min(p_max, p_max * max(0.0, row["global_step"] - 10000) / ramp) for row in reports["init"]["per_batch"]]
    assert init_values != crossed, "init would look identical under the checkpoint's origin"
    assert checkpoint_values == pytest.approx(init_values), "both states ramp from their own origin"


def _force_origin_inside_cache_served_calls(monkeypatch, forced: int = 10000):
    """THE review's surgical mutation: the cached gradients run with a foreign origin, nothing else.

    It touches neither the report fields nor the replay -- only the value the ``_PROBE_GRAD_CACHE``
    functions are called with. That is precisely the defect a collision would produce, and it is
    what a pin reading anything other than those functions' own output cannot see.
    """
    original = s1_5._jitted_grad

    def _mutated(tag, builder, *, static_key=(), grad_shardings=None):
        compiled = original(tag, builder, static_key=static_key, grad_shardings=grad_shardings)

        def _forced(params, state, data, rng, global_step, ramp_origin):
            del ramp_origin
            return compiled(params, state, data, rng, global_step, jnp.asarray(forced, jnp.int32))

        return _forced

    monkeypatch.setattr(s1_5, "_jitted_grad", _mutated)


def test_the_collision_pin_fails_under_the_surgical_origin_mutation(monkeypatch):
    # The pin above, EXECUTED against the defect it exists for. A regression test that survives its
    # own mutation is decoration; this one is run under the mutation here, in the suite, so the
    # claim "it would catch a collision" is a result rather than an assurance.
    state, batches, config, scheduler = _toy_state_and_batches(config_shape="proxy")
    ramp, p_max = int(config.exp03_p_ss_ramp_steps), float(config.exp03_p_ss_max)
    _force_origin_inside_cache_served_calls(monkeypatch)
    report = s1_5.state_report(
        state,
        batches,
        jax.random.key(1),
        config,
        scheduler,
        state_label="init",  # origin 0; the mutation forces 10000, so the init state is the victim
        checkpoint_step=0,
        support_salts=(1, 2),
    )
    violations = []
    for row in report["per_batch"]:
        expected = min(p_max, p_max * max(0.0, row["global_step"]) / ramp)
        if row["p_ss_cache_served"] != pytest.approx(expected, abs=1e-6):
            violations.append(("p_ss", row["global_step"], row["p_ss_cache_served"], expected))
        if row["ramp_elapsed_cache_served"] != pytest.approx(row["global_step"], abs=1e-6):
            violations.append(("elapsed", row["global_step"], row["ramp_elapsed_cache_served"]))
    assert violations, (
        "the mutation ran the cached gradients at origin 10000 and the emitted evidence did not "
        "move -- the pin is reading something other than what the cache-served functions computed"
    )
    # Concretely: the emitted elapsed is the foreign origin's, at every step and every tag.
    assert all(row["ramp_elapsed"] == pytest.approx(row["global_step"] - 10000) for row in report["cache_served_ramp"])
    # ...while the replay's own p_ss is untouched, which is exactly why it could never be the pin.
    assert report["per_batch"][1]["p_ss_a"] == pytest.approx(min(p_max, p_max / ramp), abs=1e-6)


def test_the_probe_compiles_each_function_once_across_both_states():
    # REAL specializations, read from each jitted wrapper's own ``_cache_size()`` -- not a count of
    # wrapper constructions, which is bookkeeping and stays right even when the function underneath
    # is eager. A collision regression shows up as cache_size 2; an eager fallback shows up as a
    # wrapper with no cache at all (``jit_cache_size`` raises on it).
    s1_5._PROBE_GRAD_CACHE.clear()
    s1_5.PROBE_COMPILATIONS.clear()
    s1_5._RELEASED_SPECIALIZATIONS.clear()
    s1_5._RELEASED_REPLAY_NAMES.clear()
    _reset_welford_jit()  # pairs built by earlier tests carry other trees' shardings and counts
    exp03._LOSS_AND_GRAD_CACHE.clear()
    exp03.COMPILE_TIMINGS.clear()
    # The helpers are module-level and their XLA caches outlive any single run, so the rest of this
    # file's toy shapes would otherwise be counted into "this run's" census. Cleared here, which is
    # also the only way this total means what a fresh production process would report.
    for helper in s1_5.probe_jit_helpers().values():
        helper.clear_cache()
    # The PRODUCTION support draws (M=4), so the census below is the production shape rather than a
    # cheaper stand-in: K does not change the count, M does.
    reports = _run_both_states(support_salts=s1_5.S1_5_SUPPORT_SALTS)

    census = s1_5.specialization_census()
    for tag, size in census["probe"].items():
        # ONE per state. It used to be one FULL STOP -- the traced ramp origin makes a single
        # compilation correct for both states -- and releasing executables gives that up on purpose:
        # a program dropped in the first state to free its executable must compile again in the
        # second. Three would mean a program was released while still in use.
        assert size == 2, (tag, size)
    # The probe side: 7 single tags plus one per (objective, salt) variance draw.
    assert len(census["probe"]) == 7 + 4 * len(s1_5.S1_5_SUPPORT_SALTS), sorted(census["probe"])
    # The replay caches per objective PER STATE VIEW (its closure captures the view), so four
    # objectives across two states is eight entries, each compiled once. Its value-only twin is
    # never called in the probe, so it contributes zero.
    assert len(census["replay"]) == 8, sorted(census["replay"])
    assert set(census["replay"].values()) == {1}, census["replay"]
    assert set(census["helpers"]) == {"grad_stats", "tree_vdot", "welford_update", "welford_first", "relative_gap"}
    for name, size in census["helpers"].items():
        assert size >= 1, (name, size)  # a helper at zero is a helper that never ran

    # THE executed total for this run: 23 probe tags x 2 states + 8 replay + 5 helpers, at the toy's
    # float32 parameters. Production runs bfloat16 (``weights_dtype: 'bfloat16'``) and the dtype mix
    # splits THREE helpers rather than one -- ``grad_stats`` (bfloat16 gradients and the float32
    # Welford mean), ``welford_first`` (a bfloat16 gradient starts a within-batch mean, a float32
    # mean starts the between-batch one) and ``welford_update`` (float32 carry over a bfloat16 tree,
    # and over a float32 tree) -- so the production total is 46 + 8 + 8 = 62. That split is executed
    # in ``test_the_production_dtype_mix_splits_three_helpers``; the artifact carries whichever
    # number the run actually measured.
    assert census["total"] == 46 + 8 + 5 == 59, census
    assert census["total"] == census["probe_total"] + census["replay_total"] + census["helper_total"]
    # ...and it reaches the artifact, from the run itself. Each state's census is the snapshot at the
    # moment that state's JSON is written: the checkpoint state has compiled and released its own 23
    # plus 4 replay (+5 helpers = 32), the init state's report sees both states' (46 + 8 + 5 = 59).
    assert reports["checkpoint"]["specializations"]["probe_total"] == 23
    assert reports["init"]["specializations"]["probe_total"] == 46
    assert reports["checkpoint"]["specializations"]["total"] == 32
    assert reports["init"]["specializations"]["total"] == census["total"]

    # The driver's runtime refusal, executed both ways: silent on the real census, loud on a tag
    # that compiled more often than the release discipline can explain. That is the assertion that
    # fires on hardware, where no test is watching.
    s1_5.assert_specializations_within_release_budget(census)
    with pytest.raises(RuntimeError) as excinfo:
        s1_5.assert_specializations_within_release_budget({**census, "probe": {**census["probe"], "parity_trial": 3}})
    assert "parity_trial" in str(excinfo.value)

    # The compile-cost log is per (tag, STATE): the init state's first call is its own entry, not
    # the checkpoint state's. Probe tags are timed too, which they were not before.
    keys = set(exp03.COMPILE_TIMINGS)
    assert all(isinstance(key, tuple) and len(key) == 2 for key in keys), sorted(keys)[:3]
    assert {label for _, label in keys} == {"checkpoint", "init"}
    assert ("replay_a", "checkpoint") in keys and ("replay_a", "init") in keys
    assert ("probe_parity_trial", "checkpoint") in keys and ("probe_parity_trial", "init") in keys
    assert all(seconds >= 0.0 for seconds in exp03.COMPILE_TIMINGS.values())

    # COVERAGE: one timing entry per specialization per state, no compilation unmeasured. The four
    # salts of a variance objective are four compilations and now four tags; sharing one tag left
    # twelve of the sixteen variance compilations per state with no entry at all.
    # Taken from the CENSUS, not from the live cache: every probe program has been released by now,
    # so the live cache is empty and only the census still knows they existed.
    assert not s1_5._PROBE_GRAD_CACHE, "the release discipline should have emptied the probe cache"
    expected_tags = {f"probe_{label.split('(')[0]}" for label in census["probe"]}
    assert len(expected_tags) == len(census["probe"]) == 23
    for label in ("checkpoint", "init"):
        timed = {tag for tag, state in keys if state == label}
        missing = expected_tags - timed
        assert not missing, (label, sorted(missing))
        # ...and nothing beyond them but the four replay objectives this state compiled.
        assert len(timed) == len(expected_tags) + 4, (label, sorted(timed - expected_tags))


def test_an_eager_wrapper_cannot_pass_as_a_compiled_one():
    # The census is only evidence if it refuses to count something that is not jitted at all.
    assert s1_5.jit_cache_size(s1_5.welford_fns()[0]) >= 0
    with pytest.raises(TypeError):
        s1_5.jit_cache_size(lambda x: x)


def test_the_measured_grad_tree_peak_is_recorded_and_capped():
    # MEASURED from jax.live_arrays() by buffer IDENTITY, in GRADIENT-TREE EQUIVALENTS (new bytes
    # over one parameter tree's bytes) -- so the "at most three resident gradient trees" contract is
    # assertable at toy scale exactly as at 5B, and a float32 accumulator over bfloat16 parameters
    # reads as the two tree-equivalents it actually costs instead of vanishing.
    reports = _run_both_states()
    for label, report in reports.items():
        assert report["param_leaves"] >= 1.0, label
        assert report["one_grad_tree_bytes"] >= 1.0, label
        assert report["gauge_samples"] >= 8, (label, report["gauge_samples"])
        # The sample points must include the ones INSIDE peak windows; sampling after a release
        # measures the trough.
        points = set(report["gauge_sample_points"])
        assert {"baseline", "parity_peak", "forced_peak", "after_release"} <= points, sorted(points)
        assert any(point.startswith("replay_") for point in points), sorted(points)
        assert any(point == "variance_draw" for point in points), sorted(points)
        # THE contract: at most three gradient trees co-resident anywhere in the probe. The
        # tolerance is for the scalars (losses, sigmas, the emitted ramp aux) that ride along --
        # 56 bytes against a 256-byte toy tree; a fourth gradient tree is +1.0 and cannot hide in
        # it. The mutation that removes a `del` moves this number.
        assert report["grad_tree_equivalents_peak"] == pytest.approx(3.0, abs=0.5), (
            label,
            report["grad_tree_equivalents_peak"],
            report["gauge_sample_points"],
        )
        assert report["grad_tree_equivalents_peak"] >= 3.0, label  # ...and the three ARE resident
        # The secondary, exact because it counts buffers rather than bytes: three parameter-shaped
        # arrays new since baseline, no more.
        assert report["new_param_shaped_peak"] == 3.0, (label, report["new_param_shaped_peak"])
        # The identity mechanism did not silently degrade to object identity anywhere.
        assert report["gauge_identity_fallbacks"] == 0.0, label
        assert report["gauge_baseline_buffers"] >= 1.0, label
        # The baseline is recorded, not dropped -- and by construction it is empty, because it is
        # the snapshot everything else is measured against.
        assert report["gauge_baseline"]["where"] == "baseline"
        assert report["gauge_baseline"]["grad_tree_equivalents"] == pytest.approx(0.0, abs=1e-9), label
        # THE stale-loop-variable detector: at the top of each variance draw nothing from the
        # previous iteration may still be resident.
        entries = [reading for reading in report["gauge_readings"] if reading["where"] == "variance_draw_entry"]
        assert entries, "the entry sample point is missing"
        # The floor here is the decomposition's own honest footprint -- the within-batch mean and
        # the between-batch mean. What must NOT happen is growth beyond it: a gradient the previous
        # iteration failed to release adds a third.
        assert max(reading["grad_tree_equivalents"] for reading in entries) <= 2.5, [
            reading for reading in entries if reading["grad_tree_equivalents"] > 2.5
        ][:3]
        assert max(reading["new_param_shaped"] for reading in entries) <= 2.0, label
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
    assert "welford_fns(tree_shardings(tree))" in source  # ...and the carry is pinned, not inferred

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


def test_the_gauge_counts_what_the_probe_made_resident_not_what_it_inherited():
    params = {"big": jnp.zeros((16, 16), dtype=jnp.float32), "small": jnp.zeros((3,), dtype=jnp.float32)}
    gauge = s1_5.LiveBufferGauge(params)
    assert gauge.num_param_leaves == 2
    assert gauge.one_tree_bytes == 16 * 16 * 4 + 3 * 4
    baseline = gauge.sample("baseline")
    # The parameters are in the baseline snapshot, so a fresh gauge reads nothing new.
    assert baseline["new_buffers"] == 0.0 and baseline["grad_tree_equivalents"] == 0.0

    # One full "gradient tree": one buffer of each parameter leaf's shape and dtype.
    grad = {"big": jnp.ones((16, 16), dtype=jnp.float32), "small": jnp.ones((3,), dtype=jnp.float32)}
    reading = gauge.sample("one_tree")
    assert reading["grad_tree_equivalents"] == pytest.approx(1.0)
    assert reading["new_param_shaped"] == 2.0
    second = {"big": jnp.full((16, 16), 2.0, dtype=jnp.float32), "small": jnp.full((3,), 2.0, dtype=jnp.float32)}
    assert gauge.sample("two_trees")["grad_tree_equivalents"] == pytest.approx(2.0)
    del grad, second
    import gc

    gc.collect()
    assert gauge.sample("released")["grad_tree_equivalents"] == pytest.approx(0.0)
    # ...and the high-water mark survives the release, which is the point of a gauge.
    assert gauge.report()["grad_tree_equivalents_peak"] == pytest.approx(2.0)
    assert gauge.report()["gauge_baseline"]["where"] == "baseline"
    assert gauge.report()["gauge_identity_fallbacks"] == 0.0


def test_adam_moments_alive_at_baseline_read_as_zero_not_as_two_gradient_trees():
    # INVERSION 1, the review's own case run backwards. It executed the shape-matching gauge against
    # a state that had restored its optimizer and read TWO resident gradient trees before a single
    # gradient existed -- AdamW's mu and nu are parameter-shaped. Identity, not shape, is what tells
    # a phantom from a gradient.
    import optax

    params = {"big": jnp.zeros((16, 16), dtype=jnp.float32), "small": jnp.zeros((3,), dtype=jnp.float32)}
    opt_state = optax.adamw(1e-4).init(params)
    moments = jax.tree_util.tree_leaves(opt_state)
    assert len(moments) >= 2 * len(jax.tree_util.tree_leaves(params)), "no moment trees to be fooled by"
    # The shape-and-dtype rule the old gauge used, applied here so the inversion is a comparison
    # rather than a claim about code that no longer exists.
    param_shapes = {(tuple(leaf.shape), str(leaf.dtype)) for leaf in jax.tree_util.tree_leaves(params)}
    phantoms = [leaf for leaf in moments if (tuple(leaf.shape), str(leaf.dtype)) in param_shapes]
    assert len(phantoms) == 4, phantoms  # exactly the two phantom "trees" the review measured

    gauge = s1_5.LiveBufferGauge(params)  # baseline taken WITH the moments alive
    assert gauge.sample("baseline")["grad_tree_equivalents"] == 0.0
    assert gauge.sample("still_nothing_happened")["grad_tree_equivalents"] == 0.0
    assert gauge.report()["grad_tree_equivalents_peak"] == 0.0
    assert jax.tree_util.tree_leaves(opt_state), "the moments must still be alive, or nothing was proved"


def test_float32_welford_buffers_over_bfloat16_params_are_counted():
    # INVERSION 2. Production parameters are bfloat16 (``weights_dtype: 'bfloat16'``) while the
    # Welford accumulator is float32 by construction, so under the old (shape, dtype) rule the one
    # buffer class most likely to break the memory budget matched NO parameter leaf and was counted
    # as zero. It costs two tree-equivalents, and the gauge must say so.
    params = {"w": jnp.zeros((16, 16), dtype=jnp.bfloat16), "b": jnp.zeros((16,), dtype=jnp.bfloat16)}
    gauge = s1_5.LiveBufferGauge(params)
    gauge.sample("baseline")
    old_rule_matches = {(tuple(leaf.shape), str(leaf.dtype)) for leaf in jax.tree_util.tree_leaves(params)}

    # The shardings come from the PARAMS (already alive at baseline); the tree being absorbed is a
    # temporary, so only the float32 accumulator it produces is new when the gauge samples.
    welford_first = s1_5.welford_fns(s1_5.tree_shardings(params))[0]
    welford_mean = welford_first({"w": jnp.ones((16, 16), jnp.bfloat16), "b": jnp.ones((16,), jnp.bfloat16)})
    leaves = jax.tree_util.tree_leaves(welford_mean)
    assert {str(leaf.dtype) for leaf in leaves} == {"float32"}  # the accumulator really is wider
    assert not any((tuple(leaf.shape), str(leaf.dtype)) in old_rule_matches for leaf in leaves), "the old rule would"

    reading = gauge.sample("welford_resident")
    assert reading["new_param_shaped"] == 2.0, reading  # SHAPE-matched, dtype recorded not required
    assert reading["new_param_shaped_by_dtype"] == {"float32": 2.0}, reading
    # float32 over bfloat16 is exactly twice the parameter tree's bytes.
    assert reading["grad_tree_equivalents"] == pytest.approx(2.0), reading
    assert welford_mean is not None  # the buffers were alive for the whole measurement


# =============================================================================================
# 12. The optimizer the probe never uses. A no-update probe restores AdamW's moments through the
# checkpoint's own layout and then carries two parameter-shaped float32 trees -- ~40 GB at 5B --
# that nothing will ever read.
# =============================================================================================


def _toy_state_with_adam_moments():
    """The toy state, rebuilt on AdamW so its optimizer state really has moment trees."""
    import optax

    state, batches, config, scheduler = _toy_state_and_batches(config_shape="proxy")
    tx = optax.adamw(1e-4)
    state = state.replace(tx=tx, opt_state=tx.init(state.params))
    assert jax.tree_util.tree_leaves(state.opt_state), "the fixture must actually carry moments"
    return state, batches, config, scheduler


def test_the_moments_are_dropped_after_the_exact_step_restore_and_the_buffers_go():
    # ORDER is the contract: the restore needs the optimizer tree as its target structure and the
    # required-step check is what makes this the right state at all, so the drop happens after both
    # and before anything measures memory.
    source = inspect.getsource(s1_5.build_probe_state)
    assert source.index("restore_exact_step(manager, state, required=required)") < source.index(
        "drop_optimizer_moments(state)"
    )
    assert source.index("if int(start_step) != int(required):") < source.index("drop_optimizer_moments(state)")
    assert source.index("drop_optimizer_moments(state)") < source.index("return state, state_shardings")
    # ...and the restore itself is untouched: it still restores the optimizer it was handed.
    restorer = inspect.getsource(s1_5.restore_exact_step)
    assert "opt_state=ocp.args.StandardRestore(state.opt_state)" in restorer
    assert 'opt_state=restored["opt_state"]' in restorer

    # EXECUTED: the moments go, the parameters do not, and the buffers are actually released.
    import gc

    def _pairs(arrays):
        found = set()
        for array in arrays:
            found |= set(s1_5.buffer_shards(array)[0])
        return found

    state, _, _, _ = _toy_state_with_adam_moments()
    moments = jax.tree_util.tree_leaves(state.opt_state)
    moment_ids = _pairs(moments)
    param_ids = _pairs(jax.tree_util.tree_leaves(state.params))
    assert moment_ids and not (moment_ids & param_ids)
    del moments

    state = s1_5.drop_optimizer_moments(state)
    gc.collect()
    assert jax.tree_util.tree_leaves(state.opt_state) == []
    assert state.opt_state == s1_5.EMPTY_OPT_STATE
    live = _pairs(jax.live_arrays())
    assert not (moment_ids & live), "the moment buffers are still resident"
    assert param_ids <= live, "the parameters were released along with the moments"
    # It stays a TrainState, with everything the probe does read.
    assert isinstance(state, parent.Overfit100TrainState)
    assert state.context_table is not None and state.graphdef is not None
    # ...and dropping an already-empty optimizer is a no-op rather than an error.
    assert s1_5.drop_optimizer_moments(state) is state


def test_nothing_downstream_reads_the_optimizer_state():
    # The claim "the moments are dead weight" is only true if the measurements do not depend on
    # them. Run the WHOLE report both ways and compare, rather than asserting it in a comment.
    state, batches, config, scheduler = _toy_state_with_adam_moments()
    kwargs = {"state_label": "checkpoint", "checkpoint_step": 10000, "support_salts": (1,)}
    with_moments = s1_5.state_report(state, batches, jax.random.key(1), config, scheduler, **kwargs)
    dropped = s1_5.drop_optimizer_moments(state)
    without = s1_5.state_report(dropped, batches, jax.random.key(1), config, scheduler, **kwargs)

    for index, (left, right) in enumerate(zip(with_moments["per_batch"], without["per_batch"])):
        assert set(left) == set(right), index
        for key in left:
            assert left[key] == pytest.approx(right[key], rel=1e-6, abs=1e-9), (index, key)
    assert without["p_ss_zero_parity"]["passes"] == with_moments["p_ss_zero_parity"]["passes"]
    assert without["label_isolation"]["grad_cosine"] == pytest.approx(
        with_moments["label_isolation"]["grad_cosine"], rel=1e-6
    )
    for objective in s1_5.S1_5_OBJECTIVES:
        assert without["support_variance"][objective]["support_variance"] == pytest.approx(
            with_moments["support_variance"][objective]["support_variance"], rel=1e-6, abs=1e-12
        ), objective
    # ...and the residency measurement is the same too: the identity gauge was never fooled by the
    # moments, so dropping them is a memory saving rather than a change of what is being measured.
    assert without["grad_tree_equivalents_peak"] == pytest.approx(with_moments["grad_tree_equivalents_peak"], abs=0.5)


def test_the_production_dtype_mix_splits_three_helpers():
    # WHY the production census is 39 and the toy's is 36, executed rather than asserted in a
    # comment. Production parameters are bfloat16 while every accumulator is float32, and the mix
    # reaches THREE helpers, not one. This models the production call pattern at toy size: the
    # variance decomposition over bfloat16 gradients, then the statistics the replay takes on them.
    _reset_welford_jit()  # so the counts below are this test's, not the whole file's
    for helper in s1_5.probe_jit_helpers().values():
        helper.clear_cache()

    def _tree(value, dtype=jnp.bfloat16):
        return {"w": jnp.full((4,), value, dtype=dtype)}

    # Two batches x two draws, exactly as the probe streams them.
    s1_5.variance_decomposition((_tree(1.0 + index + draw) for draw in range(2)) for index in range(2))
    exp03.grad_stats(_tree(1.0))  # the replay's per-gradient statistics, on a bfloat16 gradient
    exp03.grad_cosine(_tree(1.0), _tree(2.0))
    s1_5._relative_gradient_gap(_tree(1.0), _tree(2.0))

    sizes = {name: s1_5.jit_cache_size(fn) for name, fn in s1_5.probe_jit_helpers().items()}
    # bfloat16 gradients AND the float32 means: two specializations each.
    assert sizes["grad_stats"] == 2, sizes
    assert sizes["welford_first"] == 2, sizes
    assert sizes["welford_update"] == 2, sizes
    # These only ever see gradients, so they stay at one.
    assert sizes["tree_vdot"] == 1, sizes
    assert sizes["relative_gap"] == 1, sizes
    assert sum(sizes.values()) == 8, sizes
    # 23 probe + 8 replay + 8 helpers. The float32 toy reads 36; the difference is exactly the
    # three helpers above.
    assert 23 + 8 + sum(sizes.values()) == 39


# The 8-device inversion runs in a SUBPROCESS: the device count is fixed when the backend
# initialises, so it cannot be changed inside a session that has already used jax.
_EIGHT_DEVICE_SCRIPT = """
import json, sys, types

_grain = types.ModuleType("grain")
_grain_python = types.ModuleType("grain.python")
_grain_python.MapTransform = type("MapTransform", (), {})
_grain_python.RandomAccessDataSource = type("RandomAccessDataSource", (), {})
_grain.python = _grain_python
sys.modules["grain"] = _grain
sys.modules["grain.python"] = _grain_python

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import maxdiffusion.probe_exp03_s1_5 as s1_5

assert jax.device_count() == 8, jax.device_count()
mesh = Mesh(jax.devices(), ("fsdp",))
sharding = NamedSharding(mesh, PartitionSpec("fsdp"))
params = {"w": jax.device_put(jnp.zeros((64,), jnp.float32), sharding)}

gauge = s1_5.LiveBufferGauge(params)
gauge.sample("baseline")
grad = {"w": jax.device_put(jnp.ones((64,), jnp.float32), sharding)}
reading = gauge.sample("one_sharded_gradient")

# What summing over ARRAY OBJECTS reports -- the defect, measured in the same process.
live = [a for a in jax.live_arrays() if not a.is_deleted()]
baseline_pairs = gauge.baseline_ids
naive_bytes = 0
objects = 0
for array in live:
    pairs, _ = s1_5.buffer_shards(array)
    if any(pair not in baseline_pairs for pair in pairs):
        naive_bytes += int(array.nbytes)
        objects += 1

print("RESULT " + json.dumps({
    "devices": jax.device_count(),
    "shards": len(grad["w"].addressable_shards),
    "one_tree_bytes": gauge.one_tree_bytes,
    "tree_equivalents": reading["grad_tree_equivalents"],
    "new_bytes": reading["new_bytes"],
    "new_buffers": reading["new_buffers"],
    "param_shaped": reading["new_param_shaped"],
    "naive_object_sum_bytes": naive_bytes,
    "naive_object_count": objects,
    "fallbacks": gauge.identity_fallbacks,
}))
"""


def test_one_sharded_gradient_over_eight_devices_reads_one_tree_not_two():
    # THE blocker, inverted and EXECUTED. On an 8-device mesh a single sharded gradient is reachable
    # as the global array and as its eight shard arrays, so anything summing ``array.nbytes`` over
    # live arrays double-counts it -- 2.0 tree-equivalents for one gradient, which would have made
    # the residency cap meaningless on every real (v6e-8) run. Deduplicated by physical
    # (device, pointer) allocation, one gradient is one tree.
    import os

    env = dict(os.environ)
    env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = str(_REPO / "src")
    proc = subprocess.run(
        [sys.executable, "-c", _EIGHT_DEVICE_SCRIPT], capture_output=True, text=True, timeout=600, env=env
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(line for line in proc.stdout.splitlines() if line.startswith("RESULT "))
    result = json.loads(line[len("RESULT ") :])

    assert result["devices"] == 8 and result["shards"] == 8, result
    # The denominator is the parameter tree's own physical bytes, measured the same way: 64 float32
    # elements spread over eight devices is still 256 bytes.
    assert result["one_tree_bytes"] == 256, result
    assert result["new_bytes"] == 256, result
    assert result["new_buffers"] == 8, result  # eight physical allocations, one logical gradient
    assert result["tree_equivalents"] == pytest.approx(1.0), result
    assert result["param_shaped"] == 1, result  # credited to the global array, not to its shards
    assert result["fallbacks"] == 0, result
    # ...and the defect is real in this very process: summing over array objects reports double.
    assert result["naive_object_count"] == 9, result  # the global array plus its eight shards
    assert result["naive_object_sum_bytes"] == 512, result
    assert result["naive_object_sum_bytes"] / result["one_tree_bytes"] == pytest.approx(2.0), result


# =============================================================================================
# 13. The dead weight, and the ledger that survives a job dying. Job 8d reached the replay at 5B
# and then failed to LOAD A's program: 15.11G of scratch wanted against 9.50G free (E0101), with
# no artifact written because the failure came before one existed.
#
# The 5B reclaim itself cannot be shown in CI -- there is no TPU here and no 5B pipeline. What is
# executed here is the mechanism: the references really are dropped, the buffers really do go, and
# the ledger lines really are emitted and well formed. The reclaim is proven by the 8e stdout
# ledger (post_state_creation / post_restore / post_moment_drop / post_dead_weight_free / pre_replay).
# =============================================================================================


class _FakePipeline:
    """A pipeline shaped like the real one where it matters: plain attributes, one hidden alias.

    ``vae_cache.module`` is the real ``AutoencoderKLWanCache`` behaviour (``self.module = module``),
    which is why dropping ``vae`` alone frees nothing and the drop order is not cosmetic.
    """

    def __init__(self):
        self.vae = {"w": jnp.zeros((32, 32), jnp.float32)}
        self.vae_cache = SimpleNamespace(module=self.vae)
        self.text_encoder = {"w": jnp.zeros((48, 48), jnp.float32)}
        self.tokenizer = SimpleNamespace(vocab={"a": 1})
        self.video_processor = SimpleNamespace()
        self.transformer = {"w": jnp.zeros((64, 64), jnp.float32)}
        self.scheduler = SimpleNamespace(name="flow-match")
        self.scheduler_state = SimpleNamespace(step=0)
        self.mesh = SimpleNamespace(shape="toy")
        self.config = SimpleNamespace(run_name="toy")


def _live_pairs():
    found = set()
    for array in jax.live_arrays():
        found |= set(s1_5.buffer_shards(array)[0])
    return found


def _pairs_of(tree):
    found = set()
    for leaf in jax.tree_util.tree_leaves(tree):
        found |= set(s1_5.buffer_shards(leaf)[0])
    return found


def test_the_probe_frees_the_encode_time_models_and_their_buffers_go():
    # EXECUTED with the same buffer-identity machinery as the moment-release test: the attributes go
    # AND the device buffers behind them stop being live.
    import gc

    pipeline = _FakePipeline()
    vae_pairs = _pairs_of(pipeline.vae)
    encoder_pairs = _pairs_of(pipeline.text_encoder)
    transformer_pairs = _pairs_of(pipeline.transformer)
    assert vae_pairs and encoder_pairs and transformer_pairs
    assert vae_pairs <= _live_pairs() and encoder_pairs <= _live_pairs()

    dropped = s1_5.free_encode_time_models(pipeline)
    gc.collect()
    assert dropped == ["vae_cache", "vae", "text_encoder", "tokenizer", "video_processor"], dropped
    for attr in ("vae", "vae_cache", "text_encoder", "tokenizer", "video_processor"):
        assert not hasattr(pipeline, attr), attr
    live = _live_pairs()
    assert not (vae_pairs & live), "the VAE buffers are still resident"
    assert not (encoder_pairs & live), "the text-encoder buffers are still resident"
    # What the probe still needs is untouched -- the scheduler above all, which every objective uses.
    assert pipeline.scheduler.name == "flow-match" and pipeline.scheduler_state.step == 0
    assert pipeline.mesh.shape == "toy" and pipeline.config.run_name == "toy"
    assert transformer_pairs <= live, "the transformer is not encode-time and must survive this call"
    # Idempotent: a second call finds nothing and does not raise.
    assert s1_5.free_encode_time_models(pipeline) == []


def test_the_probe_drops_the_pipelines_own_copy_of_the_transformer():
    # The pipeline's module aliases the state's parameters at ``nnx.split`` time, and becomes a
    # SECOND ~5B tree the moment Orbax swaps new arrays in. Training donates those buffers away on
    # step 0; a no-update probe never donates anything, so it drops them by name.
    import gc

    pipeline = _FakePipeline()
    transformer_pairs = _pairs_of(pipeline.transformer)
    assert transformer_pairs <= _live_pairs()
    assert s1_5.free_pipeline_transformer(pipeline) is True
    gc.collect()
    assert not hasattr(pipeline, "transformer")
    assert not (transformer_pairs & _live_pairs()), "the pipeline's transformer buffers are still resident"
    assert s1_5.free_pipeline_transformer(pipeline) is False  # nothing left to drop

    # ...and the probe does both, in the only order that works: the context table is built while the
    # encoder is alive, the split happens while the transformer is, and each dies immediately after.
    source = inspect.getsource(s1_5.build_probe_state)
    assert source.index("_build_context_table") < source.index("free_encode_time_models(pipeline)")
    assert source.index("free_encode_time_models(pipeline)") < source.index("nnx.split(pipeline.transformer")
    assert source.index("nnx.split(pipeline.transformer") < source.index("free_pipeline_transformer(pipeline)")


def test_the_memory_ledger_lines_are_well_formed(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(s1_5.max_logging, "log", lambda line: lines.append(str(line)))

    gib = 1024**3
    fake = SimpleNamespace(
        memory_stats=lambda: {
            "bytes_in_use": 21 * gib,
            "peak_bytes_in_use": 23 * gib,
            "bytes_limit": 31 * gib,
            "something_else": 5,
        }
    )
    monkeypatch.setattr(s1_5.jax, "local_devices", lambda: [fake])
    stats = s1_5.log_memory("post_restore", state_label="checkpoint")
    # The runtime's own three numbers survive verbatim; the ledger adds what it can attribute to
    # arrays and the difference it cannot.
    for key, value in (("bytes_in_use", 21 * gib), ("peak_bytes_in_use", 23 * gib), ("bytes_limit", 31 * gib)):
        assert stats[key] == value, key
    assert "something_else" not in stats
    assert stats["unattributed_bytes"] == stats["bytes_in_use"] - stats["array_bytes"]
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("[exp03][mem] where=post_restore state=checkpoint"), line
    assert "in_use=21.00G" in line and "peak=23.00G" in line and "limit=31.00G" in line, line
    assert "arrays=" in line and "unattributed=" in line, line

    # A backend that reports nothing (CPU returns None) still emits a line and never raises -- and
    # still reports the array half, which needs no device statistics at all.
    lines.clear()
    monkeypatch.setattr(s1_5.jax, "local_devices", lambda: [SimpleNamespace(memory_stats=lambda: None)])
    assert s1_5.log_memory("pre_replay") == {"array_bytes": 0}  # no device -> nothing on that chip
    assert lines and "no device memory stats" in lines[0] and "where=pre_replay" in lines[0]

    # ...and so does a backend that raises when asked.
    lines.clear()

    def _boom():
        raise RuntimeError("no stats here")

    monkeypatch.setattr(s1_5.jax, "local_devices", lambda: [SimpleNamespace(memory_stats=_boom)])
    assert s1_5.memory_snapshot() == {}
    assert "bytes_in_use" not in s1_5.log_memory("post_moment_drop")
    assert lines and "no device memory stats" in lines[0]


def test_the_headroom_check_names_the_largest_live_arrays_and_never_raises(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(s1_5.max_logging, "log", lambda line: lines.append(str(line)))
    gib = 1024**3
    hog = jnp.zeros((256, 256), jnp.float32)  # the biggest thing this test makes

    # Resolved BEFORE the patch: asking for it inside the fake would re-enter the patched
    # ``local_devices`` and recurse.
    device_name = s1_5.chip_device_key()

    class _FakeDevice:
        """Named as the device the arrays are really on — the residue listing is keyed by device."""

        def __init__(self, in_use):
            self._in_use = in_use

        def __str__(self):
            return device_name

        def memory_stats(self):
            return {"bytes_in_use": self._in_use}

    monkeypatch.setattr(s1_5.jax, "local_devices", lambda: [_FakeDevice(21 * gib)])
    assert s1_5.warn_if_no_headroom("pre_replay", state_label="checkpoint", budget_bytes=8 * gib) is True
    warnings = [line for line in lines if "WARNING" in line]
    assert warnings, lines
    assert "21.00G already in use at pre_replay" in warnings[0], warnings[0]
    assert any("largest live arrays" in line for line in lines), lines
    # The residue is NAMED: dtype and shape, at per-chip size.
    assert any("float32[256, 256]" in line for line in lines), [line for line in lines if "#1" in line]

    # Under the budget: the ledger line still lands, the warning does not.
    lines.clear()
    monkeypatch.setattr(s1_5.jax, "local_devices", lambda: [_FakeDevice(2 * gib)])
    assert s1_5.warn_if_no_headroom("pre_replay", budget_bytes=8 * gib) is False
    assert lines and not any("WARNING" in line for line in lines), lines

    # A backend with no stats cannot warn, and must not crash the probe on the way past.
    lines.clear()
    monkeypatch.setattr(s1_5.jax, "local_devices", lambda: [SimpleNamespace(memory_stats=lambda: None)])
    assert s1_5.warn_if_no_headroom("pre_replay", budget_bytes=1) is False
    assert hog is not None  # kept alive across the whole measurement


def test_the_largest_live_arrays_report_local_and_global_bytes():
    # Sorted by LOCAL bytes, because ``bytes_in_use`` is per device and the two numbers have to be
    # comparable; the global size rides along so a sharded tree is still recognisable.
    keep = jnp.zeros((512, 512), jnp.float32)  # 1 MiB, larger than anything else this test holds
    rows = s1_5.largest_live_arrays(limit=5)
    assert rows and len(rows) <= 5
    assert rows[0]["local_bytes"] >= 512 * 512 * 4
    assert rows[0]["shape"] == (512, 512) and rows[0]["dtype"] == "float32"
    assert rows[0]["global_bytes"] == 512 * 512 * 4
    assert all(rows[index]["local_bytes"] >= rows[index + 1]["local_bytes"] for index in range(len(rows) - 1))
    assert keep is not None


def test_the_heavy_phase_logs_the_ledger_before_re_raising():
    # A failure in the heavy phase must leave the ledger and the residue behind it -- that is the
    # whole point of the instrumentation -- and must then re-raise untouched.
    source = inspect.getsource(s1_5._run_one_state)
    assert "except Exception as failure:" in source
    body = source[source.index("except Exception as failure:") :]
    # Through the BEST-EFFORT wrapper, never the raw calls: this path runs with an exception in
    # flight and anything that can raise here can replace the failure everyone needs to see.
    assert "log_diagnostics_best_effort(" in body
    assert "log_largest_live_arrays(" not in body and "log_memory(" not in body
    assert body.rstrip().count("raise") >= 1 and "raise\n" in body
    # The ledger points the 8e run must carry, each emitted from the path that reaches them.
    build = inspect.getsource(s1_5.build_probe_state)
    for where in ("post_encode_time_free", "post_state_creation", "post_restore", "post_moment_drop"):
        assert f'log_memory("{where}"' in build, where
    assert 'log_memory("post_dead_weight_free"' in source
    report_source = inspect.getsource(s1_5.state_report)
    assert 'warn_if_no_headroom("pre_replay"' in report_source
    # ...and the points that show what each program costs and whether releasing it gives anything
    # back: one per objective as it loads, and one after every release.
    assert 'log_memory(f"post_replay_{name}"' in report_source
    for where in ('"post_replay_release"', 'f"post_variance_{objective}_release"', '"post_parity_release"'):
        assert f"log_memory({where}" in report_source, where
    # ...and none of it reaches the immutable artifact, for the same reason the timings do not.
    assert "mem" not in s1_5.s1_5_artifact(
        SimpleNamespace(run_name="r", checkpoint_dir="", train_data_dir="", model_manifest_path="", seed=0),
        state_label="checkpoint",
        checkpoint_step=10000,
        report={},
    )


# =============================================================================================
# 14. Executable residency: the memory `jax.live_arrays()` cannot see. A loaded 5B program and its
# scratch are gigabytes and no array accounts for them, which is why the ledger reports
# `unattributed = in_use - arrays` and why programs are dropped once their last call is behind us.
# =============================================================================================


def _reset_welford_jit():
    """Drop the compiled Welford pair AND the cache it shares with every other pair.

    jax keys its pjit cache by the wrapped FUNCTION rather than by the wrapper: two
    ``jax.jit(_welford_first_impl)`` share entries, and clearing either clears both. Dropping the
    cache dict alone would leave a freshly built pair reporting earlier tests' specializations.
    """
    s1_5._WELFORD_JIT.clear()
    first, update = s1_5.welford_fns(s1_5.tree_shardings({"w": jnp.zeros((1,), jnp.float32)}))
    first.clear_cache()
    update.clear_cache()
    s1_5._WELFORD_JIT.clear()


def test_the_walk_order_does_not_change_a_single_number():
    # The premise of ANY reordering of the replay walk: the per-(batch, objective) losses and
    # gradients must not depend on the order the loops are nested in. They do not -- every RNG the
    # objectives consume is a pure function of (seed, global_step, purpose), never of a stream that
    # advances as the walk proceeds -- and this executes that claim rather than asserting it.
    state, batches, config, scheduler = _toy_state_and_batches(config_shape="proxy")
    view = s1_5.state_view(config, "checkpoint")
    rng = jax.random.key(1)
    first_step = s1_5.S1_5_STATE_PLAN["checkpoint"]["first_global_step"]

    def _one(index, objective):
        """One (batch, objective) cell, computed in isolation from any walk."""
        global_step = jnp.asarray(first_step + index, jnp.int32)
        loss_fn = parent._denoising_loss if objective == "control" else exp03.EXP03_LOSSES[objective]
        kwargs = {} if objective == "control" else {"global_step": global_step}
        value, _ = loss_fn(
            state.params, state, batches[index], jax.random.fold_in(rng, index), view, scheduler, **kwargs
        )
        return float(value)

    # Order A: batch outer, objective inner (what the probe does today).
    order_a = {}
    for index in range(len(batches)):
        for objective in s1_5.S1_5_OBJECTIVES:
            order_a[(index, objective)] = _one(index, objective)
    # Order B: objective outer, batch inner (what per-objective executable release would require).
    order_b = {}
    for objective in s1_5.S1_5_OBJECTIVES:
        for index in range(len(batches)):
            order_b[(index, objective)] = _one(index, objective)

    assert order_a.keys() == order_b.keys()
    for key in order_a:
        # BIT for bit, not approximately: nothing in the loss path carries state across cells.
        assert order_a[key] == order_b[key], key
    # ...and the cells really are different from one another, or the equality above is vacuous.
    assert len(set(order_a.values())) > 1, order_a


def test_releasing_a_probe_program_drops_the_wrapper_and_keeps_the_count():
    # What a release can promise is that the probe holds no reference: the cache entry goes and the
    # jitted wrapper -- whose C++ cache owns the executable -- becomes collectable. Whether XLA then
    # hands the memory back is NOT observable from here (jax exposes no live-executable ledger), and
    # the ledger lines around each release are what answer that on hardware.
    import gc
    import weakref

    s1_5._PROBE_GRAD_CACHE.clear()
    s1_5._RELEASED_SPECIALIZATIONS.clear()
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
    # Every probe program is released by the end of a state -- nothing is left loaded.
    assert not s1_5._PROBE_GRAD_CACHE, sorted(s1_5._PROBE_GRAD_CACHE)
    # ...and the census still knows they existed, which is the whole point of snapshotting first.
    assert report["specializations"]["probe_total"] == 7 + 4 * 2
    assert all(size == 1 for size in report["specializations"]["probe"].values())

    # The wrapper really does become collectable once the cache entry is gone.
    s1_5._PROBE_GRAD_CACHE.clear()
    compiled = s1_5._jitted_grad("weakref_probe", lambda origin: (lambda *a: (jnp.asarray(0.0), {})))
    reference = weakref.ref(compiled)
    del compiled
    s1_5._PROBE_GRAD_CACHE.pop(("weakref_probe", ()))
    gc.collect()
    assert reference() is None, "the jitted wrapper is still reachable after its cache entry was dropped"


def test_the_release_points_are_where_the_programs_die():
    # STRUCTURAL, because the ordering is the contract: a program may only be dropped after its last
    # call, and the variance salts are re-entered on every batch, so their release belongs after the
    # objective's decomposition rather than inside the batch loop.
    source = inspect.getsource(s1_5.state_report)
    assert source.index("release_replay_programs()") > source.index("exp03.exp03_frozen_replay(")
    assert source.index("release_replay_programs()") < source.index("for objective in S1_5_OBJECTIVES:")
    assert source.index("variance_decomposition(_draws") < source.index("release_probe_programs(*[variance_tag")
    for tag in ('f"isolation_{label}"', '"parity_trial"', '"parity_comparator"', '"parity_production_control"'):
        assert f"release_probe_programs({tag})" in source, tag
    assert 'release_probe_programs(f"forced_{objective}")' in source
    # The trial's gradient outlives its program: the release sits between the call and the report
    # that consumes the gradient.
    assert source.index('release_probe_programs("parity_trial")') < source.index("parity = parity_report(")


def test_the_ledger_separates_arrays_from_the_executables_it_cannot_see(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(s1_5.max_logging, "log", lambda line: lines.append(str(line)))
    gib = 1024**3
    keep = jnp.zeros((1024, 1024), jnp.float32)  # 4 MiB of honestly attributable array

    arrays = s1_5.chip_array_bytes()
    assert arrays >= 4 * 1024 * 1024, arrays

    class _FakeDevice:
        """Reports stats, and NAMES ITSELF as the device the arrays are really on.

        The name matters: the ledger's array half is keyed by device, so a fake that called itself
        something else would report the arrays as belonging to another chip and read zero.
        """

        def __init__(self, name):
            self._name = name

        def __str__(self):
            return self._name

        def memory_stats(self):
            return {"bytes_in_use": arrays + 3 * gib, "bytes_limit": 31 * gib}

    device_name = s1_5.chip_device_key()
    monkeypatch.setattr(s1_5.jax, "local_devices", lambda: [_FakeDevice(device_name)])
    stats = s1_5.log_memory("post_replay_control", state_label="checkpoint")
    # THE quantity the 8d post-mortem turned on: what is resident that no array explains.
    assert stats["array_bytes"] == arrays
    assert stats["unattributed_bytes"] == 3 * gib
    assert "unattributed=3.00G" in lines[0], lines[0]
    assert "arrays=" in lines[0] and "where=post_replay_control" in lines[0]
    assert keep is not None


def test_the_per_chip_accounting_does_not_multiply_by_the_mesh():
    # The previous version summed every addressable shard (all eight chips) and counted the same
    # buffer again through each shard wrapper. Both are fixed by keying on one device and
    # deduplicating physical allocations, so this number is comparable with ``bytes_in_use``.
    keep = jnp.zeros((1024, 1024), jnp.float32)
    key = s1_5.chip_device_key()
    allocations, owners = s1_5.chip_allocation_table()
    assert allocations, "no allocations found on this chip"
    assert all(pair[0] == key for pair in allocations), "an allocation from another device leaked in"
    assert len(set(allocations)) == len(allocations)  # a dict cannot double-count by construction
    assert s1_5.chip_array_bytes() == sum(allocations.values())
    # Every allocation is credited to exactly one owner array, and the biggest one is ours.
    assert set(owners) == set(allocations)
    rows = s1_5.largest_live_arrays(limit=3)
    assert rows[0]["shape"] == (1024, 1024) and rows[0]["local_bytes"] == 1024 * 1024 * 4
    assert rows[0]["global_bytes"] == 1024 * 1024 * 4
    assert keep is not None


def test_the_failure_diagnostics_can_never_replace_the_failure(monkeypatch):
    # THE blocker: this code runs with an exception in flight. If it raises, the traceback everyone
    # needs -- the OOM -- is replaced by one about the instrumentation.
    lines: list[str] = []
    monkeypatch.setattr(s1_5.max_logging, "log", lambda line: lines.append(str(line)))

    def _explode(*args, **kwargs):
        raise RuntimeError("array inspection blew up")

    monkeypatch.setattr(s1_5, "largest_live_arrays", _explode)
    s1_5.log_diagnostics_best_effort("failure (RuntimeProgramAllocationFailure)", state_label="checkpoint")
    assert any("diagnostics failed" in line and "array inspection blew up" in line for line in lines), lines

    # ...and the same protection on the healthy path, where a warning must not become a failure.
    lines.clear()
    gib = 1024**3
    monkeypatch.setattr(
        s1_5.jax, "local_devices", lambda: [SimpleNamespace(memory_stats=lambda: {"bytes_in_use": 21 * gib})]
    )
    assert s1_5.warn_if_no_headroom("pre_replay", budget_bytes=1) is True
    assert any("diagnostics failed" in line for line in lines), lines

    # The real thing does not raise either, with no monkeypatching at all.
    monkeypatch.undo()
    s1_5.log_diagnostics_best_effort("failure (smoke)")


# =============================================================================================
# 15. THE Job 8e defect: an unpinned jitted gradient comes back REPLICATED. The ledger caught it on
# hardware -- post_replay_control read arrays 11.07G per chip where ~2.9G was expected, with the
# failure top-10 naming weight-shaped arrays whose local size equalled their global size. Every
# chip was holding the whole ~9.5G gradient instead of its 1.25G FSDP shard.
# =============================================================================================

# Run in a subprocess for the same reason as the dedupe inversion: the device count is fixed when
# the backend initialises. The loss shape here is the toy stub's own -- a scalar gain broadcast over
# the input (``jnp.mean(gain) * tanh(x)``) -- which is exactly the shape whose gradient XLA chooses
# to replicate, so the defect is reproduced rather than imagined.
_PIN_SCRIPT = """
import json, sys, types

_grain = types.ModuleType("grain")
_grain_python = types.ModuleType("grain.python")
_grain_python.MapTransform = type("MapTransform", (), {})
_grain_python.RandomAccessDataSource = type("RandomAccessDataSource", (), {})
_grain.python = _grain_python
sys.modules["grain"] = _grain
sys.modules["grain.python"] = _grain_python

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import maxdiffusion.probe_exp03_s1_5 as s1_5

assert jax.device_count() == 8, jax.device_count()
mesh = Mesh(jax.devices(), ("fsdp",))
sharded = NamedSharding(mesh, PartitionSpec("fsdp"))
params = {"w": jax.device_put(jnp.ones((64, 8), jnp.float32), sharded)}
data = jax.device_put(jnp.ones((64, 8), jnp.float32), sharded)


def loss(p, d):
    # THE toy stub's shape: a scalar gain broadcast over the input.
    return jnp.sum(jnp.mean(p["w"]) * jnp.tanh(d)), {"p_ss_used": jnp.float32(0.5)}


def local0(array):
    key = str(jax.devices()[0])
    return sum(int(s.data.nbytes) for s in array.addressable_shards if str(s.device) == key)


shardings = s1_5.tree_shardings(params)
vg = jax.value_and_grad(loss, has_aux=True)
(_, _), unpinned = jax.jit(vg)(params, data)
(_, _), pinned = jax.jit(vg, out_shardings=((None, None), shardings))(params, data)

# The residency gauge's own reading of each, from a baseline taken before either existed.
def tree_equivalents(tree):
    gauge = s1_5.LiveBufferGauge(params)
    held = jax.tree_util.tree_map(lambda x: x + 0.0, tree)  # a fresh copy, made after the baseline
    reading = gauge.sample("grad")
    del held
    return reading["grad_tree_equivalents"]

print("RESULT " + json.dumps({
    "devices": jax.device_count(),
    "param_global": int(params["w"].nbytes),
    "param_local0": local0(params["w"]),
    "unpinned_global": int(unpinned["w"].nbytes),
    "unpinned_local0": local0(unpinned["w"]),
    "unpinned_spec": str(unpinned["w"].sharding.spec),
    "pinned_global": int(pinned["w"].nbytes),
    "pinned_local0": local0(pinned["w"]),
    "pinned_spec": str(pinned["w"].sharding.spec),
    "welford_local0": local0(s1_5.welford_fns(shardings)[0](pinned)["w"]),
}))
"""


def test_an_unpinned_gradient_replicates_and_the_pin_is_what_stops_it():
    import os

    env = dict(os.environ)
    env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = str(_REPO / "src")
    proc = subprocess.run([sys.executable, "-c", _PIN_SCRIPT], capture_output=True, text=True, timeout=600, env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = json.loads(next(l for l in proc.stdout.splitlines() if l.startswith("RESULT "))[len("RESULT ") :])

    assert result["devices"] == 8, result
    # The parameters are sharded: one eighth of the tree on this chip.
    assert result["param_global"] == 64 * 8 * 4 and result["param_local0"] == result["param_global"] // 8, result
    # WITHOUT the pin the gradient comes back REPLICATED -- local equals global, the exact signature
    # the 8e failure top-10 showed (fp32[3072,18432] with local == global).
    assert result["unpinned_local0"] == result["unpinned_global"], result
    assert result["unpinned_spec"] == "P()", result  # replicated: no axis mentioned at all
    # WITH the pin it is one shard, eight times smaller on this chip, and the values are unchanged
    # (a sharding is a layout, not a computation -- the walk-order test pins the values).
    assert result["pinned_local0"] == result["pinned_global"] // 8, result
    assert result["pinned_spec"] == "P('fsdp',)", result
    assert result["pinned_local0"] * 8 == result["unpinned_local0"], result
    # ...and the Welford accumulator built from the pinned gradient inherits the same one-eighth.
    assert result["welford_local0"] == result["pinned_local0"], result


def test_the_context_table_is_a_replicated_input_and_is_not_duplicated_per_call():
    # The third array in the 8e failure top-10, bf16[100, 512, 4096] at 0.391G, is the CONTEXT
    # TABLE: num_text_slots(100) x wan_max_sequence_length(512) x text_dim(4096) x 2 bytes =
    # 419,430,400 = 0.390625 GiB exactly. It is replicated ON PURPOSE -- every device gathers
    # arbitrary rows of it -- and it is an INPUT, so unlike the gradients it is neither an output
    # XLA got to lay out nor something a pin belongs on. What must be true is that it is not
    # re-materialised on every call.
    assert 100 * 512 * 4096 * 2 == 419430400  # the arithmetic that identifies it
    config_text = _CONFIG.read_text()
    assert "num_text_slots: 100" in config_text and "text_dim: 4096" in config_text
    assert "wan_max_sequence_length: 512" in config_text
    # Replication is a deliberate, documented choice in the state shardings, not an accident.
    shardings_source = inspect.getsource(parent._overfit100_state_shardings)
    assert "replace(context_table=replicated)" in shardings_source
    assert "replicating the ~5B" in shardings_source  # the reason params are NOT replicated

    state, batches, config, scheduler = _toy_state_and_batches(config_shape="proxy")
    table_pairs = set(s1_5.buffer_shards(state.context_table)[0])
    assert table_pairs

    gauge = s1_5.LiveBufferGauge(state.params)  # baseline taken WITH the table alive
    gauge.sample("baseline")
    view = s1_5.state_view(config, "checkpoint")
    for index in range(2):
        row = exp03.exp03_frozen_replay(
            state,
            batches[index],
            jax.random.fold_in(jax.random.key(1), index),
            view,
            scheduler,
            global_step=jnp.asarray(10000 + index, jnp.int32),
            include_control=True,
        )
        assert row["loss_control"] == row["loss_control"]  # finite, i.e. the call really ran
    # The table's own buffers are untouched: same physical allocations, still live.
    live = set()
    for array in jax.live_arrays():
        live |= set(s1_5.buffer_shards(array)[0])
    assert table_pairs <= live, "the context table was re-materialised"
    # ...and nothing of its shape was allocated beside it: exactly one such array is live.
    extra = [row for row in s1_5.largest_live_arrays(limit=20) if row["shape"] == tuple(state.context_table.shape)]
    assert len(extra) == 1, extra
    assert extra[0]["global_bytes"] == int(state.context_table.nbytes)

    # It is not in the gradient tree either: gradients are taken with respect to params alone.
    grad_shardings = s1_5.tree_shardings(state.params)
    assert jax.tree_util.tree_structure(grad_shardings) == jax.tree_util.tree_structure(state.params)
    assert "context_table" not in str(jax.tree_util.tree_structure(state.params))


# =============================================================================================
# 16. THE Job 8f failure: the sigma trace is the only batch-1 forward in the probe, and the
# transformer partitions its activations over ('data','fsdp') = 8. One row cannot be split eight
# ways. Every other phase runs the global batch (eight, one per chip) and 8f proved them at 5B --
# this was simply the first launch ever to reach the trace.
# =============================================================================================

_TRACE_TILING_SCRIPT = '''
import json, sys, types

_grain = types.ModuleType("grain")
_grain_python = types.ModuleType("grain.python")
_grain_python.MapTransform = type("MapTransform", (), {})
_grain_python.RandomAccessDataSource = type("RandomAccessDataSource", (), {})
_grain.python = _grain_python
sys.modules["grain"] = _grain
sys.modules["grain.python"] = _grain_python

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from maxdiffusion.diagnostics_exp03 import sigma_trajectory_trace as trace

assert jax.device_count() == 8, jax.device_count()
mesh = Mesh(jax.devices(), ("fsdp",))
batch_spec = NamedSharding(mesh, PartitionSpec("fsdp", None, None))

WEIGHT = jax.device_put(jnp.linspace(-1.0, 1.0, 3072 * 8, dtype=jnp.float32).reshape(3072, 8), NamedSharding(mesh, PartitionSpec()))


class _Toy:
    """A stand-in for the transformer's shape: a per-token constraint that splits the BATCH axis.

    It reproduces the failure exactly -- ``with_sharding_constraint`` on an (batch, tokens, dim)
    activation whose leading axis is partitioned eight ways -- and it does per-token arithmetic with
    a feature-axis reduction, so a cross-batch mistake would show up in row 0.
    """

    def __call__(self, *, hidden_states, timestep, encoder_hidden_states, deterministic):
        x = hidden_states.astype(jnp.float32)
        x = jax.lax.with_sharding_constraint(x, batch_spec)   # <- the 8f line, in miniature
        x = x + timestep.astype(jnp.float32)[..., None]
        x = x / (jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True)) + 1e-6)  # per-token, like RMSNorm
        x = jnp.einsum("btd,de->bte", x, WEIGHT[: x.shape[-1], :])
        x = x + jnp.mean(encoder_hidden_states.astype(jnp.float32), axis=-1, keepdims=True)
        return jax.lax.with_sharding_constraint(x, batch_spec)


toy = _Toy()
one = jnp.asarray(np.random.default_rng(0).normal(size=(1, 16, 8)), jnp.float32)
timestep = jnp.asarray(np.random.default_rng(1).normal(size=(1, 16)), jnp.float32)
context = jnp.asarray(np.random.default_rng(2).normal(size=(1, 16, 4)), jnp.float32)

# A state shaped like the real one: a PYTREE whose params are traced, with graphdef and
# rest_of_state static. The forward merges them exactly as the 5B one does -- and it has to be a
# pytree, because that is how the real state crosses the jit boundary.
from flax import struct


@struct.dataclass
class _State:
    params: dict
    graphdef: object = struct.field(pytree_node=False, default=None)
    rest_of_state: object = struct.field(pytree_node=False, default=None)


state = _State(params={"unused": jnp.zeros((1,), jnp.float32)})
trace.gen.nnx.merge = lambda graphdef, params, rest: toy  # the merge the real forward performs

result = {"devices": jax.device_count()}

# 1a. THE DEFECT exactly as Job 8f hit it: EAGER, batch 1, against an 8-way batch split. Eager
# means every constraint is its own pjit, and a pjit OUTPUT sharding must divide.
try:
    toy(hidden_states=one, timestep=timestep, encoder_hidden_states=context, deterministic=True)
    result["eager_untiled_error"] = None
except Exception as exc:
    result["eager_untiled_error"] = type(exc).__name__

# 1b. ...and what the same batch-1 shape does once COMPILED. Recorded rather than assumed: inside
# one jit the constraint is an internal annotation that GSPMD can satisfy by replicating a size-1
# dimension, so the error goes away on its own. That is WHY the tiling is not redundant -- it is
# what puts one real row on each chip instead of one row replicated across eight.
try:
    trace.velocity_fn_for(trace.jitted_tiled_forward(state, replicas=1), state)(one, timestep, context)
    result["jitted_untiled_error"] = None
except Exception as exc:
    result["jitted_untiled_error"] = type(exc).__name__

# 2. THE FIX: eight identical rows through ONE COMPILED forward, row 0 read back outside it.
replicas = trace.batch_replicas(
    type("C", (), {"logical_axis_rules": [["batch", ["fsdp"]], ["embed", ["tensor"]]]})(), mesh
)
result["replicas"] = replicas
forward = trace.jitted_tiled_forward(state, replicas=replicas)
velocity = trace.velocity_fn_for(forward, state)
tiled_out = velocity(one, timestep, context)
result["tiled_shape"] = list(tiled_out.shape)
# ...and it really is compiled, and reused rather than recompiled on every step.
for _ in range(4):
    velocity(one, timestep, context)
result["forward_cache_size"] = int(forward._cache_size())

# 3. THE REFERENCE: the same window on ONE device, unsharded, batch 1.
with jax.default_device(jax.devices()[0]):
    plain = jnp.einsum(
        "btd,de->bte",
        (lambda x: x / (jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True)) + 1e-6))(
            one.astype(jnp.float32) + timestep.astype(jnp.float32)[..., None]
        ),
        np.asarray(WEIGHT)[: one.shape[-1], :],
    ) + jnp.mean(context.astype(jnp.float32), axis=-1, keepdims=True)

left = np.asarray(tiled_out, np.float64)
right = np.asarray(plain, np.float64)
result["bitwise_equal"] = bool(np.array_equal(np.asarray(tiled_out), np.asarray(plain)))
result["max_abs_diff"] = float(np.max(np.abs(left - right)))
result["max_rel_diff"] = float(np.max(np.abs(left - right) / (np.abs(right) + 1e-30)))
# ...and every tiled row is the same row, which is what makes reading row 0 meaningful.
raw = forward(state.params, state, one, timestep, context)
result["rows_identical"] = bool(np.array_equal(np.asarray(raw[0]), np.asarray(raw[replicas - 1])))
print("RESULT " + json.dumps(result))
'''


def test_the_trace_forward_is_tiled_to_the_mesh_and_row_zero_is_the_answer():
    # EXECUTED on the forced 8-device mesh: the defect reproduced, the fix applied, and row 0
    # checked against an unsharded batch-1 reference.
    #
    # WHAT THIS IS NOT. The model here is a toy with the right SHAPE -- a batch-partitioned sharding
    # constraint and a per-token feature-axis reduction -- not the real WAN forward under real FSDP.
    # It proves the tiling mechanism and the row-0 slice, and it reproduces the failure mode; it
    # cannot prove that the 5B forward's row 0 is right. The real proof is the 8g run's own trace
    # output: finite, sane, and starting at zero (the trace begins AT the interpolant by
    # construction, so index 0 is arithmetic-free and any drift there is a broken forward).
    import os

    env = dict(os.environ)
    env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = str(_REPO / "src")
    proc = subprocess.run(
        [sys.executable, "-c", _TRACE_TILING_SCRIPT], capture_output=True, text=True, timeout=600, env=env
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = json.loads(next(l for l in proc.stdout.splitlines() if l.startswith("RESULT "))[len("RESULT ") :])

    assert result["devices"] == 8, result
    # 1a. The defect is REAL and reproduced in the mode 8f ran in: EAGER batch 1 against an 8-way
    # batch split raises IndivisibleError.
    assert result["eager_untiled_error"] is not None, result
    assert "Indivisible" in result["eager_untiled_error"] or "Sharding" in result["eager_untiled_error"], result
    # 1b. MEASURED, and worth knowing: compiling alone makes the error go away, because inside one
    # jit the constraint is an annotation GSPMD can satisfy by replicating a size-1 dimension. So
    # the jit is what unblocks the run and the TILING is what makes the batch axis honestly
    # divisible -- one real row per chip. If a future XLA raises here instead, this line says so.
    assert result["jitted_untiled_error"] is None, result
    # 2. The fix runs, and hands the sampler back the single row it passed in.
    assert result["replicas"] == 8, result
    assert result["tiled_shape"][0] == 1, result
    # ONE compilation, reused for every step -- the fix for the eager dispatch storm.
    assert result["forward_cache_size"] == 1, result
    # 3. Row 0 IS the batch-1 answer. Measured rather than assumed: the tolerance below is what the
    # 8-device run actually produced against an unsharded reference, and the assertion records
    # whether it was bitwise or merely floating-point equal.
    assert result["rows_identical"], result  # the tiled rows really are copies of one another
    assert result["max_abs_diff"] <= 1e-5, result
    assert result["max_rel_diff"] <= 1e-5, result
    # MEASURED: on this backend it is not merely within tolerance, it is bitwise. Asserted so that a
    # future XLA that schedules the batch-8 reductions differently announces itself here rather than
    # drifting a diagnostic silently; if that day comes, the tolerance above is the real contract and
    # this line is the one to loosen, deliberately.
    assert result["bitwise_equal"] is True, result


def test_the_replica_count_comes_from_the_runs_own_axis_rules():
    # Derived, not guessed: the mesh axes that 'batch' maps to, multiplied out. A single-device host
    # gets 1, which is what keeps this a no-op off the TPU and leaves earlier traces untouched.
    config = SimpleNamespace(logical_axis_rules=[["batch", ["data", "fsdp"]], ["embed", ["tensor"]]])
    assert s1_5.trace.batch_replicas(config, None) == max(1, jax.device_count())

    class _Mesh:
        shape = {"data": 2, "fsdp": 4, "tensor": 8}

    assert s1_5.trace.batch_replicas(config, _Mesh()) == 8  # data x fsdp, and NOT tensor
    # An axis the mesh does not have contributes 1 rather than exploding.
    assert s1_5.trace.batch_replicas(SimpleNamespace(logical_axis_rules=[["batch", ["nope"]]]), _Mesh()) == 1
    # No rule at all: fall back to the device count rather than silently to 1.
    assert s1_5.trace.batch_replicas(SimpleNamespace(logical_axis_rules=[]), _Mesh()) == max(1, jax.device_count())
    # The real config's rule is the one this was built for.
    assert "['batch', ['data', 'fsdp']]" in _CONFIG.read_text().replace('"', "'")


def test_the_probe_hands_the_mesh_to_the_trace():
    # The failure was reached with the whole report already computed; the wiring that prevents it is
    # the mesh reaching the trace, so it is pinned structurally as well as executed above.
    per_state = inspect.getsource(s1_5._run_one_state)
    assert "trace_in_memory_state(\n" in per_state and "mesh=mesh" in per_state
    tracer = inspect.getsource(s1_5.trace_in_memory_state)
    assert "trace.batch_replicas(config, mesh)" in tracer
    assert "trace_forward_for(state, replicas=replicas, state_label=state_label)" in tracer
    assert "trace.velocity_fn_for(forward, state)" in tracer
    # ...and the standalone entry point got the same fix, since it has the same batch-1 forward.
    standalone = inspect.getsource(s1_5.trace.run_trace)
    assert "jitted_tiled_forward(state, replicas=batch_replicas(config, mesh))" in standalone
    assert "velocity_fn_for(forward, state)" in standalone


def _nested_callables(function) -> list:
    """Every callable DEFINED inside ``function`` — def, async def or lambda.

    Syntax-independent by construction, which the previous guard was not: it matched the string
    ``def velocity_fn`` and then the AST shape ``hidden_states=`` + ``deterministic=``, so a rename,
    a positional call, ``**kwargs`` or a ``functools.partial`` all walked straight past it. The rule
    here does not care how a forward is spelled. Neither trace entry point needs a closure of any
    kind, so the honest structural statement is that it may not define one at all -- and you cannot
    hand-roll a forward you cannot define.
    """
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    nested = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != function.__name__:
            nested.append(node.name)
        elif isinstance(node, ast.Lambda):
            nested.append("<lambda>")
    return nested


def test_neither_trace_entry_point_defines_a_callable_of_its_own():
    # The static half of the guard, and it is about DEFINITION rather than spelling.
    for owner in (s1_5.trace_in_memory_state, s1_5.trace.run_trace):
        assert _nested_callables(owner) == [], (owner.__name__, _nested_callables(owner))
    # The helper that IS allowed to build one defines exactly the one.
    assert _nested_callables(s1_5.trace.jitted_tiled_forward) == ["_forward"]
    assert _nested_callables(s1_5.trace.velocity_fn_for) == ["velocity_fn"]

    # ...and the rule bites on every spelling the old shape-matcher missed.
    def _renamed(transformer):
        def _compute_step(h, t, c):
            return transformer(hidden_states=h, timestep=t, encoder_hidden_states=c, deterministic=True)

        return _compute_step

    def _positional(transformer):
        return lambda h, t, c: transformer(h, t, c, True)

    def _via_partial(transformer):
        import functools

        def _bound(h, t, c):
            return functools.partial(transformer, deterministic=True)(h, timestep=t, encoder_hidden_states=c)

        return _bound

    assert _nested_callables(_renamed) == ["_compute_step"]
    assert _nested_callables(_positional) == ["<lambda>"]
    assert _nested_callables(_via_partial) == ["_bound"]


def test_only_the_blessed_helper_ever_reaches_the_transformer(monkeypatch):
    # THE execution-based half, immune to spelling entirely: whatever the source looks like, the
    # transformer object records who called it. A hand-rolled forward -- positional, **kwargs,
    # functools.partial, any name -- shows up as a different calling frame.
    reached: list = []

    class _Recording:
        """Stands in for the merged transformer and names its caller."""

        def __call__(self, *args, **kwargs):
            reached.append(inspect.stack()[1].function)
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            return hidden

    monkeypatch.setattr(s1_5.trace.gen.nnx, "merge", lambda *a, **k: _Recording())

    state, batches, config, scheduler = _toy_state_and_batches(config_shape="proxy")
    hidden = batches[0]["z_video"][:1]
    timestep = jnp.zeros((1, 4), jnp.float32)
    context = state.context_table[:1]

    forward = s1_5.trace.jitted_tiled_forward(state, replicas=1)
    s1_5.trace.velocity_fn_for(forward, state)(hidden, timestep, context)
    # The ONLY frame that ever reaches the transformer is the blessed helper's own inner function.
    assert reached, "the transformer was never reached -- the seam is not measuring anything"
    assert set(reached) == {"_forward"}, reached

    # BITE, executed, in the three spellings the AST guard could not see. Each one reaches the same
    # transformer and is caught by its calling frame, not by how the call is written.
    import functools

    def _renamed_hand_rolled(t):
        def _compute_step(h, ts, c):
            return t(hidden_states=h, timestep=ts, encoder_hidden_states=c, deterministic=True)

        return _compute_step

    def _positional_hand_rolled(t):
        return lambda h, ts, c: t(h, ts, c, True)

    def _partial_hand_rolled(t):
        def _bound(h, ts, c):
            return functools.partial(t, deterministic=True)(hidden_states=h, timestep=ts, encoder_hidden_states=c)

        return _bound

    transformer = _Recording()
    for builder, expected in (
        (_renamed_hand_rolled, "_compute_step"),
        (_positional_hand_rolled, "<lambda>"),
        (_partial_hand_rolled, "_bound"),
    ):
        reached.clear()
        builder(transformer)(hidden, timestep, context)
        assert reached == [expected], (builder.__name__, reached)
        assert reached != ["_forward"], builder.__name__


def test_the_probe_takes_its_trace_forward_from_the_shared_helper(monkeypatch):
    # DISCRIMINATING, where the previous identity check compared an attribute with itself and could
    # not fail: patch the shared helper and prove the probe's own path is what calls it.
    calls: list = []
    sentinel = object()

    def _fake_builder(state, *, replicas):
        calls.append(int(replicas))
        return sentinel

    monkeypatch.setattr(s1_5.trace, "jitted_tiled_forward", _fake_builder)
    s1_5._TRACE_FORWARD_CACHE.clear()
    state, _, _, _ = _toy_state_and_batches(config_shape="proxy")

    wrapper = s1_5.trace_forward_for(state, replicas=8, state_label="checkpoint")
    assert calls == [8], calls  # the probe asked the SHARED helper, with the mesh's replica count
    assert getattr(wrapper, "jitted", None) is sentinel  # ...and wrapped exactly what it returned
    s1_5._TRACE_FORWARD_CACHE.clear()


def test_the_trace_forward_compiles_once_per_state_and_is_released():
    # The trace runs ~750 forwards per state. ONE compilation serves all of them, it is timed under
    # its own (tag, state) key like every other compiled function, it appears in the census, and it
    # is released when the phase ends -- the same discipline as the gradient programs.
    s1_5._TRACE_FORWARD_CACHE.clear()
    s1_5._RELEASED_SPECIALIZATIONS.clear()
    exp03.COMPILE_TIMINGS.clear()
    state, batches, config, scheduler = _toy_state_and_batches(config_shape="proxy")

    forward = s1_5.trace_forward_for(state, replicas=1, state_label="checkpoint")
    hidden = batches[0]["z_video"][:1]
    timestep = jnp.zeros((1, 4), jnp.float32)
    context = state.context_table[:1]
    out = s1_5.trace.velocity_fn_for(forward, state)(hidden, timestep, context)
    assert out.shape[0] == 1, out.shape  # row 0, sliced outside the compiled boundary

    for _ in range(5):  # ...and every later step REUSES it rather than recompiling
        s1_5.trace.velocity_fn_for(s1_5.trace_forward_for(state, replicas=1, state_label="checkpoint"), state)(
            hidden, timestep, context
        )
    assert len(s1_5._TRACE_FORWARD_CACHE) == 1, s1_5._TRACE_FORWARD_CACHE
    assert s1_5.jit_cache_size(forward) == 1, s1_5.jit_cache_size(forward)
    # Timed per (tag, state), so the init state's compile is not merged into the checkpoint's.
    assert ("probe_trace_forward", "checkpoint") in exp03.COMPILE_TIMINGS, sorted(exp03.COMPILE_TIMINGS)

    census = s1_5.specialization_census()
    assert census["trace"] == {"('trace_forward', 1)": 1}, census["trace"]
    assert census["trace_total"] == 1

    released = s1_5.release_trace_programs()
    assert released and not s1_5._TRACE_FORWARD_CACHE
    # ...and the census still knows it existed, which is the point of snapshotting before the drop.
    after = s1_5.specialization_census()
    assert after["trace_total"] == 1, after["trace"]
    assert after["total"] == census["total"], (after["total"], census["total"])
    # The release point is where the phase ends, and the ledger says whether the executable went.
    tracer = inspect.getsource(s1_5.trace_in_memory_state)
    assert tracer.index("trace.trace_rows(") < tracer.index("release_trace_programs()")
    assert 'log_memory("post_trace_release"' in tracer


def test_the_trace_release_actually_releases_because_the_locals_go_first():
    # RELEASE TRUTH. Dropping the cache entry frees nothing while any name still holds the compiled
    # object: velocity_fn closes over the timing wrapper, and the wrapper closes over the compiled
    # function. Before this, post_trace_release logged a release that had not happened.
    import gc
    import weakref

    s1_5._TRACE_FORWARD_CACHE.clear()
    s1_5._RELEASED_SPECIALIZATIONS.clear()
    state, _, _, _ = _toy_state_and_batches(config_shape="proxy")

    # 1. THE DEFECT, executed: hold the locals across the release and the executable survives it.
    wrapper = s1_5.trace_forward_for(state, replicas=1, state_label="checkpoint")
    velocity_fn = s1_5.trace.velocity_fn_for(wrapper, state)
    reference = weakref.ref(wrapper.jitted)
    s1_5.release_trace_programs()
    gc.collect()
    assert reference() is not None, "the retained locals should still be pinning it -- test is vacuous otherwise"
    assert velocity_fn is not None

    # 2. THE FIX: drop the locals FIRST, exactly as trace_in_memory_state now does, and it goes.
    del velocity_fn, wrapper
    gc.collect()
    assert reference() is None, "the compiled forward is still reachable after its last name was dropped"

    # ...and the code really does it in that order, before the ledger line that reports it.
    tracer = inspect.getsource(s1_5.trace_in_memory_state)
    assert tracer.index("del velocity_fn, forward") < tracer.index("release_trace_programs()")
    assert tracer.index("release_trace_programs()") < tracer.index('log_memory("post_trace_release"')


def test_the_persisted_census_is_refreshed_after_the_trace_phase():
    # The artifact's census used to be whatever state_report captured -- BEFORE the trace compiled
    # anything -- so the trace section reached the JSON empty. It is recomputed after the phase.
    source = inspect.getsource(s1_5._run_one_state)
    assert source.index("trace_in_memory_state(") < source.index('report["specializations"] = specialization_census()')
    assert source.index('report["specializations"] = specialization_census()') < source.index("s1_5_artifact(")

    # The logged census names the trace family too, and the runtime budget assertion covers it.
    assert "trace {census['trace_total']}" in inspect.getsource(s1_5.log_compile_costs)
    census = s1_5.specialization_census()
    for key in ("trace", "trace_total"):
        assert key in census, sorted(census)
    assert (
        census["total"]
        == census["probe_total"] + census["replay_total"] + census["trace_total"] + census["helper_total"]
    )
    # A trace program compiled more often than there are states is a release-discipline failure, and
    # the driver refuses it exactly as it does for the gradient tags.
    s1_5.assert_specializations_within_release_budget(census)
    with pytest.raises(RuntimeError) as excinfo:
        s1_5.assert_specializations_within_release_budget({**census, "trace": {"('trace_forward', 8)": 3}})
    assert "trace_forward" in str(excinfo.value)
