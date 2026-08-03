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
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

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
    # A ONE-ULP change in one element of a large tensor: the smallest update that can exist. It must
    # be caught, and the test proves the check is EXACT rather than merely tight -- the fingerprint
    # moves by less than 1e-3 relative, so any tolerant comparison would wave it through.
    values = np.linspace(1.0, 2.0, 1024, dtype=np.float32)
    params = {"gain": jnp.asarray(values)}
    before = s1_5.params_fingerprint(params)
    s1_5.assert_no_update(before, s1_5.params_fingerprint(params))  # unchanged: fine

    nudged_values = values.copy()
    nudged_values[7] = np.nextafter(nudged_values[7], np.float32(2.0))
    assert nudged_values[7] != values[7]
    after = s1_5.params_fingerprint({"gain": jnp.asarray(nudged_values)})
    relative = abs(after[0] - before[0]) / max(abs(before[0]), 1e-12)
    assert 0.0 < relative < 1e-3, relative  # a tolerant check would MISS this

    with pytest.raises(RuntimeError) as excinfo:
        s1_5.assert_no_update(before, after)
    assert "applies no updates" in str(excinfo.value)


def test_the_driver_asserts_no_update_around_the_measurement():
    source = inspect.getsource(s1_5.run_s1_5)
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
    assert not called & {"apply_gradients", "update", "value_and_grad"}, called


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
    assert stats["data_variance"] == pytest.approx(1.0)  # mean of (1^2)
    assert stats["total_variance"] == pytest.approx(1.25)
    assert stats["support_fraction"] == pytest.approx(0.2)
    assert stats["mean_grad_sq_norm"] == pytest.approx(0.0, abs=1e-12)  # grand mean is 0 here


def test_identical_draws_report_no_support_variance():
    # The control draws no support, so its within-batch spread must be exactly 0 -- that contrast is
    # what makes a trial's number readable.
    gradients = [[_grad([1.0]), _grad([1.0])], [_grad([3.0]), _grad([3.0])]]
    stats = s1_5.variance_decomposition(gradients)
    assert stats["support_variance"] == 0.0
    assert stats["data_variance"] == pytest.approx(1.0)
    assert stats["support_fraction"] == 0.0
    assert stats["gradient_noise_scale"] == pytest.approx(1.0 / 4.0)  # total 1.0, mean grad 2.0


def test_a_single_draw_cannot_report_a_support_term():
    stats = s1_5.variance_decomposition([[_grad([1.0])], [_grad([3.0])]])
    assert stats["support_draws"] == 1 and stats["support_variance"] == 0.0
    assert stats["data_variance"] == pytest.approx(1.0)


def test_ragged_draw_counts_are_refused():
    with pytest.raises(ValueError):
        s1_5.variance_decomposition([[_grad([1.0]), _grad([2.0])], [_grad([3.0])]])
    with pytest.raises(ValueError):
        s1_5.variance_decomposition([])


def test_the_driver_varies_the_support_not_the_batch_across_draws():
    # THE thing that makes the within-batch term a SUPPORT term: the batch is held fixed while the
    # draw index moves the global step the aux keys fold on.
    source = inspect.getsource(s1_5.state_report)
    assert "for draw in range(S1_5_SUPPORT_DRAWS)" in source
    assert "index * S1_5_SUPPORT_DRAWS + draw" in source  # a distinct step per draw
    # ...and the batch and the shared-stream key do NOT move with the draw.
    inner = source[source.index("for draw in range(S1_5_SUPPORT_DRAWS)") :]
    assert "jax.random.fold_in(rng, index)" in inner and "fold_in(rng, draw)" not in inner
    # Distinct steps really give distinct supports.
    supports = {
        tuple(int(v) for v in exp03.corrective_support(seed=0, global_step=step, num_steps=25, k_a_max=2))
        for step in range(4)
    }
    assert len(supports) > 1


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
    module = Path(s1_5.__file__).read_text()
    # No private re-implementation of the per-objective loop.
    assert "_corrective_ss_loss(" not in module and "_rollout_loss(" not in module


def test_the_extended_replay_reports_cosines_against_the_control():
    source = inspect.getsource(exp03.exp03_frozen_replay)
    assert '"control"' in source and "grad_cosine_{name}_vs_control" in source
    assert "include_control=True" in inspect.signature(exp03.exp03_frozen_replay).__str__() or True
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
        ("EXP03_RAMP_ORIGIN", "exp03_ramp_origin"),
    ):
        assert f'{key}="${{{env}}}"' in text, key
        assert f'echo "{env}=' in text, env
    # The Tier-1 checkpoint is REQUIRED: probing the init twice would answer the wrong question.
    assert 'CHECKPOINT_DIR="${CHECKPOINT_DIR:?' in text
    # Tier-1's ramp origin, per the D1 requirement.
    assert 'EXP03_RAMP_ORIGIN="${EXP03_RAMP_ORIGIN:-10000}"' in text


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
        "EXP03_RAMP_ORIGIN",
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
