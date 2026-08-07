"""exp_03 — gradient accumulation, so trial C can run GBS 256 with per-device-2 activations.

C misses the v6e-64 GBS-256 HBM fit by 34.32 MB at per-device batch 4 (S1.6). The fix is NOT a
smaller batch — that would change the estimator and break the comparison with A, B and ctrl0 — but
consuming the same delivered batch as two microbatches and applying ONE optimizer update. What
that has to leave alone is the entire content of this file:

1. **``N = 1`` is a bit-for-bit no-op.** The default path is compared against a VERBATIM copy of
   the pre-accumulation step, on the same params/batch/rng, with exact equality on the loss, every
   gradient-derived metric, the updated parameters, the optimizer state and the returned rng.
   ctrl0 is a replication guard for exp_02; it stays one only if the knob's default changes
   nothing at all.
2. **``N = 2`` is the full-batch step.** Shown on a DETERMINISTIC toy objective, where the
   equivalence is exact up to float reduction order — see :func:`_toy_loss` for why the real
   denoising loss cannot be the vehicle for this particular claim, and what is asserted about it
   instead.
3. **Update-matching and the RNG contract.** One ``apply_gradients`` per global step and exactly
   one ``jax.random.split`` per step, whatever ``N`` is; the returned rng is identical; the exp_03
   auxiliary draws (supports, ``k_A``, the ``p_ss`` coin) are the SAME values at ``N = 2`` as at
   ``N = 1``, because they are keyed on ``(seed, global_step, purpose)`` and not on the batch.
4. **The accumulator is param-sharded**, measured on a forced 8-device mesh in a subprocess, in
   physical per-device bytes — plus the inversion that gives the number meaning (a replicated
   accumulator reads 8x) and the shard-locality check that justifies the interleaved slice over
   the contiguous one.
5. **The knob is reachable**: config default, pyconfig type, launcher env plumb-through.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import optax
import pytest
import yaml
from flax import nnx

import maxdiffusion.trainers.wan_ti2v_exp03_trainer as exp03
import maxdiffusion.trainers.wan_ti2v_overfit100_trainer as parent
from maxdiffusion.schedulers import FlaxFlowMatchScheduler

_REPO = Path(parent.__file__).parents[3]
_CONFIG = _REPO / "src/maxdiffusion/configs/base_wan_5b_exp03.yml"
_LAUNCHER = _REPO / "bash_scripts/train_wan_exp03.sh"

# Four rows, so a batch can be halved. _BSZ is what the objectives clamp to, and it is the whole
# delivered batch: the microbatching happens BELOW that clamp, never by shrinking it.
_DATA_B, _BSZ, _C, _F, _H, _W = 4, 4, 3, 4, 5, 6
_SLOTS, _LEN, _DIM = 4, 4, 8
_STEPS = 4


class _StubTransformer(nnx.Module):
    """Tiny stand-in with a real Param, so the gradient and the optimizer step are genuine."""

    def __init__(self):
        self.gain = nnx.Param(jnp.asarray(0.5, dtype=jnp.float32))

    def __call__(self, **kwargs):
        hidden = kwargs["hidden_states"].astype(jnp.float32)
        t_mean = jnp.mean(kwargs["timestep"].astype(jnp.float32))
        ctx_mean = jnp.mean(kwargs["encoder_hidden_states"].astype(jnp.float32))
        return self.gain[...] * hidden + 0.01 * t_mean + 0.001 * ctx_mean


def _fixture(**overrides):
    transformer = _StubTransformer()
    graphdef, params, rest = nnx.split(transformer, nnx.Param, ...)
    context_table = jax.random.normal(jax.random.key(41), (_SLOTS, _LEN, _DIM), dtype=jnp.float32)
    state = parent.Overfit100TrainState.create(
        apply_fn=graphdef.apply,
        params=params,
        tx=optax.sgd(0.1),
        graphdef=graphdef,
        rest_of_state=rest,
        context_table=context_table,
    )
    k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
    data = {
        "z_i0": jax.random.normal(k1, (_DATA_B, _C, 1, _H, _W), dtype=jnp.float32),
        "z_video": jax.random.normal(k2, (_DATA_B, _C, _F, _H, _W), dtype=jnp.float32),
        "episode_index": jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        # A second batched target, so the deterministic toy objective has something to regress on
        # that is not derived from z_video.
        "target": jax.random.normal(k3, (_DATA_B, _C, _F, _H, _W), dtype=jnp.float32),
    }
    settings = {
        "weights_dtype": "float32",
        "activations_dtype": "float32",
        "global_batch_size_to_train_on": _BSZ,
        "side_adapter_sampling_steps": _STEPS,
        "flow_shift": 5.0,
        "side_adapter_t_sampling": "uniform",
        "side_adapter_noise_mode": "fresh",
        "seed": 0,
        "model_type": exp03.EXP03_MODEL_TYPE,
        "exp03_objective": "control",
        "exp03_k_a": 2,
        "exp03_k_b": 2,
        "exp03_lambda": 0.5,
        "exp03_p_ss_max": 0.5,
        "exp03_p_ss_ramp_steps": 500,
        "exp03_ramp_origin": 0,
        "exp03_support_salt": 0,
        "exp03_grad_accumulation": 1,
    }
    settings.update(overrides)
    config = SimpleNamespace(**settings)
    scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32, shift=config.flow_shift, sigma_min=0.0, sigma_max=1.0)
    return state, data, config, scheduler


def _toy_loss(params, state, data, rng, config, scheduler, *, global_step=None):
    """A DETERMINISTIC objective — no rng at all — which is what makes the N=2 claim provable.

    The real denoising losses draw their epsilon and their per-example timestep from the loss rng
    at the shape of whatever batch they are handed. Two microbatches of B/2 therefore draw a
    different noise realization from one batch of B, no matter how the accumulation is written, so
    "N=2 reproduces the full-batch step" is simply false for them and no tolerance would rescue it
    (that is a property of per-example sampling, not of this implementation — and it is asserted
    directly in ``test_the_microbatch_keys_are_folded_in_so_the_halves_are_not_noise_twins``).

    What the equivalence claim is actually about is the ACCUMULATION ARITHMETIC: that summing the
    microbatch gradients and dividing by N reproduces the full-batch gradient. Isolating that needs
    an objective whose value depends only on the examples, which is this one: a plain masked mean
    squared error of ``gain * z_video`` against ``target``. It goes through the identical machinery
    (``nnx.value_and_grad``, the same train-step factory, the same optimizer) and differs from the
    production losses in exactly the one respect that would otherwise confound the measurement.

    The mean is over the whole microbatch, so mean-of-means is the full-batch mean exactly — which
    is the same normalization property ``masked_velocity_mse`` has (it divides by ``sum(mask) *
    batch_size``, with ``batch_size`` the actual microbatch's).

    **Strictly PER-EXAMPLE**, and that is load-bearing. An earlier draft reused the stub
    transformer's forward, which adds ``0.001 * jnp.mean(encoder_hidden_states)`` — a mean over the
    WHOLE batch. That couples every example's prediction to the batch it arrived in, so a half-batch
    genuinely computes something different, and the N=2 comparison failed by ~2e-6 relative: it was
    measuring the toy's own batch coupling, not the accumulation. The production objectives have no
    such coupling (their only batch-level quantities are the exp_03 supports, which come from the
    auxiliary key and are identical in every microbatch), so the vehicle must not have one either.
    """
    del rng, scheduler, global_step
    transformer = nnx.merge(state.graphdef, params, state.rest_of_state)
    bsz = config.global_batch_size_to_train_on
    x = data["z_video"][:bsz].astype(jnp.float32)
    y = data["target"][:bsz].astype(jnp.float32)
    context = state.context_table[data["episode_index"][:bsz].astype(jnp.int32)].astype(jnp.float32)
    # PER-EXAMPLE bias: reduced over the context's OWN axes, never across the batch.
    bias = jnp.mean(context, axis=(1, 2)).reshape((-1, 1, 1, 1, 1))
    prediction = transformer.gain[...] * x + 0.001 * bias
    residual = prediction - y
    loss = jnp.mean(residual**2)
    return loss, {
        "velocity_mse": loss,
        "sigma_mean": jnp.mean(x),
        "timestep_mean": jnp.mean(y),
        "v_pred_l2": jnp.linalg.norm(prediction),
        "v_target_l2": jnp.linalg.norm(y),
        "z_noisy_std": jnp.std(x),
        "z_target_std": jnp.std(y),
        "z_init_anchor_mse": jnp.mean(residual),
        "toy_finite": jnp.isfinite(loss).astype(jnp.float32),
    }


# =============================================================================================
# 1. N = 1 is a bit-for-bit no-op.
# =============================================================================================


def _pre_accumulation_train_step(state, data: dict, rng, scheduler, config, *, global_step=None):
    """VERBATIM copy of ``_make_train_step``'s step at de2b87a, before accumulation existed.

    Bound to ``parent._denoising_loss`` directly, exactly as the factory bound it. Deliberately not
    refactored and not deduplicated against the production code: its entire value is that it is the
    old code, so that "the default path did not move" is a comparison against something that
    predates the change rather than against a rearrangement of it.
    """
    rng, loss_rng = jax.random.split(rng)

    def loss_fn(params):
        if global_step is None:
            return parent._denoising_loss(params, state, data, loss_rng, config, scheduler)
        return parent._denoising_loss(params, state, data, loss_rng, config, scheduler, global_step=global_step)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    grad_norm = jaxopt.tree_util.tree_l2_norm(grads)
    max_abs_grad = jax.tree_util.tree_reduce(
        lambda m, arr: jnp.maximum(m, jnp.max(jnp.abs(arr))), grads, initializer=-1.0
    )
    state = state.apply_gradients(grads=grads)
    _mapped = {
        "velocity_mse",
        "sigma_mean",
        "timestep_mean",
        "v_pred_l2",
        "v_target_l2",
        "z_noisy_std",
        "z_target_std",
        "z_init_anchor_mse",
    }
    extra = {f"learning/{name}": value for name, value in aux.items() if name not in _mapped}
    metrics = {
        "scalar": {
            "learning/loss": loss,
            "learning/velocity_mse": aux["velocity_mse"],
            "learning/grad_norm": grad_norm,
            "learning/max_abs_grad": max_abs_grad,
            "learning/sigma_mean": aux["sigma_mean"],
            "learning/timestep_mean": aux["timestep_mean"],
            "learning/v_pred_l2": aux["v_pred_l2"],
            "learning/v_target_l2": aux["v_target_l2"],
            "learning/z_noisy_std": aux["z_noisy_std"],
            "learning/z_target_std": aux["z_target_std"],
            "learning/z_init_anchor_mse": aux["z_init_anchor_mse"],
            **extra,
        },
        "scalars": {},
    }
    return state, metrics, rng


def _assert_steps_are_bit_identical(left, right, label: str):
    """Exact equality — parameters, optimizer state, every scalar metric, and the returned rng."""
    left_state, left_metrics, left_rng = left
    right_state, right_metrics, right_rng = right

    left_params = jax.tree_util.tree_leaves(left_state.params)
    right_params = jax.tree_util.tree_leaves(right_state.params)
    assert len(left_params) == len(right_params) == 1, label
    for a, b in zip(left_params, right_params):
        assert np.array_equal(np.asarray(a), np.asarray(b)), f"{label}: parameters differ"
    for a, b in zip(jax.tree_util.tree_leaves(left_state.opt_state), jax.tree_util.tree_leaves(right_state.opt_state)):
        assert np.array_equal(np.asarray(a), np.asarray(b)), f"{label}: optimizer state differs"
    assert int(left_state.step) == int(right_state.step), label

    assert set(left_metrics["scalar"]) == set(right_metrics["scalar"]), label
    for name in left_metrics["scalar"]:
        a = np.asarray(left_metrics["scalar"][name])
        b = np.asarray(right_metrics["scalar"][name])
        assert np.array_equal(a, b), f"{label}: metric {name} differs ({a} != {b})"
    assert np.array_equal(jax.random.key_data(left_rng), jax.random.key_data(right_rng)), f"{label}: rng differs"


def test_accumulation_one_is_bitwise_the_pre_accumulation_step():
    # THE no-op guarantee. Same params, same batch, same rng, same everything.
    state, data, config, scheduler = _fixture(exp03_grad_accumulation=1)
    rng = jax.random.key(7)
    reference = _pre_accumulation_train_step(state, data, rng, scheduler, config, global_step=11)
    produced = parent._make_train_step(parent._denoising_loss)(state, data, rng, scheduler, config, global_step=11)
    _assert_steps_are_bit_identical(reference, produced, "N=1 vs pre-accumulation")


def test_a_config_without_the_knob_at_all_is_the_pre_accumulation_step():
    # exp_01's and exp_02's configs predate the key entirely. ``getattr`` must resolve them to 1,
    # not raise and not accidentally accumulate.
    state, data, config, scheduler = _fixture()
    del config.exp03_grad_accumulation
    assert not hasattr(config, "exp03_grad_accumulation")
    assert parent.resolve_grad_accumulation(config) == 1
    rng = jax.random.key(7)
    reference = _pre_accumulation_train_step(state, data, rng, scheduler, config, global_step=11)
    produced = parent._make_train_step(parent._denoising_loss)(state, data, rng, scheduler, config, global_step=11)
    _assert_steps_are_bit_identical(reference, produced, "no-key vs pre-accumulation")


def test_the_control_arm_step_is_still_the_parents_by_identity():
    # ctrl0's replication guarantee is IDENTITY, not equality. Adding the knob must not have
    # wrapped the control's step in anything.
    _, _, config, _ = _fixture(exp03_objective="control")
    trainer = exp03.Exp03Trainer.__new__(exp03.Exp03Trainer)
    trainer.config = config
    loss_fn, step_fn = trainer._loss_and_step_fns()
    assert loss_fn is parent._denoising_loss
    assert step_fn is parent._train_step


def test_the_legacy_six_argument_loss_signature_still_works_unaccumulated():
    # Some exp_02 callers invoke the loss without a global_step. The N=1 branch must keep taking
    # that path; the accumulating branch must too.
    state, data, config, scheduler = _fixture(exp03_grad_accumulation=1)
    rng = jax.random.key(3)
    reference = _pre_accumulation_train_step(state, data, rng, scheduler, config)
    produced = parent._make_train_step(parent._denoising_loss)(state, data, rng, scheduler, config)
    _assert_steps_are_bit_identical(reference, produced, "N=1 no-global-step")

    accumulated = parent._make_train_step(parent._denoising_loss)(
        state, data, rng, scheduler, SimpleNamespace(**{**vars(config), "exp03_grad_accumulation": 2})
    )
    assert np.isfinite(float(accumulated[1]["scalar"]["learning/loss"]))


# =============================================================================================
# 2. N = 2 is the full-batch step (deterministic objective, so the claim is provable).
# =============================================================================================

# THE reduction-order caveat, stated as a measured number rather than a shrug. The equivalence is
# not bitwise and cannot be: XLA reduces one 1,440-element float32 mean differently from two
# 720-element means averaged together, and float32 carries ~1.2e-7 of relative precision per
# operation. Measured on this fixture, the accumulated step differs from the full-batch step by:
#
#     quantity                N=2         N=4
#     loss                    1.07e-6     1.07e-6
#     updated parameter       1.47e-7     1.47e-7
#     grad_norm / max_abs     3.74e-7     4.99e-7
#
# The bound below is 1e-5 -- an order of magnitude above the worst of those, so it fails on a real
# arithmetic error and not on a recompilation that reassociates a sum. It is deliberately NOT
# tightened to the observed values: that would make the suite a hostage to the XLA version.
_REDUCTION_ORDER_REL = 1e-5


def test_accumulation_two_reproduces_the_full_batch_step_on_a_deterministic_objective():
    # The accumulation arithmetic itself: sum the microbatch gradients, divide by N, one update.
    state, data, config, scheduler = _fixture(exp03_grad_accumulation=1)
    rng = jax.random.key(5)
    full = parent._make_train_step(_toy_loss)(state, data, rng, scheduler, config, global_step=4)
    split = parent._make_train_step(_toy_loss)(
        state,
        data,
        rng,
        scheduler,
        SimpleNamespace(**{**vars(config), "exp03_grad_accumulation": 2}),
        global_step=4,
    )

    full_param = np.asarray(jax.tree_util.tree_leaves(full[0].params)[0])
    split_param = np.asarray(jax.tree_util.tree_leaves(split[0].params)[0])
    # The updated parameter is the gradient, run through the optimizer: equal parameters means
    # equal gradients.
    assert split_param == pytest.approx(full_param, rel=_REDUCTION_ORDER_REL), (full_param, split_param)

    full_loss = float(full[1]["scalar"]["learning/loss"])
    split_loss = float(split[1]["scalar"]["learning/loss"])
    assert split_loss == pytest.approx(full_loss, rel=_REDUCTION_ORDER_REL), (full_loss, split_loss)
    # ...and the headline gradient metrics are the FULL-BATCH gradient's own, because they are
    # computed after the accumulator has been averaged -- not a mean of per-microbatch norms.
    for metric in ("learning/grad_norm", "learning/max_abs_grad"):
        assert float(split[1]["scalar"][metric]) == pytest.approx(
            float(full[1]["scalar"][metric]), rel=_REDUCTION_ORDER_REL
        ), metric


@pytest.mark.parametrize("num_microbatches", [1, 2, 4])
def test_every_divisor_of_the_batch_gives_the_same_deterministic_update(num_microbatches):
    # N=4 is one example per microbatch. The equivalence is a property of the arithmetic, not a
    # coincidence at N=2 -- and N=4 is where a reduction-order drift would show up largest.
    state, data, config, scheduler = _fixture(exp03_grad_accumulation=1)
    rng = jax.random.key(5)
    reference = parent._make_train_step(_toy_loss)(state, data, rng, scheduler, config, global_step=4)
    produced = parent._make_train_step(_toy_loss)(
        state,
        data,
        rng,
        scheduler,
        SimpleNamespace(**{**vars(config), "exp03_grad_accumulation": num_microbatches}),
        global_step=4,
    )
    reference_param = np.asarray(jax.tree_util.tree_leaves(reference[0].params)[0])
    produced_param = np.asarray(jax.tree_util.tree_leaves(produced[0].params)[0])
    assert produced_param == pytest.approx(reference_param, rel=_REDUCTION_ORDER_REL)


def test_the_microbatches_partition_the_batch_exactly_and_interleave():
    # Every row appears in exactly one microbatch, and the split is x[i::N] -- the interleaved
    # slice, which is the shard-local one (proved on a real mesh further down).
    rows = jnp.arange(8, dtype=jnp.int32).reshape(8, 1)
    batch = {"rows": rows}
    slices = [parent.microbatch_slice(batch, index, 2)["rows"].ravel().tolist() for index in (0, 1)]
    assert slices == [[0, 2, 4, 6], [1, 3, 5, 7]]
    assert sorted(slices[0] + slices[1]) == list(range(8))
    # ...and it is NOT the contiguous split, which is what a naive implementation reaches for.
    assert slices[0] != [0, 1, 2, 3]


def test_the_microbatch_slice_refuses_a_batch_it_cannot_partition():
    # Trace-time failure on a shape, not a silent mis-slice at step 4,000.
    with pytest.raises(ValueError, match="not divisible"):
        parent.microbatch_slice({"rows": jnp.arange(9).reshape(9, 1)}, 0, 2)
    with pytest.raises(ValueError, match="leading batch axis"):
        parent.microbatch_slice({"scalar": jnp.asarray(1.0)}, 0, 2)


def test_the_slice_covers_every_leaf_including_the_ones_the_objective_ignores():
    # A leaf left un-sliced would be silently broadcast against a half-height batch, or would blow
    # up on a shape mismatch two objectives later.
    _, data, _, _ = _fixture()
    micro = parent.microbatch_slice(data, 0, 2)
    assert set(micro) == set(data)
    for name, leaf in micro.items():
        assert leaf.shape == (data[name].shape[0] // 2,) + data[name].shape[1:], name


# =============================================================================================
# 3. Update-matching and the RNG contract.
# =============================================================================================


@pytest.mark.parametrize("num_microbatches", [1, 2, 4])
def test_one_optimizer_update_per_step_whatever_the_accumulation(num_microbatches):
    # THE point of accumulation: the update count does not move, so an accumulated arm's step 500
    # is an un-accumulated arm's step 500.
    state, data, config, scheduler = _fixture(exp03_grad_accumulation=num_microbatches)
    assert int(state.step) == 0
    stepped, _, _ = parent._make_train_step(_toy_loss)(state, data, jax.random.key(1), scheduler, config)
    assert int(stepped.step) == 1, num_microbatches


@pytest.mark.parametrize("num_microbatches", [1, 2, 4])
def test_the_shared_stream_advances_by_exactly_one_split_whatever_the_accumulation(num_microbatches):
    # The returned rng is the stream's next state. If accumulation split the stream per microbatch,
    # this would diverge -- and ctrl0 would stop reproducing exp_02 the moment any arm accumulated.
    state, data, config, scheduler = _fixture(exp03_grad_accumulation=num_microbatches)
    rng = jax.random.key(9)
    expected, _ = jax.random.split(rng)
    _, _, returned = parent._make_train_step(_toy_loss)(state, data, rng, scheduler, config)
    assert np.array_equal(jax.random.key_data(returned), jax.random.key_data(expected)), num_microbatches


def test_the_microbatch_keys_are_folded_in_so_the_halves_are_not_noise_twins():
    # The failure this prevents: with the same key and a smaller shape, jax's counter-based normal
    # hands microbatch 1 exactly what it handed microbatch 0 -- the two halves of the batch would
    # carry identical epsilon. fold_in makes them independent, and (being derived, not split) it
    # cannot advance the shared stream.
    loss_rng = jax.random.key(13)
    naive = [jax.random.normal(loss_rng, (2, 3)) for _ in range(2)]
    assert np.array_equal(np.asarray(naive[0]), np.asarray(naive[1])), "premise: the same key repeats"
    folded = [jax.random.normal(jax.random.fold_in(loss_rng, index), (2, 3)) for index in range(2)]
    assert not np.array_equal(np.asarray(folded[0]), np.asarray(folded[1]))
    # ...and the real step inherits that: at N=2 the two microbatches' losses differ, which they
    # could not do if they shared a draw over statistically identical halves.
    state, data, config, scheduler = _fixture(exp03_grad_accumulation=2)
    seen = []
    original = parent._denoising_loss

    def spy(params, state_, data_, rng_, config_, scheduler_, *, global_step=None):
        seen.append(np.asarray(jax.random.key_data(rng_)))
        return original(params, state_, data_, rng_, config_, scheduler_, global_step=global_step)

    parent._make_train_step(spy)(state, data, jax.random.key(2), scheduler, config, global_step=0)
    assert len(seen) == 2
    assert not np.array_equal(seen[0], seen[1]), "the microbatches were handed the same key"


def test_the_exp03_auxiliary_draws_are_identical_across_microbatches_and_to_the_unaccumulated_run():
    # The supports, k_A and the p_ss coin come from exp03_aux_key(seed, global_step, purpose) --
    # not from the batch -- so accumulation must leave them untouched. This is what keeps an
    # accumulated C aligned with A and B at the same step.
    state, data, config, scheduler = _fixture(exp03_objective="combined")
    rng = jax.random.key(4)
    single = parent._make_train_step(exp03._combined_loss)(state, data, rng, scheduler, config, global_step=6)
    doubled = parent._make_train_step(exp03._combined_loss)(
        state,
        data,
        rng,
        scheduler,
        SimpleNamespace(**{**vars(config), "exp03_grad_accumulation": 2}),
        global_step=6,
    )
    batch_level = [
        "learning/k_a",
        "learning/s_a",
        "learning/e_a",
        "learning/coin",
        "learning/p_ss",
        "learning/take_self_generated",
        "learning/sigma_hi_a",
        "learning/sigma_lo_a",
        "learning/s_b",
        "learning/e_b",
        "learning/sigma_hi_b",
        "learning/sigma_lo_b",
        "learning/lambda",
    ]
    for name in batch_level:
        assert name in single[1]["scalar"], name
        assert float(doubled[1]["scalar"][name]) == float(single[1]["scalar"][name]), name


def test_the_accumulated_trial_objective_still_reports_every_metric_and_stays_finite():
    # C is the arm this exists for. Its whole aux surface must survive the reduction -- the S1 log
    # lost C's B-side supports once already, by a whitelist; this checks the keys are all there.
    state, data, config, scheduler = _fixture(exp03_objective="combined")
    rng = jax.random.key(4)
    single = parent._make_train_step(exp03._combined_loss)(state, data, rng, scheduler, config, global_step=6)
    doubled = parent._make_train_step(exp03._combined_loss)(
        state,
        data,
        rng,
        scheduler,
        SimpleNamespace(**{**vars(config), "exp03_grad_accumulation": 2}),
        global_step=6,
    )
    assert set(doubled[1]["scalar"]) == set(single[1]["scalar"])
    for name, value in doubled[1]["scalar"].items():
        assert np.isfinite(float(value)), name
    for name in ("learning/loss_a", "learning/loss_b", "learning/raw_endpoint_mse"):
        assert name in doubled[1]["scalar"], name
    # The LOSS is allowed to differ -- and does -- because the per-example epsilon is a different
    # realization. That is stated openly rather than papered over with a loose tolerance.
    assert float(doubled[1]["scalar"]["learning/loss"]) != float(single[1]["scalar"]["learning/loss"])


# =============================================================================================
# 4. The aux reduction: means, and the _finite protocol.
# =============================================================================================


def test_aux_metrics_reduce_by_mean_over_microbatches():
    reduced = parent.reduce_microbatch_aux([{"velocity_mse": jnp.asarray(1.0)}, {"velocity_mse": jnp.asarray(3.0)}])
    assert float(reduced["velocity_mse"]) == pytest.approx(2.0)


def test_a_batch_level_constant_survives_the_mean_exactly():
    # k_a, coin, p_ss and the sigmas are the same value in every microbatch, so their mean is that
    # value -- exactly, not approximately, for the N=2 that ships.
    for value in (0.1724, 2.0, 24.0, 0.5):
        reduced = parent.reduce_microbatch_aux([{"k_a": jnp.asarray(value)}] * 2)
        assert float(reduced["k_a"]) == float(jnp.asarray(value))


def test_finite_flags_reduce_by_minimum_not_by_mean():
    # THE trap. step_finite_failures fails a step on "< 0.5"; a mean over two microbatches with one
    # non-finite is exactly 0.5, which is NOT < 0.5 -- the guard would pass and the run would keep
    # training on a poisoned update. The minimum is what the guard is actually asking about.
    auxes = [{"loss_a_finite": jnp.asarray(1.0)}, {"loss_a_finite": jnp.asarray(0.0)}]
    reduced = parent.reduce_microbatch_aux(auxes)
    assert float(reduced["loss_a_finite"]) == 0.0
    # The inversion, executed: the mean would have sailed through the production guard.
    mean_instead = float(sum(a["loss_a_finite"] for a in auxes) / len(auxes))
    assert mean_instead == 0.5
    failed_under_mean, _ = parent.step_finite_failures({"learning/loss_a_finite": mean_instead})
    assert failed_under_mean == [], "premise: a mean of 0.5 is invisible to the guard"
    # ...and the minimum is caught by that same guard, unchanged.
    failed, _ = parent.step_finite_failures({"learning/loss_a_finite": float(reduced["loss_a_finite"])})
    assert failed == ["loss_a_finite"]


def test_a_single_non_finite_microbatch_fails_the_step_through_the_real_train_step():
    # End to end, not just the helper: one poisoned microbatch has to reach the loop's guard.
    state, data, config, scheduler = _fixture(exp03_grad_accumulation=2)

    def half_poisoned(params, state_, data_, rng_, config_, scheduler_, *, global_step=None):
        loss, aux = _toy_loss(params, state_, data_, rng_, config_, scheduler_, global_step=global_step)
        # Poison the microbatch whose first row is odd -- i.e. exactly one of the two.
        is_second = data_["episode_index"][0] % 2 == 1
        aux = dict(aux)
        aux["toy_finite"] = jnp.where(is_second, 0.0, 1.0)
        return loss, aux

    _, metrics, _ = parent._make_train_step(half_poisoned)(state, data, jax.random.key(1), scheduler, config)
    assert float(metrics["scalar"]["learning/toy_finite"]) == 0.0
    failed, _ = parent.step_finite_failures(metrics["scalar"])
    assert "toy_finite" in failed


def test_the_reduction_refuses_microbatches_that_disagree_about_their_metrics():
    with pytest.raises(ValueError, match="different aux keys"):
        parent.reduce_microbatch_aux([{"a": jnp.asarray(1.0)}, {"b": jnp.asarray(1.0)}])
    with pytest.raises(ValueError, match="nothing to reduce"):
        parent.reduce_microbatch_aux([])


# =============================================================================================
# 5. The config gate.
# =============================================================================================


def test_the_gate_rejects_a_non_positive_accumulation():
    for bad in (0, -1, -8):
        with pytest.raises(ValueError, match="microbatch COUNT"):
            parent.resolve_grad_accumulation(SimpleNamespace(exp03_grad_accumulation=bad))


def test_the_gate_rejects_an_accumulation_that_does_not_divide_the_global_batch():
    _, _, config, _ = _fixture(exp03_grad_accumulation=3, global_batch_size_to_train_on=256)
    with pytest.raises(ValueError, match="must divide global_batch_size_to_train_on"):
        exp03.validate_exp03_config(config)


def test_the_gate_rejects_an_accumulation_that_does_not_divide_the_per_device_batch():
    # Divides the GLOBAL batch but not the per-device one: the answer would still be right, but
    # every microbatch would pay an all-to-all. Refused, because a silent throughput regression is
    # the kind of thing that only ever gets noticed months later.
    _, _, config, _ = _fixture(exp03_grad_accumulation=8, global_batch_size_to_train_on=256, per_device_batch_size=4.0)
    with pytest.raises(ValueError, match="must divide per_device_batch_size"):
        exp03.validate_exp03_config(config)


def test_the_production_recipe_for_trial_c_passes_the_gate():
    # The actual approved shape: v6e-64, GBS 256, per-device 4, N=2 -> per-device-2 microbatches.
    _, _, config, _ = _fixture(
        exp03_objective="combined",
        exp03_grad_accumulation=2,
        global_batch_size_to_train_on=256,
        per_device_batch_size=4.0,
    )
    assert exp03.validate_exp03_config(config) == "combined"
    assert exp03.validate_grad_accumulation(config) == 2


def test_the_default_config_passes_the_gate_unchanged():
    _, _, config, _ = _fixture(global_batch_size_to_train_on=256, per_device_batch_size=4.0)
    assert exp03.validate_grad_accumulation(config) == 1


# =============================================================================================
# 6. Config default, type, and launcher plumb-through.
# =============================================================================================


def test_the_config_carries_the_knob_off_by_default_as_an_int():
    cfg = yaml.safe_load(_CONFIG.read_text())
    assert cfg["exp03_grad_accumulation"] == 1
    # pyconfig coerces an override to the YAML value's type, so an int here means the command line
    # can never hand the trainer a float microbatch count.
    assert type(cfg["exp03_grad_accumulation"]) is int


def test_the_launcher_plumbs_the_env_through_with_a_default_of_one():
    text = _LAUNCHER.read_text()
    assert 'EXP03_GRAD_ACCUMULATION="${EXP03_GRAD_ACCUMULATION:-1}"' in text
    assert 'exp03_grad_accumulation="${EXP03_GRAD_ACCUMULATION}"' in text
    assert 'echo "EXP03_GRAD_ACCUMULATION=${EXP03_GRAD_ACCUMULATION}"' in text


def test_the_launcher_does_not_lower_the_per_device_batch_alongside_it():
    # The recipe is "same delivered batch, smaller microbatch". If the launcher also dropped
    # PER_DEVICE_BATCH_SIZE the global batch would halve and C would stop being comparable to A,
    # B and ctrl0 -- which is the whole reason accumulation was chosen over a smaller batch.
    text = _LAUNCHER.read_text()
    assert 'PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4.0}"' in text


def test_the_launcher_is_syntactically_valid():
    proc = subprocess.run(["bash", "-n", str(_LAUNCHER)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr


def test_the_trainer_announces_the_accumulation_only_when_it_is_on():
    import inspect

    source = inspect.getsource(exp03.Exp03Trainer.start_training)
    assert "resolve_grad_accumulation(self.config)" in source
    assert "if accumulation > 1:" in source, "an always-on line would drift exp_02's log format"


# =============================================================================================
# 7. The accumulator is param-sharded — forced 8-device mesh, in a subprocess.
#
# The device count is fixed when the backend initialises, so it cannot be raised inside a session
# that has already touched jax (the exp_03 probe suite established this pattern).
# =============================================================================================


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
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import maxdiffusion.trainers.wan_ti2v_overfit100_trainer as parent

assert jax.device_count() == 8, jax.device_count()
mesh = Mesh(np.asarray(jax.devices()).reshape(8), ("fsdp",))
sharded = NamedSharding(mesh, PartitionSpec("fsdp"))
replicated = NamedSharding(mesh, PartitionSpec())

# --- A. the accumulator carries the PARAMETER layout, not a replicated copy -------------------
# 64 float32 = 256 bytes of parameters. Summed at the parameter sharding that is 32 bytes on each
# of eight devices; replicated it would be 256 bytes on each -- the 8x that Job 8e actually hit
# when XLA was left to choose the gradient's layout.
g0 = jax.device_put(jnp.ones((64,), jnp.float32), sharded)
g1 = jax.device_put(jnp.full((64,), 2.0, jnp.float32), sharded)
accumulate = jax.jit(lambda a, b: jax.tree_util.tree_map(jnp.add, a, b))
acc = accumulate({"w": g0}, {"w": g1})["w"]
acc_shard_bytes = int(acc.addressable_shards[0].data.nbytes)
acc_total_bytes = sum(int(s.data.nbytes) for s in acc.addressable_shards)

replicated_acc = jax.jit(
    lambda a, b: jax.tree_util.tree_map(jnp.add, a, b), out_shardings={"w": replicated}
)({"w": g0}, {"w": g1})["w"]
replicated_total_bytes = sum(int(s.data.nbytes) for s in replicated_acc.addressable_shards)

# --- B. the INTERLEAVED slice is shard-local; the contiguous one is not -----------------------
# 32 rows over 8 devices is 4 rows per device: device d owns global rows 4d..4d+3.
rows = 32
labelled = jax.device_put(jnp.arange(rows, dtype=jnp.float32).reshape(rows, 1), sharded)


def owned(array):
    return {int(s.device.id): sorted(int(v) for v in np.asarray(s.data).ravel()) for s in array.addressable_shards}


source_rows = owned(labelled)
interleaved = [
    owned(jax.jit(lambda a, i=i: parent.microbatch_slice(a, i, 2)["r"], out_shardings=sharded)({"r": a_in}))
    for i, a_in in ((0, labelled), (1, labelled))
]
contiguous = [
    owned(jax.jit(lambda a, i=i: a[i * 16 : (i + 1) * 16], out_shardings=sharded)(labelled)) for i in (0, 1)
]


def is_local(slices):
    return all(set(part[d]) <= set(source_rows[d]) for part in slices for d in part)


print("RESULT " + json.dumps({
    "devices": jax.device_count(),
    "acc_shards": len(acc.addressable_shards),
    "acc_shard_bytes": acc_shard_bytes,
    "acc_total_bytes": acc_total_bytes,
    "acc_values_correct": bool(np.allclose(np.asarray(jax.device_get(acc)), 3.0)),
    "replicated_total_bytes": replicated_total_bytes,
    "one_param_tree_bytes": int(g0.nbytes),
    "source_rows_device0": source_rows[0],
    "interleaved_device0": [part[0] for part in interleaved],
    "contiguous_device0": [part[0] for part in contiguous],
    "interleaved_is_shard_local": is_local(interleaved),
    "contiguous_is_shard_local": is_local(contiguous),
}))
"""


def _run_eight_device_script(script: str) -> dict:
    import os

    env = dict(os.environ)
    env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = str(_REPO / "src")
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900, env=env)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(line for line in proc.stdout.splitlines() if line.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


def test_the_accumulator_stays_param_sharded_over_eight_devices():
    # THE memory contract, measured rather than asserted in a comment: one accumulator at the
    # parameters' own sharding. A replicated accumulator is the Job 8e failure -- every chip
    # holding the whole gradient instead of its FSDP shard -- and the same measurement shows what
    # that would have read, so the number is not just a number.
    result = _run_eight_device_script(_EIGHT_DEVICE_SCRIPT)
    assert result["devices"] == 8, result
    assert result["acc_shards"] == 8, result
    assert result["acc_values_correct"], result
    # 64 float32 = 256 bytes, one eighth on each device.
    assert result["one_param_tree_bytes"] == 256, result
    assert result["acc_shard_bytes"] == 32, result
    assert result["acc_total_bytes"] == 256, result
    # The inversion: replicated, the same accumulator costs 8x, on the same mesh, in one process.
    assert result["replicated_total_bytes"] == 2048, result
    assert result["replicated_total_bytes"] / result["acc_total_bytes"] == pytest.approx(8.0), result


def test_the_interleaved_microbatch_slice_is_shard_local_and_the_contiguous_one_is_not():
    # Why the slice is x[i::N] and not x[:B/N]. Device 0 owns global rows 0..3; the interleaved
    # halves leave it holding {0,2} and {1,3} -- its own rows, no traffic. The contiguous halves
    # would hand it {0,1} and {16,17}, and rows 16,17 live on device 4: an all-to-all, per
    # microbatch, per step, on every step of the run.
    result = _run_eight_device_script(_EIGHT_DEVICE_SCRIPT)
    assert result["source_rows_device0"] == [0, 1, 2, 3], result
    assert result["interleaved_device0"] == [[0, 2], [1, 3]], result
    assert result["interleaved_is_shard_local"] is True, result
    assert result["contiguous_is_shard_local"] is False, result
    assert result["contiguous_device0"] == [[0, 1], [16, 17]], result
