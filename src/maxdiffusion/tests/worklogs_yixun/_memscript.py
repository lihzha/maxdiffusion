"""Compiled-layout scratch measurement for the accumulation designs — run in a SUBPROCESS.

Not a test module (the leading underscore keeps pytest from collecting it): the device count is
fixed when the JAX backend initialises, so an 8-device mesh needs its own process. Compiles the
REAL train step -- so ``value_and_grad``'s layout is whatever XLA actually chooses, rather than
something a pre-sharded toy asserted into place -- and reports
``compiled.memory_analysis().temp_size_in_bytes``, i.e. program scratch: the quantity that
regressed by 755.16 MB on Job 15.

Argv: ``dim depth tokens``. Prints one ``RESULT {json}`` line.
"""

from __future__ import annotations

import json
import sys
import types

_grain = types.ModuleType("grain")
_grain_python = types.ModuleType("grain.python")
_grain_python.MapTransform = type("MapTransform", (), {})
_grain_python.RandomAccessDataSource = type("RandomAccessDataSource", (), {})
_grain.python = _grain_python
sys.modules["grain"] = _grain
sys.modules["grain.python"] = _grain_python

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flax import nnx  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec  # noqa: E402

import maxdiffusion.trainers.wan_ti2v_overfit100_trainer as parent  # noqa: E402

DIM, DEPTH, TOKENS = (int(value) for value in sys.argv[1:4])
PER_DEVICE_BATCH = 4
MESH = Mesh(np.asarray(jax.devices()).reshape(8), ("fsdp",))
GLOBAL_BATCH = PER_DEVICE_BATCH * 8


def _block(x, w1, w2):
    return jnp.tanh(x @ w1) @ w2


# The production model sets remat_policy: FULL, so the measured model rematerializes too --
# otherwise the harness would be answering a question about a model nobody runs.
_block_remat = jax.checkpoint(_block)


class Stack(nnx.Module):
    def __init__(self, dim, depth, key):
        keys = jax.random.split(key, depth * 2)
        self.w1 = nnx.List(
            [nnx.Param(jax.random.normal(keys[2 * i], (dim, dim), jnp.float32) * 0.02) for i in range(depth)]
        )
        self.w2 = nnx.List(
            [nnx.Param(jax.random.normal(keys[2 * i + 1], (dim, dim), jnp.float32) * 0.02) for i in range(depth)]
        )

    def __call__(self, **kwargs):
        x = kwargs["hidden_states"].astype(jnp.float32)
        for w1, w2 in zip(self.w1, self.w2):
            x = x + _block_remat(x, w1[...], w2[...])
        return x


def loss_fn(params, state, data, rng, config, scheduler, *, global_step=None):
    del rng, scheduler, global_step
    model = nnx.merge(state.graphdef, params, state.rest_of_state)
    prediction = model(hidden_states=data["x"].astype(jnp.float32))
    loss = jnp.mean((prediction - data["y"].astype(jnp.float32)) ** 2)
    zero = jnp.asarray(0.0, jnp.float32)
    return loss, {
        "velocity_mse": loss,
        "sigma_mean": zero,
        "timestep_mean": zero,
        "v_pred_l2": zero,
        "v_target_l2": zero,
        "z_noisy_std": zero,
        "z_target_std": zero,
        "z_init_anchor_mse": zero,
        "loss_finite": jnp.isfinite(loss).astype(jnp.float32),
    }


def build():
    graphdef, params, rest = nnx.split(Stack(DIM, DEPTH, jax.random.key(0)), nnx.Param, ...)
    state = parent.Overfit100TrainState.create(
        apply_fn=graphdef.apply,
        params=params,
        tx=optax.adam(1e-4),
        graphdef=graphdef,
        rest_of_state=rest,
        context_table=jnp.zeros((1, 1, 1), jnp.float32),
    )

    # Real FSDP: parameters and both Adam moments sharded on their leading axis, batch sharded too.
    # Adam's step counter is a rank-0 leaf and can only be replicated, so the spec is rank-aware.
    def put(leaf):
        spec = PartitionSpec("fsdp") if jnp.ndim(leaf) >= 1 else PartitionSpec()
        return jax.device_put(leaf, NamedSharding(MESH, spec))

    state = state.replace(
        params=jax.tree_util.tree_map(put, state.params),
        opt_state=jax.tree_util.tree_map(put, state.opt_state),
    )
    data = {
        "x": put(jax.random.normal(jax.random.key(1), (GLOBAL_BATCH, TOKENS, DIM), jnp.float32)),
        "y": put(jax.random.normal(jax.random.key(2), (GLOBAL_BATCH, TOKENS, DIM), jnp.float32)),
    }
    return state, data


def unrolled_step(denoising_loss, num_microbatches):
    """THE REGRESSION WITNESS: the pre-redesign unrolled+barrier step, kept so the fix is a measured
    comparison against the thing it replaced rather than an assertion about it."""

    def _train_step(state, data, rng, scheduler, config, *, global_step=None):
        rng, loss_rng = jax.random.split(rng)
        grads = None
        losses = []
        for index in range(num_microbatches):
            micro = parent.microbatch_slice(data, index, num_microbatches)
            if grads is not None:
                grads, micro = jax.lax.optimization_barrier((grads, micro))

            def one(params, batch=micro, key=jax.random.fold_in(loss_rng, index)):
                return denoising_loss(params, state, batch, key, config, None, global_step=global_step)

            (micro_loss, _), micro_grads = nnx.value_and_grad(one, has_aux=True)(state.params)
            grads = micro_grads if grads is None else jax.tree_util.tree_map(jnp.add, grads, micro_grads)
            losses.append(micro_loss)
        grads = jax.tree_util.tree_map(lambda leaf: leaf / num_microbatches, grads)
        state = state.apply_gradients(grads=grads)
        return state, {"scalar": {"learning/loss": sum(losses) / num_microbatches}}, rng

    return _train_step


def temp_bytes(step_fn, state, data, accumulation):
    config = types.SimpleNamespace(exp03_grad_accumulation=accumulation)

    def wrapped(state_, data_, rng_):
        return step_fn(state_, data_, rng_, None, config, global_step=jnp.asarray(0, jnp.int32))

    compiled = jax.jit(wrapped, donate_argnums=(0,)).lower(state, data, jax.random.key(3)).compile()
    return int(compiled.memory_analysis().temp_size_in_bytes)


def main():
    state, data = build()
    result = {
        "devices": jax.device_count(),
        "params": int(sum(x.size for x in jax.tree_util.tree_leaves(state.params))),
        "dim": DIM,
        "depth": DEPTH,
        "tokens": TOKENS,
        "n1": temp_bytes(parent._make_train_step(loss_fn), state, data, 1),
        "n2_scan": temp_bytes(parent._make_train_step(loss_fn), state, data, 2),
        "n2_unrolled": temp_bytes(unrolled_step(loss_fn, 2), state, data, 2),
        "n4_scan": temp_bytes(parent._make_train_step(loss_fn), state, data, 4),
    }
    print("RESULT " + json.dumps(result))


main()
