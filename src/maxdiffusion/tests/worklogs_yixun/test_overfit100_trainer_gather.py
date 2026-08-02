"""CPU-only tests for the exp_02 overfit100 per-episode context gather (plan D7 / G1).

G1 (Codex plan-review v2, BLOCKER) established that the full-FT trainer's step
functions are **module-level** and jit-bound inside ``start_training``, so a subclass
method cannot swap the null text embedding for a per-episode table. The overfit100
module therefore owns its own ``Overfit100TrainState`` (``context_table`` REPLACES
``null_context``), its own module-level ``_denoising_loss`` / ``_train_step`` /
``_eval_step``, and G1's mandated test lives here:

  (A) ROW-DISTINCT GATHER -- a batch whose two examples carry DIFFERENT
      ``episode_index`` values must reach the transformer with row-distinct
      ``encoder_hidden_states``, each equal to its own table row. The fixture is
      deliberately built so ``episode_index != episode_id`` (the batch even carries a
      decoy ``episode_id`` column whose values would index different rows), so a
      mutant that gathers on the wrong column -- or broadcasts row 0 -- fails.
  (B) OBJECTIVE PARITY -- with every table row equal to the full-FT null embedding,
      ``overfit100._denoising_loss`` must equal ``full_ft._denoising_loss``
      **bitwise** on identical inputs/rng/params. That is the byte-parity claim of
      plan D7: same shared helpers, same frame-0 pin, same ``eps - z_video`` target,
      same fresh noise, no actions / adapter / CFG -- the gather is the ONLY delta.

CPU-only: a tiny recording ``nnx.Module`` stub with a real ``Param`` stands in for the
5B transformer; no weights, no pipeline, no mesh. The darwin grain import stub lives
in ``conftest.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

import maxdiffusion.trainers.wan_ti2v_full_ft_trainer as full_ft
import maxdiffusion.trainers.wan_ti2v_overfit100_trainer as overfit100
from maxdiffusion.schedulers import FlaxFlowMatchScheduler

# Tiny fixed shapes: data batch 3 (sliced to bsz 2), channels C, latent frames F, H, W.
_DATA_B, _BSZ, _C, _F, _H, _W = 3, 2, 3, 4, 5, 6
_SLOTS, _LEN, _DIM = 5, 4, 8  # context table [slots, seq_len, text_dim]
_STUB_GAIN = 0.5

# The recording stub appends every call's kwargs here (module-level so it survives
# nnx.split/merge -- a method's global namespace is untouched by nnx graph ops).
_STUB_CALLS: list[dict] = []


@pytest.fixture(autouse=True)
def _clear_stub_calls():
    _STUB_CALLS.clear()
    yield
    _STUB_CALLS.clear()


class _RecordingStubTransformer(nnx.Module):
    """Stand-in for the trainable WAN transformer: a real ``Param`` + call recording."""

    def __init__(self, gain=_STUB_GAIN):
        self.gain = nnx.Param(jnp.asarray(gain, dtype=jnp.float32))

    def __call__(self, **kwargs):
        _STUB_CALLS.append(kwargs)
        hidden = kwargs["hidden_states"].astype(jnp.float32)
        ctx_rows = jnp.mean(kwargs["encoder_hidden_states"].astype(jnp.float32), axis=(1, 2))
        # Row-dependent velocity: the per-example context actually influences the output,
        # so a gather bug is observable in the loss as well as in the recorded kwargs.
        return self.gain[...] * hidden + ctx_rows.reshape((-1, 1, 1, 1, 1))


def _config(*, weights_dtype="float32", activations_dtype="float32"):
    return SimpleNamespace(
        weights_dtype=weights_dtype,
        activations_dtype=activations_dtype,
        global_batch_size_to_train_on=_BSZ,
        side_adapter_sampling_steps=4,
        flow_shift=5.0,
        side_adapter_t_sampling="uniform",
        side_adapter_noise_mode="fresh",
        seed=0,
    )


def _context_table(seed=7):
    """Row-distinct table: row i is a constant-i offset plus noise, so rows never collide."""
    base = jax.random.normal(jax.random.key(seed), (_SLOTS, _LEN, _DIM), dtype=jnp.float32)
    return base + jnp.arange(_SLOTS, dtype=jnp.float32).reshape((_SLOTS, 1, 1)) * 10.0


def _data(*, episode_index, data_dtype=jnp.float32, decoy_ids=True):
    k1, k2 = jax.random.split(jax.random.key(42), 2)
    data = {
        "z_i0": jax.random.normal(k1, (_DATA_B, _C, 1, _H, _W), dtype=jnp.float32).astype(data_dtype),
        "z_video": jax.random.normal(k2, (_DATA_B, _C, _F, _H, _W), dtype=jnp.float32).astype(data_dtype),
        "episode_index": jnp.asarray(episode_index, dtype=jnp.int32),
    }
    if decoy_ids:
        # index != id: the raw DROID episode ids are 5-digit (manifest index 0 -> id 25189).
        # Reduced mod _SLOTS they would select DIFFERENT rows, so gathering on the wrong
        # column cannot accidentally agree with gathering on episode_index.
        data["episode_id"] = jnp.asarray([25189, 31007, 40961], dtype=jnp.int32)
    return data


def _make_state(context_table, *, tx=None, gain=_STUB_GAIN):
    transformer = _RecordingStubTransformer(gain)
    graphdef, params, rest = nnx.split(transformer, nnx.Param, ...)
    if tx is None:
        return overfit100.Overfit100TrainState(
            step=0,
            apply_fn=None,
            params=params,
            tx=None,
            opt_state=None,
            graphdef=graphdef,
            rest_of_state=rest,
            context_table=context_table,
        )
    return overfit100.Overfit100TrainState.create(
        apply_fn=graphdef.apply,
        params=params,
        tx=tx,
        graphdef=graphdef,
        rest_of_state=rest,
        context_table=context_table,
    )


def _scheduler(config):
    return FlaxFlowMatchScheduler(dtype=jnp.float32, shift=config.flow_shift, sigma_min=0.0, sigma_max=1.0)


_RNG = jax.random.key(123)


# =======================================================================================
# (A) G1's mandated test: row-distinct per-episode context reaches the transformer.
# =======================================================================================


def test_batch_with_two_episode_indices_gets_row_distinct_context():
    table = _context_table()
    state = _make_state(table)
    config = _config()
    # Row 3 and row 1: distinct, non-zero, and neither is row 0 (kills a row-0 broadcast).
    data = _data(episode_index=[3, 1, 4])
    overfit100._denoising_loss(state.params, state, data, _RNG, config, _scheduler(config))

    assert len(_STUB_CALLS) == 1
    ctx = _STUB_CALLS[0]["encoder_hidden_states"]
    assert ctx.shape == (_BSZ, _LEN, _DIM)
    # Each row is EXACTLY its own episode's table row...
    np.testing.assert_array_equal(np.asarray(ctx[0]), np.asarray(table[3]))
    np.testing.assert_array_equal(np.asarray(ctx[1]), np.asarray(table[1]))
    # ...the two rows genuinely differ (non-vacuity for "row-distinct")...
    assert not np.allclose(np.asarray(ctx[0]), np.asarray(ctx[1]))
    # ...and neither equals row 0 (a broadcast-row-0 mutant fails here).
    assert not np.allclose(np.asarray(ctx[0]), np.asarray(table[0]))
    assert not np.allclose(np.asarray(ctx[1]), np.asarray(table[0]))


def test_gather_uses_episode_index_not_episode_id():
    # The decoy episode_id column (25189, 31007, ...) would select different rows mod
    # _SLOTS; the gather must ignore it entirely.
    table = _context_table()
    state = _make_state(table)
    config = _config()
    data = _data(episode_index=[2, 0, 1])
    overfit100._denoising_loss(state.params, state, data, _RNG, config, _scheduler(config))
    ctx = _STUB_CALLS[0]["encoder_hidden_states"]
    np.testing.assert_array_equal(np.asarray(ctx[0]), np.asarray(table[2]))
    np.testing.assert_array_equal(np.asarray(ctx[1]), np.asarray(table[0]))
    # The would-be rows if episode_id had been used (mod slots) differ from what we got.
    assert 25189 % _SLOTS != 2 and 31007 % _SLOTS != 0


def test_context_slice_follows_the_trained_batch_size():
    # Data batch is 3 while global_batch_size_to_train_on is 2: episode_index must be
    # sliced with the latents, so the context has _BSZ rows built from the FIRST two.
    table = _context_table()
    state = _make_state(table)
    config = _config()
    data = _data(episode_index=[3, 1, 4])
    overfit100._denoising_loss(state.params, state, data, _RNG, config, _scheduler(config))
    ctx = _STUB_CALLS[0]["encoder_hidden_states"]
    assert ctx.shape[0] == _BSZ  # not _DATA_B
    np.testing.assert_array_equal(np.asarray(ctx[1]), np.asarray(table[1]))  # index 1, not 4


def test_context_cast_follows_activations_dtype():
    # Parity with full-FT's null-context handling (its accepted deviation): the context
    # is cast with activations_dtype, NOT weights_dtype.
    table = _context_table()
    state = _make_state(table)
    config = _config(weights_dtype="float32", activations_dtype="bfloat16")
    data = _data(episode_index=[3, 1, 4])
    overfit100._denoising_loss(state.params, state, data, _RNG, config, _scheduler(config))
    ctx = _STUB_CALLS[0]["encoder_hidden_states"]
    assert ctx.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(ctx[0]), np.asarray(table[3].astype(jnp.bfloat16)))


def test_exactly_one_plain_transformer_call_without_actions():
    table = _context_table()
    state = _make_state(table)
    config = _config()
    data = _data(episode_index=[3, 1, 4])
    data["actions"] = jax.random.normal(jax.random.key(9), (_DATA_B, 32, 7), dtype=jnp.float32)
    overfit100._denoising_loss(state.params, state, data, _RNG, config, _scheduler(config))
    assert len(_STUB_CALLS) == 1  # no CFG second branch
    call = _STUB_CALLS[0]
    assert set(call.keys()) == {"hidden_states", "timestep", "encoder_hidden_states", "deterministic", "rngs"}
    assert not any("adapter" in k or "action" in k for k in call)
    assert call["deterministic"] is False


# =======================================================================================
# (B) Objective parity: table row == null embedding => bitwise-identical to full-FT.
# =======================================================================================


def _paired_states(null_row):
    """A full-FT state and an overfit100 state sharing params/graphdef, table rows == null."""
    transformer = _RecordingStubTransformer()
    graphdef, params, rest = nnx.split(transformer, nnx.Param, ...)
    null_context = null_row.reshape((1, _LEN, _DIM))
    table = jnp.broadcast_to(null_context, (_SLOTS, _LEN, _DIM))
    ft_state = full_ft.FullFTTrainState(
        step=0,
        apply_fn=None,
        params=params,
        tx=None,
        opt_state=None,
        graphdef=graphdef,
        rest_of_state=rest,
        null_context=null_context,
    )
    of_state = overfit100.Overfit100TrainState(
        step=0,
        apply_fn=None,
        params=params,
        tx=None,
        opt_state=None,
        graphdef=graphdef,
        rest_of_state=rest,
        context_table=table,
    )
    return ft_state, of_state


@pytest.mark.parametrize("dtypes", [("float32", "float32"), ("bfloat16", "bfloat16")])
def test_objective_parity_with_full_ft_when_table_rows_equal_null_context(dtypes):
    weights_dtype, activations_dtype = dtypes
    null_row = jax.random.normal(jax.random.key(41), (_LEN, _DIM), dtype=jnp.float32)
    ft_state, of_state = _paired_states(null_row)
    config = _config(weights_dtype=weights_dtype, activations_dtype=activations_dtype)
    scheduler = _scheduler(config)
    # Different episode indices -- with equal rows they must STILL produce the full-FT value.
    data = _data(episode_index=[3, 1, 4])

    ft_loss, ft_aux = full_ft._denoising_loss(ft_state.params, ft_state, data, _RNG, config, scheduler)
    of_loss, of_aux = overfit100._denoising_loss(of_state.params, of_state, data, _RNG, config, scheduler)

    np.testing.assert_array_equal(np.asarray(of_loss), np.asarray(ft_loss))
    assert set(of_aux) == set(ft_aux)
    for key in ft_aux:
        np.testing.assert_array_equal(np.asarray(of_aux[key]), np.asarray(ft_aux[key]), err_msg=key)
    # Non-vacuity: the two calls really ran (2 forwards recorded) and the loss is finite.
    assert len(_STUB_CALLS) == 2
    assert np.isfinite(float(ft_loss))


def test_parity_breaks_when_table_rows_differ():
    # Guard against a vacuous parity test: with a genuinely row-distinct table the
    # overfit100 loss must MOVE OFF the full-FT (null-broadcast) value.
    null_row = jax.random.normal(jax.random.key(41), (_LEN, _DIM), dtype=jnp.float32)
    ft_state, _ = _paired_states(null_row)
    of_state = _make_state(_context_table())
    of_state = of_state.replace(
        params=ft_state.params, graphdef=ft_state.graphdef, rest_of_state=ft_state.rest_of_state
    )
    config = _config()
    scheduler = _scheduler(config)
    data = _data(episode_index=[3, 1, 4])
    ft_loss, _ = full_ft._denoising_loss(ft_state.params, ft_state, data, _RNG, config, scheduler)
    of_loss, _ = overfit100._denoising_loss(of_state.params, of_state, data, _RNG, config, scheduler)
    assert float(of_loss) != float(ft_loss)


# =======================================================================================
# (C) The module-level step functions bind THIS module's loss.
# =======================================================================================


def test_train_step_updates_params_and_reports_finite_metrics():
    state = _make_state(_context_table(), tx=optax.sgd(0.1))
    config = _config()
    data = _data(episode_index=[3, 1, 4])
    before = np.asarray(jax.tree_util.tree_leaves(state.params)[0])
    new_state, metrics, _ = overfit100._train_step(state, data, _RNG, _scheduler(config), config)
    after = np.asarray(jax.tree_util.tree_leaves(new_state.params)[0])
    assert not np.array_equal(before, after)  # the optimizer moved the param
    assert float(metrics["scalar"]["learning/grad_norm"]) > 0.0
    for key in ("learning/loss", "learning/velocity_mse", "learning/max_abs_grad", "learning/z_init_anchor_mse"):
        assert np.isfinite(float(metrics["scalar"][key])), key
    # The context table is carried through the optimizer step untouched.
    np.testing.assert_array_equal(np.asarray(new_state.context_table), np.asarray(state.context_table))


def test_train_and_eval_steps_bind_this_modules_loss(monkeypatch):
    calls = []
    real = overfit100._denoising_loss

    def spy(params, state, data, rng, config, scheduler):
        calls.append("overfit100")
        return real(params, state, data, rng, config, scheduler)

    monkeypatch.setattr(overfit100, "_denoising_loss", spy)

    def boom(*args, **kwargs):
        raise AssertionError("overfit100 step functions must not call the full-FT loss")

    monkeypatch.setattr(full_ft, "_denoising_loss", boom)
    state = _make_state(_context_table(), tx=optax.sgd(0.1))
    config = _config()
    data = _data(episode_index=[3, 1, 4])
    overfit100._train_step(state, data, _RNG, _scheduler(config), config)
    overfit100._eval_step(state, data, _RNG, _scheduler(config), config)
    assert calls == ["overfit100", "overfit100"]


def test_eval_step_reports_per_example_loss_vector():
    state = _make_state(_context_table())
    config = _config()
    data = _data(episode_index=[3, 1, 4])
    metrics, _ = overfit100._eval_step(state, data, _RNG, _scheduler(config), config)
    losses = np.asarray(metrics["scalar"]["learning/eval_loss"])
    assert losses.shape == (_BSZ,)
    assert np.all(np.isfinite(losses))
