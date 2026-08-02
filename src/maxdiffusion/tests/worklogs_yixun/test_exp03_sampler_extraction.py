"""exp_03 cycle A round 1 — the sampler-step extraction is provably inert.

The eval rollout's bitwise reproducibility is a measured asset (the exp_02 sampling probe reproduced
the landed 30-window scalars to 0.0, twice; the video passes reproduced per-window bitwise). This
round moves the step out of ``generate_wan_side_adapter.py`` so a trainer can unroll the SAME
operator under ``jax.grad``. Three claims are pinned here, and only the third needs hardware:

1. **Numerical parity (exact).** A reference implementation of the step is copied VERBATIM into this
   file from the pre-extraction source; the extracted function must reproduce it bit for bit --
   ``array_equal``, not ``allclose`` -- over the real 25-step grid, at every index, in float32 and in
   bfloat16.
2. **No duplicate implementation.** An AST-level test asserts each eval rollout body calls the
   extracted step and that neither the Euler update nor the per-token-timestep construction survives
   anywhere in ``generate_wan_side_adapter.py``. One sampler, one definition (plan §3).
3. The existing synthetic-driver tests (``test_overfit100_context_modes.py``,
   ``test_overfit100_sampling_probe.py``) execute the real loop and are NOT modified by this round --
   they are the behavioural half of the argument. The on-hardware bitwise gate against the landed
   scalars happens at S1.5.

Plus the trainer-facing contract: the step must be usable under ``jax.grad``/``jax.remat``, which
means no Python side effects inside it, and it must not import the eval module (a trainer must be
able to take the sampler without taking the evaluator).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import maxdiffusion.generate_wan_side_adapter as gen
from maxdiffusion.models.wan import overfit100_sampling as sampling
from maxdiffusion.models.wan.side_adapter_wan import (
    _build_per_token_timestep,
    apply_first_frame_pin,
    build_rollout_sigmas,
    rollout_timesteps_from_sigmas,
)

_GEN_SOURCE = Path(gen.__file__).read_text()

# The eval's real geometry (12x20 latent grid, 9 latent frames) shrunk to something a laptop can
# run: what matters for parity is the op sequence and the dtypes, not the sizes.
_B, _C, _F, _H, _W = 1, 4, 3, 4, 6
_STEPS, _SHIFT, _SIGMA_MIN, _SIGMA_MAX, _NUM_TRAIN_TIMESTEPS = 25, 5.0, 0.0, 1.0, 1000


# ---------------------------------------------------------------------------------------------
# The reference: copied verbatim from generate_wan_side_adapter.py BEFORE the extraction
# (commit eb7a59c, ``_rollout_overfit100_sample``). Do not refactor it -- its value is that it is
# the old code, not that it is pretty.
# ---------------------------------------------------------------------------------------------


def _reference_grid():
    sigmas = build_rollout_sigmas(_STEPS, _SHIFT, _SIGMA_MIN, _SIGMA_MAX)
    timesteps = rollout_timesteps_from_sigmas(sigmas, _NUM_TRAIN_TIMESTEPS)
    return sigmas, timesteps


def _reference_body(i, current, *, transformer, sigmas, timesteps, context, z_i0, b, f_lat, h_lat, w_lat):
    step_t = jnp.broadcast_to(timesteps[i], (b,))
    timestep_2d = _build_per_token_timestep(step_t, f_lat, h_lat, w_lat, n_hist=1)
    v = transformer(
        hidden_states=current,
        timestep=timestep_2d,
        encoder_hidden_states=context,
        deterministic=True,
    )
    return apply_first_frame_pin(current + (sigmas[i + 1] - sigmas[i]).astype(current.dtype) * v, z_i0)


class _StubTransformer:
    """A deterministic stand-in for the ~5B velocity model.

    It must READ every input the real one reads -- latents, per-token timestep, context -- or the
    parity test would not see a change in how they are built.
    """

    def __call__(self, *, hidden_states, timestep, encoder_hidden_states, deterministic=True):
        b, c, f_lat, h_lat, w_lat = hidden_states.shape
        tokens = timestep.reshape(b, -1).astype(jnp.float32)
        # One scalar per example that depends on EVERY per-token timestep entry.
        t_scalar = jnp.mean(jnp.sin(tokens / 137.0), axis=-1)[:, None, None, None, None]
        ctx = jnp.mean(encoder_hidden_states.astype(jnp.float32))
        v = jnp.tanh(hidden_states.astype(jnp.float32) * 0.5) + t_scalar + 0.01 * ctx
        del c, f_lat, h_lat, w_lat, deterministic
        return v.astype(hidden_states.dtype)


def _inputs(dtype):
    key = jax.random.key(0)
    k_z, k_i0, k_ctx = jax.random.split(key, 3)
    z = jax.random.normal(k_z, (_B, _C, _F, _H, _W), dtype=jnp.float32).astype(dtype)
    z_i0 = jax.random.normal(k_i0, (_B, _C, 1, _H, _W), dtype=jnp.float32).astype(dtype)
    context = jax.random.normal(k_ctx, (_B, 7, 8), dtype=jnp.float32).astype(dtype)
    return apply_first_frame_pin(z, z_i0), z_i0, context


# ---------------------------------------------------------------------------------------------
# 1. Numerical parity -- exact.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_the_extracted_step_reproduces_the_pre_extraction_step_bit_for_bit(dtype):
    sigmas, timesteps = _reference_grid()
    z, z_i0, context = _inputs(dtype)
    transformer = _StubTransformer()

    def velocity_fn(hidden_states, timestep, encoder_hidden_states):
        return transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            deterministic=True,
        )

    for index in range(_STEPS):
        expected = _reference_body(
            index,
            z,
            transformer=transformer,
            sigmas=sigmas,
            timesteps=timesteps,
            context=context,
            z_i0=z_i0,
            b=_B,
            f_lat=_F,
            h_lat=_H,
            w_lat=_W,
        )
        got = sampling.overfit100_sampler_step(
            z,
            index,
            velocity_fn=velocity_fn,
            sigmas=sigmas,
            timesteps=timesteps,
            context=context,
            z_i0=z_i0,
        )
        assert got.dtype == expected.dtype
        assert np.array_equal(np.asarray(got), np.asarray(expected)), f"index {index} drifted ({dtype})"


def test_the_whole_rollout_reproduces_the_pre_extraction_loop_bit_for_bit():
    # Compounding is the point: 25 chained steps amplify any drift the single-step test could miss.
    sigmas, timesteps = _reference_grid()
    z0, z_i0, context = _inputs(jnp.float32)
    transformer = _StubTransformer()

    def velocity_fn(hidden_states, timestep, encoder_hidden_states):
        return transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            deterministic=True,
        )

    reference = jax.lax.fori_loop(
        0,
        _STEPS,
        lambda i, current: _reference_body(
            i,
            current,
            transformer=transformer,
            sigmas=sigmas,
            timesteps=timesteps,
            context=context,
            z_i0=z_i0,
            b=_B,
            f_lat=_F,
            h_lat=_H,
            w_lat=_W,
        ),
        z0,
    )
    got = jax.lax.fori_loop(
        0,
        _STEPS,
        lambda i, current: sampling.overfit100_sampler_step(
            current,
            i,
            velocity_fn=velocity_fn,
            sigmas=sigmas,
            timesteps=timesteps,
            context=context,
            z_i0=z_i0,
        ),
        z0,
    )
    assert np.array_equal(np.asarray(got), np.asarray(reference))


def test_the_grid_helper_reproduces_the_pre_extraction_grid_exactly():
    sigmas, timesteps = _reference_grid()
    got_sigmas, got_timesteps = sampling.overfit100_sampler_grid(
        num_inference_steps=_STEPS,
        flow_shift=_SHIFT,
        sigma_min=_SIGMA_MIN,
        sigma_max=_SIGMA_MAX,
        num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
    )
    assert np.array_equal(np.asarray(got_sigmas), np.asarray(sigmas))
    assert np.array_equal(np.asarray(got_timesteps), np.asarray(timesteps))
    # The grid the trials draw their supports from: 26 sigmas, descending, terminal exactly 0, and
    # the smallest POSITIVE sigma is the reviewer-verified 0.1724 (no clamp needed).
    values = np.asarray(got_sigmas, dtype=np.float64)
    assert values.shape == (_STEPS + 1,)
    assert values[0] == 1.0 and values[-1] == 0.0
    assert np.all(np.diff(values) < 0.0)
    assert abs(values[-2] - 0.1724137931) < 1e-6


def test_the_euler_update_is_exposed_on_its_own_and_pins_frame_zero():
    # Trial A's corrective target is derived from THIS contraction, so the update is importable
    # without a network: z_next - z_gt = (sigma_next / sigma) * (z - z_gt).
    sigmas, _ = _reference_grid()
    z, z_i0, _ = _inputs(jnp.float32)
    z_gt = jnp.zeros_like(z)
    z_gt = apply_first_frame_pin(z_gt, z_i0)
    index = 3
    sigma, sigma_next = float(sigmas[index]), float(sigmas[index + 1])
    v_star = (z - z_gt) / sigma
    z_next = sampling.overfit100_euler_update(z, v_star, sigmas, index, z_i0)
    expected = z_gt + (sigma_next / sigma) * (z - z_gt)
    assert np.allclose(np.asarray(z_next), np.asarray(expected), atol=1e-6)
    assert np.array_equal(np.asarray(z_next[:, :, :1]), np.asarray(z_i0))


# ---------------------------------------------------------------------------------------------
# 2. One sampler, one definition.
# ---------------------------------------------------------------------------------------------


def _function_node(source: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _called_names(node: ast.AST) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


@pytest.mark.parametrize("rollout", ["_rollout_overfit100_sample", "_rollout_sample"])
def test_each_eval_rollout_calls_the_extracted_step(rollout):
    called = _called_names(_function_node(_GEN_SOURCE, rollout))
    assert "overfit100_sampler_step" in called, f"{rollout} does not route through the extracted step"
    assert "overfit100_sampler_grid" in called, f"{rollout} builds its own sigma grid"
    # ...and does NOT re-implement the tail itself. (The INITIAL pin before the loop stays: it is
    # the rollout's own initialization, not part of the step.)
    assert "_build_per_token_timestep" not in called
    body = ast.unparse(_function_node(_GEN_SOURCE, rollout))
    assert "sigmas[i + 1] - sigmas[i]" not in body


def test_no_duplicate_step_implementation_survives_in_the_eval_module():
    # The whole file, not just the two rollouts: a copy anywhere is a second sampler.
    assert "sigmas[i + 1] - sigmas[i]" not in _GEN_SOURCE
    assert "_build_per_token_timestep(" not in _GEN_SOURCE
    assert "n_hist=1" not in _GEN_SOURCE
    # build_rollout_sigmas/rollout_timesteps_from_sigmas are the grid's ONE construction and now
    # live behind overfit100_sampler_grid.
    assert "build_rollout_sigmas(" not in _GEN_SOURCE
    assert "rollout_timesteps_from_sigmas(" not in _GEN_SOURCE


def test_the_extracted_module_does_not_import_the_evaluator():
    # A trainer must be able to take the sampler without dragging in the eval stack (which imports
    # tensorflow, orbax, the verdict statistic...).
    tree = ast.parse(Path(sampling.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert imported, "no imports parsed -- the AST walk is not looking at the right file"
    for name in imported:
        assert "generate_wan_side_adapter" not in name, name
        assert "overfit100_success_statistic" not in name, name
        assert not name.startswith("tensorflow"), name
        assert not name.startswith("orbax"), name


# ---------------------------------------------------------------------------------------------
# 3. The trainer-facing contract: differentiable, remat-able, side-effect free.
# ---------------------------------------------------------------------------------------------


def test_the_step_carries_no_python_side_effects():
    # Anything that talks to the host is illegal inside a step a trainer will remat and scan over.
    source = Path(sampling.__file__).read_text()
    for forbidden in ("print(", "max_logging", "jax.debug", "open(", "np.save", "logging."):
        assert forbidden not in source, forbidden
    for name in ("overfit100_sampler_step", "overfit100_euler_update", "overfit100_step_timestep"):
        body = _function_node(source, name)
        called = _called_names(body)
        assert not called & {"print", "log", "debug_print", "callback"}


def test_the_step_is_differentiable_and_rematable():
    sigmas, timesteps = _reference_grid()
    z, z_i0, context = _inputs(jnp.float32)
    scale = jnp.asarray(0.3, dtype=jnp.float32)

    def loss(scale_value):
        def velocity_fn(hidden_states, timestep, encoder_hidden_states):
            del timestep, encoder_hidden_states
            return hidden_states * scale_value

        z1 = sampling.overfit100_sampler_step(
            z, 0, velocity_fn=velocity_fn, sigmas=sigmas, timesteps=timesteps, context=context, z_i0=z_i0
        )
        z2 = sampling.overfit100_sampler_step(
            z1, 1, velocity_fn=velocity_fn, sigmas=sigmas, timesteps=timesteps, context=context, z_i0=z_i0
        )
        return jnp.sum(z2**2)

    grad = jax.grad(loss)(scale)
    assert np.isfinite(float(grad))
    assert float(grad) != 0.0  # gradient really flows through the unrolled steps
    remat_grad = jax.grad(jax.remat(loss))(scale)
    assert np.allclose(float(remat_grad), float(grad), rtol=1e-5)


def test_the_step_signature_keeps_the_state_explicit():
    # "explicit args in, state out": nothing is read from module state or a config object.
    signature = inspect.signature(sampling.overfit100_sampler_step)
    assert list(signature.parameters)[:2] == ["z", "index"]
    for required in ("velocity_fn", "sigmas", "timesteps", "context", "z_i0"):
        assert required in signature.parameters
        assert signature.parameters[required].kind is inspect.Parameter.KEYWORD_ONLY
    assert "config" not in signature.parameters and "state" not in signature.parameters
