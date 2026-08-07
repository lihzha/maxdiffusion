"""exp_06 `rollout_adapter` — T2 `loss-kernels`: the R-B endpoint loss, pinned three ways.

T1 gave exp_06 a pinned copy of exp_03's sampler and its RNG/support primitives. T2 gives it the
loss those primitives feed: exp_03's ``_rollout_loss`` (pin lines 435-510), extracted into
``maxdiffusion.pos_rollout_losses`` with its config coupling removed, plus the two helpers it stands
on re-homed out of a file exp_06's tree does not carry them in.

Three independent kinds of evidence, because "the same function exp_03 validated" is the whole
claim of this round:

1. **Double equivalence (§5-2, the heart of the round).** ``masked_velocity_mse`` and
   ``build_noisy_pinned_latents`` are proven **bitwise** equal to (a) verbatim copies of exp_03's
   pinned construction at ``side_adapter_wan.py:537-580`` of the pin, AND (b) verbatim copies of
   exp_06's OWN inline trainer math at ``trainers/wan_ti2v_side_adapter_trainer.py:161-165,192-196``
   — on the same fixtures, in one test each, so a disagreement is impossible to miss. Copy (b) is
   held honest by an AST drift tripwire against the trainer it was copied from.
2. **Analytic oracles.** Hand-computed closed forms in which the horizon normalization, the frame-0
   mask, the per-example reduction and the sigma interval are each *individually* falsifiable — not
   a single end-to-end number that any of four mistakes could produce. The two identities:
   *at the optimum ``v = eps - z_gt`` the loss is exactly zero at every support*, and *for a
   constant velocity offset ``c`` the loss is exactly ``mean(c**2)``, independent of the support* —
   the second is precisely what horizon-normalization means, and it fails loudly if the divisor is
   dropped, inverted, or taken over the wrong sigma interval.
3. **Contract pins.** The kernel is jittable / differentiable / rematable (T3a's precondition), it
   applies NO ``stop_gradient`` of its own (the §3a contract belongs to T3a's step, and a
   kernel-level stop-grad would silently pre-empt it), and the module performs NO config access at
   all (issue #11 — exp_03's version read `seed`/`k_b`/`salt` through three-argument ``getattr``,
   which raises on a pyconfig ``HyperParameters``).

Not this round: the CFG rollout step and its gradient contract (T3a), the trainer loop (T3b), any
config/YAML, any launcher.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion import pos_rollout_losses as losses
from maxdiffusion import pos_rollout_support as support
from maxdiffusion.models.wan.overfit100_sampling import overfit100_sampler_grid
from maxdiffusion.models.wan.side_adapter_wan import apply_first_frame_pin

_MODULE_PATH = Path(losses.__file__).resolve()
_PACKAGE_ROOT = _MODULE_PATH.parent
_TRAINER_PATH = _PACKAGE_ROOT / "trainers" / "wan_ti2v_side_adapter_trainer.py"
_THIS_PATH = Path(__file__).resolve()

# The production rollout grid (25 steps, flow shift 5.0, sigma in [0, 1], 1000 train timesteps) --
# the support's legal range and the terminal-sigma exclusion are facts about THIS grid (plan §3b).
_STEPS, _SHIFT, _SIGMA_MIN, _SIGMA_MAX, _NUM_TRAIN_TIMESTEPS = 25, 5.0, 0.0, 1.0, 1000
_B, _C, _F, _H, _W = 2, 4, 3, 4, 6
_K = 2  # the predeclared primary horizon (plan §3)


def _grid():
    return overfit100_sampler_grid(
        num_inference_steps=_STEPS,
        flow_shift=_SHIFT,
        sigma_min=_SIGMA_MIN,
        sigma_max=_SIGMA_MAX,
        num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
    )


def _batch(seed=0):
    key = jax.random.key(seed)
    k_v, k_i0, k_eps, k_ctx = jax.random.split(key, 4)
    z_video = jax.random.normal(k_v, (_B, _C, _F, _H, _W), dtype=jnp.float32)
    z_i0 = jax.random.normal(k_i0, (_B, _C, 1, _H, _W), dtype=jnp.float32)
    eps = jax.random.normal(k_eps, (_B, _C, _F, _H, _W), dtype=jnp.float32)
    context = jax.random.normal(k_ctx, (_B, 7, 8), dtype=jnp.float32)
    return z_video, z_i0, eps, context


def _run(velocity_fn, *, global_step=0, k_b=_K, batch=None, weights_dtype=jnp.float32):
    z_video, z_i0, eps, context = batch if batch is not None else _batch()
    sigmas, timesteps = _grid()
    return losses.rollout_endpoint_loss(
        z_video_f32=z_video,
        z_i0_f32=z_i0,
        eps_f32=eps,
        sigmas=sigmas,
        timesteps=timesteps,
        context=context,
        velocity_fn=velocity_fn,
        weights_dtype=weights_dtype,
        num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
        seed=0,
        global_step=global_step,
        num_steps=_STEPS,
        k_b=k_b,
    )


def _optimal_velocity(z_video, eps, extra=None):
    """``v* = eps - z_gt`` (the flow-matching optimum), optionally plus a fixed offset tensor."""

    def velocity_fn(hidden_states, timestep, encoder_hidden_states):
        del timestep, encoder_hidden_states
        v = (eps - z_video).astype(jnp.float32)
        if extra is not None:
            v = v + extra
        return v.astype(hidden_states.dtype)

    return velocity_fn


def _function_node(source: str, name: str, *, within: str | None = None) -> ast.FunctionDef:
    scope = ast.parse(source)
    if within is not None:
        scope = _function_node(source, within)
    for node in ast.walk(scope):
        if isinstance(node, ast.FunctionDef) and node.name == name and node is not scope:
            return node
    raise AssertionError(f"{name} not found{'' if within is None else f' inside {within}'}")


# =============================================================================================
# The two verbatim references for the DOUBLE equivalence obligation.
# =============================================================================================


def _pinned_build_noisy_pinned_latents(z_video_f32, z_i0_f32, eps, sigma_t):
    """VERBATIM from exp_03 @ 2ef9b8a, ``side_adapter_wan.py:555-558``. Do not refactor."""
    b = z_video_f32.shape[0]
    sigma_b = sigma_t.astype(jnp.float32).reshape((b, 1, 1, 1, 1))
    z_t_f32 = (1.0 - sigma_b) * z_video_f32.astype(jnp.float32) + sigma_b * eps.astype(jnp.float32)
    return apply_first_frame_pin(z_t_f32, z_i0_f32.astype(jnp.float32))


def _pinned_masked_velocity_mse(v_pred, v_target, batch_size):
    """VERBATIM from exp_03 @ 2ef9b8a, ``side_adapter_wan.py:574-580``. Do not refactor."""
    if v_pred.shape != v_target.shape:
        raise ValueError(f"masked_velocity_mse: v_pred shape {v_pred.shape} != v_target shape {v_target.shape}")
    mask = jnp.ones((1, *v_target.shape[1:]), dtype=jnp.float32)
    mask = mask.at[:, :, :1, :, :].set(0.0)
    diff = (v_pred.astype(jnp.float32) - v_target.astype(jnp.float32)) * mask
    n_valid = jnp.maximum(jnp.sum(mask) * batch_size, 1.0)
    return jnp.sum(diff**2) / n_valid


def _inline_noisy_latents(z_video_f32, z_i0_f32, eps, sigma_t, b):
    """VERBATIM from exp_06's own ``_denoising_loss``, trainer lines 162-164. Do not refactor."""
    sigma_b = sigma_t.reshape((b, 1, 1, 1, 1))
    z_t_f32 = (1.0 - sigma_b) * z_video_f32 + sigma_b * eps
    z_t_f32 = apply_first_frame_pin(z_t_f32, z_i0_f32)
    return z_t_f32


def _inline_masked_mse(v_pred, v_target, z_video_f32, f_lat, h_lat, w_lat, b):
    """VERBATIM from exp_06's own ``_denoising_loss``, trainer lines 192-196. Do not refactor."""
    mask = jnp.ones((1, z_video_f32.shape[1], f_lat, h_lat, w_lat), dtype=jnp.float32)
    mask = mask.at[:, :, :1, :, :].set(0.0)
    diff = (v_pred.astype(jnp.float32) - v_target.astype(jnp.float32)) * mask
    n_valid = jnp.maximum(jnp.sum(mask) * b, 1.0)
    loss = jnp.sum(diff**2) / n_valid
    return loss


# =============================================================================================
# 1. Provenance.
# =============================================================================================


def test_the_extracted_kernel_records_where_every_piece_came_from():
    assert set(losses.EXP03_T2_EXTRACTED_SYMBOLS) == {
        "rollout_endpoint_loss",
        "interpolant_at",
        "_endpoint_aux",
        "build_noisy_pinned_latents",
        "masked_velocity_mse",
    }
    for name, record in losses.EXP03_T2_EXTRACTED_SYMBOLS.items():
        assert hasattr(losses, name), f"{name} is declared extracted but is not defined"
        assert record["source_path"] in {
            "src/maxdiffusion/trainers/wan_ti2v_exp03_trainer.py",
            "src/maxdiffusion/models/wan/side_adapter_wan.py",
        }
        start, end = record["source_lines"]
        assert isinstance(start, int) and isinstance(end, int) and 0 < start <= end
        assert len(record["sha256"]) == 64 and set(record["sha256"]) <= set("0123456789abcdef")
    # The kernel is exp_03's `_rollout_loss` at the range T1 located, under an exp_06 name.
    kernel = losses.EXP03_T2_EXTRACTED_SYMBOLS["rollout_endpoint_loss"]
    assert kernel["source_symbol"] == "_rollout_loss"
    assert kernel["source_lines"] == (435, 510)


def test_the_kernel_module_is_pinned_to_the_same_exp03_commit_as_t1():
    # One pin for the whole experiment: T2 must not silently import from a different exp_03 SHA.
    assert losses.EXP03_SOURCE_COMMIT == support.EXP03_SOURCE_COMMIT
    assert losses.EXP03_SOURCE_COMMIT.startswith("2ef9b8a")


def test_the_per_example_helper_was_not_rehomed():
    # "Take the minimum": the kernel does not need `masked_velocity_mse_per_example`, so re-homing it
    # would be unpinned dead code carrying a second equivalence obligation nobody discharged.
    assert not hasattr(losses, "masked_velocity_mse_per_example")
    assert "masked_velocity_mse_per_example" not in losses.EXP03_T2_EXTRACTED_SYMBOLS


def test_no_inherited_module_was_edited_to_host_the_rehomed_helpers():
    # The Planner's ruling: the helpers live in the exp_06-owned module, and `side_adapter_wan.py`
    # keeps NOT having them. If someone later adds them there, we would have two definitions and no
    # guarantee they stay equal -- exactly the drift T1's zero-inherited-files property prevents.
    side_adapter = (_PACKAGE_ROOT / "models" / "wan" / "side_adapter_wan.py").read_text(encoding="utf-8")
    assert "def masked_velocity_mse" not in side_adapter
    assert "def build_noisy_pinned_latents" not in side_adapter


# =============================================================================================
# 2. THE DOUBLE EQUIVALENCE OBLIGATION.
# =============================================================================================


def test_the_rehomed_masked_mse_equals_both_the_pin_and_exp06s_inline_math_bitwise():
    """(a) exp_03's pinned construction and (b) exp_06's inline trainer math, same fixtures.

    If (a) and (b) disagreed, the two branches' losses would never have been the same function and
    this round would have to stop. They agree, so the re-homed helper is faithful to both at once.
    """
    z_video, z_i0, eps, _ = _batch(seed=3)
    for scale in (0.0, 1.0, 7.5):
        v_pred = (eps * scale - z_video).astype(jnp.float32)
        v_target = (eps - z_video).astype(jnp.float32)
        got = losses.masked_velocity_mse(v_pred, v_target, _B)
        pinned = _pinned_masked_velocity_mse(v_pred, v_target, _B)
        inline = _inline_masked_mse(v_pred, v_target, z_video, _F, _H, _W, _B)
        assert np.array_equal(np.asarray(got), np.asarray(pinned)), f"differs from the PIN at {scale}"
        assert np.array_equal(np.asarray(got), np.asarray(inline)), f"differs from exp_06 INLINE at {scale}"
        # ...and the two sources agree with each other, which is the claim that lets both stand.
        assert np.array_equal(np.asarray(pinned), np.asarray(inline))


def test_the_rehomed_noisy_latents_equal_both_the_pin_and_exp06s_inline_math_bitwise():
    z_video, z_i0, eps, _ = _batch(seed=4)
    sigmas, _ = _grid()
    for index in (0, 7, 24):
        sigma_t = jnp.full((_B,), sigmas[index].astype(jnp.float32))
        got = losses.build_noisy_pinned_latents(z_video, z_i0, eps, sigma_t)
        pinned = _pinned_build_noisy_pinned_latents(z_video, z_i0, eps, sigma_t)
        inline = _inline_noisy_latents(z_video, z_i0, eps, sigma_t, _B)
        assert np.array_equal(np.asarray(got), np.asarray(pinned)), f"differs from the PIN at {index}"
        assert np.array_equal(np.asarray(got), np.asarray(inline)), f"differs from exp_06 INLINE at {index}"
        assert np.array_equal(np.asarray(pinned), np.asarray(inline))


def test_exp06s_inline_trainer_math_has_not_drifted_from_the_copies_pinned_here():
    # The equivalence above is only evidence while copy (b) still matches the trainer it came from.
    # Structural, not string-matching: the trainer's statements are unparsed and the copies' bodies
    # must appear as contiguous subsequences.
    trainer_body = _function_node(_TRAINER_PATH.read_text(encoding="utf-8"), "_denoising_loss").body
    trainer_statements = [ast.unparse(node) for node in trainer_body]
    this_source = _THIS_PATH.read_text(encoding="utf-8")
    for reference in ("_inline_noisy_latents", "_inline_masked_mse"):
        wanted = [
            ast.unparse(node)
            for node in _function_node(this_source, reference).body
            if not isinstance(node, (ast.Expr, ast.Return))
            or not isinstance(getattr(node, "value", None), ast.Constant)
        ]
        wanted = [statement for statement in wanted if not statement.startswith("return ")]
        assert wanted, f"{reference} has no statements to compare"
        found = any(
            trainer_statements[i : i + len(wanted)] == wanted for i in range(len(trainer_statements) - len(wanted) + 1)
        )
        assert found, f"{reference} no longer matches a contiguous run of _denoising_loss statements"


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
    "low_precision, must_agree",
    [
        # Latents in bf16 with a float32 sigma: JAX PROMOTES the bf16 operands before multiplying,
        # which is exactly what the pinned `.astype(jnp.float32)` does explicitly -- so the two
        # constructions still agree bitwise. The dtype that actually decides is the sigma's.
        (("z_video", "z_i0", "eps"), True),
        (("sigma",), True),  # a bf16 sigma alone is re-promoted by the float32 latents
        (("z_video", "sigma"), False),  # ...but with a bf16 latent, `1.0 - sigma_b` rounds in bf16
        (("eps", "sigma"), False),
        (("z_video", "z_i0", "eps", "sigma"), False),  # everything low: the outputs differ in dtype too
    ],
)
def test_where_the_two_equivalence_sources_actually_part_company(low_precision, must_agree):
    """Why the precondition below is a guard and not a comment -- and exactly where the edge is.

    exp_03's construction casts to float32 internally; exp_06's inline trainer math does not. That
    difference is invisible while the SIGMA is float32, because JAX promotes any bf16 latent before
    the arithmetic. It becomes visible as soon as a bf16 sigma meets a bf16 latent: the inline
    version evaluates ``1.0 - sigma_b`` in bf16 and the pinned one in float32. So the equivalence
    this module rests on has a real boundary, not a hypothetical one, and the guard enforces
    float32 on all four inputs rather than only on the one that happens to trip it today.
    """
    z_video, z_i0, eps, _ = _batch(seed=11)
    sigmas, _ = _grid()
    arguments = {
        "z_video": z_video,
        "z_i0": z_i0,
        "eps": eps,
        "sigma": jnp.full((_B,), sigmas[7].astype(jnp.float32)),
    }
    for name in low_precision:
        arguments[name] = arguments[name].astype(jnp.bfloat16)
    ordered = (arguments["z_video"], arguments["z_i0"], arguments["eps"], arguments["sigma"])
    pinned = np.asarray(_pinned_build_noisy_pinned_latents(*ordered), dtype=np.float32)
    inline = np.asarray(_inline_noisy_latents(*ordered, _B), dtype=np.float32)
    if must_agree:
        assert np.array_equal(pinned, inline), f"{low_precision}: expected bitwise agreement"
    else:
        assert not np.array_equal(pinned, inline), f"{low_precision}: expected a measurable divergence"
        assert float(np.max(np.abs(pinned - inline))) > 1e-5


@pytest.mark.parametrize("offender", ["z_video_f32", "z_i0_f32", "eps", "sigma_t"])
def test_the_noisy_latent_construction_refuses_non_float32_inputs(offender):
    z_video, z_i0, eps, _ = _batch(seed=12)
    sigmas, _ = _grid()
    arguments = {
        "z_video_f32": z_video,
        "z_i0_f32": z_i0,
        "eps": eps,
        "sigma_t": jnp.full((_B,), sigmas[7].astype(jnp.float32)),
    }
    arguments[offender] = arguments[offender].astype(jnp.bfloat16)
    with pytest.raises(ValueError, match="must be float32"):
        losses.build_noisy_pinned_latents(**arguments)


@pytest.mark.parametrize("offender", ["z_video_f32", "z_i0_f32", "eps_f32", "sigmas"])
def test_the_kernel_refuses_non_float32_latents_noise_or_sigmas(offender):
    # The entry point checks too, so a new caller (T3b) is told at ITS call site rather than three
    # frames deep. `weights_dtype` is deliberately NOT constrained -- see the next test.
    z_video, z_i0, eps, context = _batch(seed=13)
    sigmas, timesteps = _grid()
    arguments = {
        "z_video_f32": z_video,
        "z_i0_f32": z_i0,
        "eps_f32": eps,
        "sigmas": sigmas,
        "timesteps": timesteps,
        "context": context,
        "velocity_fn": _optimal_velocity(z_video, eps),
        "weights_dtype": jnp.float32,
        "num_train_timesteps": _NUM_TRAIN_TIMESTEPS,
        "seed": 0,
        "global_step": 3,
        "num_steps": _STEPS,
        "k_b": _K,
    }
    arguments[offender] = arguments[offender].astype(jnp.bfloat16)
    with pytest.raises(ValueError, match="must be float32"):
        losses.rollout_endpoint_loss(**arguments)


def test_the_rollout_state_dtype_stays_free_because_it_is_not_a_loss_input():
    # weights_dtype is what the rollout STATE is carried in -- bf16 in production, matching the
    # deployed evaluator. Constraining it would break the production path; it is a different thing
    # from the float32 precondition on the latents.
    batch = _batch(seed=14)
    z_video, _, eps, _ = batch
    loss, aux = _run(_optimal_velocity(z_video, eps), batch=batch, weights_dtype=jnp.bfloat16)
    assert np.isfinite(float(loss))
    assert float(aux["z_end_finite"]) == 1.0


def test_an_explicit_none_global_step_is_rejected_with_a_clear_message():
    # Restored from the pin (exp_03 raised the same way): None would otherwise fail deep inside the
    # key derivation, where the message says nothing about the missing plumbing.
    batch = _batch(seed=15)
    z_video, _, eps, _ = batch
    with pytest.raises(ValueError, match="global_step"):
        _run(_optimal_velocity(z_video, eps), global_step=None, batch=batch)


def test_the_deliberate_divergences_from_the_pinned_bodies_are_recorded():
    # A re-pin diff must not have to guess which differences were intended. The arithmetic is
    # verbatim; these two are the reviewed exceptions.
    kernel = losses.EXP03_T2_EXTRACTED_SYMBOLS["rollout_endpoint_loss"]
    latents = losses.EXP03_T2_EXTRACTED_SYMBOLS["build_noisy_pinned_latents"]
    assert "config reads replaced by explicit arguments" in kernel["divergences"]
    assert "float32 precondition enforced" in kernel["divergences"]
    assert "float32 precondition enforced" in latents["divergences"]
    # ...and the ones with no divergence say so by carrying none.
    assert "divergences" not in losses.EXP03_T2_EXTRACTED_SYMBOLS["masked_velocity_mse"]


def test_the_masked_mse_rejects_a_broadcastable_but_malformed_prediction():
    # Inherited from the pin: a [1, ...] prediction would broadcast and be silently mis-normalized.
    z_video, _, eps, _ = _batch(seed=5)
    v_target = (eps - z_video).astype(jnp.float32)
    with pytest.raises(ValueError):
        losses.masked_velocity_mse(v_target[:1], v_target, _B)


# =============================================================================================
# 3. Analytic oracles -- each mistake individually falsifiable.
# =============================================================================================


@pytest.mark.parametrize("global_step", [0, 1, 17, 250])
@pytest.mark.parametrize("k_b", [1, 2, 4])
def test_the_loss_is_exactly_zero_at_the_optimum_for_every_support(global_step, k_b):
    # v* = eps - z_gt maps the interpolant at index i to the interpolant at i+1 exactly, so after k
    # steps the endpoint IS the ideal point. Zero at every support is what horizon-normalization
    # buys; a wrong sigma interval or a reversed Euler step destroys it.
    batch = _batch()
    z_video, _, eps, _ = batch
    loss, aux = _run(_optimal_velocity(z_video, eps), global_step=global_step, k_b=k_b, batch=batch)
    assert float(loss) < 1e-8, (global_step, k_b, float(loss))
    assert float(aux["raw_endpoint_mse"]) < 1e-10
    assert float(aux["loss_b_finite"]) == 1.0 and float(aux["z_end_finite"]) == 1.0


@pytest.mark.parametrize("global_step", [0, 1, 17, 250, 4321])
def test_a_constant_velocity_offset_costs_exactly_its_mean_square_whatever_the_support(global_step):
    """THE normalization oracle, and the reason the divisor is ``(sigma_hi - sigma_lo)**2``.

    With ``v = v* + c`` the frame-0 pin absorbs the error at frame 0 and the rest telescopes, so
    ``z_end - z_ideal = (sigma_lo - sigma_hi) * c`` exactly; the raw endpoint MSE is therefore
    ``(sigma_hi - sigma_lo)**2 * mean(c**2)`` and the normalized loss is ``mean(c**2)`` -- the SAME
    number at every support, though the raw MSE spans two orders of magnitude across the grid.
    Dropping the divisor leaves ``(sigma_hi-sigma_lo)**2 * c**2``; inverting it gives a fourth power.
    (Tolerance, not equality: the unroll and the closed form are different op sequences in float32.)
    """
    batch = _batch()
    z_video, _, eps, _ = batch
    c = 2.0
    offset = jnp.full_like(z_video, c)
    loss, aux = _run(_optimal_velocity(z_video, eps, extra=offset), global_step=global_step, batch=batch)
    assert np.allclose(float(loss), c**2, rtol=2e-3), (global_step, float(loss))
    # ...and the raw number really does move with the support, so the invariance above is the
    # normalizer's doing and not an accident of a flat grid.
    span = float(aux["sigma_hi_b"]) - float(aux["sigma_lo_b"])
    assert np.allclose(float(aux["raw_endpoint_mse"]), span**2 * c**2, rtol=2e-3)
    assert np.allclose(float(aux["horizon_sq"]), span**2, rtol=1e-6)
    assert np.allclose(float(loss), float(aux["raw_endpoint_mse"]) / float(aux["horizon_sq"]), rtol=1e-6)


def test_the_normalizer_is_the_squared_span_of_the_drawn_interval_not_a_neighbouring_one():
    # An off-by-one in the interval (start+1, or end+1) would still normalize by "a" squared span.
    # This pins WHICH one: exactly sigmas[start] and sigmas[start + k] for the support T1 draws.
    sigmas, _ = _grid()
    for global_step in (0, 3, 99, 1234):
        start, end = support.rollout_support(seed=0, global_step=global_step, num_steps=_STEPS, k_b=_K)
        batch = _batch()
        z_video, _, eps, _ = batch
        _, aux = _run(_optimal_velocity(z_video, eps), global_step=global_step, batch=batch)
        assert int(aux["s_b"]) == int(start) and int(aux["e_b"]) == int(end)
        assert int(aux["e_b"]) - int(aux["s_b"]) == _K
        assert float(aux["sigma_hi_b"]) == float(sigmas[int(start)])
        assert float(aux["sigma_lo_b"]) == float(sigmas[int(end)])
        assert np.allclose(float(aux["horizon_sq"]), (float(sigmas[int(start)]) - float(sigmas[int(end)])) ** 2)


def test_an_error_confined_to_latent_frame_zero_costs_nothing():
    # Frame 0 is the image condition: pinned after every step and masked out of the loss. An error
    # injected there must be invisible -- twice over.
    batch = _batch()
    z_video, _, eps, _ = batch
    offset = jnp.zeros_like(z_video).at[:, :, :1, :, :].set(5.0)
    loss, _ = _run(_optimal_velocity(z_video, eps, extra=offset), batch=batch)
    assert float(loss) < 1e-8, float(loss)


def test_an_error_on_one_future_frame_costs_exactly_its_share_of_the_masked_mean():
    # The mask oracle. The mean runs over the F-1 NON-frame-0 frames, so an error on exactly one of
    # them costs c**2 / (F-1). An inverted mask scores frame 0 only and returns 0 here; an absent
    # mask would divide by F instead of F-1 (1/3 vs 1/2 at F=3).
    batch = _batch()
    z_video, _, eps, _ = batch
    c = 2.0
    offset = jnp.zeros_like(z_video).at[:, :, 1:2, :, :].set(c)
    loss, _ = _run(_optimal_velocity(z_video, eps, extra=offset), batch=batch)
    assert np.allclose(float(loss), c**2 / (_F - 1), rtol=2e-3), float(loss)


def test_the_reduction_averages_over_examples_rather_than_summing_them():
    # The per-example oracle. An error on example 0 only costs c**2 / B, because n_valid counts one
    # example's unmasked elements times the batch size. Summing instead would give c**2; a
    # per-example vector would not be a scalar at all.
    batch = _batch()
    z_video, _, eps, _ = batch
    c = 2.0
    offset = jnp.zeros_like(z_video).at[:1, :, 1:, :, :].set(c)
    loss, _ = _run(_optimal_velocity(z_video, eps, extra=offset), batch=batch)
    assert jnp.ndim(loss) == 0
    assert np.allclose(float(loss), c**2 * (_F - 1) / (_F - 1) / _B, rtol=2e-3), float(loss)
    # ...and doubling the affected examples doubles the loss, linearly, as a mean over examples must.
    both = jnp.zeros_like(z_video).at[:, :, 1:, :, :].set(c)
    loss_both, _ = _run(_optimal_velocity(z_video, eps, extra=both), batch=batch)
    assert np.allclose(float(loss_both), _B * float(loss), rtol=2e-3)


def test_the_horizon_floor_is_the_float32_tiny_and_never_binds_on_the_production_grid():
    # `jnp.maximum(span**2, tiny)` guards a degenerate grid; on the real grid every legal support has
    # a span many orders above it, so the floor must never be what a run's loss is divided by.
    sigmas, _ = _grid()
    tiny = float(jnp.finfo(jnp.float32).tiny)
    assert "jnp.finfo(jnp.float32).tiny" in _MODULE_PATH.read_text(encoding="utf-8")
    for start in range(_STEPS - _K):
        span = float(sigmas[start]) - float(sigmas[start + _K])
        assert span > 0.0
        assert span**2 > tiny * 1e10


def test_the_kernel_reports_the_low_end_sigma_and_its_timestep():
    batch = _batch()
    z_video, _, eps, _ = batch
    _, aux = _run(_optimal_velocity(z_video, eps), global_step=11, batch=batch)
    assert np.allclose(float(aux["sigma_mean"]), float(aux["sigma_lo_b"]))
    assert np.allclose(float(aux["timestep_mean"]), float(aux["sigma_lo_b"]) * _NUM_TRAIN_TIMESTEPS, rtol=1e-6)
    for key in ("velocity_mse", "z_target_std", "z_init_anchor_mse", "v_pred_l2", "v_target_l2", "z_noisy_std"):
        assert key in aux, key


def test_the_endpoint_is_anchored_to_the_first_frame_condition():
    # Every step re-pins frame 0, so the rollout endpoint carries the condition exactly.
    batch = _batch()
    z_video, _, eps, _ = batch
    _, aux = _run(_optimal_velocity(z_video, eps, extra=jnp.full_like(z_video, 3.0)), batch=batch)
    assert float(aux["z_init_anchor_mse"]) < 1e-12


# =============================================================================================
# 4. Contract pins: T3a's precondition, and the boundaries this round must not cross.
# =============================================================================================


def _state_coupled_loss(k_b, *, global_step=7):
    """A loss whose velocity DEPENDS ON THE ROLLOUT STATE, so the unroll really couples its steps.

    A state-independent velocity would make each step's contribution additive and independent, and a
    truncated-graph bug would be invisible. Here ``v = v* + scale * z``, so the gradient with respect
    to ``scale`` can only be right if it flows back through every step's state.
    """
    z_video, z_i0, eps, context = _batch()
    sigmas, timesteps = _grid()

    def loss_of(scale):
        def velocity_fn(hidden_states, timestep, encoder_hidden_states):
            del timestep, encoder_hidden_states
            return (eps - z_video).astype(hidden_states.dtype) + scale.astype(hidden_states.dtype) * hidden_states

        value, _ = losses.rollout_endpoint_loss(
            z_video_f32=z_video,
            z_i0_f32=z_i0,
            eps_f32=eps,
            sigmas=sigmas,
            timesteps=timesteps,
            context=context,
            velocity_fn=velocity_fn,
            weights_dtype=jnp.float32,
            num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
            seed=0,
            global_step=global_step,
            num_steps=_STEPS,
            k_b=k_b,
        )
        return value

    return loss_of


def test_the_kernel_is_jittable_differentiable_and_rematable():
    # T3a unrolls this under jax.grad with remat; without all three the §3a contract has no substrate.
    loss_of = _state_coupled_loss(_K)
    scale = jnp.asarray(0.3, dtype=jnp.float32)
    eager = float(loss_of(scale))
    assert np.isfinite(eager)
    assert np.allclose(float(jax.jit(loss_of)(scale)), eager, rtol=1e-5)
    grad = float(jax.grad(loss_of)(scale))
    assert np.isfinite(grad) and grad != 0.0, grad
    assert np.allclose(float(jax.grad(jax.remat(loss_of))(scale)), grad, rtol=1e-3)


@pytest.mark.parametrize("k_b", [1, 2, 4])
def test_the_unrolled_gradient_matches_a_central_finite_difference(k_b):
    """The actual proof that the gradient traverses the whole unroll (the R11 lesson, T2's share).

    A ``stop_gradient`` on the scan carry, or a graph truncated to the final forward, changes the
    analytic derivative but not the function's values -- so only a value-based oracle catches it.
    Central differences, on a state-coupled velocity, at k = 1, 2 and 4.
    """
    loss_of = _state_coupled_loss(k_b)
    scale = jnp.asarray(0.3, dtype=jnp.float32)
    step = 1e-3
    analytic = float(jax.grad(loss_of)(scale))
    numeric = (float(loss_of(scale + step)) - float(loss_of(scale - step))) / (2.0 * step)
    assert np.isfinite(analytic) and analytic != 0.0
    assert np.allclose(analytic, numeric, rtol=2e-2), (k_b, analytic, numeric)


def test_the_kernel_applies_no_stop_gradient_of_its_own():
    # The §3a contract (which CFG branch sees the rollout state's gradient) is T3a's to write. A
    # stop_gradient here would silently pre-empt it -- exactly the one-step trainer's pattern that
    # plan §3a says must NOT be copied.
    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    offenders = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "stop_gradient"
        or isinstance(node, ast.Name)
        and node.id == "stop_gradient"
    ]
    assert not offenders, "the T2 kernel must not stop_gradient; that contract belongs to T3a"
    assert "lax.stop_gradient" not in source


def test_the_kernel_module_performs_no_config_access_at_all():
    """Issue #11 as a structural gate.

    exp_03's `_rollout_loss` read seed/k/salt via three-argument ``getattr(config, ...)``; on a
    pyconfig ``HyperParameters`` that raises instead of falling back, which killed two TPU jobs last
    campaign. The extracted kernel takes them as explicit arguments, and nothing in this module may
    touch a config object or a scheduler by attribute.
    """
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    banned_bases = {"config", "cfg", "scheduler"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert node.value.id not in banned_bases, f"config-style attribute access: {ast.unparse(node)}"
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "getattr":
            assert len(node.args) < 3, f"three-argument getattr is forbidden: {ast.unparse(node)}"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
            assert not names & banned_bases, f"{node.name} takes a config-style argument"
    parameters = inspect.signature(losses.rollout_endpoint_loss).parameters
    assert not set(parameters) & banned_bases
    # ...and everything exp_03 read from config is a first-class, keyword-only argument here.
    for required in ("seed", "global_step", "num_steps", "k_b", "support_salt", "num_train_timesteps"):
        assert parameters[required].kind is inspect.Parameter.KEYWORD_ONLY, required


def test_the_kernel_takes_the_velocity_function_from_its_caller():
    # T3a supplies the CFG branch; T2 must not bake a forward in. No transformer, no adapters, no
    # nnx merge anywhere in the module.
    parameters = inspect.signature(losses.rollout_endpoint_loss).parameters
    assert parameters["velocity_fn"].kind is inspect.Parameter.KEYWORD_ONLY
    source = _MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in ("nnx.merge", "guide_scale", "wan_action_adapter_forward", "transformer("):
        assert forbidden not in source, forbidden


def test_the_kernel_module_is_side_effect_free():
    source = _MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in ("print(", "max_logging", "jax.debug", "open(", "np.save", "logging."):
        assert forbidden not in source, forbidden


def test_the_support_draw_is_the_t1_primitive_not_a_second_copy():
    """One support construction for the whole experiment, checked where a copy would have to show.

    A private re-implementation is numerically indistinguishable from T1's while it agrees, so a
    value test cannot see it -- and it is exactly what §3b's legal-range and per-batch guarantees
    would silently stop covering. Structural locks only, and none of them defeated by renaming the
    copy or by REBINDING the imported name to it: the object the module will actually call must BE
    T1's function; the name may not be re-bound anywhere; the call site must spell it; no second
    support-shaped function may exist; and the module may draw NO randomness of its own.
    """
    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = [
        (alias.asname or alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("pos_rollout_support")
        for alias in node.names
    ]
    assert "rollout_support" in imported
    assert imported.count("rollout_support") == 1, "bound by more than one import"

    # (0) THE lock no textual trick survives: at run time the module's `rollout_support` IS T1's
    # function object. A later `rollout_support = _draw_window` passes every syntactic check ever
    # written and fails right here.
    assert losses.rollout_support is support.rollout_support

    # (0b) ...and statically, the name is never re-bound: no assignment, no `for`/`with` target, no
    # parameter, no `del`, no second import. (Belt and braces: the identity check above is evaluated
    # at import time, so a rebinding executed lazily inside a function would still be caught here.)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "rollout_support":
            assert isinstance(node.ctx, ast.Load), f"rollout_support is re-bound: {ast.dump(node.ctx)}"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            names = {
                argument.arg
                for argument in arguments.args + arguments.kwonlyargs + arguments.posonlyargs
                if argument is not None
            }
            for extra in (arguments.vararg, arguments.kwarg):
                if extra is not None:
                    names.add(extra.arg)
            assert "rollout_support" not in names, f"{node.name} shadows rollout_support with a parameter"

    # (1) The kernel's support really is that call, not something that merely looks like it.
    kernel = _function_node(source, "rollout_endpoint_loss")

    def _flat_names(target):
        if isinstance(target, ast.Tuple) and all(isinstance(element, ast.Name) for element in target.elts):
            return [element.id for element in target.elts]
        return None

    draws = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Assign) and _flat_names(node.targets[0]) == ["start", "end"]
    ]
    assert len(draws) == 1, "the support is bound more than once -- a second, private draw"
    assert isinstance(draws[0].value, ast.Call) and isinstance(draws[0].value.func, ast.Name)
    assert draws[0].value.func.id == "rollout_support", ast.unparse(draws[0])

    # (2) No second support-shaped definition anywhere in the module, whatever it is called.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert "support" not in node.name, f"{node.name} looks like a private support draw"

    # (3) The module draws no randomness at all -- a copy of the draw would have to.
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {
            "randint",
            "uniform",
            "normal",
            "fold_in",
            "key",
            "split",
        }:
            raise AssertionError(f"the kernel module must not draw randomness: {ast.unparse(node)}")
        if isinstance(node, ast.Name):
            assert node.id != "exp03_aux_key", "the key derivation belongs to T1's module, not here"
    assert "jax.random" not in source

    # ...and the values agree with T1's primitive, which is what the structure is protecting.
    batch = _batch()
    z_video, _, eps, _ = batch
    for global_step in (0, 5, 77):
        start, end = support.rollout_support(seed=0, global_step=global_step, num_steps=_STEPS, k_b=_K)
        _, aux = _run(_optimal_velocity(z_video, eps), global_step=global_step, batch=batch)
        assert (int(aux["s_b"]), int(aux["e_b"])) == (int(start), int(end))
