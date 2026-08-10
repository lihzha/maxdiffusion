"""exp_06 `rollout_adapter` — T3b-2: the two arms (plan §3, §3b; Yixun decision 1).

**One contract: both objectives are the same function on the same stream, and the arm selects only
the loss.** exp_06's claim is causal — that R-B beats matched-C0 *because of the objective*. That
claim survives exactly as long as nothing else differs between the arms, so "the arm is a parameter,
not a fork" is not a style preference here; it is the experimental control, and it is enforced and
tested as one.

**R-B (`"rollout"`)** composes committed pieces and re-implements nothing: T2's
:func:`~maxdiffusion.pos_rollout_losses.rollout_endpoint_loss` over T3a's
:func:`~maxdiffusion.pos_rollout_step.build_cfg_velocity_fn`. Its CFG forward carries no
``stop_gradient`` (plan §3a).

**matched-C0 (`"one_step"`)** is the DEPLOYED one-step denoising objective, and it carries the
double-equivalence obligation T2 established: it is proven bitwise-equal — in VALUE and in GRADIENT —
to the settled inline ``_denoising_loss`` of ``trainers/wan_ti2v_side_adapter_trainer.py`` on shared
fixtures, with an AST drift tripwire binding the verbatim copy to its source. Comparing the gradient
as well as the value is what catches a divergence no value can show (the S16 lesson) — a control arm
whose gradients were quietly wrong would make R-B look better, the most dangerous failure available
here.

**But note what the gradient comparison canNOT do, because it was tempting to claim otherwise.**
C0's stop-grads — on the latent entering the unconditional branch and on ``v_uncond`` itself — are
*inert* in this objective: ``z_t`` is built from data and noise alone, so it and ``v_uncond`` are
parameter-independent, and cutting a gradient that is already zero changes nothing. That is precisely
why they are free (and correct) here and truncating in a rollout. So removing them changes neither
value nor gradient, and no numerical oracle can see it; they are pinned STRUCTURALLY, and the
inertness itself is proven rather than asserted (``test_the_one_step_stop_gradients_are_inert``).

A matched control that is not the deployed objective would invalidate the comparison Yixun's
decision 1 exists to support, so a disagreement here is a STOP-and-report condition, not something to
reconcile in code.

**Dropout.** Production dropout is zero, so matched-C0's ``dropout_rng`` is inert — proven, not
assumed (:data:`PRODUCTION_DROPOUT_RATE` and the tests). R-B discards the key because it has nothing
to consume; C0 passes it on because the deployed forward takes it. Neither arm is therefore
randomised by it, which is what makes "both arms draw from one stream" true rather than approximate.

**The stream supplies BOTH arms' randomness.** matched-C0's per-example sigma index is a stream
draw (``StepDraws.t_idx``), not something the loss samples: left to the arm it would be
accumulation- and resume-dependent and would confound the very comparison it exists to anchor
(T3b-1 review, BLOCKER 1). Each arm reads the fields it needs from one shared ``StepDraws``; that
both are HANDED the same object is the pairing, not that both consume every field.

**What the arm may NOT change**, pinned by :func:`build_arm` returning the adapter parameters from
one split of one module and by the tests: the stream draws, the batch consumed, the parameter tree,
and accumulation behaviour. Every random quantity either arm consumes -- the support window, the
noise and the one-step timestep index -- arrives as an ARGUMENT on ``StepDraws``; neither arm draws
anything, because the purposes that generate them belong to the stream layer.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from maxdiffusion.models.wan.side_adapter_wan import (
    _build_per_token_timestep,
    wan_action_adapter_forward,
)
from maxdiffusion.pos_rollout_losses import build_noisy_pinned_latents, masked_velocity_mse, rollout_endpoint_loss
from maxdiffusion.pos_rollout_step import CFG_IDENTITY_TOLERANCE, build_cfg_velocity_fn
from maxdiffusion.pos_rollout_stream import StepDraws

__all__ = [
    "ARMS",
    "ArmContext",
    "build_arm",
    "build_one_step_velocity_fn",
    "one_step_denoising_loss",
    "rollout_arm_loss",
]

ARMS = ("rollout", "one_step")

#: Production dropout for this architecture is ZERO (``configs/base_wan_5b_side_adapter.yml``:
#: ``dropout: 0.0``, and every transformer/adapter default is 0.0). matched-C0 keeps the deployed
#: ``deterministic=False`` call so its forward is byte-for-byte the settled one, and accepts a
#: ``dropout_rng`` for the same reason — but at this rate the key is INERT: a test proves that
#: changing it moves neither the value nor the gradient. That is a stronger and cheaper claim than
#: plumbing a stream draw nobody consumes, and it is what closes "the arms differ in randomness"
#: (T3b-2 review, BLOCKER 2). If the rate ever becomes nonzero this stops being true, so the config
#: value is pinned by a tripwire and the dropout key must then become a stream draw like the others.
PRODUCTION_DROPOUT_RATE = 0.0


@dataclasses.dataclass(frozen=True)
class ArmContext:
    """Everything both arms share, held in one object so neither can be handed a different value.

    If the guide scale, the sigma grid or the batch geometry could differ between arms, "matched"
    would be a claim about intent rather than about code. Passing one context to both is what makes
    it a fact.
    """

    sigmas: jax.Array
    #: The T5("") embedding in its DEPLOYED shape ``[1, L, D]`` -- one null context broadcast to the
    #: batch, exactly as the settled trainer and evaluator hold it. Storing it per-batch instead
    #: would silently break under accumulation, where a microbatch is narrower than the batch.
    null_context: jax.Array
    timesteps: jax.Array
    guide_scale: float
    weights_dtype: Any
    num_train_timesteps: int
    num_steps: int
    k_b: int

    def broadcast_null_context(self, batch_size: int) -> jax.Array:
        """``[b, L, D]`` in the caller's weights dtype — the settled trainer's own construction."""
        return jnp.broadcast_to(
            self.null_context.astype(self.weights_dtype),
            (batch_size, self.null_context.shape[1], self.null_context.shape[2]),
        )


def build_one_step_velocity_fn(transformer, adapters):
    """``(make_velocity_fn, adapter_params, frozen)`` for the DEPLOYED one-step CFG forward.

    Same freeze seam as T3a — the frozen backbone is keyword-only and therefore undifferentiable,
    and only its array-free graph definition is captured (F3; see
    :class:`~maxdiffusion.pos_rollout_step.FrozenBackbone`) — but the CFG branch is the one-step
    trainer's, stop-grads included. Those stop-grads are CORRECT here — a one-step objective's noisy
    latent is parameter-independent, so cutting the gradient through the unconditional branch costs
    nothing and matches ``../Wan2.2`` — and they are exactly what T3a's rollout forward must not
    copy. Keeping the two forwards in separate builders is how the experiment holds both truths at
    once.
    """
    from flax import nnx

    from maxdiffusion.pos_rollout_step import split_frozen_backbone

    adapter_graphdef, adapter_params, adapter_rest = nnx.split(adapters, nnx.Param, ...)
    frozen = split_frozen_backbone(transformer)
    frozen_graphdef = frozen.graphdef

    def make_velocity_fn(params, *, frozen_state, actions, guide_scale, deterministic: bool = False, dropout_rng=None):
        # Same enforcement as the rollout builder, same reason (F3b, review MAJOR-1): keyword-only
        # placement is syntax; `stop_gradient` is autodiff. Identity in the forward pass, so
        # matched-C0's bitwise equality with the settled `_denoising_loss` is unaffected.
        frozen_state = jax.tree.map(jax.lax.stop_gradient, frozen_state)
        model = nnx.merge(frozen_graphdef, frozen_state)
        adapter = nnx.merge(adapter_graphdef, params, adapter_rest)
        guided = abs(float(guide_scale) - 1.0) > CFG_IDENTITY_TOLERANCE
        rngs = None if dropout_rng is None else nnx.Rngs(dropout=dropout_rng)

        def velocity_fn(hidden_states, timestep, encoder_hidden_states):
            v_cond = wan_action_adapter_forward(
                model,
                adapter,
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                actions=actions,
                deterministic=deterministic,
                rngs=rngs,
            )
            if not guided:
                return v_cond
            v_uncond = model(
                hidden_states=jax.lax.stop_gradient(hidden_states),
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                deterministic=True,
            )
            v_uncond = jax.lax.stop_gradient(v_uncond)
            return v_uncond + guide_scale * (v_cond - v_uncond)

        return velocity_fn

    return make_velocity_fn, adapter_params, frozen


def one_step_denoising_loss(
    params,
    make_velocity_fn,
    batch: Mapping[str, jax.Array],
    context: ArmContext,
    *,
    frozen_state,
    draws: StepDraws,
    dropout_rng=None,
) -> tuple[jax.Array, dict]:
    """matched-C0 — the deployed one-step denoising objective, on the shared stream.

    ``z_t = (1 - sigma_t) * z_video + sigma_t * eps`` with frame 0 pinned, one CFG forward at that
    sigma, target ``eps - z_video``, frame-0-masked MSE. Built from T2's re-homed constructions, so
    the two pieces the settled trainer computes inline are the ones T2 already proved equal to it.

    Both the noise and the per-example sigma index come from the shared ``StepDraws``: the loss
    samples nothing, so it cannot make an arm's randomness differ from its partner's.
    """
    epsilon, t_idx = draws.epsilon, draws.t_idx
    z_video_f32 = batch["z_video"].astype(jnp.float32)
    z_i0_f32 = batch["z_i0"].astype(jnp.float32)
    actions = batch["actions"].astype(context.weights_dtype)
    b, _, f_lat, h_lat, w_lat = z_video_f32.shape

    sigma_t = context.sigmas[t_idx]
    step_t = sigma_t * jnp.asarray(context.num_train_timesteps, dtype=jnp.float32)
    timestep_2d = _build_per_token_timestep(step_t, f_lat, h_lat, w_lat, n_hist=1)
    null_context = context.broadcast_null_context(b)

    z_t_f32 = build_noisy_pinned_latents(z_video_f32, z_i0_f32, epsilon.astype(jnp.float32), sigma_t)
    z_t = z_t_f32.astype(context.weights_dtype)

    velocity_fn = make_velocity_fn(
        params,
        frozen_state=frozen_state,
        actions=actions,
        guide_scale=context.guide_scale,
        deterministic=False,
        dropout_rng=dropout_rng,
    )
    v_pred = velocity_fn(z_t, timestep_2d, null_context)

    v_target = epsilon.astype(jnp.float32) - z_video_f32
    loss = masked_velocity_mse(v_pred, v_target, b)
    aux = {
        "velocity_mse": loss,
        "sigma_mean": jnp.mean(sigma_t.astype(jnp.float32)),
        "timestep_mean": jnp.mean(step_t.astype(jnp.float32)),
        "v_pred_l2": jnp.linalg.norm(v_pred.astype(jnp.float32)),
        "v_target_l2": jnp.linalg.norm(v_target.astype(jnp.float32)),
        "z_noisy_std": jnp.std(z_t_f32),
        "z_target_std": jnp.std(z_video_f32),
        "z_init_anchor_mse": jnp.mean((z_t_f32[:, :, :1] - z_i0_f32[:, :, :1]) ** 2),
    }
    return loss, aux


def rollout_arm_loss(
    params,
    make_velocity_fn,
    batch: Mapping[str, jax.Array],
    context: ArmContext,
    *,
    frozen_state,
    draws: StepDraws,
) -> tuple[jax.Array, dict]:
    """R-B — T2's endpoint kernel over T3a's CFG velocity. Composition only; nothing re-derived.

    BOTH halves of the step's randomness come from the one ``StepDraws``: the epsilon and the
    support window. The kernel no longer derives a support of its own, so there is no way for R-B to
    score an endpoint whose noise and whose sigma interval came from different steps (T3b-2 review,
    BLOCKER 1).
    """
    epsilon = draws.epsilon
    z_video_f32 = batch["z_video"].astype(jnp.float32)
    return rollout_endpoint_loss(
        z_video_f32=z_video_f32,
        z_i0_f32=batch["z_i0"].astype(jnp.float32),
        eps_f32=epsilon.astype(jnp.float32),
        sigmas=context.sigmas,
        timesteps=context.timesteps,
        # Broadcast for the SAME reason C0 does: a microbatch is narrower than the batch, and a
        # per-batch null context would make accumulation change the objective.
        context=context.broadcast_null_context(z_video_f32.shape[0]),
        velocity_fn=make_velocity_fn(
            params,
            frozen_state=frozen_state,
            actions=batch["actions"].astype(context.weights_dtype),
            guide_scale=context.guide_scale,
        ),
        weights_dtype=context.weights_dtype,
        num_train_timesteps=context.num_train_timesteps,
        support_start=draws.support_start,
        support_end=draws.support_end,
        k_b=context.k_b,
    )


def assert_dropout_is_zero(*modules, dropout_rate=None) -> None:
    """FAIL CLOSED unless dropout really is zero — enforcement, not a demonstration.

    Proving the key inert AT zero is not the same as guaranteeing zero: command-line overrides beat
    YAML (``pyconfig.py``) and production passes ``config.dropout`` straight into ``WanModel``, so a
    launch with ``dropout=0.1`` would have reached training with the tests still green (T3b-2 review,
    BLOCKER A2). Both the declared rate AND the constructed modules are checked, because a claim
    about a config is not a fact about the object that was built.
    """
    if dropout_rate is not None and float(dropout_rate) != PRODUCTION_DROPOUT_RATE:
        raise ValueError(
            f"dropout must be {PRODUCTION_DROPOUT_RATE} for exp_06: got {dropout_rate}. matched-C0's dropout key "
            f"is inert only at zero; a nonzero rate makes the arms differ in randomness and must first become a "
            f"stream draw (plan §3b)."
        )
    from flax import nnx

    for module in modules:
        if module is None:
            continue
        for path, node in nnx.iter_graph(module):
            if not isinstance(node, nnx.Dropout):
                continue
            rate = node.rate
            if rate is not None and float(rate) != PRODUCTION_DROPOUT_RATE:
                raise ValueError(
                    f"dropout must be {PRODUCTION_DROPOUT_RATE} for exp_06: the constructed module has rate "
                    f"{rate} at {'.'.join(str(part) for part in path)}"
                )


def build_arm(arm: str, transformer, adapters, *, dropout_rate=PRODUCTION_DROPOUT_RATE):
    """``(loss_fn, adapter_params, frozen)`` for one arm — and the parameter tree is arm-independent.

    Both arms split THE SAME adapter module, so the trainable tree, its structure and its initial
    values are identical by construction rather than by matching two configurations. The only thing
    that differs downstream of this call is which loss runs.

    ``frozen`` is the third return value for the same reason the adapter parameters are the second:
    it is DATA the caller must thread, not something this module may keep. Both arms' ``loss_fn``
    take ``frozen_state`` keyword-only, so both are undifferentiable in the backbone and neither can
    be jitted into a module carrying the weights as literals (F3;
    :class:`~maxdiffusion.pos_rollout_step.FrozenBackbone`).
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; declared arms are {list(ARMS)}")
    # Refused HERE, before any training or model wiring downstream of this call.
    assert_dropout_is_zero(transformer, adapters, dropout_rate=dropout_rate)

    if arm == "rollout":
        make_velocity_fn, adapter_params, frozen = build_cfg_velocity_fn(transformer, adapters)

        def loss_fn(
            params, batch, context, *, frozen_state, draws, seed=None, global_step=None, dropout_rng=None, **extra
        ):
            # seed/global_step are accepted so both arms share one call signature, and DELETED here:
            # every random quantity R-B uses is already in `draws`.
            del seed, global_step, dropout_rng, extra
            return rollout_arm_loss(params, make_velocity_fn, batch, context, frozen_state=frozen_state, draws=draws)

    else:
        make_velocity_fn, adapter_params, frozen = build_one_step_velocity_fn(transformer, adapters)

        def loss_fn(
            params, batch, context, *, frozen_state, draws, seed=None, global_step=None, dropout_rng=None, **extra
        ):
            del seed, global_step, extra
            if draws.t_idx is None:
                raise ValueError("the one_step arm needs its sigma index t_idx from the stream layer")
            return one_step_denoising_loss(
                params,
                make_velocity_fn,
                batch,
                context,
                frozen_state=frozen_state,
                draws=draws,
                dropout_rng=dropout_rng,
            )

    return loss_fn, adapter_params, frozen
