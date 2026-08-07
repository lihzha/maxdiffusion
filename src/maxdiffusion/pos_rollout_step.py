"""exp_06 `rollout_adapter` — the differentiable CFG rollout step (plan §3a).

T2's endpoint kernel takes ``velocity_fn`` from its caller and deliberately knows nothing about how a
velocity is produced. This module supplies that function, and it is where the experiment's central
correctness claim lives: **the gradients the unrolled loss sees must be the ones deployment's math
implies.**

Deployment computes, at every rollout step,

    v = v_unc(frozen backbone, T5("")) + w * (v_cond(adapter) - v_unc),    w = 5

then takes the Euler step and re-pins latent frame 0. The same three things happen here, through the
same shared sampler step T1 imported, so the training trajectory and the evaluation trajectory are
the same operator rather than two copies that agree today.

**The gradient contract (§3a), clause by clause.**

* **(i) The gradient tree contains ADAPTER PARAMETERS ONLY, structurally.** The frozen backbone's
  split state is captured in :func:`build_cfg_velocity_fn`'s closure and never appears as an
  argument to anything differentiated — the S7 closure-seam pattern, reused deliberately. This is
  *not* a filter over a combined tree: there is no combined tree to filter, so a frozen leaf cannot
  receive a gradient even by misconfiguration.
* **(ii) BOTH branches' dependence on the current rollout state is differentiated.** ``hidden_states``
  (the rollout state ``z_i``) enters the conditional and the unconditional branch with no
  ``stop_gradient`` on either. **The site NOT copied is**
  ``trainers/wan_ti2v_side_adapter_trainer.py:180-187``, which stop-gradients both ``z_t`` going into
  the unconditional branch and ``v_uncond`` itself. That is correct for a one-step objective — there
  ``z_t`` is parameter-independent, so the stop-grad is free — and it is gradient-TRUNCATING for a
  rollout objective, where ``z_i`` carries the whole inter-step dependence on the adapter. This
  module contains no ``stop_gradient`` at all, and the two-step finite-difference oracle in
  ``tests/worklogs_yixun/test_pos_rollout_step.py`` is the actual proof: at k=2 the endpoint depends
  on the adapter through ``v_unc(z_1)`` as well as ``v_cond(z_1)``, and a stop-grad on either branch
  moves the measured derivative far outside tolerance.
* **(iii) The architecture's internal block-0 stop-grad REMAINS.** ``_first_block_self_attention_``
  ``features`` ends in ``jax.lax.stop_gradient`` and the pre_context head is conditioned on those
  detached features. The architecture is held fixed FOR ISOLATION of the objective variable (plan
  §2/§3), so that stop-grad is not ours to "fix"; a tripwire test pins that it is still there.

**Seams kept.** The loss is not folded in — this module returns a ``velocity_fn`` of exactly the
signature T1's sampler and T2's kernel already consume, so T3b composes the three without any of
them knowing about the others. The settled evaluator is NOT rewired (standing ruling, plan §5-5):
parity with it is proven by test, against a verbatim copy held honest by an AST drift tripwire.
"""

from __future__ import annotations

import jax

from maxdiffusion.models.wan.overfit100_sampling import overfit100_sampler_step
from maxdiffusion.models.wan.side_adapter_wan import wan_action_adapter_forward

__all__ = [
    "CFG_IDENTITY_TOLERANCE",
    "build_cfg_velocity_fn",
    "cfg_rollout",
    "combine_cfg",
]

# The deployed evaluator treats a guide scale within this of 1.0 as "no CFG" and skips the second
# forward entirely. Kept identical because parity is asserted bitwise against that code path.
CFG_IDENTITY_TOLERANCE = 1e-6


def combine_cfg(v_uncond: jax.Array, v_cond: jax.Array, guide_scale) -> jax.Array:
    """``v_unc + w * (v_cond - v_unc)`` — deployment's classifier-free-guidance combination.

    Written in exactly this form, not the algebraically equal ``w * v_cond + (1 - w) * v_unc``,
    because parity with the deployed evaluator is asserted BITWISE and the two forms differ in
    float32. The argument order matters as much as the sign: ``v_uncond`` is the base and the guided
    direction is ``v_cond - v_uncond``.
    """
    return v_uncond + guide_scale * (v_cond - v_uncond)


def build_cfg_velocity_fn(transformer, adapters):
    """``(make_velocity_fn, adapter_params)`` — the frozen split, made structural.

    ``transformer``'s split state is bound HERE, in this closure, and never becomes an argument of a
    differentiated function; only ``adapter_params`` is passed back for the caller to differentiate.
    That is the whole of clause (i): the backbone is not filtered out of a gradient tree, it is never
    in one.

    ``make_velocity_fn(params, *, actions, guide_scale, ...)`` returns a
    ``velocity_fn(hidden_states, timestep, encoder_hidden_states)`` — the signature T1's sampler step
    and T2's endpoint kernel already consume, so nothing downstream needs to know a CFG combination
    happened. Both branches read ``hidden_states`` directly (clause (ii)).
    """
    from flax import nnx

    adapter_graphdef, adapter_params, adapter_rest = nnx.split(adapters, nnx.Param, ...)
    frozen = nnx.split(transformer, nnx.Param, ...)

    def make_velocity_fn(params, *, actions, guide_scale, deterministic: bool = True, rngs=None):
        model = nnx.merge(*frozen)
        adapter = nnx.merge(adapter_graphdef, params, adapter_rest)
        guided = abs(float(guide_scale) - 1.0) > CFG_IDENTITY_TOLERANCE

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
            # NOTE: `hidden_states` goes in unmodified. The one-step trainer wraps this argument and
            # the result in `stop_gradient`; doing so here would delete the inter-step gradient path
            # this experiment exists to create (§3a(ii)).
            v_uncond = model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                deterministic=True,
            )
            return combine_cfg(v_uncond, v_cond, guide_scale)

        return velocity_fn

    return make_velocity_fn, adapter_params


def cfg_rollout(
    z: jax.Array,
    *,
    velocity_fn,
    sigmas: jax.Array,
    timesteps: jax.Array,
    context: jax.Array,
    z_i0: jax.Array,
    start=0,
    num_steps: int,
) -> jax.Array:
    """``num_steps`` sampler steps from grid index ``start`` — deployment's rollout, composable.

    Nothing here but the loop: every step is T1's shared :func:`overfit100_sampler_step`, so the
    Euler update and the frame-0 pin have exactly one definition in the experiment. Used at
    ``start=0, num_steps=25`` for the deployed-parity proof and at ``num_steps=2`` for the
    finite-difference oracle; T2's kernel runs the same step under ``lax.scan`` with ``remat``.
    """
    return jax.lax.fori_loop(
        start,
        start + int(num_steps),
        lambda index, current: overfit100_sampler_step(
            current,
            index,
            velocity_fn=velocity_fn,
            sigmas=sigmas,
            timesteps=timesteps,
            context=context,
            z_i0=z_i0,
        ),
        z,
    )
