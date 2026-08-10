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

* **(i) The gradient tree contains ADAPTER PARAMETERS ONLY, structurally.** The frozen backbone
  reaches every loss as a **keyword-only** ``frozen_state`` argument, and ``jax.value_and_grad``
  takes ``argnums`` over *positional* arguments only — so a frozen leaf cannot receive a gradient
  even by misconfiguration, because the call that would ask for one cannot be written. This is
  *not* a filter over a combined tree: there is no combined tree to filter.

  **Round F3 changed HOW this is achieved, and strengthened it.** Until F3 the backbone was captured
  in :func:`build_cfg_velocity_fn`'s closure (the S7 closure-seam pattern). That delivered clause (i)
  and simultaneously caused the production failure F3 exists to fix: ``jax.jit`` bakes captured
  arrays into the lowered module as literals, so the real 5B entered XLA as **10.18 GB of
  constants** and three M1 attempts died in compilation without reaching a single step. The arrays
  are now threaded as data (:class:`FrozenBackbone`) while only the array-free graph definition is
  captured — the differentiation guarantee is preserved *and* the module is small.
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

import dataclasses
from typing import Any

import jax

from maxdiffusion.models.wan.overfit100_sampling import overfit100_sampler_step
from maxdiffusion.models.wan.side_adapter_wan import wan_action_adapter_forward

__all__ = [
    "CFG_IDENTITY_TOLERANCE",
    "FrozenBackbone",
    "build_cfg_velocity_fn",
    "cfg_rollout",
    "combine_cfg",
    "split_frozen_backbone",
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


@dataclasses.dataclass(frozen=True)
class FrozenBackbone:
    """The frozen 5B, split into the part that may be CAPTURED and the part that must be PASSED.

    **Registered as a pytree, and that registration is a correctness property, not convenience**
    (F3b, review MAJOR-3a). As a plain dataclass this object was OPAQUE to ``jax.tree`` — so
    ``jax.tree.leaves(frozen)`` returned **zero leaves while holding every weight in the model**, and
    a detector that walked closures looking for arrays would have reported a clean 0 bytes with the
    whole backbone sitting inside. A guard that cannot see the thing it guards is worse than no
    guard, because it is believed. ``state`` is the single child; ``graphdef`` is static metadata,
    which is also what makes this object safe to pass through a jit boundary if anyone ever does.

    **This split is the whole of round F3, and it is a hardware lesson.** The M1 fit probe died three
    times on ``TPU_VM_HEALTH_TIMEOUT`` without ever finishing its first XLA compile, each attempt's
    log ending on ``A large amount of constants were captured during lowering (10.18GB total)``. The
    backbone was bound into the loss closure, so ``jax.jit`` promoted every bf16 weight to a LITERAL
    inside the lowered module and XLA was asked to serialize and optimize a ten-gigabyte program.

    So the two halves are separated by what they are, and named for what may be done with them:

    * :attr:`graphdef` is STRUCTURE -- module classes, attribute names, shapes. It holds no arrays,
      so capturing it in a closure costs nothing and it is what makes the tracing side work.
    * :attr:`state` is the ARRAYS, and it crosses every ``jax.jit`` boundary as an **argument**. As
      an argument the weights are a device-resident input the compiler sees only by shape; as a
      capture they are ten gigabytes of literal.

    The freeze split the earlier rounds established is preserved, but **not by the argument position**
    -- F3 claimed that and F3b's review disproved it. Keyword-only placement stops ``argnums`` from
    naming the backbone in the production update; it does NOT stop a caller from wrapping the builder
    and differentiating the state it passes in, which the reviewer demonstrated (42 frozen gradient
    leaves, aggregate norm ~2209). What actually holds the freeze is the leafwise ``stop_gradient``
    applied to ``frozen_state`` inside every velocity builder, where no caller can opt out of it.
    """

    graphdef: Any
    state: Any

    def merge(self):
        """The live module, rebuilt from this pair. For callers that hold both halves already.

        Deliberately NOT the seam a jitted function uses: inside a transform the state arrives as an
        argument, and the merge there is ``nnx.merge(graphdef, that_argument)``. A convenience that
        merged ``self.state`` inside a traced function would silently re-capture the weights, which
        is the whole defect F3 removed.
        """
        from flax import nnx

        return nnx.merge(self.graphdef, self.state)


# ``state`` is the DATA child and ``graphdef`` is static structure. Registered explicitly rather than
# by bare decorator: the decorator's default treats every field as data, which would put the graph
# definition into the leaf list and hand it to any `tree.map` -- structure is not data.
jax.tree_util.register_dataclass(FrozenBackbone, data_fields=["state"], meta_fields=["graphdef"])


def split_frozen_backbone(transformer) -> FrozenBackbone:
    """THE one splitter, so structure and arrays have a single origin.

    A second ``nnx.split`` somewhere else would be a second construction that agrees by coincidence
    -- the failure mode W1 and W3 both closed for the adapter and the program. One function, one
    pair, and every caller threads the pair it was given.
    """
    from flax import nnx

    graphdef, state = nnx.split(transformer)
    return FrozenBackbone(graphdef=graphdef, state=state)


def build_cfg_velocity_fn(transformer, adapters):
    """``(make_velocity_fn, adapter_params, frozen)`` — the frozen split, made structural.

    Only ``adapter_params`` is differentiable, and only the backbone's GRAPHDEF is captured: its
    arrays leave through ``frozen`` for the caller to pass back as an argument. That is clause (i)
    made stronger than it was -- the backbone is not filtered out of a gradient tree, it is never in
    one, AND it is never in the compiled module either (see :class:`FrozenBackbone`).

    ``make_velocity_fn(params, *, frozen_state, actions, guide_scale, ...)`` returns a
    ``velocity_fn(hidden_states, timestep, encoder_hidden_states)`` — the signature T1's sampler step
    and T2's endpoint kernel already consume, so nothing downstream needs to know a CFG combination
    happened. Both branches read ``hidden_states`` directly (clause (ii)).

    ``frozen_state`` is REQUIRED and keyword-only. A default would have been convenient and is
    exactly the defect to avoid: it would keep the arrays in this closure, so a caller who forgot to
    thread them would silently recreate the ten-gigabyte capture and nothing would fail until a TPU
    did. Forgetting is a ``TypeError`` instead.
    """
    from flax import nnx

    adapter_graphdef, adapter_params, adapter_rest = nnx.split(adapters, nnx.Param, ...)
    frozen = split_frozen_backbone(transformer)
    frozen_graphdef = frozen.graphdef

    def make_velocity_fn(params, *, frozen_state, actions, guide_scale, deterministic: bool = True, rngs=None):
        # THE FREEZE, ENFORCED RATHER THAN CLAIMED (F3b, review MAJOR-1). F3 argued that a
        # keyword-only argument made differentiating the backbone "unspellable". That was FALSE, and
        # the reviewer spelled it: a caller wraps this function and makes `frozen_state` its OWN
        # differentiated positional argument --
        #     jax.grad(lambda s: make_velocity_fn(p, frozen_state=s, ...)(...).sum())(frozen.state)
        # -- which produced 42 frozen gradient leaves, aggregate norm ~2209, on the tiny Wan stack.
        # Keyword-only is a fact about SYNTAX; it was never a fact about autodiff.
        #
        # `stop_gradient` leafwise IS the fact about autodiff. Applied at the BUILDER boundary so no
        # caller can opt out; identity in the forward pass, so no value, bitwise-parity or
        # finite-difference claim moves; and it cuts only the path INTO the weights -- the gradient
        # with respect to `hidden_states`, which clause (ii) exists to preserve, is untouched.
        frozen_state = jax.tree.map(jax.lax.stop_gradient, frozen_state)
        model = nnx.merge(frozen_graphdef, frozen_state)
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

    return make_velocity_fn, adapter_params, frozen


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
