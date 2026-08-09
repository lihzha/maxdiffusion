"""exp_06 F1b: THE logical optimizer update — one definition, shared by the trainer and by M1.

**Why this module exists.** The fit probe timed one MICROBATCH and `project_wall_clock` then multiplied
that by the number of logical updates a run performs. At GBS 256 with microbatch 8 that understates
the computation by 32x, and it never makes the accumulation state resident, so the peak it reported
was the peak of a program nobody runs. The reviewer's instruction was not "fix the probe" but
"**share one production logical-update primitive between trainer and probe**" — because a probe that
measures its own private construction is measuring the wrong thing by construction, however carefully
that construction is written.

So the unit is defined once, here:

* :func:`build_optimizer` builds the optimizer **production builds** — `max_utils.create_optimizer`
  over the configured Adam betas/eps/weight decay and the configured gradient clipping, on the
  configured warmup+decay schedule. The probe used a bare ``optax.adamw(learning_rate=...)``, which
  silently dropped clipping and the schedule, and clipping in particular allocates.
* :func:`build_logical_update` returns the step a training loop takes: accumulate the gradient over
  **every** microbatch of the logical batch, then apply **one** optimizer update. That is the unit
  `pos_logical_batch` names, the unit `max_train_steps` counts, and therefore the unit M1 must time.

Both callers get the same function, so "the probe measured what training runs" is a fact about the
call graph rather than a claim in a docstring.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Any, Sequence

import jax

from maxdiffusion.pos_rollout_stream import StepDraws

__all__ = [
    "DRAW_FIELDS",
    "LoadedBackbone",
    "TrainingProgram",
    "arm_context",
    "build_adapter_stack",
    "build_logical_update",
    "build_optimizer",
    "build_training_program",
    "draws_from_arrays",
    "draws_to_arrays",
    "load_backbone",
    "place_step_inputs",
    "production_batch_sharding",
    "program_scope",
    "replicated_sharding",
]

#: ``StepDraws`` is a frozen dataclass, not a pytree, so its arrays cross a jit boundary one by one.
DRAW_FIELDS = ("support_start", "support_end", "epsilon", "t_idx")


def draws_to_arrays(draws: StepDraws) -> tuple:
    return tuple(getattr(draws, field) for field in DRAW_FIELDS)


def draws_from_arrays(values: Sequence) -> StepDraws:
    return StepDraws(**dict(zip(DRAW_FIELDS, values)))


def build_optimizer(config, *, num_steps: int):
    """The optimizer PRODUCTION builds, from the configured schedule and clipping.

    ``max_utils.create_optimizer`` reads ``adam_b1``/``adam_b2``/``adam_eps``/``adam_weight_decay``
    and wraps the result in ``clip_by_global_norm`` / ``clip`` when the config enables them; the
    schedule is ``create_learning_rate_schedule`` over ``warmup_steps_fraction`` and
    ``learning_rate_schedule_steps``. A probe that substituted ``optax.adamw`` alone would measure a
    step with no clipping state and a constant learning rate.
    """
    from maxdiffusion import max_utils

    declared_steps = int(getattr(config, "learning_rate_schedule_steps"))
    schedule_steps = declared_steps if declared_steps > 0 else int(num_steps)
    schedule = max_utils.create_learning_rate_schedule(
        float(getattr(config, "learning_rate")),
        schedule_steps,
        float(getattr(config, "warmup_steps_fraction")),
        int(num_steps),
    )
    return max_utils.create_optimizer(config, schedule), schedule


def build_logical_update(loss_fn, optimizer, context) -> Any:
    """ONE logical optimizer update: accumulate every microbatch, then apply once.

    The returned callable takes the microbatch tuples the stream layer produced — never a single
    microbatch — so the accumulation is part of the compiled program: the gradient accumulator is
    resident for the whole update, which is exactly the state the previous per-microbatch timing left
    out of the footprint.

    The mean over microbatches is the accumulation identity the trainer relies on: with equal-width
    microbatches, the mean of the per-microbatch gradients is the gradient of the mean loss over the
    logical batch, so accumulation does not change the objective.
    """

    def update(params, opt_state, micro_batches, micro_draws):
        grads = None
        total = 0.0
        for batch, values in zip(micro_batches, micro_draws):
            (loss, _), grad = jax.value_and_grad(loss_fn, has_aux=True)(
                params, batch, context, draws=draws_from_arrays(values)
            )
            grads = grad if grads is None else jax.tree.map(lambda a, b: a + b, grads, grad)
            total = total + loss
        count = len(micro_batches)
        grads = jax.tree.map(lambda leaf: leaf / count, grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        import optax

        return optax.apply_updates(params, updates), opt_state, total / count

    return update


def build_adapter_stack(config, transformer):
    """THE adapter the run trains — one factory, called by the trainer AND by M1 (review W1, A2).

    The probe used to rebuild this construction by hand and **omitted ``dtype``, ``weights_dtype`` and
    ``precision``**. The reviewer's finding is exact: the two agreed only by coincidence of the
    pinned defaults, so the day someone sets ``activations_dtype: float32`` for a debugging run, M1
    would measure a bf16 adapter and authorize an fp32 one. A copy that agrees by coincidence is a
    copy that disagrees on the first edit, so there is now one function and both callers enter it.

    Mirrors ``wan_ti2v_side_adapter_trainer._build_adapters`` argument for argument, including the
    three dtype/precision arguments that decide the activation footprint M1 exists to measure.
    """
    import jax
    from flax import nnx

    from maxdiffusion.models.wan.side_adapter_wan import NNXWanSideAdapterStack

    def _dtype(name):
        import jax.numpy as jnp

        return jnp.dtype(name)

    return NNXWanSideAdapterStack(
        rngs=nnx.Rngs(jax.random.key(int(getattr(config, "seed")))),
        num_layers=transformer.config.num_layers,
        model_dim=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
        text_dim=int(getattr(config, "text_dim")),
        action_adapter_type=str(getattr(config, "action_adapter_type")),
        action_dim=int(getattr(config, "action_dim")),
        action_len=int(getattr(config, "action_len")),
        action_repr=str(getattr(config, "action_repr")),
        action_tokens=int(getattr(config, "action_tokens")),
        action_hidden=int(getattr(config, "action_hidden")),
        action_heads=int(getattr(config, "action_heads")),
        side_adapter_layers=str(getattr(config, "side_adapter_layers")),
        side_adapter_hidden=int(getattr(config, "side_adapter_hidden")),
        side_adapter_heads=int(getattr(config, "side_adapter_heads")),
        pre_context_tokens=int(getattr(config, "pre_context_tokens")),
        pre_context_heads=int(getattr(config, "pre_context_heads")),
        dtype=_dtype(getattr(config, "activations_dtype")),
        weights_dtype=_dtype(getattr(config, "weights_dtype")),
        precision=getattr(jax.lax.Precision, str(getattr(config, "precision"))),
    )


# =================================================================================================
# W3 — THE PROGRAM-FINALIZATION BOUNDARY, shared whole (final review, BLOCKER A2).
#
# W1 shared the adapter factory and W2 shared the optimizer and the logical update, and the reviewer's
# finding is that this was one level too low: **shared factories are not a shared PROGRAM.** The
# trainer replicated its adapter parameters and compiled under `logical_axis_rules`; M1 returned an
# UNSHARDED tree and measured under a bare `with mesh`. The Wan blocks translate logical axes during
# the forward, so the two sides could compile different activation shardings -- and the per-device HBM
# peak that M1 authorizes is exactly what that changes. Measured on an 8-device CPU mesh before this
# existed: the trainer's leaves carried `NamedSharding(mesh, P())` while M1's carried
# `SingleDeviceSharding(CpuDevice(0))`, and M1's null context was zeros where the trainer's was the
# loader's real T5 embedding.
#
# So the boundary moves up one level. Everything that decides WHAT IS COMPILED -- the backbone load,
# the deployed grid and null context, the replicated parameter/optimizer shardings, the mesh, the
# axis-rule scope and the jitted update -- is produced by ONE function that both callers enter.
# There is no longer a place for the two paths to differ, because there is no longer a second path.
# =================================================================================================


@dataclasses.dataclass(frozen=True)
class LoadedBackbone:
    """What the WEIGHTS seam produced: the frozen 5B and the three deployed quantities around it.

    ``scheduler`` is the authority on the sigma grid and the training-timestep count, ``null_context``
    is the T5("") embedding the deployed forward actually uses, and ``mesh`` is the pipeline's own --
    all three because M1 must not invent any of them (it used to build a zero context and its own
    grid, so it measured a program deployment never runs).
    """

    transformer: Any
    mesh: Any
    null_context: Any
    scheduler: Any
    loaders: Any = None


def load_backbone(config) -> LoadedBackbone:
    """The frozen 5B and the deployed null context/scheduler/mesh, through the SETTLED seams.

    One loader for the trainer and for M1. The pipeline is loaded under
    ``nn_partitioning.axis_rules(config.logical_axis_rules)`` because ``_load_wan_pipeline`` does that
    itself -- production's own construction rather than a re-statement of it -- and the encoder and
    VAE are dropped once the null context exists, exactly as the settled trainer drops them.

    Imported inside the function because that module pulls ``jaxopt`` and the diffusers stack, and
    resolved through the MODULE so a test can stand in for the WEIGHTS while no production caller can
    pass a model (the ``DevBatchReader`` lesson: an injectable model source is a forgeable one).
    """
    from maxdiffusion.trainers import wan_ti2v_side_adapter_trainer as settled

    loaders = settled.WanTI2VSideAdapterTrainer(config)
    pipeline = loaders._load_wan_pipeline()
    mesh = pipeline.mesh
    null_context = loaders._compute_null_context(pipeline, mesh)
    for attribute in ("vae", "vae_cache", "text_encoder", "tokenizer"):
        if hasattr(pipeline, attribute):
            delattr(pipeline, attribute)
    scheduler, _ = loaders._create_scheduler()
    return LoadedBackbone(
        transformer=pipeline.transformer,
        mesh=mesh,
        null_context=null_context,
        scheduler=scheduler,
        loaders=loaders,
    )


@contextlib.contextmanager
def program_scope(config, mesh):
    """THE execution scope: the mesh AND the deployed logical axis rules, together, always.

    The mesh alone is not the scope. The adapter's first-block feature extraction issues a sharding
    constraint against a LOGICAL axis name, and the rules are what translate it -- so a program built
    or run under `with mesh` alone can compile a different activation sharding than one built under
    both. Production wraps every step in this pair; now so does everything else.
    """
    from flax.linen import partitioning as nn_partitioning

    rules = nn_partitioning.axis_rules(getattr(config, "logical_axis_rules"))
    if mesh is None:
        with rules:
            yield
    else:
        with mesh, rules:
            yield


def arm_context(config, backbone: LoadedBackbone, *, k_b: int, num_steps: int):
    """The one ``ArmContext`` both arms and BOTH PATHS are handed.

    The grid comes from the loaded scheduler and the context from the loader, never from the YAML and
    never from a zero tensor: the sigma limits and the training-timestep count are properties of the
    pretrained model, and deployment reads them there.
    """
    from maxdiffusion.models.wan.overfit100_sampling import overfit100_sampler_grid
    from maxdiffusion.pos_rollout_arms import ArmContext

    import jax.numpy as jnp

    sigmas, timesteps = overfit100_sampler_grid(
        num_inference_steps=int(num_steps),
        flow_shift=float(getattr(config, "flow_shift")),
        sigma_min=float(backbone.scheduler.config.sigma_min),
        sigma_max=float(backbone.scheduler.config.sigma_max),
        num_train_timesteps=int(backbone.scheduler.config.num_train_timesteps),
    )
    return ArmContext(
        sigmas=sigmas,
        timesteps=timesteps,
        null_context=backbone.null_context,
        guide_scale=float(getattr(config, "side_adapter_guide_scale")),
        weights_dtype=jnp.dtype(getattr(config, "weights_dtype")),
        num_train_timesteps=int(backbone.scheduler.config.num_train_timesteps),
        num_steps=int(num_steps),
        k_b=int(k_b),
    )


def production_batch_sharding(mesh):
    """The sharding the DEPLOYED loader hands a training step, derived exactly as it derives it.

    ``multihost_dataloading._build_global_shape_and_sharding`` builds
    ``NamedSharding(mesh, PartitionSpec(mesh.axis_names))`` — the example axis split across every
    mesh axis — and that is what arrives at the step in production. M1 built ordinary
    single-device ``jnp.zeros`` instead, and because the jitted update declares no input shardings
    each side SPECIALIZED ON ITS OWN ARGUMENTS: measured on an 8-device mesh, the two paths' lowered
    input shardings differed. Sharing the function was not enough; the compiled-input contract has to
    be owned too (final re-ruling, BLOCKER 1).

    Taken from the MESH rather than from ``config.data_sharding`` because the mesh is what the loader
    reads; :func:`assert_batch_contract_matches_config` pins that the config agrees, so a drift
    between the two is a failure rather than a silent difference.
    """
    if mesh is None:
        return None
    from jax.sharding import NamedSharding, PartitionSpec

    return NamedSharding(mesh, PartitionSpec(mesh.axis_names))


def replicated_sharding(mesh):
    """Adapter parameters and their optimizer slots: small, and carrying action-length axes the
    transformer's logical rules would otherwise shard over context."""
    if mesh is None:
        return None
    from jax.sharding import NamedSharding, PartitionSpec as P

    return NamedSharding(mesh, P())


def assert_batch_contract_matches_config(config, mesh) -> None:
    """The loader's sharding and ``config.data_sharding`` must denote the same split."""
    if mesh is None:
        return
    from jax.sharding import NamedSharding, PartitionSpec as P

    declared = NamedSharding(mesh, P(*getattr(config, "data_sharding")))
    if declared != production_batch_sharding(mesh):
        raise ValueError(
            f"the deployed loader shards the example axis as {production_batch_sharding(mesh).spec} but "
            f"data_sharding declares {declared.spec}: the batch a step COMPILES against would not be "
            f"the batch the loader hands it."
        )


def place_step_inputs(mesh, *, params, opt_state, micro_batches, micro_draws):
    """THE compiled-input contract, applied by BOTH callers to their own inputs.

    Parameters and every optimizer-state leaf are REPLICATED; every batch leaf and the two per-example
    draws (``epsilon``, ``t_idx``) carry the loader's example-axis split; the two scalar support
    bounds are replicated. Both paths call this, so neither can specialize on a placement the other
    does not have — and the specs are absolute, so an oracle can assert them rather than only assert
    that the two sides agree (the symmetric-move blind spot the W3 oracle had).

    **Judgment call, recorded:** ``epsilon`` is per-example and batch-major, so it is placed like the
    data rather than replicated. At GBS 256 it is ~106 MB, and replicating that on every chip would
    be a footprint M1 then measures and authorizes. Production previously handed it to the step
    unplaced and let XLA choose; now both paths state the choice.
    """
    import jax

    replicated = replicated_sharding(mesh)
    batch = production_batch_sharding(mesh)
    if replicated is None:
        return params, opt_state, tuple(micro_batches), tuple(micro_draws)

    def _place(tree, sharding):
        return jax.tree.map(lambda leaf: jax.device_put(leaf, sharding), tree)

    placed_batches = tuple(_place(part, batch) for part in micro_batches)
    placed_draws = tuple(
        tuple(jax.device_put(value, replicated if index < 2 else batch) for index, value in enumerate(draws))
        for draws in micro_draws
    )
    return _place(params, replicated), _place(opt_state, replicated), placed_batches, placed_draws


@dataclasses.dataclass(frozen=True)
class TrainingProgram:
    """The compiled thing a run takes a step of — and the thing M1 measures. One object, one origin."""

    loss_fn: Any
    context: Any
    params: Any
    opt_state: Any
    optimizer: Any
    #: The jitted logical update, EXPOSED rather than closed over, so both callers can lower the same
    #: function and an oracle can compare the two lowerings' input/output shardings.
    step: Any
    #: THE shared DEV scorer, jitted, returning ``(loss, aux)``. M1 used to jit a PRIVATE scalar-only
    #: variant whose ``[0]`` let XLA prune the aux norms, understating the evaluation component of
    #: the wall projection (final re-ruling, MAJOR 3).
    score: Any
    update_fn: Any
    dev_loss_fn: Any
    example_shape: tuple
    mesh: Any
    scope: Any


def build_training_program(config, backbone: LoadedBackbone, *, arm: str, k_b: int, num_steps: int, dropout_rate=None):
    """Adapter, optimizer, shardings, scope and jitted step — the whole program, built ONCE.

    The FREEZE SPLIT is structural and comes from ``build_arm``: the frozen transformer is captured in
    the loss closure and never enters the differentiated tree, so the optimizer is initialised on
    adapter parameters alone. There is no "exclude the backbone" step for a later edit to forget.
    """
    from maxdiffusion.pos_rollout_arms import PRODUCTION_DROPOUT_RATE, build_arm

    rate = PRODUCTION_DROPOUT_RATE if dropout_rate is None else dropout_rate
    context = arm_context(config, backbone, k_b=k_b, num_steps=num_steps)

    with program_scope(config, backbone.mesh):
        adapters = build_adapter_stack(config, backbone.transformer)
        loss_fn, adapter_params = build_arm(str(arm), backbone.transformer, adapters, dropout_rate=rate)
        # REPLICATED, as the settled trainer replicates them (`_shard_state`): adapter parameters are
        # small and carry action-length axes the transformer's logical rules would otherwise shard
        # over context. M1 used to leave them on one device, which is a different program.
        assert_batch_contract_matches_config(config, backbone.mesh)
        replicated = replicated_sharding(backbone.mesh)
        if replicated is not None:
            adapter_params = jax.device_put(adapter_params, replicated)
        optimizer, _ = build_optimizer(config, num_steps=int(getattr(config, "max_train_steps")))
        opt_state = optimizer.init(adapter_params)
        if replicated is not None:
            # EVERY leaf, explicitly: `optimizer.init` does not promise to inherit its argument's
            # sharding, and the scalar count leaves demonstrably do not (final re-ruling, MAJOR 2).
            opt_state = jax.tree.map(lambda leaf: jax.device_put(leaf, replicated), opt_state)
        compiled = jax.jit(build_logical_update(loss_fn, optimizer, context))

        def step(params, opt_state_, micro_batches, micro_draws):
            """The step BOTH paths take, over inputs BOTH paths place identically."""
            return compiled(
                *place_step_inputs(
                    backbone.mesh,
                    params=params,
                    opt_state=opt_state_,
                    micro_batches=micro_batches,
                    micro_draws=micro_draws,
                )
            )

        def _lower(params, opt_state_, micro_batches, micro_draws):
            """Lowering goes through the same placement, so an oracle sees the compiled contract."""
            return compiled.lower(
                *place_step_inputs(
                    backbone.mesh,
                    params=params,
                    opt_state=opt_state_,
                    micro_batches=micro_batches,
                    micro_draws=micro_draws,
                )
            )

        step.lower = _lower
        scorer = jax.jit(
            lambda params, batch, values: loss_fn(params, batch, context, draws=draws_from_arrays(values))
        )

    def update_fn(state, batch_parts, draw_parts, schedule, global_step):
        """``run_loop``'s update signature over the shared step. No arm branch, no second program."""
        del schedule, global_step
        with program_scope(config, backbone.mesh):
            params, opt_state_next, loss = step(
                state.params,
                state.opt_state,
                tuple(batch_parts),
                tuple(draws_to_arrays(draws) for draws in draw_parts),
            )
        return dataclasses.replace(state, params=params, opt_state=opt_state_next), loss

    def dev_loss_fn(params, batch, scored_context, *, draws):
        if scored_context is not context:
            raise ValueError(
                "the DEV instrument was handed a context this program was not compiled against: "
                "two contexts mean two grids, and two grids mean two estimands"
            )
        with program_scope(config, backbone.mesh):
            return scorer(params, batch, draws_to_arrays(draws))

    return TrainingProgram(
        loss_fn=loss_fn,
        context=context,
        params=adapter_params,
        opt_state=opt_state,
        optimizer=optimizer,
        step=step,
        score=scorer,
        update_fn=update_fn,
        dev_loss_fn=dev_loss_fn,
        example_shape=(
            int(getattr(config, "latent_channels")),
            int(getattr(config, "latent_frames")),
            int(getattr(config, "latent_height")),
            int(getattr(config, "latent_width")),
        ),
        mesh=backbone.mesh,
        scope=lambda: program_scope(config, backbone.mesh),
    )
