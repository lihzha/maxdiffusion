"""exp_06 `rollout_adapter` — F3: the frozen 5B must never be a CAPTURED CONSTANT.

**The production failure this file exists to prevent, in the form it actually took.** The M1 fit
probe was submitted three times to v6e-8 against the real Wan2.2 TI2V 5B. All three attempts died on
``TPU_VM_HEALTH_TIMEOUT`` at roughly two hours, and *none of them ever finished its first XLA
compile*. Every attempt's log ends on the same line::

    UserWarning: A large amount of constants were captured during lowering (10.18GB total)

The frozen backbone was bound into the loss closure, so ``jax.jit`` promoted ~10.18 GB of bf16
weights to **literals inside the lowered module**. XLA then had to serialize and optimize a
ten-gigabyte program; the queue's health window reaped the VM long before it could. Not one
optimizer step was ever reached, and the failure was invisible to every oracle the experiment had:
``Lowered`` reports input shardings, ``Compiled.memory_analysis()`` reports argument bytes, and a
captured constant is neither.

**Why 2113 green tests could not see it.** The CPU suite builds fake backbones whose parameters are
kilobytes, so the pathology was present in every one of those tests and cost nothing in all of them.
The defect was not a wrong value anywhere — it was a *scaling* property, and nothing in the suite
scaled. So the guard below does not test a value: it **marks the fake backbone**, giving it
parameters deliberately larger than the threshold, and then asserts that those bytes do not appear
in the traced program. A fake that is 4 MB where production is 10 GB reproduces the defect
faithfully because the mechanism is proportional, not absolute.

**The measure is JAX's own.** ``jax/_src/interpreters/mlir.py:check_jaxpr_constants`` computes
``sum(getattr(c, "nbytes", 0) for c in closed_jaxpr.consts)`` and warns above
``JAX_CAPTURED_CONSTANTS_WARN_BYTES``. :func:`captured_constant_bytes` is that expression, so this
guard and the warning that appeared on the worker are reading the same number.

**Three detectors, deliberately, and the third exists because the first two were not enough.**
:func:`captured_constant_bytes` is what the compiler would bake; :func:`array_bytes_in_closure` is
what the function object holds — **a function that drives the frozen backbone may hold its graph
definition, never its arrays**; and :func:`backbone_attributable_bytes` asks the only question that
scales, *are any of these constants weights*, because a byte budget calibrated on a 16-wide fake
cannot express a leak that is 1 KB here and 50 MiB at production width.

**Each detector carries a positive control that proves it can still fail** — including the third,
whose control is the reviewer's own pre-fix evaluator construction. This is not ceremony. Round F3
shipped an evaluator guard that was green while production was broken, because the test passed
``frozen_state`` explicitly while production closed over it: the guard tested a safer program than
the one that ships. A guard whose ability to detect its defect is untested is not a guard, it is a
reassurance, and :func:`test_the_guards_production_shape_is_still_the_loaders_shape` now pins the
guard's shape to the loader's so that divergence cannot recur silently.

**Corrected in F3b:** an earlier version of this file claimed the evaluator "runs op-by-op" and was
therefore exempt from trace-level capture. That was wrong — ``cfg_rollout``'s loop body is staged and
compiled — and the wrongness was load-bearing, so it is recorded here rather than quietly deleted.
"""

from __future__ import annotations

import functools
import hashlib
import types as _types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from flax import nnx

from maxdiffusion import pos_rollout_arms as arms
from maxdiffusion import pos_rollout_step as step_module
from maxdiffusion import pos_rollout_update as update

_PACKAGE_ROOT = Path(update.__file__).resolve().parent
_CONFIG_PATH = _PACKAGE_ROOT / "configs" / "base_wan_5b_pos_rollout.yml"

# PRODUCTION latent geometry, as everywhere else in exp_06.
_C, _F, _H, _W = 48, 9, 12, 20
_ACTION_LEN, _ACTION_DIM = 32, 7
_TEXT, _NULL_LEN = 32, 8
_STEPS, _K, _LOGICAL, _MICRO = 4, 2, 4, 2

#: The budget, and it is deliberately generous. Everything a correct program may legitimately bake in
#: -- the sigma grid, the timestep grid, the null context at test width, scalar literals -- is
#: kilobytes. Anything at megabyte scale is a weight tensor that escaped into the module.
CONSTANT_BUDGET_BYTES = 1_000_000

#: The MARK. ``ffn_dim`` is the cheapest way to make a backbone heavy without making it slow to
#: trace: it adds two large matrices and no extra operations. At 32768 the fake backbone carries
#: ~4.37 MB of parameters -- 4.4x the budget -- so a capture cannot hide inside the margin, while
#: the traced graph stays a one-layer model that a laptop handles in seconds.
_MARK_FFN_DIM = 32768
_MIN_MARK_BYTES = 4_000_000


def _requires_backend():
    pytest.importorskip("torch")
    pytest.importorskip("aqt")


# =============================================================================================
# The two detectors.
# =============================================================================================


def captured_constant_bytes(closed_jaxpr) -> int:
    """JAX's own accounting, verbatim: ``check_jaxpr_constants``'s sum over ``jaxpr.consts``.

    The ``getattr`` default is JAX's too -- consts are not always arrays -- and copying the
    expression rather than paraphrasing it is what makes "this guard measures what the worker
    warned about" a fact rather than an intention.
    """
    return sum(getattr(const, "nbytes", 0) for const in closed_jaxpr.consts)


def _leaf_nbytes(leaf) -> int:
    """``leaf.nbytes`` or zero, defensively — and neither ``getattr`` form is safe enough here.

    A closure legitimately holds arbitrary objects, and exp_06's own ``HyperParameters`` stand-in
    raises **ValueError** from ``__getattr__`` on an unknown name (issue #11, reproduced deliberately
    in the fake). Three-argument ``getattr`` does not fall back on a raise, and ``hasattr`` only
    swallows ``AttributeError`` — both propagate the ValueError and kill the walk partway through.
    A detector that dies on a hostile leaf silently under-reports every leaf after it.
    """
    try:
        value = leaf.nbytes
    except Exception:  # noqa: BLE001 -- any failure to answer means "not an array"
        return 0
    return int(value) if isinstance(value, (int, np.integer)) else 0


def array_bytes_in_closure(function) -> int:
    """Total bytes of every array reachable from ``function``'s closure, following nested functions.

    Walks ``__closure__`` transitively, because the chain that mattered in production had three
    links: ``update`` closed over ``loss_fn``, which closed over ``make_velocity_fn``, which closed
    over the split backbone. A one-level check would have reported zero and been wrong.

    ``jax.tree.leaves`` descends the containers that carry weights (``nnx.State`` is a pytree, and a
    plain tuple of them flattens); anything unregistered -- a ``GraphDef``, a module, a dtype -- is
    an opaque leaf and contributes nothing, which is exactly right: a graph definition holds
    structure, not arrays.
    """
    seen: set[int] = set()
    total = 0
    pending = [function]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        cells = current.__closure__ if hasattr(current, "__closure__") else ()
        cells = cells or ()
        for cell in cells:
            try:
                value = cell.cell_contents
            except ValueError:  # an empty cell in a recursive definition
                continue
            if callable(value) and hasattr(value, "__closure__"):
                pending.append(value)
                continue
            for leaf in jax.tree.leaves(value):
                total += _leaf_nbytes(leaf)
    return total


# =============================================================================================
# The marked fake backbone.
# =============================================================================================


class _Config(dict):
    """``pyconfig.HyperParameters``' declared contract: attribute reads, ``ValueError`` when absent."""

    def __getattr__(self, key):
        if key not in self:
            raise ValueError(f"Key {key} not in config")
        return self[key]

    def get_keys(self):
        return dict(self)


def _mesh():
    return jax.sharding.Mesh(np.array(jax.devices()).reshape(1, 1, 1, 1), ("data", "fsdp", "context", "tensor"))


def _config(**overrides):
    values = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    values.update(
        {
            "run_name": "f3-captured-constants",
            "text_dim": _TEXT,
            "wan_max_sequence_length": _NULL_LEN,
            "action_len": _ACTION_LEN,
            "action_dim": _ACTION_DIM,
            "action_tokens": 4,
            "action_hidden": 16,
            "action_heads": 2,
            "pre_context_tokens": 4,
            "pre_context_heads": 2,
            "side_adapter_layers": "0",
            "side_adapter_hidden": 16,
            "side_adapter_heads": 2,
            "side_adapter_sampling_steps": _STEPS,
            "weights_dtype": "float32",
            "activations_dtype": "float32",
            "pos_logical_batch": _LOGICAL,
            "pos_microbatch": _MICRO,
            "pos_rollout_k": _K,
            "max_train_steps": 4,
        }
    )
    values.update(overrides)
    return _Config(values)


def backbone_attributable_bytes(closed_jaxpr) -> int:
    """Bytes of captured constants that ARE backbone weights — the relative test (F3b, MAJOR-3b).

    **Why a byte budget was not enough.** The old guard asked only whether total constants stayed
    under 1 MB on a 1-layer, 16-wide fake. Production is 40 layers at 5120 wide, so a 16x16 fp32
    projection that is 1 KB here is **50 MiB** there: a capture of every attention projection across
    40 layers would sail under the fake's 1 MB budget and still put multiple gigabytes of literals
    into the real module. An absolute budget on a small fake cannot express "no weights leaked".

    So this asks the question directly. Every parameter leaf of the fake backbone is MARKED with
    distinctive random content (:func:`_marked_transformer`), and a constant is attributed to the
    backbone when its bytes hash to one of those marks. The assertion is then ZERO — not "small" —
    and it scales, because it does not depend on how big the fake happens to be.
    """
    marks = _marked_leaf_digests()
    total = 0
    for const in closed_jaxpr.consts:
        try:
            payload = np.asarray(const)
        except Exception:  # noqa: BLE001 -- a non-array const carries no weights
            continue
        if hashlib.sha256(payload.tobytes()).hexdigest() in marks:
            total += int(payload.nbytes)
    return total


@functools.lru_cache(maxsize=1)
def _marked_leaf_digests() -> frozenset:
    """The content digest of every marked backbone leaf — the identity the attribution matches on."""
    leaves = jax.tree.leaves(nnx.split(_marked_transformer())[1])
    return frozenset(hashlib.sha256(np.asarray(leaf).tobytes()).hexdigest() for leaf in leaves)


@functools.lru_cache(maxsize=1)
def _marked_transformer():
    """A REAL ``WanModel`` whose EVERY parameter leaf is marked — heavy, and individually traceable.

    Two marks, because the two detectors ask different questions. ``ffn_dim`` is enlarged so the
    total comfortably exceeds any byte budget; then **every leaf is overwritten with distinctive
    random content**, so no leaf is a zero tensor or a repeat that could collide with a legitimate
    constant, and any captured array can be attributed to this backbone by its bytes alone. Marking
    only the FFN (as F3 did) left every projection indistinguishable from ordinary program data.
    """
    from flax import nnx as _nnx
    from flax.linen import partitioning as nn_partitioning

    from maxdiffusion.models.wan.transformers.transformer_wan import WanModel

    with nn_partitioning.axis_rules(()), _mesh():
        model = WanModel(
            rngs=nnx.Rngs(jax.random.key(0)),
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=_C,
            out_channels=_C,
            text_dim=_TEXT,
            freq_dim=16,
            ffn_dim=_MARK_FFN_DIM,
            num_layers=1,
            attention="dot_product",
            rope_max_seq_len=64,
            scan_layers=False,
            dtype=jnp.float32,
            weights_dtype=jnp.float32,
        )
    # THE MARK, on every leaf. Distinctive random content per leaf: no zeros, no repeats, so a
    # captured array can be attributed to this backbone by its bytes and nothing else can be
    # mistaken for one.
    graphdef, state = _nnx.split(model)
    leaves, treedef = jax.tree.flatten(state)
    keys = jax.random.split(jax.random.key(20260810), len(leaves))
    marked = [
        jax.random.normal(key, leaf.shape, leaf.dtype) + jnp.asarray(index + 1, leaf.dtype)
        for index, (key, leaf) in enumerate(zip(keys, leaves))
    ]
    _nnx.update(model, jax.tree.unflatten(treedef, marked))
    return model


#: The guide scale T3a's own oracles use, so the gradient claims here sit on their operating point.
_STEP_GUIDE = 5.0


def _reviewer_stack():
    """The tiny real Wan stack the REVIEWER ran its attack against, plus matching inputs.

    Borrowed from ``test_pos_rollout_step`` rather than rebuilt: the freeze claim must be tested on
    the fixture whose gradients T3a already characterised, and rebuilding it here would be a second
    construction agreeing by coincidence — the failure mode this campaign has closed four times.
    """
    from maxdiffusion.tests.worklogs_yixun.test_pos_rollout_step import (
        _grid,
        _inputs,
        _mesh_context,
        _tiny_cfg_stack,
    )

    transformer, adapters = _tiny_cfg_stack()
    sigmas, timesteps = _grid()
    rules, mesh = _mesh_context()
    return transformer, adapters, _inputs(11), sigmas, timesteps, rules, mesh


def _marked_backbone():
    scheduler = _types.SimpleNamespace(
        config=_types.SimpleNamespace(sigma_min=0.0, sigma_max=1.0, num_train_timesteps=1000)
    )
    return update.LoadedBackbone(
        transformer=_marked_transformer(),
        mesh=_mesh(),
        null_context=jnp.full((1, _NULL_LEN, _TEXT), 0.25, jnp.float32),
        scheduler=scheduler,
    )


def _marked_bytes() -> int:
    return sum(int(getattr(leaf, "nbytes", 0) or 0) for leaf in jax.tree.leaves(nnx.split(_marked_transformer())[1]))


def _adapters(config):
    return update.build_adapter_stack(config, _marked_transformer())


def _micro(batch_size=_MICRO):
    return {
        "z_video": jnp.zeros((batch_size, _C, _F, _H, _W), jnp.float32),
        "z_i0": jnp.zeros((batch_size, _C, 1, _H, _W), jnp.float32),
        "actions": jnp.zeros((batch_size, _ACTION_LEN, _ACTION_DIM), jnp.float32),
    }


def _draws(batch_size=_MICRO):
    return (
        jnp.asarray(0, jnp.int32),
        jnp.asarray(_K, jnp.int32),
        jnp.zeros((batch_size, _C, _F, _H, _W), jnp.float32),
        jnp.zeros((batch_size,), jnp.int32),
    )


def _program(arm="rollout"):
    config = _config()
    return config, update.build_training_program(config, _marked_backbone(), arm=arm, k_b=_K, num_steps=_STEPS)


# =============================================================================================
# 0. The mark is real, and both detectors can still fail.
# =============================================================================================


def test_the_marked_backbone_is_actually_heavier_than_the_budget():
    """If the mark ever shrinks below the budget every test in this file passes for free."""
    _requires_backend()
    marked = _marked_bytes()
    assert marked >= _MIN_MARK_BYTES, f"the mark is only {marked} bytes; it must exceed the budget by a margin"
    assert marked > CONSTANT_BUDGET_BYTES * 4


def test_the_trace_detector_sees_a_deliberate_capture():
    """Positive control for :func:`captured_constant_bytes` — it must fail on a real capture."""
    weights = jnp.ones((1024, 1024), jnp.float32)  # 4 MB, over budget on purpose

    def captures(x):
        return x + weights.sum()

    baked = captured_constant_bytes(jax.jit(captures).trace(jnp.float32(0.0)).jaxpr)
    assert baked >= weights.nbytes, f"the detector reported {baked} for a {weights.nbytes}-byte capture"

    def threads(x, w):
        return x + w.sum()

    threaded = captured_constant_bytes(jax.jit(threads).trace(jnp.float32(0.0), weights).jaxpr)
    assert threaded < CONSTANT_BUDGET_BYTES, "the same array passed as an ARGUMENT must not be a constant"


def test_the_attribution_detector_sees_the_reviewers_pre_fix_evaluator_form():
    """Positive control for :func:`backbone_attributable_bytes` — and it is the REAL red construction.

    A detector that returned zero unconditionally would make every "attributable == 0" assertion in
    this file vacuous, which is the failure mode F3 shipped in a different costume. So the control is
    not a synthetic array: it is **the exact pre-fix evaluator shape** the reviewer traced — a
    ``velocity_for`` that closes over ``frozen.state``, wrapped in an outer jit. The reviewer measured
    4,373,412 bytes against a 4,372,352-byte backbone; this asserts the detector attributes the whole
    marked backbone to the backbone, so we know it can see what it is asked to forbid.
    """
    _requires_backend()
    config = _config()
    from maxdiffusion.models.wan.overfit100_sampling import overfit100_sampler_grid

    with update.program_scope(config, _mesh()):
        make_velocity_fn, adapter_params, frozen = step_module.build_cfg_velocity_fn(
            _marked_transformer(), _adapters(config)
        )
    sigmas, timesteps = overfit100_sampler_grid(
        num_inference_steps=_STEPS, flow_shift=5.0, sigma_min=0.0, sigma_max=1.0, num_train_timesteps=1000
    )
    batch = _micro(1)
    context = jnp.zeros((1, _NULL_LEN, _TEXT), jnp.float32)

    # THE DEFECT, reconstructed: the weights are bound into the closure instead of passed.
    def bound_velocity_for(params, actions):
        return make_velocity_fn(params, frozen_state=frozen.state, actions=actions, guide_scale=5.0)

    def scored(params, z, z_i0, actions):
        return step_module.cfg_rollout(
            z,
            velocity_fn=bound_velocity_for(params, actions),
            sigmas=sigmas,
            timesteps=timesteps,
            context=context,
            z_i0=z_i0,
            start=0,
            num_steps=1,
        )

    with update.program_scope(config, _mesh()):
        traced = jax.jit(scored).trace(adapter_params, batch["z_video"], batch["z_i0"], batch["actions"])
    attributable = backbone_attributable_bytes(traced.jaxpr)
    assert attributable >= _MIN_MARK_BYTES, (
        f"the attribution detector reported {attributable} bytes for the pre-fix evaluator form, which "
        f"the reviewer measured at 4,373,412: a detector that cannot see this capture makes every "
        f"'attributable == 0' assertion in this file meaningless"
    )


def test_the_closure_detector_sees_a_deliberate_capture_through_nesting():
    """Positive control for :func:`array_bytes_in_closure`, at the depth production actually had."""
    weights = jnp.ones((1024, 1024), jnp.float32)

    def outer():
        def middle():
            def inner(x):
                return x + weights.sum()

            return inner

        return middle

    assert array_bytes_in_closure(outer()()) >= weights.nbytes
    assert array_bytes_in_closure(lambda x: x) == 0


# =============================================================================================
# 1. The velocity builders: the closure may hold the GRAPH, never the WEIGHTS.
# =============================================================================================


@pytest.mark.parametrize("builder_name", ["build_cfg_velocity_fn", "build_one_step_velocity_fn"])
def test_the_velocity_builders_do_not_hold_the_backbones_arrays(builder_name):
    """The exact link in the chain that put 10.18 GB into the module."""
    _requires_backend()
    config = _config()
    builder = getattr(step_module, builder_name, None) or getattr(arms, builder_name)
    with update.program_scope(config, _mesh()):
        make_velocity_fn, _adapter_params, _frozen = builder(_marked_transformer(), _adapters(config))
    held = array_bytes_in_closure(make_velocity_fn)
    assert held < CONSTANT_BUDGET_BYTES, (
        f"{builder_name}'s velocity closure holds {held} bytes of arrays (the marked backbone is "
        f"{_marked_bytes()}): the frozen weights must cross every jit boundary as an ARGUMENT"
    )


@pytest.mark.parametrize("arm", list(arms.ARMS))
def test_the_arm_losses_do_not_hold_the_backbones_arrays(arm):
    """...and the property survives ``build_arm``, which is what the trainer and M1 actually call."""
    _requires_backend()
    config = _config()
    with update.program_scope(config, _mesh()):
        loss_fn, _adapter_params, _frozen = arms.build_arm(arm, _marked_transformer(), _adapters(config))
    held = array_bytes_in_closure(loss_fn)
    assert held < CONSTANT_BUDGET_BYTES, f"the {arm} loss closure holds {held} bytes of arrays"


# =============================================================================================
# 2. The compiled programs: nothing backbone-scale reaches `jaxpr.consts`.
# =============================================================================================


@pytest.mark.parametrize("arm", list(arms.ARMS))
def test_the_training_program_traces_without_baking_the_backbone(arm):
    """THE regression guard. Against the pre-fix code this reported the whole marked backbone."""
    _requires_backend()
    _config_, program = _program(arm)
    micro_batches = (_micro(), _micro())
    micro_draws = (_draws(), _draws())
    with program.scope():
        traced = program.step.trace(program.params, program.opt_state, micro_batches, micro_draws)
    attributable = backbone_attributable_bytes(traced.jaxpr)
    assert attributable == 0, (
        f"the {arm} update bakes {attributable} bytes of BACKBONE weights (marked backbone: "
        f"{_marked_bytes()}). On the real 5B this is the 10.18 GB that made XLA lowering outlive the "
        f"TPU health window. Zero is the only acceptable amount; a byte budget cannot express this "
        f"because a leak that is 1 KB on this fake is 50 MiB at production width (F3b, MAJOR-3b)."
    )
    baked = captured_constant_bytes(traced.jaxpr)
    assert baked < CONSTANT_BUDGET_BYTES, f"the {arm} update bakes {baked} bytes of constants overall"


@pytest.mark.parametrize("arm", list(arms.ARMS))
def test_the_dev_scorer_traces_without_baking_the_backbone(arm):
    """The DEV instrument compiles its own program; it had the identical defect."""
    _requires_backend()
    _config_, program = _program(arm)
    with program.scope():
        traced = program.score.trace(program.params, _micro(), _draws())
    attributable = backbone_attributable_bytes(traced.jaxpr)
    assert attributable == 0, f"the {arm} DEV scorer bakes {attributable} bytes of BACKBONE weights"
    baked = captured_constant_bytes(traced.jaxpr)
    assert baked < CONSTANT_BUDGET_BYTES, f"the {arm} DEV scorer bakes {baked} bytes of constants overall"


def _production_kernel(config, make_velocity_fn, frozen):
    """The PRODUCTION kernel, from the production builders — not a reconstruction (F3c).

    This is the whole point of the round. Twice now the evaluator guard was green while production
    was broken, because the guard built its own shape: F3 passed ``frozen_state`` explicitly where
    production closed over it, and F3b jitted its own function where production jitted nothing. So
    the guard no longer builds anything. It calls ``build_velocity_builder`` and
    ``build_rollout_kernel`` -- the same two functions ``load_device_backend`` calls, in the same
    order, with the same arguments -- and traces the object that comes back.
    """
    from maxdiffusion import eval_wan_pos_rollout as evaluator

    builder = evaluator.build_velocity_builder(
        make_velocity_fn=make_velocity_fn,
        frozen_graphdef=frozen.graphdef,
        guide_scale=float(config["side_adapter_guide_scale"]),
    )
    return builder, evaluator.build_rollout_kernel(builder, num_steps=1)


@pytest.mark.parametrize("adapter_enabled", [True, False])
def test_the_production_eval_rollout_does_not_bake_the_backbone(adapter_enabled):
    """THE evaluator guard, against the form the evaluator ACTUALLY uses (F3b, review MAJOR-2).

    F3 asserted the evaluator was exempt because it "runs op-by-op". Two things were wrong. The
    rollout reaches ``cfg_rollout``'s loop, **whose body is staged and compiled**; and the guard that
    was supposed to catch this tested a calling convention production did not use, so it was green
    while the reviewer traced the real shape capturing **4,373,412 bytes** of a 4,372,352-byte marked
    backbone.

    This traces the production path end to end: ``DeviceBackend.score``'s factory-plus-weights call
    into ``rollout_prediction``, which builds the velocity **inside** the compiled region from the
    ``frozen_state`` argument. Both branches of ``velocity_for`` are covered, because the null branch
    merges its own module and could re-capture independently.
    """
    _requires_backend()
    config = _config()
    from maxdiffusion.models.wan.overfit100_sampling import overfit100_sampler_grid

    with update.program_scope(config, _mesh()):
        make_velocity_fn, adapter_params, frozen = step_module.build_cfg_velocity_fn(
            _marked_transformer(), _adapters(config)
        )
    velocity_builder, kernel = _production_kernel(config, make_velocity_fn, frozen)
    sigmas, timesteps = overfit100_sampler_grid(
        num_inference_steps=_STEPS, flow_shift=5.0, sigma_min=0.0, sigma_max=1.0, num_train_timesteps=1000
    )
    batch = _micro(1)
    context = jnp.zeros((1, _NULL_LEN, _TEXT), jnp.float32)

    # THE PRODUCTION KERNEL ITSELF, traced on the arguments `rollout_prediction` hands it.
    with update.program_scope(config, _mesh()):
        traced = kernel.trace(
            adapter_params,
            frozen.state,
            batch["z_video"],
            batch["z_i0"],
            context,
            batch["actions"],
            sigmas,
            timesteps,
            adapter_enabled=adapter_enabled,
        )
    attributable = backbone_attributable_bytes(traced.jaxpr)
    assert attributable == 0, (
        f"the production eval kernel (adapter_enabled={adapter_enabled}) bakes {attributable} bytes "
        f"of BACKBONE weights. On the real 5B this is the ~10 GB that killed M1, at M4 instead."
    )
    # ...and the builder the kernel closes over holds no arrays either: holding a registered
    # `FrozenBackbone` for its graphdef made all 4 MiB of state reachable in the reviewer's probe.
    held = array_bytes_in_closure(velocity_builder)
    assert held == 0, f"the production velocity builder's closure reaches {held} bytes of arrays"
    held_kernel = array_bytes_in_closure(kernel)
    assert held_kernel == 0, f"the production kernel's closure reaches {held_kernel} bytes of arrays"
    baked = captured_constant_bytes(traced.jaxpr)
    assert baked < CONSTANT_BUDGET_BYTES, f"the eval rollout bakes {baked} bytes of constants overall"


# =============================================================================================
# 3. The defects the params-as-argument design introduces, each pinned.
# =============================================================================================


def test_the_frozen_state_is_not_donated_so_a_second_step_still_runs():
    """Defect (a): a donated argument is freed after one call, and the run dies on step 2.

    Behavioural rather than introspective on purpose -- ``donate_argnums`` is one way to lose the
    buffer and aliasing is another, and a second successful step refutes both at once.
    """
    _requires_backend()
    _config_, program = _program("one_step")
    micro_batches = (_micro(), _micro())
    micro_draws = (_draws(), _draws())
    with program.scope():
        params, opt_state, first = program.step(program.params, program.opt_state, micro_batches, micro_draws)
        _params2, _opt2, second = program.step(params, opt_state, micro_batches, micro_draws)
    # Both steps EXECUTED -- concrete arrays came back, so the second call really ran against buffers
    # the first call did not free. Finiteness is deliberately NOT asserted: the marked backbone is
    # built for byte attribution, with random content on every leaf at ffn_dim=32768, so it overflows
    # when actually evaluated. Numerical claims live on the sane fixture
    # (`test_the_stop_gradient_does_not_cut_the_rollout_states_gradient`); this test's property is
    # buffer liveness, and a NaN scalar demonstrates liveness exactly as well as a finite one.
    assert first.shape == () and second.shape == (), "both steps must return a concrete scalar loss"
    # ...and the frozen arrays are still alive and readable AFTER both steps.
    for leaf in jax.tree.leaves(program.frozen.state):
        assert not leaf.is_deleted(), "a frozen-backbone buffer was freed: the argument was donated"
    assert float(jnp.sum(jax.tree.leaves(program.frozen.state)[0])) is not None, "buffers remain readable"


def test_the_frozen_state_is_the_backbones_own_buffers_not_a_copy():
    """Defect (d): passing the weights as an argument must not duplicate 10 GB of HBM."""
    _requires_backend()
    _config_, program = _program("rollout")
    held = jax.tree.leaves(program.frozen.state)
    live = jax.tree.leaves(nnx.split(_marked_transformer())[1])
    assert held, "the program must carry the frozen state it passes"
    assert len(held) == len(live)
    for mine, theirs in zip(held, live):
        assert mine is theirs, "the program copied the frozen backbone instead of referencing it"


def test_the_frozen_state_is_never_replaced_per_call():
    """Defect (b)/(d): the step must not ``device_put`` the backbone on every call.

    ``place_step_inputs`` re-places its four inputs at every step, which is free for arrays that are
    already where they belong and would be a 10 GB reshard for the backbone. The frozen state must
    therefore not pass through it -- pinned here on the function's declared inputs.
    """
    _requires_backend()
    import inspect

    signature = inspect.signature(update.place_step_inputs)
    assert "frozen" not in signature.parameters and "frozen_state" not in signature.parameters, (
        "the frozen backbone must not enter the per-call placement contract: it is placed once, at "
        "load, by the pipeline that produced it"
    )
    source = inspect.getsource(update.build_training_program)
    assert "frozen.state" in source, "the built program must pass the frozen state it split"


def test_the_optimizer_is_built_on_the_adapter_tree_alone():
    """Defect (c): the freeze split must stay a fact, not a convention.

    It is structural first -- the frozen state is a KEYWORD-ONLY argument of the loss, and
    ``jax.value_and_grad`` takes ``argnums`` over POSITIONAL arguments only, so no caller can even
    spell "differentiate the backbone". This is the loud second line the round asked for.
    """
    _requires_backend()
    _config_, program = _program("rollout")
    opt_leaves = jax.tree.leaves(program.opt_state)
    param_leaves = jax.tree.leaves(program.params)
    assert param_leaves, "the adapter tree must be non-empty"
    frozen_bytes = sum(int(getattr(leaf, "nbytes", 0) or 0) for leaf in jax.tree.leaves(program.frozen.state))
    opt_bytes = sum(int(getattr(leaf, "nbytes", 0) or 0) for leaf in opt_leaves)
    assert frozen_bytes >= _MIN_MARK_BYTES
    assert opt_bytes < frozen_bytes, "the optimizer carries slots for the frozen backbone"


def test_the_guards_production_shape_is_still_the_loaders_shape():
    """The tripwire that would have caught F3's and F3b's mistakes BEFORE hardware did.

    Both earlier evaluator guards passed while production was broken, for the same reason each time:
    nothing bound the guard's shape to the loader's, so they drifted and the drift was invisible.
    The guard now *calls* the production builders, and this pins that the loader still calls them
    too -- and that the hazardous spellings have not come back.
    """
    import inspect

    from maxdiffusion import eval_wan_pos_rollout as evaluator

    loader = inspect.getsource(evaluator.load_device_backend)
    assert "build_velocity_builder(" in loader, "the loader must use the SHARED velocity builder"
    assert "build_rollout_kernel(" in loader, "the loader must build the shared compiled kernel"
    assert "frozen_graphdef = frozen.graphdef" in loader, "the graphdef must be extracted before any closure"
    assert "del frozen" in loader, (
        "the loader must drop the FrozenBackbone before defining closures: it is a registered pytree, "
        "so retaining it for its graphdef keeps every weight reachable (the reviewer measured 4 MiB)"
    )

    prediction = inspect.signature(evaluator.rollout_prediction).parameters
    assert "kernel" in prediction and "frozen_state" in prediction
    for gone in ("velocity_fn", "make_velocity"):
        assert gone not in prediction, f"rollout_prediction accepts {gone} again: that spelling captured the backbone"

    # The bind-then-jit hazard has no public seam: the backend exposes a compiled kernel, not a
    # callable with the weights bound into it.
    backend_init = inspect.signature(evaluator.DeviceBackend.__init__).parameters
    assert "kernel" in backend_init
    assert "velocity_for" not in backend_init, (
        "DeviceBackend exposes a bound-velocity seam again: `jax.jit(lambda z: cfg_rollout(z, "
        "velocity_fn=bound, ...))` captured 4,372,352 bytes in the reviewer's construction"
    )
    score = inspect.getsource(evaluator.DeviceBackend.score)
    assert "kernel=self.kernel" in score and "frozen_state=self.frozen_state" in score


def test_the_backbone_cannot_receive_a_gradient_under_the_reviewers_own_attack():
    """THE freeze test — the reviewer's adversarial construction, verbatim (F3b, review MAJOR-1).

    **What this replaces, and why.** F3 tested that ``frozen_state`` is keyword-only and concluded
    that differentiating the backbone was therefore "a call that cannot be spelled". The reviewer
    spelled it: wrap the builder in a lambda so the frozen state becomes the *wrapper's* positional
    argument, and ``jax.grad`` differentiates it happily. Executed against this same tiny stack it
    produced **42 frozen gradient leaves with aggregate norm ~2209**. A signature is a fact about
    syntax; it was never a fact about autodiff, and the old test proved only the former.

    The fix is ``stop_gradient`` applied leafwise inside every velocity builder, where no caller can
    opt out. So the attack still *runs* -- it must, or the test would be measuring nothing -- and now
    returns exactly zero for every leaf.
    """
    _requires_backend()
    # THE REVIEWER'S OWN STACK. The marked backbone exists for byte attribution and is deliberately
    # numerically extreme (random weights at ffn_dim=32768 overflow to NaN when actually executed),
    # so a gradient claim must be made on the sane fixture the reviewer used -- and a NaN gradient
    # would read as "zero" to a careless assertion, which is exactly the trap to avoid here.
    transformer, adapters, data, sigmas, timesteps, rules, mesh = _reviewer_stack()

    with rules, mesh:
        make_velocity_fn, params, frozen = step_module.build_cfg_velocity_fn(transformer, adapters)

        def attacked(state):
            velocity_fn = make_velocity_fn(
                params, frozen_state=state, actions=data["actions"], guide_scale=_STEP_GUIDE
            )
            return step_module.cfg_rollout(
                data["z"],
                velocity_fn=velocity_fn,
                sigmas=sigmas,
                timesteps=timesteps,
                context=data["null_context"],
                z_i0=data["z_i0"],
                start=0,
                num_steps=1,
            ).sum()

        grads = jax.grad(attacked)(frozen.state)

    leaves = jax.tree.leaves(grads)
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves), "a NaN gradient is not a zero one"
    assert leaves, "the attack must actually differentiate a non-empty frozen tree, or it proves nothing"
    norm = float(jnp.sqrt(sum(jnp.sum(jnp.asarray(leaf, jnp.float32) ** 2) for leaf in leaves)))
    assert norm == 0.0, (
        f"the reviewer's attack recovered a frozen-backbone gradient of norm {norm} over {len(leaves)} "
        f"leaves: `stop_gradient` at the builder boundary is what makes the freeze a fact about "
        f"autodiff rather than about argument syntax"
    )


def test_the_stop_gradient_does_not_cut_the_rollout_states_gradient():
    """...and the cut is SURGICAL: clause (ii) must survive it.

    ``stop_gradient`` on the weights must not also sever the dependence on ``hidden_states``, which
    is the entire inter-step gradient path R-B exists to create. A freeze that silently truncated
    that would make the rollout objective quietly equal to a one-step one — the exact failure T3a's
    finite-difference oracle was built to detect — so it is asserted here too, next to the freeze.
    """
    _requires_backend()
    transformer, adapters, data, sigmas, timesteps, rules, mesh = _reviewer_stack()

    with rules, mesh:
        make_velocity_fn, params, frozen = step_module.build_cfg_velocity_fn(transformer, adapters)

        def through_state(z):
            velocity_fn = make_velocity_fn(
                params, frozen_state=frozen.state, actions=data["actions"], guide_scale=_STEP_GUIDE
            )
            return step_module.cfg_rollout(
                z,
                velocity_fn=velocity_fn,
                sigmas=sigmas,
                timesteps=timesteps,
                context=data["null_context"],
                z_i0=data["z_i0"],
                start=0,
                num_steps=2,
            ).sum()

        grad_z = jax.grad(through_state)(data["z"])
    norm = float(jnp.linalg.norm(grad_z))
    assert np.isfinite(norm) and norm > 0.0, "the rollout state's gradient was severed by the freeze"
