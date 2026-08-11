"""exp_06 `rollout_adapter` — F4: the microbatch accumulation is a ``lax.scan``, and it is the SAME
accumulation.

**The production failure this round answers.** M1-2 cleared F3's constants fix on real hardware — four
attempts loaded the frozen 5B and printed ``[M1] entering rollout microbatch=8 k=2: building and
compiling``, a line M1-1 never reached in twelve tries — and then all four died 2–10 minutes into that
FIRST compile, on four different VMs, reported as ``TPU_VM_HEALTH_UNHEALTHY_MAINTENANCE``. Same-phase
death across four VMs is a workload signature, not a fleet one. The killer was the GRAPH: the logical
update accumulated its microbatches with a Python ``for``, so ``jax.jit`` emitted a fresh
forward+backward of the 5B into the jaxpr *per microbatch*. The pilot's 32 microbatches (GBS 256 at
microbatch 8) was therefore a ~118,000-equation program, and XLA exhausted the host compiling it.

``build_logical_update`` now scans over stacked microbatch chunks: one gradient block, compiled once,
graph size O(1) in the microbatch count. :func:`test_the_update_graph_stays_flat_as_microbatches_grow`
is that claim as a standing guard, red side included — the unrolled form is re-derived in-test and
shown to grow while production stays flat.

**Everything else here exists because a graph-shape fix must not be a NUMERICS change.** The argument
for parity is easy and this campaign has already punished three easy arguments: scan is sequential,
the carry starts at exact zeros, ``0 + g1`` is exactly ``g1``, so the summation order is the Python
loop's. That argument is correct and it is not the whole story, so it is measured rather than trusted.

**What the measurement found, stated before the tests state it.**

* For the **rollout** arm — the arm the pilot runs — the scanned update is **bitwise identical** to
  the unrolled one at every cell measured (logical 4 and 8; 1, 2, 4 and 8 microbatches), on the
  gradients, the loss, the parameters and the optimizer state.
* For the **one_step** (matched-C0) arm it is bitwise identical at 1 microbatch and at microbatch
  width 1, and departs on **exactly one leaf** — ``pre_context_head.norm_features.layer_norm.scale``,
  a sum-reduction gradient — by at most ``2.4e-07`` absolute (≈2 float32 eps relative to that leaf)
  in the three cells with microbatch width ≥ 2 and ≥ 2 microbatches.
* **That departure is not the accumulation**, and this file proves it twice rather than arguing it:
  with the accumulation *removed entirely* (the scan body's gradient emitted per iteration through
  ``ys``, nothing summed anywhere) the same single leaf already departs by the same amount; and with
  two **byte-identical** microbatches — where both implementations compute exactly ``g + g`` — it
  departs identically. What is left is XLA choosing a different, equally valid reduction schedule for
  the *unchanged* gradient block when that block is a scan body rather than inlined N times.

The departure is pinned by :func:`test_the_only_departure_from_bitwise_is_one_layernorm_scale_leaf`
so that it cannot grow, spread to another leaf, or reach the loss without a test failing. **Pinning it
is not the same as accepting it** — it is a MEASURED DEVIATION reported to the Planner for
adjudication, and this docstring is where the measurement lives.

**Eager execution is not the contract and is not measured here.** Production compiles this update
(``build_training_program``: ``compiled = jax.jit(build_logical_update(loss_fn, optimizer, context))``),
so every comparison below is jitted. Run op-by-op instead, the unrolled reference differs from *any*
staged form — jitted-unrolled and eager-scan included — on all 39 leaves at ~3e-7 relative, because
eager dispatch and XLA fusion are different evaluators of the same arithmetic. Comparing an eager
reference against a staged one would have measured that, not this round.
"""

from __future__ import annotations

import ast
import functools
import inspect
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from maxdiffusion import pos_rollout_update as update_module
from maxdiffusion.pos_rollout_arms import ArmContext, build_arm
from maxdiffusion.pos_rollout_stream import draw_step_for_batch
from maxdiffusion.pos_rollout_update import build_logical_update, draws_from_arrays, draws_to_arrays
from maxdiffusion.tests.worklogs_yixun.test_pos_rollout_step import _grid, _mesh_context, _tiny_cfg_stack

_UPDATE_PATH = Path(update_module.__file__).resolve()

# The tiny real-architecture geometry T3a characterised, reused verbatim (`_tiny_cfg_stack`).
_C, _F, _H, _W, _TEXT = 4, 2, 4, 6, 32
_STEPS, _K, _GUIDE, _NULL_LEN, _ACTION_LEN, _ACTION_DIM = 25, 2, 5.0, 7, 4, 7
_SEED, _GLOBAL_STEP = 1234, 11

#: The ONE leaf on which the scanned and unrolled updates are measured to disagree, and the bound the
#: disagreement is held to. Measured maximum over the three affected cells: 2.384e-07 absolute, on a
#: leaf whose own gradient reaches ~0.96-1.4 — about two float32 eps. The bound is 4x the worst
#: measurement, tight enough that a real numerical change cannot hide under it.
_KNOWN_DEPARTURE_LEAF = "['pre_context_head']['norm_features']['layer_norm']['scale'].value"
_KNOWN_DEPARTURE_BOUND = 1e-6

#: Cells where the two implementations are measured BITWISE equal. ``(arm, logical_batch, microbatch)``.
_BITWISE_CELLS = (
    ("rollout", 4, 4),
    ("rollout", 4, 2),
    ("rollout", 4, 1),
    ("rollout", 8, 2),
    ("one_step", 4, 4),
    ("one_step", 4, 1),
)
#: Cells where the one_step arm shows the single-leaf departure characterised above.
_DEPARTING_CELLS = (("one_step", 4, 2), ("one_step", 8, 4), ("one_step", 8, 2))


def _requires_backend():
    pytest.importorskip("torch")
    pytest.importorskip("aqt")


# =============================================================================================
# The reference: the accumulation as it stood at HEAD 4dfbc1b, copied verbatim.
#
# The old implementation no longer exists in the tree, so parity has to be measured against a copy.
# `test_the_production_builder_no_longer_unrolls_the_accumulation` binds the copy to reality from the
# other side: it asserts production is NOT this any more, so the two things being compared cannot
# quietly become one thing.
# =============================================================================================


def _unrolled_logical_update(loss_fn, optimizer, context):
    """``build_logical_update`` as of commit 4dfbc1b — the Python-unrolled accumulation."""

    def update(params, frozen_state, opt_state, micro_batches, micro_draws):
        grads = None
        total = 0.0
        for batch, values in zip(micro_batches, micro_draws):
            (loss, _), grad = jax.value_and_grad(loss_fn, has_aux=True)(
                params, batch, context, frozen_state=frozen_state, draws=draws_from_arrays(values)
            )
            grads = grad if grads is None else jax.tree.map(lambda a, b: a + b, grads, grad)
            total = total + loss
        count = len(micro_batches)
        grads = jax.tree.map(lambda leaf: leaf / count, grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, total / count

    return update


def _grad_capturing_optimizer():
    """An optimizer whose OUTPUT STATE is the accumulated mean gradient, bitwise.

    The gradients are what parity is about, and they are internal to the update. Reading them back
    out of ``new_params`` would put an ``apply_updates`` addition between the measurement and the
    quantity; this transformation applies no arithmetic to them at all, so ``opt_state`` out **is**
    the tree the update computed. Both implementations call ``optimizer.update`` at the same point,
    so both are instrumented identically.
    """

    def init(params):
        return jax.tree.map(jnp.zeros_like, params)

    def update(grads, state, params=None):
        del state, params
        return jax.tree.map(jnp.zeros_like, grads), grads

    return optax.GradientTransformation(init, update)


def _production_shaped_optimizer():
    """Clipping + AdamW, the shape ``build_optimizer`` produces — clipping in particular COUPLES the
    leaves through the global norm, so a one-leaf gradient difference is not a one-leaf parameter
    difference. Used where the question is what reaches the trained parameters."""
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-4))


class _Scope:
    """``_mesh_context()`` yields single-use context managers; take a fresh pair per ``with``."""

    def __enter__(self):
        rules, mesh = _mesh_context()
        self._entered = [rules.__enter__(), mesh.__enter__()]
        self._managers = (rules, mesh)
        return self

    def __exit__(self, *exc):
        for manager in reversed(self._managers):
            manager.__exit__(*exc)
        return False


def _batch(count, seed=3):
    keys = jax.random.split(jax.random.key(seed), 3)
    return {
        "z_video": jax.random.normal(keys[0], (count, _C, _F, _H, _W), jnp.float32),
        "z_i0": jax.random.normal(keys[1], (count, _C, 1, _H, _W), jnp.float32),
        "actions": jax.random.normal(keys[2], (count, _ACTION_LEN, _ACTION_DIM), jnp.float32),
    }


def _context(k_b=_K):
    sigmas, timesteps = _grid()
    return ArmContext(
        sigmas=sigmas,
        timesteps=timesteps,
        null_context=jax.random.normal(jax.random.key(9), (1, _NULL_LEN, _TEXT), jnp.float32),
        guide_scale=_GUIDE,
        weights_dtype=jnp.float32,
        num_train_timesteps=1000,
        num_steps=_STEPS,
        k_b=k_b,
    )


@functools.lru_cache(maxsize=8)
def _cell(arm, logical=4, micro=2, global_step=_GLOBAL_STEP, k_b=_K):
    """One measured cell: the real arm, the real stream draws, split the way production splits them.

    The draws come from ``draw_step_for_batch`` rather than being invented here, because contract 3
    is about the pairing between a microbatch and ITS draw and a hand-built draw would not have one.
    """
    transformer, adapters = _tiny_cfg_stack()
    loss_fn, params, frozen = build_arm(arm, transformer, adapters)
    _, micro_draws, micro_batches = draw_step_for_batch(
        _batch(logical),
        seed=_SEED,
        global_step=global_step,
        logical_batch=logical,
        microbatch=micro,
        num_steps=_STEPS,
        k_b=k_b,
    )
    return (
        loss_fn,
        params,
        frozen,
        _context(k_b),
        tuple(micro_batches),
        tuple(draws_to_arrays(draws) for draws in micro_draws),
    )


def _departures(got, want):
    """``(leaf name, max |difference|)`` for every leaf that is not BITWISE equal."""
    named = jax.tree_util.tree_flatten_with_path(got)[0]
    out = []
    for (path, left), right in zip(named, jax.tree.leaves(want)):
        left, right = np.asarray(left), np.asarray(right)
        if not np.array_equal(left, right):
            gap = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
            out.append((jax.tree_util.keystr(path), gap))
    return out


def _run_both(arm, logical, micro, optimizer_factory=_grad_capturing_optimizer, k_b=_K, draws=None):
    """The scanned and the unrolled update, jitted, over identical inputs. Returns both triples."""
    loss_fn, params, frozen, context, micro_batches, micro_draws = _cell(arm, logical, micro, k_b=k_b)
    micro_draws = micro_draws if draws is None else draws
    optimizer = optimizer_factory()
    opt_state = optimizer.init(params)
    with _Scope():
        scanned = jax.jit(build_logical_update(loss_fn, optimizer, context))(
            params, frozen.state, opt_state, micro_batches, micro_draws
        )
        unrolled = jax.jit(_unrolled_logical_update(loss_fn, optimizer, context))(
            params, frozen.state, opt_state, micro_batches, micro_draws
        )
    return scanned, unrolled


def _total_eqns(jaxpr) -> int:
    """Every equation in the program, sub-jaxprs included and each counted ONCE.

    The scan body appears once however many times the loop runs, which is exactly the property under
    test: the compiler's work is the body, not the trip count. Counting only the top level would
    flatter the scan (its body would vanish); counting the body per iteration would be a fiction.
    """
    total = 0
    for equation in jaxpr.eqns:
        total += 1
        for value in equation.params.values():
            leaves = jax.tree_util.tree_leaves(
                value, is_leaf=lambda node: hasattr(node, "eqns") or hasattr(node, "jaxpr")
            )
            for leaf in leaves:
                inner = getattr(leaf, "jaxpr", leaf)
                if hasattr(inner, "eqns"):
                    total += _total_eqns(inner)
    return total


# =============================================================================================
# Contract 1 — the accumulation is the SAME accumulation.
# =============================================================================================


@pytest.mark.parametrize("arm,logical,micro", _BITWISE_CELLS)
def test_the_scanned_update_reproduces_the_unrolled_gradient_and_loss_bitwise(arm, logical, micro):
    """``np.array_equal``, not ``allclose``: the accumulation identity is exact or it is not held."""
    _requires_backend()
    scanned, unrolled = _run_both(arm, logical, micro)
    assert _departures(scanned[1], unrolled[1]) == [], f"{arm} logical={logical} micro={micro}"
    assert np.array_equal(np.asarray(scanned[2]), np.asarray(unrolled[2])), "the reported mean loss"


def test_the_pilots_cell_is_bitwise_through_the_optimizer_it_will_actually_run():
    """The pilot's arm at its accumulation, end to end: parameters, optimizer state and loss.

    Gradients are the quantity, but parameters are the consequence, and the production-shaped
    optimizer is where a difference would become one — ``clip_by_global_norm`` divides every leaf by
    a norm computed over all of them, so any single-leaf gradient change reaches the whole tree.
    """
    _requires_backend()
    scanned, unrolled = _run_both("rollout", 8, 2, optimizer_factory=_production_shaped_optimizer)
    assert _departures(scanned[0], unrolled[0]) == [], "parameters after one logical update"
    assert _departures(scanned[1], unrolled[1]) == [], "optimizer state after one logical update"
    assert np.array_equal(np.asarray(scanned[2]), np.asarray(unrolled[2])), "the reported mean loss"


@pytest.mark.parametrize("arm,logical,micro", _DEPARTING_CELLS)
def test_the_only_departure_from_bitwise_is_one_layernorm_scale_leaf(arm, logical, micro):
    """The MEASURED DEVIATION, pinned so it cannot grow, spread or reach the loss unnoticed.

    This is a characterization of a real difference, not a relaxed parity test: every other leaf is
    still required to be bitwise equal, the loss is still required to be bitwise equal, and the one
    leaf that departs is named and bounded. Three separate failures are possible and each means
    something different — the departure SPREAD to another leaf, it GREW past two float32 eps, or it
    VANISHED (a compiler change, in which case the pin is stale and the cell table wants re-measuring,
    not a defect).
    """
    _requires_backend()
    scanned, unrolled = _run_both(arm, logical, micro)
    departures = _departures(scanned[1], unrolled[1])
    names = [name for name, _ in departures]
    assert set(names) <= {_KNOWN_DEPARTURE_LEAF}, f"the departure SPREAD: {departures}"
    assert all(gap <= _KNOWN_DEPARTURE_BOUND for _, gap in departures), f"the departure GREW: {departures}"
    assert np.array_equal(np.asarray(scanned[2]), np.asarray(unrolled[2])), "the loss does NOT depart"
    assert names == [_KNOWN_DEPARTURE_LEAF], (
        f"the measured departure VANISHED at {arm} logical={logical} micro={micro}: the accumulation is now "
        f"bitwise everywhere, which is better than the pin claims — re-measure the cell table and move this "
        f"cell into _BITWISE_CELLS rather than deleting the pin"
    )


def test_the_departure_is_the_compilers_schedule_for_the_block_not_the_accumulation():
    """Two controls, each of which removes the accumulation as a possible cause.

    **Control 1 — no accumulation exists.** The scan body's gradient is emitted per iteration through
    ``ys``; nothing is summed, divided or carried. The same single leaf already departs from the same
    block inlined, by the same amount. An accumulation that never runs cannot have caused it.

    **Control 2 — the accumulation order is identical by construction.** Two BYTE-IDENTICAL
    microbatches make both implementations compute ``g + g`` over the same operands in the same
    order. The departure survives that too.

    What remains is the compiler: a sum-reduction gradient scheduled one way inside a scan body and
    another way inlined, both correct in float32.
    """
    _requires_backend()
    loss_fn, params, frozen, context, micro_batches, micro_draws = _cell("one_step", 4, 2)

    def per_iteration(parameters, frozen_state, batches, draw_arrays):
        stacked_batches = jax.tree.map(lambda *parts: jnp.stack(parts), *batches)
        stacked_draws = jax.tree.map(lambda *parts: jnp.stack(parts), *draw_arrays)

        def body(carry, chunk):
            batch, values = chunk
            (loss, _), grad = jax.value_and_grad(loss_fn, has_aux=True)(
                parameters, batch, context, frozen_state=frozen_state, draws=draws_from_arrays(values)
            )
            return carry, (grad, loss)

        _, ys = jax.lax.scan(body, jnp.zeros((), jnp.float32), (stacked_batches, stacked_draws))
        return ys

    def inlined(parameters, frozen_state, batches, draw_arrays):
        out = []
        for batch, values in zip(batches, draw_arrays):
            (loss, _), grad = jax.value_and_grad(loss_fn, has_aux=True)(
                parameters, batch, context, frozen_state=frozen_state, draws=draws_from_arrays(values)
            )
            out.append((grad, loss))
        return out

    with _Scope():
        scanned_ys = jax.jit(per_iteration)(params, frozen.state, micro_batches, micro_draws)
        inline_blocks = jax.jit(inlined)(params, frozen.state, micro_batches, micro_draws)

    without_accumulation = []
    for index in range(len(micro_batches)):
        one = jax.tree.map(lambda leaf: leaf[index], scanned_ys[0])
        departures = _departures(one, inline_blocks[index][0])
        names = [name for name, _ in departures]
        assert set(names) <= {_KNOWN_DEPARTURE_LEAF}, (index, departures)
        assert all(gap <= _KNOWN_DEPARTURE_BOUND for _, gap in departures), (index, departures)
        assert np.array_equal(np.asarray(scanned_ys[1][index]), np.asarray(inline_blocks[index][1])), index
        without_accumulation.extend(names)
    assert without_accumulation, (
        "with the accumulation removed entirely the block no longer departs at all — the attribution "
        "measured in F4 no longer reproduces and must be re-run before the pin is trusted"
    )

    # Control 2: identical microbatches, so the accumulation is `g + g` on both sides.
    optimizer = _grad_capturing_optimizer()
    opt_state = optimizer.init(params)
    same_batches, same_draws = (micro_batches[0], micro_batches[0]), (micro_draws[0], micro_draws[0])
    with _Scope():
        scanned = jax.jit(build_logical_update(loss_fn, optimizer, context))(
            params, frozen.state, opt_state, same_batches, same_draws
        )
        unrolled = jax.jit(_unrolled_logical_update(loss_fn, optimizer, context))(
            params, frozen.state, opt_state, same_batches, same_draws
        )
    departures = _departures(scanned[1], unrolled[1])
    assert [name for name, _ in departures] == [_KNOWN_DEPARTURE_LEAF], departures
    assert all(gap <= _KNOWN_DEPARTURE_BOUND for _, gap in departures), departures


# =============================================================================================
# Contract 3 — every microbatch still meets ITS OWN draw, in order.
#
# There is no rng key inside this update and F4 did not add one: the stream layer draws once per
# OPTIMIZER STEP from `(seed, loop global step)` and hands the update pre-drawn per-microbatch views
# (`pos_rollout_stream._split_draws`). So the folding is untouched by construction, and what F4 could
# actually have broken is the PAIRING — stacking four draws and slicing them back inside a scan is a
# new opportunity to hand microbatch i the draw of microbatch j, silently and without any shape error.
# =============================================================================================


def test_stacking_the_microbatch_draws_preserves_every_fields_dtype_shape_and_order():
    """The mechanism, checked directly — including dtype, which nothing else would notice.

    ``support_start``/``support_end`` are int32 SCALARS and ``t_idx`` is an int32 vector; a stack that
    promoted any of them would change the sigma index arithmetic downstream while every shape still
    lined up and every test still passed.
    """
    _requires_backend()
    *_, micro_draws = _cell("one_step", 8, 2)
    stacked = jax.tree.map(lambda *parts: jnp.stack(parts), *micro_draws)
    for index, field in enumerate(update_module.DRAW_FIELDS):
        part = micro_draws[0][index]
        assert stacked[index].dtype == part.dtype, f"{field} was promoted by stacking"
        assert stacked[index].shape == (len(micro_draws), *part.shape), field
        for position in range(len(micro_draws)):
            assert np.array_equal(
                np.asarray(stacked[index][position]), np.asarray(micro_draws[position][index])
            ), f"{field} lost its order at position {position}"


def test_each_microbatch_meets_its_own_draw_and_the_check_has_teeth():
    """The pairing, through the production update, plus proof the comparison could see a break.

    A permutation of the pairs would be invisible to a sum, so the control is a REVERSAL of the draws
    against a fixed batch order: it re-pairs every microbatch with someone else's noise and sigma
    index, and must move the answer. If it did not, the parity assertions above would be measuring
    nothing.
    """
    _requires_backend()
    scanned, unrolled = _run_both("one_step", 8, 2)
    _, _, _, _, _, micro_draws = _cell("one_step", 8, 2)
    mispaired, _ = _run_both("one_step", 8, 2, draws=tuple(reversed(micro_draws)))

    moved = _departures(scanned[1], mispaired[1])
    assert len(moved) == len(jax.tree.leaves(scanned[1])), "re-pairing must move EVERY gradient leaf"
    assert not np.array_equal(np.asarray(scanned[2]), np.asarray(mispaired[2])), "and the loss"
    # ...and with the pairing intact, the two implementations still agree to the pinned departure.
    intact = _departures(scanned[1], unrolled[1])
    assert {name for name, _ in intact} <= {_KNOWN_DEPARTURE_LEAF}, intact
    assert all(gap <= _KNOWN_DEPARTURE_BOUND for _, gap in intact), intact


@pytest.mark.parametrize("global_step", [100, 101])
def test_the_stream_draws_reach_the_scan_unchanged_across_a_resume_boundary(global_step):
    """Two adjacent LOOP steps — the boundary a restart lands on — draw differently and still pair.

    The draws are keyed on the loop's global step, so consecutive steps get different supports and
    different noise; the point is that the scanned update consumes each step's draws exactly as the
    unrolled loop did, at both sides of a boundary, rather than (say) capturing one step's stacked
    draws into the compiled program.
    """
    _requires_backend()
    loss_fn, params, frozen, context, micro_batches, micro_draws = _cell("rollout", 8, 2, global_step=global_step)
    optimizer = _grad_capturing_optimizer()
    opt_state = optimizer.init(params)
    with _Scope():
        scanned = jax.jit(build_logical_update(loss_fn, optimizer, context))(
            params, frozen.state, opt_state, micro_batches, micro_draws
        )
        unrolled = jax.jit(_unrolled_logical_update(loss_fn, optimizer, context))(
            params, frozen.state, opt_state, micro_batches, micro_draws
        )
    assert _departures(scanned[1], unrolled[1]) == [], global_step
    assert np.array_equal(np.asarray(scanned[2]), np.asarray(unrolled[2]))

    # The two steps really are different draws — otherwise this test would pass on a frozen stream.
    other = _cell("rollout", 8, 2, global_step=201 - global_step)[5]
    assert not np.array_equal(np.asarray(micro_draws[0][2]), np.asarray(other[0][2])), "epsilon must move"


# =============================================================================================
# Contract 4 — the fingerprint is a property of the RECIPE, and this was not a recipe change.
# =============================================================================================


def test_the_recipe_fingerprint_is_untouched_by_the_accumulation_rewrite():
    """Swapping the accumulation must not move what M1's authorization is keyed on.

    Measured out of band as well, by rebuilding the checkout at 4dfbc1b and recomputing: both
    builders give ``5492b40236ba0801f9055673d599e60e8cdd23edfc3b82db30cdab0d7bc27134`` over 177
    recipe keys. In-test the same claim is made structurally — the fingerprint reads config keys and
    nothing else, so installing the old builder on the module leaves the payload byte-identical.
    """
    import yaml

    from maxdiffusion import pos_rollout_fit_probe as probe

    values = yaml.safe_load((_UPDATE_PATH.parent / "configs" / "base_wan_5b_pos_rollout.yml").read_text())

    class _Config:
        def __init__(self, mapping):
            self.__dict__.update(mapping)

        def get_keys(self):
            return dict(self.__dict__)

    config = _Config(values)
    before_payload, before = probe.config_recipe(config), probe.recipe_fingerprint(config)

    original = update_module.build_logical_update
    try:
        update_module.build_logical_update = _unrolled_logical_update
        assert probe.config_recipe(config) == before_payload, "the recipe payload moved"
        assert probe.recipe_fingerprint(config) == before, "the fingerprint moved"
    finally:
        update_module.build_logical_update = original
    assert probe.recipe_fingerprint(config) == before

    # And the accumulation is not a fingerprint INPUT in the first place: no excluded-key reasoning,
    # no module import, just config keys.
    source = inspect.getsource(probe.recipe_fingerprint) + inspect.getsource(probe.config_recipe)
    assert "pos_rollout_update" not in source and "build_logical_update" not in source


# =============================================================================================
# Contract 5 — the rollout loss rematerializes INSIDE the scan, and the trace stays cheap.
# =============================================================================================


@pytest.mark.parametrize("micro,k_b", [(2, _K), (1, _K), (2, 4)])
def test_the_rematted_rollout_loss_still_stages_under_the_scan(micro, k_b):
    """R-B's kernel is ``lax.scan`` + ``jax.remat``; F4 puts a second scan around it.

    Nested staging is where a trace goes quadratic or a remat silently vanishes, so the check is
    threefold: the rematerialization primitive is still in the program, the trace/lower/compile
    complete in seconds rather than minutes, and the frozen backbone is still an ARGUMENT — a scan
    hoists closed-over tracers into its constants, and if F3's threading had degraded there, the
    weights would be back in ``jaxpr.consts`` where they killed three M1 attempts.
    """
    _requires_backend()
    loss_fn, params, frozen, context, micro_batches, micro_draws = _cell("rollout", 8, micro, k_b=k_b)
    optimizer = _production_shaped_optimizer()
    opt_state = optimizer.init(params)
    with _Scope():
        started = time.monotonic()
        jaxpr = jax.make_jaxpr(build_logical_update(loss_fn, optimizer, context))(
            params, frozen.state, opt_state, micro_batches, micro_draws
        )
        traced = time.monotonic() - started
        compiled = (
            jax.jit(build_logical_update(loss_fn, optimizer, context))
            .lower(params, frozen.state, opt_state, micro_batches, micro_draws)
            .compile()
        )
        elapsed = time.monotonic() - started

    text = str(jaxpr)
    assert "remat" in text, "the rollout unroll is no longer rematerialized under the scan"
    assert "scan" in text
    assert compiled is not None
    # Generous by an order of magnitude against the measurement (trace ~0.3 s, through compile ~1.5 s
    # on this fixture): the guard is against a pathological blow-up, not a timing regression.
    assert traced < 30.0, f"tracing took {traced:.1f}s"
    assert elapsed < 120.0, f"trace through compile took {elapsed:.1f}s"
    assert update_module.__name__ and sum(getattr(const, "nbytes", 0) for const in jaxpr.consts) < 1_000_000


# =============================================================================================
# The permanent guard — the defect that killed four TPU hosts, as a test.
# =============================================================================================


def test_the_update_graph_stays_flat_as_microbatches_grow():
    """O(1) in the microbatch count, with the O(N) form re-derived in-test as the red side.

    The historical measurement, recorded in the round's worklog before this guard existed, is the
    unrolled builder at **4,813 / 8,472 / 15,790** equations for 1 / 2 / 4 microbatches — ~3,660 per
    microbatch, so the pilot's 32 was a ~118,000-equation program. Rather than cite that, the guard
    reproduces the growth here: the HEAD builder is measured alongside production on the same inputs,
    so the red side is evidence in the same run as the green side and cannot go stale.
    """
    _requires_backend()
    scanned_sizes, unrolled_sizes = {}, {}
    for micro, count in ((8, 1), (2, 4), (1, 8)):
        loss_fn, params, frozen, context, micro_batches, micro_draws = _cell("rollout", 8, micro)
        optimizer = _production_shaped_optimizer()
        opt_state = optimizer.init(params)
        arguments = (params, frozen.state, opt_state, micro_batches, micro_draws)
        with _Scope():
            scanned_sizes[count] = _total_eqns(
                jax.make_jaxpr(build_logical_update(loss_fn, optimizer, context))(*arguments).jaxpr
            )
            unrolled_sizes[count] = _total_eqns(
                jax.make_jaxpr(_unrolled_logical_update(loss_fn, optimizer, context))(*arguments).jaxpr
            )

    biggest, smallest = max(scanned_sizes.values()), min(scanned_sizes.values())
    assert biggest <= 1.10 * smallest, f"the scanned update grew with the microbatch count: {scanned_sizes}"

    # RED: the form this replaced. Eight microbatches must cost multiples of one, or the fixture is
    # too small to exhibit the defect and the green side above proves nothing.
    assert unrolled_sizes[8] > 3 * unrolled_sizes[1], f"the unrolled reference did not grow: {unrolled_sizes}"
    assert unrolled_sizes[8] > 3 * biggest, f"scanned {scanned_sizes} vs unrolled {unrolled_sizes}"


def test_the_production_builder_no_longer_unrolls_the_accumulation():
    """The direction, pinned. Everything above compares production against a COPY of what production
    used to be; if production reverted, the comparison would silently become a self-comparison and
    every parity assertion would pass while the graph defect was back."""
    source = inspect.getsource(update_module.build_logical_update)
    assert "lax.scan" in source, "the accumulation must stay a scan"
    body = ast.parse(inspect.getsource(_unrolled_logical_update))
    assert any(isinstance(node, ast.For) for node in ast.walk(body)), "the reference must be the LOOP form"
    assert not any(
        isinstance(node, ast.For)
        for node in ast.walk(ast.parse(inspect.getsource(update_module)))
        if "micro_batches" in ast.unparse(node)
    ), "production accumulates with a Python `for` again"
