"""exp_06 `rollout_adapter` — T3b-1 `step-stream`: the per-step training stream (plan §3b).

The contract under test: **every random quantity an optimizer step consumes is a pure function of
``(seed, LOOP global step)``** — the same for both arms, unchanged by microbatching, reproducible
after a restart. exp_06's causal claim is R-B against matched-C0 at identical data and identical
randomness, so any way an arm or an accumulation factor could move a draw is a way the comparison
stops being paired.

Three things are proven here that could not be proven anywhere else as cleanly:

1. **The epsilon purpose is additive.** Adding ``"rollout_epsilon"`` to T1's declared purposes must
   leave every draw ever made from the existing four bit-identical. That is checked against an
   INDEPENDENT re-derivation (sha256 of the name, the fold order re-implemented here), not against
   the module's own arithmetic — otherwise a change to the derivation would move both sides
   together. T1's whole suite is the second, integrated check: it still passes untouched.
2. **The restored-``state.step`` hazard is illustrated here, and the obligation is NOT discharged.**
   This module cannot make the mistake (it has no access to a state object, pinned below) and the
   demonstration below shows what the wrong number costs. But that demonstration exercises no state
   object, no loader and no restore path, and its "replays the opening stream" equality is
   arithmetically tautological given step-keying. **The real obligation — a production-callsite AST
   pin plus an interrupted-vs-uninterrupted execution through the actual restore path — belongs to
   T3b-4's integrated oracle and remains OPEN** (T3b-1 review, MAJOR 3). Nothing in this file should
   be read as closing it; S7's failure class stays live at loop level.
3. **Accumulation cannot move a draw**, because a draw never sees a microbatch — proven both
   structurally (no microbatch index reaches the derivation) and by value across factors.

Applying the standing method note from T3a — *isolate the term rather than lowering the bar* — the
oracles here compare the DRAWS themselves rather than end-to-end training outcomes: at this layer
the disputed quantities are available exactly, so nothing needs to be inferred from a noisy proxy.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion import pos_rollout_stream as stream
from maxdiffusion import pos_rollout_support as support

_MODULE_PATH = Path(stream.__file__).resolve()
_SUPPORT_PATH = Path(support.__file__).resolve()

# The four purposes that existed before T3b-1, with the ids they had. Hard-coded rather than read
# from the module: this is the "before" side of the additivity claim and must not move with it.
_PRE_EXISTING_PURPOSES = ("p_ss_coin", "k_a_draw", "index_support", "index_support_rollout")
_STEPS, _K = 25, 2
_SHAPE = (2, 4, 3, 4, 6)


def _independent_aux_key(seed, global_step, purpose):
    """T1's derivation, re-implemented from the written contract rather than imported."""
    key = jax.random.key(int(seed) + 1_000_003)
    key = jax.random.fold_in(key, jnp.asarray(global_step, dtype=jnp.uint32))
    return jax.random.fold_in(key, int.from_bytes(hashlib.sha256(purpose.encode("utf-8")).digest()[:4], "big"))


def _key_bits(key):
    return np.asarray(jax.random.key_data(key))


def _freshly_loaded(module):
    # Registered in sys.modules before execution: `dataclasses` resolves a field annotation via
    # `sys.modules[cls.__module__]`, so a module defining a dataclass cannot be re-executed without
    # it. Harmless here today, load-bearing the moment this module grew `StepDraws`.
    name = f"_restarted_{module.__name__.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(name, Path(module.__file__))
    fresh = importlib.util.module_from_spec(spec)
    sys.modules[name] = fresh
    try:
        spec.loader.exec_module(fresh)
    finally:
        sys.modules.pop(name, None)
    return fresh


# =============================================================================================
# 1. The epsilon purpose is ADDITIVE.
# =============================================================================================


def test_the_new_purpose_is_declared_and_distinct():
    assert stream.POS_ROLLOUT_EPSILON_PURPOSE == "rollout_epsilon"
    assert stream.POS_ROLLOUT_EPSILON_PURPOSE in support.EXP03_AUX_PURPOSES
    ids = {purpose: support._purpose_id(purpose) for purpose in support.EXP03_AUX_PURPOSES}
    assert len(set(ids.values())) == len(ids), "a purpose id collided"
    assert tuple(support.EXP03_AUX_PURPOSES)[: len(_PRE_EXISTING_PURPOSES)] == _PRE_EXISTING_PURPOSES


@pytest.mark.parametrize("purpose", _PRE_EXISTING_PURPOSES)
@pytest.mark.parametrize("global_step", [0, 1, 7, 250, 12_499])
def test_adding_the_purpose_left_every_existing_draw_bit_identical(purpose, global_step):
    # The additivity claim, against an INDEPENDENT re-derivation of T1's arithmetic. If appending a
    # name had renumbered anything -- the failure mode name-hashing exists to prevent -- these keys
    # would move, and with them every support draw the campaign has made.
    got = support.exp03_aux_key(seed=0, global_step=global_step, purpose=purpose)
    assert np.array_equal(_key_bits(got), _key_bits(_independent_aux_key(0, global_step, purpose)))


def test_the_epsilon_purpose_id_is_the_hash_of_its_name():
    expected = int.from_bytes(hashlib.sha256(b"rollout_epsilon").digest()[:4], "big")
    assert support._purpose_id(stream.POS_ROLLOUT_EPSILON_PURPOSE) == expected


def test_the_inherited_undeclared_purpose_tripwire_still_bites():
    # T1's suite asserts that `"epsilon"` raises. Taking that name for the new purpose would have
    # defanged an inherited tripwire silently, which is why the new one is `rollout_epsilon`.
    assert "epsilon" not in support.EXP03_AUX_PURPOSES
    with pytest.raises(ValueError):
        support.exp03_aux_key(seed=0, global_step=0, purpose="epsilon")


def test_the_epsilon_draw_is_independent_of_the_support_draw():
    # Different purposes, so independence is structural rather than lucky: the support and the noise
    # of one step must not be derivable from each other.
    for global_step in (0, 5, 91):
        support_key = support.exp03_aux_key(seed=0, global_step=global_step, purpose="index_support_rollout")
        epsilon_key = support.exp03_aux_key(seed=0, global_step=global_step, purpose="rollout_epsilon")
        assert not np.array_equal(_key_bits(support_key), _key_bits(epsilon_key))


# =============================================================================================
# 2. The step stream: keyed on the LOOP step, fresh per step, arm-blind.
# =============================================================================================


def test_the_step_stream_is_a_pure_function_of_seed_and_loop_step():
    first = stream._draw_step_stream(seed=0, global_step=17, num_steps=_STEPS, k_b=_K, shape=_SHAPE)
    again = stream._draw_step_stream(seed=0, global_step=17, num_steps=_STEPS, k_b=_K, shape=_SHAPE)
    assert int(first.support_start) == int(again.support_start)
    assert int(first.support_end) == int(again.support_end)
    assert np.array_equal(np.asarray(first.epsilon), np.asarray(again.epsilon))
    # ...and it is genuinely fresh per step and per seed.
    other_step = stream._draw_step_stream(seed=0, global_step=18, num_steps=_STEPS, k_b=_K, shape=_SHAPE)
    other_seed = stream._draw_step_stream(seed=1, global_step=17, num_steps=_STEPS, k_b=_K, shape=_SHAPE)
    assert not np.array_equal(np.asarray(first.epsilon), np.asarray(other_step.epsilon))
    assert not np.array_equal(np.asarray(first.epsilon), np.asarray(other_seed.epsilon))


def test_the_step_stream_agrees_with_the_primitives_it_composes():
    # The pairing must be the T1 support and the T3b epsilon of the SAME step -- not a support from
    # one call and a noise from another.
    draws = stream._draw_step_stream(seed=3, global_step=41, num_steps=_STEPS, k_b=_K, shape=_SHAPE)
    start, end = support.rollout_support(seed=3, global_step=41, num_steps=_STEPS, k_b=_K)
    assert (int(draws.support_start), int(draws.support_end)) == (int(start), int(end))
    assert np.array_equal(
        np.asarray(draws.epsilon),
        np.asarray(stream._rollout_epsilon(seed=3, global_step=41, shape=_SHAPE)),
    )


def test_the_epsilon_has_the_requested_shape_and_dtype():
    for dtype in (jnp.float32, jnp.bfloat16):
        epsilon = stream._rollout_epsilon(seed=0, global_step=2, shape=_SHAPE, dtype=dtype)
        assert epsilon.shape == _SHAPE
        assert epsilon.dtype == dtype


def test_the_stream_module_has_no_notion_of_an_arm():
    """Arm-independence, enforced by absence: a draw cannot depend on what the code cannot see.

    "Arm-independent" means no draw VARIES with the arm -- not that the stream is forbidden from
    drawing something only one arm reads. Both arms are handed the same ``StepDraws``; each takes
    the fields it needs.

    This is the layer at which the R-B / matched-C0 pairing is guaranteed. The end-to-end
    arm-switch test belongs to the sub-round that owns both losses; here the guarantee is that
    nothing in the stream could express an arm-dependent draw in the first place.
    """
    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = {argument.arg for argument in node.args.args + node.args.kwonlyargs}
            assert not names & {"arm", "objective", "loss", "loss_fn", "arm_name"}, node.name
        # No branching on an arm-shaped name either.
        if isinstance(node, ast.Name):
            assert node.id not in {"arm", "ARMS", "objective"}, ast.unparse(node)
    # Neither arm's loss may be imported or called here. NOTE the deliberate absence of a ban on
    # the substring "one_step": the stream draws a sigma index that only matched-C0 consumes, and
    # naming a field after its consumer is not arm-DEPENDENCE. The property is that the stream
    # produces the same values whoever reads them -- checked by value in the arms' suite -- and that
    # nothing here branches on an arm, checked structurally above.
    # Structural, over imports AND calls, naming the arm-side symbols exactly (review MINOR i):
    # a substring ban was both too weak (it missed `rollout_arm_loss`) and too strong (it tripped on
    # the `one_step_index` purpose name, which is not arm-dependence).
    arm_symbols = {
        "rollout_endpoint_loss",
        "_denoising_loss",
        "rollout_arm_loss",
        "one_step_denoising_loss",
        "build_arm",
        "ARMS",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "pos_rollout_arms" not in (node.module or ""), "the stream must not import an arm"
            assert not {alias.name for alias in node.names} & arm_symbols, ast.unparse(node)
        if isinstance(node, ast.Import):
            assert not any("pos_rollout_arms" in alias.name for alias in node.names), ast.unparse(node)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in arm_symbols:
            raise AssertionError(f"the stream must not call an arm: {ast.unparse(node)}")


def test_the_stream_cannot_read_a_state_object():
    """The restored-``state.step`` hazard cannot originate here (T1 reviewer's obligation).

    The derivation takes ``global_step`` as an explicit argument; no function in this module accepts
    a state, and nothing reads a ``.step`` attribute. The caller can still pass the wrong number --
    which is why the next test shows what that costs.
    """
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = {argument.arg for argument in node.args.args + node.args.kwonlyargs}
            assert not names & {"state", "train_state", "opt_state"}, node.name
        if isinstance(node, ast.Attribute):
            assert node.attr != "step", f"a state's step must never be read here: {ast.unparse(node)}"
    for name in ("_draw_step_stream", "_rollout_epsilon", "draw_step_for_batch"):
        assert "global_step" in inspect.signature(getattr(stream, name)).parameters


def test_keying_on_a_restored_step_instead_of_the_loop_step_replays_the_wrong_randomness():
    """The hazard ILLUSTRATED at primitive level -- this does NOT discharge the obligation.

    What follows is arithmetic on this module's pure function: it exercises no state object, no
    loader and no restore, so it cannot catch a loop that passes the wrong number. The binding
    proof is T3b-4's (a production-callsite pin plus interrupted-vs-uninterrupted through the real
    restore path). Kept because it makes the cost concrete and cheap to re-read.

    A run preempted at 4,000 and resumed restores params/opt_state/step; a freshly built state's
    ``step`` counts from 0 again. Keying the draw on that instead of the loop's position makes the
    resumed segment re-consume steps 0..N's randomness while the dataloader serves step 4,000+'s
    data -- silently, and identically in both arms, so no comparison catches it.
    """
    loop_steps = [4000, 4001, 4002]
    restored_steps = [0, 1, 2]  # what a freshly built state would report after a restore
    correct = [
        stream._draw_step_stream(seed=0, global_step=s, num_steps=_STEPS, k_b=_K, shape=_SHAPE) for s in loop_steps
    ]
    wrong = [
        stream._draw_step_stream(seed=0, global_step=s, num_steps=_STEPS, k_b=_K, shape=_SHAPE) for s in restored_steps
    ]
    for right, bad in zip(correct, wrong):
        assert not np.array_equal(np.asarray(right.epsilon), np.asarray(bad.epsilon))
    # ...and the wrong stream is exactly the run's own opening segment, replayed:
    opening = [
        stream._draw_step_stream(seed=0, global_step=s, num_steps=_STEPS, k_b=_K, shape=_SHAPE) for s in (0, 1, 2)
    ]
    for bad, first in zip(wrong, opening):
        assert np.array_equal(np.asarray(bad.epsilon), np.asarray(first.epsilon))


@pytest.mark.parametrize("prefix", ["cold", "walk", "resume"])
def test_the_stream_survives_a_restart_that_reached_the_step_differently(prefix):
    # T1's fresh-namespace technique: module-level state (a counter, a memoized first step) is stable
    # within a process and silently wrong across the restart a preempted TPU job performs.
    restarted = _freshly_loaded(stream)
    walk = {"cold": (), "walk": range(400), "resume": range(300, 400)}[prefix]
    for earlier in walk:
        restarted._draw_step_stream(seed=0, global_step=earlier, num_steps=_STEPS, k_b=_K, shape=_SHAPE)
    got = restarted._draw_step_stream(seed=0, global_step=400, num_steps=_STEPS, k_b=_K, shape=_SHAPE)
    want = stream._draw_step_stream(seed=0, global_step=400, num_steps=_STEPS, k_b=_K, shape=_SHAPE)
    assert np.array_equal(np.asarray(got.epsilon), np.asarray(want.epsilon))
    assert (int(got.support_start), int(got.support_end)) == (int(want.support_start), int(want.support_end))


def test_the_draws_are_tracer_safe():
    jitted = jax.jit(
        lambda value: stream._draw_step_stream(
            seed=0, global_step=value, num_steps=_STEPS, k_b=_K, shape=_SHAPE
        ).epsilon
    )
    for global_step in (0, 250, 9999):
        eager = stream._draw_step_stream(seed=0, global_step=global_step, num_steps=_STEPS, k_b=_K, shape=_SHAPE)
        assert np.array_equal(np.asarray(eager.epsilon), np.asarray(jitted(jnp.asarray(global_step, jnp.int32))))


# =============================================================================================
# 3. Accumulation cannot move a draw, and cannot change the logical batch.
# =============================================================================================


@pytest.mark.parametrize("microbatch", [8, 4, 2, 1])
def test_every_accumulation_factor_reconstructs_the_same_logical_draw(microbatch):
    """The accumulation-invariance proof, and it can actually FAIL.

    An earlier version of this test deleted the accumulation factor and repeated an identical call,
    so it asserted nothing (T3b-1 review, MAJOR 2). The real hazard the reviewer demonstrated is that
    drawing once PER MICROBATCH and concatenating gives an epsilon exactly unequal to the factor-1
    draw -- accumulation would then silently change the objective. So the oracle reconstructs: draw
    once at the logical width through the orchestration seam, split, concatenate the parts, and
    require EXACT equality with the single logical draw, at every factor a run might use.
    """
    logical = 8
    batch = {"z_video": jnp.zeros((logical, *_SHAPE[1:]), jnp.float32)}
    logical_draws, parts, batch_parts = stream.draw_step_for_batch(
        batch, seed=0, global_step=123, logical_batch=logical, microbatch=microbatch, num_steps=_STEPS, k_b=_K
    )
    assert len(batch_parts) == len(parts)
    assert len(parts) == logical // microbatch
    assert np.array_equal(
        np.asarray(jnp.concatenate([part.epsilon for part in parts], axis=0)), np.asarray(logical_draws.epsilon)
    )
    assert np.array_equal(
        np.asarray(jnp.concatenate([part.t_idx for part in parts], axis=0)), np.asarray(logical_draws.t_idx)
    )
    # The support is per-BATCH: every microbatch shares the one the logical step drew.
    for part in parts:
        assert int(part.support_start) == int(logical_draws.support_start)
        assert int(part.support_end) == int(logical_draws.support_end)
    # ...and the logical draw itself does not depend on the factor at all.
    reference, _, _ = stream.draw_step_for_batch(
        batch, seed=0, global_step=123, logical_batch=logical, microbatch=None, num_steps=_STEPS, k_b=_K
    )
    assert np.array_equal(np.asarray(reference.epsilon), np.asarray(logical_draws.epsilon))
    assert np.array_equal(np.asarray(reference.t_idx), np.asarray(logical_draws.t_idx))
    signature = inspect.signature(stream._draw_step_stream).parameters
    assert not set(signature) & {"accumulation_steps", "microbatch", "microbatch_index", "part"}
    # ...and the raw helper is NOT part of the public surface: the safe seam is (review MAJOR).
    assert "draw_step_stream" not in stream.__all__ and "split_draws" not in stream.__all__
    assert "draw_step_for_batch" in stream.__all__


def test_drawing_per_microbatch_is_exactly_what_the_seam_prevents():
    """The failure mode, exhibited: this is why `draw_logical_step` exists.

    Drawing at the microbatch width and concatenating is the natural-looking thing a loop author
    would write, and it produces a DIFFERENT epsilon from the logical draw -- not approximately, but
    entirely. The seam is the fix; this test is the evidence that it was needed.
    """
    logical, microbatch = 8, 2
    logical_draws, _ = stream._draw_logical_step(
        seed=0,
        global_step=123,
        logical_batch=logical,
        microbatch=microbatch,
        num_steps=_STEPS,
        k_b=_K,
        example_shape=_SHAPE[1:],
    )
    naive = jnp.concatenate(
        [
            stream._draw_step_stream(
                seed=0, global_step=123, num_steps=_STEPS, k_b=_K, shape=(microbatch, *_SHAPE[1:])
            ).epsilon
            for _ in range(logical // microbatch)
        ],
        axis=0,
    )
    assert naive.shape == logical_draws.epsilon.shape
    assert not np.array_equal(np.asarray(naive), np.asarray(logical_draws.epsilon))


def test_the_one_step_index_is_a_stream_draw_at_the_logical_width():
    # matched-C0's sigma index is drawn here, not by the arm (T3b-1 review, BLOCKER 1): left to the
    # loss it would be accumulation- and resume-dependent and would confound the R-B-vs-C0 pairing.
    assert stream.POS_ROLLOUT_TIMESTEP_PURPOSE == "one_step_index"
    assert stream.POS_ROLLOUT_TIMESTEP_PURPOSE in support.EXP03_AUX_PURPOSES
    draws = stream._draw_step_stream(seed=0, global_step=17, num_steps=_STEPS, k_b=_K, shape=_SHAPE)
    assert draws.t_idx.shape == (_SHAPE[0],)
    indices = np.asarray(draws.t_idx)
    assert indices.min() >= 0 and indices.max() < _STEPS
    assert np.array_equal(
        indices,
        np.asarray(stream._one_step_timestep_indices(seed=0, global_step=17, batch_size=_SHAPE[0], num_steps=_STEPS)),
    )
    # Fresh per step, and independent of the other two purposes.
    other = stream._draw_step_stream(seed=0, global_step=18, num_steps=_STEPS, k_b=_K, shape=_SHAPE)
    assert not np.array_equal(indices, np.asarray(other.t_idx))


def test_the_accumulation_plan_preserves_the_logical_batch_or_refuses():
    assert stream.accumulation_plan(256, None) == (256, 1)
    assert stream.accumulation_plan(256, 256) == (256, 1)
    assert stream.accumulation_plan(256, 64) == (64, 4)
    assert stream.accumulation_plan(1, 1) == (1, 1)
    # Each rejection is matched on its OWN message, not merely on ValueError. A misconfiguration is
    # diagnosed from a worker log, so "which rule did I break" is part of the contract -- and
    # without this, the too-large-microbatch guard would be indistinguishable from the non-divisor
    # guard that happens to catch the same inputs (they overlap: size > logical implies
    # logical % size != 0), i.e. an untested and silently removable line.
    for logical, microbatch, message in (
        (256, 96, "does not divide the logical batch"),
        (256, 512, "larger than the logical batch"),
        (256, 0, "microbatch must be positive"),
        (256, -4, "microbatch must be positive"),
        (0, 1, "logical batch must be positive"),
        (-1, 1, "logical batch must be positive"),
    ):
        with pytest.raises(ValueError, match=message):
            stream.accumulation_plan(logical, microbatch)


def test_the_microbatch_windows_are_contiguous_ordered_and_complete():
    windows = stream.microbatch_slices(12, 3)
    assert [(w.start, w.stop) for w in windows] == [(0, 4), (4, 8), (8, 12)]
    assert stream.microbatch_slices(8, 1) == (slice(0, 8),)
    for logical, count in ((12, 5), (12, 0), (0, 1)):
        with pytest.raises(ValueError):
            stream.microbatch_slices(logical, count)


def test_splitting_a_batch_preserves_every_field_and_the_example_order():
    batch = {
        "z_video": jnp.arange(8 * 3, dtype=jnp.float32).reshape(8, 3),
        "actions": jnp.arange(8 * 2, dtype=jnp.float32).reshape(8, 2),
    }
    parts = stream.split_batch(batch, 4)
    assert len(parts) == 4
    assert set(parts[0]) == set(batch), "a field was dropped by the split"
    for name, value in batch.items():
        assert np.array_equal(np.asarray(jnp.concatenate([part[name] for part in parts], axis=0)), np.asarray(value))
    with pytest.raises(ValueError):
        stream.split_batch({"a": jnp.zeros((8, 2)), "b": jnp.zeros((4, 2))}, 2)
    with pytest.raises(ValueError):
        stream.split_batch({}, 1)


def test_a_batch_that_is_not_the_declared_logical_batch_is_refused():
    # S7's BLOCKER 2, generalized: the configuration being expressible is not the iterator obeying it.
    good = {"z_video": jnp.zeros((256, 3)), "actions": jnp.zeros((256, 2))}
    assert stream.checked_logical_batch(good, logical_batch=256, accumulation_steps=4, microbatch=64) is good
    short = {"z_video": jnp.zeros((128, 3)), "actions": jnp.zeros((128, 2))}
    with pytest.raises(ValueError, match="logical batch"):
        stream.checked_logical_batch(short, logical_batch=256, accumulation_steps=4, microbatch=64)
    with pytest.raises(ValueError, match="microbatch width"):
        stream.checked_logical_batch(good, logical_batch=256, accumulation_steps=4, microbatch=32)
    with pytest.raises(ValueError):
        stream.checked_logical_batch({}, logical_batch=256, accumulation_steps=4, microbatch=64)


def test_the_stream_module_is_side_effect_free_and_reads_no_config():
    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for forbidden in ("print(", "max_logging", "jax.debug", "open(", "logging."):
        assert forbidden not in source, forbidden
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert node.value.id not in {"config", "cfg", "scheduler"}, ast.unparse(node)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "getattr":
            assert len(node.args) < 3, f"three-argument getattr is forbidden (issue #11): {ast.unparse(node)}"


def test_the_support_draw_is_still_t1s_primitive():
    # One support construction for the experiment: the stream composes T1's rather than re-deriving.
    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("pos_rollout_support")
        for alias in node.names
    }
    assert {"rollout_support", "exp03_aux_key"} <= imported
    assert stream.rollout_support is support.rollout_support
    assert stream.exp03_aux_key is support.exp03_aux_key
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in {"rollout_support", "exp03_aux_key"}:
            assert isinstance(node.ctx, ast.Load), f"{node.id} is re-bound"
    assert "def rollout_support" not in source
    assert "1_000_003" not in source, "the key derivation belongs to T1, not here"
