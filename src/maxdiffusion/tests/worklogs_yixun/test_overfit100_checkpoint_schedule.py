"""CPU-only tests for the exp_02 overfit100 checkpoint contract (plan D7 / H2).

H2 (Codex plan-review v3, BLOCKER) established that D10's checkpoint LISTS are not
executable through the inherited machinery: ``wan_ti2v_full_ft_trainer`` saves only on one
periodic ``checkpoint_every`` cadence, and ``wan_ti2v_side_adapter_trainer``'s manager
hard-codes ``max_to_keep=3`` -- so ``{250,500,1000,2500}`` / ``{250,500,1000,1750,2500}``
can neither be produced nor retained. The rewritten trainer therefore owns:

  (A) an explicit ``checkpoint_steps`` schedule -- a fake loop over the steps with a
      recording saver must emit EXACTLY the configured set (both the S2 and S3 lists),
      including across a resume, with ``checkpoint_every`` still honored when set > 0 and
      the end-of-run final save de-duplicated against the in-loop emissions;
  (B) its OWN ``CheckpointManager`` with ``max_to_keep=None`` -- every listed checkpoint is
      retained, so segment-final checkpoints are never garbage-collected (the inherited
      ``max_to_keep=3`` would evict step 250 by the time step 2500 lands);
  (C) a save/restore round trip that EXCLUDES ``context_table``: the deterministic 400 MiB
      text table is rebuilt from ``episodes.json`` on every start and must never enter the
      checkpoint (~30 GB/checkpoint is budgeted for params + both Adam moments only), so
      restoring into a state whose table differs must leave that table untouched while
      params / opt_state / step come back from disk.

CPU-only: a tiny ``nnx.Module`` stub transformer with a real ``Param`` plus a real
``optax.adamw``, checkpoints in ``tmp_path``. The darwin grain import stub lives in
``conftest.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml
from flax import nnx

import maxdiffusion.trainers.wan_ti2v_overfit100_trainer as overfit100

_REPO = Path(overfit100.__file__).parents[3]
_CONFIG = _REPO / "src/maxdiffusion/configs/base_wan_5b_overfit100.yml"

_S3_STEPS = [250, 500, 1000, 1750, 2500]
_S2_STEPS = [250, 500, 1000, 2500]
_INIT_GAIN = 0.5


# =======================================================================================
# (A) the emitted step set -- fake loop with a recording saver
# =======================================================================================


def _fake_loop(checkpoint_steps, *, max_train_steps=2500, checkpoint_every=0, save_final=True, start_step=0):
    """Drive the trainer's own scheduler exactly as ``start_training``'s loop does."""
    scheduler = overfit100.CheckpointScheduler(
        checkpoint_steps=checkpoint_steps,
        checkpoint_every=checkpoint_every,
        max_train_steps=max_train_steps,
        save_final=save_final,
    )
    saved: list[int] = []
    for step in range(start_step, max_train_steps):
        if scheduler.should_save(step + 1):
            saved.append(step + 1)
    final = scheduler.final_step()
    if final is not None:
        saved.append(final)
    return saved


@pytest.mark.parametrize("steps", [_S3_STEPS, _S2_STEPS])
def test_fake_loop_emits_exactly_the_configured_checkpoint_steps(steps):
    assert _fake_loop(steps) == steps


def test_checkpoint_steps_alone_produce_no_periodic_extras():
    saved = _fake_loop(_S3_STEPS)
    # A `% checkpoint_every` mutant (e.g. every 100) would emit 25 steps here.
    assert len(saved) == 5
    assert saved == sorted(set(saved))


def test_empty_checkpoint_steps_falls_back_to_checkpoint_every():
    # Backwards compatibility with the inherited cadence knob.
    assert _fake_loop([], max_train_steps=2500, checkpoint_every=1000) == [1000, 2000, 2500]


def test_non_empty_list_SUPPRESSES_the_cadence(tmp_path=None):
    # C4 (cycle-C review, MINOR): H2's contract is an EXACT retained set. Union semantics let
    # an accidental nonzero cadence mint unplanned ~30 GB checkpoints, so a non-empty list
    # takes PRECEDENCE and the cadence is ignored entirely.
    saved = _fake_loop([250, 1750], max_train_steps=2000, checkpoint_every=1000)
    assert saved == [250, 1750, 2000]  # the final save is the only non-listed step
    assert 1000 not in saved  # the cadence contributed nothing


def test_precedence_is_reported_when_both_knobs_are_set():
    scheduler = overfit100.CheckpointScheduler(
        checkpoint_steps=[250, 1750], checkpoint_every=1000, max_train_steps=2000, save_final=True
    )
    note = scheduler.precedence_note()
    assert note is not None
    assert "checkpoint_steps" in note and "1000" in note  # names the ignored cadence


def test_no_precedence_note_when_only_one_knob_is_set():
    listed = overfit100.CheckpointScheduler(
        checkpoint_steps=_S3_STEPS, checkpoint_every=0, max_train_steps=2500, save_final=True
    )
    cadence = overfit100.CheckpointScheduler(
        checkpoint_steps=[], checkpoint_every=1000, max_train_steps=2500, save_final=True
    )
    assert listed.precedence_note() is None
    assert cadence.precedence_note() is None


def test_final_save_is_not_duplicated_when_max_step_is_listed():
    saved = _fake_loop(_S3_STEPS, max_train_steps=2500, save_final=True)
    assert saved.count(2500) == 1


def test_final_save_is_appended_when_max_step_is_not_listed():
    saved = _fake_loop([250, 500], max_train_steps=1000, save_final=True)
    assert saved == [250, 500, 1000]


def test_final_save_can_be_disabled():
    assert _fake_loop([250, 500], max_train_steps=1000, save_final=False) == [250, 500]


def test_resume_still_emits_the_remaining_segment_finals():
    # Restarting at step 1000 (checkpoint 1000 already on disk): the later listed steps
    # must still be emitted, and the earlier ones are not re-emitted.
    assert _fake_loop(_S3_STEPS, start_step=1000) == [1750, 2500]


def test_steps_beyond_max_train_steps_are_never_emitted():
    assert _fake_loop([250, 5000, 7500], max_train_steps=2500) == [250, 2500]


def test_planned_steps_matches_the_fake_loop():
    # The startup log's predeclared plan and the loop's behavior are the same set.
    for steps, every, max_steps in ((_S3_STEPS, 0, 2500), (_S2_STEPS, 0, 2500), ([], 1000, 2500)):
        assert overfit100.planned_checkpoint_steps(
            max_train_steps=max_steps, checkpoint_steps=steps, checkpoint_every=every, save_final=True
        ) == _fake_loop(steps, max_train_steps=max_steps, checkpoint_every=every)


def test_config_lists_are_the_ones_under_test():
    cfg = yaml.safe_load(_CONFIG.read_text())
    assert overfit100.parse_checkpoint_steps(cfg["checkpoint_steps"]) == tuple(_S3_STEPS)
    assert cfg["max_train_steps"] == 2500
    # The documented S2 override string parses to the S2 list through pyconfig's parser.
    from maxdiffusion.pyconfig import string_to_list

    assert overfit100.parse_checkpoint_steps(string_to_list("[250,500,1000,2500]")) == tuple(_S2_STEPS)


@pytest.mark.parametrize("raw", [[250, 250, 500], (500, 250), "250,500", " 250 , 500 "])
def test_parse_checkpoint_steps_normalizes_sorted_unique(raw):
    assert overfit100.parse_checkpoint_steps(raw) == (250, 500)


@pytest.mark.parametrize("raw", [[0], [-5], ["x"], [1.5]])
def test_parse_checkpoint_steps_rejects_invalid_entries(raw):
    with pytest.raises(ValueError):
        overfit100.parse_checkpoint_steps(raw)


def test_parse_checkpoint_steps_accepts_empty():
    assert overfit100.parse_checkpoint_steps([]) == ()
    assert overfit100.parse_checkpoint_steps("") == ()
    assert overfit100.parse_checkpoint_steps(None) == ()


# =======================================================================================
# (B) retention: max_to_keep=None
# =======================================================================================


def _ckpt_config(ckpt_dir, *, checkpoint_keep_period=-1):
    return SimpleNamespace(checkpoint_dir=str(ckpt_dir), output_dir="", checkpoint_keep_period=checkpoint_keep_period)


def test_checkpoint_manager_options_keep_everything(tmp_path):
    options = overfit100.WanTI2VOverfit100Trainer(_ckpt_config(tmp_path))._checkpoint_manager_options()
    assert options.max_to_keep is None  # H2: segment-final checkpoints are never GC'd
    assert options.create is True
    # Unconditional: it does NOT depend on checkpoint_keep_period (see the next test for
    # why leaning on that knob is unsafe).
    other = overfit100.WanTI2VOverfit100Trainer(
        _ckpt_config(tmp_path, checkpoint_keep_period=0)
    )._checkpoint_manager_options()
    assert other.max_to_keep is None


def test_parent_max_to_keep_3_evicts_the_gate_baseline(tmp_path):
    # Non-vacuity for the retention claim: the INHERITED manager keeps only 3, so the S2/S3
    # gate baseline (step 250) is GONE by the time the segment final lands. The parent can
    # only rescue it via keep_period, which cannot express D10's non-uniform list.
    from maxdiffusion.trainers.wan_ti2v_side_adapter_trainer import WanTI2VSideAdapterTrainer

    parent = WanTI2VSideAdapterTrainer(_ckpt_config(tmp_path / "parent", checkpoint_keep_period=0))
    mgr = parent._build_checkpoint_manager(str(tmp_path / "parent"))
    state = _make_state(_INIT_GAIN, optax.adamw(0.1), table_value=1.0)
    for step in _S3_STEPS:
        parent._save_checkpoint(mgr, step, state)
    mgr.wait_until_finished()
    kept = sorted(mgr.all_steps())
    mgr.close()
    assert kept == [1000, 1750, 2500]  # step 250 -- the S2 gate baseline -- is GONE


def test_parent_keep_period_minus_one_retention_is_undefined_behavior(tmp_path):
    # Documented reason the overfit100 manager does NOT reuse the parent's construction:
    # the parent passes `checkpoint_keep_period or None`, so the repo-wide default -1 is
    # forwarded verbatim and retention then hinges on `step % -1 == 0` being true in
    # Python -- everything survives BY ACCIDENT, at the mercy of Orbax's validation. The
    # exp_02 contract must be explicit (max_to_keep=None), not incidental.
    from maxdiffusion.trainers.wan_ti2v_side_adapter_trainer import WanTI2VSideAdapterTrainer

    parent = WanTI2VSideAdapterTrainer(_ckpt_config(tmp_path / "parent", checkpoint_keep_period=-1))
    mgr = parent._build_checkpoint_manager(str(tmp_path / "parent"))
    state = _make_state(_INIT_GAIN, optax.adamw(0.1), table_value=1.0)
    for step in _S3_STEPS:
        parent._save_checkpoint(mgr, step, state)
    mgr.wait_until_finished()
    kept = sorted(mgr.all_steps())
    mgr.close()
    # Observed on the pinned Orbax: all five survive, but only because of the modulo quirk.
    assert kept == _S3_STEPS


def test_overfit100_manager_retains_every_listed_checkpoint(tmp_path):
    trainer = overfit100.WanTI2VOverfit100Trainer(_ckpt_config(tmp_path / "ckpts"))
    mgr = trainer._build_checkpoint_manager(str(tmp_path / "ckpts"))
    state = _make_state(_INIT_GAIN, optax.adamw(0.1), table_value=1.0)
    for step in _S3_STEPS:
        trainer._save_checkpoint(mgr, step, state)
    mgr.wait_until_finished()
    kept = sorted(mgr.all_steps())
    mgr.close()
    assert kept == _S3_STEPS  # all five, including the 250 the parent would have evicted


# =======================================================================================
# (C) save/restore round trip EXCLUDING context_table
# =======================================================================================


class _CkptStubTransformer(nnx.Module):
    def __init__(self, gain=_INIT_GAIN):
        self.gain = nnx.Param(jnp.asarray(gain, dtype=jnp.float32))

    def __call__(self, **kwargs):  # pragma: no cover - unused here
        return self.gain[...] * kwargs["hidden_states"]


def _make_state(gain, tx, *, table_value):
    transformer = _CkptStubTransformer(gain)
    graphdef, params, rest = nnx.split(transformer, nnx.Param, ...)
    table = jnp.full((4, 3, 8), table_value, dtype=jnp.bfloat16)
    return overfit100.Overfit100TrainState.create(
        apply_fn=graphdef.apply,
        params=params,
        tx=tx,
        graphdef=graphdef,
        rest_of_state=rest,
        context_table=table,
    )


def _first_leaf(tree):
    return np.asarray(jax.tree_util.tree_leaves(tree)[0])


def test_save_targets_exclude_context_table(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    trainer = overfit100.WanTI2VOverfit100Trainer(_ckpt_config(ckpt_dir))
    mgr = trainer._build_checkpoint_manager(str(ckpt_dir))
    state = _make_state(_INIT_GAIN, optax.adamw(0.1), table_value=1.0)
    trainer._save_checkpoint(mgr, 250, state)
    mgr.wait_until_finished()
    mgr.close()
    written = set(os.listdir(ckpt_dir / "250"))
    assert {"params", "opt_state", "step"} <= written
    assert not any("context" in name for name in written)


def test_restore_rebuilds_nothing_and_leaves_the_table_untouched(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    trainer = overfit100.WanTI2VOverfit100Trainer(_ckpt_config(ckpt_dir))

    # A saved state whose params AND Adam moments have moved off their init values, with
    # table value 1.0.
    saved = _make_state(_INIT_GAIN, optax.adamw(0.1), table_value=1.0)
    saved = saved.apply_gradients(grads=jax.tree.map(jnp.ones_like, saved.params))
    saved_param = _first_leaf(saved.params)
    from maxdiffusion.trainers.wan_ti2v_full_ft_trainer import _adam_moment_trees

    saved_mu = _first_leaf(_adam_moment_trees(saved.opt_state)[0])
    assert saved_param != np.float32(_INIT_GAIN) and saved_mu != 0.0  # non-vacuity

    mgr = trainer._build_checkpoint_manager(str(ckpt_dir))
    trainer._save_checkpoint(mgr, 250, saved)
    mgr.wait_until_finished()
    mgr.close()

    # A FRESH state: init params, zero moments, and a DIFFERENT table (7.0) -- the table a
    # rebuilt-from-episodes.json start would produce.
    fresh = _make_state(_INIT_GAIN, optax.adamw(0.1), table_value=7.0)
    assert _first_leaf(_adam_moment_trees(fresh.opt_state)[0]) == 0.0

    mgr = trainer._build_checkpoint_manager(str(ckpt_dir))
    restored, start_step = trainer._maybe_restore(mgr, fresh)
    mgr.close()

    assert start_step == 250
    np.testing.assert_array_equal(_first_leaf(restored.params), saved_param)
    np.testing.assert_array_equal(_first_leaf(_adam_moment_trees(restored.opt_state)[0]), saved_mu)
    # The context table is NEITHER saved NOR restored: it is still the freshly-built one.
    np.testing.assert_array_equal(
        np.asarray(restored.context_table, dtype=np.float32), np.full((4, 3, 8), 7.0, dtype=np.float32)
    )
    assert restored.context_table.dtype == jnp.bfloat16


def test_restore_on_empty_dir_is_a_no_op_at_step_zero(tmp_path):
    ckpt_dir = tmp_path / "empty"
    trainer = overfit100.WanTI2VOverfit100Trainer(_ckpt_config(ckpt_dir))
    mgr = trainer._build_checkpoint_manager(str(ckpt_dir))
    fresh = _make_state(_INIT_GAIN, optax.adamw(0.1), table_value=7.0)
    restored, start_step = trainer._maybe_restore(mgr, fresh)
    mgr.close()
    assert start_step == 0
    assert restored is fresh


def test_start_training_uses_the_scheduler_not_the_inherited_cadence():
    import inspect

    src = inspect.getsource(overfit100.WanTI2VOverfit100Trainer.start_training)
    assert "CheckpointScheduler" in src
    assert "Overfit100TrainState.create" in src
    assert "_build_context_table" in src
    # The inherited `% config.checkpoint_every` in-loop test must be gone (H2).
    assert "% config.checkpoint_every" not in src
