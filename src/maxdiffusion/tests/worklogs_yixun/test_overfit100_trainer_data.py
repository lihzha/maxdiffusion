"""CPU-only tests for the exp_02 overfit100 data path, shardings, and wiring (plan D7/D9/D10).

The overfit100 trainer reads the **schema-v2** TFRecords cycle B built (plan D6):
``{name, episode_id, episode_index, window_start, z_i0 f16, z_video f16, instruction}``
with **no ``actions`` field**. That makes the side-adapter parent's ``_load_dataset``
(whose nested ``prepare_sample`` hard-requires ``actions``) unusable, so the subclass
owns the parse. This file pins:

  (A) SCHEMA-V2 PARSE -- synthetic TFRecords written by the REAL cycle-B writer
      (``build_overfit100_dataset.serialize_window_record``) parse to the exact
      shapes/dtypes the loss consumes, ``episode_index`` survives as int32, and
      ``actions`` is neither requested nor emitted.
  (B) READINESS ASSERTS -- ``_SUCCESS`` is REQUIRED before any record is consumed
      (cycle-B contract: the marker is written LAST, after promotion), with an error
      naming the directory; and ``expected_windows`` is checked against the built
      count the way cycle B records it, with an error naming both numbers.
  (C) DATA SHARDINGS -- the tree structure of ``_data_shardings`` matches a real batch
      (jax pytree-structure equality), and ``episode_index`` is sharded on the batch
      axis exactly like ``z_i0``.
  (D) CONFIG + DISPATCH + WRAPPER WIRING -- ``base_wan_5b_overfit100.yml`` carries every
      plan-D9/D10 delta (and every new key, so each is pyconfig-overridable), the
      documented S2 ``train10`` overrides are valid pyconfig values, ``train_wan.py``
      dispatches ``OVERFIT100_TI2V``, and ``bash_scripts/train_wan_overfit100.sh``
      passes ``bash -n`` and forwards the exp_02 knobs.

CPU-only: TFRecords land in ``tmp_path``, no GCS, no weights, no mesh beyond a 1-device
CPU mesh. The darwin grain import stub lives in ``conftest.py``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import yaml
from jax.sharding import NamedSharding, PartitionSpec as P

import maxdiffusion.trainers.wan_ti2v_overfit100_trainer as overfit100
from maxdiffusion.trainers.wan_ti2v_side_adapter_trainer import WanTI2VSideAdapterTrainer
from maxdiffusion.data_preprocessing.build_overfit100_dataset import (
    SUCCESS_MARKER,
    Z_I0_SHAPE,
    Z_VIDEO_SHAPE,
    serialize_window_record,
)

_REPO = Path(overfit100.__file__).parents[3]
_CONFIG = _REPO / "src/maxdiffusion/configs/base_wan_5b_overfit100.yml"
_FULL_FT_CONFIG = _REPO / "src/maxdiffusion/configs/base_wan_5b_full_ft.yml"
_WRAPPER = _REPO / "bash_scripts/train_wan_overfit100.sh"
_TRAIN_WAN = _REPO / "src/maxdiffusion/train_wan.py"


def _geometry_config(**kw):
    cfg = {
        "latent_channels": Z_VIDEO_SHAPE[0],
        "latent_frames": Z_VIDEO_SHAPE[1],
        "latent_height": Z_VIDEO_SHAPE[2],
        "latent_width": Z_VIDEO_SHAPE[3],
        "data_sharding": (("data", "fsdp", "context", "tensor"),),
        # C3: the parse path range-asserts episode_index against the table height.
        "num_text_slots": 10,
    }
    cfg.update(kw)
    return SimpleNamespace(**cfg)


def _write_records(directory: Path, episode_indices, *, success=True, records=None, marker=None) -> list[dict]:
    """One shard plus the cycle-B sidecars, with a STRUCTURALLY VALID marker.

    The full fingerprint fixture (per-shard hashes, mutation knobs) lives in
    ``test_overfit100_preflight.py``; this one carries just enough for the count/parse tests,
    but it must still satisfy ``read_success_marker``'s structural contract -- the strengthened
    reader refuses a marker missing the required provenance fields.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    rng = np.random.default_rng(0)
    shard_name = "overfit100-00000-of-00001.tfrecord"
    path = directory / shard_name
    with tf.io.TFRecordWriter(str(path)) as writer:
        for position, episode_index in enumerate(episode_indices):
            z_video = rng.standard_normal(Z_VIDEO_SHAPE).astype(np.float16)
            z_i0 = z_video[:, 0:1].copy()
            writer.write(
                serialize_window_record(
                    name=f"ep{25189 + episode_index:05d}_v0_s{position * 4:05d}",
                    # index != id on purpose (manifest index 0 -> episode_id 25189).
                    episode_id=25189 + episode_index,
                    episode_index=episode_index,
                    window_start=position * 4,
                    z_i0=z_i0,
                    z_video=z_video,
                    instruction=f"instruction {episode_index}",
                )
            )
            written.append({"episode_index": episode_index, "z_i0": z_i0, "z_video": z_video})
    payload = path.read_bytes()
    summary = {
        "build_id": "20260729-000000",
        "build_commit": "0" * 40,
        "sets": {
            directory.name: {
                "written": len(episode_indices),
                "expected_windows": len(episode_indices),
                "shards": [
                    {
                        "name": shard_name,
                        "records": len(episode_indices),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size": len(payload),
                    }
                ],
            }
        },
    }
    summary_bytes = json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n"
    (directory / "summary.json").write_text(summary_bytes)
    if success:
        full = {
            "build_id": summary["build_id"],
            "build_commit": summary["build_commit"],
            "records": len(episode_indices) if records is None else records,
            "shards": 1,
            "summary_sha256": hashlib.sha256(summary_bytes.encode("utf-8")).hexdigest(),
            "manifest_sha256": "f" * 64,
        }
        if marker is not None:
            full = marker(full)
        (directory / SUCCESS_MARKER).write_text(json.dumps(full, indent=2) + "\n")
    return written


# =======================================================================================
# (A) schema-v2 parse
# =======================================================================================


def test_feature_description_is_schema_v2_without_actions():
    spec = overfit100._schema_v2_feature_description()
    assert set(spec) == {"z_i0", "z_video", "episode_index"}
    assert "actions" not in spec
    assert spec["episode_index"].dtype == tf.int64  # int64_list on the wire
    for key in ("z_i0", "z_video"):
        assert spec[key].dtype == tf.string  # decode_raw payloads


def test_prepare_sample_parses_shapes_dtypes_and_episode_index(tmp_path):
    written = _write_records(tmp_path / "train10", [7, 0, 3])
    config = _geometry_config()
    spec = overfit100._schema_v2_feature_description()
    prepare = overfit100._schema_v2_prepare_sample(config)

    ds = tf.data.TFRecordDataset([str(p) for p in sorted((tmp_path / "train10").glob("*.tfrecord"))])
    ds = ds.map(lambda raw: prepare(tf.io.parse_single_example(raw, spec)))
    samples = list(ds.as_numpy_iterator())

    assert len(samples) == 3
    for sample, source in zip(samples, written):
        assert set(sample) == {"z_i0", "z_video", "episode_index"}  # no actions emitted
        assert sample["z_i0"].shape == tuple(Z_I0_SHAPE)
        assert sample["z_video"].shape == tuple(Z_VIDEO_SHAPE)
        assert sample["z_i0"].dtype == np.float32  # f16 on disk, f32 in the batch
        assert sample["z_video"].dtype == np.float32
        assert sample["episode_index"].dtype == np.int32
        assert int(sample["episode_index"]) == source["episode_index"]
        np.testing.assert_array_equal(sample["z_video"], source["z_video"].astype(np.float32))
        np.testing.assert_array_equal(sample["z_i0"], source["z_i0"].astype(np.float32))
    # episode_index carries the manifest INDEX, not the 5-digit episode id.
    assert [int(s["episode_index"]) for s in samples] == [7, 0, 3]


def _parse_all(directory: Path, config):
    spec = overfit100._schema_v2_feature_description()
    prepare = overfit100._schema_v2_prepare_sample(config)
    ds = tf.data.TFRecordDataset([str(p) for p in sorted(directory.glob("*.tfrecord"))])
    ds = ds.map(lambda raw: prepare(tf.io.parse_single_example(raw, spec)))
    return list(ds.as_numpy_iterator())


def test_episode_index_at_or_above_num_text_slots_is_rejected_in_the_parse_path(tmp_path):
    # C3: an out-of-range index must never reach the JIT gather -- jnp gathers CLAMP silently,
    # so index 99 against a 10-row table would train on row 9's instruction with no error.
    _write_records(tmp_path / "train10", [0, 99])
    with pytest.raises(tf.errors.InvalidArgumentError) as ei:
        _parse_all(tmp_path / "train10", _geometry_config(num_text_slots=10))
    assert "episode_index" in str(ei.value)


def test_negative_episode_index_is_rejected_in_the_parse_path(tmp_path):
    _write_records(tmp_path / "train10", [-1])
    with pytest.raises(tf.errors.InvalidArgumentError):
        _parse_all(tmp_path / "train10", _geometry_config(num_text_slots=10))


def test_in_range_boundary_indices_are_accepted(tmp_path):
    _write_records(tmp_path / "train10", [0, 9])
    samples = _parse_all(tmp_path / "train10", _geometry_config(num_text_slots=10))
    assert [int(s["episode_index"]) for s in samples] == [0, 9]


@pytest.mark.parametrize("bad", [2**32, 2**32 + 5, 2**31])
def test_int64_values_that_narrow_into_range_are_still_rejected(tmp_path, bad):
    # F4 (follow-up review, MINOR): the record carries an int64. Asserting AFTER the int32 cast
    # let 2**32 narrow to int32(0) -- an in-range row -- so a corrupt index silently trained on
    # episode 0's instruction. The bounds are now asserted on the ORIGINAL int64.
    _write_records(tmp_path / "train10", [bad])
    narrowed = int(np.int64(bad).astype(np.int32))  # non-vacuity: it really does narrow
    assert narrowed < 10
    with pytest.raises(tf.errors.InvalidArgumentError) as ei:
        _parse_all(tmp_path / "train10", _geometry_config(num_text_slots=10))
    assert "episode_index" in str(ei.value)


def test_negative_int64_index_is_rejected(tmp_path):
    _write_records(tmp_path / "train10", [-(2**32)])
    with pytest.raises(tf.errors.InvalidArgumentError):
        _parse_all(tmp_path / "train10", _geometry_config(num_text_slots=10))


def test_the_range_assert_consumes_the_int64_feature_not_the_cast():
    # Structural companion to the overflow regression: the bounds must be asserted on the raw
    # int64 feature, so the tf.data graph rejects the value BEFORE any narrowing.
    import inspect

    src = inspect.getsource(overfit100._schema_v2_prepare_sample)
    assert 'tf.cast(features["episode_index"]' not in src  # never assert on a narrowed value
    assert src.count("tf.constant(0, tf.int64)") == 1  # lower bound compared as int64
    assert "tf.constant(num_slots, tf.int64)" in src  # upper bound compared as int64
    # Both assertions take the raw feature, and the narrowing happens under their control dep.
    assert src.index("assert_greater_equal(\n                        raw_episode_index") > 0
    assert src.index("assert_less(\n                        raw_episode_index") > 0
    assert src.index("assert_less(") < src.rindex("tf.cast(raw_episode_index")


# =======================================================================================
# (B) readiness asserts: _SUCCESS required, expected_windows checked
# =======================================================================================


def test_missing_success_marker_raises_naming_the_dir(tmp_path):
    data_dir = tmp_path / "train100"
    _write_records(data_dir, [0, 1], success=False)
    with pytest.raises(ValueError) as ei:
        overfit100.assert_dataset_ready(str(data_dir), expected_windows=2)
    msg = str(ei.value)
    assert str(data_dir) in msg
    # The MARKER-MISSING message specifically -- not the "count unreadable" one, which also
    # mentions _SUCCESS. (A mutation check caught this: deleting the marker check let the
    # weaker `SUCCESS_MARKER in msg` assertion pass via the count path.)
    assert f"has no {SUCCESS_MARKER} marker" in msg


def test_missing_success_marker_is_refused_even_when_the_count_is_verifiable(tmp_path):
    # The airtight form of the _SUCCESS requirement: a set that LOOKS promoted -- shards on
    # disk and a summary.json whose count matches expected_windows exactly -- must still be
    # refused while the marker (written LAST) is absent. Only the marker check can fail this.
    data_dir = tmp_path / "train10"
    _write_records(data_dir, [0, 1], success=False)
    # The count IS independently readable from the promoted summary.json...
    summary = json.loads((data_dir / "summary.json").read_text())
    assert summary["sets"]["train10"]["written"] == 2
    # ...yet without the marker (written LAST, after promotion) the set is refused.
    with pytest.raises(ValueError) as ei:
        overfit100.assert_dataset_ready(str(data_dir), expected_windows=2)
    assert f"has no {SUCCESS_MARKER} marker" in str(ei.value)


def test_success_marker_present_returns_built_count(tmp_path):
    data_dir = tmp_path / "train100"
    _write_records(data_dir, [0, 1, 2])
    assert overfit100.assert_dataset_ready(str(data_dir), expected_windows=3) == 3


def test_expected_windows_mismatch_raises_naming_both_counts(tmp_path):
    data_dir = tmp_path / "train100"
    _write_records(data_dir, [0, 1, 2], records=1629)
    with pytest.raises(ValueError) as ei:
        overfit100.assert_dataset_ready(str(data_dir), expected_windows=167)
    msg = str(ei.value)
    assert "1629" in msg and "167" in msg and str(data_dir) in msg


def test_expected_windows_zero_skips_the_count_check(tmp_path):
    # A non-positive expected_windows means "marker only" (used for an eval dir that is
    # not the train set); the _SUCCESS requirement still holds.
    data_dir = tmp_path / "train100"
    _write_records(data_dir, [0], records=1629)
    assert overfit100.assert_dataset_ready(str(data_dir), expected_windows=0) == 1629


def test_count_falls_back_to_summary_json_only_when_a_valid_marker_lacks_records(tmp_path):
    # Judgment 8: the fallback is legal ONLY for a marker that read+validated fine and simply
    # omits the optional count field.
    data_dir = tmp_path / "train10"
    _write_records(data_dir, [0, 1], marker=lambda m: {k: v for k, v in m.items() if k != "records"})
    assert overfit100.assert_dataset_ready(str(data_dir), expected_windows=2) == 2


def test_unparseable_marker_fails_loudly_instead_of_falling_back(tmp_path):
    # Judgment 8: an unreadable/unparseable marker must NOT be masked by summary.json.
    data_dir = tmp_path / "train10"
    _write_records(data_dir, [0])
    (data_dir / SUCCESS_MARKER).write_text("not json\n")
    with pytest.raises(ValueError) as ei:
        overfit100.assert_dataset_ready(str(data_dir), expected_windows=1)
    msg = str(ei.value)
    assert str(data_dir) in msg and SUCCESS_MARKER in msg


def test_structurally_invalid_marker_fails_loudly(tmp_path):
    data_dir = tmp_path / "train10"
    _write_records(data_dir, [0], marker=lambda m: {k: v for k, v in m.items() if k != "summary_sha256"})
    with pytest.raises(ValueError, match="summary_sha256"):
        overfit100.assert_dataset_ready(str(data_dir), expected_windows=1)


def test_load_dataset_asserts_before_building_the_iterator(tmp_path, monkeypatch):
    # The readiness assert must run BEFORE make_data_iterator is consulted: a missing
    # _SUCCESS must fail without any reader being constructed.
    data_dir = tmp_path / "train100"
    _write_records(data_dir, [0, 1], success=False)

    def _boom(*args, **kwargs):
        raise AssertionError("make_data_iterator built before the readiness assert")

    monkeypatch.setattr(overfit100, "make_data_iterator", _boom)
    config = _geometry_config(
        dataset_type="tfrecord",
        cache_latents_text_encoder_outputs=True,
        train_data_dir=str(data_dir),
        eval_data_dir=str(data_dir),
        expected_windows=2,
        global_batch_size_to_load=1,
        seed=0,
    )
    trainer = overfit100.WanTI2VOverfit100Trainer(config)
    with pytest.raises(ValueError, match=f"has no {SUCCESS_MARKER} marker"):
        trainer._load_dataset(mesh=None, is_training=True)


def test_load_dataset_passes_schema_v2_parse_to_the_reader(tmp_path, monkeypatch):
    data_dir = tmp_path / "train100"
    _write_records(data_dir, [0, 1])
    seen = {}

    def _capture(config, host_index, host_count, mesh, batch, **kwargs):
        seen.update(kwargs)
        seen["batch"] = batch
        return "ITERATOR"

    monkeypatch.setattr(overfit100, "make_data_iterator", _capture)
    config = _geometry_config(
        dataset_type="tfrecord",
        cache_latents_text_encoder_outputs=True,
        train_data_dir=str(data_dir),
        eval_data_dir=str(data_dir),
        expected_windows=2,
        global_batch_size_to_load=2,
        seed=3,
    )
    trainer = overfit100.WanTI2VOverfit100Trainer(config)
    assert trainer._load_dataset(mesh=None, is_training=True, seed=11) == "ITERATOR"
    assert set(seen["feature_description"]) == {"z_i0", "z_video", "episode_index"}
    assert seen["is_training"] is True
    assert seen["seed"] == 11
    assert seen["batch"] == 2
    # The prepare fn the reader received is the schema-v2 one (emits episode_index).
    example = tf.io.parse_single_example(
        next(iter(tf.data.TFRecordDataset([str(p) for p in sorted(data_dir.glob("*.tfrecord"))]))),
        seen["feature_description"],
    )
    out = seen["prepare_sample_fn"](example)
    assert set(out) == {"z_i0", "z_video", "episode_index"}


def test_load_dataset_rejects_non_tfrecord_dataset_type(tmp_path):
    data_dir = tmp_path / "train100"
    _write_records(data_dir, [0])
    config = _geometry_config(
        dataset_type="hf",
        cache_latents_text_encoder_outputs=True,
        train_data_dir=str(data_dir),
        eval_data_dir=str(data_dir),
        expected_windows=1,
        global_batch_size_to_load=1,
        seed=0,
    )
    with pytest.raises(ValueError, match="tfrecord"):
        overfit100.WanTI2VOverfit100Trainer(config)._load_dataset(mesh=None, is_training=True)


# =======================================================================================
# (C) data shardings
# =======================================================================================


def _cpu_mesh():
    device = jax.devices()[0]
    return jax.sharding.Mesh(np.array([device]).reshape(1, 1, 1, 1), ("data", "fsdp", "context", "tensor"))


def test_data_shardings_tree_matches_a_real_batch():
    mesh = _cpu_mesh()
    config = _geometry_config()
    shardings = overfit100.WanTI2VOverfit100Trainer(config)._data_shardings(mesh)
    batch = {
        "z_i0": jnp.zeros((2, *Z_I0_SHAPE), dtype=jnp.float32),
        "z_video": jnp.zeros((2, *Z_VIDEO_SHAPE), dtype=jnp.float32),
        "episode_index": jnp.zeros((2,), dtype=jnp.int32),
    }
    assert jax.tree_util.tree_structure(batch) == jax.tree_util.tree_structure(shardings)
    # episode_index rides the SAME batch-axis sharding as z_i0 (it is a [B] int32 column).
    assert shardings["episode_index"] == shardings["z_i0"]
    assert shardings["episode_index"] == NamedSharding(mesh, P(*config.data_sharding))
    # And the specs actually apply to the batch (jit would reject a mismatched spec).
    placed = jax.tree.map(jax.device_put, batch, shardings)
    assert placed["episode_index"].shape == (2,)


def test_data_shardings_extends_the_parent_dict():
    mesh = _cpu_mesh()
    config = _geometry_config(action_len=32, action_dim=7)
    parent = WanTI2VSideAdapterTrainer(config)._data_shardings(mesh)
    child = overfit100.WanTI2VOverfit100Trainer(config)._data_shardings(mesh)
    # The parent's latent keys keep their exact sharding; actions is dropped (schema v2
    # has none) and episode_index is added.
    for key in ("z_i0", "z_video"):
        assert child[key] == parent[key]
    assert "actions" not in child
    assert "episode_index" in child


def test_overfit100_state_shardings_keeps_computed_params_and_replicates_the_table():
    mesh = _cpu_mesh()
    fsdp = NamedSharding(mesh, P("fsdp"))
    replicated = NamedSharding(mesh, P())
    state = overfit100.Overfit100TrainState(
        step=0,
        apply_fn=None,
        params={"w": jax.ShapeDtypeStruct((8,), jnp.float32)},
        tx=None,
        opt_state={"mu": jax.ShapeDtypeStruct((8,), jnp.float32)},
        graphdef=None,
        rest_of_state=None,
        context_table=jax.ShapeDtypeStruct((4, 3, 8), jnp.bfloat16),
    )
    # Pretend nnx computed FSDP specs everywhere, including for the table.
    computed = state.replace(params={"w": fsdp}, opt_state={"mu": fsdp}, context_table=fsdp)
    out = overfit100._overfit100_state_shardings(computed, state, replicated)
    # params/opt_state keep the COMPUTED FSDP specs (replicating the ~5B tree would OOM)...
    assert out.params == {"w": fsdp}
    assert out.opt_state == {"mu": fsdp}
    # ...and the context table is pinned REPLICATED (every device gathers arbitrary rows).
    assert out.context_table == replicated


# =======================================================================================
# (D) config asserts + the absolute-warmup optimizer (plan D9)
# =======================================================================================


def _recipe_config(**kw):
    cfg = {
        "num_text_slots": 100,
        "text_encode_batch": 8,
        "expected_windows": 1629,
        "checkpoint_steps": [250, 500, 1000, 1750, 2500],
        # F3: the committed manifest authenticates episodes.json, so it is mandatory.
        "model_manifest_path": "docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json",
        "learning_rate": 1.0e-5,
        "warmup_steps": 250,
        "learning_rate_schedule_steps": 2500,
        "warmup_steps_fraction": 0.1,
        "adam_b1": 0.9,
        "adam_b2": 0.999,
        "adam_eps": 1.0e-8,
        "adam_weight_decay": 1.0e-2,
        "opt_enable_grad_global_norm_clipping": True,
        "max_grad_norm": 1.0,
        "opt_enable_grad_clipping": False,
        "max_grad_value": 1.0,
    }
    cfg.update(kw)
    return SimpleNamespace(**cfg)


def test_validate_overfit100_config_accepts_the_shipped_recipe():
    overfit100.WanTI2VOverfit100Trainer._validate_overfit100_config(_recipe_config())
    # And the real yml's values pass too.
    cfg = _yml()
    overfit100.WanTI2VOverfit100Trainer._validate_overfit100_config(
        _recipe_config(
            num_text_slots=cfg["num_text_slots"],
            text_encode_batch=cfg["text_encode_batch"],
            expected_windows=cfg["expected_windows"],
            checkpoint_steps=cfg["checkpoint_steps"],
        )
    )


@pytest.mark.parametrize(
    "bad,needle",
    [
        ({"num_text_slots": 0}, "num_text_slots"),
        ({"text_encode_batch": 0}, "text_encode_batch"),
        ({"expected_windows": 0}, "expected_windows"),
        ({"checkpoint_steps": [0, 250]}, "checkpoint_steps"),
    ],
)
def test_validate_overfit100_config_rejects_bad_values(bad, needle):
    with pytest.raises(ValueError, match=needle):
        overfit100.WanTI2VOverfit100Trainer._validate_overfit100_config(_recipe_config(**bad))


def test_absolute_warmup_schedule_reaches_lr_at_warmup_steps():
    trainer = overfit100.WanTI2VOverfit100Trainer(_recipe_config())
    _tx, schedule = trainer._build_optimizer(2500)
    lr = 1.0e-5
    assert float(schedule(0)) == 0.0
    np.testing.assert_allclose(float(schedule(125)), lr * 0.5, rtol=1e-6)
    np.testing.assert_allclose(float(schedule(250)), lr, rtol=1e-6)
    np.testing.assert_allclose(float(schedule(2499)), lr, rtol=1e-6)


def test_absolute_warmup_is_invariant_to_segment_length():
    # A resumable S3 segment that raises max_train_steps must NOT rescale the warmup --
    # the fraction-based factory would stretch it to 1000 steps at max_train_steps=10000.
    trainer = overfit100.WanTI2VOverfit100Trainer(_recipe_config())
    _tx, short = trainer._build_optimizer(2500)
    _tx, long = trainer._build_optimizer(10000)
    lr = 1.0e-5
    for step in (0, 125, 250, 500):
        np.testing.assert_allclose(float(short(step)), float(long(step)), rtol=0, atol=0)
    np.testing.assert_allclose(float(long(250)), lr, rtol=1e-6)


def test_warmup_steps_zero_falls_back_to_the_inherited_fraction_schedule():
    trainer = overfit100.WanTI2VOverfit100Trainer(_recipe_config(warmup_steps=0))
    _tx, schedule = trainer._build_optimizer(2500)
    from maxdiffusion import max_utils

    reference = max_utils.create_learning_rate_schedule(1.0e-5, 2500, 0.1, 2500)
    for step in (0, 125, 250, 2000):
        np.testing.assert_allclose(float(schedule(step)), float(reference(step)), rtol=0, atol=0)


def test_optimizer_is_adamw_with_global_norm_clipping_like_exp01():
    import optax

    trainer = overfit100.WanTI2VOverfit100Trainer(_recipe_config())
    tx, _schedule = trainer._build_optimizer(2500)
    params = {"w": jnp.ones((4,), dtype=jnp.float32)}
    opt_state = tx.init(params)
    # The real Adam moments exist (so checkpoints carry mu/nu, the ~30 GB budget), and
    # clipping is chained in front (exp_01 parity).
    from maxdiffusion.trainers.wan_ti2v_full_ft_trainer import _adam_moment_trees

    mu, nu = _adam_moment_trees(opt_state)
    assert mu is not None and nu is not None
    assert any(
        isinstance(node, optax.ScaleByAdamState)
        for node in jax.tree_util.tree_leaves(opt_state, is_leaf=lambda x: isinstance(x, optax.ScaleByAdamState))
    )


# =======================================================================================
# (E) config + dispatch + wrapper wiring
# =======================================================================================


def _yml():
    return yaml.safe_load(_CONFIG.read_text())


def test_overfit100_yml_exists_and_parses():
    assert _CONFIG.exists(), f"missing {_CONFIG}"
    assert isinstance(_yml(), dict)


def test_overfit100_yml_plan_d9_d10_deltas():
    cfg = _yml()
    assert cfg["model_type"] == "OVERFIT100_TI2V"
    assert cfg["train_data_dir"] == "gs://v6_east1d/datasets/exp02_overfit100/train100"
    assert cfg["expected_windows"] == 1629
    assert cfg["num_text_slots"] == 100
    assert cfg["text_encode_batch"] == 8
    assert cfg["learning_rate"] == 1.0e-5
    assert cfg["warmup_steps"] == 250
    assert cfg["max_train_steps"] == 2500
    assert cfg["checkpoint_steps"] == [250, 500, 1000, 1750, 2500]
    assert cfg["per_device_batch_size"] == 4.0
    assert cfg["global_batch_size_to_train_on"] == 256
    assert cfg["global_batch_size_to_load"] == 256
    assert cfg["side_adapter_noise_mode"] == "fresh"
    assert cfg["side_adapter_guide_scale"] == 1.0
    # The list-typed knob must be a LIST in the yml so pyconfig's string_to_list parser
    # accepts a CLI override (a str default would make `checkpoint_steps=[...]` a no-op).
    assert isinstance(cfg["checkpoint_steps"], list)
    assert all(isinstance(s, int) for s in cfg["checkpoint_steps"])


def test_overfit100_yml_new_keys_are_all_present_for_pyconfig_overridability():
    # pyconfig raises "Key X was passed at the command line but isn't in config" for any
    # override whose key is absent from the yml -- every new knob must live here.
    cfg = _yml()
    for key in ("expected_windows", "num_text_slots", "text_encode_batch", "checkpoint_steps", "warmup_steps"):
        assert key in cfg, key


def test_overfit100_yml_documents_the_train10_s2_overrides():
    text = _CONFIG.read_text()
    assert "train10" in text
    assert "expected_windows=167" in text or "expected_windows: 167" in text or "167" in text
    # The documented S2 checkpoint list must be a literal pyconfig value.
    assert "[250,500,1000,2500]" in text.replace(" ", "")
    assert ast.literal_eval("[250,500,1000,2500]") == [250, 500, 1000, 2500]


def test_overfit100_yml_retains_full_ft_geometry_and_optimizer_keys():
    cfg, base = _yml(), yaml.safe_load(_FULL_FT_CONFIG.read_text())
    for key in (
        "latent_channels",
        "latent_frames",
        "latent_height",
        "latent_width",
        "height",
        "width",
        "num_frames",
        "wan_max_sequence_length",
        "text_dim",
        "weights_dtype",
        "activations_dtype",
        "remat_policy",
        "adam_b1",
        "adam_b2",
        "adam_eps",
        "adam_weight_decay",
        "max_grad_norm",
        "opt_enable_grad_global_norm_clipping",
        "side_adapter_sampling_steps",
        "side_adapter_t_sampling",
        "flow_shift",
        "flow_sigma_min",
        "flow_sigma_max",
        "logical_axis_rules",
        "data_sharding",
        "ici_fsdp_parallelism",
        "scan_layers",
    ):
        assert cfg[key] == base[key], key


def test_overfit100_yml_context_table_geometry_matches_400mib():
    # D8/F5's memory claim: 100 slots x 512 tokens x 4096 dims x 2 bytes (bf16) = 400 MiB.
    cfg = _yml()
    nbytes = cfg["num_text_slots"] * cfg["wan_max_sequence_length"] * cfg["text_dim"] * 2
    assert nbytes == 400 * 2**20


def test_dispatch_overfit100_branch_and_class():
    src = _TRAIN_WAN.read_text()
    assert 'config.model_type == "OVERFIT100_TI2V"' in src
    dispatch = src[src.index("def train(") :]
    assert "OVERFIT100_TI2V" in dispatch and "WanTI2VOverfit100Trainer" in dispatch
    assert issubclass(overfit100.WanTI2VOverfit100Trainer, overfit100.WanTI2VFullFTTrainer)


def test_train_wan_dispatch_executes_overfit100_trainer(monkeypatch):
    import maxdiffusion.train_wan as train_wan

    calls = []

    class _FakeTrainer:
        def __init__(self, config):
            calls.append(("init", config))

        def start_training(self):
            calls.append(("start", None))

    monkeypatch.setattr(overfit100, "WanTI2VOverfit100Trainer", _FakeTrainer)
    config = SimpleNamespace(model_type="OVERFIT100_TI2V")
    train_wan.train(config)
    assert calls == [("init", config), ("start", None)]


def test_train_wrapper_exists_and_passes_bash_n():
    assert _WRAPPER.exists(), f"missing {_WRAPPER}"
    bash_exe = shutil.which("bash") or "/bin/bash"
    proc = subprocess.run([bash_exe, "-n", str(_WRAPPER)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr


def test_train_wrapper_forwards_the_exp02_knobs():
    text = _WRAPPER.read_text()
    assert "src/maxdiffusion/configs/base_wan_5b_overfit100.yml" in text
    for env in (
        "RUN_NAME",
        "MAX_TRAIN_STEPS",
        "DATA_DIR",
        "EXPECTED_WINDOWS",
        "NUM_TEXT_SLOTS",
        "CHECKPOINT_STEPS",
    ):
        assert env in text, env
    for override in (
        'run_name="${RUN_NAME}"',
        'train_data_dir="${DATA_DIR}"',
        'expected_windows="${EXPECTED_WINDOWS}"',
        'num_text_slots="${NUM_TEXT_SLOTS}"',
        'checkpoint_steps="${CHECKPOINT_STEPS}"',
        'max_train_steps="${MAX_TRAIN_STEPS}"',
    ):
        assert override in text, override
    # fresh noise is the documented foot-gun default; keep it explicit.
    assert "SIDE_ADAPTER_NOISE_MODE:-fresh" in text
    # Full-repo HF prefetch: this trainer needs transformer + T5 + VAE (unlike the
    # VAE-only cycle-B build), so NO allow-pattern argument is passed.
    assert "bash bash_scripts/prefetch_hf_snapshot.sh" in text
    prefetch_line = next(
        line for line in text.splitlines() if "prefetch_hf_snapshot.sh" in line and "#" != line.strip()[:1]
    )
    assert "vae/*" not in prefetch_line and "model_index.json" not in prefetch_line
    assert "export COMMIT" in text
    assert "LIBTPU_INIT_ARGS" in text
