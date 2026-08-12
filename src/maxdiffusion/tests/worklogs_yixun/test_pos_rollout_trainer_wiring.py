"""exp_06 `rollout_adapter` — W2: `WanPosRolloutTrainer.start_training()` is WIRED (review A2).

**The finding this round closes, in the reviewer's words:** *"accumulation, optimizer behaviour,
batch-one x 64 timing and checkpoint payload are all individually correct — but 'the probe measures
the trainer's step' is FALSE because the production trainer is still unwired."* Every piece existed;
nothing called them together. So this file's job is not to re-prove the pieces (their own rounds did
that) but to prove the **composition**: that a config becomes a running training job, that the job's
step is the step M1 measures, and that the properties the earlier rounds established survive being
wired together.

**What is REAL here.** A real ``WanModel`` and a real ``NNXWanSideAdapterStack`` at the **production
latent geometry** ``[48, 9, 12, 20]`` (the DEV instrument's canonical reader enforces exactly that
geometry, so a smaller one would have meant not running the real instrument); the real
``build_adapter_stack``/``build_optimizer``/``build_logical_update``; the real ``build_arm`` loss; the
real ``draw_step_for_batch`` stream; the real 64-example DEV cohort loaded from exp_04's published J0
manifest by digest; the real ``run_loop`` with real Orbax resume and selection trees; the real recipe
lock, the real attempt publication and the real SHA-bound resume adoption.

**What is not, and why — two seams, both named.**

* **The 5B weights and the T5 encoder.** ``_load_wan_pipeline`` downloads and loads a 5B checkpoint
  and ``_compute_null_context`` runs a T5 encoder; neither is a host operation. The settled trainer
  MODULE is therefore replaced in this process (the technique
  ``test_pos_rollout_eval_end_to_end`` uses for the same seam, for the same reason — importing it
  needs ``jaxopt`` and the diffusers stack, which this environment does not have). Production takes
  no model argument, so this substitution is a property of the test process and not of the API.
* **``make_data_iterator``.** exp_06 reads its batches through the settled ``_load_dataset``, which
  imports the maxdiffusion input pipeline — unavailable here (``datasets`` is not installed). The
  stand-in reads **real TFRecord files written in the deployed schema** with the deployed
  ``feature_description``/``prepare_sample``, so the decode contract is exercised even though the
  multi-host iterator around it is not.

Both stand-ins are structural risks only if the names they stand in for drift, so
``test_every_seam_the_trainer_calls_exists_on_the_REAL_settled_trainer`` pins each one against the
real file's source and against this trainer's own call sites.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import inspect
import json
import os
import subprocess
import sys
import textwrap
import types as _types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from maxdiffusion import pos_rollout_dev_instrument as instrument
from maxdiffusion import pos_rollout_fit_probe as probe
from maxdiffusion import pos_rollout_loop as loop
from maxdiffusion import pos_rollout_update
from maxdiffusion.tests.worklogs_yixun.test_pos_rollout_fit_probe import _install_import_shims

_PACKAGE_ROOT = Path(loop.__file__).resolve().parent
_CONFIG_PATH = _PACKAGE_ROOT / "configs" / "base_wan_5b_pos_rollout.yml"
_TRAINER_PATH = _PACKAGE_ROOT / "trainers" / "wan_pos_rollout_trainer.py"
_SETTLED_PATH = _PACKAGE_ROOT / "trainers" / "wan_ti2v_side_adapter_trainer.py"
_MANIFEST_DIR = _PACKAGE_ROOT.parents[1] / "docs" / "worklogs_yixun" / "exp_04_null_adapter_claude" / "j0_manifests"
_DEV = str(_MANIFEST_DIR / "dev64.json")

# PRODUCTION latent geometry -- `run_wan_null_inversion.build_read_batch` checks every DEV record
# against `PRODUCTION_GEOMETRY`, so the instrument can only ever hand the loss these shapes.
_C, _F, _H, _W = 48, 9, 12, 20
_ACTION_LEN, _ACTION_DIM = 32, 7
# ...and a tiny transformer/adapter around them, which is what keeps this runnable on a laptop.
_TEXT, _NULL_LEN, _MODEL_DIM = 32, 8, 16
_STEPS, _K, _LOGICAL, _MICRO = 4, 2, 4, 2


def _requires_backend():
    pytest.importorskip("torch")
    pytest.importorskip("aqt")
    pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("tensorflow")


def _mesh():
    """The deployed axes at one device: the config's own ``mesh_axes``, so its rules resolve."""
    return jax.sharding.Mesh(np.array(jax.devices()).reshape(1, 1, 1, 1), ("data", "fsdp", "context", "tensor"))


class _Config(dict):
    """A stand-in for ``pyconfig.HyperParameters``: attribute reads, ``ValueError`` on an unknown key.

    The ``ValueError`` is the point (issue #11): it is why three-argument ``getattr`` never falls
    back, and a fake raising ``AttributeError`` would quietly let that defect back in.
    """

    def __getattr__(self, key):
        if key not in self:
            raise ValueError(f"Key {key} not in config")
        return self[key]

    def get_keys(self):
        return dict(self)


@functools.lru_cache(maxsize=1)
def _model_dir() -> str:
    """A content-bound local snapshot: ``derive_model_revision`` fails closed on an unresolvable name."""
    import tempfile

    directory = Path(tempfile.mkdtemp(prefix="exp06_w2_model_")) / "snapshot"
    (directory / "transformer").mkdir(parents=True)
    (directory / "transformer" / "weights.safetensors").write_bytes(b"w" * 256)
    (directory / "model_index.json").write_text('{"_class_name": "test"}')
    return str(directory)


def _config(tmp_path, *, attempt="att-ONE", **overrides):
    values = yaml.safe_load(_CONFIG_PATH.read_text())
    values.update(
        {
            "pretrained_model_name_or_path": _model_dir(),
            "run_name": "w2",
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
            "global_batch_size_to_load": _LOGICAL,
            "global_batch_size_to_train_on": _LOGICAL,
            "max_train_steps": 4,
            "eval_every": 2,
            "checkpoint_every": 2,
            "pos_dev_manifest": _DEV,
            "train_data_dir": str(tmp_path / "shards"),
            "checkpoint_dir": str(tmp_path / "attempts" / attempt / "checkpoints"),
            "pos_resume_parent": str(tmp_path / "attempts"),
            "pos_recipe_lock": str(tmp_path / "recipe_lock.json"),
            "pos_fit_authorization": str(tmp_path / "m1.json"),
        }
    )
    values.update(overrides)
    return _Config(values)


# ---------------------------------------------------------------------------------------------
# The two seams: the 5B weights (with the T5 encoder), and the multi-host input pipeline.
# ---------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _tiny_transformer():
    """A REAL ``WanModel`` at production latent geometry, one layer wide enough to run on a CPU."""
    from flax import nnx

    from maxdiffusion.models.wan.transformers.transformer_wan import WanModel

    with _mesh():
        return WanModel(
            rngs=nnx.Rngs(jax.random.key(0)),
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=_C,
            out_channels=_C,
            text_dim=_TEXT,
            freq_dim=16,
            ffn_dim=32,
            num_layers=1,
            attention="dot_product",
            rope_max_seq_len=64,
            scan_layers=False,
            dtype=jnp.float32,
            weights_dtype=jnp.float32,
        )


class _Scheduler:
    """What ``_create_scheduler`` returns in the part the trainer reads: ``scheduler.config``."""

    def __init__(self, *, sigma_min=0.0, sigma_max=1.0, num_train_timesteps=1000):
        self.config = _types.SimpleNamespace(
            sigma_min=sigma_min, sigma_max=sigma_max, num_train_timesteps=num_train_timesteps
        )


def _write_shards(directory: Path, count: int = 16) -> str:
    """REAL TFRecords in the deployed side-adapter schema (fp16 latents, fp32 actions)."""
    import tensorflow as tf

    directory.mkdir(parents=True, exist_ok=True)
    with tf.io.TFRecordWriter(str(directory / "train-00000.tfrecord")) as writer:
        for index in range(count):
            value = 0.01 * (index + 1)
            payload = {
                "z_i0": np.full((_C, 1, _H, _W), value, np.float16).tobytes(),
                "z_video": np.full((_C, _F, _H, _W), value, np.float16).tobytes(),
                "actions": np.full((_ACTION_LEN, _ACTION_DIM), value, np.float32).tobytes(),
            }
            feature = {
                name: tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw])) for name, raw in payload.items()
            }
            writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
    return str(directory)


def _install_settled_trainer(monkeypatch, *, scheduler=None, on_load=None, on_dataset=None, cursor=True):
    """Stand in for the WEIGHTS and the input pipeline; everything else stays real.

    Replaces the module in ``sys.modules`` rather than patching the class, because importing the real
    one needs ``jaxopt`` and the diffusers pipeline stack. Declared boundary: this proves exp_06's
    own composition, not that the settled class exposes these four methods — the tripwire below pins
    those names against the real file.

    ``cursor`` selects which of two loader behaviours the seed means. ``True`` is the CONVENTION
    ``resume_seed`` encodes and T3b-4's oracle assumes — the batch at offset ``i`` of a stream opened
    at ``s`` is the ``(s + i)``-th batch, so opening at ``seed + start_step`` continues the sequence.
    ``False`` is what the DEPLOYED pipeline actually does — it seeds a file permutation and a shuffle
    buffer, so a resumed run re-draws rather than continues (CLAUDE.md: "Resume is partial"). Both
    are exercised, in the two tests that care.
    """
    mesh = _mesh()
    transformer = _tiny_transformer()

    class _Pipeline:
        def __init__(self):
            self.transformer = transformer
            self.mesh = mesh
            self.vae = object()
            self.text_encoder = object()

    class _Trainer:
        def __init__(self, config):
            self.config = config

        def _load_wan_pipeline(self):
            if on_load is not None:
                on_load()
            return _Pipeline()

        def _compute_null_context(self, pipeline, mesh_):
            del pipeline, mesh_
            # NON-ZERO deliberately: M1 used to close over zeros, so a zero stand-in here would make
            # "both paths use the same context" unfalsifiable.
            return jnp.full((1, _NULL_LEN, _TEXT), 0.25, jnp.float32)

        def _create_scheduler(self):
            return (scheduler or _Scheduler()), None

        def _load_dataset(self, mesh_, is_training, seed=None):
            del mesh_
            assert is_training, "the training loop opens the TRAIN split"
            if on_dataset is not None:
                on_dataset(seed)
            return _tfrecord_iterator(self.config, seed=seed, cursor=cursor)

    stub = _types.ModuleType("maxdiffusion.trainers.wan_ti2v_side_adapter_trainer")
    stub.WanTI2VSideAdapterTrainer = _Trainer
    monkeypatch.setitem(sys.modules, "maxdiffusion.trainers.wan_ti2v_side_adapter_trainer", stub)
    # The EXACT objects this seam will hand production, returned so a test can hold the very
    # transformer the trained closure captured (final review, MAJOR A3: comparing two freshly
    # constructed models proved nothing).
    return _types.SimpleNamespace(mesh=mesh, transformer=transformer, pipeline_cls=_Pipeline)


def _tfrecord_iterator(config, *, seed, cursor=True):
    """The deployed decode, over real TFRecords: exactly ``_load_dataset``'s feature/prepare pair."""
    import tensorflow as tf

    feature_description = {
        "z_i0": tf.io.FixedLenFeature([], tf.string),
        "z_video": tf.io.FixedLenFeature([], tf.string),
        "actions": tf.io.FixedLenFeature([], tf.string),
    }
    c, f = int(config.latent_channels), int(config.latent_frames)
    h, w = int(config.latent_height), int(config.latent_width)

    def prepare_sample(features):
        return {
            "z_i0": tf.cast(tf.reshape(tf.io.decode_raw(features["z_i0"], tf.float16), [c, 1, h, w]), tf.float32),
            "z_video": tf.cast(
                tf.reshape(tf.io.decode_raw(features["z_video"], tf.float16), [c, f, h, w]), tf.float32
            ),
            "actions": tf.cast(
                tf.reshape(
                    tf.io.decode_raw(features["actions"], tf.float32), [int(config.action_len), int(config.action_dim)]
                ),
                tf.float32,
            ),
        }

    files = sorted(tf.io.gfile.glob(f"{config.train_data_dir}/*.tfrecord"))
    width = int(config.global_batch_size_to_load)
    parsed = (
        tf.data.TFRecordDataset(files)
        .map(lambda raw: tf.io.parse_single_example(raw, feature_description))
        .map(prepare_sample)
    )
    if cursor:
        dataset = parsed.repeat(-1).batch(width, drop_remainder=True).skip(int(seed))
    else:
        dataset = parsed.shuffle(64, seed=int(seed)).batch(width, drop_remainder=True).repeat(-1)
    return ({key: jnp.asarray(value.numpy()) for key, value in batch.items()} for batch in dataset)


def _authorize(config, *, microbatch=_MICRO, k_b=_K, arm="rollout"):
    """Publish the M1 authorization this exact program would need, at this program's own context."""
    context = probe.derive_probe_context(config)
    measurement = probe.CellMeasurement(
        cell=probe.FitCell(arm=arm, microbatch=microbatch, k_b=k_b),
        context_digest=context.digest(),
        compile_seconds=1.0,
        step_seconds=1.0,
        eval_seconds=1.0,
        checkpoint_seconds=1.0,
        peak_bytes=1024,
        capacity_bytes=1024**3,
        reservation_failures=0,
        peak_source=probe.PEAK_SOURCE_RUNTIME_RESET,
    )
    evidence = probe.build_evidence(
        context,
        [measurement],
        max_train_steps=int(config.max_train_steps),
        eval_every=int(config.eval_every),
        checkpoint_every=int(config.checkpoint_every),
    )
    probe.publish_authorization(str(config.pos_fit_authorization), evidence)
    return context


_GEOMETRY = {"z_i0": (_C, 1, _H, _W), "z_video": (_C, _F, _H, _W), "actions": (_ACTION_LEN, _ACTION_DIM)}


def _install_dev_records(monkeypatch, rows):
    """Point the instrument's OWN decoder at in-memory records (the module seam, never an argument)."""
    from maxdiffusion import null_adapter_manifest_io, run_wan_null_inversion

    by_shard: dict[str, list[dict]] = {}
    for row in rows:
        by_shard.setdefault(str(row["shard_path"]), []).append(dict(row))

    def reader(shard_path, wanted):
        for row in by_shard[str(shard_path)]:
            if str(row["name"]) not in set(wanted):
                continue
            fill = float(int(hashlib.sha256(str(row["name"]).encode()).hexdigest()[:4], 16) % 97) / 100.0 + 0.01
            yield (
                str(row["name"]),
                int(row["ordinal"]),
                np.full(_GEOMETRY["z_i0"], fill, np.float32),
                np.full(_GEOMETRY["z_video"], fill, np.float32),
                np.full(_GEOMETRY["actions"], fill, np.float32),
            )

    def binder(shard_path):
        row = by_shard[str(shard_path)][0]
        return {"generation": str(row["shard_generation"]), "size": int(row["shard_size"])}

    monkeypatch.setattr(run_wan_null_inversion, "_tfrecord_reader", reader)
    monkeypatch.setattr(null_adapter_manifest_io, "shard_binding", binder)


def _trainer(config):
    from maxdiffusion.trainers.wan_pos_rollout_trainer import WanPosRolloutTrainer

    return WanPosRolloutTrainer(config)


def _prepared(tmp_path, monkeypatch, *, cursor=True, **overrides):
    """A config, its shards, its DEV records, its M1 authorization and the two stood-in seams."""
    _requires_backend()
    _install_import_shims()
    config = _config(tmp_path, **overrides)
    _write_shards(Path(config.train_data_dir))
    _install_dev_records(monkeypatch, instrument.load_dev_cohort(_DEV).rows)
    _install_settled_trainer(monkeypatch, cursor=cursor)
    _authorize(config)
    return config


def _function_node(source: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


# =============================================================================================
# 1. The round's headline: `start_training` runs, end to end, on a real model.
# =============================================================================================


def test_start_training_runs_the_whole_job_end_to_end(tmp_path, monkeypatch):
    """Config in, trained adapter out — with every artifact the run is supposed to leave behind.

    The `NotImplementedError` this replaces was the reviewer's A2 blocker: the probe claimed to
    measure "the trainer's step" while the trainer had no step. Now the trainer takes four real
    optimizer steps over real TFRecords, evaluates the real DEV-64 instrument twice, writes both
    checkpoint trees, and publishes its attempt.
    """
    config = _prepared(tmp_path, monkeypatch)
    trainer = _trainer(config)
    initial = jax.tree.leaves(trainer.build_program(trainer.load_backbone()).params)
    report = trainer.start_training()

    assert report.steps_run == 4, "four optimizer steps, one per logical batch"
    assert [record.step for record in report.history] == [2, 4], "plan §3d's cadence, through the loop"
    assert all(np.isfinite(record.dev_metric) for record in report.history)
    assert int(report.state.step) == 4 and report.retained_step in (2, 4)

    resume = loop.build_checkpoint_manager(config.checkpoint_dir)
    selection = loop.build_selection_manager(config.checkpoint_dir)
    assert tuple(resume.all_steps()) == (2, 4), "a checkpoint at every evaluation boundary"
    assert selection.all_steps(), "the selection sibling holds the best DEV checkpoint"
    metadata = loop.read_checkpoint_json(resume, 4)
    assert metadata["arm"] == "rollout" and metadata["k_b"] == 2, "a checkpoint says which arm made it"

    # ...and the parameters actually moved: a wired loop that returned its initial state would
    # satisfy every count above.
    trained = jax.tree.leaves(report.state.params)
    assert any(not np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(initial, trained))


def test_the_attempt_is_PUBLISHED_after_the_loop_and_its_id_is_derived(tmp_path, monkeypatch):
    """`publish_attempt` was implemented and never called (review A2). It is called here, LAST, and
    the attempt it names is derived from the tree it describes rather than supplied beside it."""
    config = _prepared(tmp_path, monkeypatch)
    trainer = _trainer(config)
    assert trainer.attempt_identity() == (str(tmp_path / "attempts"), "att-ONE")

    report = trainer.start_training()
    marker = Path(config.pos_resume_parent) / "att-ONE" / "publication.json"
    assert marker.exists(), "an attempt nothing published can never be resumed from"
    published = json.loads(marker.read_text())["payload"]
    assert published["step"] == int(report.state.step) == 4
    assert published["arm"] == "rollout" and published["complete"] is True
    assert published["checkpoint_dir"] == config.checkpoint_dir

    from maxdiffusion.trainers.wan_pos_rollout_trainer import select_resume_publication

    # F5c: the selector matches the WHOLE derived context, not a commit label. Passing the running
    # context's digest here is what the trainer's own `resume_source` does.
    running = trainer.running_context()
    adopted = select_resume_publication(
        config.pos_resume_parent, code_sha=running.code_sha, arm="rollout", context_digest=running.digest()
    )
    assert adopted is not None and adopted["attempt"] == "att-ONE"
    assert (
        select_resume_publication(
            config.pos_resume_parent, code_sha=running.code_sha, arm="rollout", context_digest="f" * 64
        )
        is None
    ), "a foreign context adopts nothing, even at the same SHA"


@pytest.mark.parametrize(
    "checkpoint_dir, why",
    [
        ("{root}/attempts/att-ONE/ckpts", "the leaf must be the derived `checkpoints` directory"),
        ("{root}/attempts/whatever/checkpoints", "`whatever` is not an attempt id"),
        ("{root}/elsewhere/att-ONE/checkpoints", "the parent is not this run's attempts root"),
    ],
)
def test_a_checkpoint_root_the_publication_cannot_describe_is_refused(tmp_path, monkeypatch, checkpoint_dir, why):
    """A marker naming an attempt whose tree it does not describe is how a resume adopts the wrong
    state, so the identity is DERIVED from the output root and any other shape fails closed."""
    _install_import_shims()
    config = _config(tmp_path, checkpoint_dir=checkpoint_dir.format(root=tmp_path))
    with pytest.raises(ValueError, match="not this attempt's derived output root"):
        _trainer(config).attempt_identity()


# =============================================================================================
# 2. THE call-graph proof: M1 and the live trainer enter the SAME three factories.
# =============================================================================================


def _calls_in(function) -> set:
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def test_BOTH_M1_and_the_live_trainer_call_the_SAME_shared_factories(tmp_path, monkeypatch):
    """The two-sided proof, at W3's boundary: both paths enter ONE program-finalization function.

    W1 shared the adapter factory and W2 the optimizer and the logical update; the final review's
    blocker was that this was a level too low — the two sides still finished their programs
    differently (unsharded vs replicated, bare mesh vs axis rules, zero context vs the loader's).
    So the property is now that **both callers enter ``build_training_program``**, and that the three
    factories are reached from inside it rather than from either caller.

    Identity first, then recorded execution: the trainer binds the shared names at import and M1
    resolves them at call time, so the test asserts the two are the SAME function OBJECT — which no
    patching can manufacture — before installing one recorder under both names.
    """
    from maxdiffusion.trainers import wan_pos_rollout_trainer as trainer_module

    config = _prepared(tmp_path, monkeypatch)
    # The AST half is taken BEFORE the recorders are installed: `inspect.getsource` of a patched
    # module attribute reads the RECORDER, which would have made this assertion about the test.
    finalizer = _calls_in(pos_rollout_update.build_training_program)
    caller_sources = {
        "probe": inspect.getsource(probe.build_probe_program),
        "trainer": inspect.getsource(
            __import__(
                "maxdiffusion.trainers.wan_pos_rollout_trainer", fromlist=["x"]
            ).WanPosRolloutTrainer.build_program
        ),
    }
    caller_calls = {
        "probe": _calls_in(probe.build_probe_program),
        "trainer": _calls_in(
            __import__(
                "maxdiffusion.trainers.wan_pos_rollout_trainer", fromlist=["x"]
            ).WanPosRolloutTrainer.build_program
        ),
    }
    seen: list[tuple[str, str]] = []
    side = ["m1"]

    def _recorder(name, original):
        def wrapper(*args, **kwargs):
            seen.append((side[0], name))
            return original(*args, **kwargs)

        return wrapper

    shared_names = ("build_adapter_stack", "build_optimizer", "build_logical_update")
    for name in shared_names + ("build_training_program", "load_backbone"):
        shared = getattr(pos_rollout_update, name)
        if hasattr(trainer_module, name):
            assert getattr(trainer_module, name) is shared, f"the trainer's {name} is not the shared one"
        recorder = _recorder(name, shared)
        monkeypatch.setattr(pos_rollout_update, name, recorder)
        if hasattr(trainer_module, name):
            monkeypatch.setattr(trainer_module, name, recorder)

    class _WeightsSeam:
        def load(self, config_):
            return pos_rollout_update.load_backbone(config_)

    probe.build_probe_program(config, probe.FitCell("rollout", _MICRO, _K), model_source=_WeightsSeam())

    side[0] = "trainer"
    trainer = _trainer(config)
    trainer.build_program(trainer.load_backbone())

    for name in shared_names + ("build_training_program", "load_backbone"):
        assert ("m1", name) in seen, f"M1 did not call {name}"
        assert ("trainer", name) in seen, f"the live trainer did not call {name}"

    # ...and the three factories are entered from INSIDE the shared finalizer, so neither caller has
    # its own construction sitting beside the import.
    assert {"build_adapter_stack", "build_optimizer", "build_logical_update", "build_arm"} <= finalizer
    for name in ("probe", "trainer"):
        assert "NNXWanSideAdapterStack" not in caller_sources[name], "no caller may construct the adapter itself"
        assert "build_training_program" in caller_calls[name], "every caller enters the shared finalizer"


def test_the_trainer_and_M1_build_the_SAME_adapter_and_the_SAME_optimizer(tmp_path, monkeypatch):
    """Calling one function is not the same as getting one answer — and the first version of this
    test did not check that either: it compared shapes, dtypes and tree STRUCTURE and called it a
    value comparison (final review, MAJOR A2). Every adapter and optimizer leaf is now compared by
    VALUE. The sharding half of the claim needs more than one device and lives in the W3 oracle.
    """
    config = _prepared(tmp_path, monkeypatch)
    trainer = _trainer(config)
    left = trainer.build_program(trainer.load_backbone())

    class _WeightsSeam:
        def load(self, config_):
            from maxdiffusion.pos_rollout_update import load_backbone

            return load_backbone(config_)

    right = probe.build_probe_program(config, probe.FitCell("rollout", _MICRO, _K), model_source=_WeightsSeam())

    left_params, right_params = jax.tree.leaves(left.params), jax.tree.leaves(right.params)
    assert left_params and len(left_params) == len(right_params)
    for mine, theirs in zip(left_params, right_params):
        assert np.array_equal(np.asarray(mine), np.asarray(theirs)), "the two paths' adapter weights differ"
    left_opt, right_opt = jax.tree.leaves(left.opt_state), jax.tree.leaves(right.opt_state)
    assert left_opt and len(left_opt) == len(right_opt)
    for mine, theirs in zip(left_opt, right_opt):
        assert np.array_equal(np.asarray(mine), np.asarray(theirs)), "the two paths' optimizer states differ"

    # ...and the deployed context, which M1 used to invent: a zero null context and its own grid.
    assert np.array_equal(np.asarray(left.context.null_context), np.asarray(right.context.null_context))
    assert np.array_equal(np.asarray(left.context.sigmas), np.asarray(right.context.sigmas))
    assert float(np.abs(np.asarray(left.context.null_context)).sum()) > 0.0, "a zero context proves nothing"


# =============================================================================================
# 3. The properties earlier rounds established, re-asserted THROUGH the wiring.
# =============================================================================================


def test_one_wired_step_is_the_shared_logical_update_over_the_seams_own_split(tmp_path, monkeypatch):
    """The composition oracle: one wired step's loss is the MEAN of the per-microbatch losses, each
    scored on the microbatch and the draws the stream seam produced for it.

    This is the assertion that makes "the trainer's step is M1's step" checkable by value rather than
    by call graph. It fails if the update uses one microbatch, if it reuses one microbatch's draws for
    all of them, or if it sums where it should average — three plausible accumulation bugs that no
    count of optimizer steps could see.
    """
    from maxdiffusion.pos_rollout_stream import draw_step_for_batch

    config = _prepared(tmp_path, monkeypatch)
    trainer = _trainer(config)
    backbone = trainer.load_backbone()
    program = trainer.build_program(backbone)

    batch = next(trainer.batches(backbone)(int(config.seed)))
    _, draw_parts, batch_parts = draw_step_for_batch(
        batch,
        seed=int(config.seed),
        global_step=1,
        logical_batch=_LOGICAL,
        microbatch=_MICRO,
        num_steps=_STEPS,
        k_b=_K,
    )
    assert len(batch_parts) == _LOGICAL // _MICRO == 2, "the split must actually accumulate"
    per_micro = [
        float(program.dev_loss_fn(program.params, part, program.context, draws=draws)[0])
        for part, draws in zip(batch_parts, draw_parts)
    ]
    assert per_micro[0] != pytest.approx(
        per_micro[1], rel=1e-6
    ), "the two microbatches must be distinguishable, or reusing one's draws for both would be invisible"

    _, loss = program.update_fn(
        loop.RolloutTrainState(params=program.params, opt_state=program.opt_state, step=0),
        batch_parts,
        draw_parts,
        trainer.schedule,
        1,
    )
    assert float(loss) == pytest.approx(sum(per_micro) / len(per_micro), rel=1e-4)


def test_the_grid_is_the_SCHEDULERS_by_VALUE_and_not_only_by_spelling(tmp_path, monkeypatch):
    """The value-side twin of the source pin below (the G07 lesson: every guard family needs one).

    A scheduler disagreeing with the YAML is not hypothetical — deployment reads the sigma limits and
    the training-timestep count off the loaded model precisely because the model is their authority.
    """
    from maxdiffusion.models.wan.side_adapter_wan import build_rollout_sigmas

    config = _prepared(tmp_path, monkeypatch)
    assert float(config.flow_sigma_max) == 1.0, "the YAML's limit, which the scheduler below disagrees with"
    _install_settled_trainer(monkeypatch, scheduler=_Scheduler(num_train_timesteps=500, sigma_max=0.8))
    trainer = _trainer(config)
    program = trainer.build_program(trainer.load_backbone())

    assert program.context.num_train_timesteps == 500, "the scheduler's count, not the YAML's 1000"
    assert float(jnp.max(program.context.timesteps)) <= 500.0, "timesteps scale with the scheduler's count"
    expected = build_rollout_sigmas(_STEPS, float(config.flow_shift), 0.0, 0.8)
    assert np.allclose(np.asarray(program.context.sigmas), np.asarray(expected)), "sigma_max came from the config"


def test_a_config_whose_data_sharding_disagrees_with_the_loader_is_refused(tmp_path, monkeypatch):
    """Battery V08. The step compiles against the sharding the DEPLOYED loader produces; if
    ``data_sharding`` declares a different split, the batch a step compiles against is not the batch
    it will be handed, and nothing downstream would notice. The check existed and nothing ran it."""
    from maxdiffusion.pos_rollout_update import assert_batch_contract_matches_config

    config = _prepared(tmp_path, monkeypatch, data_sharding=[["data"]])
    trainer = _trainer(config)
    with pytest.raises(ValueError, match="would not be the batch the loader hands it"):
        trainer.build_program(trainer.load_backbone())

    # ...and the checked-in config agrees, so the guard is not merely unreachable.
    mesh = _mesh()
    assert_batch_contract_matches_config(_config(tmp_path), mesh)


def test_the_freeze_split_survives_the_wiring(tmp_path, monkeypatch):
    """~128M adapter trains, ~5B backbone does not — proved on the EXACT backbone that trained.

    The first version built one ``_tiny_transformer()`` before the run and another after, while
    ``start_training`` captured a third; identity-disjointness between two freshly extracted state
    trees is automatic, so the assertion could not fail (final review, MAJOR A3). Now the loader seam
    hands back the object it will give production, that identity is asserted, and the SAME object's
    parameter leaves are compared by value across the run.
    """
    from flax import nnx

    config = _prepared(tmp_path, monkeypatch)
    seam = _install_settled_trainer(monkeypatch)
    trainer = _trainer(config)
    backbone = trainer.load_backbone()
    assert backbone.transformer is seam.transformer, "the run must train against the seam's own backbone"

    def _paths(module):
        flat, _ = jax.tree_util.tree_flatten_with_path(nnx.state(module, nnx.Param))
        return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in flat}

    frozen_before = _paths(seam.transformer)
    report = trainer.start_training()
    frozen_after = _paths(seam.transformer)

    assert frozen_before and set(frozen_before) == set(frozen_after)
    for path, before in frozen_before.items():
        assert np.array_equal(before, frozen_after[path]), f"the frozen backbone parameter {path} MOVED"

    # ...and the trainable tree is the ADAPTER's, compared by PATH rather than by the identity of two
    # freshly extracted arrays (which is what made the old assertion vacuous).
    trainable = jax.tree.leaves(report.state.params)
    assert trainable, "the adapter has parameters"
    assert len(trainable) < len(frozen_after), "the adapter is the small half of the split"
    assert len(jax.tree.leaves(report.state.opt_state)) > 0, "the optimizer holds slots for the adapter"
    # The optimizer's slots must match the TRAINABLE tree's structure, not the backbone's: a leaked
    # backbone leaf would change this count.
    slots = [leaf for leaf in jax.tree.leaves(report.state.opt_state) if hasattr(leaf, "shape") and leaf.ndim > 0]
    assert len(slots) % len(trainable) == 0, "the optimizer state is built over the adapter tree alone"


def test_the_draws_the_wired_loop_consumed_are_the_stream_seams_own(tmp_path, monkeypatch):
    """The loop's randomness must be reproducible from ``(seed, global step)`` alone — recomputed
    here from ``draw_step_for_batch`` with no trainer in the picture and compared value by value."""
    from maxdiffusion.pos_rollout_stream import draw_step_for_batch

    config = _prepared(tmp_path, monkeypatch)
    report = _trainer(config).start_training()

    batch = {
        "z_video": jnp.zeros((_LOGICAL, _C, _F, _H, _W), jnp.float32),
        "z_i0": jnp.zeros((_LOGICAL, _C, 1, _H, _W), jnp.float32),
        "actions": jnp.zeros((_LOGICAL, _ACTION_LEN, _ACTION_DIM), jnp.float32),
    }
    expected = []
    for global_step in range(1, 5):
        _, parts, _ = draw_step_for_batch(
            batch,
            seed=int(config.seed),
            global_step=global_step,
            logical_batch=_LOGICAL,
            microbatch=_MICRO,
            num_steps=_STEPS,
            k_b=_K,
        )
        first = parts[0]
        expected.append(
            (global_step, int(first.support_start), int(first.t_idx[0]), round(float(first.epsilon.sum()), 6))
        )
    assert list(report.draw_log) == expected, "the wired loop drew something the stream seam did not"


def test_the_dev_metric_is_the_INSTRUMENTS_number_not_the_training_streams(tmp_path, monkeypatch):
    """plan §3d: selection is a fixed-draw DEV-64 estimand. Re-scored here from the instrument
    directly on the run's own final parameters and compared with what the loop recorded — and the
    training stream's own loss on the same parameters is shown to be a DIFFERENT number, so 'the
    metric came from the instrument' is falsifiable rather than merely stated."""
    config = _prepared(tmp_path, monkeypatch, max_train_steps=2, eval_every=2)
    trainer = _trainer(config)
    report = trainer.start_training()
    assert len(report.history) == 1

    program = trainer.build_program(trainer.load_backbone())
    cohort = trainer.load_dev_cohort()
    measured = instrument.score_dev_cohort(
        cohort,
        program.dev_loss_fn,
        params=report.state.params,
        context=program.context,
        example_shape=program.example_shape,
        eval_index=2,
        arm="rollout",
    )
    assert measured["cohort"] == "dev64" and measured["example_count"] == 64
    assert measured["manifest_sha256"] == instrument.J0_DEV64_SHA256
    assert float(measured["metric"]) == pytest.approx(report.history[-1].dev_metric, rel=1e-6)
    assert report.history[-1].train_metric != pytest.approx(report.history[-1].dev_metric, rel=1e-3)


def test_the_dev_instrument_refuses_a_context_the_program_was_not_compiled_against(tmp_path, monkeypatch):
    """Two contexts mean two grids, and two grids mean two estimands — so the scorer refuses one it
    did not close over rather than silently measuring something else."""
    config = _prepared(tmp_path, monkeypatch)
    trainer = _trainer(config)
    program = trainer.build_program(trainer.load_backbone())
    foreign = dataclasses.replace(program.context, guide_scale=1.0)
    with pytest.raises(ValueError, match="was not compiled against"):
        program.dev_loss_fn(program.params, {}, foreign, draws=None)


# =============================================================================================
# 4. Interruption: the T3b-4 oracle, lifted to `start_training`.
# =============================================================================================


def test_an_interrupted_run_reproduces_the_uninterrupted_one_THROUGH_start_training(tmp_path, monkeypatch):
    """T3b-4 proved this for ``run_loop`` with a stub update. Here it is the whole job: a real model,
    a real optimizer, real TFRecords, a real publication, and the production RESUME shape — attempt
    two adopts attempt one's published checkpoint into its OWN fresh root and continues.

    The two halves carry different recipe locks, and deliberately: the interruption is simulated by
    a shorter budget, and ``max_train_steps`` is part of the normalized recipe, so one shared lock
    would (correctly) refuse the second half. The lock's own contract is tested on its own below.

    The loader here honours the cursor convention ``resume_seed`` encodes. The DEPLOYED pipeline does
    not, and the next test says so out loud rather than letting this one imply otherwise.
    """
    whole = _prepared(tmp_path / "whole", monkeypatch)
    uninterrupted = _trainer(whole).start_training()

    first_config = _prepared(tmp_path / "split", monkeypatch, max_train_steps=2)
    first = _trainer(first_config).start_training()
    assert first.steps_run == 2

    second_config = _config(
        tmp_path / "split",
        attempt="att-TWO",
        max_train_steps=4,
        pos_recipe_lock=str(tmp_path / "split" / "lock_two.json"),
    )
    second = _trainer(second_config).start_training()

    assert second.steps_run == 2, "the resumed attempt continues; it does not restart"
    assert first.draw_log + second.draw_log == uninterrupted.draw_log, "the resumed stream diverged"
    assert second.history == uninterrupted.history, "the eval history did not survive the interruption"
    assert second.retained_step == uninterrupted.retained_step
    assert int(second.state.step) == int(uninterrupted.state.step) == 4
    for resumed, straight in zip(jax.tree.leaves(second.state.params), jax.tree.leaves(uninterrupted.state.params)):
        assert np.allclose(np.asarray(resumed), np.asarray(straight), atol=1e-5)

    # The adoption wrote into attempt TWO's OWN root and left attempt ONE's alone (issue #13).
    assert tuple(loop.build_checkpoint_manager(second_config.checkpoint_dir).all_steps()) == (2, 4)
    assert tuple(loop.build_checkpoint_manager(first_config.checkpoint_dir).all_steps()) == (2,)


def test_the_deployed_loader_RESEEDS_rather_than_continuing_and_the_draws_survive_anyway(tmp_path, monkeypatch):
    """The honest boundary of the oracle above, stated as its own executable claim.

    ``_load_dataset`` passes the seed to a file permutation AND a shuffle buffer, so
    ``seed + start_step`` gives a resumed run a DIFFERENT draw of data rather than the continuation
    of the previous one — CLAUDE.md's "Resume is partial", inherited from the settled trainer and not
    a property W2 introduced. What survives the interruption regardless is the part exp_06's
    comparison depends on: the per-step randomness and the decision state. Both are asserted here
    against a loader that behaves the way the deployed one does.
    """
    whole = _prepared(tmp_path / "whole", monkeypatch, cursor=False)
    uninterrupted = _trainer(whole).start_training()

    first_config = _prepared(tmp_path / "split", monkeypatch, cursor=False, max_train_steps=2)
    first = _trainer(first_config).start_training()
    second_config = _config(
        tmp_path / "split",
        attempt="att-TWO",
        max_train_steps=4,
        pos_recipe_lock=str(tmp_path / "split" / "lock_two.json"),
    )
    second = _trainer(second_config).start_training()

    assert first.draw_log + second.draw_log == uninterrupted.draw_log, "the draws are seed-keyed and must survive"
    assert [record.step for record in second.history] == [record.step for record in uninterrupted.history]
    assert second.verdict.stop == uninterrupted.verdict.stop
    assert int(second.state.step) == 4
    # ...and the METRICS differ, because the data did. Asserted rather than tolerated, so that a
    # future loader that DOES continue the cursor makes this test fail and be re-decided.
    assert second.history[-1].train_metric != pytest.approx(uninterrupted.history[-1].train_metric, rel=1e-6)


def test_the_resumed_iterator_is_opened_at_the_RESUMED_seed(tmp_path, monkeypatch):
    """The dataloader cursor is not serialized, so the seed carries the position. A resumed attempt
    that reopened at ``config.seed`` would silently retrain the opening segment's data."""
    _requires_backend()
    _install_import_shims()
    config = _prepared(tmp_path, monkeypatch, max_train_steps=2)
    _trainer(config).start_training()

    seeds: list[int] = []
    _install_settled_trainer(monkeypatch, on_dataset=seeds.append)
    resumed = _config(tmp_path, attempt="att-TWO", max_train_steps=4, pos_recipe_lock=str(tmp_path / "lock_two.json"))
    _trainer(resumed).start_training()
    assert seeds == [int(config.seed) + 2], "the resumed run must open at seed + start_step"


# =============================================================================================
# 5. The gates run BEFORE the load, and every refusal still refuses.
# =============================================================================================


def test_no_job_M1_refused_ever_reaches_the_pipeline_load(tmp_path, monkeypatch):
    """M1's gate is the only door into a training run, so it must fire before the expensive half —
    executed here with a loader that raises if it is so much as entered."""
    _requires_backend()
    _install_import_shims()
    config = _config(tmp_path)
    _write_shards(Path(config.train_data_dir))
    _install_dev_records(monkeypatch, instrument.load_dev_cohort(_DEV).rows)

    def _boom():
        raise AssertionError("the pipeline was loaded for a job M1 did not authorize")

    _install_settled_trainer(monkeypatch, on_load=_boom)
    _authorize(config, microbatch=_MICRO)

    for index, (override, message, authorize) in enumerate(
        (
            ({"pos_microbatch": _LOGICAL}, "M1 did not authorize", False),
            ({"pos_rollout_arm": "one_step"}, "M1 did not authorize", False),
            ({"pos_fit_authorization": ""}, "published fit-probe authorization", False),
            # These two pass M1's gate and must reach their OWN refusal, so each is authorized at its
            # own config: `pos_dev_manifest` is inside the recipe fingerprint, so an override that was
            # not re-measured would (correctly) be refused by the gate before its own check ran.
            ({"pos_dev_manifest": ""}, "never a directory", True),
            ({"pos_recipe_lock": ""}, "must name this run's recipe lock", True),
            ({"global_batch_size_to_load": 8}, "measured a different program", False),
        )
    ):
        candidate = _config(tmp_path, **override)
        if authorize:
            candidate["pos_fit_authorization"] = str(tmp_path / f"m1_{index}.json")
            _authorize(candidate)
        with pytest.raises(ValueError, match=message):
            _trainer(candidate).start_training()


def test_a_loader_that_would_not_yield_the_logical_batch_is_refused_before_the_load(tmp_path, monkeypatch):
    """``checked_logical_batch`` refuses a wrong-width batch at step 1, with the 5B already on
    device and the reservation spent. The width is a CONFIGURATION fact, so the trainer decides it in
    seconds instead — and the refusal names the knob (``per_device_batch_size``) rather than the
    symptom. Measured at a config M1 authorized, so this is the trainer's own check firing.
    """
    _requires_backend()
    _install_import_shims()
    config = _config(tmp_path, global_batch_size_to_load=8)
    _write_shards(Path(config.train_data_dir))
    _install_dev_records(monkeypatch, instrument.load_dev_cohort(_DEV).rows)

    def _boom():
        raise AssertionError("the pipeline was loaded for a job whose loader cannot feed it")

    _install_settled_trainer(monkeypatch, on_load=_boom)
    _authorize(config)

    with pytest.raises(ValueError, match="per_device_batch_size") as refusal:
        _trainer(config).start_training()
    assert "loads 8 examples per step but pos_logical_batch is 4" in str(refusal.value)


def test_the_gates_run_in_the_declared_order_before_anything_expensive():
    """The refusal must come from configuration. Asserted on the ORDER of ``start_training``'s own
    statements, because "the call exists somewhere in the function" is not the property that matters
    — a gate after the load is a gate that costs a reservation."""
    body = _function_node(_TRAINER_PATH.read_text(encoding="utf-8"), "start_training").body
    order = [ast.unparse(statement) for statement in body]
    # LITERALLY first (final re-ruling, LOW 4): relative order let `running_context()` run ahead of
    # the gate. The gate is now one named statement and it is body[0].
    assert "authorized_context" in order[0], f"M1's gate must be the first executable statement, got {order[0]!r}"
    index = {
        name: next(position for position, text in enumerate(order) if name in text)
        for name in (
            "load_dev_cohort",
            "authorized_context",
            "assert_paired_recipe",
            "resume_source",
            "assert_loader_yields_the_logical_batch",
            "load_backbone",
            "build_program",
            "run_loop",
            "publish_this_attempt",
        )
    }
    assert (
        index["authorized_context"]
        < index["load_dev_cohort"]
        < index["assert_paired_recipe"]
        < index["resume_source"]
        < index["assert_loader_yields_the_logical_batch"]
        < index["load_backbone"]
        < index["build_program"]
        < index["run_loop"]
        < index["publish_this_attempt"]
    ), order
    assert not [statement for statement in body if isinstance(statement, ast.Raise)], "the boundary is gone"


def test_a_divergent_second_arm_still_refuses_to_start_after_the_wiring(tmp_path, monkeypatch):
    """Yixun's decision-1 control: the recipe lock is published by the first arm and adopted by the
    second, and a run that differs anywhere outside the arm and its destinations does not start.

    The divergence chosen is ``eval_every``, and deliberately: plan §3d says the cadence is
    "identical for R-B and matched-C0", and it is one of the keys M1's fingerprint EXCLUDES (a
    cadence does not change what one step compiles to), so the authorization still matches and the
    refusal has to come from the lock rather than from the gate before it.
    """
    config = _prepared(tmp_path, monkeypatch)
    _trainer(config).start_training()

    divergent = _config(tmp_path, attempt="att-TWO", eval_every=4)
    with pytest.raises(ValueError, match="DIFFERENT recipe") as refusal:
        _trainer(divergent).start_training()
    assert "eval_every" in str(refusal.value), "the refusal must name the key that differs"


# =============================================================================================
# 6. The seams, pinned against the file they stand in for.
# =============================================================================================


def test_every_seam_the_trainer_calls_exists_on_the_REAL_settled_trainer():
    """The stub above cannot prove these names exist. This does — and it also pins that the trainer
    reaches each of them, so a rename on either side fails here rather than on a TPU."""
    from maxdiffusion.trainers.wan_pos_rollout_trainer import WanPosRolloutTrainer

    settled = _SETTLED_PATH.read_text(encoding="utf-8")
    for method in ("_load_wan_pipeline", "_compute_null_context", "_create_scheduler", "_load_dataset"):
        assert f"def {method}(" in settled, f"exp_06 calls {method}, which no longer exists"
    backbone = inspect.getsource(pos_rollout_update.load_backbone)
    for method in ("_load_wan_pipeline", "_compute_null_context", "_create_scheduler"):
        assert f".{method}(" in backbone, "the SHARED loader must reach the settled seam"
    assert "._load_dataset(" in inspect.getsource(WanPosRolloutTrainer.batches)
    # ...and the trainer reaches them only through that shared loader, so M1 cannot take another way.
    assert "load_backbone(self.config)" in inspect.getsource(WanPosRolloutTrainer.load_backbone)
    # ...and the settled file is READ, never edited: exp_06 owns no copy of it.
    assert "class WanTI2VSideAdapterTrainer" not in _TRAINER_PATH.read_text(encoding="utf-8")


def test_the_pipeline_is_loaded_under_the_deployed_axis_rules_and_the_scheduler_is_the_authority():
    """Two production facts the wiring must not paraphrase: ``_load_wan_pipeline`` performs the
    ``axis_rules`` load itself, and the sigma limits/timestep count come from the SCHEDULER — reading
    them off the config is the issue-#11 family that has now claimed three jobs in this campaign."""

    settled = _SETTLED_PATH.read_text(encoding="utf-8")
    loader = _function_node(settled, "_load_wan_pipeline")
    assert "axis_rules" in ast.unparse(loader), "the settled loader no longer enters the deployed rules"

    grid = inspect.getsource(pos_rollout_update.arm_context)
    for attribute in (
        "scheduler.config.sigma_min",
        "scheduler.config.sigma_max",
        "scheduler.config.num_train_timesteps",
    ):
        assert attribute in grid, f"the grid must come from {attribute}"
    assert "config.num_train_timesteps" not in grid.replace("scheduler.config.num_train_timesteps", "")
    # ...and every step runs inside the mesh AND the rules, as the settled loop wraps its own.
    # Every step, every score and the construction itself run inside the SHARED scope, which is the
    # mesh AND the rules together -- a bare mesh is what let M1 compile a different sharding.
    scope = inspect.getsource(pos_rollout_update.program_scope)
    assert "axis_rules" in scope and "with mesh, rules" in scope
    program = inspect.getsource(pos_rollout_update.build_training_program)
    # Battery Z04 SURVIVED a `>= 3` count, because the finalizer names the scope FOUR times and the
    # mutant only had to drop one. Counting is the wrong assertion: the property is that the
    # CONSTRUCTION happens inside the scope, so that is what is checked, on the AST.
    finalizer = ast.parse(textwrap.dedent(program)).body[0]
    constructed_in_scope = [
        node
        for node in ast.walk(finalizer)
        if isinstance(node, ast.With)
        and "program_scope" in ast.unparse(node.items[0].context_expr)
        and "build_adapter_stack" in ast.unparse(node)
        and "build_optimizer" in ast.unparse(node)
    ]
    assert constructed_in_scope, "the adapter and optimizer must be BUILT inside the deployed scope"
    assert program.count("program_scope(config, backbone.mesh)") >= 4, "build, step, score and the exported scope"

    # ...and M1's weights seam reaches the backbone through the SHARED loader (battery Z12: a seam
    # returning a hand-built record survived, because nothing executes that production-only path).
    assert "load_backbone" in _calls_in(probe.ProductionModelSource.load), "M1's seam must call the shared loader"
    assert "LoadedBackbone(" not in inspect.getsource(probe.ProductionModelSource.load)


def test_the_trainer_reads_no_key_the_rollout_yaml_does_not_declare(tmp_path, monkeypatch):
    """The F3a-fix contract test, applied to this round's file: every config key the wiring reads is
    declared by the checked-in YAML, so an undeclared read fails the moment it is written."""
    from maxdiffusion.trainers import wan_pos_rollout_trainer

    declared = set(yaml.safe_load(_CONFIG_PATH.read_text()))
    source = _TRAINER_PATH.read_text(encoding="utf-8")
    read: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
            if node.value.attr == "config" and getattr(node.value.value, "id", "") == "self":
                read.add(node.attr)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "optional_config_value":
            read.add(ast.literal_eval(node.args[1]))
    assert read, "the AST scan found no config reads at all, so it is proving nothing"
    assert read <= declared, sorted(read - declared)
    for node in ast.walk(ast.parse(inspect.getsource(wan_pos_rollout_trainer))):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "getattr":
            assert len(node.args) < 3, f"issue #11: {ast.unparse(node)}"


# =============================================================================================
# 7. W3 — THE MULTI-DEVICE ORACLE: M1 compiles the trainer's exact sharded program.
#
# The final review's BLOCKER: *shared factories are not a shared PROGRAM.* The trainer replicated its
# adapter parameters and compiled under `logical_axis_rules`; M1 did neither, and the Wan blocks
# translate logical axes during the forward — so M1 could compile different activation shardings and
# report a different per-device HBM peak, which is the one number it exists to authorize. Measured on
# an 8-device CPU mesh before the fix: trainer leaves `NamedSharding(mesh, P())`, M1 leaves
# `SingleDeviceSharding(CpuDevice(0))`, and M1's null context zeros against the loader's real one.
#
# One device cannot see any of that, so this runs in a SUBPROCESS with
# `XLA_FLAGS=--xla_force_host_platform_device_count=8` (the flag is read at backend initialization,
# before any test in this process imported jax). The oracle compares leaf VALUES, leaf SHARDINGS, the
# compiled step's INPUT/OUTPUT shardings, and the deployed context both sides closed over.
# =============================================================================================

_ORACLE = """
import json, sys, types
import jax, jax.numpy as jnp, numpy as np, yaml
from flax import nnx
from maxdiffusion.tests.worklogs_yixun.test_pos_rollout_fit_probe import _install_import_shims

_install_import_shims()
from maxdiffusion import pos_rollout_fit_probe as probe
from maxdiffusion.models.wan.transformers.transformer_wan import WanModel

CONFIG, DEVICES = sys.argv[1], int(sys.argv[2])
_C, _F, _H, _W, _TEXT, _NULL = 48, 9, 12, 20, 32, 8
#: The DEPLOYED logical-axis rules, read from the same YAML the loader reads (F3d, MINOR-1).
LOGICAL_AXIS_RULES = yaml.safe_load(open(CONFIG).read())["logical_axis_rules"]
_LOGICAL, _MICRO, _STEPS, _K = DEVICES, DEVICES, 4, 2


class Config(dict):
    def __getattr__(self, key):
        if key not in self:
            raise ValueError(f"Key {key} not in config")
        return self[key]

    def get_keys(self):
        return dict(self)


def mesh():
    return jax.sharding.Mesh(
        np.array(jax.devices()).reshape(1, DEVICES, 1, 1), ("data", "fsdp", "context", "tensor")
    )


TRANSFORMER = []


def transformer():
    if not TRANSFORMER:
        with mesh():
            model = WanModel(
                rngs=nnx.Rngs(jax.random.key(0)), num_attention_heads=2, attention_head_dim=8,
                in_channels=_C, out_channels=_C, text_dim=_TEXT, freq_dim=16, ffn_dim=32,
                num_layers=1, attention="dot_product", rope_max_seq_len=64, scan_layers=False,
                dtype=jnp.float32, weights_dtype=jnp.float32)
            # COMMIT the weights to the mesh, as the deployed pipeline does when it loads them
            # (F3b, review MINOR-1). Without this the fake backbone's leaves are UNCOMMITTED
            # single-device arrays, so "the compiled contract equals the sharding the weights
            # actually carry" would be checked against a placement production never has, and the
            # frozen-input assertions below would be testing the fixture rather than the program.
            # THE LOADER'S OWN MAPPING, not a rule of our own (F3d, review MINOR-1).
            # F3c committed by a synthetic rule -- P("fsdp") when the leading dim divided the mesh --
            # and only 18 of 42 placements matched what production actually produces: biases and
            # kernels were frequently split on the wrong axis. An oracle that asserts the compiled
            # contract against a sharding production never creates is checking a fiction. This is
            # `wan_pipeline.py`'s own three lines (`get_partition_spec` -> `logical_to_mesh_sharding`
            # over the config's `logical_axis_rules`), so `checks["frozen"]` now asserts the LOADER's
            # contract.
            from flax import linen as nn_linen

            _graphdef, state = nnx.split(model)
            logical_spec = nnx.get_partition_spec(state)
            state_sharding = nn_linen.logical_to_mesh_sharding(
                logical_spec, mesh(), LOGICAL_AXIS_RULES
            )
            nnx.update(
                model,
                jax.tree.map(lambda leaf, sh: jax.device_put(leaf, sh), state, state_sharding),
            )
            TRANSFORMER.append(model)
    return TRANSFORMER[0]


class _Scheduler:
    def __init__(self):
        self.config = types.SimpleNamespace(sigma_min=0.0, sigma_max=1.0, num_train_timesteps=1000)


def install(cfg):
    model, m = transformer(), mesh()

    class _Pipeline:
        transformer = model
        mesh = m

    class _Trainer:
        def __init__(self, c):
            self.config = c

        def _load_wan_pipeline(self):
            return _Pipeline()

        def _compute_null_context(self, pipeline, mesh_):
            # NON-ZERO on purpose: M1 used to close over zeros, and a zero-vs-zero comparison could
            # not tell a shared context from two coincidentally equal ones.
            return jnp.full((1, _NULL, _TEXT), 0.25, jnp.float32)

        def _create_scheduler(self):
            return _Scheduler(), None

        def _load_dataset(self, *a, **k):
            raise AssertionError("the oracle compares PROGRAMS, not data")

    stub = types.ModuleType("maxdiffusion.trainers.wan_ti2v_side_adapter_trainer")
    stub.WanTI2VSideAdapterTrainer = _Trainer
    sys.modules["maxdiffusion.trainers.wan_ti2v_side_adapter_trainer"] = stub


def leaves(tree):
    return [(np.asarray(x).tolist(), str(getattr(x, "sharding", "<none>"))) for x in jax.tree.leaves(tree)]


def main():
    values = yaml.safe_load(open(CONFIG).read())
    values.update({
        "text_dim": _TEXT, "wan_max_sequence_length": _NULL, "action_tokens": 4, "action_hidden": 16,
        "action_heads": 2, "pre_context_tokens": 4, "pre_context_heads": 2, "side_adapter_layers": "0",
        "side_adapter_hidden": 16, "side_adapter_heads": 2, "side_adapter_sampling_steps": _STEPS,
        "weights_dtype": "float32", "activations_dtype": "float32", "pos_logical_batch": _LOGICAL,
        "pos_microbatch": _MICRO, "pos_rollout_k": _K, "max_train_steps": 4,
        "global_batch_size_to_load": _LOGICAL,
    })
    cfg = Config(values)
    install(cfg)

    from maxdiffusion.trainers.wan_pos_rollout_trainer import WanPosRolloutTrainer

    trainer = WanPosRolloutTrainer(cfg)
    left = trainer.build_program(trainer.load_backbone())

    class _WeightsSeam:
        def load(self, c):
            from maxdiffusion.pos_rollout_update import load_backbone

            return load_backbone(c)

    right = probe.build_probe_program(cfg, probe.FitCell("rollout", _MICRO, _K), model_source=_WeightsSeam())

    # W5: each side lowers a COMPLETE operand tuple OF ITS OWN — params, optimizer state, batch and
    # draws — and the compiled contract is checked against expectations this test CONSTRUCTS from the
    # mesh, not from either path. The W4 version still shared params/opt_state/draws and compared the
    # two lowerings to each other, so a defect that moved both sides, or one that lived in an operand
    # neither side owned, was invisible.
    from jax.sharding import NamedSharding, PartitionSpec

    from maxdiffusion.pos_rollout_stream import draw_step_for_batch
    from maxdiffusion.pos_rollout_update import (
        draws_to_arrays,
        place_step_inputs,
        production_batch_sharding,
        replicated_sharding,
    )

    m = jax.tree.leaves(left.params)[0].sharding.mesh
    # INDEPENDENTLY CONSTRUCTED, in this test, from the mesh alone. Nothing below reads either path's
    # opinion of what the right sharding is.
    expected_replicated = NamedSharding(m, PartitionSpec())
    expected_batch = NamedSharding(m, PartitionSpec(m.axis_names))

    # The trainer's operands: a batch as the DEPLOYED loader hands it, and draws the stream seam
    # produced for that batch. M1's operands are its own synthetic zeros and its own draws.
    trainer_batch = tuple(
        {key: jax.device_put(value, expected_batch) for key, value in part.items()} for part in right.batch
    )
    logical = sum(int(jnp.shape(part["z_video"])[0]) for part in trainer_batch)
    whole = {
        key: jnp.concatenate([part[key] for part in trainer_batch], axis=0) for key in trainer_batch[0]
    }
    _, trainer_draw_parts, _ = draw_step_for_batch(
        whole,
        seed=int(cfg.seed),
        global_step=1,
        logical_batch=logical,
        microbatch=int(cfg.pos_microbatch),
        num_steps=_STEPS,
        k_b=_K,
    )
    trainer_draws = tuple(draws_to_arrays(part) for part in trainer_draw_parts)

    operands = {
        "trainer": (left.params, left.opt_state, trainer_batch, trainer_draws),
        "m1": (right.params, right.opt_state, right.batch, right.draws),
    }
    lowered = {}
    for name, program in (("trainer", left), ("m1", right)):
        with program.scope():
            lowered[name] = program.step.lower(*operands[name]).compile()

    def _compiled_reference(program, operand):
        # F3: `step.lower` still takes the four operands above, but the program it lowers takes
        # `frozen_state` as its SECOND POSITIONAL argument, so the tuple the compiled input shardings
        # are read against carries it between `params` and `opt_state`.
        #
        # EACH SIDE'S OWN frozen state, never one side's for both. The two do reach the backbone
        # through the shared loader, but "they must be the same" is the claim this oracle exists to
        # TEST, and an oracle that assumes it cannot fail when it stops being true -- the exact
        # symmetric-move blind spot W3 and W4 closed for the other operands.
        return (operand[0], program.frozen.state, operand[1], operand[2], operand[3])

    compiled_operands = {
        name: _compiled_reference(program, operands[name]) for name, program in (("trainer", left), ("m1", right))
    }

    def _absolute(compiled, operand):
        # Every compiled input sharding against the INDEPENDENT expectation for its own argument.
        args = compiled.input_shardings[0]
        checks = {}
        for index, (name, want) in (
            (0, ("params", expected_replicated)),
            (2, ("opt", expected_replicated)),
            (3, ("batch", expected_batch)),
        ):
            leaves_ = jax.tree.leaves(args[index])
            shapes = jax.tree.leaves(operand[index])
            checks[name] = bool(leaves_) and all(
                got.is_equivalent_to(want, np.ndim(ref)) for got, ref in zip(leaves_, shapes)
            )
        # INDEX 1 IS THE FROZEN BACKBONE, and skipping it was the gap (F3b, review MINOR-1). It is
        # asserted against the sharding the state ACTUALLY CARRIES rather than against a constant
        # expectation, because that is the property worth having: the compiled program must accept
        # the backbone exactly where the loader already put it. Any divergence here is a per-call
        # reshard of the whole 5B -- the regression a "10 GB copy" would show up as -- and the
        # reviewer demonstrated that with a sharded-backbone fake this oracle stayed green purely
        # because index 1 was omitted.
        frozen_leaves = jax.tree.leaves(args[1])
        frozen_refs = jax.tree.leaves(operand[1])
        checks["frozen"] = bool(frozen_leaves) and all(
            got.is_equivalent_to(ref.sharding, np.ndim(ref)) for got, ref in zip(frozen_leaves, frozen_refs)
        )
        draw_leaves = jax.tree.leaves(args[4])
        draw_refs = jax.tree.leaves(operand[4])
        checks["draws"] = bool(draw_leaves) and all(
            got.is_equivalent_to(expected_replicated if position % 4 < 2 else expected_batch, np.ndim(ref))
            for position, (got, ref) in enumerate(zip(draw_leaves, draw_refs))
        )
        return checks

    def _contract(params, opt_state, batch, draws):
        params_p, opt_p, batches_p, draws_p = place_step_inputs(
            m, params=params, opt_state=opt_state, micro_batches=batch, micro_draws=draws
        )
        return {
            "params": all(leaf.sharding == expected_replicated for leaf in jax.tree.leaves(params_p)),
            "opt": all(leaf.sharding == expected_replicated for leaf in jax.tree.leaves(opt_p)),
            "batch": bool(jax.tree.leaves(batches_p))
            and all(leaf.sharding == expected_batch for leaf in jax.tree.leaves(batches_p)),
            "draws": bool(draws_p)
            and all(
                value.sharding == (expected_replicated if index < 2 else expected_batch)
                for part in draws_p
                for index, value in enumerate(part)
            ),
        }

    # The SHARED scorer, lowered on each side's OWN scoring operands (W5 item 3).
    def _eval_operands(program, batch, draws):
        one = {key: value[:1] for key, value in batch[0].items()}
        one_draws = tuple(
            value[:1] if getattr(value, "ndim", 0) and value.shape[0] == batch[0]["z_video"].shape[0] else value
            for value in draws[0]
        )
        return program.params, one, one_draws

    def _scorer_reference(program, operand):
        # F3, on the scorer: `score.lower` takes the three operands above and threads `frozen_state`
        # second, so the reference tuple its input shardings are read against carries it there too --
        # and it is THIS program's own state, for the reason `_compiled_reference` gives.
        return (operand[0], program.frozen.state, operand[1], operand[2])

    scorers = {}
    for name, program, batch, draws in (
        ("trainer", left, trainer_batch, trainer_draws),
        ("m1", right, right.batch, right.draws),
    ):
        with program.scope():
            scorers[name] = program.score.lower(*_eval_operands(program, batch, draws)).compile()

    # Battery U02/U03: `place_step_inputs` was only ever handed operands that were ALREADY placed
    # (the finalizer replicates params and optimizer state at construction), so dropping its
    # placement was a no-op and survived. The contract has to be tested on operands that VIOLATE it,
    # or the test is measuring the coincidence rather than the function.
    def _unplaced(tree):
        return jax.tree.map(lambda leaf: jax.device_put(leaf, jax.devices()[0]), tree)

    stray = _contract(
        _unplaced(left.params),
        _unplaced(left.opt_state),
        tuple(_unplaced(part) for part in right.batch),
        tuple(_unplaced(part) for part in right.draws),
    )

    def _scorer_absolute(compiled, operand):
        # Every scorer input against an INDEPENDENTLY constructed expectation. The DEV instrument
        # feeds batch-one host arrays and the parameters are the replicated adapter tree, so those
        # inputs are replicated -- measured, then asserted, rather than compared side to side.
        #
        # ...but NOT the frozen state at argument 1 (F3b, review MINOR-1). "Everything is replicated"
        # was true only while the backbone was invisible to this oracle; it is not the load-time
        # contract for a SHARDED backbone, and asserting it would either fail on a real FSDP load or,
        # worse, pass while a per-call reshard of the 5B went unnoticed. Argument 1 is therefore
        # checked against the sharding its own leaves carry, exactly as the step oracle does.
        per_argument = compiled.input_shardings[0]
        frozen_leaves = jax.tree.leaves(per_argument[1])
        frozen_refs = jax.tree.leaves(operand[1])
        frozen_ok = bool(frozen_leaves) and all(
            got.is_equivalent_to(ref.sharding, np.ndim(ref)) for got, ref in zip(frozen_leaves, frozen_refs)
        )
        others = [value for index, value in enumerate(per_argument) if index != 1]
        other_refs = [value for index, value in enumerate(operand) if index != 1]
        leaves_ = jax.tree.leaves(others)
        refs = jax.tree.leaves(other_refs)
        return (
            frozen_ok
            and bool(leaves_)
            and all(got.is_equivalent_to(expected_replicated, np.ndim(ref)) for got, ref in zip(leaves_, refs))
        )

    def _leafwise_equivalent(one, other, operand):
        # A SEMANTIC side-to-side comparison, replacing the string gates: two shardings that print
        # differently can be the same placement, and two that print identically under a different
        # normalization are not evidence.
        left_leaves = jax.tree.leaves(one.input_shardings[0])
        right_leaves = jax.tree.leaves(other.input_shardings[0])
        refs = jax.tree.leaves(operand)
        return len(left_leaves) == len(right_leaves) and all(
            a.is_equivalent_to(b, np.ndim(ref)) for a, b, ref in zip(left_leaves, right_leaves, refs)
        )

    report = {
        "devices": jax.device_count(),
        "contract_repairs_unplaced_operands": stray,
        "param_values_equal": leaves(left.params) == leaves(right.params),
        "opt_values_equal": leaves(left.opt_state) == leaves(right.opt_state),
        "context_equal": bool(
            np.array_equal(np.asarray(left.context.null_context), np.asarray(right_context(right).null_context))
            and np.array_equal(np.asarray(left.context.sigmas), np.asarray(right_context(right).sigmas))
            and np.array_equal(np.asarray(left.context.timesteps), np.asarray(right_context(right).timesteps))
        ),
        "null_context_is_nonzero": float(np.abs(np.asarray(left.context.null_context)).sum()) > 0.0,
        "opt_leaf_count": len(jax.tree.leaves(left.opt_state)),
        "trainer_contract": _contract(*operands["trainer"]),
        "m1_contract": _contract(*operands["m1"]),
        # ABSOLUTE, per side, against expectations built in this test.
        "trainer_absolute": _absolute(lowered["trainer"], compiled_operands["trainer"]),
        "m1_absolute": _absolute(lowered["m1"], compiled_operands["m1"]),
        # Each side really did lower a COMPLETE tuple of its own, and the tuples really differ.
        "operand_trees_match": str(jax.tree.structure(operands["trainer"]))
        == str(jax.tree.structure(operands["m1"])),
        # W5b: the BATCH is the operand the whole blocker was about, and it was the one field the
        # ownership check omitted -- rewiring the trainer's lowering to `right.batch` left all three
        # assertions green. Every element of both tuples is pinned now.
        "operands_are_own": (
            operands["trainer"][0] is left.params
            and operands["trainer"][1] is left.opt_state
            and operands["trainer"][2] is trainer_batch
            and operands["trainer"][3] is trainer_draws
            and operands["m1"][0] is right.params
            and operands["m1"][1] is right.opt_state
            and operands["m1"][2] is right.batch
            and operands["m1"][3] is right.draws
        ),
        # ...and the difference is read off the operands ACTUALLY LOWERED, not off the standalone
        # locals, which a rewiring would have left untouched and still-correct.
        "operands_differ_before_placement": (
            jax.tree.leaves(operands["trainer"][2])[0].sharding == expected_batch
            and jax.tree.leaves(operands["m1"][2])[0].sharding != expected_batch
        ),
        "step_leafwise_equivalent": _leafwise_equivalent(
            lowered["trainer"], lowered["m1"], compiled_operands["m1"]
        ),
        # Both shared scorers LOWERED and compared, plus M1's arity on M1's OWN parameters.
        "score_absolute_trainer": _scorer_absolute(
            scorers["trainer"], _scorer_reference(left, _eval_operands(left, trainer_batch, trainer_draws))
        ),
        "score_absolute_m1": _scorer_absolute(
            scorers["m1"], _scorer_reference(right, _eval_operands(right, right.batch, right.draws))
        ),
        "score_leafwise_equivalent": _leafwise_equivalent(
            scorers["trainer"], scorers["m1"], _scorer_reference(right, _eval_operands(right, right.batch, right.draws))
        ),
        "score_returns_aux": _score_arity(left, right),
        "m1_score_returns_aux": _probe_score_arity(right),
    }
    print("ORACLE " + json.dumps(report))


def right_context(program):
    return program.context


def _score_arity(left, right):
    # The shared scorer returns (loss, aux); a scalar-only one lets XLA prune the aux work.
    import jax

    with left.scope():
        out = jax.eval_shape(
            lambda p, b, d: left.loss_fn(p, b, left.context, frozen_state=left.frozen.state, draws=_draws(d)),
            left.params,
            {k: v[:1] for k, v in right.batch[0].items()},
            tuple(x[:1] if getattr(x, "ndim", 0) else x for x in right.draws[0]),
        )
    return isinstance(out, tuple) and len(out) == 2 and bool(out[1])


def _draws(values):
    from maxdiffusion.pos_rollout_update import draws_from_arrays

    return draws_from_arrays(values)


def _probe_score_arity(right):
    # Executes M1's OWN score callable on M1's OWN parameters (W5: it used the trainer's).
    import jax

    with right.scope():
        out = jax.eval_shape(right.score, right.params, right.eval_batch, right.eval_draws)
    return isinstance(out, tuple) and len(out) == 2 and bool(out[1])


main()
"""


def _run_oracle(tmp_path, devices=8):
    script = tmp_path / "oracle.py"
    script.write_text(_ORACLE)
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "PYTHONPATH": str(_PACKAGE_ROOT.parents[0]),
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
        "TF_CPP_MIN_LOG_LEVEL": "3",
    }
    proc = subprocess.run(
        [sys.executable, str(script), str(_CONFIG_PATH), str(devices)],
        capture_output=True,
        text=True,
        timeout=1800,
        env=environment,
    )
    line = next((l for l in proc.stdout.splitlines() if l.startswith("ORACLE ")), None)
    assert line, proc.stdout[-4000:] + proc.stderr[-4000:]
    return json.loads(line[len("ORACLE ") :])


def test_M1_compiles_the_TRAINERS_EXACT_sharded_program_on_a_multi_device_mesh(tmp_path):
    """The W3/W4 oracle. Eight devices, each path lowering with ITS OWN inputs, and the assertions
    are ABSOLUTE production specs — the shape the W3 version got wrong by passing M1's batch to both
    sides and proving only that one function lowers the same way on one set of operands."""
    _requires_backend()
    report = _run_oracle(tmp_path)

    assert report["devices"] == 8, "the oracle needs a real multi-device mesh to mean anything"
    assert report["opt_leaf_count"] > 0

    # Each side lowered a COMPLETE tuple OF ITS OWN, and the two tuples genuinely differ (W5 item 1).
    assert report["operands_are_own"], "each side must lower its own params, opt_state, batch AND draws"
    assert report["operand_trees_match"], "the two operand tuples must have the same tree structure"
    assert report["operands_differ_before_placement"], "the two operand sets must genuinely differ"

    # ABSOLUTE: the compiled contract on each side against expectations this TEST constructed from
    # the mesh — not against the other side's opinion of it (W5 item 2).
    for side in ("trainer_absolute", "m1_absolute"):
        contract = report[side]
        assert contract["params"], f"{side}: parameters must be compiled against a replicated sharding"
        assert contract["opt"], f"{side}: every optimizer-state leaf must be compiled replicated"
        assert contract["batch"], f"{side}: the batch must be compiled against the loader's example split"
        assert contract["draws"], f"{side}: scalar supports replicated, per-example draws split like the data"
        # F3b computed this and never asserted it, so a false result was silently ignored -- the
        # oracle reported on the frozen input without ever gating on it (F3c, review MINOR-1).
        assert contract["frozen"], (
            f"{side}: the frozen backbone must be compiled against the sharding it already carries; "
            f"any difference is a per-call reshard of the whole 5B"
        )
    for side in ("trainer_contract", "m1_contract"):
        contract = report[side]
        assert all(contract.values()), f"{side}: the placement contract is {contract}"
    # ...and the contract PLACES rather than merely passing through: fed operands that violate it,
    # every group comes back correct (battery U02/U03).
    stray = report["contract_repairs_unplaced_operands"]
    assert all(stray.values()), f"place_step_inputs did not place unplaced operands: {stray}"

    # ...and only THEN is agreement between the two sides meaningful.
    assert report["param_values_equal"], "the two paths' adapter parameters differ in value or sharding"
    assert report["opt_values_equal"], "the two paths' optimizer states differ in value or sharding"
    assert report["context_equal"], "the two paths compiled against different grids or null contexts"
    assert report["null_context_is_nonzero"], "a zero context would make the comparison vacuous"

    # The SHARED DEV scorer: ABSOLUTE per argument on each side, then semantic agreement (W5b item 2).
    assert report["score_absolute_trainer"], "the trainer's scorer must compile against replicated inputs"
    assert report["score_absolute_m1"], "M1's scorer must compile against the same replicated contract"
    assert report["score_leafwise_equivalent"], "the two scorers' input placements differ semantically"
    assert report["score_returns_aux"], "a scalar-only scorer lets XLA prune the aux work M1 is timing"
    assert report["m1_score_returns_aux"], "M1's own scorer, on M1's own parameters, must carry the aux"

    assert report["step_leafwise_equivalent"], "the compiled step's input placements differ semantically"
