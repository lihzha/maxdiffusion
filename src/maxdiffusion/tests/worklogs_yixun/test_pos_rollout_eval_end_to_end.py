"""exp_06 `rollout_adapter` — the END-TO-END evaluation test (review pass 2's launch precondition).

**Why this file exists, in the reviewer's own words:** *"the tests manufacture ideal scalar tables and
never exercise table production, artifact round-tripping, GCS I/O, real certificate consumption, VAE
decode layout, Orbax restore templates, or bf16/sharding. A small end-to-end fake-model artifact test
should precede the first real checkpoint smoke run."* The Planner adopted it as a **launch
precondition**, so this runs the protocol — restore → rollout → decode → summarize → certificate →
gate → TEST door — before any real checkpoint is ever loaded.

**What is REAL here.** A real ``WanModel`` and a real ``NNXWanSideAdapterStack`` in its ``pre_context``
configuration (T3a's fixtures, reused rather than re-declared); a real Orbax checkpoint tree written
and restored through the production template; real TFRecord shards written and read by the anchor's
own reader; the real 25-step deployed sigma grid; the real ``DeviceBackend``; the real
``rollout_prediction``/``sample_metrics``/``summarize_samples``; the real digest-bound artifacts,
published and re-loaded through a ``gs://`` URI; the real cohorts, the real derangement derived from
the records' own action bytes, the real table producer, and both real gates.

**What is not, and why.** The VAE is a deterministic stand-in: it needs the 5B pipeline's weights,
and what this test must exercise is the LAYOUT its output has to have — ``(batch, frames, H, W, 3)``,
both sides through one seam — which the stand-in reproduces exactly. And the second half's device
compute is a stub, because 64 examples × seven condition tables × a 25-step rollout is a TPU-shaped
amount of work: what the second half is testing is the ORCHESTRATION (restores, cohorts, production,
publication, gate consumption, the TEST door), while the first half runs one complete measurement
through the real model. Both halves are named for what they cover.

**The first half's most useful assertion is the one that fails the anchor.** A randomly initialised
tiny adapter does not reproduce the deployed 0.2946, so the verdict is ``reproduced=False`` — and the
test then shows that every later phase REFUSES to run. That is plan §3c's ordering, executed rather
than described.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion import eval_wan_pos_rollout as anchor
from maxdiffusion import pos_rollout_dev_instrument as instrument
from maxdiffusion import pos_rollout_gates as gates
from maxdiffusion import pos_rollout_loop as loop
from maxdiffusion.models.wan.overfit100_sampling import overfit100_sampler_grid
from maxdiffusion.pos_rollout_step import build_cfg_velocity_fn
from maxdiffusion.tests.worklogs_yixun.test_pos_rollout_gates import _install_records, _StubBackend
from maxdiffusion.tests.worklogs_yixun.test_pos_rollout_step import _mesh_context, _tiny_cfg_stack

_MANIFEST_DIR = Path(anchor.__file__).resolve().parents[2] / "docs" / "worklogs_yixun"
_MANIFEST_DIR = _MANIFEST_DIR / "exp_04_null_adapter_claude" / "j0_manifests"
_DEV = str(_MANIFEST_DIR / "dev64.json")
_TEST = str(_MANIFEST_DIR / "test64.json")
_SHA = "a" * 40

# The tiny fixture's geometry (exp_05's `_tiny_pre_context`, via T3a).
_C, _TEXT, _F, _H, _W, _ACTION_LEN, _ACTION_DIM = 4, 32, 2, 4, 6, 4, 7
_GUIDE, _STEPS = 5.0, 25


class _Config(dict):
    """A stand-in for ``pyconfig.HyperParameters``: attribute reads, ``ValueError`` on an unknown key.

    The ``ValueError`` matters — it is why ``getattr(config, key, default)`` never falls back and has
    killed two TPU jobs in this campaign (issue #11). A dict-like fake that raised ``AttributeError``
    would quietly let that bug back in.
    """

    def __getattr__(self, key):
        if key not in self:
            raise ValueError(f"Key {key} not in config")
        return self[key]

    def get_keys(self):
        return dict(self)


def _decode(latents):
    """A deterministic stand-in for ``_decode_latents_to_video(_denormalize_latents(x))``.

    The VAE needs the pipeline's weights; what this path must get right is the LAYOUT — one seam, both
    sides, ``(batch, frames, H, W, 3)`` with enough spatial extent for SSIM's 7-pixel window.
    """
    values = np.asarray(latents, np.float32)
    batch, channels, frames, height, width = values.shape
    frame = values.mean(axis=1)  # (batch, frames, H, W)
    frame = np.repeat(np.repeat(frame, 2, axis=2), 2, axis=3)
    frame = 1.0 / (1.0 + np.exp(-frame))  # into [0, 1], as a decoded video is
    del channels, height, width
    return jnp.asarray(np.repeat(frame[..., None], 3, axis=-1).reshape(batch, frames, 2 * 4, 2 * 6, 3))


def _backend(*, params=None):
    """The REAL ``DeviceBackend`` over the real tiny transformer + adapter stack."""
    transformer, adapters = _tiny_cfg_stack()
    rules, mesh = _mesh_context()
    with rules, mesh:
        make_velocity_fn, adapter_params = build_cfg_velocity_fn(transformer, adapters)
    from flax import nnx

    frozen = nnx.split(transformer, nnx.Param, ...)

    def velocity_for(bound, actions, adapter_enabled):
        if adapter_enabled:
            return make_velocity_fn(bound, actions=actions, guide_scale=_GUIDE)
        model = nnx.merge(*frozen)

        def velocity_fn(hidden_states, timestep, encoder_hidden_states):
            return model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                deterministic=True,
            )

        return velocity_fn

    sigmas, timesteps = overfit100_sampler_grid(
        num_inference_steps=_STEPS, flow_shift=5.0, sigma_min=0.0, sigma_max=1.0, num_train_timesteps=1000
    )
    template = loop.RolloutTrainState(params=adapter_params, opt_state={"mu": jnp.zeros((2,), jnp.float32)}, step=0)
    return anchor.DeviceBackend(
        velocity_for=velocity_for,
        decode_fn=_decode,
        sigmas=sigmas,
        timesteps=timesteps,
        context=jnp.zeros((1, 7, _TEXT), jnp.float32),
        guide_scale=_GUIDE,
        template=template,
        params=params if params is not None else adapter_params,
        scope=_mesh_context,
    )


def _write_checkpoint(root, *, step, template, arm="rollout", dev_metric=0.5):
    """A REAL Orbax tree at ``root``, written through the production saver."""
    manager = loop.build_checkpoint_manager(root)
    state = loop.RolloutTrainState(params=template.params, opt_state=template.opt_state, step=step)
    loop.save_checkpoint(
        manager,
        state,
        dev_metric=dev_metric,
        history=[loop.EvalRecord(step=step, dev_metric=dev_metric, train_metric=1.0)],
        arm=arm,
        k_b=2,
    )
    return root


def _write_selection(root, *, step, template, arm, dev_metric):
    manager = loop.build_selection_manager(root)
    state = loop.RolloutTrainState(params=template.params, opt_state=template.opt_state, step=step)
    loop.save_checkpoint(
        manager,
        state,
        dev_metric=dev_metric,
        history=[loop.EvalRecord(step=step, dev_metric=dev_metric, train_metric=1.0)],
        arm=arm,
        k_b=2,
    )


def _write_records(directory: Path, names):
    """REAL TFRecord shards in the deployed schema, so the anchor's own reader is exercised."""
    import tensorflow as tf

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "val-00000.tfrecord"
    with tf.io.TFRecordWriter(str(path)) as writer:
        for index, name in enumerate(names):
            value = 0.05 * (index + 1)
            z_i0 = np.full((_C, 1, _H, _W), value, np.float16)
            z_video = np.full((_C, _F, _H, _W), value, np.float16)
            actions = np.full((_ACTION_LEN, _ACTION_DIM), value, np.float32)
            feature = {
                "name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode()])),
                "z_i0": tf.train.Feature(bytes_list=tf.train.BytesList(value=[z_i0.tobytes()])),
                "z_video": tf.train.Feature(bytes_list=tf.train.BytesList(value=[z_video.tobytes()])),
                "actions": tf.train.Feature(bytes_list=tf.train.BytesList(value=[actions.tobytes()])),
            }
            writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
    return str(directory)


def _config(tmp_path, **overrides):
    values = {
        "run_name": "e2e",
        "code_sha": _SHA,
        "pretrained_model_name_or_path": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "pos_eval_phase": "anchor",
        "base_output_directory": "gs://bucket/e2e/eval_anchor_att-1",
        "checkpoint_dir": "",
        "pos_run_report": "",
        "pos_control_checkpoint_dir": "",
        "pos_control_run_report": "",
        "pos_dev_manifest": _DEV,
        "pos_test_manifest": _TEST,
        "eval_data_dir": "",
        "latent_channels": _C,
        "latent_frames": _F,
        "latent_height": _H,
        "latent_width": _W,
        "action_len": _ACTION_LEN,
        "action_dim": _ACTION_DIM,
        "validation_seed": 0,
        "validation_start_index": 0,
        "side_adapter_sampling_steps": _STEPS,
        "side_adapter_guide_scale": _GUIDE,
    }
    values.update(overrides)
    del tmp_path
    return _Config(values)


def _requires_backend():
    pytest.importorskip("torch")
    pytest.importorskip("aqt")
    pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("tensorflow")


# =============================================================================================
# 1. The anchor phase, on a real (tiny) model: restore -> rollout -> decode -> summarize -> certify.
# =============================================================================================


def test_the_anchor_phase_runs_the_WHOLE_path_on_a_real_tiny_model(tmp_path, fake_gs):
    """Every stage executes: an Orbax restore through the production template, four TFRecords read by
    the anchor's own reader, four 25-step CFG rollouts through the real transformer + adapter, the
    decode seam on both sides, the aggregation, the reproduction verdict, and a published certificate.
    """
    _requires_backend()
    backend = _backend()
    run = anchor.HISTORICAL_ANCHOR.run_name
    ckpt_dir = str(tmp_path / run / "checkpoints")
    _write_checkpoint(ckpt_dir, step=anchor.HISTORICAL_ANCHOR.checkpoint_step, template=backend.template)
    data_dir = _write_records(tmp_path / "val", anchor.HISTORICAL_ANCHOR.sample_names)

    config = _config(tmp_path, checkpoint_dir=ckpt_dir, eval_data_dir=data_dir)
    result = anchor.run_evaluation(config, backend=backend)

    certificate, measurement = result["certificate"], result["measurement"]
    assert result["phase"] == "anchor"
    # The identity is DERIVED from the tree that was opened, not from anything the config said.
    assert certificate["checkpoint"]["run_name"] == run and certificate["checkpoint"]["step"] == 30000
    assert certificate["checkpoint"]["source"] == "historical"
    assert certificate["sample_names"] == list(anchor.HISTORICAL_ANCHOR.sample_names)
    assert certificate["num_steps"] == anchor.DEPLOYED_SAMPLING_STEPS
    assert certificate["measurement_sha256"] == measurement.digest
    assert all(np.isfinite(list(measurement.means.values())))

    # The artifact landed in the BUCKET (gs:// through the storage layer), and reads back identically.
    published = "gs://bucket/e2e/anchor_certificate.json"
    assert published in fake_gs.blobs and not Path("gs:").exists()
    assert anchor.load_certificate(published, protocol=anchor.ANCHOR_PROTOCOL)["measurement_sha256"] == (
        measurement.digest
    )

    # A randomly initialised adapter is not the deployed one, so the wiring proof FAILS -- and every
    # later phase is then refused. That refusal is plan §3c's ordering, executed.
    assert not certificate["reproduced"], "a random tiny adapter must not reproduce the deployed 0.2946"
    for phase in ("benchmark", "gates", "confirm"):
        with pytest.raises(ValueError, match="did not reproduce"):
            anchor.run_evaluation(
                _config(tmp_path, pos_eval_phase=phase, base_output_directory=f"gs://bucket/e2e/eval_{phase}_att-1"),
                backend=backend,
            )


def test_the_rollout_the_anchor_phase_runs_is_the_deployed_grid_and_the_seam_is_the_only_stub(tmp_path):
    """The horizon cannot be shortened to make this test cheap: ``rollout_prediction`` takes no
    ``num_steps``, and it refuses a grid that is not the deployed one."""
    _requires_backend()
    backend = _backend()
    z_i0 = jnp.zeros((1, _C, 1, _H, _W), jnp.float32)
    z_video = jnp.zeros((1, _C, _F, _H, _W), jnp.float32)
    execution, metrics = backend.score(
        z_i0=z_i0,
        z_video=z_video,
        actions=jnp.zeros((1, _ACTION_LEN, _ACTION_DIM), jnp.float32),
        key=anchor.evaluation_draw_key("example"),
    )
    assert execution.num_steps == 25 and execution.grid_size == 26
    assert execution.draw_key_sha256 == anchor.draw_key_digest(anchor.evaluation_draw_key("example"))
    assert np.isfinite(metrics["ssim_avg"]) and np.isfinite(metrics["latent_mse"])
    # frame 0 stays pinned to the conditioning frame, all the way through the rollout
    assert np.allclose(np.asarray(execution.z_pred[:, :, :1], np.float32), np.asarray(z_i0, np.float32))

    short, short_t = overfit100_sampler_grid(
        num_inference_steps=10, flow_shift=5.0, sigma_min=0.0, sigma_max=1.0, num_train_timesteps=1000
    )
    with pytest.raises(ValueError, match="sigmas"):
        anchor.rollout_prediction(
            velocity_fn=lambda *a, **k: z_video,
            sigmas=short,
            timesteps=short_t,
            context=backend.context,
            z_i0=z_i0,
            z_video=z_video,
            key=anchor.evaluation_draw_key("x"),
            guide_scale=_GUIDE,
        )


# =============================================================================================
# 2. The rest of the protocol, from a reproducing anchor: benchmark -> gates -> TEST door.
# =============================================================================================


def _publish_reproducing_anchor(tmp_path, backend, root="gs://bucket/e2e"):
    """A REAL anchor certificate for a run that did reproduce — built through the real path.

    The restore, the aggregation, the verdict and the publication are all production code; only the
    per-sample numbers are the recorded ones rather than a tiny model's. That is the honest way to put
    the later phases in the state a successful anchor leaves behind.
    """
    run = anchor.HISTORICAL_ANCHOR.run_name
    ckpt_dir = str(tmp_path / run / "checkpoints")
    _write_checkpoint(ckpt_dir, step=anchor.HISTORICAL_ANCHOR.checkpoint_step, template=backend.template)
    _, identity = anchor.restore_anchor_checkpoint(ckpt_dir, backend.template)
    rows = [
        {
            "name": name,
            "latent_mse": anchor.HISTORICAL_ANCHOR.mean_latent_mse,
            "pixel_mse": anchor.HISTORICAL_ANCHOR.mean_pixel_mse,
            "ssim_avg": anchor.HISTORICAL_ANCHOR.mean_ssim,
            "num_steps": anchor.DEPLOYED_SAMPLING_STEPS,
        }
        for name in anchor.HISTORICAL_ANCHOR.sample_names
    ]
    measurement = anchor.summarize_samples(
        rows, checkpoint=identity, code_sha=_SHA, model_revision="tiny", test_manifest_path=_TEST
    )
    verdict = anchor.reproduce_anchor(measurement)
    assert verdict.reproduced
    anchor.publish_certificate(anchor.anchor_certificate_path(root), anchor.anchor_certificate(verdict))
    return ckpt_dir


def _stub(high, low, template=None):
    """A device stand-in whose SSIM depends on WHICH actions it was fed and WHICH parameters are bound.

    ``_install_records`` gives an example's z_video and its actions the same fill, so "these actions
    are this example's" is visible to the device without anyone telling it a name — a fair simulation
    of action conditioning. And a checkpoint whose adapter parameters are all zero scores 0.10 lower,
    which is how the matched-C0 arm is made distinguishable from R-B without touching identity.
    """

    def values(*, z_video, actions, adapter_enabled, params=None):
        if not adapter_enabled:
            return 0.10, 2.0
        matched = np.isclose(float(np.mean(np.asarray(actions))), float(np.mean(np.asarray(z_video))))
        alive = params is None or any(float(np.abs(np.asarray(leaf)).sum()) > 0 for leaf in jax.tree.leaves(params))
        base = high if matched else low
        return (base if alive else base - 0.10), 1.0

    return _StubBackend(values, template=template), None


def test_the_protocol_runs_end_to_end_from_a_reproducing_anchor(tmp_path, monkeypatch, fake_gs):
    """benchmark → gates → confirm, with real Orbax restores, real cohorts, the real producer, real
    digest-bound artifacts published through ``gs://``, real certificate consumption, and one TEST door.
    """
    _requires_backend()
    real_backend = _backend()
    _install_records(monkeypatch, instrument.load_dev_cohort(_DEV).rows)
    historical = _publish_reproducing_anchor(tmp_path, real_backend)

    stub, _ = _stub(0.40, 0.30, template=real_backend.template)

    # --- benchmark: the deployed checkpoint's DEV-64 row, frozen -----------------------------
    benchmark = anchor.run_evaluation(
        _config(
            tmp_path,
            pos_eval_phase="benchmark",
            base_output_directory="gs://bucket/e2e/eval_benchmark_att-1",
            checkpoint_dir=historical,
        ),
        backend=stub,
    )
    row = benchmark["benchmark_row"]
    assert row["cohort"] == "dev64" and row["example_count"] == 64 and row["num_steps"] == 25
    assert row["checkpoint"]["run_name"] == anchor.HISTORICAL_ANCHOR.run_name
    assert anchor.load_benchmark_row(anchor.benchmark_row_path("gs://bucket/e2e")) == row
    assert "gs://bucket/e2e/eval_benchmark_att-1/tables/historical_true.json" in fake_gs.blobs
    reloaded = anchor.load_score_table("gs://bucket/e2e/eval_benchmark_att-1/tables/historical_true.json")
    assert reloaded.digest == benchmark["table"].digest, "a published table round-trips to the same artifact"

    # --- gates: both arms, the full battery, the DEV certificate ------------------------------
    arm_root = str(tmp_path / "rb" / "checkpoints")
    control_root = str(tmp_path / "c0" / "checkpoints")
    zeroed = loop.RolloutTrainState(
        params=jax.tree.map(jnp.zeros_like, real_backend.template.params),
        opt_state=real_backend.template.opt_state,
        step=0,
    )
    for root, arm, metric, template in (
        (arm_root, "rollout", 0.40, real_backend.template),
        (control_root, "control", 0.44, zeroed),
    ):
        _write_checkpoint(root, step=1000, template=template, arm=arm, dev_metric=metric)
        _write_selection(root, step=1000, template=template, arm=arm, dev_metric=metric)
    reports = {}
    for label, (root, arm, metric) in {
        "arm": (arm_root, "rollout", 0.40),
        "control": (control_root, "control", 0.44),
    }.items():
        report = loop.RunReport(
            state=loop.RolloutTrainState(params=None, opt_state=None, step=1000),
            history=(loop.EvalRecord(step=1000, dev_metric=metric, train_metric=1.0),),
            verdict=loop.stop_verdict([loop.EvalRecord(step=1000, dev_metric=metric, train_metric=1.0)]),
            retained_step=1000,
            steps_run=1000,
            draw_log=(),
        )
        reports[label] = f"gs://bucket/e2e/{label}_run_report.json"
        anchor.publish_run_report(reports[label], report, arm=arm, k_b=2, num_steps=25)

    gates_config = _config(
        tmp_path,
        pos_eval_phase="gates",
        base_output_directory="gs://bucket/e2e/eval_gates_att-1",
        checkpoint_dir=arm_root,
        pos_run_report=reports["arm"],
        pos_control_checkpoint_dir=control_root,
        pos_control_run_report=reports["control"],
    )
    outcome = anchor.run_evaluation(gates_config, backend=stub)
    certificate = outcome["certificate"]
    assert certificate["certificate"] == gates.GATE_CERTIFICATE
    assert certificate["cohort"] == "dev64" and certificate["coverage_ok"] and certificate["num_steps"] == 25
    assert certificate["rollout_checkpoint"]["run_name"] == "rb"
    assert certificate["control_checkpoint"]["run_name"] == "c0"
    # The certificate is a real artifact: it re-loads through the strict schema and re-decides.
    assert gates.load_dev_certificate(anchor.dev_certificate_path("gs://bucket/e2e"))["passed"] == (
        certificate["passed"]
    )
    report = outcome["action_use"]
    assert report["gate"].passed, "true actions beat the deranged donor's on identical noise"
    assert report["reported"]["coverage_ok"] and "control_mean_delta_true_minus_zero" in report["reported"]
    assert "passed" not in report["reported"]
    for label in ("rollout", "control"):
        for condition in ("true", "wrong", "zero"):
            assert f"gs://bucket/e2e/eval_gates_att-1/tables/{label}_{condition}.json" in fake_gs.blobs

    # --- confirm: the one TEST door, both gates, an independent TEST derangement ---------------
    _install_records(monkeypatch, gates.load_test_cohort(_TEST).rows)
    test_stub, _ = _stub(0.40, 0.30, template=real_backend.template)
    confirmation = anchor.run_evaluation(
        _config(
            tmp_path,
            pos_eval_phase="confirm",
            base_output_directory="gs://bucket/e2e/eval_confirm_att-1",
            checkpoint_dir=arm_root,
            pos_run_report=reports["arm"],
            pos_control_checkpoint_dir=control_root,
            pos_control_run_report=reports["control"],
        ),
        backend=test_stub,
    )["confirmation"]
    assert confirmation["cohort"] == "test64"
    assert confirmation["primary"].numbers["manifest_sha256"] == anchor.J0_TEST64_SHA256
    assert confirmation["action_use"].numbers["derangement_sha256"] == confirmation["derangement_sha256"]
    assert confirmation["dev_certificate_sha256"]
    published = json.loads(fake_gs.blobs["gs://bucket/e2e/eval_confirm_att-1/test_confirmation.json"])
    assert published["payload"]["primary"]["passed"] is confirmation["primary"].passed
    assert published["payload"]["action_use"]["passed"] is confirmation["action_use"].passed


def test_no_phase_can_be_reached_without_its_prerequisites(tmp_path, monkeypatch, fake_gs):
    """The protocol's order, as refusals: the benchmark needs the anchor, and TEST needs the DEV gate."""
    _requires_backend()
    real_backend = _backend()
    _install_records(monkeypatch, instrument.load_dev_cohort(_DEV).rows)
    historical = _publish_reproducing_anchor(tmp_path, real_backend, root="gs://bucket/ordered")
    stub, _ = _stub(0.40, 0.30, template=real_backend.template)

    # The confirm phase runs only from a DEV certificate the gates phase issued -- none exists yet.
    with pytest.raises(ValueError, match="there is no exp06.gates.v1 artifact"):
        anchor.run_evaluation(
            _config(
                tmp_path,
                pos_eval_phase="confirm",
                base_output_directory="gs://bucket/ordered/eval_confirm_att-1",
                checkpoint_dir=historical,
                pos_run_report="gs://bucket/ordered/missing.json",
                pos_control_checkpoint_dir=historical,
                pos_control_run_report="gs://bucket/ordered/missing.json",
            ),
            backend=stub,
        )
    # ...and an artifact root that is not attempt-scoped for its phase is refused before anything runs.
    with pytest.raises(ValueError, match="attempt-scoped"):
        anchor.run_evaluation(
            _config(tmp_path, pos_eval_phase="benchmark", base_output_directory="gs://bucket/ordered/whatever"),
            backend=stub,
        )
