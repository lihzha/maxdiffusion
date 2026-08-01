"""CPU-only tests for the standalone sampling-steps probe (H1 redesign).

``src/maxdiffusion/probe_overfit100_sampling_steps.py`` answers one question -- does a longer
sampler close the gap at checkpoint 2500? -- and is deliberately **verdict-isolated**: it never
touches role validation, the aggregation artifact, staging or ``eval_pass_role``, so it cannot
produce an admissible artifact by construction. What is pinned here:

  * COHORT determinism -- the same seed and manifest always pick the same 30 canonical windows,
    a different seed picks a different set, and the chosen names are recorded in the output.
  * PER-ARM CORRECTNESS -- each arm really rolls out with ITS step count (asserted on the sigma
    schedule the arm builds, not on a comment), via its own jitted rollout rather than a mutated
    global config.
  * OUTPUT -- schema, per-row keys, the paired-delta summary, an immutable write, and a path
    that can never land inside a verdict role directory.
  * ISOLATION -- a source guard that the module imports/calls nothing from the verdict path.
  * The launcher's placement contract, and why it carries no ffmpeg-ensure block.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import maxdiffusion.generate_wan_side_adapter as gen
import maxdiffusion.probe_overfit100_sampling_steps as probe

_REPO = Path(gen.__file__).parents[2]
_MANIFEST = _REPO / "docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json"
_LAUNCHER = _REPO / "bash_scripts/probe_wan_overfit100_sampling.sh"
_CONFIG = _REPO / "src/maxdiffusion/configs/base_wan_5b_overfit100.yml"


# --------------------------------------------------------------------------------------
# Cohort.
# --------------------------------------------------------------------------------------


def test_probe_cohort_is_deterministic_in_the_seed():
    a = probe.probe_cohort(str(_MANIFEST), seed=0, num_windows=30)
    b = probe.probe_cohort(str(_MANIFEST), seed=0, num_windows=30)
    assert [w["name"] for w in a] == [w["name"] for w in b]
    assert len(a) == 30
    assert len({w["name"] for w in a}) == 30
    # Every pick is a canonical window of the committed cohort.
    canonical = {gen.overfit100_window_name(*key) for key in gen.manifest_canonical_cohort(str(_MANIFEST))}
    assert {w["name"] for w in a} <= canonical
    # Output order is stable and by episode_index (so the JSON reads predictably).
    assert [w["episode_index"] for w in a] == sorted(w["episode_index"] for w in a)


def test_a_different_seed_picks_a_different_cohort():
    a = {w["name"] for w in probe.probe_cohort(str(_MANIFEST), seed=0, num_windows=30)}
    b = {w["name"] for w in probe.probe_cohort(str(_MANIFEST), seed=1, num_windows=30)}
    assert a != b


def test_the_whole_cohort_is_returned_when_asked_for_everything():
    every = probe.probe_cohort(str(_MANIFEST), seed=0, num_windows=100)
    assert len(every) == 100
    with pytest.raises(ValueError):
        probe.probe_cohort(str(_MANIFEST), seed=0, num_windows=101)  # more than the cohort has
    with pytest.raises(ValueError):
        probe.probe_cohort(str(_MANIFEST), seed=0, num_windows=0)


# --------------------------------------------------------------------------------------
# Arms.
# --------------------------------------------------------------------------------------


def test_parse_probe_steps_accepts_a_list_or_a_string():
    assert probe.parse_probe_steps([25, 50, 100]) == (25, 50, 100)
    assert probe.parse_probe_steps("[25,50,100]") == (25, 50, 100)
    assert probe.parse_probe_steps("25,50") == (25, 50)
    for bad in ("", [], [0], [-5], "junk"):
        with pytest.raises(ValueError):
            probe.parse_probe_steps(bad)


def test_each_arm_gets_its_own_config_view_without_mutating_the_base():
    base = SimpleNamespace(side_adapter_sampling_steps=25, flow_shift=5.0, weights_dtype="float32")
    arm = probe.arm_config(base, 100)
    assert arm.side_adapter_sampling_steps == 100
    assert arm.flow_shift == 5.0 and arm.weights_dtype == "float32"  # everything else proxied
    assert base.side_adapter_sampling_steps == 25  # the base is NEVER mutated
    assert probe.arm_config(base, 50).side_adapter_sampling_steps == 50
    assert base.side_adapter_sampling_steps == 25


def test_each_arm_builds_its_own_sigma_schedule(monkeypatch):
    # The per-arm correctness claim, asserted on what the rollout actually constructs: one sigma
    # schedule per arm, of that arm's length. A shared/stale jit would show the same count twice.
    seen: list[int] = []
    real = gen.build_rollout_sigmas
    monkeypatch.setattr(gen, "build_rollout_sigmas", lambda n, *a, **kw: seen.append(int(n)) or real(n, *a, **kw))
    base = SimpleNamespace(
        side_adapter_sampling_steps=25, flow_shift=5.0, weights_dtype="float32", activations_dtype="float32"
    )
    scheduler = SimpleNamespace(config=SimpleNamespace(sigma_min=0.0, sigma_max=1.0, num_train_timesteps=1000))
    for steps in (25, 50, 100):
        gen.build_rollout_sigmas(
            probe.arm_config(base, steps).side_adapter_sampling_steps,
            base.flow_shift,
            scheduler.config.sigma_min,
            scheduler.config.sigma_max,
        )
    assert seen == [25, 50, 100]
    assert len(real(50, 5.0, 0.0, 1.0)) == 51  # n + 1 sigmas, so the count is observable


# --------------------------------------------------------------------------------------
# Output.
# --------------------------------------------------------------------------------------


def _rows(steps=(25, 50, 100), n=3):
    rows = []
    for index in range(n):
        for arm in steps:
            rows.append(
                {
                    "name": f"ep{100 + index}_v0_s00000",
                    "episode_id": 100 + index,
                    "episode_index": index,
                    "window_start": 0,
                    "sampling_steps": arm,
                    "ssim": 0.80 + 0.01 * steps.index(arm) + 0.001 * index,
                    "latent_mse": 0.05,
                    "pixel_mse": 0.004,
                }
            )
    return rows


def test_probe_summary_reports_per_arm_stats_and_paired_deltas():
    summary = probe.probe_summary(_rows(), baseline=25)
    assert summary["baseline_sampling_steps"] == 25
    assert sorted(summary["per_arm"]) == ["100", "25", "50"]
    for arm in ("25", "50", "100"):
        assert summary["per_arm"][arm]["n"] == 3
        assert "mean_ssim" in summary["per_arm"][arm] and "median_ssim" in summary["per_arm"][arm]
    # Paired per-window deltas against the in-probe baseline.
    assert summary["paired_deltas"]["50-25"]["n"] == 3
    assert summary["paired_deltas"]["50-25"]["mean_delta_ssim"] == pytest.approx(0.01)
    assert summary["paired_deltas"]["100-25"]["mean_delta_ssim"] == pytest.approx(0.02)


def test_probe_summary_refuses_an_incomplete_pairing():
    rows = [row for row in _rows() if not (row["sampling_steps"] == 50 and row["episode_index"] == 1)]
    with pytest.raises(ValueError) as ei:
        probe.probe_summary(rows, baseline=25)
    assert "ep101_v0_s00000" in str(ei.value)


def test_probe_output_path_is_canonical_and_has_no_override():
    import inspect

    config = SimpleNamespace(output_dir="gs://bucket/out", run_name="ovf-s3")
    path = probe.probe_output_path(config, 2500)
    assert path == "gs://bucket/out/ovf-s3/validation_probe_sampling/probe_steps_ckpt2500.json"
    assert not any(part.startswith("step_") for part in path.split("/"))
    # There is NO output-root override: an unrestricted one could drop a diagnostic inside a
    # verdict role directory, which no amount of testing downstream can undo.
    source = inspect.getsource(probe)
    assert "validation_probe_output_dir" not in source


@pytest.mark.parametrize(
    "run_name",
    [
        "step_002500_s3_segment_final",
        "x/step_002500_s3_full_set",
        "../step_002500_s3_segment_final",
        "ok/../step_002500_s3_intermediate",
    ],
)
def test_a_hostile_run_name_cannot_steer_the_probe_into_a_verdict_directory(run_name):
    config = SimpleNamespace(output_dir="gs://bucket/out", run_name=run_name)
    with pytest.raises(ValueError) as ei:
        probe.probe_output_path(config, 2500)
    message = str(ei.value)
    assert "step_" in message and "verdict" in message.lower()


def test_a_hostile_output_dir_cannot_steer_the_probe_either():
    config = SimpleNamespace(output_dir="gs://bucket/out/step_002500_s3_full_set", run_name="ovf")
    with pytest.raises(ValueError):
        probe.probe_output_path(config, 2500)


# --------------------------------------------------------------------------------------
# The approved experiment is pinned: any deviation is a startup failure.
# --------------------------------------------------------------------------------------


def test_the_approved_design_is_pinned():
    assert probe.APPROVED_SAMPLING_ARMS == (25, 50, 100)
    assert probe.APPROVED_CHECKPOINT_STEP == 2500
    assert probe.BASELINE_SAMPLING_STEPS == 25
    probe.assert_approved_design(steps=(25, 50, 100), checkpoint_step=2500)
    probe.assert_approved_design(steps=(100, 25, 50), checkpoint_step=2500)  # order-canonicalized


@pytest.mark.parametrize(
    "steps,step,needle",
    [
        ((25, 50), 2500, "arms"),
        ((25, 50, 100, 200), 2500, "arms"),
        ((10, 20, 30), 2500, "arms"),
        ((25, 50, 100), 1000, "checkpoint"),
        ((25, 50, 100), 5000, "checkpoint"),
    ],
)
def test_a_deviation_from_the_approved_design_is_refused(steps, step, needle):
    with pytest.raises(ValueError) as ei:
        probe.assert_approved_design(steps=steps, checkpoint_step=step)
    message = str(ei.value)
    assert needle in message.lower()
    assert "approv" in message.lower()  # says WHY it is pinned


def test_the_baseline_is_fixed_not_derived_from_the_arms():
    import inspect

    source = inspect.getsource(probe)
    assert "baseline=BASELINE_SAMPLING_STEPS" in source
    assert "baseline=min(" not in source


def test_probe_artifact_is_written_immutably(tmp_path):
    payload = probe.probe_artifact(
        SimpleNamespace(
            output_dir=str(tmp_path),
            run_name="r",
            validation_output_dir="",
            seed=0,
            model_manifest_path=str(_MANIFEST),
            eval_data_dir="gs://x/train100",
            checkpoint_dir="gs://x/ck",
        ),
        checkpoint_step=2500,
        cohort=[{"name": "ep100_v0_s00000", "episode_id": 100, "episode_index": 0, "window_start": 0}],
        steps=(25, 50, 100),
        rows=_rows(steps=(25, 50, 100), n=1),
    )
    assert payload["schema"] == probe.PROBE_SCHEMA
    assert payload["checkpoint_step"] == 2500
    assert payload["sampling_steps_arms"] == [25, 50, 100]
    assert payload["cohort"] == ["ep100_v0_s00000"]
    assert "summary" in payload and "rows" in payload
    # Verdict fields must NOT be present: this is not an admissible artifact.
    for forbidden in ("eval_pass_role", "canonical_cohort", "role_validation", "run_signature"):
        assert forbidden not in payload

    path = str(tmp_path / "probe.json")
    probe.write_probe_artifact(path, payload)
    probe.write_probe_artifact(path, payload)  # identical rewrite tolerated
    with pytest.raises(ValueError):
        probe.write_probe_artifact(path, {**payload, "checkpoint_step": 9999})


# --------------------------------------------------------------------------------------
# Isolation from the verdict machinery.
# --------------------------------------------------------------------------------------


def test_the_probe_touches_no_verdict_machinery():
    source = Path(probe.__file__).read_text()
    for forbidden in (
        "overfit100_aggregation_artifact",
        "pass_role_plan_reasons",
        "assert_pass_role_plan",
        "validate_artifact_role",
        "read_staged_rows",
        "write_staged_row",
        "overfit100_publication_state",
        "overfit100_published_marker",
        "parse_eval_pass_role",
        "eval_pass_role",
        "OVERFIT100_PUBLISHED_MARKER",
        "overfit100_success_statistic",
    ):
        assert forbidden not in source, f"the probe references verdict machinery: {forbidden}"
    assert "overfit100_step_root" not in source  # cannot even name a role directory


def test_the_probe_reuses_the_eval_paths_rollout_and_rng():
    source = Path(probe.__file__).read_text()
    assert "_rollout_overfit100_sample" in source or "_overfit100_rollout_fn" in source
    assert "window_fold_key" in source
    assert "overfit100_context_for_mode" in source


# --------------------------------------------------------------------------------------
# Launcher + config.
# --------------------------------------------------------------------------------------


def test_the_probe_launcher_exists_and_passes_bash_n():
    assert _LAUNCHER.exists(), f"missing {_LAUNCHER}"
    bash_exe = shutil.which("bash") or "/bin/bash"
    proc = subprocess.run([bash_exe, "-n", str(_LAUNCHER)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr


def test_the_probe_launcher_forwards_its_knobs():
    text = _LAUNCHER.read_text()
    assert "src/maxdiffusion/probe_overfit100_sampling_steps.py" in text
    assert "src/maxdiffusion/configs/base_wan_5b_overfit100.yml" in text
    for env in ("CHECKPOINT_STEP", "PROBE_STEPS", "PROBE_NUM_WINDOWS"):
        assert f"{env}=" in text, env
    for override in (
        'checkpoint_step="${CHECKPOINT_STEP}"',
        'probe_sampling_steps_list="${PROBE_STEPS}"',
        'probe_num_windows="${PROBE_NUM_WINDOWS}"',
    ):
        assert override in text, override
    assert 'PROBE_STEPS="${PROBE_STEPS:-[25,50,100]}"' in text
    assert 'PROBE_NUM_WINDOWS="${PROBE_NUM_WINDOWS:-30}"' in text
    # Manifest-pinned model, like every other arm.
    assert "MANIFEST_PATH" in text and "export COMMIT" in text and "local_files_only=True" in text


def test_neither_the_launcher_nor_the_config_still_offers_an_output_override():
    # The override is gone at every layer -- an env knob or a live YAML key would be a way back in.
    text = _LAUNCHER.read_text()
    for token in ("PROBE_OUTPUT_DIR", "validation_probe_output_dir"):
        assert token not in text, token
    config_text = (Path(probe.__file__).parent / "configs" / "base_wan_5b_overfit100.yml").read_text()
    assert "validation_probe_output_dir" not in config_text


def test_the_probe_launcher_documents_why_it_needs_no_ffmpeg():
    # Mirrors the loss-arm treatment: the probe scores against the VAE decode only -- no MP4 is
    # ever pulled or decoded -- so the ffmpeg-ensure block is deliberately absent.
    text = _LAUNCHER.read_text()
    assert "# >>> ffmpeg ensure" not in text
    assert "ffmpeg" in text  # ...but it says WHY it is absent
    assert "eval_aux_rgb" in text or "aux" in text


def test_the_probe_never_requests_the_auxiliary_rgb_path():
    source = Path(probe.__file__).read_text()
    for forbidden in ("overfit100_aux_rgb", "decode_mp4_frames", "fetch_pinned", "_save_video"):
        assert forbidden not in source, forbidden


def test_config_carries_the_probe_keys():
    import yaml

    cfg = yaml.safe_load(_CONFIG.read_text())
    assert cfg["probe_sampling_steps_list"] == [25, 50, 100]
    assert cfg["probe_num_windows"] == 30


# --------------------------------------------------------------------------------------
# Gap-closing (found by the mutation pass): a GOLDEN cohort, and the arm wiring exercised
# through the real driver rather than through the helper in isolation.
# --------------------------------------------------------------------------------------

# The first five windows seed 0 selects from the committed manifest. Pinned as a golden value: a
# same-seed comparison cannot notice a drifted selection rule, but this can.
_GOLDEN_SEED0_HEAD = [
    "ep45109_v0_s00068",
    "ep61233_v0_s00080",
    "ep35500_v0_s00004",
    "ep34499_v0_s00080",
    "ep61291_v0_s00032",
]


def test_the_seed0_cohort_is_the_golden_selection():
    cohort = probe.probe_cohort(str(_MANIFEST), seed=0, num_windows=30)
    names = [window["name"] for window in cohort]
    # Order is by episode_index, so compare as a set against the recorded selection...
    assert set(_GOLDEN_SEED0_HEAD) <= set(names)
    # ...and pin the whole selection by a stable digest of the sorted names.
    import hashlib

    digest = hashlib.sha256("|".join(sorted(names)).encode()).hexdigest()
    assert (
        digest == "6ef44151725ee3dc0ce9fc0221ea8bbf5576b9117d4513cc8c3cf08be6557c4a"
    )  # golden: a drifted selection rule changes this
    assert len(names) == 30


def _probe_stub_stack(monkeypatch, tmp_path, *, episodes=2):
    """The minimal stack ``run_probe`` needs on one CPU device."""
    import optax
    from flax import nnx
    from jax.sharding import Mesh

    import maxdiffusion.trainers.wan_ti2v_overfit100_trainer as overfit100
    from maxdiffusion.schedulers import FlaxFlowMatchScheduler

    C, F, H, W = 2, 3, 4, 4

    class _Stub(nnx.Module):
        def __init__(self):
            self.gain = nnx.Param(jnp.asarray(0.2, dtype=jnp.float32))

        def __call__(self, **kwargs):
            return self.gain[...] * kwargs["hidden_states"].astype(jnp.float32)

    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1, 1), ("data", "fsdp", "context", "tensor"))
    graphdef, params, rest = nnx.split(_Stub(), nnx.Param, ...)
    table = jnp.stack([jnp.full((2, 8), float(i) + 1.0, dtype=jnp.float32) for i in range(episodes)])
    state = overfit100.Overfit100TrainState.create(
        apply_fn=graphdef.apply,
        params=params,
        tx=optax.adamw(0.1),
        graphdef=graphdef,
        rest_of_state=rest,
        context_table=table,
    )

    class _Pipeline:
        def _denormalize_latents(self, latents):
            return latents

        def _decode_latents_to_video(self, latents):
            value = float(jnp.mean(latents.astype(jnp.float32)))
            return np.full((1, 5, 8, 8, 3), np.tanh(value) * 0.5 + 0.5, dtype=np.float32)

    scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32, shift=5.0, sigma_min=0.0, sigma_max=1.0)
    trainer = SimpleNamespace(_create_scheduler=lambda: (scheduler, None))
    pipeline = _Pipeline()
    monkeypatch.setattr(
        gen,
        "_restore_overfit100_validation_state",
        lambda cfg: (trainer, pipeline, mesh, state, "SH", jnp.zeros((1, 2, 8)), 2500),
    )
    monkeypatch.setattr(gen, "assert_ssim_available", lambda: None)
    monkeypatch.setattr(gen, "_frame_ssim", lambda pred, target: float(1.0 - np.mean(np.abs(pred - target))))
    samples = [
        gen.Overfit100EvalSample(
            name=gen.overfit100_window_name(100 + i, 0),
            episode_id=100 + i,
            episode_index=i,
            window_start=0,
            canonical=True,
            position=i,
            z_i0=np.full((C, 1, H, W), 0.1 * (i + 1), np.float32),
            z_video=np.full((C, F, H, W), 0.1 * (i + 1), np.float32),
            instruction="t",
        )
        for i in range(episodes)
    ]
    monkeypatch.setattr(gen, "read_overfit100_samples", lambda config, windows: samples)
    monkeypatch.setattr(
        probe,
        "probe_cohort",
        lambda manifest_path, *, seed, num_windows: [
            {
                "name": s.name,
                "episode_id": s.episode_id,
                "episode_index": s.episode_index,
                "window_start": s.window_start,
                "canonical": True,
            }
            for s in samples
        ],
    )
    return SimpleNamespace(
        model_type="OVERFIT100_TI2V",
        run_name="probe",
        output_dir=str(tmp_path / "out"),
        seed=0,
        checkpoint_step=2500,
        probe_sampling_steps_list=[25, 50, 100],
        probe_num_windows=episodes,
        model_manifest_path=str(_MANIFEST),
        eval_data_dir="gs://x/train100",
        checkpoint_dir=str(tmp_path / "ck"),
        logical_axis_rules=(),
        weights_dtype="float32",
        activations_dtype="float32",
        flow_shift=5.0,
        side_adapter_sampling_steps=25,
        side_adapter_guide_scale=1.0,
    )


def test_the_driver_gives_each_arm_its_own_step_count(tmp_path, monkeypatch):
    # THE wiring claim, through run_probe: each arm's rollout is built from a config view carrying
    # THAT arm's step count. Sharing the base config (or pinning the view) must be visible here.
    config = _probe_stub_stack(monkeypatch, tmp_path)
    seen: list[int] = []
    real_fn = gen._overfit100_rollout_fn

    def _recording(state_shardings, data_shardings, replicated, scheduler, cfg):
        seen.append(int(cfg.side_adapter_sampling_steps))
        return lambda s, batch, rng: gen._rollout_overfit100_sample(s, batch, rng, scheduler, cfg)

    monkeypatch.setattr(gen, "_overfit100_rollout_fn", _recording)
    payload = probe.run_probe(config)
    assert seen == [25, 50, 100], seen
    assert sorted({row["sampling_steps"] for row in payload["rows"]}) == [25, 50, 100]
    assert len(payload["rows"]) == 2 * 3
    # Distinct step counts really produce distinct rollouts (not a stale jit).
    per_arm = {row["sampling_steps"]: row["latent_mse"] for row in payload["rows"] if row["name"].endswith("s00000")}
    assert len({round(value, 9) for value in per_arm.values()}) > 1
    del real_fn


def test_the_driver_writes_only_to_the_probe_directory(tmp_path, monkeypatch):
    config = _probe_stub_stack(monkeypatch, tmp_path)
    monkeypatch.setattr(
        gen,
        "_overfit100_rollout_fn",
        lambda ss, ds, rep, sched, cfg: (
            lambda s, batch, rng: gen._rollout_overfit100_sample(s, batch, rng, sched, cfg)
        ),
    )
    probe.run_probe(config)
    written = [str(p.relative_to(tmp_path)) for p in (tmp_path).rglob("*.json")]
    assert written == ["out/probe/validation_probe_sampling/probe_steps_ckpt2500.json"], written
    assert not any("step_" in path for path in written)


def test_the_docstring_records_the_acceptance_rule_and_the_landed_reference():
    # The scientific anchor the reviewer supplied: the same 30 windows' landed 25-step numbers, so
    # the in-probe control arm can be checked against them offline.
    doc = probe.__doc__ or ""
    assert "0.8100125855" in doc and "0.8059329625" in doc
    assert "acceptance" in doc.lower()


def test_the_artifact_carries_everything_the_offline_validity_check_needs():
    payload = probe.probe_artifact(
        SimpleNamespace(
            output_dir="/tmp/x",
            run_name="r",
            seed=0,
            model_manifest_path=str(_MANIFEST),
            eval_data_dir="gs://x/train100",
            checkpoint_dir="gs://x/ck",
        ),
        checkpoint_step=2500,
        cohort=[{"name": "ep100_v0_s00000", "episode_id": 100, "episode_index": 0, "window_start": 0}],
        steps=(25, 50, 100),
        rows=_rows(steps=(25, 50, 100), n=1),
    )
    # Row-level identity + per-arm ssim is what an offline join against the segment-final rows needs.
    for row in payload["rows"]:
        for field in ("name", "episode_id", "window_start", "sampling_steps", "ssim"):
            assert field in row, field
    assert payload["rollout_seed"] == 0 and payload["context_mode"] == "correct"


def test_the_module_and_launcher_call_it_a_discriminator_not_just_a_diagnostic():
    doc = probe.__doc__ or ""
    assert "H1/H2" in doc or "discriminator" in doc
    assert "discriminator" in _LAUNCHER.read_text()
