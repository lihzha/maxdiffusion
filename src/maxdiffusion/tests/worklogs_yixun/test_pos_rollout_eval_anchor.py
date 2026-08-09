"""exp_06 `rollout_adapter` — T5a `eval-anchor`: restore, and reproduce the baseline (plan §3c/§5-5).

The contract: **a checkpoint can be restored and scored, and the historical baseline is reproduced
before anything new is measured.**

Two halves, and the second is the headline. **Restore** must reach the immutable earliest-best
SELECTION artifact and nothing else — evaluating a checkpoint the stop rule did not select produces a
plausible number for the wrong model, which is the worst failure available here (exp_05's S9). The
**anchor protocol** then reproduces the deployed checkpoint's recorded 4-sample validation before any
new arm is scored: if exp_06's evaluation path cannot recover 0.2946 on the checkpoint that produced
it, every number exp_06 later reports is measured with an instrument nobody has calibrated.

What is NOT re-implemented is pinned as hard as what is: the prediction is T3a's ``cfg_rollout``, the
SSIM is the deployed definition held by a drift tripwire, and the noise convention is deployment's
sequential split. A wiring proof written against a private copy of the wiring proves nothing.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion import eval_wan_pos_rollout as anchor
from maxdiffusion import pos_rollout_loop as loop
from maxdiffusion import pos_rollout_dev_instrument as instrument
from maxdiffusion.pos_rollout_loop import EvalRecord, RolloutTrainState

_MODULE_PATH = Path(anchor.__file__).resolve()
_DEPLOYED = _MODULE_PATH.parent / "generate_wan_side_adapter.py"
_MANIFEST_DIR = _MODULE_PATH.parents[2] / "docs" / "worklogs_yixun" / "exp_04_null_adapter_claude" / "j0_manifests"
_DEV = str(_MANIFEST_DIR / "dev64.json")
_TEST = str(_MANIFEST_DIR / "test64.json")
_SHA = "a" * 40
_REVISION = "Wan-AI/Wan2.2-TI2V-5B-Diffusers@" + "b" * 40


def _state(step=0):
    return RolloutTrainState(
        params={"w": jnp.zeros((2,), jnp.float32)}, opt_state={"mu": jnp.zeros((2,), jnp.float32)}, step=step
    )


def _tree(tmp_path, name, *, resume_steps=((4, 0.9),), selection=(2, 0.5), arm="rollout", k_b=2):
    """A checkpoint root shaped like a finished run: a recency resume tree and a sibling selection."""
    directory = str(tmp_path / name)
    resume = loop.build_checkpoint_manager(directory)
    for step, metric in resume_steps:
        loop.save_checkpoint(
            resume,
            dataclasses.replace(_state(), step=step),
            dev_metric=metric,
            history=[EvalRecord(step=step, dev_metric=metric, train_metric=1.0)],
            arm=arm,
            k_b=k_b,
        )
    if selection is not None:
        step, metric = selection
        loop.save_checkpoint(
            loop.build_selection_manager(directory),
            dataclasses.replace(_state(), step=step),
            dev_metric=metric,
            # A metric-less artifact is a real shape (an interrupted write), so its history is empty too.
            history=[] if metric is None else [EvalRecord(step=step, dev_metric=metric, train_metric=1.0)],
            arm=arm,
            k_b=k_b,
        )
    return directory


def _restore(directory, **overrides):
    kwargs = {"expected_step": 2, "expected_dev_metric": 0.5, "expected_arm": "rollout", "expected_k_b": 2}
    kwargs.update(overrides)
    return anchor.restore_selected_checkpoint(directory, _state(), **kwargs)


_MEAN_TO_ROW = {"mean_latent_mse": "latent_mse", "mean_pixel_mse": "pixel_mse", "mean_ssim": "ssim_avg"}


def _rows(names=None, *, num_steps=25, grid=None, **fields):
    names = anchor.HISTORICAL_ANCHOR.sample_names if names is None else names
    row = {
        "latent_mse": 1.496,
        "pixel_mse": 0.0983,
        "ssim_avg": 0.2946,
        "num_steps": num_steps,
        "grid_sha256": anchor.DEPLOYED_GRID_SHA256 if grid is None else grid,
    }
    row.update(fields)
    return [{"name": name, **row} for name in names]


def _historical_identity(step=30000, run_name=None):
    """What ``restore_anchor_checkpoint`` yields: derived from the root it opened and the step it read."""
    name = anchor.HISTORICAL_ANCHOR.run_name if run_name is None else run_name
    return anchor.CheckpointIdentity(run_name=name, step=step, root=f"gs://b/{name}/checkpoints", source="historical")


def _measured(names=None, *, checkpoint_step=30000, rows=None, **overrides):
    """A real Measurement, produced by the real aggregation — every anchor test runs the whole path.

    Mean overrides are applied to the ROWS, because a mean is now derived from what was scored rather
    than patched into a mapping afterwards — which is the whole point of this round.
    """
    count = overrides.pop("num_samples", None)
    fields = {_MEAN_TO_ROW[key]: overrides.pop(key) for key in list(overrides) if key in _MEAN_TO_ROW}
    names = list(anchor.HISTORICAL_ANCHOR.sample_names if names is None else names)
    if count is not None:
        names = names[:count]
    return anchor.summarize_samples(
        _rows(names, **fields) if rows is None else rows,
        checkpoint=_historical_identity(step=checkpoint_step),
        code_sha=_SHA,
        model_revision=_REVISION,
        test_manifest_path=_TEST,
        **overrides,
    )


def _function_node(source: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


# =============================================================================================
# 1. Restore: the selection artifact, and structurally nothing else.
# =============================================================================================


def test_restore_takes_a_checkpoint_ROOT_so_the_resume_tree_has_no_way_in(tmp_path):
    """exp_05's S9 said "there is no parameter through which the resume tree could reach it" — but it
    took a MANAGER, so that was a claim about what callers would pass. Here the function takes the
    root and derives the sibling itself: handing it the resume root reaches ``<root>_selection``, an
    empty tree, and is refused. The wrong thing is unconstructible rather than merely undocumented.
    """
    pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("tensorflow")
    parameters = list(inspect.signature(anchor.restore_selected_checkpoint).parameters)
    assert parameters[0] == "ckpt_dir"
    assert not {"manager", "selection_manager", "checkpoint_manager"} & set(parameters)
    source = _MODULE_PATH.read_text(encoding="utf-8")
    restore = _function_node(source, "restore_selected_checkpoint")
    called = {node.func.id for node in ast.walk(restore) if isinstance(node, ast.Call) and hasattr(node.func, "id")}
    assert "build_selection_manager" in called
    assert "build_checkpoint_manager" not in called, "the resume tree must not be constructible here"

    directory = _tree(tmp_path, "root")
    state, identity = _restore(directory)
    assert int(state.step) == 2, "the resume tree's latest is step 4; selection is step 2"
    # The identity is DERIVED: the step from the artifact, the run name from the root it opened.
    assert isinstance(identity, anchor.CheckpointIdentity)
    assert identity.step == 2 and identity.dev_metric == 0.5 and identity.source == "selection"
    assert identity.run_name == "root" and identity.arm == "rollout" and identity.k_b == 2


def test_a_missing_selection_artifact_fails_loudly_and_never_falls_back(tmp_path):
    pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("tensorflow")
    directory = _tree(tmp_path, "nosel", selection=None)
    assert loop.build_checkpoint_manager(directory).latest_step() == 4, "the resume tree IS populated"
    with pytest.raises(ValueError, match="no selection artifact"):
        _restore(directory)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"expected_step": 3}, "is at step 2 but the report named 3"),
        ({"expected_dev_metric": 0.4}, "not the reported"),
        ({"expected_arm": "one_step"}, "declares arm"),
        ({"expected_k_b": 4}, "declares k_b"),
    ],
)
def test_every_metadata_field_the_report_named_must_match(tmp_path, override, message):
    """Two runs can select the same step; the metric, the arm and the horizon are what tell their
    artifacts apart. A matched-C0 checkpoint scored as R-B would invert the experiment's claim."""
    pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("tensorflow")
    directory = _tree(tmp_path, f"meta_{abs(hash(message)) % 9999}")
    with pytest.raises(ValueError, match=message):
        _restore(directory, **override)


def test_a_nonfinite_expectation_or_a_metricless_artifact_is_refused(tmp_path):
    pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("tensorflow")
    directory = _tree(tmp_path, "nonfinite")
    with pytest.raises(ValueError, match="must be finite"):
        _restore(directory, expected_dev_metric=float("nan"))
    bare = _tree(tmp_path, "bare", selection=(2, None))
    with pytest.raises(ValueError, match="carries no"):
        _restore(bare)


# =============================================================================================
# 2. THE ANCHOR PROTOCOL — the round's headline (plan §3c(a)).
# =============================================================================================


def test_the_historical_anchor_is_the_recorded_four_sample_validation():
    """The tracker's record, transcribed once and pinned so it cannot drift into something else."""
    record = anchor.HISTORICAL_ANCHOR
    assert record.mean_ssim == 0.2946
    assert record.mean_latent_mse == 1.496 and record.mean_pixel_mse == 0.0983
    assert record.num_samples == 4 and record.checkpoint_step == 30000
    assert record.run_name == "wan-pre_context-v6e64-full-gbs512-fresh-20260629-034110"
    assert dataclasses.is_dataclass(record) and record.__dataclass_params__.frozen


def test_the_anchor_is_reproduced_when_every_recorded_metric_lands_in_the_band():
    verdict = anchor.reproduce_anchor(_measured())
    assert verdict.reproduced and not verdict.failed
    assert verdict.deviations["mean_ssim"] == pytest.approx(0.0, abs=1e-12)
    # ...and a run that differs only by float rounding still reproduces.
    nudged = _measured(mean_ssim=0.2946 * 1.01, mean_pixel_mse=0.0983 * 0.99, mean_latent_mse=1.496 * 1.005)
    assert anchor.reproduce_anchor(nudged).reproduced


@pytest.mark.parametrize("metric", ["mean_ssim", "mean_pixel_mse", "mean_latent_mse"])
def test_any_one_recorded_metric_leaving_the_band_fails_the_anchor(metric):
    """All three, not just SSIM: a latent path that is wrong while SSIM happens to land is exactly
    the coincidence a single-metric check would wave through."""
    value = getattr(anchor.HISTORICAL_ANCHOR, metric) * 1.05
    verdict = anchor.reproduce_anchor(_measured(**{metric: value}))
    assert not verdict.reproduced
    assert verdict.failed == (metric,)


def test_the_tolerance_is_a_constant_no_caller_can_widen():
    """A tolerance that is an argument is a tolerance that gets widened at 2am when the anchor misses."""
    parameters = inspect.signature(anchor.reproduce_anchor).parameters
    assert list(parameters) == ["measurement"]
    assert not {"tolerance", "rel_tol", "anchor", "record", "atol"} & set(parameters)
    assert anchor.ANCHOR_REL_TOLERANCE == 0.02


def test_the_band_discriminates_every_wiring_failure_this_path_can_have():
    """The tolerance's job is to separate float rounding from a different computation, and every
    wiring failure available here is order-unity: the frozen-context control sits at ~0.25, the
    per-clip oracle at ~0.92, and a missing frame-0 pin or a wrong noise convention is worse still.
    """
    low, high = 0.2946 * 0.98, 0.2946 * 1.02
    assert (low, high) == pytest.approx((0.288708, 0.300492), rel=1e-6)
    for wrong in (0.25, 0.92, 0.0, 0.35):
        assert not anchor.reproduce_anchor(_measured(mean_ssim=wrong)).reproduced, wrong


@pytest.mark.parametrize(
    "override, message",
    [
        ({"num_samples": 3}, "4 samples"),
        ({"checkpoint_step": 29000}, "step 30000"),
        ({"mean_ssim": float("nan")}, "non-finite ssim"),
        ({"rows": _rows(num_steps=1)}, "horizons"),
    ],
)
def test_a_measurement_that_is_not_the_anchor_protocol_is_refused(override, message):
    """Every one of these is now refused where the measurement is FORMED, not where it is quoted."""
    with pytest.raises(ValueError, match=message):
        anchor.reproduce_anchor(_measured(**override))


def test_the_anchor_certificate_is_published_once_and_carries_its_provenance(tmp_path):
    path = str(tmp_path / "anchor.json")
    measurement = _measured()
    published = anchor.publish_certificate(path, anchor.anchor_certificate(anchor.reproduce_anchor(measurement)))
    assert published["protocol"] == anchor.ANCHOR_PROTOCOL and published["reproduced"] is True
    assert published["code_sha"] == _SHA and published["model_revision"] == _REVISION
    assert published["checkpoint"]["step"] == 30000
    assert published["checkpoint"]["run_name"] == anchor.HISTORICAL_ANCHOR.run_name
    assert published["sample_names"] == list(anchor.HISTORICAL_ANCHOR.sample_names), "the names the summary screened"
    assert published["tolerance"] == anchor.ANCHOR_REL_TOLERANCE
    assert published["measurement_sha256"] == measurement.digest
    assert json.loads(Path(path).read_text())["sha256"] == published["sha256"]


# =============================================================================================
# 3. Scoring: composed from the committed pieces, and unable to be otherwise.
# =============================================================================================


def test_the_prediction_is_T3as_rollout_and_this_module_cannot_re_implement_one():
    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("pos_rollout_step")
        for alias in node.names
    }
    assert "cfg_rollout" in imported, "the scoring rollout must be T3a's"
    predict = _function_node(source, "rollout_prediction")
    calls = {node.func.id for node in ast.walk(predict) if isinstance(node, ast.Call) and hasattr(node.func, "id")}
    assert "cfg_rollout" in calls
    # No second loop can exist in this module: a private rollout is exactly the drift §5-5 forbids.
    for forbidden in ("fori_loop", "while_loop", "lax.scan", "overfit100_sampler_step", "overfit100_euler_update"):
        assert forbidden not in source, forbidden


def test_the_initial_latent_is_deploymentsnoise_convention(tmp_path):
    """The anchor was produced by ``key(seed)`` split sequentially per sample, with the noise drawn
    in ``z_video``'s dtype and frame 0 pinned. Every clause matters: a different convention gives a
    different trajectory and the anchor simply will not reproduce."""
    deployed = _DEPLOYED.read_text(encoding="utf-8")
    for fragment in (
        "z = jax.random.normal(rng, z_video.shape, dtype=z_video.dtype)",
        "z = apply_first_frame_pin(z, z_i0)",
        "rng, sample_rng = jax.random.split(rng)",
    ):
        assert fragment in deployed, "the deployed convention moved; this copy is stale"

    keys = anchor.anchor_sample_keys(7, 4)
    rng = jax.random.key(7)
    expected = []
    for _ in range(4):
        rng, sample_rng = jax.random.split(rng)
        expected.append(sample_rng)
    assert len(keys) == 4
    for got, want in zip(keys, expected):
        assert np.array_equal(np.asarray(jax.random.key_data(got)), np.asarray(jax.random.key_data(want)))

    z_video = jnp.zeros((1, 4, 3, 2, 2), jnp.bfloat16)
    z_i0 = jnp.ones((1, 4, 1, 2, 2), jnp.bfloat16)
    latents = anchor.initial_latents(keys[0], z_video, z_i0)
    assert latents.dtype == z_video.dtype, "the noise is drawn in the video latents' dtype"
    # ...and the DRAW is compared, not merely its dtype. Drawing in fp32 and casting down produces
    # entirely different bf16 noise (0.387 vs 1.625 on the first element at key(0)) while satisfying
    # every dtype assertion -- the T3b-1 lesson: compare the drawn quantity, never a proxy for it.
    reference = jax.random.normal(keys[0], z_video.shape, dtype=z_video.dtype)
    downcast = jax.random.normal(keys[0], z_video.shape, dtype=jnp.float32).astype(z_video.dtype)
    drawn = np.asarray(latents[:, :, 1:], np.float32)
    assert np.array_equal(drawn, np.asarray(reference[:, :, 1:], np.float32)), "not deployment's draw"
    assert not np.array_equal(
        drawn, np.asarray(downcast[:, :, 1:], np.float32)
    ), "fp32-then-cast is a different draw and must not be mistaken for this one"
    assert np.array_equal(np.asarray(latents[:, :, :1], np.float32), np.asarray(z_i0, np.float32))
    assert not np.array_equal(np.asarray(latents[:, :, 1:]), np.zeros_like(np.asarray(latents[:, :, 1:])))


def test_the_ssim_is_the_deployed_definition_held_by_a_drift_tripwire():
    """§5-5's ruling: the settled evaluator is not rewired, and parity is discharged BY TEST — a
    verbatim copy, plus a tripwire that fails the moment the original changes."""
    deployed = _DEPLOYED.read_text(encoding="utf-8")
    copied = inspect.getsource(anchor.frame_ssim)
    body = copied.split('"""')[-1]
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    assert len(lines) >= 12, "the copy must be the whole definition, not a paraphrase"
    for line in lines:
        assert line in deployed, f"this line is not in the deployed evaluator: {line!r}"

    identical = np.random.default_rng(0).random((3, 16, 16, 3)).astype(np.float32)
    assert anchor.frame_ssim(identical, identical) == pytest.approx(1.0, abs=1e-6)
    assert anchor.frame_ssim(np.zeros((3, 16, 16, 3), np.float32), identical) < 0.5


def test_a_nonfinite_ssim_is_refused_rather_than_quietly_dropped():
    """DECLARED DEVIATION from the deployed summary, which filters non-finite SSIMs so a worker
    without scikit-image still produces a number. exp_06 refuses instead: a mean over whichever
    samples happened to work is a different estimand. It cannot change the anchor value — all four
    anchor samples are finite — but it changes what a broken worker does, which is the point."""
    rows = _rows(ssim_avg=0.3)
    assert _measured(rows=rows).means["mean_ssim"] == pytest.approx(0.3)
    rows[2]["ssim_avg"] = float("nan")
    with pytest.raises(ValueError, match="non-finite ssim"):
        _measured(rows=rows)
    assert "finite_ssim" not in _MODULE_PATH.read_text(encoding="utf-8"), "the deployed filter is not inherited"


def test_the_summary_is_the_deployed_aggregation():
    deployed = _DEPLOYED.read_text(encoding="utf-8")
    for fragment in ('"mean_latent_mse": mean_latent', '"mean_pixel_mse": mean_pixel'):
        assert fragment in deployed
    grid = anchor.DEPLOYED_GRID_SHA256
    rows = [
        {"name": "a", "latent_mse": 1.0, "pixel_mse": 0.10, "ssim_avg": 0.20, "num_steps": 25, "grid_sha256": grid},
        {"name": "b", "latent_mse": 2.0, "pixel_mse": 0.20, "ssim_avg": 0.40, "num_steps": 25, "grid_sha256": grid},
    ]
    summary = _measured(rows=rows)
    assert summary.means == {
        "mean_latent_mse": 1.5,
        "mean_pixel_mse": pytest.approx(0.15),
        "mean_ssim": pytest.approx(0.30),
    }
    assert summary.num_samples == 2 and summary.sample_names == ["a", "b"]
    assert summary.payload["checkpoint_step"] == 30000 and summary.num_steps == 25
    # The digest covers every one of those facts, so a re-typed summary is a different artifact.
    assert summary.digest == hashlib.sha256(json.dumps(summary.payload, sort_keys=True).encode()).hexdigest()


def _execution(z_pred, *, num_steps=25, grid=None):
    return anchor.RolloutExecution(
        z_pred=z_pred,
        num_steps=num_steps,
        grid_size=26,
        guide_scale=5.0,
        draw_key_sha256="0" * 64,
        grid_sha256=anchor.DEPLOYED_GRID_SHA256 if grid is None else grid,
    )


def test_sample_metrics_decodes_both_sides_through_the_one_seam():
    """The decode is deployment's ``_decode_latents_to_video(_denormalize_latents(...))``; the metric
    definitions are deployment's. Both sides go through the SAME callable, so a decode that
    normalizes differently cannot flatter the prediction."""
    seen = []

    def decode_fn(latents):
        seen.append(np.asarray(latents, np.float32).sum())
        frames = np.asarray(latents, np.float32).reshape(1, 2, 4, 4, 3)
        return jnp.asarray(np.clip(frames, 0.0, 1.0))

    z_pred = jnp.full((1, 2, 4, 4, 3), 0.5, jnp.float32).reshape(1, 96)
    z_video = jnp.full((1, 2, 4, 4, 3), 0.5, jnp.float32).reshape(1, 96)
    with pytest.raises(TypeError, match="RolloutExecution"):
        anchor.sample_metrics(z_pred, z_video, decode_fn=decode_fn)
    metrics = anchor.sample_metrics(_execution(z_pred), z_video, decode_fn=decode_fn)
    assert metrics["num_steps"] == 25, "the horizon travels with the prediction into the row"
    assert len(seen) == 2, "prediction and ground truth are decoded by the same seam"
    assert metrics["latent_mse"] == pytest.approx(0.0)
    assert metrics["pixel_mse"] == pytest.approx(0.0)
    assert metrics["ssim_avg"] == pytest.approx(1.0, abs=1e-6)


# =============================================================================================
# 4. The frozen DEV-64 benchmark row (plan §3c(b)) — published once, adopted, never re-derived.
# =============================================================================================


def _score_table(cohort, per_example, *, condition="true", arm="historical", identity=None, incomplete=False):
    """A bound table with every identity a produced one carries — the artifact the row derives from."""
    rows = {
        name: {
            "ssim": value,
            "mse": 1.0,
            "actions_from": name,
            "actions_sha256": hashlib.sha256(str(name).encode()).hexdigest(),
            "draw_key_sha256": anchor.draw_key_digest(anchor.evaluation_draw_key(name)),
            "num_steps": 25,
        }
        for name, value in per_example.items()
    }
    return anchor.build_score_table(
        rows=rows,
        cohort=cohort,
        condition=condition,
        arm=arm,
        checkpoint=_historical_identity() if identity is None else identity,
        num_steps=25,
        allow_incomplete=incomplete,
    )


def _row(tmp_path, name="benchmark.json", per_example=None, cohort=None):
    cohort = instrument.load_dev_cohort(_DEV) if cohort is None else cohort
    table = per_example or {example: 0.25 + index * 1e-4 for index, example in enumerate(cohort.names)}
    return (
        str(tmp_path / name),
        cohort,
        anchor.freeze_benchmark_row(str(tmp_path / name), table=_score_table(cohort, table)),
    )


def test_the_benchmark_row_is_frozen_with_full_provenance(tmp_path):
    path, cohort, row = _row(tmp_path)
    assert row["protocol"] == anchor.BENCHMARK_PROTOCOL
    assert row["manifest_sha256"] == cohort.manifest_sha256 and row["cohort"] == "dev64"
    assert row["example_count"] == 64 and len(row["per_example"]) == 64
    assert row["checkpoint"]["step"] == 30000 and row["checkpoint"]["run_name"] == anchor.HISTORICAL_ANCHOR.run_name
    assert row["num_steps"] == 25 and row["score_table_sha256"], "the row names the table it was derived from"
    assert row["mean_ssim"] == pytest.approx(sum(row["per_example"].values()) / 64)
    assert Path(path).exists()


def test_re_freezing_adopts_an_identical_row_and_refuses_a_different_one(tmp_path):
    """Issue #10's discipline: a published artifact is adopted, never rewritten. Re-deriving the
    benchmark per call is how a 'frozen' baseline silently tracks the code that is being judged
    against it."""
    path, cohort, first = _row(tmp_path)
    again = anchor.freeze_benchmark_row(
        path, table=_score_table(cohort, {name: first["per_example"][name] for name in cohort.names})
    )
    assert again == first, "an identical republication is adopted"
    with pytest.raises(ValueError, match="already published"):
        anchor.freeze_benchmark_row(path, table=_score_table(cohort, dict.fromkeys(cohort.names, 0.9)))


def test_a_loaded_benchmark_row_is_digest_checked_AND_semantically_validated(tmp_path):
    """A digest proves nobody edited the file; it does not prove the file is a benchmark row. The
    reviewer's MAJOR: ``load_benchmark_row`` verified the digest but not even the protocol."""
    path, cohort, row = _row(tmp_path)
    assert anchor.load_benchmark_row(path) == row
    payload = json.loads(Path(path).read_text())
    payload["payload"]["mean_ssim"] = 0.99
    Path(path).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        anchor.load_benchmark_row(path)

    # A CONSISTENTLY re-digested artifact still has to be a benchmark row.
    for mutate, message in (
        (lambda p: p.update(protocol="exp06.something.v1"), "is not a"),
        (lambda p: p.update(example_count=63), "count and the table"),
        (lambda p: p.update(num_steps=10), "horizon"),
        (lambda p: p.update(mean_ssim=0.99), "not the mean of its own table"),
        (lambda p: p.update(checkpoint={}), "names the checkpoint"),
    ):
        target = str(tmp_path / f"edited_{abs(hash(message)) % 9999}.json")
        edited = dict(row)
        edited.pop("sha256", None)
        mutate(edited)
        Path(target).write_text(
            json.dumps(
                {"payload": edited, "sha256": hashlib.sha256(json.dumps(edited, sort_keys=True).encode()).hexdigest()}
            )
        )
        with pytest.raises(ValueError, match=message):
            anchor.load_benchmark_row(target)


def test_the_benchmark_row_can_only_be_derived_from_a_bound_score_table(tmp_path):
    """It used to take the table, the checkpoint, the code SHA and the horizon as four independent
    caller assertions. TEST still cannot reach this path: a table is built from a LOADED cohort, and
    the DEV instrument issues none for a TEST manifest."""
    with pytest.raises(TypeError, match="ScoreTable"):
        anchor.freeze_benchmark_row(str(tmp_path / "forged.json"), table={"ep61399_v0_s00000": 0.9})
    cohort = instrument.load_dev_cohort(_DEV)
    with pytest.raises(ValueError, match="condition"):
        anchor.freeze_benchmark_row(
            str(tmp_path / "wrongcond.json"),
            table=_score_table(cohort, dict.fromkeys(cohort.names, 0.3), condition="zero"),
        )
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        instrument.load_dev_cohort(_TEST)


def test_the_row_must_cover_the_cohort_exactly(tmp_path):
    cohort = instrument.load_dev_cohort(_DEV)
    with pytest.raises(ValueError, match="does not cover"):
        _score_table(cohort, dict.fromkeys(cohort.names[:63], 0.3))
    with pytest.raises(ValueError, match="does not cover|outside the cohort"):
        _score_table(cohort, {**dict.fromkeys(cohort.names, 0.3), "stranger": 0.3})
    with pytest.raises(ValueError, match="must be finite"):
        anchor.freeze_benchmark_row(
            str(tmp_path / "nan.json"), table=_score_table(cohort, {name: float("nan") for name in cohort.names})
        )


# =============================================================================================
# 5. TEST may not reach the anchor path, and the unwired boundary says what it is.
# =============================================================================================


def test_a_test_example_reaching_the_anchor_samples_is_refused():
    """The anchor's four samples come from the val DIRECTORY in file order — and exp_04 drew BOTH
    DEV-64 and TEST-64 from that same directory, so "the first four records" can legitimately land
    on a TEST example. Names are used here only to REFUSE, never to score.
    """
    test_names = [str(row["name"]) for row in json.loads(Path(_TEST).read_text())["rows"]]
    anchor.assert_no_test_examples(("not_a_test_name", "another"), test_manifest_path=_TEST)
    with pytest.raises(ValueError, match="TEST-64"):
        anchor.assert_no_test_examples(("fine", test_names[0]), test_manifest_path=_TEST)


def test_the_only_remaining_seam_is_the_DEVICE_and_it_is_implemented(tmp_path):
    """The Planner's ruling, as a test: "name the boundary in the error" was accepted for DEVICE work
    and does NOT extend to the orchestration. So the module raises no ``NotImplementedError``
    anywhere, and the one function that cannot run without weights is ``load_device_backend`` — which
    is implemented against the deployed pipeline rather than stubbed."""
    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "NotImplementedError" not in source
    backend_source = inspect.getsource(anchor.load_device_backend)
    for token in ("_load_wan_pipeline", "_decode_latents_to_video", "build_cfg_velocity_fn"):
        assert token in backend_source, token
    # ...and DeviceBackend itself is pure composition, so the whole path is exercisable off-device.
    assert "pragma: no cover" not in inspect.getsource(anchor.DeviceBackend)


def test_config_is_read_only_through_the_declared_helpers_never_three_argument_getattr():
    """Issue #11: ``getattr(config, key, default)`` on a pyconfig ``HyperParameters`` never falls back
    — ``__getattr__`` raises ``ValueError`` — and it has killed two TPU jobs in this campaign."""
    source = _MODULE_PATH.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "getattr":
            assert len(node.args) < 3, f"issue #11: {ast.unparse(node)}"
    assert "optional_config_value" in source, "genuinely optional keys go through exp_04's reader"
    assert "generate_wan_null_adapter" not in source, "exp_05's S9 tripwire asserts that file does not exist"
    digest = hashlib.sha256(Path(_DEV).read_bytes()).hexdigest()
    assert instrument.J0_DEV64_SHA256 == digest and math.isfinite(anchor.ANCHOR_REL_TOLERANCE)


# =============================================================================================
# 6. Self-review strengthenings (no reviewer was available this round; these are the attacks I
#    would have run against my own first draft, each now a test).
# =============================================================================================


def test_the_certificate_cannot_name_samples_the_summary_did_not_screen():
    """First draft took ``sample_names`` as a free argument — so a certificate could name four
    innocuous samples for a measurement taken on four others. The names now come from the verdict."""
    assert "sample_names" not in inspect.signature(anchor.anchor_certificate).parameters
    assert list(inspect.signature(anchor.anchor_certificate).parameters) == ["verdict"]
    certificate = anchor.anchor_certificate(anchor.reproduce_anchor(_measured()))
    assert certificate["sample_names"] == list(anchor.HISTORICAL_ANCHOR.sample_names)
    # ...and there is no verdict a caller can hand-build that carries different names, because the
    # only thing a certificate reads is the Measurement, whose names came from the aggregation.
    with pytest.raises(TypeError, match="AnchorVerdict"):
        anchor.anchor_certificate({"reproduced": True, "sample_names": ["s0", "s1", "s2", "s3"]})


def test_the_summary_screens_test_examples_and_refuses_an_unnamed_sample():
    """The screen lives INSIDE the aggregation, so no route reaches an anchor summary without it."""
    assert "test_manifest_path" in inspect.signature(anchor.summarize_samples).parameters
    intruder = str(json.loads(Path(_TEST).read_text())["rows"][0]["name"])
    with pytest.raises(ValueError, match="TEST-64"):
        _measured(rows=_rows(("s0", intruder)))
    nameless = _rows(("s0", "s1"))
    nameless[1].pop("name")
    with pytest.raises(ValueError, match="must carry its name"):
        _measured(rows=nameless)


def test_the_test_screen_is_pinned_to_the_published_manifest(tmp_path):
    """A guard that reads whichever file it is handed is defanged by handing it an empty one."""
    doctored = tmp_path / "test64.json"
    payload = json.loads(Path(_TEST).read_text())
    payload["rows"] = []
    doctored.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not exp_04's published TEST-64 manifest"):
        anchor.assert_no_test_examples(("anything",), test_manifest_path=str(doctored))
    assert anchor.J0_TEST64_SHA256 == hashlib.sha256(Path(_TEST).read_bytes()).hexdigest()


def test_an_edited_publication_cannot_be_adopted(tmp_path):
    """Adoption compared two self-declared digests, so a hand-edited payload whose header still
    carried the original hash was adopted as if it were the published one."""
    path = str(tmp_path / "cert.json")
    payload = {"protocol": anchor.ANCHOR_PROTOCOL, "value": 1}
    anchor.publish_certificate(path, payload)
    stored = json.loads(Path(path).read_text())
    stored["payload"]["value"] = 999  # the header keeps the original digest
    Path(path).write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="does not describe its own payload"):
        anchor.publish_certificate(path, payload)


def test_sample_metrics_refuses_a_sample_whose_ssim_cannot_be_computed():
    """The deployed helper returns NaN for frames too small to window; a NaN that reaches a summary
    is a silently dropped sample, so it is refused where it is produced."""

    def decode_fn(latents):
        return jnp.asarray(np.asarray(latents, np.float32).reshape(1, 1, 2, 2, 3))

    tiny = jnp.zeros((1, 12), jnp.float32)
    assert not math.isfinite(anchor.frame_ssim(np.zeros((1, 2, 2, 3), np.float32), np.zeros((1, 2, 2, 3), np.float32)))
    with pytest.raises(ValueError, match="not finite"):
        anchor.sample_metrics(_execution(tiny), tiny, decode_fn=decode_fn)


def test_the_anchor_is_documented_as_a_wiring_check_and_never_a_quality_baseline():
    """Standing rule (Planner, after pulling the run's summary.csv): the four anchor samples are four
    windows of ONE episode (ep10099_v0_s00000/_s00004/_s00008/_s00012, per-sample SSIM 0.175-0.484),
    so 0.2946 is a mean over correlated windows of a single clip. It certifies the wiring; the DEV-64
    benchmark row is the quality baseline. A future reader must not be able to miss this."""
    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "MUST NEVER BE QUOTED AS A QUALITY BASELINE" in source
    assert "ep10099_v0_s00000" in source and "single episode".lower() in source.lower()
    flowed = " ".join(source.split())  # the docstring wraps; the claim must survive rewrapping
    assert "8× tighter than the between-sample SEM" in flowed, "the tolerance's basis is stated"
    assert "SSIM sd 0.1336" in flowed and "22.7%" in flowed
    assert "0.29460108026184817" in source, "the record is verified against the run's own summary.json"


# =============================================================================================
# 7. T5a review items 6 and 7, carried into T5b.
# =============================================================================================


def test_the_expectations_come_from_the_run_report_ARTIFACT_not_from_kwargs(tmp_path):
    """Item 6: ``expected_step/metric/arm/k`` supplied as loose kwargs make the artifact-vs-report
    check only as strong as the caller's memory. The report is now a published artifact, produced
    from the loop's own RunReport, and the evaluator reads it."""
    pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("tensorflow")
    directory = _tree(tmp_path, "reported")
    report = loop.RunReport(
        state=_state(2),
        history=(EvalRecord(step=2, dev_metric=0.5, train_metric=1.0),),
        verdict=loop.stop_verdict([EvalRecord(step=2, dev_metric=0.5, train_metric=1.0)]),
        retained_step=2,
        steps_run=2,
        draw_log=(),
    )
    path = str(tmp_path / "run_report.json")
    published = anchor.publish_run_report(path, report, arm="rollout", k_b=2, num_steps=25)
    assert published["retained_step"] == 2 and published["arm"] == "rollout" and published["k_b"] == 2

    state, identity = anchor.restore_from_report(directory, _state(), report_path=path)
    assert int(state.step) == 2 and identity.dev_metric == 0.5 and identity.run_name == "reported"
    assert list(inspect.signature(anchor.restore_from_report).parameters) == ["ckpt_dir", "template", "report_path"]

    # An edited report cannot be used to bless a checkpoint it does not describe.
    stored = json.loads(Path(path).read_text())
    stored["payload"]["retained_step"] = 4
    Path(path).write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="digest"):
        anchor.restore_from_report(directory, _state(), report_path=path)


def test_a_report_whose_retained_step_has_no_metric_is_refused(tmp_path):
    report = loop.RunReport(
        state=_state(0),
        history=(),
        verdict=loop.stop_verdict([]),
        retained_step=None,
        steps_run=0,
        draw_log=(),
    )
    with pytest.raises(ValueError, match="selected no checkpoint"):
        anchor.publish_run_report(str(tmp_path / "empty.json"), report, arm="rollout", k_b=2, num_steps=25)


def test_the_certificate_stamps_and_CHECKS_the_rollout_horizon(tmp_path):
    """Item 7 became review-pass-2's MAJOR: a wrong horizon failed loudly but recorded silently, and
    then the recorded value turned out to be whatever the certifier typed. Now it can only be what
    ran — and the refusal happens where the rows are formed, before anything is certified."""
    assert anchor.DEPLOYED_SAMPLING_STEPS == 25
    certificate = anchor.anchor_certificate(anchor.reproduce_anchor(_measured()))
    assert certificate["num_steps"] == 25
    with pytest.raises(ValueError, match="horizons"):
        _measured(rows=_rows(num_steps=20))


# =============================================================================================
# 8. Remote artifacts (review pass 2, T5a BLOCKER 2): every exp_06 root is a gs:// URI.
# =============================================================================================


def test_pathlib_would_have_silently_written_a_LOCAL_file_for_a_gs_uri():
    """The defect, demonstrated: ``pathlib`` does not raise on a URI, it drops a slash. Every
    publication and load in exp_06 went through it, so a run's report, certificates and benchmark row
    would have been written to a local directory named ``gs:`` and been invisible in the bucket."""
    from maxdiffusion import pos_rollout_support

    assert str(Path("gs://bucket/run/report.json")) == "gs:/bucket/run/report.json"
    assert pos_rollout_support.is_remote("gs://bucket/x") and not pos_rollout_support.is_remote("/tmp/x")


def test_a_certificate_round_trips_through_a_remote_uri(fake_gs, tmp_path):
    remote = "gs://v6_east1d/artifacts/exp06/run/anchor.json"
    published = anchor.publish_certificate(remote, anchor.anchor_certificate(anchor.reproduce_anchor(_measured())))
    assert remote in fake_gs.blobs, "the artifact must land in the bucket, not beside it"
    assert "gs://v6_east1d/artifacts/exp06/run" in fake_gs.dirs
    assert not Path("gs:").exists(), "a local gs: directory means the URI was treated as a path"
    # ...and the adopt-on-republication path reads the same bytes back.
    assert anchor.publish_certificate(remote, {k: v for k, v in published.items() if k != "sha256"}) == published


def test_the_run_report_and_benchmark_row_round_trip_remotely(fake_gs):
    report = loop.RunReport(
        state=_state(2),
        history=(EvalRecord(step=2, dev_metric=0.5, train_metric=1.0),),
        verdict=loop.stop_verdict([EvalRecord(step=2, dev_metric=0.5, train_metric=1.0)]),
        retained_step=2,
        steps_run=2,
        draw_log=(),
    )
    remote_report = "gs://bucket/run/run_report.json"
    anchor.publish_run_report(remote_report, report, arm="rollout", k_b=2, num_steps=25)
    assert anchor.load_run_report(remote_report)["retained_step"] == 2

    cohort = instrument.load_dev_cohort(_DEV)
    remote_row = "gs://bucket/run/benchmark.json"
    frozen = anchor.freeze_benchmark_row(remote_row, table=_score_table(cohort, dict.fromkeys(cohort.names, 0.25)))
    assert anchor.load_benchmark_row(remote_row) == frozen
    assert set(fake_gs.blobs) == {remote_report, remote_row}


# =============================================================================================
# 9. STAMPED != BOUND (review pass 2). Provenance must be DERIVED from the artifact that produced
#    it. Every test below reproduces one of the reviewer's EXECUTED attacks.
# =============================================================================================


def _identity(*, run_name=None, step=30000, source="historical"):
    """A restore's own account of what it opened. Production builds these only inside a restore."""
    name = anchor.HISTORICAL_ANCHOR.run_name if run_name is None else run_name
    return anchor.CheckpointIdentity(run_name=name, step=step, root=f"gs://b/{name}/checkpoints", source=source)


def _anchor_rows(names=None, num_steps=25):
    names = anchor.HISTORICAL_ANCHOR.sample_names if names is None else names
    return [
        {
            "name": n,
            "latent_mse": 1.496,
            "pixel_mse": 0.0983,
            "ssim_avg": 0.2946,
            "num_steps": num_steps,
            "grid_sha256": anchor.DEPLOYED_GRID_SHA256,
        }
        for n in names
    ]


def _measurement(rows=None, identity=None):
    return anchor.summarize_samples(
        _anchor_rows() if rows is None else rows,
        checkpoint=_identity() if identity is None else identity,
        code_sha=_SHA,
        model_revision=_REVISION,
        test_manifest_path=_TEST,
    )


def test_the_anchor_is_bound_to_its_exact_four_sample_names_in_order():
    """THE reviewer's attack: the recorded means with FOUR UNRELATED NAMES returned reproduced=True.

    The anchor is four windows of one episode, in the order the val directory yielded them. Any other
    four samples are a different measurement that happens to average to the same number.
    """
    assert anchor.HISTORICAL_ANCHOR.sample_names == (
        "ep10099_v0_s00000",
        "ep10099_v0_s00004",
        "ep10099_v0_s00008",
        "ep10099_v0_s00012",
    )
    assert anchor.reproduce_anchor(_measurement()).reproduced

    with pytest.raises(ValueError, match="sample"):
        anchor.reproduce_anchor(_measurement(rows=_anchor_rows(("s0", "s1", "s2", "s3"))))
    shuffled = list(anchor.HISTORICAL_ANCHOR.sample_names)
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    with pytest.raises(ValueError, match="order"):
        anchor.reproduce_anchor(_measurement(rows=_anchor_rows(tuple(shuffled))))


def test_the_anchor_refuses_a_foreign_checkpoint_identity():
    """The reviewer certified checkpoint ``{"run_name": "some-other-run", "step": 1}``."""
    with pytest.raises(ValueError, match="some-other-run|historical run"):
        anchor.reproduce_anchor(_measurement(identity=_identity(run_name="some-other-run")))
    with pytest.raises(ValueError, match="step"):
        anchor.reproduce_anchor(_measurement(identity=_identity(step=1)))


def test_a_measurement_cannot_be_described_by_a_caller_mapping():
    with pytest.raises(TypeError, match="CheckpointIdentity"):
        anchor.summarize_samples(
            _anchor_rows(),
            checkpoint={"run_name": anchor.HISTORICAL_ANCHOR.run_name, "step": 30000},
            code_sha=_SHA,
            model_revision=_REVISION,
            test_manifest_path=_TEST,
        )
    with pytest.raises(TypeError, match="Measurement"):
        anchor.reproduce_anchor({"mean_ssim": 0.2946, "num_samples": 4, "checkpoint_step": 30000})


def test_the_horizon_is_ENFORCED_at_the_scoring_boundary_not_supplied_by_a_caller():
    """``rollout_prediction(num_steps=1)`` executed and was later certified as ``num_steps=25``."""
    assert "num_steps" not in inspect.signature(anchor.rollout_prediction).parameters
    rows = _anchor_rows(num_steps=1)
    with pytest.raises(ValueError, match="25|horizon"):
        anchor.summarize_samples(
            rows, checkpoint=_identity(), code_sha=_SHA, model_revision=_REVISION, test_manifest_path=_TEST
        )
    mixed = _anchor_rows()
    mixed[1] = {**mixed[1], "num_steps": 24}
    with pytest.raises(ValueError, match="horizon|25"):
        anchor.summarize_samples(
            mixed, checkpoint=_identity(), code_sha=_SHA, model_revision=_REVISION, test_manifest_path=_TEST
        )


def test_the_certificate_is_DERIVED_from_the_measurement_and_takes_nothing_else():
    assert list(inspect.signature(anchor.anchor_certificate).parameters) == ["verdict"]
    measurement = _measurement()
    certificate = anchor.anchor_certificate(anchor.reproduce_anchor(measurement))
    assert certificate["checkpoint"] == measurement.checkpoint.payload()
    assert certificate["sample_names"] == list(anchor.HISTORICAL_ANCHOR.sample_names)
    assert certificate["num_steps"] == 25 and certificate["measurement_sha256"] == measurement.digest
    assert certificate["code_sha"] == _SHA and certificate["model_revision"] == _REVISION


# =============================================================================================
# 10. The evaluator must actually RUN (review pass 2, T5a BLOCKER 1). Phase dispatch, and the
#     anchor completing before any new-arm scoring.
# =============================================================================================


class _Config(dict):
    """A stand-in for ``pyconfig.HyperParameters``: attribute reads, and ValueError on an unknown key."""

    def __getattr__(self, key):
        if key not in self:
            raise ValueError(f"Key {key} not in config")
        return self[key]

    def get_keys(self):
        return dict(self)


def _config(**overrides):
    values = {
        "pos_eval_phase": "anchor",
        "base_output_directory": "gs://bucket/run-x/eval_anchor_att-20260808T000000Z",
        "checkpoint_dir": "gs://bucket/run-x/checkpoints",
        "pos_run_report": "gs://bucket/run-x/run_report.json",
        "pos_dev_manifest": _DEV,
        "pos_test_manifest": _TEST,
        "side_adapter_sampling_steps": 25,
        "side_adapter_guide_scale": 5.0,
        "pos_rollout_k": 2,
        "seed": 0,
        "run_name": "run-x",
    }
    values.update(overrides)
    return _Config(values)


def test_the_entry_point_dispatches_phases_and_nothing_in_the_module_raises_notimplemented():
    """The reviewer's BLOCKER: ``main()`` always reached a seam that always raised, so no phase could
    restore, score, reproduce the anchor or freeze the benchmark. "Name the boundary in the error"
    was accepted for DEVICE work and does not extend to the ORCHESTRATION."""
    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "NotImplementedError" not in source
    assert anchor.EVAL_PHASES == ("anchor", "benchmark", "gates", "confirm")
    for name in ("run_evaluation", "run_anchor_phase", "run_benchmark_phase", "run_gates_phase"):
        assert callable(getattr(anchor, name))
    with pytest.raises(ValueError, match="not a phase"):
        anchor.run_evaluation(_config(pos_eval_phase="everything"), backend=object())
    with pytest.raises(ValueError, match="not a phase"):
        anchor.run_evaluation(_config(pos_eval_phase=""), backend=object())


def test_the_protocol_root_is_derived_from_the_attempt_scoped_artifact_root_and_fails_closed():
    """Issue #13 gives each phase its own attempt-scoped root; the cross-phase artifacts live one
    level up. A root that does not carry the phase is refused rather than guessed at."""
    root = "gs://bucket/run-x/eval_gates_att-20260808T101010Z"
    assert anchor.protocol_root_for(root, phase="gates") == "gs://bucket/run-x"
    with pytest.raises(ValueError, match="attempt-scoped"):
        anchor.protocol_root_for("gs://bucket/run-x/somewhere-else", phase="gates")
    with pytest.raises(ValueError, match="attempt-scoped"):
        anchor.protocol_root_for(root, phase="confirm")


def test_no_new_arm_phase_can_run_before_the_anchor_has_been_reproduced(fake_gs):
    """Plan §3c: the anchor is the wiring proof, so it completes BEFORE anything new is measured.
    That ordering is enforced by requiring the anchor's own certificate, not by the launcher."""
    for phase in ("benchmark", "gates", "confirm"):
        config = _config(
            pos_eval_phase=phase,
            base_output_directory=f"gs://bucket/run-x/eval_{phase}_att-1",
        )
        with pytest.raises(ValueError, match="anchor"):
            anchor.run_evaluation(config, backend=object())

    # A published-but-failing anchor is not a licence either.
    measurement = _measurement(rows=_anchor_rows()[:3] + [{**_anchor_rows()[3], "ssim_avg": 0.9}])
    anchor.publish_certificate(
        "gs://bucket/run-x/anchor_certificate.json", anchor.anchor_certificate(anchor.reproduce_anchor(measurement))
    )
    with pytest.raises(ValueError, match="did not reproduce"):
        anchor.require_anchor("gs://bucket/run-x")


# =============================================================================================
# 11. Holes THIS ROUND'S battery found in this round's own tests. Each mutant below survived the
#     first battery run, and each survivor was a test weakness rather than a code weakness.
# =============================================================================================


def test_the_anchor_names_a_WRONG_SET_and_a_WRONG_ORDER_differently():
    """Battery G01: dropping the set check left the ORDER check firing, so a foreign-sample anchor
    still failed — but for the wrong reason, and with a message that would send an operator hunting
    an ordering bug. The two failures are distinct and are now asserted distinctly."""
    with pytest.raises(ValueError, match="recorded samples"):
        anchor.reproduce_anchor(_measured(names=("s0", "s1", "s2", "s3")))
    shuffled = list(anchor.HISTORICAL_ANCHOR.sample_names)
    shuffled[0], shuffled[3] = shuffled[3], shuffled[0]
    with pytest.raises(ValueError, match="in the order the val directory yielded them"):
        anchor.reproduce_anchor(_measured(names=shuffled))


def test_a_restored_identity_takes_its_run_name_from_the_ROOT_and_from_nothing_else():
    """Battery G06: reading ``run_name`` out of the checkpoint's own JSON survived, because today's
    checkpoints carry no such key — it would become live the moment one did. The derivation is pinned
    structurally instead: the only expression that can produce a run name is the root's."""
    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    produced = [
        ast.unparse(keyword.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "CheckpointIdentity"
        for keyword in node.keywords
        if keyword.arg == "run_name"
    ]
    assert produced, "no CheckpointIdentity is constructed in this module?"
    assert set(produced) == {"_run_name_from_root(ckpt_dir)", "run_name"}, produced
    # ...and `run_name` in the anchor restore is itself the root's, checked against the record.
    restore = _function_node(source, "restore_anchor_checkpoint")
    assigned = [
        ast.unparse(node.value)
        for node in ast.walk(restore)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "run_name" for t in node.targets)
    ]
    assert assigned == ["_run_name_from_root(ckpt_dir)"], assigned


def test_the_anchor_restores_step_30000_and_not_whatever_is_latest(tmp_path):
    """Battery G07: falling back to ``manager.latest_step()`` survived because the fixture wrote only
    one step. A real run has many, and the anchor is a fact about ONE of them."""
    pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("tensorflow")
    root = str(tmp_path / anchor.HISTORICAL_ANCHOR.run_name / "checkpoints")
    manager = loop.build_checkpoint_manager(root)
    for step in (30000, 40000):
        loop.save_checkpoint(
            manager,
            dataclasses.replace(_state(), step=step),
            dev_metric=0.5,
            history=[EvalRecord(step=step, dev_metric=0.5, train_metric=1.0)],
            arm="rollout",
            k_b=2,
        )
    assert manager.latest_step() == 40000, "the resume tree's latest is NOT the anchor"
    state, identity = anchor.restore_anchor_checkpoint(root, _state())
    assert int(state.step) == 30000 and identity.step == 30000
    assert identity.run_name == anchor.HISTORICAL_ANCHOR.run_name and identity.source == "historical"

    stranger = str(tmp_path / "some-other-run" / "checkpoints")
    loop.save_checkpoint(
        loop.build_checkpoint_manager(stranger),
        dataclasses.replace(_state(), step=30000),
        dev_metric=0.5,
        history=[EvalRecord(step=30000, dev_metric=0.5, train_metric=1.0)],
        arm="rollout",
        k_b=2,
    )
    with pytest.raises(ValueError, match="belongs to run"):
        anchor.restore_anchor_checkpoint(stranger, _state())


def test_the_measurement_must_carry_the_horizon_it_publishes():
    """Battery G10 was EQUIVALENT (the horizon set is already pinned to {25}, so ``max`` of it is 25).
    What is not equivalent is dropping the field: a certificate with no horizon cannot be compared
    with a benchmark row, so the schema carries it and the anchor checks it."""
    measurement = _measured()
    assert measurement.payload["num_steps"] == anchor.DEPLOYED_SAMPLING_STEPS
    stripped = anchor.Measurement(
        payload={k: v for k, v in measurement.payload.items() if k != "num_steps"},
        digest=measurement.digest,
        checkpoint=measurement.checkpoint,
    )
    with pytest.raises(KeyError):
        anchor.reproduce_anchor(stripped)


# =============================================================================================
# 12. F3a — the DEVICE/MEASUREMENT boundary (re-review EV-1, EV-2, EV-6). Each test below
#     reproduces a probe the reviewer EXECUTED against this module.
# =============================================================================================


def test_a_nonfinite_latent_or_pixel_mse_can_never_certify_as_reproduced():
    """EV-2 BLOCKER, executed: ``abs(NaN) > tolerance`` is FALSE, so a ``latent_mse=NaN``
    measurement returned ``reproduced=True``. Only SSIM was ever checked for finiteness."""
    for metric, field in (("mean_latent_mse", "latent_mse"), ("mean_pixel_mse", "pixel_mse")):
        rows = _rows()
        rows[1] = {**rows[1], field: float("nan")}
        with pytest.raises(ValueError, match="not finite"):
            _measured(rows=rows)
        rows[1] = {**rows[1], field: float("inf")}
        with pytest.raises(ValueError, match="not finite"):
            _measured(rows=rows)
        del metric
    # ...and the same at the aggregate/deviation level: a mean that is not finite decides nothing.
    finite = _measured()
    assert all(math.isfinite(value) for value in finite.means.values())
    verdict = anchor.reproduce_anchor(finite)
    assert all(math.isfinite(value) for value in verdict.deviations.values())


def test_sample_metrics_refuses_every_nonfinite_metric_not_only_the_ssim():
    """The refusal belongs where the number is produced — the six-entry-point discipline."""

    def decode_fn(latents):
        # Deliberately FINITE whatever it is handed: a decoder that sanitises its output is exactly
        # how a NaN latent MSE reaches a summary while the SSIM check waves it through.
        frames = np.nan_to_num(np.asarray(latents, np.float32)).reshape(1, 2, 8, 8, 3)
        return jnp.asarray(np.clip(frames, 0.0, 1.0))

    z_video = jnp.full((1, 384), 0.5, jnp.float32)
    poisoned = jnp.asarray(np.concatenate([[np.nan], np.full(383, 0.5, np.float32)])).reshape(1, 384)
    metrics = anchor.sample_metrics(_execution(z_video), z_video, decode_fn=decode_fn)
    assert math.isfinite(metrics["ssim_avg"]) and math.isfinite(metrics["latent_mse"])
    with pytest.raises(ValueError, match="not finite"):
        anchor.sample_metrics(_execution(poisoned), z_video, decode_fn=decode_fn)


def test_the_grid_is_bound_to_the_DEPLOYED_VALUES_not_merely_to_its_length():
    """EV-6 MAJOR, executed: an all-ones 26-sigma grid with terminal sigma 1.0 and 25 zero timesteps
    was accepted and labelled a deployed 25-step execution."""
    sigmas, timesteps = anchor.deployed_grid()
    assert int(sigmas.shape[0]) == anchor.DEPLOYED_SAMPLING_STEPS + 1
    assert float(sigmas[-1]) == 0.0 and float(sigmas[0]) == 1.0
    assert anchor.grid_digest(sigmas, timesteps) == anchor.DEPLOYED_GRID_SHA256

    ones = jnp.ones((anchor.DEPLOYED_SAMPLING_STEPS + 1,), jnp.float32)
    zeros = jnp.zeros((anchor.DEPLOYED_SAMPLING_STEPS,), jnp.float32)
    z_i0 = jnp.zeros((1, 4, 1, 2, 2), jnp.float32)
    z_video = jnp.zeros((1, 4, 2, 2, 2), jnp.float32)
    for bad_sigmas, bad_timesteps, message in (
        (ones, zeros, "terminal|deployed grid"),
        (sigmas, zeros, "timestep"),
        (jnp.flip(sigmas), timesteps, "descend|deployed grid"),
    ):
        with pytest.raises(ValueError, match=message):
            anchor.rollout_prediction(
                velocity_fn=lambda *a, **k: z_video,
                sigmas=bad_sigmas,
                timesteps=bad_timesteps,
                context=jnp.zeros((1, 7, 8), jnp.float32),
                z_i0=z_i0,
                z_video=z_video,
                key=anchor.evaluation_draw_key("x"),
                guide_scale=5.0,
            )


def test_the_grid_identity_is_published_downstream():
    """The reviewer's own words: publish that grid identity downstream in the Measurement."""
    measurement = _measured()
    assert measurement.payload["grid_sha256"] == anchor.DEPLOYED_GRID_SHA256
    certificate = anchor.anchor_certificate(anchor.reproduce_anchor(measurement))
    assert certificate["grid_sha256"] == anchor.DEPLOYED_GRID_SHA256


def test_the_backend_casts_its_inputs_to_the_configured_evaluation_dtype():
    """EV-1 BLOCKER: the reader hands float32 and the historical evaluator casts z_i0, z_video and
    actions to ``weights_dtype`` — bf16 in the pinned config — BEFORE drawing the noise."""
    deployed = _DEPLOYED.read_text(encoding="utf-8")
    for fragment in (
        "weights_dtype = _dtype(config.weights_dtype)",
        'z_i0 = data["z_i0"].astype(weights_dtype)',
        'z_video = data["z_video"].astype(weights_dtype)',
        'actions = data["actions"].astype(weights_dtype)',
        "z = jax.random.normal(rng, z_video.shape, dtype=z_video.dtype)",
    ):
        assert fragment in deployed, f"the deployed boundary moved: {fragment!r}"
    assert "eval_dtype" in inspect.signature(anchor.DeviceBackend.__init__).parameters
    assert "eval_dtype" in inspect.getsource(anchor.load_device_backend)
    assert "weights_dtype" in inspect.getsource(anchor.load_device_backend)


# =============================================================================================
# 13. F3a battery survivors. Eleven mutants survived the first run, and every one survived for the
#     same reason: the finiteness and grid checks are LAYERED, so each is masked by the next and no
#     test isolated any single layer. The layering is the design (it is the loop's six-entry-point
#     discipline); the tests below make each entry point answer for itself.
# =============================================================================================


def _measurement_from(payload_overrides):
    """A Measurement built directly, to reach a layer the aggregation would have caught first."""
    base = _measured()
    payload = {**base.payload, **payload_overrides}
    return anchor.Measurement(payload=payload, digest=base.digest, checkpoint=base.checkpoint)


def test_the_PER_SAMPLE_finiteness_check_answers_for_itself(monkeypatch):
    """F11: the per-row loop was masked by the aggregate. Its refusal must name the SAMPLE."""
    rows = _rows()
    rows[2] = {**rows[2], "latent_mse": float("nan")}
    with pytest.raises(ValueError, match=r"sample 2 .*'s latent_mse"):
        _measured(rows=rows)
    rows = _rows()
    rows[0] = {**rows[0], "pixel_mse": float("-inf")}
    with pytest.raises(ValueError, match=r"sample 0 .*'s pixel_mse"):
        _measured(rows=rows)


def test_the_AGGREGATE_finiteness_check_answers_for_itself():
    """F12: every per-sample value can be finite while their mean is not. ``np.mean`` sums first, so
    two representable-but-enormous values overflow to infinity in the aggregate."""
    huge = 1.7e308
    rows = _rows()
    rows = [{**row, "latent_mse": huge} for row in rows]
    assert all(math.isfinite(row["latent_mse"]) for row in rows), "each sample is finite on its own"
    with pytest.raises(ValueError, match="mean_latent_mse"):
        _measured(rows=rows)


def test_the_ANCHOR_re_checks_the_means_it_was_handed():
    """F13: a Measurement can reach ``reproduce_anchor`` without passing through the aggregation that
    built the tested ones — a loaded artifact is exactly that case."""
    with pytest.raises(ValueError, match="the measured mean_ssim"):
        anchor.reproduce_anchor(_measurement_from({"mean_ssim": float("nan")}))
    with pytest.raises(ValueError, match="the measured mean_latent_mse"):
        anchor.reproduce_anchor(_measurement_from({"mean_latent_mse": float("inf")}))


def test_the_DEVIATION_is_required_to_be_finite_even_when_the_value_is():
    """F14: ``(value - recorded) / recorded`` overflows for a finite but enormous value, and an
    infinite deviation compares FALSE against the band — i.e. it passes."""
    measurement = _measurement_from({"mean_ssim": 1.7e308})
    assert math.isfinite(measurement.means["mean_ssim"]), "the value itself is finite"
    assert not math.isfinite((1.7e308 - 0.2946) / 0.2946), "...but its deviation is not"
    with pytest.raises(ValueError, match="deviation"):
        anchor.reproduce_anchor(measurement)


def test_every_metric_sample_metrics_publishes_goes_through_require_finite():
    """F10: a non-finite pixel MSE cannot be produced without a non-finite SSIM coming with it, so
    the guard is pinned STRUCTURALLY — the four metric values in the returned mapping are
    ``require_finite`` calls, and a future edit that drops one is a visible edit."""
    source = inspect.getsource(anchor.sample_metrics)
    returned = _function_node(source.replace("\n    ", "\n", 1) if False else source, "sample_metrics")
    mapping = [node for node in ast.walk(returned) if isinstance(node, ast.Dict)][0]
    guarded = {
        key.value: isinstance(value, ast.Call) and getattr(value.func, "id", "") == "require_finite"
        for key, value in zip(mapping.keys, mapping.values)
        if isinstance(key, ast.Constant)
    }
    for metric in ("latent_mse", "pixel_mse", "pixel_mae", "ssim_avg"):
        assert guarded.get(metric), f"{metric} is published without require_finite: {guarded}"


def test_the_MONOTONICITY_clause_answers_for_itself():
    """F16: a flipped grid is caught by the terminal-zero clause first, so monotonicity was never
    reached. A grid that still ends at zero, with its timesteps swapped to match, reaches it."""
    sigmas, timesteps = anchor.deployed_grid()
    swapped = np.asarray(sigmas, np.float64).copy()
    swapped[3], swapped[7] = swapped[7], swapped[3]
    swapped_t = np.asarray(timesteps, np.float64).copy()
    swapped_t[3], swapped_t[7] = swapped_t[7], swapped_t[3]
    assert float(swapped[-1]) == 0.0, "the terminal-zero clause cannot be what fires"
    with pytest.raises(ValueError, match="descend"):
        anchor.assert_deployed_grid(jnp.asarray(swapped, jnp.float32), jnp.asarray(swapped_t, jnp.float32))


def test_the_GRID_DIGEST_answers_for_itself():
    """F18: every forged grid the earlier tests used trips a structural clause first. A grid built by
    the deployed constructor at a DIFFERENT flow shift passes all of them — right length, terminal
    zero, strictly descending, corresponding timesteps — and is a different schedule."""
    from maxdiffusion.models.wan.overfit100_sampling import overfit100_sampler_grid

    sigmas, timesteps = overfit100_sampler_grid(
        num_inference_steps=anchor.DEPLOYED_SAMPLING_STEPS,
        flow_shift=3.0,
        sigma_min=anchor.DEPLOYED_SIGMA_MIN,
        sigma_max=anchor.DEPLOYED_SIGMA_MAX,
        num_train_timesteps=anchor.DEPLOYED_NUM_TRAIN_TIMESTEPS,
    )
    values = np.asarray(sigmas, np.float64)
    assert len(values) == 26 and values[-1] == 0.0 and np.all(np.diff(values) < 0)
    assert np.allclose(np.asarray(timesteps, np.float64), values[:-1] * 1000, rtol=1e-6, atol=1e-3)
    with pytest.raises(ValueError, match="different VALUES"):
        anchor.assert_deployed_grid(sigmas, timesteps)


def test_the_SUMMARY_and_the_ANCHOR_each_check_the_grid_a_measurement_ran_on():
    """F20 and F21: the two layers were masked by always being handed the right grid."""
    foreign = "b" * 64
    with pytest.raises(ValueError, match="rolled out on grids"):
        _measured(rows=_rows(grid=foreign))
    with pytest.raises(ValueError, match="taken on grid"):
        anchor.reproduce_anchor(_measurement_from({"grid_sha256": foreign}))


# =============================================================================================
# 14. F3a-fix — the LOADER's config contract (re-review C3).
#
# The one place this module talks to a real ``HyperParameters``, and the one class on which
# three-argument ``getattr`` lies and a missing key RAISES. Every fake config in this suite is a
# dict or a SimpleNamespace, where an undeclared read is harmless — which is exactly what hid a
# ``config.num_train_timesteps`` that no rollout YAML declares. The class is lifted out of
# ``pyconfig.py`` by AST (importing it drags in the whole TPU stack) and bound to the YAML's own
# keys, so the semantics under test are byte-for-byte the deployed ones. This is exp_04's fix-round
# technique, applied to the surface that cost this experiment a review round.
# =============================================================================================

_PYCONFIG = _MODULE_PATH.parent / "pyconfig.py"
_ROLLOUT_YAML = _MODULE_PATH.parent / "configs" / "base_wan_5b_pos_rollout.yml"


def _real_hyperparameters(keys):
    """The deployed ``HyperParameters``, extracted by AST and bound to a fake key store."""
    import types

    tree = ast.parse(_PYCONFIG.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HyperParameters")
    namespace = {"_config": types.SimpleNamespace(keys=dict(keys))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(_PYCONFIG), "exec"), namespace)  # noqa: S102
    return namespace["HyperParameters"]()


def _yaml_keys():
    import yaml

    return yaml.safe_load(_ROLLOUT_YAML.read_text(encoding="utf-8"))


def _direct_reads(function_name):
    """Every ``config.X`` this function reads directly — the reads that must be DECLARED keys."""
    source = _MODULE_PATH.read_text(encoding="utf-8")
    node = _function_node(source, function_name)
    return sorted(
        {
            child.attr
            for child in ast.walk(node)
            if isinstance(child, ast.Attribute) and getattr(child.value, "id", "") == "config"
        }
    )


def test_the_real_HyperParameters_raises_on_an_undeclared_key_which_is_why_this_test_exists():
    """The property every fake config in this suite lacks, stated once so the rest can rely on it."""
    config = _real_hyperparameters(_yaml_keys())
    assert float(config.flow_shift) == 5.0
    with pytest.raises(ValueError, match="Requested key num_train_timesteps"):
        _ = config.num_train_timesteps
    # ...and the three-argument form does NOT save a caller here (issue #11), which is the whole trap.
    with pytest.raises(ValueError, match="Requested key num_train_timesteps"):
        getattr(config, "num_train_timesteps", 1000)


def test_every_config_key_the_LOADER_reads_is_declared_by_the_rollout_yaml():
    """C3 BLOCKER: ``load_device_backend`` read ``config.num_train_timesteps``, which no rollout YAML
    declares, so the production loader died before the anchor or the grid check could run. Deployment
    never had that bug — it takes the value from ``scheduler.config``.

    This is a CONTRACT test over the loader's actual reads rather than a spot fix: a future key added
    to this function is checked against the YAML the moment it is added.
    """
    declared = set(_yaml_keys())
    for function in ("load_device_backend", "run_anchor_phase", "read_anchor_samples", "run_evaluation"):
        undeclared = [key for key in _direct_reads(function) if key not in declared]
        assert undeclared == [], f"{function} reads undeclared config keys {undeclared} (issue #11)"


def test_the_loader_takes_the_grid_parameters_from_the_SCHEDULER_as_deployment_does():
    """Deployment builds its sigmas from ``scheduler.config.sigma_min``/``sigma_max`` and its
    timesteps from ``scheduler.config.num_train_timesteps`` — the scheduler is the authority on the
    diffusion schedule, and the YAML never claimed to be."""
    deployed = _DEPLOYED.read_text(encoding="utf-8")
    assert "scheduler.config.sigma_min" in deployed and "scheduler.config.sigma_max" in deployed
    assert "rollout_timesteps_from_sigmas(sigmas, scheduler.config.num_train_timesteps)" in deployed

    loader = inspect.getsource(anchor.load_device_backend)
    assert "_create_scheduler()" in loader, "the loader must obtain the deployed scheduler"
    for expression in (
        "scheduler.config.num_train_timesteps",
        "scheduler.config.sigma_min",
        "scheduler.config.sigma_max",
    ):
        assert expression in loader, f"the loader must read {expression}"
    # Checked over the CODE, not the text: the comment above the fix names the defect it fixed, and a
    # textual scan would flag the explanation as the bug. `_direct_reads` walks the AST.
    assert "num_train_timesteps" not in _direct_reads("load_device_backend")
    # ...and the grid it builds is checked against the deployed one BEFORE the backend is returned,
    # so a scheduler that disagrees fails closed at load rather than at the first scored sample.
    assert "assert_deployed_grid(" in loader
