"""exp_02 overfit100 — standalone SAMPLING-STEPS H1/H2 DISCRIMINATOR, deliberately verdict-isolated.

One question: at checkpoint 2500, does a longer sampler close the gap? The segment-final pass ran
the plan's 25-step sampler and reported ``mean m_corr = 0.8133`` with 0/100 windows at 0.95, so
before concluding anything about memorization we want to know whether the sampler, rather than the
weights, is the limiter.

**This script cannot produce an admissible artifact, by construction.** It never imports or calls
the verdict machinery — no role validation, no aggregation artifact, no staging, no publication
marker, no pass-role key — and it writes a plainly-named diagnostic JSON to its own
``validation_probe_sampling/`` directory, which is not a role directory and can never be mistaken
for one. A test asserts every one of those names is absent from this file.

What it DOES reuse, so the numbers are comparable with the eval path rather than merely similar:

* the same restore (``_restore_overfit100_validation_state``: pinned-snapshot check, manifest
  binding, dataset preflight, context table + null embedding, Orbax restore),
* the same rollout body (``_rollout_overfit100_sample`` through the shared jit seam),
* the same per-window rng (``window_fold_key(0, episode_id, window_start)``) and the same
  correct-mode context, so an arm's 25-step row is directly comparable to the segment-final row for
  that window — which is why the 25-step arm is INCLUDED as an in-probe validity control.

The only thing that varies across arms is ``side_adapter_sampling_steps``, supplied per arm through
a read-only config VIEW (:func:`arm_config`). The base config is never mutated: each arm closes over
its own view, so each gets its own jitted rollout with its own sigma schedule, and no jitted
function can silently reuse another arm's step count.

Why no ffmpeg: the probe scores against the VAE decode of the stored ``z_video`` only (latent MSE,
pixel MSE, SSIM). It never pulls or decodes a source MP4, so — like the one-step loss arm — its
launcher deliberately carries no ffmpeg-ensure block.

**What it discriminates.** H1 (the sampler is the limiter) predicts the longer arms lift SSIM
materially; H2 (the weights are the limiter) predicts they do not. The paired per-window deltas are
the read-out, because the arms share a cohort, a context and a per-window rng.

**Acceptance criteria, fixed in advance.**

* *Validity control.* The 25-step arm must reproduce the landed segment-final rows for these same
  30 windows, whose reference is **mean SSIM 0.8100125855, median 0.8059329625**. That row-level
  join is done OFFLINE against the published aggregation -- this module never reads a verdict
  artifact, which is what keeps it isolated -- so every row here carries its window identity and its
  per-arm SSIM. A 25-step arm that does not match those numbers invalidates the probe, not the
  checkpoint.
* *Read-out.* ``paired_deltas`` for ``50-25`` and ``100-25``. Deltas that leave the cohort far below
  the 0.95 canonical bar mean the sampler is not the limiter and the segment-final reading stands.

**The design is approval-pinned** (:func:`assert_approved_design`): checkpoint 2500, arms exactly
{25, 50, 100}, baseline 25. Only ``probe_num_windows`` is flexible -- 15 is the pre-declared
fallback for bad spot weather. Anything else is a startup failure, because the approved experiment
is what was approved.
"""

from __future__ import annotations

import ast
import statistics
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from absl import app
from flax.linen import partitioning as nn_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P

import maxdiffusion.generate_wan_side_adapter as gen
from maxdiffusion import max_logging, pyconfig

PROBE_SCHEMA = "overfit100_sampling_steps_probe_v1"
PROBE_OUTPUT_DIR = "validation_probe_sampling"
PROBE_CONTEXT_MODE = "correct"
PROBE_SEED = 0  # the segment-final pass's seed 0, so an arm's 25-step row is comparable to it

# The APPROVED experiment. Pinned rather than parameterised: what was approved is a specific
# comparison at a specific checkpoint, and a probe that quietly ran something else would answer a
# question nobody sanctioned.
APPROVED_SAMPLING_ARMS = (25, 50, 100)
APPROVED_CHECKPOINT_STEP = 2500
BASELINE_SAMPLING_STEPS = 25


class _ArmConfig:
    """A read-only view of the run config with one overridden ``side_adapter_sampling_steps``.

    A view rather than a mutation: mutating the shared config between jitted calls is exactly how an
    arm would silently inherit another arm's step count (the value is baked into the traced
    function). Each arm closes over its own view, so each arm gets its own jit entry.
    """

    def __init__(self, base, sampling_steps: int):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "side_adapter_sampling_steps", int(sampling_steps))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_base"), name)

    def __setattr__(self, name, value):  # pragma: no cover - the view is read-only by contract
        raise AttributeError(f"{type(self).__name__} is read-only; build another view instead of setting {name!r}")


def arm_config(base, sampling_steps: int):
    """The per-arm config view (see :class:`_ArmConfig`)."""
    return _ArmConfig(base, sampling_steps)


def parse_probe_steps(raw) -> tuple[int, ...]:
    """``probe_sampling_steps_list`` -> an ordered, duplicate-free tuple of positive step counts.

    Accepts the yml list, a ``"[25,50,100]"`` literal, or a bare comma-separated string, so the knob
    survives both pyconfig override styles.
    """
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError("probe_sampling_steps_list must not be empty")
        items = ast.literal_eval(text) if text.startswith(("[", "(")) else text.split(",")
    else:
        items = list(raw or ())
    steps: list[int] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, str)):
            raise ValueError(f"probe_sampling_steps_list entries must be positive integers; got {item!r}")
        try:
            value = int(str(item).strip())
        except ValueError as exc:
            raise ValueError(f"probe_sampling_steps_list entries must be positive integers; got {item!r}") from exc
        if value <= 0:
            raise ValueError(f"probe_sampling_steps_list entries must be positive integers; got {item!r}")
        if value not in steps:
            steps.append(value)
    if not steps:
        raise ValueError("probe_sampling_steps_list parsed to empty; give at least one step count")
    return tuple(steps)


def probe_cohort(manifest_path: str, *, seed: int, num_windows: int) -> list[dict]:
    """A deterministic seeded subsample of the manifest's canonical windows.

    Selection is a sha256-keyed ordering of the window NAMES (the same platform- and
    version-independent scheme the shuffled-context derangement uses -- no RNG stream to drift), so
    the same seed and manifest always pick the same cohort. The result is returned in
    ``episode_index`` order so the artifact reads predictably, and the chosen names are recorded in
    the output.
    """
    count = int(num_windows)
    if count <= 0:
        raise ValueError(f"probe_num_windows must be positive; got {num_windows!r}")
    episodes = gen.manifest_episode_windows(manifest_path)
    canonical = gen.select_eval_windows(("canonical", ()), episodes)
    if count > len(canonical):
        raise ValueError(
            f"probe_num_windows={count} exceeds the {len(canonical)} canonical windows the manifest defines"
        )
    chosen = set(gen._seeded_order([window["name"] for window in canonical], int(seed))[:count])
    return [window for window in canonical if window["name"] in chosen]


def probe_output_path(config, checkpoint_step: int) -> str:
    """``<output_dir>/<run_name>/validation_probe_sampling/probe_steps_ckpt<step>.json``.

    CANONICAL ONLY -- there is deliberately no output-root override. An unrestricted one could drop
    a diagnostic inside a ``step_<n>_<role>`` verdict directory, where nothing downstream could undo
    the confusion, and the launcher never needed it. The operator-controlled components that remain
    (``output_dir``, ``run_name``) are still checked: any path component that looks like a role
    directory is refused, so neither a typo nor a hostile value can steer the probe into the
    evidence tree.
    """
    root = f"{str(config.output_dir).rstrip('/')}/{config.run_name}/{PROBE_OUTPUT_DIR}"
    path = f"{root.rstrip('/')}/probe_steps_ckpt{int(checkpoint_step)}.json"
    offenders = [part for part in path.split("/") if part.startswith("step_")]
    if offenders:
        raise ValueError(
            f"refusing the probe output path {path!r}: component(s) {offenders} name a verdict role directory. A "
            f"diagnostic must never be written where the verdict's evidence lives -- fix output_dir/run_name."
        )
    parts = path.split("/")
    if len(parts) < 2 or parts[-2] != PROBE_OUTPUT_DIR:
        raise ValueError(f"refusing the probe output path {path!r}: it is not directly under {PROBE_OUTPUT_DIR}/")
    return path


def assert_approved_design(*, steps, checkpoint_step: int) -> None:
    """Refuse anything but the approved experiment (arms {25, 50, 100} at checkpoint 2500)."""
    arms = tuple(sorted({int(step) for step in steps}))
    if arms != tuple(sorted(APPROVED_SAMPLING_ARMS)):
        raise ValueError(
            f"the sampling probe is approval-pinned: arms must be exactly {list(APPROVED_SAMPLING_ARMS)}, got "
            f"{list(arms)}. Running a different comparison would answer a question that was not approved -- get the "
            f"new design approved and change APPROVED_SAMPLING_ARMS deliberately."
        )
    if int(checkpoint_step) != APPROVED_CHECKPOINT_STEP:
        raise ValueError(
            f"the sampling probe is approval-pinned to checkpoint {APPROVED_CHECKPOINT_STEP}, but got "
            f"{int(checkpoint_step)}. The landed segment-final reference this probe is validated against belongs to "
            f"that checkpoint; probing another one needs its own approval and its own reference."
        )


def probe_summary(rows: Sequence[dict], *, baseline: int) -> dict:
    """Per-arm SSIM stats plus PAIRED per-window deltas against the in-probe baseline arm.

    Paired, not marginal: the arms share a cohort and a per-window rng, so the per-window difference
    is the quantity with the least noise in it. A window missing from any arm is refused rather than
    silently dropped -- an unbalanced comparison is what would make a null result unreadable.
    """
    arms = sorted({int(row["sampling_steps"]) for row in rows})
    if int(baseline) not in arms:
        raise ValueError(f"baseline arm {baseline} is not among the probe's arms {arms}")
    by_arm: dict[int, dict[str, dict]] = {arm: {} for arm in arms}
    for row in rows:
        by_arm[int(row["sampling_steps"])][str(row["name"])] = row
    names = sorted({str(row["name"]) for row in rows})
    for arm in arms:
        missing = [name for name in names if name not in by_arm[arm]]
        if missing:
            raise ValueError(
                f"probe arm {arm} is missing {len(missing)} window(s), e.g. {missing[:3]}; every arm must cover the "
                f"same cohort or the paired comparison is not meaningful"
            )

    per_arm = {}
    for arm in arms:
        ssims = [float(by_arm[arm][name]["ssim"]) for name in names]
        per_arm[str(arm)] = {
            "n": len(ssims),
            "mean_ssim": statistics.fmean(ssims),
            "median_ssim": statistics.median(ssims),
            "mean_latent_mse": statistics.fmean([float(by_arm[arm][name]["latent_mse"]) for name in names]),
            "mean_pixel_mse": statistics.fmean([float(by_arm[arm][name]["pixel_mse"]) for name in names]),
        }
    paired = {}
    for arm in arms:
        if arm == int(baseline):
            continue
        deltas = [float(by_arm[arm][name]["ssim"]) - float(by_arm[int(baseline)][name]["ssim"]) for name in names]
        paired[f"{arm}-{int(baseline)}"] = {
            "n": len(deltas),
            "mean_delta_ssim": statistics.fmean(deltas),
            "median_delta_ssim": statistics.median(deltas),
            "n_improved": sum(1 for value in deltas if value > 0.0),
        }
    return {"baseline_sampling_steps": int(baseline), "per_arm": per_arm, "paired_deltas": paired}


def probe_artifact(config, *, checkpoint_step: int, cohort, steps, rows) -> dict:
    """The diagnostic JSON. Carries NO verdict fields -- it is not admissible evidence."""
    return {
        "schema": PROBE_SCHEMA,
        "kind": "diagnostic",
        "note": (
            "Sampling-steps diagnostic. NOT an evaluation artifact: no role, no cohort denominator, no "
            "aggregation schema. It cannot enter the success statistic."
        ),
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_dir": str(getattr(config, "checkpoint_dir", "")),
        "run_name": str(getattr(config, "run_name", "")),
        "commit": gen._eval_code_commit(),
        "eval_data_dir": str(getattr(config, "eval_data_dir", "")),
        "model_manifest_path": str(getattr(config, "model_manifest_path", "")),
        "context_mode": PROBE_CONTEXT_MODE,
        "rollout_seed": PROBE_SEED,
        "cohort_seed": int(getattr(config, "seed", 0)),
        "sampling_steps_arms": [int(step) for step in steps],
        "cohort": [str(window["name"]) for window in cohort],
        "summary": probe_summary(rows, baseline=BASELINE_SAMPLING_STEPS),
        "rows": list(rows),
    }


def write_probe_artifact(path: str, payload: dict) -> None:
    """Immutable write, reusing the eval path's compare-or-refuse writer."""
    gen._write_json_immutable(path, payload)


def _log_summary(summary: dict) -> None:
    max_logging.log("[overfit100_probe] sampling-steps probe summary")
    max_logging.log(f"[overfit100_probe]   {'steps':>6} {'n':>4} {'mean_ssim':>10} {'median_ssim':>12}")
    for arm, stats in sorted(summary["per_arm"].items(), key=lambda item: int(item[0])):
        max_logging.log(
            f"[overfit100_probe]   {arm:>6} {stats['n']:>4} {stats['mean_ssim']:>10.4f} {stats['median_ssim']:>12.4f}"
        )
    for pair, stats in sorted(summary["paired_deltas"].items()):
        max_logging.log(
            f"[overfit100_probe]   paired {pair}: mean_delta_ssim={stats['mean_delta_ssim']:+.4f} "
            f"median={stats['median_delta_ssim']:+.4f} improved={stats['n_improved']}/{stats['n']}"
        )


def run_probe(config) -> dict:
    """Roll out one cohort under several samplers and write the diagnostic JSON."""
    gen.assert_ssim_available()  # SSIM is the whole point of the probe
    steps = parse_probe_steps(getattr(config, "probe_sampling_steps_list", ()))
    manifest_path = str(getattr(config, "model_manifest_path", ""))
    cohort = probe_cohort(
        manifest_path, seed=int(getattr(config, "seed", 0)), num_windows=int(config.probe_num_windows)
    )
    # Cheap refusal first: the arms and the REQUESTED step, before the ~5B load.
    assert_approved_design(steps=steps, checkpoint_step=int(getattr(config, "checkpoint_step", -1)))

    (
        trainer,
        pipeline,
        mesh,
        state,
        state_shardings,
        null_context,
        checkpoint_step,
    ) = gen._restore_overfit100_validation_state(config)
    # Re-checked against the RESTORED checkpoint: the requested step is not proof of what Orbax
    # handed back.
    assert_approved_design(steps=steps, checkpoint_step=checkpoint_step)
    scheduler, _ = trainer._create_scheduler()
    replicated = NamedSharding(mesh, P())
    data_shardings = {"z_i0": replicated, "z_video": replicated, "context": replicated}
    # One jitted rollout PER ARM, each closing over its own config view -- so an arm cannot inherit
    # another arm's step count through a retrace that never happened.
    rollouts = {
        step: gen._overfit100_rollout_fn(
            state_shardings, data_shardings, replicated, scheduler, arm_config(config, step)
        )
        for step in steps
    }
    samples = gen.read_overfit100_samples(config, cohort)
    if jax.process_index() == 0:
        max_logging.log(
            f"[overfit100_probe] step={checkpoint_step} windows={len(samples)} arms={list(steps)} "
            f"seed={PROBE_SEED} mode={PROBE_CONTEXT_MODE}"
        )

    rows: list[dict] = []
    for sample in samples:
        batch_base = {
            "z_i0": jnp.asarray(sample.z_i0[None]),
            "z_video": jnp.asarray(sample.z_video[None]),
        }
        gt_video = pipeline._decode_latents_to_video(
            pipeline._denormalize_latents(batch_base["z_video"].astype(jnp.float32))
        )
        gt0 = np.asarray(gt_video[0], dtype=np.float32)
        context = gen.overfit100_context_for_mode(
            state.context_table, null_context, PROBE_CONTEXT_MODE, sample.episode_index, None
        )
        for step in steps:
            batch = jax.tree.map(lambda x: jax.device_put(x, replicated), {**batch_base, "context": context})
            # The eval path's per-window key, so the 25-step arm reproduces the segment-final rollout.
            rng = gen.window_fold_key(PROBE_SEED, sample.episode_id, sample.window_start)
            with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
                z_pred, metrics = rollouts[step](state, batch, rng)
                z_pred.block_until_ready()
            pred_video = pipeline._decode_latents_to_video(pipeline._denormalize_latents(z_pred.astype(jnp.float32)))
            if jax.process_index() != 0:
                continue
            pred0 = np.asarray(pred_video[0], dtype=np.float32)
            rows.append(
                {
                    "name": sample.name,
                    "episode_id": int(sample.episode_id),
                    "episode_index": int(sample.episode_index),
                    "window_start": int(sample.window_start),
                    "sampling_steps": int(step),
                    "ssim": gen._frame_ssim(pred0, gt0),
                    "latent_mse": float(jax.device_get(metrics["latent_mse"])),
                    "pixel_mse": float(np.mean((pred0 - gt0) ** 2)),
                }
            )
            max_logging.log(
                f"[overfit100_probe] {sample.name} steps={step} ssim={rows[-1]['ssim']:.4f} "
                f"latent_mse={rows[-1]['latent_mse']:.6f}"
            )

    if jax.process_index() != 0:
        return {}
    payload = probe_artifact(config, checkpoint_step=checkpoint_step, cohort=cohort, steps=steps, rows=rows)
    path = probe_output_path(config, checkpoint_step)
    write_probe_artifact(path, payload)
    _log_summary(payload["summary"])
    max_logging.log(f"[overfit100_probe] wrote {path} ({len(rows)} rows over {len(steps)} arms)")
    return payload


def run(argv: Sequence[str]):
    pyconfig.initialize(argv)
    config = pyconfig.config
    if str(getattr(config, "model_type", "")) != gen.OVERFIT100_MODEL_TYPE:
        raise ValueError(
            f"the sampling probe requires model_type == {gen.OVERFIT100_MODEL_TYPE!r}; got {config.model_type!r}"
        )
    return run_probe(config)


def main(argv: Sequence[str]) -> None:
    run(argv)


if __name__ == "__main__":
    app.run(main)
