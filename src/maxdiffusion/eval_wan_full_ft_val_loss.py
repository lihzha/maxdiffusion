"""Full-finetune (FULL_FT_TI2V) validation-loss evaluator (exp_01 Part II, Query 8, T1).

Per-checkpoint one-step validation loss -- the EXACT training objective (velocity MSE,
frame-0 masked) over ALL held-out windows, with per-example ``(t, eps)`` held FIXED across
checkpoints so the loss curve is a pure model effect.

Cycle A contributed the PURE, CPU-testable core:

* :func:`per_example_rng` -- deterministic per-position ``(t_idx, eps)`` draw (D1/F1),
* :func:`plan_batches`   -- exactly-once position coverage with a masked padded tail (D2/F2),
* :func:`aggregate`      -- validity-masked mean / sample-stderr / count (D2/F2).

Cycle B ("val-loss-evaluator") completes the config-driven evaluator on top of those:

* :func:`load_all_records` -- drain the reader to EOF into a host-RAM cache, tag each record
  with its dataset POSITION, and assert the count == ``validation_expected_count`` BEFORE any
  state/restore work (D2/F2);
* :func:`assemble_batch` -- host batch assembly that draws ``(t, eps)`` per POSITION via
  :func:`per_example_rng` (D1);
* :func:`_eval_batch_per_example_loss` -- the jitted eval step: one plain transformer call
  (null context, activations dtype; NO actions/adapter/CFG), objective parity with training
  via the shared ``build_noisy_pinned_latents`` / ``masked_velocity_mse_per_example`` (D3);
* :func:`_evaluate_all_checkpoints` -- the sequential per-checkpoint restore loop (D4);
* :func:`write_outputs` / :func:`plot_rows` -- the 9-column JSON/CSV + guarded PNG (D5/F6);
* :func:`evaluate` / :func:`run` / :func:`main` -- config init, guards, and the CLI (incl.
  the ``plot-only`` local plot-regeneration mode).

Offline evaluation only: this module never trains, never mutates a checkpoint, and (T1)
deletes the pipeline's rollout-only modules right after state construction (plan F5).

Strengthened per the cycle-B Codex review (three findings): :func:`_loss_to_host` (F1 --
per-batch COLLECTIVE gather of the sharded ``[B]`` losses; v6e-8 is a two-host topology, so
a direct ``np.asarray`` on the jitted output would raise), :func:`_require_train_commit`
(F2 -- a FULL run refuses to write artifacts without the training run's SHA; smoke is
exempt with the rationale documented on the helper), and :func:`_build_sigma_grid` (F3 --
the four-argument sigma-grid wiring is a pinned, testable seam).
"""

from __future__ import annotations

import csv
import functools
import io
import json
import os
import tempfile
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P

import maxdiffusion.generate_wan_side_adapter as gen
from maxdiffusion import max_logging, pyconfig
from maxdiffusion.models.wan.side_adapter_wan import (
    build_noisy_pinned_latents,
    build_rollout_sigmas,
    masked_velocity_mse_per_example,
    _build_per_token_timestep,
    _dtype,
)
from maxdiffusion.trainers.wan_ti2v_full_ft_trainer import WanTI2VFullFTTrainer


def per_example_rng(
    seed: int,
    position: int,
    num_steps: int,
    example_shape: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    """Deterministic ``(t_idx, eps)`` for one validation example, keyed by POSITION.

    Counter-based (stateless) draw: ``k = fold_in(key(seed), position)``, split into
    ``(k_t, k_eps)``; ``t_idx = randint(k_t, (), 0, num_steps, int32)`` and
    ``eps = normal(k_eps, example_shape, float32)`` generated at the UNBATCHED
    example shape (the caller stacks per-example ``eps`` into a batch).

    DISTRIBUTION PARITY, not draw replay. Training samples a uniform step index
    ``randint([0, num_steps))`` and independent fresh Gaussian noise per example
    (see ``_sample_step_indices`` / ``_build_noise`` in
    ``wan_ti2v_side_adapter_trainer``). Held-out examples have no training draws to
    replay, so this reproduces the same MARGINAL law -- independent uniform ``t`` and
    standard-normal ``eps`` per example -- deliberately NOT training's stateful
    per-step key sequence. Being a pure function of ``(seed, position)``, the draw is
    identical across checkpoints, batches, batch order, and hosts, which is what
    makes the per-checkpoint loss comparison a pure model effect.

    ``position`` is the 0-based DATASET POSITION (the reader coordinate); the stored
    per-record ``ordinal`` field is NEVER used as this index. Raises ``ValueError``
    if ``num_steps <= 0``. Returns ``t_idx`` as an int32 scalar and ``eps`` as a
    float32 array of shape ``example_shape``.
    """
    if num_steps <= 0:
        raise ValueError(f"per_example_rng: num_steps must be positive, got {num_steps}")
    k = jax.random.fold_in(jax.random.key(seed), position)
    k_t, k_eps = jax.random.split(k)
    t_idx = jax.random.randint(k_t, (), 0, num_steps, dtype=jnp.int32)
    eps = jax.random.normal(k_eps, example_shape, jnp.float32)
    return t_idx, eps


def plan_batches(total: int, batch: int) -> list[tuple[list[int], list[bool]]]:
    """Plan fixed, position-ordered evaluation batches covering ``0..total-1`` once.

    Returns one ``(positions, validity)`` pair per batch in position order, with no
    shuffling and no repetition of the dataset. Every batch has exactly ``batch``
    entries; the final partial batch is padded by REPEATING THE LAST REAL POSITION
    with ``validity == False`` for the pads. Repeating a real record is safe because
    the transformer forward and the per-example loss are batch-independent -- the pad
    only fills the fixed-shape batch and is dropped in aggregation.

    Invariants (checked by the tests): the union of positions whose validity is True
    is exactly ``range(total)``, each real position appears once and only in a valid
    slot, and padded slots alias the last real position (never a fresh index).

    Raises ``ValueError`` if ``total`` or ``batch`` is non-positive.
    """
    if total <= 0:
        raise ValueError(f"plan_batches: total must be positive, got {total}")
    if batch <= 0:
        raise ValueError(f"plan_batches: batch must be positive, got {batch}")
    plan: list[tuple[list[int], list[bool]]] = []
    for start in range(0, total, batch):
        positions = list(range(start, min(start + batch, total)))
        validity = [True] * len(positions)
        while len(positions) < batch:
            positions.append(positions[-1])  # repeat the last REAL position
            validity.append(False)
        plan.append((positions, validity))
    return plan


def _concat_1d(x, dtype) -> np.ndarray:
    """Normalize the aggregate() input into a flat 1-D numpy array of ``dtype``.

    Accepts the canonical contract -- a list/tuple of per-batch 1-D array-likes,
    concatenated in the given order -- and, for convenience, a single already-1-D
    array-like (treated as one batch). A flat Python list of scalars also flattens
    correctly (each scalar becomes a length-1 segment).
    """
    if isinstance(x, (list, tuple)):
        parts = [np.asarray(p, dtype=dtype).reshape(-1) for p in x]
        if not parts:
            return np.zeros((0,), dtype=dtype)
        return np.concatenate(parts)
    return np.asarray(x, dtype=dtype).reshape(-1)


def aggregate(
    per_example_losses,
    validity,
    expected_count: int,
) -> dict:
    """Validity-masked mean / sample-stderr / count over per-example losses.

    Input contract (chosen, per plan D2): ``per_example_losses`` is a list of
    per-batch 1-D loss arrays and ``validity`` is the matching list of per-batch 1-D
    boolean masks -- exactly what the evaluator collects from its per-batch loop
    (each jitted eval step returns a ``[batch]`` loss vector; each batch's validity
    comes from :func:`plan_batches`). A single already-concatenated 1-D array is also
    accepted for either argument. Both are flattened in the given order and must have
    equal total length.

    ONLY mask-valid entries are counted: padded duplicates (``validity == False``)
    are excluded from the count, the mean, AND the stderr. ``stderr`` is the sample
    standard deviation (``ddof=1``) divided by ``sqrt(n)``. Raises ``ValueError`` if
    the flattened losses and validity differ in length, or if the number of valid
    entries ``n`` does not equal ``expected_count`` (catches both a short read and a
    duplicated/over-long source). Reductions run in float64 for host-side stability.

    Returns ``{"mean_loss": float, "stderr": float, "n": int}``.
    """
    losses = _concat_1d(per_example_losses, np.float64)
    valid = _concat_1d(validity, bool)
    if losses.shape != valid.shape:
        raise ValueError(f"aggregate: losses length {losses.shape[0]} != validity length {valid.shape[0]}")
    valid_losses = losses[valid]
    n = int(valid_losses.shape[0])
    if n != expected_count:
        raise ValueError(f"aggregate: counted {n} valid entries != expected_count {expected_count}")
    mean_loss = float(np.mean(valid_losses))
    if n >= 2:
        stderr = float(np.std(valid_losses, ddof=1) / np.sqrt(n))
    else:
        stderr = float("nan")
    return {"mean_loss": mean_loss, "stderr": stderr, "n": n}


# --------------------------------------------------------------------------------------
# Cycle B: record cache, host batch assembly, jitted eval step, checkpoint loop, outputs.
# --------------------------------------------------------------------------------------


def load_all_records(config) -> list[dict]:
    """Drain the val reader to EOF into a host-RAM cache; assert the count BEFORE any restore.

    Consumes :func:`generate_wan_side_adapter._iter_parsed_records` (the tested, file-ordered,
    no-shuffle reader seam) ONCE, decoding each record's ``z_i0`` / ``z_video`` into fp16 host
    arrays (actions are unused in a full-FT run) and tagging it with its 0-based enumeration
    POSITION -- the reader coordinate that :func:`per_example_rng` keys on. The stored per-record
    ``ordinal`` rides along for provenance only and is NEVER used as an index.

    The reader is drained fully (so an over-long / duplicated source is caught, not only a short
    read), then ``len(records)`` is asserted equal to ``config.validation_expected_count`` with a
    ``ValueError`` naming BOTH numbers. This runs before the evaluator builds any state or restores
    any checkpoint (a wrong-count dataset must fail fast, not after a costly model load). A
    non-positive ``validation_expected_count`` is refused up front.
    """
    expected = int(getattr(config, "validation_expected_count", 0))
    if expected <= 0:
        raise ValueError(f"validation_expected_count must be a positive integer; got {expected}")
    c = int(config.latent_channels)
    f = int(config.latent_frames)
    h = int(config.latent_height)
    w = int(config.latent_width)

    records: list[dict] = []
    for position, raw in enumerate(gen._iter_parsed_records(config)):
        records.append(
            {
                "position": position,
                "ordinal": int(raw["ordinal"]),  # provenance only -- never an index
                "z_i0": np.frombuffer(raw["z_i0"], dtype=np.float16).reshape(c, 1, h, w),
                "z_video": np.frombuffer(raw["z_video"], dtype=np.float16).reshape(c, f, h, w),
            }
        )

    n = len(records)
    if n != expected:
        raise ValueError(
            f"validation dataset yielded {n} records but validation_expected_count is {expected}; "
            "the reader was drained to EOF, so this is a real size mismatch (short read, or a "
            "duplicated / over-long source)."
        )
    return records


def assemble_batch(records, positions, seed, num_steps, sigmas):
    """Stack one fixed-shape eval batch, drawing ``(t, eps)`` per dataset POSITION (D1).

    ``records`` is the position-indexed host cache from :func:`load_all_records`; ``positions``
    is one batch's position list from :func:`plan_batches` (padded slots repeat the last real
    position). For each position ``p`` the ``(t_idx, eps)`` come from ``per_example_rng(seed, p,
    num_steps, example_shape)`` -- keyed on the POSITION, never on ``records[p]["ordinal"]`` --
    which is exactly what makes the per-checkpoint loss a pure model effect (identical draws
    across checkpoints, batch sizes, batch order, and hosts). ``eps`` is drawn at the UNBATCHED
    example shape and stacked. ``sigma_t = sigmas[t_idx]`` mirrors training's
    ``sigmas[_sample_step_indices(...)]``. Returns numpy ``(z_i0[B], z_video[B], eps[B],
    sigma_t[B])``; the eval step casts/pins on device.
    """
    example_shape = tuple(int(d) for d in records[positions[0]]["z_video"].shape)
    z_i0 = np.stack([records[p]["z_i0"] for p in positions])
    z_video = np.stack([records[p]["z_video"] for p in positions])
    eps_list = []
    t_indices = []
    for p in positions:
        t_idx, eps = per_example_rng(seed, p, num_steps, example_shape)
        eps_list.append(np.asarray(eps))
        t_indices.append(int(t_idx))
    eps = np.stack(eps_list)
    sigma_t = np.asarray(sigmas)[np.asarray(t_indices)]
    return z_i0, z_video, eps, sigma_t


def _build_sigma_grid(config, scheduler) -> np.ndarray:
    """Training's EXACT sigma grid for the evaluator (D1/D3 parity; wiring pinned by test, F3).

    Mirrors ``wan_ti2v_full_ft_trainer._denoising_loss`` line-for-line:
    ``build_rollout_sigmas(side_adapter_sampling_steps, flow_shift, scheduler sigma_min,
    scheduler sigma_max)`` -- each of the FOUR arguments read from the same source the
    trainer reads it from, so a drift in any one of them breaks the wiring test rather
    than silently skewing the loss curve. Returns the host-side ``N+1`` float grid that
    :func:`assemble_batch` indexes with the per-position ``t_idx``.
    """
    return np.asarray(
        build_rollout_sigmas(
            int(config.side_adapter_sampling_steps),
            config.flow_shift,
            scheduler.config.sigma_min,
            scheduler.config.sigma_max,
        )
    )


def _eval_batch_per_example_loss(state, z_i0, z_video, eps, sigma_t, *, config, num_train_timesteps):
    """Jitted eval step: per-example one-step velocity MSE for one batch (D3).

    Objective parity with ``wan_ti2v_full_ft_trainer._denoising_loss`` by SHARED code: the noisy
    frame-0-pinned latents come from ``build_noisy_pinned_latents`` and the loss from
    ``masked_velocity_mse_per_example`` (the per-example twin of the training scalar). Exactly ONE
    plain ``transformer(...)`` forward -- null text context broadcast over the batch in the
    activations dtype, ``deterministic=True`` -- with NO actions, NO adapter, and NO CFG. The
    interpolation / pin math is float32; only the transformer's ``hidden_states`` is cast to
    ``weights_dtype`` (matching the trainer). Returns a ``[B]`` float32 per-example loss vector;
    the host-side validity mask (padded tail) is applied later in :func:`aggregate`.
    """
    weights_dtype = _dtype(config.weights_dtype)
    activations_dtype = _dtype(config.activations_dtype)
    transformer = nnx.merge(state.graphdef, state.params, state.rest_of_state)

    z_i0_f32 = z_i0.astype(jnp.float32)
    z_video_f32 = z_video.astype(jnp.float32)
    eps_f32 = eps.astype(jnp.float32)
    b, _, f_lat, h_lat, w_lat = z_video_f32.shape

    z_t_f32 = build_noisy_pinned_latents(z_video_f32, z_i0_f32, eps_f32, sigma_t)
    step_t = sigma_t.astype(jnp.float32) * jnp.asarray(num_train_timesteps, dtype=jnp.float32)
    timestep_2d = _build_per_token_timestep(step_t, f_lat, h_lat, w_lat, n_hist=1)
    null_context = jnp.broadcast_to(
        state.null_context.astype(activations_dtype),
        (b, state.null_context.shape[1], state.null_context.shape[2]),
    )

    v_pred = transformer(
        hidden_states=z_t_f32.astype(weights_dtype),
        timestep=timestep_2d,
        encoder_hidden_states=null_context,
        deterministic=True,
    )
    v_target = eps_f32 - z_video_f32
    return masked_velocity_mse_per_example(v_pred, v_target)


# The 9-column output schema (plan D5/F6), identical in JSON and CSV, in this exact order.
_VAL_LOSS_COLUMNS: Sequence[str] = (
    "checkpoint_step",
    "mean_loss",
    "stderr",
    "n",
    "validation_seed",
    "dataset_path",
    "checkpoint_path",
    "train_commit",
    "eval_commit",
)


def _make_row(step, agg, seed, dataset_path, checkpoint_path, train_commit, eval_commit) -> dict:
    """One output row in the EXACT 9-column order (plan D5/F6). Dict order is the schema."""
    return {
        "checkpoint_step": int(step),
        "mean_loss": float(agg["mean_loss"]),
        "stderr": float(agg["stderr"]),
        "n": int(agg["n"]),
        "validation_seed": int(seed),
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "train_commit": str(train_commit),
        "eval_commit": str(eval_commit),
    }


def _loss_to_host(loss) -> np.ndarray:
    """Bring one batch's ``[B]`` per-example losses to EVERY host as numpy (F1).

    v6e-8 is a TWO-host topology (4 chips per host): the jitted eval step's output is
    sharded over the GLOBAL mesh, so each process can address only its own shards and a
    direct ``np.asarray`` on the device array would raise (or truncate) on multi-host.
    Not-fully-addressable values are therefore gathered COLLECTIVELY on all processes --
    ``multihost_utils.process_allgather(..., tiled=True)`` (tiled: concatenate the shards
    back into the one ``[B]`` vector, not stack per process); being a collective, every
    host must reach the call in the same order, which the deterministic batch loop
    guarantees -- and then copied to host. Fully-addressable values (single-process CPU
    tests, single-host TPUs) skip the collective and convert directly.
    """
    if getattr(loss, "is_fully_addressable", True):
        return np.asarray(jax.device_get(loss))
    return np.asarray(jax.device_get(multihost_utils.process_allgather(loss, tiled=True)))


def _evaluate_all_checkpoints(
    config,
    records,
    steps,
    ckpt_dir,
    state,
    eval_step_fn,
    *,
    seed,
    num_steps,
    sigmas,
    batch,
    expected_count,
    dataset_path,
    checkpoint_path,
    train_commit,
    eval_commit,
    smoke_limit=None,
):
    """Sequential per-checkpoint restore + full-pass loop over ONE built state (D4).

    For each requested step: restore params/opt_state/step in place via
    ``generate_wan_side_adapter._restore_checkpoint_state(..., cohort_mode=True,
    requested_step=step)`` (the kwarg overrides the immutable ``config.checkpoint_step``); assert
    BOTH the returned step and ``state.step`` equal the request; run every planned batch through
    ``eval_step_fn``, gathering each batch's ``[B]`` losses to every host through
    :func:`_loss_to_host` (F1 -- the gather also materializes the batch, so the final batch is
    complete before the next restore, D4); and aggregate the per-batch losses with the plan's
    per-batch validity so the padded tail is excluded.

    SMOKE (plan F5): a positive ``smoke_limit`` truncates to the first ``smoke_limit`` batches and
    the FIRST checkpoint only, and the ``n == expected`` assertion is SKIPPED -- the aggregate uses
    the actual valid count of the smoke subset (everything else is identical). Returns the 9-column
    rows, one per evaluated checkpoint.
    """
    plan = plan_batches(len(records), batch)
    smoke = smoke_limit is not None and int(smoke_limit) > 0
    if smoke:
        plan = plan[: int(smoke_limit)]
        steps = list(steps)[:1]

    rows = []
    for step in steps:
        state, restored_step = gen._restore_checkpoint_state(
            config, state, ckpt_dir, cohort_mode=True, requested_step=step
        )
        if restored_step != step or int(state.step) != step:
            raise ValueError(
                f"checkpoint restore mismatch: requested step {step} but restored step "
                f"{restored_step} / state.step {int(state.step)}"
            )

        batch_losses = []
        batch_validity = []
        for positions, validity in plan:
            z_i0, z_video, eps, sigma_t = assemble_batch(records, positions, seed, num_steps, sigmas)
            loss = eval_step_fn(state, z_i0, z_video, eps, sigma_t)
            # F1: gather THIS batch's [B] losses to every host before any numpy use -- the
            # jitted output is sharded over the global mesh on the two-host v6e-8. The gather
            # also materializes the batch, so the final one is complete before the next
            # restore (D4 loop discipline).
            batch_losses.append(_loss_to_host(loss))
            batch_validity.append(np.asarray(validity, dtype=bool))

        # Full run enforces n == expected (14636); smoke skips it by counting its own subset.
        agg_expected = int(sum(int(v.sum()) for v in batch_validity)) if smoke else expected_count
        agg = aggregate(batch_losses, batch_validity, agg_expected)
        rows.append(_make_row(step, agg, seed, dataset_path, checkpoint_path, train_commit, eval_commit))
    return rows


def _parse_checkpoint_steps(text) -> list[int]:
    """Parse ``validation_checkpoint_steps`` into a non-empty list of positive ints.

    Empty / whitespace-only is rejected (the evaluator needs at least one checkpoint), and each
    token must be a positive integer (0 would be the pretrained baseline, which is not part of the
    T1 loss curve; negatives are meaningless).
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("validation_checkpoint_steps must be a non-empty comma-separated list of positive ints")
    steps: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"validation_checkpoint_steps must be comma-separated integers; got {token!r}") from exc
        if value <= 0:
            raise ValueError(f"validation_checkpoint_steps entries must be positive; got {value}")
        steps.append(value)
    if not steps:
        raise ValueError("validation_checkpoint_steps parsed to empty; provide at least one positive step")
    return steps


def _free_rollout_modules(pipeline) -> None:
    """Delete the pipeline's rollout-only modules (VAE / vae_cache / text encoder / tokenizer).

    T1 computes losses in latent space only -- it never decodes video or embeds text -- so these
    heavy modules are freed right after state construction to relieve v6e-8 HBM (plan F5). Missing
    attributes are tolerated (the shared builder already drops the text encoder / tokenizer).
    """
    for attr in ("vae", "vae_cache", "text_encoder", "tokenizer"):
        if hasattr(pipeline, attr):
            delattr(pipeline, attr)


def _build_and_free_state(config):
    """Build the full-FT validation state via the shared builder, then free rollout-only modules."""
    trainer, pipeline, mesh, state, state_shardings = gen._build_full_ft_validation_state(config)
    _free_rollout_modules(pipeline)  # T1: no rollouts -- release the VAE / text modules (plan F5)
    return trainer, pipeline, mesh, state, state_shardings


def _resolve_output_dir(config, smoke=False) -> str:
    """Resolve the output root (plan D5). Smoke writes to an isolated sibling (plan F5)."""
    explicit = (getattr(config, "validation_loss_output_dir", "") or "").strip()
    if explicit:
        base = explicit.rstrip("/")
        return f"{base}_smoke" if smoke else base
    leaf = "validation_loss_smoke" if smoke else "validation_loss"
    return f"{config.output_dir.rstrip('/')}/{config.run_name}/{leaf}"


def _git_sha() -> str:
    try:
        import subprocess

        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _resolve_commits(config) -> tuple[str, str]:
    """(train_commit, eval_commit) for the output rows (plan F6).

    ``train_commit`` = the run's recorded training SHA, from the ``TRAIN_COMMIT`` env (or a
    ``train_commit`` config field), else ``"unknown"``. ``eval_commit`` = the eval code SHA, from
    the ``COMMIT`` env, else ``git rev-parse HEAD``, else ``"unknown"``.
    """
    train = (os.environ.get("TRAIN_COMMIT") or getattr(config, "train_commit", "") or "").strip() or "unknown"
    eval_commit = (os.environ.get("COMMIT") or "").strip() or _git_sha() or "unknown"
    return train, eval_commit


def _require_train_commit(train_commit: str, *, smoke: bool) -> None:
    """F2: refuse to run a FULL evaluation whose artifacts would lack the training SHA.

    A full run's rows are the T1 acceptance artifact; ``train_commit`` empty/"unknown" would
    make the loss curve unattributable to a training commit, so it is rejected up front --
    BEFORE the dataset drain, the model load, and hours of evaluation, and therefore before
    any artifact write. (The wrapper independently enforces ``TRAIN_COMMIT`` before python;
    this guard covers direct module invocations.)

    SMOKE relaxation, documented per the review's change-order: smoke outputs go to the
    isolated ``validation_loss_smoke/`` directory and serve ONLY as the storage-light
    fit-probe gate (plan F5) -- they are never T1 acceptance evidence, and the fit probe must
    stay runnable in ad-hoc debugging where the training-SHA bookkeeping may not be threaded
    through yet. Everything else about a smoke run is identical, including stamping whatever
    ``train_commit`` WAS resolved into the smoke rows.
    """
    if smoke:
        return
    if not train_commit or train_commit == "unknown":
        raise ValueError(
            "train_commit is required for a FULL validation-loss run: set the TRAIN_COMMIT env var "
            "(or the train_commit config field) to the training run's recorded commit SHA; refusing "
            f"to write provenance-less artifacts (got {train_commit!r})."
        )


def _smoke_limit():
    """The positive ``SMOKE_LIMIT`` env (batches-per-checkpoint cap) or ``None``."""
    raw = os.environ.get("SMOKE_LIMIT", "").strip()
    if not raw:
        return None
    value = int(raw)
    return value if value > 0 else None


def _write_text(path, text) -> None:
    parent = os.path.dirname(path)
    if parent:
        tf.io.gfile.makedirs(parent)
    with tf.io.gfile.GFile(path, "w") as fh:
        fh.write(text)


def _rows_to_csv(rows) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(_VAL_LOSS_COLUMNS), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in _VAL_LOSS_COLUMNS})
    return buf.getvalue()


def plot_rows(rows, path) -> None:
    """Pure single-series plot of held-out one-step loss vs checkpoint step. Writes a LOCAL PNG.

    Requires matplotlib (Agg backend). Callers guard the import (see :func:`write_outputs`); the
    ``plot-only`` CLI calls this directly for local regeneration when the worker lacked matplotlib.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(r["checkpoint_step"]) for r in rows]
    means = [float(r["mean_loss"]) for r in rows]
    errs = [float(r["stderr"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.errorbar(steps, means, yerr=errs, marker="o", capsize=3)
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("held-out one-step velocity MSE (mean ± stderr)")
    ax.set_title("Full-FT validation loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _emit_plot(rows, out_path) -> None:
    """Render ``plot_rows`` to a local temp PNG and copy it to ``out_path`` (GCS-safe)."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        local = fh.name
    try:
        plot_rows(rows, local)
        gen._copy_local_to_output(local, out_path)
    finally:
        if os.path.exists(local):
            os.remove(local)


def write_outputs(rows, out_dir) -> None:
    """Write ``val_loss.json`` + ``val_loss.csv`` (EXACT 9-column schema) and, if matplotlib is
    available, ``val_loss_plot.png`` (plan D5/F6).

    The plot is import-guarded: JSON and CSV are always written; if matplotlib is missing the PNG
    is skipped (not fatal) and a note points at the ``plot-only`` regeneration mode -- T1 is only
    accepted once the PNG exists at ``val_loss_plot.png`` (a mandatory recorded post-step then).
    """
    out_dir = out_dir.rstrip("/")
    ordered = [{key: row[key] for key in _VAL_LOSS_COLUMNS} for row in rows]
    _write_text(f"{out_dir}/val_loss.json", json.dumps(ordered, indent=2) + "\n")
    _write_text(f"{out_dir}/val_loss.csv", _rows_to_csv(rows))

    try:
        import matplotlib  # noqa: F401
    except Exception:
        max_logging.log(
            "[wan_full_ft_val] matplotlib unavailable; skipped val_loss_plot.png. Regenerate locally: "
            f"python -m maxdiffusion.eval_wan_full_ft_val_loss plot-only {out_dir}/val_loss.json "
            f"{out_dir}/val_loss_plot.png"
        )
        return
    _emit_plot(rows, f"{out_dir}/val_loss_plot.png")


def _plot_only(json_path, png_path) -> None:
    with tf.io.gfile.GFile(json_path, "r") as fh:
        rows = json.load(fh)
    _emit_plot(rows, png_path)


def _assert_full_ft(config) -> None:
    """Guard: this evaluator is FULL_FT_TI2V-only and enforces the probe invariants the trainer does."""
    model_type = getattr(config, "model_type", "")
    if model_type != "FULL_FT_TI2V":
        raise ValueError(f"eval_wan_full_ft_val_loss requires model_type == 'FULL_FT_TI2V'; got {model_type!r}")
    # Same guide-scale (1.0) + fresh-noise asserts the trainer applies (CFG bypassed; plan §2.1/F1).
    WanTI2VFullFTTrainer._validate_probe_config(config)


def evaluate(config):
    """Run the full-validation loss sweep for one config; write the outputs. Returns (rows, out_dir).

    Order matters, cheapest guard first: parse the checkpoint steps, REQUIRE the training SHA
    in full mode (F2 -- fail before the drain, the model load, and any artifact write), DRAIN +
    count-assert the dataset (:func:`load_all_records`) BEFORE building any state, THEN build
    the state and evaluate. A wrong-count dataset therefore fails before a costly model load.
    """
    steps = _parse_checkpoint_steps(getattr(config, "validation_checkpoint_steps", ""))
    smoke_limit = _smoke_limit()
    train_commit, eval_commit = _resolve_commits(config)
    _require_train_commit(train_commit, smoke=smoke_limit is not None)  # F2: full mode needs the SHA
    records = load_all_records(config)  # drain + count assert BEFORE any state/restore work

    trainer, pipeline, mesh, state, state_shardings = _build_and_free_state(config)
    scheduler, _ = trainer._create_scheduler()
    num_train_timesteps = int(scheduler.config.num_train_timesteps)
    num_steps = int(config.side_adapter_sampling_steps)
    sigmas = _build_sigma_grid(config, scheduler)  # F3: the pinned four-argument wiring seam
    batch = int(config.global_batch_size_to_train_on)
    seed = int(getattr(config, "validation_seed", getattr(config, "seed", 0)))
    ckpt_dir = config.checkpoint_dir or os.path.join(config.output_dir, config.run_name, "checkpoints")

    data_sharding = NamedSharding(mesh, P(*config.data_sharding))
    p_eval = jax.jit(
        functools.partial(_eval_batch_per_example_loss, config=config, num_train_timesteps=num_train_timesteps),
        in_shardings=(state_shardings, data_sharding, data_sharding, data_sharding, data_sharding),
        out_shardings=data_sharding,
    )

    def eval_step_fn(current_state, z_i0, z_video, eps, sigma_t):
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            z_i0 = jax.device_put(jnp.asarray(z_i0), data_sharding)
            z_video = jax.device_put(jnp.asarray(z_video), data_sharding)
            eps = jax.device_put(jnp.asarray(eps), data_sharding)
            sigma_t = jax.device_put(jnp.asarray(sigma_t), data_sharding)
            return p_eval(current_state, z_i0, z_video, eps, sigma_t)

    rows = _evaluate_all_checkpoints(
        config,
        records,
        steps,
        ckpt_dir,
        state,
        eval_step_fn,
        seed=seed,
        num_steps=num_steps,
        sigmas=sigmas,
        batch=batch,
        expected_count=int(config.validation_expected_count),
        dataset_path=config.eval_data_dir,
        checkpoint_path=ckpt_dir,
        train_commit=train_commit,
        eval_commit=eval_commit,
        smoke_limit=smoke_limit,
    )

    out_dir = _resolve_output_dir(config, smoke=smoke_limit is not None)
    if jax.process_index() == 0:
        write_outputs(rows, out_dir)
        for row in rows:
            max_logging.log(
                f"[wan_full_ft_val] step={row['checkpoint_step']} mean_loss={row['mean_loss']:.6f} "
                f"stderr={row['stderr']:.6f} n={row['n']}"
            )
        max_logging.log(f"[wan_full_ft_val] wrote outputs under {out_dir}")
    return rows, out_dir


def run(argv: Sequence[str]):
    pyconfig.initialize(argv)
    config = pyconfig.config
    _assert_full_ft(config)
    return evaluate(config)


def main(argv: Sequence[str]) -> None:
    # ``plot-only <val_loss.json> <out.png>`` regenerates the PNG locally (no pyconfig / TPU),
    # for the mandatory post-step when the worker lacked matplotlib (plan F6).
    if len(argv) >= 2 and argv[1] == "plot-only":
        if len(argv) != 4:
            raise ValueError("plot-only usage: eval_wan_full_ft_val_loss.py plot-only <val_loss.json> <out.png>")
        _plot_only(argv[2], argv[3])
        return
    run(argv)


if __name__ == "__main__":
    app.run(main)
