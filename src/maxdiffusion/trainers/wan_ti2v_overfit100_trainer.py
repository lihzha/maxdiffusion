"""WAN TI2V overfit100 trainer -- per-episode TEXT-conditioned full finetune (exp_02).

This is the exp_01 full-FT trainer with exactly ONE experimental variable added: the
single reused null text embedding is replaced by a **per-episode context table**, so each
training window is conditioned on its own DROID language instruction (plan
``docs/worklogs_yixun/exp_02_overfit100_claude/plan_overfit100.md`` D7/D8). Everything
else -- the fully-trainable 5B transformer, the one-step flow-matching objective, the
frame-0 pin, fresh noise, the CFG bypass, FSDP shardings, the Orbax Composite
``params``/``opt_state``/``step`` checkpoint layout -- is inherited or byte-parity code.

Why a whole module instead of subclass overrides (Codex plan review v2, **G1**): the
full-FT ``_denoising_loss`` / ``_train_step`` / ``_eval_step`` are *module-level*
functions that ``WanTI2VFullFTTrainer.start_training`` jit-binds directly, so subclass
methods can never replace them. This module therefore owns:

* ``Overfit100TrainState`` -- ``FullFTTrainState`` with ``context_table``
  ``[num_text_slots, L, text_dim]`` REPLACING ``null_context``;
* module-level ``_denoising_loss`` -- byte-parity with the full-FT loss through the shared
  ``side_adapter_wan`` helpers (``build_noisy_pinned_latents`` for the frame-0 pin,
  ``eps - z_video`` target, ``masked_velocity_mse``, fresh per-example noise, no actions /
  adapter / CFG). The ONLY delta is ``context = state.context_table[episode_index]``
  (batched gather) instead of broadcasting one null embedding;
* module-level ``_train_step`` / ``_eval_step`` binding that loss;
* ``WanTI2VOverfit100Trainer(WanTI2VFullFTTrainer)`` overriding the genuine seams --
  ``_load_dataset`` (schema-v2 parse; the parent's parse fn is nested, so the override
  owns it), ``_data_shardings`` (adds ``episode_index``), ``_build_optimizer`` (absolute
  ``warmup_steps``, plan D9), ``_shard_state`` (context table replicated),
  ``_build_checkpoint_manager`` (``max_to_keep=None``, H2) -- plus a REWRITTEN
  ``start_training`` that builds the text table, constructs ``Overfit100TrainState`` and
  jit-binds **this module's** step functions.

Checkpoint contract (Codex plan review v3, **H2**): the loop saves when
``(step + 1) in config.checkpoint_steps`` (an explicit list; ``checkpoint_every`` is still
honored when > 0 for compatibility) and the manager keeps EVERYTHING
(``max_to_keep=None``), so segment-final checkpoints survive resumes. The context table is
NOT part of ``params`` or ``opt_state``, hence not part of the Composite save/restore
targets: it is rebuilt deterministically from ``episodes.json`` on every start, keeping the
~30 GB/checkpoint budget to params + both Adam moments.

``train_wan.py`` dispatches ``OVERFIT100_TI2V`` here.
"""

from __future__ import annotations

import ast
import base64
import datetime
import functools
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import orbax.checkpoint as ocp
import tensorflow as tf
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state
from jax.sharding import NamedSharding, PartitionSpec as P

from maxdiffusion import max_logging, max_utils
from maxdiffusion.input_pipeline.input_pipeline_interface import make_data_iterator
from maxdiffusion.models.wan.side_adapter_wan import (
    adapter_param_count,
    build_noisy_pinned_latents,
    build_rollout_sigmas,
    masked_velocity_mse,
    _build_per_token_timestep,
    _dtype,
)
from maxdiffusion.schedulers import FlaxFlowMatchScheduler
from maxdiffusion.train_utils import load_next_batch
from maxdiffusion.trainers.wan_ti2v_full_ft_trainer import (
    WanTI2VFullFTTrainer,
    _adam_moment_trees,
    _dtype_summary,
    _format_dtype_summary,
    _full_ft_state_shardings,
    _leaf_bytes,
)
from maxdiffusion.trainers.wan_ti2v_side_adapter_trainer import (
    _apply_actual_sharding_for_tpu,
    _build_noise,
    _sample_step_indices,
    _to_target_if_cpu,
)

# Cycle-B sidecar contract (``data_preprocessing/build_overfit100_dataset.py``). Mirrored
# as literals rather than imported so the TPU training path never pulls in the builder's
# ffmpeg/gsutil-oriented dependencies; the tests cross-check them against the builder's
# own constants.
SUCCESS_MARKER = "_SUCCESS"
EPISODES_SIDECAR = "episodes.json"
SUMMARY_SIDECAR = "summary.json"


class Overfit100TrainState(train_state.TrainState):
    """Full-FT train state whose text conditioning is a PER-EPISODE table.

    Identical to ``FullFTTrainState`` except that the single ``null_context`` embedding is
    replaced by ``context_table`` ``[num_text_slots, L, text_dim]``: row ``i`` is the T5
    embedding of manifest ``episode_index`` ``i``'s instruction. The table is rebuilt
    deterministically at every start and is deliberately NOT checkpointed.
    """

    graphdef: nnx.GraphDef
    rest_of_state: nnx.State
    context_table: jax.Array


def _denoising_loss(
    params,
    state: Overfit100TrainState,
    data: dict,
    rng: jax.Array,
    config,
    scheduler: FlaxFlowMatchScheduler,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """One-step flow-matching velocity MSE with per-episode text context.

    Byte-parity with ``wan_ti2v_full_ft_trainer._denoising_loss`` (same shared helpers,
    same frame-0 pin, same ``eps - z_video`` target, same fresh noise, one plain
    transformer forward, no actions / adapter / CFG); the only difference is that
    ``encoder_hidden_states`` is a batched GATHER of ``state.context_table`` at the
    example's ``episode_index`` instead of one broadcast null embedding. With every table
    row equal to the null embedding the two losses agree bitwise (tested).
    """
    noise_rng, step_rng, dropout_rng = jax.random.split(rng, 3)
    weights_dtype = _dtype(config.weights_dtype)
    activations_dtype = _dtype(config.activations_dtype)
    bsz = config.global_batch_size_to_train_on

    transformer = nnx.merge(state.graphdef, params, state.rest_of_state)

    z_i0_f32 = data["z_i0"][:bsz].astype(jnp.float32)
    z_video_f32 = data["z_video"][:bsz].astype(jnp.float32)

    b, _, f_lat, h_lat, w_lat = z_video_f32.shape
    num_steps = int(config.side_adapter_sampling_steps)
    sigmas = build_rollout_sigmas(
        num_steps,
        config.flow_shift,
        scheduler.config.sigma_min,
        scheduler.config.sigma_max,
    )
    t_idx = _sample_step_indices(step_rng, b, num_steps, sigmas, config)
    sigma_t = sigmas[t_idx]
    step_t = sigma_t * jnp.asarray(scheduler.config.num_train_timesteps, dtype=jnp.float32)
    timestep_2d = _build_per_token_timestep(step_t, f_lat, h_lat, w_lat, n_hist=1)
    # The experiment variable: per-example text context, gathered by manifest episode
    # index (NOT the raw DROID episode id, which is not a table position).
    episode_index = data["episode_index"][:bsz].astype(jnp.int32)
    context = state.context_table[episode_index].astype(activations_dtype)

    eps = _build_noise(noise_rng, z_video_f32.shape, jnp.float32, config)
    z_t_f32 = build_noisy_pinned_latents(z_video_f32, z_i0_f32, eps, sigma_t)

    v_pred = transformer(
        hidden_states=z_t_f32.astype(weights_dtype),
        timestep=timestep_2d,
        encoder_hidden_states=context,
        deterministic=False,
        rngs=nnx.Rngs(dropout=dropout_rng),
    )

    v_target = eps - z_video_f32
    loss = masked_velocity_mse(v_pred, v_target, b)
    aux = {
        "velocity_mse": loss,
        "sigma_mean": jnp.mean(sigma_t.astype(jnp.float32)),
        "timestep_mean": jnp.mean(step_t.astype(jnp.float32)),
        "v_pred_l2": jnp.linalg.norm(v_pred.astype(jnp.float32)),
        "v_target_l2": jnp.linalg.norm(v_target.astype(jnp.float32)),
        "z_noisy_std": jnp.std(z_t_f32),
        "z_target_std": jnp.std(z_video_f32),
        "z_init_anchor_mse": jnp.mean((z_t_f32[:, :, :1] - z_i0_f32[:, :, :1]) ** 2),
    }
    return loss, aux


def _train_step(state: Overfit100TrainState, data: dict, rng: jax.Array, scheduler, config):
    rng, loss_rng = jax.random.split(rng)

    def loss_fn(params):
        return _denoising_loss(params, state, data, loss_rng, config, scheduler)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    grad_norm = jaxopt.tree_util.tree_l2_norm(grads)
    max_abs_grad = jax.tree_util.tree_reduce(
        lambda m, arr: jnp.maximum(m, jnp.max(jnp.abs(arr))), grads, initializer=-1.0
    )
    state = state.apply_gradients(grads=grads)
    metrics = {
        "scalar": {
            "learning/loss": loss,
            "learning/velocity_mse": aux["velocity_mse"],
            "learning/grad_norm": grad_norm,
            "learning/max_abs_grad": max_abs_grad,
            "learning/sigma_mean": aux["sigma_mean"],
            "learning/timestep_mean": aux["timestep_mean"],
            "learning/v_pred_l2": aux["v_pred_l2"],
            "learning/v_target_l2": aux["v_target_l2"],
            "learning/z_noisy_std": aux["z_noisy_std"],
            "learning/z_target_std": aux["z_target_std"],
            "learning/z_init_anchor_mse": aux["z_init_anchor_mse"],
        },
        "scalars": {},
    }
    return state, metrics, rng


def _eval_step(state: Overfit100TrainState, data: dict, rng: jax.Array, scheduler, config):
    losses = jnp.zeros((config.global_batch_size_to_train_on,), dtype=jnp.float32)
    loss, aux = _denoising_loss(state.params, state, data, rng, config, scheduler)
    losses = losses.at[:].set(loss)
    metrics = {
        "scalar": {
            "learning/eval_loss": losses,
            "learning/eval_sigma_mean": aux["sigma_mean"],
            "learning/eval_v_pred_l2": aux["v_pred_l2"],
            "learning/eval_v_target_l2": aux["v_target_l2"],
            "learning/eval_z_noisy_std": aux["z_noisy_std"],
            "learning/eval_z_target_std": aux["z_target_std"],
        },
        "scalars": {},
    }
    return metrics, rng


# --------------------------------------------------------------------------------------
# Pure helpers: dataset readiness, the text-table sources, shardings, and the checkpoint
# schedule. All CPU-testable with tmp dirs / fake trees -- no mesh, weights, or GCS.
# --------------------------------------------------------------------------------------


def _join(data_dir: str, name: str) -> str:
    """Join a local path or ``gs://`` prefix with a sidecar file name."""
    return f"{str(data_dir).rstrip('/')}/{name}"


def _read_json_strict(path: str, what: str):
    """Read one JSON sidecar, or raise a ValueError that names the object and the cause.

    Cycle-C review judgment 8: a read/permission/parse failure on a provenance sidecar is a
    LOUD failure, never a reason to fall back to a weaker source. Everything that reads
    ``_SUCCESS`` / ``summary.json`` / ``episodes.json`` goes through here.
    """
    if not tf.io.gfile.exists(path):
        raise ValueError(f"overfit100 {what}: {path} does not exist")
    try:
        with tf.io.gfile.GFile(path, "r") as handle:
            payload = json.load(handle)
    except ValueError as exc:  # json.JSONDecodeError is a ValueError
        raise ValueError(f"overfit100 {what}: {path} is not valid JSON ({exc})") from exc
    except Exception as exc:  # noqa: BLE001 -- surface the transport error, do not mask it
        raise ValueError(f"overfit100 {what}: {path} could not be read ({type(exc).__name__}: {exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"overfit100 {what}: {path} is not a JSON object")
    return payload


# The provenance fields cycle B always writes into `_SUCCESS` (build_overfit100_dataset.py's
# promote stage). `records` is the only OPTIONAL one -- it is the sole sanctioned fallback to
# `summary.json` (judgment 8).
_MARKER_REQUIRED_KEYS = ("build_id", "build_commit", "shards", "summary_sha256", "manifest_sha256")


def read_success_marker(data_dir: str) -> dict:
    """The validated ``_SUCCESS`` marker, or a loud failure.

    ``_SUCCESS`` is written LAST, after the whole set has been promoted, so its absence means
    the set is partial/in-flight and must not be read (plan D6: "every reader must require
    ``_SUCCESS``"). Structure is validated here so downstream code can trust the fields.
    """
    path = _join(data_dir, SUCCESS_MARKER)
    if not tf.io.gfile.exists(path):
        raise ValueError(
            f"overfit100 dataset {data_dir} has no {SUCCESS_MARKER} marker: the build either did not finish or "
            f"the directory is wrong. The marker is written LAST (after promotion), so a missing marker means "
            f"the set is partial -- refusing to read it."
        )
    marker = _read_json_strict(path, f"{SUCCESS_MARKER} marker in {data_dir}")
    missing = [key for key in _MARKER_REQUIRED_KEYS if key not in marker]
    if missing:
        raise ValueError(
            f"overfit100 {SUCCESS_MARKER} marker in {data_dir} is structurally invalid: missing {missing}. "
            f"A cycle-B marker always carries {list(_MARKER_REQUIRED_KEYS)} (+ optional 'records')."
        )
    return marker


def _set_name(data_dir: str) -> str:
    return str(data_dir).rstrip("/").rsplit("/", 1)[-1]


def _sha256_and_md5(path: str, chunk_bytes: int = 8 * 2**20) -> tuple[str, str, int]:
    """Stream one object once; return (sha256 hex, md5 base64, byte length)."""
    sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
    with tf.io.gfile.GFile(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), base64.b64encode(md5.digest()).decode("ascii"), size


def verify_dataset_integrity(data_dir: str, expected_windows: int, *, verify_bytes: bool = True) -> dict:
    """Bind the bytes this run will train on to cycle B's published fingerprints (C2).

    Fail-closed chain, cheapest first:

    1. ``_SUCCESS`` exists, reads, parses, and carries its provenance fields;
    2. ``summary.json``'s RAW BYTES hash to ``_SUCCESS.summary_sha256`` -- this single check
       binds the entire summary (every shard fingerprint, every count) to the marker, so
       nothing downstream can be edited without detection;
    3. the canonical shard set on disk is EXACTLY the set the summary lists (no strays, none
       missing) -- a stray ``*.tfrecord`` would silently join the training stream, since the
       reader globs the directory;
    4. every shard's byte length matches (remote metadata only), and with ``verify_bytes`` its
       ``sha256`` and ``md5`` match the published fingerprints;
    5. the record counts agree with each other and with ``config.expected_windows``.

    On ``generation``: cycle B stats the STAGING object, then ``promote()`` COPIES it to the
    canonical prefix, minting a new generation. The recorded generation therefore belongs to an
    object that no longer exists at the canonical URI, so it is deliberately NOT checked; md5
    (content-derived) survives the copy and is checked instead. Verifying the sha256 of the
    bytes we are about to read is strictly stronger than trusting any remote metadata field,
    and needs no external binary (``tf.io.gfile`` only) -- so this works identically on a local
    path and on GCS.

    Returns a report dict for the startup log.
    """
    marker = read_success_marker(data_dir)

    summary_path = _join(data_dir, SUMMARY_SIDECAR)
    if not tf.io.gfile.exists(summary_path):
        raise ValueError(f"overfit100 dataset {data_dir}: {SUMMARY_SIDECAR} is missing but {SUCCESS_MARKER} exists")
    with tf.io.gfile.GFile(summary_path, "rb") as handle:
        summary_bytes = handle.read()
    observed = hashlib.sha256(summary_bytes).hexdigest()
    if observed != str(marker["summary_sha256"]):
        raise ValueError(
            f"overfit100 dataset {data_dir}: {SUMMARY_SIDECAR} does not match {SUCCESS_MARKER}.summary_sha256 "
            f"(recorded {marker['summary_sha256']}, observed {observed}) -- the published metadata was modified "
            f"after the build; refusing to train."
        )
    summary = _read_json_strict(summary_path, f"{SUMMARY_SIDECAR} in {data_dir}")

    name = _set_name(data_dir)
    sets = summary.get("sets") or {}
    if name not in sets:
        raise ValueError(
            f"overfit100 dataset {data_dir}: {SUMMARY_SIDECAR} describes sets {sorted(sets)}, not {name!r}. The "
            f"directory name IS the set key (cycle B writes the same summary into every set)."
        )
    entry = sets[name]
    shards = list(entry.get("shards") or [])
    if not shards:
        raise ValueError(f"overfit100 dataset {data_dir}: {SUMMARY_SIDECAR} lists no shards for set {name!r}")

    listed = [str(shard["name"]) for shard in shards]
    if len(set(listed)) != len(listed):
        raise ValueError(f"overfit100 dataset {data_dir}: {SUMMARY_SIDECAR} lists a duplicate shard name")
    on_disk = sorted(path.rsplit("/", 1)[-1] for path in _shard_uris(data_dir))
    if sorted(listed) != on_disk:
        extra = sorted(set(on_disk) - set(listed))
        absent = sorted(set(listed) - set(on_disk))
        raise ValueError(
            f"overfit100 dataset {data_dir}: canonical shard set does not match {SUMMARY_SIDECAR} "
            f"(unlisted on disk: {extra}; listed but absent: {absent}). The reader globs the directory, so a "
            f"stray shard would silently join the training stream."
        )

    total_records = 0
    for shard in shards:
        uri = _join(data_dir, str(shard["name"]))
        recorded_size = int(shard["size"])
        actual_size = int(tf.io.gfile.stat(uri).length)
        if actual_size != recorded_size:
            raise ValueError(
                f"overfit100 dataset {data_dir}: shard {shard['name']} size {actual_size} != recorded "
                f"{recorded_size} -- the published bytes changed; refusing to train."
            )
        if verify_bytes:
            sha256, md5_b64, streamed = _sha256_and_md5(uri)
            if streamed != recorded_size:
                raise ValueError(f"overfit100 dataset {data_dir}: shard {shard['name']} size changed while reading")
            if sha256 != str(shard["sha256"]):
                raise ValueError(
                    f"overfit100 dataset {data_dir}: shard {shard['name']} sha256 {sha256} != recorded "
                    f"{shard['sha256']} -- the shard's CONTENT was modified (record count and byte length can "
                    f"both be preserved by such an edit); refusing to train."
                )
            if shard.get("md5") and md5_b64 != str(shard["md5"]):
                raise ValueError(
                    f"overfit100 dataset {data_dir}: shard {shard['name']} md5 {md5_b64} != recorded {shard['md5']}"
                )
        total_records += int(shard["records"])

    written = int(entry.get("written", total_records))
    if written != total_records:
        raise ValueError(
            f"overfit100 dataset {data_dir}: {SUMMARY_SIDECAR} says written={written} but its shard records sum "
            f"to {total_records}"
        )
    records_source = SUMMARY_SIDECAR
    if "records" in marker:
        records_source = SUCCESS_MARKER
        if int(marker["records"]) != total_records:
            raise ValueError(
                f"overfit100 dataset {data_dir}: {SUCCESS_MARKER}.records={int(marker['records'])} disagrees with "
                f"the {total_records} records its shards contain"
            )
    if int(expected_windows) > 0 and total_records != int(expected_windows):
        raise ValueError(
            f"overfit100 dataset {data_dir} contains {total_records} windows but config.expected_windows="
            f"{int(expected_windows)}. Either the wrong set is configured (train100=1629 vs train10=167) or the "
            f"build drifted; refusing to train."
        )
    return {
        "data_dir": str(data_dir),
        "set_name": name,
        "records": total_records,
        "records_source": records_source,
        "shards": len(shards),
        "bytes_verified": bool(verify_bytes),
        "build_id": marker["build_id"],
        "build_commit": marker["build_commit"],
        "summary_sha256": observed,
    }


def _shard_uris(data_dir: str) -> list[str]:
    """The canonical ``*.tfrecord`` objects in ``data_dir`` (what the reader will glob)."""
    return sorted(tf.io.gfile.glob(_join(data_dir, "*.tfrecord")))


def assert_dataset_ready(data_dir: str, expected_windows: int) -> int:
    """The light re-check every iterator build does (integrity ran in the startup preflight).

    Requires a structurally VALID ``_SUCCESS`` (never a silent fallback for a marker that
    failed to read or parse -- judgment 8) and the expected window count.
    ``expected_windows <= 0`` checks the marker only.
    """
    marker = read_success_marker(data_dir)
    if "records" in marker:
        count = int(marker["records"])
    else:
        summary = _read_json_strict(_join(data_dir, SUMMARY_SIDECAR), f"{SUMMARY_SIDECAR} in {data_dir}")
        value = (summary.get("sets") or {}).get(_set_name(data_dir), {}).get("written")
        if value is None:
            raise ValueError(
                f"overfit100 dataset {data_dir}: {SUCCESS_MARKER} omits 'records' and {SUMMARY_SIDECAR} has no "
                f"'sets.{_set_name(data_dir)}.written' -- refusing to read an uncountable set."
            )
        count = int(value)
    if int(expected_windows) > 0 and count != int(expected_windows):
        raise ValueError(
            f"overfit100 dataset {data_dir} contains {count} windows but config.expected_windows="
            f"{int(expected_windows)}. Either the wrong set is configured (train100=1629 vs train10=167) or the "
            f"build drifted; refusing to train."
        )
    return count


def read_episode_mapping(data_dir: str) -> dict[int, dict]:
    """``episode_index -> {episode_id, used_text}`` from the set's ``episodes.json``.

    This mapping IS the context table's row assignment, so it is the thing that must agree
    between any two sets used in one run (C3).
    """
    payload = _read_json_strict(_join(data_dir, EPISODES_SIDECAR), f"{EPISODES_SIDECAR} in {data_dir}")
    episodes = payload["episodes"] if "episodes" in payload else []
    mapping: dict[int, dict] = {}
    for entry in episodes:
        index = int(entry["episode_index"])
        if index in mapping:
            raise ValueError(f"{EPISODES_SIDECAR} in {data_dir} repeats episode_index {index}")
        mapping[index] = {"episode_id": int(entry["episode_id"]), "used_text": str(entry["used_text"])}
    if not mapping:
        raise ValueError(f"{EPISODES_SIDECAR} in {data_dir} lists no episodes")
    return mapping


def read_episode_texts(data_dir: str, num_text_slots: int) -> list[str]:
    """The instruction of every episode in ``data_dir``, ordered by ``episode_index``.

    Enforces the table's structural contract: exactly ``num_text_slots`` entries whose
    ``episode_index`` values are index-CONTIGUOUS ``0..N-1`` (the loss gathers the table
    with that index, so a gap or a duplicate would silently mis-condition examples), and
    no empty instruction.
    """
    mapping = read_episode_mapping(data_dir)
    if len(mapping) != int(num_text_slots):
        raise ValueError(
            f"{EPISODES_SIDECAR} in {data_dir} lists {len(mapping)} episodes but config.num_text_slots="
            f"{int(num_text_slots)}; the context table must have exactly one row per built episode."
        )
    if set(mapping) != set(range(int(num_text_slots))):
        raise ValueError(
            f"{EPISODES_SIDECAR} in {data_dir} has non-contiguous episode_index values "
            f"{sorted(mapping)}; expected exactly 0..{int(num_text_slots) - 1}"
        )
    texts = [mapping[index]["used_text"] for index in range(int(num_text_slots))]
    for index, text in enumerate(texts):
        if not text.strip():
            raise ValueError(f"{EPISODES_SIDECAR} in {data_dir}: empty instruction at episode_index {index}")
    return texts


def assert_context_map_compatible(train_data_dir: str, eval_data_dir: str) -> int:
    """Every eval ``episode_index`` must mean the SAME episode+text as in the training set (C3).

    The context table is always built from ``train_data_dir``, so a different eval set whose
    ``episode_index`` values are individually valid but map to different episodes would be
    scored against the wrong instruction with no error anywhere. Checked as full triple
    equality on the shared indices, and every eval index must exist in the training mapping.
    """
    train_map = read_episode_mapping(train_data_dir)
    eval_map = read_episode_mapping(eval_data_dir)
    for index in sorted(eval_map):
        if index not in train_map:
            raise ValueError(
                f"overfit100 eval set {eval_data_dir} uses episode_index {index}, which the training set "
                f"{train_data_dir} does not define (its context table has rows {min(train_map)}..{max(train_map)}); "
                f"the gathered text would be wrong."
            )
        for field in ("episode_id", "used_text"):
            if eval_map[index][field] != train_map[index][field]:
                raise ValueError(
                    f"overfit100 context-map mismatch at episode_index {index}: eval set {eval_data_dir} has "
                    f"{field}={eval_map[index][field]!r} but the training set {train_data_dir} (which builds the "
                    f"context table) has {field}={train_map[index][field]!r}; refusing to evaluate against the "
                    f"wrong instruction."
                )
    return len(eval_map)


def encode_positive_prompts(pipeline, prompts: Sequence[str], max_sequence_length: int) -> jax.Array:
    """``encode_prompt``'s POSITIVE branch, replicated exactly -- and nothing else.

    ``WanPipeline.encode_prompt`` always also encodes a negative prompt (``""`` when none
    is given), which for a 100-prompt table would double the T5 work for embeddings this
    trainer never uses. So the positive half is called directly: ``_get_t5_prompt_embeds``
    (prompt_clean -> tokenize to ``max_sequence_length`` -> mask through UMT5 -> truncate
    to the true length -> zero-pad) followed by ``encode_prompt``'s own float32
    conversion. ``encoder_attention_mask`` stays absent, as in the null-context path.
    """
    embeds = pipeline._get_t5_prompt_embeds(
        prompt=list(prompts),
        num_videos_per_prompt=1,
        max_sequence_length=int(max_sequence_length),
    )
    if hasattr(embeds, "detach"):  # torch tensor, as the real pipeline returns
        embeds = embeds.detach().float().numpy()
    return jnp.asarray(np.asarray(embeds), dtype=jnp.float32)


def context_table_audit_line(table) -> str:
    """The startup memory-audit line for the text table (plan D8/F5: 400 MiB predeclared)."""
    global_bytes, addressable_bytes = _leaf_bytes(table)
    return (
        f"[wan_overfit100] context table: shape={tuple(int(d) for d in table.shape)} "
        f"dtype={jnp.dtype(table.dtype).name} global={global_bytes / 2**20:.1f} MiB "
        f"per-host physical={addressable_bytes / 2**20:.1f} MiB"
    )


def _overfit100_state_shardings(computed, state, replicated):
    """Full-FT's shard selection plus a REPLICATED context table.

    Keeps the computed FSDP shardings for ``params``/``opt_state`` (replicating the ~5B
    transformer would OOM) and pins the text table to a fully-replicated spec: every
    device gathers arbitrary rows of it, and at 400 MiB bf16 replication is affordable
    (it is not in ``params``, so the full-FT >100 MB-replicated audit does not apply).
    """
    return _full_ft_state_shardings(computed, state).replace(context_table=replicated)


def _schema_v2_feature_description() -> dict:
    """The schema-v2 fields this trainer reads. No ``actions`` -- schema v2 has none."""
    return {
        "z_i0": tf.io.FixedLenFeature([], tf.string),
        "z_video": tf.io.FixedLenFeature([], tf.string),
        "episode_index": tf.io.FixedLenFeature([], tf.int64),
    }


def _schema_v2_prepare_sample(config):
    """Build the ``prepare_sample`` the TFRecord reader maps over parsed schema-v2 examples.

    NOTE for cycle D: the evaluator additionally needs ``name``, ``episode_id``, and
    ``window_start`` for its per-window aggregation artifact (review judgment 7). They are on
    every schema-v2 record but are deliberately NOT parsed here -- the training objective does
    not use them, and every extra batch field costs a sharded device array per step. Cycle D
    must extend BOTH this function and ``_data_shardings``.
    """
    c = int(config.latent_channels)
    f = int(config.latent_frames)
    h = int(config.latent_height)
    w = int(config.latent_width)
    num_slots = int(getattr(config, "num_text_slots", 0) or 0)

    def prepare_sample(features):
        z_i0 = tf.reshape(tf.io.decode_raw(features["z_i0"], tf.float16), [c, 1, h, w])
        z_video = tf.reshape(tf.io.decode_raw(features["z_video"], tf.float16), [c, f, h, w])
        # int32: the gather index rides the batch as a [B] column.
        episode_index = tf.cast(features["episode_index"], tf.int32)
        if num_slots > 0:
            # C3: bound the index HERE, in the tf.data graph, where it is a cheap scalar
            # compare that fails loudly. A jnp gather CLAMPS out-of-range indices silently,
            # so index 99 against a 10-row table would train on row 9's instruction with no
            # error anywhere -- exactly the kind of wrong-text run this experiment cannot
            # afford to discover after the fact.
            with tf.control_dependencies(
                [
                    tf.debugging.assert_greater_equal(
                        episode_index,
                        tf.constant(0, tf.int32),
                        message="overfit100: negative episode_index in a schema-v2 record",
                    ),
                    tf.debugging.assert_less(
                        episode_index,
                        tf.constant(num_slots, tf.int32),
                        message=(
                            "overfit100: episode_index >= num_text_slots in a schema-v2 record; the context "
                            "table has no such row (wrong set/num_text_slots pairing?)"
                        ),
                    ),
                ]
            ):
                episode_index = tf.identity(episode_index)
        return {
            "z_i0": tf.cast(z_i0, tf.float32),
            "z_video": tf.cast(z_video, tf.float32),
            "episode_index": episode_index,
        }

    return prepare_sample


def parse_checkpoint_steps(raw) -> tuple[int, ...]:
    """Normalize ``config.checkpoint_steps`` to a sorted, unique tuple of positive ints.

    Accepts the yaml/pyconfig list (``[250,500,...]``, incl. a CLI override parsed by
    ``pyconfig.string_to_list``) and a comma-separated string, so the knob survives both
    override styles. Rejects non-integers and non-positive steps loudly: a silently
    dropped step means a missing gate checkpoint.
    """
    if raw is None:
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return ()
        items = ast.literal_eval(text) if text.startswith(("[", "(")) else text.split(",")
    else:
        items = list(raw)
    steps: list[int] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, str)):
            raise ValueError(f"checkpoint_steps entries must be positive integers; got {item!r}")
        try:
            step = int(str(item).strip())
        except ValueError as exc:
            raise ValueError(f"checkpoint_steps entries must be positive integers; got {item!r}") from exc
        if step <= 0:
            raise ValueError(f"checkpoint_steps entries must be positive integers; got {item!r}")
        steps.append(step)
    return tuple(sorted(set(steps)))


class CheckpointScheduler:
    """H2's explicit save schedule: an exact step LIST, not one periodic cadence.

    ``should_save(step)`` is the in-loop decision (``step`` is the 1-based ``step + 1``);
    ``final_step()`` is the end-of-run save, de-duplicated against what the loop already
    emitted so a listed ``max_train_steps`` is written exactly once.

    PRECEDENCE, not union (cycle-C review C4): a non-empty ``checkpoint_steps`` SUPPRESSES
    ``checkpoint_every`` entirely. H2's contract is an *exact retained* set with a predeclared
    storage budget (~30 GB per checkpoint, nothing ever GC'd), so an accidental nonzero cadence
    must not be able to mint unplanned checkpoints. The cadence remains the fallback when no
    list is configured; ``precedence_note()`` surfaces the suppression in the startup log.
    """

    def __init__(self, *, checkpoint_steps, checkpoint_every, max_train_steps, save_final):
        self.steps = parse_checkpoint_steps(checkpoint_steps)
        self.every = max(0, int(checkpoint_every or 0))
        self.max_train_steps = int(max_train_steps)
        self.save_final = bool(save_final)
        self.emitted: set[int] = set()

    def precedence_note(self) -> str | None:
        """One log line when both knobs are set non-trivially, naming the ignored cadence."""
        if self.steps and self.every > 0:
            return (
                f"[wan_overfit100] checkpoint_steps {list(self.steps)} GOVERNS; checkpoint_every="
                f"{self.every} is IGNORED (H2: the explicit list is the exact retained set)"
            )
        return None

    def should_save(self, step: int) -> bool:
        step = int(step)
        if step in self.emitted:
            return False
        due = step in self.steps if self.steps else (self.every > 0 and step % self.every == 0)
        if due:
            self.emitted.add(step)
            return True
        return False

    def final_step(self) -> int | None:
        if not self.save_final or self.max_train_steps in self.emitted:
            return None
        self.emitted.add(self.max_train_steps)
        return self.max_train_steps


def planned_checkpoint_steps(*, max_train_steps, checkpoint_steps, checkpoint_every, save_final=True, start_step=0):
    """The steps a run WILL checkpoint at -- logged at startup, asserted by the tests."""
    scheduler = CheckpointScheduler(
        checkpoint_steps=checkpoint_steps,
        checkpoint_every=checkpoint_every,
        max_train_steps=max_train_steps,
        save_final=save_final,
    )
    planned = [step for step in range(int(start_step) + 1, int(max_train_steps) + 1) if scheduler.should_save(step)]
    final = scheduler.final_step()
    if final is not None:
        planned.append(final)
    return planned


class WanTI2VOverfit100Trainer(WanTI2VFullFTTrainer):
    """exp_02 overfit100: full-FT backbone + per-episode text conditioning.

    Inherits the pipeline load, scheduler, probe-config asserts, sharding audit, Orbax
    save/restore layout, and the eval loop from ``WanTI2VFullFTTrainer``; overrides only
    genuine method seams and rewrites ``start_training`` (G1: the step functions are
    module-level and jit-bound there, so they cannot be overridden as methods).
    """

    @staticmethod
    def _validate_overfit100_config(config) -> None:
        """Enforce the exp_02 config invariants before any weights load (CPU-testable)."""
        slots = int(getattr(config, "num_text_slots", 0) or 0)
        if slots <= 0:
            raise ValueError(
                "overfit100 requires num_text_slots > 0 (one context-table row per built episode: "
                f"train100 -> 100, train10 -> 10); got {slots}"
            )
        batch = int(getattr(config, "text_encode_batch", 0) or 0)
        if batch <= 0:
            raise ValueError(f"overfit100 requires text_encode_batch > 0 (D8's bounded loop); got {batch}")
        windows = int(getattr(config, "expected_windows", 0) or 0)
        if windows <= 0:
            raise ValueError(
                "overfit100 requires expected_windows > 0 (the built window count the reader must match: "
                f"train100 -> 1629, train10 -> 167); got {windows}"
            )
        # Raises on a malformed list before any TPU time is spent.
        parse_checkpoint_steps(getattr(config, "checkpoint_steps", ()))

    @staticmethod
    def _validate_pinned_snapshot(config) -> str:
        """Bind training to the model revision the DATASET was built with (C1).

        The manifest pins ``{hf_repo, revision}``; the dataset's latents and this run's context
        table must come from that exact snapshot, or the run silently claims a recipe it did not
        execute. Loading a bare repo id (``Wan-AI/Wan2.2-TI2V-5B-Diffusers``) resolves to the
        MUTABLE hub default, so it is refused: the launcher must prefetch the pinned revision
        and pass the resolved local snapshot directory, whose HF layout
        (``…/snapshots/<revision>/``) carries the revision as a path component -- that is what
        is asserted here, for the pipeline path AND the transformer path the Wan loader uses.

        ``model_manifest_path``, when set, is the authority: the committed manifest's revision
        must agree with ``expected_model_revision``, so a launcher cannot claim a pin the
        manifest does not carry. Returns the verified revision.
        """
        expected = str(getattr(config, "expected_model_revision", "") or "").strip()
        manifest_path = str(getattr(config, "model_manifest_path", "") or "").strip()
        if manifest_path:
            manifest = _read_json_strict(manifest_path, "model manifest")
            pinned = str((manifest.get("vae_fingerprint") or {}).get("revision", "") or "").strip()
            if not pinned:
                raise ValueError(f"overfit100: model manifest {manifest_path} carries no vae_fingerprint.revision")
            if expected and expected != pinned:
                raise ValueError(
                    f"overfit100: expected_model_revision={expected} disagrees with the committed manifest "
                    f"{manifest_path}, which pins {pinned}. The manifest is the authority -- fix the launcher."
                )
            expected = pinned
        if not expected:
            raise ValueError(
                "overfit100 requires a pinned model revision: set expected_model_revision (or model_manifest_path, "
                "whose vae_fingerprint.revision supplies it). Training against the mutable hub default would let a "
                "future repo update change the transformer or the T5 that builds the context table while the run "
                "still claimed the reviewed recipe (C1)."
            )
        if not re.fullmatch(r"[0-9a-f]{40}", expected):
            raise ValueError(f"overfit100: expected_model_revision {expected!r} is not a 40-hex commit sha")

        for key in ("pretrained_model_name_or_path", "wan_transformer_pretrained_model_name_or_path"):
            path = str(getattr(config, key, "") or "")
            if not path:
                continue  # pyconfig copies the pipeline path into the transformer key when empty
            if expected not in path:
                raise ValueError(
                    f"overfit100: {key}={path!r} is not the pinned snapshot for revision {expected}. Pass the "
                    f"RESOLVED local snapshot directory (…/snapshots/{expected}/) -- a bare repo id resolves to the "
                    f"mutable hub default, which is not reproducible (C1)."
                )
        max_logging.log(f"[wan_overfit100] model revision pinned: {expected}")
        max_logging.log(f"[wan_overfit100]   snapshot: {config.pretrained_model_name_or_path}")
        return expected

    def _preflight_dataset(self) -> dict:
        """Verify the published dataset bytes BEFORE any model work (C2/C3).

        Runs as the first I/O of ``start_training``: cheap, local-metadata-plus-one-streaming-
        read, and it fails before the ~5B pipeline, the optimizer, and the state exist -- the
        old placement (inside ``_load_dataset``) discovered a bad dataset only after all of that
        had been built. When ``eval_data_dir`` differs from ``train_data_dir`` it is verified
        too, AND its episode mapping must agree with the training set's, because the context
        table is always built from the training set (C3).
        """
        config = self.config
        verify_bytes = bool(getattr(config, "dataset_verify_bytes", True))
        report = verify_dataset_integrity(
            config.train_data_dir, int(getattr(config, "expected_windows", 0) or 0), verify_bytes=verify_bytes
        )
        max_logging.log(
            f"[wan_overfit100] train set integrity OK: {report['data_dir']} set={report['set_name']} "
            f"records={report['records']} (from {report['records_source']}) shards={report['shards']} "
            f"bytes_verified={report['bytes_verified']} build={report['build_commit'][:12]} "
            f"summary_sha256={report['summary_sha256'][:12]} -- {SUCCESS_MARKER} verified"
        )
        eval_dir = getattr(config, "eval_data_dir", "") or ""
        if eval_dir and eval_dir != config.train_data_dir:
            eval_report = verify_dataset_integrity(eval_dir, 0, verify_bytes=verify_bytes)
            shared = assert_context_map_compatible(config.train_data_dir, eval_dir)
            max_logging.log(
                f"[wan_overfit100] eval set integrity OK: {eval_report['data_dir']} records={eval_report['records']} "
                f"context map compatible over {shared} episode(s)"
            )
        return report

    def _build_optimizer(self, num_steps: int):
        """AdamW + clipping exactly as exp_01, but with an ABSOLUTE warmup (plan D9).

        D9 locks warmup at 250 steps. The shared factory derives warmup from
        ``learning_rate_schedule_steps * warmup_steps_fraction``, which would rescale the
        warmup every time a resumable segment changes ``max_train_steps`` (S3 extends
        toward 10k). Feeding the absolute count as the schedule length with fraction 1.0
        reuses the SAME factory (linear 0 -> lr over ``warmup_steps``, then constant lr)
        while making the warmup segment-invariant. ``warmup_steps <= 0`` falls back to the
        inherited fraction behavior.
        """
        warmup_steps = int(getattr(self.config, "warmup_steps", 0) or 0)
        if warmup_steps <= 0:
            return super()._build_optimizer(num_steps)
        lr_schedule = max_utils.create_learning_rate_schedule(
            self.config.learning_rate,
            warmup_steps,
            1.0,
            num_steps,
        )
        tx = max_utils.create_optimizer(self.config, lr_schedule)
        return tx, lr_schedule

    def _load_dataset(self, mesh, is_training: bool, seed: int | None = None):
        """Schema-v2 TFRecord reader (plan D6): z_i0 / z_video / episode_index, no actions.

        The parent's nested ``prepare_sample`` hard-requires an ``actions`` feature that
        schema v2 does not have, so this override owns the parse -- and gates the read on
        the cycle-B ``_SUCCESS`` marker plus the expected window count BEFORE any reader is
        constructed.
        """
        config = self.config
        if config.dataset_type != "tfrecord" or not config.cache_latents_text_encoder_outputs:
            raise ValueError(
                "WanTI2VOverfit100Trainer requires dataset_type='tfrecord' and "
                "cache_latents_text_encoder_outputs=True."
            )
        data_dir = config.train_data_dir if is_training else config.eval_data_dir
        if not data_dir:
            raise ValueError(
                "train_data_dir/eval_data_dir must point to an exp_02 overfit100 schema-v2 TFRecord set "
                "(gs://v6_east1d/datasets/exp02_overfit100/{train100,train10})"
            )
        expected = int(getattr(config, "expected_windows", 0) or 0)
        if not (is_training or data_dir == config.train_data_dir):
            expected = 0  # a non-train eval dir has its own count; only require the marker
        count = assert_dataset_ready(data_dir, expected)
        max_logging.log(f"[wan_overfit100] dataset {data_dir}: {count} windows, {SUCCESS_MARKER} verified")

        return make_data_iterator(
            config,
            jax.process_index(),
            jax.process_count(),
            mesh,
            config.global_batch_size_to_load,
            feature_description=_schema_v2_feature_description(),
            prepare_sample_fn=_schema_v2_prepare_sample(config),
            is_training=is_training,
            seed=seed if seed is not None else config.seed,
        )

    def _data_shardings(self, mesh) -> dict:
        """The parent's latent shardings plus ``episode_index`` on the same batch axis."""
        sharding = NamedSharding(mesh, P(*self.config.data_sharding))
        return {"z_i0": sharding, "z_video": sharding, "episode_index": sharding}

    def _build_context_table(self, pipeline, mesh) -> jax.Array:
        """Encode the built episodes' instructions into the ``[N, L, text_dim]`` table (D8)."""
        config = self.config
        texts = read_episode_texts(config.train_data_dir, int(config.num_text_slots))
        max_len = int(getattr(config, "wan_max_sequence_length", 512))
        chunk_size = max(1, int(getattr(config, "text_encode_batch", 8)))
        chunks = []
        for start in range(0, len(texts), chunk_size):
            chunks.append(encode_positive_prompts(pipeline, texts[start : start + chunk_size], max_len))
        table = jnp.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

        expected_shape = (len(texts), max_len, int(config.text_dim))
        if tuple(table.shape) != expected_shape:
            raise ValueError(
                f"overfit100 context table shape {tuple(table.shape)} != expected {expected_shape} "
                f"(num_text_slots x wan_max_sequence_length x text_dim)"
            )
        table = table.astype(_dtype(config.weights_dtype))
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            table = jax.device_put(table, NamedSharding(mesh, P()))
        if jax.process_index() == 0:
            max_logging.log(context_table_audit_line(table))
        return table

    def _checkpoint_manager_options(self) -> ocp.CheckpointManagerOptions:
        """H2: keep EVERY checkpoint. ``max_to_keep=3`` (inherited) would evict the S2/S3
        gate baselines -- step 250 is gone by the time step 2500 lands -- and ``keep_period``
        cannot express a non-uniform list. Storage is predeclared: ~30 GB per checkpoint
        (bf16 params + both Adam moments) => S2 ~120 GB, S3 segment 1 ~150 GB on GCS."""
        return ocp.CheckpointManagerOptions(
            create=True,
            max_to_keep=None,
            enable_async_checkpointing=True,
        )

    def _build_checkpoint_manager(self, ckpt_dir: str) -> ocp.CheckpointManager:
        """The parent's Composite layout (params/opt_state/step) with keep-everything options.

        The item names and handlers are byte-identical to the parent's, which is exactly
        why the inherited ``_save_checkpoint`` / ``_maybe_restore`` (exp_01's proven path)
        keep working -- and why ``context_table``, being in neither ``params`` nor
        ``opt_state``, is excluded from both the save and the restore targets.
        """
        tf.io.gfile.makedirs(ckpt_dir)
        return ocp.CheckpointManager(
            ckpt_dir,
            item_names=("params", "opt_state", "step"),
            item_handlers={
                "params": ocp.StandardCheckpointHandler(),
                "opt_state": ocp.StandardCheckpointHandler(),
                "step": ocp.JsonCheckpointHandler(),
            },
            options=self._checkpoint_manager_options(),
        )

    def _shard_state(self, mesh, state: Overfit100TrainState) -> tuple[Overfit100TrainState, Any]:
        with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            state_shardings = nnx.get_named_sharding(state, mesh)
            state_shardings = _apply_actual_sharding_for_tpu(state, state_shardings)
            state_shardings = _overfit100_state_shardings(state_shardings, state, NamedSharding(mesh, P()))
            self._log_and_audit_shardings(state, state_shardings, mesh)
            state = _to_target_if_cpu(state, state_shardings)
            state = jax.device_put(state, state_shardings)
        return state, state_shardings

    def start_training(self):
        config = self.config
        # ---- PREFLIGHT (cycle-C review C1/C2/C3): everything that can fail cheaply fails
        # here, before a single weight is touched. Order matters: config gates, then the
        # model-revision pin, then the dataset bytes + context-map compatibility.
        self._validate_probe_config(config)  # guide_scale == 1.0, fresh noise (full-FT)
        self._validate_overfit100_config(config)
        self._validate_pinned_snapshot(config)
        self._preflight_dataset()
        pipeline = self._load_wan_pipeline()
        mesh = pipeline.mesh

        context_table = self._build_context_table(pipeline, mesh)
        # Free the text encoder / VAE exactly as the parent does after the null context.
        for attr in ("vae", "vae_cache", "text_encoder", "tokenizer"):
            if hasattr(pipeline, attr):
                delattr(pipeline, attr)

        # The transformer IS the trainable module -- no adapter is built.
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            transformer_graphdef, transformer_params, transformer_rest = nnx.split(
                pipeline.transformer, nnx.Param, ...
            )

        if jax.process_index() == 0:
            n_trainable = adapter_param_count(transformer_params)
            max_logging.log(f"[wan_overfit100] trainable transformer params: {n_trainable / 1e9:.2f}B")

        tx, lr_schedule = self._build_optimizer(config.max_train_steps)
        state = Overfit100TrainState.create(
            apply_fn=transformer_graphdef.apply,
            params=transformer_params,
            tx=tx,
            graphdef=transformer_graphdef,
            rest_of_state=transformer_rest,
            context_table=context_table,
        )

        if jax.process_index() == 0:
            mu_tree, nu_tree = _adam_moment_trees(state.opt_state)
            max_logging.log(f"[wan_overfit100] param dtypes: {_format_dtype_summary(_dtype_summary(state.params))}")
            max_logging.log(f"[wan_overfit100] adam mu dtypes: {_format_dtype_summary(_dtype_summary(mu_tree))}")
            max_logging.log(f"[wan_overfit100] adam nu dtypes: {_format_dtype_summary(_dtype_summary(nu_tree))}")
            max_logging.log(f"[wan_overfit100] activations dtype: {config.activations_dtype}")
            max_logging.log(context_table_audit_line(state.context_table))

        state, state_shardings = self._shard_state(mesh, state)
        data_shardings = self._data_shardings(mesh)

        scheduler, _ = self._create_scheduler()
        ckpt_dir = config.checkpoint_dir or os.path.join(config.output_dir, "checkpoints")
        ckpt_mgr = self._build_checkpoint_manager(ckpt_dir)
        state, start_step = self._maybe_restore(ckpt_mgr, state)
        if start_step:
            max_logging.log(
                f"[wan_overfit100] resumed at step {start_step} "
                "(params/opt_state/step restored; the context table is REBUILT, never checkpointed)"
            )

        train_iter = self._load_dataset(mesh, is_training=True, seed=config.seed + start_step)
        p_train_step = jax.jit(
            functools.partial(_train_step, scheduler=scheduler, config=config),
            in_shardings=(state_shardings, data_shardings, None),
            out_shardings=(state_shardings, None, None),
            donate_argnums=(0,),
        )
        p_eval_step = jax.jit(
            functools.partial(_eval_step, scheduler=scheduler, config=config),
            in_shardings=(state_shardings, data_shardings, None),
            out_shardings=(None, None),
        )

        ckpt_scheduler = CheckpointScheduler(
            checkpoint_steps=getattr(config, "checkpoint_steps", ()),
            checkpoint_every=getattr(config, "checkpoint_every", 0),
            max_train_steps=config.max_train_steps,
            save_final=bool(config.save_final_checkpoint),
        )

        if jax.process_index() == 0:
            max_logging.log("***** Running WAN TI2V overfit100 text-conditioned full finetune *****")
            max_logging.log(f"  Per-device batch size: {config.per_device_batch_size}")
            max_logging.log(f"  Devices: {jax.device_count()}")
            max_logging.log(f"  Max train steps: {config.max_train_steps}")
            max_logging.log(f"  Output dir: {config.output_dir}")
            max_logging.log(f"  Train data dir: {config.train_data_dir}")
            max_logging.log(f"  Eval data dir: {config.eval_data_dir}")
            max_logging.log(f"  Expected windows: {config.expected_windows}")
            max_logging.log(f"  Text slots: {config.num_text_slots} (encode batch {config.text_encode_batch})")
            max_logging.log(f"  Warmup steps: {getattr(config, 'warmup_steps', 0)}")
            max_logging.log(f"  Denoising sigma steps: {config.side_adapter_sampling_steps}")
            max_logging.log(f"  Timestep sampling: {getattr(config, 'side_adapter_t_sampling', 'uniform')}")
            max_logging.log(f"  Noise mode: {getattr(config, 'side_adapter_noise_mode', 'fresh')}")
            max_logging.log(f"  Guidance scale: {config.side_adapter_guide_scale}")
            max_logging.log(
                "  Checkpoint steps (planned, all retained): "
                f"{planned_checkpoint_steps(max_train_steps=config.max_train_steps, checkpoint_steps=getattr(config, 'checkpoint_steps', ()), checkpoint_every=getattr(config, 'checkpoint_every', 0), save_final=bool(config.save_final_checkpoint), start_step=start_step)}"
            )
            # C4: when both knobs are set, say which one governs.
            precedence = ckpt_scheduler.precedence_note()
            if precedence is not None:
                max_logging.log(precedence)

        wandb_run = None
        if jax.process_index() == 0 and getattr(config, "wandb_project", ""):
            import wandb

            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=getattr(config, "wandb_entity", None) or None,
                name=config.run_name or None,
                settings=wandb.Settings(start_method="thread"),
            )

        rng = jax.random.key(config.seed + 1)
        recent_loss: list[float] = []
        recent_grad: list[float] = []
        last_log_time = datetime.datetime.now()
        batch = load_next_batch(train_iter, None, config)

        with ThreadPoolExecutor(max_workers=1) as executor:
            for step in range(start_step, config.max_train_steps):
                next_batch_future = executor.submit(load_next_batch, train_iter, batch, config)
                with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
                    state, metrics, rng = p_train_step(state, batch, rng)
                    metrics["scalar"]["learning/loss"].block_until_ready()

                recent_loss.append(float(metrics["scalar"]["learning/loss"]))
                recent_grad.append(float(metrics["scalar"]["learning/grad_norm"]))

                if (step + 1) % config.log_period == 0 and jax.process_index() == 0:
                    now = datetime.datetime.now()
                    avg_loss = sum(recent_loss) / len(recent_loss)
                    avg_grad = sum(recent_grad) / len(recent_grad)
                    sps = len(recent_loss) / max(1e-6, (now - last_log_time).total_seconds())
                    lr = float(lr_schedule(step))
                    max_logging.log(
                        f"step {step + 1}/{config.max_train_steps} "
                        f"loss={avg_loss:.6f} grad_norm={avg_grad:.3f} "
                        f"lr={lr:.2e} steps/s={sps:.3f}"
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": avg_loss,
                                "train/grad_norm": avg_grad,
                                "train/lr": lr,
                                "train/steps_per_sec": sps,
                            },
                            step=step + 1,
                        )
                    recent_loss.clear()
                    recent_grad.clear()
                    last_log_time = now

                if config.eval_every > 0 and config.eval_data_dir and (step + 1) % config.eval_every == 0:
                    self._run_eval(mesh, p_eval_step, state, data_shardings, step + 1, rng, wandb_run)

                # H2: the explicit step LIST decides, not one periodic cadence.
                if ckpt_scheduler.should_save(step + 1):
                    self._save_checkpoint(ckpt_mgr, step + 1, state)

                batch = next_batch_future.result()

        final_step = ckpt_scheduler.final_step()
        if final_step is not None:
            self._save_checkpoint(ckpt_mgr, final_step, state)
        ckpt_mgr.wait_until_finished()
        if wandb_run is not None:
            wandb_run.finish()
