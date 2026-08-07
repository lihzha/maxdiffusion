"""J1 driver for exp_04 null-text inversion (plan §5 item 5; round R10).

``python src/maxdiffusion/run_wan_null_inversion.py <config.yml> key=value …``

This is the wiring that turns R1-R9 into a launchable job, and it is deliberately split in two:

- **Orchestration decisions are pure functions** -- mode dispatch, the batching plan, the decode
  wrapper's range contract, the divergence seam, the free-space floor, the manifest-bound reader's
  binding and coverage rules, the provenance digests. They take fakes and are unit-tested without a
  TPU or a model, and ``main`` itself is composed of them behind injectable seams, so a launch's
  whole decision surface -- including *which pipeline class loads* -- is asserted in this repo rather
  than discovered on a TPU (R10 review, finding 9).
- **Pipeline glue is thin**: the literal ``from_pretrained``/``encode_prompt``/decode calls, the
  TFRecord read and the hub snapshot lookup. Those, and only those, carry ``# pragma: no cover``.

**The backend is ``WanPipelineTI2V_2_2``, loaded under the trainer's ``axis_rules``.** Base
``WanPipeline`` has no ``from_pretrained`` -- that classmethod is the TI2V subclass's
(``wan_pipeline_ti2v_2p2.py:82``) -- and the load must happen inside
``nn_partitioning.axis_rules(config.logical_axis_rules)``, exactly as
``WanTI2VSideAdapterTrainer._load_wan_pipeline`` does (trainer lines 286-287), or the 5B transformer's
logical annotations have no rules to resolve against.

**Every example is read through its manifest row.** ``build_read_batch`` never scans for a name: it
looks the name up in the validated J0 cohort, re-checks that the source shard still carries the
generation and size the manifest bound it to, and refuses a record whose ordinal disagrees with its
row. A cohort is evidence, and evidence that silently re-resolves is not evidence.

**Decoded pixels are already ``[0, 1]``.** ``WanPipelineTI2V_2_2._decode_latents_to_video``
(``pipelines/wan/wan_pipeline.py:663-671``) ends in ``video_processor.postprocess_video(…, "np")``,
which per batch item calls ``VaeImageProcessor.postprocess`` (``video_processor.py:99-113``); that
denormalizes with ``(images / 2 + 0.5).clamp(0, 1)`` (``image_processor.py:185-189``, reached because
``do_normalize`` defaults True at ``image_processor.py:116``) and returns float32 ``[F, H, W, C]``
via ``pt_to_numpy`` (``image_processor.py:170-175``), stacked to ``[B, F, H, W, 3]``. That is exactly
R7's ``decode_fn`` contract, so the wrapper's default converts nothing, and it is pinned at **zero
tolerance**: the postprocessor clamps, so any excursion at all is a wiring bug rather than rounding.

**The convention is declared, and the declaration is trusted.** The check catches a decoder that
leaves its declared range -- a ``[-1, 1]`` backend declared ``unit`` -- but it cannot catch every
wrong declaration, because the ranges nest: ``[0, 1]`` pixels declared ``byte`` pass and are quietly
squeezed to ``[0, 0.0039]`` (R10 review, finding 10). Only ``unit`` is a claim about *this* backend,
verified against the source above; the other two are conveniences for a future pipeline, and a run
that declares one is trusting its operator, not being checked.

**Per-example pathology becomes ``ExampleDivergenceError``** at the orchestration seam, and only for
the R6 trace-finiteness signatures. Everything else -- geometry, provenance, OOM, preemption -- is
systemic and propagates, which is the condition R8's quarantine ratification rests on.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from maxdiffusion.null_adapter_cache_policy import FIDELITY_SUBSET_SIZE, ExampleDivergenceError
from maxdiffusion.null_adapter_records import PRODUCTION_GEOMETRY, SOURCE_DTYPES


NULL_MODES = ("capacity", "cache", "verify_replay", "adequacy_probe", "direct_opt", "transfer_probe")
# Plan §4-P2 fixes the fp16 decision to the first eight DEV examples whatever cohort is being cached,
# so a cache run reads from two manifests rather than one.
FIDELITY_COHORT = "dev64"
PIXEL_CONVENTIONS = {"unit": (0.0, 1.0), "signed": (-1.0, 1.0), "byte": (0.0, 255.0)}
# Exactly the R6 trace guards. A per-example optimization that diverged says one of these; nothing
# else may be laundered into a data gap (R8 review, finding 5).
DIVERGENCE_SIGNATURES = (
    "tracking_losses must be finite",
    "grad_norms must be finite",
    "per_step_final_losses must be finite",
    "final_losses must be finite",
)
# The per-example fields a record needs and the arms never see; the runner's own contract, restated
# here because this is where they are assembled from manifest rows.
EXAMPLE_FIELDS = ("ordinal", "split", "episode", "actions")
_SNAPSHOT = re.compile(r"snapshots/([0-9a-f]{40})")
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


def resolve_mode(mode: Any) -> str:
    """The four plan §5 modes, and nothing else -- a typo must not silently run the wrong job."""
    if not isinstance(mode, str) or mode not in NULL_MODES:
        raise ValueError(f"null_mode must be one of {list(NULL_MODES)}, got {mode!r}")
    return mode


def batching_plan(names: Sequence[str], batch_size: int) -> tuple[tuple[str, ...], ...]:
    """Manifest order, fixed-size batches, a short final batch -- deterministic and total.

    Manifest order is kept rather than sorted: the manifest *is* the cohort's order, and a resumed
    run that batched differently would produce different shard membership for the same work.
    """
    names = tuple(names)
    if not names:
        raise ValueError("the cohort is empty: there is nothing to run")
    if len(set(names)) != len(names):
        raise ValueError("cohort names must be unique")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch size must be an integer >= 1, got {batch_size!r}")
    return tuple(tuple(names[start : start + batch_size]) for start in range(0, len(names), batch_size))


def pixel_decoder(
    raw_decode: Callable[[Any], Any], convention: str = "unit", *, atol: float = 0.0
) -> Callable[[Any], np.ndarray]:
    """Wrap a pipeline decode into R7's ``decode_fn``: float32 ``[B, 33, H, W, 3]`` in ``[0, 1]``.

    The convention is declared, never detected, and for this pipeline it is ``unit`` with **no
    excursion allowance**: the postprocessor's own ``clamp(0, 1)`` is what makes zero tolerance the
    right number, and forgiving ``1.0005`` would reopen the strict-boundary exception R7 closed --
    a 5e-4 excursion already moves SSIM measurably. ``atol`` stays available for an alternate backend
    whose rounding genuinely needs it, but it is not something this run uses.

    See the module docstring on why a *declaration* cannot be fully checked: the three ranges nest.
    """
    if convention not in PIXEL_CONVENTIONS:
        raise ValueError(f"pixel convention must be one of {sorted(PIXEL_CONVENTIONS)}, got {convention!r}")
    if not np.isfinite(atol) or float(atol) < 0.0:
        raise ValueError(f"atol must be finite and non-negative, got {atol!r}")
    low, high = PIXEL_CONVENTIONS[convention]

    def decode_fn(latents):
        pixels = np.asarray(raw_decode(latents), dtype=np.float32)
        if not np.all(np.isfinite(pixels)):
            raise ValueError("the pipeline decode returned non-finite pixels")
        if pixels.min() < low - atol or pixels.max() > high + atol:
            raise ValueError(
                f"the pipeline decode returned {pixels.min():g}..{pixels.max():g}, outside the declared "
                f"{convention!r} range [{low:g}, {high:g}]"
            )
        return np.clip((pixels - low) / (high - low), 0.0, 1.0).astype(np.float32)

    return decode_fn


def divergent(error: BaseException) -> bool:
    """Whether a failure is one example's pathology rather than the job's problem."""
    message = str(error)
    return isinstance(error, (ValueError, RuntimeError)) and any(sig in message for sig in DIVERGENCE_SIGNATURES)


def guard_example_divergence(run_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Re-raise the R6 trace-finiteness failures as ``ExampleDivergenceError``; pass the rest through.

    This is the seam R8's quarantine policy is defined against: only what comes out of here as
    ``ExampleDivergenceError`` may become a recorded gap, so the translation is a short, explicit
    list rather than a blanket ``except Exception``.
    """

    def guarded(*args, **kwargs):
        try:
            return run_fn(*args, **kwargs)
        except ExampleDivergenceError:
            raise
        except (ValueError, RuntimeError) as error:
            if divergent(error):
                raise ExampleDivergenceError(f"{type(error).__name__}: {error}") from error
            raise

    return guarded


def check_free_space(path: str, minimum_bytes: int, *, statvfs: Callable[[str], Any] | None = None) -> int:
    """The SOP's storage guardrail: refuse to start a long write below the declared floor."""
    if isinstance(minimum_bytes, bool) or not isinstance(minimum_bytes, int) or minimum_bytes < 0:
        raise ValueError(f"the free-space floor must be a non-negative integer, got {minimum_bytes!r}")
    if path.startswith("gs://"):
        return -1  # object storage has no floor to check; the bucket is the operator's problem
    stat = (statvfs or os.statvfs)(path)
    free = int(stat.f_bavail) * int(stat.f_frsize)
    if free < minimum_bytes:
        raise RuntimeError(f"{path} has {free} bytes free, below the declared floor of {minimum_bytes}")
    return free


def sweep_stale_staging(staging_prefix: str, *, lister=None, remover=None) -> tuple[str, ...]:
    """Remove attempt directories a killed process left behind (R8: the runner owns this)."""
    if not staging_prefix:
        return ()
    if lister is None or remover is None:
        from tensorflow.io import gfile  # pragma: no cover -- object-store glue

        lister = lister or (lambda path: gfile.glob(posixpath.join(path, "*.staging")))  # pragma: no cover
        remover = remover or gfile.rmtree  # pragma: no cover
    stale = tuple(sorted(lister(staging_prefix)))
    for path in stale:
        remover(path)
    return stale


def published_shards(artifact_dir: str, *, lister: Callable[[str], Iterable[str]] | None = None) -> tuple[str, ...]:
    """Every shard directory under the artifact root that carries a completion marker."""
    from maxdiffusion.null_adapter_shards import MARKER_NAME

    if lister is None:
        from tensorflow.io import gfile  # pragma: no cover -- object-store listing glue

        lister = gfile.glob  # pragma: no cover
    markers = lister(posixpath.join(artifact_dir, "**", MARKER_NAME))
    return tuple(sorted(posixpath.dirname(path) for path in markers))


def optional_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Read a config key that may legitimately not exist, on every config object this driver sees.

    **The three-argument ``getattr`` does not work here.** ``pyconfig.HyperParameters.__getattr__``
    raises ``ValueError`` for an unknown key (``pyconfig.py:316-319``), and ``getattr``'s default
    only swallows ``AttributeError`` -- so the fallback never runs, it propagates. J1's smoke died
    exactly there, at ``code_sha``, after the 5B model was already loaded and its revision resolved.

    Resolution order: the object's own key mapping (``get_keys()``, which is what ``HyperParameters``
    exposes and the only *declared* way to ask it what it has), then a plain ``Mapping``, then
    attribute access guarded against **both** exception types.

    This is for genuinely optional keys. A key the YAML declares is required, and is read directly:
    the config ships in the same tarball as this file, so a missing one is a broken deployment that
    should say so loudly rather than silently take a default.
    """
    getter = getattr(type(config), "get_keys", None)
    if callable(getter):
        try:
            keys = config.get_keys()
        except Exception:  # noqa: BLE001 -- an uninitialized config must not break an optional read
            keys = None
        if isinstance(keys, Mapping):
            return keys.get(key, default)
    if isinstance(config, Mapping):
        return config.get(key, default)
    try:
        return getattr(config, key)
    except (AttributeError, ValueError):
        return default


def resolved_code_sha(value: Any) -> str:
    """The 40-hex commit every published record is stamped with.

    Required, and required to *look* like a commit. R10 shipped
    ``os.environ.get("COMMIT", "unknown")`` against a launcher that printed ``COMMIT`` but never
    exported it, so every record would have carried ``code_sha="unknown"`` -- a cache whose
    provenance says nothing, discovered only when someone tried to reproduce it (review, finding 7).
    """
    text = str(value or "").strip()
    if not _COMMIT_SHA.fullmatch(text):
        raise ValueError(
            f"code_sha must be a 40-character hex commit, got {value!r}; the launcher exports COMMIT "
            f"and a run without it has no provenance worth publishing"
        )
    return text


def model_revision(pretrained_path: str, *, resolved: str | None = None) -> str:
    """The revision string headers and ``verify_replay`` are checked against.

    An HF snapshot path carries its commit hash; anything else (a local directory, a ``gs://``
    mirror) is recorded verbatim, because an unidentifiable revision must not silently become one.
    """
    match = _SNAPSHOT.search(resolved or "")
    if match:
        return f"{pretrained_path}@{match.group(1)}"
    return f"{pretrained_path}@unresolved"


def manifest_rows(manifests: Mapping[str, Any], cohort: str) -> dict[str, dict[str, Any]]:
    """The cohort's validated rows, keyed by name, in manifest order."""
    if cohort not in manifests or cohort == "header":
        raise ValueError(
            f"the manifest set has no cohort {cohort!r}; it has {sorted(k for k in manifests if k != 'header')}"
        )
    rows = {}
    for row in manifests[cohort]["rows"]:
        if row["name"] in rows:
            raise ValueError(f"the {cohort} manifest names {row['name']!r} twice")
        rows[row["name"]] = dict(row)
    return rows


def cohort_names(manifests: Mapping[str, Any], cohort: str) -> tuple[str, ...]:
    """The cohort's names, in manifest order, from an already-validated manifest set."""
    return tuple(manifest_rows(manifests, cohort))


def reader_rows(manifests: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Every manifest row this run's reader has to be able to serve, keyed by name.

    Capacity, verification and the probe read only their own cohort. **Cache reads two.** It caches
    the selected cohort, but before a single example of it is written the plan's fp16 fidelity gate
    replays the first eight **DEV** examples -- which for a TRAIN-2000 job are not in the cohort at
    all. A reader built from the cohort alone refuses them by name, correctly, against the wrong row
    set, and J2 dies in its first phase (follow-up review, finding 1).

    The union is keyed by name, so a DEV cohort -- where the two lists overlap -- is unaffected, and
    every row still carries its own shard binding: spanning two manifests widens what may be read,
    never what may be read *unchecked*.
    """
    rows = manifest_rows(manifests, plan["cohort"])
    if plan["mode"] != "cache":
        return rows
    dev = manifest_rows(manifests, FIDELITY_COHORT)
    probe = {name: dev[name] for name in list(dev)[:FIDELITY_SUBSET_SIZE]}
    return {**probe, **rows}


def manifest_digest(manifests: Mapping[str, Any], cohort: str) -> str:
    """sha256 over the canonical JSON of the manifest set this run actually selected.

    The header's ``shard_listing_checksum`` is a digest of the *listing the builder walked*, not of
    the cohort a run published against: two cohorts cut from one scan share it, and a row edited
    after publication does not move it (review, finding 7). This hashes the header and the selected
    rows together, so the number in a shard header identifies the evidence the shard was built from.
    """
    payload = {"header": manifests["header"], "cohort": cohort, "rows": manifests[cohort]["rows"]}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_grid(spec: str) -> tuple[tuple[int, float], ...]:
    """``"10:0.01,25:0.03"`` -> ``((10, 0.01), (25, 0.03))`` -- the adequacy grid, from config."""
    cells = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        inner, _, lr = chunk.partition(":")
        if not inner.strip().isdigit() or not lr.strip():
            raise ValueError(f"adequacy grid cells must look like 'J:lr', got {chunk!r}")
        cells.append((int(inner), float(lr)))
    if not cells:
        raise ValueError("the adequacy grid is empty")
    return tuple(cells)


def _smoke_limit(config: Any) -> int:
    limit = config.null_smoke_examples
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError(f"null_smoke_examples must be a non-negative integer, got {limit!r}")
    return limit


def plan_run(config: Any, manifests: Mapping[str, Any]) -> dict[str, Any]:
    """Everything the run decides before touching a model: mode, cohort, batches, recipe, seeds.

    Pure, so the whole decision surface of a J1 launch can be asserted in a unit test.

    ``null_smoke_examples`` truncates **the declared cohort**, not just the work: J1's first rung is
    a two-example smoke, and a limiter that left ``names`` at 64 while running 2 would hand the gates
    a manifest with 62 holes and call the resulting coverage failure a result (review, finding 2).
    Zero -- the default -- means the full cohort.
    """
    mode = resolve_mode(config.null_mode)
    cohort = str(config.null_cohort)
    names = cohort_names(manifests, cohort)
    smoke = _smoke_limit(config)
    if smoke:
        names = names[:smoke]
    batches = batching_plan(names, int(config.null_batch_size))
    plan = {
        "mode": mode,
        "cohort": cohort,
        "names": names,
        "batches": batches,
        "smoke_examples": smoke,
        "params": {
            "inner_iters": int(config.null_inner_iters),
            "lr": float(config.null_lr),
            "guide_scale": float(config.null_guide_scale),
            "l_null": int(config.null_L),
        },
        "noise_convention": str(config.null_noise_convention),
        "latent_dtype": str(config.null_latent_dtype),
        "eval_seed": int(config.null_eval_seed),
        "decode_subset": tuple(names[: int(config.null_decode_subset)]),
        "decode_batch_size": int(config.null_decode_batch_size),
    }
    if mode == "adequacy_probe":
        plan["grid"] = _parse_grid(config.null_adequacy_grid)
    # The R6 header contract: optimization_config declares exactly these two keys.
    plan["optimization_config"] = {"inner_iters": plan["params"]["inner_iters"], "lr": plan["params"]["lr"]}
    return plan


def _checked_source(array: Any, label: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.shape != shape:
        raise ValueError(f"{label} must have the production shape {shape}, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be finite")
    return value


def build_read_batch(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    reader: Callable[[str, tuple[str, ...]], Iterable[tuple]],
    binder: Callable[[str], Mapping[str, Any]] | None = None,
) -> Callable[[Sequence[str]], tuple[Any, dict[str, dict[str, Any]]]]:
    """The manifest-bound reader the modes call: names in, ``(CapacityBatch, example_fields)`` out.

    Args:
      rows: the cohort's validated manifest rows, keyed by name (``manifest_rows``).
      reader: ``(shard_path, wanted) -> iterable of (name, ordinal, z_i0, z_video, actions)``,
        injected so every rule below is testable without TensorFlow.
      binder: ``shard_path -> {"generation", "size", …}``, defaulting to R9's ``shard_binding``. The
        binding is re-checked against the row before the shard's bytes are used, and **verified once
        per shard per process**: a GCS generation is immutable -- an overwrite mints a new one -- so
        re-statting per batch would add a subprocess call per batch for no new information.

    The returned reader is total about its own contract: it refuses a name outside the cohort, a
    record whose ordinal disagrees with its row, a shard that has moved since J0 bound it, a
    duplicate, and any name the shard did not yield. Rows come back in the **requested** order, which
    is what makes ``names[i]`` line up with row ``i`` of the stacked tensors downstream.
    """
    from maxdiffusion.null_adapter_runner_core import CapacityBatch  # lazy: keeps module scope jax-free

    if binder is None:
        from maxdiffusion.null_adapter_manifest_io import shard_binding

        binder = shard_binding
    verified: set[str] = set()

    def _verify_binding(shard_path: str, row: Mapping[str, Any]) -> None:
        if shard_path in verified:
            return
        binding = binder(shard_path)
        found = (str(binding.get("generation")), int(binding.get("size", -1)))
        declared = (str(row["shard_generation"]), int(row["shard_size"]))
        if found != declared:
            raise RuntimeError(
                f"{shard_path} is not the object the manifest bound: it now reports "
                f"generation/size {found}, the cohort was built against {declared}"
            )
        verified.add(shard_path)

    def read_batch(names: Sequence[str]) -> tuple[Any, dict[str, dict[str, Any]]]:
        names = tuple(names)
        if not names:
            raise ValueError("read_batch was asked for no examples")
        if len(set(names)) != len(names):
            raise ValueError(f"read_batch names must be unique, got {list(names)}")
        strangers = [name for name in names if name not in rows]
        if strangers:
            raise ValueError(f"these names are not in the cohort manifest: {strangers}")

        by_shard: dict[str, list[str]] = {}
        for name in names:
            by_shard.setdefault(str(rows[name]["shard_path"]), []).append(name)

        found: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for shard_path, wanted in sorted(by_shard.items()):
            _verify_binding(shard_path, rows[wanted[0]])
            for name, ordinal, z_i0, z_video, actions in reader(shard_path, tuple(wanted)):
                if name not in set(wanted):
                    continue
                if name in found:
                    raise ValueError(f"{name!r} appears more than once in {shard_path}")
                declared = rows[name]["ordinal"]
                if int(ordinal) != int(declared):
                    raise ValueError(
                        f"{name!r} is ordinal {int(ordinal)} in {shard_path} but the manifest binds it "
                        f"to ordinal {int(declared)}: this is not the window the cohort selected"
                    )
                found[name] = (
                    _checked_source(z_i0, f"{name}: z_i0", PRODUCTION_GEOMETRY.z_i0),
                    _checked_source(z_video, f"{name}: z_video", PRODUCTION_GEOMETRY.z_video),
                    _checked_source(actions, f"{name}: actions", PRODUCTION_GEOMETRY.actions),
                )
        missing = [name for name in names if name not in found]
        if missing:
            raise ValueError(f"the manifest's shards did not yield {missing}")

        batch = CapacityBatch(
            names=names,
            z_i0=np.stack([found[name][0] for name in names]),
            z_video=np.stack([found[name][1] for name in names]),
        )
        fields = {
            name: {
                "ordinal": int(rows[name]["ordinal"]),
                "split": str(rows[name]["split"]),
                "episode": str(rows[name]["episode"]),
                "actions": found[name][2].astype(SOURCE_DTYPES["actions"]),
            }
            for name in names
        }
        return batch, fields

    return read_batch


def _tfrecord_reader(shard_path: str, wanted: tuple[str, ...]):  # pragma: no cover -- TFRecord IO glue
    """Yield ``(name, ordinal, z_i0, z_video, actions)`` for the wanted names of one source shard."""
    import tensorflow as tf

    features = {
        "name": tf.io.FixedLenFeature([], tf.string),
        "ordinal": tf.io.FixedLenFeature([], tf.int64),
        "z_i0": tf.io.FixedLenFeature([], tf.string),
        "z_video": tf.io.FixedLenFeature([], tf.string),
        "actions": tf.io.FixedLenFeature([], tf.string),
    }
    remaining = set(wanted)
    for raw in tf.data.TFRecordDataset([shard_path]):
        if not remaining:
            break
        parsed = tf.io.parse_single_example(raw, features)
        name = parsed["name"].numpy().decode("utf-8")
        if name not in remaining:
            continue
        remaining.discard(name)
        yield (
            name,
            int(parsed["ordinal"].numpy()),
            np.frombuffer(parsed["z_i0"].numpy(), np.float16).reshape(PRODUCTION_GEOMETRY.z_i0),
            np.frombuffer(parsed["z_video"].numpy(), np.float16).reshape(PRODUCTION_GEOMETRY.z_video),
            np.frombuffer(parsed["actions"].numpy(), np.float32).reshape(PRODUCTION_GEOMETRY.actions),
        )


def _resolved_snapshot(pretrained_path: str) -> str:  # pragma: no cover -- hub cache lookup
    """The local snapshot directory the launcher prefetched, whose name carries the commit."""
    try:
        from huggingface_hub import snapshot_download

        return str(snapshot_download(pretrained_path, local_files_only=True))
    except Exception:  # noqa: BLE001 -- a local dir or gs:// mirror simply has no snapshot
        return ""


def _load_backend(config, rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Load the pipeline, compute T5("") once, free the text encoder, and return the seams.

    Mirrors ``WanTI2VSideAdapterTrainer._compute_null_context`` (trainer lines 313-325) and its
    ``_load_wan_pipeline`` (lines 286-287) without importing the trainer, and never imports
    ``generate_wan_side_adapter``, whose module-level
    ``jax.config.update("jax_use_shardy_partitioner", True)`` would follow the import in.
    """
    import jax.numpy as jnp
    from flax.linen import partitioning as nn_partitioning

    from maxdiffusion.models.wan.side_adapter_wan import _dtype
    from maxdiffusion.pipelines.wan.wan_pipeline_ti2v_2p2 import WanPipelineTI2V_2_2

    with nn_partitioning.axis_rules(config.logical_axis_rules):  # pragma: no cover -- model load
        pipeline = WanPipelineTI2V_2_2.from_pretrained(config)  # pragma: no cover
    mesh = pipeline.mesh
    prompt_embeds, _ = pipeline.encode_prompt(  # pragma: no cover -- T5 forward
        prompt=[""],
        negative_prompt=[""],
        num_videos_per_prompt=1,
        max_sequence_length=int(config.wan_max_sequence_length),
    )
    base_context = jnp.asarray(prompt_embeds.astype(_dtype(config.activations_dtype))).astype(jnp.float32)[0]
    for attribute in ("text_encoder", "tokenizer"):
        if hasattr(pipeline, attribute):
            delattr(pipeline, attribute)

    transformer = pipeline.transformer

    def velocity_fn(latents, timestep_2d, encoder_hidden_states):
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            return transformer(  # pragma: no cover -- the 5B forward
                hidden_states=latents.astype(_dtype(config.activations_dtype)),
                timestep=timestep_2d,
                encoder_hidden_states=encoder_hidden_states.astype(_dtype(config.activations_dtype)),
                deterministic=True,
            ).astype(jnp.float32)

    def raw_decode(latents):  # pragma: no cover -- the VAE decode
        return pipeline._decode_latents_to_video(pipeline._denormalize_latents(jnp.asarray(latents, jnp.float32)))

    return {
        "pipeline": pipeline,
        "mesh": mesh,
        "base_context": base_context,
        "velocity_fn": velocity_fn,
        "decode_fn": pixel_decoder(raw_decode, str(config.null_pixel_convention)),
        "read_batch": build_read_batch(rows, reader=_tfrecord_reader),
        "resolved": _resolved_snapshot(str(config.pretrained_model_name_or_path)),
    }


def _configure(argv: Sequence[str]):  # pragma: no cover -- pyconfig glue
    from maxdiffusion import pyconfig

    pyconfig.initialize(argv)
    return pyconfig.config


def _artifact_exists(uri: str) -> bool:  # pragma: no cover -- object-store existence glue
    from tensorflow.io import gfile

    return bool(gfile.exists(uri))


def _load_manifests(manifest_dir: str):  # pragma: no cover -- artifact IO glue
    from maxdiffusion.null_adapter_manifest_io import load_manifests

    return load_manifests(manifest_dir)


def load_adoption(uri: str, *, exists: Callable[[str], bool], read_json: Callable[[str], Any]) -> dict | None:
    """The adequacy artifact's adoption block, or ``None`` when there is no artifact at all.

    **Fail-closed against silent recipe drift.** Plan §4-P1 says that if a recipe is adopted, A1/A2
    are re-run under it *before* gating. An adequacy artifact sitting at the configured URI while
    capacity quietly runs at (J=10, lr=1e-2) is the failure that produces a perfectly plausible set
    of gate verdicts for an experiment nobody chose (follow-up review, finding 3). So: no artifact is
    fine, an artifact is consumed, and an artifact that cannot be understood stops the run.
    """
    if not uri or not exists(uri):
        return None
    payload = read_json(uri)  # any read failure propagates: the artifact exists, so it must be read
    adoption = payload.get("adopted") if isinstance(payload, Mapping) else None
    if not isinstance(adoption, Mapping) or not {"inner_iters", "lr", "adopted"} <= set(adoption):
        raise ValueError(
            f"{uri} exists but carries no usable adoption block: capacity will not fall back to the "
            f"default recipe while an adequacy result is sitting next to it"
        )
    return dict(adoption)


def mode_kwargs(
    config,
    plan: Mapping[str, Any],
    manifests: Mapping[str, Any],
    *,
    shards_for,
    adoption: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Exactly the arguments each mode takes, decided from config -- pure, so it is testable whole."""
    common: dict[str, Any] = {"artifact_dir": str(config.null_artifact_dir)}
    if plan["mode"] == "adequacy_probe":
        return common
    if plan["mode"] in ("capacity", "cache"):
        common.update(
            staging_dir=str(config.null_staging_dir),
            manifest_hash=manifest_digest(manifests, plan["cohort"]),
            code_sha=resolved_code_sha(optional_config_value(config, "code_sha", "") or os.environ.get("COMMIT")),
        )
    if plan["mode"] == "capacity":
        common.update(
            decode_batch_size=int(plan["decode_batch_size"]),
            adopted_recipe=adoption,
            a3_measure=bool(config.null_a3_measure),
        )
    # J1b: separately approved, so it takes only what it needs and stamps what it writes.
    if plan["mode"] == "transfer_probe":
        # J1c replays J1b's published tensors: the npz URI is the mode's whole input, and the run's
        # provenance travels with the table it produces.
        common.update(
            manifest_hash=manifest_digest(manifests, plan["cohort"]),
            code_sha=resolved_code_sha(optional_config_value(config, "code_sha", "") or os.environ.get("COMMIT")),
            nulls_uri=str(optional_config_value(config, "null_transfer_nulls_uri", "")),
            decode_batch_size=int(plan["decode_batch_size"]),
        )
    if plan["mode"] == "direct_opt":
        common.update(
            manifest_hash=manifest_digest(manifests, plan["cohort"]),
            code_sha=resolved_code_sha(optional_config_value(config, "code_sha", "") or os.environ.get("COMMIT")),
            iters=int(config.null_a3_iters),
        )
    # The selection was made on DEV-64, so that is the digest it is bound to -- not the digest of
    # whatever cohort this job happens to be caching.
    if plan["mode"] in ("cache", "verify_replay"):
        common.update(
            selection_uri=str(config.null_selection_uri) or None,
            selection_digest=manifest_digest(manifests, FIDELITY_COHORT),
        )
    if plan["mode"] == "cache":
        common.update(
            existing_shards=shards_for(str(config.null_artifact_dir)),
            dev_manifest=cohort_names(manifests, FIDELITY_COHORT),
        )
    if plan["mode"] == "verify_replay":
        common.update(
            shard_paths=shards_for(str(config.null_artifact_dir)),
            atol=float(config.null_verify_atol),
        )
    return common


def main(
    argv: Sequence[str],
    *,
    configure: Callable[[Sequence[str]], Any] | None = None,
    load_manifests: Callable[[str], Mapping[str, Any]] | None = None,
    load_backend: Callable[..., Mapping[str, Any]] | None = None,
    sinks: Any = None,
    shards_for: Callable[[str], tuple[str, ...]] | None = None,
    artifact_exists: Callable[[str], bool] | None = None,
) -> int:
    """Compose a J1 launch and run it. Every seam is injectable, so this path has real tests.

    The four seams -- config, manifests, backend, sinks -- are the only things here that touch a TPU,
    a bucket or a YAML file. R10 shipped them hard-wired behind a function-level ``no cover``, and
    that is exactly where the wrong pipeline class, the missing reader, the wrong resume fingerprint
    and the hard-coded arm all survived eighty tests (review, finding 9).
    """
    # Imported here, not at module scope: ``null_adapter_modes`` imports this module's pure decisions,
    # so a top-level import in either direction closes the cycle.
    from maxdiffusion.null_adapter_modes import Backend, default_sinks, execute

    config = (configure or _configure)(argv)
    check_free_space(str(config.null_artifact_dir), int(config.null_min_free_bytes))
    sweep_stale_staging(str(config.null_staging_dir))
    manifests = (load_manifests or _load_manifests)(str(config.null_manifest_dir))
    plan = plan_run(config, manifests)
    rows = reader_rows(manifests, plan)
    print(
        f"[null-inversion] mode={plan['mode']} cohort={plan['cohort']} examples={len(plan['names'])} "
        f"batches={len(plan['batches'])} smoke={plan['smoke_examples'] or 'off'}"
    )

    backend = (load_backend or _load_backend)(config, rows)
    revision = model_revision(str(config.pretrained_model_name_or_path), resolved=backend.get("resolved"))
    print(f"[null-inversion] revision={revision}")
    seams = Backend(
        velocity_fn=backend["velocity_fn"],
        decode_fn=backend["decode_fn"],
        read_batch=backend["read_batch"],
        base_context=backend["base_context"],
        model_revision=revision,
    )
    active_sinks = sinks or default_sinks()
    adoption = load_adoption(
        str(config.null_adequacy_uri),
        exists=artifact_exists or _artifact_exists,
        read_json=active_sinks.read_json,
    )
    if adoption:
        print(f"[null-inversion] adopted recipe from the adequacy artifact: {adoption}")
    kwargs = mode_kwargs(config, plan, manifests, shards_for=shards_for or published_shards, adoption=adoption)
    report, code = execute(plan["mode"], plan, seams, active_sinks, **kwargs)
    print(f"[null-inversion] {json.dumps(report, sort_keys=True, default=str)[:2000]}")
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
