"""GT-vs-prediction comparison videos for published exp_04 / exp_05 capacity artifacts.

Post-STOP visualization runbook (2026-08-09, Yixun-approved): render the arms the capacity jobs
never decoded to pixels. The decode-only arms (exp_04 A2, exp_05 B1/B2) come straight from each
record's ``expected_final_latent`` -- the writer's cast->replay->record discipline
(``null_adapter_runner_core`` / ``pos_context_modes.build_pos_capacity_records``) makes that tensor
the arm's deployed latent, so no sampler pass is needed. The probe arms (A1-probe, B1-probe) were
never published as latents anywhere, so they are replayed here from ``keyed(name, k)`` starts with
the record's stored per-step conditioning -- the same operators and the same noise construction the
capacity runs used.

Every recomputed ``future_ssim`` is cross-checked against the experiment's published
``gate_tables.json`` entry in the report, so a wrong-arm or wrong-clip wiring error cannot pass
silently: the published tables were computed by the original jobs from their own in-memory latents.

This module is orchestration only -- numpy in, seams out. Everything that touches a TPU, a bucket
or ffmpeg is injected (``replay``, ``decode_fn``, ``save_video``, ``write_json``, ``read_json``,
``list_shards``, ``read_shard``), mirroring ``run_wan_null_inversion.main``'s seam discipline.
"""

from __future__ import annotations

import dataclasses
import posixpath
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from maxdiffusion.null_adapter_pixels import (
    PROBE_K_SET,
    SINGLE_SEED_KEY,
    _decoded,
    _score_pair,
    comparison_video_frames,
)

VIDEO_FPS = 16
REPORT_NAME = "videos_report.json"
TABLES_NAME = "gate_tables.json"


@dataclasses.dataclass(frozen=True)
class SectionSpec:
    """What one experiment contributes: which arms decode, which arm's conditioning probes."""

    label: str  # "exp_04_null" | "exp_05_pos"
    decode_arms: tuple[str, ...]  # arms rendered from stored expected_final_latent
    probe_arm: str  # arm whose stored conditioning is replayed from keyed noise
    embed_field: str  # "nulls" | "pos_embeds"

    def __post_init__(self) -> None:
        if self.probe_arm not in self.decode_arms and self.probe_arm not in ("a1",):
            # a1 already has published own-basin videos; its record still feeds the probe replay.
            raise ValueError(f"probe arm {self.probe_arm!r} must be a decode arm or 'a1'")


NULL_SPEC = SectionSpec(label="exp_04_null", decode_arms=("a2",), probe_arm="a1", embed_field="nulls")
POS_SPEC = SectionSpec(label="exp_05_pos", decode_arms=("b1", "b2"), probe_arm="b1", embed_field="pos_embeds")


def load_arm(
    root: str, arm: str, *, list_shards: Callable[[str], Sequence[str]], read_shard: Callable[[str], tuple[Any, tuple]]
) -> tuple[Any, dict[str, Any]]:
    """Every record of one published arm, by name, plus the (shared) shard header."""
    shards = tuple(list_shards(posixpath.join(root, arm)))
    if not shards:
        raise ValueError(f"no published shards under {root}/{arm}: wrong root or unpublished arm")
    header, records = None, {}
    for shard in shards:
        shard_header, shard_records = read_shard(shard)
        header = header if header is not None else shard_header
        for record in shard_records:
            if record.name in records:
                raise ValueError(f"{arm}: record {record.name!r} appears in more than one shard")
            records[record.name] = record
    return header, records


def subset_records(records_by_name: Mapping[str, Any], subset: int) -> tuple[Any, ...]:
    """The first ``subset`` records in manifest order -- ``ordinal`` is the manifest ordinal, so this
    reproduces the capacity runs' ``decode_subset`` (the clips behind exp_04's published 00-07)."""
    if subset < 1:
        raise ValueError(f"subset must be >= 1, got {subset}")
    ordered = sorted(records_by_name.values(), key=lambda record: int(record.ordinal))
    return tuple(ordered[:subset])


def _stacked(records: Sequence[Any], field: str) -> np.ndarray:
    return np.stack([np.asarray(getattr(record, field), dtype=np.float32) for record in records])


def _chunks(count: int, size: int):
    for start in range(0, count, size):
        yield start, min(start + size, count)


def published_future_ssim(tables: Any, method: str, name: str, seed_key: str) -> float | None:
    """The original job's future_ssim for (method, clip, seed) -- ``None`` when absent, never a throw:
    the cross-check is a report field for the human, not a gate (announcement 03)."""
    try:
        value = tables[method][name][seed_key]["future_ssim"]
    except (KeyError, TypeError):
        return None
    return float(value)


def render_section(
    spec: SectionSpec,
    *,
    source_root: str,
    out_root: str,
    subset: int,
    probe_k: int,
    replay: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray],
    keyed_noise_fn: Callable[[str, int], Any],
    decode_fn: Callable[[np.ndarray], Any],
    save_video: Callable[[Any, str, int], str],
    write_json: Callable[[str, Any], str],
    read_json: Callable[[str], Any],
    decode_batch: int = 8,
    fps: int = VIDEO_FPS,
    list_shards: Callable[[str], Sequence[str]],
    read_shard: Callable[[str], tuple[Any, tuple]],
) -> dict[str, Any]:
    """Render one experiment's missing arms and publish mp4s + the run-level report (written last)."""
    if probe_k not in PROBE_K_SET:
        raise ValueError(f"probe_k must be one of {PROBE_K_SET}, got {probe_k}")

    arms: dict[str, tuple[Any, dict[str, Any]]] = {}
    for arm in dict.fromkeys(spec.decode_arms + (spec.probe_arm,)):
        arms[arm] = load_arm(source_root, arm, list_shards=list_shards, read_shard=read_shard)

    name_sets = {arm: frozenset(records) for arm, (_, records) in arms.items()}
    if len(set(name_sets.values())) != 1:
        raise ValueError(f"{spec.label}: arm cohorts disagree: " + str({a: len(s) for a, s in name_sets.items()}))

    probe_header, probe_records = arms[spec.probe_arm]
    chosen = subset_records(probe_records, subset)
    names = tuple(record.name for record in chosen)
    guide_scale = float(probe_header.guide_scale)

    # Ground truth comes from the records themselves -- byte-identical across arms by construction.
    z_video = _stacked(chosen, "z_video")

    latents_by_method: dict[str, tuple[str, str, np.ndarray]] = {}  # method -> (published method, seed, latents)
    for arm in spec.decode_arms:
        records = [arms[arm][1][name] for name in names]
        latents_by_method[arm] = (arm, SINGLE_SEED_KEY, _stacked(records, "expected_final_latent"))

    starts = np.stack([np.asarray(keyed_noise_fn(name, probe_k), dtype=np.float32) for name in names])
    embeds = np.stack([np.asarray(getattr(r, spec.embed_field), dtype=np.float32) for r in chosen], axis=1)
    probe_latents = np.asarray(replay(starts, _stacked(chosen, "z_i0"), embeds, guide_scale), dtype=np.float32)
    probe_method = f"{spec.probe_arm}_probe"
    latents_by_method[f"{probe_method}_k{probe_k}"] = (probe_method, str(probe_k), probe_latents)

    try:
        tables = read_json(posixpath.join(source_root, TABLES_NAME))
    except Exception as error:  # the cross-check is best-effort; the report says when it was impossible
        tables = None
        tables_error = repr(error)
    else:
        tables_error = None

    report_arms: dict[str, Any] = {}
    gt_pixels = np.concatenate([_decoded(decode_fn, z_video[lo:hi]) for lo, hi in _chunks(len(names), decode_batch)])
    for method, (published_method, seed_key, latents) in latents_by_method.items():
        pixels = np.concatenate([_decoded(decode_fn, latents[lo:hi]) for lo, hi in _chunks(len(names), decode_batch)])
        videos, metrics = {}, {}
        for index, record in enumerate(chosen):
            stacked = comparison_video_frames(gt_pixels[index], pixels[index])
            # Manifest names carry '/' (e.g. "droid_ep_000001/w0"); flatten so one arm dir holds its clips.
            safe_name = str(record.name).replace("/", "_")
            path = posixpath.join(out_root, method, f"{int(record.ordinal):03d}_{safe_name}.mp4")
            videos[record.name] = save_video(stacked, path, fps)
            scored = _score_pair(pixels[index], gt_pixels[index])
            reference = published_future_ssim(tables, published_method, record.name, seed_key)
            scored["published_future_ssim"] = reference
            scored["future_ssim_delta_vs_published"] = None if reference is None else scored["future_ssim"] - reference
            metrics[record.name] = scored
        report_arms[method] = {"videos": videos, "metrics": metrics}

    report = {
        "experiment": spec.label,
        "source_root": source_root,
        "out_root": out_root,
        "subset": list(names),
        "ordinals": [int(record.ordinal) for record in chosen],
        "probe_k": probe_k,
        "fps": fps,
        "gate_tables_cross_check": "loaded" if tables is not None else f"unavailable: {tables_error}",
        "provenance": {
            "source_code_sha": str(probe_header.code_sha),
            "model_revision": str(probe_header.model_revision),
            "guide_scale": guide_scale,
            "dtype_policy": str(probe_header.dtype_policy),
            "optimization_config": dict(probe_header.optimization_config),
        },
        "arms": report_arms,
    }
    write_json(posixpath.join(out_root, REPORT_NAME), report)  # last write = the root is authoritative
    return report
