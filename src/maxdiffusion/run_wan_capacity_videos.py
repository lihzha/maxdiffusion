"""Entrypoint: render the never-decoded capacity arms of exp_04 / exp_05 as comparison videos.

One backend load serves both experiments: ``_load_backend``'s ``velocity_fn`` casts contexts to the
activation dtype exactly as exp_04's capacity runs did, and the positive replays additionally go
through ``casting_velocity_fn`` -- idempotent over the backend's own cast -- so each experiment's
probe replay reproduces its own production semantics. Sections are selected by which roots are
configured; an empty root skips that experiment.

Usage (the launcher ``bash_scripts/run_wan_capacity_videos.sh`` sets all of this):

  python src/maxdiffusion/run_wan_capacity_videos.py \\
      src/maxdiffusion/configs/base_wan_5b_pos_inversion.yml \\
      videos_null_capacity_root=gs://... videos_null_out=gs://... \\
      videos_pos_capacity_root=gs://... videos_pos_out=gs://... run_name=videos ...
"""

from __future__ import annotations

import json
import posixpath
import sys
from typing import Any, Sequence

import numpy as np

from maxdiffusion.capacity_videos import NULL_SPEC, POS_SPEC, render_section
from maxdiffusion.null_adapter_verify import canonical_sigmas
from maxdiffusion.run_wan_null_inversion import _load_backend, published_shards


def pos_read_shard(shard_path: str) -> tuple[Any, tuple[Any, ...]]:
    """``null_adapter_modes.read_shard``, with the positive-slot parsers: the null reader would
    (correctly) refuse a B-shard's members, so the pos slot gets its own three lines."""
    from tensorflow.io import gfile

    from maxdiffusion.null_adapter_shards import HEADER_NAME, MARKER_NAME, marker_from_json
    from maxdiffusion.pos_context_records import pos_header_from_json, pos_record_from_bytes

    with gfile.GFile(posixpath.join(shard_path, MARKER_NAME), "r") as handle:
        marker = marker_from_json(handle.read())
    with gfile.GFile(posixpath.join(shard_path, HEADER_NAME), "r") as handle:
        header = pos_header_from_json(handle.read())
    records = []
    for name in marker.names:
        with gfile.GFile(posixpath.join(shard_path, marker.files[name]), "rb") as handle:
            records.append(pos_record_from_bytes(handle.read()))
    return header, tuple(records)


def main(argv: Sequence[str]) -> int:  # pragma: no cover -- TPU/bucket composition glue
    from maxdiffusion import pyconfig
    from maxdiffusion.models.wan.null_inversion_wan import keyed_noise, replay_with_nulls
    from maxdiffusion.models.wan.pos_context_inversion_wan import replay_with_positive
    from maxdiffusion.models.wan.side_adapter_wan import _dtype
    from maxdiffusion.null_adapter_modes import publish_video, read_json, read_shard, write_json
    from maxdiffusion.pos_context_modes import casting_velocity_fn

    pyconfig.initialize(argv)
    config = pyconfig.config

    null_root = str(config.videos_null_capacity_root)
    pos_root = str(config.videos_pos_capacity_root)
    null_out = str(config.videos_null_out)
    pos_out = str(config.videos_pos_out)
    subset = int(config.videos_subset)
    probe_k = int(config.videos_probe_k)
    decode_batch = int(config.null_decode_batch_size)
    if not null_root and not pos_root:
        raise ValueError("nothing to do: set videos_null_capacity_root and/or videos_pos_capacity_root")
    for root, out in ((null_root, null_out), (pos_root, pos_out)):
        if bool(root) != bool(out):
            raise ValueError("each configured capacity root needs its matching *_out root, and vice versa")

    backend = _load_backend(config, {})
    sigmas = canonical_sigmas()
    base_context = backend["base_context"]
    velocity_fn = backend["velocity_fn"]
    pos_velocity_fn = casting_velocity_fn(velocity_fn, _dtype(str(config.activations_dtype)))

    def null_replay(z_start, z_i0, embeds, guide_scale):
        return replay_with_nulls(velocity_fn, z_start, z_i0, sigmas, embeds, base_context, guide_scale=guide_scale)

    def pos_replay(z_start, z_i0, embeds, guide_scale):
        return replay_with_positive(
            pos_velocity_fn, z_start, z_i0, sigmas, embeds, base_context, guide_scale=guide_scale
        )

    seams = {
        "subset": subset,
        "probe_k": probe_k,
        "decode_batch": decode_batch,
        "keyed_noise_fn": lambda name, k: np.asarray(keyed_noise(name, k), dtype=np.float32),
        "decode_fn": backend["decode_fn"],
        "save_video": lambda frames, path, fps: publish_video(frames, path, fps),
        "write_json": write_json,
        "read_json": read_json,
        "list_shards": published_shards,
    }
    reports = {}
    if null_root:
        print(f"[capacity-videos] exp_04 null slot: {null_root} -> {null_out}")
        reports["exp_04_null"] = render_section(
            NULL_SPEC, source_root=null_root, out_root=null_out, replay=null_replay, read_shard=read_shard, **seams
        )
    if pos_root:
        print(f"[capacity-videos] exp_05 positive slot: {pos_root} -> {pos_out}")
        reports["exp_05_pos"] = render_section(
            POS_SPEC, source_root=pos_root, out_root=pos_out, replay=pos_replay, read_shard=pos_read_shard, **seams
        )
    for label, report in reports.items():
        summary = {
            method: {name: round(entry["future_ssim"], 4) for name, entry in arm["metrics"].items()}
            for method, arm in report["arms"].items()
        }
        print(f"[capacity-videos] {label} future_ssim: {json.dumps(summary, sort_keys=True)}")
    print(f"[capacity-videos] done: {sum(len(r['arms']) for r in reports.values())} arms rendered")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
