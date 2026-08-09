"""Tests for the post-STOP capacity-videos runbook (capacity_videos.py).

Everything here is CPU-only and seam-injected: fake shards, a shape-honest fake decoder, and the
REAL replay operators driven by a zero-velocity function -- so the noise construction, the
[N, B, L, D] embed stacking and the first-frame pin run for real, without a TPU or a bucket.
"""

import posixpath
from types import SimpleNamespace

import numpy as np
import pytest

from maxdiffusion.capacity_videos import (
    NULL_SPEC,
    POS_SPEC,
    REPORT_NAME,
    TABLES_NAME,
    load_arm,
    published_future_ssim,
    render_section,
    subset_records,
)
from maxdiffusion.models.wan.null_inversion_wan import keyed_noise, replay_with_nulls
from maxdiffusion.models.wan.pos_context_inversion_wan import replay_with_positive
from maxdiffusion.null_adapter_verify import canonical_sigmas

Z_VIDEO = (48, 9, 12, 20)
Z_I0 = (48, 1, 12, 20)
PIXELS = (33, 4, 4, 3)


def _record(name, ordinal, embed_field, l_embed, seed):
    rng = np.random.default_rng(seed)
    fields = {
        "name": name,
        "ordinal": ordinal,
        "z_i0": rng.uniform(-1, 1, Z_I0).astype(np.float16),
        "z_video": rng.uniform(-1, 1, Z_VIDEO).astype(np.float16),
        "z_start": rng.uniform(-1, 1, Z_VIDEO).astype(np.float16),
        "expected_final_latent": rng.uniform(-1, 1, Z_VIDEO).astype(np.float16),
        embed_field: rng.uniform(-1, 1, (25, l_embed, 4096)).astype(np.float16),
    }
    return SimpleNamespace(**fields)


def _header():
    return SimpleNamespace(
        guide_scale=5.0,
        code_sha="a" * 40,
        model_revision="rev-test",
        dtype_policy="fp16",
        optimization_config={"inner_iters": 10, "lr": 0.01},
    )


def _fake_shards(spec, names_ordinals, l_embed):
    """One shard per arm, records shared across arms except each arm's own prediction tensors."""
    shards = {}
    for arm in dict.fromkeys(spec.decode_arms + (spec.probe_arm,)):
        records = []
        for index, (name, ordinal) in enumerate(names_ordinals):
            record = _record(name, ordinal, spec.embed_field, l_embed, seed=hash((arm, index)) % 2**32)
            base = _record(name, ordinal, spec.embed_field, l_embed, seed=index)  # shared GT across arms
            record.z_video, record.z_i0 = base.z_video, base.z_i0
            records.append(record)
        shards[f"root/{arm}"] = (_header(), tuple(records))
    return shards


def _list_shards(shards):
    return lambda prefix: tuple(sorted(path for path in shards if path.startswith(prefix)))


def _read_shard(shards):
    return lambda path: shards[path]


def _decode_fn(latents):
    """Shape-honest, deterministic, in [0, 1]: each clip's pixels depend on its latents."""
    latents = np.asarray(latents, dtype=np.float32)
    values = 0.5 + 0.4 * np.tanh(latents.mean(axis=(1, 2, 3, 4), keepdims=False))
    return np.tile(values[:, None, None, None, None], (1,) + PIXELS).astype(np.float32)


class _Sink:

    def __init__(self, tables=None):
        self.videos, self.jsons, self.tables = {}, {}, tables

    def save_video(self, frames, path, fps):
        assert frames.shape == (PIXELS[0], 2 * PIXELS[1], PIXELS[2], PIXELS[3]), frames.shape  # GT stacked over pred
        self.videos[path] = fps
        return path

    def write_json(self, path, payload):
        self.jsons[path] = payload
        return path

    def read_json(self, path):
        if self.tables is None:
            raise FileNotFoundError(path)
        assert path.endswith(TABLES_NAME)
        return self.tables


def _zero_velocity(latents, timestep_2d, encoder_hidden_states):
    import jax.numpy as jnp

    return jnp.zeros_like(latents)


def _null_replay(z_start, z_i0, embeds, guide_scale):
    return replay_with_nulls(
        _zero_velocity,
        z_start,
        z_i0,
        canonical_sigmas(),
        embeds,
        np.zeros((512, 4096), np.float32),
        guide_scale=guide_scale,
    )


def _pos_replay(z_start, z_i0, embeds, guide_scale):
    return replay_with_positive(
        _zero_velocity,
        z_start,
        z_i0,
        canonical_sigmas(),
        embeds,
        np.zeros((512, 4096), np.float32),
        guide_scale=guide_scale,
    )


NAMES = [(f"clip_{i}/w0", 10 * i) for i in range(3)]


def _run(spec, replay, *, tables=None, subset=2, probe_k=0):
    shards = _fake_shards(spec, NAMES, l_embed=16 if spec is NULL_SPEC else 8)
    sink = _Sink(tables)
    report = render_section(
        spec,
        source_root="root",
        out_root="out",
        subset=subset,
        probe_k=probe_k,
        replay=replay,
        keyed_noise_fn=lambda name, k: np.asarray(keyed_noise(name, k)),
        decode_fn=_decode_fn,
        save_video=sink.save_video,
        write_json=sink.write_json,
        read_json=sink.read_json,
        list_shards=_list_shards(shards),
        read_shard=_read_shard(shards),
    )
    return report, sink


class TestSubsetAndLoading:

    def test_subset_orders_by_ordinal_and_truncates(self):
        records = {n: SimpleNamespace(name=n, ordinal=o) for n, o in [("b", 5), ("a", 20), ("c", 1)]}
        chosen = subset_records(records, 2)
        assert [record.name for record in chosen] == ["c", "b"]

    def test_load_arm_refuses_empty_and_duplicates(self):
        with pytest.raises(ValueError, match="no published shards"):
            load_arm("root", "a2", list_shards=lambda prefix: (), read_shard=lambda path: None)
        dup = SimpleNamespace(name="x", ordinal=0)
        with pytest.raises(ValueError, match="more than one shard"):
            load_arm(
                "root",
                "a2",
                list_shards=lambda prefix: ("s0", "s1"),
                read_shard=lambda path: (_header(), (dup,)),
            )

    def test_published_lookup_is_total(self):
        tables = {"a2": {"clip": {"0": {"future_ssim": 0.5}}}}
        assert published_future_ssim(tables, "a2", "clip", "0") == 0.5
        assert published_future_ssim(tables, "a2", "missing", "0") is None
        assert published_future_ssim(None, "a2", "clip", "0") is None


class TestNullSection:

    def test_renders_a2_and_a1_probe_with_cross_check(self):
        subset_names = [name for name, _ in NAMES[:2]]
        tables = {
            "a2": {name: {"0": {"future_ssim": 0.9}} for name in subset_names},
            "a1_probe": {name: {"0": {"future_ssim": 0.2}} for name in subset_names},
        }
        report, sink = _run(NULL_SPEC, _null_replay, tables=tables)
        assert set(report["arms"]) == {"a2", "a1_probe_k0"}
        assert report["subset"] == subset_names and report["ordinals"] == [0, 10]
        # 2 clips x 2 arms rendered, named by manifest ordinal + name, plus the report written last.
        assert sorted(sink.videos) == [
            "out/a1_probe_k0/000_clip_0_w0.mp4",
            "out/a1_probe_k0/010_clip_1_w0.mp4",
            "out/a2/000_clip_0_w0.mp4",
            "out/a2/010_clip_1_w0.mp4",
        ]
        assert list(sink.jsons) == [posixpath.join("out", REPORT_NAME)]
        for name in subset_names:
            entry = report["arms"]["a2"]["metrics"][name]
            assert entry["published_future_ssim"] == 0.9
            assert entry["future_ssim_delta_vs_published"] == pytest.approx(entry["future_ssim"] - 0.9)
            probe = report["arms"]["a1_probe_k0"]["metrics"][name]
            assert probe["published_future_ssim"] == 0.2  # looked up under ("a1_probe", k-seed), not the video key
        assert report["gate_tables_cross_check"] == "loaded"
        assert report["provenance"]["source_code_sha"] == "a" * 40

    def test_zero_velocity_probe_reproduces_pinned_keyed_noise(self):
        # With v == 0 the replay is the identity on frames 1.. and the pin on frame 0, so the probe's
        # decoded pixels must equal decoding keyed_noise(name, 0) with frame 0 replaced by z_i0 -- this
        # pins the noise construction and the pin, the two silent ways to render the wrong probe.
        report, _ = _run(NULL_SPEC, _null_replay)
        shards = _fake_shards(NULL_SPEC, NAMES, l_embed=16)
        _, records = shards["root/a1"]
        for index, name in enumerate(report["subset"]):
            record = records[index]
            expected = np.asarray(keyed_noise(name, 0)).copy()
            expected[:, :1] = np.asarray(record.z_i0, np.float32)
            pixels = _decode_fn(expected[None])[0]
            gt = _decode_fn(np.asarray(record.z_video, np.float32)[None])[0]
            from maxdiffusion.null_adapter_pixels import _score_pair

            assert report["arms"]["a1_probe_k0"]["metrics"][name]["future_ssim"] == pytest.approx(
                _score_pair(pixels, gt)["future_ssim"], abs=1e-6
            )

    def test_missing_tables_is_reported_not_fatal(self):
        report, _ = _run(NULL_SPEC, _null_replay, tables=None)
        assert report["gate_tables_cross_check"].startswith("unavailable")
        entry = next(iter(report["arms"]["a2"]["metrics"].values()))
        assert entry["published_future_ssim"] is None and entry["future_ssim_delta_vs_published"] is None


class TestPosSection:

    def test_renders_b1_b2_and_b1_probe(self):
        report, sink = _run(POS_SPEC, _pos_replay)
        assert set(report["arms"]) == {"b1", "b2", "b1_probe_k0"}
        assert len(sink.videos) == 6  # 2 clips x 3 arms
        assert all(path.startswith(("out/b1/", "out/b2/", "out/b1_probe_k0/")) for path in sink.videos)

    def test_probe_k_selects_the_keyed_start(self):
        report_k0, _ = _run(POS_SPEC, _pos_replay, probe_k=0)
        report_k2, _ = _run(POS_SPEC, _pos_replay, probe_k=2)
        name = report_k0["subset"][0]
        assert report_k0["arms"]["b1_probe_k0"]["metrics"][name]["future_ssim"] != pytest.approx(
            report_k2["arms"]["b1_probe_k2"]["metrics"][name]["future_ssim"]
        )

    def test_bad_probe_k_refused(self):
        with pytest.raises(ValueError, match="probe_k"):
            _run(POS_SPEC, _pos_replay, probe_k=7)


class TestGuards:

    def test_arm_cohort_mismatch_refused(self):
        shards = _fake_shards(NULL_SPEC, NAMES, l_embed=16)
        header, records = shards["root/a2"]
        shards["root/a2"] = (header, records[:-1])  # drop one clip from a2 only
        sink = _Sink()
        with pytest.raises(ValueError, match="cohorts disagree"):
            render_section(
                NULL_SPEC,
                source_root="root",
                out_root="out",
                subset=2,
                probe_k=0,
                replay=_null_replay,
                keyed_noise_fn=lambda name, k: np.asarray(keyed_noise(name, k)),
                decode_fn=_decode_fn,
                save_video=sink.save_video,
                write_json=sink.write_json,
                read_json=sink.read_json,
                list_shards=_list_shards(shards),
                read_shard=_read_shard(shards),
            )
