"""exp_05 S4 — the runner's positive-slot extension (plan §5 items 2-3, §4-P1').

Four things carry this round:

- **Additivity.** ``run_wan_null_inversion.py`` is a class-(c) dual-touch file under plan §6's merge
  policy and ``null_adapter_modes.py`` is untouched entirely. ``test_the_null_slot_path_is_unchanged``
  is the proof the merge rule rests on: at the default slot the plan, the dispatch and an end-to-end
  fake-backend capacity report are exactly exp_04's.
- **The pivots differ.** exp_05 inverts at the deployed **8-token** context, not the 512-row T5("").
  ``test_the_inversion_runs_at_the_eight_token_context`` captures the inversion call and pins its
  context shape -- the guarantee behind plan §3's "no artifact reuse from exp_04".
- **THE S4 MUST.** The runner-built closure casts **both** branches' contexts to the activation dtype.
  S2/S3 could only prove the operators do *not* cast; this is the other half, at bf16, on a real
  tiny WanModel.
- **``l_pos``, not ``l_null``.** The positive header names the slot and the row count honestly.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import types

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bit_test_helpers import f32_bits as _f32_bits
from maxdiffusion.models.wan.pos_context_inversion_wan import POS_L, pos_context_from_t5, replay_with_positive
from maxdiffusion.null_adapter_gates import Target
from maxdiffusion.null_adapter_modes import Backend, Sinks, execute
from maxdiffusion.null_adapter_records import PRODUCTION_GEOMETRY
from maxdiffusion.null_adapter_runner_core import PROBE_K_SET, CapacityBatch, CapacityParams
from maxdiffusion.null_adapter_verify import canonical_sigmas
from maxdiffusion.pos_context_modes import (
    POS_METHODS,
    positive_plan,
    build_pos_capacity_records,
    casting_velocity_fn,
    pos_default_sinks,
    pos_execute,
    pos_header_for,
    run_pos_adequacy,
    run_pos_capacity_example_batch,
)
from maxdiffusion.pos_context_records import pos_header_from_json, pos_header_to_json
from maxdiffusion.run_wan_null_inversion import EMBEDDING_SLOTS, main, plan_run, resolve_embedding_slot


_S, _D = 512, 4096
_LATENT = PRODUCTION_GEOMETRY.z_video
_NAMES = ("ep1_v0_s00000", "ep2_v0_s00004")
_SHA = "a" * 40
_RNG = np.random.default_rng(20260805)
_BASE_CONTEXT = jnp.asarray(_RNG.standard_normal((_S, _D), dtype=np.float32) * 0.02)
_WEIGHTS = jnp.asarray(_RNG.standard_normal((POS_L, _D), dtype=np.float32) * 0.02)
_PATTERN = jnp.asarray(_RNG.standard_normal(_LATENT, dtype=np.float32) * 0.01)


class _Fake:
    """A toy backend in exp_04's mode-test shape: coupled oracle, stand-in VAE, deterministic reader."""

    def __init__(self):
        self.contexts: list[tuple[int, ...]] = []
        self.context_dtypes: list[Any] = []

    def read_batch(self, names):
        names = tuple(names)
        rng = np.random.default_rng(abs(hash(names)) % (2**32))
        batch = CapacityBatch(
            names=names,
            z_i0=jnp.asarray(rng.standard_normal((len(names), 48, 1, 12, 20), dtype=np.float32) * 0.5),
            z_video=jnp.asarray(rng.standard_normal((len(names), *_LATENT), dtype=np.float32) * 0.5),
        )
        fields = {
            name: {
                "ordinal": index,
                "split": "dev64",
                "episode": name.split("_")[0][2:],
                "actions": rng.standard_normal((32, 7), dtype=np.float32),
            }
            for index, name in enumerate(names)
        }
        return batch, fields

    def velocity_fn(self, z, timestep_2d, context):
        self.contexts.append(tuple(context.shape))
        self.context_dtypes.append(jnp.asarray(context).dtype)
        scale = jnp.sum(context[:, :POS_L] * _WEIGHTS, axis=(1, 2))
        return scale[:, None, None, None, None] * _PATTERN + 0.5 * z + 1e-5 * jnp.mean(timestep_2d)

    def decode_fn(self, latents):
        z = np.asarray(latents, np.float32)
        per_frame = z.mean(axis=1)
        frames = np.concatenate([per_frame[:, :1], np.repeat(per_frame[:, 1:], 4, axis=1)], axis=1)
        return np.repeat((0.5 + 0.4 * np.tanh(frames))[..., None], 3, axis=-1).astype(np.float32)

    def backend(self):
        return Backend(
            velocity_fn=self.velocity_fn,
            decode_fn=self.decode_fn,
            read_batch=self.read_batch,
            base_context=_BASE_CONTEXT,
            model_revision="Wan2.2-TI2V-5B@" + "b" * 40,
        )


class _Recorder:
    """Fake sinks that keep what would have been published."""

    def __init__(self):
        self.shards, self.json, self.json_paths = [], {}, []

    def sinks(self):
        return Sinks(
            write_shard=self._write_shard,
            write_json=self._write_json,
            save_video=lambda frames, path, fps: path,
            read_shard=lambda path: (None, ()),
            resume_plan=lambda *a, **k: None,
            validate_shard=lambda *a, **k: None,
            read_json=lambda path: {},
            read_marker=lambda path: None,
        )

    def _write_shard(self, records, header, shard_path, staging, *, quarantined=None):
        self.shards.append({"path": shard_path, "records": list(records), "header": header})
        return self.shards[-1]

    def _write_json(self, path, payload):
        self.json_paths.append(path)
        self.json[path.rsplit("/", 1)[-1]] = json.loads(json.dumps(payload, default=str))
        return path


def _manifests(names=_NAMES):
    rows = [
        {
            "name": name,
            "ordinal": index,
            "split": "dev64",
            "episode": name.split("_")[0][2:],
            "shard_path": "gs://bucket/shard-0.tfrecord",
            "shard_generation": "1700000000000000",
            "shard_size": 4096,
        }
        for index, name in enumerate(names)
    ]
    return {"header": {"built": "2026-08-05"}, "dev64": {"rows": rows}}


def _config(**overrides):
    base = {
        "null_mode": "capacity",
        "null_cohort": "dev64",
        "null_batch_size": 2,
        "null_inner_iters": 2,
        "null_lr": 0.01,
        "null_guide_scale": 5.0,
        "null_L": 16,
        "null_noise_convention": "keyed",
        "null_latent_dtype": "fp16",
        "null_eval_seed": 0,
        "null_decode_subset": 0,
        "null_decode_batch_size": 2,
        "null_smoke_examples": 0,
        "null_artifact_dir": "gs://bucket/artifacts",
        "null_staging_dir": "",
        "null_manifest_dir": "gs://bucket/manifests",
        "null_min_free_bytes": 0,
        "code_sha": _SHA,
        "pretrained_model_name_or_path": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    }
    return types.SimpleNamespace(**{**base, **overrides})


_POSITIVE = {
    "embedding_slot": "positive",
    "pos_artifact_dir": "gs://bucket/pos-artifacts",
    "pos_staging_dir": "gs://bucket/pos-staging",
    "pos_L": POS_L,
    "pos_ablation_L": "1,8",
    "pos_adequacy_uri": "",
    "activations_dtype": "bfloat16",
}


def _header(fake, **overrides):
    plan = {
        "params": {"guide_scale": 5.0, "l_pos": POS_L},
        "optimization_config": {"inner_iters": 1, "lr": 0.01},
        "latent_dtype": "fp16",
        **overrides,
    }
    return pos_header_for(plan, fake.backend(), manifest_hash="m" * 8, code_sha=_SHA)


def _run_main(fake, recorder, observed=None, **overrides):
    config = _config(**overrides)
    observed = {"swept": [], "free_space": []} if observed is None else observed
    return main(
        ["prog", "config.yml"],
        configure=lambda argv: config,
        load_manifests=lambda path: _manifests(),
        load_backend=lambda cfg, rows: {
            "velocity_fn": fake.velocity_fn,
            "decode_fn": fake.decode_fn,
            "read_batch": fake.read_batch,
            "base_context": _BASE_CONTEXT,
            "resolved": "",
        },
        sinks=recorder.sinks(),
        shards_for=lambda path: (),
        artifact_exists=lambda uri: False,
        sweep=lambda root: observed["swept"].append(root) or (),
        free_space=lambda root, floor: observed["free_space"].append(root) or 0,
    )


# --------------------------------------------------------------------------------------------------
# 1. Slot dispatch and the additivity proof.
# --------------------------------------------------------------------------------------------------


def test_the_default_slot_is_null_and_an_unknown_slot_is_refused():
    assert EMBEDDING_SLOTS == ("null", "positive")
    assert resolve_embedding_slot(types.SimpleNamespace()) == "null"  # absent key -> exp_04's path
    assert resolve_embedding_slot(_config()) == "null"  # ... as does an unset one
    # Unset in any of its spellings -- absent, empty, None -- is "not configured", i.e. exp_04's path.
    assert resolve_embedding_slot(_config(embedding_slot="")) == "null"
    assert resolve_embedding_slot(_config(embedding_slot=None)) == "null"
    assert resolve_embedding_slot(_config(embedding_slot="positive")) == "positive"
    for bad in ("pos", "POSITIVE", "nulls", 1, True, ["positive"]):
        with pytest.raises(ValueError, match="embedding_slot must be one of"):
            resolve_embedding_slot(_config(embedding_slot=bad))


def test_the_null_slot_path_is_unchanged():
    """**THE ADDITIVITY PROOF** the merge-2 rule depends on (plan §6, class-(c) files).

    At the default slot the run must be indistinguishable from exp_04's: the same plan dict (no new
    keys), the same mode dispatch, and the same end-to-end report from the same fakes. The report is
    compared against ``null_adapter_modes.execute`` called directly, so a routing change of any kind
    -- including one that merely *added* something to the null path -- shows up here.
    """
    plan = plan_run(_config(), _manifests())
    assert set(plan) == {
        "mode",
        "cohort",
        "names",
        "batches",
        "smoke_examples",
        "params",
        "noise_convention",
        "latent_dtype",
        "eval_seed",
        "decode_subset",
        "decode_batch_size",
        "optimization_config",
    }
    assert "embedding_slot" not in plan and "l_pos" not in plan["params"]

    routed = _Recorder()
    exit_code = _run_main(_Fake(), routed)
    direct, _ = execute(
        "capacity",
        plan_run(_config(), _manifests()),
        _Fake().backend(),
        _Recorder().sinks(),
        artifact_dir="gs://bucket/artifacts",
        staging_dir="",
        manifest_hash=_digest(),
        code_sha=_SHA,
        decode_batch_size=2,
        adopted_recipe=None,
    )

    assert exit_code == 0
    routed_report = routed.json["run_report.json"]
    assert routed_report["mode"] == "capacity" and "embedding_slot" not in routed_report
    for key in ("cohort", "declared", "examples", "recipe", "target", "shards", "tables"):
        assert routed_report[key] == json.loads(json.dumps(direct[key], default=str)), key


def _digest():
    from maxdiffusion.run_wan_null_inversion import manifest_digest

    return manifest_digest(_manifests(), "dev64")


def test_the_positive_slot_routes_to_the_b_arms():
    fake = _Fake()
    recorder = _Recorder()

    code = _run_main(fake, recorder, **_POSITIVE)

    report = recorder.json["run_report.json"]
    assert code == 0 and report["embedding_slot"] == "positive" and report["l_pos"] == POS_L
    assert sorted(report["tables"]) == sorted(POS_METHODS)  # B-arms, not A-arms
    assert [shard["path"].rsplit("/", 2)[-2] for shard in recorder.shards] == ["b1", "b2"]
    assert recorder.json["selection.json"]["embedding_slot"] == "positive"

    # S4 review, finding 3: no positive write may land in the null slot's tree.
    written = [shard["path"] for shard in recorder.shards] + list(recorder.json_paths)
    assert written and all(path.startswith(_POSITIVE["pos_artifact_dir"]) for path in written), written
    assert not any(path.startswith(_config().null_artifact_dir) for path in written)

    # S4 review, finding 6: the closure the modes actually call is the casting one, observed through
    # ``main`` -- not a wrapper with no call site.
    assert fake.context_dtypes and all(dtype == jnp.bfloat16 for dtype in fake.context_dtypes)


def test_pos_execute_refuses_a_mode_it_has_not_wired():
    """A positive-slot cache/verify must not fall through to the null-slot implementation."""
    for mode in ("cache", "verify_replay", "nonsense"):
        with pytest.raises(ValueError, match="positive slot wires"):
            pos_execute(mode, {}, _Fake().backend(), _Recorder().sinks())


def test_pos_default_sinks_wires_this_slot_s_own_shard_writer():
    """S4b, folded in: exp_04's writer hard-codes its own codec, so the positive slot has its own --
    and the production sink points at it instead of at a late refusal (S4 review, finding 5)."""
    from maxdiffusion.pos_context_modes import pos_write_shard

    assert pos_default_sinks().write_shard is pos_write_shard


@pytest.mark.parametrize(
    "mangle, message",
    [
        (lambda h, r: (dataclasses.replace(h, embedding_slot="null"), r), "positive-slot header"),
        (lambda h, r: (dataclasses.replace(h, l_pos=3), r), "context rows but the header"),
        (lambda h, r: (h, r + r), "may not carry a name twice"),
        (lambda h, r: (dataclasses.replace(h, dtype_policy="fp32"), r), "but the header declares"),
    ],
)
def test_the_positive_shard_writer_refuses_a_shard_that_disagrees_with_its_header(mangle, message):
    """The writer preflight, before a byte is staged: slot, row count, dtype policy, name bijection."""
    from maxdiffusion.pos_context_modes import pos_write_shard

    fake = _Fake()
    batch, fields = fake.read_batch(_NAMES)
    result = run_pos_capacity_example_batch(fake.velocity_fn, batch, _BASE_CONTEXT, CapacityParams(inner_iters=1))
    header = _header(fake)
    records = build_pos_capacity_records(fake.velocity_fn, result, batch, _BASE_CONTEXT, header, fields, arm="b1")
    header, records = mangle(header, records)

    with pytest.raises(ValueError, match=message):
        pos_write_shard(records, header, "gs://bucket/pos/b1/shard_00000", "gs://bucket/pos/staging")


def test_the_run_level_json_is_published_only_after_every_shard_has_landed():
    """**S4 review, finding 5.** A failing writer must not leave an authoritative-looking
    ``selection.json`` from a run that published no records at all."""

    class _FailingWriter(_Recorder):
        def _write_shard(self, records, header, shard_path, staging, *, quarantined=None):
            raise RuntimeError("the object store rejected the write")

    recorder = _FailingWriter()
    with pytest.raises(RuntimeError, match="rejected the write"):
        _run_main(_Fake(), recorder, **_POSITIVE)

    assert recorder.json == {}, recorder.json  # no tables, no selection, no report


def test_the_real_config_object_without_the_key_resolves_to_the_null_slot():
    """**S4 review, finding 1 -- the BLOCKER.** ``HyperParameters.__getattr__`` raises ``ValueError``
    for an undeclared key, so ``getattr(config, key, default)`` never reaches its default and every
    exp_04 YAML -- none of which declare ``embedding_slot`` -- would have crashed a **null** launch.

    Exercised against the class's **real source**, extracted from ``pyconfig.py`` and executed here,
    rather than against a ``SimpleNamespace`` (which is what let the first characterization test miss
    this) and rather than by importing ``pyconfig``, whose module chain needs the full TPU stack.
    """
    import ast
    import pathlib

    source = (pathlib.Path(__file__).resolve().parents[2] / "pyconfig.py").read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HyperParameters")
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "pyconfig-extract", "exec"), namespace)

    class _Raw:
        def __init__(self, keys):
            self.keys = keys

    namespace["_config"] = _Raw({"null_mode": "capacity"})  # an exp_04 config: no embedding_slot
    config = namespace["HyperParameters"]()

    with pytest.raises(ValueError, match="not in config"):
        config.embedding_slot
    with pytest.raises(ValueError, match="not in config"):
        getattr(config, "embedding_slot", "sentinel")  # even WITH a default -- the bug S4 shipped
    assert resolve_embedding_slot(config) == "null"  # ... and the fix rides over it

    namespace["_config"] = _Raw({"embedding_slot": "positive"})
    assert resolve_embedding_slot(namespace["HyperParameters"]()) == "positive"


def test_the_positive_yaml_declares_every_key_the_positive_route_reads():
    """CLI overrides are rejected for keys a YAML does not declare, so the positive config must carry
    them -- and the null YAML must stay untouched."""
    import pathlib

    import yaml

    configs = pathlib.Path(__file__).resolve().parents[2] / "configs"
    positive = yaml.safe_load((configs / "base_wan_5b_pos_inversion.yml").read_text())
    null = yaml.safe_load((configs / "base_wan_5b_null_inversion.yml").read_text())

    assert positive["embedding_slot"] == "positive" and positive["pos_L"] == POS_L
    for key in ("pos_artifact_dir", "pos_staging_dir", "pos_adequacy_uri", "pos_ablation_L"):
        assert key in positive, key
    assert "embedding_slot" not in null and not any(k.startswith("pos_") for k in null)
    assert {k: v for k, v in positive.items() if not k.startswith("pos_") and k != "embedding_slot"} == null


def test_l_pos_threads_from_the_config_into_the_contexts_and_the_header():
    """**S4 review, finding 4.** A run asking for ``pos_L=1`` produced eight-row contexts and an
    ``l_pos=8`` header. One number now reaches the inversion context, the warm start, the frozen arms
    and the optimizer's initialization."""
    plan = positive_plan(_config(**{**_POSITIVE, "pos_L": 1}), plan_run(_config(), _manifests()))

    assert plan["params"]["l_pos"] == 1 and plan["embedding_slot"] == "positive"
    fake = _Fake()
    batch, _ = fake.read_batch(_NAMES)

    result = run_pos_capacity_example_batch(
        fake.velocity_fn, batch, _BASE_CONTEXT, CapacityParams(inner_iters=1), l_pos=1
    )

    assert fake.contexts[0] == (len(_NAMES), 1, _D)  # the inversion context is one row, not eight
    assert result["pos_embeds"]["b1"].shape[2] == 1 and result["l_pos"] == 1
    header = pos_header_for(
        {"params": {"guide_scale": 5.0, "l_pos": plan["params"]["l_pos"]}, "optimization_config": {},
         "latent_dtype": "fp16"},
        fake.backend(), manifest_hash="m" * 8, code_sha=_SHA,
    )
    assert header.l_pos == 1  # ... and the header says what actually ran


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"pos_artifact_dir": ""}, "needs its own pos_artifact_dir"),
        ({"pos_artifact_dir": "gs://bucket/artifacts"}, "must not be the null slot"),
        ({"pos_staging_dir": "gs://bucket/artifacts"}, "must not be the null slot"),
        ({"pos_L": 0}, "pos_L must be a positive integer"),
        ({"pos_ablation_L": "1,0"}, "pos_ablation_L cells must be positive"),
    ],
)
def test_a_positive_run_must_have_its_own_slot_isolated_roots(overrides, message):
    """**S4 review, finding 3.** A positive run writing under ``null_artifact_dir`` would put two
    experiments' ``selection.json`` in one directory."""
    with pytest.raises(ValueError, match=message):
        positive_plan(_config(**{**_POSITIVE, **overrides}), plan_run(_config(), _manifests()))


# --------------------------------------------------------------------------------------------------
# 2. The B-arms: the 8-token inversion, and composition with the S2/S3 operators.
# --------------------------------------------------------------------------------------------------


def test_the_inversion_runs_at_the_eight_token_context():
    """**THE PIVOTS-DIFFER PIN** (plan §3: exp_05 shares no artifacts with exp_04).

    The inversion's conditional context is the deployed 8-token one, so exp_05's trajectories are
    different tensors from exp_04's 512-row-context pivots. Inversion is the batch's **first** forward
    -- every arm depends on its output -- so the first context the backend sees is the pin. (The calls
    are counted by trace, not by iteration: the operators run their steps inside ``lax.scan``, so each
    call *site* records once.)
    """
    fake = _Fake()
    batch, _ = fake.read_batch(_NAMES)

    run_pos_capacity_example_batch(fake.velocity_fn, batch, _BASE_CONTEXT, CapacityParams(inner_iters=1))

    eight, full = (len(_NAMES), POS_L, _D), (len(_NAMES), _S, _D)
    assert fake.contexts[0] == eight, fake.contexts[:3]  # THE pin: inversion conditions on 8 tokens
    assert full in fake.contexts  # ... and the frozen 512-row T5("") is still the uncond branch
    assert fake.contexts.index(eight) < fake.contexts.index(full)


def test_every_b_arm_composes_with_the_s2_s3_operators():
    """B0/B1/B2 are the S3 replay operator driven by the S2 optimizer's output -- not a re-derivation.

    B0 replays the *frozen* warm-start context from traj[0]; B1 replays the optimized contexts from
    the same pivot; B2 starts from eps_0. Each is recomputed here by calling ``replay_with_positive``
    directly on the arm's own recorded inputs, so a cross-wired arm moves the result by O(1).
    """
    fake = _Fake()
    batch, _ = fake.read_batch(_NAMES)
    params = CapacityParams(inner_iters=1)

    result = run_pos_capacity_example_batch(fake.velocity_fn, batch, _BASE_CONTEXT, params)

    sigmas = jnp.asarray(canonical_sigmas())
    steps = sigmas.shape[0] - 1
    frozen = jnp.broadcast_to(pos_context_from_t5(_BASE_CONTEXT), (steps, len(_NAMES), POS_L, _D))
    traj0 = jnp.asarray(result["z_start"]["b1"])
    common = {"z_i0": jnp.asarray(batch.z_i0), "sigmas": sigmas, "base_context": _BASE_CONTEXT}
    for method, z_start, embeds in (
        ("b0", traj0, frozen),
        ("b1", traj0, jnp.asarray(result["pos_embeds"]["b1"])),
        ("b2", jnp.asarray(result["z_start"]["b2"]), jnp.asarray(result["pos_embeds"]["b2"])),
    ):
        direct = replay_with_positive(
            fake.velocity_fn,
            z_start,
            common["z_i0"],
            common["sigmas"],
            embeds,
            common["base_context"],
            guide_scale=params.guide_scale,
        )
        np.testing.assert_allclose(np.asarray(direct), result["final_latents"][method], rtol=1e-5, atol=1e-6)

    from maxdiffusion.models.wan.null_inversion_wan import global_noise

    np.testing.assert_array_equal(  # B2 really starts from the single canonical noise
        _f32_bits(result["z_start"]["b2"]), _f32_bits(jnp.broadcast_to(global_noise(0), (len(_NAMES), *_LATENT)))
    )
    for probe in ("b1_probe", "b2_probe"):
        assert result["final_latents"][probe].shape == (len(PROBE_K_SET), len(_NAMES), *_LATENT)
    assert result["z_bar_states"]["b1"].shape == (steps, len(_NAMES), *_LATENT)  # the K2 cache's states


def test_pos_records_round_trip_through_the_s5_codec_in_the_runner_flow():
    """The records the runner builds are S5 records: they serialize, parse back and carry the states."""
    from maxdiffusion.pos_context_records import pos_record_from_bytes, pos_record_to_bytes

    fake = _Fake()
    batch, fields = fake.read_batch(_NAMES)
    result = run_pos_capacity_example_batch(fake.velocity_fn, batch, _BASE_CONTEXT, CapacityParams(inner_iters=1))
    header = pos_header_for(
        {
            "params": {"guide_scale": 5.0, "l_pos": POS_L},
            "optimization_config": {"inner_iters": 1, "lr": 0.01},
            "latent_dtype": "fp16",
        },
        fake.backend(),
        manifest_hash="m" * 8,
        code_sha=_SHA,
    )

    records = build_pos_capacity_records(fake.velocity_fn, result, batch, _BASE_CONTEXT, header, fields, arm="b1")

    assert [record.name for record in records] == list(_NAMES)
    for record in records:
        parsed = pos_record_from_bytes(pos_record_to_bytes(record))
        assert parsed.pos_embeds.shape == (25, POS_L, _D) and parsed.z_bar_states.shape == (25, *_LATENT)
        assert parsed.arm == "B1" and parsed.noise_convention == "keyed"
        np.testing.assert_array_equal(parsed.z_bar_states.view(np.uint8), record.z_bar_states.view(np.uint8))


def test_build_pos_records_refuses_a_foreign_or_null_slot_header():
    fake = _Fake()
    batch, fields = fake.read_batch(_NAMES)
    result = run_pos_capacity_example_batch(fake.velocity_fn, batch, _BASE_CONTEXT, CapacityParams(inner_iters=1))
    header = pos_header_for(
        {"params": {"guide_scale": 5.0}, "optimization_config": {}, "latent_dtype": "fp16"},
        fake.backend(),
        manifest_hash="m" * 8,
        code_sha=_SHA,
    )
    import dataclasses

    with pytest.raises(ValueError, match="does not match this base_context"):
        build_pos_capacity_records(
            fake.velocity_fn,
            result,
            batch,
            _BASE_CONTEXT,
            dataclasses.replace(header, base_context_fingerprint="0" * 64),
            fields,
            arm="b1",
        )
    with pytest.raises(ValueError, match="arm must be one of"):
        build_pos_capacity_records(fake.velocity_fn, result, batch, _BASE_CONTEXT, header, fields, arm="a1")


# --------------------------------------------------------------------------------------------------
# 3. The l_pos header decision, and THE S4 MUST.
# --------------------------------------------------------------------------------------------------


def test_the_header_names_the_slot_and_carries_l_pos_not_l_null():
    """**THE l_pos DECISION.** exp_04's ``l_null`` would read as a claim about a 512-row null context;
    a positive shard states its own row count under its own name, plus the slot outright."""
    header = pos_header_for(
        {
            "params": {"guide_scale": 5.0, "l_pos": POS_L},
            "optimization_config": {"inner_iters": 10, "lr": 0.01},
            "latent_dtype": "fp16",
        },
        _Fake().backend(),
        manifest_hash="m" * 8,
        code_sha=_SHA,
    )

    assert header.l_pos == POS_L and header.embedding_slot == "positive"
    assert not hasattr(header, "l_null")
    parsed = pos_header_from_json(pos_header_to_json(header))
    assert parsed.l_pos == POS_L and parsed.embedding_slot == "positive"
    assert json.loads(pos_header_to_json(header))["l_pos"] == POS_L

    import dataclasses

    with pytest.raises(ValueError, match="embedding_slot must be"):
        pos_header_to_json(dataclasses.replace(header, embedding_slot="null"))
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="l_pos must be a positive integer"):
            pos_header_to_json(dataclasses.replace(header, l_pos=bad))


def test_the_runner_closure_casts_both_branches_at_bf16():
    """**THE S4 MUST, DISCHARGED.** The runner-built ``velocity_fn`` casts the context to the
    activation dtype immediately before the transformer call -- for the 8-token conditional branch
    **and** the 512-row unconditional one.

    S2/S3 proved the operators pass fp32 through untouched, which is exactly why nothing until now
    proved the wiring casts. Here a deliberately non-casting raw closure is wrapped, and each branch's
    output must equal the same call made with a pre-cast context and differ from the uncast one.
    """
    pytest.importorskip("torch")
    pytest.importorskip("aqt")
    from flax import nnx
    from flax.linen import partitioning as nn_partitioning
    from maxdiffusion.models.wan.transformers.transformer_wan import WanModel

    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(1, 1), ("data", "fsdp"))
    set_mesh = getattr(jax, "set_mesh", None)
    mesh_ctx = set_mesh(mesh) if set_mesh is not None else mesh
    channels, text_dim, f_lat, h_lat, w_lat = 4, 32, 2, 4, 6
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (2, channels, f_lat, h_lat, w_lat), jnp.float32)
    timestep = jnp.full((2, f_lat * (h_lat // 2) * (w_lat // 2)), 700.0, jnp.float32).at[:, :6].set(0.0)
    contexts = {
        "conditional (8 tokens)": jax.random.normal(key, (2, POS_L, text_dim), jnp.float32),
        "unconditional (512 rows)": jax.random.normal(key, (2, 64, text_dim), jnp.float32),
    }

    with nn_partitioning.axis_rules(()), mesh_ctx:
        model = WanModel(
            rngs=nnx.Rngs(jax.random.key(0)),
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=channels,
            out_channels=channels,
            text_dim=text_dim,
            freq_dim=16,
            ffn_dim=32,
            num_layers=1,
            attention="dot_product",
            rope_max_seq_len=64,
            scan_layers=False,
            dtype=jnp.bfloat16,
            weights_dtype=jnp.bfloat16,
        )

        def raw(latents, timestep_2d, context):  # a closure that does NOT cast
            return model(hidden_states=latents, timestep=timestep_2d, encoder_hidden_states=context)

        wrapped = casting_velocity_fn(raw, jnp.bfloat16)
        for label, context in contexts.items():
            cast = np.asarray(wrapped(z, timestep, context))
            reference = np.asarray(raw(z, timestep, context.astype(jnp.bfloat16)))
            uncast = np.asarray(raw(z, timestep, context))
            np.testing.assert_array_equal(cast, reference, err_msg=f"{label}: the wrapper must cast")
            assert not np.array_equal(cast, uncast), f"{label}: the cast must be observable"


def test_the_positive_shard_writer_publishes_a_readable_shard_marker_last(tmp_path):
    """**S4b, end to end on a real filesystem.** The fakes replace ``write_shard`` everywhere else, so
    the writer's happy path is exercised only here -- which is how a missing import in it survived
    every other test in this file until ruff caught it.

    Data before marker: the marker is written last, and everything it names is present and readable
    through the S5 codec when it appears.
    """
    from maxdiffusion.null_adapter_shards import HEADER_NAME, MARKER_NAME
    from maxdiffusion.pos_context_modes import pos_write_shard
    from maxdiffusion.pos_context_records import pos_header_from_json, pos_record_from_bytes

    fake = _Fake()
    batch, fields = fake.read_batch(_NAMES)
    result = run_pos_capacity_example_batch(fake.velocity_fn, batch, _BASE_CONTEXT, CapacityParams(inner_iters=1))
    header = _header(fake)
    records = build_pos_capacity_records(fake.velocity_fn, result, batch, _BASE_CONTEXT, header, fields, arm="b1")
    shard = str(tmp_path / "b1" / "shard_00000")

    marker = pos_write_shard(records, header, shard, str(tmp_path / "staging"))

    published = {path.name for path in (tmp_path / "b1" / "shard_00000").iterdir()}
    assert MARKER_NAME in published and HEADER_NAME in published
    assert published == {MARKER_NAME, HEADER_NAME, *marker.files.values()}
    assert marker.names == tuple(sorted(_NAMES)) and marker.header_fingerprint
    assert not (tmp_path / "staging").exists() or not any((tmp_path / "staging").iterdir())  # staging swept

    parsed_header = pos_header_from_json((tmp_path / "b1" / "shard_00000" / HEADER_NAME).read_text())
    assert parsed_header.l_pos == POS_L and parsed_header.embedding_slot == "positive"
    for name, filename in marker.files.items():
        record = pos_record_from_bytes((tmp_path / "b1" / "shard_00000" / filename).read_bytes())
        assert record.name == name and record.z_bar_states.shape == (25, *_LATENT)


# --------------------------------------------------------------------------------------------------
# 4. The writer preflight (S4 review, finding 2) -- the reviewer's three acceptance probes.
# --------------------------------------------------------------------------------------------------


def _arms_and_header(inner_iters=1, lr=0.01, guide_scale=5.0, l_pos=POS_L):
    fake = _Fake()
    batch, fields = fake.read_batch(_NAMES)
    result = run_pos_capacity_example_batch(
        fake.velocity_fn, batch, _BASE_CONTEXT,
        CapacityParams(inner_iters=inner_iters, lr=lr, guide_scale=guide_scale), l_pos=l_pos,
    )
    header = _header(
        fake,
        params={"guide_scale": guide_scale, "l_pos": l_pos},
        optimization_config={"inner_iters": inner_iters, "lr": lr},
    )
    return fake, batch, fields, result, header


def test_the_writer_refuses_a_header_that_misdeclares_the_recipe():
    """**Reviewer probe 1.** Embeddings made at w=5, J=1, lr=.01 were accepted under a header claiming
    w=7, J=50, lr=.03 -- and would then have *verified*, because a verifier checks a record against
    its own header."""
    fake, batch, fields, result, header = _arms_and_header()

    for mangled, message in (
        (dataclasses.replace(header, guide_scale=7.0), "guide_scale 7.0 does not match the run's 5.0"),
        (dataclasses.replace(header, optimization_config={"inner_iters": 50, "lr": 0.01}), "optimization_config"),
        (dataclasses.replace(header, optimization_config={"inner_iters": 1, "lr": 0.03}), "optimization_config"),
    ):
        with pytest.raises(ValueError, match=message):
            build_pos_capacity_records(fake.velocity_fn, result, batch, _BASE_CONTEXT, mangled, fields, arm="b1")


def test_the_writer_refuses_a_header_whose_l_pos_is_not_what_ran():
    """**Reviewer probe 2.** A header claiming ``l_pos=1`` was accepted while the record stored eight
    rows. The three numbers -- declared, produced, stored -- must be one number."""
    fake, batch, fields, result, header = _arms_and_header()

    with pytest.raises(ValueError, match="must equal the l_pos the arms ran at"):
        build_pos_capacity_records(
            fake.velocity_fn, result, batch, _BASE_CONTEXT, dataclasses.replace(header, l_pos=1), fields, arm="b1"
        )


def test_the_writer_refuses_a_batch_that_only_shares_the_names():
    """**Reviewer probe 3.** The same arm results paired with different ``z_i0``/``z_video`` were
    accepted, publishing someone else's tensors under this run's metrics."""
    fake, batch, fields, result, header = _arms_and_header()
    other, other_fields = _Fake().read_batch(("ep9_v0_s00009", "ep8_v0_s00008"))
    swapped = CapacityBatch(names=batch.names, z_i0=other.z_i0, z_video=other.z_video)

    with pytest.raises(ValueError, match="the same names carry different tensors"):
        build_pos_capacity_records(fake.velocity_fn, result, swapped, _BASE_CONTEXT, header, fields, arm="b1")

    renamed = CapacityBatch(names=other.names, z_i0=batch.z_i0, z_video=batch.z_video)
    with pytest.raises(ValueError, match="describe different examples"):
        build_pos_capacity_records(fake.velocity_fn, result, renamed, _BASE_CONTEXT, header, other_fields, arm="b1")


def test_the_writer_refuses_a_non_canonical_sigma_grid_and_incomplete_example_fields():
    fake, batch, fields, result, header = _arms_and_header()
    grid = np.asarray(canonical_sigmas(), np.float32).copy()
    grid[3] = 0.5

    with pytest.raises(ValueError, match="does not match the canonical grid"):
        build_pos_capacity_records(
            fake.velocity_fn, result, batch, _BASE_CONTEXT, dataclasses.replace(header, sigma_vector=grid),
            fields, arm="b1",
        )
    thin = {name: {k: v for k, v in row.items() if k != "actions"} for name, row in fields.items()}
    with pytest.raises(ValueError, match="example_fields must carry exactly"):
        build_pos_capacity_records(fake.velocity_fn, result, batch, _BASE_CONTEXT, header, thin, arm="b1")


# --------------------------------------------------------------------------------------------------
# 5. The adequacy probe, the L_pos ablation, and the selection artifact.
# --------------------------------------------------------------------------------------------------


def _adequacy_plan(**overrides):
    from maxdiffusion.null_adapter_runner_core import ADEQUACY_GRID

    names = tuple(f"ep{i}_v0_s0000{i}" for i in range(8))
    plan = {
        "mode": "adequacy_probe",
        "cohort": "dev64",
        "names": names,
        "grid": tuple(ADEQUACY_GRID),
        "params": {"inner_iters": 1, "lr": 0.01, "guide_scale": 5.0, "l_pos": POS_L},
        "ablation_l_pos": (1, 8),
    }
    return {**plan, **overrides}


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"cohort": "train2000"}, "adequacy probe is defined on"),
        ({"names": tuple(f"ep{i}" for i in range(3))}, "needs the first 8 DEV names"),
        ({"grid": ((10, 0.01), (11, 0.02))}, "the approved"),
    ],
)
def test_the_positive_adequacy_probe_preflights_before_reading_any_data(overrides, message):
    """The probe is a fixed experiment; a wrong request costs the same as a right one."""
    fake = _Fake()

    with pytest.raises(ValueError, match=message):
        run_pos_adequacy(_adequacy_plan(**overrides), fake.backend(), _Recorder().sinks(), artifact_dir="gs://a")
    assert fake.contexts == []  # nothing was read and no forward was made


def test_the_positive_adequacy_probe_persists_its_evidence_and_an_adoption():
    """Mirrors the null probe: per-example scores, the [N,J,B] traces, the [B,N] final losses, the
    per-cell wall time the re-run projection needs, and exp_04's adoption rule -- over the S2
    optimizer and the 8-token pivots."""
    from maxdiffusion.null_adapter_runner_core import ADEQUACY_GRID

    recorder = _Recorder()

    payload = run_pos_adequacy(_adequacy_plan(), _Fake().backend(), recorder.sinks(), artifact_dir="gs://a")

    assert payload["embedding_slot"] == "positive" and payload["l_pos"] == POS_L
    assert len(payload["scores"]) == len(ADEQUACY_GRID)
    for score in payload["scores"]:
        assert len(score["per_example"]) == 8 and np.isfinite(score["score"])
        assert np.asarray(score["tracking_losses"]).shape[0] == 25  # [N, J, B]
        assert np.asarray(score["final_losses"]).shape == (8, 25)  # [B, N]
        assert score["seconds"] >= 0.0
    assert {"inner_iters", "lr", "adopted", "projection_seconds_per_example"} <= set(payload["adopted"])
    assert recorder.json["adequacy_report.json"]["mode"] == "adequacy_probe"


def test_the_l_pos_ablation_is_diagnostic_only_and_never_reaches_publication():
    """Plan §4: ``L_pos in {1, 8}`` is a diagnostic. It reports at every row count while publication
    stays fixed at ``plan["params"]["l_pos"]`` -- and the writer's own preflight refuses anything else,
    so an ablation output cannot become a cached target even by mistake."""
    recorder = _Recorder()

    payload = run_pos_adequacy(_adequacy_plan(), _Fake().backend(), recorder.sinks(), artifact_dir="gs://a")

    ablation = payload["l_pos_ablation"]
    assert ablation["diagnostic_only"] and ablation["published_l_pos"] == POS_L
    assert sorted(ablation["cells"]) == ["1", "8"]
    for key, cell in ablation["cells"].items():
        assert cell["l_pos"] == int(key) and np.isfinite(cell["final_tracking_loss"])
    assert recorder.shards == []  # the probe publishes no records at all

    # ... and a one-row arm result cannot be written under the published header.
    fake, batch, fields, result, header = _arms_and_header(l_pos=1)
    with pytest.raises(ValueError, match="must equal the l_pos the arms ran at"):
        build_pos_capacity_records(
            fake.velocity_fn, result, batch, _BASE_CONTEXT, dataclasses.replace(header, l_pos=POS_L), fields, arm="b1"
        )


def test_the_selection_artifact_speaks_in_b_arms_and_h_gates():
    """**S4 review, finding 3.** S4 serialized ``target="A1/keyed"`` beside ``arm="b1"``."""
    recorder = _Recorder()

    _run_main(_Fake(), recorder, **_POSITIVE)

    selection = recorder.json["selection.json"]
    assert selection["embedding_slot"] == "positive" and selection["l_pos"] == POS_L
    assert selection["target"] in ("B1/keyed", "B2/global", "stop")
    assert selection["arm"] in ("b1", "b2", None) and selection["label"] in ("B1", "B2", None)
    # The gate *reasons* keep exp_04's wording: H1/H2 are its G1/G2 functions verbatim (plan §4-P1'),
    # so what has to be in this slot's names is the artifact's own structure.
    assert sorted(k for k in selection["gates"] if k != "selection") == ["h1", "h2"]
    assert not {selection["arm"], selection["label"]} & {"a1", "a2", "A1", "A2"}
    assert not selection["target"].startswith(("A1", "A2"))
    for key in ("cohort", "manifest_hash", "smoke_examples", "manifest", "reasons"):
        assert key in selection, key


@pytest.mark.parametrize(
    "mangle, message",
    [
        (lambda s: {**s, "embedding_slot": "null"}, "does not authorize a positive-slot cache"),
        (lambda s: {**s, "target": "stop", "arm": None}, "stopped after P1"),
        (lambda s: {**s, "label": "A1"}, "names an unusable arm"),
        (lambda s: {**s, "target": "B2/global"}, "disagrees with its arm"),
        (lambda s: {**s, "cohort": "train2000"}, "does not authorize a cache"),
        (lambda s: {**s, "manifest_hash": ""}, "carries no manifest binding"),
        (lambda s: {**s, "smoke_examples": 2}, "smoke run"),
    ],
)
def test_the_consuming_side_refuses_a_selection_that_does_not_authorize_this_job(mangle, message):
    from maxdiffusion.pos_context_modes import pos_selected_arm

    good = {
        "embedding_slot": "positive", "cohort": "dev64", "manifest_hash": "d" * 64, "smoke_examples": 0,
        "target": "B1/keyed", "arm": "b1", "label": "B1", "noise_convention": "keyed", "reasons": [],
    }

    assert pos_selected_arm(good) == "b1"
    with pytest.raises(ValueError, match=message):
        pos_selected_arm(mangle(good))


@pytest.mark.parametrize(
    "target, arm, label, convention",
    [
        (Target.A1_KEYED, "b1", "B1", "keyed"),
        (Target.A2_GLOBAL, "b2", "B2", "global"),
        (Target.STOP, None, None, None),
    ],
)
def test_the_selection_payload_maps_every_verdict_into_this_slot_s_names(target, arm, label, convention):
    """The serializer, exercised at each verdict directly -- the end-to-end fixture only ever reaches
    STOP, where exp_04's payload and this one agree, which is how an A-labelled target survived the
    first battery (S4 review, finding 3)."""
    from maxdiffusion.null_adapter_gates import GateVerdict, TargetSelection
    from maxdiffusion.pos_context_modes import pos_selection_payload, pos_selected_arm

    gate = GateVerdict(passed=True, reasons=("ok",), numbers={"n": 1.0})
    verdicts = {"g1": gate, "g2": gate, "selection": TargetSelection(target, ("because",), {"n": 2.0})}
    plan = {"cohort": "dev64", "names": list(_NAMES), "smoke_examples": 0, "params": {"l_pos": POS_L}}

    payload = pos_selection_payload(verdicts, plan, manifest_hash="d" * 64)

    assert payload["target"] == (f"{label}/{convention}" if arm else "stop")
    assert (payload["arm"], payload["label"], payload["noise_convention"]) == (arm, label, convention)
    assert payload["embedding_slot"] == "positive" and payload["l_pos"] == POS_L
    assert sorted(k for k in payload["gates"] if k != "selection") == ["h1", "h2"]
    if arm:
        assert pos_selected_arm(payload) == arm  # the round trip the consumer performs
    else:
        with pytest.raises(ValueError, match="stopped after P1"):
            pos_selected_arm(payload)


# --------------------------------------------------------------------------------------------------
# 6. The follow-up blockers: storage roots, slot-bound adoption, ablation binding.
# --------------------------------------------------------------------------------------------------


def test_a_positive_run_free_space_checks_and_sweeps_only_its_own_roots():
    """**Follow-up finding 1.** ``main`` ran exp_04's free-space check and staging sweep against the
    NULL roots whatever the slot: with the checked-in positive YAML (whose null roots are empty) that
    is ``os.statvfs("")`` before the plan exists, and the positive run's own staging is never swept."""
    observed = {"swept": [], "free_space": []}

    _run_main(_Fake(), _Recorder(), observed=observed, **_POSITIVE)

    assert observed["free_space"] == [_POSITIVE["pos_artifact_dir"]]
    assert observed["swept"] == [_POSITIVE["pos_staging_dir"]]
    null_roots = {_config().null_artifact_dir, _config().null_staging_dir}
    assert not (set(observed["swept"]) | set(observed["free_space"])) & null_roots


def test_the_checked_in_positive_yaml_reaches_main_without_touching_a_null_root():
    """The shipped config is the launch surface: its null roots are empty, so anything that resolved
    storage from them would die in ``os.statvfs('')`` before the plan existed."""
    import pathlib as _pathlib

    import yaml

    configs = _pathlib.Path(__file__).resolve().parents[2] / "configs"
    declared = yaml.safe_load((configs / "base_wan_5b_pos_inversion.yml").read_text())
    assert declared["null_artifact_dir"] == "" and declared["null_staging_dir"] == ""

    observed = {"swept": [], "free_space": []}
    overrides = {
        **{key: declared[key] for key in ("embedding_slot", "pos_L", "pos_ablation_L", "null_artifact_dir",
                                          "null_staging_dir")},
        "pos_artifact_dir": "gs://bucket/k1",
        "pos_staging_dir": "gs://bucket/k1-staging",
        "activations_dtype": "bfloat16",
        "pos_adequacy_uri": "",
    }

    assert _run_main(_Fake(), _Recorder(), observed=observed, **overrides) == 0
    assert observed["free_space"] == ["gs://bucket/k1"] and observed["swept"] == ["gs://bucket/k1-staging"]


@pytest.mark.parametrize(
    "pos_artifact, pos_staging, null_artifact, message",
    [
        ("gs://b/x/", "gs://b/s", "gs://b/x", "must not be the null slot"),  # trailing slash
        ("gs://b/x", "gs://b/s/", "gs://b/s", "must not be the null slot"),
        ("", "gs://b/s", "gs://b/n", "pos_artifact_dir"),
        ("gs://b/x", "", "gs://b/n", "pos_staging_dir"),
    ],
)
def test_positive_roots_are_normalized_before_they_are_compared(pos_artifact, pos_staging, null_artifact, message):
    """A trailing slash must not smuggle a positive root onto the null slot's tree."""
    from maxdiffusion.pos_context_modes import positive_roots

    config = _config(
        pos_artifact_dir=pos_artifact, pos_staging_dir=pos_staging,
        null_artifact_dir=null_artifact, null_staging_dir="gs://b/s",
    )

    with pytest.raises(ValueError, match=message):
        positive_roots(config)


def test_positive_roots_normalize_trailing_slashes_on_the_way_out():
    from maxdiffusion.pos_context_modes import positive_roots

    assert positive_roots(
        _config(pos_artifact_dir="gs://b/x/", pos_staging_dir="gs://b/s/", null_artifact_dir="", null_staging_dir="")
    ) == ("gs://b/x", "gs://b/s")


def _adequacy_artifact(**overrides):
    payload = {
        "mode": "adequacy_probe",
        "embedding_slot": "positive",
        "cohort": "dev64",
        "l_pos": POS_L,
        "guide_scale": 5.0,
        "manifest_hash": "d" * 64,
        "adopted": {"inner_iters": 25, "lr": 0.03, "adopted": True, "projection_seconds_per_example": 0.1},
    }
    return {**payload, **overrides}


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"embedding_slot": "null"}, "was produced in the 'null' slot"),
        ({"mode": "capacity"}, "not an adequacy probe"),
        ({"cohort": "train2000"}, "was probed on"),
        ({"l_pos": 1}, "different representation"),
        ({"guide_scale": 7.0}, "was probed at w="),
        ({"manifest_hash": "e" * 64}, "does not authorize this job"),
        ({"adopted": {"lr": 0.03}}, "no usable adoption block"),
    ],
)
def test_a_positive_run_refuses_an_adoption_artifact_that_is_not_its_own(overrides, message):
    """**Follow-up finding 2.** exp_04's ``load_adoption`` would consume a null-slot artifact sitting
    at the positive URI and re-run the B-arms at a recipe chosen for a different experiment."""
    from maxdiffusion.pos_context_modes import pos_adoption

    plan = {"cohort": "dev64", "params": {"l_pos": POS_L, "guide_scale": 5.0}}
    good = _adequacy_artifact()

    assert pos_adoption("gs://a/x.json", plan, exists=lambda u: True, read_json=lambda u: good,
                        manifest_hash="d" * 64)["inner_iters"] == 25
    assert pos_adoption("", plan, exists=lambda u: True, read_json=lambda u: good) is None  # no URI, no adoption
    with pytest.raises(ValueError, match=message):
        pos_adoption(
            "gs://a/x.json", plan, exists=lambda u: True, read_json=lambda u: _adequacy_artifact(**overrides),
            manifest_hash="d" * 64,
        )


def test_the_ablation_runs_at_the_adopted_recipe_and_the_selected_arm():
    """**Follow-up finding 3.** The diagnostic ran at the default recipe while K1 proceeds at whatever
    the probe adopted -- a comparison about a configuration the experiment is not using."""
    from maxdiffusion.null_adapter_runner_core import RecipeAdoption

    recorder = _Recorder()
    plan = _adequacy_plan(params={"inner_iters": 1, "lr": 0.01, "guide_scale": 5.0, "l_pos": POS_L})
    forced = RecipeAdoption(inner_iters=50, lr=0.03, adopted=True, plateau="recipe-limited", reasons=(), numbers={})

    with mock_adopt_recipe(forced):
        payload = run_pos_adequacy(plan, _Fake().backend(), recorder.sinks(), artifact_dir="gs://a", manifest_hash="d")

    ablation = payload["l_pos_ablation"]
    assert ablation["recipe"] == {"inner_iters": 50, "lr": 0.03}  # the ADOPTED recipe, not (1, 0.01)
    assert ablation["arm"] == "b1" and ablation["published_l_pos"] == POS_L
    assert payload["manifest_hash"] == "d"


@contextlib.contextmanager
def mock_adopt_recipe(adoption):
    """Force an adoption verdict, so the ablation's binding is observable independently of the grid."""
    from maxdiffusion import pos_context_modes

    original = pos_context_modes.adopt_recipe
    pos_context_modes.adopt_recipe = lambda report: adoption
    try:
        yield
    finally:
        pos_context_modes.adopt_recipe = original
