"""exp_04 R10 — the four J1 mode bodies, run end to end against fakes.

The model and the filesystem reach the mode bodies through two seams of callables, so everything the
J1 smoke rung will do can be done here first: a real ``run_capacity_example_batch`` over production
geometry with a toy oracle, a real decode/score/fill, real gate evaluation, real record building, and
fake sinks that capture what would have been published.

What carries the round, after the R10 strengthening:

- **Nothing becomes immutable before the run knows it succeeded.** A decode failure at the last batch
  must leave zero published shards, or the job is unre-runnable and the cohort is half-published
  forever.
- **Fill before gate, gates before publish.** R6 leaves ``future_ssim`` absent and the gates read
  absent as *invalid*, so the published tables are asserted to carry it -- and the gates are asserted
  to run against the **declared** cohort, so a quarantined example fails coverage instead of quietly
  shrinking the experiment.
- **Quarantine really is the seam.** A poisoned example makes R6 raise a plain ``ValueError`` about a
  non-finite trace; only because the divergence guard sits *inside* the quarantine call does that
  become one recorded gap instead of a dead job.
- **The arm is selected, never defaulted.** Cache and verify read it from the J1 artifact; a run that
  stopped after P1 has nothing to cache and says so.
- **"Nothing to check" is not a pass.** Empty inputs, unvalidated shards, missing names, duplicates
  and recorded quarantines are all exit 1.
"""

from __future__ import annotations

import json
import os
import tempfile
import types

import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion.null_adapter_gates import Target
from maxdiffusion.null_adapter_modes import (
    ADEQUACY_EXAMPLES,
    RERUN_BUDGET_SECONDS,
    Backend,
    Sinks,
    apply_adopted_recipe,
    decode_cohort,
    execute,
    header_for,
    merge_latents,
    merge_tables,
    natural_context_length,
    run_cache,
    run_capacity,
    selected_arm,
)
from maxdiffusion.null_adapter_records import PRODUCTION_GEOMETRY, make_record
from maxdiffusion.null_adapter_runner_core import (
    ADEQUACY_GRID,
    METHODS,
    PROBE_K_SET,
    CapacityBatch,
    run_capacity_example_batch as _RUNNER,
)
from maxdiffusion.null_adapter_shards import ResumePlan, ShardReport, header_fingerprint
from maxdiffusion.null_adapter_verify import canonical_sigmas


_L, _S, _D = 16, 512, 4096
_LATENT = PRODUCTION_GEOMETRY.z_video
_PIXEL_FRAMES, _H, _W = 33, 12, 20
_NAMES = ("ep1_v0_s00000", "ep2_v0_s00004")

_RNG = np.random.default_rng(20260805)
_BASE_CONTEXT = jnp.asarray(_RNG.standard_normal((_S, _D), dtype=np.float32) * 0.02)
_WEIGHTS = jnp.asarray(_RNG.standard_normal((_L, _D), dtype=np.float32) * 0.02)
_PATTERN = jnp.asarray(_RNG.standard_normal(_LATENT, dtype=np.float32) * 0.01)


class _Fake:
    """A toy backend: a coupled velocity oracle, a stand-in VAE, and a deterministic reader."""

    def __init__(self, poisoned=None, decode_fails_after=None):
        self.poisoned = poisoned
        self.decode_fails_after = decode_fails_after
        self.last_names: tuple[str, ...] = ()
        self.reads: list[tuple[str, ...]] = []
        self.decodes: list[int] = []

    def read_batch(self, names):
        names = tuple(names)
        self.last_names = names
        self.reads.append(names)
        rng = np.random.default_rng(abs(hash(names)) % (2**32))
        z_video = rng.standard_normal((len(names), *_LATENT), dtype=np.float32) * 0.5
        z_i0 = rng.standard_normal((len(names), 48, 1, 12, 20), dtype=np.float32) * 0.5
        batch = CapacityBatch(names=names, z_i0=jnp.asarray(z_i0), z_video=jnp.asarray(z_video))
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
        scale = jnp.sum(context[:, :_L] * _WEIGHTS, axis=(1, 2))
        value = scale[:, None, None, None, None] * _PATTERN + 0.5 * z + 1e-5 * jnp.mean(timestep_2d)
        if self.poisoned in self.last_names:
            row = self.last_names.index(self.poisoned)
            value = value.at[row].set(jnp.nan)  # one example's optimization diverges
        return value

    def decode_fn(self, latents):
        z = np.asarray(latents, np.float32)
        self.decodes.append(z.shape[0])
        if self.decode_fails_after is not None and len(self.decodes) > self.decode_fails_after:
            raise RuntimeError("the VAE ran out of memory")
        per_frame = z.mean(axis=1)
        frames = np.concatenate([per_frame[:, :1], np.repeat(per_frame[:, 1:], 4, axis=1)], axis=1)
        squashed = 0.5 + 0.4 * np.tanh(frames)
        return np.repeat(squashed[..., None], 3, axis=-1).astype(np.float32)

    def backend(self):
        return Backend(
            velocity_fn=self.velocity_fn,
            decode_fn=self.decode_fn,
            read_batch=self.read_batch,
            base_context=_BASE_CONTEXT,
            model_revision="Wan2.2-TI2V-5B@" + "a" * 40,
        )


class _Recorder:
    """Fake sinks that keep what would have been published."""

    def __init__(self, resume=None, selection=None, shard=None, reports=None, marker=None):
        self.marker = marker
        self.markers = []
        self.shards, self.json, self.videos = [], {}, []
        self.resume = resume
        self.resume_calls = []
        self.selection = selection
        self.shard = shard
        self.reports = reports or {}
        self.validated = []

    def sinks(self, **overrides):
        base = {
            "write_shard": self._write_shard,
            "write_json": self._write_json,
            "save_video": self._save_video,
            "read_shard": self._read_shard,
            "resume_plan": self._resume_plan,
            "validate_shard": self._validate_shard,
            "read_json": self._read_json,
            "read_marker": self._read_marker,
        }
        return Sinks(**{**base, **overrides})

    def _write_shard(self, records, header, shard_path, staging, *, quarantined=None):
        self.shards.append(
            {
                "path": shard_path,
                "names": [r.name for r in records],
                "quarantined": dict(quarantined or {}),
                "dtype": header.dtype_policy,
            }
        )
        return self.shards[-1]

    def _write_json(self, path, payload):
        self.json[path.rsplit("/", 1)[-1]] = json.loads(json.dumps(payload, default=str))
        return path

    def _save_video(self, frames, path, fps):
        self.videos.append((path, np.asarray(frames).shape, fps))
        return path

    def _read_shard(self, path):
        if self.shard is None:
            raise AssertionError("no shard reads in this mode")
        return self.shard

    def _resume_plan(self, manifest, shards, **kwargs):
        self.resume_calls.append((tuple(manifest), tuple(shards), kwargs))
        return self.resume or ResumePlan(todo=tuple(manifest), covered=(), quarantined={}, shards=())

    def _validate_shard(self, path, **kwargs):
        self.validated.append((path, kwargs))
        return self.reports.get(path, ShardReport(path, True, _NAMES, {}, ()))

    def _read_json(self, path):
        if self.selection is None:
            raise FileNotFoundError(path)
        return self.selection

    def _read_marker(self, path):
        self.markers.append(path)
        if self.marker is not None:
            return self.marker
        return types.SimpleNamespace(header_fingerprint=_DIGEST)


_DIGEST = "d" * 64


def _selection(target=Target.A1_KEYED.value, arm="a1", label="A1", **extra):
    """A well-formed J1 artifact: an arm plus the provenance that makes it authoritative."""
    return {
        "target": target,
        "arm": arm,
        "label": label,
        "reasons": [],
        "cohort": "dev64",
        "manifest_hash": _DIGEST,
        "smoke_examples": 0,
        **extra,
    }


def _plan(names=_NAMES, batches=None, mode="capacity", **extra):
    plan = {
        "mode": mode,
        "cohort": "dev64",
        "names": tuple(names),
        "batches": batches or (tuple(names),),
        "smoke_examples": 0,
        "params": {"inner_iters": 1, "lr": 0.01, "guide_scale": 5.0, "l_null": _L},
        "optimization_config": {"inner_iters": 1, "lr": 0.01},
        "noise_convention": "keyed",
        "latent_dtype": "fp32",
        "eval_seed": 2026,
        "decode_subset": tuple(names[:1]),
        "decode_batch_size": 8,
    }
    plan.update(extra)
    return plan


def _capacity(fake=None, recorder=None, plan=None, **kwargs):
    fake = fake or _Fake()
    recorder = recorder or _Recorder()
    report = run_capacity(
        plan or _plan(),
        fake.backend(),
        recorder.sinks(),
        artifact_dir="gs://bucket/exp04/run",
        staging_dir="gs://bucket/exp04/run/_staging",
        manifest_hash="m" * 64,
        code_sha="c" * 40,
        **kwargs,
    )
    return report, fake, recorder


def test_capacity_publishes_filled_tables_records_videos_and_a_selection():
    report, fake, recorder = _capacity()

    assert report["mode"] == "capacity" and report["examples"] == len(_NAMES) and report["quarantined"] == {}
    tables = recorder.json["gate_tables.json"]
    assert set(tables) == set(METHODS)
    for method, rows in tables.items():
        assert sorted(rows) == sorted(_NAMES)
        for entry in rows.values():
            for seed, metrics in entry.items():
                # The fill happened: absent SSIM is what the gates read as invalid.
                assert {"future_mse", "full_mse", "future_ssim", "full_ssim"} <= set(metrics), (method, seed)
        assert set(next(iter(rows.values()))) == (
            {str(k) for k in PROBE_K_SET} if method.endswith("_probe") else {"0"}
        )
    assert [shard["names"] for shard in recorder.shards] == [list(_NAMES), list(_NAMES)]  # A1 and A2
    assert [shard["path"].rsplit("/", 2)[-2] for shard in recorder.shards] == ["a1", "a2"]
    assert len(recorder.videos) == 1
    path, shape, fps = recorder.videos[0]
    assert shape == (_PIXEL_FRAMES, 2 * _H, _W, 3) and fps == 16  # ground truth stacked over prediction
    selection = recorder.json["selection.json"]
    assert selection["target"] in {t.value for t in Target} and selection["manifest"] == list(_NAMES)
    assert set(selection["gates"]) == {"g1", "g2", "selection"}


def test_capacity_gates_and_selects_before_it_publishes_anything():
    """The order is the point: a gate verdict computed after publication cannot stop a bad cache."""
    order = []
    fake = _Fake()
    recorder = _Recorder()
    sinks = recorder.sinks(
        write_shard=lambda *a, **k: order.append("shard"),
        write_json=lambda path, payload: order.append(path.rsplit("/", 1)[-1]) or path,
    )

    run_capacity(
        _plan(),
        fake.backend(),
        sinks,
        artifact_dir="gs://b/run",
        staging_dir="gs://b/run/_stage",
        manifest_hash="m" * 64,
        code_sha="c" * 40,
    )

    assert order.index("selection.json") < order.index("shard")
    assert order.index("gate_tables.json") < order.index("shard")


def test_a_decode_failure_leaves_no_published_shard():
    """R10's first cut published per batch inside the arm loop, so a late decode failure left
    completed immutable shards a retry could never overwrite -- the job became unre-runnable."""
    fake = _Fake(decode_fails_after=0)
    recorder = _Recorder()

    with pytest.raises(RuntimeError, match="out of memory"):
        _capacity(fake=fake, recorder=recorder)

    assert recorder.shards == [] and recorder.json == {}


def test_capacity_decodes_the_whole_cohort_in_bounded_batches():
    """§4-P1 asks for a full-cohort decode; that is coverage, not one B=64 VAE call."""
    fake = _Fake()
    report, _, recorder = _capacity(fake=fake, plan=_plan(batches=((_NAMES[0],), (_NAMES[1],))), decode_batch_size=1)

    assert report["examples"] == 2
    assert recorder.json["gate_tables.json"]["a1"].keys() == set(_NAMES)  # coverage is still the cohort
    assert max(fake.decodes) == 1  # ... and no single decode was ever wider than the declared bound

    wide = _Fake()
    _capacity(fake=wide, plan=_plan(batches=((_NAMES[0],), (_NAMES[1],))), decode_batch_size=8)
    assert max(wide.decodes) == 2  # the bound is what changed, not the coverage


def test_the_cohort_decode_covers_every_name_whatever_the_bound():
    fake = _Fake()
    batch, _ = fake.read_batch(_NAMES)
    latents = {m: np.zeros((len(_NAMES), *_LATENT), np.float32) for m in METHODS if not m.endswith("_probe")}
    latents.update({m: np.zeros((3, len(_NAMES), *_LATENT), np.float32) for m in METHODS if m.endswith("_probe")})

    for bound in (1, 2, 8):
        metrics = decode_cohort(fake.decode_fn, fake.read_batch, latents, _NAMES, batch_size=bound)
        assert set(metrics) == set(METHODS)
        assert all(sorted(rows) == sorted(_NAMES) for rows in metrics.values())


@pytest.mark.parametrize("bound", [0, -1, True, 2.0])
def test_an_unusable_decode_bound_is_refused(bound):
    fake = _Fake()
    with pytest.raises(ValueError, match="decode batch size"):
        decode_cohort(fake.decode_fn, fake.read_batch, {}, _NAMES, batch_size=bound)


def test_the_gates_are_evaluated_against_the_declared_cohort_not_the_surviving_one():
    """A quarantined example must cost the run its coverage, not silently shrink the experiment."""
    fake = _Fake(poisoned=_NAMES[1])

    report, _, recorder = _capacity(fake=fake)

    assert report["examples"] == 1 and list(report["quarantined"]) == [_NAMES[1]]
    assert recorder.json["selection.json"]["manifest"] == list(_NAMES)  # both names, not one
    coverage = recorder.json["selection.json"]["gates"]["g1"]
    assert coverage["numbers"]["missing_names"]["method"] == [_NAMES[1]]
    assert not coverage["passed"] and "coverage" in coverage["reasons"]


def test_a_diverging_example_is_quarantined_and_the_rest_still_publish():
    fake = _Fake(poisoned=_NAMES[1])

    report, _, recorder = _capacity(fake=fake)

    assert list(report["quarantined"]) == [_NAMES[1]]
    assert "must be finite" in report["quarantined"][_NAMES[1]]
    assert [shard["names"] for shard in recorder.shards] == [[_NAMES[0]], [_NAMES[0]]]
    assert all(shard["quarantined"] == {_NAMES[1]: report["quarantined"][_NAMES[1]]} for shard in recorder.shards)
    assert sorted(recorder.json["gate_tables.json"]["a1"]) == [_NAMES[0]]


def test_a_cohort_that_entirely_diverges_stops_the_run():
    fake = _Fake(poisoned=_NAMES[0])

    with pytest.raises(Exception) as caught:  # noqa: PT011 -- the policy re-raises the batch error
        _capacity(fake=fake, plan=_plan(names=(_NAMES[0],), batches=((_NAMES[0],),)))
    assert "finite" in str(caught.value) or "nothing to gate" in str(caught.value)


def test_shard_identities_do_not_collide_across_arms():
    """A1's and A2's shards are separate immutable paths; sharing a counter made them A1/00000 and
    A2/00001, so a resume comparing indices across arms saw a cohort with a hole."""
    _, _, recorder = _capacity(plan=_plan(batches=((_NAMES[0],), (_NAMES[1],))))

    paths = [shard["path"] for shard in recorder.shards]
    assert len(set(paths)) == len(paths)
    assert sorted(p.rsplit("/", 1)[-1] for p in paths if "/a1/" in p) == ["shard_00000", "shard_00001"]
    assert sorted(p.rsplit("/", 1)[-1] for p in paths if "/a2/" in p) == ["shard_00000", "shard_00001"]


def test_the_header_carries_the_r6_contract():
    fake = _Fake()

    header = header_for(_plan(), fake.backend(), manifest_hash="m" * 64, code_sha="c" * 40)

    assert sorted(header.optimization_config) == ["inner_iters", "lr"]
    assert header.guide_scale == 5.0 and header.l_null == _L and header.dtype_policy == "fp32"
    assert header.model_revision.endswith("a" * 40)
    np.testing.assert_array_equal(np.asarray(header.sigma_vector), canonical_sigmas())


def test_merging_refuses_to_double_count_a_name():
    table = {method: {"ep1_v0_s00000": {"0": {"future_mse": 1.0}}} for method in METHODS}

    with pytest.raises(ValueError, match="appears in two batches"):
        merge_tables([table, table])


def test_merging_latents_stacks_probes_on_the_batch_axis():
    class _Stub:
        final_latents = {
            **{m: np.zeros((2, 48, 9, 12, 20), np.float32) for m in METHODS if not m.endswith("_probe")},
            **{m: np.zeros((3, 2, 48, 9, 12, 20), np.float32) for m in METHODS if m.endswith("_probe")},
        }

    merged = merge_latents([_Stub(), _Stub()])

    assert merged["a1"].shape == (4, 48, 9, 12, 20)
    assert merged["a1_probe"].shape == (3, 4, 48, 9, 12, 20)


# --------------------------------------------------------------------------- the adopted-recipe seam


def test_an_adopted_recipe_reaches_both_the_arms_and_the_header():
    """Half of this is the trap: records built at the adopted recipe under a header advertising the
    default are refused by ``build_capacity_records``, so the plan and the header move together."""
    adopted = {"inner_iters": 25, "lr": 0.03, "adopted": True, "projection_seconds_per_example": 1.0}

    updated = apply_adopted_recipe(_plan(), adopted)

    assert updated["params"]["inner_iters"] == 25 and updated["params"]["lr"] == 0.03
    assert updated["optimization_config"] == {"inner_iters": 25, "lr": 0.03}
    assert updated["params"]["guide_scale"] == 5.0  # untouched: only J and lr are adopted


def test_a_recipe_that_did_not_qualify_leaves_the_plan_alone():
    updated = apply_adopted_recipe(_plan(), {"inner_iters": 50, "lr": 0.03, "adopted": False})

    assert updated["optimization_config"] == {"inner_iters": 1, "lr": 0.01}


def test_a_rerun_over_the_two_hour_budget_stops_instead_of_running():
    over = RERUN_BUDGET_SECONDS / (len(_NAMES) * 2) + 1.0
    adopted = {"inner_iters": 50, "lr": 0.03, "adopted": True, "projection_seconds_per_example": over}

    with pytest.raises(RuntimeError, match="over the 2 h budget"):
        apply_adopted_recipe(_plan(), adopted)


def test_a_rerun_inside_the_budget_is_allowed_and_recorded():
    under = RERUN_BUDGET_SECONDS / (len(_NAMES) * 2) - 1.0
    adopted = {"inner_iters": 25, "lr": 0.01, "adopted": True, "projection_seconds_per_example": under}

    updated = apply_adopted_recipe(_plan(), adopted)

    assert updated["adopted_recipe"]["projected_seconds"] <= RERUN_BUDGET_SECONDS


def test_the_natural_context_length_is_the_last_non_padding_row():
    context = np.zeros((_S, _D), np.float32)
    context[:5] = 0.3

    assert natural_context_length(context) == 5
    with pytest.raises(ValueError, match="entirely zero"):
        natural_context_length(np.zeros((_S, _D), np.float32))


# --------------------------------------------------------------------------- adequacy


def _adequacy_plan(**overrides):
    names = tuple(overrides.pop("names", _dev_names()))
    fields = {
        "names": names,
        "batches": (names,),
        "mode": "adequacy_probe",
        "decode_subset": names,
        "grid": ADEQUACY_GRID,
        "params": {"inner_iters": 1, "lr": 0.01, "guide_scale": 5.0, "l_null": _L},
    }
    return _plan(**{**fields, **overrides})


def _dev_names(count=ADEQUACY_EXAMPLES):
    return tuple(f"ep{index}_v0_s00000" for index in range(count))


def test_adequacy_persists_every_trace_the_probe_produced():
    fake, recorder = _Fake(), _Recorder()
    names = _dev_names()

    report, code = execute(
        "adequacy_probe",
        _adequacy_plan(names=names),
        fake.backend(),
        recorder.sinks(),
        artifact_dir="gs://bucket/probe",
    )

    assert code == 0 and report["names"] == list(names)
    assert [(cell["inner_iters"], cell["lr"]) for cell in report["scores"]] == list(ADEQUACY_GRID)
    for cell in report["scores"]:
        assert len(cell["per_example"]) == len(names)
        # [N, J, B] optimizer traces and the [B, N] post-inner-loop losses -- the R6 evidence contract.
        assert np.asarray(cell["tracking_losses"]).shape == (25, cell["inner_iters"], len(names))
        assert np.asarray(cell["grad_norms"]).shape == (25, cell["inner_iters"], len(names))
        assert np.asarray(cell["final_losses"]).shape == (len(names), 25)
        assert cell["seconds"] >= 0.0
    assert report["adopted"]["projection_seconds_per_example"] >= 0.0
    assert set(report["numbers"]) >= {"threshold", "default_score", "adopted_score", "scores"}
    assert report["l_null_ablation"]["diagnostic_only"] is True
    assert recorder.json["adequacy_report.json"]["names"] == list(names)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"cohort": "test64"}, "defined on dev64"),
        ({"names": _dev_names(3)}, "first 8 DEV names"),
        ({"grid": ((10, 0.01), (25, 0.03))}, "approved"),
        ({"grid": tuple((j, lr) for j, lr in ADEQUACY_GRID if j != 10)}, "approved"),
        ({"grid": (*ADEQUACY_GRID, (5, 0.01))}, "approved"),
    ],
)
def test_adequacy_preflights_the_experiment_before_reading_any_data(overrides, message):
    """A probe on three arbitrary examples and an arbitrary grid costs the same as the right one."""
    fake = _Fake()

    with pytest.raises(ValueError, match=message):
        execute(
            "adequacy_probe",
            _adequacy_plan(**overrides),
            fake.backend(),
            _Recorder().sinks(),
            artifact_dir="gs://bucket/probe",
        )
    assert fake.reads == []  # ... and it costs nothing


# --------------------------------------------------------------------------- cache


def _fake_fidelity(passing=True):
    def fidelity(backend, names, params, header, *, arm):
        fp32 = {name: {"future_ssim": 0.9, "future_mse": 1.0} for name in names}
        drop = 0.001 if passing else 0.5
        fp16 = {name: {"future_ssim": 0.9 - drop, "future_mse": 1.0} for name in names}
        return fp32, fp16

    return fidelity


def _cache(recorder, plan=None, names=None, **kwargs):
    fake = _Fake()
    names = names or _dev_names()
    return (
        run_cache(
            plan or _plan(names=names, batches=(names,), mode="cache"),
            fake.backend(),
            recorder.sinks(),
            artifact_dir="gs://bucket/cache",
            staging_dir="gs://bucket/cache/_staging",
            manifest_hash="m" * 64,
            code_sha="c" * 40,
            dev_manifest=names,
            fidelity=kwargs.pop("fidelity", _fake_fidelity()),
            **kwargs,
        ),
        fake,
    )


def test_cache_takes_its_arm_from_the_j1_selection_artifact():
    names = _dev_names()
    recorder = _Recorder(
        selection=_selection(target=Target.A2_GLOBAL.value, arm="a2", label="A2"),
        resume=ResumePlan(todo=(), covered=names, quarantined={}, shards=()),
    )

    report, _ = _cache(recorder, names=names)

    assert report["arm"] == "A2" and report["noise_convention"] == "global"
    _, _, kwargs = recorder.resume_calls[0]
    assert kwargs["expected_arm"] == "A2" and kwargs["expected_noise_convention"] == "global"


def test_cache_refuses_to_run_when_j1_stopped_after_p1():
    recorder = _Recorder(selection={"target": Target.STOP.value, "arm": None, "reasons": ["G1 failed"]})

    with pytest.raises(ValueError, match="no selected arm"):
        _cache(recorder)


def test_cache_refuses_to_run_without_a_selection_artifact():
    with pytest.raises(FileNotFoundError):
        _cache(_Recorder())


def test_the_fidelity_verdict_decides_the_dtype_before_anything_is_written():
    names = _dev_names()
    recorder = _Recorder(selection=_selection())

    report, _ = _cache(recorder, names=names, fidelity=_fake_fidelity(passing=False))

    assert report["fidelity"]["passed"] is False and report["latent_dtype"] == "fp32"
    # The published records carry the verdict's dtype, not the plan's declared one.
    assert {shard["dtype"] for shard in recorder.shards} == {"fp32"}


def test_a_passing_fidelity_gate_keeps_the_cache_at_fp16():
    names = _dev_names()
    recorder = _Recorder(selection=_selection())

    report, _ = _cache(recorder, names=names)

    assert report["fidelity"]["passed"] and report["latent_dtype"] == "fp16"
    assert {shard["dtype"] for shard in recorder.shards} == {"fp16"}


def test_the_fidelity_probe_runs_before_the_resume_and_on_the_first_eight_dev_names():
    seen = {}
    names = _dev_names(10)

    def fidelity(backend, probe_names, params, header, *, arm):
        seen["names"] = tuple(probe_names)
        seen["before_resume"] = not recorder.resume_calls
        return _fake_fidelity()(backend, probe_names, params, header, arm=arm)

    recorder = _Recorder(selection=_selection(), resume=ResumePlan(todo=(), covered=names, quarantined={}, shards=()))
    _cache(recorder, names=names, fidelity=fidelity)

    assert seen["names"] == names[:8] and seen["before_resume"]


def test_cache_resumes_against_the_canonical_full_header_fingerprint():
    """``base_context_fingerprint`` is 64 hex too, so the wrong one passes every type check and then
    rejects every valid shard as another run's."""
    names = _dev_names()
    recorder = _Recorder(selection=_selection(), resume=ResumePlan(todo=(), covered=names, quarantined={}, shards=()))

    _cache(recorder, names=names)

    _, _, kwargs = recorder.resume_calls[0]
    fake = _Fake()
    header = header_for(
        {**_plan(names=names), "latent_dtype": "fp16"}, fake.backend(), manifest_hash="m" * 64, code_sha="c" * 40
    )
    assert kwargs["expected_header_fingerprint"] == header_fingerprint(header)
    assert kwargs["expected_header_fingerprint"] != header.base_context_fingerprint


def test_cache_skips_what_a_validated_shard_already_covers():
    names = _dev_names()
    covered = ShardReport(path="gs://b/shard_00000", valid=True, names=names[:-1], quarantined={}, reasons=())
    recorder = _Recorder(
        selection=_selection(),
        resume=ResumePlan(todo=(names[-1],), covered=names[:-1], quarantined={}, shards=(covered,)),
    )

    report, _ = _cache(recorder, names=names, existing_shards=("gs://bucket/cache/shard_00000",))

    assert report["already_covered"] == len(names) - 1 and report["todo"] == [names[-1]]
    assert [shard["names"] for shard in recorder.shards] == [[names[-1]]]
    # ... and the new shard does not reuse the identity the covered one already owns.
    assert recorder.shards[0]["path"].endswith("shard_00001")


def test_a_retry_that_succeeded_is_not_still_counted_as_a_gap():
    names = _dev_names()
    recorder = _Recorder(
        selection=_selection(),
        resume=ResumePlan(
            todo=(names[-1],), covered=names[:-1], quarantined={names[-1]: "ExampleDivergenceError: boom"}, shards=()
        ),
    )

    report, _ = _cache(recorder, names=names)

    assert report["quarantined"] == {} and report["superseded"] == [names[-1]]
    assert report["written"] == 1


def test_cache_executes_only_the_selected_arm():
    """One arm's records, and -- the point -- one arm's compute: the other five are not run."""
    names = _dev_names()
    recorder = _Recorder(selection=_selection(target=Target.A2_GLOBAL.value, arm="a2", label="A2"))
    seen = {}

    def spy_arms(velocity_fn, batch, base_context, params, *, arms=None):
        seen.setdefault("arms", []).append(arms)
        return _RUNNER(velocity_fn, batch, base_context, params, arms=arms)

    import maxdiffusion.null_adapter_modes as modes

    original = modes.run_capacity_example_batch
    modes.run_capacity_example_batch = spy_arms
    try:
        report, _ = _cache(recorder, names=names)
    finally:
        modes.run_capacity_example_batch = original

    assert report["arm"] == "A2"
    assert seen["arms"] and all(arms == ("a2",) for arms in seen["arms"])
    # One flat shard per batch -- not the capacity mode's per-arm a1/ and a2/ subdirectories.
    assert len(recorder.shards) == 1
    assert recorder.shards[0]["path"].rsplit("/", 2)[-2] == "cache"


# --------------------------------------------------------------------------- verify


def _record(name, arm="A1", convention="keyed"):
    shapes = PRODUCTION_GEOMETRY.shapes()
    arrays = {field: np.zeros(shape, np.float32) for field, shape in shapes.items()}
    return make_record(
        name=name,
        ordinal=0,
        split="dev64",
        episode="1",
        latent_dtype="fp32",
        noise_convention=convention,
        arm=arm,
        final_future_mse=0.5,
        **arrays,
    )


def _verify(recorder, plan=None, backend=None, shard_paths=("gs://b/s0",), **kwargs):
    fake = _Fake()
    return execute(
        "verify_replay",
        plan or _plan(mode="verify_replay"),
        backend or fake.backend(),
        recorder.sinks(),
        artifact_dir="gs://bucket/cache",
        shard_paths=shard_paths,
        atol=1e-2,
        **kwargs,
    )


def test_verify_replay_exits_non_zero_when_a_record_fails():
    """A cache that cannot reproduce itself must not pass quietly in a pipeline."""
    fake = _Fake()
    header = header_for(_plan(), fake.backend(), manifest_hash="m" * 64, code_sha="c" * 40)
    records = (_record(_NAMES[0], arm="A2"), _record(_NAMES[1]))  # the first lies about its arm
    recorder = _Recorder(selection=_selection(), shard=(header, records))

    def forbidden(*args, **kwargs):
        raise AssertionError("provenance must be judged before any model call")

    backend = Backend(
        velocity_fn=forbidden,
        decode_fn=fake.decode_fn,
        read_batch=fake.read_batch,
        base_context=_BASE_CONTEXT,
        model_revision=fake.backend().model_revision,
    )

    report, code = _verify(recorder, backend=backend)

    assert code == 1
    assert _NAMES[0] in report["failures"]
    assert "record arm" in report["verdicts"][_NAMES[0]]
    assert recorder.json["verify_report.json"]["records"] == 2


def test_verify_validates_every_shard_before_reading_a_record():
    fake = _Fake()
    header = header_for(_plan(), fake.backend(), manifest_hash="m" * 64, code_sha="c" * 40)
    broken = ShardReport("gs://b/s0", False, (), {}, ("record sha256 does not match the marker",))
    recorder = _Recorder(selection=_selection(), shard=(header, ()), reports={"gs://b/s0": broken})

    report, code = _verify(recorder)

    assert code == 1
    assert any("record sha256" in failure for failure in report["failures"])
    assert report["invalid_shards"] == ["gs://b/s0"]
    # The expectations a validated shard is judged against come from the selection, not a default.
    _, kwargs = recorder.validated[0]
    assert kwargs["expected_arm"] == "A1" and kwargs["expected_noise_convention"] == "keyed"
    assert len(kwargs["expected_header_fingerprint"]) == 64


def test_verify_refuses_to_certify_nothing():
    """R10's own test asserted that zero records exit 0 -- an empty cache is not a verified one."""
    recorder = _Recorder(selection=_selection(), shard=None)

    report, code = _verify(recorder, shard_paths=())

    assert code == 1 and report["records"] == 0
    assert any("nothing to verify" in failure for failure in report["failures"])


def test_verify_requires_exact_cohort_coverage():
    fake = _Fake()
    header = header_for(_plan(), fake.backend(), manifest_hash="m" * 64, code_sha="c" * 40)
    recorder = _Recorder(
        selection=_selection(),
        shard=(header, (_record(_NAMES[0]),)),
        reports={"gs://b/s0": ShardReport("gs://b/s0", True, (_NAMES[0],), {}, ())},
    )

    report, code = _verify(recorder)

    assert code == 1 and report["missing"] == [_NAMES[1]]
    assert f"missing {_NAMES[1]}" in report["failures"]


def test_verify_treats_a_recorded_quarantine_as_a_failure():
    fake = _Fake()
    header = header_for(_plan(), fake.backend(), manifest_hash="m" * 64, code_sha="c" * 40)
    recorder = _Recorder(
        selection=_selection(),
        shard=(header, (_record(_NAMES[0]),)),
        reports={"gs://b/s0": ShardReport("gs://b/s0", True, (_NAMES[0],), {_NAMES[1]: "boom"}, ())},
    )

    report, code = _verify(recorder)

    assert code == 1
    assert any(failure.startswith(f"quarantined {_NAMES[1]}") for failure in report["failures"])


def test_verify_refuses_a_duplicate_name_across_shards():
    fake = _Fake()
    header = header_for(_plan(), fake.backend(), manifest_hash="m" * 64, code_sha="c" * 40)
    records = (_record(_NAMES[0]), _record(_NAMES[1]))
    recorder = _Recorder(selection=_selection(), shard=(header, records))

    report, code = _verify(recorder, shard_paths=("gs://b/s0", "gs://b/s1"))

    assert code == 1
    assert any(failure.startswith("duplicate ") for failure in report["failures"])


def _reproducing_backend():
    """A backend whose replay really does land on the all-zero ``expected_final_latent``.

    ``replay_with_nulls`` advances by the velocity, so a zero velocity leaves a zero ``z_start`` at
    zero -- which is exactly what ``_record`` stores. Anything less than a genuine replay here and
    the exit-zero path would be asserting nothing.
    """
    fake = _Fake()
    return Backend(
        velocity_fn=lambda z, timestep_2d, context: jnp.zeros_like(z),
        decode_fn=fake.decode_fn,
        read_batch=fake.read_batch,
        base_context=_BASE_CONTEXT,
        model_revision=fake.backend().model_revision,
    )


def test_verify_exits_zero_only_when_the_whole_cohort_reproduces():
    backend = _reproducing_backend()
    header = header_for(_plan(), backend, manifest_hash="m" * 64, code_sha="c" * 40)
    records = tuple(_record(name) for name in _NAMES)
    recorder = _Recorder(selection=_selection(), shard=(header, records))

    report, code = _verify(recorder, backend=backend)

    assert code == 0 and report["failures"] == [] and report["records"] == len(_NAMES)
    assert report["missing"] == [] and report["verdicts"] == dict.fromkeys(_NAMES, "ok")


def test_a_record_whose_replay_drifts_is_a_failure():
    """The same cohort, the same provenance -- only the dynamics differ, and that is enough."""
    drifting = _Fake().backend()
    header = header_for(_plan(), drifting, manifest_hash="m" * 64, code_sha="c" * 40)
    recorder = _Recorder(selection=_selection(), shard=(header, tuple(_record(name) for name in _NAMES)))

    report, code = _verify(recorder, backend=drifting)

    assert code == 1 and sorted(report["failures"]) == sorted(_NAMES)
    assert all("does not reproduce" in verdict for verdict in report["verdicts"].values())


@pytest.mark.parametrize(
    "selection, message",
    [
        ({"target": "A1/keyed", "arm": "a3", "label": "A3"}, "unusable arm"),
        ({"target": "A1/keyed", "arm": "a2", "label": "A2"}, "disagrees with its arm"),
        ({"target": "A2/global", "arm": "a2", "label": "A1"}, "unusable arm"),
        ("not-an-object", "must be a JSON object"),
    ],
)
def test_a_malformed_selection_artifact_is_refused(selection, message):
    with pytest.raises(ValueError, match=message):
        selected_arm(selection)


def test_execute_refuses_an_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        execute("rollout", _plan(), _Fake().backend(), _Recorder().sinks(), artifact_dir="gs://b")


# --------------------------------------------------------------------------- the video sink


class _FakeGfile:
    """Just enough of ``tf.io.gfile`` to watch a transactional upload happen."""

    def __init__(self):
        self.files = {}
        self.order = []

    def makedirs(self, path):
        self.order.append(("makedirs", path))

    def exists(self, path):
        return path in self.files

    def remove(self, path):
        self.order.append(("remove", path))
        self.files.pop(path, None)

    def copy(self, source, destination, overwrite=False):
        with open(source, "rb") as handle:
            self.files[destination] = handle.read()
        self.order.append(("copy", destination))

    def rename(self, source, destination, overwrite=False):
        self.files[destination] = self.files.pop(source)
        self.order.append(("rename", destination))


def _frames(count=4, size=8):
    return np.linspace(0.0, 1.0, count * size * size * 3, dtype=np.float32).reshape(count, size, size, 3)


def test_a_local_video_path_is_written_directly(tmp_path):
    from maxdiffusion.null_adapter_modes import publish_video

    produced = publish_video(_frames(), str(tmp_path / "clip.mp4"), fps=4)

    assert produced.startswith(str(tmp_path))
    assert os.path.exists(produced)


def test_a_gs_video_path_is_encoded_locally_then_uploaded(tmp_path):
    """``save_video_mp4`` is an ``os.makedirs``/``os.replace`` writer: handed a ``gs://`` path it
    created a local directory literally named ``gs:`` and published nothing (review, finding 8)."""
    from maxdiffusion.null_adapter_modes import publish_video

    gfile = _FakeGfile()
    staged = []

    class _Workdir:
        def __enter__(self):
            self.dir = tempfile.mkdtemp(dir=str(tmp_path))
            return self.dir

        def __exit__(self, *args):
            staged.append(sorted(os.listdir(self.dir)))
            return False

    produced = publish_video(_frames(), "gs://bucket/run/videos/00.mp4", fps=4, gfile=gfile, workdir=_Workdir)

    assert produced == "gs://bucket/run/videos/00.mp4"
    assert produced in gfile.files and gfile.files[produced]
    # Nothing partial was ever visible at the published path.
    assert [kind for kind, _ in gfile.order if kind in ("copy", "rename")] == ["copy", "rename"]
    copied = [path for kind, path in gfile.order if kind == "copy"][0]
    assert copied.endswith(".partial") and copied != produced
    assert not os.path.exists("gs:")  # the bug this replaces
    assert staged and staged[0]  # the encode really happened locally, in the working directory


def test_the_png_fallback_is_published_too(tmp_path, monkeypatch):
    """The fallback is the path taken on a host without ffmpeg -- the host most likely to need it."""
    import maxdiffusion.null_adapter_modes as modes

    frames_dir = tmp_path / "local_frames"
    frames_dir.mkdir()
    for index in range(3):
        (frames_dir / f"frame_{index:04d}.png").write_bytes(b"png" + bytes([index]))
    monkeypatch.setattr(modes, "save_video_mp4", lambda frames, path, fps: str(frames_dir))
    gfile = _FakeGfile()

    produced = modes.publish_video(_frames(), "gs://bucket/run/videos/00.mp4", gfile=gfile)

    assert produced == "gs://bucket/run/videos/00_frames"
    assert sorted(gfile.files) == [f"gs://bucket/run/videos/00_frames/frame_{i:04d}.png" for i in range(3)]


def test_an_upload_replaces_a_stale_staged_file(tmp_path):
    from maxdiffusion.null_adapter_modes import upload_artifact

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fresh")
    gfile = _FakeGfile()
    gfile.files["gs://b/videos/00.mp4.partial"] = b"stale"

    upload_artifact(str(source), "gs://b/videos/00.mp4", gfile=gfile)

    assert gfile.files["gs://b/videos/00.mp4"] == b"fresh"
    assert ("remove", "gs://b/videos/00.mp4.partial") in gfile.order


def test_the_default_sinks_route_videos_through_the_publishing_writer():
    from maxdiffusion.null_adapter_modes import default_sinks

    sinks = default_sinks()

    # The production sink calls ``publish_video``, not ``save_video_mp4``: that is the whole finding.
    assert "publish_video" in sinks.save_video.__code__.co_names
    assert "save_video_mp4" not in sinks.save_video.__code__.co_names
    # ... and the two seams verification needs are wired rather than left to a default.
    assert sinks.validate_shard is not None and sinks.read_json is not None


def test_the_fidelity_probe_reads_dev_even_when_the_cohort_is_not_dev():
    """J2 caches TRAIN-2000, and the fp16 decision is still the plan's first eight **DEV** examples.

    While the cache cohort happens to be DEV the two lists coincide, so ``plan["names"]`` looks
    correct and is not; the mutant that swaps them survives every same-cohort test.
    """
    train = tuple(f"tr{index}_v0_s00000" for index in range(12))
    dev = _dev_names(10)
    seen = {}

    def fidelity(backend, probe_names, params, header, *, arm):
        seen["names"] = tuple(probe_names)
        return _fake_fidelity()(backend, probe_names, params, header, arm=arm)

    recorder = _Recorder(
        selection=_selection(), resume=ResumePlan(todo=(), covered=train, quarantined={}, shards=())
    )
    run_cache(
        _plan(names=train, batches=(train,), mode="cache", cohort="train2000"),
        _Fake().backend(),
        recorder.sinks(),
        artifact_dir="gs://bucket/cache",
        staging_dir="gs://bucket/cache/_staging",
        manifest_hash="m" * 64,
        code_sha="c" * 40,
        dev_manifest=dev,
        fidelity=fidelity,
    )

    assert seen["names"] == dev[:8]
    assert not any(name.startswith("tr") for name in seen["names"])


def test_records_inside_an_invalid_shard_are_never_verified():
    """R8's validated-shard boundary is the whole point: a replacement record beside a stale marker
    must not be able to certify itself just because it happens to replay correctly."""
    backend = _reproducing_backend()
    header = header_for(_plan(), backend, manifest_hash="m" * 64, code_sha="c" * 40)
    records = tuple(_record(name) for name in _NAMES)  # these WOULD verify, if anything read them
    broken = ShardReport("gs://b/s0", False, (), {}, ("record sha256 does not match the marker",))
    recorder = _Recorder(selection=_selection(), shard=(header, records), reports={"gs://b/s0": broken})

    report, code = _verify(recorder, backend=backend)

    assert code == 1
    assert report["records"] == 0 and report["verdicts"] == {}
    assert report["missing"] == list(_NAMES)  # an unvalidated shard covers nothing


def test_an_unreadable_first_marker_is_a_verdict_rather_than_a_crash():
    def explode(path):
        raise OSError("the object store returned 503")

    recorder = _Recorder(selection=_selection())
    report, code = execute(
        "verify_replay",
        _plan(mode="verify_replay"),
        _Fake().backend(),
        recorder.sinks(read_marker=explode),
        artifact_dir="gs://bucket/cache",
        shard_paths=("gs://b/s0",),
        atol=1e-2,
    )

    assert code == 1 and any("marker unreadable" in failure for failure in report["failures"])


def test_no_shard_is_read_before_it_has_been_validated():
    """R8's boundary: bootstrapping the expected fingerprint from ``read_shard`` crossed it for shard
    zero -- the one shard whose records were read before anything had judged the shard."""
    backend = _reproducing_backend()
    reads = []

    def forbidden_read(path):
        reads.append(path)
        raise AssertionError(f"{path} was read before it was validated")

    recorder = _Recorder(selection=_selection())
    recorder.reports["gs://b/s0"] = ShardReport("gs://b/s0", False, (), {}, ("record sha256 mismatch",))

    report, code = execute(
        "verify_replay",
        _plan(mode="verify_replay"),
        backend,
        recorder.sinks(read_shard=forbidden_read),
        artifact_dir="gs://bucket/cache",
        shard_paths=("gs://b/s0",),
        atol=1e-2,
    )

    assert reads == []  # the marker carried the expectation; no record bytes were touched
    assert code == 1 and report["records"] == 0
    assert recorder.markers == ["gs://b/s0"]


# --------------------------------------------------------------------------- selection provenance


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"cohort": "train2000"}, "does not authorize a cache"),
        ({"manifest_hash": ""}, "carries no manifest binding"),
        ({"manifest_hash": None}, "carries no manifest binding"),
        ({"manifest_hash": "f" * 64}, "different dev64 manifest"),
        ({"smoke_examples": 2}, "2-example smoke run"),
    ],
)
def test_a_selection_that_does_not_bind_to_this_job_is_refused(overrides, message):
    """The artifact is the authority to cache two thousand examples; a file that says "A1" is not."""
    with pytest.raises(ValueError, match=message):
        selected_arm(_selection(**overrides), expected_manifest_hash=_DIGEST)


def test_a_bound_non_smoke_selection_is_accepted():
    assert selected_arm(_selection(), expected_manifest_hash=_DIGEST) == "a1"


def test_verification_may_name_the_arm_from_a_smoke_selection():
    """A smoke cache is still a cache, and verification checks coverage against its own plan."""
    assert selected_arm(_selection(smoke_examples=2), expected_manifest_hash=_DIGEST, allow_smoke=True) == "a1"


def test_a_cache_run_refuses_a_smoke_authored_selection():
    with pytest.raises(ValueError, match="2-example smoke run"):
        _cache(_Recorder(selection=_selection(smoke_examples=2)), names=_dev_names())


def test_a_cache_run_refuses_a_selection_from_another_manifest():
    recorder = _Recorder(selection=_selection(manifest_hash="f" * 64))

    with pytest.raises(ValueError, match="different dev64 manifest"):
        _cache(recorder, names=_dev_names(), selection_digest=_DIGEST)


def test_the_published_selection_carries_its_own_provenance():
    _, _, recorder = _capacity(plan=_plan(smoke_examples=2))

    selection = recorder.json["selection.json"]

    assert selection["cohort"] == "dev64"
    assert selection["manifest_hash"] == "m" * 64  # the capacity run's own manifest digest
    assert selection["smoke_examples"] == 2  # ... and it admits it was a smoke


def test_the_default_sinks_read_a_marker_without_reading_records():
    """Verification's fingerprint bootstrap must not be the record-reading path in disguise."""
    from maxdiffusion.null_adapter_modes import default_sinks, read_marker, read_shard

    sinks = default_sinks()

    assert sinks.read_marker is read_marker
    assert sinks.read_marker is not read_shard
    assert sinks.read_shard is read_shard
