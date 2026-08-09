"""exp_04 R10 — ``run_wan_null_inversion``: the decisions a J1 launch makes, and the one it loads with.

The entrypoint is split so that everything a launch can get *wrong* is a pure function: which mode
runs, which names in which batches, which example comes out of which shard, what a decoded pixel
means, which failures may become data gaps, what a record's provenance is bound to, and whether there
is room on disk to write at all. All of that is tested here with fakes -- including ``main`` itself,
which composes them behind four injectable seams.

**The backend class is pinned statically.** ``WanPipelineTI2V_2_2.from_pretrained`` cannot be
imported in this environment (it pulls ``transformers`` through ``pyconfig``), and it certainly
cannot be *called* without a TPU and 5B weights. So the property the BLOCKER turned on is asserted
against the repo's own syntax trees instead: the subclass defines ``from_pretrained``, the base class
does not, and the entrypoint loads the subclass inside ``axis_rules``. That is exactly the fact R10
got wrong, and it is checkable here rather than at the first launch.

The decode wrapper deserves its own note. The pipeline emits ``[0, 1]`` already -- verified in the
module docstring against ``wan_pipeline.py``/``image_processor.py`` line by line -- so the default
conversion is the identity, at zero tolerance. The wrapper takes the convention as a *declaration*,
and the declaration is trusted rather than verified: the three ranges nest, so ``[0, 1]`` pixels
declared ``byte`` pass the check and are quietly squeezed. Only ``unit`` is a claim about this
backend.
"""

from __future__ import annotations

import ast
import types

import numpy as np
import pytest
import yaml

from maxdiffusion.null_adapter_cache_policy import ExampleDivergenceError
from maxdiffusion.null_adapter_records import PRODUCTION_GEOMETRY
from maxdiffusion.run_wan_null_inversion import (
    DIVERGENCE_SIGNATURES,
    EXAMPLE_FIELDS,
    FIDELITY_COHORT,
    NULL_MODES,
    PIXEL_CONVENTIONS,
    batching_plan,
    build_read_batch,
    check_free_space,
    cohort_names,
    divergent,
    guard_example_divergence,
    main,
    load_adoption,
    manifest_digest,
    manifest_rows,
    mode_kwargs,
    model_revision,
    pixel_decoder,
    plan_run,
    published_shards,
    reader_rows,
    resolve_mode,
    resolved_code_sha,
    sweep_stale_staging,
)

_CONFIG_PATH = "src/maxdiffusion/configs/base_wan_5b_null_inversion.yml"
_ENTRYPOINT_PATH = "src/maxdiffusion/run_wan_null_inversion.py"
_TI2V_PATH = "src/maxdiffusion/pipelines/wan/wan_pipeline_ti2v_2p2.py"
_BASE_PIPELINE_PATH = "src/maxdiffusion/pipelines/wan/wan_pipeline.py"
_DOCUMENTED_KEYS = (
    "null_mode",
    "null_L",
    "null_inner_iters",
    "null_lr",
    "null_guide_scale",
    "inversion_guide_scale",
    "null_noise_convention",
    "null_batch_size",
    "null_pixel_convention",
    "null_manifest_dir",
    "null_cohort",
    "null_artifact_dir",
    "null_staging_dir",
    "null_data_dir",
    "null_arms",
    "null_eval_seed",
    "null_decode_subset",
    "null_decode_batch_size",
    "null_fidelity_subset",
    "null_latent_dtype",
    "null_adequacy_grid",
    "null_min_free_bytes",
    "null_verify_atol",
    "null_smoke_examples",
    "null_selection_uri",
    "null_adequacy_uri",
)


def _source(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _names(count, prefix="ep"):
    return tuple(f"{prefix}{index}_v0_s{index * 4:05d}" for index in range(count))


def _rows(names=_names(10), cohort="dev64", shard="gs://bucket/data/val-00000.tfrecord"):
    return [
        {
            "split": cohort,
            "name": name,
            "episode": name.split("_")[0][2:],
            "ordinal": index,
            "shard_path": shard,
            "shard_generation": "17000000000000",
            "shard_size": 4096,
        }
        for index, name in enumerate(names)
    ]


def _manifests(names=_names(10), cohort="dev64", extra=None, **header):
    """A manifest set. ``extra`` adds a second cohort, which is what a J2 cache job actually loads."""
    manifests = {
        "header": {"schema_version": 1, "shard_listing_checksum": "z" * 64, **header},
        cohort: {"schema_version": 1, "cohort": cohort, "rows": _rows(names, cohort)},
    }
    for name, rows in (extra or {}).items():
        manifests[name] = {"schema_version": 1, "cohort": name, "rows": _rows(rows, name)}
    return manifests


def _config(**overrides):
    fields = {
        "null_mode": "capacity",
        "null_cohort": "dev64",
        "null_batch_size": 4,
        "null_inner_iters": 10,
        "null_lr": 0.01,
        "null_guide_scale": 5.0,
        "null_L": 16,
        "null_noise_convention": "keyed",
        "null_latent_dtype": "fp16",
        "null_eval_seed": 2026,
        "null_decode_subset": 8,
        "null_decode_batch_size": 8,
        "null_adequacy_grid": "10:0.01,25:0.03",
        "null_smoke_examples": 0,
        "null_artifact_dir": "gs://bucket/exp04/run",
        "null_staging_dir": "",
        "null_manifest_dir": "gs://bucket/manifests/j0",
        "null_min_free_bytes": 0,
        "null_verify_atol": 0.01,
        "null_selection_uri": "",
        "null_adequacy_uri": "",
        "null_a3_measure": False,
        "null_a3_iters": 300,
        # Read only inside ``_load_backend``, which every test here injects -- carried anyway so the
        # fake is a faithful stand-in for a real config rather than only for the paths we happen to
        # exercise today.
        "null_pixel_convention": "unit",
        "activations_dtype": "bfloat16",
        "logical_axis_rules": [],
        "wan_max_sequence_length": 512,
        "pretrained_model_name_or_path": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "code_sha": "a" * 40,
        **overrides,
    }
    return types.SimpleNamespace(**fields)


@pytest.mark.parametrize("mode", NULL_MODES)
def test_every_planned_mode_resolves(mode):
    assert resolve_mode(mode) == mode


@pytest.mark.parametrize("mode", ["", "Capacity", "cache ", None, 3, "rollout"])
def test_an_unknown_mode_is_refused(mode):
    with pytest.raises(ValueError, match="null_mode must be one of"):
        resolve_mode(mode)


def test_the_batching_plan_keeps_manifest_order_and_carries_the_remainder():
    # Hash-ordered, like a real J0 cohort: emphatically not the lexicographic order, so a plan that
    # sorted "for determinism" would produce different shard membership for the same work.
    names = (
        "ep7_v0_s00000",
        "ep2_v0_s00004",
        "ep10_v0_s00008",
        "ep1_v0_s00012",
        "ep30_v0_s00016",
        "ep4_v0_s00020",
        "ep25_v0_s00024",
        "ep3_v0_s00028",
        "ep9_v0_s00032",
        "ep6_v0_s00036",
    )
    assert names != tuple(sorted(names))

    batches = batching_plan(names, 4)

    assert batches == (names[0:4], names[4:8], names[8:10])
    assert tuple(name for batch in batches for name in batch) == names  # order preserved exactly
    assert batching_plan(names, 10) == (names,)
    assert batching_plan(names, 1) == tuple((name,) for name in names)


@pytest.mark.parametrize(
    "names, size, message",
    [
        ((), 4, "cohort is empty"),
        (("a", "a"), 4, "unique"),
        (_names(3), 0, "integer >= 1"),
        (_names(3), -1, "integer >= 1"),
        (_names(3), True, "integer >= 1"),
        (_names(3), 2.0, "integer >= 1"),
    ],
)
def test_an_unusable_batching_request_is_refused(names, size, message):
    with pytest.raises(ValueError, match=message):
        batching_plan(names, size)


def _frames(value, shape=(2, 33, 4, 6, 3)):
    return np.full(shape, value, np.float32)


def test_the_unit_convention_is_the_identity_because_the_pipeline_already_emits_it():
    pixels = np.linspace(0.0, 1.0, 2 * 33 * 4 * 6 * 3, dtype=np.float32).reshape(2, 33, 4, 6, 3)

    decoded = pixel_decoder(lambda latents: pixels)(None)

    np.testing.assert_array_equal(decoded, pixels)
    assert decoded.dtype == np.float32


@pytest.mark.parametrize(
    "convention, raw, expected",
    [
        ("unit", 0.25, 0.25),
        ("signed", -1.0, 0.0),
        ("signed", 0.0, 0.5),
        ("signed", 1.0, 1.0),
        ("byte", 0.0, 0.0),
        ("byte", 127.5, 0.5),
        ("byte", 255.0, 1.0),
    ],
)
def test_each_declared_convention_lands_in_zero_one(convention, raw, expected):
    decoded = pixel_decoder(lambda latents: _frames(raw), convention)(None)

    assert decoded.min() == pytest.approx(expected) and decoded.max() == pytest.approx(expected)
    assert 0.0 <= decoded.min() and decoded.max() <= 1.0


@pytest.mark.parametrize(
    "convention, raw, message",
    [
        ("unit", -0.5, r"outside the declared 'unit' range"),  # a signed decoder under a unit declaration
        ("unit", 200.0, r"outside the declared 'unit' range"),
        ("signed", 200.0, r"outside the declared 'signed' range"),
        ("byte", -3.0, r"outside the declared 'byte' range"),
    ],
)
def test_a_decoder_that_does_not_match_its_declaration_is_refused(convention, raw, message):
    """Declared, never sniffed: rescaling whatever arrives would hide the wiring bug it came from."""
    with pytest.raises(ValueError, match=message):
        pixel_decoder(lambda latents: _frames(raw), convention)(None)


@pytest.mark.parametrize("raw", [1.0005, 1.0 + 1e-6, -1e-6])
def test_the_production_backend_is_pinned_to_unit_with_zero_tolerance(raw):
    """The postprocessor already clamps, so any excursion is a wiring bug, not rounding -- and a
    5e-4 excursion is already enough to move SSIM measurably (R7's strict-boundary rule)."""
    with pytest.raises(ValueError, match="outside the declared 'unit' range"):
        pixel_decoder(lambda latents: _frames(raw))(None)


def test_a_tolerance_is_available_but_is_not_what_this_run_uses():
    assert pixel_decoder(lambda latents: _frames(1.0005), atol=1e-3)(None).max() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="atol must be finite and non-negative"):
        pixel_decoder(lambda latents: _frames(0.5), atol=-1.0)


def test_non_finite_pixels_are_refused():
    with pytest.raises(ValueError, match="non-finite pixels"):
        pixel_decoder(lambda latents: _frames(np.nan))(None)


def test_an_unknown_pixel_convention_is_refused():
    assert set(PIXEL_CONVENTIONS) == {"unit", "signed", "byte"}
    with pytest.raises(ValueError, match="pixel convention must be one of"):
        pixel_decoder(lambda latents: _frames(0.5), "srgb")


@pytest.mark.parametrize("signature", DIVERGENCE_SIGNATURES)
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_the_r6_trace_failures_become_example_divergence(signature, error_type):
    def run_fn(names):
        raise error_type(f"{signature}: a diagnostic trace carrying NaN or inf is not evidence")

    with pytest.raises(ExampleDivergenceError, match=signature):
        guard_example_divergence(run_fn)(("ep1_v0_s00000",))


@pytest.mark.parametrize(
    "error",
    [
        ValueError("z_video must have the production shape (2, 48, 9, 12, 20), got (2, 48, 9, 12, 21)"),
        ValueError("header guide_scale 7.0 does not match the run's 5.0"),
        # Says "finite", but it is a *configuration* error from R3's argument guard: matching on the
        # word rather than on the R6 trace names would turn a bad recipe into a quarantined example.
        ValueError("guide_scale must be finite, got nan"),
        ValueError("lr must be finite and non-negative, got -0.001"),
        RuntimeError("gsutil stat failed for gs://bucket/x"),
        MemoryError("RESOURCE_EXHAUSTED: Out of memory while allocating"),
        KeyboardInterrupt(),
    ],
)
def test_every_other_failure_passes_through_untouched(error):
    """A geometry bug, a provenance mismatch, an OOM or a preemption is the job's problem, not one
    example's -- laundering any of them into a data gap is what R8's ratification forbids."""

    def run_fn(names):
        raise error

    with pytest.raises(type(error)):
        guard_example_divergence(run_fn)(("ep1_v0_s00000",))
    assert not divergent(error)


def test_a_divergence_error_from_below_is_passed_along_unchanged():
    original = ExampleDivergenceError("tracking_losses must be finite")

    def run_fn(names):
        raise original

    with pytest.raises(ExampleDivergenceError) as caught:
        guard_example_divergence(run_fn)(("ep1_v0_s00000",))
    assert caught.value is original


def test_a_clean_run_passes_its_result_through():
    assert guard_example_divergence(lambda names: {"ok": names})(("a",)) == {"ok": ("a",)}


def _statvfs(free_bytes, block=4096):
    return lambda path: types.SimpleNamespace(f_bavail=free_bytes // block, f_frsize=block)


def test_the_free_space_floor_admits_a_roomy_disk_and_refuses_a_full_one():
    assert check_free_space("/data", 100, statvfs=_statvfs(1 << 30)) >= 1 << 30

    with pytest.raises(RuntimeError, match="below the declared floor"):
        check_free_space("/data", 1 << 40, statvfs=_statvfs(1 << 30))


def test_object_storage_has_no_local_floor():
    assert check_free_space("gs://bucket/artifacts", 1 << 40, statvfs=_statvfs(0)) == -1


@pytest.mark.parametrize("floor", [-1, True, 1.5, "100"])
def test_a_malformed_floor_is_a_caller_error(floor):
    with pytest.raises(ValueError, match="free-space floor"):
        check_free_space("/data", floor, statvfs=_statvfs(1 << 30))


def test_stale_attempt_directories_are_swept():
    removed = []
    stale = ["gs://b/staging/shard_00001.abc.staging", "gs://b/staging/shard_00000.def.staging"]

    swept = sweep_stale_staging("gs://b/staging", lister=lambda prefix: stale, remover=removed.append)

    assert swept == tuple(sorted(stale)) and removed == sorted(stale)
    assert sweep_stale_staging("", lister=lambda prefix: stale, remover=removed.append) == ()


def test_published_shards_are_the_directories_that_carry_a_marker():
    markers = [
        "gs://b/run/a2/shard_00000/_COMPLETE.json",
        "gs://b/run/a1/shard_00000/_COMPLETE.json",
    ]

    found = published_shards("gs://b/run", lister=lambda pattern: markers)

    assert found == ("gs://b/run/a1/shard_00000", "gs://b/run/a2/shard_00000")
    assert published_shards("gs://b/run", lister=lambda pattern: []) == ()


def test_the_model_revision_carries_the_snapshot_hash_when_there_is_one():
    resolved = "/root/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/" + "a" * 40 + "/x"

    assert model_revision("Wan-AI/Wan2.2-TI2V-5B-Diffusers", resolved=resolved) == (
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers@" + "a" * 40
    )
    assert model_revision("gs://v6_east1d/models/wan", resolved="/local/dir").endswith("@unresolved")
    assert model_revision("x", resolved=None) == "x@unresolved"


# --------------------------------------------------------------------------- provenance


def test_the_code_sha_must_look_like_a_commit():
    assert resolved_code_sha("b" * 40) == "b" * 40
    assert resolved_code_sha(f"  {'c' * 40}\n") == "c" * 40


@pytest.mark.parametrize("value", ["", None, "unknown", "b" * 39, "b" * 41, "B" * 40, "z" * 40, 42])
def test_a_run_without_a_real_commit_refuses_to_publish(value):
    """R10 printed COMMIT and never exported it, so every record would have said ``code_sha=unknown``
    -- a cache whose provenance means nothing, discovered by whoever tried to reproduce it."""
    with pytest.raises(ValueError, match="40-character hex commit"):
        resolved_code_sha(value)


def test_the_manifest_hash_is_a_digest_of_the_selected_cohort_not_of_the_scan():
    manifests = _manifests()

    digest = manifest_digest(manifests, "dev64")

    assert len(digest) == 64 and int(digest, 16) >= 0
    assert digest != manifests["header"]["shard_listing_checksum"]
    # It moves with the rows ...
    edited = _manifests()
    edited["dev64"]["rows"][0]["ordinal"] = 999
    assert manifest_digest(edited, "dev64") != digest
    # ... and with the header, which the listing checksum alone would not do.
    assert manifest_digest(_manifests(builder_sha="d" * 40), "dev64") != digest


def test_the_manifest_hash_is_stable_for_the_same_evidence():
    assert manifest_digest(_manifests(), "dev64") == manifest_digest(_manifests(), "dev64")


# --------------------------------------------------------------------------- the manifest-bound reader


def _example(name, ordinal, *, scale=1.0):
    return (
        name,
        ordinal,
        np.full(PRODUCTION_GEOMETRY.z_i0, np.float32(scale)),
        np.full(PRODUCTION_GEOMETRY.z_video, np.float32(scale)),
        np.full(PRODUCTION_GEOMETRY.actions, np.float32(scale)),
    )


def _reader_for(rows, *, shuffle=False, drop=(), ordinal_shift=0, duplicate=()):
    def reader(shard_path, wanted):
        produced = [
            _example(row["name"], row["ordinal"] + ordinal_shift, scale=index + 1)
            for index, row in enumerate(rows)
            if row["name"] in set(wanted) and row["name"] not in set(drop)
        ]
        produced.extend(
            _example(row["name"], row["ordinal"] + ordinal_shift)
            for row in rows
            if row["name"] in set(duplicate) and row["name"] in set(wanted)
        )
        return reversed(produced) if shuffle else produced

    return reader


def _binder_for(rows, *, generation=None, size=None):
    def binder(shard_path):
        return {
            "kind": "gcs",
            "generation": generation or rows[0]["shard_generation"],
            "size": size if size is not None else rows[0]["shard_size"],
        }

    return binder


def _read_batch(rows=None, **kwargs):
    rows = rows or _rows(_names(4))
    return build_read_batch(
        {row["name"]: row for row in rows},
        reader=kwargs.pop("reader", None) or _reader_for(rows, **kwargs),
        binder=kwargs.pop("binder", None) or _binder_for(rows),
    )


def test_the_reader_returns_the_requested_names_in_the_requested_order():
    rows = _rows(_names(4))
    names = (rows[2]["name"], rows[0]["name"])

    batch, fields = _read_batch(rows, shuffle=True)(names)

    assert batch.names == names
    assert np.asarray(batch.z_i0).shape == (2, *PRODUCTION_GEOMETRY.z_i0)
    assert np.asarray(batch.z_video).shape == (2, *PRODUCTION_GEOMETRY.z_video)
    # Row i of the tensors is names[i]: the third example was written with scale 3, the first with 1.
    assert np.asarray(batch.z_video)[0].mean() == pytest.approx(3.0)
    assert np.asarray(batch.z_video)[1].mean() == pytest.approx(1.0)
    assert sorted(fields) == sorted(names)
    assert sorted(fields[names[0]]) == sorted(EXAMPLE_FIELDS)
    assert fields[names[0]]["ordinal"] == 2 and fields[names[0]]["split"] == "dev64"
    assert np.asarray(fields[names[0]]["actions"]).shape == PRODUCTION_GEOMETRY.actions


def test_the_reader_refuses_a_name_outside_the_cohort():
    with pytest.raises(ValueError, match="not in the cohort manifest"):
        _read_batch()(("ep0_v0_s00000", "stranger"))


def test_the_reader_refuses_a_record_whose_ordinal_is_not_its_rows():
    """The manifest binds one window per episode; a different ordinal is a different window."""
    with pytest.raises(ValueError, match="is not the window the cohort selected"):
        _read_batch(ordinal_shift=1)(("ep0_v0_s00000",))


def test_the_reader_refuses_a_shard_that_moved_since_the_manifest_bound_it():
    rows = _rows(_names(4))
    reader = build_read_batch(
        {row["name"]: row for row in rows},
        reader=_reader_for(rows),
        binder=_binder_for(rows, generation="17999999999999"),
    )

    with pytest.raises(RuntimeError, match="not the object the manifest bound"):
        reader(("ep0_v0_s00000",))


def test_the_reader_refuses_a_shard_whose_size_moved():
    rows = _rows(_names(4))
    reader = build_read_batch(
        {row["name"]: row for row in rows}, reader=_reader_for(rows), binder=_binder_for(rows, size=99)
    )

    with pytest.raises(RuntimeError, match="not the object the manifest bound"):
        reader(("ep0_v0_s00000",))


def test_a_shard_binding_is_verified_once_per_process_not_once_per_batch():
    """A GCS generation is immutable, so re-statting per batch buys a subprocess call and nothing."""
    rows = _rows(_names(4))
    calls = []

    def binder(shard_path):
        calls.append(shard_path)
        return {"kind": "gcs", "generation": rows[0]["shard_generation"], "size": rows[0]["shard_size"]}

    reader = build_read_batch({row["name"]: row for row in rows}, reader=_reader_for(rows), binder=binder)
    reader(("ep0_v0_s00000",))
    reader(("ep1_v0_s00004",))

    assert len(calls) == 1


def test_the_reader_refuses_a_name_its_shard_never_yielded():
    with pytest.raises(ValueError, match="did not yield"):
        _read_batch(drop=("ep1_v0_s00004",))(("ep0_v0_s00000", "ep1_v0_s00004"))


def test_the_reader_refuses_a_duplicated_record():
    with pytest.raises(ValueError, match="appears more than once"):
        _read_batch(duplicate=("ep0_v0_s00000",))(("ep0_v0_s00000",))


@pytest.mark.parametrize("names, message", [((), "asked for no examples"), (("a", "a"), "must be unique")])
def test_the_reader_refuses_an_unusable_request(names, message):
    with pytest.raises(ValueError, match=message):
        _read_batch()(names)


def test_the_reader_refuses_an_example_at_the_wrong_geometry():
    rows = _rows(_names(2))

    def reader(shard_path, wanted):
        for row in rows:
            if row["name"] in set(wanted):
                yield (
                    row["name"],
                    row["ordinal"],
                    np.zeros((48, 1, 12, 21), np.float32),
                    np.zeros(PRODUCTION_GEOMETRY.z_video, np.float32),
                    np.zeros(PRODUCTION_GEOMETRY.actions, np.float32),
                )

    bound = build_read_batch({row["name"]: row for row in rows}, reader=reader, binder=_binder_for(rows))
    with pytest.raises(ValueError, match="z_i0 must have the production shape"):
        bound(("ep0_v0_s00000",))


def test_the_reader_refuses_a_non_finite_example():
    rows = _rows(_names(2))

    def reader(shard_path, wanted):
        for row in rows:
            if row["name"] in set(wanted):
                yield (
                    row["name"],
                    row["ordinal"],
                    np.zeros(PRODUCTION_GEOMETRY.z_i0, np.float32),
                    np.full(PRODUCTION_GEOMETRY.z_video, np.nan, np.float32),
                    np.zeros(PRODUCTION_GEOMETRY.actions, np.float32),
                )

    bound = build_read_batch({row["name"]: row for row in rows}, reader=reader, binder=_binder_for(rows))
    with pytest.raises(ValueError, match="z_video must be finite"):
        bound(("ep0_v0_s00000",))


def test_the_reader_groups_a_request_by_source_shard():
    rows = _rows(_names(2))
    rows[1]["shard_path"] = "gs://bucket/data/val-00001.tfrecord"
    visited = []

    def reader(shard_path, wanted):
        visited.append((shard_path, tuple(wanted)))
        return [_example(row["name"], row["ordinal"]) for row in rows if row["name"] in set(wanted)]

    bound = build_read_batch({row["name"]: row for row in rows}, reader=reader, binder=_binder_for(rows))
    bound((rows[0]["name"], rows[1]["name"]))

    assert sorted(shard for shard, _ in visited) == sorted({row["shard_path"] for row in rows})
    assert all(len(wanted) == 1 for _, wanted in visited)


# --------------------------------------------------------------------------- the plan


def test_the_run_plan_is_decided_before_any_model_loads():
    names = _names(10)

    plan = plan_run(_config(), _manifests(names))

    assert plan["mode"] == "capacity" and plan["cohort"] == "dev64"
    assert plan["names"] == names and plan["batches"] == (names[0:4], names[4:8], names[8:10])
    assert plan["params"] == {"inner_iters": 10, "lr": 0.01, "guide_scale": 5.0, "l_null": 16}
    # The R6 header contract: exactly these two keys, nothing else.
    assert plan["optimization_config"] == {"inner_iters": 10, "lr": 0.01}
    assert plan["decode_subset"] == names[:8] and plan["eval_seed"] == 2026
    assert plan["smoke_examples"] == 0 and plan["decode_batch_size"] == 8
    assert "grid" not in plan


def test_the_smoke_limiter_truncates_the_declared_cohort_not_just_the_work():
    """A limiter that left ``names`` at 64 while running 2 would hand the gates a manifest with 62
    holes and call the resulting coverage failure a result."""
    names = _names(10)

    plan = plan_run(_config(null_smoke_examples=2, null_batch_size=2), _manifests(names))

    assert plan["names"] == names[:2] and plan["batches"] == (names[:2],)
    assert plan["decode_subset"] == names[:2] and plan["smoke_examples"] == 2


def test_a_smoke_limit_of_zero_runs_the_whole_cohort():
    names = _names(10)

    assert plan_run(_config(null_smoke_examples=0), _manifests(names))["names"] == names


@pytest.mark.parametrize("limit", [-1, True, 1.5, "2"])
def test_a_malformed_smoke_limit_is_refused(limit):
    with pytest.raises(ValueError, match="null_smoke_examples"):
        plan_run(_config(null_smoke_examples=limit), _manifests())


def test_the_adequacy_grid_is_parsed_only_for_its_own_mode():
    plan = plan_run(_config(null_mode="adequacy_probe"), _manifests())

    assert plan["grid"] == ((10, 0.01), (25, 0.03))


@pytest.mark.parametrize(
    "grid, message",
    [("", "grid is empty"), ("10", "J:lr"), ("x:0.01", "J:lr"), ("10:", "J:lr")],
)
def test_a_malformed_adequacy_grid_is_refused(grid, message):
    with pytest.raises(ValueError, match=message):
        plan_run(_config(null_mode="adequacy_probe", null_adequacy_grid=grid), _manifests())


def test_an_unknown_cohort_is_refused():
    with pytest.raises(ValueError, match="no cohort 'test64'"):
        cohort_names(_manifests(), "test64")
    with pytest.raises(ValueError, match="no cohort 'header'"):
        manifest_rows(_manifests(), "header")


def test_a_manifest_that_names_an_example_twice_is_refused():
    manifests = _manifests(_names(3))
    manifests["dev64"]["rows"].append(dict(manifests["dev64"]["rows"][0]))

    with pytest.raises(ValueError, match="names 'ep0_v0_s00000' twice"):
        manifest_rows(manifests, "dev64")


def test_the_plan_refuses_an_unknown_mode_before_anything_else():
    with pytest.raises(ValueError, match="null_mode must be one of"):
        plan_run(_config(null_mode="rollout"), _manifests())


# --------------------------------------------------------------------------- mode arguments


def _mode_kwargs(mode, **overrides):
    config = _config(null_mode=mode, **overrides)
    manifests = _manifests()
    manifests["dev64"]["rows"] = _rows(_names(10))
    plan = plan_run(config, manifests)
    return mode_kwargs(config, plan, manifests, shards_for=lambda root: (f"{root}/shard_00000",))


def test_capacity_gets_its_staging_provenance_and_decode_bound():
    kwargs = _mode_kwargs("capacity")

    assert set(kwargs) == {
        "artifact_dir",
        "staging_dir",
        "manifest_hash",
        "code_sha",
        "decode_batch_size",
        "adopted_recipe",
        "a3_measure",
    }
    assert kwargs["adopted_recipe"] is None  # no adequacy artifact, no adoption
    assert kwargs["a3_measure"] is False  # the A3 stage is opt-in
    assert kwargs["code_sha"] == "a" * 40 and len(kwargs["manifest_hash"]) == 64


def test_cache_gets_the_selection_the_dev_manifest_and_the_existing_shards():
    manifests = _manifests()
    manifests["dev64"]["rows"] = _rows(_names(10))
    config = _config(null_mode="cache")
    kwargs = mode_kwargs(config, plan_run(config, manifests), manifests, shards_for=lambda root: ("gs://b/s0",))

    assert kwargs["existing_shards"] == ("gs://b/s0",)
    assert kwargs["dev_manifest"] == _names(10)
    assert kwargs["selection_uri"] is None  # unset means "next to the artifacts"


def test_verify_gets_the_shards_the_tolerance_and_the_selection():
    kwargs = _mode_kwargs("verify_replay", null_selection_uri="gs://b/j1/selection.json")

    assert kwargs["atol"] == 0.01 and kwargs["shard_paths"] == ("gs://bucket/exp04/run/shard_00000",)
    assert kwargs["selection_uri"] == "gs://b/j1/selection.json"
    assert "code_sha" not in kwargs  # verification publishes no records, so it stamps none


def test_the_adequacy_probe_takes_only_an_artifact_directory():
    assert set(_mode_kwargs("adequacy_probe")) == {"artifact_dir"}


def test_a_publishing_mode_without_a_commit_refuses_before_the_model_loads():
    config = _config(null_mode="capacity", code_sha="")
    manifests = _manifests()

    with pytest.raises(ValueError, match="40-character hex commit"):
        mode_kwargs(config, plan_run(config, manifests), manifests, shards_for=lambda root: ())


# --------------------------------------------------------------------------- the backend class (the BLOCKER)


def _class_defs(path, class_name):
    tree = ast.parse(_source(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {child.name for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))}
    raise AssertionError(f"{class_name} is not defined in {path}")


def test_from_pretrained_lives_on_the_ti2v_subclass_and_not_on_the_base_pipeline():
    """The BLOCKER's premise, checked against the repo rather than assumed."""
    assert "from_pretrained" in _class_defs(_TI2V_PATH, "WanPipelineTI2V_2_2")
    assert "from_pretrained" not in _class_defs(_BASE_PIPELINE_PATH, "WanPipeline")


def _load_backend_ast():
    tree = ast.parse(_source(_ENTRYPOINT_PATH))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_load_backend":
            return node
    raise AssertionError("_load_backend is not defined in the entrypoint")


def test_the_entrypoint_loads_the_ti2v_pipeline_class():
    """``WanPipeline.from_pretrained`` does not exist; R10 shipped it, and it would have raised on
    the TPU at the first launch after the model had already been prefetched."""
    node = _load_backend_ast()
    imported = {alias.name for child in ast.walk(node) if isinstance(child, ast.ImportFrom) for alias in child.names}

    assert "WanPipelineTI2V_2_2" in imported
    assert "WanPipeline" not in imported
    modules = {child.module for child in ast.walk(node) if isinstance(child, ast.ImportFrom)}
    assert "maxdiffusion.pipelines.wan.wan_pipeline_ti2v_2p2" in modules


def test_the_pipeline_is_constructed_inside_the_axis_rules_context():
    """The trainer loads it this way (lines 286-287); outside the context the 5B transformer's
    logical annotations have no rules to resolve against."""
    node = _load_backend_ast()
    guarded = []
    for child in ast.walk(node):
        if not isinstance(child, ast.With):
            continue
        items = ast.dump(ast.Module(body=[ast.Expr(item.context_expr) for item in child.items], type_ignores=[]))
        if "axis_rules" in items:
            guarded.append(ast.dump(ast.Module(body=child.body, type_ignores=[])))

    assert guarded, "no axis_rules context in _load_backend"
    assert any("from_pretrained" in body for body in guarded)


def test_the_backend_returns_every_seam_the_modes_require():
    """A dictionary missing ``read_batch`` is how ``main`` used to raise ``KeyError`` deterministically."""
    node = _load_backend_ast()
    returned = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
            returned |= {key.value for key in child.value.keys if isinstance(key, ast.Constant)}

    assert {"velocity_fn", "decode_fn", "read_batch", "base_context", "resolved"} <= returned


def test_the_entrypoint_never_imports_the_side_adapter_generator():
    """Importing it would set ``jax_use_shardy_partitioner`` globally as a side effect."""
    source = _source(_ENTRYPOINT_PATH)

    assert "generate_wan_side_adapter" in source  # only as the docstring's warning
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "generate_wan_side_adapter" not in stripped, stripped
            assert "wan_ti2v_side_adapter_trainer" not in stripped, stripped


def test_coverage_pragmas_stay_on_the_literal_glue():
    """Function-level exclusions are where the wrong class, the missing reader and the wrong
    fingerprint all survived eighty tests (review, finding 9)."""
    source = _source(_ENTRYPOINT_PATH)
    excluded = {
        line.split("def ", 1)[1].split("(", 1)[0]
        for line in source.splitlines()
        if line.startswith("def ") and "pragma: no cover" in line
    }

    assert "main" not in excluded and "_load_backend" not in excluded
    assert excluded <= {
        "_tfrecord_reader",
        "_resolved_snapshot",
        "_configure",
        "_load_manifests",
        "_artifact_exists",
    }


# --------------------------------------------------------------------------- main, composed


class _MainRecorder:
    def __init__(self):
        self.calls = []

    def sinks(self):
        """``main`` reads the adequacy artifact through this seam, so it is a real object."""
        return types.SimpleNamespace(read_json=lambda uri: {})


def _fake_backend(recorder, resolved):
    def load_backend(config, rows):
        recorder.calls.append(("load_backend", tuple(rows)))
        return {
            "velocity_fn": lambda *a, **k: None,
            "decode_fn": lambda latents: latents,
            "read_batch": lambda names: (names, {}),
            "base_context": np.zeros((4, 4), np.float32),
            "resolved": resolved,
        }

    return load_backend


def test_main_composes_a_two_example_run_end_to_end(monkeypatch):
    """J1's first rung, without a TPU: config -> manifests -> plan -> backend -> mode -> exit code."""
    import maxdiffusion.null_adapter_modes as modes

    recorder = _MainRecorder()
    seen = {}

    def fake_execute(mode, plan, backend, sinks, **kwargs):
        seen.update(mode=mode, plan=plan, backend=backend, kwargs=kwargs)
        return {"mode": mode, "examples": len(plan["names"])}, 0

    monkeypatch.setattr(modes, "execute", fake_execute)
    names = _names(10)
    resolved = "/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/" + "e" * 40

    code = main(
        ["run", "config.yml"],
        configure=lambda argv: _config(null_smoke_examples=2, null_batch_size=2),
        load_manifests=lambda uri: _manifests(names),
        load_backend=_fake_backend(recorder, resolved),
        sinks=recorder.sinks(),
        shards_for=lambda root: (),
    )

    assert code == 0
    assert seen["mode"] == "capacity" and seen["plan"]["names"] == names[:2]
    assert seen["backend"].model_revision == "Wan-AI/Wan2.2-TI2V-5B-Diffusers@" + "e" * 40
    assert seen["kwargs"]["code_sha"] == "a" * 40 and len(seen["kwargs"]["manifest_hash"]) == 64
    # The backend was handed the cohort's manifest rows, which is what binds the reader.
    assert recorder.calls[0][1] == names


def test_main_returns_the_mode_exit_code(monkeypatch):
    import maxdiffusion.null_adapter_modes as modes

    monkeypatch.setattr(modes, "execute", lambda *a, **k: ({"mode": "verify_replay", "failures": ["x"]}, 1))

    code = main(
        ["run", "config.yml"],
        configure=lambda argv: _config(null_mode="verify_replay"),
        load_manifests=lambda uri: _manifests(),
        load_backend=_fake_backend(_MainRecorder(), ""),
        sinks=_MainRecorder().sinks(),
        shards_for=lambda root: ("gs://b/s0",),
    )

    assert code == 1


def test_main_checks_the_free_space_floor_before_it_loads_anything(monkeypatch, tmp_path):
    import maxdiffusion.null_adapter_modes as modes

    monkeypatch.setattr(modes, "execute", lambda *a, **k: ({}, 0))
    loaded = []

    with pytest.raises(RuntimeError, match="below the declared floor"):
        main(
            ["run", "config.yml"],
            configure=lambda argv: _config(null_artifact_dir=str(tmp_path), null_min_free_bytes=1 << 62),
            load_manifests=lambda uri: _manifests(),
            load_backend=lambda config, rows: loaded.append(1),
            sinks=_MainRecorder().sinks(),
            shards_for=lambda root: (),
        )
    assert loaded == []


# --------------------------------------------------------------------------- config


def test_the_config_carries_every_documented_key():
    with open(_CONFIG_PATH, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    missing = [key for key in _DOCUMENTED_KEYS if key not in config]
    assert not missing, f"pyconfig can only override keys that exist in the YAML; missing {missing}"
    assert config["null_mode"] in NULL_MODES
    assert config["null_pixel_convention"] in PIXEL_CONVENTIONS
    assert config["null_L"] == 16 and config["null_inner_iters"] == 10 and config["null_lr"] == 0.01
    assert config["null_guide_scale"] == 5.0 and config["inversion_guide_scale"] == 1.0
    assert config["null_eval_seed"] == 2026 and config["null_decode_subset"] == 8
    assert config["null_smoke_examples"] == 0 and config["null_selection_uri"] == ""
    assert config["model_name"] == "wan2.2" and config["height"] == 192 and config["width"] == 320
    assert config["num_frames"] == 32 and config["wan_max_sequence_length"] == 512


# --------------------------------------------------------------------------- the launcher


_LAUNCHER_PATH = "bash_scripts/run_wan_null_inversion.sh"
_SIDE_ADAPTER_LAUNCHER = "bash_scripts/train_wan_side_adapter.sh"


def _launcher():
    return _source(_LAUNCHER_PATH)


def _position(needle, source=None):
    source = source if source is not None else _launcher()
    index = source.find(needle)
    assert index >= 0, f"{needle!r} is not in the launcher"
    return index


def test_the_log_is_all_terminal_output():
    """R10 emitted prefetch, preflight, config and git state before the pipe was open, so a run that
    died in prefetch left a log that never mentioned prefetch (review, finding 11)."""
    source = _launcher()

    tee = _position('exec > >(tee -a "${LOG_FILE}") 2>&1', source)
    for later in ("prefetch_hf_snapshot.sh", "PREFLIGHT", "COMMIT=", "git status", "run_wan_null_inversion.py"):
        assert tee < _position(later, source), f"{later} is emitted before tee opens the log"


def test_the_commit_is_exported_and_validated_before_anything_runs():
    """Printed but never exported is how every record would have carried code_sha='unknown'.

    Asserted on a live statement rather than a substring: ``# export COMMIT`` contains
    ``export COMMIT`` too, and a commented-out export is precisely the regression being guarded.
    """
    source = _launcher()
    statements = [line.strip() for line in source.splitlines() if not line.strip().startswith("#")]

    assert "export COMMIT" in statements
    assert "grep -Eq '^[0-9a-f]{40}$'" in source
    assert _position("export COMMIT", source) < _position("run_wan_null_inversion.py", source)


def test_the_default_manifest_uri_is_the_ratified_mirror():
    assert "gs://v6_east1d/datasets/droid_wan_null_adapter/manifests/j0/" in _launcher()
    assert "gs://v6_east1d/manifests/exp04_j0" not in _launcher()


def test_the_ffmpeg_check_requires_an_executable_not_merely_a_file():
    """A non-executable file passes an existence check and then fails inside the writer, after the
    arms have already run."""
    source = _launcher()

    assert "os.access(ffmpeg, os.X_OK)" in source
    assert "shutil.which" not in source


def test_the_preflight_imports_the_pipeline_class_the_driver_loads():
    source = _launcher()

    assert "from maxdiffusion.pipelines.wan.wan_pipeline_ti2v_2p2 import WanPipelineTI2V_2_2" in source
    assert 'getattr(WanPipelineTI2V_2_2, "from_pretrained", None)' in source


def test_the_r1_noise_golden_is_asserted_on_device_before_any_arm_runs():
    """Every cached target is keyed to this draw; if the device disagrees, the artifacts are keyed to
    noise nothing else can reproduce."""
    source = _launcher()

    golden = _position("keyed_noise(name, k)", source)
    assert golden < _position("run_wan_null_inversion.py", source)
    # The pinned values are the R1 ones, not a fresh transcription.
    assert "1.392072319984436" in source and "0.18953724205493927" in source


def test_the_launcher_passes_every_new_config_key():
    source = _launcher()

    for key in ("null_smoke_examples", "null_decode_batch_size", "null_selection_uri", "null_adequacy_uri"):
        assert f'{key}="${{{key.upper()}' in source, key


def test_the_launcher_maps_the_adequacy_uri_env_onto_the_config_key():
    """Issue #15: the J1-4 runbook exported ``NULL_ADEQUACY_URI`` and this launcher had no mapping for
    it, so the YAML default ``''`` applied, ``load_adoption`` returned None, and capacity ran at the
    default J=10 while the adequacy probe had adopted J=50/lr=0.01.

    exp_04's launcher style is a three-place mapping -- an env default, an echo, and one CLI override
    -- and dropping any one of them breaks it: no default is fatal under ``set -u``, no override is
    the silent discard that happened, and no echo means the log cannot show which recipe ran.
    """
    source = _launcher()

    assert 'NULL_ADEQUACY_URI="${NULL_ADEQUACY_URI:-}"' in source
    assert 'echo "NULL_ADEQUACY_URI=${NULL_ADEQUACY_URI}"' in source
    assert 'null_adequacy_uri="${NULL_ADEQUACY_URI}"' in source
    # ``set -u`` aborts on an unset expansion, so the default has to be established first.
    default = _position('NULL_ADEQUACY_URI="${NULL_ADEQUACY_URI:-}"', source)
    assert default < _position('echo "NULL_ADEQUACY_URI=${NULL_ADEQUACY_URI}"', source)
    assert default < _position("run_wan_null_inversion.py", source)


def test_an_exported_adequacy_uri_is_not_silently_dropped():
    """The negative that bit: the env was exported at launch and never reached pyconfig.

    Asserted on the entrypoint's own argument list -- a mapping anywhere else in the file is not an
    override -- and on the spelling, because a mistyped expansion overrides with the empty string,
    which is indistinguishable from the default that hid the bug for four jobs.
    """
    import re

    source = _launcher()
    invocation = source.split("run_wan_null_inversion.py", 1)[1]

    assert 'null_adequacy_uri="${NULL_ADEQUACY_URI}"' in invocation
    assert set(re.findall(r"NULL_ADEQUACY\w*", source)) == {"NULL_ADEQUACY_URI"}


def test_the_xla_flags_stay_identical_to_the_side_adapter_launcher():
    """Same 5B transformer, same mesh: a divergence here is a performance mystery, not a feature."""

    def flags(path):
        block = _source(path).split("LIBTPU_INIT_ARGS=", 1)[1]
        return {line.strip().rstrip("\\}\"").strip() for line in block.split("}")[0].splitlines() if "--xla" in line}

    assert flags(_LAUNCHER_PATH) == flags(_SIDE_ADAPTER_LAUNCHER)


# --------------------------------------------------------------------------- the cache reader's rows


def _train_names(count=12):
    return tuple(f"tr{index}_v0_s{index * 4:05d}" for index in range(count))


def _two_cohort_manifests():
    return _manifests(_train_names(), cohort="train2000", extra={FIDELITY_COHORT: _names(10)})


def test_a_cache_reader_spans_the_cohort_and_the_first_eight_dev_rows():
    """J2's fidelity gate replays the first eight DEV examples before it caches a single TRAIN
    window; a reader built from the cohort alone refuses them by name, and the job dies in its first
    phase (follow-up review, finding 1)."""
    manifests = _two_cohort_manifests()
    plan = plan_run(_config(null_mode="cache", null_cohort="train2000"), manifests)

    rows = reader_rows(manifests, plan)

    assert set(_train_names()) <= set(rows)  # everything it caches ...
    assert set(_names(10)[:8]) <= set(rows)  # ... and everything the fp16 gate reads
    assert _names(10)[8] not in rows  # but no further: the subset is eight, not the cohort


def test_a_capacity_reader_spans_only_its_own_cohort():
    manifests = _two_cohort_manifests()
    plan = plan_run(_config(null_mode="capacity", null_cohort="train2000"), manifests)

    assert set(reader_rows(manifests, plan)) == set(_train_names())


def test_the_union_keeps_every_rows_own_binding():
    """Spanning two manifests widens what may be read, never what may be read unchecked."""
    manifests = _two_cohort_manifests()
    plan = plan_run(_config(null_mode="cache", null_cohort="train2000"), manifests)

    rows = reader_rows(manifests, plan)

    assert all(row["shard_generation"] and row["shard_size"] for row in rows.values())
    assert rows[_names(10)[0]]["split"] == FIDELITY_COHORT
    assert rows[_train_names()[0]]["split"] == "train2000"


def test_a_dev_cache_is_unaffected_by_the_union():
    manifests = _manifests(_names(10))
    plan = plan_run(_config(null_mode="cache"), manifests)

    assert set(reader_rows(manifests, plan)) == set(_names(10))


def test_the_reviewers_probe_a_train_cache_can_read_the_dev_fidelity_names():
    """The reviewer's exact probe: build the reader the way a J2 cache builds it, then ask it for the
    DEV names the fidelity phase asks for. It failed with 'these names are not in the cohort manifest'."""
    manifests = _two_cohort_manifests()
    plan = plan_run(_config(null_mode="cache", null_cohort="train2000"), manifests)
    rows = reader_rows(manifests, plan)
    read_batch = build_read_batch(
        rows, reader=_reader_for(list(rows.values())), binder=_binder_for(list(rows.values()))
    )

    batch, fields = read_batch(_names(10)[:8])

    assert batch.names == _names(10)[:8]
    assert sorted(fields) == sorted(_names(10)[:8])


# --------------------------------------------------------------------------- the adoption seam


def _adequacy_payload(**adopted):
    return {"mode": "adequacy_probe", "adopted": {"inner_iters": 25, "lr": 0.03, "adopted": True, **adopted}}


def test_no_adequacy_artifact_means_no_adopted_recipe():
    assert load_adoption("", exists=lambda uri: True, read_json=lambda uri: {}) is None
    assert load_adoption("gs://b/a.json", exists=lambda uri: False, read_json=lambda uri: {}) is None


def test_an_adequacy_artifact_that_exists_is_consumed():
    adoption = load_adoption(
        "gs://b/adequacy_report.json", exists=lambda uri: True, read_json=lambda uri: _adequacy_payload()
    )

    assert adoption["inner_iters"] == 25 and adoption["lr"] == 0.03 and adoption["adopted"] is True


@pytest.mark.parametrize("payload", [{}, {"adopted": {}}, {"adopted": {"inner_iters": 25}}, "nonsense", None])
def test_an_unusable_adequacy_artifact_stops_the_run(payload):
    """Fail-closed against silent recipe drift: an artifact sitting there while capacity runs at the
    default recipe produces plausible verdicts for an experiment nobody chose."""
    with pytest.raises(ValueError, match="no usable adoption block"):
        load_adoption("gs://b/a.json", exists=lambda uri: True, read_json=lambda uri: payload)


def test_an_unreadable_adequacy_artifact_propagates_rather_than_defaulting():
    def explode(uri):
        raise OSError("the object store returned 503")

    with pytest.raises(OSError):
        load_adoption("gs://b/a.json", exists=lambda uri: True, read_json=explode)


def test_capacity_is_handed_the_adopted_recipe():
    manifests = _manifests()
    config = _config(null_mode="capacity")
    adoption = _adequacy_payload()["adopted"]

    kwargs = mode_kwargs(
        config, plan_run(config, manifests), manifests, shards_for=lambda root: (), adoption=adoption
    )

    assert kwargs["adopted_recipe"] == adoption


def test_main_threads_the_adopted_recipe_into_capacity(monkeypatch):
    import maxdiffusion.null_adapter_modes as modes

    seen = {}
    monkeypatch.setattr(modes, "execute", lambda mode, plan, backend, sinks, **kw: (seen.update(kw) or ({}, 0)))

    code = main(
        ["run", "config.yml"],
        configure=lambda argv: _config(null_adequacy_uri="gs://b/adequacy_report.json"),
        load_manifests=lambda uri: _manifests(),
        load_backend=_fake_backend(_MainRecorder(), ""),
        sinks=types.SimpleNamespace(read_json=lambda uri: _adequacy_payload()),
        shards_for=lambda root: (),
        artifact_exists=lambda uri: True,
    )

    assert code == 0
    assert seen["adopted_recipe"] == {"inner_iters": 25, "lr": 0.03, "adopted": True}


def test_main_refuses_to_run_capacity_beside_an_unusable_adequacy_artifact(monkeypatch):
    import maxdiffusion.null_adapter_modes as modes

    monkeypatch.setattr(modes, "execute", lambda *a, **k: ({}, 0))

    with pytest.raises(ValueError, match="no usable adoption block"):
        main(
            ["run", "config.yml"],
            configure=lambda argv: _config(null_adequacy_uri="gs://b/adequacy_report.json"),
            load_manifests=lambda uri: _manifests(),
            load_backend=_fake_backend(_MainRecorder(), ""),
            sinks=types.SimpleNamespace(read_json=lambda uri: {"mode": "adequacy_probe"}),
            shards_for=lambda root: (),
            artifact_exists=lambda uri: True,
        )


def test_a_cache_run_is_bound_to_the_dev_manifest_digest():
    manifests = _two_cohort_manifests()
    config = _config(null_mode="cache", null_cohort="train2000")

    kwargs = mode_kwargs(config, plan_run(config, manifests), manifests, shards_for=lambda root: ())

    # The selection was made on DEV, so that is the digest it is bound to -- not the cached cohort's.
    assert kwargs["selection_digest"] == manifest_digest(manifests, FIDELITY_COHORT)
    assert kwargs["selection_digest"] != kwargs["manifest_hash"]


def test_verification_is_bound_to_the_same_dev_digest():
    manifests = _two_cohort_manifests()
    config = _config(null_mode="verify_replay", null_cohort="train2000")

    kwargs = mode_kwargs(config, plan_run(config, manifests), manifests, shards_for=lambda root: ())

    assert kwargs["selection_digest"] == manifest_digest(manifests, FIDELITY_COHORT)


def test_main_builds_a_cache_reader_that_spans_both_manifests(monkeypatch):
    """The composition, not just the helper: ``main`` must hand the backend the union, or J2's
    fidelity phase asks a cohort-only reader for DEV names and is refused by name."""
    import maxdiffusion.null_adapter_modes as modes

    monkeypatch.setattr(modes, "execute", lambda *a, **k: ({}, 0))
    recorder = _MainRecorder()
    manifests = _two_cohort_manifests()

    main(
        ["run", "config.yml"],
        configure=lambda argv: _config(null_mode="cache", null_cohort="train2000"),
        load_manifests=lambda uri: manifests,
        load_backend=_fake_backend(recorder, ""),
        sinks=recorder.sinks(),
        shards_for=lambda root: (),
        artifact_exists=lambda uri: False,
    )

    served = set(recorder.calls[0][1])
    assert set(_train_names()) <= served  # everything it caches ...
    assert set(_names(10)[:8]) <= served  # ... and everything the fp16 gate reads


def test_main_builds_a_capacity_reader_from_its_cohort_alone(monkeypatch):
    import maxdiffusion.null_adapter_modes as modes

    monkeypatch.setattr(modes, "execute", lambda *a, **k: ({}, 0))
    recorder = _MainRecorder()

    main(
        ["run", "config.yml"],
        configure=lambda argv: _config(null_mode="capacity", null_cohort="train2000"),
        load_manifests=lambda uri: _two_cohort_manifests(),
        load_backend=_fake_backend(recorder, ""),
        sinks=recorder.sinks(),
        shards_for=lambda root: (),
        artifact_exists=lambda uri: False,
    )

    assert set(recorder.calls[0][1]) == set(_train_names())


# --------------------------------------------------------------------------- A3 / J1b integration


def test_direct_opt_is_a_launchable_mode():
    """R11's optimizer was unreachable: nothing outside its own tests imported it, and there was no
    mode to dispatch a separately-approved J1b through (review, finding 1)."""
    assert "direct_opt" in NULL_MODES
    assert resolve_mode("direct_opt") == "direct_opt"


def test_the_capacity_a3_stage_is_config_gated():
    manifests = _manifests()
    enabled = _config(null_mode="capacity", null_a3_measure=True)

    kwargs = mode_kwargs(enabled, plan_run(enabled, manifests), manifests, shards_for=lambda root: ())

    assert kwargs["a3_measure"] is True


def test_direct_opt_gets_its_provenance_and_iteration_count():
    manifests = _manifests()
    config = _config(null_mode="direct_opt", null_a3_iters=17)

    kwargs = mode_kwargs(config, plan_run(config, manifests), manifests, shards_for=lambda root: ())

    assert kwargs["iters"] == 17
    assert kwargs["code_sha"] == "a" * 40 and len(kwargs["manifest_hash"]) == 64
    assert "staging_dir" not in kwargs  # J1b publishes no shards


def test_main_composes_a_direct_opt_run(monkeypatch):
    import maxdiffusion.null_adapter_modes as modes

    seen = {}
    monkeypatch.setattr(
        modes, "execute", lambda mode, plan, backend, sinks, **kw: (seen.update(mode=mode, kw=kw) or ({}, 0))
    )
    recorder = _MainRecorder()

    code = main(
        ["run", "config.yml"],
        configure=lambda argv: _config(null_mode="direct_opt", null_a3_iters=5),
        load_manifests=lambda uri: _manifests(),
        load_backend=_fake_backend(recorder, ""),
        sinks=recorder.sinks(),
        shards_for=lambda root: (),
        artifact_exists=lambda uri: False,
    )

    assert code == 0 and seen["mode"] == "direct_opt" and seen["kw"]["iters"] == 5


def test_main_propagates_a_refusing_fit_probe_as_a_non_zero_exit(monkeypatch):
    import maxdiffusion.null_adapter_modes as modes

    monkeypatch.setattr(modes, "execute", lambda *a, **k: ({"continued": False}, 1))

    code = main(
        ["run", "config.yml"],
        configure=lambda argv: _config(null_mode="direct_opt"),
        load_manifests=lambda uri: _manifests(),
        load_backend=_fake_backend(_MainRecorder(), ""),
        sinks=_MainRecorder().sinks(),
        shards_for=lambda root: (),
        artifact_exists=lambda uri: False,
    )

    assert code == 1


def test_the_launcher_exposes_the_a3_keys_and_an_external_watchdog():
    """The plan's hard stops cannot be enforced in-process: a synchronous XLA compile cannot be
    cancelled by the code blocked inside it (review, finding 2)."""
    source = _launcher()

    assert 'null_a3_measure="${NULL_A3_MEASURE}"' in source
    assert 'null_a3_iters="${NULL_A3_ITERS}"' in source
    assert "NULL_A3_WATCHDOG_SECONDS" in source
    assert "timeout --signal=TERM" in source
    assert '"${WATCHDOG[@]}" python src/maxdiffusion/run_wan_null_inversion.py' in source


def test_the_watchdog_is_phase_aware_and_on_by_default_for_j1b():
    """J1b is exactly one A3 optimization, so it gets a hard ceiling derived from the plan's own
    per-phase budgets. The capacity run does not: an A3-sized timeout around a multi-hour job would
    kill the arms rather than the measurement, and its protection is the queue's own timeout."""
    source = _launcher()

    assert 'if [ "${NULL_MODE}" = "direct_opt" ]; then' in source
    # ON by default for J1b: the default expands to the derived ceiling, not to 0.
    assert "NULL_A3_WATCHDOG_SECONDS:-$((A3_COMPILE_BUDGET + NULL_A3_ITERS * A3_UPDATE_BUDGET" in source
    # ... and OFF by default everywhere else, said out loud rather than left implicit.
    assert 'NULL_A3_WATCHDOG_SECONDS="${NULL_A3_WATCHDOG_SECONDS:-0}"' in source
    assert "job-level protection is the queue timeout" in source
    assert "A3_WATCHDOG_MARGIN" in source


def test_the_launcher_does_not_claim_the_in_capacity_stage_can_hard_stop():
    """The stage-level numbers are verdicts; saying otherwise is the thing the review objected to."""
    source = _launcher()

    assert "only *reported* by measure_single_update" in source
    assert "stage-level verdicts" in source


def test_the_config_carries_the_a3_keys():
    with open(_CONFIG_PATH, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["null_a3_measure"] is False and config["null_a3_iters"] == 300
    assert "direct_opt" in config["null_mode"] or config["null_mode"] in NULL_MODES


def test_the_fake_config_declares_every_key_the_driver_reads_directly():
    """The fixture drifted from the YAML and a three-argument ``getattr`` hid it -- which is the same
    shape of bug J1 hit, one layer down. Required keys are read directly now, so the fake has to
    carry them or the test lies about what a launch does."""
    import ast as _ast

    source = _source(_ENTRYPOINT_PATH)
    tree = _ast.parse(source)
    called = {
        node.func for node in _ast.walk(tree) if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Attribute)
    }
    read = {
        node.attr
        for node in _ast.walk(tree)
        if isinstance(node, _ast.Attribute)
        and isinstance(node.value, _ast.Name)
        and node.value.id == "config"
        and node not in called
    }

    fake = _config()
    missing = sorted(key for key in read if not hasattr(fake, key))
    assert missing == [], f"the fake config is missing keys the driver reads: {missing}"
