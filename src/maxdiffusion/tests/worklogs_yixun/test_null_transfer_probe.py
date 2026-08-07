"""exp_04 R12-lite `transfer-probe` — do J1b's jointly-optimized nulls survive a change of basin?

J1b answered "can a null tensor reach own-basin quality from fresh noise" (3/8 yes). J1c asks the
question that killed A1/A2: are those tensors a property of the *example*, or of the *noise draw they
were optimized against*? The mode replays EXTERNAL nulls -- loaded from J1b's published npz, never
re-optimized -- from four noise settings and scores each.

Three things carry the round:

- **The rows are the example's own.** ``a3_nulls.npz`` is step-major (``[N, B, L, D]``), so example i
  lives at ``nulls[:, i]``. A crossover -- example i replayed under example j's nulls -- would produce
  a complete, plausible table showing no transfer, which is the exact answer the probe is asking for.
  The replay seam is captured and compared element-by-element against the loaded array.
- **The noise is the convention's, not a fresh draw.** ``global(0)`` and ``keyed(name, k)`` come from
  the golden-pinned helpers by import; a probe on noise nobody else can reproduce measures nothing.
- **The artifact is bound to what produced it.** The npz URI *and its sha256*, code_sha, manifest
  hash, l_null and guide scale ride in the JSON: J1c's table only means something next to the J1b run
  it was computed from.
"""

from __future__ import annotations

import io
import json

import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion.models.wan.null_inversion_wan import global_noise, keyed_noise
from maxdiffusion.null_adapter_records import PRODUCTION_GEOMETRY
from maxdiffusion.null_adapter_runner_core import SINGLE_SEED_KEY
from maxdiffusion.null_transfer_probe import (
    NULLS_FIELD,
    TRANSFER_NAME,
    TRANSFER_SETTINGS,
    load_transfer_nulls,
    run_transfer_probe,
    setting_label,
    transfer_start_latents,
)

_NAMES = tuple(f"ep{index}_v0_s{index * 4:05d}" for index in range(4))
_STEPS, _L = 25, 16
# The noise helpers are pinned to the PRODUCTION latent shape (R1's golden), so a probe fixture cannot
# shrink the latents. The context width is free, and a narrow one keeps the npz fixtures small.
_GEOMETRY = PRODUCTION_GEOMETRY.z_video
_WIDTH = 64


def _npz_bytes(nulls: np.ndarray, *, field: str = NULLS_FIELD, extra: dict | None = None) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **{field: nulls, **(extra or {})})
    return buffer.getvalue()


def _nulls(examples=len(_NAMES), steps=_STEPS, l_null=_L, width=_WIDTH):
    """Step-major, as ``direct_optimize_nulls`` returns and ``write_arrays`` stores: ``[N, B, L, D]``.

    Every example's block is a distinct constant, so a crossover is visible in the replayed value.
    """
    rows = np.arange(examples, dtype=np.float32)[None, :, None, None] + 1.0
    return np.broadcast_to(rows, (steps, examples, l_null, width)).astype(np.float32).copy()


class _Backend:
    """A toy backend: the reader, a decoder and a velocity function nothing here differentiates."""

    def __init__(self, names=_NAMES):
        self.names = tuple(names)
        self.base_context = np.zeros((32, _WIDTH), np.float32)
        self.model_revision = "Wan2.2-TI2V-5B@" + "b" * 40
        self.decoded = []

    def read_batch(self, names):
        names = tuple(names)
        rng = np.random.default_rng(abs(hash(names)) % (2**32))
        batch = type(
            "Batch",
            (),
            {
                "names": names,
                "z_i0": rng.standard_normal((len(names), _GEOMETRY[0], 1, *_GEOMETRY[2:])).astype(np.float32),
                "z_video": rng.standard_normal((len(names), *_GEOMETRY)).astype(np.float32),
            },
        )()
        return batch, {name: {"ordinal": index} for index, name in enumerate(names)}

    def velocity_fn(self, latents, timestep_2d, context):
        # jnp, not np: the replay is scanned, so this runs on tracers.
        return jnp.asarray(latents, jnp.float32) * 0.5 + jnp.mean(jnp.asarray(context, jnp.float32)) * 1e-3

    def decode_fn(self, latents):
        """9 latent frames -> the 33 raw frames the pixel path expects (1 + 8x4)."""
        self.decoded.append(np.asarray(latents, np.float32))
        per_frame = np.asarray(latents, np.float32).mean(axis=1)
        frames = np.concatenate([per_frame[:, :1], np.repeat(per_frame[:, 1:], 4, axis=1)], axis=1)
        return np.repeat((0.5 + 0.4 * np.tanh(frames))[..., None], 3, axis=-1).astype(np.float32)


class _Sinks:
    def __init__(self):
        self.json = {}

    def write_json(self, path, payload):
        self.json[path.rsplit("/", 1)[-1]] = json.loads(json.dumps(payload, default=str))
        return path


def _plan(names=_NAMES, cohort="dev64", guide_scale=5.0, l_null=_L):
    return {
        "mode": "transfer_probe",
        "cohort": cohort,
        "names": tuple(names),
        "params": {"guide_scale": guide_scale, "l_null": l_null},
    }


def _run(tmp_nulls=None, *, plan=None, backend=None, sinks=None, replay=None, **overrides):
    backend = backend or _Backend()
    sinks = sinks or _Sinks()
    payload = _npz_bytes(_nulls()) if tmp_nulls is None else tmp_nulls
    kwargs = {
        "artifact_dir": "gs://bucket/artifacts/j1c",
        "nulls_uri": "gs://bucket/artifacts/j1b/a3_nulls.npz",
        "manifest_hash": "m" * 64,
        "code_sha": "a" * 40,
        "read_bytes": lambda uri: payload,
        "sigmas": np.linspace(1.0, 0.0, _STEPS + 1).astype(np.float32),
    }
    kwargs.update(overrides)
    if replay is not None:
        kwargs["replay"] = replay
    return run_transfer_probe(plan or _plan(), backend, sinks, **kwargs), sinks


# --------------------------------------------------------------------------------------------------
# 1. The npz: J1b's actual schema, verified rather than assumed.
# --------------------------------------------------------------------------------------------------


def test_the_published_schema_is_step_major():
    """``direct_optimize_nulls`` returns ``[N, B, L, D]`` and ``write_arrays`` stores it unchanged --
    steps first, examples second. Reading it as ``[B, N, L, D]`` would silently transpose the study."""
    nulls, digest = load_transfer_nulls(
        "gs://b/a3_nulls.npz",
        read_bytes=lambda uri: _npz_bytes(_nulls()),
        steps=_STEPS,
        l_null=_L,
        width=_WIDTH,
        examples=len(_NAMES),
    )

    assert nulls.shape == (_STEPS, len(_NAMES), _L, _WIDTH) and nulls.dtype == np.float32
    assert len(digest) == 64 and int(digest, 16) >= 0  # a sha256 of the bytes that were read


def test_the_digest_is_of_the_bytes_that_were_read():
    payload = _npz_bytes(_nulls())
    expected = __import__("hashlib").sha256(payload).hexdigest()

    _, digest = load_transfer_nulls(
        "gs://b/x.npz", read_bytes=lambda uri: payload, steps=_STEPS, l_null=_L, width=_WIDTH, examples=len(_NAMES)
    )

    assert digest == expected


@pytest.mark.parametrize(
    "nulls, message",
    [
        (_nulls(steps=24), "sampler-step"),
        (_nulls(l_null=8), "l_null"),
        (_nulls(width=2048), "width"),
        (_nulls()[0], "rank"),
        (_nulls(examples=3), "examples"),
    ],
)
def test_a_geometry_that_is_not_j1bs_is_refused(nulls, message):
    with pytest.raises(ValueError, match=message):
        load_transfer_nulls(
            "gs://b/x.npz",
            read_bytes=lambda uri: _npz_bytes(nulls),
            steps=_STEPS,
            l_null=_L,
            width=_WIDTH,
            examples=len(_NAMES),
        )


def test_an_npz_without_the_nulls_array_is_refused():
    with pytest.raises(ValueError, match=NULLS_FIELD):
        load_transfer_nulls(
            "gs://b/x.npz",
            read_bytes=lambda uri: _npz_bytes(_nulls(), field="weights"),
            steps=_STEPS,
            l_null=_L,
            width=_WIDTH,
            examples=len(_NAMES),
        )


def test_nonfinite_nulls_are_refused():
    poisoned = _nulls()
    poisoned[3, 1, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        load_transfer_nulls(
            "gs://b/x.npz",
            read_bytes=lambda uri: _npz_bytes(poisoned),
            steps=_STEPS,
            l_null=_L,
            width=_WIDTH,
            examples=len(_NAMES),
        )


def test_an_unreadable_uri_propagates():
    def explode(uri):
        raise FileNotFoundError(uri)

    with pytest.raises(FileNotFoundError):
        load_transfer_nulls(
            "gs://b/missing.npz", read_bytes=explode, steps=_STEPS, l_null=_L, width=_WIDTH, examples=len(_NAMES)
        )


# --------------------------------------------------------------------------------------------------
# 2. The noise settings come from the golden-pinned helpers.
# --------------------------------------------------------------------------------------------------


def test_the_settings_are_the_four_the_probe_is_defined_on():
    assert TRANSFER_SETTINGS == (("global", 0), ("keyed", 0), ("keyed", 1), ("keyed", 2))
    assert [setting_label(*setting) for setting in TRANSFER_SETTINGS] == [
        "global_0",
        "keyed_0",
        "keyed_1",
        "keyed_2",
    ]


def test_the_global_setting_is_the_canonical_epsilon_zero():
    latents = transfer_start_latents(_NAMES, "global", 0, geometry=PRODUCTION_GEOMETRY.z_video)

    assert np.array_equal(np.asarray(latents[0]), np.asarray(global_noise(0)))
    assert all(np.array_equal(np.asarray(row), np.asarray(global_noise(0))) for row in latents)


@pytest.mark.parametrize("k", [0, 1, 2])
def test_the_keyed_settings_are_the_examples_own_draws(k):
    latents = transfer_start_latents(_NAMES, "keyed", k, geometry=PRODUCTION_GEOMETRY.z_video)

    for index, name in enumerate(_NAMES):
        assert np.array_equal(np.asarray(latents[index]), np.asarray(keyed_noise(name, k)))


def test_an_unknown_convention_is_refused():
    with pytest.raises(ValueError, match="convention"):
        transfer_start_latents(_NAMES, "fresh", 0, geometry=PRODUCTION_GEOMETRY.z_video)


# --------------------------------------------------------------------------------------------------
# 3. End to end: the rows replayed are the rows that were loaded.
# --------------------------------------------------------------------------------------------------


def test_every_setting_replays_the_loaded_nulls_unchanged():
    """**The off-by-one killer.** Each example's block is a distinct constant, so a crossover, a
    transpose or a reordering all show up as a different array reaching the replay."""
    captured = []
    loaded = _nulls()

    def replay(velocity_fn, z_start, z_i0, sigmas, nulls, base_context, *, guide_scale, **kwargs):
        captured.append(np.asarray(nulls, np.float32))
        return np.asarray(z_start, np.float32)

    report, _ = _run(replay=replay)

    assert len(captured) == len(TRANSFER_SETTINGS)
    for seen in captured:
        assert seen.shape == loaded.shape
        np.testing.assert_array_equal(seen, loaded)
    assert report["settings"] == [setting_label(*setting) for setting in TRANSFER_SETTINGS]


def test_each_setting_starts_from_its_own_noise():
    starts = {}

    def replay(velocity_fn, z_start, z_i0, sigmas, nulls, base_context, *, guide_scale, **kwargs):
        starts[len(starts)] = np.asarray(z_start, np.float32)
        return np.asarray(z_start, np.float32)

    _run(replay=replay)

    assert not np.array_equal(starts[0], starts[1])  # global(0) is not keyed(name, 0)
    assert not np.array_equal(starts[1], starts[2])  # keyed 0 is not keyed 1
    assert not np.array_equal(starts[2], starts[3])


def test_the_real_replay_runs_and_frame_zero_comes_back_pinned():
    """**The pin, observed.** Every other end-to-end test injects a replay stub; this one delegates to
    the module's actual operator and inspects what it returned. ``replay_with_nulls`` pins latent frame
    0 to ``z_i0`` at every step, so a rollout that lost the pin comes back with noise in frame 0."""
    from maxdiffusion.models.wan.null_inversion_wan import replay_with_nulls

    backend = _Backend()
    expected_z_i0 = backend.read_batch(_NAMES)[0].z_i0
    captured = []

    def recording(*args, **kwargs):
        out = replay_with_nulls(*args, **kwargs)
        captured.append(np.asarray(out, np.float32))
        return out

    _run(backend=backend, replay=recording)

    assert len(captured) == len(TRANSFER_SETTINGS)
    for rolled in captured:
        np.testing.assert_allclose(rolled[:, :, :1], expected_z_i0, rtol=0, atol=1e-5)


def test_the_default_replay_seam_is_the_deployment_operator():
    """And the seam itself is the deployment replay, not something with the same shape: swapping it
    is how the pin would disappear without any test noticing."""
    import inspect

    from maxdiffusion.models.wan.null_inversion_wan import replay_with_nulls

    assert inspect.signature(run_transfer_probe).parameters["replay"].default is replay_with_nulls


def test_the_table_scores_every_example_under_every_setting():
    report, sinks = _run()

    table = report["table"]
    assert sorted(table) == sorted(setting_label(*setting) for setting in TRANSFER_SETTINGS)
    for label, per_example in table.items():
        assert sorted(per_example) == sorted(_NAMES), label
        for name, per_seed in per_example.items():
            # exp_04's table shape, inherited whole: method -> name -> seed key -> metrics. Each
            # setting is a single noise draw, so it carries exactly SINGLE_SEED_KEY.
            assert list(per_seed) == [SINGLE_SEED_KEY], (label, name)
            metrics = per_seed[SINGLE_SEED_KEY]
            assert set(metrics) >= {"future_mse", "future_ssim"}, (label, name)
            assert np.isfinite(metrics["future_mse"]) and np.isfinite(metrics["future_ssim"])
    assert TRANSFER_NAME in sinks.json


def test_the_artifact_is_bound_to_the_run_that_produced_the_nulls():
    payload = _npz_bytes(_nulls())
    digest = __import__("hashlib").sha256(payload).hexdigest()

    report, sinks = _run(payload)

    provenance = report["provenance"]
    assert provenance["nulls_uri"].endswith("a3_nulls.npz")
    assert provenance["nulls_sha256"] == digest
    assert provenance["code_sha"] == "a" * 40 and provenance["manifest_hash"] == "m" * 64
    assert provenance["l_null"] == _L and provenance["guide_scale"] == 5.0
    assert provenance["cohort"] == "dev64" and provenance["names"] == list(_NAMES)
    assert sinks.json[TRANSFER_NAME]["provenance"] == provenance


def test_the_report_is_json_publishable():
    report, sinks = _run()

    assert json.loads(json.dumps(report, default=str)) == sinks.json[TRANSFER_NAME]


def test_the_artifact_lands_under_the_given_directory_only():
    _, sinks = _run(artifact_dir="gs://bucket/attempt-42")

    assert list(sinks.json) == [TRANSFER_NAME]


# --------------------------------------------------------------------------------------------------
# 4. Refusals.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("uri", ["", None])
def test_a_missing_nulls_uri_is_refused(uri):
    with pytest.raises(ValueError, match="null_transfer_nulls_uri"):
        _run(nulls_uri=uri)


def test_an_npz_whose_example_count_is_not_the_cohorts_is_refused():
    with pytest.raises(ValueError, match="examples"):
        _run(_npz_bytes(_nulls(examples=3)))


@pytest.mark.parametrize("field", ["code_sha", "manifest_hash"])
def test_an_artifact_without_its_provenance_is_refused(field):
    with pytest.raises(ValueError, match=field):
        _run(**{field: ""})


def test_a_cohort_larger_than_the_published_nulls_is_refused():
    """J1b published the first eight DEV examples; pointing the probe at the whole cohort is a
    mismatch, not a subset to be silently truncated."""
    plan = _plan(names=tuple(f"ep{i}_v0_s{i * 4:05d}" for i in range(6)))

    with pytest.raises(ValueError, match="examples"):
        _run(plan=plan, backend=_Backend(names=plan["names"]))


# --------------------------------------------------------------------------------------------------
# 5. The wiring: the mode exists, dispatches, and is launchable.
# --------------------------------------------------------------------------------------------------


def test_the_mode_joins_the_enum_and_the_dispatch():
    from maxdiffusion.null_adapter_modes import execute
    from maxdiffusion.run_wan_null_inversion import NULL_MODES

    assert "transfer_probe" in NULL_MODES
    assert NULL_MODES[:5] == ("capacity", "cache", "verify_replay", "adequacy_probe", "direct_opt")  # additive

    seen = {}
    backend, sinks = _Backend(), _Sinks()
    report, code = execute(
        "transfer_probe",
        _plan(),
        backend,
        sinks,
        artifact_dir="gs://bucket/j1c",
        nulls_uri="gs://bucket/j1b/a3_nulls.npz",
        manifest_hash="m" * 64,
        code_sha="a" * 40,
        read_bytes=lambda uri: seen.setdefault("uri", uri) and _npz_bytes(_nulls()) or _npz_bytes(_nulls()),
        sigmas=np.linspace(1.0, 0.0, _STEPS + 1).astype(np.float32),
    )

    assert code == 0 and report["mode"] == "transfer_probe"
    assert TRANSFER_NAME in sinks.json


def test_the_config_declares_the_key_the_mode_requires():
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parents[4]
    config = yaml.safe_load((root / "src/maxdiffusion/configs/base_wan_5b_null_inversion.yml").read_text())

    assert config["null_transfer_nulls_uri"] == ""  # empty by default; the mode refuses it loudly
    assert config["null_mode"] in ("capacity",)  # unchanged default


def test_the_mode_kwargs_carry_the_npz_uri_and_the_provenance():
    import types

    import yaml
    import pathlib

    from maxdiffusion.run_wan_null_inversion import mode_kwargs

    root = pathlib.Path(__file__).resolve().parents[4]
    keys = yaml.safe_load((root / "src/maxdiffusion/configs/base_wan_5b_null_inversion.yml").read_text())
    config = types.SimpleNamespace(
        **{
            **keys,
            "null_artifact_dir": "gs://b/run",
            "null_staging_dir": "",
            "code_sha": "a" * 40,
            "null_transfer_nulls_uri": "gs://b/j1b/a3_nulls.npz",
        }
    )
    plan = {"mode": "transfer_probe", "cohort": "dev64", "names": _NAMES, "decode_batch_size": 8}
    manifests = {"header": {"h": 1}, "dev64": {"rows": [{"name": name} for name in _NAMES]}}

    kwargs = mode_kwargs(config, plan, manifests, shards_for=lambda root: ())

    assert kwargs["nulls_uri"] == "gs://b/j1b/a3_nulls.npz"
    assert kwargs["code_sha"] == "a" * 40 and len(kwargs["manifest_hash"]) == 64
    assert kwargs["decode_batch_size"] == 8


def test_the_launcher_maps_the_env_onto_the_config_key():
    """exp_04's launcher style: an env default, an echo, and one CLI override line."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[4]
    source = (root / "bash_scripts/run_wan_null_inversion.sh").read_text()

    assert 'NULL_TRANSFER_NULLS_URI="${NULL_TRANSFER_NULLS_URI:-}"' in source
    assert 'null_transfer_nulls_uri="${NULL_TRANSFER_NULLS_URI}"' in source
    assert 'echo "NULL_TRANSFER_NULLS_URI=${NULL_TRANSFER_NULLS_URI}"' in source
    assert source.index("NULL_TRANSFER_NULLS_URI=") < source.index("run_wan_null_inversion.py")
