"""exp_04 R4c — ``verify_replay``: the independent verifier that closes the artifact contract.

The claim under test is the one the whole caching phase rests on: **a published record, together with
its shard provenance header, regenerates its own claimed output.** The reference verifier
(``third_party/Wan2.2/scripts/verify_reconstruction_from_null.py``, submodule pin f370228) makes the
same point by what it refuses to open -- no ground-truth frames, no inversion trajectory, no
scheduler RNG, only ``I_0`` and ``null_embeddings.pt``. This port keeps that discipline mechanically:
the verifier touches ``nulls``, ``z_start``, ``z_i0``, ``expected_final_latent`` and ``latent_dtype``,
and nothing else on the record.

Two tests carry the security argument:

- **The tampered-nulls case.** A record whose nulls were altered *and whose hash was recomputed* is
  perfectly valid to the reader -- every byte matches its digest. It is a well-formed artifact that
  lies. Only replaying it catches that, which is exactly the value the verifier adds over
  ``record_from_bytes``.
- **The restricted-proxy case.** The verifier runs against an object exposing only the five fields it
  is allowed to read. If it ever reached for ``z_video`` (the trainer's copy of the answer) the call
  would raise ``AttributeError`` instead of quietly passing, so "never reads the ground truth" is
  proven by construction rather than by inspection.
"""

from __future__ import annotations

import dataclasses
import types

import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion.models.wan.null_inversion_wan import base_context_fingerprint, replay_with_nulls
from maxdiffusion.models.wan.side_adapter_wan import build_rollout_sigmas
from maxdiffusion.null_adapter_records import (
    ProvenanceHeader,
    _Geometry,
    _make_record,
    _record_from_bytes,
    _record_to_bytes,
    make_record,
)
from maxdiffusion.null_adapter_verify import ALLOWED_RECORD_FIELDS, verify_replay


# The sigma grid is fixed at 25 sampler steps by plan §3 regardless of latent size, so the tiny seam
# shrinks the tensors but keeps 25 nulls and 25 recorded losses.
_STEPS, _L, _DIM, _S = 25, 3, 8, 8
_GEOMETRY = _Geometry(
    z_video=(2, 3, 4, 6),
    z_i0=(2, 1, 4, 6),
    actions=(5, 7),
    nulls=(_STEPS, _L, _DIM),
    per_step_final_losses=(_STEPS,),
)
_REVISION = "Wan2.2-TI2V-5B@f370228"
_GUIDE_SCALE = 5.0
_SHIFT = 5.0  # the sigma grid's shift; numerically equal to the CFG weight, semantically unrelated


def _forbidden_velocity(*args, **kwargs):
    """Any provenance failure must be raised before a single model forward."""
    raise AssertionError("provenance must be checked before any compute")


class _RestrictedRecord:
    """A record proxy that exposes exactly ``ALLOWED_RECORD_FIELDS`` and nothing else.

    ``AssertionError``, deliberately, not ``AttributeError``: a verifier written as
    ``getattr(record, "z_video", None)`` would swallow the latter and quietly read the answer. This
    also blocks ``__dict__``, so ``vars()``-style inspection cannot route around the allowlist.
    """

    def __init__(self, record, allowed=ALLOWED_RECORD_FIELDS):
        object.__setattr__(self, "_allowed", tuple(allowed))
        object.__setattr__(self, "_values", {field: getattr(record, field) for field in allowed})
        object.__setattr__(self, "_log", [])

    def __getattribute__(self, name):
        allowed = object.__getattribute__(self, "_allowed")
        if name not in allowed:
            raise AssertionError(f"the verifier must not touch record.{name}")
        object.__getattribute__(self, "_log").append(name)
        return object.__getattribute__(self, "_values")[name]


def _access_log(proxy):
    return object.__getattribute__(proxy, "_log")


def _velocity(seed=3):
    keys = np.random.default_rng(seed)
    weights = keys.standard_normal((_L, _DIM), dtype=np.float32)
    pattern = keys.standard_normal(_GEOMETRY.z_video, dtype=np.float32)

    def velocity_fn(z, timestep_2d, context):
        scale = jnp.sum(context[:, :_L] * weights, axis=(1, 2))
        return scale[:, None, None, None, None] * pattern + 0.05 * z

    return velocity_fn


def _artifacts(latent_dtype="fp32", seed=0, **record_overrides):
    """A consistent (record, header, base_context, velocity_fn) quadruple, built like the runner will."""
    rng = np.random.default_rng(seed)
    velocity_fn = _velocity()
    sigmas = np.asarray(build_rollout_sigmas(_STEPS, _SHIFT, 0.0, 1.0))
    base_context = rng.standard_normal((_S, _DIM), dtype=np.float32)
    shapes = _GEOMETRY.shapes()
    arrays = {field: rng.standard_normal(shapes[field], dtype=np.float32) for field in shapes}

    # The runner must compute expected_final_latent from the values the record will STORE, not from
    # the pre-cast inputs: z_i0 is always kept at the source fp16, so a replay from the fp32 original
    # misses by ~5e-4 and could never verify exactly. Round-trip through the storage dtypes first.
    def _as_stored(field):
        dtype = np.float16 if field == "z_i0" or latent_dtype == "fp16" else np.float32
        return arrays[field].astype(dtype).astype(np.float32)

    expected = np.asarray(
        replay_with_nulls(
            velocity_fn,
            _as_stored("z_start")[None],
            _as_stored("z_i0")[None],
            jnp.asarray(sigmas),
            _as_stored("nulls"),
            base_context,
            guide_scale=_GUIDE_SCALE,
        )
    )[0]
    fields = {
        "name": "droid_ep_000001/w0",
        "ordinal": 3,
        "split": "dev",
        "episode": "droid_ep_000001",
        "latent_dtype": latent_dtype,
        "noise_convention": "keyed",
        "arm": "A1",
        "final_future_mse": 0.0123,
        **arrays,
        "expected_final_latent": expected,
    }
    fields.update(record_overrides)
    record = _make_record(geometry=_GEOMETRY, **fields)
    header = ProvenanceHeader(
        manifest_hash="a" * 64,
        code_sha="b" * 40,
        model_revision=_REVISION,
        sigma_vector=sigmas,
        guide_scale=_GUIDE_SCALE,
        base_context_fingerprint=base_context_fingerprint(base_context),
        optimization_config={"inner_iters": 10, "lr": 0.01},
        dtype_policy=latent_dtype,
        l_null=_L,
    )
    return record, header, base_context, velocity_fn


def _verify(
    record,
    header,
    base_context,
    velocity_fn,
    *,
    atol=1e-5,
    revision=_REVISION,
    w=_GUIDE_SCALE,
    convention="keyed",
    arm="A1",
):
    return verify_replay(
        record,
        header,
        velocity_fn,
        base_context,
        expected_model_revision=revision,
        expected_guide_scale=w,
        expected_noise_convention=convention,
        expected_arm=arm,
        atol=atol,
    )


def test_a_consistent_record_and_header_verify():
    record, header, base_context, velocity_fn = _artifacts()

    assert _verify(record, header, base_context, velocity_fn, atol=0.0) is None


def test_verification_survives_a_serialization_round_trip():
    record, header, base_context, velocity_fn = _artifacts()

    restored = _record_from_bytes(_record_to_bytes(record, _GEOMETRY), _GEOMETRY)

    _verify(restored, header, base_context, velocity_fn, atol=0.0)


def test_fp16_storage_verifies_within_a_tolerance_but_not_exactly():
    """fp16 rounding is real and the atol argument is where the caller declares what it accepts."""
    record, header, base_context, velocity_fn = _artifacts(latent_dtype="fp16")

    _verify(record, header, base_context, velocity_fn, atol=1e-2)
    with pytest.raises(ValueError, match="replay does not reproduce"):
        _verify(record, header, base_context, velocity_fn, atol=0.0)


@pytest.mark.parametrize(
    "mutate_header, message",
    [
        (lambda h: {"sigma_vector": np.concatenate([h.sigma_vector[:5] + 1e-3, h.sigma_vector[5:]])}, "sigma vector"),
        (lambda h: {"guide_scale": h.guide_scale + 0.5}, "guide_scale"),  # caught as provenance, not as drift
        (lambda h: {"base_context_fingerprint": "d" * 64}, "base_context fingerprint"),
        (lambda h: {"l_null": h.l_null + 1}, "l_null"),
        (lambda h: {"model_revision": "Wan2.2-TI2V-5B@deadbee"}, "model_revision"),
    ],
)
def test_provenance_mismatches_are_refused(mutate_header, message):
    """The velocity callback raises if invoked, so each of these pins provenance-before-compute."""
    record, header, base_context, _ = _artifacts()

    with pytest.raises(ValueError, match=message):
        _verify(record, dataclasses.replace(header, **mutate_header(header)), base_context, _forbidden_velocity)


def test_a_different_base_context_is_refused_before_any_replay():
    record, header, base_context, _ = _artifacts()
    other_context = base_context + 1.0

    with pytest.raises(ValueError, match="base_context fingerprint"):
        _verify(record, header, other_context, _forbidden_velocity)


def test_a_record_whose_latent_dtype_lies_is_refused():
    """The reader enforces this too, but the verifier accepts records from anywhere -- including one
    a caller built in memory -- so it re-asserts rather than assuming a reader ran first."""
    record, header, base_context, _ = _artifacts(latent_dtype="fp16")
    values = {field: getattr(record, field) for field in ALLOWED_RECORD_FIELDS}
    liar = types.SimpleNamespace(**{**values, "latent_dtype": "fp32"})  # ... arrays still fp16

    with pytest.raises(ValueError, match="does not describe its stored arrays"):
        _verify(liar, dataclasses.replace(header, dtype_policy="fp32"), base_context, _forbidden_velocity)


def test_a_malformed_header_is_refused():
    record, header, base_context, _ = _artifacts()

    with pytest.raises(ValueError, match="unsupported header schema_version"):
        _verify(record, dataclasses.replace(header, schema_version=99), base_context, _forbidden_velocity)


@pytest.mark.parametrize(
    "kwargs, header_overrides, message",
    [
        ({}, {"dtype_policy": "fp32"}, "dtype_policy"),  # header and record disagree about storage
        ({"convention": "global"}, {}, "noise_convention"),
        ({"arm": "A2"}, {}, "record arm"),
    ],
)
def test_pair_level_provenance_is_refused_before_compute(kwargs, header_overrides, message):
    """The record and its header must describe the same run, and both must match what was declared."""
    record, header, base_context, _ = _artifacts(latent_dtype="fp16")

    with pytest.raises(ValueError, match=message):
        _verify(record, dataclasses.replace(header, **header_overrides), base_context, _forbidden_velocity, **kwargs)


def test_tampered_nulls_with_a_recomputed_hash_are_caught_by_the_replay():
    """The record is internally valid -- the reader accepts it. Only replaying it exposes the lie."""
    record, header, base_context, velocity_fn = _artifacts()
    lying_nulls = np.asarray(record.nulls).copy()
    lying_nulls[7] += 0.5

    lying = _make_record(
        geometry=_GEOMETRY,
        **{
            **{f.name: getattr(record, f.name) for f in dataclasses.fields(record) if f.name != "nulls"},
            "nulls": lying_nulls,
        },
    )
    _record_from_bytes(_record_to_bytes(lying, _GEOMETRY), _GEOMETRY)  # a well-formed artifact ...

    with pytest.raises(ValueError, match="replay does not reproduce"):  # ... that lies
        _verify(lying, header, base_context, velocity_fn)


def test_tampered_expected_final_latent_is_caught_by_the_reader():
    """The other half of the contract: the reader's hash covers what the verifier compares against."""
    record, _, _, _ = _artifacts()
    tampered = np.asarray(record.expected_final_latent).copy()
    tampered[0, 0, 0, 0] += np.float32(1.0)

    with pytest.raises(ValueError, match="expected_final_latent sha256 mismatch"):
        _record_to_bytes(dataclasses.replace(record, expected_final_latent=tampered), _GEOMETRY)


def test_ground_truth_fields_are_never_read():
    """z_video and actions are the trainer's payload; a verifier that consults them proves nothing."""
    record, header, base_context, velocity_fn = _artifacts()
    garbage = _make_record(
        geometry=_GEOMETRY,
        **{
            **{f.name: getattr(record, f.name) for f in dataclasses.fields(record)},
            "z_video": np.full(_GEOMETRY.z_video, 7.5, np.float32),
            "actions": np.full(_GEOMETRY.actions, -3.25, np.float32),
            "per_step_final_losses": np.full((_STEPS,), 99.0, np.float32),
            "final_future_mse": 42.0,
        },
    )

    _verify(garbage, header, base_context, velocity_fn, atol=0.0)


def test_verifier_touches_only_the_fields_it_is_allowed_to():
    """Structural proof: every read outside the allowlist aborts the run rather than returning data."""
    record, header, base_context, velocity_fn = _artifacts()
    proxy = _RestrictedRecord(record)

    _verify(proxy, header, base_context, velocity_fn, atol=0.0)

    assert set(_access_log(proxy)) <= set(ALLOWED_RECORD_FIELDS)
    assert {"nulls", "z_start", "z_i0", "expected_final_latent"} <= set(_access_log(proxy))
    assert not {"z_video", "actions", "per_step_final_losses", "final_future_mse"} & set(ALLOWED_RECORD_FIELDS)


def test_the_proxy_defeats_optional_lookups_and_vars():
    """Guard the guard: the two ways a verifier could sneak a look at the answer both abort."""
    record, _, _, _ = _artifacts()
    proxy = _RestrictedRecord(record)

    with pytest.raises(AssertionError, match="must not touch record.z_video"):
        getattr(proxy, "z_video", None)  # AttributeError would have been swallowed by the default
    with pytest.raises(AssertionError, match="must not touch record.__dict__"):
        vars(proxy)
    with pytest.raises(AssertionError, match="must not touch record.final_future_mse"):
        proxy.final_future_mse


def test_atol_is_the_only_slack_and_it_is_required():
    record, header, base_context, velocity_fn = _artifacts()
    drifted = np.asarray(record.expected_final_latent).copy()
    drifted[0, 0, 0, 0] += np.float32(1e-3)
    lying = _make_record(
        geometry=_GEOMETRY,
        **{
            **{
                f.name: getattr(record, f.name)
                for f in dataclasses.fields(record)
                if f.name != "expected_final_latent"
            },
            "expected_final_latent": drifted,
        },
    )

    _verify(lying, header, base_context, velocity_fn, atol=2e-3)
    with pytest.raises(ValueError, match="replay does not reproduce"):
        _verify(lying, header, base_context, velocity_fn, atol=5e-4)
    with pytest.raises(TypeError):
        verify_replay(
            record,
            header,
            velocity_fn,
            base_context,
            expected_model_revision=_REVISION,
            expected_guide_scale=_GUIDE_SCALE,
        )


def test_production_geometry_smoke():
    """One pass at deployment shapes, so the tiny seam is not the only thing ever exercised."""
    rng = np.random.default_rng(11)
    sigmas = np.asarray(build_rollout_sigmas(_STEPS, _SHIFT, 0.0, 1.0))
    shapes = _Geometry().shapes()
    base_context = rng.standard_normal((512, 4096), dtype=np.float32) * 0.01
    arrays = {field: np.zeros(shapes[field], np.float32) for field in shapes}
    arrays["nulls"] = rng.standard_normal(shapes["nulls"], dtype=np.float32) * 0.01

    def velocity_fn(z, timestep_2d, context):
        return 0.01 * jnp.mean(context[:, :16], axis=(1, 2))[:, None, None, None, None] + 0.05 * z

    expected = np.asarray(
        replay_with_nulls(
            velocity_fn,
            arrays["z_start"][None],
            arrays["z_i0"][None],
            jnp.asarray(sigmas),
            arrays["nulls"],
            base_context,
            guide_scale=_GUIDE_SCALE,
        )
    )[0]
    record = make_record(
        name="n",
        ordinal=0,
        split="dev",
        episode="e",
        latent_dtype="fp32",
        noise_convention="keyed",
        arm="A1",
        final_future_mse=0.5,
        **{**arrays, "expected_final_latent": expected},
    )
    header = ProvenanceHeader(
        manifest_hash="a" * 64,
        code_sha="b" * 40,
        model_revision=_REVISION,
        sigma_vector=sigmas,
        guide_scale=_GUIDE_SCALE,
        base_context_fingerprint=base_context_fingerprint(base_context),
        optimization_config={"inner_iters": 10},
        dtype_policy="fp32",
        l_null=16,
    )

    verify_replay(
        record,
        header,
        velocity_fn,
        base_context,
        expected_model_revision=_REVISION,
        expected_guide_scale=_GUIDE_SCALE,
        expected_noise_convention="keyed",
        expected_arm="A1",
        atol=0.0,
    )
