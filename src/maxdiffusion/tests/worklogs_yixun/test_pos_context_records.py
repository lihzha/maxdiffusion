"""exp_05 S5 — the positive-slot record codec and its states-fidelity policy (plan §4-P2').

The schema is exp_04's with ``nulls`` replaced by ``pos_embeds [25, 8, 4096]`` and the per-step
teacher-forcing states ``z_bar_states [25, 48, 9, 12, 20]`` added, both governed by ``latent_dtype``.
Every R4b-era property is re-tested **on this schema** rather than assumed to transfer: the sibling's
tests cannot see a field the sibling does not have, and ``z_bar_states`` is 69% of the record's bytes.

**THE BF16 FINDING (read this before trusting plan §4-P2's F7 expectation).** The plan expects the
states-fidelity gate to pass trivially, "since fp16 -> bf16 is value-preserving at bf16 precision for
our range". **It is not.** fp32 -> fp16 -> bf16 is a *double rounding*, and it differs from a direct
fp32 -> bf16 cast whenever the fp16 step lands on a bf16 midpoint that round-half-even then resolves
away from the true value. The three tests named below are the evidence, and they state exactly what
was measured -- the first version of this file overstated all three claims (S5 review, finding 1):

* ``..._premise_is_false_in_the_normal_range`` -- ~6.2-6.3% of fp16-**normal** latent-like values
  differ (6.217% here over 200k, 6.27% in the reviewer's 6M sweep), each by exactly one bf16 ulp.
* ``test_subnormal_inputs_diverge_by_far_more_than_one_ulp`` -- "always one ulp" is **false** below
  2^-14, where fp16's absolutely-spaced subnormal grid is far coarser than bf16's relative one:
  deterministic witnesses diverge by 4, 64 and 13056 bf16 ulps. Their *absolute* error nonetheless
  stays under ~6e-8, which is why the gate reports ``max_abs_delta`` too and treats underflow as
  benign while overflow is fatal.
* ``test_the_corrected_fp16_range_thresholds`` -- inf at and above **65520** (not "above 65504"), zero
  at and below **2^-25** (not "below ~6e-8").

So the bit-identity path is expected to FAIL on real data and the feature-tolerance path is
load-bearing -- and the gate that judges it fails closed, per the S5 review's finding 2.
"""

from __future__ import annotations

import numpy as np
import pytest

from maxdiffusion.null_adapter_records import LATENT_DTYPES, SOURCE_DTYPES
from maxdiffusion.pos_context_records import (
    POS_ARRAY_FIELDS,
    POS_LATENT_SCOPED_FIELDS,
    PRODUCTION_POS_GEOMETRY,
    STATES_FEATURE_MAX_ABS_DELTA,
    PosContextRecord,
    _PosGeometry,
    _make_pos_record,
    _pos_record_from_bytes,
    _pos_record_to_bytes,
    make_pos_record,
    pos_record_from_bytes,
    pos_record_to_bytes,
    record_storage_bytes,
    states_fidelity_check,
    to_bfloat16,
)


# A tiny geometry with every axis distinct, exercising the same validation path as production.
_TINY = _PosGeometry(
    z_video=(2, 3, 2, 2),
    z_i0=(2, 1, 2, 2),
    actions=(4, 7),
    pos_embeds=(3, 8, 5),
    z_bar_states=(3, 2, 3, 2, 2),
    per_step_final_losses=(3,),
)


def _fields(geometry=_TINY, latent_dtype="fp16", seed=0):
    rng = np.random.default_rng(seed)
    shapes = geometry.shapes()
    arrays = {field: rng.standard_normal(shapes[field]).astype(np.float32) for field in POS_ARRAY_FIELDS}
    return {
        "name": "episode-000",
        "ordinal": 7,
        "split": "dev",
        "episode": "ep-42",
        "latent_dtype": latent_dtype,
        "noise_convention": "keyed",
        "arm": "B1",
        "final_future_mse": 0.125,
        **arrays,
    }


def _record(geometry=_TINY, **overrides):
    fields = _fields(geometry, **overrides)
    return _make_pos_record(geometry=geometry, **fields)


def _roundtrip(record, geometry=_TINY):
    return _pos_record_from_bytes(_pos_record_to_bytes(record, geometry), geometry)


# --------------------------------------------------------------------------------------------------
# 1. The schema and its round trip.
# --------------------------------------------------------------------------------------------------


def test_the_schema_is_exp_04_s_with_pos_embeds_and_the_states_added():
    """``nulls`` is gone, ``pos_embeds`` and ``z_bar_states`` are present, and both are latent-scoped."""
    assert "nulls" not in POS_ARRAY_FIELDS
    assert set(POS_LATENT_SCOPED_FIELDS) == {"pos_embeds", "z_start", "expected_final_latent", "z_bar_states"}
    assert set(POS_ARRAY_FIELDS) == set(SOURCE_DTYPES) | set(POS_LATENT_SCOPED_FIELDS) | {"per_step_final_losses"}
    assert PRODUCTION_POS_GEOMETRY.pos_embeds == (25, 8, 4096)  # plan §4-P2': L_pos = 8, T5 width
    assert PRODUCTION_POS_GEOMETRY.z_bar_states == (25, 48, 9, 12, 20)  # one state per sampler step
    assert {f.name for f in PosContextRecord.__dataclass_fields__.values()} >= set(POS_ARRAY_FIELDS)


@pytest.mark.parametrize("latent_dtype", ["fp16", "fp32"])
def test_round_trip_is_bitwise_per_field(latent_dtype):
    record = _record(latent_dtype=latent_dtype)

    parsed = _roundtrip(record)

    for field in POS_ARRAY_FIELDS:
        original, read = getattr(record, field), getattr(parsed, field)
        assert read.dtype == original.dtype and read.shape == original.shape, field
        np.testing.assert_array_equal(read.view(np.uint8), original.view(np.uint8), err_msg=field)
    for field in ("name", "ordinal", "split", "episode", "latent_dtype", "noise_convention", "arm"):
        assert getattr(parsed, field) == getattr(record, field), field
    assert parsed.expected_final_latent_sha256 == record.expected_final_latent_sha256
    assert parsed.final_future_mse == record.final_future_mse


@pytest.mark.parametrize("latent_dtype", ["fp16", "fp32"])
def test_the_latent_dtype_scope_covers_the_states_and_spares_the_source_arrays(latent_dtype):
    """The fp32 fallback must flip ``z_bar_states`` too -- it is the field the plan added, and 69% of
    the bytes. ``z_i0``/``z_video``/``actions`` keep their source dtypes under every policy, because
    that is what the source data *is*, not a precision choice."""
    record = _record(latent_dtype=latent_dtype)

    for field in POS_LATENT_SCOPED_FIELDS:
        assert getattr(record, field).dtype == LATENT_DTYPES[latent_dtype], field
    for field, dtype in SOURCE_DTYPES.items():
        assert getattr(record, field).dtype == dtype, field
    assert record.per_step_final_losses.dtype == np.dtype("<f4")


def test_arrays_are_frozen_and_bytes_are_deterministic():
    record = _record()

    for field in POS_ARRAY_FIELDS:
        assert not getattr(record, field).flags.writeable, field
        with pytest.raises(ValueError):
            getattr(record, field)[...] = 0
    assert _pos_record_to_bytes(record, _TINY) == _pos_record_to_bytes(record, _TINY)
    assert not _roundtrip(record).z_bar_states.flags.writeable


def test_the_hash_describes_the_stored_bytes_not_the_input():
    """``make`` casts first and hashes the cast bytes, so a reader's recomputation agrees at fp16."""
    fields = _fields(latent_dtype="fp16")
    record = _make_pos_record(geometry=_TINY, **fields)

    from maxdiffusion.null_adapter_records import sha256_of_array

    assert record.expected_final_latent_sha256 == sha256_of_array(record.expected_final_latent)
    assert record.expected_final_latent_sha256 != sha256_of_array(fields["expected_final_latent"])  # fp32 input
    assert _roundtrip(record).expected_final_latent_sha256 == record.expected_final_latent_sha256


def test_the_production_geometry_round_trips():
    """The artifact boundary itself, at full size -- 8 members, ~7 MiB of fp16."""
    record = make_pos_record(**_fields(PRODUCTION_POS_GEOMETRY))

    parsed = pos_record_from_bytes(pos_record_to_bytes(record))

    assert parsed.pos_embeds.shape == (25, 8, 4096)
    assert parsed.z_bar_states.shape == (25, 48, 9, 12, 20)
    np.testing.assert_array_equal(parsed.z_bar_states.view(np.uint8), record.z_bar_states.view(np.uint8))


# --------------------------------------------------------------------------------------------------
# 2. Fail-closed reads and writes, on THIS schema.
# --------------------------------------------------------------------------------------------------


def _rewrite(blob, replace=None, meta_update=None, mutate=None):
    """Rebuild a record blob with array members replaced (and their declared shapes patched to match).

    Patching the shape metadata as well is what makes the tampered blob **self-consistent**: it agrees
    with its own declaration, so only validation against the geometry can still reject it.
    """
    import io
    import json
    import zipfile

    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    meta = json.loads(bytes(arrays.pop("__meta__")).decode("utf-8"))
    for name, array in (replace or {}).items():
        arrays[name] = array
        meta["shapes"][name] = list(array.shape)
    meta.update(meta_update or {})
    if mutate is not None:
        mutate(meta, arrays)
    arrays["__meta__"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name, array in arrays.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.ascontiguousarray(array), allow_pickle=False)
            archive.writestr(f"{name}.npy", payload.getvalue())
    return buffer.getvalue()


def test_a_truncated_states_member_is_rejected_even_when_the_blob_agrees_with_itself():
    """The byte-length check must cover ``z_bar_states`` -- the field a truncated write would most
    likely damage, and one the sibling's validator has never heard of.

    Both the member and its declared shape are shrunk, so the blob is internally consistent and the
    only thing standing between it and the trainer is validation against the geometry.
    """
    record = _record()
    truncated = np.asarray(record.z_bar_states)[..., :1]
    tampered = _rewrite(_pos_record_to_bytes(record, _TINY), replace={"z_bar_states": truncated})

    assert truncated.shape != _TINY.z_bar_states
    with pytest.raises(ValueError, match="z_bar_states"):
        _pos_record_from_bytes(tampered, _TINY)


def test_a_tampered_hash_is_rejected_on_write_and_on_read():
    import dataclasses

    record = _record()
    tampered = dataclasses.replace(record, expected_final_latent_sha256="0" * 64)

    with pytest.raises(ValueError, match="sha256 mismatch"):
        _pos_record_to_bytes(tampered, _TINY)


def test_a_mutated_array_invalidates_the_record_hash():
    """``_freeze`` blocks the easy path; a reader that unfroze a copy must still be caught."""
    import dataclasses

    record = _record()
    corrupted = np.array(record.expected_final_latent)
    corrupted[(0,) * corrupted.ndim] += np.array(1.0, corrupted.dtype)

    with pytest.raises(ValueError, match="sha256 mismatch"):
        _pos_record_to_bytes(dataclasses.replace(record, expected_final_latent=corrupted), _TINY)


@pytest.mark.parametrize("field", ["pos_embeds", "z_bar_states", "z_video"])
def test_non_finite_data_is_refused(field):
    fields = _fields()
    poisoned = np.array(fields[field])
    poisoned.reshape(-1)[0] = np.inf

    with pytest.raises(ValueError, match=f"{field} must be finite"):
        _make_pos_record(geometry=_TINY, **{**fields, field: poisoned})


@pytest.mark.parametrize("field", ["pos_embeds", "z_bar_states"])
def test_a_wrong_shape_is_refused(field):
    fields = _fields()

    with pytest.raises(ValueError, match=f"{field} must have shape"):
        _make_pos_record(geometry=_TINY, **{**fields, field: np.zeros((1, 1), np.float32)})


def test_unknown_and_missing_archive_members_are_rejected():
    import io
    import zipfile

    record = _record()
    blob = _pos_record_to_bytes(record, _TINY)
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(blob)) as source, zipfile.ZipFile(buffer, "w") as extended:
        for item in source.infolist():
            extended.writestr(item, source.read(item.filename))
        extended.writestr("nulls.npy", b"whatever")  # the sibling's field has no business here

    with pytest.raises(ValueError, match="archive members"):
        _pos_record_from_bytes(buffer.getvalue(), _TINY)


def test_duplicate_archive_members_are_rejected():
    import io
    import warnings
    import zipfile

    blob = _pos_record_to_bytes(_record(), _TINY)
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(blob)) as source, zipfile.ZipFile(buffer, "w") as duplicated:
        for item in source.infolist():
            duplicated.writestr(item, source.read(item.filename))
        with warnings.catch_warnings():  # zipfile warns about the duplicate we are deliberately making
            warnings.simplefilter("ignore", UserWarning)
            duplicated.writestr("z_bar_states.npy", source.read("z_bar_states.npy"))

    with pytest.raises(ValueError, match="duplicated"):
        _pos_record_from_bytes(buffer.getvalue(), _TINY)


def test_an_unknown_metadata_key_is_rejected():
    """``l_null`` is the sibling's key: a positive-slot record must not silently accept it."""
    tampered = _rewrite(_pos_record_to_bytes(_record(), _TINY), meta_update={"l_null": 16})

    with pytest.raises(ValueError, match="metadata keys"):
        _pos_record_from_bytes(tampered, _TINY)


def _drop_member(meta, arrays, name):
    arrays.pop(name)
    meta["shapes"].pop(name)


@pytest.mark.parametrize(
    "mutate, message",
    [
        # The four R4b-style namespace closures, restated on THIS schema (S5 review, finding 3).
        (lambda meta, arrays: _drop_member(meta, arrays, "z_bar_states"), "archive members"),
        (lambda meta, arrays: meta.pop("arm"), "metadata keys"),
        (lambda meta, arrays: meta["shapes"].update({"nulls": [25, 16, 4096]}), "shape keys"),
        (lambda meta, arrays: meta["shapes"].pop("pos_embeds"), "shape keys"),
    ],
)
def test_the_record_namespaces_are_closed_on_both_sides(mutate, message):
    """Members, metadata keys and shape keys are each an exact set: a record that is missing one or
    carries an extra is a different schema, and a reader that shrugged would hand the trainer a
    record whose meaning it never checked."""
    tampered = _rewrite(_pos_record_to_bytes(_record(), _TINY), mutate=mutate)

    with pytest.raises(ValueError, match=message):
        _pos_record_from_bytes(tampered, _TINY)


def test_a_geometry_whose_states_disagree_with_z_video_is_refused():
    """``z_bar_states`` is ``(N, *z_video)`` by construction; a table where it is not is a typo that
    would otherwise be validated against faithfully for the rest of the job."""
    with pytest.raises(ValueError, match="z_bar_states"):
        _PosGeometry(**{**_TINY.as_dict(), "z_bar_states": (3, 9, 9, 9, 9)})
    with pytest.raises(ValueError, match="one entry per sampler step"):  # N disagrees across fields
        _PosGeometry(**{**_TINY.as_dict(), "per_step_final_losses": (4,)})


# --------------------------------------------------------------------------------------------------
# 3. THE BF16 FINDING and the states-fidelity policy.
# --------------------------------------------------------------------------------------------------


def test_the_module_s_bf16_rounding_is_the_one_jax_uses():
    """``to_bfloat16`` must be the model's own conversion, or the gate judges the wrong quantity."""
    import jax.numpy as jnp

    values = np.random.default_rng(5).standard_normal(4096).astype(np.float32)

    np.testing.assert_array_equal(
        to_bfloat16(values).view(np.uint16), np.asarray(jnp.asarray(values).astype(jnp.bfloat16)).view(np.uint16)
    )


def _bf16_bits(values):
    return to_bfloat16(values).view(np.uint16).astype(np.int64)


def _via_fp16_bits(values):
    with np.errstate(over="ignore"):
        return _bf16_bits(np.asarray(values, np.float32).astype(np.float16).astype(np.float32))


def test_the_bf16_value_preservation_premise_is_false_in_the_normal_range():
    """**THE FINDING, part 1.** plan §4-P2's F7 expects bit-identity to hold trivially because
    "fp16 -> bf16 is value-preserving at bf16 precision". It is a double rounding, and it is not.

    ``1 + 2^-8 + 2^-11`` is strictly above the bf16 midpoint, so a direct cast rounds it UP; the fp16
    step lands exactly ON that midpoint and round-half-even then takes it DOWN. For fp16-**normal**
    inputs the disagreement is exactly one bf16 ulp -- but it is not rare: 6.217% of 200k N(0,1)
    values here, 6.233% over 1M, and 6.27% in the S5 reviewer's independent 6M sweep.
    """
    witness = np.float32(1.0 + 2**-8 + 2**-11)
    assert float(to_bfloat16(witness)) == 1.0078125
    assert float(to_bfloat16(witness.astype(np.float16).astype(np.float32))) == 1.0

    sampled = np.random.default_rng(0).standard_normal(200_000).astype(np.float32)
    # Restricted to fp16-NORMAL inputs *by construction*: N(0,1) does put a handful of samples into
    # the subnormal region (11 of these 200k), and those are the next test's subject, not this one's.
    values = sampled[np.abs(sampled) >= 2.0**-14]
    divergence = np.abs(_bf16_bits(values) - _via_fp16_bits(values))

    assert values.size > 199_000 and np.all(np.abs(values) >= 2.0**-14)
    assert 0.060 < float(np.mean(divergence != 0)) < 0.065  # measured 6.217%; reviewer 6.27% over 6M
    assert int(divergence.max()) == 1  # one ulp -- but ONLY because every input here is normal


@pytest.mark.parametrize(
    "value, expected_ulps",
    [
        (24.5 * 2.0**-24, 4),  # fp16's subnormal grid (spacing 2^-24) is far coarser than bf16's here
        (1.5 * 2.0**-25, 64),
        (2.0**-25, 13056),  # ties-to-even sends it to zero, so the whole exponent field disagrees
    ],
)
def test_subnormal_inputs_diverge_by_far_more_than_one_ulp(value, expected_ulps):
    """**THE FINDING, part 2 (S5 review, finding 1 -- my "always exactly 1 ulp" claim was WRONG).**

    Below 2^-14 fp16 goes subnormal: its grid is spaced 2^-24 *absolutely*, while bf16 keeps ~2^-8
    *relative* precision all the way down. So the double rounding stops being a 1-ulp affair. These
    witnesses are deterministic rather than sampled -- the reviewer observed 2-35 bit spans from
    N(0,1) tails, and the last one is the extreme where fp16 annihilates the value outright.

    The absolute error nonetheless stays under 2^-24 (~6e-8): bit distance overstates the harm down
    here, which is why the gate reports ``max_abs_delta`` beside ``max_ulp_delta`` and why underflow
    is not treated as fatal while overflow is.
    """
    value = np.float32(value)

    assert abs(float(value)) < 2.0**-14, "the witness must be in the fp16 SUBNORMAL region to mean anything"
    assert int(np.abs(_bf16_bits(value) - _via_fp16_bits(value))) == expected_ulps
    assert abs(float(to_bfloat16(value)) - float(np.float32(value).astype(np.float16))) < 2.0**-24


def test_the_corrected_fp16_range_thresholds():
    """**THE FINDING, part 3 (S5 review, finding 1 -- my thresholds were wrong).**

    Overflow is at **65520**, the round-to-nearest halfway, not "above 65504": 65519 still rounds down
    to the largest finite fp16. Underflow is at **2^-25** (~2.98e-8), the ties-to-even halfway, not
    "below ~6e-8": 2^-25 goes to zero, but 2^-25 * 1.001 rounds up to 2^-24.
    """
    with np.errstate(over="ignore"):
        assert np.float32(65519.0).astype(np.float16) == np.float16(65504.0)
        assert np.isinf(np.float32(65520.0).astype(np.float16))
        assert np.isinf(np.float32(70000.0).astype(np.float16)) and np.isfinite(to_bfloat16(np.float32(70000.0)))

    assert np.float32(2.0**-25).astype(np.float16) == 0.0
    assert np.float32(2.0**-25 * 1.001).astype(np.float16) == np.float16(2.0**-24)
    assert to_bfloat16(np.float32(2.0**-25)) != 0.0


# The gate runs over the predeclared first 8 DEV names; the manifest is longer, so "first 8" bites.
_MANIFEST = tuple(f"dev-{i}" for i in range(11))
_SUBSET = _MANIFEST[:8]


def _states(seed=1, exact_grid=False, geometry=_TINY):
    values = np.random.default_rng(seed).standard_normal(geometry.z_bar_states).astype(np.float32)
    return np.round(values * 4).astype(np.float32) if exact_grid else values


def _put(call, name, array):
    """Write one example on BOTH sides, so a pair-consistency error cannot mask the clause under test."""
    call["fp32"][name] = array
    with np.errstate(over="ignore"):
        call["stored"][name] = np.asarray(array).astype(np.float16)


def _gate(fp32=None, stored=None, manifest=_MANIFEST, deltas=None, **kwargs):
    fp32 = {name: _states(seed=i, **kwargs) for i, name in enumerate(_SUBSET)} if fp32 is None else fp32
    stored = {name: array.astype(np.float16) for name, array in fp32.items()} if stored is None else stored
    return states_fidelity_check(manifest, fp32, stored, feature_deltas=deltas, geometry=_TINY)


def test_bit_identical_states_pass_on_the_bit_identity_path():
    """Values that survive the fp16 hop exactly (all fp16-grid values do) take the first path."""
    verdict = _gate(exact_grid=True)

    assert verdict.bit_identical and verdict.passed and verdict.latent_dtype == "fp16"
    assert verdict.path == "bit-identical" and verdict.mismatched_elements == 0 and verdict.reasons == ()
    assert verdict.subset == _SUBSET and verdict.total_elements == 8 * int(np.prod(_TINY.z_bar_states))


def test_double_rounded_states_fail_bit_identity_and_fall_back_to_the_tolerance_path():
    without = _gate()
    within = _gate(deltas=dict.fromkeys(_SUBSET, 1e-3))
    beyond = _gate(deltas={**dict.fromkeys(_SUBSET, 1e-3), "dev-5": 0.5})

    assert not without.bit_identical and without.mismatched_elements > 0
    assert not without.passed and without.latent_dtype == "fp32" and "no feature deltas" in without.reasons[0]
    assert within.passed and within.latent_dtype == "fp16" and within.path == "feature-tolerance"
    assert not beyond.passed and beyond.latent_dtype == "fp32" and beyond.path == "feature-tolerance"
    assert beyond.worst_feature_delta == 0.5  # the worst example decides, not the mean


def test_the_feature_tolerance_boundary_is_inclusive():
    at = _gate(deltas=dict.fromkeys(_SUBSET, STATES_FEATURE_MAX_ABS_DELTA))
    over = _gate(deltas=dict.fromkeys(_SUBSET, STATES_FEATURE_MAX_ABS_DELTA * 1.01))

    assert STATES_FEATURE_MAX_ABS_DELTA == 1e-2  # plan §4-P2' F7
    assert at.passed and not over.passed
    assert at.worst_feature_delta == STATES_FEATURE_MAX_ABS_DELTA


def test_an_overflowing_state_selects_fp32_immediately_whatever_the_deltas_say():
    """**S5 review, finding 2(d): the 70000.0 probe.** A finite fp32 state that serializes to inf must
    never be excused by a caller-chosen feature delta -- the model would be handed an inf."""
    fp32 = {name: _states(seed=i) for i, name in enumerate(_SUBSET)}
    poisoned = np.array(fp32["dev-3"])
    poisoned.reshape(-1)[0] = np.float32(70000.0)  # finite in fp32 ...
    fp32["dev-3"] = poisoned
    with np.errstate(over="ignore"):
        stored = {name: array.astype(np.float16) for name, array in fp32.items()}  # ... inf in fp16

    verdict = _gate(fp32=fp32, stored=stored, deltas=dict.fromkeys(_SUBSET, 0.0))

    assert np.isinf(stored["dev-3"]).any() and np.all(np.isfinite(poisoned))
    assert not verdict.passed and verdict.latent_dtype == "fp32"
    assert verdict.path == "nonfinite-serialization" and "dev-3" in verdict.reasons[0]


def test_the_check_compares_the_stored_bytes_and_not_the_inputs_to_themselves():
    """A gate handed fp32 on both sides would pass vacuously; the serialized side must be the fp16
    one that was actually written, and a real difference must be visible."""
    fp32 = {name: _states(seed=i) for i, name in enumerate(_SUBSET)}
    corrupted = {name: array.astype(np.float16) for name, array in fp32.items()}
    corrupted["dev-0"] = corrupted["dev-0"].copy()
    corrupted["dev-0"].reshape(-1)[0] = np.float16(9.0)

    assert _gate(fp32=fp32, stored=corrupted).max_abs_delta > 1.0
    with pytest.raises(ValueError, match="serialized_states\\['dev-0'\\] must be float16"):
        _gate(fp32=fp32, stored=fp32)  # fp32-vs-fp32 is not a fidelity measurement


@pytest.mark.parametrize(
    "mangle, message",
    [
        # (a) coverage: the subset is derived, and the evidence must match it exactly.
        (lambda c: c.update(manifest=_MANIFEST[:7]), "at least 8 names"),
        (lambda c: c.update(manifest=("dup", "dup", *_MANIFEST)), "must be unique"),
        (lambda c: c["fp32"].pop("dev-7"), "fp32_states must cover exactly"),
        (lambda c: c["stored"].pop("dev-2"), "serialized_states must cover exactly"),
        (lambda c: c["fp32"].update({"dev-9": _states()}), "fp32_states must cover exactly"),
        (lambda c: c.update(deltas=dict.fromkeys(_SUBSET[:7], 1e-3)), "feature_deltas must cover exactly"),
        (lambda c: c.update(deltas={**dict.fromkeys(_SUBSET, 1e-3), "dev-9": 1e-3}), "feature_deltas must cover"),
        (lambda c: c.update(deltas={}), "feature_deltas must cover exactly"),
        (lambda c: c.update(deltas={**dict.fromkeys(_SUBSET, 1e-3), "dev-0": float("nan")}), "must be finite"),
        (lambda c: c.update(deltas={**dict.fromkeys(_SUBSET, 1e-3), "dev-0": -1.0}), "non-negative"),
        # (b) geometry, (c) empty/non-finite inputs. ``_put`` writes both sides, so the pair agrees on
        # shape and dtype and the clause under test is the one that fires.
        (lambda c: _put(c, "dev-1", np.zeros((3, 2, 3, 2, 1), np.float32)), "states geometry"),
        (lambda c: _put(c, "dev-1", np.zeros((0, 2, 3, 2, 2), np.float32)), "states geometry"),
        (lambda c: _put(c, "dev-1", np.zeros(_TINY.z_bar_states, np.float64)), "must be float32"),
        (lambda c: _put(c, "dev-4", np.full(_TINY.z_bar_states, np.nan, np.float32)), "must be finite"),
        (lambda c: c["stored"].update({"dev-6": c["fp32"]["dev-6"]}), "must be float16"),
    ],
)
def test_the_gate_fails_closed_on_malformed_evidence(mangle, message):
    fp32 = {name: _states(seed=i) for i, name in enumerate(_SUBSET)}
    call = {
        "fp32": fp32,
        "stored": {name: array.astype(np.float16) for name, array in fp32.items()},
        "manifest": _MANIFEST,
        "deltas": dict.fromkeys(_SUBSET, 1e-3),
    }
    mangle(call)

    with pytest.raises(ValueError, match=message):
        _gate(**call)


def test_an_empty_states_geometry_cannot_be_smuggled_in_through_the_seam():
    """(c) The emptiness check is independent of the geometry check: a caller who also supplies an
    empty geometry must still be refused, or a zero-element gate would pass vacuously."""
    empty = _PosGeometry(
        z_video=(0, 3, 2, 2),
        z_i0=(0, 1, 2, 2),
        actions=(4, 7),
        pos_embeds=(3, 8, 5),
        z_bar_states=(3, 0, 3, 2, 2),
        per_step_final_losses=(3,),
    )
    fp32 = {name: np.zeros(empty.z_bar_states, np.float32) for name in _SUBSET}

    with pytest.raises(ValueError, match="must not be empty"):
        states_fidelity_check(_MANIFEST, fp32, {n: a.astype(np.float16) for n, a in fp32.items()}, geometry=empty)


# --------------------------------------------------------------------------------------------------
# 4. Storage accounting (the pre-build free-space check consumes this).
# --------------------------------------------------------------------------------------------------


def test_storage_arithmetic_is_exact_and_matches_the_hand_computation():
    """Hand-computed from the schema: 8 arrays, no compression, npy/zip headers excluded."""
    fp16 = record_storage_bytes("fp16")
    fp32 = record_storage_bytes("fp32")

    assert fp16["pos_embeds"] == 25 * 8 * 4096 * 2 == 1_638_400
    assert fp16["z_bar_states"] == 25 * 48 * 9 * 12 * 20 * 2 == 5_184_000
    assert fp16["z_i0"] == 48 * 1 * 12 * 20 * 2 and fp16["actions"] == 32 * 7 * 4
    assert fp16["total"] == 7_468_516  # = 7.122 MiB
    # The fp32 fallback doubles exactly the four latent-scoped fields and nothing else.
    for field in POS_LATENT_SCOPED_FIELDS:
        assert fp32[field] == 2 * fp16[field], field
    for field in SOURCE_DTYPES:
        assert fp32[field] == fp16[field], field
    assert fp32["total"] == 14_705_636
    assert 1.9 < fp32["total"] / fp16["total"] < 2.0  # the plan's "fp32 fallback ~ 2x"


def test_the_states_dominate_the_record_and_the_cohort_total_is_reported():
    """Plan §4-P2' sizes the K2 build from these numbers, so they are pinned rather than described."""
    fp16 = record_storage_bytes("fp16")

    assert fp16["z_bar_states"] / fp16["total"] > 0.65  # ~69%: the states ARE the record
    assert record_storage_bytes("fp16", records=2128)["total"] == 2128 * 7_468_516
    assert 14.7 < record_storage_bytes("fp16", records=2128)["total"] / 1024**3 < 14.9  # GiB


def test_record_storage_bytes_rejects_an_unknown_policy_or_count():
    with pytest.raises(ValueError, match="latent_dtype"):
        record_storage_bytes("fp8")
    with pytest.raises(ValueError, match="records"):
        record_storage_bytes("fp16", records=0)
