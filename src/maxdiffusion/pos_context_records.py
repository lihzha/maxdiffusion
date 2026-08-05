"""Cached-artifact records for exp_05 positive-context inversion (plan §4-P2').

The positive-slot sibling of ``null_adapter_records``: the same record contract, with ``nulls``
replaced by the deployed 8-token ``pos_embeds [25, 8, 4096]`` and the per-step teacher-forcing states
``z_bar_states [25, 48, 9, 12, 20]`` added. Both are governed by ``latent_dtype``, so the fp32
fallback covers ``{pos_embeds, z_start, expected_final_latent, z_bar_states}`` while ``z_i0``,
``z_video`` and ``actions`` keep their source dtypes under every policy.

``z_bar_states`` is what makes this schema more than a rename: the P3' regression trainer is
teacher-forced on ``(z_bar_states[i], pos_embeds[i])`` pairs, so a state that is silently truncated,
mis-typed or unvalidated becomes a training target that means something else. It is also 69% of the
record's bytes, which is why the byte-length validation and the storage accounting below both name it
explicitly rather than inheriting a rule written for a schema that never had it.

**What is imported and what is restated (the ``_Geometry`` decision).** Every schema-independent
primitive comes from exp_04's codec **by import** -- ``sha256_of_array``, the dtype tables, the freeze
helper, the deterministic zip writer, the namespace check. Its ``_Geometry`` seam, however, cannot be
reused: it is a frozen dataclass whose fields and ``shapes()`` hardcode the null field set (including
``nulls`` itself), so admitting ``pos_embeds``/``z_bar_states`` would mean editing exp_04's settled
module -- which plan §5/F6 forbids. **``_PosGeometry`` is therefore a local table with the identical
validation structure**, plus two invariants the sibling has no reason to state: the states are
``(N, *z_video)`` and every per-step array agrees on N.

**THE BF16 FINDING -- plan §4-P2's F7 premise is false; the tolerance path is load-bearing.**
F7 expects the states-fidelity gate to pass trivially because "fp16 -> bf16 is value-preserving at
bf16 precision for our range". It is not. ``fp32 -> fp16 -> bf16`` is a **double rounding**: when the
fp16 hop lands exactly on a bf16 midpoint, round-half-even resolves it away from the value a direct
cast would have chosen. What is actually measured (S5 review verified the frequency independently and
corrected the rest of this paragraph):

* **Frequency.** ~6.2-6.3% of elements differ on latent-like N(0,1) data -- 6.217% over 200k values
  here, 6.233% over 1M, 6.27% in the reviewer's 6M sweep. Not a corner case.
* **Size, for fp16-NORMAL inputs (|x| >= 2^-14).** Exactly one bf16 ulp.
* **Size, for fp16-SUBNORMAL inputs (|x| < 2^-14).** Larger: fp16's subnormal grid is spaced 2^-24
  regardless of magnitude, so it is far coarser than bf16's ~2^-8 *relative* step down there.
  Deterministic witnesses: ``24.5 * 2^-24`` diverges by 4 bf16 ulps, ``1.5 * 2^-25`` by 64, and
  ``2^-25`` -- which fp16 ties-to-even straight to zero -- by 13056. The reviewer observed 2-35 bit
  spans from N(0,1) tails. **Note what this does and does not mean:** the *absolute* error stays
  below 2^-24 (~6e-8) throughout, because the values themselves are that small, so bit distance
  overstates the harm here. ``max_abs_delta`` is the honest quantity in this regime and is reported
  alongside ``max_ulp_delta``.
* **Range (corrected thresholds, verified by test).** fp16 rounds to **inf at and above 65520** (not
  "above 65504": 65519 still rounds down to 65504), and to **zero at and below 2^-25** (~2.98e-8, the
  ties-to-even halfway), while 2^-25 * 1.001 rounds up to 2^-24. bf16 carries fp32's exponent range
  and does neither.

So ``states_fidelity_check`` is written to *expect* the bit-identity path to fail on real data, and
the feature-tolerance fallback is the decisive one. Overflow is treated as fatal on sight -- a finite
state that serializes to inf can never be excused by a caller-supplied delta. Underflow deliberately
is **not**: flushing a ~1e-8 element to zero is an absolute change of ~1e-8, far below anything bf16
resolves at signal scale, and forcing fp32 for it would double the cache for no fidelity.

Numpy-only, like its sibling: record IO must not need jax. ``ml_dtypes`` supplies the bfloat16 type
(it is jax's own dtype library, not jax), and a test pins its rounding to ``jnp.bfloat16``'s.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import zipfile
from typing import Any, Mapping, Optional, Sequence

import ml_dtypes
import numpy as np

from maxdiffusion.null_adapter_cache_policy import FIDELITY_SUBSET_SIZE, _checked_names
from maxdiffusion.null_adapter_records import (
    LATENT_DTYPES,
    NOISE_CONVENTIONS,
    SCHEMA_VERSION,
    SOURCE_DTYPES,
    _check_namespace,
    _freeze,
    _is_integral,
    _META_KEYS,
    _zip_arrays,
    sha256_of_array,
)


POS_LATENT_SCOPED_FIELDS = ("pos_embeds", "z_start", "expected_final_latent", "z_bar_states")
POS_ARRAY_FIELDS = (*SOURCE_DTYPES, *POS_LATENT_SCOPED_FIELDS, "per_step_final_losses")
# plan §4-P2' F7: the fallback tolerance on block-0 feature deltas, judged here, measured by the K job.
STATES_FEATURE_MAX_ABS_DELTA = 1e-2
# Inclusive thresholds, as in exp_04's fidelity gate: 1e-2 computed by subtraction is not exactly 1e-2.
FIDELITY_BOUNDARY_ATOL = 1e-9


@dataclasses.dataclass(frozen=True)
class _PosGeometry:
    """The size table the codec validates against; the public API only ever uses production."""

    z_video: tuple[int, ...] = (48, 9, 12, 20)
    z_i0: tuple[int, ...] = (48, 1, 12, 20)
    actions: tuple[int, ...] = (32, 7)
    pos_embeds: tuple[int, ...] = (25, 8, 4096)
    z_bar_states: tuple[int, ...] = (25, 48, 9, 12, 20)
    per_step_final_losses: tuple[int, ...] = (25,)

    def __post_init__(self) -> None:
        steps = self.pos_embeds[0]
        if self.z_bar_states != (steps, *self.z_video):
            raise ValueError(f"z_bar_states must be (N, *z_video) = {(steps, *self.z_video)}, got {self.z_bar_states}")
        if self.per_step_final_losses != (steps,):
            raise ValueError(f"per-step arrays must carry one entry per sampler step (N = {steps})")

    def as_dict(self) -> dict[str, tuple[int, ...]]:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "z_i0": self.z_i0,
            "actions": self.actions,
            "z_video": self.z_video,
            "pos_embeds": self.pos_embeds,
            "z_start": self.z_video,
            "expected_final_latent": self.z_video,
            "z_bar_states": self.z_bar_states,
            "per_step_final_losses": self.per_step_final_losses,
        }


PRODUCTION_POS_GEOMETRY = _PosGeometry()
EMBEDDING_SLOT = "positive"


@dataclasses.dataclass(frozen=True)
class PosProvenanceHeader:
    """What every positive-slot record in a shard was produced by (S4's l_pos decision).

    **Why this exists instead of exp_04's ``ProvenanceHeader``.** That header's ninth field is
    ``l_null`` -- the number of optimized rows in the *null* slot. The positive slot optimizes 8 rows
    of a conditional context, which is the same *quantity* under a name that would be read as a
    different *claim*: a reader (or the S9 evaluator) seeing ``l_null=8`` in a shard would reasonably
    conclude it was looking at a 512-row null-slot artifact whose leading 8 rows were optimized, which
    is not what these records are. Reusing the field would make every positive shard's provenance
    quietly wrong, so the field is named ``l_pos`` and the header additionally states its
    ``embedding_slot`` outright -- a reader never has to infer the slot from a field name.

    Everything else matches exp_04's header field for field, deliberately, so the two artifact
    families stay diff-able and the shared verifier can be extended additively later.
    """

    manifest_hash: str
    code_sha: str
    model_revision: str
    sigma_vector: np.ndarray
    guide_scale: float
    base_context_fingerprint: str
    optimization_config: dict[str, Any]
    dtype_policy: str
    l_pos: int
    embedding_slot: str = EMBEDDING_SLOT
    schema_version: int = SCHEMA_VERSION


def _validate_pos_header(header: PosProvenanceHeader) -> np.ndarray:
    """Shared by writer and reader; returns the canonical sigma vector (exp_04's rules, plus the slot)."""
    if not _is_integral(header.schema_version) or int(header.schema_version) != SCHEMA_VERSION:
        raise ValueError(f"unsupported header schema_version {header.schema_version!r}, expected {SCHEMA_VERSION}")
    if header.embedding_slot != EMBEDDING_SLOT:
        raise ValueError(
            f"embedding_slot must be {EMBEDDING_SLOT!r} in a positive-slot header, got {header.embedding_slot!r}"
        )
    sigma = np.ascontiguousarray(np.asarray(header.sigma_vector).astype(np.dtype("<f4")))
    if sigma.ndim != 1 or sigma.shape[0] != PRODUCTION_POS_GEOMETRY.pos_embeds[0] + 1:
        raise ValueError(
            f"sigma_vector must be 1-D with {PRODUCTION_POS_GEOMETRY.pos_embeds[0] + 1} entries, got {sigma.shape}"
        )
    if not np.all(np.isfinite(sigma)):
        raise ValueError("sigma_vector must be finite")
    if not np.isfinite(header.guide_scale):
        raise ValueError(f"guide_scale must be finite, got {header.guide_scale}")
    if header.dtype_policy not in LATENT_DTYPES:
        raise ValueError(f"dtype_policy must be one of {sorted(LATENT_DTYPES)}, got {header.dtype_policy!r}")
    if not _is_integral(header.l_pos) or int(header.l_pos) < 1:
        raise ValueError(
            f"l_pos must be a positive integer -- the optimized context's row count, got {header.l_pos!r}"
        )
    if not isinstance(header.optimization_config, dict):
        raise ValueError(f"optimization_config must be a dict, got {type(header.optimization_config)}")
    for key, value in header.optimization_config.items():
        if not isinstance(key, str) or not (value is None or isinstance(value, (str, bool, int, float))):
            raise ValueError("optimization_config must be flat: values must be str, int, float, bool or None")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"optimization_config[{key!r}] must be finite, got {value}")
    return sigma


def pos_header_to_json(header: PosProvenanceHeader) -> str:
    """Deterministic JSON: sorted keys, sigma vector as exact float32 values."""
    sigma = _validate_pos_header(header)
    payload = {**dataclasses.asdict(header), "sigma_vector": [float(v) for v in sigma]}
    return json.dumps(payload, sort_keys=True)


def pos_header_from_json(text: str) -> PosProvenanceHeader:
    """Parse a positive-slot header under exactly the writer's contract."""
    payload = json.loads(text)
    _check_namespace(payload, [f.name for f in dataclasses.fields(PosProvenanceHeader)], "header fields")
    header = PosProvenanceHeader(**payload)
    return dataclasses.replace(header, sigma_vector=_validate_pos_header(header))


def pos_header_fingerprint(header: PosProvenanceHeader) -> str:
    """The one digest a positive shard is bound to (exp_04's ``header_fingerprint`` rule)."""
    return hashlib.sha256(pos_header_to_json(header).encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class PosContextRecord:
    """One cached example of the positive slot, per plan §4-P2'."""

    name: str
    ordinal: int
    split: str
    episode: str
    z_i0: np.ndarray
    actions: np.ndarray
    z_video: np.ndarray
    latent_dtype: str
    pos_embeds: np.ndarray
    z_start: np.ndarray
    expected_final_latent: np.ndarray
    expected_final_latent_sha256: str
    noise_convention: str
    arm: str
    per_step_final_losses: np.ndarray
    final_future_mse: float
    z_bar_states: np.ndarray


@dataclasses.dataclass(frozen=True)
class StatesFidelityVerdict:
    """What the F7 states gate decided, on which path, and over which predeclared examples."""

    passed: bool
    latent_dtype: str
    path: str
    subset: tuple[str, ...]
    bit_identical: bool
    mismatched_elements: int
    total_elements: int
    max_abs_delta: float
    max_ulp_delta: int
    worst_feature_delta: Optional[float]
    reasons: tuple[str, ...]


def to_bfloat16(values) -> np.ndarray:
    """Round to bfloat16 exactly as the model's own cast does (pinned against ``jnp.bfloat16``)."""
    return np.asarray(values, dtype=np.float32).astype(ml_dtypes.bfloat16)


def _stored_dtypes(latent_dtype: str) -> dict[str, np.dtype]:
    if latent_dtype not in LATENT_DTYPES:
        raise ValueError(f"latent_dtype must be one of {sorted(LATENT_DTYPES)}, got {latent_dtype!r}")
    return {
        **SOURCE_DTYPES,
        **dict.fromkeys(POS_LATENT_SCOPED_FIELDS, LATENT_DTYPES[latent_dtype]),
        "per_step_final_losses": np.dtype("<f4"),
    }


def _validate(record: PosContextRecord, geometry: _PosGeometry = PRODUCTION_POS_GEOMETRY) -> None:
    """The one validator, shared by the writer, the serializer and the reader."""
    dtypes = _stored_dtypes(record.latent_dtype)
    shapes = geometry.shapes()
    if record.noise_convention not in NOISE_CONVENTIONS:
        raise ValueError(f"noise_convention must be one of {list(NOISE_CONVENTIONS)}, got {record.noise_convention!r}")
    if not _is_integral(record.ordinal):
        raise ValueError(f"ordinal must be an integer, got {record.ordinal!r}")
    for field in POS_ARRAY_FIELDS:
        array = getattr(record, field)
        if not isinstance(array, np.ndarray) or array.dtype != dtypes[field]:
            raise ValueError(
                f"{field} must be a {dtypes[field].name} array, got {getattr(array, 'dtype', type(array))}"
            )
        if array.shape != shapes[field]:
            raise ValueError(f"{field} must have shape {shapes[field]}, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{field} must be finite: a cached artifact may not carry NaN or inf")
    if not np.isfinite(record.final_future_mse):
        raise ValueError(f"final_future_mse must be finite, got {record.final_future_mse}")
    if sha256_of_array(record.expected_final_latent) != record.expected_final_latent_sha256:
        raise ValueError("expected_final_latent sha256 mismatch: the stored bytes are not the ones that were hashed")


def _make_pos_record(*, geometry: _PosGeometry, **fields: Any) -> PosContextRecord:
    """Cast every tensor to its storage dtype, hash the stored bytes, freeze, validate fail-closed."""
    dtypes = _stored_dtypes(fields["latent_dtype"])

    def cast(field: str) -> np.ndarray:
        return _freeze(np.ascontiguousarray(np.asarray(fields[field]).astype(dtypes[field])))

    stored_expected = cast("expected_final_latent")
    record = PosContextRecord(
        name=str(fields["name"]),
        ordinal=fields["ordinal"],
        split=str(fields["split"]),
        episode=str(fields["episode"]),
        z_i0=cast("z_i0"),
        actions=cast("actions"),
        z_video=cast("z_video"),
        latent_dtype=fields["latent_dtype"],
        pos_embeds=cast("pos_embeds"),
        z_start=cast("z_start"),
        expected_final_latent=stored_expected,
        expected_final_latent_sha256=sha256_of_array(stored_expected),
        noise_convention=fields["noise_convention"],
        arm=str(fields["arm"]),
        per_step_final_losses=cast("per_step_final_losses"),
        final_future_mse=float(fields["final_future_mse"]),
        z_bar_states=cast("z_bar_states"),
    )
    _validate(record, geometry)
    return record


def make_pos_record(**fields: Any) -> PosContextRecord:
    """Build a record at the production geometry (plan §4-P2')."""
    return _make_pos_record(geometry=PRODUCTION_POS_GEOMETRY, **fields)


def _pos_record_to_bytes(record: PosContextRecord, geometry: _PosGeometry) -> bytes:
    _validate(record, geometry)
    meta = {"schema_version": SCHEMA_VERSION, "shapes": {f: list(getattr(record, f).shape) for f in POS_ARRAY_FIELDS}}
    meta.update(
        {f.name: getattr(record, f.name) for f in dataclasses.fields(record) if f.name not in POS_ARRAY_FIELDS}
    )
    members = {f: getattr(record, f) for f in POS_ARRAY_FIELDS}
    members["__meta__"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    return _zip_arrays(members)


def pos_record_to_bytes(record: PosContextRecord) -> bytes:
    """Serialize a production-geometry record; refuses anything the reader would refuse."""
    return _pos_record_to_bytes(record, PRODUCTION_POS_GEOMETRY)


def _pos_record_from_bytes(blob: bytes, geometry: _PosGeometry) -> PosContextRecord:
    """Parse a record, refusing anything whose bytes do not match its own declared schema."""
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError(f"archive members are duplicated: {sorted({n for n in names if names.count(n) > 1})}")
    _check_namespace(names, [f"{f}.npy" for f in (*POS_ARRAY_FIELDS, "__meta__")], "archive members")

    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        members = {name: data[name] for name in data.files}
    meta = json.loads(bytes(members.pop("__meta__")).decode("utf-8"))
    expected_meta = (
        *_META_KEYS,
        *(f.name for f in dataclasses.fields(PosContextRecord) if f.name not in POS_ARRAY_FIELDS),
    )
    _check_namespace(meta, expected_meta, "record metadata keys")
    _check_namespace(meta["shapes"], POS_ARRAY_FIELDS, "record shape keys")
    if meta["schema_version"] != SCHEMA_VERSION or not _is_integral(meta["schema_version"]):
        raise ValueError(f"unsupported record schema_version {meta['schema_version']!r}, expected {SCHEMA_VERSION}")
    dtypes = _stored_dtypes(meta["latent_dtype"])
    shapes = geometry.shapes()

    for field in POS_ARRAY_FIELDS:
        array, shape = members[field], tuple(meta["shapes"][field])
        # Against the GEOMETRY rather than the blob's own declaration (the sibling uses the
        # declaration here). Both forms reject a self-consistently truncated ``z_bar_states`` --
        # ``_validate`` below re-checks every shape against the geometry anyway, which is why a
        # mutation of this line alone changes no outcome -- but this one fails earlier, before a
        # dataclass is built from the bytes, and says so in byte terms.
        expected_bytes = int(np.prod(shapes[field])) * dtypes[field].itemsize
        if array.dtype != dtypes[field] or array.shape != shape or array.nbytes != expected_bytes:
            raise ValueError(
                f"{field} is {array.nbytes} bytes of {array.dtype} {array.shape}, but latent_dtype "
                f"{meta['latent_dtype']!r} and the geometry imply {expected_bytes} bytes of "
                f"{dtypes[field]} {shapes[field]}"
            )
        _freeze(array)

    scalars = {k: v for k, v in meta.items() if k not in _META_KEYS}
    record = PosContextRecord(**scalars, **members)
    _validate(record, geometry)
    return record


def pos_record_from_bytes(blob: bytes) -> PosContextRecord:
    """Parse a production-geometry record, failing closed on anything off-contract."""
    return _pos_record_from_bytes(blob, PRODUCTION_POS_GEOMETRY)


def _checked_coverage(table: Any, subset: tuple[str, ...], label: str) -> dict[str, Any]:
    """Exactly the predeclared names -- no gaps to hide behind, no extras to dilute the worst case."""
    if not isinstance(table, Mapping) or set(table) != set(subset):
        raise ValueError(
            f"{label} must cover exactly the predeclared first {len(subset)} DEV examples {list(subset)}, "
            f"got {sorted(table) if isinstance(table, Mapping) else type(table)}"
        )
    return dict(table)


def _checked_feature_deltas(feature_deltas: Any, subset: tuple[str, ...]) -> float:
    table = _checked_coverage(feature_deltas, subset, "feature_deltas")
    worst = 0.0
    for name in subset:
        value = float(table[name])
        if not math.isfinite(value):
            raise ValueError(f"feature_deltas[{name!r}] must be finite, got {table[name]}")
        if value < 0.0:
            raise ValueError(f"feature_deltas[{name!r}] must be non-negative (it is a |delta|), got {table[name]}")
        worst = max(worst, value)
    return worst


def _checked_states_pair(exact: Any, stored: Any, name: str, geometry: _PosGeometry) -> tuple[np.ndarray, np.ndarray]:
    """One example's in-memory and serialized states, at the production states geometry."""
    exact = np.asarray(exact)
    stored = np.asarray(stored)
    if exact.dtype != np.float32:
        raise ValueError(f"fp32_states[{name!r}] must be float32, got {exact.dtype}")
    if stored.dtype != np.float16:
        raise ValueError(
            f"serialized_states[{name!r}] must be float16 -- the dtype actually written, got {stored.dtype}"
        )
    if exact.shape != stored.shape:
        raise ValueError(
            f"fp32_states[{name!r}] {exact.shape} and serialized_states[{name!r}] {stored.shape} must have the same shape"
        )
    if exact.shape != geometry.z_bar_states:
        raise ValueError(f"states[{name!r}] must have the states geometry {geometry.z_bar_states}, got {exact.shape}")
    if exact.size == 0:
        raise ValueError(f"states[{name!r}] must not be empty: an empty gate measures nothing")
    if not np.all(np.isfinite(exact)):
        raise ValueError(f"fp32_states[{name!r}] must be finite: a non-finite state is not a fidelity question")
    return exact, stored


def states_fidelity_check(
    dev_manifest: Sequence[str],
    fp32_states: Mapping[str, np.ndarray],
    serialized_states: Mapping[str, np.ndarray],
    *,
    feature_deltas: Optional[Mapping[str, float]] = None,
    geometry: _PosGeometry = PRODUCTION_POS_GEOMETRY,
) -> StatesFidelityVerdict:
    """Plan §4-P2' F7: may ``z_bar_states`` be stored at fp16, given what the model actually consumes?

    The model consumes bf16, so the question is not whether fp16 storage is lossless -- it is whether
    the *bf16 inputs* differ. Path 1 asks exactly that, per example: ``bf16(serialized_fp16)`` vs
    ``bf16(fp32)``, bit for bit.

    **Do not expect path 1 to pass** (see the module docstring): fp32 -> fp16 -> bf16 double-rounds,
    and ~6.2-6.3% of latent-like elements come out a bf16 ulp away from a direct cast. That is why
    path 2 exists and is load-bearing. Path 2 judges **precomputed block-0 feature deltas** -- the K
    job measures them by running the frozen block on both versions of the states; this function only
    decides, exactly as exp_04's ``fidelity_gate`` decides on precomputed metric deltas. The cache
    stays fp16 iff ``max |delta| <= 1e-2`` (inclusive); otherwise ``latent_dtype`` flips to fp32 for
    the four latent-scoped fields.

    **This gate fails closed, and the S5 review's fail-open findings are why** (finding 2):

    * The **subset is derived here** from the DEV manifest -- the predeclared first
      ``FIDELITY_SUBSET_SIZE`` names, exp_04's rule -- and never accepted from a caller, who could
      otherwise choose the eight examples that give the answer they want.
    * Both state tables **and** the feature deltas must cover **exactly** that subset: a missing name
      is a gap that hides an example, an extra one dilutes a worst case that is supposed to be a max.
    * Every pair must carry the production **states geometry**, be non-empty, and be finite on the
      fp32 side -- an empty or malformed input otherwise measures nothing and passes.
    * **Overflow is fatal on sight.** If any finite fp32 state serializes to a non-finite fp16 value
      (|x| >= 65520), the verdict is fp32 immediately, on its own path, whatever the deltas say: no
      caller-supplied number can excuse handing the model an inf. Underflow is deliberately *not*
      treated this way -- see the module docstring for the measured reason.
    * Absent evidence, the answer is fp32. A bit-identity failure with no feature deltas is undecided,
      and undecided means the conservative dtype.

    Args:
      dev_manifest: the DEV manifest, in order; the first ``FIDELITY_SUBSET_SIZE`` names are the gate.
      fp32_states: ``{name: [25, 48, 9, 12, 20] float32}`` in-memory states, exactly for the subset.
      serialized_states: the same states after the float16 round trip -- float16, same shapes. Passing
        float32 here would compare a value with itself and pass vacuously, so it is rejected.
      feature_deltas: ``{name: max |delta|}`` block-0 feature deltas, exactly for the subset.
      geometry: the states size table to require; production by default. The tests use the tiny table,
        exactly as the codec's private seam lets them, so this stays a real geometry check either way.

    Returns:
      ``StatesFidelityVerdict`` -- the decision, the path, the subset it was decided over, and the
      measured bf16-level divergence aggregated over the subset (element counts, max |delta|, max ulp).
    """
    manifest = _checked_names(dev_manifest, "the DEV manifest")
    if len(manifest) < FIDELITY_SUBSET_SIZE:
        raise ValueError(f"the DEV manifest must carry at least {FIDELITY_SUBSET_SIZE} names, got {len(manifest)}")
    subset = manifest[:FIDELITY_SUBSET_SIZE]
    exact_table = _checked_coverage(fp32_states, subset, "fp32_states")
    stored_table = _checked_coverage(serialized_states, subset, "serialized_states")

    mismatched = total = max_ulp_delta = 0
    max_abs_delta = 0.0
    overflowed: list[str] = []
    for name in subset:
        exact_states, stored_states = _checked_states_pair(exact_table[name], stored_table[name], name, geometry)
        if not np.all(np.isfinite(stored_states.astype(np.float32))):
            overflowed.append(name)
        exact = to_bfloat16(exact_states)
        stored = to_bfloat16(stored_states.astype(np.float32))
        exact_bits = exact.view(np.uint16).astype(np.int64)
        stored_bits = stored.view(np.uint16).astype(np.int64)
        mismatched += int(np.count_nonzero(exact_bits != stored_bits))
        total += int(exact.size)
        finite = np.isfinite(stored.astype(np.float32))  # inf deltas are reported by ``overflowed``
        if np.any(finite):
            deltas = np.abs(exact.astype(np.float32) - stored.astype(np.float32))[finite]
            max_abs_delta = max(max_abs_delta, float(np.max(deltas)))
        # A bit-pattern distance; both operands share a sign, so it is a plain grid distance. It sizes
        # the rounding disagreement and is never the decision -- ``max_abs_delta`` is the honest
        # quantity in the subnormal regime, where the bit distance is large but the values are ~1e-8.
        max_ulp_delta = max(max_ulp_delta, int(np.max(np.abs(exact_bits - stored_bits))))

    worst_feature = _checked_feature_deltas(feature_deltas, subset) if feature_deltas is not None else None
    if overflowed:
        path, passed = "nonfinite-serialization", False
        reasons = (f"fp16 serialization produced non-finite values for {sorted(overflowed)}",)
    elif not mismatched:
        path, passed, reasons = "bit-identical", True, ()
    elif worst_feature is None:
        path, passed = "bit-identity", False
        reasons = ("bit-identity failed and no feature deltas were supplied to judge the fallback",)
    else:
        path = "feature-tolerance"
        passed = worst_feature <= STATES_FEATURE_MAX_ABS_DELTA + FIDELITY_BOUNDARY_ATOL
        reasons = () if passed else ("block-0 feature delta exceeds the fp16 tolerance",)
    return StatesFidelityVerdict(
        passed=passed,
        latent_dtype="fp16" if passed else "fp32",
        path=path,
        subset=subset,
        bit_identical=not mismatched,
        mismatched_elements=mismatched,
        total_elements=total,
        max_abs_delta=max_abs_delta,
        max_ulp_delta=max_ulp_delta,
        worst_feature_delta=worst_feature,
        reasons=reasons,
    )


def record_storage_bytes(
    latent_dtype: str, *, records: int = 1, geometry: _PosGeometry = PRODUCTION_POS_GEOMETRY
) -> dict[str, int]:
    """Exact stored bytes per field (and ``total``), for the pre-K2 free-space check (plan §4-P2').

    Array payload only: npy/zip headers add a few hundred bytes per record and no compression is used
    (``ZIP_STORED``), so this is a floor that is tight to well under 0.1%. At fp16 one record is
    7,468,516 B = 7.122 MiB, of which ``z_bar_states`` is 5,184,000 B (69%); the fp32 fallback is
    14,705,636 B = 1.97x. The 2,128-record K2 cohort is therefore **14.80 GiB** at fp16 and 29.1 GiB
    at fp32 -- the plan's "~8.2 MiB/record, ~17.1 GiB" is the conservative figure quoted at planning
    time and this exact arithmetic supersedes it (it agrees with the reviewer's independent ~14.8 GiB).
    """
    dtypes = _stored_dtypes(latent_dtype)
    if not _is_integral(records) or int(records) < 1:
        raise ValueError(f"records must be a positive integer, got {records!r}")
    shapes = geometry.shapes()
    sizes = {field: int(np.prod(shapes[field])) * dtypes[field].itemsize * int(records) for field in POS_ARRAY_FIELDS}
    return {**sizes, "total": sum(sizes.values())}
