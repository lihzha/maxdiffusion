"""exp_04 R1 — ``embed_null_tokens`` / ``base_context_fingerprint`` contracts.

The null branch of CFG is fed a full ``[B, 512, 4096]`` T5 context. Inversion only ever optimizes
the first ``L_null`` rows (plan §3, ``L_null = 16``); the remaining ``512 - L`` rows must stay the
T5("") padding the frozen backbone was trained and evaluated with. Two properties are load-bearing:

1. **Row replacement, not concatenation.** Rows ``[0:L]`` come from the optimized tokens, rows
   ``[L:512]`` are the base context's own rows, bit for bit. Anything else changes the null branch's
   sequence length or its padding statistics, which changes the CFG direction everywhere.
2. **Branch equality at the bf16 boundary (plan §3).** Initializing ``∅_init = T5("")[0:L]`` must
   make the null branch *identical* to the untouched frozen branch -- so that step 0 of the
   optimization starts exactly at the model's own behaviour, and the A0 control is genuinely
   "frozen ∅". Since the model consumes bf16, the equality that matters is after the bf16 cast, and
   it is asserted bitwise, not with a tolerance.

``base_context_fingerprint`` is the guard that a cached artifact's nulls were optimized against the
same T5("") the reader is about to replay with (a different tokenizer/revision silently changes the
padding rows). Its byte convention is pinned here by a hardcoded digest.
"""

from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion.models.wan.null_inversion_wan import base_context_fingerprint, embed_null_tokens


_SEQ, _DIM, _L = 512, 4096, 16

# sha256 of np.ascontiguousarray(np.arange(6, dtype=np.float32).reshape(2, 3).astype("<f4")).tobytes(),
# i.e. the 24 little-endian float32 bytes 0.0,1.0,...,5.0 in C order. Hardcoded so the byte
# convention cannot drift (a re-derived digest would silently follow any change to the convention).
_GOLDEN_FINGERPRINT = "e2c0a71510b5394df7773b63fb5f54372b84c3564e67811bde7d665be227976d"

# Values whose float32 bit patterns a numeric ``==`` cannot tell apart (+0/-0) or compares as unequal
# to themselves (NaN), plus a subnormal that a lossy round-trip would flush. Planted in both the
# replaced and the untouched row ranges.
_EDGE_VALUES = (0.0, -0.0, np.float32(np.inf), np.float32(-np.inf), np.float32(np.nan), np.float32(1e-45), -1.0, 1.0)


def _bits(x, storage_dtype, bit_dtype):
    """Raw bit patterns, so +0 != -0 and NaN == NaN -- what "bitwise equal" actually means."""
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x).astype(storage_dtype), bit_dtype))


def _bf16_bits(x):
    return _bits(x, jnp.bfloat16, jnp.uint16)


def _f32_bits(x):
    return _bits(x, jnp.float32, jnp.uint32)


def _base_context(seq=_SEQ, dim=_DIM, seed=0, dtype=jnp.float32):
    return jax.random.normal(jax.random.PRNGKey(seed), (seq, dim), dtype=jnp.float32).astype(dtype)


def _edge_case_base_context(seq=_SEQ, dim=_DIM):
    """A base context carrying signed zeros / inf / NaN / subnormals in replaced and tail rows."""
    base = np.asarray(_base_context(seq=seq, dim=dim)).copy()
    edges = np.asarray(_EDGE_VALUES, dtype=np.float32)
    base[0, : len(edges)] = edges  # inside the replaced range [0:L]
    base[_L, : len(edges)] = edges  # first untouched row
    base[-1, : len(edges)] = edges  # last untouched row
    return jnp.asarray(base)


def _null_tokens(batch, length=_L, dim=_DIM, seed=1, dtype=jnp.float32):
    return jax.random.normal(jax.random.PRNGKey(seed), (batch, length, dim), dtype=jnp.float32).astype(dtype)


def test_embed_replaces_leading_rows_and_leaves_the_tail_untouched():
    base = _base_context()
    nulls = _null_tokens(batch=3)

    out = embed_null_tokens(nulls, base)

    assert out.shape == (3, _SEQ, _DIM)
    np.testing.assert_array_equal(np.asarray(out[:, :_L]), np.asarray(nulls))
    for b in range(3):
        np.testing.assert_array_equal(np.asarray(out[b, _L:]), np.asarray(base[_L:]))


def test_embed_broadcasts_unbatched_tokens_and_squeezes_the_leading_base_axis():
    base_2d = _base_context(seq=12, dim=8)
    base_3d = base_2d[None]
    nulls_2d = _null_tokens(batch=1, length=3, dim=8)[0]

    from_2d_base = embed_null_tokens(nulls_2d, base_2d)
    from_3d_base = embed_null_tokens(nulls_2d, base_3d)
    from_batched = embed_null_tokens(nulls_2d[None], base_2d)

    assert from_2d_base.shape == (1, 12, 8)
    np.testing.assert_array_equal(np.asarray(from_3d_base), np.asarray(from_2d_base))
    np.testing.assert_array_equal(np.asarray(from_batched), np.asarray(from_2d_base))


@pytest.mark.parametrize(
    "null_dtype, base_dtype, expected",
    [
        (jnp.float32, jnp.float32, jnp.float32),
        (jnp.bfloat16, jnp.float32, jnp.float32),
        (jnp.float32, jnp.bfloat16, jnp.float32),
        (jnp.bfloat16, jnp.bfloat16, jnp.bfloat16),
    ],
)
def test_embed_output_dtype_is_the_promotion_of_its_inputs(null_dtype, base_dtype, expected):
    base = _base_context(seq=12, dim=8, dtype=base_dtype)
    nulls = _null_tokens(batch=2, length=3, dim=8, dtype=null_dtype)

    out = embed_null_tokens(nulls, base)

    assert out.dtype == expected


def test_bit_comparison_helpers_distinguish_what_numeric_equality_cannot():
    """Guard the guard: these helpers, unlike ``==``, separate +0/-0 and match NaN to itself."""
    signed_zeros = jnp.asarray([0.0, -0.0], dtype=jnp.float32)
    nans = jnp.asarray([np.nan, np.nan], dtype=jnp.float32)

    assert bool(jnp.all(signed_zeros[0] == signed_zeros[1]))  # numeric == cannot see the sign bit
    assert _f32_bits(signed_zeros)[0] != _f32_bits(signed_zeros)[1]
    assert _bf16_bits(signed_zeros)[0] != _bf16_bits(signed_zeros)[1]
    assert not bool(jnp.any(nans[0] == nans[1]))  # numeric == calls NaN unequal to itself
    np.testing.assert_array_equal(_f32_bits(nans)[:1], _f32_bits(nans)[1:])
    np.testing.assert_array_equal(_bf16_bits(nans)[:1], _bf16_bits(nans)[1:])


def test_embed_with_base_rows_as_tokens_is_bitwise_equal_to_the_base_at_bfloat16():
    """Plan §3 branch-equality: ∅_init = T5("")[0:L] must reproduce the frozen branch exactly.

    Asserted on raw bit patterns (not ``==``) over a context carrying signed zeros, infinities, NaN
    and a subnormal, so "identical to the frozen branch" means what the model would actually see.
    """
    base = _edge_case_base_context()

    out = embed_null_tokens(base[:_L], base)

    np.testing.assert_array_equal(_bf16_bits(out[0]), _bf16_bits(base))
    np.testing.assert_array_equal(_f32_bits(out[0]), _f32_bits(base))


def test_embed_preserves_edge_value_bits_in_both_row_ranges():
    """Row replacement is a copy: no flush-to-zero, no NaN canonicalization, no sign-bit loss."""
    base = _edge_case_base_context()
    nulls = jnp.broadcast_to(base[0], (2, _L, _DIM))

    out = embed_null_tokens(nulls, base)

    for row in range(2):
        np.testing.assert_array_equal(_f32_bits(out[row, :_L]), _f32_bits(nulls[row]))
        np.testing.assert_array_equal(_f32_bits(out[row, _L:]), _f32_bits(base[_L:]))


@pytest.mark.parametrize(
    "nulls_shape, base_shape",
    [
        ((2, 3, 8), (12, 9)),  # feature dim mismatch
        ((2, 13, 8), (12, 8)),  # more null tokens than context rows
        ((2, 3, 8), (2, 12, 8)),  # base context with a non-unit leading axis
        ((8,), (12, 8)),  # rank-1 tokens
        ((2, 2, 3, 8), (12, 8)),  # rank-4 tokens
        ((2, 3, 8), (12,)),  # rank-1 base context
    ],
)
def test_embed_rejects_inconsistent_shapes(nulls_shape, base_shape):
    nulls = jnp.zeros(nulls_shape, dtype=jnp.float32)
    base = jnp.zeros(base_shape, dtype=jnp.float32)

    with pytest.raises(ValueError):
        embed_null_tokens(nulls, base)


def test_base_context_fingerprint_matches_the_pinned_byte_convention():
    context = np.arange(6, dtype=np.float32).reshape(2, 3)

    assert base_context_fingerprint(context) == _GOLDEN_FINGERPRINT
    # Same bytes, same digest: a [1, 512, 4096] context fingerprints as its [512, 4096] squeeze.
    assert base_context_fingerprint(context[None]) == _GOLDEN_FINGERPRINT
    assert base_context_fingerprint(jnp.asarray(context)) == _GOLDEN_FINGERPRINT


def test_base_context_fingerprint_hashes_the_float32_upcast():
    """The digest identifies the context, not the dtype it happened to be held in."""
    context = np.arange(6, dtype=np.float32).reshape(2, 3)  # exactly representable in bfloat16

    assert base_context_fingerprint(context.astype(np.float64)) == _GOLDEN_FINGERPRINT
    assert base_context_fingerprint(jnp.asarray(context).astype(jnp.bfloat16)) == _GOLDEN_FINGERPRINT


def test_base_context_fingerprint_canonicalizes_byte_order():
    """A cached artifact's digest must not depend on the host that wrote it.

    The big-endian input below carries the same values in reversed bytes; hashing it verbatim gives
    fb92147e... instead of the golden, so this pins the explicit "<f4" canonicalization rather than
    whatever ``float32`` happens to mean on the running machine.
    """
    big_endian = np.arange(6, dtype=np.float32).reshape(2, 3).astype(np.dtype(">f4"))

    assert big_endian.dtype.byteorder == ">"
    assert base_context_fingerprint(big_endian) == _GOLDEN_FINGERPRINT


def test_base_context_fingerprint_is_deterministic_and_element_sensitive():
    base = _base_context(seq=12, dim=8)
    perturbed = base.at[7, 3].add(np.float32(1e-3))

    assert base_context_fingerprint(base) == base_context_fingerprint(base)
    assert base_context_fingerprint(base) != base_context_fingerprint(perturbed)
    canonical = np.ascontiguousarray(np.asarray(base).astype(np.dtype("<f4")))
    assert base_context_fingerprint(base) == hashlib.sha256(canonical.tobytes()).hexdigest()
