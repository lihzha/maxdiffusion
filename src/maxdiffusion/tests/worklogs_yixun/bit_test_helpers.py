"""Bit-level comparison helpers shared by the exp_04 worklog tests.

Hoisted verbatim from ``test_null_adapter_embed_tokens.py`` once a fourth consumer appeared, per the
R2 review's ruling ("extract it only when a third consumer appears"). Raw bit patterns, not numeric
``==``: this is what lets a test tell +0 from -0 and call NaN equal to itself, which is the standard
every "bitwise" assertion in these files is written against.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def bits(x, storage_dtype, bit_dtype):
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x).astype(storage_dtype), bit_dtype))


def bf16_bits(x):
    return bits(x, jnp.bfloat16, jnp.uint16)


def f32_bits(x):
    return bits(x, jnp.float32, jnp.uint32)
