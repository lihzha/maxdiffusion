"""exp_04 R1 — the noise conventions are a fixed, reproducible mapping (plan §3, N3/M1).

Every exp_04 claim is a *paired* comparison at matched noise: A1 vs A0 on the same per-example draw,
A2 vs A2-0 on the single canonical eps_0 = global(0), adapter vs pre_context on identical ``z_start``.
Cached targets under ``noise=keyed`` are only valid for the exact tensor the optimizer saw, and the
evaluator rejects convention mismatches -- so ``(name, k) -> noise`` must be a fixed function of the
manifest name alone: independent of batch composition, iteration order, host count, and of any other
RNG stream in the process.

The convention is **one float32 ``(48, 9, 12, 20)`` draw per ``(name, k)``, full stop.** The helpers
take no shape argument, because a batch-shaped draw would silently redefine the mapping: asking for
``(B, 48, 9, 12, 20)`` under the global convention yields B *sequential* draws, of which only row 0
is eps_0 -- so the "one canonical noise shared by all examples" deliverable would quietly become B
different noises (Codex R1 review, finding 1). Batches are assembled by the caller: stack keyed
draws, broadcast the global draw. Both assembly rules are pinned below.

The GOLDEN constants pin the mapping forever. They were generated from an independent transcription
of plan §3's derivation (a scratch script written from the plan text, not from the module under
test) and are compared bitwise -- float32 values round-trip exactly through Python float literals.
One golden name is deliberately non-ASCII, so that swapping the specified UTF-8 encoding for
anything else -- including a silently lossy ``encode("ascii", "ignore")`` -- changes the draw and
fails here. If a refactor changes the fold_in order, the domain constant, the seed, the sha256
slicing or the encoding, these fail; regenerating them is a decision that invalidates every cached
target and belongs to the Planner, not to a refactor.
"""

from __future__ import annotations

import hashlib
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion.models.wan.null_inversion_wan import LATENT_SHAPE, NOISE_DOMAIN, global_noise, keyed_noise, noise_key


# A manifest name outside ASCII: "é" is Latin-1-encodable, "七" is not, so every non-UTF-8 encoding
# either raises or produces different bytes -- and therefore a different draw.
_NON_ASCII_NAME = "droid_ép_000七/w0"

# sha256("droid_ép_000七/w0".encode("utf-8"))[0:4] and [4:8] as big-endian uint32, and the resulting
# threefry key data after the full fold_in chain. Both are independent-transcription outputs.
_NON_ASCII_WORDS = (3608796426, 2981090583)
_NON_ASCII_KEY_DATA = {0: (4069681520, 1033188866), 1: (2062723132, 3965242868)}

# First 8 values of the flattened (48, 9, 12, 20) draw, per (name, k).
_GOLDEN_HEADS = {
    ("droid_ep_000001/w0", 0): (
        1.392072319984436,
        0.18953724205493927,
        -0.06578119099140167,
        -0.0243215449154377,
        0.2619726359844208,
        -0.3992597460746765,
        1.1612269878387451,
        0.13727153837680817,
    ),
    ("droid_ep_000001/w0", 1): (
        -0.26154080033302307,
        0.8063388466835022,
        1.614790439605713,
        -0.22073400020599365,
        1.955387830734253,
        -0.3821457326412201,
        0.6670055985450745,
        0.15746262669563293,
    ),
    ("droid_ep_000042/w3", 0): (
        1.430148720741272,
        1.256081223487854,
        0.9240360856056213,
        -0.38739997148513794,
        0.09611265361309052,
        -0.14979973435401917,
        1.3174430131912231,
        0.2699679732322693,
    ),
    ("droid_ep_000042/w3", 1): (
        -0.9673252105712891,
        1.148789644241333,
        -1.2398324012756348,
        0.42471134662628174,
        -0.9332324266433716,
        -0.29245564341545105,
        1.857926607131958,
        -0.8013700246810913,
    ),
    ("GLOBAL", 0): (
        0.05884796380996704,
        -0.360533207654953,
        -0.684444010257721,
        -1.219874382019043,
        -0.5272910594940186,
        0.3335162103176117,
        0.8607488870620728,
        -0.18842674791812897,
    ),
    ("GLOBAL", 1): (
        -0.599536657333374,
        0.33515113592147827,
        0.17817501723766327,
        0.7211692929267883,
        1.1957013607025146,
        -1.3844702243804932,
        0.4273970127105713,
        -0.2727961540222168,
    ),
    (_NON_ASCII_NAME, 0): (
        0.02791585586965084,
        1.911646842956543,
        -0.28721025586128235,
        -0.42538556456565857,
        1.039413332939148,
        0.5277562141418457,
        0.4848426580429077,
        -0.25868046283721924,
    ),
    (_NON_ASCII_NAME, 1): (
        -1.835070252418518,
        0.6399882435798645,
        -0.04103163257241249,
        -0.29834258556365967,
        -1.8721718788146973,
        -0.7093086242675781,
        0.8521747589111328,
        -0.943516194820404,
    ),
}


def test_noise_constants_are_pinned():
    assert NOISE_DOMAIN == 0x4E4F4953
    assert LATENT_SHAPE == (48, 9, 12, 20)


@pytest.mark.parametrize("name, k", sorted(_GOLDEN_HEADS))
def test_keyed_noise_matches_the_golden_fingerprints(name, k):
    head = np.asarray(keyed_noise(name, k)).reshape(-1)[:8]

    np.testing.assert_array_equal(head, np.asarray(_GOLDEN_HEADS[(name, k)], dtype=np.float32))


@pytest.mark.parametrize("k", [0, 1])
def test_non_ascii_name_is_hashed_as_utf8_down_to_the_key_words(k):
    """The encoding is part of the derivation: these words come from the UTF-8 bytes, nothing else."""
    digest = hashlib.sha256(_NON_ASCII_NAME.encode("utf-8")).digest()
    assert (int.from_bytes(digest[0:4], "big"), int.from_bytes(digest[4:8], "big")) == _NON_ASCII_WORDS

    key_data = tuple(int(v) for v in np.asarray(jax.random.key_data(noise_key(_NON_ASCII_NAME, k))))
    assert key_data == _NON_ASCII_KEY_DATA[k]


def test_keyed_noise_shape_and_dtype():
    draw = keyed_noise("droid_ep_000001/w0", 0)

    assert draw.shape == LATENT_SHAPE
    assert draw.dtype == np.float32
    assert global_noise(0).shape == LATENT_SHAPE
    assert global_noise(0).dtype == np.float32


def test_noise_helpers_expose_no_shape_argument():
    """One (name, k) -> one canonical latent-shaped draw; the shape is not a caller's choice."""
    assert list(inspect.signature(keyed_noise).parameters) == ["name", "k"]
    assert list(inspect.signature(global_noise).parameters) == ["k"]

    with pytest.raises(TypeError):
        keyed_noise("GLOBAL", 0, (2, 3))
    with pytest.raises(TypeError):
        global_noise(0, (2, 3))


def test_keyed_noise_is_deterministic_across_calls():
    first = keyed_noise("droid_ep_000042/w3", 2)
    second = keyed_noise("droid_ep_000042/w3", 2)

    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_noise_separates_names_and_seed_indices():
    a0 = np.asarray(keyed_noise("droid_ep_000001/w0", 0))
    b0 = np.asarray(keyed_noise("droid_ep_000042/w3", 0))
    a1 = np.asarray(keyed_noise("droid_ep_000001/w0", 1))
    # Names one character apart must not share a draw either (sha256 decorrelates them).
    near = np.asarray(keyed_noise("droid_ep_000001/w1", 0))

    assert not np.array_equal(a0, b0)
    assert not np.array_equal(a0, a1)
    assert not np.array_equal(a0, near)


@pytest.mark.parametrize("k", [0, 1, 2])
def test_global_noise_is_the_keyed_draw_of_the_literal_global_name(k):
    np.testing.assert_array_equal(np.asarray(global_noise(k)), np.asarray(keyed_noise("GLOBAL", k)))


def test_noise_is_invariant_to_the_order_names_are_drawn_in():
    names = ["droid_ep_000001/w0", "droid_ep_000042/w3", "GLOBAL", _NON_ASCII_NAME]

    forward = {n: np.asarray(keyed_noise(n, 0)) for n in names}
    backward = {n: np.asarray(keyed_noise(n, 0)) for n in reversed(names)}

    for name in names:
        np.testing.assert_array_equal(forward[name], backward[name])


def test_keyed_batches_are_stacked_per_name_and_are_permutation_equivariant():
    """Batch assembly rule under ``noise=keyed``: one draw per name, stacked -- never a batch draw."""
    names = ["droid_ep_000042/w3", "droid_ep_000001/w0", _NON_ASCII_NAME]

    batch = np.asarray(jnp.stack([keyed_noise(n, 0) for n in names]))
    shuffled = ["droid_ep_000001/w0", _NON_ASCII_NAME, "droid_ep_000042/w3"]
    shuffled_batch = np.asarray(jnp.stack([keyed_noise(n, 0) for n in shuffled]))

    assert batch.shape == (len(names), *LATENT_SHAPE)
    for row, name in enumerate(names):
        np.testing.assert_array_equal(batch[row], np.asarray(keyed_noise(name, 0)))
        np.testing.assert_array_equal(batch[row], shuffled_batch[shuffled.index(name)])


@pytest.mark.parametrize("k", [0, 1])
def test_global_batches_are_one_broadcast_tensor_not_sequential_draws(k):
    """Batch assembly rule under ``noise=global``: eps_k broadcast, so every row is the same tensor."""
    canonical = global_noise(k)

    batch = np.asarray(jnp.broadcast_to(canonical, (3, *LATENT_SHAPE)))

    for row in range(3):
        np.testing.assert_array_equal(batch[row], np.asarray(canonical))
    # Why the helpers refuse a shape argument: a batch-shaped draw from the same key is a *different*
    # tensor whose rows 1.. are not eps_k at all, so cached targets would not match the noise used.
    batch_shaped_draw = np.asarray(jax.random.normal(noise_key("GLOBAL", k), (3, *LATENT_SHAPE), dtype=jnp.float32))
    assert not np.array_equal(batch_shaped_draw[1], np.asarray(canonical))
    assert not np.array_equal(batch_shaped_draw[1], batch_shaped_draw[0])
