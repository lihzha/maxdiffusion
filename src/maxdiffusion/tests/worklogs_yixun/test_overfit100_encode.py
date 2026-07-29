"""CPU-only tests for the exp_02 encode contract (cycle B, deliverable B1).

Plan v4 D4 locks the encode path to `wan_pipeline.py`'s video-encode code path *verbatim*:
frames laid out `[B, C, T, H, W]` in the pipeline's value range, `vae.encode(x, cache)[0]`
**`.mode()`** (deterministic posterior mode -- never `.sample()`, so no RNG enters the
dataset), the pipeline's `latents_mean/latents_std` normalization, channels-first transpose,
and only then the float16 cast. A silent break anywhere here (a missed mean/std, a
channels-last store, an f16 cast before the normalization) produces a dataset that trains
without crashing and generates garbage -- exactly the exp_01 failure mode this experiment
exists to rule out.

The VAE is stubbed: no weights are downloaded and no accelerator is used. The pixel-side
preprocessing is additionally CHARACTERIZED against the real `VideoProcessor.preprocess_video`
(importable without weights) at the corpus geometry 192x320, so the parity claim is executable
rather than asserted in prose.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion.data_preprocessing.build_overfit100_dataset import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    BuildError,
    decode_latents_to_frames,
    encode_pixels_to_latents,
    encode_window_latents,
    preprocess_frames,
)

# Tiny stand-in geometry for the real (48, 9, 12, 20). Every axis length is distinct so a
# transposed/reshaped result cannot accidentally pass.
_Z_DIM = 3
_LATENT_T, _LATENT_H, _LATENT_W = 4, 2, 5
_MEAN = [0.5, -1.0, 2.0]
_STD = [2.0, 4.0, 0.5]


class _StubDistribution:
    """Stands in for `WanDiagonalGaussianDistribution`; `.sample()` is a hard error."""

    def __init__(self, latents):
        self._latents = latents

    def mode(self):
        return self._latents

    def sample(self, *args, **kwargs):
        raise AssertionError("the exp_02 encode path must use .mode(); sampling would inject RNG")


class _StubVae:
    """Records what the encode wrapper hands the VAE and returns a fixed latent block."""

    def __init__(self, latents, dtype=jnp.float32, decoded=None):
        self.z_dim = _Z_DIM
        self.latents_mean = list(_MEAN)
        self.latents_std = list(_STD)
        self.dtype = dtype
        self._latents = latents
        self._decoded = decoded
        self.encode_calls = []
        self.decode_calls = []

    def encode(self, x, feat_cache, return_dict=True):
        self.encode_calls.append((x, feat_cache))
        return (_StubDistribution(self._latents),)

    def decode(self, z, feat_cache, return_dict=True):
        self.decode_calls.append((z, feat_cache))
        return (self._decoded,)


def _raw_latents(seed=0):
    """Channels-last `[B, T', H', W', C]` -- the layout `vae.encode(...).mode()` returns."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(1, _LATENT_T, _LATENT_H, _LATENT_W, _Z_DIM)), dtype=jnp.float32)


def _expected_normalized(raw):
    """The pipeline's normalization + channels-first transpose, computed independently."""
    mean = np.asarray(_MEAN).reshape(1, 1, 1, 1, _Z_DIM)
    std = np.asarray(_STD).reshape(1, 1, 1, 1, _Z_DIM)
    return np.transpose((np.asarray(raw, dtype=np.float32) - mean) / std, (0, 4, 1, 2, 3))


# ----------------------------------------------------------------------------------
# 1. Pixel preprocessing -- layout, range, dtype, and parity with the real processor.
# ----------------------------------------------------------------------------------


def test_preprocess_frames_layout_and_dtype():
    frames = np.zeros((5, 6, 8, 3), dtype=np.uint8)
    out = preprocess_frames(frames)
    assert out.shape == (1, 3, 5, 6, 8)  # [B, C, T, H, W]
    assert out.dtype == np.float32


def test_preprocess_frames_maps_uint8_onto_minus_one_to_one():
    frames = np.array([0, 128, 255], dtype=np.uint8).reshape(1, 1, 3, 1).repeat(3, axis=3)
    out = preprocess_frames(frames)
    np.testing.assert_allclose(out[0, 0, 0, 0, 0], -1.0)
    np.testing.assert_allclose(out[0, 0, 0, 0, 2], 1.0)
    # float32 arithmetic: 2*(128/255)-1 and 128/127.5-1 agree to ~1e-7 absolute, not to 1e-6 relative.
    np.testing.assert_allclose(out[0, 0, 0, 0, 1], 128.0 / 127.5 - 1.0, atol=1e-6)
    assert float(out.min()) >= -1.0 and float(out.max()) <= 1.0


def test_preprocess_frames_keeps_channels_and_time_distinct():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(4, 6, 8, 3), dtype=np.uint8)
    out = preprocess_frames(frames)
    # out[0, c, t, y, x] must be frame t, pixel (y, x), channel c.
    for t, c, y, x in ((0, 0, 1, 2), (3, 2, 5, 7), (2, 1, 0, 0)):
        assert out[0, c, t, y, x] == pytest.approx(frames[t, y, x, c] / 127.5 - 1.0, rel=1e-6)


def test_preprocess_frames_rejects_a_non_rgb_or_wrong_rank_input():
    with pytest.raises(BuildError):
        preprocess_frames(np.zeros((4, 6, 8), dtype=np.uint8))
    with pytest.raises(BuildError):
        preprocess_frames(np.zeros((4, 6, 8, 4), dtype=np.uint8))


def test_preprocess_frames_matches_the_real_video_processor_at_corpus_geometry():
    # Characterization against the code path `wan_pipeline` actually uses, at the exact
    # 192x320 DROID geometry (do_resize is a no-op there, so this compares normalization
    # and layout, which is what D4 pins). No weights, no accelerator -- torch only.
    torch = pytest.importorskip("torch")
    from maxdiffusion.video_processor import VideoProcessor

    rng = np.random.default_rng(7)
    frames = rng.integers(0, 256, size=(2, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    processor = VideoProcessor(vae_scale_factor=16)
    reference = processor.preprocess_video(
        (frames.astype(np.float32) / 255.0)[None], height=FRAME_HEIGHT, width=FRAME_WIDTH
    )
    assert isinstance(reference, torch.Tensor)
    np.testing.assert_array_equal(preprocess_frames(frames), reference.numpy())


# ----------------------------------------------------------------------------------
# 2. The encode wrapper -- .mode(), normalization constants, channels-first output.
# ----------------------------------------------------------------------------------


def test_encode_uses_mode_and_never_samples():
    raw = _raw_latents()
    vae = _StubVae(raw)
    out = encode_pixels_to_latents(np.zeros((1, 3, 13, 8, 20), dtype=np.float32), vae, feat_cache="CACHE")
    assert len(vae.encode_calls) == 1
    assert vae.encode_calls[0][1] == "CACHE"  # the vae_cache is threaded through
    np.testing.assert_allclose(np.asarray(out), _expected_normalized(raw), rtol=1e-6, atol=1e-6)


def test_encode_applies_the_pipeline_normalization_constants():
    raw = _raw_latents(1)
    out = np.asarray(encode_pixels_to_latents(np.zeros((1, 3, 13, 8, 20), dtype=np.float32), _StubVae(raw), None))
    # Independent recomputation: normalization happens channels-LAST, transpose after.
    np.testing.assert_allclose(out, _expected_normalized(raw), rtol=1e-6, atol=1e-6)
    # A path that skipped the std division would be off by exactly the std factors.
    assert not np.allclose(out, np.transpose(np.asarray(raw) - np.asarray(_MEAN), (0, 4, 1, 2, 3)))


def test_encode_output_is_channels_first_float32():
    raw = _raw_latents(2)
    out = encode_pixels_to_latents(np.zeros((1, 3, 13, 8, 20), dtype=np.float32), _StubVae(raw), None)
    assert out.shape == (1, _Z_DIM, _LATENT_T, _LATENT_H, _LATENT_W)
    assert out.dtype == jnp.float32


def test_encode_casts_the_video_to_the_vae_dtype():
    raw = _raw_latents(3)
    vae = _StubVae(raw, dtype=jnp.bfloat16)
    encode_pixels_to_latents(np.zeros((1, 3, 13, 8, 20), dtype=np.float32), vae, None)
    assert vae.encode_calls[0][0].dtype == jnp.bfloat16


# ----------------------------------------------------------------------------------
# 3. `encode_window_latents` -- shape gate, f16 cast LAST, z_i0 as a bitwise slice.
# ----------------------------------------------------------------------------------


def _window_stub(seed=0):
    raw = _raw_latents(seed)
    return _StubVae(raw), raw


def test_encode_window_returns_float16_arrays_of_the_declared_shape():
    vae, _ = _window_stub()
    z_video, z_i0 = encode_window_latents(
        np.zeros((13, 8, 20, 3), dtype=np.uint8),
        vae,
        None,
        expected_shape=(_Z_DIM, _LATENT_T, _LATENT_H, _LATENT_W),
    )
    assert z_video.shape == (_Z_DIM, _LATENT_T, _LATENT_H, _LATENT_W)
    assert z_i0.shape == (_Z_DIM, 1, _LATENT_H, _LATENT_W)
    assert z_video.dtype == np.float16 and z_i0.dtype == np.float16


def test_encode_window_casts_to_float16_only_after_normalizing():
    vae, raw = _window_stub(5)
    z_video, _ = encode_window_latents(
        np.zeros((13, 8, 20, 3), dtype=np.uint8),
        vae,
        None,
        expected_shape=(_Z_DIM, _LATENT_T, _LATENT_H, _LATENT_W),
    )
    expected = _expected_normalized(raw)[0].astype(np.float16)
    np.testing.assert_array_equal(z_video, expected)
    # Casting to f16 BEFORE the normalization rounds twice -- a bitwise-different tensor.
    early_cast = np.asarray(raw, dtype=np.float32).astype(np.float16).astype(np.float32)
    early_cast = np.transpose((early_cast - np.asarray(_MEAN)) / np.asarray(_STD), (0, 4, 1, 2, 3))
    assert not np.array_equal(z_video, early_cast[0].astype(np.float16))


def test_encode_window_z_i0_is_bitwise_the_first_latent_frame():
    vae, _ = _window_stub(6)
    z_video, z_i0 = encode_window_latents(
        np.zeros((13, 8, 20, 3), dtype=np.uint8),
        vae,
        None,
        expected_shape=(_Z_DIM, _LATENT_T, _LATENT_H, _LATENT_W),
    )
    assert z_i0.tobytes() == np.ascontiguousarray(z_video[:, :1]).tobytes()


def test_encode_window_rejects_a_latent_block_of_the_wrong_shape():
    vae, _ = _window_stub(7)
    with pytest.raises(BuildError):
        encode_window_latents(
            np.zeros((13, 8, 20, 3), dtype=np.uint8),
            vae,
            None,
            expected_shape=(_Z_DIM, _LATENT_T + 1, _LATENT_H, _LATENT_W),
        )


# ----------------------------------------------------------------------------------
# 4. Decode (V1/V3 need the inverse path) -- denormalize, then [0, 1] frames.
# ----------------------------------------------------------------------------------


def test_decode_denormalizes_channels_first_then_maps_to_zero_one():
    rng = np.random.default_rng(11)
    # Decoder output is channels-last pixels in [-1, 1] (the VAE clips to that range).
    decoded = jnp.asarray(rng.uniform(-1.0, 1.0, size=(1, 5, 6, 8, 3)), dtype=jnp.float32)
    vae = _StubVae(_raw_latents(), decoded=decoded)
    z = np.asarray(rng.normal(size=(_Z_DIM, _LATENT_T, _LATENT_H, _LATENT_W)), dtype=np.float16)

    frames = decode_latents_to_frames(z, vae, "CACHE")

    assert frames.shape == (5, 6, 8, 3)
    assert frames.dtype == np.float32
    assert float(frames.min()) >= 0.0 and float(frames.max()) <= 1.0
    np.testing.assert_allclose(frames, np.asarray(decoded)[0] / 2.0 + 0.5, rtol=1e-6, atol=1e-6)

    # The latents handed to the VAE are denormalized channels-first: z * std + mean.
    sent = np.asarray(vae.decode_calls[0][0])
    mean = np.asarray(_MEAN).reshape(1, _Z_DIM, 1, 1, 1)
    std = np.asarray(_STD).reshape(1, _Z_DIM, 1, 1, 1)
    np.testing.assert_allclose(sent, z.astype(np.float32)[None] * std + mean, rtol=1e-3, atol=1e-3)
    assert vae.decode_calls[0][1] == "CACHE"


# ----------------------------------------------------------------------------------
# 5. B7 -- V3's decode must be bfloat16-parity with the rollout evaluator.
#
# `_decode_latents_to_video` casts to torch bfloat16 before postprocessing, so a float32
# postprocess yields a slightly DIFFERENT number than the generation-time metric that V3 is
# supposed to be the ceiling for. The gated value therefore mirrors the pipeline exactly;
# the float32 number survives only as a separately named diagnostic.
# ----------------------------------------------------------------------------------


def test_decode_bfloat16_postprocess_matches_the_pipeline_video_processor():
    torch = pytest.importorskip("torch")
    from maxdiffusion.video_processor import VideoProcessor

    rng = np.random.default_rng(3)
    decoded = jnp.asarray(rng.uniform(-1.0, 1.0, size=(1, 3, 8, 12, 3)), dtype=jnp.float32)
    vae = _StubVae(_raw_latents(), decoded=decoded)
    z = np.asarray(rng.normal(size=(_Z_DIM, _LATENT_T, _LATENT_H, _LATENT_W)), dtype=np.float16)

    ours = decode_latents_to_frames(z, vae, None, postprocess="bfloat16")

    # The pipeline path verbatim: [B,T,H,W,C] -> [B,C,T,H,W] -> torch bf16 -> postprocess_video.
    reference = np.transpose(np.asarray(decoded, dtype=np.float32), (0, 4, 1, 2, 3))
    reference = torch.from_numpy(reference).to(dtype=torch.bfloat16)
    reference = VideoProcessor(vae_scale_factor=16).postprocess_video(reference, output_type="np")[0]
    np.testing.assert_array_equal(ours, reference)


def test_decode_bfloat16_and_float32_postprocess_differ_but_stay_close():
    rng = np.random.default_rng(4)
    decoded = jnp.asarray(rng.uniform(-1.0, 1.0, size=(1, 3, 8, 12, 3)), dtype=jnp.float32)
    vae = _StubVae(_raw_latents(), decoded=decoded)
    z = np.zeros((_Z_DIM, _LATENT_T, _LATENT_H, _LATENT_W), dtype=np.float16)
    bf16 = decode_latents_to_frames(z, vae, None, postprocess="bfloat16")
    f32 = decode_latents_to_frames(z, vae, None, postprocess="float32")
    assert not np.array_equal(bf16, f32)  # the bf16 rounding is real, not cosmetic
    assert np.abs(bf16 - f32).max() < 0.01
    assert bf16.dtype == np.float32 and f32.dtype == np.float32


def test_decode_defaults_to_the_float32_diagnostic_postprocess():
    rng = np.random.default_rng(5)
    decoded = jnp.asarray(rng.uniform(-1.0, 1.0, size=(1, 3, 8, 12, 3)), dtype=jnp.float32)
    vae = _StubVae(_raw_latents(), decoded=decoded)
    z = np.zeros((_Z_DIM, _LATENT_T, _LATENT_H, _LATENT_W), dtype=np.float16)
    np.testing.assert_array_equal(
        decode_latents_to_frames(z, vae, None), decode_latents_to_frames(z, vae, None, postprocess="float32")
    )


def test_decode_rejects_an_unknown_postprocess_mode():
    vae = _StubVae(_raw_latents(), decoded=jnp.zeros((1, 2, 3, 4, 3)))
    with pytest.raises(BuildError):
        decode_latents_to_frames(
            np.zeros((_Z_DIM, _LATENT_T, _LATENT_H, _LATENT_W), np.float16), vae, None, postprocess="float64"
        )
