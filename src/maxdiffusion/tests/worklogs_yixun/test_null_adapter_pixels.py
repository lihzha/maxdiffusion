"""exp_04 R7 — ``null_adapter_pixels``: the decoded-pixel metrics that complete the gates tables.

R6 emitted tables with ``future_ssim`` deliberately absent, which the gates read as *invalid*: every
observation, every pair, every verdict. This round discharges that contract, so the headline test is
an end-to-end one -- a real ``run_capacity_example_batch`` result, decoded through a toy VAE, scored,
filled, and handed to the real ``gate_g1``, which must now report **zero invalid pairs**.

What carries the round:

- **The frame mapping.** The Wan VAE upsamples 9 latent frames to 33 pixel frames at temporal stride
  4, and latent frame 0 -- the pinned image condition -- becomes pixel frame 0 *alone*. So "future"
  is pixel frames 1..32, and dropping exactly one frame is the whole difference between measuring the
  method and measuring the pin. The toy decoder here performs the same 1 + 4*(F-1) upsampling, so the
  bookkeeping under test is the real bookkeeping.
- **SSIM parity with the deployed metric.** ``generate_wan_side_adapter._frame_ssim`` (lines 255-279)
  is the metric this experiment's earlier numbers were computed with: ``channel_axis=-1``,
  ``data_range=1.0``, ``win_size = min(7, H, W)`` forced odd. The tests pin those semantics against
  direct ``skimage`` calls rather than against our own wrapper -- including the tiny-image case, where
  a hardcoded ``win_size=7`` raises inside skimage instead of quietly scoring something else.
- **Decode once, decode in range.** The ground truth is decoded exactly once per call (counted), and a
  decoder that returns ``[-1, 1]`` -- the other common VAE post-processing convention -- is rejected
  with a matched message instead of silently halving every SSIM.
- **Filling is fail-closed.** ``fill_pixel_metrics`` refuses any (method, name, seed) namespace that is
  not exactly the tables', because a partial fill is indistinguishable, at the gate, from a run that
  legitimately produced no SSIM.

No real VAE and no real transformer: the decoder is injected, and the one real R6 run uses the same
toy velocity oracle R6's own tests use.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from skimage.metrics import structural_similarity

from maxdiffusion.null_adapter_gates import NoiseConvention, gate_g1, parse_table
from maxdiffusion.null_adapter_pixels import (
    LATENT_FRAMES,
    PIXEL_FRAMES,
    PIXEL_METRICS,
    PROBE_K_SET,
    SINGLE_SEED_KEY,
    TEMPORAL_STRIDE,
    comparison_video_frames,
    decode_and_score,
    fill_pixel_metrics,
    frame_ssim_series,
    full_pixel_mse,
    full_ssim,
    future_frame_pixel_mse,
    future_frame_ssim,
    save_video_mp4,
)
from maxdiffusion.null_adapter_records import PRODUCTION_GEOMETRY


_LATENT_SHAPE = PRODUCTION_GEOMETRY.z_video  # (48, 9, 12, 20)
_H, _W = _LATENT_SHAPE[2], _LATENT_SHAPE[3]  # the toy decoder keeps the latent grid as its pixel grid
_NAMES = ("droid_ep_000001/w0", "droid_ep_000042/w3")


def _toy_decode(latents):
    """A stand-in VAE: channel mean per latent frame, Wan temporal upsampling, squashed into [0, 1].

    Latent frame 0 becomes pixel frame 0 alone and each later latent frame becomes four pixel frames,
    i.e. the same 9 -> 33 mapping the real decoder performs.
    """
    z = np.asarray(latents, np.float32)
    per_latent_frame = z.mean(axis=1)  # [B, 9, 12, 20]
    frames = np.concatenate(
        [per_latent_frame[:, :1], np.repeat(per_latent_frame[:, 1:], TEMPORAL_STRIDE, axis=1)], axis=1
    )
    squashed = 0.5 + 0.4 * np.tanh(frames)  # strictly inside [0, 1], so range checks stay honest
    return np.repeat(squashed[..., None], 3, axis=-1).astype(np.float32)


def _one_pixel(pixels, value):
    """One pixel moved out of range -- the smallest possible violation of the decode contract."""
    pixels = np.asarray(pixels).copy()
    pixels[0, 1, 2, 3, 0] = value
    return pixels


def _counting_decode(inner=_toy_decode):
    calls = []

    def decode_fn(latents):
        calls.append(np.asarray(latents).copy())
        return inner(latents)

    return decode_fn, calls


def _latents(names=_NAMES, seed=0):
    return np.random.default_rng(seed).standard_normal((len(names), *_LATENT_SHAPE), dtype=np.float32)


def _arm_latents(names=_NAMES):
    """The shape ``ArmResults.final_latents`` has: [B, ...] per single-noise arm, [K, B, ...] per probe."""
    return {
        "a0": _latents(names, 1),
        "a1": _latents(names, 2),
        "a2": _latents(names, 3),
        "a2_0": _latents(names, 4),
        "a1_probe": np.stack([_latents(names, 10 + k) for k in PROBE_K_SET]),
        "a2_probe": np.stack([_latents(names, 20 + k) for k in PROBE_K_SET]),
    }


def _video(frames=PIXEL_FRAMES, height=_H, width=_W, seed=0):
    return np.random.default_rng(seed).random((frames, height, width, 3), dtype=np.float32)


def _direct_series(pred, gt, win_size=7):
    """Straight from skimage, in the reference's argument order (target first, prediction second)."""
    return np.array(
        [structural_similarity(g, p, channel_axis=-1, data_range=1.0, win_size=win_size) for p, g in zip(pred, gt)]
    )


def test_identical_videos_score_exactly_one_every_frame():
    video = _video()

    series = frame_ssim_series(video, video)

    assert series.shape == (PIXEL_FRAMES,)
    np.testing.assert_array_equal(series, np.ones(PIXEL_FRAMES))


def test_the_series_matches_a_direct_skimage_call():
    """Guard the guard: the wrapper is compared against skimage itself, not against another wrapper."""
    gt, pred = _video(seed=1), _video(seed=2)

    series = frame_ssim_series(pred, gt)

    np.testing.assert_allclose(series, _direct_series(pred, gt, win_size=7), rtol=0, atol=0)


@pytest.mark.parametrize("height, expected_win", [(12, 7), (7, 7), (5, 5), (4, 3), (3, 3)])
def test_the_window_follows_the_smaller_spatial_dimension_and_stays_odd(height, expected_win):
    """``min(7, H, W)`` forced odd -- ``_frame_ssim``'s rule. A hardcoded 7 raises inside skimage for
    H < 7, which is exactly how the mutation battery catches a broken window."""
    gt, pred = _video(frames=3, height=height, seed=3), _video(frames=3, height=height, seed=4)

    series = frame_ssim_series(pred, gt)

    np.testing.assert_allclose(series, _direct_series(pred, gt, win_size=expected_win), rtol=0, atol=0)
    if expected_win != 7:
        with pytest.raises(ValueError):  # ... and 7 genuinely does not work at this size
            _direct_series(pred, gt, win_size=7)


def test_a_degenerate_frame_is_refused_rather_than_scored_as_nan():
    """The reference returns NaN here; a NaN metric reaches the gates as a merely-invalid observation,
    so this round fails loudly at the source instead (documented deviation)."""
    gt, pred = _video(frames=2, height=2, width=2), _video(frames=2, height=2, width=2, seed=5)

    with pytest.raises(ValueError, match="window of at least 3"):
        frame_ssim_series(pred, gt)


def test_the_metric_layer_does_not_clip_because_the_reference_does_not():
    """Parity is the point: ``_frame_ssim`` hands skimage whatever it was given, so clipping here --
    however tempting for out-of-range pixels -- would make these numbers a different metric from the
    ones already recorded for pre_context (Codex R7 review, finding 2). The seam keeps pixels in
    range; the metric keeps its parity."""
    gt = _video(seed=6)
    hot = gt.copy()
    hot[2] = gt[2] + 0.5  # a whole frame pushed above 1

    series = frame_ssim_series(hot, gt)

    np.testing.assert_array_equal(series, _direct_series(hot, gt, win_size=7))
    # ... and clipping really would have produced a different number, so the assertion has teeth.
    assert not np.allclose(series, _direct_series(np.clip(hot, 0.0, 1.0), gt, win_size=7))
    unclipped_mse = float(np.mean((hot.astype(np.float64) - gt.astype(np.float64)) ** 2))
    assert full_pixel_mse(hot, gt) == pytest.approx(unclipped_mse, rel=1e-12)
    assert full_pixel_mse(hot, gt) != pytest.approx(
        float(np.mean((np.clip(hot, 0.0, 1.0).astype(np.float64) - gt.astype(np.float64)) ** 2)), rel=1e-6
    )


@pytest.mark.parametrize("shape", [(PIXEL_FRAMES, _H, _W), (PIXEL_FRAMES, _H, _W, 4)])
def test_the_series_rejects_videos_that_are_not_frames_of_rgb(shape):
    video = np.zeros(shape, np.float32)

    with pytest.raises(ValueError, match=r"\[F, H, W, 3\]"):
        frame_ssim_series(video, video)


def test_mismatched_videos_are_refused():
    with pytest.raises(ValueError, match="same shape"):
        frame_ssim_series(_video(), _video(frames=PIXEL_FRAMES - 1))


def test_future_metrics_exclude_exactly_pixel_frame_zero():
    """Frame 0 is the pinned image condition: identical in every arm, so it measures the pin."""
    gt = _video(seed=7)
    pred = gt.copy()
    pred[0] = 1.0 - gt[0]  # frame 0 destroyed, frames 1.. untouched

    series = frame_ssim_series(pred, gt)

    assert future_frame_ssim(series) == pytest.approx(1.0)
    assert full_ssim(series) < 1.0
    assert full_ssim(series) == pytest.approx(float(np.mean(series)))
    assert future_frame_pixel_mse(pred, gt) == pytest.approx(0.0)
    # float64 on both sides: the module accumulates 33*12*20*3 squares, where float32 drifts by ~4e-8.
    expected = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    assert full_pixel_mse(pred, gt) == pytest.approx(expected, rel=1e-12)
    assert full_pixel_mse(pred, gt) > 0.0


def test_the_frame_mapping_is_the_wan_temporal_stride():
    assert (LATENT_FRAMES, TEMPORAL_STRIDE, PIXEL_FRAMES) == (9, 4, 33)
    assert PIXEL_FRAMES == 1 + TEMPORAL_STRIDE * (LATENT_FRAMES - 1)
    assert LATENT_FRAMES == PRODUCTION_GEOMETRY.z_video[1]


def test_the_seed_key_convention_agrees_with_the_runner():
    """Restated locally so this module stays numpy-only (the R4b ruling); pinned equal here."""
    from maxdiffusion import null_adapter_runner_core as runner

    assert (PROBE_K_SET, SINGLE_SEED_KEY) == (runner.PROBE_K_SET, runner.SINGLE_SEED_KEY)
    assert set(PIXEL_METRICS) == {"future_ssim", "full_ssim", "future_pixel_mse", "full_pixel_mse"}


def test_decode_and_score_metrics_are_hand_computable():
    arms, z_video = _arm_latents(), _latents(seed=99)

    metrics = decode_and_score(_toy_decode, arms, z_video, _NAMES)

    gt, pred = _toy_decode(z_video), _toy_decode(arms["a1"])
    for index, name in enumerate(_NAMES):
        entry = metrics["a1"][name][SINGLE_SEED_KEY]
        series = _direct_series(pred[index], gt[index])

        assert entry["future_ssim"] == pytest.approx(float(series[1:].mean()))
        assert entry["full_ssim"] == pytest.approx(float(series.mean()))
        assert entry["future_pixel_mse"] == pytest.approx(float(np.mean((pred[index, 1:] - gt[index, 1:]) ** 2)))
        assert entry["full_pixel_mse"] == pytest.approx(float(np.mean((pred[index] - gt[index]) ** 2)))


def test_probe_arms_are_scored_per_seed():
    arms, z_video = _arm_latents(), _latents(seed=99)

    metrics = decode_and_score(_toy_decode, arms, z_video, _NAMES)

    gt = _toy_decode(z_video)
    assert set(metrics) == set(arms)
    assert set(metrics["a1_probe"][_NAMES[0]]) == {str(k) for k in PROBE_K_SET}
    assert set(metrics["a1"][_NAMES[0]]) == {SINGLE_SEED_KEY}
    for position, k in enumerate(PROBE_K_SET):
        pred = _toy_decode(arms["a1_probe"][position])
        expected = float(_direct_series(pred[0], gt[0])[1:].mean())
        assert metrics["a1_probe"][_NAMES[0]][str(k)]["future_ssim"] == pytest.approx(expected)


def test_the_ground_truth_is_decoded_exactly_once():
    """Ten arm-seeds share one GT decode; decoding it per arm would be ten times the VAE cost."""
    arms, z_video = _arm_latents(), _latents(seed=99)
    decode_fn, calls = _counting_decode()

    decode_and_score(decode_fn, arms, z_video, _NAMES)

    single, probes = 4, 2 * len(PROBE_K_SET)
    assert len(calls) == 1 + single + probes
    np.testing.assert_array_equal(calls[0], z_video)  # ... and the ground truth goes first
    assert sum(1 for call in calls if np.array_equal(call, z_video)) == 1


@pytest.mark.parametrize(
    "decode_fn, message",
    [
        (lambda z: np.tanh(_toy_decode(z) * 2 - 1), r"outside \[0, 1\]"),  # the [-1, 1] VAE convention
        # A single pixel 5e-4 past the boundary: previously inside the excursion tolerance, and it
        # shifts SSIM by ~1e-5 (Codex R7 review, finding 2). The contract is now exact.
        (lambda z: _one_pixel(_toy_decode(z), 1.0 + 5e-4), r"outside \[0, 1\]"),
        (lambda z: _one_pixel(_toy_decode(z), -5e-4), r"outside \[0, 1\]"),
        (lambda z: _toy_decode(z)[:, :-1], r"\[B, 33, H, W, 3\]"),  # 32 frames: an off-by-one upsampler
        (lambda z: _toy_decode(z)[..., :1], r"\[B, 33, H, W, 3\]"),  # greyscale
        (lambda z: _toy_decode(z)[0], r"\[B, 33, H, W, 3\]"),  # batch axis dropped
        (lambda z: _toy_decode(z) * np.nan, "non-finite"),
    ],
)
def test_a_misbehaving_decoder_is_refused(decode_fn, message):
    with pytest.raises(ValueError, match=message):
        decode_and_score(decode_fn, _arm_latents(), _latents(seed=99), _NAMES)


def _forbidden_decode(latents):
    raise AssertionError("inputs must be validated before any decode")


def _black_frames(latents):
    """An input-independent decoder: whatever it is handed, it returns finite mid-grey frames.

    Real VAEs are not this pathological, but they are also not guaranteed to propagate NaN -- and
    with a decoder like this one, a corrupt arm scores a *perfect* reconstruction. That is the
    failure the finiteness check exists to prevent (Codex R7 review, finding 1).
    """
    return np.full((np.asarray(latents).shape[0], PIXEL_FRAMES, _H, _W, 3), 0.5, np.float32)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("target", ["z_video", "a0", "a1_probe"])
def test_non_finite_latents_are_refused_before_any_decode(target, bad):
    """Every arm and every probe seed, checked before the first VAE pass -- ``_forbidden_decode``
    proves no decode happened, including the ground truth's."""
    arms, z_video = _arm_latents(), _latents(seed=99)
    if target == "z_video":
        z_video = z_video.copy()
        z_video[0, 0, 0, 0, 0] = bad
    elif target == "a0":
        arms["a0"] = arms["a0"].copy()
        arms["a0"][1, 5, 3, 2, 1] = bad
    else:
        arms["a1_probe"] = arms["a1_probe"].copy()
        arms["a1_probe"][1, 0, 7, 4, 3, 2] = bad  # the middle probe seed only

    with pytest.raises(ValueError, match=f"{target} must be finite"):
        decode_and_score(_forbidden_decode, arms, z_video, _NAMES)


def test_a_corrupt_arm_cannot_be_scored_as_a_perfect_reconstruction():
    """The reviewer's probe, kept as a regression: NaN in a0 + an input-independent decoder used to
    produce ``future_ssim == 1.0`` as valid gate evidence."""
    arms, z_video = _arm_latents(), _latents(seed=99)
    arms["a0"] = arms["a0"].copy()
    arms["a0"][0, 0, 0, 0, 0] = np.nan

    assert decode_and_score(_black_frames, _arm_latents(), z_video, _NAMES)["a0"][_NAMES[0]]["0"]["future_ssim"] == 1.0
    with pytest.raises(ValueError, match="a0 must be finite"):
        decode_and_score(_black_frames, arms, z_video, _NAMES)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda arms, z, names: (arms, z[:, :, :4], names), "z_video must have the production shape"),
        (lambda arms, z, names: ({**arms, "a1": arms["a1"][:1]}, z, names), "a1"),
        (lambda arms, z, names: (arms, z, names[:1]), "z_video must have the production shape"),
        (lambda arms, z, names: (arms, z, (names[0], names[0])), "unique"),
        (lambda arms, z, names: (arms, z, ()), "at least one example"),
    ],
)
def test_the_latent_geometry_is_enforced_before_any_decode(mutate, message):
    arms, z_video, names = mutate(_arm_latents(), _latents(seed=99), _NAMES)

    with pytest.raises(ValueError, match=message):
        decode_and_score(_forbidden_decode, arms, z_video, names)


def _tables(names=_NAMES):
    """R6-shaped latent tables: the exact schema ``emit_metric_tables`` produces."""
    tables = {}
    for method in ("a0", "a1", "a2", "a2_0", "a1_probe", "a2_probe"):
        keys = [str(k) for k in PROBE_K_SET] if method.endswith("_probe") else [SINGLE_SEED_KEY]
        tables[method] = {
            name: {key: {"future_mse": 0.5 + index, "full_mse": 1.5 + index} for key in keys}
            for index, name in enumerate(names)
        }
    return tables


def test_filling_merges_the_pixel_metrics_into_the_latent_tables():
    tables = _tables()
    pixels = decode_and_score(_toy_decode, _arm_latents(), _latents(seed=99), _NAMES)

    filled = fill_pixel_metrics(tables, pixels)

    for method, table in filled.items():
        for name, seeds in table.items():
            for key, entry in seeds.items():
                assert set(entry) == {"future_mse", "full_mse", *PIXEL_METRICS}
                assert entry["future_mse"] == tables[method][name][key]["future_mse"]
                assert entry["future_ssim"] == pixels[method][name][key]["future_ssim"]
    assert set(tables["a1"][_NAMES[0]][SINGLE_SEED_KEY]) == {"future_mse", "full_mse"}  # input untouched


def test_the_completed_tables_are_gate_ready():
    """The R6 contract this round exists to discharge: no observation is invalid any more."""
    filled = fill_pixel_metrics(_tables(), decode_and_score(_toy_decode, _arm_latents(), _latents(seed=99), _NAMES))

    parsed = {method: parse_table(json.dumps(table)) for method, table in filled.items()}
    verdict = gate_g1(parsed["a1"], parsed["a0"], list(_NAMES), NoiseConvention.GLOBAL)

    assert verdict.numbers["coverage_ok"]
    assert verdict.numbers["invalid_pairs"] == 0 and verdict.numbers["invalid_fraction"] == 0.0
    assert "invalid_fraction" not in verdict.reasons
    assert np.isfinite(verdict.numbers["mean_ssim"])


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda pixels: {k: v for k, v in pixels.items() if k != "a2_probe"}, "methods"),
        (lambda pixels: {**pixels, "a3": pixels["a1"]}, "methods"),
        (lambda pixels: {**pixels, "a1": {**pixels["a1"], "stranger": pixels["a1"][_NAMES[0]]}}, "names"),
        (
            lambda pixels: {**pixels, "a1_probe": {n: {"0": s["0"]} for n, s in pixels["a1_probe"].items()}},
            "seed keys",
        ),
    ],
)
def test_filling_refuses_a_namespace_that_is_not_the_tables(mutate, message):
    pixels = decode_and_score(_toy_decode, _arm_latents(), _latents(seed=99), _NAMES)

    with pytest.raises(ValueError, match=message):
        fill_pixel_metrics(_tables(), mutate(pixels))


def test_filling_refuses_a_nonfinite_pixel_metric():
    pixels = decode_and_score(_toy_decode, _arm_latents(), _latents(seed=99), _NAMES)
    pixels["a1"][_NAMES[0]][SINGLE_SEED_KEY]["future_ssim"] = float("nan")

    with pytest.raises(ValueError, match="must be finite"):
        fill_pixel_metrics(_tables(), pixels)


def test_end_to_end_a_real_arm_run_becomes_a_gate_ready_table():
    """R6 -> R7 -> gates, with nothing synthetic between the runner and the verdict."""
    from maxdiffusion.null_adapter_runner_core import emit_metric_tables
    from maxdiffusion.tests.worklogs_yixun import test_null_adapter_runner_core as r6

    batch, _, results = r6._cached_run()
    tables = emit_metric_tables(results)

    pixels = decode_and_score(_toy_decode, results.final_latents, np.asarray(batch.z_video), batch.names)
    filled = fill_pixel_metrics(tables, pixels)

    parsed = {method: parse_table(json.dumps(table)) for method, table in filled.items()}
    verdict = gate_g1(parsed["a1"], parsed["a0"], list(batch.names), NoiseConvention.GLOBAL)
    assert verdict.numbers["invalid_pairs"] == 0, verdict.numbers
    assert all(np.isfinite(entry["future_ssim"]) for seeds in filled["a1"].values() for entry in seeds.values())


def test_comparison_frames_stack_ground_truth_on_top():
    gt, pred = _video(frames=4, seed=8), _video(frames=4, seed=9)

    stacked = comparison_video_frames(gt, pred)

    assert stacked.shape == (4, 2 * _H, _W, 3)
    np.testing.assert_array_equal(stacked[:, :_H], gt)
    np.testing.assert_array_equal(stacked[:, _H:], pred)
    with pytest.raises(ValueError, match="same shape"):
        comparison_video_frames(gt, pred[:2])


def test_saving_falls_back_to_a_frame_sequence_without_an_mp4_backend(tmp_path, monkeypatch):
    """Documented fallback: no imageio (or no ffmpeg plugin) still leaves the frames on disk."""
    from maxdiffusion import null_adapter_pixels as pixels

    monkeypatch.setattr(pixels, "_imageio", lambda: None)
    frames = _video(frames=3, seed=10)

    written = save_video_mp4(frames, str(tmp_path / "clip.mp4"), fps=8)

    assert written != str(tmp_path / "clip.mp4") and os.path.isdir(written)
    assert sorted(os.listdir(written)) == ["frame_0000.png", "frame_0001.png", "frame_0002.png"]


def test_saving_writes_an_mp4_when_a_backend_is_available(tmp_path):
    imageio = pytest.importorskip("imageio")
    try:
        imageio.get_writer(str(tmp_path / "probe.mp4"), format="FFMPEG", fps=8).close()
    except Exception as error:  # no ffmpeg plugin in this environment
        pytest.skip(f"imageio ffmpeg backend unavailable: {error}")

    written = save_video_mp4(_video(frames=3, seed=11), str(tmp_path / "clip.mp4"), fps=8)

    assert written == str(tmp_path / "clip.mp4") and os.path.getsize(written) > 0
    assert not os.path.exists(str(tmp_path / "clip.mp4.partial.mp4"))  # the staged name is consumed


class _PartialThenFailingWriter:
    """An mp4 backend that dies mid-clip, having already put bytes on disk."""

    def __init__(self, path):
        self.path = path
        self.written = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def append_data(self, frame):
        with open(self.path, "ab") as handle:
            handle.write(b"partial-mp4-bytes")
        self.written += 1
        if self.written == 2:
            raise RuntimeError("ffmpeg pipe broke")


def _fake_imageio(writer_factory):
    class _Module:
        @staticmethod
        def get_writer(path, format=None, fps=None):  # noqa: A002 -- imageio's own keyword
            return writer_factory(path)

    return _Module


def test_a_backend_that_dies_mid_clip_leaves_no_partial_mp4_and_warns(tmp_path, monkeypatch):
    """Transactional publish: the staged file is removed, the failure is audible, the frames survive.

    A previously published clip at the same path is removed too -- after this call the path holds this
    call's output or nothing, never the last run's.
    """
    from maxdiffusion import null_adapter_pixels as pixels

    monkeypatch.setattr(pixels, "_imageio", lambda: _fake_imageio(_PartialThenFailingWriter))
    path = str(tmp_path / "clip.mp4")
    open(path, "wb").write(b"a previously published clip")

    with pytest.warns(RuntimeWarning, match="ffmpeg pipe broke"):
        written = save_video_mp4(_video(frames=3, seed=12), path, fps=8)

    assert not os.path.exists(path) and not os.path.exists(path + ".partial.mp4")
    assert os.path.isdir(written) and len(os.listdir(written)) == 3


def test_the_mp4_is_staged_so_the_published_path_appears_only_when_complete(tmp_path, monkeypatch):
    """Atomic publication: a reader watching the output path never observes a half-encoded clip.

    Cleanup-on-failure alone cannot give this -- it only tidies up afterwards, while the window in
    which a truncated file sits at the published path is exactly what staging closes.
    """
    from maxdiffusion import null_adapter_pixels as pixels

    final = str(tmp_path / "clip.mp4")
    seen = {"paths": [], "published_mid_write": []}

    class _ObservingWriter:
        def __init__(self, path):
            seen["paths"].append(path)
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def append_data(self, frame):
            with open(self.path, "ab") as handle:
                handle.write(b"encoded-bytes")
            seen["published_mid_write"].append(os.path.exists(final))

    monkeypatch.setattr(pixels, "_imageio", lambda: _fake_imageio(_ObservingWriter))

    written = save_video_mp4(_video(frames=3, seed=17), final, fps=8)

    assert written == final and os.path.getsize(final) > 0
    assert seen["paths"] == [final + ".partial.mp4"]  # encoded under a staged name ...
    assert seen["published_mid_write"] == [False, False, False]  # ... so the path appears only at the end


def test_a_shorter_rewrite_does_not_publish_frames_of_the_previous_clip(tmp_path, monkeypatch):
    from maxdiffusion import null_adapter_pixels as pixels

    monkeypatch.setattr(pixels, "_imageio", lambda: None)
    path = str(tmp_path / "clip.mp4")

    first = save_video_mp4(_video(frames=3, seed=13), path, fps=8)
    second = save_video_mp4(_video(frames=2, seed=14), path, fps=8)

    assert first == second
    assert sorted(os.listdir(second)) == ["frame_0000.png", "frame_0001.png"]


def test_publishing_refuses_a_frame_path_blocked_by_a_file(tmp_path, monkeypatch):
    from maxdiffusion import null_adapter_pixels as pixels

    monkeypatch.setattr(pixels, "_imageio", lambda: None)
    (tmp_path / "clip_frames").write_text("not a directory")

    with pytest.raises(RuntimeError, match="not a directory"):
        save_video_mp4(_video(frames=2, seed=15), str(tmp_path / "clip.mp4"), fps=8)


def test_saving_creates_the_parent_directory(tmp_path, monkeypatch):
    from maxdiffusion import null_adapter_pixels as pixels

    monkeypatch.setattr(pixels, "_imageio", lambda: None)

    written = save_video_mp4(_video(frames=2, seed=16), str(tmp_path / "nested" / "dir" / "clip.mp4"), fps=8)

    assert os.path.isdir(written) and len(os.listdir(written)) == 2
