"""Per-view temporal-frequency weighting of the world-model denoising loss.

The loss is supervised on the future slots only. Their per-element error e (along
the frame axis) is placed after ``n_hist`` zero-error history slots — the history
is ground truth, so a predicted chunk that jumps away from it reads as a fast
error at the seam — mirror-padded at the far end, and weighted in the DFT domain by

    w_v(f) = high_v + (low_v - high_v) * exp(-f**2 / (2 * sigma**2)),

with f in cycles per window slot: ``low_v`` weights view v's slow error and
``high_v`` its fast error. Everything is linear and the window is a handful of
slots, so the weighting reduces to a fixed PSD matrix per view,

    loss_v = e^T Q_v e / n_fut,   Q_v = P^T F^-1 diag(w_v) F P / 2,

with P the zero-history + mirror padding. All weights 1 gives Q_v = I, i.e. the
plain MSE, so the knobs interpolate around the existing objective. Views are the
camera latents stacked along H, in stored order (DROID TFRecords: wrist, ext1, ext2).
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np


class FreqLossSettings(NamedTuple):
    low_weights: tuple
    high_weights: tuple
    sigma: float
    zero_history: bool


def band_weight_matrices(
    n_hist: int,
    n_fut: int,
    low_weights: Sequence[float],
    high_weights: Sequence[float],
    sigma: float,
    zero_history: bool = True,
) -> np.ndarray:
    """``(n_views, n_fut, n_fut)`` float32 Q_v with ``e^T Q_v e`` the weighted error energy."""
    low = np.asarray(low_weights, np.float64)
    high = np.asarray(high_weights, np.float64)
    if low.ndim != 1 or low.shape != high.shape:
        raise ValueError(f"low/high weights must be equal-length lists, got {low_weights} / {high_weights}")
    if (low < 0).any() or (high < 0).any():
        raise ValueError("frequency loss weights must be >= 0 (Q must stay PSD)")
    if sigma <= 0:
        raise ValueError(f"freq_loss_sigma must be > 0, got {sigma}")

    n_lead = n_hist if zero_history else 0
    n = 2 * (n_lead + n_fut)
    # P: e → [0] * n_lead, e, reversed(e), [0] * n_lead.
    pad = np.zeros((n, n_fut))
    cols = np.arange(n_fut)
    pad[n_lead + cols, cols] = 1.0
    pad[n - 1 - n_lead - cols, cols] = 1.0

    w_low = np.exp(-np.fft.fftfreq(n) ** 2 / (2 * sigma ** 2))
    fourier = np.fft.fft(np.eye(n), axis=0)
    mats = []
    for lo, hi in zip(low, high):
        w = hi + (lo - hi) * w_low
        circulant = np.fft.ifft(w[:, None] * fourier, axis=0).real  # F^-1 diag(w) F, symmetric
        mats.append(pad.T @ circulant @ pad / 2.0)
    return np.stack(mats).astype(np.float32)


def apply_band_weights(diff: jax.Array, q: np.ndarray, frame_axis: int, view_axis: int) -> jax.Array:
    """``Q_v diff`` along ``frame_axis``, with ``q.shape[0]`` views stacked on ``view_axis``.

    ``diff * apply_band_weights(diff, ...)`` summed over frames is ``e^T Q_v e`` per
    element (``diff ** 2`` when Q = I), so it drops in where ``diff ** 2`` was;
    ``2 * apply_band_weights(...)`` is the matching gradient w.r.t. the prediction.
    """
    n_views = q.shape[0]
    x = jnp.moveaxis(diff, (frame_axis, view_axis), (0, 1))
    f, h = x.shape[:2]
    if h % n_views:
        raise ValueError(f"axis {view_axis} has {h} rows, not divisible into {n_views} stacked views")
    x = x.reshape((f, n_views, h // n_views) + x.shape[2:])
    qx = jnp.einsum("vst,tv...->sv...", jnp.asarray(q, x.dtype), x, precision=jax.lax.Precision.HIGHEST)
    return jnp.moveaxis(qx.reshape((f, h) + qx.shape[3:]), (0, 1), (frame_axis, view_axis))


def settings_from_config(config, n_views: int) -> Optional[FreqLossSettings]:
    """``FreqLossSettings`` from the ``freq_loss_*`` keys, or None when disabled."""
    from maxdiffusion import max_logging, max_utils  # local: keeps model modules free of trainer imports

    if not max_utils.config_get(config, "freq_loss_weighting", False):
        return None
    low = tuple(float(v) for v in max_utils.config_get(config, "freq_loss_low_weights", [1.0] * n_views))
    high = tuple(float(v) for v in max_utils.config_get(config, "freq_loss_high_weights", [1.0] * n_views))
    if len(low) != n_views or len(high) != n_views:
        raise ValueError(f"freq_loss_low_weights / freq_loss_high_weights need {n_views} entries "
                         f"(one per stacked view), got {list(low)} / {list(high)}")
    settings = FreqLossSettings(
        low_weights=low,
        high_weights=high,
        sigma=float(max_utils.config_get(config, "freq_loss_sigma", 0.2)),
        zero_history=bool(max_utils.config_get(config, "freq_loss_zero_history", True)),
    )
    max_logging.log(f"[freq_loss] per-view temporal-frequency loss weighting on: {settings}")
    return settings
