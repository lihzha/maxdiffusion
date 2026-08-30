"""Action-conditioned WAN (Ctrl-World style) trainer.

Initialises from the WAN 2.2 Ti2V 5B diffusers checkpoint, adds a small
NNX action encoder, and fine-tunes both jointly on robot trajectory data.

Architecture
------------
``WanCtrlWorldModel`` wraps the ``WanModel`` transformer together with
``NNXWanActionEncoder``. Action tokens ``(B, T_act, 4096)`` are fed through
the WAN transformer's text projection as the sole cross-attention sequence —
no changes to the transformer itself.

Training objective (per-token timestep)
----------------------------------------
Uses the same per-token timestep scheme introduced by WAN Ti2V:

* History latent frames are kept **clean** (no noise added) — unless history
  noise augmentation is enabled (``history_noise_max_timestep > 0``), in which
  case each history frame is corrupted at its own small ``t_hist`` so the
  model tolerates imperfect (AR-generated) history at inference. By default
  the corruption is *blind* (history tokens stay declared t=0); set
  ``history_noise_conditioned: True`` to expose ``t_hist`` to AdaLN instead.
* Future latent frames receive flow-matching noise at a sampled global ``t``.
* A ``(B, seq_len)`` timestep array is passed to the transformer: history
  frame tokens get ``t=0`` (or ``t_hist``), future frame tokens get the
  sampled ``t``.
* The WAN model's AdaLN modulation therefore tells each block whether it is
  looking at a clean reference frame (history) or a frame being denoised
  (future), exactly as in Ti2V inference.
* MSE loss is computed **only on future latent frames**.

Conditioning
------------
Action tokens are the sole conditioning signal. Text is not used: the action
encoder is called with ``text_embed=None``, matching the text-free inference
path. 5 % of samples have their action tokens zeroed for classifier-free
guidance.

``action_cond_mode`` (config) selects how the action tokens reach the
transformer:

* ``"cross_attn"`` (default): action tokens are used as the sole
  cross-attention K/V sequence, exactly as in the original design.
* ``"adaln"``: cross-attention instead receives all-zero tokens (the same
  no-op state used for the CFG-uncond branch) and the action tokens are
  projected (``NNXWanActionAdaLNProjector``) and summed into the per-token
  timestep embedding that drives AdaLN modulation.
* ``"skeleton"``: the 7-dim vector actions are not used at all. Conditioning
  comes from a rendered 2D-kinematic-skeleton video, VAE-encoded into latents
  that are token-for-token aligned with the video latents; a separate patch
  embedding (``NNXWanSkeletonPatchEmbed``) projects them and the result is
  *added* onto the video tokens inside the transformer. This is the OSCAR
  recipe (``oscar-public``: ``Wan2pt1I2VConcat.addition_patch_embedding`` +
  ``additional_embed_alpha``). Requires a dataset carrying ``skeleton_cam*``
  features, e.g. ``droid_wan_skeletal_192_320``; ``NNXWanActionEncoder`` is
  not built in this mode, so no dead action-encoder weights land in the
  checkpoint.
* ``"skeleton_adaln"``: the same skeleton latents as ``"skeleton"``, but
  patch-embedded (``NNXWanSkeletonAdaLNEmbed``) into per-token AdaLN conditioning
  instead of a video-token bias. The additive route injects once, after the patch
  embedding, and leaves the residual stream to carry the signal through the
  remaining blocks; this one re-modulates every block. It uses the *same*
  transformer argument as ``"adaln"`` (``action_hidden_states``, summed into the
  per-token time embedding), so "adaln" names one site regardless of
  representation and a cross-representation comparison is not confounded by the
  wiring. No projection or spatial repeat is needed — those exist in the vector
  route only because a 7-dim action has no spatial extent — since the
  transformer's per-token AdaLN, built for TI2V's per-token timesteps, is already
  spatially varying and the skeleton grid lines up with it token for token.
  Carries no ``skeleton_embed_alpha``.

* ``"skeleton_cross_attn"``: the same skeleton latents again, patch-embedded
  (``NNXWanSkeletonCrossAttnEmbed``) into the cross-attention K/V — the site the
  vector-action ``"cross_attn"`` mode uses. Unlike the other two skeleton routes
  this one does not get token-for-token alignment for free, because softmax over
  keys is permutation-invariant, so two mechanisms restore it structurally:
  ``frame_level_cond=True`` with ``cond_tokens_per_frame`` set to the spatial
  patch count locks latent frame k's video tokens to latent frame k's skeleton
  tokens, and ``cross_attn_rope=True`` gives ``attn2`` the same 3D RoPE
  ``attn1`` already uses, sliced per frame, so Q and K carry identical phases at
  identical grid cells and the logit peaks on the diagonal. No positional
  parameters are learned. Because the K/V is frame-locked there is no room for a
  second sequence, so the instruction is pooled onto the skeleton tokens exactly
  as ``"cross_attn"`` pools it onto the action tokens.

These two axes are independent in principle — an action *representation* (vector
actions or rendered-skeleton latents) crossed with a conditioning *site*
(cross-attention K/V, AdaLN modulation, additive in video-token space) — and the
five modes above are five of those six cells. The one not implemented is
vector-actions-as-additive (needs a broadcast from a 7-dim vector to the latent
grid).

The modes are mutually exclusive and none of their checkpoints are compatible
with each other.

Checkpointing
-------------
Combined params (transformer + action encoder) are saved with plain orbax.
Cold-start loads from the HF diffusers checkpoint via
``WanPipelineTI2V_2_2.from_pretrained``; warm restarts load the combined
orbax checkpoint.
"""

from __future__ import annotations

import datetime
import functools
import os
from typing import Any

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state
from jax.sharding import NamedSharding, PartitionSpec as P

from maxdiffusion import max_logging, max_utils
from maxdiffusion.models.wan.action_encoder_wan import (
    NNXWanActionEncoder,
    NNXWanActionAdaLNProjector,
    NNXWanSkeletonPatchEmbed,
    NNXWanSkeletonAdaLNEmbed,
    NNXWanSkeletonCrossAttnEmbed,
)
from maxdiffusion.pipelines.wan.wan_pipeline_ti2v_2p2 import WanPipelineTI2V_2_2
from maxdiffusion.schedulers import FlaxFlowMatchScheduler
from maxdiffusion.train_utils import load_next_batch


# ── Action grouping ───────────────────────────────────────────────────────────


def _group_actions(actions: jnp.ndarray, F_lat: int) -> jnp.ndarray:
    """Map (B, 4*F_lat, 7) raw actions → (B, F_lat, 4, 7).

    The dataset pre-aligns actions so that index 4k..4k+3 corresponds to
    latent frame k (both history and future).
    """
    B = actions.shape[0]
    return actions.reshape(B, F_lat, 4, 7)


# ── Combined model ────────────────────────────────────────────────────────────


class WanCtrlWorldModel(nnx.Module):
    """WAN transformer plus whichever conditioning modules the current
    ``action_cond_mode`` needs, held together for nnx.split/merge.

    Exactly one conditioning route is live per mode, and the modules the other
    routes would need are ``None`` so they contribute no parameters to the
    checkpoint:

    * ``"cross_attn"`` — ``action_encoder`` only.
    * ``"adaln"``      — ``action_encoder`` + ``action_adaln_proj``.
    * ``"skeleton"``   — ``skeleton_embed`` only; the vector actions are unused,
      so ``action_encoder`` is ``None`` too.
    * ``"skeleton_adaln"`` — ``skeleton_adaln_embed`` only, likewise with no
      ``action_encoder``. Same conditioning signal as ``"skeleton"``, injected as
      per-token AdaLN conditioning instead of a video-token bias. The two modules
      have the same shape but are separate attributes, so their param-tree paths
      differ and the checkpoints are not interchangeable.
    * ``"skeleton_cross_attn"`` — ``skeleton_cross_attn_embed`` only, again with
      no ``action_encoder``. Same conditioning signal once more, injected at the
      cross-attention K/V site the vector-action ``"cross_attn"`` mode uses. This
      module's output width is ``wan_text_dim`` rather than ``inner_dim`` (it
      passes through the transformer's pretrained ``text_embedder``), so its
      shape differs from the other two skeleton modules as well as its path.

    Every submodule other than the transformer lives here rather than inside
    ``WanModel`` because ``create_sharded_logical_transformer`` materialises the
    transformer's params by replacing its ``nnx.eval_shape`` pytree with what it
    finds in the pretrained safetensors — anything with no counterpart there
    would survive as an unmaterialised ``ShapeDtypeStruct``.
    """

    def __init__(
        self,
        transformer,
        action_encoder: NNXWanActionEncoder | None = None,
        action_adaln_proj: NNXWanActionAdaLNProjector | None = None,
        skeleton_embed: NNXWanSkeletonPatchEmbed | None = None,
        skeleton_adaln_embed: NNXWanSkeletonAdaLNEmbed | None = None,
        skeleton_cross_attn_embed: NNXWanSkeletonCrossAttnEmbed | None = None,
    ):
        self.transformer = transformer
        self.action_encoder = action_encoder if action_encoder is not None else nnx.data(None)
        self.action_adaln_proj = action_adaln_proj if action_adaln_proj is not None else nnx.data(None)
        self.skeleton_embed = skeleton_embed if skeleton_embed is not None else nnx.data(None)
        self.skeleton_adaln_embed = (
            skeleton_adaln_embed if skeleton_adaln_embed is not None else nnx.data(None)
        )
        self.skeleton_cross_attn_embed = (
            skeleton_cross_attn_embed if skeleton_cross_attn_embed is not None else nnx.data(None)
        )


# ── TrainState ────────────────────────────────────────────────────────────────


class TrainState(train_state.TrainState):
    graphdef: nnx.GraphDef
    rest_of_state: nnx.State


# ── Helpers ───────────────────────────────────────────────────────────────────


def _dtype(name: str | jnp.dtype) -> jnp.dtype:
    if isinstance(name, jnp.dtype):
        return name
    return {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}[name]


def _apply_cfg_dropout(
    rng: jax.Array,
    action_tokens: jnp.ndarray,
    drop_prob: float,
) -> jnp.ndarray:
    """Zero action tokens for a Bernoulli-drop_prob fraction of samples."""
    b = action_tokens.shape[0]
    keep = (jax.random.uniform(rng, (b, 1, 1)) >= drop_prob).astype(action_tokens.dtype)
    return action_tokens * keep


def _pool_text_tokens(text_embeds: jnp.ndarray) -> jnp.ndarray:
    """``(B, S, D)`` T5 token sequence → ``(B, D)`` masked mean.

    Only needed in ``cross_attn`` mode, where the per-frame locking in
    ``WanTransformerBlock`` reshapes the cross-attention K/V to
    ``(B*F_lat, K, D)`` — a shared 512-token text sequence cannot be
    concatenated into that layout without breaking the reshape. Pooling to one
    vector lets the text ride along as a per-sample bias on the action tokens,
    which is exactly how the SVD arm's cross_attn mode carries text.

    Padding rows are all-zero in the stored T5 tensor, so mean over only the
    non-zero tokens keeps short prompts from being scaled down by the padding.
    Falls back to the plain mean if a row is entirely zero (empty prompt).
    """
    keep = (jnp.abs(text_embeds).sum(axis=-1) > 0).astype(text_embeds.dtype)  # (B, S)
    n = jnp.maximum(keep.sum(axis=-1, keepdims=True), 1.0)                    # (B, 1)
    return (text_embeds * keep[..., None]).sum(axis=1) / n                    # (B, D)


def _text_routes(
    text_embeds: jnp.ndarray | None,
    use_task_instructions: bool,
    action_cond_mode: str,
) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
    """Decide where the task instruction goes, if anywhere.

    Returns ``(pooled_bias, cross_attn_tokens)``:

    * ``pooled_bias`` — ``(B, wan_text_dim)`` pooled text, added to the action
      tokens by ``_add_text_bias``. Used in ``cross_attn`` mode only.
    * ``cross_attn_tokens`` — the raw ``(B, S, wan_text_dim)`` T5 sequence, used
      as the cross-attention context in ``adaln`` mode (where it replaces the
      all-zero placeholder, since the action has moved to the timestep
      embedding).

    Both are ``None`` when instructions are off, which reproduces the original
    action-only behaviour exactly. This split mirrors the SVD trainer's, so
    "text on" means the same thing in both arms of the comparison.
    """
    if not use_task_instructions or text_embeds is None:
        return None, None
    if action_cond_mode in ("adaln", "skeleton", "skeleton_adaln"):
        # All three leave cross-attention free (adaln moves the action to the
        # timestep embedding; skeleton moves it to the video tokens;
        # skeleton_adaln to the AdaLN site), so the instruction can be the full
        # T5 sequence rather than a pooled bias.
        #
        # skeleton_cross_attn is deliberately NOT in this list: it occupies
        # cross-attention with the skeleton grid and locks it per frame, so like
        # the vector `cross_attn` route it has no room for a second K/V sequence
        # and falls through to the pooled bias below.
        return None, text_embeds
    return _pool_text_tokens(text_embeds), None


def _add_text_bias(action_tokens: jnp.ndarray, text_bias: jnp.ndarray | None) -> jnp.ndarray:
    """Broadcast-add the pooled instruction onto every action token.

    Called AFTER ``_apply_cfg_dropout``, deliberately: the instruction is
    present in every training sample and at every inference call, so there is
    nothing to gain from teaching the model a no-text branch. Keeping it out of
    the dropout also means CFG scales the *action* alone — the instruction is
    identical in the cond and uncond branches and cancels out of the delta.

    In ``adaln`` mode there is nothing to do here: the instruction is its own
    cross-attention sequence and never passes through the dropout either.
    """
    if text_bias is None:
        return action_tokens
    return action_tokens + text_bias[:, None, :].astype(action_tokens.dtype)


def _frame_level_cond(action_cond_mode: str) -> bool:
    """Whether cross-attention should be locked per latent frame.

    Two modes put a per-frame sequence in the K/V and so want the
    ``(B*F_lat, K, D)`` reshape: ``cross_attn`` (a handful of action tokens per
    frame) and ``skeleton_cross_attn`` (a full spatial grid per frame).
    ``adaln``, ``skeleton`` and ``skeleton_adaln`` leave cross-attention carrying
    a single shared sequence (the instruction, or all-zero tokens), where the
    reshape is a pure waste — a B*F_lat batch expansion over identical K/V.
    """
    return action_cond_mode in ("cross_attn", "skeleton_cross_attn")


def _xattn_tokens_per_frame(
    action_cond_mode: str,
    action_tokens_per_frame: int,
    H_lat: int,
    W_lat: int,
    patch_hw: int = 2,
) -> int:
    """Cross-attention K/V tokens per latent frame, i.e. the transformer's
    ``cond_tokens_per_frame``.

    Distinct from ``action_tokens_per_latent_frame``, which stays the *action*
    grouping (how many tokens ``NNXWanActionEncoder`` emits per frame, and how
    ``_route_action_conditioning``'s adaln branch regroups them). In
    ``skeleton_cross_attn`` the K/V is the skeleton's spatial grid instead, one
    token per video token, so the per-frame count is the spatial patch count —
    which is also exactly what makes the frame-locked reshape congruent and lets
    cross-attention RoPE line Q and K up cell for cell.
    """
    if action_cond_mode == "skeleton_cross_attn":
        return (H_lat // patch_hw) * (W_lat // patch_hw)
    return action_tokens_per_frame


def _cross_attn_rope(action_cond_mode: str) -> bool:
    """Whether ``attn2`` should apply the transformer's 3D RoPE to its Q/K.

    Only ``skeleton_cross_attn``. It is the one mode whose cross-attention K/V
    is a spatial grid congruent with the queries, so position is both meaningful
    and shared — RoPE then peaks the logit on the diagonal for free. The vector
    ``cross_attn`` route's action tokens sit on no grid (and are fewer than the
    queries), so there is no position to encode and the rope array would not even
    be shape-compatible.
    """
    return action_cond_mode == "skeleton_cross_attn"


def _placeholder_action_tokens(
    b: int,
    F_lat: int,
    tokens_per_frame: int,
    wan_text_dim: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Zero stand-in for the action tokens in ``skeleton`` mode.

    That mode has no action encoder — the conditioning is the skeleton video —
    but the cross-attention K/V sequence still has to be *some* tensor. Shaped
    like the action tokens the other modes produce so every downstream call site
    (CFG dropout, text bias, routing) stays shape-identical across modes, and
    matching what ``adaln`` already feeds cross-attention (``zeros_like`` of the
    action tokens) so the transformer sees an already-exercised sequence length.
    """
    return jnp.zeros((b, F_lat * tokens_per_frame, wan_text_dim), dtype=dtype)


def _encode_skeleton(
    skeleton_embed,
    skeleton_latents: jnp.ndarray | None,
    cfg_rng: jax.Array | None,
    drop_prob: float,
    dtype: jnp.dtype,
) -> jnp.ndarray | None:
    """Patch-embed skeleton latents into this mode's conditioning tensor.

    Shared by both skeleton routes; ``skeleton_embed`` is whichever module
    ``_skeleton_module`` selected. Either way ``(B, C, F_lat, H_lat, W_lat)``
    becomes ``(B, seq_len, inner_dim)`` — the two modules differ only in whether
    the result is alpha-scaled (``skeleton``) or not (``skeleton_adaln``), and in
    where it is injected: ``_skeleton_bias`` sends ``skeleton``'s to the
    video-token bias, ``_route_action_conditioning`` sends ``skeleton_adaln``'s
    to the AdaLN slot. Returns ``None`` when the mode is off, which is also the
    transformer's "skip the injection" signal.

    CFG dropout zeroes the *token contribution* — i.e. skips the injection — for a
    ``drop_prob`` fraction of samples, which is the true no-conditioning state.
    Zeroing the skeleton *latents* instead would not be: an empty (all-black)
    skeleton frame encodes to a perfectly ordinary nonzero latent, so a
    zero latent is off-manifold input rather than an absent condition.
    Pass ``cfg_rng=None`` (eval, and the cond branch of a guided rollout) to
    skip the mask.
    """
    if skeleton_embed is None or skeleton_latents is None:
        return None
    tokens = skeleton_embed(skeleton_latents.astype(dtype))
    if cfg_rng is not None and drop_prob > 0.0:
        keep = (jax.random.uniform(cfg_rng, (tokens.shape[0], 1, 1)) >= drop_prob)
        tokens = tokens * keep.astype(tokens.dtype)
    return tokens


def _is_skeleton_mode(action_cond_mode: str) -> bool:
    """Whether the conditioning is the rendered-skeleton video rather than the
    vector actions. True for both skeleton routes, which differ only in where
    the encoded skeleton is injected, not in what the dataset must carry or in
    whether an action encoder exists."""
    return action_cond_mode in ("skeleton", "skeleton_adaln", "skeleton_cross_attn")


def _skeleton_module(model, action_cond_mode: str):
    """The live skeleton encoder for this mode, or None in the action modes.

    ``skeleton`` and ``skeleton_adaln`` build *different* modules (different
    output widths, different injection sites) and only ever one at a time, so a
    checkpoint from one mode cannot be restored into the other.
    """
    if action_cond_mode == "skeleton":
        return model.skeleton_embed
    if action_cond_mode == "skeleton_adaln":
        return model.skeleton_adaln_embed
    if action_cond_mode == "skeleton_cross_attn":
        return model.skeleton_cross_attn_embed
    return None


def _skeleton_bias(action_cond_mode: str, skeleton_tokens):
    """The additive video-token bias, i.e. ``WanModel``'s ``skeleton_hidden_states``.

    Only ``skeleton`` uses that site. ``skeleton_adaln`` encodes the *same*
    tensor but routes it through ``_route_action_conditioning`` into
    ``action_hidden_states`` — the AdaLN site the vector-action ``adaln`` route
    already uses — so it must not also be added to the video tokens. Returns
    ``None`` for every other mode, which jit traces as an absent pytree so those
    modes are untouched.
    """
    return skeleton_tokens if action_cond_mode == "skeleton" else None


def _route_action_conditioning(
    action_tokens: jnp.ndarray,
    action_adaln_proj: NNXWanActionAdaLNProjector | None,
    action_cond_mode: str,
    tokens_per_frame_k: int,
    H_lat: int,
    W_lat: int,
    text_tokens: jnp.ndarray | None = None,
    skeleton_tokens: jnp.ndarray | None = None,
    text_bias: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Route encoded conditioning to cross-attention or AdaLN.

    ``"cross_attn"`` (default): action tokens pass through unchanged as the
    transformer's cross-attention K/V; no AdaLN conditioning is added.
    ``"adaln"``: the action tokens are projected per latent frame, then
    repeated across each frame's spatial patch tokens to align with the
    per-token timestep embedding they get summed into. Cross-attention is then
    free, so it carries ``text_tokens`` when task instructions are enabled, and
    all-zero tokens otherwise (a no-op — the same state used for CFG-uncond).
    ``"skeleton"``: nothing to route into AdaLN. The conditioning is the encoded
    skeleton, which reaches the transformer by its own ``skeleton_hidden_states``
    argument (see ``_skeleton_bias``), so ``action_tokens`` here is only the zero
    placeholder from ``_placeholder_action_tokens``.

    ``"skeleton_adaln"``: ``skeleton_tokens`` becomes the AdaLN conditioning
    directly — no projection and no spatial repeat, because a skeleton's token
    grid is already per-token and already ``inner_dim`` wide. This returns it in
    the same slot ``adaln`` returns its projected action vectors, so both
    representations reach AdaLN through one transformer argument and "adaln"
    names one site rather than two.

    ``"skeleton_cross_attn"``: the encoded skeleton *is* the cross-attention
    K/V, replacing the zero placeholder entirely. Nothing goes to AdaLN and
    nothing goes to the video-token bias. The caller pairs this with
    ``frame_level_cond=True``, ``cond_tokens_per_frame`` = the spatial patch
    count (see ``_xattn_tokens_per_frame``) and ``cross_attn_rope=True``, which
    together make latent frame k's video tokens attend to latent frame k's
    skeleton tokens with matching rotary phases.

    ``skeleton`` and ``skeleton_adaln`` handle cross-attention exactly as
    ``adaln`` does: the instruction when enabled, all-zero tokens otherwise.

    ``text_tokens`` is ignored in cross-attention mode: there the instruction
    has already been folded into ``action_tokens`` as a pooled bias, because
    per-frame locking leaves no room for a second K/V sequence.

    Returns ``(encoder_hidden_states, action_hidden_states)`` — the second
    element is ``None`` in cross-attention mode.
    """
    if action_cond_mode == "skeleton_cross_attn":
        # Cross-attention is occupied by the skeleton, and it is frame-locked, so
        # a shared 512-token T5 sequence cannot be concatenated into the
        # (B*F_lat, K, D) layout — the same constraint the vector `cross_attn`
        # route faces. `_text_routes` therefore hands this mode a POOLED bias,
        # broadcast here onto every skeleton token exactly as `_add_text_bias`
        # does for action tokens. Added after `_encode_skeleton`'s CFG mask so
        # the instruction is never dropped and cancels out of the CFG delta.
        #
        # No zero-placeholder fallback here, unlike the other skeleton modes.
        # Theirs is shape-compatible because cross-attention still carries the
        # action-shaped (B, F_lat*K, D) sequence; this mode's K/V is the skeleton
        # grid, (B, F_lat*Sp, D), and the caller has already told the transformer
        # cond_tokens_per_frame=Sp. Falling back to the placeholder would make
        # the block's `F = encoder.shape[1] // K` reshape silently wrong, so fail
        # loudly instead. Reaching this means the module was not built or the
        # batch carried no "skeleton" — a misconfiguration, not a CFG branch (the
        # uncond branch passes zeros of the correct shape, not None).
        if skeleton_tokens is None:
            raise ValueError(
                "action_cond_mode='skeleton_cross_attn' requires encoded skeleton "
                "tokens as the cross-attention K/V, but got None. Check that "
                "skeleton_cross_attn_embed was built and that the dataset carries "
                "the 'skeleton' feature (load_skeleton=True)."
            )
        enc = skeleton_tokens
        if text_bias is not None:
            enc = enc + text_bias[:, None, :].astype(enc.dtype)
        return enc, None
    if action_cond_mode in ("skeleton", "skeleton_adaln"):
        enc = (
            text_tokens.astype(action_tokens.dtype) if text_tokens is not None
            else jnp.zeros_like(action_tokens)
        )
        # skeleton_adaln shares the AdaLN slot with the vector-action `adaln`
        # route; `skeleton` leaves it empty and takes the video-token bias
        # instead. `skeleton_tokens` is None on the CFG-uncond branch, which is
        # exactly the no-conditioning state (temb left untouched).
        if action_cond_mode == "skeleton_adaln":
            return enc, skeleton_tokens
        return enc, None
    if action_cond_mode == "adaln":
        b, fk, d = action_tokens.shape
        f_lat = fk // tokens_per_frame_k
        grouped = action_tokens.reshape(b, f_lat, tokens_per_frame_k, d)
        action_temb = action_adaln_proj(grouped)                       # (B, F_lat, inner_dim)
        spatial_tokens_per_frame = (H_lat // 2) * (W_lat // 2)
        action_hidden_states = jnp.repeat(action_temb, spatial_tokens_per_frame, axis=1)
        if text_tokens is not None:
            enc = text_tokens.astype(action_tokens.dtype)
        else:
            enc = jnp.zeros_like(action_tokens)
        return enc, action_hidden_states
    return action_tokens, None


def _build_per_token_timestep(
    timesteps: jnp.ndarray,
    F_lat: int,
    H_lat: int,
    W_lat: int,
    n_hist_lat: int,
    hist_timesteps: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Build ``(B, seq_len)`` timestep array for per-token Ti2V training.

    History frame tokens receive ``t=0`` (treated as clean by AdaLN), or
    ``hist_timesteps`` — ``(B,)`` shared or ``(B, n_hist_lat)`` per-frame —
    when conditioned history noise augmentation is active; future frame
    tokens receive the sampled global timestep ``t``.

    WAN patches spatially at 2×2, so tokens_per_frame = (H_lat//2)*(W_lat//2).
    """
    tokens_per_frame = (H_lat // 2) * (W_lat // 2)
    b = timesteps.shape[0]
    n_fut = F_lat - n_hist_lat
    if hist_timesteps is None:
        hist_t = jnp.zeros((b, n_hist_lat), dtype=timesteps.dtype)
    else:
        hist_t = jnp.broadcast_to(hist_timesteps, (b, n_hist_lat)).astype(timesteps.dtype)
    fut_t = jnp.broadcast_to(timesteps[:, None], (b, n_fut))
    per_frame = jnp.concatenate([hist_t, fut_t], axis=1)          # (B, F_lat)
    return jnp.repeat(per_frame, tokens_per_frame, axis=1)        # (B, seq_len)


def _apply_history_noise(
    rng: jax.Array,
    hist_latents: jnp.ndarray,
    scheduler,
    config,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """History noise augmentation (GameNGen / diffusion-forcing style).

    Corrupts each history latent frame with flow-matching noise at its own
    independently sampled timestep t_hist ~ Uniform(0, history_noise_max_timestep).
    A history_noise_clean_prob fraction of samples keeps the entire history
    window exactly clean (t_hist=0), so the cold-start / ground-truth-history
    case stays in-distribution.

    Returns (noised_hist, hist_timesteps) with hist_timesteps shaped
    (B, F_hist) — one level per (sample, history frame).
    """
    b, _, f_hist = hist_latents.shape[:3]
    t_rng, clean_rng, noise_rng = jax.random.split(rng, 3)

    max_t = float(config.history_noise_max_timestep)
    t_hist = jax.random.uniform(t_rng, (b, f_hist), minval=0.0, maxval=max_t)
    clean = jax.random.bernoulli(
        clean_rng, float(getattr(config, "history_noise_clean_prob", 0.2)), (b, 1)
    )
    t_hist = jnp.where(clean, 0.0, t_hist)

    # Same formula as scheduler.apply_flow_match, but with an independent
    # noise level per history frame (its timesteps arg is per-sample only).
    t_norm = t_hist / float(scheduler.config.num_train_timesteps)
    sigma = (1.0 - t_norm) * scheduler.config.sigma_min + t_norm * scheduler.config.sigma_max
    sigma = sigma[:, None, :, None, None].astype(hist_latents.dtype)   # (B,1,F_hist,1,1)
    noise = jax.random.normal(noise_rng, hist_latents.shape, dtype=hist_latents.dtype)
    noised = (1.0 - sigma) * hist_latents + sigma * noise
    # The sigma formula has a sigma_min floor even at t=0; keep t_hist=0 frames
    # exactly clean instead.
    exact_clean = (t_hist == 0.0)[:, None, :, None, None]
    noised = jnp.where(exact_clean, hist_latents, noised).astype(hist_latents.dtype)
    return noised, t_hist


# ── Attention-sharpness diagnostics ────────────────────────────────────────────


def _attn_param_layer_stats(params) -> dict:
    """Per-layer stats of the self-attention (attn1) tensors that drive attention
    logit growth — the hypothesised late-training instability at the final block.

    ``scan_layers`` builds the blocks with ``nnx.vmap`` over num_layers, so each
    matched leaf is stacked as ``(num_layers, ...)`` with layer index -1 = the
    last block (L29 for the 30-block WAN 2.2 TI2V-5B). We track the qk-norm
    scales (``norm_q``/``norm_k``, whose growth directly inflates the logits and
    which the freeze fix targets) via per-layer absmax, and the Q/K projection
    kernels via per-layer Frobenius norm.

    Flash attention never materialises QK^T, so the logits/entropy themselves
    aren't observable without breaking the kernel; these parameter magnitudes are
    the cheap, kernel-agnostic proxy for the same thing. Returns {} (metrics then
    simply absent — never crashes training) if the names don't match.
    """
    def elem(p):
        for attr in ("key", "name", "idx"):
            v = getattr(p, attr, None)
            if v is not None:
                return str(v)
        return str(p)

    targets = {
        "norm_q_scale": (("attn1", "norm_q", "scale"), "absmax"),
        "norm_k_scale": (("attn1", "norm_k", "scale"), "absmax"),
        "q_kernel":     (("attn1", "query", "kernel"), "fro"),
        "k_kernel":     (("attn1", "key", "kernel"), "fro"),
    }
    stats = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        key = ".".join(elem(p) for p in path)
        for name, (needles, kind) in targets.items():
            if name in stats or not all(n in key for n in needles) or leaf.ndim < 2:
                continue
            flat = leaf.reshape(leaf.shape[0], -1).astype(jnp.float32)  # (num_layers, -1)
            per_layer = (jnp.max(jnp.abs(flat), axis=1) if kind == "absmax"
                         else jnp.sqrt(jnp.sum(flat ** 2, axis=1)))
            per_layer = jax.lax.with_sharding_constraint(per_layer, P())
            stats[f"attn/{name}_last"] = per_layer[-1]                  # final block
            stats[f"attn/{name}_max"] = jnp.max(per_layer)              # worst block
            stats[f"attn/{name}_argmax"] = jnp.argmax(per_layer).astype(jnp.float32)
    return stats


# ── Training step ─────────────────────────────────────────────────────────────



def _train_step(state: TrainState, data: dict, rng: jax.Array,
                scheduler_state, scheduler, config) -> tuple:
    """
    When grad_accum_steps == 1 (default): data leaves have shape [bsz, ...].
    When grad_accum_steps > 1: data leaves have shape [grad_accum_steps, bsz, ...];
    gradients are accumulated via jax.lax.scan before a single optimizer update.
    """
    _, noise_rng, timestep_rng, drop_rng, new_rng = jax.random.split(rng, 5)

    bsz = config.global_batch_size_to_train_on
    weights_dtype = _dtype(config.weights_dtype)
    n_hist = config.num_history_latent_frames
    grad_accum_steps = getattr(config, "grad_accum_steps", 1)

    def compute_loss(params, micro_data, n_rng, t_rng, d_rng):
        model: WanCtrlWorldModel = nnx.merge(state.graphdef, params, state.rest_of_state)

        latents = micro_data["latent"][:bsz].astype(weights_dtype)               # (B,C,F_lat,H,W)
        actions = micro_data["action"][:bsz].astype(weights_dtype)               # (B,4*F_lat,7)
        frame_positions = micro_data["frame_positions"][:bsz]                    # (B, W) int32

        b, _, F_lat, H_lat, W_lat = latents.shape

        actions_grouped = _group_actions(actions, F_lat)             # (B, F_lat, 4, 7)

        timesteps = scheduler.sample_timesteps(t_rng, b)

        future_latents = latents[:, :, n_hist:]
        n_rng, hist_rng = jax.random.split(n_rng)
        noise = jax.random.normal(n_rng, future_latents.shape, dtype=future_latents.dtype)
        noisy_future, target_future, training_weight = scheduler.apply_flow_match(
            noise, future_latents, timesteps
        )

        hist_latents = latents[:, :, :n_hist]
        hist_timesteps = None
        if getattr(config, "history_noise_max_timestep", 0) > 0:
            hist_latents, hist_t = _apply_history_noise(
                hist_rng, hist_latents, scheduler, config
            )
            # Conditioned: AdaLN is told each history frame's noise level.
            # Blind (default): history stays declared clean (t=0) — the model
            # learns to mildly distrust "clean" history, matching AR inference
            # where the true corruption level is unknown.
            if getattr(config, "history_noise_conditioned", False):
                hist_timesteps = hist_t
        noisy_latents = jnp.concatenate([hist_latents, noisy_future], axis=2)

        timestep_2d = _build_per_token_timestep(
            timesteps, F_lat, H_lat, W_lat, n_hist, hist_timesteps=hist_timesteps
        )
        timestep_2d = jax.lax.with_sharding_constraint(timestep_2d, P(("data", "fsdp", "context"), None))

        action_cond_mode = getattr(config, "action_cond_mode", "cross_attn")
        text_bias, text_tokens = _text_routes(
            micro_data.get("text_embeds", None)[:bsz]
            if micro_data.get("text_embeds", None) is not None else None,
            getattr(config, "use_task_instructions", False),
            action_cond_mode,
        )
        cond_tokens_per_frame = getattr(config, "action_tokens_per_latent_frame", 1)
        # The cross-attention K/V count differs from the action grouping in
        # skeleton_cross_attn, where the K/V is the skeleton's spatial grid.
        xattn_tokens_per_frame = _xattn_tokens_per_frame(
            action_cond_mode, cond_tokens_per_frame, H_lat, W_lat
        )
        cfg_rng, do_rng = jax.random.split(d_rng)

        if _is_skeleton_mode(action_cond_mode):
            # No action encoder in these modes; the conditioning is the skeleton
            # video, patch-embedded and injected inside the transformer (into the
            # video tokens for `skeleton`, into the shared AdaLN slot for
            # `skeleton_adaln`). CFG drops the SKELETON here, which is what the
            # rollout's uncond branch also drops.
            action_tokens = _placeholder_action_tokens(
                b, F_lat, cond_tokens_per_frame, config.wan_text_dim, latents.dtype
            )
            skeleton_tokens = _encode_skeleton(
                _skeleton_module(model, action_cond_mode),
                micro_data["skeleton"][:bsz],
                cfg_rng,
                config.ctrl_cfg_drop_prob,
                weights_dtype,
            )
        else:
            action_tokens = model.action_encoder(actions_grouped, None)  # (B, F_lat*K, 4096)
            action_tokens = _apply_cfg_dropout(cfg_rng, action_tokens, config.ctrl_cfg_drop_prob)
            skeleton_tokens = None
        action_tokens = _add_text_bias(action_tokens, text_bias)  # after dropout — never dropped

        enc_tokens, action_hidden_states = _route_action_conditioning(
            action_tokens, model.action_adaln_proj, action_cond_mode,
            cond_tokens_per_frame, H_lat, W_lat, text_tokens=text_tokens,
            skeleton_tokens=skeleton_tokens, text_bias=text_bias,
        )

        want_attn_diag = bool(getattr(config, "log_attn_activation_stats", False))
        transformer_out = model.transformer(
            hidden_states=noisy_latents,
            timestep=timestep_2d,           # (B, seq_len) → per-token AdaLN
            encoder_hidden_states=enc_tokens,
            action_hidden_states=action_hidden_states,
            skeleton_hidden_states=_skeleton_bias(action_cond_mode, skeleton_tokens),
            deterministic=False,
            rngs=nnx.Rngs(dropout=do_rng),
            # Per-frame cross-attn locking only applies when action tokens flow
            # through cross-attention. In adaln and skeleton modes
            # _route_action_conditioning leaves cross-attn carrying only the
            # instruction (or zeros), so the per-frame reshape is a wasted no-op
            # (B*F_lat batch expansion over a shared K/V) — disable it.
            frame_level_cond=_frame_level_cond(action_cond_mode),
            cond_tokens_per_frame=xattn_tokens_per_frame,
            cross_attn_rope=_cross_attn_rope(action_cond_mode),
            frame_positions=frame_positions,
            return_attn_diag=want_attn_diag,
        )
        # (num_blocks, 3) per-block [logit_absmax, rms_q_min, rms_k_min] for the
        # attention A/B diagnostic, or None when disabled. Carried through has_aux
        # (no gradient path), so it does not affect the trained model.
        model_pred, attn_diags = transformer_out if want_attn_diag else (transformer_out, None)

        diff = target_future - model_pred[:, :, n_hist:]
        sq = diff ** 2
        # d(loss)/d(pred) = -2*diff*weight; its per-sample L2 norm is the cheap
        # causal proxy for how much each episode drives the parameter gradient
        # (shared Jacobian d(pred)/d(params) cancels out of the ranking), unlike
        # latent_absmax which only measures input magnitude.
        outgrad = 2.0 * diff
        if not config.disable_training_weights:
            w = jnp.expand_dims(training_weight, (1, 2, 3, 4))
            sq = sq * w
            outgrad = outgrad * w
        axes = tuple(range(1, sq.ndim))
        per_sample_loss = jnp.mean(sq, axis=axes)                       # (B,)
        per_sample_outgrad = jnp.sqrt(jnp.sum(outgrad ** 2, axis=axes)) # (B,)
        return jnp.mean(per_sample_loss), (per_sample_loss, per_sample_outgrad, attn_diags)

    if grad_accum_steps == 1:
        def loss_fn(params):
            return compute_loss(params, data, noise_rng, timestep_rng, drop_rng)

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, (per_sample_loss, per_sample_outgrad, attn_diags)), grads = grad_fn(state.params)
        total_loss = loss
    else:
        # data leaves: [grad_accum_steps, bsz, ...]
        # Split each RNG into per-microbatch keys so every microbatch sees different noise.
        noise_rngs = jax.random.split(noise_rng, grad_accum_steps)
        timestep_rngs = jax.random.split(timestep_rng, grad_accum_steps)
        drop_rngs = jax.random.split(drop_rng, grad_accum_steps)

        # Python loop: unrolled at JIT trace time, keeping nnx.value_and_grad at
        # JIT trace level (avoids the cross-trace-level NNX graph inspection error
        # that occurs inside jax.lax.scan).
        acc_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
        total_loss = jnp.zeros((), dtype=jnp.float32)
        ps_losses, ps_outgrads, micro_diags = [], [], []

        for i in range(grad_accum_steps):
            micro_data = jax.tree_util.tree_map(lambda x, _i=i: x[_i], data)
            n_i, t_i, d_i = noise_rngs[i], timestep_rngs[i], drop_rngs[i]

            def loss_fn(params, _d=micro_data, _n=n_i, _t=t_i, _dr=d_i):
                return compute_loss(params, _d, _n, _t, _dr)

            grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
            (micro_loss, (micro_ps_loss, micro_ps_outgrad, micro_diag)), micro_grads = grad_fn(state.params)
            acc_grads = jax.tree_util.tree_map(
                lambda a, g: a + g / grad_accum_steps, acc_grads, micro_grads
            )
            total_loss = total_loss + micro_loss / grad_accum_steps
            ps_losses.append(micro_ps_loss)
            ps_outgrads.append(micro_ps_outgrad)
            if micro_diag is not None:
                micro_diags.append(micro_diag)

        grads = acc_grads
        # Concatenate micro-batches to match the flattened (accum*B,) episode_ids.
        per_sample_loss = jnp.concatenate(ps_losses, axis=0)
        per_sample_outgrad = jnp.concatenate(ps_outgrads, axis=0)
        # Attention diagnostics: mean the per-block stats across micro-batches.
        attn_diags = jnp.mean(jnp.stack(micro_diags, axis=0), axis=0) if micro_diags else None

    grad_norm = jaxopt.tree_util.tree_l2_norm(grads)
    new_state = state.apply_gradients(grads=grads)

    # Bad-batch guard: if the raw (pre-clip) global grad norm is non-finite or
    # exceeds grad_norm_skip_threshold, revert the whole update (params,
    # opt_state, step) so one bad batch can't poison Adam's moments — even
    # clipped, a spike's direction lingers in m/v for ~1/(1-b2) steps.
    skip_threshold = float(getattr(config, "grad_norm_skip_threshold", 0.0) or 0.0)
    update_skipped = jnp.zeros((), dtype=jnp.float32)
    if skip_threshold > 0.0:
        is_bad = jnp.logical_or(jnp.logical_not(jnp.isfinite(grad_norm)), grad_norm > skip_threshold)
        keep = lambda new, old: jax.tree_util.tree_map(lambda n, o: jnp.where(is_bad, o, n), new, old)
        new_state = new_state.replace(
            step=jnp.where(is_bad, state.step, new_state.step),
            params=keep(new_state.params, state.params),
            opt_state=keep(new_state.opt_state, state.opt_state),
        )
        update_skipped = is_bad.astype(jnp.float32)

    # Data diagnostics: per-sample episode ids, latent |max|, loss, and
    # output-grad norm (the causal spike-attribution signal), replicated so
    # every host can attribute a grad spike to specific batch samples. Leading
    # dims ((accum,) B) are flattened to line up with episode_ids.
    lat_abs = jnp.abs(data["latent"])
    latent_absmax_per_sample = jnp.max(lat_abs, axis=tuple(range(lat_abs.ndim - 4, lat_abs.ndim))).reshape(-1)
    latent_absmax_per_sample = jax.lax.with_sharding_constraint(latent_absmax_per_sample, P())
    episode_ids = jax.lax.with_sharding_constraint(data["episode_id"].reshape(-1), P())
    loss_per_sample = jax.lax.with_sharding_constraint(per_sample_loss.reshape(-1), P())
    outgrad_per_sample = jax.lax.with_sharding_constraint(per_sample_outgrad.reshape(-1), P())

    # Attention-sharpness tracking (computed on the post-update weights). Reveals
    # whether the qk-norm scales / Q-K kernels at the final block are growing —
    # the hypothesised driver of the late-training grad spikes.
    attn_stats = {}
    if getattr(config, "log_attn_param_stats", True):
        attn_stats = _attn_param_layer_stats(new_state.params)

    # Activation-level attention diagnostic (A vs B). attn_diags is (num_blocks, 3)
    # = per-block [logit_absmax, rms_q_min, rms_k_min]; index -1 = final block
    # (L29). logit_absmax large ⇒ logit growth (A); rms_*_min → sqrt(eps)=1e-3 ⇒
    # qk-norm denominator singularity (B). Per-step values (windowed in the loop).
    act_diag_stats = {}
    if attn_diags is not None:
        attn_diags = jax.lax.with_sharding_constraint(attn_diags, P())
        act_diag_stats = {
            "actdiag/logit_absmax_l29": attn_diags[-1, 0],
            "actdiag/logit_absmax_maxblk": jnp.max(attn_diags[:, 0]),
            "actdiag/rms_q_min_l29": attn_diags[-1, 1],
            "actdiag/rms_k_min_l29": attn_diags[-1, 2],
            "actdiag/rms_q_min_minblk": jnp.min(attn_diags[:, 1]),
            "actdiag/rms_k_min_minblk": jnp.min(attn_diags[:, 2]),
        }

    metrics = {
        "scalar": {
            "learning/loss": total_loss,
            "learning/grad_norm": grad_norm,
            "learning/latent_absmax": jnp.max(latent_absmax_per_sample),
            "learning/loss_max": jnp.max(loss_per_sample),
            "learning/outgrad_max": jnp.max(outgrad_per_sample),
            "learning/update_skipped": update_skipped,
            **attn_stats,
            **act_diag_stats,
        },
        "scalars": {
            "learning/latent_absmax_per_sample": latent_absmax_per_sample,
            "learning/loss_per_sample": loss_per_sample,
            "learning/outgrad_per_sample": outgrad_per_sample,
            "learning/episode_ids": episode_ids,
        },
    }
    return new_state, scheduler_state, metrics, new_rng


# ── Trainer ───────────────────────────────────────────────────────────────────


VALID_ACTION_COND_MODES = (
    "cross_attn",
    "adaln",
    "skeleton",
    "skeleton_adaln",
    "skeleton_cross_attn",
)


class WanCtrlWorldTrainer:
    """Self-contained trainer for action-conditioned WAN video generation."""

    def __init__(self, config):
        self.config = config
        mode = getattr(config, "action_cond_mode", "cross_attn")
        if mode not in VALID_ACTION_COND_MODES:
            raise ValueError(
                f"action_cond_mode={mode!r} is not one of {VALID_ACTION_COND_MODES}."
            )

    # ── Scheduler ─────────────────────────────────────────────────────────────

    def _create_scheduler(self):
        sched = FlaxFlowMatchScheduler(dtype=jnp.float32)
        state = sched.create_state()
        state = sched.set_timesteps(state, num_inference_steps=1000, training=True)
        return sched, state

    # ── Dataset ───────────────────────────────────────────────────────────────

    def _load_dataset(self, mesh, is_training: bool, seed: int = None):
        from maxdiffusion.input_pipeline.robot.wan_ctrl_world_dataset import (
            WanCtrlWorldDroidDataset,
        )
        from maxdiffusion.multihost_dataloading import MultiHostDataLoadIterator
        config = self.config
        split = "train" if is_training else "val"
        max_latent_frames = config.num_predicted_latents + config.num_history_latent_frames
        per_host_batch = max(1, config.global_batch_size_to_load // jax.process_count())
        ds = WanCtrlWorldDroidDataset(
            data_dir=config.train_data_dir if is_training else config.eval_data_dir,
            stats_path=config.action_stats_path,
            n_hist=config.num_history_latent_frames,
            max_latent_frames=max_latent_frames,
            action_dim=config.action_dim,
            batch_size=per_host_batch,
            split=split,
            seed=seed if seed is not None else config.seed,
            shuffle=is_training,
            shard_for_training=jax.process_count() > 1,
            load_skeleton=_is_skeleton_mode(getattr(config, "action_cond_mode", "cross_attn")),
            # Eval windows are anchored at the episode start (history = frame 0
            # repeated), matching a deployment-style cold-start rollout.
            first_window_only=not is_training,
            # Eval repeats too — see the `repeat` docstring: a draining val
            # iterator desynchronises the hosts on the collective eval step.
            repeat=True,
        )
        return MultiHostDataLoadIterator(ds.dataset, mesh)

    # ── Pipeline / model loading ───────────────────────────────────────────────

    def _load_wan_pipeline(self) -> WanPipelineTI2V_2_2:
        max_logging.log("[wan_ctrl_world] loading WAN Ti2V pipeline from pretrained")
        with nn_partitioning.axis_rules(self.config.logical_axis_rules):
            pipeline = WanPipelineTI2V_2_2.from_pretrained(self.config)
        return pipeline

    def _build_action_encoder(self) -> NNXWanActionEncoder | None:
        """The vector-action encoder, or None in the skeleton modes.

        Those modes condition on the rendered-skeleton video instead, so an
        encoder here would receive zero gradient forever — dead weights in the
        checkpoint and dead optimizer moments in HBM.
        """
        if _is_skeleton_mode(getattr(self.config, "action_cond_mode", "cross_attn")):
            return None
        return NNXWanActionEncoder(
            rngs=nnx.Rngs(jax.random.key(self.config.seed)),
            action_dim=self.config.action_dim,
            num_actions=4,  # WAN 4× temporal compression → 4 raw frames per latent frame
            hidden_dim=self.config.wan_action_encoder_hidden_dim,
            out_dim=self.config.wan_text_dim,
            tokens_per_frame=getattr(self.config, "action_tokens_per_latent_frame", 1),
            dtype=_dtype(self.config.activations_dtype),
            weights_dtype=_dtype(self.config.weights_dtype),
        )

    def _build_action_adaln_proj(self, transformer_config) -> NNXWanActionAdaLNProjector | None:
        """Only built when action_cond_mode == "adaln"; unused (None) otherwise.

        ``inner_dim`` must come from the loaded transformer's own registered
        config, not ``self.config.num_attention_heads``/``attention_head_dim``
        — those top-level yaml fields are stale for this pipeline: the real
        architecture is loaded from the pretrained checkpoint's config.json
        (see ``create_sharded_logical_transformer`` in wan_pipeline.py), which
        can (and does, for WAN 2.2 TI2V-5B) differ from the yaml defaults.
        """
        if getattr(self.config, "action_cond_mode", "cross_attn") != "adaln":
            return None
        inner_dim = transformer_config.num_attention_heads * transformer_config.attention_head_dim
        return NNXWanActionAdaLNProjector(
            rngs=nnx.Rngs(jax.random.key(self.config.seed + 1)),
            tokens_per_frame=getattr(self.config, "action_tokens_per_latent_frame", 1),
            wan_text_dim=self.config.wan_text_dim,
            inner_dim=inner_dim,
            dtype=_dtype(self.config.activations_dtype),
            weights_dtype=_dtype(self.config.weights_dtype),
        )

    def _build_skeleton_embed(self, transformer_config) -> NNXWanSkeletonPatchEmbed | None:
        """Only built when action_cond_mode == "skeleton"; None otherwise.

        Both ``inner_dim`` and ``patch_size`` come from the *loaded*
        transformer's registered config rather than the top-level yaml, for the
        same reason ``_build_action_adaln_proj`` does: this pipeline reads its
        real architecture from the pretrained checkpoint's config.json, which
        differs from the yaml defaults for WAN 2.2 TI2V-5B. ``patch_size``
        especially has to match exactly, or the skeleton token grid would not
        line up with the video token grid it is added to.

        ``in_channels`` is the transformer's own ``in_channels`` — the skeleton
        video goes through the same VAE as the RGB video, so its latents have
        the same channel count by construction.
        """
        if getattr(self.config, "action_cond_mode", "cross_attn") != "skeleton":
            return None
        inner_dim = transformer_config.num_attention_heads * transformer_config.attention_head_dim
        return NNXWanSkeletonPatchEmbed(
            rngs=nnx.Rngs(jax.random.key(self.config.seed + 2)),
            in_channels=transformer_config.in_channels,
            inner_dim=inner_dim,
            patch_size=tuple(transformer_config.patch_size),
            alpha=float(getattr(self.config, "skeleton_embed_alpha", 0.1)),
            dtype=_dtype(self.config.activations_dtype),
            weights_dtype=_dtype(self.config.weights_dtype),
        )

    def _build_skeleton_adaln_embed(self, transformer_config) -> NNXWanSkeletonAdaLNEmbed | None:
        """Only built when action_cond_mode == "skeleton_adaln"; None otherwise.

        Same reason as ``_build_skeleton_embed`` for reading ``inner_dim``,
        ``patch_size`` and ``in_channels`` off the *loaded* transformer config
        rather than the yaml: those top-level fields are stale for WAN 2.2
        TI2V-5B. ``patch_size`` has to match exactly or the modulation token grid
        would not line up with the video token grid it modulates.

        Takes no ``skeleton_embed_alpha`` — see ``NNXWanSkeletonAdaLNEmbed``'s
        docstring for why this route drops it. Setting that flag has no effect in
        this mode, which is why the startup banner says so explicitly.
        """
        if getattr(self.config, "action_cond_mode", "cross_attn") != "skeleton_adaln":
            return None
        inner_dim = transformer_config.num_attention_heads * transformer_config.attention_head_dim
        return NNXWanSkeletonAdaLNEmbed(
            rngs=nnx.Rngs(jax.random.key(self.config.seed + 3)),
            in_channels=transformer_config.in_channels,
            inner_dim=inner_dim,
            patch_size=tuple(transformer_config.patch_size),
            dtype=_dtype(self.config.activations_dtype),
            weights_dtype=_dtype(self.config.weights_dtype),
        )

    def _build_skeleton_cross_attn_embed(
        self, transformer_config
    ) -> NNXWanSkeletonCrossAttnEmbed | None:
        """Only built when action_cond_mode == "skeleton_cross_attn"; None otherwise.

        ``patch_size`` and ``in_channels`` come off the *loaded* transformer
        config for the same reason as the other two skeleton builders (the yaml
        fields are stale for WAN 2.2 TI2V-5B). ``patch_size`` matters more here
        than anywhere else: it fixes the skeleton token grid, and the frame-locked
        cross-attention reshape plus its RoPE both assume that grid is congruent
        with the video token grid cell for cell.

        Note this one takes ``wan_text_dim``, not ``inner_dim``. Cross-attention
        context enters ``WanModel`` *before* ``condition_embedder.text_embedder``
        projects 4096 -> inner_dim, so emitting 4096 routes the skeleton through
        the identical path the action tokens take. Takes no
        ``skeleton_embed_alpha``, matching ``skeleton_adaln``.
        """
        if getattr(self.config, "action_cond_mode", "cross_attn") != "skeleton_cross_attn":
            return None
        return NNXWanSkeletonCrossAttnEmbed(
            rngs=nnx.Rngs(jax.random.key(self.config.seed + 4)),
            in_channels=transformer_config.in_channels,
            wan_text_dim=self.config.wan_text_dim,
            patch_size=tuple(transformer_config.patch_size),
            dtype=_dtype(self.config.activations_dtype),
            weights_dtype=_dtype(self.config.weights_dtype),
        )

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _build_checkpoint_manager(self, ckpt_dir: str) -> ocp.CheckpointManager:
        os.makedirs(ckpt_dir, exist_ok=True)
        keep_period = getattr(self.config, "checkpoint_keep_period", -1) or None
        options = ocp.CheckpointManagerOptions(
            create=True,
            max_to_keep=3,
            enable_async_checkpointing=True,
            keep_period=keep_period,
        )
        return ocp.CheckpointManager(
            ckpt_dir,
            item_names=("params", "opt_state", "step"),
            item_handlers={
                "params":    ocp.StandardCheckpointHandler(),
                "opt_state": ocp.StandardCheckpointHandler(),
                "step":      ocp.JsonCheckpointHandler(),
            },
            options=options,
        )

    def _save_checkpoint(self, mgr: ocp.CheckpointManager, step: int, state: TrainState):
        if jax.process_index() == 0:
            max_logging.log(f"[wan_ctrl_world] saving checkpoint at step {step}")
        mgr.save(
            step,
            args=ocp.args.Composite(
                params=ocp.args.StandardSave(state.params),
                opt_state=ocp.args.StandardSave(state.opt_state),
                step=ocp.args.JsonSave({"step": int(step)}),
            ),
        )

    def _maybe_restore(
        self,
        mgr: ocp.CheckpointManager,
        state: TrainState,
    ) -> tuple[TrainState, int]:
        latest = mgr.latest_step()
        if latest is None:
            return state, 0
        max_logging.log(f"[wan_ctrl_world] restoring combined checkpoint at step {latest}")
        restored = mgr.restore(
            latest,
            args=ocp.args.Composite(
                params=ocp.args.StandardRestore(state.params),
                opt_state=ocp.args.StandardRestore(state.opt_state),
                step=ocp.args.JsonRestore(),
            ),
        )
        restored_step = int(restored["step"]["step"])
        # step isn't consumed by the optimizer (bias correction and the LR
        # schedule run off counts inside opt_state, restored above) but keep it
        # consistent for anything that reads state.step.
        new_state = state.replace(
            params=restored["params"],
            opt_state=restored["opt_state"],
            step=jnp.asarray(restored_step, dtype=jnp.int32),
        )
        return new_state, restored_step

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def _build_optimizer(self, num_steps: int):
        schedule_steps = (
            self.config.learning_rate_schedule_steps
            if self.config.learning_rate_schedule_steps > 0
            else num_steps
        )
        schedule_type = getattr(self.config, "learning_rate_schedule_type", "constant")
        end_ratio = getattr(self.config, "learning_rate_end_ratio", 0.0)
        lr_schedule = max_utils.create_learning_rate_schedule(
            self.config.learning_rate,
            schedule_steps,
            self.config.warmup_steps_fraction,
            num_steps,
            schedule_type=schedule_type,
            end_value=self.config.learning_rate * end_ratio,
        )
        tx = max_utils.create_optimizer(self.config, lr_schedule)
        return tx, lr_schedule

    # ── Sharding ──────────────────────────────────────────────────────────────

    def _shard_state(self, mesh, state: TrainState) -> tuple[TrainState, Any]:
        with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            state_spec = nnx.get_partition_spec(state)
            state_shardings = nnx.get_named_sharding(state, mesh)
            state = jax.lax.with_sharding_constraint(state, state_spec)
        return state, state_shardings

    def _data_shardings(self, mesh, for_eval: bool = False) -> dict:
        if not for_eval and getattr(self.config, "grad_accum_steps", 1) > 1:
            # Leading dim is the micro-batch index (local, not distributed).
            pspec = NamedSharding(mesh, P(None, *self.config.data_sharding))
        else:
            pspec = NamedSharding(mesh, P(*self.config.data_sharding))
        shardings = {
            "latent":          pspec,
            "action":          pspec,
            "text_embeds":     pspec,
            "frame_positions": pspec,
            "episode_id":      pspec,
        }
        # Must mirror the dataset's key set exactly — jit's in_shardings is a
        # pytree prefix match, so a key the batch carries but this dict omits
        # (or vice versa) is a trace-time structure mismatch, not a warning.
        if _is_skeleton_mode(getattr(self.config, "action_cond_mode", "cross_attn")):
            shardings["skeleton"] = pspec
        return shardings

    # ── Main training entry point ─────────────────────────────────────────────

    def start_training(self):
        config = self.config

        # 1. Load WAN pipeline (transformer + mesh)
        pipeline = self._load_wan_pipeline()
        mesh = pipeline.mesh

        # Free VAE — we use pre-encoded latents. When W&B video logging is
        # enabled, keep only the decoder weights (in bf16) for rollout decode.
        self._pipeline = pipeline
        if getattr(config, "wandb_video_every", 0) > 0:
            self._slim_vae_for_video_logging(pipeline)
        else:
            if hasattr(pipeline, "vae"):
                del pipeline.vae
            if hasattr(pipeline, "vae_cache"):
                del pipeline.vae_cache

        # 2. Build combined model
        action_encoder = self._build_action_encoder()
        action_adaln_proj = self._build_action_adaln_proj(pipeline.transformer.config)
        skeleton_embed = self._build_skeleton_embed(pipeline.transformer.config)
        skeleton_adaln_embed = self._build_skeleton_adaln_embed(pipeline.transformer.config)
        skeleton_xattn_embed = self._build_skeleton_cross_attn_embed(pipeline.transformer.config)
        combined = WanCtrlWorldModel(
            pipeline.transformer, action_encoder, action_adaln_proj,
            skeleton_embed, skeleton_adaln_embed, skeleton_xattn_embed,
        )

        # 3. Split combined model into (graphdef, params, rest_of_state)
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            graphdef, params, rest_of_state = nnx.split(combined, nnx.Param, ...)

        if jax.process_index() == 0:
            n_params = sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(params))
            max_logging.log(f"[wan_ctrl_world] trainable params: {n_params / 1e6:.1f}M")

        # 4. Build optimizer and train state
        tx, lr_schedule = self._build_optimizer(config.max_train_steps)
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            state = TrainState.create(
                apply_fn=graphdef.apply,
                params=params,
                tx=tx,
                graphdef=graphdef,
                rest_of_state=rest_of_state,
            )

        # 5. Shard state across devices
        state, state_shardings = self._shard_state(mesh, state)

        # Free the pre-shard param copy after sharding; GPU JAX needs it pinned until then.
        if config.hardware != "gpu":
            max_utils.delete_pytree(params)
        data_shardings = self._data_shardings(mesh)

        # 6. Scheduler
        scheduler, scheduler_state = self._create_scheduler()

        # 7. Checkpoint manager + possible restore
        ckpt_dir = config.checkpoint_dir or os.path.join(config.output_dir, "checkpoints")
        ckpt_mgr = self._build_checkpoint_manager(ckpt_dir)
        state, start_step = self._maybe_restore(ckpt_mgr, state)
        if start_step:
            max_logging.log(f"[wan_ctrl_world] resumed at step {start_step}")

        # 8. Data iterator — offset seed by start_step so resume sees a fresh shuffle order.
        train_iter = self._load_dataset(mesh, is_training=True, seed=config.seed + start_step)

        grad_accum_steps = getattr(config, "grad_accum_steps", 1)

        def _next_batch(iterator):
            """Return one batch (grad_accum_steps == 1) or a stacked micro-batch tensor."""
            if grad_accum_steps == 1:
                return next(iterator)
            bufs = [next(iterator) for _ in range(grad_accum_steps)]
            return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *bufs)

        # 9. Compile train step
        p_train_step = jax.jit(
            functools.partial(_train_step, scheduler=scheduler, config=config),
            in_shardings=(state_shardings, data_shardings, None, None),
            out_shardings=(state_shardings, None, None, None),
            donate_argnums=(0,),
        )

        # 10. Training loop
        if jax.process_index() == 0:
            max_logging.log("***** Running WAN Ctrl-World training *****")
            max_logging.log(f"  Per-device batch size: {config.per_device_batch_size}")
            max_logging.log(f"  Devices: {jax.device_count()}")
            max_logging.log(f"  Grad accum steps: {grad_accum_steps}")
            max_logging.log(f"  Effective batch size: {config.global_batch_size_to_train_on * grad_accum_steps}")
            max_logging.log(f"  Max train steps: {config.max_train_steps}")
            max_logging.log(f"  Output dir: {config.output_dir}")
            _acm = getattr(config, "action_cond_mode", "cross_attn")
            if _acm == "skeleton":
                max_logging.log(
                    f"  Action conditioning: skeleton — VAE latents of the rendered "
                    f"2D skeleton video, patch-embedded (alpha="
                    f"{float(getattr(config, 'skeleton_embed_alpha', 0.1))}) and added "
                    f"to the video tokens; vector actions unused"
                )
            elif _acm == "skeleton_adaln":
                max_logging.log(
                    "  Action conditioning: skeleton_adaln — VAE latents of the "
                    "rendered 2D skeleton video, patch-embedded (no alpha; "
                    "zero-init) and summed into the per-token timestep embedding, "
                    "the same AdaLN site the vector-action 'adaln' mode uses, so it "
                    "re-modulates every block; vector actions unused"
                )
            elif _acm == "skeleton_cross_attn":
                max_logging.log(
                    "  Action conditioning: skeleton_cross_attn — VAE latents of the "
                    "rendered 2D skeleton video, patch-embedded (no alpha; zero-init) "
                    "into the cross-attention K/V, the same site the vector-action "
                    "'cross_attn' mode uses; frame-locked (one skeleton token per "
                    "video token, per latent frame) with 3D RoPE on the cross-attn "
                    "Q/K so the grids line up cell for cell; vector actions unused"
                )
            else:
                max_logging.log(f"  Action conditioning: {_acm} — 7-dim vector actions")
            if max_utils.config_get(config, "use_task_instructions", False):
                # cross_attn is the only mode that pools; adaln and skeleton both
                # leave cross-attention free for the full T5 sequence (see
                # _text_routes).
                if _acm == "cross_attn":
                    _route = "pooled into the action tokens"
                elif _acm == "skeleton_cross_attn":
                    # Same constraint as cross_attn: the K/V is frame-locked, so
                    # a shared T5 sequence does not fit the (B*F, K, D) layout.
                    _route = "pooled into the skeleton tokens"
                else:
                    _route = "the cross-attention context"
                max_logging.log(
                    f"  Task instructions: ON — T5 text is {_route}; not CFG-dropped"
                )
            elif _acm == "skeleton_cross_attn":
                max_logging.log(
                    "  Task instructions: OFF — the rendered skeleton video is the "
                    "ONLY conditioning signal (and it *is* the cross-attention K/V)"
                )
            elif _is_skeleton_mode(_acm):
                max_logging.log(
                    "  Task instructions: OFF — the rendered skeleton video is the "
                    "ONLY conditioning signal (cross-attention gets zero tokens)"
                )
            else:
                max_logging.log("  Task instructions: OFF — action-only conditioning")

        self._wandb_run = None
        if jax.process_index() == 0 and getattr(config, "wandb_project", ""):
            import wandb
            self._wandb_run = wandb.init(
                project=config.wandb_project,
                entity=getattr(config, "wandb_entity", None) or None,
                name=config.run_name or None,
            )
        wandb_run = self._wandb_run

        rng = jax.random.key(config.seed + 1)
        recent_loss: list[float] = []
        recent_grad: list[float] = []
        recent_absmax: list[float] = []
        recent_loss_max: list[float] = []
        recent_outgrad_max: list[float] = []
        # Attention A/B activation diagnostic: window-reduced so a transient spike
        # step (logit max ↑ or rms min ↓) is never averaged away. Empty unless
        # log_attn_activation_stats is on.
        recent_logit_l29: list[float] = []
        recent_logit_maxblk: list[float] = []
        recent_rms_q_l29: list[float] = []
        recent_rms_k_l29: list[float] = []
        skipped_count = 0
        skip_threshold = float(getattr(config, "grad_norm_skip_threshold", 0.0) or 0.0)
        last_step_time = datetime.datetime.now()

        profiler = None
        first_profiling_step = config.skip_first_n_steps_for_profiler
        if max_utils.profiler_enabled(config) and first_profiling_step >= config.max_train_steps:
            raise ValueError("Profiling requested but initial profiling step set past training final step")
        last_profiling_step = np.clip(
            first_profiling_step + config.profiler_steps - 1,
            first_profiling_step,
            config.max_train_steps - 1,
        )

        example_batch = _next_batch(train_iter)

        for step in range(start_step, config.max_train_steps):
            if max_utils.profiler_enabled(config) and step == first_profiling_step:
                profiler = max_utils.Profiler(config)
                profiler.start()
            step_start = datetime.datetime.now()
            with (
                jax.profiler.StepTraceAnnotation("train", step_num=step),
                mesh,
                nn_partitioning.axis_rules(config.logical_axis_rules),
            ):
                state, scheduler_state, metrics, rng = p_train_step(
                    state, example_batch, rng, scheduler_state
                )

            # Measure compute time before prefetching the next batch.
            # _next_batch calls jax.device_put which allocates HBM buffers; on a
            # busy TPU this forces an implicit sync with the ongoing step and
            # inflates step_secs.  We measure first, then prefetch so that the
            # load overlaps with logging / eval / checkpoint logic below.
            metrics["scalar"]["learning/loss"].block_until_ready()
            now = datetime.datetime.now()
            step_secs = (now - step_start).total_seconds()

            if profiler is not None and step == last_profiling_step:
                profiler.stop()
                profiler = None

            example_batch = _next_batch(train_iter)

            recent_loss.append(float(metrics["scalar"]["learning/loss"]))
            recent_grad.append(float(metrics["scalar"]["learning/grad_norm"]))
            recent_absmax.append(float(metrics["scalar"]["learning/latent_absmax"]))
            recent_loss_max.append(float(metrics["scalar"]["learning/loss_max"]))
            recent_outgrad_max.append(float(metrics["scalar"]["learning/outgrad_max"]))
            if "actdiag/logit_absmax_l29" in metrics["scalar"]:
                recent_logit_l29.append(float(metrics["scalar"]["actdiag/logit_absmax_l29"]))
                recent_logit_maxblk.append(float(metrics["scalar"]["actdiag/logit_absmax_maxblk"]))
                recent_rms_q_l29.append(float(metrics["scalar"]["actdiag/rms_q_min_l29"]))
                recent_rms_k_l29.append(float(metrics["scalar"]["actdiag/rms_k_min_l29"]))
            skipped_count += int(float(metrics["scalar"]["learning/update_skipped"]))

            # Grad-spike attribution: on a skipped step, name the batch's
            # episodes ranked by output-grad norm (the causal proxy for each
            # episode's gradient contribution). A single dominant episode ⇒ a
            # data/episode cause; a flat ranking ⇒ intrinsic instability with no
            # single culprit. loss and latent_absmax shown alongside.
            grad_val = recent_grad[-1]
            if skip_threshold > 0.0 and (not np.isfinite(grad_val) or grad_val > skip_threshold):
                eids = np.asarray(metrics["scalars"]["learning/episode_ids"])
                absmax = np.asarray(metrics["scalars"]["learning/latent_absmax_per_sample"])
                ps_loss = np.asarray(metrics["scalars"]["learning/loss_per_sample"])
                outgrad = np.asarray(metrics["scalars"]["learning/outgrad_per_sample"])
                if jax.process_index() == 0:
                    order = np.argsort(-outgrad)
                    share = outgrad[order[0]] / (outgrad.sum() + 1e-12)
                    offenders = ", ".join(
                        f"ep{int(eids[i])}:g={outgrad[i]:.3e},l={ps_loss[i]:.3e},a={absmax[i]:.2f}"
                        for i in order
                    )
                    max_logging.log(
                        f"[wan_ctrl_world] GRAD SPIKE step {step}: raw grad_norm={grad_val:.3e} "
                        f"(threshold {skip_threshold:g}) — update skipped. top episode "
                        f"ep{int(eids[order[0]])} holds {share:.0%} of batch output-grad. "
                        f"episode_id:outgrad(g)/loss(l)/latent_absmax(a) desc-by-g: {offenders}"
                    )

            if jax.process_index() == 0 and (step < 5 or (step + 1) % config.log_period == 0):
                max_logging.log(f"step {step} s/step={step_secs:.2f}")

            if (step + 1) % config.log_period == 0 and jax.process_index() == 0:
                lr = float(lr_schedule(step))
                avg_loss = sum(recent_loss) / len(recent_loss)
                avg_grad = sum(recent_grad) / len(recent_grad)
                sps = config.log_period / (now - last_step_time).total_seconds()
                max_logging.log(
                    f"step {step + 1}/{config.max_train_steps} "
                    f"loss={avg_loss:.4f} grad_norm={avg_grad:.3f} "
                    f"lr={lr:.2e} steps/s={sps:.2f} s/step={1/sps:.2f}"
                )
                # Attention-sharpness params vary slowly, so log the current
                # step's value (not a window reduction). "_last" = final block,
                # "_max"/"_argmax" = worst block + its index.
                attn_log = {f"train/{k.split('/', 1)[1]}": float(v)
                            for k, v in metrics["scalar"].items() if k.startswith("attn/")}
                if attn_log:
                    ql, qm, qa = (attn_log.get("train/norm_q_scale_last"),
                                  attn_log.get("train/norm_q_scale_max"),
                                  attn_log.get("train/norm_q_scale_argmax"))
                    if ql is not None:
                        max_logging.log(
                            f"  attn qk-norm scale absmax: L-last={ql:.3f} "
                            f"max={qm:.3f}@L{int(qa)}")
                elif step < config.log_period * 2:
                    max_logging.log(
                        "[wan_ctrl_world] WARN: no attn/* param stats emitted — "
                        "tensor-name match in _attn_param_layer_stats may need fixing")
                # Attention A/B diagnostic, window-reduced (max for logit-growth
                # signals, min for the qk-norm denominator signals) so the spike
                # step's extreme survives the window.
                act_log = {}
                if recent_logit_l29:
                    act_log = {
                        "train/logit_absmax_l29": max(recent_logit_l29),
                        "train/logit_absmax_maxblk": max(recent_logit_maxblk),
                        "train/rms_q_min_l29": min(recent_rms_q_l29),
                        "train/rms_k_min_l29": min(recent_rms_k_l29),
                    }
                    max_logging.log(
                        f"  attn A/B: logit_absmax L29={act_log['train/logit_absmax_l29']:.2f} "
                        f"maxblk={act_log['train/logit_absmax_maxblk']:.2f} | "
                        f"rms_min L29 q={act_log['train/rms_q_min_l29']:.4f} "
                        f"k={act_log['train/rms_k_min_l29']:.4f}  "
                        f"(logit↑⇒A; rms→1e-3⇒B)")
                if wandb_run is not None:
                    wandb_run.log({"train/loss": avg_loss, "train/grad_norm": avg_grad,
                                   "train/grad_norm_max": max(recent_grad),
                                   "train/latent_absmax": max(recent_absmax),
                                   "train/loss_max": max(recent_loss_max),
                                   "train/outgrad_max": max(recent_outgrad_max),
                                   "train/updates_skipped": skipped_count,
                                   "train/lr": lr, "train/steps_per_sec": sps,
                                   **attn_log, **act_log}, step=step + 1)
                recent_loss.clear()
                recent_grad.clear()
                recent_absmax.clear()
                recent_loss_max.clear()
                recent_outgrad_max.clear()
                recent_logit_l29.clear()
                recent_logit_maxblk.clear()
                recent_rms_q_l29.clear()
                recent_rms_k_l29.clear()
                skipped_count = 0
                last_step_time = now

            if (
                config.eval_every > 0
                and (step + 1) % config.eval_every == 0
            ):
                self._run_eval(mesh, state, state_shardings,
                               data_shardings, scheduler, scheduler_state, step + 1, rng)

            if (
                getattr(config, "wandb_video_every", 0) > 0
                and getattr(config, "wandb_project", "")
                and (step + 1) % config.wandb_video_every == 0
            ):
                self._run_video_log(mesh, state, state_shardings, scheduler, step + 1, rng)

            if (
                config.checkpoint_every > 0
                and (step + 1) % config.checkpoint_every == 0
            ):
                self._save_checkpoint(ckpt_mgr, step + 1, state)

        if config.save_final_checkpoint:
            self._save_checkpoint(ckpt_mgr, config.max_train_steps, state)
        ckpt_mgr.wait_until_finished()
        if wandb_run is not None:
            wandb_run.finish()

    # ── Eval ──────────────────────────────────────────────────────────────────

    def _run_eval(
        self,
        mesh,
        state: TrainState,
        state_shardings,
        data_shardings,
        scheduler,
        scheduler_state,
        step: int,
        rng: jax.Array,
    ):
        config = self.config
        if not config.eval_data_dir:
            max_logging.log("[wan_ctrl_world] eval_every>0 but eval_data_dir not set; skipping")
            return

        if not hasattr(self, "_eval_iter"):
            self._eval_iter = self._load_dataset(mesh, is_training=False)
        eval_iter = self._eval_iter

        if not hasattr(self, "_p_eval_step"):
            eval_data_shardings = self._data_shardings(mesh, for_eval=True)
            self._p_eval_step = jax.jit(
                functools.partial(_eval_step, scheduler=scheduler, config=config),
                in_shardings=(state_shardings, eval_data_shardings, None, None),
                out_shardings=None,
            )
        p_eval_step = self._p_eval_step

        losses: list[float] = []
        for _ in range(max(1, int(getattr(config, "eval_max_batches", 50)))):
            try:
                batch = next(eval_iter)
            except StopIteration:
                break
            rng, sub = jax.random.split(rng)
            with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
                loss = p_eval_step(state, batch, sub, scheduler_state)
                loss.block_until_ready()
            losses.append(float(loss))

        if losses and jax.process_index() == 0:
            mean_loss = sum(losses) / len(losses)
            max_logging.log(
                f"[wan_ctrl_world] eval step={step} batches={len(losses)} "
                f"mean_loss={mean_loss:.4f}"
            )
            if getattr(self, "_wandb_run", None) is not None:
                self._wandb_run.log({"eval/loss": mean_loss}, step=step)

    # ── W&B video logging ─────────────────────────────────────────────────────

    def _slim_vae_for_video_logging(self, pipeline):
        """Keep only the VAE weights video logging needs.

        Video logging only calls ``vae.decode``, which touches
        ``post_quant_conv`` and ``decoder`` — the encoder and ``quant_conv``
        weights (~150M params) are dropped, and the remaining ~555M decoder
        params are cast to bf16 (plenty for preview videos). Cuts the VAE
        footprint from ~2.8 GB fp32 to ~1.1 GB across the mesh.
        """
        vae = pipeline.vae
        vae.encoder = None
        vae.quant_conv = None
        graphdef, state = nnx.split(vae)
        # Cast inside jit: eager ops on multi-host globally-sharded arrays
        # are unsafe.
        state = jax.jit(
            functools.partial(
                jax.tree_util.tree_map,
                lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            )
        )(state)
        pipeline.vae = nnx.merge(graphdef, state)
        # The cache only uses its conv counts after construction; repoint its
        # module ref so the old fp32 VAE can be garbage-collected.
        pipeline.vae_cache.module = pipeline.vae
        max_logging.log("[wan_ctrl_world] VAE slimmed for video logging: encoder dropped, decoder cast to bf16")

    def _run_video_log(
        self,
        mesh,
        state: TrainState,
        state_shardings,
        scheduler,
        step: int,
        rng: jax.Array,
    ):
        """Rollout one eval batch, VAE-decode gen vs GT, log videos to W&B.

        Runs on every host (the rollout and decode are collective ops);
        only process 0 writes to W&B.
        """
        config = self.config
        pipeline = self._pipeline
        if not config.eval_data_dir:
            max_logging.log("[wan_ctrl_world] wandb_video_every>0 but eval_data_dir not set; skipping")
            return

        if not hasattr(self, "_eval_iter"):
            self._eval_iter = self._load_dataset(mesh, is_training=False)
        try:
            batch = next(self._eval_iter)
        except StopIteration:
            self._eval_iter = self._load_dataset(mesh, is_training=False)
            batch = next(self._eval_iter)

        if not hasattr(self, "_p_video_rollout"):
            lat_mean = jnp.array(pipeline.vae.latents_mean).reshape(1, -1, 1, 1, 1)
            lat_std = jnp.array(pipeline.vae.latents_std).reshape(1, -1, 1, 1, 1)
            self._p_video_rollout = jax.jit(
                functools.partial(
                    _video_rollout,
                    scheduler=scheduler,
                    config=config,
                    num_steps=int(getattr(config, "wandb_video_inference_steps", 20)),
                    guidance_scale=float(getattr(config, "wandb_video_guidance_scale", 1.0)),
                    num_samples=int(getattr(config, "wandb_video_samples", 1)),
                    lat_mean=lat_mean,
                    lat_std=lat_std,
                ),
                in_shardings=(state_shardings, self._data_shardings(mesh, for_eval=True), None),
                out_shardings=None,
            )

        rng, sub = jax.random.split(rng)
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            gen_lat, gt_lat, future_mse = self._p_video_rollout(state, batch, sub)
            gen_lat.block_until_ready()

        # Spatially-sharded VAE decode + cross-host allgather → host numpy
        # (B*3cams, T, H, W, C) in [0, 1].
        gen_np = np.asarray(pipeline._decode_latents_to_video(gen_lat))
        gt_np = np.asarray(pipeline._decode_latents_to_video(gt_lat))

        if jax.process_index() != 0 or getattr(self, "_wandb_run", None) is None:
            return

        import wandb

        n = gen_np.shape[0] // 3
        t, h, w, c = gen_np.shape[1:]
        gen_np = gen_np.reshape(n, 3, t, h, w, c)
        gt_np = gt_np.reshape(n, 3, t, h, w, c)

        logs = {"eval/video_rollout_latent_mse": float(future_mse)}
        for i in range(n):
            gen_grid = np.concatenate(list(gen_np[i]), axis=1)              # cameras stacked on H
            gt_grid = np.concatenate(list(gt_np[i]), axis=1)
            side_by_side = np.concatenate([gen_grid, gt_grid], axis=2)      # gen | GT on W
            # wandb.Video expects uint8 (T, C, H, W)
            frames = (side_by_side * 255).clip(0, 255).astype(np.uint8).transpose(0, 3, 1, 2)
            logs[f"eval/video/sample_{i}"] = wandb.Video(
                frames, fps=config.output_video_fps, format="mp4"
            )
        self._wandb_run.log(logs, step=step)
        max_logging.log(f"[wan_ctrl_world] logged {n} rollout video(s) to W&B at step {step}")


def _eval_step(state: TrainState, data: dict, rng: jax.Array,
               scheduler_state, scheduler, config) -> jax.Array:
    """Eval-only forward pass — no gradient computation, same per-token scheme."""
    _, noise_rng, timestep_rng = jax.random.split(rng, 3)
    bsz = config.global_batch_size_to_train_on
    weights_dtype = _dtype(config.weights_dtype)
    n_hist = config.num_history_latent_frames

    model: WanCtrlWorldModel = nnx.merge(state.graphdef, state.params, state.rest_of_state)

    latents = data["latent"][:bsz].astype(weights_dtype)
    actions = data["action"][:bsz].astype(weights_dtype)
    frame_positions = data["frame_positions"][:bsz]                  # (B, W) int32

    b, _, F_lat, H_lat, W_lat = latents.shape

    actions_grouped = _group_actions(actions, F_lat)                # (B, F_lat, 4, 7)

    timesteps = scheduler.sample_timesteps(timestep_rng, b)

    future_latents = latents[:, :, n_hist:]
    noise = jax.random.normal(noise_rng, future_latents.shape, dtype=future_latents.dtype)
    noisy_future, target_future, training_weight = scheduler.apply_flow_match(
        noise, future_latents, timesteps
    )
    noisy_latents = jnp.concatenate([latents[:, :, :n_hist], noisy_future], axis=2)

    timestep_2d = _build_per_token_timestep(timesteps, F_lat, H_lat, W_lat, n_hist)
    timestep_2d = jax.lax.with_sharding_constraint(timestep_2d, P(("data", "fsdp", "context"), None))

    action_cond_mode = getattr(config, "action_cond_mode", "cross_attn")
    text_bias, text_tokens = _text_routes(
        data.get("text_embeds", None)[:bsz] if data.get("text_embeds", None) is not None else None,
        getattr(config, "use_task_instructions", False),
        action_cond_mode,
    )
    cond_tokens_per_frame = getattr(config, "action_tokens_per_latent_frame", 1)
    xattn_tokens_per_frame = _xattn_tokens_per_frame(
        action_cond_mode, cond_tokens_per_frame, H_lat, W_lat
    )
    # No CFG dropout on the eval path, so this is just the plain conditional.
    if _is_skeleton_mode(action_cond_mode):
        action_tokens = _placeholder_action_tokens(
            b, F_lat, cond_tokens_per_frame, config.wan_text_dim, latents.dtype
        )
        skeleton_tokens = _encode_skeleton(
            _skeleton_module(model, action_cond_mode),
            data["skeleton"][:bsz], None, 0.0, weights_dtype,
        )
    else:
        action_tokens = model.action_encoder(actions_grouped, None)  # (B, F_lat*K, 4096)
        skeleton_tokens = None
    action_tokens = _add_text_bias(action_tokens, text_bias)

    enc_tokens, action_hidden_states = _route_action_conditioning(
        action_tokens, model.action_adaln_proj, action_cond_mode,
        cond_tokens_per_frame, H_lat, W_lat, text_tokens=text_tokens,
        skeleton_tokens=skeleton_tokens, text_bias=text_bias,
    )

    model_pred = model.transformer(
        hidden_states=noisy_latents,
        timestep=timestep_2d,
        encoder_hidden_states=enc_tokens,
        action_hidden_states=action_hidden_states,
        skeleton_hidden_states=_skeleton_bias(action_cond_mode, skeleton_tokens),
        deterministic=True,
        frame_level_cond=_frame_level_cond(action_cond_mode),
        cond_tokens_per_frame=xattn_tokens_per_frame,
        cross_attn_rope=_cross_attn_rope(action_cond_mode),
        frame_positions=frame_positions,
    )

    diff = target_future - model_pred[:, :, n_hist:]
    loss = diff ** 2
    if not config.disable_training_weights:
        loss = loss * jnp.expand_dims(training_weight, (1, 2, 3, 4))
    return jnp.mean(loss)


def _video_rollout(state: TrainState, data: dict, rng: jax.Array,
                   scheduler, config, num_steps: int, guidance_scale: float,
                   num_samples: int, lat_mean: jnp.ndarray, lat_std: jnp.ndarray) -> tuple:
    """Euler rollout for W&B video logging.

    History latent frames stay clean; future frames are denoised from pure
    noise conditioned on action tokens (same per-token timestep scheme as
    training). Returns ``(gen, gt, future_mse)`` where gen/gt are
    denormalized VAE latents of shape (num_samples*3, C, F_lat, H_cam, W) —
    the 3 H-stacked cameras unstacked into the batch axis (sample-major) so
    they can be VAE-decoded in one call.
    """
    bsz = config.global_batch_size_to_train_on
    weights_dtype = _dtype(config.weights_dtype)
    n_hist = config.num_history_latent_frames

    model: WanCtrlWorldModel = nnx.merge(state.graphdef, state.params, state.rest_of_state)

    latents = data["latent"][:bsz].astype(weights_dtype)          # (B,C,F_lat,3H,W)
    actions = data["action"][:bsz].astype(weights_dtype)
    frame_positions = data["frame_positions"][:bsz]

    b, _, F_lat, H_lat, W_lat = latents.shape
    actions_grouped = _group_actions(actions, F_lat)
    text_bias, text_tokens = _text_routes(
        data.get("text_embeds", None)[:bsz] if data.get("text_embeds", None) is not None else None,
        getattr(config, "use_task_instructions", False),
        getattr(config, "action_cond_mode", "cross_attn"),
    )
    action_cond_mode = getattr(config, "action_cond_mode", "cross_attn")
    cond_tokens_per_frame = getattr(config, "action_tokens_per_latent_frame", 1)
    xattn_tokens_per_frame = _xattn_tokens_per_frame(
        action_cond_mode, cond_tokens_per_frame, H_lat, W_lat
    )

    if _is_skeleton_mode(action_cond_mode):
        action_tokens = _placeholder_action_tokens(
            b, F_lat, cond_tokens_per_frame, config.wan_text_dim, latents.dtype
        )
        skel_tokens = _encode_skeleton(
            _skeleton_module(model, action_cond_mode),
            data["skeleton"][:bsz], None, 0.0, weights_dtype,
        )
    else:
        action_tokens = model.action_encoder(actions_grouped, None)
        skel_tokens = None
    action_tokens = _add_text_bias(action_tokens, text_bias)

    num_train_t = scheduler.config.num_train_timesteps
    # Shift-warped sigma schedule — the same warp FlaxFlowMatchScheduler.set_timesteps
    # applies at inference; official Wan2.2 TI2V-5B ships sample_shift=5.0.
    shift = float(getattr(config, "inference_sigma_shift", 5.0))
    t_uniform = jnp.linspace(1.0, 0.0, num_steps + 1)
    sigmas = scheduler.config.sigma_min + (scheduler.config.sigma_max - scheduler.config.sigma_min) * t_uniform
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    rollout_ts = (sigmas * num_train_t).astype(jnp.int32)

    history = latents[:, :, :n_hist]
    future_gt = latents[:, :, n_hist:]
    gen_init = jax.random.normal(rng, future_gt.shape, dtype=latents.dtype)

    def scan_body(lat, step_idx):
        t_from = rollout_ts[step_idx]
        sig_from = sigmas[step_idx]
        sig_to = sigmas[step_idx + 1]

        roll_input = jnp.concatenate([history, lat], axis=2)
        ts_2d = _build_per_token_timestep(
            jnp.broadcast_to(t_from, (b,)), F_lat, H_lat, W_lat, n_hist
        )
        ts_2d = jax.lax.with_sharding_constraint(ts_2d, P(("data", "fsdp", "context"), None))

        def _velocity(tokens, skel):
            enc_tokens, action_hidden_states = _route_action_conditioning(
                tokens, model.action_adaln_proj, action_cond_mode,
                cond_tokens_per_frame, H_lat, W_lat, text_tokens=text_tokens,
                skeleton_tokens=skel, text_bias=text_bias,
            )
            return model.transformer(
                hidden_states=roll_input,
                timestep=ts_2d,
                encoder_hidden_states=enc_tokens,
                action_hidden_states=action_hidden_states,
                skeleton_hidden_states=_skeleton_bias(action_cond_mode, skel),
                deterministic=True,
                frame_level_cond=_frame_level_cond(action_cond_mode),
                cond_tokens_per_frame=xattn_tokens_per_frame,
                cross_attn_rope=_cross_attn_rope(action_cond_mode),
                frame_positions=frame_positions,
            )

        v_pred = _velocity(action_tokens, skel_tokens)
        if guidance_scale > 1.0:
            # Uncond drops the CONDITION only, keeping the instruction — the same
            # split training uses, where _add_text_bias runs after the dropout.
            # In skeleton mode that means skipping the skeleton token add (what
            # the training-time CFG mask does), leaving the action tokens alone
            # since they are already the zero placeholder. In the action modes it
            # means zeroing the action tokens and re-adding the text bias:
            # zeros_like alone would also blank the pooled text baked into them,
            # guiding on action+text and putting the uncond branch out of
            # distribution.
            if _is_skeleton_mode(action_cond_mode):
                v_uncond = _velocity(action_tokens, None)
            else:
                v_uncond = _velocity(
                    _add_text_bias(jnp.zeros_like(action_tokens), text_bias), None
                )
            v_pred = v_uncond + guidance_scale * (v_pred - v_uncond)

        # Euler step: x_{t_to} = x_{t_from} + (σ_{t_to} - σ_{t_from}) * v
        v_future = v_pred[:, :, n_hist:]
        new_lat = (lat + (sig_to - sig_from) * v_future).astype(lat.dtype)
        return new_lat, None

    final_future, _ = jax.lax.scan(scan_body, gen_init, jnp.arange(num_steps))

    future_mse = jnp.mean(
        (final_future.astype(jnp.float32) - future_gt.astype(jnp.float32)) ** 2
    )
    gen = jnp.concatenate([history, final_future], axis=2)

    def _prep_for_decode(lat):
        # Denormalize with VAE stats, then unstack the 3 cameras from the H
        # axis into the batch axis. Done inside jit: eager ops on multi-host
        # globally-sharded arrays are unsafe.
        lat = lat[:num_samples].astype(jnp.float32) * lat_std + lat_mean
        n, c, f, h3, w = lat.shape
        lat = lat.reshape(n, c, f, 3, h3 // 3, w)
        lat = jnp.transpose(lat, (0, 3, 1, 2, 4, 5))          # (n, cam, C, F, H, W)
        return lat.reshape(n * 3, c, f, h3 // 3, w)

    return _prep_for_decode(gen), _prep_for_decode(latents), future_mse
