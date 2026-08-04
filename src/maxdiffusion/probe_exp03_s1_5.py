"""exp_03 S1.5 — the NO-UPDATE discriminator probe (plan v3.2 §4, as amended by the rulings).

Before any v6e-64 training commitment, S1.5 asks what the three objectives actually *do* to the
gradient, at a state nobody has trained with them yet. It runs **no optimizer updates at all**: the
parameters that go in are bit-identical to the parameters that come out (asserted, not assumed), so
whatever it reports is a property of the objectives rather than of a trajectory they created.

It runs at **both states** the experiment will start from (Tier-2 delta adoption):

* ``checkpoint`` — the exp_02 step-10,000 checkpoint that Tier 1 continues from;
* ``init`` — the pinned pretrained snapshot that Tier 2 starts from, restored through the same code
  with an empty checkpoint directory, so "from init" is the production path and not a special case.

What it measures, per state, over ``K`` fixed batches:

1. **Per-objective diagnostics** — raw and horizon-normalized losses for control/A/B/C, gradient
   norms, max-abs gradients, finite-leaf counts, and the **cosine of each trial's gradient against
   the plain objective's**. Computed by the trainer's own ``exp03_frozen_replay``, extended rather
   than re-implemented, so the probe cannot drift from the thing it is probing.
2. **A's label isolation (review P6)** — on *identical* self-generated states, the corrective label
   against the same-epsilon label. A wins by changing two things at once (exposure and supervision);
   this is the number that says how much of it is the label.
3. **``p_ss=0`` parity** — A with the scheduled-sampling probability at zero must be the plain
   objective, in loss *and* gradient, to floating-point tolerance. If that fails, every A result is
   about a bug rather than an objective.
4. **Support-gradient variance (D2 adoption)** — the trials draw one sigma support per batch, so
   their gradient carries noise the control's does not. The decomposition is deliberately simple and
   predeclared: each batch is replayed under ``M`` independent support draws, the *within-batch*
   variance is the support term, the *between-batch* variance is the data term, and their sum is the
   total. The control's support term is ~0 by construction, which is what makes the trials' number
   readable.
5. **Sigma-trajectory baseline traces (mechanism B)** — ``diagnostics_exp03.sigma_trajectory_trace``
   at both states, so the trials have a before-picture to be compared against later.

**Approval-pinned**, in the manner of the exp_02 sampling probe: ``K``, ``M`` and the two state
labels are constants, and a config that asks for anything else is a startup failure rather than a
quietly different experiment. Output is one immutable JSON per state under the run's canonical
``validation_probe_sampling/`` root, with the same refusal of any ``step_`` path component -- a
diagnostic must never be written where a verdict's evidence lives.

Why no ffmpeg in its launcher: like the one-step loss arm, this probe decodes no MP4 and writes no
video. It scores gradients, not pixels.
"""

from __future__ import annotations

import functools
import gc
import hashlib
import re
import math
import statistics
from types import SimpleNamespace
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import tensorflow as tf
from absl import app
from flax import nnx
from flax.linen import partitioning as nn_partitioning

import maxdiffusion.generate_wan_side_adapter as gen
import maxdiffusion.trainers.wan_ti2v_exp03_trainer as exp03
from maxdiffusion import max_logging, pyconfig
from maxdiffusion.diagnostics_exp03 import sigma_trajectory_trace as trace
from maxdiffusion.models.wan.side_adapter_wan import apply_first_frame_pin, masked_velocity_mse
from maxdiffusion.train_utils import load_next_batch
from maxdiffusion.trainers.wan_ti2v_overfit100_trainer import (
    Overfit100TrainState,
    _denoising_loss,
    adapter_param_count,
)

S1_5_SCHEMA = "exp03_s1_5_discriminator_v1"
S1_5_OUTPUT_DIR = "validation_probe_sampling"  # the same non-role diagnostic root

# The APPROVED experiment. Pinned rather than parameterised, for the reason the exp_02 probe review
# gave: constants in a module do not constrain a run whose config supplies the values.
S1_5_NUM_BATCHES = 8  # K
S1_5_SUPPORT_DRAWS = 4  # M
S1_5_SUPPORT_SALTS = (1, 2, 3, 4)  # the M salts; ONLY the support purposes consume them
S1_5_STATES = ("checkpoint", "init")
S1_5_OBJECTIVES = ("control", "corrective_ss", "rollout_loss", "combined")
S1_5_PARITY_TOLERANCE = 1e-5  # relative, for the p_ss=0 identity


def assert_approved_s1_5_design(*, num_batches: int, support_draws: int, states: Sequence[str]) -> None:
    """Refuse anything but the approved probe (K=8 batches, M=4 support draws, both states)."""
    if int(num_batches) != S1_5_NUM_BATCHES:
        raise ValueError(
            f"S1.5 is approval-pinned to K={S1_5_NUM_BATCHES} batches, got {int(num_batches)}; a different K is a "
            f"different measurement of gradient noise and needs its own approval."
        )
    if int(support_draws) != S1_5_SUPPORT_DRAWS:
        raise ValueError(
            f"S1.5 is approval-pinned to M={S1_5_SUPPORT_DRAWS} support draws per batch, got {int(support_draws)}; "
            f"the support-variance decomposition is defined at that M."
        )
    if tuple(states) != S1_5_STATES:
        raise ValueError(
            f"S1.5 runs at exactly {list(S1_5_STATES)} -- Tier 1 continues from the checkpoint and Tier 2 starts "
            f"from init, so a probe of one of them answers half the question. Got {list(states)}."
        )


def s1_5_output_path(config, *, state_label: str, checkpoint_step: int) -> str:
    """``<output_dir>/<run_name>/validation_probe_sampling/s1_5_<state>_ckpt<step>.json``.

    Canonical only, with the exp_02 probe review's rule enforced: any path component naming a
    ``step_<n>_<role>`` verdict directory is refused outright.
    """
    if state_label not in S1_5_STATES:
        raise ValueError(f"unknown S1.5 state {state_label!r}; the approved states are {list(S1_5_STATES)}")
    root = f"{str(config.output_dir).rstrip('/')}/{config.run_name}/{S1_5_OUTPUT_DIR}"
    path = f"{root.rstrip('/')}/s1_5_{state_label}_ckpt{int(checkpoint_step)}.json"
    offenders = [part for part in path.split("/") if part.startswith("step_")]
    if offenders:
        raise ValueError(
            f"refusing the S1.5 output path {path!r}: component(s) {offenders} name a verdict role directory. A "
            f"diagnostic must never be written where the verdict's evidence lives -- fix output_dir/run_name."
        )
    parts = path.split("/")
    if len(parts) < 2 or parts[-2] != S1_5_OUTPUT_DIR:
        raise ValueError(f"refusing the S1.5 output path {path!r}: it is not directly under {S1_5_OUTPUT_DIR}/")
    return path


# ------------------------------------------------------------------------------- the no-update pin


def params_fingerprint(params) -> list[str]:
    """A sha256 digest per parameter leaf.

    A digest, not a byte SUM: a sum collides on any permutation -- swap two parameter values and it
    does not move -- so it could certify "unchanged" for a state that had in fact been rewritten.
    The digest is over the raw bytes, so it is bit identity, and it is O(1) memory per leaf rather
    than a retained copy of a 5B-parameter tree.
    """
    digests = []
    for leaf in jax.tree_util.tree_leaves(params):
        array = np.asarray(leaf)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
        digests.append(digest.hexdigest())
    return digests


def assert_no_update(before: Sequence[str], after: Sequence[str]) -> None:
    """The probe's defining property: it changed nothing, bit for bit.

    Digest equality, with no tolerance anywhere: a probe that nudged the parameters would report the
    objectives' behaviour at a state that no longer exists, and a tolerance would hide exactly the
    small update an accidental ``apply_gradients`` produces.
    """
    if list(before) != list(after):
        differing = [index for index, (x, y) in enumerate(zip(before, after)) if x != y]
        raise RuntimeError(
            f"S1.5 modified the parameters it was measuring: {len(differing)} leaf/leaves changed "
            f"(first at index {differing[:3]}). This probe applies no updates by construction."
        )


# ------------------------------------------------------------------------ support-variance (D2)


@jax.jit
def _welford_first(tree):
    """The first observation becomes the running mean (in float32)."""
    return jax.tree_util.tree_map(lambda leaf: leaf.astype(jnp.float32), tree)


@functools.partial(jax.jit, donate_argnums=(0,))
def _welford_update(mean, tree, count):
    """One compiled Welford step; the mean buffer is DONATED so it is updated in place.

    Eagerly this was three whole-tree traversals plus a dot, each materializing its own tree -- the
    same shape as the replay's OOM, in the accumulator that runs K x M times.
    """
    delta = jax.tree_util.tree_map(lambda g, m: g.astype(jnp.float32) - m, tree, mean)
    new_mean = jax.tree_util.tree_map(lambda m, d: m + d / count, mean, delta)
    delta2 = jax.tree_util.tree_map(lambda g, m: g.astype(jnp.float32) - m, tree, new_mean)
    increment = sum(
        jnp.sum(x * y) for x, y in zip(jax.tree_util.tree_leaves(delta), jax.tree_util.tree_leaves(delta2))
    )
    return new_mean, increment


def buffer_shards(array) -> tuple[dict, bool]:
    """The PHYSICAL allocations behind one live array: ``{(device, pointer): bytes}``.

    The unit is the allocation, not the array object, because one allocation is reachable through
    several array objects. The review executed an 8-device mesh and found exactly that: a single
    256-byte gradient appears in ``jax.live_arrays()`` as the global array AND as its eight 32-byte
    shard arrays, so anything summing ``array.nbytes`` over live arrays reported 512 bytes -- two
    gradient trees for one gradient, on every sharded run, which is every real one.

    Keyed by ``(device, pointer, bytes)``. The device because two devices' allocators are
    independent address spaces and can hand out the same numeric address for different memory; the
    size because a freed small buffer's address can become the start of a much larger allocation,
    and treating that as "already at baseline" would lose a whole gradient tree. The same physical
    buffer always reports the same triple, so deduplication is unaffected.

    Not ``(shape, dtype)`` and not ``id()``. Shape matching was the gauge's original mechanism and
    it was wrong in both directions, as the review demonstrated by executing it: AdamW's moment
    trees are parameter-shaped, so a probe that merely restored an optimizer read two phantom
    gradient trees at baseline; and a float32 Welford accumulator over bfloat16 parameters matches
    no parameter ``(shape, dtype)`` at all, so the buffer class the memory contract most needs to
    see was invisible. ``id()`` is worse still -- CPython recycles object addresses, and
    ``jax.live_arrays()`` hands back fresh wrapper objects.

    Failure modes, handled rather than assumed away:

    * **Address reuse.** A freed buffer's address can be handed to a new allocation, which then
      reads as "already at baseline". That is the RIGHT answer for this gauge: it measures resident
      bytes, and memory reused in place is not additional memory. Donation (``_welford_update``)
      is the common case and is deliberately not counted as growth.
    * **No pointer available.** A backend that refuses the call falls back to object identity for
      that shard, and the fallback is COUNTED into the report, so a run that silently degraded is
      visible rather than quietly mis-measured.
    * **No addressable shards at all.** In multi-process runs an array can have no local shard. It
      then holds no local memory, but rather than assume that, the array is counted once at its own
      size under a fallback key -- conservative, and flagged.
    * **Deleted buffers.** A donated-away array is still an object; it holds no memory and is
      skipped by the caller.

    Cost, since it is paid per sample point: one pointer read per addressable shard. At 5B that is
    on the order of 10^4 reads per sample against a probe whose compilations take minutes, so the
    exact per-shard key is bought rather than approximated by the first shard's address.
    """
    try:
        shards = list(array.addressable_shards)
    except Exception:  # pragma: no cover - backend-dependent
        shards = []
    allocations: dict = {}
    fell_back = False
    for index, shard in enumerate(shards):
        data = shard.data
        nbytes = int(data.nbytes)
        try:
            allocations[(str(shard.device), int(data.unsafe_buffer_pointer()), nbytes)] = nbytes
        except Exception:  # pragma: no cover - backend-dependent
            fell_back = True
            allocations[(str(shard.device), ("obj", id(array), index), nbytes)] = nbytes
    if not allocations:  # pragma: no cover - multi-process only
        fell_back = True
        allocations[("?", ("obj", id(array)), int(array.nbytes))] = int(array.nbytes)
    return allocations, fell_back


class LiveBufferGauge:
    """New-since-baseline device memory, by PHYSICAL ALLOCATION, from ``jax.live_arrays()``.

    The baseline is snapshotted at construction: every allocation alive before the probe computes
    anything -- parameters, the batches, the context table, and anything an optimizer or a caller
    happened to be holding. Every later sample counts only allocations NOT in that set, so the gauge
    measures what the probe ITSELF made resident and cannot be poisoned by what it inherited.

    Identity and BYTES are the same set: a ``(device, pointer)`` pair contributes its shard's bytes
    exactly once, however many array objects reference it. That is what makes the reading survive
    sharding -- an FSDP-sharded gradient reads 1.0 tree-equivalents on 8 devices exactly as it does
    on 1, instead of 2.0 for the global array plus its shards.

    The primary number is ``new_bytes / one_grad_tree_bytes`` -- GRADIENT-TREE EQUIVALENTS -- which
    is the probe's actual contract (at most three trees) and is scale-free. The denominator is the
    parameter tree measured THE SAME WAY (its own physical allocations), so replication does not
    change the ratio either. It is also dtype-honest, which the count-based version was not: a
    float32 accumulator over bfloat16 parameters is two tree-equivalents, because that is what it
    costs. Parameter-SHAPED new arrays are still counted, bucketed by dtype, as secondary evidence
    of what the bytes are made of; each allocation is credited to the LARGEST array that covers it,
    so a shard view of a bigger array is not counted a second time there either.

    Only identity KEYS are retained, never the arrays: a gauge that held references would keep the
    buffers it is measuring alive, which is the failure it exists to detect.

    ``identity_fallbacks`` is CUMULATIVE over every array inspected at every sample: it is a
    "did this ever degrade to object identity" alarm, not a count of distinct buffers.
    """

    def __init__(self, params):
        leaves = jax.tree_util.tree_leaves(params)
        self.param_shapes = {tuple(leaf.shape) for leaf in leaves}
        self.num_param_leaves = max(1, len(leaves))
        self.identity_fallbacks = 0
        # ONE gradient tree in bytes = the parameter tree's own PHYSICAL bytes, deduplicated the
        # same way the numerator is. A gradient has the parameters' shapes and sharding; if it
        # carries a wider dtype the ratio says so, which is the point.
        param_allocations, _ = self._allocations(leaves)
        self.one_tree_bytes = max(1, sum(param_allocations.values()))
        self.baseline_ids = set(self._allocations(jax.live_arrays())[0])
        self.baseline_buffers = len(self.baseline_ids)
        self.peak_new_bytes = 0
        self.peak_tree_equivalents = 0.0
        self.peak_new_param_shaped = 0
        self.samples: list[dict] = []

    def _allocations(self, arrays) -> tuple[dict, dict]:
        """``({(device, pointer): bytes}, {(device, pointer): owner})`` over live arrays.

        The owner is the array covering the most allocations -- the global array rather than one of
        its shards -- so the secondary per-dtype count credits each buffer to one array.
        """
        allocations: dict = {}
        owners: dict = {}
        for array in arrays:
            if getattr(array, "is_deleted", lambda: False)():
                continue
            shards, fell_back = buffer_shards(array)
            self.identity_fallbacks += int(fell_back)
            owner = (len(shards), id(array), tuple(array.shape), str(array.dtype))
            for pair, nbytes in shards.items():
                allocations[pair] = nbytes
                if pair not in owners or owner[0] > owners[pair][0]:
                    owners[pair] = owner
        return allocations, owners

    def sample(self, where: str) -> dict:
        allocations, owners = self._allocations(jax.live_arrays())
        new = {pair: nbytes for pair, nbytes in allocations.items() if pair not in self.baseline_ids}
        new_bytes = sum(new.values())
        equivalents = new_bytes / self.one_tree_bytes
        by_owner: dict = {}
        for pair in new:
            _, owner_id, shape, dtype = owners[pair]
            if shape in self.param_shapes:
                by_owner[owner_id] = dtype
        by_dtype: dict[str, int] = {}
        for dtype in by_owner.values():
            by_dtype[dtype] = by_dtype.get(dtype, 0) + 1
        param_shaped = len(by_owner)
        self.peak_new_bytes = max(self.peak_new_bytes, new_bytes)
        self.peak_tree_equivalents = max(self.peak_tree_equivalents, equivalents)
        self.peak_new_param_shaped = max(self.peak_new_param_shaped, param_shaped)
        reading = {
            "where": where,
            "new_buffers": float(len(new)),
            "new_bytes": float(new_bytes),
            "grad_tree_equivalents": float(equivalents),
            "new_param_shaped": float(param_shaped),
            "new_param_shaped_by_dtype": {name: float(count) for name, count in sorted(by_dtype.items())},
        }
        self.samples.append(reading)
        return reading

    def report(self) -> dict:
        return {
            "grad_tree_equivalents_peak": float(self.peak_tree_equivalents),
            "new_bytes_peak": float(self.peak_new_bytes),
            "one_grad_tree_bytes": float(self.one_tree_bytes),
            "new_param_shaped_peak": float(self.peak_new_param_shaped),
            "param_leaves": float(self.num_param_leaves),
            "gauge_baseline_buffers": float(self.baseline_buffers),
            "gauge_identity_fallbacks": float(self.identity_fallbacks),
            "gauge_samples": float(len(self.samples)),
            "gauge_sample_points": [reading["where"] for reading in self.samples],
            "gauge_baseline": self.samples[0] if self.samples else {},
            "gauge_readings": list(self.samples),
        }


class _TreeWelford:
    """Streaming mean/variance over gradient TREES, retaining O(1) trees.

    Welford's algorithm applied leafwise on device: the running mean is one tree, and the sum of
    squared deviations is a scalar accumulated with ``tree_dot``. At 5B parameters the alternative --
    retaining K x M gradient trees and flattening each to host float64 -- is about 1.28 TB, i.e. the
    difference between a probe that runs and one that kills the host.
    """

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = 0.0

    def update(self, tree) -> None:
        """One compiled Welford step. TRANSACTIONAL: state advances only if the call returns.

        The earlier version cleared ``self.mean`` into a local first, on the theory that a Python
        alias suppresses JAX donation. That theory was WRONG -- donation is declared at the jit
        boundary, and a surviving reference produces a use-after-donate error on later access, not a
        silent decline. The dance is reverted. It also left a real bug: ``count`` was incremented
        before the call, so a raised exception left the accumulator claiming an observation it never
        absorbed, with ``mean`` set to ``None``. Count advances last now.

        Honest footprint: while this runs, the accumulator holds ONE fp32 mean tree, and the caller
        holds the gradient being absorbed. Two trees is the floor for a streaming mean, and the
        budget counts it as such.
        """
        if self.mean is None:
            self.mean = _welford_first(tree)
            self.count += 1
            return
        next_count = self.count + 1
        self.mean, increment = _welford_update(self.mean, tree, jnp.asarray(next_count, jnp.float32))
        self.m2 += float(increment)
        self.count = next_count

    def population_variance(self) -> float:
        return self.m2 / self.count if self.count else 0.0

    def sample_variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0


def variance_decomposition(gradients) -> dict:
    """Split gradient variance into its SUPPORT and BATCH+SHARED-RNG parts.

    ``gradients[k][m]`` is the gradient on batch ``k`` under support draw ``m`` -- where only the
    support SALT differs across ``m``, so the batch, the shared-stream epsilon and dropout key, the
    logical global step, ``p_ss`` and A's branch coin are all held fixed. The within-batch spread is
    therefore attributable to the sigma support alone.

    The between-batch term is named **batch+shared-RNG variance**, not "data variance": at finite
    ``M`` each batch mean still carries support Monte-Carlo error, and the batches differ in their
    shared-stream draws as well as in their examples. Both the population and the unbiased (``M-1``,
    ``K-1``) estimates are reported; the unbiased pair is the honest one at ``M=4``, and the finite-M
    inflation of the between term is stated rather than corrected away.
    """
    between = _TreeWelford()
    within_population, within_sample, draws = [], [], set()
    # STREAMING: ``gradients`` may be a generator of generators, and each gradient is consumed and
    # released as it arrives. Nothing accumulates but one mean tree per accumulator, so the peak
    # retained tree count is O(1) rather than K x M (32 trees, ~1.28 TB at 5B).
    for row in gradients:
        accumulator = _TreeWelford()
        for grad in row:
            accumulator.update(grad)
            del grad
        if accumulator.count == 0:
            raise ValueError("variance_decomposition needs at least one draw per batch")
        draws.add(accumulator.count)
        within_population.append(accumulator.population_variance())
        within_sample.append(accumulator.sample_variance())
        between.update(accumulator.mean)
        del accumulator
    if not within_population:
        raise ValueError("variance_decomposition needs at least one batch with at least one draw")
    if len(draws) != 1:
        raise ValueError(f"every batch must contribute the same number of support draws; got {sorted(draws)}")

    rows_count = len(within_population)
    support = float(np.mean(within_population))
    support_unbiased = float(np.mean(within_sample))
    batch_variance = between.population_variance()
    batch_variance_unbiased = between.sample_variance()
    total = support + batch_variance
    mean_sq = exp03.tree_sq_norm(between.mean)
    num_draws = draws.pop()
    return {
        "num_batches": rows_count,
        "support_draws": num_draws,
        "support_variance": support,
        "support_variance_unbiased": support_unbiased,
        "batch_shared_rng_variance": batch_variance,
        "batch_shared_rng_variance_unbiased": batch_variance_unbiased,
        "total_variance": total,
        "support_fraction": (support / total) if total > 0 else 0.0,
        "mean_grad_sq_norm": mean_sq,
        "gradient_noise_scale": (total / mean_sq) if mean_sq > 0 else float("nan"),
        "finite_m_note": (
            f"the between term still contains support Monte-Carlo error at M={num_draws} "
            f"(order support_variance/M); it is reported as measured, not corrected"
        ),
    }


# ---------------------------------------------------------------------------- A's label isolation


@jax.jit
def _jit_relative_gap(left, right):
    """``||left - right||`` and ``||right||`` in ONE compiled pass (no eager difference tree)."""
    difference_sq = sum(
        jnp.sum((a.astype(jnp.float32) - b.astype(jnp.float32)) ** 2)
        for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right))
    )
    right_sq = sum(jnp.sum(b.astype(jnp.float32) ** 2) for b in jax.tree_util.tree_leaves(right))
    return jnp.sqrt(difference_sq), jnp.sqrt(right_sq)


def _relative_gradient_gap(left, right) -> float:
    """``||left - right|| / ||right||`` — compiled, so no whole difference tree is ever materialized."""
    difference, magnitude = _jit_relative_gap(left, right)
    return float(difference) / max(float(magnitude), 1e-12)


def same_eps_label(z_lo, z_gt, eps):
    """The plain objective's label ``eps - z_gt`` — what A would supervise WITHOUT the correction."""
    del z_lo
    return eps - z_gt


def corrective_label(z_lo, z_gt, sigma_lo):
    """A's label ``(z_lo - z_gt) / sigma_lo`` — exact under the Euler rule, on or off path."""
    return (z_lo - z_gt) / sigma_lo


def label_isolation(*, corrective_loss, same_eps_loss, corrective_grad, same_eps_grad) -> dict:
    """The isolated effect of A's LABEL, given both losses/gradients on identical states.

    A changes exposure and supervision together, so a win is ambiguous until this number exists:
    same states, same draws, only the label differs.
    """
    return {
        "loss_corrective": float(corrective_loss),
        "loss_same_eps": float(same_eps_loss),
        "loss_delta": float(corrective_loss) - float(same_eps_loss),
        "grad_norm_corrective": float(np.sqrt(exp03.tree_sq_norm(corrective_grad))),
        "grad_norm_same_eps": float(np.sqrt(exp03.tree_sq_norm(same_eps_grad))),
        "grad_cosine": exp03.grad_cosine(corrective_grad, same_eps_grad),
        "grad_relative_delta": _relative_gradient_gap(corrective_grad, same_eps_grad),
    }


# ------------------------------------------------------------------------------- p_ss = 0 parity


def parity_report(*, trial_loss, plain_loss, trial_grad, plain_grad, tolerance: float = S1_5_PARITY_TOLERANCE) -> dict:
    """A at ``p_ss=0`` against the plain objective — loss AND gradient, relative tolerance."""
    loss_gap = abs(float(trial_loss) - float(plain_loss)) / max(abs(float(plain_loss)), 1e-12)
    grad_gap = _relative_gradient_gap(trial_grad, plain_grad)
    return {
        "loss_relative_gap": loss_gap,
        "grad_relative_gap": grad_gap,
        "tolerance": float(tolerance),
        "passes": bool(loss_gap <= tolerance and grad_gap <= tolerance),
    }


# -------------------------------------------------------------------------------------- driver


# The canonical per-state mapping (review V3/V5): the checkpoint state is Tier 1 continuing at
# global steps 10000.., with the ramp origin there; the init state is Tier 2 at 0.., origin 0. The
# probe applies no updates, but the ramp still decides A/C's branch, so the mapping changes what is
# measured and must be per-state rather than one launcher env.
S1_5_STATE_PLAN = {
    "checkpoint": {"first_global_step": 10000, "ramp_origin": 10000, "required_checkpoint_step": 10000},
    "init": {"first_global_step": 0, "ramp_origin": 0, "required_checkpoint_step": 0},
}


# The key that must be present in any real run config. It is the sentinel because it is the one the
# hardware crashed on: ``_denoising_loss`` reads ``config.weights_dtype`` as a hard attribute, so a
# view missing it fails deep inside a 5B replay rather than at construction.
CONFIG_SENTINEL_KEY = "weights_dtype"


def _config_key_dict(config) -> dict:
    """Every key of ``config``, whatever kind of config it is — and never silently empty.

    ``vars()`` is WRONG for a production config. ``pyconfig.config`` is the ``HyperParameters``
    proxy: its keys live in ``_config.keys`` behind ``__getattr__`` and its instance ``__dict__`` is
    empty, so ``vars(config)`` returns ``{}`` and a view built from it carries only its overrides.
    Job 8b died exactly that way -- every ``getattr(view, k, default)`` in the replay path quietly
    returned its default, and the first hard read (``config.weights_dtype``) raised, 5B restore and
    dataset pin already behind it.

    So: ``get_keys()`` when the config offers it (the proxy does), ``vars()`` otherwise (test and
    view-of-view namespaces, where it is correct and lossless). Either way the result is checked --
    an empty dict, or one without the sentinel, is a construction-time failure with a message that
    says which shape of config it was handed.
    """
    getter = getattr(type(config), "get_keys", None)
    keys = dict(config.get_keys()) if callable(getter) else dict(vars(config))
    if not keys:
        raise ValueError(
            f"refusing to build a config view from {type(config).__name__}: it exposes no keys. A pyconfig proxy "
            f"keeps its keys behind __getattr__ (vars() is empty), so a view built from vars() would carry only "
            f"its overrides and every default-valued read downstream would be a lie."
        )
    if CONFIG_SENTINEL_KEY not in keys:
        raise ValueError(
            f"refusing to build a config view without {CONFIG_SENTINEL_KEY!r}: got {len(keys)} keys "
            f"({sorted(keys)[:5]}...). That key is read as a hard attribute inside the replay, so its absence "
            f"surfaces as an AttributeError deep in a 5B forward instead of here."
        )
    return keys


def state_view(config, state_label: str, **overrides):
    """A read-only config view carrying THIS state's ramp origin (and any forced knobs).

    It also carries the state LABEL, which is not a knob: the replay's compile-time log keys its
    first-call timings by it, so the init state's compile is reported as its own number instead of
    being merged into the checkpoint state's entry.
    """
    plan = S1_5_STATE_PLAN[state_label]
    return SimpleNamespace(
        **{
            **_config_key_dict(config),
            "exp03_ramp_origin": plan["ramp_origin"],
            "exp03_state_label": state_label,
            **overrides,
        }
    )


_PROBE_GRAD_CACHE: dict = {}
PROBE_COMPILATIONS: list = []  # every distinct compiled function, for the executable count test


def variance_tag(objective: str, salt) -> str:
    """The tag of ONE support-variance compilation — objective AND salt.

    One tag per compilation, because the tag is what the compile-cost ledger is keyed by: the four
    salts are four separate specializations, and a per-objective tag timed the first of them and
    silently attributed nothing to the rest.
    """
    return f"variance_{objective}_salt{int(salt)}"


def _jitted_grad(tag: str, builder, *, static_key=()):
    """A cached, JITTED value-and-gradient whose per-state knob is a TRACED argument.

    Two things this shape fixes. First the memory one (Job 8c): an unjitted backward through a 5B
    forward runs op by op, so nothing is scheduled or freed.

    Second, and worse because it was silent: the previous version cached by ``tag`` alone while its
    closures captured state-specific config views. The init state would have reused the checkpoint
    state's compiled functions and run with ``ramp_origin=10000`` instead of 0 -- wrong branch, wrong
    p_ss, wrong science, no crash. The fix is not a wider key: the varying quantity is now an
    explicit traced argument (``ramp_origin``), so ONE compilation legitimately serves both states
    and the collision cannot exist. ``static_key`` carries only the knobs that genuinely change the
    traced program (forced ``p_ss`` constants, salts, objective), each of which is identical across
    states by construction.

    ``builder(ramp_origin)`` returns the loss callable for that origin; it is invoked at trace time
    with the tracer, so the config view it builds carries the traced origin.
    """
    key = (tag, tuple(static_key))
    if key not in _PROBE_GRAD_CACHE:

        def _call(params, state, data, rng, global_step, ramp_origin, make=builder):
            return make(ramp_origin)(params, state, data, rng, global_step)

        compiled = jax.jit(jax.value_and_grad(_call, has_aux=True))
        _PROBE_GRAD_CACHE[key] = compiled
        PROBE_COMPILATIONS.append(key)
    return _PROBE_GRAD_CACHE[key]


def traced_view(config, base_label: str, ramp_origin, **overrides):
    """A config view whose ramp origin is whatever was threaded in — tracer or int."""
    return SimpleNamespace(
        **{
            **_config_key_dict(config),
            "exp03_ramp_origin": ramp_origin,
            "exp03_state_label": base_label,
            **overrides,
        }
    )


# The jitted helpers that are not gradient functions. Named here rather than in the test, so the
# census the artifact carries and the count the test asserts come from ONE list.
def probe_jit_helpers() -> dict:
    return {
        "grad_stats": exp03._jit_grad_stats,
        "tree_vdot": exp03._jit_tree_vdot,
        "welford_update": _welford_update,
        "welford_first": _welford_first,
        "relative_gap": _jit_relative_gap,
    }


def jit_cache_size(fn) -> int:
    """A jitted wrapper's REAL specialization count — and a refusal if it is not jitted at all.

    ``._cache_size()`` is XLA's own ledger. Counting wrapper CONSTRUCTIONS instead (which the probe
    also keeps, in ``PROBE_COMPILATIONS``) stays right even when the thing underneath has silently
    become eager, which is the regression that costs a 5B probe its memory budget.
    """
    target = getattr(fn, "jitted", fn)
    sizer = getattr(target, "_cache_size", None)
    if not callable(sizer):
        raise TypeError(f"{fn!r} exposes no _cache_size(): it is not a jitted function (an eager fallback looks so)")
    return int(sizer())


def specialization_census() -> dict:
    """Every jitted wrapper in the probe's world, with its specialization count and their TOTAL.

    Three families: the probe's cached gradient functions, the replay's cached
    ``(value_and_grad, value)`` pair per objective per state view, and the jitted helpers. Keys are
    derived from tags and insertion order, never from ``id()``, so the census is identical across
    runs and the artifact stays byte-reproducible.

    Two specializations for one probe tag is the cache-collision regression coming back; a helper
    with zero is a helper that never ran.
    """
    probe = {}
    for (tag, static_key), fn in _PROBE_GRAD_CACHE.items():
        label = tag if not static_key else f"{tag}({','.join(str(part) for part in static_key)})"
        probe[label] = jit_cache_size(fn)
    replay: dict = {}
    seen: dict = {}
    for (name, _, _), (grad_fn, value_fn) in exp03._LOSS_AND_GRAD_CACHE.items():
        seen[name] = seen.get(name, 0) + 1
        replay[f"{name}#{seen[name]}"] = jit_cache_size(grad_fn) + jit_cache_size(value_fn)
    helpers = {name: jit_cache_size(fn) for name, fn in probe_jit_helpers().items()}
    return {
        "probe": probe,
        "replay": replay,
        "helpers": helpers,
        "probe_total": sum(probe.values()),
        "replay_total": sum(replay.values()),
        "helper_total": sum(helpers.values()),
        "total": sum(probe.values()) + sum(replay.values()) + sum(helpers.values()),
    }


def assert_one_specialization_per_probe_tag(census: dict) -> None:
    """Every probe tag compiles ONCE, serving both states — the collision, asserted at runtime.

    Called from the driver rather than from ``state_report``: ``_PROBE_GRAD_CACHE`` is module-level,
    so in a process that runs the report more than once (a test file) a second set of shapes adds a
    legitimate specialization. A production job runs the driver once, which is where the number
    means what it says.
    """
    offenders = {tag: size for tag, size in census["probe"].items() if size != 1}
    if offenders:
        raise RuntimeError(
            f"probe gradient tags with more than one specialization: {offenders}. One compilation must serve BOTH "
            f"states -- the per-state ramp origin is a traced argument precisely so that it can. More than one "
            f"means a state-specific value was captured in a closure again."
        )


def _ramp_aux(view, global_step) -> dict:
    """The ramp quantities read off the VERY VIEW the loss is being evaluated on.

    ``view`` is the config view the cache-served function built from its traced origin and handed
    to the loss; both numbers therefore come from the same object the measurement came from, not
    from a recomputation standing beside it.
    """
    origin = getattr(view, "exp03_ramp_origin", 0)
    return {
        "p_ss_used": exp03.exp03_p_ss(view, global_step),
        "ramp_elapsed": jnp.asarray(global_step, jnp.float32) - jnp.asarray(origin, jnp.float32),
    }


def _with_ramp_aux(view, loss_fn):
    """Make a cache-served loss EMIT the ramp quantities it computed with, as aux.

    THE fix for a pin that could not bite. The evidence was previously a parallel jitted
    recomputation of ``p_ss`` sitting next to the cache: the review's surgical mutation -- force the
    origin to 10000 inside cache-served calls only -- changed what the gradients were computed with
    and left the recomputation untouched, so the regression test passed while the probe was
    measuring the wrong state. Emitting from INSIDE the cached function removes that gap: the
    number in the report is the number the compiled program used, and there is nowhere for a
    corrupted origin to hide.
    """

    def _fn(params, state, data, rng, global_step):
        return loss_fn(params, state, data, rng, global_step), _ramp_aux(view, global_step)

    return _fn


class RampWitness:
    """Every cache-served function's own account of the ramp it ran with.

    One row per ``(tag, global_step)``: the M support draws re-enter the same compiled function at
    the same step, and recording each of them would pad the artifact without adding evidence.
    """

    def __init__(self):
        self.rows: dict = {}

    def record(self, tag: str, global_step, aux) -> dict:
        key = (str(tag), int(global_step))
        if key not in self.rows:
            self.rows[key] = {
                "tag": str(tag),
                "global_step": float(int(global_step)),
                "p_ss_used": float(aux["p_ss_used"]),
                "ramp_elapsed": float(aux["ramp_elapsed"]),
            }
        return self.rows[key]

    def at(self, tag: str, global_step) -> dict:
        return self.rows[(str(tag), int(global_step))]

    def report(self) -> list:
        return [self.rows[key] for key in sorted(self.rows)]


def _objective_config(config, objective: str, **overrides):
    """A read-only config view selecting one objective.

    NOT the exp_02 probe's pattern -- that claim was false and is removed. exp_02's ``arm_config``
    is a read-only PROXY that forwards ``__getattr__`` to the base config, which is why it never met
    this failure; these views are flat namespaces, which is what makes a view-of-view work, and the
    price of that is having to enumerate the base's keys correctly.
    """
    return SimpleNamespace(**{**_config_key_dict(config), "exp03_objective": objective, **overrides})


def plain_fixed_support_loss(params, state, data, rng, config, scheduler, *, global_step):
    """The plain objective at A's SINGLE end support — the conditional-parity comparator.

    Production A at ``p_ss=0`` is not the production control: A scores one scalar sigma per batch
    while the control samples a timestep per example from the shared stream, so their gradients are
    not the same quantity and demanding equality would be demanding the wrong thing. This comparator
    holds the sigma, epsilon, dropout key, batch and state identical to A's and differs only in the
    label, which is what makes the comparison an identity rather than a coincidence.
    """
    ctx = exp03._exp03_prologue(params, state, data, rng, config, scheduler)
    _, end, _ = exp03.corrective_support(
        seed=int(getattr(config, "seed", 0)),
        global_step=global_step,
        num_steps=ctx.num_steps,
        k_a_max=int(getattr(config, "exp03_k_a", 2)),
        support_salt=int(getattr(config, "exp03_support_salt", 0)),
    )
    z_lo = exp03._interpolant_at(ctx, end)
    v_pred = exp03._forward_velocity(ctx, z_lo, end)
    v_target = ctx.eps - ctx.z_video  # the PLAIN label, at A's support
    return masked_velocity_mse(v_pred, v_target, ctx.b), {"sigma_lo": ctx.sigmas[end]}


def corrective_at_support(params, state, data, rng, config, scheduler, *, global_step, same_eps_label_flag):
    """A's supervised forward on the SELF-GENERATED state, under either label.

    ``p_ss`` is forced to 1 by the caller, so the state scored here is the off-path one -- which is
    the only place the two labels differ, and therefore the only place the isolation is meaningful.
    """
    ctx = exp03._exp03_prologue(params, state, data, rng, config, scheduler)
    start, end, k_a = exp03.corrective_support(
        seed=int(getattr(config, "seed", 0)),
        global_step=global_step,
        num_steps=ctx.num_steps,
        k_a_max=int(getattr(config, "exp03_k_a", 2)),
        support_salt=int(getattr(config, "exp03_support_salt", 0)),
    )
    sigma_lo = ctx.sigmas[end].astype(jnp.float32)
    advanced = jax.lax.stop_gradient(
        exp03._advance_with_sampler(
            ctx,
            jax.lax.stop_gradient(exp03._interpolant_at(ctx, start)).astype(ctx.weights_dtype),
            start,
            k_a,
            velocity_fn=exp03._sampling_velocity_fn(ctx),
            k_max=int(getattr(config, "exp03_k_a", 2)),
        )
    ).astype(jnp.float32)
    z_lo = apply_first_frame_pin(advanced, ctx.z_i0)
    v_pred = exp03._forward_velocity(ctx, z_lo, end)
    v_target = (
        same_eps_label(z_lo, ctx.z_video, ctx.eps)
        if same_eps_label_flag
        else corrective_label(z_lo, ctx.z_video, sigma_lo)
    )
    return masked_velocity_mse(v_pred, v_target, ctx.b), {"sigma_lo": sigma_lo}


def build_probe_state(config, trainer, pipeline, mesh, *, state_label: str):
    """The training state, built as ``start_training`` builds it, restored through the SAME path.

    Both states go through the checkpoint manager: the checkpoint state must come back at the
    REQUIRED step (``latest_step()`` is not a pin -- a stray later checkpoint would silently become
    the probe's subject), and the init state goes through the empty-restore path production uses,
    so "from init" is the production behaviour rather than a special case.
    """
    plan = S1_5_STATE_PLAN[state_label]
    context_table = trainer._build_context_table(pipeline, mesh)
    # The encode-time models die with the line above: the table is built, and S1.5 decodes no pixels
    # and encodes no text ever again.
    free_encode_time_models(pipeline)
    log_memory("post_encode_time_free", state_label=state_label)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        graphdef, params, rest = nnx.split(pipeline.transformer, nnx.Param, ...)
    # ...and the pipeline's own handle on the transformer dies HERE, with the split: the state owns
    # the parameters from this line on, and after the restore below the pipeline's copy would be a
    # second ~5B tree that nothing reads.
    free_pipeline_transformer(pipeline)
    tx, _ = trainer._build_optimizer(config.max_train_steps)
    state = Overfit100TrainState.create(
        apply_fn=graphdef.apply,
        params=params,
        tx=tx,
        graphdef=graphdef,
        rest_of_state=rest,
        context_table=context_table,
    )
    if jax.process_index() == 0:
        max_logging.log(f"[exp03_s1_5] trainable params: {adapter_param_count(params) / 1e9:.2f}B")
    state, state_shardings = trainer._shard_state(mesh, state)
    log_memory("post_state_creation", state_label=state_label)

    directory = config.checkpoint_dir if state_label == "checkpoint" else _empty_checkpoint_dir(config)
    manager = trainer._build_checkpoint_manager(directory)
    required = plan["required_checkpoint_step"]
    state, start_step = restore_exact_step(manager, state, required=required)
    if int(start_step) != int(required):
        raise ValueError(
            f"S1.5 state {state_label!r} restored step {int(start_step)}, but this state is pinned to "
            f"{int(required)}. Tier 1 is about the step-10,000 checkpoint and Tier 2 about the untouched init; "
            f"a state that is neither would answer a question nobody asked."
        )
    log_memory("post_restore", state_label=state_label)
    # AFTER the exact-step restore and its check, never before: the restore needs the optimizer tree
    # as its target structure, and the step contract is what makes this the right state at all.
    state = drop_optimizer_moments(state)
    release()  # the moment buffers are unreachable now; collect BEFORE anything measures memory
    # Whether the moments were REALLY reclaimed on the device is not something a toy test can show;
    # this line against the one above it is the on-hardware answer.
    log_memory("post_moment_drop", state_label=state_label)
    return state, state_shardings, int(start_step)


# ------------------------------------------------------- the dead weight, and the on-hardware ledger

# Everything the pipeline loads that a LATENT-SPACE probe never touches once the context table
# exists. ``vae_cache`` is dropped BEFORE ``vae`` deliberately: ``AutoencoderKLWanCache.__init__``
# keeps its own strong reference (``self.module = module``), so dropping ``vae`` alone frees nothing.
PIPELINE_ENCODE_TIME_ATTRS = (
    "vae_cache",
    "vae",
    "text_encoder",
    "tokenizer",
    "image_encoder",
    "image_processor",
    "video_processor",
)
MEM_LEDGER_TAG = "[exp03][mem]"
HEADROOM_BUDGET_BYTES = 8 * 1024**3  # per chip, before the first 5B replay


def free_encode_time_models(pipeline) -> list:
    """Drop the encode/decode-time models and COLLECT.

    S1.5 reads cached latents and scores gradients: after ``_build_context_table`` it encodes no
    text and it decodes no pixels anywhere, so the VAE and the text encoder are dead weight from
    that line on. The training loop drops the same set; the probe additionally has to collect,
    because these are Python objects in reference cycles and a ``delattr`` alone only makes them
    unreachable, not freed.

    What must SURVIVE: ``scheduler``/``scheduler_state`` (every objective uses them), ``mesh``,
    ``devices_array`` and ``config`` -- metadata the whole probe runs on.
    """
    dropped = []
    for attr in PIPELINE_ENCODE_TIME_ATTRS:
        if getattr(pipeline, attr, None) is not None:
            delattr(pipeline, attr)
            dropped.append(attr)
    release()
    if dropped and jax.process_index() == 0:
        max_logging.log(f"[exp03_s1_5] freed encode-time models: {dropped}")
    return dropped


def free_pipeline_transformer(pipeline) -> bool:
    """Drop the pipeline's own handle on the ~5B transformer, once the state owns the parameters.

    ``nnx.split`` does not copy: immediately after it, the pipeline's module and the state's params
    are the same buffers, so this frees nothing yet. It matters at the RESTORE -- Orbax returns
    brand-new arrays and the pipeline is left holding the pre-restore tree, a second ~5B copy that
    nothing will ever read. The training loop survives that by accident (its train step declares
    ``donate_argnums`` and the buffers are donated away on step 0); a probe that applies no updates
    donates nothing and must drop them itself.
    """
    if getattr(pipeline, "transformer", None) is None:
        return False
    delattr(pipeline, "transformer")
    release()
    return True


def memory_snapshot() -> dict:
    """``bytes_in_use`` / ``peak_bytes_in_use`` / ``bytes_limit`` for this process's first device.

    Empty on any backend that reports nothing -- CPU returns ``None`` -- so every caller treats the
    result as optional rather than assuming the keys are there.
    """
    try:
        devices = jax.local_devices()
        stats = devices[0].memory_stats() if devices else None
    except Exception:  # pragma: no cover - backend-dependent
        stats = None
    if not stats:
        return {}
    return {key: int(stats[key]) for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit") if key in stats}


def log_memory(where: str, *, state_label: str = "") -> dict:
    """ONE stdout line at a named point — the ledger a job that dies leaves behind.

    Job 8d died loading a compiled program (E0101, 15.11G of scratch against 9.50G free) before any
    artifact was written, so everything the probe knew about its own residency died with it. These
    lines survive that: they go to the log, never into the artifact, for the same reason compile
    timings do not -- the artifact is compare-or-refuse immutable and these numbers are machine
    state, not measurements of the objectives.
    """
    stats = memory_snapshot()
    prefix = f"{MEM_LEDGER_TAG} where={where}" + (f" state={state_label}" if state_label else "")
    if not stats:
        max_logging.log(f"{prefix} (no device memory stats on this backend)")
        return stats
    gib = 1024**3
    max_logging.log(
        f"{prefix} in_use={stats.get('bytes_in_use', 0) / gib:.2f}G "
        f"peak={stats.get('peak_bytes_in_use', 0) / gib:.2f}G "
        f"limit={stats.get('bytes_limit', 0) / gib:.2f}G"
    )
    return stats


def largest_live_arrays(limit: int = 10) -> list:
    """The biggest live arrays by LOCAL bytes — what is actually holding this chip's memory.

    Local, not global: ``bytes_in_use`` is per device, so a sharded 5B tree must be compared at its
    per-chip size or the two numbers cannot be reconciled. Reuses the residency gauge's physical
    allocation view, so the two instruments cannot disagree about what a byte is.
    """
    rows = []
    for array in jax.live_arrays():
        if getattr(array, "is_deleted", lambda: False)():
            continue
        allocations, _ = buffer_shards(array)
        rows.append(
            {
                "local_bytes": int(sum(allocations.values())),
                "global_bytes": int(array.nbytes),
                "shape": tuple(array.shape),
                "dtype": str(array.dtype),
            }
        )
    rows.sort(key=lambda row: row["local_bytes"], reverse=True)
    return rows[:limit]


def log_largest_live_arrays(reason: str, *, limit: int = 10) -> list:
    """Name the residue, in the log, at the moment it matters."""
    rows = largest_live_arrays(limit)
    gib = 1024**3
    max_logging.log(f"{MEM_LEDGER_TAG} {reason}: the {len(rows)} largest live arrays on this chip are")
    for index, row in enumerate(rows):
        max_logging.log(
            f"{MEM_LEDGER_TAG}   #{index + 1} local={row['local_bytes'] / gib:.3f}G "
            f"global={row['global_bytes'] / gib:.3f}G {row['dtype']}{list(row['shape'])}"
        )
    return rows


def warn_if_no_headroom(where: str, *, state_label: str = "", budget_bytes: int = HEADROOM_BUDGET_BYTES) -> bool:
    """SOFT check before the heavy phase: name the residue rather than die on an opaque E0101.

    Deliberately not a refusal. The budget is generous and approximate (params plus a couple of
    gradients), and a probe that stops itself on a heuristic would be worse than one that runs and
    says why it failed. What it buys is that the next allocation failure arrives with the ten
    largest live arrays printed next to it.
    """
    stats = log_memory(where, state_label=state_label)
    in_use = int(stats.get("bytes_in_use", 0))
    if not stats or in_use <= int(budget_bytes):
        return False
    gib = 1024**3
    max_logging.log(
        f"{MEM_LEDGER_TAG} WARNING {in_use / gib:.2f}G already in use at {where} "
        f"(budget {int(budget_bytes) / gib:.2f}G) -- a 5B replay and its program scratch need the rest"
    )
    log_largest_live_arrays("WARNING residue")
    return True


EMPTY_OPT_STATE = ()  # a valid, leafless pytree: the state stays a TrainState, minus the moments


def drop_optimizer_moments(state):
    """Release the AdamW moments a no-update probe has no use for.

    They cost ~40 GB of HBM at 5B (two parameter-shaped float32 trees) in a probe that applies no
    updates by construction -- and they are not merely dead weight. The review executed the older
    shape-matching residency gauge against a restored state and read TWO gradient trees at baseline
    from these buffers alone, because moment trees are parameter-shaped. The identity-based gauge no
    longer confuses them for gradients, but a probe that carries 40 GB it will never read is still
    40 GB closer to the OOM that produced this whole line of work.

    The state stays a ``TrainState`` (nothing downstream constructs a new one) and keeps ``tx``; only
    the moments go. Correctness rests on the fact that the probe never calls ``apply_gradients`` --
    which is asserted independently, bit for bit, by ``assert_no_update``.
    """
    if not jax.tree_util.tree_leaves(state.opt_state):
        return state
    return state.replace(opt_state=EMPTY_OPT_STATE)


def restore_exact_step(manager, state, *, required: int):
    """Restore the REQUIRED step, selecting it — not "latest, then complain".

    The exp_02 run directory really does hold later checkpoints (12500 among them), so a
    ``latest_step()`` restore would silently hand S1.5 the wrong state and the required-step check
    would only tell us afterwards. ``required == 0`` means "no checkpoint": the init state, which
    must go through the same empty-restore path production uses.
    """
    steps = sorted(int(step) for step in (manager.all_steps() or []))
    if int(required) == 0:
        if steps:
            raise ValueError(
                f"the init state must restore nothing, but this directory holds checkpoints {steps}. "
                f"Point it at an empty directory -- S1.5's init is the untouched pretrained snapshot."
            )
        return state, 0
    if int(required) not in steps:
        raise ValueError(
            f"S1.5 needs checkpoint step {int(required)}, which is not in {steps or '[]'}. It is SELECTED, not "
            f"taken as the latest -- this directory holds later steps and the latest one is not the Tier-1 state."
        )
    restored = manager.restore(
        int(required),
        args=ocp.args.Composite(
            params=ocp.args.StandardRestore(state.params),
            opt_state=ocp.args.StandardRestore(state.opt_state),
            step=ocp.args.JsonRestore(),
        ),
    )
    return state.replace(params=restored["params"], opt_state=restored["opt_state"]), int(restored["step"]["step"])


def release(*objects) -> None:
    """Drop references and collect, so the next 5B state is not built beside the previous one."""
    del objects
    gc.collect()


def _empty_checkpoint_dir(config) -> str:
    """A guaranteed-empty directory, so the init state takes production's empty-restore path."""
    directory = f"{str(config.output_dir).rstrip('/')}/{config.run_name}/s1_5_init_empty_ckpt"
    tf.io.gfile.makedirs(directory)
    return directory


def state_report(
    state, batches, rng, config, scheduler, *, state_label: str, checkpoint_step: int, support_salts
) -> dict:
    """Every S1.5 measurement at ONE state (no updates; the caller pins that)."""
    plan = S1_5_STATE_PLAN[state_label]
    first_step = plan["first_global_step"]
    # The per-state knob, threaded as a TRACED value rather than captured in a closure: this is what
    # makes one compilation correct for both states instead of silently reusing the other's.
    ramp_origin = jnp.asarray(plan["ramp_origin"], jnp.int32)
    witness = RampWitness()
    gauge = LiveBufferGauge(state.params)
    gauge.sample("baseline")  # the state alone, before any gradient exists
    # The last cheap moment before the first 5B forward/backward: if anything is still resident that
    # should not be, this is where it can still be named.
    warn_if_no_headroom("pre_replay", state_label=state_label)
    view = state_view(config, state_label)

    def call_grad(tag: str, compiled, batch, batch_rng, global_step):
        """One cache-served gradient: TIMED on its first call at this state, and WITNESSED.

        The aux the compiled function emits is recorded here, at the call site, from the same
        return value the gradient came out of -- so the ramp this state actually ran with is
        evidence produced by the measurement rather than beside it.
        """
        timed = exp03._timed_first_call(f"probe_{tag}", compiled, state_label=state_label)
        (loss, ramp_aux), grad = timed(state.params, state, batch, batch_rng, global_step, ramp_origin)
        witness.record(tag, global_step, ramp_aux)
        return loss, grad

    rows = []
    for index, batch in enumerate(batches):
        global_step = jnp.asarray(first_step + index, jnp.int32)
        step_rng = jax.random.fold_in(rng, index)
        row = exp03.exp03_frozen_replay(
            state,
            batch,
            step_rng,
            view,
            scheduler,
            global_step=global_step,
            include_control=True,
            gauge=gauge,
        )
        row["batch_index"] = float(index)
        row["global_step"] = float(first_step + index)
        rows.append(row)

    # SUPPORT variance: only the salt moves, so the batch, epsilon, dropout key, logical step,
    # p_ss and A's coin are identical across the M draws.
    variance = {}
    for objective in S1_5_OBJECTIVES:
        loss_fn = _denoising_loss if objective == "control" else exp03.EXP03_LOSSES[objective]

        def _draws(batch, index, fn=loss_fn, objective=objective):
            global_step = jnp.asarray(first_step + index, jnp.int32)
            batch_rng = jax.random.fold_in(rng, index)
            for salt in support_salts:
                uses_step = objective != "control"

                def _build(origin, f=fn, salt=int(salt), obj=objective, uses_step=uses_step):
                    # ONE view, handed to the loss AND to the aux, so the emitted ramp numbers are
                    # by construction the ones this compiled program computed with.
                    traced = traced_view(config, state_label, origin, exp03_objective=obj, exp03_support_salt=salt)
                    return _with_ramp_aux(
                        traced,
                        lambda params, st, b, key, gs: f(
                            params,
                            st,
                            b,
                            key,
                            traced,
                            scheduler,
                            **({"global_step": gs} if uses_step else {}),
                        )[0],
                    )

                # ENTRY sample: whatever the previous iteration failed to release is still bound
                # here, before this iteration allocates anything. With the `del` below this shows
                # only Welford's mean; without it, the stale gradient shows up beside the mean.
                gauge.sample("variance_draw_entry")
                # The SALT is part of the tag, not only of the static key. Each salt is its own
                # compilation, and a tag shared across the M salts timed one of them and left the
                # other M-1 -- twelve of sixteen compilations per state -- with no entry at all.
                tag = variance_tag(objective, salt)
                compiled = _jitted_grad(tag, _build, static_key=(objective, int(salt)))
                _, gradient = call_grad(tag, compiled, batch, batch_rng, global_step)
                # SAMPLE INSIDE the peak window, and at the discriminating moment: this gradient is
                # live alongside Welford's mean AND anything the previous iteration failed to drop.
                # A stale loop variable shows up here as one extra tree.
                gauge.sample("variance_draw")
                yield gradient
                # ...and DROP it before the next reverse pass starts. Without this the previous
                # draw's gradient stayed bound while the next one was being computed -- a four-tree
                # peak on every iteration that no counter saw.
                del gradient

        # Generators, so a gradient exists only while Welford is consuming it.
        variance[objective] = variance_decomposition(_draws(batch, index) for index, batch in enumerate(batches))

    # THE cache-collision evidence, merged into the per-batch rows: A's ramp as the compiled
    # function that produced A's gradient reports it. The replay's own ``p_ss_a`` never had the
    # collision (its cache is keyed per state view), which is why asserting on that proved nothing.
    # The first salt, arbitrarily but deterministically: p_ss does not depend on the support salt,
    # and every salt's function emitted the same ramp -- ``cache_served_ramp`` carries all of them.
    a_tag = variance_tag("corrective_ss", list(support_salts)[0])
    for index, row in enumerate(rows):
        emitted = witness.at(a_tag, first_step + index)
        row["p_ss_cache_served"] = emitted["p_ss_used"]
        row["ramp_elapsed_cache_served"] = emitted["ramp_elapsed"]
    per_objective = {}
    for key in sorted({name for row in rows for name in row}):
        values = [row[key] for row in rows if key in row]
        per_objective[key] = {"mean": statistics.fmean(values), "min": min(values), "max": max(values)}

    # A's LABEL ISOLATION and the CONDITIONAL parity, on the first batch, forced to the branch each
    # one is about (p_ss=1 for the off-path label question, p_ss=0 for the identity).
    batch, batch_rng = batches[0], jax.random.fold_in(rng, 0)
    global_step = jnp.asarray(first_step, jnp.int32)
    isolation_values = {}
    for flag, label in ((False, "corrective"), (True, "same_eps")):

        def _build_isolation(origin, f=flag):
            traced = traced_view(config, state_label, origin, exp03_p_ss_max=1.0, exp03_p_ss_ramp_steps=0)
            return _with_ramp_aux(
                traced,
                lambda params, st, b, key, gs: corrective_at_support(
                    params, st, b, key, traced, scheduler, global_step=gs, same_eps_label_flag=f
                )[0],
            )

        loss, grad = call_grad(
            f"isolation_{label}",
            _jitted_grad(
                f"isolation_{label}",
                _build_isolation,
                static_key=("p_ss_max=1.0", "ramp=0", f"same_eps={flag}"),
            ),
            batch,
            batch_rng,
            global_step,
        )
        isolation_values[label] = (float(loss), grad)
        gauge.sample("isolation_peak")  # both label gradients live once the second one lands
        del grad  # ...and the loop variable does NOT survive into the parity phase
    isolation = label_isolation(
        corrective_loss=isolation_values["corrective"][0],
        same_eps_loss=isolation_values["same_eps"][0],
        corrective_grad=isolation_values["corrective"][1],
        same_eps_grad=isolation_values["same_eps"][1],
    )
    del isolation_values  # two 5B gradients, consumed and dropped before the parity pair is built

    def _build_parity_trial(origin):
        traced = traced_view(config, state_label, origin, exp03_p_ss_max=0.0, exp03_p_ss_ramp_steps=0)
        return _with_ramp_aux(
            traced,
            lambda params, st, b, key, gs: exp03._corrective_ss_loss(
                params, st, b, key, traced, scheduler, global_step=gs
            )[0],
        )

    def _build_parity_comparator(origin):
        traced = traced_view(config, state_label, origin, exp03_p_ss_max=0.0, exp03_p_ss_ramp_steps=0)
        return _with_ramp_aux(
            traced,
            lambda params, st, b, key, gs: plain_fixed_support_loss(
                params, st, b, key, traced, scheduler, global_step=gs
            )[0],
        )

    def _build_production_control(origin):
        traced = traced_view(config, state_label, origin)
        return _with_ramp_aux(
            traced, lambda params, st, b, key, gs: _denoising_loss(params, st, b, key, traced, scheduler)[0]
        )

    trial_loss, trial_grad = call_grad(
        "parity_trial",
        _jitted_grad("parity_trial", _build_parity_trial, static_key=("p_ss_max=0.0", "ramp=0")),
        batch,
        batch_rng,
        global_step,
    )
    comparator_loss, comparator_grad = call_grad(
        "parity_comparator",
        _jitted_grad("parity_comparator", _build_parity_comparator, static_key=("p_ss_max=0.0", "ramp=0")),
        batch,
        batch_rng,
        global_step,
    )
    parity = parity_report(
        trial_loss=trial_loss, plain_loss=comparator_loss, trial_grad=trial_grad, plain_grad=comparator_grad
    )
    production_loss, production_grad = call_grad(
        "parity_production_control",
        _jitted_grad("parity_production_control", _build_production_control),
        batch,
        batch_rng,
        global_step,
    )
    parity["production_control_difference"] = {
        "note": (
            "A at p_ss=0 vs the PRODUCTION control: not an identity -- A scores one scalar sigma per "
            "batch while the control samples a timestep per example -- so this is reported, not gated."
        ),
        "loss_relative_gap": abs(float(trial_loss) - float(production_loss)) / max(abs(float(production_loss)), 1e-12),
        "grad_relative_gap": _relative_gradient_gap(trial_grad, production_grad),
        "grad_cosine": exp03.grad_cosine(trial_grad, production_grad),
    }

    # The parity trio is consumed into scalars and RELEASED before the forced diagnostics allocate
    # their own gradients: holding trial + comparator + production while building a fourth was a
    # real four-tree peak, outside the replay's counter and outside its cap.
    gauge.sample("parity_peak")  # trial + comparator + production, all live, BEFORE the release
    del trial_grad, comparator_grad
    forced_diagnostics = {}
    for objective in ("corrective_ss", "combined"):

        def _build_forced(origin, fn=exp03.EXP03_LOSSES[objective], obj=objective):
            traced = traced_view(
                config, state_label, origin, exp03_objective=obj, exp03_p_ss_max=1.0, exp03_p_ss_ramp_steps=0
            )
            return _with_ramp_aux(
                traced, lambda params, st, b, key, gs: fn(params, st, b, key, traced, scheduler, global_step=gs)[0]
            )

        loss, grad = call_grad(
            f"forced_{objective}",
            _jitted_grad(f"forced_{objective}", _build_forced, static_key=(objective, "p_ss_max=1.0", "ramp=0")),
            batch,
            batch_rng,
            global_step,
        )
        forced_diagnostics[objective] = {
            "loss": float(loss),
            "grad_norm": float(np.sqrt(exp03.tree_sq_norm(grad))),
            "grad_cosine_vs_production_control": exp03.grad_cosine(grad, production_grad),
            "ramp_elapsed": witness.at(f"forced_{objective}", global_step)["ramp_elapsed"],
        }
        gauge.sample("forced_peak")  # this forced gradient plus the production control, both live
        del grad  # consumed in the same iteration that produced it
    del production_grad  # the last reference: nothing survives into the report but scalars
    gauge.sample("after_release")

    branch_counts = {
        "self_generated": sum(1 for row in rows if float(row.get("take_self_generated_a", 0.0)) > 0.5),
        "teacher_forced": sum(1 for row in rows if float(row.get("take_self_generated_a", 0.0)) <= 0.5),
        "coin_values": [float(row.get("coin_a", float("nan"))) for row in rows],
        "p_ss_values": [float(row.get("p_ss_a", float("nan"))) for row in rows],
    }
    return {
        "state": state_label,
        "checkpoint_step": int(checkpoint_step),
        "branch_outcomes": branch_counts,
        "first_global_step": first_step,
        "ramp_origin": plan["ramp_origin"],
        "num_batches": len(batches),
        "support_draws": len(list(support_salts)),
        "support_salts": [int(salt) for salt in support_salts],
        "per_objective": per_objective,
        "per_batch": rows,
        # What the cache-served functions themselves reported about the ramp they ran with, one row
        # per (tag, step). Deliberately NOT accompanied by a recomputation of the same quantity: the
        # previous evidence was exactly that, and it survived the mutation it was written to catch.
        "cache_served_ramp": witness.report(),
        "specializations": specialization_census(),
        **gauge.report(),
        "support_variance": variance,
        "label_isolation": isolation,
        "p_ss_zero_parity": parity,
        "forced_p_ss_one": forced_diagnostics,
    }


def s1_5_artifact(config, *, state_label: str, checkpoint_step: int, report: dict) -> dict:
    """The diagnostic JSON. Carries NO verdict fields — it is not admissible evidence."""
    return {
        "schema": S1_5_SCHEMA,
        "kind": "diagnostic",
        "note": (
            "exp_03 S1.5 no-update discriminator. NOT an evaluation artifact: no role, no cohort "
            "denominator, no aggregation schema. It cannot enter the success statistic."
        ),
        "state": state_label,
        "checkpoint_step": int(checkpoint_step),
        "run_name": str(getattr(config, "run_name", "")),
        "commit": gen._eval_code_commit(),
        "checkpoint_dir": str(getattr(config, "checkpoint_dir", "")),
        "train_data_dir": str(getattr(config, "train_data_dir", "")),
        "model_manifest_path": str(getattr(config, "model_manifest_path", "")),
        "num_batches": S1_5_NUM_BATCHES,
        "support_draws": S1_5_SUPPORT_DRAWS,
        "support_salts": S1_5_SUPPORT_SALTS,
        "objectives": list(S1_5_OBJECTIVES),
        "parity_tolerance": S1_5_PARITY_TOLERANCE,
        "first_global_step": S1_5_STATE_PLAN[state_label]["first_global_step"],
        "ramp_origin": S1_5_STATE_PLAN[state_label]["ramp_origin"],
        "required_checkpoint_step": S1_5_STATE_PLAN[state_label]["required_checkpoint_step"],
        "restored_step": int(checkpoint_step),
        "iterator_seed": int(getattr(config, "seed", 0)) + int(checkpoint_step),
        "report": report,
    }


def assert_finite_payload(payload, trail: str = "") -> None:
    """Refuse to write a JSON that carries a non-finite number.

    A ``NaN`` in a diagnostic is not a datum, it is a bug that has learned to look like one -- and
    ``json`` writes it as the invalid literal ``NaN``, which every strict reader then rejects. Better
    to fail here, where the failing key can be named.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert_finite_payload(value, f"{trail}.{key}" if trail else str(key))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            assert_finite_payload(value, f"{trail}[{index}]")
    elif isinstance(payload, float) and not math.isfinite(payload):
        raise ValueError(f"refusing to write a non-finite value at {trail!r}: {payload!r}")


def write_s1_5_artifact(path: str, payload: dict) -> None:
    """Immutable write, reusing the eval path's compare-or-refuse writer."""
    assert_finite_payload(payload)
    gen._write_json_immutable(path, payload)


def log_summary(payload: dict) -> None:
    report = payload["report"]
    max_logging.log(f"[exp03_s1_5] ===== state={payload['state']} checkpoint_step={payload['checkpoint_step']} =====")
    max_logging.log(
        f"[exp03_s1_5] {'objective':<14}{'loss':>12}{'grad_norm':>12}{'cos_vs_ctrl':>13}{'noise_scale':>13}"
    )
    for objective in S1_5_OBJECTIVES:
        short = {"control": "control", "corrective_ss": "a", "rollout_loss": "b", "combined": "c"}[objective]
        per = report["per_objective"]
        loss = per.get(f"loss_{short}", {}).get("mean", float("nan"))
        norm = per.get(f"grad_norm_{short}", {}).get("mean", float("nan"))
        cosine = per.get(f"grad_cosine_{short}_vs_control", {}).get("mean", float("nan"))
        noise = report["support_variance"][objective]["gradient_noise_scale"]
        max_logging.log(f"[exp03_s1_5] {objective:<14}{loss:>12.6f}{norm:>12.4f}{cosine:>13.4f}{noise:>13.4g}")
    for objective, stats in report["support_variance"].items():
        max_logging.log(
            f"[exp03_s1_5]   {objective:<14} support_var={stats['support_variance']:.6g} "
            f"batch_var={stats['batch_shared_rng_variance']:.6g} support_fraction={stats['support_fraction']:.3f}"
        )
    if "label_isolation" in report:
        isolation = report["label_isolation"]
        max_logging.log(
            f"[exp03_s1_5]   label isolation: loss {isolation['loss_same_eps']:.6f} -> "
            f"{isolation['loss_corrective']:.6f} (delta {isolation['loss_delta']:+.6f}), "
            f"grad cosine {isolation['grad_cosine']:.4f}, relative delta {isolation['grad_relative_delta']:.4f}"
        )
    if "p_ss_zero_parity" in report:
        parity = report["p_ss_zero_parity"]
        verdict = "PASS" if parity["passes"] else "FAIL"
        max_logging.log(
            f"[exp03_s1_5]   p_ss=0 parity: {verdict} (loss gap {parity['loss_relative_gap']:.3g}, "
            f"grad gap {parity['grad_relative_gap']:.3g}, tolerance {parity['tolerance']:.1g})"
        )


def run_s1_5(config) -> dict:
    """Both states, no updates, one immutable JSON each."""
    assert_approved_s1_5_design(
        num_batches=int(getattr(config, "s1_5_num_batches", S1_5_NUM_BATCHES)),
        support_draws=int(getattr(config, "s1_5_support_draws", S1_5_SUPPORT_DRAWS)),
        states=S1_5_STATES,
    )
    trainer = exp03.Exp03Trainer(config)
    trainer._validate_probe_config(config)
    trainer._validate_overfit100_config(config)
    trainer._validate_pinned_snapshot(config)
    trainer._preflight_dataset()
    scheduler, _ = trainer._create_scheduler()

    assert_commit_is_pinned()
    payloads = {}
    for state_label in S1_5_STATES:
        payloads[state_label] = _run_one_state(config, trainer, scheduler, state_label)
        # The previous state's parameters, optimizer state and trace rows are released BEFORE the
        # next pipeline is built: two live 5B states is an OOM, and holding the first one across the
        # second load is exactly how that happens.
        release()
    # BOTH states are behind us, so this is the production total: one compilation per probe tag,
    # serving both. The per-state artifacts carry the same census; this is the loud version.
    census = specialization_census()
    assert_one_specialization_per_probe_tag(census)
    log_compile_costs(census)
    return {label: payload for label, payload in payloads.items() if payload}


def log_compile_costs(census: dict) -> None:
    """The specialization total and every timed first call, per (tag, state).

    The timings are LOGGED and not written into the artifact on purpose: the S1.5 JSONs are written
    through the compare-or-refuse immutable writer, and a wall-clock number would make a re-run of
    the identical code look like a conflicting result. The specialization counts, which are
    reproducible, do go into the artifact.
    """
    max_logging.log(
        f"[exp03_s1_5] specializations: total={census['total']} "
        f"(probe {census['probe_total']} over {len(census['probe'])} tags, replay {census['replay_total']}, "
        f"helpers {census['helper_total']})"
    )
    for (tag, state_label), seconds in sorted(exp03.COMPILE_TIMINGS.items()):
        max_logging.log(f"[exp03_s1_5]   first call {tag} [{state_label}]: {seconds:.1f}s")


def _run_one_state(config, trainer, scheduler, state_label: str) -> dict:
    """One state, start to finish, in its own frame so its locals die when it returns."""
    pipeline = trainer._load_wan_pipeline()
    mesh = pipeline.mesh  # metadata, and the ONE thing from the pipeline that outlives it here
    state, _, checkpoint_step = build_probe_state(config, trainer, pipeline, mesh, state_label=state_label)
    # The pipeline's weights were dropped inside ``build_probe_state``; the object itself goes here.
    # Nothing below reads it -- the sigma trace works from the state already in memory -- and a name
    # still bound to it is a name that can pin whatever it is later given.
    del pipeline
    release()
    log_memory("post_dead_weight_free", state_label=state_label)
    before = params_fingerprint(state.params)
    iterator = trainer._load_dataset(mesh, is_training=True, seed=config.seed + checkpoint_step)
    # THE call the training loop makes, from the module that actually defines it
    # (``train_utils.load_next_batch``, exactly as ``start_training`` imports and calls it). The
    # first hardware attempt died here on ``gen.load_next_batch`` -- a name the eval module has
    # never had -- because the batches in the tests were built rather than pulled.
    batches = [load_next_batch(iterator, None, config) for _ in range(S1_5_NUM_BATCHES)]
    try:
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            report = state_report(
                state,
                batches,
                jax.random.key(config.seed + 1),
                config,
                scheduler,
                state_label=state_label,
                checkpoint_step=checkpoint_step,
                support_salts=S1_5_SUPPORT_SALTS,
            )
            trace_rows = trace_in_memory_state(
                state, config, scheduler, state_label=state_label, checkpoint_step=checkpoint_step
            )
    except Exception as failure:
        # The heavy phase is where the memory goes and where Job 8d died. Whatever killed it, the
        # ledger and the residue are printed BEFORE the traceback -- then the failure is re-raised
        # untouched, because a probe that swallows an OOM is worse than one that dies loudly.
        log_memory("failure", state_label=state_label)
        log_largest_live_arrays(f"failure ({type(failure).__name__})")
        raise
    assert_no_update(before, params_fingerprint(state.params))
    payload = {}
    if jax.process_index() == 0:
        # The sigma trace lands as its OWN artifact too, under the trace module's canonical rules,
        # so mechanism B has the standalone per-state file its later comparisons expect.
        trace_payload = trace.trace_artifact(
            config, checkpoint_step=checkpoint_step, cohort=trace_rows["cohort_windows"], rows=trace_rows["rows"]
        )
        trace.write_trace_artifact(trace.trace_output_path(config, checkpoint_step), trace_payload)
        report["sigma_trace"] = {key: value for key, value in trace_rows.items() if key != "cohort_windows"}
        payload = s1_5_artifact(config, state_label=state_label, checkpoint_step=checkpoint_step, report=report)
        path = s1_5_output_path(config, state_label=state_label, checkpoint_step=checkpoint_step)
        write_s1_5_artifact(path, payload)
        log_summary(payload)
        max_logging.log(f"[exp03_s1_5] wrote {path}")
    release(state, batches, report, iterator)  # the pipeline is already gone, by name and by weight
    return payload


def assert_commit_is_pinned() -> None:
    """Refuse to run without a real 40-hex COMMIT (the exp_02 resume precedent).

    An artifact stamped ``unknown`` cannot be tied to the code that produced it, which is the one
    thing a diagnostic must never lose.
    """
    commit = str(gen._eval_code_commit())
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(
            f"refusing to run S1.5 with COMMIT={commit!r}: a 40-hex commit is required so the artifacts can be "
            f"tied to the code that produced them. Export COMMIT=$(git rev-parse HEAD) in the launcher."
        )


def trace_in_memory_state(state, config, scheduler, *, state_label: str, checkpoint_step: int) -> dict:
    """Mechanism-B sigma trace of the state in memory (no second restore, no second 5B load).

    It never took anything from the pipeline -- the transformer is rebuilt from the state's own
    graphdef -- and the ``pipeline`` parameter it used to accept was a false dependency that kept a
    ~5B module reachable across the heaviest phase of the probe. It is gone.
    """
    cohort = trace.trace_cohort(
        str(getattr(config, "model_manifest_path", "")), seed=trace.TRACE_SEED, num_windows=trace.TRACE_NUM_WINDOWS
    )
    sigmas, timesteps = trace.overfit100_sampler_grid(
        num_inference_steps=trace.TRACE_SAMPLING_STEPS,
        flow_shift=config.flow_shift,
        sigma_min=scheduler.config.sigma_min,
        sigma_max=scheduler.config.sigma_max,
        num_train_timesteps=scheduler.config.num_train_timesteps,
    )
    weights_dtype = gen._dtype(config.weights_dtype)
    transformer = nnx.merge(state.graphdef, state.params, state.rest_of_state)

    def velocity_fn(hidden_states, timestep, encoder_hidden_states):
        return transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            deterministic=True,
        )

    samples = gen.read_overfit100_samples(config, cohort)
    rows = []
    for sample in samples:
        z_i0 = jnp.asarray(sample.z_i0[None]).astype(weights_dtype)
        z_gt = jnp.asarray(sample.z_video[None]).astype(weights_dtype)
        context = jnp.broadcast_to(
            state.context_table[sample.episode_index][None].astype(weights_dtype),
            (1, state.context_table.shape[1], state.context_table.shape[2]),
        )
        eps = trace.trace_noise(
            episode_id=sample.episode_id, window_start=sample.window_start, shape=z_gt.shape, dtype=z_gt.dtype
        )
        rows.append(
            {
                "name": sample.name,
                "episode_id": int(sample.episode_id),
                "episode_index": int(sample.episode_index),
                "window_start": int(sample.window_start),
                "trace": trace.trace_rows(
                    velocity_fn, z_gt=z_gt, z_i0=z_i0, eps=eps, context=context, sigmas=sigmas, timesteps=timesteps
                ),
            }
        )
    return {
        "state": state_label,
        "checkpoint_step": int(checkpoint_step),
        "cohort": [str(window["name"]) for window in cohort],
        "cohort_windows": cohort,
        "mean_trace": trace.mean_trace(rows),
        "rows": rows,
    }


def run(argv: Sequence[str]):
    pyconfig.initialize(argv)
    config = pyconfig.config
    if str(getattr(config, "model_type", "")) != exp03.EXP03_MODEL_TYPE:
        raise ValueError(f"S1.5 requires model_type == {exp03.EXP03_MODEL_TYPE!r}; got {config.model_type!r}")
    return run_s1_5(config)


def main(argv: Sequence[str]) -> None:  # pragma: no cover - entry point
    run(argv)


if __name__ == "__main__":  # pragma: no cover
    app.run(main)
