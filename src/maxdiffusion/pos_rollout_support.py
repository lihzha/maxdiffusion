"""exp_06 `rollout_adapter` — the pinned exp_03 RNG/support primitives, exp_06-owned.

exp_06 trains the pre_context adapter through a short differentiable rollout, and plan §3b pins the
randomness policy that rollout consumes: **one support draw per batch**, the start position uniform
over the legal grid range, the terminal ``sigma = 0`` never a start (and never an endpoint either),
and a PRNG derivation that is resume- and accumulation-stable. exp_03 already built exactly those
primitives and ran them; T1's job is to give exp_06 its own pinned copy rather than to reinvent them
or to merge exp_03's branch (plan-review F2: a branch merge measured ~44k insertions with real
conflicts, and was rejected).

**What is here.** ``_purpose_id``, :func:`exp03_aux_key` and :func:`rollout_support` are LIFTED —
semantics preserved exactly, same key-derivation arithmetic, same support-draw construction — out of
exp_03's 935-line full-finetune trainer, which is not importable here (it is coupled to exp_03's
context table, train state and objective surface). The names are kept as exp_03 wrote them so that
provenance is nominal as well as behavioural: a future re-pin is then a pure content diff.

**What is NOT here.** exp_03's ``corrective_support`` (trial A / R-A) is not lifted: R-A is a
CONDITIONAL arm admitted only by a recorded decision (plan §3), and its support construction is
arm-specific. Its *purposes* stay declared in :data:`EXP03_AUX_PURPOSES` regardless, because purpose
ids are hashes of the NAME — trimming the tuple would change nothing numerically but would make a
later R-A admission a re-pin event for no gain. exp_03 drew its epsilon from the shared training stream, so no epsilon purpose existed at the pin;
exp_06 added `rollout_epsilon` (T3b-1), `one_step_index` (T3b-1 strengthen) and `dev_instrument`
(T3b-3) additively. **Seven purposes are live**: the four inherited above plus those three. Because
an id is the hash of its NAME, every addition left every earlier draw bit-identical.

**Provenance** — recorded below as module constants so it is machine-checkable, and pinned by
``tests/worklogs_yixun/test_exp03_imports.py``. exp_03 @ ``2ef9b8a`` is READ-ONLY to exp_06; nothing
in this file may drift from it without a recorded decision to advance the pin (plan §5-1).
"""

from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np

# =============================================================================================
# Provenance of everything exp_06 took from exp_03 at T1 (plan §5-1).
# =============================================================================================

EXP03_SOURCE_EXPERIMENT = "exp_03_rollout_objective"
EXP03_SOURCE_BRANCH = "claude-exp_03_rollout_objective-20260802"
# The plan pins the short SHA `2ef9b8a` — exp_03's last APPROVE+GO reviewed commit.
EXP03_SOURCE_COMMIT = "2ef9b8a56f22d412cf989654e38883f76b12f2cf"
EXP03_PLAN_PIN_SHORT = "2ef9b8a"

# Files copied wholesale. ``git_blob_sha1`` is git's own object id for the pinned blob (an
# independent second lock on the bytes); ``body_sha256`` covers the copy below its exp_06
# provenance header, which is the part that must stay byte-identical.
EXP03_IMPORTED_BLOBS = {
    "src/maxdiffusion/models/wan/overfit100_sampling.py": {
        "target_path": "src/maxdiffusion/models/wan/overfit100_sampling.py",
        "git_blob_sha1": "8104edf416d199ed6d3f4928848b2ded1fb33dea",
        "body_sha256": "7e91fbd78185c3dd6ec03b40418fe7065c9a5c54b4e8a04981db88d2f5545939",
        "header_sentinel": "# === END exp_06 PROVENANCE HEADER — everything below is the pinned blob, verbatim ===\n",
    },
}

# Symbols lifted out of a file that could NOT be copied wholesale. ``sha256`` is of the exact
# source segment at the pin (``ast.get_source_segment``), recorded so a re-pin diff is checkable;
# the behavioural pin is the independent re-derivation in the T1 tests, not this string.
_EXP03_TRAINER_PATH = "src/maxdiffusion/trainers/wan_ti2v_exp03_trainer.py"
EXP03_EXTRACTED_SYMBOLS = {
    "EXP03_AUX_PURPOSES": {
        "source_path": _EXP03_TRAINER_PATH,
        "source_lines": (70, 75),
        "sha256": "523248da4393eaa3122e0b9866c5512fef8c28be4a91db8220c083ec733bfd0b",
    },
    "EXP03_AUX_SEED_OFFSET": {
        "source_path": _EXP03_TRAINER_PATH,
        "source_lines": (82, 82),
        "sha256": "e0807354c4cbba53ebe5c73b59642c6749651178670cc6e0b339fa2e8d41cf0a",
    },
    "_purpose_id": {
        "source_path": _EXP03_TRAINER_PATH,
        "source_lines": (85, 89),
        "sha256": "e83b91a34188ab692e103e9c69a36153f11c80d6fb87271818203432041e3018",
    },
    "exp03_aux_key": {
        "source_path": _EXP03_TRAINER_PATH,
        "source_lines": (92, 124),
        "sha256": "3edc1d4ca95ae77a13e33f56d07e3ff65ea2f2db2fe48e5706dca84dc9df8ad2",
    },
    "rollout_support": {
        "source_path": _EXP03_TRAINER_PATH,
        "source_lines": (312, 320),
        "sha256": "e863cd5432291c2e596feef47089e1b1d84ec90264ececcd97da7db676add1d1",
    },
}

# =============================================================================================
# The lifted primitives. Bodies are exp_03's; only the surrounding commentary is exp_06's.
# =============================================================================================

# Auxiliary-key purposes. Named rather than numbered so adding one later cannot renumber the
# others: the purpose's id is a hash of its NAME (see _purpose_id).
EXP03_AUX_PURPOSES = (
    "p_ss_coin",  # trial A: the scheduled-sampling coin
    "k_a_draw",  # trial A: k_A ~ U{1..exp03_k_a}, drawn FIRST (plan v2.2)
    "index_support",  # trial A: the start index s ~ U{0 .. 24 - k_A}
    "index_support_rollout",  # trial B: the start index s ~ U{0 .. 22}  <- exp_06's R-B draw
    # ADDED AT T3b-1, additively (plan §3b, T1 finding 4). exp_03 drew its epsilon from the shared
    # training stream; exp_06's loop needs a resume-stable one, so it gets its own purpose. Because
    # an id is the hash of its NAME, appending here leaves every id above -- and so every draw ever
    # made from them -- bit-identical; `test_pos_rollout_stream.py` pins exactly that.
    "rollout_epsilon",  # exp_06 R-B / matched-C0: the per-step epsilon
    "one_step_index",  # exp_06 matched-C0: the per-example sigma index (T3b-1 strengthen)
    # T3b-3: the DEV-64 selection instrument. DEDICATED so selection randomness never moves
    # when the training stream does -- sharing a purpose would make the same checkpoint score
    # differently depending on which step it happened to be evaluated at.
    "dev_instrument",  # exp_06 selection: the fixed per-(example, eval) draw
)

# Offset for the auxiliary root key. The shared stream is key(seed + 1); the auxiliary root is a
# DIFFERENT key, not a descendant of it, so the two can never collide or interleave.
EXP03_AUX_SEED_OFFSET = 1_000_003


def _purpose_id(purpose: str) -> int:
    """A stable 32-bit id for a purpose NAME (sha256, so it never depends on declaration order)."""
    if purpose not in EXP03_AUX_PURPOSES:
        raise ValueError(f"unknown exp_03 rng purpose {purpose!r}; declared purposes are {list(EXP03_AUX_PURPOSES)}")
    return int.from_bytes(hashlib.sha256(purpose.encode("utf-8")).digest()[:4], "big")


def exp03_aux_key(*, seed: int, global_step, purpose: str, salt: int = 0) -> jax.Array:
    """An auxiliary key for a NEW-objective draw, derived without touching the shared stream.

    ``fold_in(fold_in(key(seed + offset), global_step), purpose_id)``. Four properties matter and
    each is tested:

    * **Non-advancing.** It is derived from the config seed, not split off the training rng, so a
      step that draws from it leaves the shared stream in exactly the state the one-step trainer
      would have left it in. In exp_06 this is what lets R-B and matched-C0 consume THE SAME batch
      stream at matched seeds (plan §3b) even though only R-B makes a support draw.
    * **Resume-stable.** Keyed on the global step, not on how many times it has been called, so a
      preempted-and-restarted segment draws the same values at the same steps. The caller must pass
      the LOOP's step: ``state.step`` is not resume-safe, because a restore brings back params and
      opt_state and leaves the freshly-built step behind.
    * **Cross-arm aligned.** The same ``(seed, step, purpose)`` gives the same key in every arm, so
      where two arms make the same kind of draw they make the *same* draw.
    * **Tracer-safe.** ``global_step`` arrives as a traced scalar from inside the compiled train
      step, so it is folded in as an array and never coerced with ``int()``. ``seed`` and
      ``purpose`` are static (config values), so they may stay Python-level. The non-negative check
      therefore applies only when the step is a concrete Python integer -- under tracing there is
      no value to check, and a fabricated one would be worse than none.

    Accumulation-stability (plan §3b) follows from the same keying: every microbatch of a gradient
    accumulation shares one ``global_step``, so they share one support -- one draw per batch.
    """
    if isinstance(global_step, (int, np.integer)) and not isinstance(global_step, bool) and int(global_step) < 0:
        raise ValueError(f"global_step must be non-negative; got {global_step}")
    key = jax.random.key(int(seed) + EXP03_AUX_SEED_OFFSET)
    key = jax.random.fold_in(key, jnp.asarray(global_step, dtype=jnp.uint32))
    key = jax.random.fold_in(key, _purpose_id(purpose))
    # ``salt`` re-draws THIS purpose while everything else about the step stays put. It is folded
    # only when non-zero, so every existing draw is bit-identical to what it was before the salt
    # existed. exp_03's S1.5 used it to vary the sigma SUPPORT alone: moving the global step instead
    # would also move A's p_ss coin and the ramp, and the resulting variance would mix three things.
    if int(salt) != 0:
        key = jax.random.fold_in(key, int(salt))
    return key


def rollout_support(*, seed: int, global_step, num_steps: int, k_b: int, support_salt: int = 0):
    """Trial B's support: ``s ~ U{0 .. num_steps-1-k_B}``, path ``s -> s+1 -> ... -> s+k_B``.

    This is exp_06's R-B support draw (plan §3b). ``num_steps`` is N, the number of sampler steps;
    the sigma grid has N+1 entries with ``sigmas[N] == 0``. The draw is a SCALAR -- one support per
    batch, not per example -- and ``end = start + k_B <= N - 1``, so the terminal zero sigma is
    unreachable as an endpoint by construction rather than by a clamp, and a fortiori is never a
    start. At the production N=25, k=2 this is ``start in {0..22}``, ``end in {2..24}``.

    §3b states the same range 1-based (``{1 .. N-k}``); this 0-based form is the identical set, and
    is the construction the pin runs.
    """
    start = jax.random.randint(
        exp03_aux_key(seed=seed, global_step=global_step, purpose="index_support_rollout", salt=support_salt),
        (),
        0,
        num_steps - int(k_b),
    )
    return start, start + int(k_b)
