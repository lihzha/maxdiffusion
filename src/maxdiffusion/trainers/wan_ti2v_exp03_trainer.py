"""exp_03 `rollout_objective` — the trainer that can carry the new objectives.

Everything about a run except the objective is inherited from the exp_02 overfit100 trainer: the
same ``Overfit100TrainState``, the same Orbax item shapes, the same preflights (config gates, pinned
snapshot, dataset byte-verify, manifest-bound context table), the same optimizer, dataset, sharding,
checkpoint schedule and loop. Only :meth:`Exp03Trainer._loss_and_step_fns` is overridden, because
that is the one thing exp_03 changes.

**This round implements ``exp03_objective: control`` only** — and implements it by returning the
parent's own functions *by identity*, not by a copy that would merely look equal. That is what makes
ctrl0 a replication guard rather than a second implementation: with the same seed, data order and
checkpoint bytes, ctrl0 through this trainer executes exactly the code exp_02 executed. The three
real objectives (``corrective_ss``, ``rollout_loss``, ``combined``) are accepted by the config
surface and raise :class:`NotImplementedError` until round 3, so a launch that names one fails at
startup instead of quietly training the control.

**RNG discipline (plan v3.1 §1b, delta-review 2).** The *shared stream* is exp_02's: a single
``jax.random.key(seed + 1)`` advanced by exactly one ``split`` per step inside the train step, whose
per-step key then splits into (noise, timestep, dropout). ctrl0 and the shared draws of every arm
must reproduce it exactly, so the new objectives' own randomness — the ``p_ss`` coins, the sigma
index supports, the self-generation noise — comes from :func:`exp03_aux_key`, an *auxiliary* key
derived from ``(seed, global_step, purpose)``. It is derived, never split off the stream, so
consuming it cannot advance the stream; and because it is keyed on the global step rather than on
call order, it survives a resume unchanged.
"""

from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion import max_logging
from maxdiffusion.trainers.wan_ti2v_overfit100_trainer import (
    WanTI2VOverfit100Trainer,
    _denoising_loss,
    _train_step,
)

EXP03_MODEL_TYPE = "EXP03_TI2V"

# The objective surface, fixed by plan v3.1 §1. "control" is the plain one-step objective, i.e.
# exp_02's; the other three are the trials.
EXP03_CONTROL_OBJECTIVE = "control"
EXP03_OBJECTIVES = (EXP03_CONTROL_OBJECTIVE, "corrective_ss", "rollout_loss", "combined")
EXP03_IMPLEMENTED_OBJECTIVES = (EXP03_CONTROL_OBJECTIVE,)

# Auxiliary-key purposes. Named rather than numbered so adding one later cannot renumber the
# others: the purpose's id is a hash of its NAME (see _purpose_id).
EXP03_AUX_PURPOSES = (
    "p_ss_coin",  # trial A: the scheduled-sampling coin
    "index_support",  # trials A/B: the sigma grid start/length draw
    "self_gen_noise",  # trial A: epsilon for the self-generated state
)

# Offset for the auxiliary root key. The shared stream is key(seed + 1); the auxiliary root is a
# DIFFERENT key, not a descendant of it, so the two can never collide or interleave.
EXP03_AUX_SEED_OFFSET = 1_000_003


def _purpose_id(purpose: str) -> int:
    """A stable 32-bit id for a purpose NAME (sha256, so it never depends on declaration order)."""
    if purpose not in EXP03_AUX_PURPOSES:
        raise ValueError(f"unknown exp_03 rng purpose {purpose!r}; declared purposes are {list(EXP03_AUX_PURPOSES)}")
    return int.from_bytes(hashlib.sha256(purpose.encode("utf-8")).digest()[:4], "big")


def exp03_aux_key(*, seed: int, global_step, purpose: str) -> jax.Array:
    """An auxiliary key for a NEW-objective draw, derived without touching the shared stream.

    ``fold_in(fold_in(key(seed + offset), global_step), purpose_id)``. Four properties matter and
    each is tested:

    * **Non-advancing.** It is derived from the config seed, not split off the training rng, so a
      step that draws from it leaves the shared stream in exactly the state exp_02 would have left
      it in. This is what lets ctrl0 be a bitwise replication guard while A/B/C add randomness.
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
    """
    if isinstance(global_step, (int, np.integer)) and not isinstance(global_step, bool) and int(global_step) < 0:
        raise ValueError(f"global_step must be non-negative; got {global_step}")
    key = jax.random.key(int(seed) + EXP03_AUX_SEED_OFFSET)
    key = jax.random.fold_in(key, jnp.asarray(global_step, dtype=jnp.uint32))
    return jax.random.fold_in(key, _purpose_id(purpose))


def validate_exp03_config(config) -> str:
    """Gate the exp_03 knobs before anything expensive happens; return the objective."""
    objective = str(getattr(config, "exp03_objective", EXP03_CONTROL_OBJECTIVE))
    if objective not in EXP03_OBJECTIVES:
        raise ValueError(f"exp03_objective must be one of {list(EXP03_OBJECTIVES)}; got {objective!r}")

    k_a = int(getattr(config, "exp03_k_a", 2))
    if k_a not in (1, 2):
        raise ValueError(f"exp03_k_a is the MAX scheduled-sampling advance and the plan fixes it at 2; got {k_a}")
    k_b = int(getattr(config, "exp03_k_b", 2))
    if k_b != 2:
        raise ValueError(f"exp03_k_b is fixed at 2 by the plan (a two-step rollout loss); got {k_b}")
    lam = float(getattr(config, "exp03_lambda", 0.5))
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"exp03_lambda must be in [0, 1]; got {lam}")
    p_ss_max = float(getattr(config, "exp03_p_ss_max", 0.5))
    if not 0.0 <= p_ss_max <= 1.0:
        raise ValueError(f"exp03_p_ss_max must be in [0, 1]; got {p_ss_max}")
    ramp = int(getattr(config, "exp03_p_ss_ramp_steps", 500))
    if ramp < 0:
        raise ValueError(f"exp03_p_ss_ramp_steps must be non-negative; got {ramp}")
    return objective


class Exp03Trainer(WanTI2VOverfit100Trainer):
    """The exp_02 overfit100 trainer with a switchable objective (``model_type: EXP03_TI2V``).

    The subclass exists for provenance as much as for behaviour: a run recorded as ``EXP03_TI2V``
    can be told apart from an exp_02 run in every artifact, even when — as with ctrl0 — it is
    deliberately executing the identical code.
    """

    def _loss_and_step_fns(self):
        objective = validate_exp03_config(self.config)
        if objective == EXP03_CONTROL_OBJECTIVE:
            # BY IDENTITY, not by copy: ctrl0's job is to reproduce exp_02, and the only way to be
            # sure it does is to run the same functions.
            return _denoising_loss, _train_step
        raise NotImplementedError(
            f"exp03_objective={objective!r} is declared by plan v3.1 but lands in round 3; only "
            f"{list(EXP03_IMPLEMENTED_OBJECTIVES)} is implemented. Refusing to start rather than silently "
            f"training the control under a trial's run name."
        )

    def start_training(self):
        # Fail on a bad objective/knob before the ~5B load, and say which one is running.
        objective = validate_exp03_config(self.config)
        if jax.process_index() == 0:
            max_logging.log(f"[wan_exp03] objective={objective} (model_type={self.config.model_type})")
            if objective == EXP03_CONTROL_OBJECTIVE:
                max_logging.log(
                    "[wan_exp03] control arm: the parent's plain one-step objective, by identity -- "
                    "this run is a replication of exp_02's recipe through the exp_03 trainer"
                )
        self._loss_and_step_fns()  # raises here, not 20 minutes in, for an unimplemented trial
        return super().start_training()
