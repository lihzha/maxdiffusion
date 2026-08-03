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

import statistics
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from absl import app
from flax import nnx
from flax.linen import partitioning as nn_partitioning

import maxdiffusion.generate_wan_side_adapter as gen
import maxdiffusion.trainers.wan_ti2v_exp03_trainer as exp03
from maxdiffusion import max_logging, pyconfig
from maxdiffusion.diagnostics_exp03 import sigma_trajectory_trace as trace
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


def params_fingerprint(params) -> list[float]:
    """A cheap bit-level fingerprint of a parameter tree (sum of the raw bits per leaf)."""
    out = []
    for leaf in jax.tree_util.tree_leaves(params):
        array = np.asarray(leaf)
        out.append(float(np.sum(array.view(np.uint8).astype(np.float64))))
    return out


def assert_no_update(before: Sequence[float], after: Sequence[float]) -> None:
    """The probe's defining property: it changed nothing.

    Bit-level, not ``allclose``: a probe that nudged the parameters would report the objectives'
    behaviour at a state that no longer exists, and a tolerance would hide exactly the small update
    an accidental ``apply_gradients`` produces.
    """
    if list(before) != list(after):
        differing = [index for index, (x, y) in enumerate(zip(before, after)) if x != y]
        raise RuntimeError(
            f"S1.5 modified the parameters it was measuring: {len(differing)} leaf/leaves changed "
            f"(first at index {differing[:3]}). This probe applies no updates by construction."
        )


# ------------------------------------------------------------------------ support-variance (D2)


def variance_decomposition(gradients) -> dict:
    """Split gradient variance into its SUPPORT and DATA parts.

    ``gradients[k][m]`` is the gradient on batch ``k`` under support draw ``m``. By the law of total
    variance, the mean within-batch spread (over the ``M`` draws) is the part the support draw is
    responsible for, and the spread of the per-batch means is the part the data is responsible for.
    Reported as squared-L2 quantities, plus a noise scale relative to the mean gradient -- the number
    that says whether a null result could be the support estimator rather than the objective.
    """
    rows = [
        [
            np.concatenate([np.asarray(leaf, np.float64).ravel() for leaf in jax.tree_util.tree_leaves(grad)])
            for grad in row
        ]
        for row in gradients
    ]
    if not rows or not rows[0]:
        raise ValueError("variance_decomposition needs at least one batch with at least one draw")
    draws = {len(row) for row in rows}
    if len(draws) != 1:
        raise ValueError(f"every batch must contribute the same number of support draws; got {sorted(draws)}")
    per_batch_mean = [np.mean(np.stack(row), axis=0) for row in rows]
    grand_mean = np.mean(np.stack(per_batch_mean), axis=0)
    support = (
        float(
            np.mean(
                [
                    np.mean([float(np.sum((grad - mean) ** 2)) for grad in row])
                    for row, mean in zip(rows, per_batch_mean)
                ]
            )
        )
        if len(rows[0]) > 1
        else 0.0
    )
    data = float(np.mean([float(np.sum((mean - grand_mean) ** 2)) for mean in per_batch_mean]))
    total = support + data
    mean_sq = float(np.sum(grand_mean**2))
    return {
        "num_batches": len(rows),
        "support_draws": len(rows[0]),
        "support_variance": support,
        "data_variance": data,
        "total_variance": total,
        "support_fraction": (support / total) if total > 0 else 0.0,
        "mean_grad_sq_norm": mean_sq,
        "gradient_noise_scale": (total / mean_sq) if mean_sq > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------- A's label isolation


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
        "grad_norm_corrective": float(np.linalg.norm(exp03._flat_gradient(corrective_grad))),
        "grad_norm_same_eps": float(np.linalg.norm(exp03._flat_gradient(same_eps_grad))),
        "grad_cosine": exp03.grad_cosine(corrective_grad, same_eps_grad),
        "grad_relative_delta": float(
            np.linalg.norm(exp03._flat_gradient(corrective_grad) - exp03._flat_gradient(same_eps_grad))
            / max(float(np.linalg.norm(exp03._flat_gradient(same_eps_grad))), 1e-12)
        ),
    }


# ------------------------------------------------------------------------------- p_ss = 0 parity


def parity_report(*, trial_loss, plain_loss, trial_grad, plain_grad, tolerance: float = S1_5_PARITY_TOLERANCE) -> dict:
    """A at ``p_ss=0`` against the plain objective — loss AND gradient, relative tolerance."""
    trial_vector, plain_vector = exp03._flat_gradient(trial_grad), exp03._flat_gradient(plain_grad)
    loss_gap = abs(float(trial_loss) - float(plain_loss)) / max(abs(float(plain_loss)), 1e-12)
    grad_gap = float(np.linalg.norm(trial_vector - plain_vector)) / max(float(np.linalg.norm(plain_vector)), 1e-12)
    return {
        "loss_relative_gap": loss_gap,
        "grad_relative_gap": grad_gap,
        "tolerance": float(tolerance),
        "passes": bool(loss_gap <= tolerance and grad_gap <= tolerance),
    }


# -------------------------------------------------------------------------------------- driver


def _objective_config(config, objective: str, **overrides):
    """A read-only config view selecting one objective (the exp_02 probe's arm-view pattern)."""
    return exp03.SimpleNamespace(**{**vars(config), "exp03_objective": objective, **overrides})


def build_probe_state(config, trainer, pipeline, mesh, *, restore: bool):
    """The training state, built exactly as ``start_training`` builds it; restored only if asked."""
    context_table = trainer._build_context_table(pipeline, mesh)
    for attr in ("vae", "vae_cache", "text_encoder", "tokenizer"):
        if hasattr(pipeline, attr):
            delattr(pipeline, attr)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        graphdef, params, rest = nnx.split(pipeline.transformer, nnx.Param, ...)
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
    start_step = 0
    if restore:
        ckpt_dir = config.checkpoint_dir
        manager = trainer._build_checkpoint_manager(ckpt_dir)
        state, start_step = trainer._maybe_restore(manager, state)
        if start_step == 0:
            raise ValueError(
                f"S1.5's checkpoint state found nothing to restore in {ckpt_dir!r}; the Tier-1 half of this probe is "
                f"about the step-10,000 checkpoint, and silently probing the init instead would answer the wrong "
                f"question twice."
            )
    return state, state_shardings, start_step


def state_report(state, batches, rng, config, scheduler, *, state_label: str, checkpoint_step: int) -> dict:
    """Every S1.5 measurement at ONE state (no updates; the caller pins that)."""
    reports = []
    for index, batch in enumerate(batches):
        step_rng = jax.random.fold_in(rng, index)
        reports.append(
            exp03.exp03_frozen_replay(
                state, batch, step_rng, config, scheduler, global_step=jnp.asarray(index, jnp.int32)
            )
        )
    per_objective = {}
    for key in sorted({name for report in reports for name in report}):
        values = [report[key] for report in reports if key in report]
        per_objective[key] = {"mean": statistics.fmean(values), "min": min(values), "max": max(values)}

    # Support-gradient variance: the SAME batch under M different support draws is the support term.
    variance = {}
    for objective in S1_5_OBJECTIVES:
        view = _objective_config(config, objective)
        loss_fn = _denoising_loss if objective == "control" else exp03.EXP03_LOSSES[objective]
        gradients = []
        for index, batch in enumerate(batches):
            row = []
            for draw in range(S1_5_SUPPORT_DRAWS):
                # A DIFFERENT global step per draw is what changes the support: the aux keys are
                # folded on it, so the batch is held fixed while the draw varies.
                global_step = jnp.asarray(index * S1_5_SUPPORT_DRAWS + draw, jnp.int32)
                kwargs = {} if objective == "control" else {"global_step": global_step}
                row.append(
                    jax.grad(
                        lambda params, fn=loss_fn, b=batch, kw=kwargs: fn(
                            params, state, b, jax.random.fold_in(rng, index), view, scheduler, **kw
                        )[0]
                    )(state.params)
                )
            gradients.append(row)
        variance[objective] = variance_decomposition(gradients)

    return {
        "state": state_label,
        "checkpoint_step": int(checkpoint_step),
        "num_batches": len(batches),
        "support_draws": S1_5_SUPPORT_DRAWS,
        "per_objective": per_objective,
        "support_variance": variance,
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
        "objectives": list(S1_5_OBJECTIVES),
        "parity_tolerance": S1_5_PARITY_TOLERANCE,
        "report": report,
    }


def write_s1_5_artifact(path: str, payload: dict) -> None:
    """Immutable write, reusing the eval path's compare-or-refuse writer."""
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
            f"data_var={stats['data_variance']:.6g} support_fraction={stats['support_fraction']:.3f}"
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

    payloads = {}
    for state_label in S1_5_STATES:
        pipeline = trainer._load_wan_pipeline()
        mesh = pipeline.mesh
        state, _, checkpoint_step = build_probe_state(
            config, trainer, pipeline, mesh, restore=state_label == "checkpoint"
        )
        before = params_fingerprint(state.params)
        iterator = trainer._load_dataset(mesh, is_training=True, seed=config.seed)
        batches = [gen.load_next_batch(iterator, None, config) for _ in range(S1_5_NUM_BATCHES)]
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            report = state_report(
                state,
                batches,
                jax.random.key(config.seed + 1),
                config,
                scheduler,
                state_label=state_label,
                checkpoint_step=checkpoint_step,
            )
        assert_no_update(before, params_fingerprint(state.params))
        if jax.process_index() != 0:
            continue
        payload = s1_5_artifact(config, state_label=state_label, checkpoint_step=checkpoint_step, report=report)
        path = s1_5_output_path(config, state_label=state_label, checkpoint_step=checkpoint_step)
        write_s1_5_artifact(path, payload)
        log_summary(payload)
        max_logging.log(f"[exp03_s1_5] wrote {path}")
        payloads[state_label] = payload
        # Mechanism-B baseline at this state, under its own canonical path rules.
        trace.run_trace(config)
    return payloads


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
