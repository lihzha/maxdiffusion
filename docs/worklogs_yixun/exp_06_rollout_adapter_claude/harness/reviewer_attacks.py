"""The reviewer's five EXECUTED attacks, re-runnable as the round's acceptance criteria.

A-B1(a) module issue token · A-B1(b) public digest override · A-B2 unrestricted batch callback ·
B-1 terminal-verdict resume · B-2 selection artifact after a crash in the write window.
"""
import dataclasses, hashlib, json, pathlib, tempfile

import jax.numpy as jnp

from maxdiffusion import pos_rollout_dev_instrument as instrument
from maxdiffusion import pos_rollout_loop as loop

MD = pathlib.Path(instrument.__file__).resolve().parents[2] / "docs/worklogs_yixun/exp_04_null_adapter_claude/j0_manifests"
DEV, TEST = str(MD / "dev64.json"), str(MD / "test64.json")
TEST_ROW = json.loads(pathlib.Path(TEST).read_text())["rows"][0]
SHAPE = (4, 3, 4, 6)


def attack_a_b1a():
    token = getattr(instrument, "_ISSUE_TOKEN", None)
    if token is None:
        return "REFUSED: there is no _ISSUE_TOKEN module attribute to hand back"
    cohort = instrument.DevCohort(token, cohort="dev64", rows=[{**TEST_ROW, "split": "dev64"}],
                                 manifest_sha256="0" * 64, manifest_path=DEV)
    drawn = cohort.draw(TEST_ROW["name"], num_steps=25, k_b=2, example_shape=SHAPE)
    return f"SUCCEEDED: drew for {TEST_ROW['name']} (support {int(drawn.support_start)})"


def attack_a_b1b():
    rows = [{**TEST_ROW, "split": "dev64"}] + [dict(r) for r in json.loads(pathlib.Path(DEV).read_text())["rows"][1:]]
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "forged.json"
        path.write_text(json.dumps({"schema_version": 1, "cohort": "dev64", "rows": rows}))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            cohort = instrument.load_dev_cohort(str(path), expected_sha256=digest)
        except TypeError as error:
            return f"REFUSED (no override to pass): {error}"
        except ValueError as error:
            return f"REFUSED: {error}"
        drawn = cohort.draw(TEST_ROW["name"], num_steps=25, k_b=2, example_shape=SHAPE)
        return f"SUCCEEDED: cohort {cohort.cohort} row0 {cohort.names[0]} support {int(drawn.support_start)}"


def attack_a_b2():
    class Ctx:
        num_steps, k_b = 25, 2

    test_batch = {"z_video": jnp.ones((1, *SHAPE), jnp.float32) * 999.0}
    cohort = instrument.load_dev_cohort(DEV)
    try:
        out = instrument.score_dev_cohort(cohort, lambda p, b, c, *, draws: (jnp.asarray(1.0), {}),
                                          lambda row: test_batch, params=jnp.asarray(1.0), context=Ctx(),
                                          example_shape=SHAPE)
    except (TypeError, ValueError) as error:
        return f"REFUSED: {error}"
    return f"SUCCEEDED: metric {out['metric']} stamped cohort={out['cohort']} sha={out['manifest_sha256'][:12]}"


def _state():
    return loop.RolloutTrainState(params={"w": jnp.zeros((2,), jnp.float32)},
                                  opt_state={"mu": jnp.zeros((2,), jnp.float32)}, step=0)


def _schedule(**over):
    base = dict(max_train_steps=6, eval_every=2, logical_batch=4, microbatch=2, seed=0, arm="rollout", k_b=2,
                num_steps=25)
    base.update(over)
    return loop.LoopSchedule(**base)


def _stream(schedule):
    import itertools

    def factory(seed):
        events.append(("iterator", int(seed)))
        return itertools.repeat({"z_video": jnp.zeros((schedule.logical_batch, *SHAPE), jnp.float32),
                                 "actions": jnp.zeros((schedule.logical_batch, 4, 7), jnp.float32)})

    return factory


events = []


def attack_b1(tmp):
    """Restore a TERMINAL history and see whether another optimizer step runs."""
    global events
    directory = str(pathlib.Path(tmp) / "b1")
    schedule = _schedule(max_train_steps=4, eval_every=1)
    values = iter([0.1, 1.0, 1.0, 1.0])
    manager = loop.build_checkpoint_manager(directory)
    events = []
    loop.run_loop(_state(), schedule, batches=_stream(schedule),
                  update_fn=lambda s, b, d, sc, gs: (dataclasses.replace(s, params=s.params), 1.0 / gs),
                  dev_metric_fn=lambda s, step: next(values), manager=manager)
    assert loop.stop_verdict(loop.restore_eval_history(loop.build_checkpoint_manager(directory))).stop
    events = []

    def update(state, batch_parts, draw_parts, sched, global_step):
        events.append(("update", global_step))
        return dataclasses.replace(state, params=state.params), 1.0

    report = loop.run_loop(_state(), _schedule(max_train_steps=30_000, eval_every=1),
                           batches=_stream(_schedule()), update_fn=update, dev_metric_fn=lambda s, step: 0.01,
                           manager=loop.build_checkpoint_manager(directory))
    return (f"steps_run={report.steps_run} events={events}"
            + ("  -> SUCCEEDED (a terminal reopen trained on)" if events else "  -> REFUSED (no step, no iterator)"))


def attack_b2(tmp):
    """Crash between the resume save and the selection update; see what the sibling ships."""
    directory = str(pathlib.Path(tmp) / "b2")
    schedule = _schedule(max_train_steps=2, eval_every=2)
    values = iter([0.5])
    loop.run_loop(_state(), schedule, batches=_stream(schedule),
                  update_fn=lambda s, b, d, sc, gs: (dataclasses.replace(s, params=s.params), 1.0),
                  dev_metric_fn=lambda s, step: next(values), manager=loop.build_checkpoint_manager(directory))
    selection = loop.build_selection_manager(directory)
    values = iter([0.9])
    schedule = _schedule(max_train_steps=4, eval_every=2)
    report = loop.run_loop(_state(), schedule, batches=_stream(schedule),
                           update_fn=lambda s, b, d, sc, gs: (dataclasses.replace(s, params=s.params), 1.0),
                           dev_metric_fn=lambda s, step: next(values),
                           manager=loop.build_checkpoint_manager(directory), selection_manager=selection)
    selection.wait_until_finished()
    shipped = selection.latest_step()
    return (f"history_best={report.retained_step} shipped_selection={shipped}"
            + ("  -> REFUSED (reconciled)" if shipped == report.retained_step else "  -> SUCCEEDED (stale/wrong)"))


# =================================================================================================
# T5a `eval-anchor` — the attacks I would run as the reviewer (no reviewer was available; these are
# self-generated). Appended per the standing rule: extend this harness, never start a fresh one.
# =================================================================================================


def _anchor_env():
    from maxdiffusion import eval_wan_pos_rollout as anchor

    return anchor, str(MD / "test64.json"), str(MD / "dev64.json")


def attack_t5a_restore_falls_back():
    """Can an evaluation be pointed at the resume tree's latest instead of the selection artifact?"""
    import dataclasses as dc
    import tempfile as tf

    from maxdiffusion import pos_rollout_loop as pl

    anchor, _, _ = _anchor_env()
    with tf.TemporaryDirectory() as tmp:
        root = str(pathlib.Path(tmp) / "run")
        resume = pl.build_checkpoint_manager(root)
        pl.save_checkpoint(resume, dc.replace(_state(), step=9), dev_metric=0.9, history=(), arm="rollout", k_b=2)
        try:
            state, _ = anchor.restore_selected_checkpoint(
                root, _state(), expected_step=9, expected_dev_metric=0.9, expected_arm="rollout", expected_k_b=2
            )
        except ValueError as error:
            return f"REFUSED: {str(error).splitlines()[0][:110]}"
        return f"SUCCEEDED: evaluated the unselected resume checkpoint at step {int(state.step)}"


def attack_t5a_widen_the_anchor():
    """Can a caller widen the band, swap the record, or pass a measurement that misses it?"""
    anchor, _, _ = _anchor_env()
    attempts = []
    try:
        anchor.reproduce_anchor({"mean_ssim": 0.20, "mean_latent_mse": 1.496, "mean_pixel_mse": 0.0983,
                                 "num_samples": 4, "checkpoint_step": 30000, "sample_names": list("abcd")},
                                tolerance=0.5)
        attempts.append("tolerance override ACCEPTED")
    except TypeError:
        attempts.append("no tolerance argument exists")
    verdict = anchor.reproduce_anchor({"mean_ssim": 0.20, "mean_latent_mse": 1.496, "mean_pixel_mse": 0.0983,
                                       "num_samples": 4, "checkpoint_step": 30000, "sample_names": list("abcd")})
    attempts.append(f"ssim 0.20 reproduced={verdict.reproduced}")
    return "REFUSED: " + "; ".join(attempts) if not verdict.reproduced else "SUCCEEDED: " + "; ".join(attempts)


def attack_t5a_test_into_the_anchor():
    """Can a TEST-64 example be scored into the anchor summary?"""
    anchor, test_manifest, _ = _anchor_env()
    intruder = json.loads(pathlib.Path(test_manifest).read_text())["rows"][0]["name"]
    rows = [{"name": n, "latent_mse": 1.0, "pixel_mse": 0.1, "ssim_avg": 0.3} for n in ("a", intruder)]
    try:
        anchor.summarize_samples(rows, checkpoint_step=30000, test_manifest_path=test_manifest)
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return f"SUCCEEDED: {intruder} scored into the anchor summary"


def attack_t5a_rederive_the_benchmark(tmp):
    """Can the frozen benchmark row be silently re-derived with different numbers?"""
    from maxdiffusion import pos_rollout_dev_instrument as instrument

    anchor, _, dev = _anchor_env()
    cohort = instrument.load_dev_cohort(dev)
    path = str(pathlib.Path(tmp) / "bench.json")
    common = dict(checkpoint={"step": 30000}, code_sha="a" * 40, model_revision="rev@" + "b" * 40)
    anchor.freeze_benchmark_row(path, cohort=cohort, per_example={n: 0.25 for n in cohort.names}, **common)
    try:
        anchor.freeze_benchmark_row(path, cohort=cohort, per_example={n: 0.95 for n in cohort.names}, **common)
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return "SUCCEEDED: the frozen baseline was rewritten with better numbers"


def attack_t5a_forge_a_dev_cohort(tmp):
    """Can a TEST row be frozen into the DEV-64 benchmark by handing over a look-alike cohort?"""
    anchor, test_manifest, _ = _anchor_env()
    row = json.loads(pathlib.Path(test_manifest).read_text())["rows"][0]

    class _LookAlike:
        cohort, manifest_sha256, manifest_path = "dev64", "0" * 64, "dev64.json"
        names = (row["name"],)

        def __len__(self):
            return 1

    try:
        anchor.freeze_benchmark_row(str(pathlib.Path(tmp) / "forged.json"), cohort=_LookAlike(),
                                    per_example={row["name"]: 0.9}, checkpoint={"step": 30000},
                                    code_sha="a" * 40, model_revision="rev")
    except (TypeError, ValueError) as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return "SUCCEEDED: a TEST row was frozen as the DEV-64 benchmark"


# =================================================================================================
# T5b `eval-gates` — five more self-generated attacks on the two gates that decide the experiment.
# =================================================================================================


def _gate_env():
    from maxdiffusion import pos_rollout_dev_instrument as instrument
    from maxdiffusion import pos_rollout_gates as g

    cohort = instrument.load_dev_cohort(str(MD / "dev64.json"))
    names = list(cohort.names)
    return g, cohort, names


def _tbl(names, ssim):
    return {n: {"ssim": float(ssim), "mse": 1.0} for n in names}


def attack_t5b_lower_the_bar():
    """Can the +0.05 margin or the CI condition be relaxed from the outside?"""
    g, cohort, names = _gate_env()
    notes = []
    try:
        g.primary_gate(rollout=_tbl(names, 0.34), control=_tbl(names, 0.30), cohort=cohort, margin=0.01)
        notes.append("margin override ACCEPTED")
    except TypeError:
        notes.append("no margin argument exists")
    verdict = g.primary_gate(rollout=_tbl(names, 0.34), control=_tbl(names, 0.30), cohort=cohort)
    notes.append(f"+0.04 with a clean CI passed={verdict.passed}")
    return ("SUCCEEDED: " if verdict.passed else "REFUSED: ") + "; ".join(notes)


def attack_t5b_score_test_first():
    """Can TEST be scored without a passing DEV gate?"""
    g, cohort, names = _gate_env()
    failing = g.dev_certificate(
        g.primary_gate(rollout=_tbl(names, 0.30), control=_tbl(names, 0.30), cohort=cohort), cohort, num_steps=25
    )
    for label, cert in (("failing certificate", failing), ("hand-written pass", {"passed": True})):
        try:
            g.confirm_on_test(cert, test_manifest_path=str(MD / "test64.json"), rollout={}, control={})
        except ValueError as error:
            last = f"REFUSED ({label}): {str(error).splitlines()[0][:80]}"
        else:
            return f"SUCCEEDED: TEST scored with a {label}"
    return last


def attack_t5b_forge_the_derangement():
    """Can a wrong-action assignment secretly hand examples their own actions back?"""
    g, cohort, names = _gate_env()
    blobs = {n: f"a{i}".encode() for i, n in enumerate(names)}
    good = g.cohort_derangement(names, cohort="dev64", action_bytes=blobs)
    sneaky = {**good, names[0]: names[0]}
    try:
        g.action_use_gate(true_table=_tbl(names, 0.36), wrong_table=_tbl(names, 0.30), cohort=cohort,
                          derangement=sneaky)
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return "SUCCEEDED: an example was scored against its own actions as the wrong-action row"


def attack_t5b_swap_arm_and_control():
    """Can the claim and its control be swapped by position?"""
    g, cohort, names = _gate_env()
    try:
        g.primary_gate(_tbl(names, 0.30), _tbl(names, 0.36), cohort)
    except TypeError as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return "SUCCEEDED: positional arguments let the control be reported as the arm"


def attack_t5b_drop_the_control_battery():
    """Can 'the adapter uses its actions' be published without matched-C0's own battery?"""
    g, cohort, names = _gate_env()
    blobs = {n: f"a{i}".encode() for i, n in enumerate(names)}
    mapping = g.cohort_derangement(names, cohort="dev64", action_bytes=blobs)
    try:
        g.action_use_report(cohort, derangement=mapping, true_table=_tbl(names, 0.36),
                            wrong_table=_tbl(names, 0.30), zero_table=_tbl(names, 0.20),
                            adapter_disabled_table=_tbl(names, 0.10), control_tables={})
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return "SUCCEEDED: the action-use finding was published without its comparison"


# =================================================================================================
# T7 `fit-probe-mode` — five more self-generated attacks on the M1 authorization contract.
# =================================================================================================


def _probe_env():
    from maxdiffusion import pos_rollout_fit_probe as fp

    return fp


_HARNESS_MODEL = pathlib.Path(tempfile.mkdtemp(prefix="exp06_attacks_model_")) / "snapshot"
_HARNESS_MODEL.mkdir(parents=True)
(_HARNESS_MODEL / "weights.safetensors").write_bytes(b"w" * 128)


def _pos_config(**over):
    """Round F1 (LS-10): provenance is CONTENT-BOUND, so a config must name a model that exists.
    `derive_model_revision` now fails closed on an unresolvable name -- pointing this helper at a
    real directory keeps every attack below testing what it was written to test rather than dying on
    the new refusal."""
    import yaml

    values = yaml.safe_load(pathlib.Path("src/maxdiffusion/configs/base_wan_5b_pos_rollout.yml").read_text())
    values["pretrained_model_name_or_path"] = str(_HARNESS_MODEL)
    values.update(over)

    class _C:
        def __init__(self, m):
            self.__dict__.update(m)

        def get_keys(self):
            return dict(self.__dict__)

    return _C(values)


class _Dev:
    device_kind = "v6e"


def _ctx(fp, config=None, **over):
    """A DERIVED context (pass 3). Provenance can no longer arrive as a publication argument."""
    context = fp.derive_probe_context(config or _pos_config(), devices=[_Dev() for _ in range(8)], environ={})
    return dataclasses.replace(context, **over) if over else context


def _fit(fp, context=None, *, arm="rollout", microbatch=32, k_b=2, **over):
    values = dict(cell=fp.FitCell(arm, microbatch, k_b), context_digest=(context or _ctx(fp)).digest(),
                  compile_seconds=480.0, step_seconds=3.5, eval_seconds=600.0, checkpoint_seconds=90.0,
                  peak_bytes=20 * 1024**3, capacity_bytes=32 * 1024**3, reservation_failures=0,
                  peak_source=fp.PEAK_SOURCE_RUNTIME_RESET)
    values.update(over)
    return fp.CellMeasurement(**values)


def _auth(fp, tmp, measurements, name="auth.json", context=None):
    context = context or _ctx(fp)
    evidence = fp.build_evidence(context, measurements, max_train_steps=10_000, eval_every=1_000,
                                 checkpoint_every=1_000)
    return fp.publish_authorization(str(pathlib.Path(tmp) / name), evidence)


def attack_t7_run_an_unmeasured_cell(tmp):
    """Can a training run reach an (arm, microbatch, k) cell M1 never measured?"""
    from maxdiffusion.trainers.wan_pos_rollout_trainer import WanPosRolloutTrainer

    fp = _probe_env()
    path = str(pathlib.Path(tmp) / "t7a.json")
    config = _pos_config(pos_fit_authorization=path, pos_rollout_k=4,
                         pos_recipe_lock=str(pathlib.Path(tmp) / "t7a_lock.json"),
                         pos_resume_parent=str(pathlib.Path(tmp) / "t7a_attempts"),
                         checkpoint_dir=str(pathlib.Path(tmp) / "t7a_attempts/att-X/checkpoints"))
    running = fp.derive_probe_context(config)
    _auth(fp, tmp, [_fit(fp, running)], name="t7a.json", context=running)
    try:
        WanPosRolloutTrainer(config).start_training()
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    except NotImplementedError:
        return "SUCCEEDED: an unmeasured cell reached the model wiring"
    return "SUCCEEDED: an unmeasured cell was accepted"


def attack_t7_forge_an_authorization(tmp):
    """Can a hand-written mapping authorize a cell?"""
    fp = _probe_env()
    try:
        fp.assert_cell_authorized({"authorized_cells": [{"arm": "rollout", "microbatch": 64, "k_b": 4}]},
                                  fp.FitCell("rollout", 64, 4), context=_ctx(fp))
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return "SUCCEEDED: a hand-written mapping authorized an unmeasured cell"


def attack_t7_edit_a_published_authorization(tmp):
    """Can a published authorization be edited to add a cell?"""
    fp = _probe_env()
    path = pathlib.Path(tmp) / "t7c.json"
    _auth(fp, tmp, [_fit(fp)], name="t7c.json")
    stored = json.loads(path.read_text())
    stored["payload"]["authorized_cells"].append({"arm": "rollout", "microbatch": 64, "k_b": 4})
    path.write_text(json.dumps(stored))
    try:
        fp.load_authorization(str(path))
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return "SUCCEEDED: an edited authorization was loaded"


def attack_t7_authorize_a_cell_that_missed(tmp):
    """Can a cell that was measured and MISSED the headroom rule be run anyway?"""
    fp = _probe_env()
    context = _ctx(fp)
    published = _auth(fp, tmp, [_fit(fp, context, microbatch=64, peak_bytes=31 * 1024**3)], name="t7d.json",
                      context=context)
    try:
        fp.assert_cell_authorized(published, fp.FitCell("rollout", 64, 2), context=context)
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return "SUCCEEDED: a cell that missed fit at 96.9% was authorized"


def attack_t7_project_what_was_not_measured(tmp):
    """Can a wall-clock be projected without measured eval/checkpoint overhead, or for a misfit cell?"""
    fp = _probe_env()
    notes = []
    try:
        fp.project_wall_clock(_fit(fp), max_train_steps=10_000, eval_every=1_000)
        notes.append("cadences defaulted ACCEPTED")
    except TypeError:
        notes.append("every cadence is a required argument")
    try:
        fp.project_wall_clock(_fit(fp), max_train_steps=10_000, eval_every=1_000, checkpoint_every=1_000,
                              eval_seconds=1.0, checkpoint_seconds=1.0)
        notes.append("caller-supplied overheads ACCEPTED")
    except TypeError:
        notes.append("there is no overhead argument to supply")
    try:
        fp.project_wall_clock(_fit(fp, peak_bytes=31 * 1024**3), max_train_steps=10_000, eval_every=1_000,
                              checkpoint_every=1_000)
        return "SUCCEEDED: " + "; ".join(notes) + "; a misfit cell was projected"
    except ValueError:
        notes.append("a misfit cell cannot be projected")
    return "REFUSED: " + "; ".join(notes)


# =================================================================================================
# Review pass 3 — the reviewer's EXECUTED attacks on the LAUNCH SURFACE, re-run against the fixes.
# T6-1/T6-2/T6-3 + T7-1/T7-2/T7-3/T7-4 and the projection MAJOR. Appended per the standing rule:
# extend this harness, never start a fresh one.
# =================================================================================================

_LAUNCH = pathlib.Path(instrument.__file__).resolve().parents[2] / "bash_scripts"
_COMMIT = "a1b2c3d4" * 5
_PY_SHIM = """#!/bin/sh
{ printf 'PYTHON'; for arg in "$@"; do printf '\\037%s' "$arg"; done; printf '\\n'; } >> "$SHIM_RECORD"
exit 0
"""
_PREFETCH_STUB = "#!/bin/sh\necho '[prefetch stub]'\nexit 0\n"


def _launch(tmp, script, **env_over):
    """Execute a launcher under real bash in a curated-PATH sandbox and return its recorded argv."""
    import os
    import shutil
    import stat
    import subprocess

    root = pathlib.Path(tmp) / f"launch_{script}_{len(list(pathlib.Path(tmp).glob('launch_*')))}"
    (root / "bash_scripts").mkdir(parents=True)
    (root / "bin").mkdir()
    (root / "home").mkdir()
    shutil.copy2(_LAUNCH / script, root / "bash_scripts" / script)
    os.symlink(_LAUNCH.parent / "src", root / "src")
    for name, body in (("python", _PY_SHIM), ("prefetch_hf_snapshot.sh", _PREFETCH_STUB)):
        target = (root / "bin" / name) if name == "python" else (root / "bash_scripts" / name)
        target.write_text(body)
        target.chmod(target.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    for tool in ("bash", "date", "mkdir", "tee", "grep", "cat", "git", "printf", "env", "sed"):
        found = shutil.which(tool)
        if found:
            os.symlink(found, root / "bin" / tool)
    # W2b made the topology a REQUIRED declaration of the training launcher, so every sandbox
    # declares one. A caller that wants the missing-declaration case overrides it explicitly.
    env = {"PATH": str(root / "bin"), "HOME": str(root / "home"), "POS_DEVICE_COUNT": "8",
           "SHIM_RECORD": str(root / "record.txt"), "COMMIT": _COMMIT}
    env.update({k: str(v) for k, v in env_over.items() if not k.startswith("_")})
    proc = subprocess.run(["/bin/bash", f"bash_scripts/{script}"], cwd=root, env=env,
                          capture_output=True, text=True, timeout=300)
    argv = []
    if (root / "record.txt").exists():
        for line in (root / "record.txt").read_text().splitlines():
            parts = line.split("\x1f")
            if parts[0] == "PYTHON" and len(parts) > 1 and parts[1] != "-":
                argv = parts[1:]
    # A launcher that never reached its entrypoint emits NOTHING, and an attack that compares two
    # nothings reads as a success. That is the harness lying about production, so it is refused here:
    # a probe must be told its launch did not happen rather than shown two equal Nones. (Found when
    # W2b made a new declaration required and P3-1 flipped to SUCCEEDED with `both arms ... None`.)
    if not argv and "_expect_failure" not in env_over:
        raise RuntimeError(
            f"{script} exited {proc.returncode} without reaching its entrypoint, so this probe measured "
            f"nothing: {(proc.stdout or proc.stderr).splitlines()[-1][:160] if (proc.stdout or proc.stderr) else ''}"
        )
    return proc, dict(w.split("=", 1) for w in argv if "=" in w and not w.startswith("--"))


def attack_p3_arms_share_checkpoint_state(tmp):
    """T6-1: running R-B then matched-C0 made C0 restore R-B's parameters, optimizer, step, history."""
    common = dict(RUN_NAME="m3", ATTEMPT="att-X", OUTPUT_DIR="gs://b/p",
                  POS_FIT_AUTHORIZATION="gs://b/m1.json")
    _, left = _launch(tmp, "train_wan_pos_rollout.sh", POS_ROLLOUT_ARM="rollout", **common)
    _, right = _launch(tmp, "train_wan_pos_rollout.sh", POS_ROLLOUT_ARM="one_step", **common)
    if left.get("checkpoint_dir") == right.get("checkpoint_dir"):
        return f"SUCCEEDED: both arms write and restore {left.get('checkpoint_dir')}"
    return (f"REFUSED: rollout -> {left['checkpoint_dir'].split('/p/')[-1]} vs "
            f"one_step -> {right['checkpoint_dir'].split('/p/')[-1]}")


def attack_p3_divergent_second_arm(tmp):
    """T6-1's other half: two submissions a day apart, differing in a seed. No shell can see that."""
    from maxdiffusion.trainers.wan_pos_rollout_trainer import publish_recipe_lock

    path = str(pathlib.Path(tmp) / "p3_lock.json")
    publish_recipe_lock(path, _pos_config(pos_rollout_arm="rollout"), arm="rollout")
    try:
        publish_recipe_lock(path, _pos_config(pos_rollout_arm="one_step", seed=7, learning_rate=1e-3),
                            arm="one_step")
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:130]}"
    return "SUCCEEDED: matched-C0 ran at another seed and learning rate and nothing noticed"


def attack_p3_confirm_without_a_certificate(tmp):
    """T6-2: `confirm` scored TEST with no anchor certificate, no benchmark row, no DEV certificate."""
    # `_expect_failure`: this probe's whole point is that the launcher does NOT reach its entrypoint,
    # so the "measured nothing" guard must not answer for it -- the probe's own check below does.
    proc, overrides = _launch(tmp, "eval_wan_pos_rollout.sh", POS_EVAL_PHASE="confirm", _expect_failure=True)
    if proc.returncode == 0:
        return f"SUCCEEDED: confirm reached the evaluator carrying {sorted(overrides)}"
    return f"REFUSED: {[l for l in proc.stdout.splitlines() if l.startswith('FATAL')][0][:120]}"


def attack_p3_flatten_the_attempt_scoping(tmp):
    """T6-3: a caller-supplied ARTIFACT_ROOT removed phase/attempt scoping entirely, in BOTH launchers."""
    notes = []
    for script, extra in (("train_wan_pos_rollout.sh", {"POS_FIT_AUTHORIZATION": "gs://b/m1.json"}),
                          ("eval_wan_pos_rollout.sh", {})):
        _, overrides = _launch(tmp, script, ARTIFACT_ROOT="gs://bucket/flat", CHECKPOINT_DIR="gs://bucket/flat",
                               OUTPUT_DIR="gs://b/p", RUN_NAME="m3", ATTEMPT="att-X", **extra)
        flattened = overrides.get("base_output_directory") == "gs://bucket/flat"
        notes.append(f"{script.split('_')[0]}={'FLAT' if flattened else 'derived'}")
    return ("SUCCEEDED: " if "FLAT" in " ".join(notes) else "REFUSED: ") + ", ".join(notes)


def attack_p3_adopt_an_incomplete_or_foreign_publication(tmp):
    """T6-3: 'select only the latest COMPLETE checkpoint whose recorded SHA matches the running code'."""
    from maxdiffusion.trainers import wan_pos_rollout_trainer as tr

    parent = str(pathlib.Path(tmp) / "p3_attempts")
    tr.publish_attempt(parent, attempt="att-mine", arm="rollout", code_sha="a" * 40,
                       context_digest="d" * 64, step=1000, checkpoint_dir=f"{parent}/att-mine/checkpoints")
    tr.publish_attempt(parent, attempt="att-foreign-sha", arm="rollout", code_sha="b" * 40,
                       context_digest="d" * 64, step=9000, checkpoint_dir=f"{parent}/att-foreign-sha/checkpoints")
    tr.publish_attempt(parent, attempt="att-other-arm", arm="one_step", code_sha="a" * 40,
                       context_digest="d" * 64, step=9000, checkpoint_dir=f"{parent}/att-other-arm/checkpoints")
    (pathlib.Path(parent) / "att-crashed" / "checkpoints").mkdir(parents=True)
    chosen = tr.select_resume_publication(parent, code_sha="a" * 40, arm="rollout")
    if chosen["attempt"] != "att-mine":
        return f"SUCCEEDED: adopted {chosen['attempt']} at step {chosen['step']}"
    return f"REFUSED: adopted only att-mine (step 1000); the 9000-step foreign-SHA and other-arm trees were skipped"


def attack_p3_authorization_from_another_program(tmp):
    """T7-1, verbatim: an authorization carrying a wrong SHA, a foreign model and the wrong device."""
    fp = _probe_env()
    here = _ctx(fp)
    elsewhere = _ctx(fp, code_sha="0" * 40, model_revision="Some-Other/Model", device_kind="v5p", device_count=256)
    published = _auth(fp, tmp, [_fit(fp, elsewhere)], name="p3_foreign.json", context=elsewhere)
    try:
        fp.assert_cell_authorized(published, fp.FitCell("rollout", 32, 2), context=here)
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:130]}"
    return "SUCCEEDED: a foreign SHA / model / device authorized this program"


def attack_p3_one_arm_authorizes_the_other(tmp):
    """T7-2: `FitCell` was `(microbatch, k)`, so a matched-C0 measurement authorized R-B."""
    fp = _probe_env()
    context = _ctx(fp)
    published = _auth(fp, tmp, [_fit(fp, context, arm="one_step")], name="p3_arm.json", context=context)
    try:
        fp.assert_cell_authorized(published, fp.FitCell("rollout", 32, 2), context=context)
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:130]}"
    return "SUCCEEDED: a matched-C0 HBM measurement authorized R-B's different backward graph"


def attack_p3_contradictory_duplicate_trials(tmp):
    """T7-3: the same cell published once fitting and once at 96.9% with a reservation failure."""
    fp = _probe_env()
    context = _ctx(fp)
    published = _auth(fp, tmp, [_fit(fp, context),
                                _fit(fp, context, peak_bytes=int(32 * 1024**3 * 0.969), reservation_failures=1)],
                      name="p3_dup.json", context=context)
    try:
        fp.assert_cell_authorized(published, fp.FitCell("rollout", 32, 2), context=context)
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:130]}"
    return f"SUCCEEDED: authorized={published['authorized_cells']} refused={published['refused_cells']}"


def attack_p3_m1_cannot_be_run(tmp):
    """T7-4: `run_fit_probe` walked no ladder and the launcher had no probe mode -- M1 was unrunnable."""
    fp = _probe_env()
    proc, overrides = _launch(tmp, "train_wan_pos_rollout.sh", POS_JOB_MODE="fit_probe", RUN_NAME="m1",
                              ATTEMPT="att-X", OUTPUT_DIR="gs://b/p")
    path = str(pathlib.Path(tmp) / "p3_m1.json")
    config = _pos_config(pos_fit_authorization=path)
    seen = []

    def measurer(*, cell, context, config):
        seen.append(cell)
        return _fit(fp, context, arm=cell.arm, microbatch=cell.microbatch, k_b=cell.k_b)

    try:
        published = fp.run_fit_probe(config, measurer=measurer, devices=[_Dev()], trials=1)
    except NotImplementedError as error:
        return f"SUCCEEDED (still unrunnable): {str(error)[:100]}"
    ran = proc.returncode == 0 and overrides.get("pos_fit_authorization", "").endswith("fit_authorization.json")
    return (f"REFUSED (M1 runs): launcher probe mode={'yes' if ran else 'no'}, ladder walked {len(seen)} cells, "
            f"published {len(published['authorized_cells'])} authorized cells")


# =================================================================================================
# Review pass 1 — the reviewer's EXECUTED attacks, re-run against the fixes.
# =================================================================================================


def attack_p1_echoed_identity_decoder():
    """The decoder echoing genuine DEV names + declared ordinals while returning 999-filled tensors,
    with a binder echoing the manifest's generation and size. It scored metric 999.0 under cohort
    dev64 with the genuine digest -- "the previous unrestricted callback one layer lower"."""
    from maxdiffusion import pos_rollout_dev_instrument as instrument

    cohort = instrument.load_dev_cohort(str(MD / "dev64.json"))
    forged = lambda shard, wanted: iter(())  # noqa: E731
    notes = []
    try:
        instrument.DevBatchReader(cohort, reader=forged, binder=lambda p: {"generation": "x", "size": 0})
        notes.append("DevBatchReader ACCEPTED a decoder")
    except TypeError:
        notes.append("DevBatchReader takes a cohort and nothing else")
    try:
        instrument.score_dev_cohort(cohort, lambda *a, **k: (1.0, {}), forged, params=1.0,
                                    context=type("C", (), {"num_steps": 25, "k_b": 2})(), example_shape=(4, 3, 4, 6))
        return "SUCCEEDED: " + "; ".join(notes) + "; scoring accepted a decoder"
    except TypeError:
        notes.append("score_dev_cohort has no batch source parameter")
    return "REFUSED: " + "; ".join(notes)


def attack_p1_poison_selection_with_nan():
    """A single NaN early in training became an unbeatable running best that no later finite value
    could displace, while preserve_selection still replaced the sibling."""
    from maxdiffusion import pos_rollout_loop as pl

    history = [pl.EvalRecord(step=1000, dev_metric=float("nan"), train_metric=1.0),
               pl.EvalRecord(step=2000, dev_metric=0.1, train_metric=0.9)]
    try:
        verdict = pl.stop_verdict(history)
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return f"SUCCEEDED: best_step={verdict.best_step} best_value={verdict.best_value}"


def attack_p1_write_a_gs_artifact_locally():
    """pathlib silently turns gs://bucket/x into the local path gs:/bucket/x."""
    import pathlib as _p

    from maxdiffusion import pos_rollout_support

    dropped = str(_p.Path("gs://bucket/run/report.json"))
    guarded = pos_rollout_support.is_remote("gs://bucket/run/report.json")
    return (f"REFUSED: pathlib would give {dropped!r}; storage layer treats it as remote={guarded}"
            if guarded else f"SUCCEEDED: {dropped}")


def _report(label, fn, *args):
    """An exception raised BY PRODUCTION is a refusal; the runner must survive it and say so.

    Added when a sibling round hardened `reproduce_anchor` into a TypeError: the attack was refused
    in the strongest possible way (no such call shape exists) and the runner still crashed, hiding
    every later attack. No attack's content changed.
    """
    try:
        print(f"{label}:", fn(*args))
    except Exception as error:  # noqa: BLE001
        print(f"{label}: REFUSED ({type(error).__name__}): {str(error).splitlines()[0][:110]}")



# =================================================================================================
# GROUPS 3-4 -- the reviewer's pass-2 EXECUTED attacks, re-run against the rework. Every one of these
# SUCCEEDED before this round; each must now be REFUSED.
# =================================================================================================


def _fake_environment():
    """In-memory records + an in-memory gs:// filesystem, installed process-wide (no pytest here)."""
    import numpy as np

    from maxdiffusion import eval_wan_pos_rollout as ev
    from maxdiffusion import null_adapter_manifest_io, pos_rollout_support, run_wan_null_inversion
    from maxdiffusion import pos_rollout_dev_instrument as inst

    geometry = {"z_i0": (48, 1, 12, 20), "z_video": (48, 9, 12, 20), "actions": (32, 7)}

    def rows_of(cohort):
        return {str(r["shard_path"]): [] for r in cohort.rows}

    def install(cohort, *, identical=False):
        by_shard = {}
        for row in cohort.rows:
            by_shard.setdefault(str(row["shard_path"]), []).append(dict(row))

        def fill(name):
            if identical:
                return 1.0
            return float(int(hashlib.sha256(str(name).encode()).hexdigest()[:6], 16) % 9973) / 10.0 + 1.0

        def reader(shard_path, wanted):
            for row in by_shard[str(shard_path)]:
                if str(row["name"]) not in set(wanted):
                    continue
                value = fill(row["name"])
                yield (
                    str(row["name"]),
                    int(row["ordinal"]),
                    np.full(geometry["z_i0"], value, np.float32),
                    np.full(geometry["z_video"], value, np.float32),
                    np.full(geometry["actions"], value, np.float32),
                )

        def binder(shard_path):
            row = by_shard[str(shard_path)][0]
            return {"generation": str(row["shard_generation"]), "size": int(row["shard_size"])}

        run_wan_null_inversion._tfrecord_reader = reader
        null_adapter_manifest_io.shard_binding = binder

    blobs = {}

    class _Gfile:
        @staticmethod
        def _remote(path):
            return pos_rollout_support.is_remote(path)

        def exists(self, path):
            return str(path) in blobs if self._remote(path) else pathlib.Path(str(path)).exists()

        def makedirs(self, path):
            if not self._remote(path):
                pathlib.Path(str(path)).mkdir(parents=True, exist_ok=True)

        def GFile(self, path, mode="rb"):  # noqa: N802
            if not self._remote(path):
                return open(str(path), mode)  # noqa: SIM115
            key = str(path)

            class _Handle:
                def read(self):
                    return blobs[key]

                def write(self, payload):
                    blobs[key] = payload

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            return _Handle()

    pos_rollout_support._gfile = lambda path: _Gfile()
    del rows_of
    return ev, inst, install, blobs


def _identity(run=None, step=30000, source="historical"):
    from maxdiffusion import eval_wan_pos_rollout as ev

    name = ev.HISTORICAL_ANCHOR.run_name if run is None else run
    return ev.CheckpointIdentity(run_name=name, step=step, root=f"gs://b/{name}/checkpoints", source=source)


def _rows(names, num_steps=25):
    from maxdiffusion import eval_wan_pos_rollout as ev

    return [
        {
            "name": n,
            "latent_mse": ev.HISTORICAL_ANCHOR.mean_latent_mse,
            "pixel_mse": ev.HISTORICAL_ANCHOR.mean_pixel_mse,
            "ssim_avg": ev.HISTORICAL_ANCHOR.mean_ssim,
            "num_steps": num_steps,
        }
        for n in names
    ]


def _summary(names=None, identity=None, num_steps=25):
    from maxdiffusion import eval_wan_pos_rollout as ev

    return ev.summarize_samples(
        _rows(ev.HISTORICAL_ANCHOR.sample_names if names is None else names, num_steps),
        checkpoint=_identity() if identity is None else identity,
        code_sha="a" * 40,
        model_revision="rev",
        test_manifest_path=TEST,
    )


def attack_g_anchor_foreign_names():
    """The reviewer passed the recorded means with FOUR UNRELATED names; reproduced came back True."""
    from maxdiffusion import eval_wan_pos_rollout as ev

    try:
        verdict = ev.reproduce_anchor(_summary(names=("s0", "s1", "s2", "s3")))
    except (TypeError, ValueError) as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return f"SUCCEEDED: reproduced={verdict.reproduced} on four unrelated samples"


def attack_g_anchor_wrong_order():
    from maxdiffusion import eval_wan_pos_rollout as ev

    names = list(ev.HISTORICAL_ANCHOR.sample_names)
    names[0], names[2] = names[2], names[0]
    try:
        verdict = ev.reproduce_anchor(_summary(names=names))
    except (TypeError, ValueError) as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return f"SUCCEEDED: reproduced={verdict.reproduced} with the samples out of order"


def attack_g_anchor_foreign_checkpoint():
    """The reviewer certified checkpoint {"run_name": "some-other-run", "step": 1}."""
    from maxdiffusion import eval_wan_pos_rollout as ev

    try:
        verdict = ev.reproduce_anchor(_summary(identity=_identity(run="some-other-run", step=1)))
        certificate = ev.anchor_certificate(verdict)
    except (TypeError, ValueError) as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return f"SUCCEEDED: certified {certificate['checkpoint']}"


def attack_g_certify_a_short_rollout():
    """`rollout_prediction(num_steps=1)` executed, and its result was certifiable as num_steps=25."""
    import inspect

    from maxdiffusion import eval_wan_pos_rollout as ev

    if "num_steps" in inspect.signature(ev.rollout_prediction).parameters:
        return "SUCCEEDED: rollout_prediction still takes a caller-supplied horizon"
    try:
        ev.reproduce_anchor(_summary(num_steps=1))
    except (TypeError, ValueError) as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return "SUCCEEDED: a 1-step measurement was certified"


def attack_g_forge_the_certificate_marker():
    """`{"certificate": GATE_CERTIFICATE, "passed": True}` unlocked TEST in the reviewer's probe."""
    from maxdiffusion import eval_wan_pos_rollout as ev
    from maxdiffusion import pos_rollout_gates as g

    ev, inst, install, blobs = _fake_environment()
    path = "gs://attack/forged_certificate.json"
    ev.publish_certificate(path, {"certificate": g.GATE_CERTIFICATE, "passed": True})
    install(inst.load_dev_cohort(DEV))
    try:
        g.confirm_on_test(
            path, test_cohort=g.load_test_cohort(TEST), derangement=None, tables={}, control_tables={}
        )
    except (TypeError, ValueError) as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return "SUCCEEDED: the marker alone unlocked TEST"


def attack_g_certificate_without_a_computed_gate():
    """`dev_certificate(GateVerdict(True, (), {}), …)` issued a pass with mean_delta=NaN."""
    import inspect

    from maxdiffusion import pos_rollout_gates as g

    parameters = inspect.signature(g.dev_certificate).parameters
    if "verdict" in parameters:
        return "SUCCEEDED: dev_certificate still accepts a caller's verdict"
    return f"REFUSED: dev_certificate computes its own gate; it takes {list(parameters)}"


def attack_g_test_seeded_derangement_for_dev():
    """A TEST-seeded mapping passed as DEV's derangement; any permutation was accepted."""
    import dataclasses as dc

    from maxdiffusion import pos_rollout_gates as g

    ev, inst, install, _ = _fake_environment()
    cohort = inst.load_dev_cohort(DEV)
    install(cohort)
    real = g.cohort_derangement(cohort)
    foreign = dc.replace(real, cohort="test64")
    try:
        g.action_use_plan(cohort, derangement=foreign)
    except (TypeError, ValueError) as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return "SUCCEEDED: a foreign-cohort derangement was accepted for DEV"


def attack_g_byte_identical_donors():
    """Byte-identical donors passed, because nothing ever received action bytes or their digests."""
    from maxdiffusion import pos_rollout_gates as g

    ev, inst, install, _ = _fake_environment()
    cohort = inst.load_dev_cohort(DEV)
    install(cohort, identical=True)
    try:
        g.cohort_derangement(cohort)
    except (TypeError, ValueError) as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return "SUCCEEDED: every example was deranged onto byte-identical actions"


def attack_g_donor_keyed_noise():
    """A scorer that keys the wrong-action row on the DONOR passed every scoped test."""
    from maxdiffusion import eval_wan_pos_rollout as ev
    from maxdiffusion import pos_rollout_gates as g

    ev, inst, install, _ = _fake_environment()
    cohort = inst.load_dev_cohort(DEV)
    install(cohort)
    art = g.cohort_derangement(cohort)
    names = list(cohort.names)

    def table(condition, ssim, donor_keyed=False):
        rows = {}
        for name in names:
            donor = art.donor(name) if condition == "wrong" else name
            drawn = donor if (donor_keyed and condition == "wrong") else name
            rows[name] = {
                "ssim": ssim,
                "mse": 1.0,
                "actions_from": donor,
                "actions_sha256": art.action_sha256[donor],
                "draw_key_sha256": ev.draw_key_digest(ev.evaluation_draw_key(drawn)),
                "num_steps": 25,
            }
        return ev.build_score_table(
            rows=rows,
            cohort=cohort,
            condition=condition,
            arm="rollout",
            checkpoint=_identity(run="rb", source="selection"),
            num_steps=25,
            derangement_sha256=art.fingerprint if condition == "wrong" else None,
        )

    try:
        g.action_use_gate(
            true_table=table("true", 0.36),
            wrong_table=table("wrong", 0.30, donor_keyed=True),
            cohort=cohort,
            derangement=art,
        )
    except (TypeError, ValueError) as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return "SUCCEEDED: the wrong-action row was scored on the donor's noise"


def attack_g_report_on_incomplete_coverage():
    """The early coverage return dropped provenance and `action_use_report` then raised KeyError."""
    from maxdiffusion import eval_wan_pos_rollout as ev
    from maxdiffusion import pos_rollout_gates as g

    ev, inst, install, _ = _fake_environment()
    cohort = inst.load_dev_cohort(DEV)
    install(cohort)
    art = g.cohort_derangement(cohort)
    names = list(cohort.names)

    def table(condition, ssim, subset=None):
        rows = {}
        for name in subset or names:
            donor = art.donor(name) if condition == "wrong" else (None if condition == "zero" else name)
            rows[name] = {
                "ssim": ssim,
                "mse": 1.0,
                "actions_from": donor,
                "actions_sha256": art.action_sha256[donor] if donor else "0" * 64,
                "draw_key_sha256": ev.draw_key_digest(ev.evaluation_draw_key(name)),
                "num_steps": 25,
            }
        return ev.build_score_table(
            rows=rows,
            cohort=cohort,
            condition=condition,
            arm="rollout",
            checkpoint=_identity(run="rb", source="selection"),
            num_steps=25,
            derangement_sha256=art.fingerprint if condition == "wrong" else None,
            allow_incomplete=subset is not None,
        )

    tables = {
        "true": table("true", 0.36, subset=names[:63]),
        "wrong": table("wrong", 0.30),
        "zero": table("zero", 0.20),
        "adapter_disabled": table("adapter_disabled", 0.10),
    }
    control = {c: table(c, 0.30) for c in ("true", "wrong", "zero")}
    try:
        report = g.action_use_report(cohort, derangement=art, tables=tables, control_tables=control)
    except KeyError as error:
        return f"SUCCEEDED: the report still crashes on incomplete coverage ({error})"
    numbers = report["gate"].numbers
    missing = [k for k in ("derangement", "derangement_sha256", "cohort", "manifest_sha256") if k not in numbers]
    deltas = [k for k in report["reported"] if k.startswith("mean_delta") or k.startswith("control_mean")]
    if missing or deltas:
        return f"SUCCEEDED: missing provenance {missing}, reported deltas {deltas}"
    return "REFUSED: the failing verdict keeps its provenance and reports no deltas"


def attack_g_control_without_zero():
    """Plan §3e requires true/wrong/zero on matched-C0; the code required only {'true','wrong'}."""
    from maxdiffusion import pos_rollout_gates as g

    ev, inst, install, _ = _fake_environment()
    cohort = inst.load_dev_cohort(DEV)
    install(cohort)
    art = g.cohort_derangement(cohort)
    names = list(cohort.names)

    def table(condition, ssim):
        rows = {}
        for name in names:
            donor = art.donor(name) if condition == "wrong" else (None if condition == "zero" else name)
            rows[name] = {
                "ssim": ssim,
                "mse": 1.0,
                "actions_from": donor,
                "actions_sha256": art.action_sha256[donor] if donor else "0" * 64,
                "draw_key_sha256": ev.draw_key_digest(ev.evaluation_draw_key(name)),
                "num_steps": 25,
            }
        return ev.build_score_table(
            rows=rows,
            cohort=cohort,
            condition=condition,
            arm="rollout",
            checkpoint=_identity(run="rb", source="selection"),
            num_steps=25,
            derangement_sha256=art.fingerprint if condition == "wrong" else None,
        )

    # A COMPLETE arm battery, so the refusal that fires is the one about matched-C0's missing zero row.
    tables = {c: table(c, v) for c, v in (("true", 0.36), ("wrong", 0.30), ("zero", 0.20), ("adapter_disabled", 0.1))}
    try:
        g.action_use_report(
            cohort,
            derangement=art,
            tables=tables,
            control_tables={"true": table("true", 0.33), "wrong": table("wrong", 0.32)},
        )
    except (TypeError, ValueError) as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return "SUCCEEDED: matched-C0 was reported without its zero-action row"


def attack_g_confirm_only_the_primary_gate():
    """`confirm_on_test` ran only the primary gate where §3e requires the action-use confirmation."""
    import inspect

    from maxdiffusion import pos_rollout_gates as g

    source = inspect.getsource(g.confirm_on_test)
    runs_action_use = "action_use_gate(" in source
    takes_derangement = "derangement" in inspect.signature(g.confirm_on_test).parameters
    if runs_action_use and takes_derangement:
        return "REFUSED: the TEST door runs both gates and demands an independent TEST derangement"
    return f"SUCCEEDED: action_use={runs_action_use} derangement={takes_derangement}"


def attack_g_the_evaluator_cannot_run():
    """`main()` always reached a raising seam, so no phase could restore, score or certify."""
    import pathlib as _p

    from maxdiffusion import eval_wan_pos_rollout as ev

    source = _p.Path(ev.__file__).read_text()
    if "NotImplementedError" in source:
        return "SUCCEEDED: the evaluator still contains a raising orchestration seam"
    missing = [n for n in ("run_evaluation", "run_anchor_phase", "run_benchmark_phase", "run_gates_phase",
                           "run_confirm_phase") if not callable(getattr(ev, n, None))]
    if missing:
        return f"SUCCEEDED: the protocol is missing {missing}"
    return "REFUSED: all four phases are implemented and dispatched"


def attack_g_skip_the_anchor():
    """Can a new arm be scored before the wiring proof has passed?"""
    from maxdiffusion import eval_wan_pos_rollout as ev

    ev, inst, install, blobs = _fake_environment()

    class Config(dict):
        def __getattr__(self, key):
            if key not in self:
                raise ValueError(key)
            return self[key]

        def get_keys(self):
            return dict(self)

    for phase in ("benchmark", "gates", "confirm"):
        config = Config(
            {"pos_eval_phase": phase, "base_output_directory": f"gs://attack/run/eval_{phase}_att-1"}
        )
        try:
            ev.run_evaluation(config, backend=object())
        except (TypeError, ValueError) as error:
            last = f"REFUSED ({phase}): {str(error).splitlines()[0][:90]}"
        else:
            return f"SUCCEEDED: {phase} ran with no anchor certificate"
    return last


def attack_g_plan_with_no_consumer():
    """The identical-noise contract used to live in a plan dictionary nothing consumed."""
    import inspect

    from maxdiffusion import pos_rollout_gates as g

    producer = inspect.getsource(g.score_condition_table)
    consumes = "plan.entries" in producer and "ActionUsePlan" in producer
    checks_draw = "draw_key_name" in producer
    if consumes and checks_draw:
        return "REFUSED: score_condition_table consumes the plan and enforces the receiver-keyed draw"
    return f"SUCCEEDED: consumes={consumes} checks_draw={checks_draw}"



# =================================================================================================
# Round F1 — the launch-surface RE-review's six M1-critical findings, re-run against the fixes.
# LS-4 fingerprint blindness · LS-5 load without re-decision · LS-6 the entrypoint that cannot run ·
# LS-7 the test layer that never ran the real parser · LS-8 projection arithmetic · LS-10 provenance.
# =================================================================================================


def attack_f1_load_without_redeciding(tmp):
    """LS-5, verbatim: edit an authorized cell's recorded measurement to a capacity-level peak plus a
    reservation failure, recompute the (unkeyed) digest, and see whether it is still authorized."""
    fp = _probe_env()
    context = _ctx(fp)
    path = pathlib.Path(tmp) / "f1_redecide.json"
    _auth(fp, tmp, [_fit(fp, context)], name="f1_redecide.json", context=context)
    stored = json.loads(path.read_text())
    stored["payload"]["measurements"][0]["peak_bytes"] = 32 * 1024**3
    stored["payload"]["measurements"][0]["reservation_failures"] = 3
    stored["sha256"] = hashlib.sha256(json.dumps(stored["payload"], sort_keys=True).encode("utf-8")).hexdigest()
    path.write_text(json.dumps(stored))
    try:
        loaded = fp.load_authorization(str(path))
        fp.assert_cell_authorized(loaded, fp.FitCell("rollout", 32, 2), context=context)
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:130]}"
    return "SUCCEEDED: a cell recorded at 100% of capacity with 3 reservation failures is authorized"


def attack_f1_fingerprint_blindness(tmp):
    """LS-4: the reviewer changed action_tokens / pre_context_tokens / flash_block_sizes /
    latent_frames and the context digest did not move."""
    fp = _probe_env()
    base = fp.recipe_fingerprint(_pos_config())
    blind = [
        key
        for key, value in (("action_tokens", 64), ("pre_context_tokens", 64),
                           ("flash_block_sizes", {"block_q": 1024}), ("latent_frames", 99),
                           ("action_dim", 14), ("logical_axis_rules", [["a", "b"]]))
        if fp.recipe_fingerprint(_pos_config(**{key: value})) == base
    ]
    return (f"SUCCEEDED: {blind} leave the fingerprint unchanged" if blind
            else "REFUSED: every graph/HBM-bearing key moves the fingerprint")


def attack_f1_projection_miscounts(tmp):
    """LS-8: the loop evaluates at the cadence AND at the final step, and never reads an independent
    checkpoint cadence."""
    fp = _probe_env()
    from maxdiffusion.pos_rollout_loop import LoopSchedule, should_evaluate

    schedule = LoopSchedule(max_train_steps=1_001, eval_every=1_000, logical_batch=256, microbatch=32,
                            seed=0, arm="rollout", k_b=2, num_steps=25)
    production = sum(1 for step in range(1, 1_002) if should_evaluate(step, schedule))
    projected = fp.project_wall_clock(_fit(fp), max_train_steps=1_001, eval_every=1_000,
                                      checkpoint_every=1_000)["evaluations"]
    notes = [f"projected={projected} production={production}"]
    try:
        fp.project_wall_clock(_fit(fp), max_train_steps=10_000, eval_every=1_000, checkpoint_every=250)
        notes.append("an independent checkpoint cadence was PROJECTED")
        return "SUCCEEDED: " + "; ".join(notes)
    except ValueError:
        notes.append("an independent checkpoint cadence is refused")
    return ("REFUSED: " if projected == production else "SUCCEEDED: ") + "; ".join(notes)


def attack_f1_two_models_compare_equal(tmp):
    """LS-10: `@local-dir` and `@no-local-snapshot:<Error>` made different models share an identity."""
    fp = _probe_env()
    notes = []
    try:
        fp.derive_model_revision(_pos_config(pretrained_model_name_or_path="No-Such-Org/No-Such-Model"))
        notes.append("an unresolvable model produced a revision")
    except ValueError:
        notes.append("an unresolvable model fails closed")
    left, right = pathlib.Path(tmp) / "f1mA", pathlib.Path(tmp) / "f1mB"
    for directory, payload in ((left, b"A"), (right, b"B-and-longer")):
        directory.mkdir(exist_ok=True)
        (directory / "model.safetensors").write_bytes(payload)
    revisions = [fp.derive_model_revision(_pos_config(pretrained_model_name_or_path=str(d)))
                 for d in (left, right)]
    if revisions[0] == revisions[1] or "an unresolvable model produced a revision" in notes:
        return "SUCCEEDED: " + "; ".join(notes + [f"two local models share {revisions[0].split('@')[-1]}"])
    return "REFUSED: " + "; ".join(notes + ["two local models get different content manifests"])


def attack_f1_entrypoint_cannot_measure(tmp):
    """LS-6: the production default measurer raised unconditionally, so M1 died before cell one."""
    import inspect

    fp = _probe_env()
    raises = "NotImplementedError" in inspect.getsource(fp.measure_cell_on_device)
    reaches = []
    try:
        fp.run_fit_probe(_pos_config(pos_fit_authorization=str(pathlib.Path(tmp) / "f1_m1.json")),
                         devices=[_Dev()], cells=[fp.FitCell("rollout", 8, 2)], trials=1)
        reaches.append("it measured")
    except NotImplementedError as error:
        return f"SUCCEEDED (M1 still dies): {str(error)[:90]}"
    except Exception as error:  # noqa: BLE001 -- reaching the real weights load is the point
        reaches.append(f"reached the real model load ({type(error).__name__})")
    return (f"SUCCEEDED: measurer still raises" if raises
            else f"REFUSED: the measurement path is real -- {reaches[0]}")


def attack_f1_real_entrypoint_never_runs(tmp):
    """LS-7 (M1 slice): does the REAL entrypoint run under the REAL config parser?"""
    import os
    import subprocess
    import sys

    repo = pathlib.Path(instrument.__file__).resolve().parents[2]
    test = repo / "src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_fit_probe.py"
    source = test.read_text()
    if "pyconfig.initialize" not in source or "_run_real_entrypoint" not in source:
        return "SUCCEEDED: no test executes the real entrypoint through the real parser"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(test), "-q", "-p", "no:cacheprovider", "--no-header",
         "-k", "real_m1_entrypoint_measures", "--tb=line"],
        cwd=repo, capture_output=True, text=True, timeout=1800,
        env={**os.environ, "PYTHONPATH": "src", "JAX_PLATFORMS": "cpu"},
    )
    tail = [line for line in proc.stdout.splitlines() if line.strip()][-1:] or ["<no output>"]
    return (f"REFUSED (it runs): {tail[0][:110]}" if proc.returncode == 0
            else f"SUCCEEDED: the real entrypoint test did not pass -- {tail[0][:110]}")



# =================================================================================================
# F3a -- the re-review's device/measurement probes (EV-1 bf16 boundary, EV-2 finiteness, EV-6 grid).
# All three SUCCEEDED against the previous round; each must now be REFUSED.
# =================================================================================================


def _f3a_rows(**over):
    from maxdiffusion import eval_wan_pos_rollout as ev

    row = {
        "latent_mse": ev.HISTORICAL_ANCHOR.mean_latent_mse,
        "pixel_mse": ev.HISTORICAL_ANCHOR.mean_pixel_mse,
        "ssim_avg": ev.HISTORICAL_ANCHOR.mean_ssim,
        "num_steps": ev.DEPLOYED_SAMPLING_STEPS,
        "grid_sha256": ev.DEPLOYED_GRID_SHA256,
    }
    row.update(over)
    return [{"name": n, **row} for n in ev.HISTORICAL_ANCHOR.sample_names]


def _f3a_measure(rows):
    from maxdiffusion import eval_wan_pos_rollout as ev

    run = ev.HISTORICAL_ANCHOR.run_name
    identity = ev.CheckpointIdentity(
        run_name=run, step=30000, root=f"gs://b/{run}/checkpoints", source="historical"
    )
    return ev.summarize_samples(
        rows, checkpoint=identity, code_sha="a" * 40, model_revision="rev", test_manifest_path=TEST
    )


def attack_f3a_nan_latent_mse_certifies():
    """EV-2, executed: `abs(NaN) > tolerance` is FALSE, so a NaN latent MSE returned reproduced=True."""
    from maxdiffusion import eval_wan_pos_rollout as ev

    try:
        verdict = ev.reproduce_anchor(_f3a_measure(_f3a_rows(latent_mse=float("nan"))))
    except (TypeError, ValueError) as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return f"SUCCEEDED: reproduced={verdict.reproduced} with a NaN latent MSE"


def attack_f3a_nan_pixel_mse_certifies():
    from maxdiffusion import eval_wan_pos_rollout as ev

    try:
        verdict = ev.reproduce_anchor(_f3a_measure(_f3a_rows(pixel_mse=float("inf"))))
    except (TypeError, ValueError) as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return f"SUCCEEDED: reproduced={verdict.reproduced} with an infinite pixel MSE"


def attack_f3a_all_ones_grid():
    """EV-6, executed: an all-ones 26-sigma grid with 25 zero timesteps passed as the deployed grid."""
    import jax.numpy as jnp

    from maxdiffusion import eval_wan_pos_rollout as ev

    ones = jnp.ones((ev.DEPLOYED_SAMPLING_STEPS + 1,), jnp.float32)
    zeros = jnp.zeros((ev.DEPLOYED_SAMPLING_STEPS,), jnp.float32)
    z_i0 = jnp.zeros((1, 4, 1, 2, 2), jnp.float32)
    z_video = jnp.zeros((1, 4, 2, 2, 2), jnp.float32)
    try:
        execution = ev.rollout_prediction(
            velocity_fn=lambda *a, **k: z_video,
            sigmas=ones,
            timesteps=zeros,
            context=jnp.zeros((1, 7, 8), jnp.float32),
            z_i0=z_i0,
            z_video=z_video,
            key=ev.evaluation_draw_key("x"),
            guide_scale=5.0,
        )
    except (TypeError, ValueError) as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return f"SUCCEEDED: an all-ones grid was labelled a {execution.num_steps}-step deployed execution"


def attack_f3a_structurally_valid_but_foreign_grid():
    """The harder version: a grid with the right length, terminal zero, monotonicity and corresponding
    timesteps -- but built at a different flow shift, so it is a DIFFERENT schedule."""
    from maxdiffusion.models.wan.overfit100_sampling import overfit100_sampler_grid

    from maxdiffusion import eval_wan_pos_rollout as ev

    sigmas, timesteps = overfit100_sampler_grid(
        num_inference_steps=ev.DEPLOYED_SAMPLING_STEPS,
        flow_shift=3.0,  # the deployed value is 5.0
        sigma_min=ev.DEPLOYED_SIGMA_MIN,
        sigma_max=ev.DEPLOYED_SIGMA_MAX,
        num_train_timesteps=ev.DEPLOYED_NUM_TRAIN_TIMESTEPS,
    )
    try:
        ev.assert_deployed_grid(sigmas, timesteps)
    except (TypeError, ValueError) as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return "SUCCEEDED: a foreign-shift grid passed as the deployed schedule"


def attack_f3a_score_in_float32_under_a_bf16_config():
    """EV-1: can a run configured at bfloat16 end up drawing its noise in float32?"""
    import jax.numpy as jnp
    import numpy as np

    from maxdiffusion import eval_wan_pos_rollout as ev

    seen = {}

    def velocity_for(params, actions, adapter_enabled):
        seen["actions"] = actions.dtype
        del params, adapter_enabled

        def velocity_fn(hidden_states, timestep, encoder_hidden_states):
            seen["context"] = encoder_hidden_states.dtype
            del timestep
            return jnp.zeros_like(hidden_states)

        return velocity_fn

    sigmas, timesteps = ev.deployed_grid()
    backend = ev.DeviceBackend(
        velocity_for=velocity_for,
        decode_fn=lambda x: jnp.asarray(
            np.repeat(np.asarray(x, np.float32).mean(axis=1)[..., None], 3, axis=-1)
        ),
        sigmas=sigmas,
        timesteps=timesteps,
        context=jnp.zeros((1, 7, 8), jnp.float32),
        guide_scale=5.0,
        params={"w": jnp.zeros((1,), jnp.float32)},
        eval_dtype=jnp.bfloat16,
    ).bound({"w": jnp.zeros((1,), jnp.float32)})
    execution, _ = backend.score(
        z_i0=jnp.zeros((1, 4, 1, 4, 6), jnp.float32),
        z_video=jnp.zeros((1, 4, 2, 4, 6), jnp.float32),
        actions=jnp.zeros((1, 4, 7), jnp.float32),
        key=ev.evaluation_draw_key("x"),
    )
    wrong = [
        name
        for name, dtype in (
            ("z_pred", execution.z_pred.dtype),
            ("actions", seen.get("actions")),
            ("context", seen.get("context")),
        )
        if dtype != jnp.bfloat16
    ]
    if wrong:
        return f"SUCCEEDED: {wrong} stayed float32 under a bfloat16 configuration"
    return "REFUSED: the restored backend casts latents, actions and context before drawing"



# =================================================================================================
# Round F1b — the M1-readiness review's six findings, re-run against the fixes. The reviewer's own
# probes: "boom in program build", "No room left on device", the same-size in-place byte change, and
# the JSON 2.0 -> 2 retype.
# =================================================================================================


def attack_f1b_wrong_adapter(tmp):
    """The pilot trains the UNCHANGED pre_context adapter; the config inherited `side_adapter`."""
    import yaml

    declared = yaml.safe_load(
        pathlib.Path("src/maxdiffusion/configs/base_wan_5b_pos_rollout.yml").read_text()
    )["action_adapter_type"]
    if declared != "pre_context":
        return f"SUCCEEDED: M1 would build action_adapter_type={declared!r}, not the approved pre_context"
    fp = _probe_env()
    moved = fp.recipe_fingerprint(_pos_config(action_adapter_type="side_adapter")) != fp.recipe_fingerprint(_pos_config())
    return f"REFUSED: the config declares pre_context and the fingerprint separates the two ({moved})"


def attack_f1b_microbatch_timed_as_update(tmp):
    """The timed unit was one microbatch; `max_train_steps` counts LOGICAL updates."""
    import f1_shims

    f1_shims.install()
    from probe_f1_smoke import TinyModelSource, _config as _tiny

    fp = _probe_env()
    config = _tiny(pretrained_model_name_or_path=str(_HARNESS_MODEL), pos_logical_batch=8, pos_microbatch=2,
                   checkpoint_dir=tempfile.mkdtemp())
    program = fp.build_probe_program(config, fp.FitCell("rollout", 2, 2), model_source=TinyModelSource())
    parts = len(program.batch) if isinstance(program.batch, tuple) else 1
    width = program.eval_batch["z_video"].shape[0] if program.eval_batch is not None else None
    if parts == 1:
        return f"SUCCEEDED: the timed unit is ONE microbatch of a {8 // 2}-microbatch logical batch"
    return f"REFUSED: the timed unit accumulates all {parts} microbatches; the eval unit is batch-{width}"


def attack_f1b_inherited_peak(tmp):
    """32 sequential cells shared one lifetime high-water mark; a later cell inherited an earlier one."""
    fp = _probe_env()

    class _NoReset(fp.DeviceTelemetry):
        def reset_peak(self):
            return False

        def peak_and_capacity(self):
            return 30 * 1024**3, 32 * 1024**3

    telemetry = _NoReset()
    before = telemetry.begin_steady_state()
    try:
        peak, _, source = telemetry.end_steady_state(before, program_bytes=None)
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:130]}"
    return f"SUCCEEDED: reported {peak} bytes from {source!r} that this cell never set"


def attack_f1b_boom_is_not_oom(tmp):
    """The reviewer's probes, verbatim: a model bug and a FULL DISK became measured HBM refusals."""
    fp = _probe_env()
    probes = {
        "boom in program build": RuntimeError("boom in program build"),
        "No room left on device": OSError("No room left on device"),
        "genuine RESOURCE_EXHAUSTED": RuntimeError("RESOURCE_EXHAUSTED: HBM"),
    }
    verdicts = {name: fp._is_resource_exhausted(error) for name, error in probes.items()}
    wrong = [name for name, hit in verdicts.items() if hit and "RESOURCE" not in name]
    if wrong:
        return f"SUCCEEDED: {wrong} classified as an allocation refusal"
    return f"REFUSED: only the genuine refusal classifies ({verdicts})"


def attack_f1b_swap_the_weights(tmp):
    """A same-size in-place byte change (or a same-shaped checkpoint swap) left identity standing."""
    fp = _probe_env()
    directory = pathlib.Path(tmp) / "f1b_model"
    directory.mkdir(exist_ok=True)
    shard = directory / "model.safetensors"
    shard.write_bytes(b"A" * 8192)
    before = fp.derive_model_revision(_pos_config(pretrained_model_name_or_path=str(directory)))
    shard.write_bytes(b"B" * 8192)
    after = fp.derive_model_revision(_pos_config(pretrained_model_name_or_path=str(directory)))
    if before == after:
        return f"SUCCEEDED: every byte changed and the identity did not: {before.split('@')[-1][:20]}"
    return "REFUSED: the identity moved with the bytes"


def attack_f1b_retype_a_duration(tmp):
    """JSON 2.0 -> 2, re-hashed: 'byte-identical reconstruction' was mapping equality."""
    fp = _probe_env()
    context = _ctx(fp)
    path = pathlib.Path(tmp) / "f1b_retype.json"
    _auth(fp, tmp, [_fit(fp, context, step_seconds=2.0)], name="f1b_retype.json", context=context)
    stored = json.loads(path.read_text())
    stored["payload"]["measurements"][0]["step_seconds"] = 2
    stored["sha256"] = hashlib.sha256(json.dumps(stored["payload"], sort_keys=True).encode("utf-8")).hexdigest()
    path.write_text(json.dumps(stored))
    try:
        fp.load_authorization(str(path))
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return "SUCCEEDED: a retyped duration survived the reconstruction check"



def attack_f3afix_loader_reads_an_undeclared_key():
    """C3, executed: `load_device_backend` read `config.num_train_timesteps`; on the real
    HyperParameters an undeclared key RAISES, so the production loader died before anything ran."""
    import ast as _ast
    import types as _types

    from maxdiffusion import eval_wan_pos_rollout as ev

    root = pathlib.Path(ev.__file__).parent
    yaml_text = (root / "configs" / "base_wan_5b_pos_rollout.yml").read_text()
    import yaml as _yaml

    declared = set(_yaml.safe_load(yaml_text))
    tree = _ast.parse(pathlib.Path(ev.__file__).read_text())
    node = next(n for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef) and n.name == "load_device_backend")
    reads = sorted(
        {
            child.attr
            for child in _ast.walk(node)
            if isinstance(child, _ast.Attribute) and getattr(child.value, "id", "") == "config"
        }
    )
    undeclared = [key for key in reads if key not in declared]

    # ...and the real class is what makes an undeclared read fatal rather than merely untidy.
    pyconfig = root / "pyconfig.py"
    cls = next(
        n for n in _ast.parse(pyconfig.read_text()).body if isinstance(n, _ast.ClassDef) and n.name == "HyperParameters"
    )
    namespace = {"_config": _types.SimpleNamespace(keys=_yaml.safe_load(yaml_text))}
    exec(compile(_ast.Module(body=[cls], type_ignores=[]), str(pyconfig), "exec"), namespace)  # noqa: S102
    config = namespace["HyperParameters"]()
    try:
        _ = config.num_train_timesteps
        fatal = False
    except ValueError:
        fatal = True
    if undeclared:
        return f"SUCCEEDED: the loader reads undeclared {undeclared} (raises on the real class: {fatal})"
    return f"REFUSED: every config key the loader reads is declared (undeclared reads would raise: {fatal})"


def attack_f3afix_loader_skips_the_grid_check():
    """Can a scheduler that disagrees with the deployed schedule get past the loader?"""
    import inspect as _inspect

    from maxdiffusion import eval_wan_pos_rollout as ev

    loader = _inspect.getsource(ev.load_device_backend)
    checks = "assert_deployed_grid(" in loader
    from_scheduler = "scheduler.config.num_train_timesteps" in loader and "scheduler.config.sigma_min" in loader
    if checks and from_scheduler:
        return "REFUSED: the loader builds from the scheduler and binds the result to the deployed grid"
    return f"SUCCEEDED: grid_check={checks} scheduler_sourced={from_scheduler}"



# =================================================================================================
# Round W1 — the peak FLOOR that could authorize, the manifest framing collision, and the shared
# adapter factory. The reviewer's own probes.
# =================================================================================================


def attack_w1_authorize_on_a_floor(tmp):
    """A compiled-memory analysis is a LOWER bound; a 90% CEILING rule cannot be cleared by one."""
    fp = _probe_env()

    class _NoReset(fp.DeviceTelemetry):
        def reset_peak(self):
            return False

        def peak_and_capacity(self):
            return 30 * 1024**3, 32 * 1024**3

    telemetry = _NoReset()
    before = telemetry.begin_steady_state()
    peak, capacity, source = telemetry.end_steady_state(before, program_bytes=7 * 1024**3)
    measurement = _fit(fp, peak_bytes=peak, capacity_bytes=capacity, peak_source=source)
    verdict = fp.cell_verdict(measurement)
    if verdict.fits:
        return f"SUCCEEDED: a {peak // 1024**3}GiB FLOOR under a 30GiB standing mark authorized the cell"
    return f"REFUSED: {verdict.reasons} -- an analysis floor may refuse a cell, never authorize one"


def attack_w1_reshuffle_a_snapshot(tmp):
    """The reviewer's framing collision: one file holding `Xb\0Y` vs two files a=X, b=Y."""
    fp = _probe_env()
    left = pathlib.Path(tmp) / "w1_left"
    right = pathlib.Path(tmp) / "w1_right"
    left.mkdir(exist_ok=True)
    right.mkdir(exist_ok=True)
    (left / "a").write_bytes(b"Xb\0Y".replace(b"\\0", b"\0"))
    (left / "a").write_bytes(b"X" + b"b" + b"\0" + b"Y")
    (right / "a").write_bytes(b"X")
    (right / "b").write_bytes(b"Y")
    if fp.snapshot_manifest_digest(str(left)) == fp.snapshot_manifest_digest(str(right)):
        return "SUCCEEDED: a reshuffled snapshot keeps its identity"
    return "REFUSED: length-framed records separate the two trees"


def attack_w1_hand_rebuild_the_adapter(tmp):
    """M1 must build the adapter through the SHARED factory, dtypes and precision included."""
    import ast as _ast
    import inspect as _inspect
    import textwrap as _textwrap

    fp = _probe_env()
    source = _textwrap.dedent(_inspect.getsource(fp.ProductionModelSource.build))
    called = {
        node.func.id
        for node in _ast.walk(_ast.parse(source))
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)
    }
    from maxdiffusion.pos_rollout_update import build_adapter_stack

    factory = _inspect.getsource(build_adapter_stack)
    missing = [a for a in ("dtype=", "weights_dtype=", "precision=") if a not in factory]
    if "build_adapter_stack" not in called or missing:
        return f"SUCCEEDED: M1 calls {sorted(called)}; factory missing {missing}"
    return "REFUSED: M1 calls the shared factory, which passes the production dtypes and precision"



# =================================================================================================
# Round W2 — the trainer wiring. The reviewer's A2: "the probe measures the trainer's step" was FALSE
# because the trainer had no step. These probe the composition rather than the pieces.
# =================================================================================================


def _trainer_module():
    from maxdiffusion.trainers import wan_pos_rollout_trainer as tm

    return tm


def attack_w2_the_trainer_still_cannot_train():
    """The A2 blocker itself: `start_training` terminating at a named boundary instead of running."""
    import ast as _ast
    import inspect as _inspect

    tm = _trainer_module()
    node = next(
        item
        for item in _ast.walk(_ast.parse(_inspect.getsource(tm)))
        if isinstance(item, _ast.FunctionDef) and item.name == "start_training"
    )
    raises = [_ast.unparse(sub) for sub in _ast.walk(node) if isinstance(sub, _ast.Raise)]
    calls = {_ast.unparse(sub.func) for sub in _ast.walk(node) if isinstance(sub, _ast.Call)}
    missing = [name for name in ("run_loop", "self.load_backbone", "self.build_program") if name not in calls]
    if raises or missing:
        return f"SUCCEEDED: start_training raises {raises} and is missing {missing}"
    return "REFUSED: start_training loads, builds and runs the loop -- no boundary is left in it"


def attack_w2_bypass_the_shared_factories():
    """A private construction beside a live import: the trainer must ENTER the factories M1 enters."""
    import ast as _ast
    import inspect as _inspect

    from maxdiffusion import pos_rollout_update as shared

    import textwrap as _textwrap

    tm = _trainer_module()
    source = _textwrap.dedent(_inspect.getsource(tm.WanPosRolloutTrainer.build_program))
    called = {
        node.func.id
        for node in _ast.walk(_ast.parse(source))
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)
    }
    finalizer = _textwrap.dedent(_inspect.getsource(shared.build_training_program))
    inside = {
        node.func.id
        for node in _ast.walk(_ast.parse(finalizer))
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)
    }
    factories = ("build_adapter_stack", "build_optimizer", "build_logical_update")
    not_called = [name for name in factories if name not in inside]
    not_shared = [name for name in factories if getattr(tm, name, getattr(shared, name)) is not getattr(shared, name)]
    if "build_training_program" not in called or not_called or not_shared or "NNXWanSideAdapterStack" in source:
        return f"SUCCEEDED: finalizer missing {not_called}; not shared {not_shared}"
    return "REFUSED: both callers enter one finalizer, which enters the three shared factories"


def attack_w2_publish_an_attempt_for_a_foreign_tree(tmp):
    """The publication marker must name the tree it describes, not a path handed alongside it."""
    tm = _trainer_module()
    config = _pos_config(
        checkpoint_dir=str(pathlib.Path(tmp) / "somewhere" / "else" / "checkpoints"),
        pos_resume_parent=str(pathlib.Path(tmp) / "attempts"),
        pos_recipe_lock=str(pathlib.Path(tmp) / "lock.json"),
    )
    try:
        tm.WanPosRolloutTrainer(config).attempt_identity()
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:96]}"
    return "SUCCEEDED: an attempt id was derived for a tree outside this run's attempts root"


def attack_w2_select_on_the_training_stream():
    """Selection must be the fixed-draw DEV-64 instrument, not whatever the training step just saw."""
    import ast as _ast
    import inspect as _inspect

    import textwrap as _textwrap

    tm = _trainer_module()
    source = _textwrap.dedent(_inspect.getsource(tm.WanPosRolloutTrainer.dev_metric_fn))
    called = {
        node.func.id
        for node in _ast.walk(_ast.parse(source))
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)
    }
    if "score_dev_cohort" not in called or 'measured["metric"]' not in source:
        return f"SUCCEEDED: the selection metric is produced by {sorted(called)}"
    return "REFUSED: the metric is the instrument's cohort mean, over the instrument's own draws"


def attack_w2_gate_after_the_load(tmp):
    """A gate that fires after the pipeline load is a gate that costs a reservation."""
    import ast as _ast
    import inspect as _inspect

    tm = _trainer_module()
    node = next(
        item
        for item in _ast.walk(_ast.parse(_inspect.getsource(tm)))
        if isinstance(item, _ast.FunctionDef) and item.name == "start_training"
    )
    order = [_ast.unparse(statement) for statement in node.body]
    where = {}
    for name in ("authorized_cell", "assert_paired_recipe", "load_backbone", "run_loop"):
        where[name] = next((index for index, text in enumerate(order) if name in text), -1)
    late = [name for name in ("authorized_cell", "assert_paired_recipe") if where[name] > where["load_backbone"]]
    if late or where["load_backbone"] < 0 or where["run_loop"] < where["load_backbone"]:
        return f"SUCCEEDED: {late} run after the 5B load ({where})"
    return "REFUSED: every configuration gate precedes the pipeline load"



def attack_w2b_launch_m2_with_the_yaml_per_device_batch(tmp):
    """The launch blocker W2 found: a submission whose per-device batch leaves the loader loading
    fewer examples than the run declares. The launcher must not be able to emit one."""
    import re as _re

    launcher = pathlib.Path("bash_scripts/train_wan_pos_rollout.sh").read_text()
    reads_env = "${PER_DEVICE_BATCH" in launcher
    derived = _re.search(r"DERIVED_PER_DEVICE_BATCH=\"\$\(\( POS_LOGICAL_BATCH / POS_DEVICE_COUNT \)\)", launcher)
    emitted = 'per_device_batch_size="${DERIVED_PER_DEVICE_BATCH}"' in launcher
    guarded = "POS_LOGICAL_BATCH % POS_DEVICE_COUNT" in launcher
    if reads_env or not derived or not emitted or not guarded:
        return (
            f"SUCCEEDED: env-read={reads_env} derived={bool(derived)} emitted={emitted} guarded={guarded}"
        )

    # ...and the trainer still refuses the value the launcher used to emit, which is what makes the
    # derivation load-bearing rather than cosmetic.
    from maxdiffusion.trainers.wan_pos_rollout_trainer import WanPosRolloutTrainer

    config = _pos_config(per_device_batch_size=1.0, global_batch_size_to_load=8, pos_logical_batch=256)
    try:
        WanPosRolloutTrainer(config).assert_loader_yields_the_logical_batch()
    except ValueError as error:
        return f"REFUSED: derived in the launcher, and re-checked on the worker -- {str(error).splitlines()[0][:70]}"
    return "SUCCEEDED: a loader width that cannot feed the logical batch was accepted"



def attack_w3_measure_an_unsharded_program():
    """The final review's BLOCKER: M1 measuring a program with different shardings than training's."""
    import ast as _ast
    import inspect as _inspect
    import textwrap as _textwrap

    from maxdiffusion import pos_rollout_fit_probe as fp
    from maxdiffusion import pos_rollout_update as shared

    finalizer = _textwrap.dedent(_inspect.getsource(shared.build_training_program))
    replicates = "NamedSharding(backbone.mesh, P())" in finalizer
    scoped = finalizer.count("program_scope(config, backbone.mesh)") >= 3
    builder = _textwrap.dedent(_inspect.getsource(fp.build_probe_program))
    shares = "build_training_program" in {
        node.func.id
        for node in _ast.walk(_ast.parse(builder))
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)
    }
    measured = _textwrap.dedent(_inspect.getsource(fp._measure_under_mesh))
    inside_scope = "with program.scope():" in measured
    private_grid = "overfit100_sampler_grid" in builder
    if not (replicates and scoped and shares and inside_scope) or private_grid:
        return (
            f"SUCCEEDED: replicated={replicates} scoped={scoped} shared={shares} "
            f"measured_in_scope={inside_scope} private_grid={private_grid}"
        )
    return "REFUSED: one finalizer replicates, scopes and jits; M1 measures inside that same scope"


def attack_w3_zero_null_context():
    """M1 used to close over a ZERO null context, so it compiled against a prompt nobody deploys."""
    import inspect as _inspect
    import textwrap as _textwrap

    from maxdiffusion import pos_rollout_update as shared

    context = _textwrap.dedent(_inspect.getsource(shared.arm_context))
    from_loader = "backbone.null_context" in context
    from_scheduler = all(
        f"backbone.scheduler.config.{field}" in context for field in ("sigma_min", "sigma_max", "num_train_timesteps")
    )
    invents = "jnp.zeros" in context
    if not (from_loader and from_scheduler) or invents:
        return f"SUCCEEDED: loader={from_loader} scheduler={from_scheduler} invents_zeros={invents}"
    return "REFUSED: the context is the loader's and the grid is the scheduler's, for both callers"


def attack_w3_the_seams_diverge():
    """Two loaders is two programs: both callers must reach the settled seams through ONE function."""
    import ast as _ast
    import inspect as _inspect
    import textwrap as _textwrap

    from maxdiffusion import pos_rollout_fit_probe as fp
    from maxdiffusion import pos_rollout_update as shared

    tm = _trainer_module()
    loader = _textwrap.dedent(_inspect.getsource(shared.load_backbone))
    seams = all(f"._{name}(" in loader for name in ("load_wan_pipeline", "compute_null_context", "create_scheduler"))
    callers = []
    for owner in (tm.WanPosRolloutTrainer.load_backbone, fp.ProductionModelSource.load):
        source = _textwrap.dedent(_inspect.getsource(owner))
        callers.append(
            "load_backbone" in {
                node.func.id
                for node in _ast.walk(_ast.parse(source))
                if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)
            }
        )
    if not seams or not all(callers):
        return f"SUCCEEDED: seams_present={seams} callers_delegating={callers}"
    return "REFUSED: one loader reaches the settled seams and both callers enter it"


if __name__ == "__main__":
    _report("A-B1(a) module issue token   :", attack_a_b1a)
    _report("A-B1(b) public digest override:", attack_a_b1b)
    _report("A-B2   unrestricted callback :", attack_a_b2)
    with tempfile.TemporaryDirectory() as tmp:
        _report("B-1    terminal resume       :", attack_b1, tmp)
        _report("B-2    selection crash window:", attack_b2, tmp)
        _report("T5a-1  restore falls back     :", attack_t5a_restore_falls_back)
        _report("T5a-2  widen/miss the anchor  :", attack_t5a_widen_the_anchor)
        _report("T5a-3  TEST into the anchor   :", attack_t5a_test_into_the_anchor)
        _report("T5a-4  re-derive the benchmark:", attack_t5a_rederive_the_benchmark, tmp)
        _report("T5a-5  forge a DEV cohort     :", attack_t5a_forge_a_dev_cohort, tmp)
        _report("T5b-1  lower the primary bar  :", attack_t5b_lower_the_bar)
        _report("T5b-2  score TEST first       :", attack_t5b_score_test_first)
        _report("T5b-3  forge the derangement  :", attack_t5b_forge_the_derangement)
        _report("T5b-4  swap arm and control   :", attack_t5b_swap_arm_and_control)
        _report("T5b-5  drop C0's battery      :", attack_t5b_drop_the_control_battery)
        _report("T7-1   run an unmeasured cell :", attack_t7_run_an_unmeasured_cell, tmp)
        _report("T7-2   forge an authorization :", attack_t7_forge_an_authorization, tmp)
        _report("T7-3   edit an authorization  :", attack_t7_edit_a_published_authorization, tmp)
        _report("T7-4   run a cell that missed :", attack_t7_authorize_a_cell_that_missed, tmp)
        _report("T7-5   project the unmeasured :", attack_t7_project_what_was_not_measured, tmp)
        _report("P1-1   echoed-identity decoder:", attack_p1_echoed_identity_decoder)
        _report("P1-2   NaN poisons selection  :", attack_p1_poison_selection_with_nan)
        _report("P1-3   gs:// written locally  :", attack_p1_write_a_gs_artifact_locally)
        _report("P3-1   arms share checkpoints :", attack_p3_arms_share_checkpoint_state, tmp)
        _report("P3-2   divergent second arm   :", attack_p3_divergent_second_arm, tmp)
        _report("P3-3   confirm w/o certificate:", attack_p3_confirm_without_a_certificate, tmp)
        _report("P3-4   flatten attempt scoping:", attack_p3_flatten_the_attempt_scoping, tmp)
        _report("P3-5   adopt incomplete/foreign:", attack_p3_adopt_an_incomplete_or_foreign_publication, tmp)
        _report("P3-6   another program's auth :", attack_p3_authorization_from_another_program, tmp)
        _report("P3-7   one arm authorizes other:", attack_p3_one_arm_authorizes_the_other, tmp)
        _report("P3-8   contradictory duplicates:", attack_p3_contradictory_duplicate_trials, tmp)
        _report("P3-9   M1 cannot be run        :", attack_p3_m1_cannot_be_run, tmp)

        # Groups 3-4: the reviewer's pass-2 EXECUTED attacks on the evaluator and the gates.
        _report("G3-1   anchor: foreign names   :", attack_g_anchor_foreign_names)
        _report("G3-2   anchor: wrong order     :", attack_g_anchor_wrong_order)
        _report("G3-3   anchor: foreign ckpt    :", attack_g_anchor_foreign_checkpoint)
        _report("G3-4   certify a short rollout :", attack_g_certify_a_short_rollout)
        _report("G3-5   forge the marker        :", attack_g_forge_the_certificate_marker)
        _report("G3-6   certificate w/o a gate  :", attack_g_certificate_without_a_computed_gate)
        _report("G3-7   TEST-seeded derangement :", attack_g_test_seeded_derangement_for_dev)
        _report("G3-8   byte-identical donors   :", attack_g_byte_identical_donors)
        _report("G3-9   donor-keyed wrong noise :", attack_g_donor_keyed_noise)
        _report("G3-10  report on bad coverage  :", attack_g_report_on_incomplete_coverage)
        _report("G3-11  C0 without its zero row :", attack_g_control_without_zero)
        _report("G3-12  TEST: primary gate only :", attack_g_confirm_only_the_primary_gate)
        _report("G3-13  the evaluator cannot run:", attack_g_the_evaluator_cannot_run)
        _report("G3-14  skip the anchor         :", attack_g_skip_the_anchor)
        _report("G4-1   a plan with no consumer :", attack_g_plan_with_no_consumer)
        _report("F1-1   load w/o re-decision  ", attack_f1_load_without_redeciding, tmp)
        _report("F1-2   fingerprint blindness ", attack_f1_fingerprint_blindness, tmp)
        _report("F1-3   projection miscounts  ", attack_f1_projection_miscounts, tmp)
        _report("F1-4   two models are equal  ", attack_f1_two_models_compare_equal, tmp)
        _report("F1-5   entrypoint cannot run ", attack_f1_entrypoint_cannot_measure, tmp)
        _report("F1-6   real parser never runs", attack_f1_real_entrypoint_never_runs, tmp)
        # F3a: the device/measurement boundary.
        _report("F3a-1  NaN latent MSE certifies:", attack_f3a_nan_latent_mse_certifies)
        _report("F3a-2  inf pixel MSE certifies :", attack_f3a_nan_pixel_mse_certifies)
        _report("F3a-3  all-ones grid accepted  :", attack_f3a_all_ones_grid)
        _report("F3a-4  foreign-shift grid      :", attack_f3a_structurally_valid_but_foreign_grid)
        _report("F3a-5  float32 under bf16 cfg  :", attack_f3a_score_in_float32_under_a_bf16_config)
        _report("F1b-1  wrong adapter type   ", attack_f1b_wrong_adapter, tmp)
        _report("F1b-2  microbatch as update ", attack_f1b_microbatch_timed_as_update, tmp)
        _report("F1b-3  inherited peak       ", attack_f1b_inherited_peak, tmp)
        _report("F1b-4  'boom' is not OOM    ", attack_f1b_boom_is_not_oom, tmp)
        _report("F1b-5  swap the weights     ", attack_f1b_swap_the_weights, tmp)
        _report("F1b-6  retype a duration    ", attack_f1b_retype_a_duration, tmp)
        _report("F3a-6  loader: undeclared key  :", attack_f3afix_loader_reads_an_undeclared_key)
        _report("F3a-7  loader: skips grid check:", attack_f3afix_loader_skips_the_grid_check)
        _report("W1-1   authorize on a floor ", attack_w1_authorize_on_a_floor, tmp)
        _report("W1-2   reshuffle a snapshot ", attack_w1_reshuffle_a_snapshot, tmp)
        _report("W1-3   hand-rebuild adapter ", attack_w1_hand_rebuild_the_adapter, tmp)
        _report("W2-1   trainer cannot train  ", attack_w2_the_trainer_still_cannot_train)
        _report("W2-2   bypass the factories  ", attack_w2_bypass_the_shared_factories)
        _report("W2-3   publish a foreign tree", attack_w2_publish_an_attempt_for_a_foreign_tree, tmp)
        _report("W2-4   select on train stream", attack_w2_select_on_the_training_stream)
        _report("W2-5   gate after the load   ", attack_w2_gate_after_the_load, tmp)
        _report("W2b-1  M2 at the YAML batch  ", attack_w2b_launch_m2_with_the_yaml_per_device_batch, tmp)
        _report("W3-1   unsharded M1 program ", attack_w3_measure_an_unsharded_program)
        _report("W3-2   zero null context     ", attack_w3_zero_null_context)
        _report("W3-3   two loaders, two progs", attack_w3_the_seams_diverge)
