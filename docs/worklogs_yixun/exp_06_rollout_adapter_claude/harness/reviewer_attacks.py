"""The reviewer's five EXECUTED attacks, re-runnable as the round's acceptance criteria.

A-B1(a) module issue token · A-B1(b) public digest override · A-B2 unrestricted batch callback ·
B-1 terminal-verdict resume · B-2 selection artifact after a crash in the write window.
"""

import contextlib, dataclasses, functools, hashlib, json, pathlib, tempfile

import jax.numpy as jnp

from maxdiffusion import pos_rollout_dev_instrument as instrument
from maxdiffusion import pos_rollout_loop as loop

MD = (
    pathlib.Path(instrument.__file__).resolve().parents[2]
    / "docs/worklogs_yixun/exp_04_null_adapter_claude/j0_manifests"
)
DEV, TEST = str(MD / "dev64.json"), str(MD / "test64.json")
TEST_ROW = json.loads(pathlib.Path(TEST).read_text())["rows"][0]
SHAPE = (4, 3, 4, 6)


def attack_a_b1a():
    token = getattr(instrument, "_ISSUE_TOKEN", None)
    if token is None:
        return "REFUSED: there is no _ISSUE_TOKEN module attribute to hand back"
    cohort = instrument.DevCohort(
        token, cohort="dev64", rows=[{**TEST_ROW, "split": "dev64"}], manifest_sha256="0" * 64, manifest_path=DEV
    )
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
        out = instrument.score_dev_cohort(
            cohort,
            lambda p, b, c, *, draws: (jnp.asarray(1.0), {}),
            lambda row: test_batch,
            params=jnp.asarray(1.0),
            context=Ctx(),
            example_shape=SHAPE,
        )
    except (TypeError, ValueError) as error:
        return f"REFUSED: {error}"
    return f"SUCCEEDED: metric {out['metric']} stamped cohort={out['cohort']} sha={out['manifest_sha256'][:12]}"


def _state():
    return loop.RolloutTrainState(
        params={"w": jnp.zeros((2,), jnp.float32)}, opt_state={"mu": jnp.zeros((2,), jnp.float32)}, step=0
    )


def _schedule(**over):
    base = dict(
        max_train_steps=6, eval_every=2, logical_batch=4, microbatch=2, seed=0, arm="rollout", k_b=2, num_steps=25
    )
    base.update(over)
    return loop.LoopSchedule(**base)


def _stream(schedule):
    import itertools

    def factory(seed):
        events.append(("iterator", int(seed)))
        return itertools.repeat(
            {
                "z_video": jnp.zeros((schedule.logical_batch, *SHAPE), jnp.float32),
                "actions": jnp.zeros((schedule.logical_batch, 4, 7), jnp.float32),
            }
        )

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
    loop.run_loop(
        _state(),
        schedule,
        batches=_stream(schedule),
        update_fn=lambda s, b, d, sc, gs: (dataclasses.replace(s, params=s.params), 1.0 / gs),
        dev_metric_fn=lambda s, step: next(values),
        manager=manager,
    )
    assert loop.stop_verdict(loop.restore_eval_history(loop.build_checkpoint_manager(directory))).stop
    events = []

    def update(state, batch_parts, draw_parts, sched, global_step):
        events.append(("update", global_step))
        return dataclasses.replace(state, params=state.params), 1.0

    report = loop.run_loop(
        _state(),
        _schedule(max_train_steps=30_000, eval_every=1),
        batches=_stream(_schedule()),
        update_fn=update,
        dev_metric_fn=lambda s, step: 0.01,
        manager=loop.build_checkpoint_manager(directory),
    )
    outcome = "SUCCEEDED: a terminal reopen trained on" if events else "REFUSED: no step, no iterator"
    return f"{outcome} (steps_run={report.steps_run} events={events})"


def attack_b2(tmp):
    """Crash between the resume save and the selection update; see what the sibling ships."""
    directory = str(pathlib.Path(tmp) / "b2")
    schedule = _schedule(max_train_steps=2, eval_every=2)
    values = iter([0.5])
    loop.run_loop(
        _state(),
        schedule,
        batches=_stream(schedule),
        update_fn=lambda s, b, d, sc, gs: (dataclasses.replace(s, params=s.params), 1.0),
        dev_metric_fn=lambda s, step: next(values),
        manager=loop.build_checkpoint_manager(directory),
    )
    selection = loop.build_selection_manager(directory)
    values = iter([0.9])
    schedule = _schedule(max_train_steps=4, eval_every=2)
    report = loop.run_loop(
        _state(),
        schedule,
        batches=_stream(schedule),
        update_fn=lambda s, b, d, sc, gs: (dataclasses.replace(s, params=s.params), 1.0),
        dev_metric_fn=lambda s, step: next(values),
        manager=loop.build_checkpoint_manager(directory),
        selection_manager=selection,
    )
    selection.wait_until_finished()
    shipped = selection.latest_step()
    outcome = "REFUSED: reconciled" if shipped == report.retained_step else "SUCCEEDED: stale/wrong selection"
    return f"{outcome} (history_best={report.retained_step} shipped_selection={shipped})"


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
    """Can a caller widen the band, swap the record, or pass a measurement that misses it?

    **F7d: this probe had been dead since the 2026-08-09/10 evaluator rework and was mislabelling its
    own failure.** `reproduce_anchor` stopped accepting a mapping — it takes a `Measurement`, so the
    hand-written record below is refused by type. The old body caught that `TypeError` and recorded it
    as "no tolerance argument exists", which was simply false, then made a second unguarded call whose
    `TypeError` escaped into `_report` and was printed as a REFUSAL.

    Rewritten so the refusal is OBSERVED and RETURNED: each attempt is executed, a production refusal
    is recorded as one, and only a verdict that actually reproduces the anchor counts as success.
    """
    anchor, _, _ = _anchor_env()
    forged = {
        "mean_ssim": 0.20,
        "mean_latent_mse": 1.496,
        "mean_pixel_mse": 0.0983,
        "num_samples": 4,
        "checkpoint_step": 30000,
        "sample_names": list("abcd"),
    }
    attempts = []
    for label, extra in (("tolerance override", {"tolerance": 0.5}), ("hand-written record", {})):
        try:
            verdict = anchor.reproduce_anchor(dict(forged), **extra)
        except TypeError as error:
            attempts.append(f"{label} refused ({str(error).splitlines()[0][:56]})")
        except ValueError as error:
            attempts.append(f"{label} refused ({str(error).splitlines()[0][:56]})")
        else:
            if getattr(verdict, "reproduced", False):
                return f"SUCCEEDED: {label} reproduced the anchor from a record nobody measured"
            attempts.append(f"{label} did not reproduce (reproduced={verdict.reproduced})")
    return "REFUSED: " + "; ".join(attempts)


def attack_t5a_test_into_the_anchor():
    """Can a TEST-64 example be scored into the anchor summary?

    **F8 — dead since 2026-08-09 (`76117df`).** ``summarize_samples`` stopped taking
    ``checkpoint_step=``: a measurement is bound to the :class:`CheckpointIdentity` a restore
    produced, and the code SHA and model revision are part of the same binding. The old call died on
    the signature, and until F7d's universal guard `_report` scored that crash as a REFUSAL.

    Re-expressed against the real call and deliberately legal in every OTHER respect: each row
    carries the deployed grid digest and the deployed horizon, so the only thing wrong with this
    summary is the held-out name inside it. A row that also failed the grid or horizon check would
    produce a refusal that says nothing about the TEST screen — which is the F5b caution (watch the
    thing the probe is named for).
    """
    from maxdiffusion import eval_wan_pos_rollout as ev

    anchor, test_manifest, _ = _anchor_env()
    intruder = json.loads(pathlib.Path(test_manifest).read_text())["rows"][0]["name"]
    legal = {
        "latent_mse": ev.HISTORICAL_ANCHOR.mean_latent_mse,
        "pixel_mse": ev.HISTORICAL_ANCHOR.mean_pixel_mse,
        "ssim_avg": ev.HISTORICAL_ANCHOR.mean_ssim,
        "num_steps": ev.DEPLOYED_SAMPLING_STEPS,
        "grid_sha256": ev.DEPLOYED_GRID_SHA256,
    }
    rows = [{"name": name, **legal} for name in ("a", intruder)]
    try:
        anchor.summarize_samples(
            rows,
            checkpoint=_identity(),
            code_sha="a" * 40,
            model_revision="rev@" + "b" * 40,
            test_manifest_path=test_manifest,
        )
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return f"SUCCEEDED: {intruder} scored into the anchor summary"


def attack_t5a_rederive_the_benchmark(tmp):
    """Can the frozen benchmark row be silently re-derived with different numbers?

    **F8 — dead since 2026-08-09 (`76117df`).** ``freeze_benchmark_row`` stopped taking the cohort,
    the per-example values, the checkpoint, the code SHA and the model revision as five independent
    caller assertions; it derives every one of them from ONE bound ``ScoreTable``. The old
    five-argument call died on ``cohort=`` and the crash was filed as a refusal.

    The attack itself is unchanged and is issue #10 in the small: freeze the baseline, then freeze a
    BETTER one at the same path. Both tables are legitimately built and differ only in their numbers,
    so a refusal here is about republication and not about the artifact being malformed.
    """
    from maxdiffusion import pos_rollout_dev_instrument as instrument

    anchor, _, dev = _anchor_env()
    cohort = instrument.load_dev_cohort(dev)
    names = list(cohort.names)
    path = str(pathlib.Path(tmp) / "bench.json")
    frozen = anchor.freeze_benchmark_row(path, table=_gate_table(names, 0.25, cohort=cohort))
    try:
        anchor.freeze_benchmark_row(path, table=_gate_table(names, 0.95, cohort=cohort))
    except ValueError as error:
        return f"REFUSED (mean stayed {frozen['mean_ssim']:.2f}): {str(error).splitlines()[0][:90]}"
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
        anchor.freeze_benchmark_row(
            str(pathlib.Path(tmp) / "forged.json"),
            cohort=_LookAlike(),
            per_example={row["name"]: 0.9},
            checkpoint={"step": 30000},
            code_sha="a" * 40,
            model_revision="rev",
        )
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


@contextlib.contextmanager
def _cohort_records(cohort):
    """In-memory records for THIS cohort's rows, installed for the block and then REMOVED.

    ``cohort_derangement`` reads every example's real action bytes through ``CohortBatchReader`` —
    that is what makes "no example received byte-identical actions" a measurement rather than a
    promise about a name list — so a derangement cannot be built on a laptop without records.

    ``_fake_environment`` supplies records, but it also installs an in-memory ``gs://`` filesystem
    and a shard binder **process-wide and permanently**, and the T5b probes run BEFORE the T7/P1/P3
    probes, which have never been measured under those fakes. Installing it here would change what
    twenty later probes are standing on. This patches the two seams a derangement actually needs and
    restores both, so the blast radius is the ``with`` block.
    """
    import numpy as np

    from maxdiffusion import null_adapter_manifest_io, run_wan_null_inversion

    geometry = {"z_i0": (48, 1, 12, 20), "z_video": (48, 9, 12, 20), "actions": (32, 7)}
    by_shard = {}
    for row in cohort.rows:
        by_shard.setdefault(str(row["shard_path"]), []).append(dict(row))

    def fill(name):
        """Distinct per example, so the cohort legitimately SUPPORTS a derangement. (The
        byte-identical case is `G3-8`'s attack, and it uses `_fake_environment(identical=True)`.)"""
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

    saved = (run_wan_null_inversion._tfrecord_reader, null_adapter_manifest_io.shard_binding)
    run_wan_null_inversion._tfrecord_reader = reader
    null_adapter_manifest_io.shard_binding = binder
    try:
        yield
    finally:
        run_wan_null_inversion._tfrecord_reader, null_adapter_manifest_io.shard_binding = saved


def _tbl(names, ssim):
    """A naked mapping. Kept for `T5b-4`, whose attack is the POSITIONAL call itself — those
    arguments never reach the gate's body, so what they are is irrelevant to what it tests."""
    return {n: {"ssim": float(ssim), "mse": 1.0} for n in names}


def _gate_table(
    names,
    ssim,
    *,
    cohort=None,
    condition="true",
    arm="rollout",
    run="rb",
    mse=1.0,
    derangement=None,
    draw_key_for=None,
    checkpoint=None,
    incomplete=False,
):
    """A LEGITIMATE built :class:`ScoreTable` — the artifact every gate has required since F8's drift.

    **Why this exists (F8).** The 2026-08-09 rework stopped letting a gate read
    ``{name: {"ssim": …}}``: a mapping cannot say which checkpoint, whose actions or which noise
    produced it, and those identities are precisely what a gate cross-checks. Five probes below were
    still handing over mappings, so every one of them died inside ``as_gate_table``'s type refusal
    **before its own attack ran**, and scored as a refusal for four days.

    So the probes need a real artifact to attack. This builds one through the real constructor, with
    per-row action digests and pinned per-example noise keys, which lets each probe apply its ONE
    mutation to an otherwise-legal table and learn what production does about *that* — rather than
    collecting a refusal that was really about the argument type.
    """
    from maxdiffusion import eval_wan_pos_rollout as ev

    cohort = _gate_env()[1] if cohort is None else cohort
    rows = {}
    for index, name in enumerate(names):
        if condition == "zero":
            donor = None
        elif condition == "wrong" and derangement is not None:
            donor = derangement.donor(name)
        else:
            donor = name
        # The digests come from the DERANGEMENT artifact whenever there is one: that is what the gate
        # cross-checks, and a hand-written digest would not be what the cohort's records hold.
        if derangement is not None and donor is not None:
            digest = derangement.action_sha256[donor]
        else:
            digest = hashlib.sha256(str(donor).encode()).hexdigest()
        drawn = name if draw_key_for is None else draw_key_for(name)
        rows[str(name)] = {
            "ssim": float(ssim(index)) if callable(ssim) else float(ssim),
            "mse": float(mse),
            "actions_from": donor,
            "actions_sha256": digest,
            "draw_key_sha256": ev.draw_key_digest(ev.evaluation_draw_key(drawn)),
            "num_steps": ev.DEPLOYED_SAMPLING_STEPS,
        }
    return ev.build_score_table(
        rows=rows,
        cohort=cohort,
        condition=condition,
        arm=arm,
        checkpoint=_identity(run=run, step=10000, source="selection") if checkpoint is None else checkpoint,
        num_steps=ev.DEPLOYED_SAMPLING_STEPS,
        derangement_sha256=derangement.fingerprint if (condition == "wrong" and derangement is not None) else None,
        allow_incomplete=incomplete,
    )


def attack_t5b_lower_the_bar():
    """Can the +0.05 margin or the CI condition be relaxed from the outside?

    **F8 — dead since 2026-08-09 (`76117df`).** Both calls handed the gate naked mappings and died
    inside ``as_gate_table``'s type refusal *before either half of the attack ran*; the ``TypeError``
    was filed as "no margin argument exists", a claim about production the probe had not tested.
    With real artifacts the two halves finally measure the MARGIN rather than the argument type.

    The second half is the one that matters: +0.04 with a spotless CI is exactly the run that would
    want a caller-supplied margin to exist. **The old body also mis-scored its own first half** — an
    ACCEPTED override was appended to the notes and the verdict was still taken from the second call,
    so the attack could have succeeded and reported REFUSED. It now returns on that branch.
    """
    g, cohort, names = _gate_env()

    def arms():
        return dict(
            rollout=_gate_table(names, 0.34, cohort=cohort),
            control=_gate_table(names, 0.30, cohort=cohort, arm="control", run="c0"),
            cohort=cohort,
        )

    notes = []
    try:
        g.primary_gate(**arms(), margin=0.01)
    except TypeError as error:
        notes.append(f"no margin argument exists ({str(error).splitlines()[0][:52]})")
    else:
        return "SUCCEEDED: the primary margin was overridden by a caller"
    verdict = g.primary_gate(**arms())
    if verdict.passed:
        return f"SUCCEEDED: a +0.04 delta passed the +{g.PRIMARY_MARGIN} gate; " + "; ".join(notes)
    ci = [round(float(value), 4) for value in verdict.numbers["ci"]]
    notes.append(f"+0.04 at CI {ci} (CI-low clean) still failed on {list(verdict.reasons)}")
    return "REFUSED: " + "; ".join(notes)


def attack_t5b_score_test_first(tmp):
    """Can TEST be scored without a passing DEV gate?

    **F8 — dead since 2026-08-09 (`76117df`).** Two signatures moved at once: ``dev_certificate``
    stopped accepting a caller's ``GateVerdict`` (it COMPUTES the gate and publishes to a path), and
    ``confirm_on_test`` takes a certificate **path** plus a ``TestCohort``. The probe died on the
    mapping tables before it ever reached the TEST door.

    Re-expressed with both attempts made real, and the second one sharpened. The old "hand-written
    pass" was ``{"passed": True}``, which `G3-5` already covers as a bare marker. The interesting
    forgery now is a certificate that is **complete, correctly digested and internally
    well-formed** — production's own failing certificate with ``passed`` flipped to True and its
    ``reasons`` erased, republished so the file's digest describes its own payload. Nothing about it
    is detectable by integrity checking; only re-deciding the verdict from its own numbers catches it.
    """
    from maxdiffusion import eval_wan_pos_rollout as ev

    g, cohort, names = _gate_env()
    test_cohort = g.load_test_cohort(TEST)
    honest = str(pathlib.Path(tmp) / "dev_cert_failing.json")
    # A genuinely failing DEV gate, computed by production from real tables: R-B does not beat C0.
    published = g.dev_certificate(
        honest,
        rollout=_gate_table(names, 0.30, cohort=cohort),
        control=_gate_table(names, 0.30, cohort=cohort, arm="control", run="c0"),
        cohort=cohort,
    )
    forged_payload = {
        **{key: value for key, value in published.items() if key != "sha256"},
        "passed": True,
        "reasons": [],
    }
    forged = str(pathlib.Path(tmp) / "dev_cert_forged.json")
    ev.publish_certificate(forged, forged_payload)  # recomputes the digest: the file is self-consistent
    # BOTH refusals are recorded, not just the last one: each is a different production rule, and a
    # probe that prints only its final attempt hides which door actually held (the T5a-2 repair).
    refusals = []
    for label, path in (("the gate's own failing certificate", honest), ("a digest-consistent forged pass", forged)):
        try:
            g.confirm_on_test(path, test_cohort=test_cohort, derangement=None, tables={}, control_tables={})
        except (TypeError, ValueError) as error:
            reason = str(error).splitlines()[0]
            refusals.append(f"{label} -> {reason.split(': ', 1)[-1][:78]}")
        else:
            return f"SUCCEEDED: TEST was scored behind {label}"
    return "REFUSED: " + "; ".join(refusals)


def attack_t5b_forge_the_derangement():
    """Can a wrong-action assignment secretly hand examples their own actions back?

    **F8 — dead since 2026-08-09 (`76117df`).** ``cohort_derangement`` became ``cohort_derangement(cohort)``
    returning a :class:`DerangementArtifact`: it reads the cohort's own action bytes, so the legality
    of the shuffle is measured. The old call passed ``names`` positionally **and** ``cohort=`` and
    died on the argument binding before any forgery was attempted.

    Re-expressed at full strength rather than as its dead letter. A naive rewrite of the permutation
    is caught by the fingerprint, which would make this a test of tamper detection instead of the
    fixed-point rule — so the forgery **recomputes the fingerprint after the edit**. The artifact that
    reaches production is internally consistent: its hash correctly describes its own permutation and
    digests, and the only thing wrong with it is that one example is its own donor. The score tables
    are built under the forged mapping as well, so a relaxation of the artifact check would still
    have to get past the row-level identity checks inside the gate.
    """
    import dataclasses as dc

    g, cohort, names = _gate_env()
    with _cohort_records(cohort):
        honest = g.cohort_derangement(cohort)
        sneaky = dc.replace(honest, permutation={**honest.permutation, names[0]: names[0]})
        sneaky = dc.replace(sneaky, fingerprint=g.derangement_fingerprint(sneaky))
        if g.derangement_fingerprint(sneaky) != sneaky.fingerprint:
            return "SUCCEEDED: the harness failed to build the self-consistent forgery it means to test"
        try:
            g.action_use_gate(
                true_table=_gate_table(names, 0.36, cohort=cohort),
                wrong_table=_gate_table(names, 0.30, cohort=cohort, condition="wrong", derangement=sneaky),
                cohort=cohort,
                derangement=sneaky,
            )
        except (TypeError, ValueError) as error:
            return (
                f"REFUSED (fingerprint re-derived, so this is not tamper detection): {str(error).splitlines()[0][:78]}"
            )
        return f"SUCCEEDED: {names[0]} was scored against its own actions as the wrong-action row"


def attack_t5b_swap_arm_and_control():
    """Can the claim and its control be swapped by position?"""
    g, cohort, names = _gate_env()
    try:
        g.primary_gate(_tbl(names, 0.30), _tbl(names, 0.36), cohort)
    except TypeError as error:
        return f"REFUSED: {str(error).splitlines()[0][:110]}"
    return "SUCCEEDED: positional arguments let the control be reported as the arm"


def _arm_battery(g, cohort, names, derangement, *, arm="rollout", run="rb", values=(0.36, 0.30, 0.20, 0.10)):
    """The arm's four legal condition tables — the shape `action_use_report` requires (F8)."""
    true_ssim, wrong_ssim, zero_ssim, disabled_ssim = values
    common = dict(cohort=cohort, arm=arm, run=run, derangement=derangement)
    return {
        "true": _gate_table(names, true_ssim, condition="true", **common),
        "wrong": _gate_table(names, wrong_ssim, condition="wrong", **common),
        "zero": _gate_table(names, zero_ssim, condition="zero", **common),
        "adapter_disabled": _gate_table(names, disabled_ssim, condition="adapter_disabled", **common),
    }


def attack_t5b_drop_the_control_battery():
    """Can 'the adapter uses its actions' be published without matched-C0's own battery?

    **F8 — dead since 2026-08-09 (`76117df`).** Two signatures moved at once: ``cohort_derangement``
    became ``cohort_derangement(cohort)`` (the probe passed ``names`` positionally *and* ``cohort=``),
    and ``action_use_report`` takes the arm's four conditions as ONE ``tables`` mapping rather than
    four keyword arguments. It died on the derangement call, before the report was ever asked for.

    Re-expressed with a COMPLETE and legal arm battery, built under the cohort's real derangement, so
    the refusal that fires is the one about matched-C0's missing battery rather than a complaint about
    the arm's own coverage. Publishing "the adapter uses its actions" without the control is publishing
    it without the comparison that says whether rollout training uses actions MORE than one-step does.
    """
    g, cohort, names = _gate_env()
    with _cohort_records(cohort):
        art = g.cohort_derangement(cohort)
        try:
            g.action_use_report(cohort, derangement=art, tables=_arm_battery(g, cohort, names, art), control_tables={})
        except (TypeError, ValueError) as error:
            return f"REFUSED: {str(error).splitlines()[0][:110]}"
        return "SUCCEEDED: the action-use finding was published without matched-C0's comparison"


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
    """A LEGITIMATE measurement of a fitting cell -- the honest input every probe here builds on.

    **F10 (Yixun's Option A) moved what "legitimate" means, so this moved with it.** The authorized
    bound is the cell's compiled memory analysis and the runtime watermark is a recorded cross-check
    that can only refuse, so a measurement carrying no analysis authorizes nothing whatever its peak
    source says. The shape below is M1-9's measured one: analysis == reported peak, watermark well
    under it. Leaving the old shape here would have turned nine probes into false refusals against a
    production refusing their SETUP rather than their attack -- the fifth caution, again.
    """
    values = dict(
        cell=fp.FitCell(arm, microbatch, k_b),
        context_digest=(context or _ctx(fp)).binding_digest(),
        compile_seconds=480.0,
        step_seconds=3.5,
        eval_seconds=600.0,
        checkpoint_seconds=90.0,
        peak_bytes=20 * 1024**3,
        capacity_bytes=32 * 1024**3,
        reservation_failures=0,
        peak_source=fp.PEAK_SOURCE_ANALYSIS,
        peak_attribution=fp.PEAK_ATTRIBUTION_NONE,
        watermark_bytes=4 * 1024**3,
        watermark_before_bytes=3 * 1024**3,
    )
    values.update(over)
    # The bound tracks the peak unless a probe sets it: every `peak_bytes=` attack in this file was
    # written to move the number the headroom rule reads, and since F10 that number is the analysis.
    values.setdefault("analysis_bytes", values["peak_bytes"])
    return fp.CellMeasurement(**values)


def _auth(fp, tmp, measurements, name="auth.json", context=None):
    context = context or _ctx(fp)
    evidence = fp.build_evidence(
        context, measurements, max_train_steps=10_000, eval_every=1_000, checkpoint_every=1_000
    )
    return fp.publish_authorization(str(pathlib.Path(tmp) / name), evidence)


def attack_t7_run_an_unmeasured_cell(tmp):
    """Can a training run reach an (arm, microbatch, k) cell M1 never measured?"""
    from maxdiffusion.trainers.wan_pos_rollout_trainer import WanPosRolloutTrainer

    fp = _probe_env()
    path = str(pathlib.Path(tmp) / "t7a.json")
    config = _pos_config(
        pos_fit_authorization=path,
        pos_rollout_k=4,
        pos_recipe_lock=str(pathlib.Path(tmp) / "t7a_lock.json"),
        pos_resume_parent=str(pathlib.Path(tmp) / "t7a_attempts"),
        checkpoint_dir=str(pathlib.Path(tmp) / "t7a_attempts/att-X/checkpoints"),
    )
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
        fp.assert_cell_authorized(
            {"authorized_cells": [{"arm": "rollout", "microbatch": 64, "k_b": 4}]},
            fp.FitCell("rollout", 64, 4),
            context=_ctx(fp),
        )
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
    published = _auth(
        fp, tmp, [_fit(fp, context, microbatch=64, peak_bytes=31 * 1024**3)], name="t7d.json", context=context
    )
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
        fp.project_wall_clock(
            _fit(fp),
            max_train_steps=10_000,
            eval_every=1_000,
            checkpoint_every=1_000,
            eval_seconds=1.0,
            checkpoint_seconds=1.0,
        )
        notes.append("caller-supplied overheads ACCEPTED")
    except TypeError:
        notes.append("there is no overhead argument to supply")
    try:
        fp.project_wall_clock(
            _fit(fp, peak_bytes=31 * 1024**3), max_train_steps=10_000, eval_every=1_000, checkpoint_every=1_000
        )
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
    env = {
        "PATH": str(root / "bin"),
        "HOME": str(root / "home"),
        "POS_DEVICE_COUNT": "8",
        "SHIM_RECORD": str(root / "record.txt"),
        "COMMIT": _COMMIT,
    }
    env.update({k: str(v) for k, v in env_over.items() if not k.startswith("_")})
    proc = subprocess.run(
        ["/bin/bash", f"bash_scripts/{script}"], cwd=root, env=env, capture_output=True, text=True, timeout=300
    )
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
    common = dict(RUN_NAME="m3", ATTEMPT="att-X", OUTPUT_DIR="gs://b/p", POS_FIT_AUTHORIZATION="gs://b/m1.json")
    _, left = _launch(tmp, "train_wan_pos_rollout.sh", POS_ROLLOUT_ARM="rollout", **common)
    _, right = _launch(tmp, "train_wan_pos_rollout.sh", POS_ROLLOUT_ARM="one_step", **common)
    if left.get("checkpoint_dir") == right.get("checkpoint_dir"):
        return f"SUCCEEDED: both arms write and restore {left.get('checkpoint_dir')}"
    return (
        f"REFUSED: rollout -> {left['checkpoint_dir'].split('/p/')[-1]} vs "
        f"one_step -> {right['checkpoint_dir'].split('/p/')[-1]}"
    )


def attack_p3_divergent_second_arm(tmp):
    """T6-1's other half: two submissions a day apart, differing in a seed. No shell can see that."""
    from maxdiffusion.trainers.wan_pos_rollout_trainer import publish_recipe_lock

    path = str(pathlib.Path(tmp) / "p3_lock.json")
    publish_recipe_lock(path, _pos_config(pos_rollout_arm="rollout"), arm="rollout")
    try:
        publish_recipe_lock(path, _pos_config(pos_rollout_arm="one_step", seed=7, learning_rate=1e-3), arm="one_step")
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
    for script, extra in (
        ("train_wan_pos_rollout.sh", {"POS_FIT_AUTHORIZATION": "gs://b/m1.json"}),
        ("eval_wan_pos_rollout.sh", {}),
    ):
        _, overrides = _launch(
            tmp,
            script,
            ARTIFACT_ROOT="gs://bucket/flat",
            CHECKPOINT_DIR="gs://bucket/flat",
            OUTPUT_DIR="gs://b/p",
            RUN_NAME="m3",
            ATTEMPT="att-X",
            **extra,
        )
        flattened = overrides.get("base_output_directory") == "gs://bucket/flat"
        notes.append(f"{script.split('_')[0]}={'FLAT' if flattened else 'derived'}")
    return ("SUCCEEDED: " if "FLAT" in " ".join(notes) else "REFUSED: ") + ", ".join(notes)


def attack_p3_adopt_an_incomplete_or_foreign_publication(tmp):
    """T6-3: 'select only the latest COMPLETE checkpoint whose recorded SHA matches the running code'.

    **F5d: this probe silently stopped executing for a whole round.** F5c gave
    `select_resume_publication` a required `context_digest` keyword; this call site was not updated,
    the resulting `TypeError` was caught by `_report`, and the probe scored
    `REFUSED (TypeError): ... missing 1 required keyword-only argument`. A standing attack had not run
    since the signature changed, and the summary counted it as coverage.

    So the call is fixed AND the failure mode is made loud locally: a `TypeError` from the selector is
    the harness failing to execute the attack, not production refusing it, and it is reported as such.
    """
    from maxdiffusion.trainers import wan_pos_rollout_trainer as tr

    parent = str(pathlib.Path(tmp) / "p3_attempts")
    tr.publish_attempt(
        parent,
        attempt="att-mine",
        arm="rollout",
        code_sha="a" * 40,
        context_digest="d" * 64,
        binding_digest="d" * 64,
        step=1000,
        checkpoint_dir=f"{parent}/att-mine/checkpoints",
    )
    tr.publish_attempt(
        parent,
        attempt="att-foreign-build",
        arm="rollout",
        code_sha="b" * 40,
        context_digest="e" * 64,
        binding_digest="e" * 64,
        step=9000,
        checkpoint_dir=f"{parent}/att-foreign-build/checkpoints",
    )
    tr.publish_attempt(
        parent,
        attempt="att-other-arm",
        arm="one_step",
        code_sha="a" * 40,
        context_digest="d" * 64,
        binding_digest="d" * 64,
        step=9000,
        checkpoint_dir=f"{parent}/att-other-arm/checkpoints",
    )
    (pathlib.Path(parent) / "att-crashed" / "checkpoints").mkdir(parents=True)
    chosen = tr.select_resume_publication(parent, arm="rollout", binding_digest="d" * 64, code_sha="a" * 40)
    if chosen is None:
        return "SUCCEEDED: nothing was adopted at all -- the probe is no longer exercising the selector"
    if chosen["attempt"] != "att-mine":
        return f"SUCCEEDED: adopted {chosen['attempt']} at step {chosen['step']}"
    # F5c added a fourth filter; the probe now also proves a foreign CONTEXT at this SHA is skipped.
    unpublished_build = tr.select_resume_publication(parent, arm="rollout", binding_digest="f" * 64, code_sha="a" * 40)
    if unpublished_build is not None:
        return f"SUCCEEDED: a build nobody published adopted {unpublished_build['attempt']}"
    return (
        "REFUSED: selection logic chose only att-mine (step 1000); the 9000-step foreign-BUILD and other-arm "
        "trees and a foreign context were all skipped"
    )


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
    """T7-3: the same cell published once fitting and once at 96.9% with a reservation failure.

    **F10b:** the two trials as written also disagree about the cell's compiled analysis, which
    production now refuses at aggregation instead of collapsing (``analysis_disagreement``). Both
    outcomes are refusals of the same attack -- the cell must not reach a training run -- so the
    probe accepts either and says which one it got, rather than pinning the mechanism.
    """
    fp = _probe_env()
    context = _ctx(fp)
    missed = int(32 * 1024**3 * 0.969)
    try:
        published = _auth(
            fp,
            tmp,
            [_fit(fp, context), _fit(fp, context, peak_bytes=missed, analysis_bytes=missed, reservation_failures=1)],
            name="p3_dup.json",
            context=context,
        )
    except ValueError as error:
        return f"REFUSED at aggregation: {str(error).splitlines()[0][:110]}"
    try:
        fp.assert_cell_authorized(published, fp.FitCell("rollout", 32, 2), context=context)
    except ValueError as error:
        return f"REFUSED: {str(error).splitlines()[0][:130]}"
    return f"SUCCEEDED: authorized={published['authorized_cells']} refused={published['refused_cells']}"


def attack_p3_m1_cannot_be_run(tmp):
    """T7-4: `run_fit_probe` walked no ladder and the launcher had no probe mode -- M1 was unrunnable."""
    fp = _probe_env()
    proc, overrides = _launch(
        tmp,
        "train_wan_pos_rollout.sh",
        POS_JOB_MODE="fit_probe",
        RUN_NAME="m1",
        ATTEMPT="att-X",
        OUTPUT_DIR="gs://b/p",
    )
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
    return (
        f"REFUSED (M1 runs): launcher probe mode={'yes' if ran else 'no'}, ladder walked {len(seen)} cells, "
        f"published {len(published['authorized_cells'])} authorized cells"
    )


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
        instrument.score_dev_cohort(
            cohort,
            lambda *a, **k: (1.0, {}),
            forged,
            params=1.0,
            context=type("C", (), {"num_steps": 25, "k_b": 2})(),
            example_shape=(4, 3, 4, 6),
        )
        return "SUCCEEDED: " + "; ".join(notes) + "; scoring accepted a decoder"
    except TypeError:
        notes.append("score_dev_cohort has no batch source parameter")
    return "REFUSED: " + "; ".join(notes)


def attack_p1_poison_selection_with_nan():
    """A single NaN early in training became an unbeatable running best that no later finite value
    could displace, while preserve_selection still replaced the sibling."""
    from maxdiffusion import pos_rollout_loop as pl

    history = [
        pl.EvalRecord(step=1000, dev_metric=float("nan"), train_metric=1.0),
        pl.EvalRecord(step=2000, dev_metric=0.1, train_metric=0.9),
    ]
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
    return (
        f"REFUSED: pathlib would give {dropped!r}; storage layer treats it as remote={guarded}"
        if guarded
        else f"SUCCEEDED: {dropped}"
    )


#: Three verdicts, counted separately (review F5c). REFUSED: production stopped the attack. DECLARED:
#: the attack SUCCEEDS BY DESIGN, inside the trust boundary this campaign has explicitly accepted and
#: written down -- it is not a refusal and must never be counted as one; if the boundary claim ever
#: changes, or authentication is added, it flips to SUCCEEDED (or the probe is rewritten) and the
#: change is forced to be deliberate. SUCCEEDED: a defect in production.
_VERDICTS: list[str] = []

#: **Which probes may say DECLARED — an allowlist, not a keyword** (review F5d, BLOCKER).
#: `_report` used to accept any return beginning with "DECLARED" and `_summarize` labelled it an
#: accepted residual, so a probe drifting into that word -- by accident or by edit -- could relabel a
#: real defect as a known one. The reviewer executed exactly that. An accepted residual is a decision
#: somebody made and wrote down, so it is enumerated here; a DECLARED from anywhere else is a harness
#: failure and is counted UNPARSED, loudly.
_MAY_DECLARE = frozenset({"F5-8"})


def _probe_id(label) -> str:
    return str(label).split()[0].rstrip(":")


def _report(label, fn, *args):
    """**A verdict is a RETURNED string. An escaping exception means the probe did not run.**

    Review F7d, and this is the harness's most expensive lesson. `_report` used to convert any
    exception into `REFUSED (...)`, on the reasoning that production raising is production refusing.
    That reasoning is wrong in the case that matters: when a probe calls an API whose signature or
    name has moved, the exception comes from the PROBE failing to execute, not from production
    refusing anything — and it landed in the pass column. The reviewer found concrete probes that had
    been dead since the F3c-era API changes, counted as coverage in every green battery since.

    So the discriminator is now structural rather than a guess about exception types: **a probe that
    means "production refused" RETURNS a verdict string saying so** — catching the production error
    itself, which also forces it to say which error it expected — **and anything that escapes the body
    is the probe's own failure.** The runner exits non-zero on it.

    This is universal by construction: every probe is invoked through here, so there is no list of
    guarded probes to keep in step (F5d guarded one call, F7c guarded a subset, and both were
    overtaken by the next signature to move).
    """
    try:
        verdict = str(fn(*args))
    except Exception as error:  # noqa: BLE001 -- an escape is a non-run, whatever its class
        verdict = (
            f"SUCCEEDED: THE PROBE DID NOT RUN -- {type(error).__name__}: "
            f"{str(error).splitlines()[0][:110]}; this attack is not covered"
        )
    if verdict.startswith("DECLARED") and _probe_id(label) not in _MAY_DECLARE:
        print(
            f"{label}: HARNESS FAILURE: {_probe_id(label)} returned DECLARED but is not on the accepted-residual "
            f"allowlist {sorted(_MAY_DECLARE)}. An accepted residual is a decision somebody wrote down, not a "
            f"word a probe can reach for. Original verdict: {verdict[:140]}"
        )
        _VERDICTS.append("UNPARSED")
        return
    print(f"{label}: {verdict}")
    for name in ("DECLARED", "SUCCEEDED", "REFUSED"):
        if verdict.startswith(name):
            _VERDICTS.append(name)
            break
    else:
        _VERDICTS.append("UNPARSED")


#: **The HONEST CONTROLS' verdicts, counted separately from the attacks' (review F8b, MAJOR 2).**
#: A control is not an attack and must never be summed with one — see :func:`_control`.
_CONTROL_VERDICTS: list[str] = []


def _control(label, fn, *args):
    """A probe's HONEST CONTROL, executed by the SAME battery run as its attack (F8b, MAJOR 2).

    **The hole this closes.** F8 revived nine dead probes and paired each with a reachability check —
    point production at the LEGITIMATE input and confirm it is *not* refused — but those checks were
    run by hand, beside the battery, and written up in the worklog. The reviewer's objection is
    exact: the recurring battery invokes only the attacks, so **a production regression that refused
    everything would still print nine green REFUSED lines**. A refusal only means something if the
    same code path can be shown to accept something. Unexecuted evidence is the F7d lesson wearing a
    different hat, and it took four days to learn the first time.

    **Why controls get their own verdict words rather than being folded into `_report`.** A control
    that fails is not "the attack succeeded" — production has not let anything through, it has
    stopped letting legitimate work through, which is a different defect with a different fix. Two
    rounds of this campaign were lost to a probe reporting one thing while its name claimed another
    (`F5-5`, `T5a-2`), so the vocabulary stays honest: attacks say REFUSED/DECLARED/SUCCEEDED,
    controls say CONTROL-PASSED/CONTROL-REFUSED, and :func:`_summarize` counts them on separate
    lines. The runner exits non-zero on either kind of failure.

    The did-not-run guard is inherited verbatim from `_report`: a verdict is a RETURNED string, and
    anything that escapes the body is the control's own failure, scored CONTROL-REFUSED so a control
    can never go quiet the way the nine probes did.
    """
    try:
        verdict = str(fn(*args))
    except Exception as error:  # noqa: BLE001 -- an escape is a non-run, whatever its class
        verdict = (
            f"CONTROL-REFUSED: THE CONTROL DID NOT RUN -- {type(error).__name__}: "
            f"{str(error).splitlines()[0][:110]}; this probe's refusal is now unwitnessed"
        )
    print(f"{label}: {verdict}")
    for name in ("CONTROL-PASSED", "CONTROL-REFUSED"):
        if verdict.startswith(name):
            _CONTROL_VERDICTS.append(name)
            break
    else:
        _CONTROL_VERDICTS.append("UNPARSED")


def _summarize():
    """The honest headline. A single "N refused" number hid a false refusal for two rounds.

    Since F8b it is TWO headlines, because the battery now runs two kinds of thing and summing them
    would be the same category error the three-way verdict split was introduced to stop.
    """
    counts = {name: _VERDICTS.count(name) for name in ("REFUSED", "DECLARED", "SUCCEEDED", "UNPARSED")}
    print(
        f"\nSUMMARY: {len(_VERDICTS)} probes -- {counts['REFUSED']} REFUSED, {counts['DECLARED']} DECLARED "
        f"(accepted residual, see the harness README), {counts['SUCCEEDED']} SUCCEEDED, "
        f"{counts['UNPARSED']} UNPARSED"
    )
    controls = {name: _CONTROL_VERDICTS.count(name) for name in ("CONTROL-PASSED", "CONTROL-REFUSED", "UNPARSED")}
    print(
        f"SUMMARY: {len(_CONTROL_VERDICTS)} honest controls -- {controls['CONTROL-PASSED']} CONTROL-PASSED, "
        f"{controls['CONTROL-REFUSED']} CONTROL-REFUSED, {controls['UNPARSED']} UNPARSED. A control asserts "
        f"production still ACCEPTS the legitimate case, so its probe's REFUSED means something."
    )
    failed = False
    if counts["SUCCEEDED"] or counts["UNPARSED"]:
        print("SUMMARY: FAILED -- a SUCCEEDED or UNPARSED line is production-guilty until you have read the probe.")
        failed = True
    if controls["CONTROL-REFUSED"] or controls["UNPARSED"]:
        print(
            "SUMMARY: FAILED -- a CONTROL-REFUSED line means production stopped accepting the legitimate case, so "
            "every refusal it witnesses is now worthless. That is a defect too, and a different one."
        )
        failed = True
    return not failed


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
            if not self._remote(path):
                return pathlib.Path(str(path)).exists()
            # A bucket has no directories: a prefix exists when an object lives under it, which is
            # what tensorflow's GCS filesystem reports. F5's adoption scan lists prefixes, and a fake
            # that only knew exact object names would make every scan report an empty tree.
            key = str(path)
            return key in blobs or any(blob.startswith(key.rstrip("/") + "/") for blob in blobs)

        def listdir(self, path):
            if not self._remote(path):
                return sorted(child.name for child in pathlib.Path(str(path)).iterdir())
            prefix = str(path).rstrip("/") + "/"
            return sorted({blob[len(prefix) :].split("/", 1)[0] for blob in blobs if blob.startswith(prefix)} - {""})

        def rename(self, source, destination, overwrite=False):
            # F5 publishes by staging and renaming, so a fake without this makes every F5 probe
            # report REFUSED for the wrong reason -- the exact staleness this harness's README warns
            # about ("a probe that goes stale reports a defect it cannot see").
            if not self._remote(source):
                pathlib.Path(str(source)).replace(str(destination))
                return
            if str(destination) in blobs and not overwrite:
                raise FileExistsError(str(destination))
            blobs[str(destination)] = blobs.pop(str(source))

        def remove(self, path):
            if self._remote(path):
                blobs.pop(str(path), None)
            else:
                pathlib.Path(str(path)).unlink(missing_ok=True)

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


def _rows(names, num_steps=25, grid_sha256=None):
    """Anchor sample rows that are legal in every respect the caller is not deliberately breaking.

    **F8b: this helper omitted ``grid_sha256``, and writing the anchor family's honest control is
    what exposed it.** ``summarize_samples`` checks the grid BEFORE the horizon and long before
    ``reproduce_anchor`` ever sees a name, so every probe built on `_summary` was refused with
    ``these samples were rolled out on grids ['']`` — and `G3-1 foreign names`, `G3-2 wrong order`,
    `G3-3 foreign checkpoint` and `G3-4 short rollout` were all scored REFUSED for a reason that has
    nothing to do with what they are named for. Three probes green, zero coverage of the rules they
    claim, which is the fourth caution (`F5-5`) in three more places.

    The rows now carry the deployed grid, so each of those probes reaches its own rule. A probe that
    wants to break the grid says so explicitly.
    """
    from maxdiffusion import eval_wan_pos_rollout as ev

    return [
        {
            "name": n,
            "latent_mse": ev.HISTORICAL_ANCHOR.mean_latent_mse,
            "pixel_mse": ev.HISTORICAL_ANCHOR.mean_pixel_mse,
            "ssim_avg": ev.HISTORICAL_ANCHOR.mean_ssim,
            "num_steps": num_steps,
            "grid_sha256": ev.DEPLOYED_GRID_SHA256 if grid_sha256 is None else grid_sha256,
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
        g.confirm_on_test(path, test_cohort=g.load_test_cohort(TEST), derangement=None, tables={}, control_tables={})
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
    missing = [
        n
        for n in ("run_evaluation", "run_anchor_phase", "run_benchmark_phase", "run_gates_phase", "run_confirm_phase")
        if not callable(getattr(ev, n, None))
    ]
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
        config = Config({"pos_eval_phase": phase, "base_output_directory": f"gs://attack/run/eval_{phase}_att-1"})
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
        for key, value in (
            ("action_tokens", 64),
            ("pre_context_tokens", 64),
            ("flash_block_sizes", {"block_q": 1024}),
            ("latent_frames", 99),
            ("action_dim", 14),
            ("logical_axis_rules", [["a", "b"]]),
        )
        if fp.recipe_fingerprint(_pos_config(**{key: value})) == base
    ]
    return (
        f"SUCCEEDED: {blind} leave the fingerprint unchanged"
        if blind
        else "REFUSED: every graph/HBM-bearing key moves the fingerprint"
    )


def attack_f1_projection_miscounts(tmp):
    """LS-8: the loop evaluates at the cadence AND at the final step, and never reads an independent
    checkpoint cadence."""
    fp = _probe_env()
    from maxdiffusion.pos_rollout_loop import LoopSchedule, should_evaluate

    schedule = LoopSchedule(
        max_train_steps=1_001,
        eval_every=1_000,
        logical_batch=256,
        microbatch=32,
        seed=0,
        arm="rollout",
        k_b=2,
        num_steps=25,
    )
    production = sum(1 for step in range(1, 1_002) if should_evaluate(step, schedule))
    projected = fp.project_wall_clock(_fit(fp), max_train_steps=1_001, eval_every=1_000, checkpoint_every=1_000)[
        "evaluations"
    ]
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
    revisions = [fp.derive_model_revision(_pos_config(pretrained_model_name_or_path=str(d))) for d in (left, right)]
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
        fp.run_fit_probe(
            _pos_config(pos_fit_authorization=str(pathlib.Path(tmp) / "f1_m1.json")),
            devices=[_Dev()],
            cells=[fp.FitCell("rollout", 8, 2)],
            trials=1,
        )
        reaches.append("it measured")
    except NotImplementedError as error:
        return f"SUCCEEDED (M1 still dies): {str(error)[:90]}"
    except Exception as error:  # noqa: BLE001 -- reaching the real weights load is the point
        # F8b (review MINOR): this used to say "reached the real model load", which stopped being true
        # when the blind-backend refusal moved ahead of the load. A hardcoded description of WHERE
        # production stopped is a second thing to keep in step with production -- so it now QUOTES the
        # refusal instead of restating it, and cannot go stale again.
        reaches.append(f"the real path ran and refused it ({type(error).__name__}: {str(error).splitlines()[0][:64]})")
    return f"SUCCEEDED: measurer still raises" if raises else f"REFUSED: the measurement path is real -- {reaches[0]}"


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
        [
            sys.executable,
            "-m",
            "pytest",
            str(test),
            "-q",
            "-p",
            "no:cacheprovider",
            "--no-header",
            "-k",
            "real_m1_entrypoint_measures",
            "--tb=line",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=1800,
        env={**os.environ, "PYTHONPATH": "src", "JAX_PLATFORMS": "cpu"},
    )
    tail = [line for line in proc.stdout.splitlines() if line.strip()][-1:] or ["<no output>"]
    return (
        f"REFUSED (it runs): {tail[0][:110]}"
        if proc.returncode == 0
        else f"SUCCEEDED: the real entrypoint test did not pass -- {tail[0][:110]}"
    )


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
    identity = ev.CheckpointIdentity(run_name=run, step=30000, root=f"gs://b/{run}/checkpoints", source="historical")
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
    """EV-1: can a run configured at bfloat16 end up drawing its noise in float32?

    **F8 — dead since ~2026-08-09 (F3c).** The probe injected ``DeviceBackend(velocity_for=…)``, and
    **F3c removed that seam deliberately**: handing out a callable with the weights already bound is
    the spelling that let ``jax.jit`` bake the 5B backbone into the lowered module as 10.18 GB of
    literals, which killed three consecutive M1 compiles on ``TPU_VM_HEALTH_TIMEOUT``. The backend
    now exposes no bound-velocity seam at all, so there is nothing left to bind.

    **The modern equivalent of "bind a foreign velocity" is** :func:`build_rollout_kernel`. Device
    work is still injectable — that is the Planner's boundary, *device work may be injected,
    orchestration may not* — but the injection point moved INSIDE the single ``jax.jit`` boundary and
    the weights cross it as arguments rather than as a closure. So this attack is **not**
    unconstructible: it changed shape, and it is re-expressed at the new seam rather than downgraded
    to an assertion.

    Two things, both EXECUTED against production rather than read off its source:

    * **The removed spelling really is gone.** The constructors are interrogated, so a re-added
      ``velocity_for`` — or a ``velocity_fn`` back on ``rollout_prediction`` — re-arms this probe
      instead of silently restoring the capture defect F3c paid three dead compiles to find.
    * **EV-1 itself.** A bf16-configured backend is handed float32 latents, actions and context (the
      cached reader really does hand this evaluator float32) and must cast BEFORE it draws, because
      :func:`initial_latents` draws in ``z_video.dtype``. Casting after the draw is a different
      measurement: T5a's A13 finding measured native ``[0.387, 0.183, -1.0]`` against fp32-routed
      ``[1.625, 2.031, -0.434]``, so the anchor would fail on WIRING and be read as model quality.
    """
    import inspect as _inspect

    import jax.numpy as jnp
    import numpy as np

    from maxdiffusion import eval_wan_pos_rollout as ev

    restored = [
        name
        for name, function in (("velocity_for", ev.DeviceBackend.__init__), ("velocity_fn", ev.rollout_prediction))
        if name in _inspect.signature(function).parameters
    ]
    if restored:
        return f"SUCCEEDED: the pre-bound velocity seam is back ({restored}); F3c removed it to stop the capture"

    seen = {}

    def velocity_builder(params, frozen_state, actions, adapter_enabled):
        """The kernel's OWN injection point — what `velocity_for` became when it moved inside jit."""
        seen["actions"] = actions.dtype
        del params, frozen_state, adapter_enabled

        def velocity_fn(hidden_states, timestep, encoder_hidden_states):
            seen["latents"] = hidden_states.dtype
            seen["context"] = encoder_hidden_states.dtype
            del timestep
            return jnp.zeros_like(hidden_states)

        return velocity_fn

    sigmas, timesteps = ev.deployed_grid()
    params = {"w": jnp.zeros((1,), jnp.float32)}
    backend = ev.DeviceBackend(
        kernel=ev.build_rollout_kernel(velocity_builder),
        decode_fn=lambda x: jnp.asarray(np.repeat(np.asarray(x, np.float32).mean(axis=1)[..., None], 3, axis=-1)),
        sigmas=sigmas,
        timesteps=timesteps,
        context=jnp.zeros((1, 7, 8), jnp.float32),
        guide_scale=5.0,
        params=params,
        eval_dtype=jnp.bfloat16,
        frozen_state=None,
    ).bound(params)
    execution, _ = backend.score(
        z_i0=jnp.zeros((1, 4, 1, 4, 6), jnp.float32),
        z_video=jnp.zeros((1, 4, 2, 4, 6), jnp.float32),
        actions=jnp.zeros((1, 4, 7), jnp.float32),
        key=ev.evaluation_draw_key("x"),
    )
    observed = {
        "z_pred": execution.z_pred.dtype,
        "latents": seen.get("latents"),
        "actions": seen.get("actions"),
        "context": seen.get("context"),
    }
    # A kernel that never ran would leave `seen` empty, and three of these would be None -- which
    # `!= bfloat16` would happily report as a float32 finding. That is the W4 "time a pruned scorer"
    # mistake in miniature, so the non-observation is separated from the observation.
    if any(dtype is None for dtype in observed.values()):
        return f"SUCCEEDED: the kernel never ran, so nothing was observed ({observed})"
    wrong = sorted(name for name, dtype in observed.items() if dtype != jnp.bfloat16)
    if wrong:
        return f"SUCCEEDED: {wrong} stayed float32 under a bfloat16 configuration"
    return "REFUSED: the bound backend casts latents, actions and context to bf16 before it draws the noise"


# =================================================================================================
# Round F1b — the M1-readiness review's six findings, re-run against the fixes. The reviewer's own
# probes: "boom in program build", "No room left on device", the same-size in-place byte change, and
# the JSON 2.0 -> 2 retype.
# =================================================================================================


def attack_f1b_wrong_adapter(tmp):
    """The pilot trains the UNCHANGED pre_context adapter; the config inherited `side_adapter`."""
    import yaml

    declared = yaml.safe_load(pathlib.Path("src/maxdiffusion/configs/base_wan_5b_pos_rollout.yml").read_text())[
        "action_adapter_type"
    ]
    if declared != "pre_context":
        return f"SUCCEEDED: M1 would build action_adapter_type={declared!r}, not the approved pre_context"
    fp = _probe_env()
    moved = fp.recipe_fingerprint(_pos_config(action_adapter_type="side_adapter")) != fp.recipe_fingerprint(
        _pos_config()
    )
    return f"REFUSED: the config declares pre_context and the fingerprint separates the two ({moved})"


def attack_f1b_microbatch_timed_as_update(tmp):
    """The timed unit was one microbatch; `max_train_steps` counts LOGICAL updates.

    **F8 — dead since ~2026-08-09.** The probe imported ``f1_shims`` and ``probe_f1_smoke``, two
    session-scratchpad modules that were never committed to the tree, so it died on
    ``ModuleNotFoundError`` at its first line and `_report` filed the crash as a refusal.

    **The intent is not obsolete, so this is re-expressed rather than deleted.** ``build_probe_program``
    still decides the unit M1 times, and ``step_seconds`` is still multiplied by ``max_train_steps``,
    which counts logical updates — so timing one microbatch would still understate GBS-256
    computation by 4-32x and never make the accumulation state resident.

    The two dead modules' maintained successors live in the canonical suite
    (``maxdiffusion.tests.worklogs_yixun.test_pos_rollout_fit_probe``): ``_install_import_shims``,
    ``_TinySource`` — the WEIGHTS seam at test dimensions — and ``_tiny_probe_config``. They are
    IMPORTED rather than copied in here deliberately: a hand-rolled tiny backbone living in the
    harness is exactly the "copy that agrees by coincidence" W1's ``build_adapter_stack`` finding was
    about, and it would drift from production the first time the seam moved. The cost is that this
    one probe needs the test package importable, which it is wherever the canonical suite runs.
    """
    from maxdiffusion.tests.worklogs_yixun.test_pos_rollout_fit_probe import (
        _TinySource,
        _install_import_shims,
        _tiny_probe_config,
    )

    fp = _probe_env()
    _install_import_shims()
    logical, microbatch = 8, 2
    expected = logical // microbatch
    config = _tiny_probe_config(pathlib.Path(tmp), pos_logical_batch=logical, pos_microbatch=microbatch)
    program = fp.build_probe_program(config, fp.FitCell("rollout", microbatch, 2), model_source=_TinySource())
    parts = len(program.batch) if isinstance(program.batch, tuple) else 1
    width = program.eval_batch["z_video"].shape[0] if program.eval_batch is not None else None
    if parts != expected:
        return f"SUCCEEDED: the timed unit is {parts} of the {expected} microbatches in one logical batch"
    if len(program.draws) != parts:
        return f"SUCCEEDED: {parts} microbatches were built but carry {len(program.draws)} draws"
    if width != 1:
        return f"SUCCEEDED: the evaluation unit is batch-{width}, not the DEV instrument's own batch-one"
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
        n
        for n in _ast.parse(pyconfig.read_text()).body
        if isinstance(n, _ast.ClassDef) and n.name == "HyperParameters"
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
    """A cell whose two accounts of itself DISAGREE must not be authorized on the friendlier one.

    **The premise of the original probe was retired by F10 and the attack was re-expressed, not
    deleted (the seventh caution).** W1's version asserted that a compiled analysis can never
    authorize, because it was a LOWER bound against a 90% CEILING rule. Yixun's Option A (plan v2.9
    §4-P1) inverts that: F3 made the frozen backbone an explicit argument so the analysis counts it,
    M1-9 measured the analysis running 2-7x above the allocator's watermark on all twelve cells, and
    the analysis is now the conservative bound the rule reads. Asserting the old conclusion here
    would be testing a contract production no longer has.

    What is still an attack -- and is the SAME attack, one layer down -- is a cell asking to be
    authorized while its own recorded numbers contradict each other: a 30 GiB standing watermark
    beside a 7 GiB claimed bound. One of the two is wrong about this cell, and authorizing it means
    picking the friendlier one. Production must refuse, and name the conflict.
    """
    fp = _probe_env()

    class _NoReset(fp.DeviceTelemetry):
        def reset_peak(self):
            return False

        def peak_and_capacity(self):
            return 30 * 1024**3, 32 * 1024**3

    telemetry = _NoReset()
    before = telemetry.begin_steady_state()
    evidence = telemetry.steady_state_evidence(before, telemetry.close_steady_state(before), program_bytes=7 * 1024**3)
    measurement = _fit(
        fp,
        peak_bytes=evidence.peak_bytes,
        capacity_bytes=evidence.capacity_bytes,
        peak_source=evidence.peak_source,
        peak_attribution=evidence.peak_attribution,
        analysis_bytes=evidence.analysis_bytes,
        watermark_bytes=evidence.watermark_bytes,
        watermark_before_bytes=evidence.watermark_before_bytes,
    )
    verdict = fp.cell_verdict(measurement)
    if verdict.fits:
        return (
            f"SUCCEEDED: a {evidence.analysis_bytes // 1024**3}GiB claimed bound under a "
            f"{int(evidence.watermark_bytes) // 1024**3}GiB standing mark authorized the cell"
        )
    if "watermark_exceeds_analysis" not in verdict.reasons:
        return f"SUCCEEDED: refused for {verdict.reasons} -- the conflict between the two numbers was not named"
    return f"REFUSED: {verdict.reasons} -- a mark above the claimed bound falsifies the bound"


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
    """M1 must build the adapter through the SHARED factory, dtypes and precision included.

    **F8 — dead since ~2026-08-09 (W3).** The probe read the source of
    ``ProductionModelSource.build``, and **W3 removed that method**: the source is the WEIGHTS seam
    and nothing more (``load`` returns the shared ``LoadedBackbone`` from ``load_backbone``), while
    the adapter is finalized inside ``build_training_program``, which the live trainer and M1 both
    enter. The probe died on ``AttributeError`` and `_report` filed the crash as a refusal.

    Re-expressed against the current seam **and upgraded from an AST check to a BEHAVIOURAL one**,
    for the reason the canonical suite's own strengthening battery recorded (G07): the first version
    of this test grepped for ``optax.adamw(``, and a private optimizer spelled ``_o.adamw(`` walked
    straight past it. A source string is not the property; the same objection retires the old
    ``"dtype=" in source`` check below it.

    Two halves, both executed against the real construction:

    * **M1 enters the shared factory.** ``build_adapter_stack`` is instrumented and M1's program is
      built; an adapter M1 rolled by hand would simply never call it.
    * **The reviewer's actual W1 finding.** The probe's and the trainer's constructions "agreed only
      by coincidence of the pinned defaults", so the day someone sets ``activations_dtype: float32``
      for a debugging run M1 would measure a bf16 adapter and authorize an fp32 one. M1's program is
      therefore built at TWO different dtypes and the adapter parameters it would measure must
      actually differ — which is the property, rather than the spelling of an argument.
    """
    import jax

    from maxdiffusion import pos_rollout_update
    from maxdiffusion.tests.worklogs_yixun.test_pos_rollout_fit_probe import (
        _TinySource,
        _install_import_shims,
        _tiny_probe_config,
    )

    fp = _probe_env()
    _install_import_shims()

    def build_m1_program(dtype, recorder=None):
        config = _tiny_probe_config(
            pathlib.Path(tempfile.mkdtemp(dir=tmp)), weights_dtype=dtype, activations_dtype=dtype
        )
        real = pos_rollout_update.build_adapter_stack
        if recorder is not None:

            def watched(*args, **kwargs):
                recorder.append(args)
                return real(*args, **kwargs)

            pos_rollout_update.build_adapter_stack = watched
        try:
            return fp.build_probe_program(config, fp.FitCell("rollout", 2, 2), model_source=_TinySource())
        finally:
            pos_rollout_update.build_adapter_stack = real

    entered = []
    programs = {"float32": build_m1_program("float32", entered), "bfloat16": build_m1_program("bfloat16")}
    if not entered:
        return "SUCCEEDED: M1 built its program without ever entering the shared adapter factory"
    dtypes = {
        label: sorted({str(leaf.dtype) for leaf in jax.tree.leaves(program.params)})
        for label, program in programs.items()
    }
    if dtypes["float32"] == dtypes["bfloat16"]:
        return f"SUCCEEDED: the adapter M1 measures is {dtypes['float32']} at BOTH configured dtypes"
    return (
        f"REFUSED: M1 enters build_adapter_stack ({len(entered)}x) and the adapter it measures follows the "
        f"config ({dtypes['float32']} vs {dtypes['bfloat16']})"
    )


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
        return f"SUCCEEDED: env-read={reads_env} derived={bool(derived)} emitted={emitted} guarded={guarded}"

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
    """M1 measuring a program whose placements differ from training's.

    W5: this EXECUTES the placement contract instead of grepping for one spelling of it. The previous
    version matched the literal `jax.device_put(adapter_params, replicated)` and reported SUCCEEDED
    against correct code the moment W4 refactored that line — a probe that goes stale reports a defect
    it cannot see.
    """
    import ast as _ast
    import inspect as _inspect
    import textwrap as _textwrap

    import jax
    import jax.numpy as jnp
    import numpy as _np
    from jax.sharding import NamedSharding, PartitionSpec

    from maxdiffusion import pos_rollout_fit_probe as fp
    from maxdiffusion import pos_rollout_update as shared

    mesh = jax.sharding.Mesh(_np.array(jax.devices()).reshape(1, 1, 1, 1), ("data", "fsdp", "context", "tensor"))
    want_rep = NamedSharding(mesh, PartitionSpec())
    want_batch = NamedSharding(mesh, PartitionSpec(mesh.axis_names))
    batch = ({"z_video": jnp.zeros((1, 2, 2)), "z_i0": jnp.zeros((1, 2, 2))},)
    draws = ((jnp.asarray(0), jnp.asarray(2), jnp.zeros((1, 2, 2)), jnp.zeros((1,), jnp.int32)),)
    params, opt_state, placed_batch, placed_draws = shared.place_step_inputs(
        mesh,
        params={"w": jnp.zeros((2,))},
        opt_state={"mu": jnp.zeros((2,))},
        micro_batches=batch,
        micro_draws=draws,
    )
    observed = {
        "params": all(leaf.sharding == want_rep for leaf in jax.tree.leaves(params)),
        "opt": all(leaf.sharding == want_rep for leaf in jax.tree.leaves(opt_state)),
        "batch": all(leaf.sharding == want_batch for leaf in jax.tree.leaves(placed_batch)),
        "draws": all(
            value.sharding == (want_rep if index < 2 else want_batch)
            for part in placed_draws
            for index, value in enumerate(part)
        ),
        "loader_spec": shared.production_batch_sharding(mesh) == want_batch,
    }
    # The call-graph half stays structural: it is a fact about WHO calls what, not about text.
    entered = {
        node.func.id
        for node in _ast.walk(_ast.parse(_textwrap.dedent(_inspect.getsource(fp.build_probe_program))))
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)
    }
    observed["m1_enters_the_finalizer"] = "build_training_program" in entered

    # W5b: the last source-SPELLING dependency, replaced. Two halves, neither of them a substring:
    #  (a) BEHAVIOURAL -- entering a program's scope really does install the deployed axis rules, so
    #      the scope object is not decorative;
    #  (b) STRUCTURAL on the AST -- the measurement enters `program.scope()` as a `with` item, which
    #      survives any equivalent refactor that still enters it and fails the moment one does not.
    from flax.linen import partitioning as nn_partitioning

    from maxdiffusion.pos_rollout_update import program_scope

    class _Cfg:
        logical_axis_rules = (("batch", "data"),)

    outside = tuple(nn_partitioning.get_axis_rules())
    with program_scope(_Cfg(), mesh):
        inside = tuple(nn_partitioning.get_axis_rules())
    observed["scope_installs_the_rules"] = inside == (("batch", "data"),) and outside != inside

    measured = _ast.parse(_textwrap.dedent(_inspect.getsource(fp._measure_under_mesh)))
    observed["measured_in_scope"] = any(
        isinstance(node, _ast.With)
        and any(
            isinstance(item.context_expr, _ast.Call)
            and isinstance(item.context_expr.func, _ast.Attribute)
            and item.context_expr.func.attr == "scope"
            for item in node.items
        )
        for node in _ast.walk(measured)
    )
    failed = sorted(name for name, ok in observed.items() if not ok)
    if failed:
        return f"SUCCEEDED: the observed contract fails {failed}"
    return "REFUSED: the executed placement contract holds and M1 enters the shared finalizer"


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
            "load_backbone"
            in {
                node.func.id
                for node in _ast.walk(_ast.parse(source))
                if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)
            }
        )
    if not seams or not all(callers):
        return f"SUCCEEDED: seams_present={seams} callers_delegating={callers}"
    return "REFUSED: one loader reaches the settled seams and both callers enter it"


def attack_w4_compile_against_a_batch_production_never_hands_it():
    """A step compiling against a batch the deployed loader never produces.

    W5: EXECUTED. `production_batch_sharding` is compared with the sharding the deployed loader
    builds (`multihost_dataloading._build_global_shape_and_sharding`), re-derived here from the mesh,
    and the config-agreement guard is exercised on a config that disagrees.
    """
    import jax
    import numpy as _np
    from jax.sharding import NamedSharding, PartitionSpec

    from maxdiffusion import pos_rollout_update as shared

    mesh = jax.sharding.Mesh(_np.array(jax.devices()).reshape(1, 1, 1, 1), ("data", "fsdp", "context", "tensor"))
    deployed = NamedSharding(mesh, PartitionSpec(mesh.axis_names))
    matches = shared.production_batch_sharding(mesh) == deployed
    shared.assert_batch_contract_matches_config(_pos_config(data_sharding=[list(mesh.axis_names)]), mesh)
    try:
        shared.assert_batch_contract_matches_config(_pos_config(data_sharding=[["data"]]), mesh)
    except ValueError:
        refuses = True
    else:
        refuses = False
    if not matches or not refuses:
        return f"SUCCEEDED: loader_match={matches} refuses_disagreement={refuses}"
    return "REFUSED: the contract is the deployed loader's, and a disagreeing config does not start"


def attack_w4_time_a_pruned_scorer():
    """M1's private `loss_fn(...)[0]` let XLA drop the aux norms it was supposed to be timing."""
    import inspect as _inspect
    import textwrap as _textwrap

    from maxdiffusion import pos_rollout_fit_probe as fp
    from maxdiffusion import pos_rollout_update as shared

    builder = _textwrap.dedent(_inspect.getsource(fp.build_probe_program))
    private = "jax.jit(score)" in builder or "draws=_draws_from(v))[0]" in builder
    shared_scorer = "score=scorer," in _textwrap.dedent(_inspect.getsource(shared.build_training_program))
    if private or not shared_scorer:
        return f"SUCCEEDED: private_scorer={private} shared_exposed={shared_scorer}"
    return "REFUSED: M1 times the shared scorer, aux included"


# =================================================================================================
# F5 -- per-cell publication and adoption. A "resume from what is already published" design adds a
# new attack surface in one move: numbers that this process did not measure now enter its artifact.
# Every probe below is an attempt to get somebody else's numbers, half-written numbers, or numbers
# that then escape the artifact's own digest, into an M1 authorization.
# =================================================================================================


def _f5_root(tmp, slot):
    root = pathlib.Path(tmp) / "f5" / slot
    root.mkdir(parents=True, exist_ok=True)
    return root


def _f5_config(fp, tmp, slot, attempt, **over):
    root = _f5_root(tmp, slot) / "m1" / "exp06-f5" / "fit_probe" / "attempts" / attempt
    root.mkdir(parents=True, exist_ok=True)
    values = dict(pos_fit_authorization=str(root / "fit_authorization.json"), run_name="exp06-f5")
    values.update(over)
    return _pos_config(**values)


def _f5_measurer(fp, calls=None):
    def measure(*, cell, context, config):
        if calls is not None:
            calls.append(cell)
        return _fit(fp, context, arm=cell.arm, microbatch=cell.microbatch, k_b=cell.k_b)

    return measure


def _f5_cell():
    return ("rollout", 8, 2)


def _f5_publish(fp, tmp, slot, attempt="att-1", **over):
    """Measure one cell into a fresh attempt root and return the path of the artifact it banked."""
    config = _f5_config(fp, tmp, slot, attempt, **over)
    cell = fp.FitCell(*_f5_cell())
    fp.run_fit_probe(config, measurer=_f5_measurer(fp), cells=[cell], trials=2, devices=[_Dev() for _ in range(8)])
    return fp.cell_marker_path(str(getattr(config, "pos_fit_authorization")), cell)


def _f5_resume(fp, tmp, slot, attempt="att-2", **over):
    """Re-run the same cell with adoption on. Returns (calls the measurer received, published table)."""
    calls = []
    config = _f5_config(fp, tmp, slot, attempt, pos_fit_adoption_root=str(_f5_root(tmp, slot)), **over)
    table = fp.run_fit_probe(
        config,
        measurer=_f5_measurer(fp, calls),
        cells=[fp.FitCell(*_f5_cell())],
        trials=2,
        devices=[_Dev() for _ in range(8)],
    )
    return calls, table


def _f5_edit(marker, mutate):
    """Forge a banked cell the way somebody who wanted it adopted would: a COMPETENT forger.

    An edit that leaves the artifact contradicting ITSELF is refused by a self-consistency check and
    proves nothing about whether adoption compares the artifact with the running program. So every
    derived field is recomputed, and since F5b the forger also REPUBLISHES properly: a new
    content-addressed object at its own digest, the marker moved to commit it, the old object
    removed. Otherwise every probe below would be refused by the content addressing rather than by
    the binding it exists to test -- the same "refused for the wrong reason" trap that made the
    `_Gfile` and the smuggling probe useless.

    **F10d: the resync is computed from the PAYLOAD, not through ``ProbeContext``.** A forgery that
    production's own parser cannot even read is a forgery this harness could not test — the
    fractional-``device_count`` attack died inside `_f5_edit` with `THE PROBE DID NOT RUN` the first
    time, because the resync went through `from_payload`, which now refuses it. A real forger has
    the payload and the digest recipe (both public) and needs neither.

    **F10e: and the derived fields are computed from the value production WOULD PARSE, not from the
    raw one.** F10d hashed the raw ``8.5`` into the context digest, the header and the trial
    bindings — so if bare-``int()`` truncation ever regressed, production would rebuild ``8``, all
    three of those would disagree, and the probe would go on printing REFUSED while being blind to
    the exact regression it exists to catch (the fourth caution: watching the wrong observable). The
    derived fields are now computed over the NORMALIZED context, which is what the original exploit
    did and what a competent forger does; the OUTER payload keeps the raw value and is hashed as-is,
    so the only thing that can refuse it is the parse.
    """
    from maxdiffusion import pos_rollout_fit_probe as fp

    def _digest(mapping):
        return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode("utf-8")).hexdigest()

    def _as_production_would_parse(context):
        """The context a bare-``int()`` reader would rebuild: numbers truncated, everything else as-is."""
        parsed = dict(context)
        try:
            parsed["device_count"] = int(context["device_count"])
        except (TypeError, ValueError):
            pass
        return parsed

    digest = pathlib.Path(marker).read_text().strip()
    content = pathlib.Path(fp._content_for_marker(marker, digest))
    payload = json.loads(content.read_text())["payload"]
    mutate(payload)
    recorded = payload["context"]
    normalized = _as_production_would_parse(recorded)
    payload["context_digest"] = _digest(normalized)
    payload["code_sha"] = normalized["code_sha"]
    payload["device_count"] = normalized["device_count"]
    payload["recipe_fingerprint"] = normalized["recipe_fingerprint"]
    binding = _digest({key: normalized[key] for key in fp.ProbeContext.BINDING_FIELDS})
    for trial in payload["trials"]:
        trial["context_digest"] = binding
    forged = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    pathlib.Path(fp._content_for_marker(marker, forged)).write_text(
        json.dumps({"payload": payload, "sha256": forged}, sort_keys=True)
    )
    content.unlink(missing_ok=True)
    pathlib.Path(marker).write_text(forged + "\n")


def attack_f5_lose_the_bank_to_a_docs_commit(tmp):
    """Can a COSMETIC commit destroy a measured ladder?

    **Rewritten in F7, and the old probe was deleted rather than kept.** It used to mutate `code_sha`
    and call the resulting refusal a success — which was the behaviour, and the behaviour was wrong:
    this campaign records every submission in a ledger commit, so consecutive attempts of one job
    never share a commit, and M1-6 refused the whole M1-5 bank with `['code_sha'] differ` while the
    manifest MATCHED. Guaranteed bank loss on every resubmission is the failure F5 exists to prevent.

    The attack is now the production one: change the label, leave the bytes alone, and see whether the
    ladder is thrown away. REFUSED means it was not. The foreign-MANIFEST refusal is a different probe
    (`F5-5`) and is untouched.
    """
    fp = _probe_env()
    published = _f5_publish(fp, tmp, "commit")
    _f5_edit(published, lambda payload: payload["context"].update(code_sha="0" * 40))
    calls, table = _f5_resume(fp, tmp, "commit")
    if calls:
        return f"SUCCEEDED: a docs-only label change re-measured {len(calls)} trials and discarded the bank"
    if not any(row["provenance"].startswith(fp.ADOPTED_PREFIX) for row in table["cell_provenance"]):
        return "SUCCEEDED: nothing was adopted, so the bank was lost by another route"
    return "REFUSED: identical bytes under a different commit label were adopted, with the drift logged"


def attack_f5_adopt_a_cell_measured_on_another_topology(tmp):
    """A v6e-8 peak is not a v6e-64 peak. Can a cell cross topologies by being published?"""
    fp = _probe_env()
    published = _f5_publish(fp, tmp, "topology")
    _f5_edit(published, lambda payload: payload["context"].update(device_count=64))
    calls, _ = _f5_resume(fp, tmp, "topology")
    if not calls:
        return "SUCCEEDED: a cell measured on 64 chips was adopted by an 8-chip probe"
    return f"REFUSED: re-measured ({len(calls)} trials) -- device_count is inside the context digest"


def attack_f5_adopt_another_jobs_cell(tmp):
    """Two jobs sharing an artifact root: can one job's cells become the other's evidence?"""
    fp = _probe_env()
    published = _f5_publish(fp, tmp, "foreign")
    _f5_edit(published, lambda payload: payload.update(job_identity="somebody-elses-run"))
    calls, _ = _f5_resume(fp, tmp, "foreign")
    if not calls:
        return "SUCCEEDED: another job's cell was adopted"
    return f"REFUSED: re-measured ({len(calls)} trials) -- every banked cell names the run that measured it"


def attack_f5_adopt_a_half_published_cell(tmp):
    """The classic: a writer died mid-publication. Is the wreckage adoptable?

    Two shapes are tried -- content whose digest sidecar never landed (the crash between the two
    writes) and content truncated in place (the crash during one).
    """
    fp = _probe_env()
    notes = []

    published = _f5_publish(fp, tmp, "sidecar")
    pathlib.Path(published).unlink()  # the marker never committed
    calls, _ = _f5_resume(fp, tmp, "sidecar")
    notes.append("no-sidecar ADOPTED" if not calls else "no-sidecar re-measured")

    published = _f5_publish(fp, tmp, "truncated")
    content = pathlib.Path(fp._content_for_marker(published, pathlib.Path(published).read_text().strip()))
    body = content.read_text()
    content.write_text(body[: len(body) // 2])
    calls, _ = _f5_resume(fp, tmp, "truncated")
    notes.append("truncated ADOPTED" if not calls else "truncated re-measured")

    if any("ADOPTED" in note for note in notes):
        return f"SUCCEEDED: {notes}"
    return f"REFUSED: {notes} -- the sidecar is the commit marker and the digest covers the content"


def attack_f5_adopt_a_fabricated_favourable_cell(tmp):
    """Can a writer FABRICATE a cheap cell, rehash it, and have M1 authorize it without measuring?

    **This probe previously lied, and the correction is the point.** Its earlier form rewrote a trial,
    watched the run-level digest move, and reported REFUSED -- but the forged artifact WAS ADOPTED.
    It verified propagation (adopted content lands inside the digest) and labelled that legality
    (adopted content is legitimate). Codex found it: "the committed attack nearly demonstrates this
    itself". The `86/0` it contributed to did not cover this hunt.

    What it asserts is that a FOREIGN-manifest artifact carrying favourable peaks and a correctly
    recomputed digest causes **REMEASUREMENT**: the forger sets both trials to a peak that trivially
    fits and carries a manifest that is not this program's, so the binding refuses it.

    **This is only half the attack, and F5-8 is the other half** (review F5c). A forger who copies the
    CURRENT manifest -- which is public in the payload -- is adopted, and `F5-8` reports that as
    DECLARED. Both cases are kept because they are different classes: one is genuinely refused, the
    other succeeds inside the declared boundary.

    **What this does NOT prove, stated because the harness is the wrong place to imply otherwise:**
    a writer who holds the deployed source tree AND write access to the bucket can reproduce the
    manifest and fabricate peaks. These artifacts are integrity-checked and content-bound, not
    authenticated; the trust anchor is the bucket ACL, exactly as it is for the final authorization
    table. See the F5b worklog entry and the module docstring for the accepted residual.
    """
    fp = _probe_env()
    published = _f5_publish(fp, tmp, "fabricate")

    def fabricate(payload):
        for trial in payload["trials"]:
            # F10: a cheap cell is one whose BOUND is cheap and whose watermark agrees with it. A
            # forger sets all three, because all three live inside the artifact it is rewriting.
            trial["peak_bytes"] = trial["analysis_bytes"] = trial["watermark_bytes"] = 1
        payload["context"]["manifest_digest"] = "0" * 64

    _f5_edit(published, fabricate)
    calls, table = _f5_resume(fp, tmp, "fabricate")
    if not calls:
        return "SUCCEEDED: a fabricated cheap cell was adopted without being measured"
    if any(entry["peak_bytes"] == 1 for entry in table["measurements"]):
        return "SUCCEEDED: the fabricated peak reached the authorization table"
    return f"REFUSED: re-measured ({len(calls)} trials) -- the artifact's bytes are not the running bytes"


def attack_f5_forge_with_the_CURRENT_manifest(tmp):
    """The in-boundary forgery: copy the published context EXACTLY, fabricate the peaks, be adopted.

    **This probe reports DECLARED, not REFUSED, and that is the point.** `F5-5` forges with a FOREIGN
    manifest and is genuinely refused. This one does what an actual attacker would: it copies the
    current artifact's context verbatim -- the manifest digest is PUBLIC in the payload -- swaps both
    trials for one-byte peaks, recomputes the payload digest and the marker, and adoption accepts it.
    No deployed source tree is required. The reviewer executed exactly this and found the previous
    probe scoring it REFUSED.

    It succeeds BY DESIGN within the declared trust boundary: these artifacts are integrity-checked
    and program-bound but NOT authenticated, and the anchor is the bucket ACL. If a publication
    authority is ever added, this probe must flip to REFUSED and the docstring, the worklog and the
    module's residual statement must move with it -- which is what the DECLARED class is for.
    """
    fp = _probe_env()
    published = _f5_publish(fp, tmp, "inboundary")

    def forge(payload):
        for trial in payload["trials"]:
            # F10: bound, peak and watermark together, so the fabricated cell is internally coherent
            # under the new rule exactly as it was under the old one.
            trial["peak_bytes"] = trial["analysis_bytes"] = trial["watermark_bytes"] = 1
        # The context, manifest digest included, is left EXACTLY as published.

    _f5_edit(published, forge)
    artifact = fp.load_cell_artifact(published)
    if artifact.context.manifest_digest != fp.deployed_manifest_digest():
        return "SUCCEEDED: the probe failed to stay in-boundary -- it is testing the foreign-manifest case again"
    calls, table = _f5_resume(fp, tmp, "inboundary")
    if calls:
        return (
            f"REFUSED: the in-boundary forgery was re-measured ({len(calls)} trials) -- a publication authority "
            f"now exists, so the declared residual is STALE: update the module docstring, the worklog and this probe"
        )
    peaks = [entry["peak_bytes"] for entry in table["measurements"]]
    return (
        f"DECLARED: an authorized bucket writer who can READ one artifact fabricated a cell (peaks {peaks}) and it "
        f"was adopted without measurement. Accepted residual: integrity- and program-bound, NOT authenticated; "
        f"the anchor is the bucket ACL. Escalated to Yixun (KMS/workload-identity signing)."
    )


def attack_f7_refuse_a_launch_over_a_docs_commit(tmp):
    """Can a COSMETIC commit block a legitimate M2 launch at the gate?

    The mirror of `F5-1`, one step later and far more expensive. M1 publishes an authorization at one
    tip; the Planner records the submission in the ledger; M2 starts at the next tip running identical
    bytes. Until F7b `assert_cell_authorized` compared the full context and refused — at startup, with
    the reservation already held. REFUSED here means the launch was NOT blocked.

    The dangerous direction is a separate probe (`F7-2`): identical label, different bytes.
    """
    fp = _probe_env()
    measured_at = _ctx(fp)
    running_now = dataclasses.replace(measured_at, code_sha="5631a36" + "0" * 33)
    if measured_at.binding_digest() != running_now.binding_digest():
        return "SUCCEEDED: the probe changed the BUILD, not just the label -- it is testing the wrong thing"
    published = _auth(fp, tmp, [_fit(fp, measured_at)], name="f7_label.json", context=measured_at)
    try:
        fp.assert_cell_authorized(published, fp.FitCell("rollout", 32, 2), context=running_now)
    except ValueError as error:
        return f"SUCCEEDED: a docs-only commit blocked the launch -- {str(error).splitlines()[0][:110]}"
    return "REFUSED: identical bytes under a different commit label were authorized, with the drift logged"


def attack_f7_authorize_a_different_build_under_the_same_label(tmp):
    """The direction that must NOT be narrowed: same `COMMIT`, different running bytes.

    A dirty tree, a stale tarball or a hand-edited module on a worker all produce exactly this, and it
    is what the manifest was introduced for in F5b. F7/F7b removed the LABEL from the binding and must
    not have removed anything else.
    """
    fp = _probe_env()
    measured_at = _ctx(fp)
    running_now = dataclasses.replace(measured_at, manifest_digest="9" * 64)
    published = _auth(fp, tmp, [_fit(fp, measured_at)], name="f7_build.json", context=measured_at)
    try:
        fp.assert_cell_authorized(published, fp.FitCell("rollout", 32, 2), context=running_now)
    except ValueError as error:
        text = str(error).splitlines()[0]
        if "manifest_digest" not in text:
            return f"SUCCEEDED (refused for the wrong reason): {text[:110]}"
        return f"REFUSED: {text[:120]}"
    return "SUCCEEDED: an authorization measured by other bytes was accepted under a matching label"


def attack_f6_quote_an_excluded_cell(tmp):
    """Can a training run reach a cell M1 DECLARED unreachable and never built?

    F6 exists because `one_step microbatch=32 k=2` faults the chip deterministically (2/2 on the same
    cell, two VMs, two days), so the ladder publishes without it. An excluded cell has no measurement
    at all -- weaker evidence than a refused one, which at least missed a rule -- so quoting it must be
    as unconstructible as quoting a refused cell, and the refusal must say it was left out ON PURPOSE
    rather than merely never seen.
    """
    fp = _probe_env()
    config = _f5_config(fp, tmp, "excluded", "att-1")
    for key, value in (
        ("pos_fit_excluded_cells", "one_step:32:2"),
        ("pos_fit_exclusion_reason", "bad_smem_address: deterministic XLA codegen fault (issue #18)"),
    ):
        setattr(config, key, value)
    excluded = fp.FitCell("one_step", 32, 2)
    path = str(getattr(config, "pos_fit_authorization"))
    fp.run_fit_probe(
        config,
        measurer=_f5_measurer(fp),
        cells=[fp.FitCell("rollout", 8, 2), excluded],
        trials=2,
        devices=[_Dev() for _ in range(8)],
    )
    published = fp.load_authorization(path)
    if any(dict(e) == excluded.as_payload() for e in published["authorized_cells"]):
        return "SUCCEEDED: an excluded cell was AUTHORIZED"
    if not published["excluded_cells"]:
        return "SUCCEEDED: the excluded cell vanished from the table instead of being recorded"
    try:
        fp.assert_cell_authorized(published, excluded, context=fp.derive_probe_context(config, devices=[_Dev()] * 8))
    except ValueError as error:
        text = str(error)
        if "EXCLUDED" not in text:
            return f"SUCCEEDED: refused, but not AS an exclusion: {text.splitlines()[0][:110]}"
        return f"REFUSED: {text.splitlines()[0][:120]}"
    return "SUCCEEDED: a declared-unreachable cell was accepted by the gate"


def attack_f6_doctor_a_table_into_two_statuses(tmp):
    """Can an edited-and-rehashed v4 table give one cell TWO statuses, and does the gate notice?

    Review F6b, MAJOR 1. The serializer emitted `authorized_cells` / `refused_cells` /
    `excluded_cells` / `skipped_cells` independently and the loader only type-checked them, so an
    editor could append an AUTHORIZED cell to `excluded_cells`, re-hash, and load — and
    `assert_cell_authorized` returns on the authorized list before it ever looks at exclusions, so the
    contradiction resolved in the attacker's favour.

    This must be refused at LOAD, which is a different and earlier refusal than `F6-1`'s (a table that
    is internally sound, refusing a cell it legitimately excludes).
    """
    fp = _probe_env()
    config = _f5_config(fp, tmp, "doctored", "att-1")
    for key, value in (
        ("pos_fit_excluded_cells", "one_step:32:2"),
        ("pos_fit_exclusion_reason", "bad_smem_address: deterministic XLA codegen fault (issue #18)"),
    ):
        setattr(config, key, value)
    path = str(getattr(config, "pos_fit_authorization"))
    fp.run_fit_probe(
        config,
        measurer=_f5_measurer(fp),
        cells=[fp.FitCell("rollout", 8, 2), fp.FitCell("one_step", 32, 2)],
        trials=2,
        devices=[_Dev() for _ in range(8)],
    )
    fp.load_authorization(path)  # sound before doctoring

    stored = json.loads(pathlib.Path(path).read_text())
    payload = stored["payload"]
    smuggled = dict(payload["authorized_cells"][0])
    payload["excluded_cells"].append({**smuggled, "reason": "smuggled"})
    stored["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    pathlib.Path(path).write_text(json.dumps(stored, sort_keys=True))

    try:
        published = fp.load_authorization(path)
    except ValueError as error:
        return f"REFUSED at load: {str(error).splitlines()[0][:120]}"
    try:
        fp.assert_cell_authorized(
            published,
            fp.FitCell(**{k: v for k, v in smuggled.items()}),
            context=fp.derive_probe_context(config, devices=[_Dev()] * 8),
        )
    except ValueError:
        return "SUCCEEDED (partly): the contradictory table LOADED; only the gate caught the cell"
    return "SUCCEEDED: a cell listed as both authorized and excluded loaded and was authorized"


def attack_f5_tear_the_pair_with_two_publishers(tmp):
    """Can two concurrent publishers leave a cell permanently unadoptable?

    Review F5b, MAJOR. The previous shape wrote a fixed-name content file and then a digest sidecar,
    so an interleaving could finish as `content-B + digest-A` and no reader could ever accept that
    cell again. Production payloads are NOT byte-identical between attempts -- measured step times
    differ in the third decimal -- so this drives two distinct payloads through the worst ordering:
    both contents, then the markers committed in the opposite order.
    """
    fp = _probe_env()
    config = _f5_config(fp, tmp, "tear", "att-1")
    cell = fp.FitCell(*_f5_cell())
    context = fp.derive_probe_context(config, devices=[_Dev() for _ in range(8)])
    marker = fp.cell_marker_path(str(getattr(config, "pos_fit_authorization")), cell)
    a, b = (
        fp.CellArtifact(
            cell=cell,
            context=context,
            job_identity="exp06-f5",
            trials=(_fit(fp, context, arm=cell.arm, microbatch=cell.microbatch, k_b=cell.k_b, step_seconds=t),),
        )
        for t in (25.347, 25.356)
    )
    fp.publish_cell_content(marker, a)
    fp.publish_cell_content(marker, b)
    fp.commit_cell_marker(marker, b)
    fp.commit_cell_marker(marker, a)
    try:
        loaded = fp.load_cell_artifact(marker)
    except Exception as error:  # noqa: BLE001
        return f"SUCCEEDED: the interleaving tore the pair -- {type(error).__name__}: {error}"
    steps = {25.347, 25.356}
    if loaded.trials[0].step_seconds not in steps:
        return f"SUCCEEDED: the marker resolved to a payload neither publisher wrote ({loaded.trials[0].step_seconds})"
    if loaded.trials[0].step_seconds != 25.347:
        return "SUCCEEDED: the marker does not name the last publisher's object"
    return "REFUSED: content objects are immutable and the marker commits one whole object -- no torn pair exists"


def attack_f5_a_killed_ladder_banks_nothing(tmp):
    """The production failure itself, as an attack: does a ladder killed mid-way still lose its work?

    Attempt 2 of M1-3 measured 24 of 32 cells and published NOTHING, because the table published only
    at completion. The probe must bank each finished cell before it starts the next compile.
    """
    fp = _probe_env()
    config = _f5_config(fp, tmp, "killed", "att-1")
    cells = [fp.FitCell("rollout", 8, 2), fp.FitCell("rollout", 16, 2), fp.FitCell("one_step", 8, 2)]
    calls = []

    def dies_after_two_cells(*, cell, context, config):
        if len(calls) >= 4:
            raise RuntimeError("TPU_VM_HEALTH_UNHEALTHY_MAINTENANCE")
        calls.append(cell)
        return _fit(fp, context, arm=cell.arm, microbatch=cell.microbatch, k_b=cell.k_b)

    try:
        fp.run_fit_probe(
            config, measurer=dies_after_two_cells, cells=cells, trials=2, devices=[_Dev() for _ in range(8)]
        )
    except RuntimeError:
        pass
    authorization = str(getattr(config, "pos_fit_authorization"))
    banked = [cell for cell in cells if pathlib.Path(fp.cell_marker_path(authorization, cell)).exists()]
    if len(banked) != 2:
        return f"SUCCEEDED: a ladder killed after 2 of 3 cells banked {len(banked)}"
    readable = all(fp.load_cell_artifact(fp.cell_marker_path(authorization, cell)) for cell in banked)
    if not readable:
        return "SUCCEEDED: the banked cells do not load"
    return "REFUSED: both finished cells were banked, digest-verified, before the cell that died"


# =================================================================================================
# Round F10 — the AUTHORIZATION EVIDENCE contract (Yixun's Option A, plan v2.9 §4-P1).
#
# The bound is the cell's compiled memory analysis; the runtime watermark is recorded beside it as a
# cross-check that can only ever REFUSE. Three ways to get authorized on evidence that does not
# support it, and one way to have an old table re-read under the new rule.
# =================================================================================================


def _f10_gate(fp, tmp, name, measurement, context):
    """Publish a table holding ONE cell, re-decide it on load, and try to gate a launch with it.

    Returns ``(reloaded_table, refusal_or_None)`` — the refusal is what `assert_cell_authorized`
    raised, and ``None`` means the cell got through to a training run.

    ``measurement`` may be one measurement or a list of TRIALS of one cell (F10b): the aggregation
    is part of the path a trial-local fault has to survive, so the probes that attack it have to go
    through this same publish → reload → gate sequence rather than stopping at the aggregate.
    """
    trials = list(measurement) if isinstance(measurement, (list, tuple)) else [measurement]
    published = _auth(fp, tmp, trials, name=name, context=context)
    reloaded = fp.load_authorization(str(pathlib.Path(tmp) / name))
    if reloaded["authorized_cells"] != published["authorized_cells"]:
        raise AssertionError("the loader and the publisher disagree about this table")
    try:
        fp.assert_cell_authorized(reloaded, trials[0].cell, context=context)
    except ValueError as error:
        return reloaded, str(error).splitlines()[0][:110]
    return reloaded, None


def attack_f10_mark_above_the_bound(tmp):
    """A cell whose recorded WATERMARK stands above its claimed analysis must not authorize.

    The watermark is a lower bound on what the process really held; the analysis is offered as an
    upper bound on what the cell needs. A watermark above it falsifies the premise the authorization
    rests on, and the contract refuses the cell rather than deciding which number to believe.
    """
    fp = _probe_env()
    context = _ctx(fp)
    measurement = _fit(
        fp,
        context,
        peak_bytes=25 * 1024**3,
        analysis_bytes=15 * 1024**3,  # 46.9% of capacity: the headroom rule alone would authorize
        watermark_bytes=25 * 1024**3,
        peak_source=fp.PEAK_SOURCE_RUNTIME_RAISED,
        peak_attribution=fp.PEAK_ATTRIBUTION_RAISED,
    )
    table, refusal = _f10_gate(fp, tmp, "f10_mark.json", measurement, context)
    if refusal is None:
        return "SUCCEEDED: a 25GiB mark over a 15GiB claimed bound authorized a training cell"
    reasons = table["refused_cells"][0]["reasons"]
    if "watermark_exceeds_analysis" not in reasons:
        return f"SUCCEEDED: the table refused for {reasons} -- the mark/bound conflict was not named"
    return f"REFUSED: {reasons}; the gate says {refusal!r}"


def attack_f10_no_bound_at_all(tmp):
    """A measurement with NO analysis has no authorization bound, however good its runtime reading.

    This is the case F10 has to keep fail-closed: the contract names exactly one number as the
    bound, so a record that does not carry it cannot be measured against the headroom rule at all.
    Reporting a clean, attributable, reset-sourced watermark does not substitute for it.
    """
    fp = _probe_env()
    context = _ctx(fp)
    measurement = _fit(
        fp,
        context,
        analysis_bytes=None,
        peak_bytes=20 * 1024**3,
        watermark_bytes=20 * 1024**3,
        peak_source=fp.PEAK_SOURCE_RUNTIME_RESET,
        peak_attribution=fp.PEAK_ATTRIBUTION_RESET,
    )
    table, refusal = _f10_gate(fp, tmp, "f10_unbounded.json", measurement, context)
    if refusal is None:
        return "SUCCEEDED: a cell with no compiled analysis was authorized on its watermark alone"
    reasons = table["refused_cells"][0]["reasons"]
    if "analysis_missing" not in reasons:
        return f"SUCCEEDED: the table refused for {reasons} -- the missing bound was not named"
    return f"REFUSED: {reasons}; the gate says {refusal!r}"


def attack_f10_peak_above_a_quiet_bound(tmp):
    """The residue: a reported peak above the analysis that the cell's own watermark cannot explain.

    ``classify_peak`` reports ``max(runtime, analysis)``, so this pairing cannot come out of the
    measurement path — which is exactly why a record carrying it was assembled somewhere else, and
    why authorizing it on the small number would be authorizing a bound its own row contradicts.
    """
    fp = _probe_env()
    context = _ctx(fp)
    measurement = _fit(
        fp,
        context,
        peak_bytes=31 * 1024**3,
        analysis_bytes=8 * 1024**3,
        watermark_bytes=1 * 1024**3,
    )
    table, refusal = _f10_gate(fp, tmp, "f10_quiet.json", measurement, context)
    if refusal is None:
        return "SUCCEEDED: a 31GiB reported peak was authorized against an 8GiB bound"
    reasons = table["refused_cells"][0]["reasons"]
    if "peak_exceeds_analysis" not in reasons:
        return f"SUCCEEDED: the table refused for {reasons} -- the peak/bound conflict was not named"
    return f"REFUSED: {reasons}; the gate says {refusal!r}"


def attack_f10_read_a_v6_table_as_a_v7_one(tmp):
    """A table decided under the OLD authorization rule must not be read under the new one.

    F10 changed how ``authorized_cells`` is derived while every field kept its name and type, so the
    protocol version is the only thing standing between a v6 table and a v7 reader. Three reader
    paths are probed: the loader, the training gate, and republication (which adopts an identical
    artifact and must not adopt one whose protocol says it was decided by another rule).
    """
    fp = _probe_env()
    context = _ctx(fp)
    path = pathlib.Path(tmp) / "f10_v6.json"
    _auth(fp, tmp, [_fit(fp, context)], name="f10_v6.json", context=context)
    stored = json.loads(path.read_text())
    stored["payload"]["protocol"] = "exp06.fit_authorization.v6"
    stored["sha256"] = hashlib.sha256(json.dumps(stored["payload"], sort_keys=True).encode("utf-8")).hexdigest()
    path.write_text(json.dumps(stored))

    survivors = []
    try:
        fp.load_authorization(str(path))
        survivors.append("the loader accepted it")
    except ValueError:
        pass
    try:
        fp.assert_cell_authorized(stored["payload"], fp.FitCell("rollout", 32, 2), context=context)
        survivors.append("the training gate accepted it")
    except ValueError:
        pass
    try:
        evidence = fp.build_evidence(
            context, [_fit(fp, context)], max_train_steps=10_000, eval_every=1_000, checkpoint_every=1_000
        )
        republished = fp.publish_authorization(str(path), evidence)
        if republished.get("protocol") != fp.AUTHORIZATION_PROTOCOL:
            survivors.append("republication adopted it and returned it under its old protocol")
        else:
            survivors.append("republication silently re-labelled it")
    except ValueError:
        pass
    if survivors:
        return f"SUCCEEDED: {'; '.join(survivors)}"
    return "REFUSED: loader, training gate and republication all refuse a table decided under another rule"


# =================================================================================================
# Round F10b — the three fail-closed gaps the F10 review found IN the new gate. Each one was
# executed by the reviewer through publication, reload and `assert_cell_authorized`; each probe
# below walks that same path rather than stopping at a verdict object.
# =================================================================================================


def attack_f10b_no_mark_at_all(tmp):
    """An analysis-bounded cell with NO watermark: nothing cross-checked it, so nothing authorizes it.

    Review MAJOR 1. F10 only compared the watermark when one was recorded, so the cross-check the
    contract RETAINS was skipped exactly when it was missing — and the case is reachable, not
    theoretical: a backend that reports ``bytes_limit`` without ``peak_bytes_in_use`` produces this
    record. The F10 test suite pinned the authorization, which is how the gap survived review-by-test.
    """
    fp = _probe_env()
    context = _ctx(fp)
    measurement = _fit(fp, context, watermark_bytes=None, watermark_before_bytes=None)
    table, refusal = _f10_gate(fp, tmp, "f10b_nomark.json", measurement, context)
    if refusal is None:
        return "SUCCEEDED: a cell with a bound and no runtime cross-check at all authorized a training run"
    reasons = table["refused_cells"][0]["reasons"]
    if "watermark_missing" not in reasons:
        return f"SUCCEEDED: the table refused for {reasons} -- the absent cross-check was not named"
    return f"REFUSED: {reasons}; the gate says {refusal!r}"


def attack_f10b_a_friendly_trial_cancels_a_contradicted_one(tmp):
    """Two trials of one cell, one of them contradicted: the cell must not authorize.

    Review MAJOR 2, the reviewer's executed exploits. Aggregation took ``max`` of the peak, the
    analysis and the watermark INDEPENDENTLY, so a second trial's friendlier numbers covered the
    first trial's fault and the aggregate — which is all the published table carries — looked clean.
    Three shapes are attacked here: a mark that cleared the bound in one trial, a peak that did, and
    a trial that recorded no bound at all beside one that did.
    """
    fp = _probe_env()
    context = _ctx(fp)
    bound = 20 * 1024**3  # both trials of every pair agree on the cell's own bound, as they must
    cases = {
        "mark over the bound in trial 2": [
            _fit(fp, context, analysis_bytes=bound, watermark_bytes=4 * 1024**3),
            _fit(fp, context, analysis_bytes=bound, watermark_bytes=25 * 1024**3),
        ],
        "peak over the bound in trial 1": [
            _fit(fp, context, peak_bytes=31 * 1024**3, analysis_bytes=bound, watermark_bytes=1 * 1024**3),
            _fit(fp, context, analysis_bytes=bound, watermark_bytes=1 * 1024**3),
        ],
        "no bound in trial 1": [_fit(fp, context, analysis_bytes=None), _fit(fp, context, analysis_bytes=bound)],
        "no mark in trial 1": [
            _fit(fp, context, analysis_bytes=bound, watermark_bytes=None),
            _fit(fp, context, analysis_bytes=bound),
        ],
    }
    survivors = []
    for index, (name, trials) in enumerate(cases.items()):
        try:
            table, refusal = _f10_gate(fp, tmp, f"f10b_trial_{index}.json", trials, context)
        except ValueError:
            continue  # refused before publication -- also a refusal of this attack
        if refusal is None:
            survivors.append(f"{name}: authorized {table['authorized_cells']}")
    if survivors:
        return f"SUCCEEDED: a friendly trial cancelled a contradicted one -- {'; '.join(survivors)}"
    return f"REFUSED: all {len(cases)} trial-local contradictions survived aggregation and refused the cell"


def attack_f10b_two_bounds_for_one_executable(tmp):
    """Trials that disagree about the analysis: ``analysis=10/watermark=11`` beside ``analysis=20``.

    Review MAJOR 2's first exploit, and the one no per-field aggregate can fix: the larger bound
    covers the smaller trial's contradicted mark. It is one compiled executable measured twice, so
    two accounts of its footprint are two programs — refused, not collapsed.
    """
    fp = _probe_env()
    context = _ctx(fp)
    trials = [
        _fit(fp, context, analysis_bytes=10 * 1024**3, watermark_bytes=11 * 1024**3),
        _fit(fp, context, analysis_bytes=20 * 1024**3),
    ]
    try:
        table, refusal = _f10_gate(fp, tmp, "f10b_twobounds.json", trials, context)
    except ValueError as error:
        if "analysis_disagreement" not in str(error):
            return f"SUCCEEDED: refused, but not as a disagreement about the cell's own bound: {str(error)[:110]}"
        return f"REFUSED at aggregation: {str(error).splitlines()[0][:120]}"
    if refusal is None:
        return f"SUCCEEDED: two different bounds for one cell authorized it ({table['measurements'][0]})"
    return f"REFUSED: {table['refused_cells'][0]['reasons']}; the gate says {refusal!r}"


def attack_f10b_fractional_bytes(tmp):
    """The reviewer's truncation construction: ``analysis = watermark = peak = 9.9`` of a 10-byte device.

    Review MODERATE. ``int()`` rounds toward zero, so a 99%-of-capacity record parsed as ``9/10`` and
    authorized. A byte count is an integer; a record carrying anything else was not measured here.
    """
    fp = _probe_env()
    context = _ctx(fp)
    survivors = []
    for name, over in (
        ("9.9 of a 10-byte device", dict(peak_bytes=9.9, analysis_bytes=9.9, watermark_bytes=9.9, capacity_bytes=10)),
        ("a boolean bound", dict(analysis_bytes=True, watermark_bytes=True, peak_bytes=True, capacity_bytes=10)),
    ):
        try:
            table, refusal = _f10_gate(
                fp, tmp, f"f10b_frac_{abs(hash(name))}.json", _fit(fp, context, **over), context
            )
        except ValueError:
            continue  # the record does not parse: refused before anything could be decided from it
        if refusal is None:
            survivors.append(f"{name}: authorized on {table['measurements'][0]['analysis_bytes']} bytes")
    if survivors:
        return f"SUCCEEDED: {'; '.join(survivors)}"
    return "REFUSED: a fractional or boolean byte count is a malformed record, not a small measurement"


def attack_f10b_round_the_headroom_boundary_down(tmp):
    """A cell truly above 90% whose float division lands exactly ON the boundary.

    Review MODERATE, second half. ``analysis/capacity`` is a float division of two byte counts; at
    this magnitude the quotient rounds to exactly ``0.9`` while ``10*analysis > 9*capacity`` — so the
    old comparison authorized a cell that is over the pinned rule. The magnitudes are absurd for HBM;
    the point is that a pinned gate must not inherit the mantissa's limits.
    """
    fp = _probe_env()
    context = _ctx(fp)
    analysis, capacity = 8106479329266895, 9007199254740994
    if not (analysis / capacity == fp.HEADROOM_FRACTION and 10 * analysis > 9 * capacity):
        return "SUCCEEDED: THE PROBE DID NOT RUN -- the chosen pair no longer separates the two comparisons"
    measurement = _fit(
        fp, context, peak_bytes=analysis, analysis_bytes=analysis, watermark_bytes=1, capacity_bytes=capacity
    )
    table, refusal = _f10_gate(fp, tmp, "f10b_boundary.json", measurement, context)
    if refusal is None:
        return "SUCCEEDED: a cell above the 90% rule authorized because the float division rounded down"
    reasons = table["refused_cells"][0]["reasons"]
    if "headroom" not in reasons:
        return f"SUCCEEDED: refused for {reasons} -- not on the headroom rule it is actually over"
    return f"REFUSED: {reasons} -- the boundary is decided in exact integers"


# =================================================================================================
# Round F10c — the two residuals the F10b review executed: a malformed count that survived the
# deserialization boundary, and a poisoned BANK that wedged every retry.
# =================================================================================================


def attack_f10c_poison_the_bank_forever(tmp):
    """Bank a cell whose trials disagree, then retry twice: issue #10's wedge, made concrete.

    Review MODERATE 2. F10b refuses to publish a table whose trials disagree about the cell's own
    compiled analysis — correct, and the reviewer then put exactly that artifact in the CACHE the
    retry reads: two consecutive attempts adopted it and died at ``build_evidence``, forever. The
    fix keeps the table-wide raise and quarantines the artifact instead: an inconsistent cached cell
    is refused for adoption and re-measured, so the outage costs one measurement, not the campaign.
    """
    fp = _probe_env()
    published = _f5_publish(fp, tmp, "poison")
    _f5_edit(published, lambda payload: payload["trials"][1].update(analysis_bytes=10 * 1024**3))
    try:
        calls, table = _f5_resume(fp, tmp, "poison")
    except ValueError as error:
        return f"SUCCEEDED: the retry DIED on a cached artifact instead of re-measuring it: {str(error)[:110]}"
    if not calls:
        return "SUCCEEDED: an artifact whose trials cannot be aggregated was adopted"
    if not table.get("authorized_cells"):
        return f"SUCCEEDED: the retry published nothing usable ({table.get('refused_cells')})"
    # ...and the attempt AFTER the repair adopts the clean cell the retry banked, so the cost is one
    # measurement rather than one per attempt.
    try:
        again, _ = _f5_resume(fp, tmp, "poison", attempt="att-3")
    except ValueError as error:
        return f"SUCCEEDED: the third attempt died on the quarantined artifact: {str(error)[:110]}"
    if again:
        return f"SUCCEEDED: the quarantine is not self-healing -- attempt 3 re-measured {len(again)} trials again"
    return f"REFUSED: the poisoned cell was quarantined and re-measured ({len(calls)} trials), then adopted clean"


def attack_f10c_bank_a_malformed_count(tmp):
    """A digest-valid banked artifact carrying ``peak_bytes: 9.9`` / ``true`` / a digit string.

    Review MODERATE (b). ``CellMeasurement.from_payload`` coerced the three counts with bare
    ``int()`` BEFORE ``_checked`` saw them, so the reviewer loaded these artifacts through
    load -> adopt -> republish -> ``assert_cell_authorized``. Parsing is exact now, so the artifact
    does not load, adoption logs it, and the cell is measured instead of rounded.
    """
    fp = _probe_env()
    survivors = []
    for index, (name, mutation) in enumerate(
        (
            ("peak_bytes 9.9", {"peak_bytes": 9.9}),
            ("peak_bytes true", {"peak_bytes": True}),
            ("reservation_failures 0.9", {"reservation_failures": 0.9}),
            ("analysis_bytes as text", {"analysis_bytes": "21474836480"}),
        )
    ):
        slot = f"malformed{index}"
        published = _f5_publish(fp, tmp, slot)
        _f5_edit(published, lambda payload, m=mutation: payload["trials"][0].update(m))
        try:
            calls, table = _f5_resume(fp, tmp, slot)
        except ValueError:
            continue  # refused on the way in, which is a refusal of this attack
        if not calls:
            survivors.append(f"{name}: adopted without measurement ({table['measurements'][0]['peak_bytes']})")
    if survivors:
        return f"SUCCEEDED: {'; '.join(survivors)}"
    return "REFUSED: every malformed banked count is refused at parse and the cell is re-measured"


# =================================================================================================
# Round F10d — the SAME truncation class at the sites the F10c sweep had not reached: the cell
# IDENTITY, the BINDING device count, the projection cadences, the trial count.
# =================================================================================================


def attack_f10d_truncate_an_identity_a_binding_and_a_count(tmp):
    """Four numbers that are not byte counts, and every one of them used to round into a valid record.

    Review MODERATE. F10b/F10c made the BYTES exact and stopped there, which left the same class
    alive one field over: ``FitCell("rollout", 8.5, 2.5)`` published and loaded as cell 8/2 (an
    identity a truncation invented, carrying the authorization of a cell nobody measured);
    ``device_count: 8.5`` in a banked artifact loaded as 8 and matched an eight-device context on a
    BINDING field; ``_positive_int`` took a fractional ``Fraction``/``Decimal`` through its own float
    round trip; and ``trial_count`` accepted ``1.9``, ``"1"`` and ``True`` as one trial.
    """
    import decimal
    import fractions

    fp = _probe_env()
    context = _ctx(fp)
    survivors = []

    for microbatch, k_b in ((8.5, 2), (8, 2.5), ("8", 2)):
        try:
            cell = fp.FitCell(arm="rollout", microbatch=microbatch, k_b=k_b)
        except ValueError:
            continue
        survivors.append(f"FitCell({microbatch!r}, {k_b!r}) constructed as {cell.as_payload()}")
    for microbatch in (8.5, "8"):
        try:
            loaded = fp.FitCell.from_payload({"arm": "rollout", "microbatch": microbatch, "k_b": 2})
        except ValueError:
            continue
        survivors.append(f"a payload identity {microbatch!r} loaded as {loaded.microbatch}")

    for value in (8.5, "8", True):
        try:
            rebuilt = fp.ProbeContext.from_payload({**context.as_payload(), "device_count": value})
        except ValueError:
            continue
        if rebuilt.binding_digest() == context.binding_digest():
            survivors.append(f"device_count={value!r} BOUND to an 8-device context")
        else:
            survivors.append(f"device_count={value!r} loaded as {rebuilt.device_count}")
    try:
        dataclasses.replace(context, device_count=8.5).as_payload()
        survivors.append("a fractional device_count SERIALIZED into a clean payload")
    except ValueError:
        pass

    for value in (fractions.Fraction(162129586585337857, 2), decimal.Decimal("81064793292668928.5"), "10000"):
        try:
            steps = fp.project_wall_clock(
                _fit(fp, context), max_train_steps=value, eval_every=1_000, checkpoint_every=1_000
            )["max_train_steps"]
        except ValueError:
            continue
        survivors.append(f"a cadence of {value!r} projected as {steps} steps")

    artifact = fp.CellArtifact(
        cell=fp.FitCell("rollout", 32, 2), context=context, job_identity="j", trials=(_fit(fp, context),)
    )
    for value in (1.9, "1", True):
        try:
            fp.CellArtifact.from_payload({**artifact.as_payload(), "trial_count": value})
        except ValueError:
            continue
        survivors.append(f"trial_count={value!r} accepted as one trial")

    if survivors:
        return f"SUCCEEDED: {'; '.join(survivors)}"
    return "REFUSED: identity, binding, cadence and trial count are all parsed exactly, in both directions"


def attack_f10d_bank_a_fractional_binding(tmp):
    """The same truncation where it pays: a banked artifact recording ``device_count: 8.5``.

    It loaded as ``8`` and was ADOPTED by an eight-device context — a topology nobody ran, matching
    one that did, on the field adoption exists to compare.
    """
    fp = _probe_env()
    published = _f5_publish(fp, tmp, "fracbinding")
    _f5_edit(published, lambda payload: payload["context"].update(device_count=8.5))
    try:
        calls, table = _f5_resume(fp, tmp, "fracbinding")
    except ValueError as error:
        return f"SUCCEEDED: the retry DIED on the artifact rather than re-measuring it: {str(error)[:110]}"
    if not calls:
        return "SUCCEEDED: an artifact whose device count is fractional was adopted"
    if table["context"]["device_count"] != 8:
        return f"SUCCEEDED: the published context records {table['context']['device_count']!r} devices"
    return f"REFUSED: the fractional binding was not adopted; the cell was re-measured ({len(calls)} trials)"


# =================================================================================================
# Round F10e — the last hole in the coercion invariant: the compiled analysis's own COMPONENTS.
# =================================================================================================


class _F10eAnalysis:
    """A compiled memory analysis with whatever component values an attack wants."""

    def __init__(self, **fields):
        for name, value in fields.items():
            setattr(self, name, value)


class _F10eStep:
    """A ``program.step`` that lowers and compiles to a chosen analysis (and is callable)."""

    def __init__(self, analysis):
        self._analysis = analysis

    def __call__(self, params, opt_state, batch, draws):
        return params, opt_state, 0.0

    def lower(self, *args, **kwargs):
        return self

    def compile(self):
        return self

    def memory_analysis(self):
        return self._analysis


def _f10e_program(fp, **fields):
    return fp.ProbeProgram(
        step=_F10eStep(_F10eAnalysis(**fields)),
        score=lambda params, batch, draws: 0.0,
        params={},
        opt_state={},
        batch={},
        draws=(),
    )


def attack_f10e_shrink_the_bound_with_a_bad_component(tmp):
    """Understate the analysis by feeding its COMPONENTS values that are not byte counts.

    Review MODERATE. ``_exact_count(value or 0, ...)`` was the last ``or 0`` in the module: ``False``
    and ``""`` became a legal-looking zero, and a NEGATIVE component was accepted and SUBTRACTED from
    the genuine ones. The reviewer's executed case is the first row here — ``(100, -90, 0, 0)`` sums
    to 10, and a 10-byte bound against a 5-byte watermark on a 100-byte device AUTHORIZES. An
    understated bound that authorizes is the severity class the whole amendment exists to prevent.

    **F10f adds the ABSENT component**, on the reviewer's ruling against F10e's skip: an analysis
    that exposes only some of the four is not a small bound, it is an unknown one, and a partial sum
    must never become the authorized number.
    """
    fp = _probe_env()
    whole = {
        "argument_size_in_bytes": 100,
        "temp_size_in_bytes": 0,
        "output_size_in_bytes": 0,
        "alias_size_in_bytes": 0,
    }
    survivors = []
    for name, fields in (
        ("a negative component", {**whole, "temp_size_in_bytes": -90}),
        ("False", {**whole, "argument_size_in_bytes": False}),
        ("an empty string", {**whole, "argument_size_in_bytes": ""}),
        ("an unfilled component", {**whole, "argument_size_in_bytes": None}),
        ("a fractional component", {**whole, "argument_size_in_bytes": 9.9}),
        # F10f: an ABSENT component is unknown evidence, not zero. The reviewer's shape: an analysis
        # exposing only `argument_size_in_bytes = 10` used to be a 10-byte bound, and a 10-byte bound
        # against a 5-byte watermark on a 100-byte device AUTHORIZED on three quarters of a bound.
        ("only one of the four components", {"argument_size_in_bytes": 10}),
        ("three of the four components", {k: v for k, v in whole.items() if k != "alias_size_in_bytes"}),
    ):
        program = _f10e_program(fp, **fields)
        analysis = fp._program_bytes(program, program.params, program.opt_state)
        if analysis is None:
            continue
        measurement = _fit(fp, peak_bytes=analysis, analysis_bytes=analysis, watermark_bytes=5, capacity_bytes=100)
        verdict = fp.cell_verdict(measurement)
        survivors.append(f"{name}: analysis {analysis}, verdict {'AUTHORIZED' if verdict.fits else verdict.reasons}")
    if survivors:
        return f"SUCCEEDED: {'; '.join(survivors)}"
    return (
        "REFUSED: one bad OR ABSENT component discards the whole analysis (7 shapes), so the cell has no "
        "bound and refuses"
    )


# =================================================================================================
# Round F8b — THE HONEST CONTROLS, executed by the battery itself (review F8b, MAJOR 2).
#
# Every one of these points production at the LEGITIMATE case and requires it to be ACCEPTED. They
# exist because a refusal proves nothing on its own: a production regression that refused everything
# would leave the attacks printing green REFUSED lines forever. F8 ran these by hand and wrote them
# up; the reviewer's objection is that unexecuted evidence is not evidence, which is the F7d lesson
# in a new costume. They are now part of the recurring run and the runner fails on CONTROL-REFUSED.
#
# **Which families have a control, and which are deliberately attack-only.** A control is only worth
# having where the legitimate case is genuinely available to the harness; manufacturing one would be
# inventing evidence. Covered: the anchor family (T5a-2, G3-1..G3-4), the anchor's TEST screen
# (T5a-3), benchmark publication (T5a-4), the primary gate (T5b-1, T5b-4), the TEST door (T5b-2,
# G3-5, G3-12), the derangement (T5b-3, G3-7, G3-8), the action-use report (T5b-5, G3-10, G3-11),
# the sigma grid (F3a-3, F3a-4), the evaluator's dtype boundary (F3a-5), M1's accumulation plan
# (F1b-2) and M1's adapter construction (W1-3).
#
# **Attack-only, and why**: the authorization/publication families (T7, P3, F5, F6, F7) — their
# "legitimate case" is a full multi-phase publish/adopt cycle against the in-memory bucket, which is
# a fixture with its own failure modes rather than a cheap witness, and several of those probes
# already assert a positive outcome internally (`F5-6` requires two cells BANKED and re-loadable,
# `F7-1` requires the launch AUTHORIZED). The launcher family (W2b) and the source-shape probes
# (G3-13, W2-1, W2-2) assert presence rather than refusal, so a control would restate the probe.
# =================================================================================================


def _m1_program(tmp, **over):
    """M1's real program at test dimensions — the shared path `F1b-2` and `W1-3` measure."""
    from maxdiffusion.tests.worklogs_yixun.test_pos_rollout_fit_probe import (
        _TinySource,
        _install_import_shims,
        _tiny_probe_config,
    )

    fp = _probe_env()
    _install_import_shims()
    microbatch = int(over.pop("pos_microbatch", 2))
    config = _tiny_probe_config(pathlib.Path(tempfile.mkdtemp(dir=tmp)), pos_microbatch=microbatch, **over)
    return fp.build_probe_program(config, fp.FitCell("rollout", microbatch, 2), model_source=_TinySource())


def control_anchor_reproduces_from_a_real_measurement():
    """T5a-2/G3-1..G3-4: the anchor's OWN samples, means and checkpoint must reproduce it.

    Without this, "the anchor refuses everything" and "the anchor refuses what it should" look
    identical — and they were not identical: this control is what exposed `_rows` omitting
    ``grid_sha256``, which had four probes refusing on the grid instead of on their own rules.
    """
    from maxdiffusion import eval_wan_pos_rollout as ev

    verdict = ev.reproduce_anchor(_summary())
    if not getattr(verdict, "reproduced", False):
        return f"CONTROL-REFUSED: the anchor does not reproduce from its own recorded measurement ({verdict})"
    return "CONTROL-PASSED: the recorded samples, means and checkpoint reproduce the anchor"


def control_a_clean_anchor_summary_is_accepted():
    """T5a-3: the same call with no held-out name must SUMMARIZE, or the TEST screen means nothing."""
    from maxdiffusion import eval_wan_pos_rollout as ev

    measurement = ev.summarize_samples(
        _rows(("a", "b")),
        checkpoint=_identity(),
        code_sha="a" * 40,
        model_revision="rev@" + "b" * 40,
        test_manifest_path=TEST,
    )
    if int(measurement.payload["num_samples"]) != 2:
        return f"CONTROL-REFUSED: a clean summary counted {measurement.payload['num_samples']} samples"
    return "CONTROL-PASSED: the same summary without a TEST name is accepted (2 samples)"


def control_an_identical_refreeze_is_adopted(tmp):
    """T5a-4: republishing the SAME benchmark row must be ADOPTED, not refused.

    Issue #10's rule has two halves and the attack only tests one. If publication refused every
    second call, a queue retry could never adopt its own published artifact — and `T5a-4` would sail
    on looking green.
    """
    from maxdiffusion import pos_rollout_dev_instrument as instrument

    anchor, _, dev = _anchor_env()
    cohort = instrument.load_dev_cohort(dev)
    names = list(cohort.names)
    path = str(pathlib.Path(tmp) / "control_bench.json")
    first = anchor.freeze_benchmark_row(path, table=_gate_table(names, 0.25, cohort=cohort))
    second = anchor.freeze_benchmark_row(path, table=_gate_table(names, 0.25, cohort=cohort))
    if first["sha256"] != second["sha256"]:
        return f"CONTROL-REFUSED: an identical re-freeze changed the digest {first['sha256'][:12]} -> {second['sha256'][:12]}"
    return f"CONTROL-PASSED: an identical re-freeze is adopted at digest {first['sha256'][:12]}"


def control_a_real_margin_passes_the_primary_gate():
    """T5b-1/T5b-4: +0.06 with a clean CI must PASS, or the +0.04 refusal is not about the margin."""
    g, cohort, names = _gate_env()
    verdict = g.primary_gate(
        rollout=_gate_table(names, 0.36, cohort=cohort),
        control=_gate_table(names, 0.30, cohort=cohort, arm="control", run="c0"),
        cohort=cohort,
    )
    if not verdict.passed:
        return f"CONTROL-REFUSED: a +0.06 delta failed the +{g.PRIMARY_MARGIN} gate on {list(verdict.reasons)}"
    return f"CONTROL-PASSED: +0.06 passes (mean_delta={verdict.numbers['mean_delta']:.2f}, CI-low > 0)"


def control_a_passing_certificate_opens_the_test_door(tmp):
    """T5b-2/G3-5/G3-12: a genuinely passing DEV certificate must UNLOCK TEST and confirm both gates."""
    g, dev, dev_names = _gate_env()
    test_cohort = g.load_test_cohort(TEST)
    path = str(pathlib.Path(tmp) / "control_dev_cert.json")
    certificate = g.dev_certificate(
        path,
        rollout=_gate_table(dev_names, 0.36, cohort=dev),
        control=_gate_table(dev_names, 0.30, cohort=dev, arm="control", run="c0"),
        cohort=dev,
    )
    if not certificate["passed"]:
        return f"CONTROL-REFUSED: a +0.06 DEV gate did not issue a passing certificate ({certificate['reasons']})"
    names = list(test_cohort.names)
    with _cohort_records(test_cohort):
        art = g.cohort_derangement(test_cohort)
        arm = _arm_battery(g, test_cohort, names, art)
        control_tables = {
            condition: table
            for condition, table in _arm_battery(
                g, test_cohort, names, art, arm="control", run="c0", values=(0.30, 0.29, 0.20, 0.10)
            ).items()
            if condition != "adapter_disabled"
        }
        confirmation = g.confirm_on_test(
            path, test_cohort=test_cohort, derangement=art, tables=arm, control_tables=control_tables
        )
    if not confirmation["confirmed"]:
        return f"CONTROL-REFUSED: TEST did not confirm behind a passing certificate ({confirmation})"
    return "CONTROL-PASSED: a passing DEV certificate opens TEST and both gates confirm"


def control_the_honest_derangement_is_accepted():
    """T5b-3/G3-7/G3-8: the cohort's OWN derangement must build and gate, or the forgery proves nothing."""
    g, cohort, names = _gate_env()
    with _cohort_records(cohort):
        art = g.cohort_derangement(cohort)
        verdict = g.action_use_gate(
            true_table=_gate_table(names, 0.36, cohort=cohort, derangement=art),
            wrong_table=_gate_table(names, 0.30, cohort=cohort, condition="wrong", derangement=art),
            cohort=cohort,
            derangement=art,
        )
    if not verdict.passed:
        return f"CONTROL-REFUSED: the honest derangement failed its own gate on {list(verdict.reasons)}"
    return f"CONTROL-PASSED: the cohort's own derangement builds and gates (delta={verdict.numbers['mean_delta']:.2f})"


def control_the_full_battery_publishes():
    """T5b-5/G3-10/G3-11: WITH matched-C0's battery the action-use report must publish."""
    g, cohort, names = _gate_env()
    with _cohort_records(cohort):
        art = g.cohort_derangement(cohort)
        control_tables = {
            condition: table
            for condition, table in _arm_battery(
                g, cohort, names, art, arm="control", run="c0", values=(0.33, 0.32, 0.20, 0.10)
            ).items()
            if condition != "adapter_disabled"
        }
        report = g.action_use_report(
            cohort, derangement=art, tables=_arm_battery(g, cohort, names, art), control_tables=control_tables
        )
    if not report["reported"].get("coverage_ok"):
        return f"CONTROL-REFUSED: a complete battery reported incomplete coverage ({report['reported']})"
    return "CONTROL-PASSED: the complete arm + matched-C0 battery publishes with full coverage"


def control_the_deployed_grid_is_accepted():
    """F3a-3/F3a-4: the grid production itself builds must PASS its own check."""
    from maxdiffusion import eval_wan_pos_rollout as ev

    digest = ev.assert_deployed_grid(*ev.deployed_grid())
    if digest != ev.DEPLOYED_GRID_SHA256:
        return f"CONTROL-REFUSED: the deployed grid hashes {digest[:16]}, pinned {ev.DEPLOYED_GRID_SHA256[:16]}"
    return f"CONTROL-PASSED: the deployed grid passes its own check ({digest[:16]})"


def control_the_eval_dtype_follows_the_config():
    """F3a-5: under a float32 configuration the SAME path must produce float32.

    The attack observes all-bf16 under a bf16 config. That is only evidence of a cast if the cast
    follows the CONFIG — a backend that hardcoded bf16 would pass the attack and silently mismeasure
    every fp32 debugging run.
    """
    import jax.numpy as jnp
    import numpy as np

    from maxdiffusion import eval_wan_pos_rollout as ev

    seen = {}

    def velocity_builder(params, frozen_state, actions, adapter_enabled):
        seen["actions"] = actions.dtype
        del params, frozen_state, adapter_enabled

        def velocity_fn(hidden_states, timestep, encoder_hidden_states):
            seen["latents"] = hidden_states.dtype
            seen["context"] = encoder_hidden_states.dtype
            del timestep
            return jnp.zeros_like(hidden_states)

        return velocity_fn

    sigmas, timesteps = ev.deployed_grid()
    params = {"w": jnp.zeros((1,), jnp.float32)}
    backend = ev.DeviceBackend(
        kernel=ev.build_rollout_kernel(velocity_builder),
        decode_fn=lambda x: jnp.asarray(np.repeat(np.asarray(x, np.float32).mean(axis=1)[..., None], 3, axis=-1)),
        sigmas=sigmas,
        timesteps=timesteps,
        context=jnp.zeros((1, 7, 8), jnp.float32),
        guide_scale=5.0,
        params=params,
        eval_dtype=jnp.float32,
        frozen_state=None,
    ).bound(params)
    execution, _ = backend.score(
        z_i0=jnp.zeros((1, 4, 1, 4, 6), jnp.float32),
        z_video=jnp.zeros((1, 4, 2, 4, 6), jnp.float32),
        actions=jnp.zeros((1, 4, 7), jnp.float32),
        key=ev.evaluation_draw_key("x"),
    )
    observed = {"z_pred": execution.z_pred.dtype, **{key: seen.get(key) for key in ("latents", "actions", "context")}}
    wrong = sorted(name for name, dtype in observed.items() if dtype != jnp.float32)
    if wrong:
        return f"CONTROL-REFUSED: under a float32 config {wrong} did not come back float32 ({observed})"
    return "CONTROL-PASSED: the cast follows the config -- a float32 configuration measures in float32"


def control_the_accumulation_tracks_the_config(tmp):
    """F1b-2: the microbatch count must FOLLOW the config, not sit at a constant that happens to pass."""
    program = _m1_program(tmp, pos_logical_batch=16, pos_microbatch=2)
    parts = len(program.batch) if isinstance(program.batch, tuple) else 1
    if parts != 8 or len(program.draws) != 8:
        return (
            f"CONTROL-REFUSED: a 16/2 logical batch built {parts} microbatches and {len(program.draws)} draws, not 8"
        )
    return "CONTROL-PASSED: the accumulation plan tracks the config (16/2 -> 8 microbatches, 8 draws)"


def control_the_adapter_follows_the_configured_dtype(tmp):
    """W1-3: M1's adapter must actually BE the configured dtype, not merely differ between two."""
    import jax

    program = _m1_program(tmp, weights_dtype="bfloat16", activations_dtype="bfloat16")
    dtypes = {str(leaf.dtype) for leaf in jax.tree.leaves(program.params)}
    if "bfloat16" not in dtypes:
        return f"CONTROL-REFUSED: a bfloat16 configuration produced an adapter carrying {sorted(dtypes)}"
    return f"CONTROL-PASSED: a bfloat16 configuration produces a bfloat16 adapter ({sorted(dtypes)})"


def control_an_analysis_bounded_cell_authorizes(tmp):
    """F10-1/F10-2/F10-3's honest case: **the shape M1-9 actually measured must AUTHORIZE.**

    This is the control the whole round exists for. The rule F10 replaced was not merely strict, it
    was unsatisfiable on this hardware — twelve measured cells, zero authorizations — so a battery
    that only showed F10 refusing things would witness nothing about whether the new rule can ever
    say yes. Here a cell with a 20 GiB analysis under a 32 GiB device and a 4 GiB watermark below it
    is published, re-decided on load, and gates a training run.
    """
    fp = _probe_env()
    context = _ctx(fp)
    cell = fp.FitCell("rollout", 32, 2)
    table, refusal = _f10_gate(fp, tmp, "f10_control.json", _fit(fp, context), context)
    if refusal is not None:
        return f"CONTROL-REFUSED: the measured M1-9 shape does not authorize -- {refusal}"
    if list(table["authorized_cells"]) != [cell.as_payload()]:
        return f"CONTROL-REFUSED: the table authorized {table['authorized_cells']}"
    numbers = table["projections"][0]
    if numbers["authorized_bytes"] != 20 * 1024**3 or numbers["watermark_bytes"] != 4 * 1024**3:
        return f"CONTROL-REFUSED: the published projection does not carry both numbers ({numbers})"
    return (
        f"CONTROL-PASSED: an analysis-bounded cell ({numbers['authorized_fraction']:.1%} of capacity, watermark "
        f"below the bound) is authorized and gates a launch"
    )


def control_a_watermark_at_the_bound_is_not_an_excess(tmp):
    """F10-1's boundary: equality is not "exceeds". A mark exactly at the analysis must authorize.

    Without this the cross-check could be off by one in the safe-looking direction and no attack in
    this battery would notice: every refusal above would still be green.
    """
    fp = _probe_env()
    context = _ctx(fp)
    measurement = _fit(
        fp,
        context,
        peak_bytes=20 * 1024**3,
        analysis_bytes=20 * 1024**3,
        watermark_bytes=20 * 1024**3,
        peak_source=fp.PEAK_SOURCE_RUNTIME_RAISED,
        peak_attribution=fp.PEAK_ATTRIBUTION_RAISED,
    )
    _, refusal = _f10_gate(fp, tmp, "f10_boundary.json", measurement, context)
    if refusal is not None:
        return f"CONTROL-REFUSED: a watermark EQUAL to the bound was treated as exceeding it -- {refusal}"
    return "CONTROL-PASSED: watermark == analysis authorizes; only a mark ABOVE the bound refuses"


def control_two_agreeing_trials_authorize(tmp):
    """F10b's honest case: **the ladder measures every cell twice**, so if the new per-trial rules
    over-refused, M1-10 would authorize nothing at all — the exact outage F10 was written to end.

    Two trials that agree on the bound, differ only in their marks (both under it) and carry a full
    evidence pair must aggregate to one authorized cell and gate a launch.
    """
    fp = _probe_env()
    context = _ctx(fp)
    trials = [
        _fit(fp, context, watermark_bytes=4 * 1024**3),
        _fit(fp, context, watermark_bytes=5 * 1024**3),
    ]
    table, refusal = _f10_gate(fp, tmp, "f10b_control_trials.json", trials, context)
    if refusal is not None:
        return f"CONTROL-REFUSED: two agreeing trials of a fitting cell do not authorize -- {refusal}"
    if len(table["measured_cells"]) != 1 or not table["authorized_cells"]:
        return f"CONTROL-REFUSED: the table is {table['measured_cells']} / {table['authorized_cells']}"
    recorded = table["measurements"][0]
    if recorded["watermark_bytes"] != 5 * 1024**3 or recorded["analysis_bytes"] != 20 * 1024**3:
        return f"CONTROL-REFUSED: the aggregate lost the worst mark or the shared bound ({recorded})"
    return "CONTROL-PASSED: two agreeing trials aggregate to one authorized cell, worst mark kept"


def control_a_consistent_banked_cell_still_adopts(tmp):
    """F10c's honest case, and the one the quarantine could have broken: **banking still works.**

    The adoption path now validates that a cached artifact's trials aggregate. If that check were
    wrong — or if banking refused to write a perfectly good cell — every restart would re-measure
    the whole ladder and F5's entire reason for existing would be gone, silently, behind two green
    refusals above.
    """
    fp = _probe_env()
    _f5_publish(fp, tmp, "adoptcontrol")
    try:
        calls, table = _f5_resume(fp, tmp, "adoptcontrol")
    except ValueError as error:
        return f"CONTROL-REFUSED: a consistent banked cell could not be resumed at all -- {str(error)[:110]}"
    if calls:
        return f"CONTROL-REFUSED: a consistent banked cell was re-measured ({len(calls)} trials)"
    adopted = [row for row in table["cell_provenance"] if row["provenance"].startswith(fp.ADOPTED_PREFIX)]
    if not adopted or not table["authorized_cells"]:
        return f"CONTROL-REFUSED: nothing was adopted or nothing authorized ({table['cell_provenance']})"
    return "CONTROL-PASSED: a consistent banked cell is still adopted without measurement, and still authorizes"


def control_a_real_analysis_still_sums(tmp):
    """F10e's control: a well-formed analysis -- INCLUDING legitimate zero components -- still sums.

    The fix discards a whole analysis on one bad component. If it discarded a good one too, every
    cell would refuse on `analysis_missing`, which is the M1-6 outage wearing a new reason, and every
    attack above would still print REFUSED.
    """
    fp = _probe_env()
    program = _f10e_program(
        fp,
        argument_size_in_bytes=40,
        temp_size_in_bytes=0,
        output_size_in_bytes=0,
        alias_size_in_bytes=0,
    )
    analysis = fp._program_bytes(program, program.params, program.opt_state)
    if analysis != 40:
        return f"CONTROL-REFUSED: a well-formed analysis with a zero component summed to {analysis!r}, not 40"
    verdict = fp.cell_verdict(
        _fit(fp, peak_bytes=analysis, analysis_bytes=analysis, watermark_bytes=5, capacity_bytes=100)
    )
    if not verdict.fits:
        return f"CONTROL-REFUSED: a 40-byte bound on a 100-byte device did not authorize ({verdict.reasons})"
    return "CONTROL-PASSED: a real analysis still sums (zero components included) and still authorizes"


def control_the_ordinary_numbers_still_parse(tmp):
    """F10d's control, and the one an exactness sweep most needs: **the honest values still work.**

    Every number in this module is now parsed by one strict parser. A parser that is too strict
    fails the same way a truncation does — silently, in production, at 3.5 hours in — and no attack
    above would notice, because they all assert refusals. So: the deployed ladder builds, a cell
    round-trips through its payload unchanged, an integral float is a whole number, the real
    cadences project, and a well-formed artifact still loads and adopts.
    """
    fp = _probe_env()
    context = _ctx(fp)
    cells = fp.ladder()
    if (
        len(cells) != 16
        or fp.cell_artifact_name(cells[0]) != f"{cells[0].arm}_m{cells[0].microbatch}_k{cells[0].k_b}.json"
    ):
        return f"CONTROL-REFUSED: the declared ladder no longer builds ({len(cells)} cells)"
    if fp.FitCell("rollout", 8.0, 2.0) != fp.FitCell("rollout", 8, 2):
        return "CONTROL-REFUSED: an integral float is no longer accepted as the whole number it is"

    measurement = _fit(fp, context)
    if fp.CellMeasurement.from_payload(measurement.as_payload()) != measurement:
        return "CONTROL-REFUSED: an honest measurement no longer round-trips through its payload"
    if fp.ProbeContext.from_payload(context.as_payload()).binding_digest() != context.binding_digest():
        return "CONTROL-REFUSED: an honest context no longer round-trips to the same binding"
    projected = fp.project_wall_clock(measurement, max_train_steps=10_000, eval_every=1_000, checkpoint_every=1_000)
    if projected["max_train_steps"] != 10_000 or projected["total_hours"] <= 0:
        return f"CONTROL-REFUSED: the real cadences no longer project ({projected.get('max_train_steps')!r})"

    _f5_publish(fp, tmp, "parsecontrol")
    calls, table = _f5_resume(fp, tmp, "parsecontrol")
    if calls or not table["authorized_cells"]:
        return f"CONTROL-REFUSED: a well-formed banked cell was re-measured ({len(calls)} trials)"
    return (
        "CONTROL-PASSED: the ladder, the payload round trip, the binding, the projection and adoption all still work"
    )


def control_exactly_ninety_percent_authorizes(tmp):
    """F10b's boundary on the other side: the headroom rule did not get stricter, only exact.

    A cell whose bound is EXACTLY 90% of capacity must still authorize — otherwise the integer
    comparison would have quietly moved the pinned line, and every refusal above it would be
    evidence of nothing.
    """
    fp = _probe_env()
    context = _ctx(fp)
    capacity = 40 * 1024**3
    exactly = capacity * fp.HEADROOM_NUMERATOR // fp.HEADROOM_DENOMINATOR
    if exactly * fp.HEADROOM_DENOMINATOR != capacity * fp.HEADROOM_NUMERATOR:
        return "CONTROL-REFUSED: the chosen capacity does not have an exact 90% point"
    measurement = _fit(
        fp, context, peak_bytes=exactly, analysis_bytes=exactly, watermark_bytes=1, capacity_bytes=capacity
    )
    table, refusal = _f10_gate(fp, tmp, "f10b_control_exact.json", measurement, context)
    if refusal is not None:
        return f"CONTROL-REFUSED: a bound at exactly {fp.HEADROOM_FRACTION:.0%} of capacity was refused -- {refusal}"
    fraction = table["projections"][0]["authorized_fraction"]
    return f"CONTROL-PASSED: a bound at exactly {fraction:.1%} of capacity authorizes; one byte above refuses"


if __name__ == "__main__":
    _report("A-B1(a) module issue token   :", attack_a_b1a)
    _report("A-B1(b) public digest override:", attack_a_b1b)
    _report("A-B2   unrestricted callback :", attack_a_b2)
    with tempfile.TemporaryDirectory() as tmp:
        _report("B-1    terminal resume       :", attack_b1, tmp)
        _report("B-2    selection crash window:", attack_b2, tmp)
        _report("T5a-1  restore falls back     :", attack_t5a_restore_falls_back)
        _report("T5a-2  widen/miss the anchor  :", attack_t5a_widen_the_anchor)
        _control("  ctrl anchor reproduces      :", control_anchor_reproduces_from_a_real_measurement)
        _report("T5a-3  TEST into the anchor   :", attack_t5a_test_into_the_anchor)
        _control("  ctrl clean summary accepted :", control_a_clean_anchor_summary_is_accepted)
        _report("T5a-4  re-derive the benchmark:", attack_t5a_rederive_the_benchmark, tmp)
        _control("  ctrl identical refreeze     :", control_an_identical_refreeze_is_adopted, tmp)
        _report("T5a-5  forge a DEV cohort     :", attack_t5a_forge_a_dev_cohort, tmp)
        _report("T5b-1  lower the primary bar  :", attack_t5b_lower_the_bar)
        _control("  ctrl +0.06 passes the gate  :", control_a_real_margin_passes_the_primary_gate)
        _report("T5b-2  score TEST first       :", attack_t5b_score_test_first, tmp)
        _control("  ctrl real cert opens TEST   :", control_a_passing_certificate_opens_the_test_door, tmp)
        _report("T5b-3  forge the derangement  :", attack_t5b_forge_the_derangement)
        _control("  ctrl honest derangement     :", control_the_honest_derangement_is_accepted)
        _report("T5b-4  swap arm and control   :", attack_t5b_swap_arm_and_control)
        _report("T5b-5  drop C0's battery      :", attack_t5b_drop_the_control_battery)
        _control("  ctrl full battery publishes :", control_the_full_battery_publishes)
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
        _control("  ctrl deployed grid accepted :", control_the_deployed_grid_is_accepted)
        _report("F3a-5  float32 under bf16 cfg  :", attack_f3a_score_in_float32_under_a_bf16_config)
        _control("  ctrl dtype follows config   :", control_the_eval_dtype_follows_the_config)
        _report("F1b-1  wrong adapter type   ", attack_f1b_wrong_adapter, tmp)
        _report("F1b-2  microbatch as update ", attack_f1b_microbatch_timed_as_update, tmp)
        _control("  ctrl accumulation tracks cfg:", control_the_accumulation_tracks_the_config, tmp)
        _report("F1b-3  inherited peak       ", attack_f1b_inherited_peak, tmp)
        _report("F1b-4  'boom' is not OOM    ", attack_f1b_boom_is_not_oom, tmp)
        _report("F1b-5  swap the weights     ", attack_f1b_swap_the_weights, tmp)
        _report("F1b-6  retype a duration    ", attack_f1b_retype_a_duration, tmp)
        _report("F3a-6  loader: undeclared key  :", attack_f3afix_loader_reads_an_undeclared_key)
        _report("F3a-7  loader: skips grid check:", attack_f3afix_loader_skips_the_grid_check)
        _report("W1-1   authorize on a floor ", attack_w1_authorize_on_a_floor, tmp)
        _report("W1-2   reshuffle a snapshot ", attack_w1_reshuffle_a_snapshot, tmp)
        _report("W1-3   hand-rebuild adapter ", attack_w1_hand_rebuild_the_adapter, tmp)
        _control("  ctrl adapter follows dtype  :", control_the_adapter_follows_the_configured_dtype, tmp)
        _report("W2-1   trainer cannot train  ", attack_w2_the_trainer_still_cannot_train)
        _report("W2-2   bypass the factories  ", attack_w2_bypass_the_shared_factories)
        _report("W2-3   publish a foreign tree", attack_w2_publish_an_attempt_for_a_foreign_tree, tmp)
        _report("W2-4   select on train stream", attack_w2_select_on_the_training_stream)
        _report("W2-5   gate after the load   ", attack_w2_gate_after_the_load, tmp)
        _report("W2b-1  M2 at the YAML batch  ", attack_w2b_launch_m2_with_the_yaml_per_device_batch, tmp)
        _report("W3-1   unsharded M1 program ", attack_w3_measure_an_unsharded_program)
        _report("W3-2   zero null context     ", attack_w3_zero_null_context)
        _report("W3-3   two loaders, two progs", attack_w3_the_seams_diverge)
        _report("W4-1   foreign batch contract ", attack_w4_compile_against_a_batch_production_never_hands_it)
        _report("W4-2   time a pruned scorer   ", attack_w4_time_a_pruned_scorer)
        # F5: per-cell publication and adoption -- the attack surface "resume from what is
        # published" adds, one probe per known defect of the pattern.
        _report("F5-1   lose bank to docs commit", attack_f5_lose_the_bank_to_a_docs_commit, tmp)
        _report("F5-2   adopt another topology", attack_f5_adopt_a_cell_measured_on_another_topology, tmp)
        _report("F5-3   adopt another job     ", attack_f5_adopt_another_jobs_cell, tmp)
        _report("F5-4   adopt a half-write    ", attack_f5_adopt_a_half_published_cell, tmp)
        _report("F5-5   forge w/ FOREIGN manifest", attack_f5_adopt_a_fabricated_favourable_cell, tmp)
        _report("F5-6   killed ladder banks 0 ", attack_f5_a_killed_ladder_banks_nothing, tmp)
        _report("F5-7   tear the pair (2 writers)", attack_f5_tear_the_pair_with_two_publishers, tmp)
        _report("F5-8   forge w/ CURRENT manifest", attack_f5_forge_with_the_CURRENT_manifest, tmp)
        _report("F6-1   quote an excluded cell", attack_f6_quote_an_excluded_cell, tmp)
        _report("F6-2   doctor two statuses", attack_f6_doctor_a_table_into_two_statuses, tmp)
        _report("F7-1   block launch on a label", attack_f7_refuse_a_launch_over_a_docs_commit, tmp)
        _report("F7-2   authorize another build", attack_f7_authorize_a_different_build_under_the_same_label, tmp)
        _report("F10-1  mark above the bound   ", attack_f10_mark_above_the_bound, tmp)
        _control("  ctrl mark AT the bound      :", control_a_watermark_at_the_bound_is_not_an_excess, tmp)
        _report("F10-2  no bound at all        ", attack_f10_no_bound_at_all, tmp)
        _report("F10-3  peak above a quiet bound", attack_f10_peak_above_a_quiet_bound, tmp)
        _control("  ctrl analysis-bounded cell  :", control_an_analysis_bounded_cell_authorizes, tmp)
        _report("F10-4  read a v6 table as v7  ", attack_f10_read_a_v6_table_as_a_v7_one, tmp)
        _report("F10b-1 no mark at all         ", attack_f10b_no_mark_at_all, tmp)
        _report("F10b-2 friendly trial cancels ", attack_f10b_a_friendly_trial_cancels_a_contradicted_one, tmp)
        _control("  ctrl two agreeing trials    :", control_two_agreeing_trials_authorize, tmp)
        _report("F10b-3 two bounds, one cell   ", attack_f10b_two_bounds_for_one_executable, tmp)
        _report("F10b-4 fractional byte counts ", attack_f10b_fractional_bytes, tmp)
        _report("F10b-5 round the boundary down", attack_f10b_round_the_headroom_boundary_down, tmp)
        _control("  ctrl exactly 90% authorizes :", control_exactly_ninety_percent_authorizes, tmp)
        _report("F10c-1 poison the bank forever", attack_f10c_poison_the_bank_forever, tmp)
        _report("F10c-2 bank a malformed count ", attack_f10c_bank_a_malformed_count, tmp)
        _control("  ctrl consistent bank adopts :", control_a_consistent_banked_cell_still_adopts, tmp)
        _report("F10d-1 truncate id/binding/cnt", attack_f10d_truncate_an_identity_a_binding_and_a_count, tmp)
        _report("F10d-2 bank a fractional bind ", attack_f10d_bank_a_fractional_binding, tmp)
        _control("  ctrl ordinary numbers parse :", control_the_ordinary_numbers_still_parse, tmp)
        _report("F10e-1 shrink the bound       ", attack_f10e_shrink_the_bound_with_a_bad_component, tmp)
        _control("  ctrl a real analysis sums   :", control_a_real_analysis_still_sums, tmp)
    if not _summarize():
        raise SystemExit(1)
