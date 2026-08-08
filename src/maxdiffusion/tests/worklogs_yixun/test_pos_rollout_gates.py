"""exp_06 `rollout_adapter` — T5b `eval-gates`: the two gates that decide the experiment (§3c/§3e).

**The primary gate** asks whether the rollout objective beat its matched control: mean PAIRED
per-example ΔSSIM(R-B − C0) on DEV-64 ≥ +0.05 with a paired-bootstrap CI-low > 0. The decision
function is exp_04's, by import — re-deriving a bootstrap is how two experiments end up with two
different definitions of the same claim — and what is imported is PINNED here, because "we reuse
exp_04's gate" is a claim about a call, not about a margin.

**The action-use gate** is the one that makes this an action-conditioned claim at all. SSIM alone
cannot distinguish a world model from a video autoencoder that ignores its actions: an adapter that
learned to reconstruct plausible robot footage from the first frame scores well and knows nothing
about the actions. So every example is evaluated three times on IDENTICAL noise — true actions, another
example's actions, and zeros — and the gate is that true beats wrong. The wrong-action assignment is a
cohort-level seeded derangement with its permutation and hash persisted, because "we shuffled the
actions" is unreproducible unless the shuffle is written down.

What is deliberately NOT a gate is reported anyway: the zero-action row, the adapter-disabled
diagnostic, and the same battery on matched-C0 — does one-step training use actions more or less than
rollout training? Either answer is a finding, and neither may quietly become a pass condition.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

from maxdiffusion import eval_wan_pos_rollout as anchor
from maxdiffusion import null_adapter_gates as exp04
from maxdiffusion import pos_rollout_dev_instrument as instrument
from maxdiffusion import pos_rollout_gates as gates

_MODULE_PATH = Path(gates.__file__).resolve()
_MANIFEST_DIR = _MODULE_PATH.parents[2] / "docs" / "worklogs_yixun" / "exp_04_null_adapter_claude" / "j0_manifests"
_DEV = str(_MANIFEST_DIR / "dev64.json")
_TEST = str(_MANIFEST_DIR / "test64.json")


def _cohort():
    return instrument.load_dev_cohort(_DEV)


def _checkpoint(run="rb", arm="rollout"):
    return anchor.CheckpointIdentity(
        run_name=run, step=10000, root=f"gs://b/{run}/checkpoints", source="selection", arm=arm, k_b=2, dev_metric=0.3
    )


def _table(
    names,
    ssim,
    mse=1.0,
    *,
    cohort=None,
    condition="true",
    arm="rollout",
    checkpoint=None,
    incomplete=False,
    derangement=None,
    draw_keys=None,
):
    """exp_06's per-example shape as the ARTIFACT the gates now require.

    Every row carries the identities a produced table carries — whose actions were fed, their digest,
    and the digest of the pinned draw — because a gate that reads bare numbers can only check numbers.
    The digests come from the DERANGEMENT artifact when one is supplied, which is exactly what the
    gate cross-checks: a hand-written digest would not match what the cohort's records actually hold.
    """
    cohort = _cohort() if cohort is None else cohort
    values = {name: (float(ssim(index)) if callable(ssim) else float(ssim)) for index, name in enumerate(names)}
    rows = {}
    for name, value in values.items():
        if condition == "zero":
            donor = None
        elif condition == "wrong" and derangement is not None:
            donor = derangement.permutation[name]
        else:
            donor = name
        if derangement is not None and donor is not None:
            digest = derangement.action_sha256[donor]
        else:
            digest = hashlib.sha256(str(donor).encode()).hexdigest()
        drawn = name if draw_keys is None else draw_keys.get(name, name)
        rows[name] = {
            "ssim": value,
            "mse": mse,
            "actions_from": donor,
            "actions_sha256": digest,
            "draw_key_sha256": anchor.draw_key_digest(anchor.evaluation_draw_key(drawn)),
            "num_steps": 25,
        }
    return anchor.build_score_table(
        rows=rows,
        cohort=cohort,
        condition=condition,
        arm=arm,
        checkpoint=_checkpoint(arm=arm) if checkpoint is None else checkpoint,
        num_steps=25,
        derangement_sha256=derangement.fingerprint if (condition == "wrong" and derangement is not None) else None,
        allow_incomplete=incomplete,
    )


def _control(names, ssim, **kwargs):
    return _table(names, ssim, arm="control", checkpoint=_checkpoint(run="c0", arm="control"), **kwargs)


def _wrong(names, ssim, derangement, **kwargs):
    return _table(names, ssim, condition="wrong", derangement=derangement, **kwargs)


def _battery(names, derangement, values, *, arm="rollout", run="rb", cohort=None):
    """One arm's condition tables at chosen SSIMs — the shape ``action_use_report`` requires."""
    return {
        condition: _table(
            names,
            value,
            condition=condition,
            arm=arm,
            cohort=cohort,
            checkpoint=_checkpoint(run=run, arm=arm),
            derangement=derangement,
        )
        for condition, value in values.items()
    }


# =============================================================================================
# 1. The primary gate (§3c): exp_04's decision function, and the numbers it decides with.
# =============================================================================================


def test_the_primary_gate_is_exp04s_decision_function_and_its_constants_are_pinned():
    """ "We reuse exp_04's gate" is a claim about a call; these are the numbers it decides with."""
    assert exp04.BOOTSTRAP_RESAMPLES == 10_000, "plan §3c: 10k resamples"
    assert exp04.BOOTSTRAP_SEED == 20260804, "plan §3c: seed 20260804"
    assert exp04.CI_PERCENTILES == (2.5, 97.5)
    assert exp04.IMPUTED_DELTA == -1.0, "an invalid pair must penalize the CLAIM, not flatter it"
    assert exp04.MAX_INVALID_FRACTION == 0.10
    assert gates.PRIMARY_MARGIN == 0.05
    assert exp04.GLOBAL_K_SET == (0,), "exp_06 scores one pinned draw per example"
    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("null_adapter_gates")
        for alias in node.names
    }
    assert "gate_g3_vs_null_only" in imported, "the paired-delta decision is imported, never re-derived"
    # `resamples`/`seed` appear as RECORDED provenance, so the scan is on computation, not on words.
    for forbidden in ("np.percentile", "_bootstrap_ci", "IMPUTED_DELTA =", "MAX_INVALID_FRACTION ="):
        assert forbidden not in source, f"{forbidden}: the bootstrap has one definition, and it is exp_04's"
    # The module owns exactly one RNG, and it is the derangement's -- not a second bootstrap.
    seeded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(call, ast.Attribute) and call.attr == "default_rng" for call in ast.walk(node))
    ]
    assert [node.name for node in seeded] == ["cohort_derangement"], [n.name for n in seeded]


def test_the_primary_gate_passes_only_on_margin_AND_ci():
    cohort = _cohort()
    names = list(cohort.names)
    passing = gates.primary_gate(rollout=_table(names, 0.36), control=_control(names, 0.30), cohort=cohort)
    assert passing.passed and passing.numbers["mean_delta"] == pytest.approx(0.06)
    assert passing.numbers["ci"][0] > 0.0
    # +0.04 is a real improvement with a clean CI and still fails: the margin is the claim.
    narrow = gates.primary_gate(rollout=_table(names, 0.34), control=_control(names, 0.30), cohort=cohort)
    assert not narrow.passed and "mean_delta" in narrow.reasons
    # A mean that CLEARS the margin but whose CI straddles zero fails on the CI alone: a +0.05 mean
    # produced by half the cohort improving 0.8 and half degrading 0.7 is not evidence of anything.
    noisy = gates.primary_gate(
        rollout=_table(names, lambda i: 0.9 if i % 2 == 0 else 0.1),
        control=_control(names, lambda i: 0.1 if i % 2 == 0 else 0.7),
        cohort=cohort,
    )
    assert noisy.numbers["mean_delta"] >= gates.PRIMARY_MARGIN, noisy.numbers["mean_delta"]
    assert not noisy.passed and noisy.reasons == ("ci_excludes_zero",)
    assert noisy.numbers["ci"][0] < 0.0


def test_the_primary_gate_refuses_to_compare_an_arm_with_itself_or_a_wrong_condition():
    """Review pass 3 found the two arms sharing checkpoint state by default. A gate that compares one
    checkpoint with itself reports a delta of exactly zero as a finding, so it is refused here too."""
    cohort = _cohort()
    names = list(cohort.names)
    with pytest.raises(ValueError, match="same|matched-C0 is a different run|checkpoint"):
        gates.primary_gate(rollout=_table(names, 0.36), control=_table(names, 0.30), cohort=cohort)
    with pytest.raises(ValueError, match="condition"):
        gates.primary_gate(rollout=_table(names, 0.36, condition="zero"), control=_control(names, 0.30), cohort=cohort)


def test_the_comparison_is_PAIRED_per_example_not_a_difference_of_means():
    """Two tables with identical means but opposite per-example structure must not be equivalent."""
    cohort = _cohort()
    names = list(cohort.names)
    rollout = _table(names, lambda i: 0.6 if i % 2 else 0.2)
    control = _control(names, lambda i: 0.2 if i % 2 else 0.6)
    verdict = gates.primary_gate(rollout=rollout, control=control, cohort=cohort)
    per_example = verdict.numbers["per_example_delta"]
    assert set(np.round(per_example, 6)) == {0.4, -0.4}, "the deltas are per example"
    assert float(np.mean(per_example)) == pytest.approx(0.0)
    assert not verdict.passed


def test_coverage_and_imputation_stay_claim_penalizing():
    cohort = _cohort()
    names = list(cohort.names)
    # a missing example is a coverage failure, never a silent 63-example mean
    holed = _table(names[:63], 0.36, incomplete=True)
    missing = gates.primary_gate(rollout=holed, control=_control(names, 0.30), cohort=cohort)
    assert not missing.passed and "coverage" in missing.reasons

    unusable = _table(names, lambda i: float("nan") if i == 0 else 0.36)
    imputed = gates.primary_gate(rollout=unusable, control=_control(names, 0.30), cohort=cohort)
    assert imputed.numbers["per_example_delta"][0] == exp04.IMPUTED_DELTA
    assert imputed.numbers["invalid_pairs"] == 1


def test_the_gate_is_bound_to_the_approved_manifest():
    """The manifest decides which examples are scored; a caller cannot bring its own name list."""
    parameters = list(inspect.signature(gates.primary_gate).parameters)
    assert parameters[-1] == "cohort"
    assert "manifest" not in parameters and "names" not in parameters
    with pytest.raises(TypeError, match="LOADED cohort"):
        gates.primary_gate(rollout={}, control={}, cohort=["ep12399_v0_s00000"])
    names = list(_cohort().names)
    verdict = gates.primary_gate(rollout=_table(names, 0.36), control=_control(names, 0.30), cohort=_cohort())
    assert verdict.numbers["examples"] == 64
    assert verdict.numbers["manifest_sha256"] == hashlib.sha256(Path(_DEV).read_bytes()).hexdigest()


# =============================================================================================
# 2. THE ACTION-USE GATE (§3e) — the cohort-level seeded derangement, now an artifact.
# =============================================================================================


@pytest.fixture
def records(monkeypatch):
    """Point the canonical reader at in-memory records, so a derangement can be DERIVED in a test."""
    _install_records(monkeypatch, instrument.load_dev_cohort(_DEV).rows)
    return monkeypatch


def _derangement(cohort=None):
    return gates.cohort_derangement(_cohort() if cohort is None else cohort)


def test_the_derangement_has_no_fixed_point_and_is_a_permutation(records):
    artifact = _derangement()
    mapping = artifact.permutation
    assert set(mapping) == set(_cohort().names) and len(mapping) == 64
    assert not [name for name, donor in mapping.items() if donor == name], "a fixed point is TRUE actions"
    assert sorted(mapping.values()) == sorted(_cohort().names), "every example donates exactly once"


def test_the_derangement_is_seeded_deterministic_and_independent_per_cohort(records):
    first = _derangement()
    assert first == _derangement(), "not reproducible"
    assert gates.DERANGEMENT_SEED == 20260804
    # The independence is a property of the SEED derivation, and it is checked directly: the two
    # cohorts' seeds must differ, so DEV's assignment tells you nothing about TEST's.
    assert gates._cohort_seed("dev64") != gates._cohort_seed("test64")


def test_a_byte_identical_replacement_is_repaired_by_a_BIJECTIVE_swap(monkeypatch):
    """Two examples with byte-identical action sequences would make 'wrong actions' mean 'the same
    actions' for that pair — a silent fixed point that the name check cannot see. Plan v2.6: the
    repair is a SWAP, so the assignment stays a rearrangement of the same action set."""
    rows = instrument.load_dev_cohort(_DEV).rows
    names = [str(row["name"]) for row in rows]
    _install_records(monkeypatch, rows)
    plain = _derangement()
    _install_records(monkeypatch, rows, collide={names[1]: names[0]})
    repaired = _derangement()
    assert sorted(repaired.permutation.values()) == sorted(names), "the repair must preserve the permutation"
    for name, donor in repaired.permutation.items():
        assert donor != name
        assert repaired.action_sha256[donor] != repaired.action_sha256[name], f"{name} got identical actions"
    assert repaired.permutation != plain.permutation or plain.permutation[names[0]] != names[1]


def test_a_cohort_with_no_legal_derangement_fails_closed(monkeypatch):
    _install_records(monkeypatch, instrument.load_dev_cohort(_DEV).rows, fill_of=lambda row: 1.0)
    with pytest.raises(ValueError, match="no legal wrong-action assignment"):
        _derangement()


def test_the_permutation_and_its_hash_are_persisted_with_the_verdict(records):
    cohort = _cohort()
    names = list(cohort.names)
    artifact = _derangement(cohort)
    verdict = gates.action_use_gate(
        true_table=_table(names, 0.36, derangement=artifact),
        wrong_table=_wrong(names, 0.30, artifact),
        cohort=cohort,
        derangement=artifact,
    )
    assert verdict.numbers["derangement"] == artifact.permutation, "the permutation itself, not just its hash"
    assert verdict.numbers["derangement_sha256"] == artifact.fingerprint
    assert len(verdict.numbers["derangement_sha256"]) == 64
    # ...and the fingerprint moves with the assignment, so a swapped permutation cannot reuse it.
    swapped = dict(artifact.permutation)
    swapped[names[0]], swapped[names[1]] = swapped[names[1]], swapped[names[0]]
    assert gates.derangement_fingerprint(dataclasses.replace(artifact, permutation=swapped)) != artifact.fingerprint


def test_the_action_use_gate_is_ci_low_only_and_carries_no_plus_005_margin(records):
    """§3e's gate is 'true beats wrong', not 'true beats wrong by the primary margin'."""
    cohort = _cohort()
    names = list(cohort.names)
    artifact = _derangement(cohort)
    assert gates.ACTION_USE_MARGIN == 0.0
    for true_value, expected in ((0.31, True), (0.30, False), (0.20, False)):
        verdict = gates.action_use_gate(
            true_table=_table(names, true_value, derangement=artifact),
            wrong_table=_wrong(names, 0.30, artifact),
            cohort=cohort,
            derangement=artifact,
        )
        assert verdict.passed is expected, true_value
        if not expected:
            assert "ci_excludes_zero" in verdict.reasons


def test_the_gate_refuses_a_derangement_that_does_not_match_the_cohort(records):
    cohort = _cohort()
    names = list(cohort.names)
    artifact = _derangement(cohort)
    broken = (
        (dataclasses.replace(artifact, permutation={**artifact.permutation, names[0]: names[0]}), "fingerprint"),
        (dataclasses.replace(artifact, permutation={n: artifact.permutation[n] for n in names[:63]}), "fingerprint"),
        (dataclasses.replace(artifact, cohort="test64"), "cohort"),
        (dataclasses.replace(artifact, seed=1), "seed"),
    )
    for artifact_variant, message in broken:
        with pytest.raises(ValueError, match=message):
            gates.action_use_gate(
                true_table=_table(names, 0.36, derangement=artifact),
                wrong_table=_wrong(names, 0.30, artifact),
                cohort=cohort,
                derangement=artifact_variant,
            )


def test_true_and_wrong_are_scored_on_the_SAME_pinned_noise(records):
    """The draw is keyed on the EVALUATED example, never on the donor: keying it on the donor would
    change the noise between the two conditions and the gate would measure noise, not actions."""
    cohort = _cohort()
    artifact = _derangement(cohort)
    plan = gates.action_use_plan(cohort, derangement=artifact)
    assert [entry["name"] for entry in plan.entries] == list(cohort.names)
    for entry in plan.entries:
        assert set(entry["conditions"]) == set(gates.SCORED_CONDITIONS)
        assert entry["conditions"]["true"]["actions_from"] == entry["name"]
        assert entry["conditions"]["wrong"]["actions_from"] == artifact.permutation[entry["name"]]
        assert entry["conditions"]["zero"]["actions_from"] is None
        draws = {condition["draw_key_name"] for condition in entry["conditions"].values()}
        assert draws == {entry["name"]}, "every condition reads the evaluated example's own draw"
    assert plan.derangement_sha256 == artifact.fingerprint


def test_the_zero_row_and_the_diagnostics_are_reported_and_never_gate(records):
    cohort = _cohort()
    names = list(cohort.names)
    artifact = _derangement(cohort)
    report = gates.action_use_report(
        cohort,
        derangement=artifact,
        tables=_battery(names, artifact, {"true": 0.36, "wrong": 0.30, "zero": 0.20, "adapter_disabled": 0.10}),
        control_tables=_battery(names, artifact, {"true": 0.33, "wrong": 0.32, "zero": 0.31}, arm="control", run="c0"),
    )
    assert report["gate"].passed
    assert report["reported"]["mean_delta_true_minus_zero"] == pytest.approx(0.16)
    assert report["reported"]["mean_ssim_adapter_disabled"] == pytest.approx(0.10)
    # matched-C0's own action battery: a finding either way, and never a pass condition.
    assert report["reported"]["control_mean_delta_true_minus_wrong"] == pytest.approx(0.01)
    assert report["reported"]["control_mean_delta_true_minus_zero"] == pytest.approx(0.02)
    assert report["reported"]["rollout_uses_actions_more_than_control"] is True
    assert set(report) == {
        "gate",
        "reported",
        "protocol",
        "cohort",
        "manifest_sha256",
        "derangement_sha256",
        "table_sha256",
        "control_table_sha256",
    }
    assert "passed" not in report["reported"], "nothing in the reported block may look like a verdict"


def test_the_control_battery_is_required_not_optional(records):
    """Skipping C0's battery is how "the adapter uses actions" gets published without its comparison."""
    cohort = _cohort()
    names = list(cohort.names)
    artifact = _derangement(cohort)
    tables = _battery(names, artifact, {"true": 0.36, "wrong": 0.30, "zero": 0.20, "adapter_disabled": 0.10})
    with pytest.raises(ValueError, match="control_tables"):
        gates.action_use_report(
            cohort,
            derangement=artifact,
            tables=tables,
            control_tables=_battery(names, artifact, {"true": 0.33, "wrong": 0.32}, arm="control", run="c0"),
        )


# =============================================================================================
# 3. TEST confirmation, structurally after DEV.
# =============================================================================================


def test_the_dev_certificate_is_computed_and_published_and_a_failing_one_locks_TEST(records, fake_gs):
    cohort = _cohort()
    names = list(cohort.names)
    passing = gates.dev_certificate(
        "gs://b/run/pass.json", rollout=_table(names, 0.36), control=_control(names, 0.30), cohort=cohort
    )
    assert passing["certificate"] == gates.GATE_CERTIFICATE and passing["passed"] is True
    assert passing["num_steps"] == 25 and passing["cohort"] == "dev64"
    failing = gates.dev_certificate(
        "gs://b/run/fail.json", rollout=_table(names, 0.30), control=_control(names, 0.30), cohort=cohort
    )
    assert failing["passed"] is False
    artifact = _derangement(cohort)
    with pytest.raises(ValueError, match="did not pass"):
        gates.confirm_on_test(
            "gs://b/run/fail.json",
            test_cohort=gates.load_test_cohort(_TEST),
            derangement=artifact,
            tables={},
            control_tables={},
        )


def test_the_manifest_is_digest_pinned_and_there_is_exactly_one_TEST_door(tmp_path):
    """Structural, not conventional: the only function that reads the TEST manifest for scoring
    demands a DEV certificate first, and the manifest itself is digest-pinned by construction."""
    doctored = tmp_path / "test64.json"
    payload = json.loads(Path(_TEST).read_text())
    payload["rows"] = payload["rows"][:1]
    doctored.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="published TEST-64 manifest"):
        gates.load_test_cohort(str(doctored))

    source = _MODULE_PATH.read_text(encoding="utf-8")
    doors = [
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and "test_cohort" in {arg.arg for arg in node.args.kwonlyargs}
    ]
    assert doors == ["confirm_on_test"], doors
    # ...and the whole evaluator has exactly one place that can turn the TEST manifest into a cohort.
    assert sum(line.count("TestCohort(") for line in source.splitlines()) == 1


def test_the_gate_module_reads_no_config_and_keeps_its_provenance(records, fake_gs):
    cohort = _cohort()
    names = list(cohort.names)
    certificate = gates.dev_certificate(
        "gs://b/run/prov.json", rollout=_table(names, 0.36), control=_control(names, 0.30), cohort=cohort
    )
    assert certificate["manifest_sha256"] == hashlib.sha256(Path(_DEV).read_bytes()).hexdigest()
    assert certificate["margin"] == gates.PRIMARY_MARGIN
    assert certificate["bootstrap"] == {"seed": exp04.BOOTSTRAP_SEED, "resamples": exp04.BOOTSTRAP_RESAMPLES}
    source = _MODULE_PATH.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "getattr":
            assert len(node.args) < 3, f"issue #11: {ast.unparse(node)}"
    assert "generate_wan_null_adapter" not in source


# =============================================================================================
# 4. Self-review strengthenings (these are the attacks on my own draft).
# =============================================================================================


def test_the_action_use_decision_is_stated_here_not_filtered_out_of_the_primary_gate(records):
    """First draft called the +0.05 gate and filtered ``mean_delta`` out of its reasons — the right
    answer for the wrong reason, and it would have silently absorbed any new reason exp_04 added."""
    cohort = _cohort()
    names = list(cohort.names)
    artifact = _derangement(cohort)
    verdict = gates.action_use_gate(
        true_table=_table(names, 0.305, derangement=artifact),
        wrong_table=_wrong(names, 0.30, artifact),
        cohort=cohort,
        derangement=artifact,
    )
    assert verdict.passed and verdict.numbers["margin"] == 0.0
    assert verdict.numbers["mean_delta"] < gates.PRIMARY_MARGIN, "far below the primary margin, and it passes"
    # A coverage failure returns early from exp_04's gate with no "ci" key; the decision must survive it.
    broken = gates.action_use_gate(
        true_table=_table(names[:63], 0.36, derangement=artifact, incomplete=True),
        wrong_table=_wrong(names, 0.30, artifact),
        cohort=cohort,
        derangement=artifact,
    )
    assert not broken.passed and broken.reasons == ("coverage",)


def test_neither_gate_can_have_its_arm_and_control_swapped_by_position():
    """Swapping the claim and its control by position reports the control winning, with a straight face."""
    for function in (gates.primary_gate, gates.action_use_gate):
        kinds = {name: parameter.kind for name, parameter in inspect.signature(function).parameters.items()}
        assert all(kind is inspect.Parameter.KEYWORD_ONLY for kind in kinds.values()), (function.__name__, kinds)
    names = list(_cohort().names)
    with pytest.raises(TypeError):
        gates.primary_gate(_table(names, 0.36), _control(names, 0.30), _cohort())


def test_a_malformed_table_entry_becomes_a_coverage_failure_not_a_silent_drop():
    """A row that is not a record at all becomes MISSING coverage, never an imputed perfect SSIM."""
    cohort = _cohort()
    names = list(cohort.names)
    table = _table(names, 0.36)
    mangled = {**table.payload, "rows": {**table.payload["rows"], names[0]: 0.36}}
    broken = anchor.ScoreTable(payload=mangled, digest="0" * 64)
    verdict = gates.primary_gate(rollout=broken, control=_control(names, 0.30), cohort=cohort)
    assert not verdict.passed and "coverage" in verdict.reasons
    assert names[0] in verdict.numbers["missing_names"]["method"]


# =============================================================================================
# 5. STAMPED != BOUND (review pass 2). The derangement is an ARTIFACT, the tables are produced by
#    one producer that consumes the plan, and every gate accepts only loaded artifacts.
# =============================================================================================

_GEOMETRY = {"z_i0": (48, 1, 12, 20), "z_video": (48, 9, 12, 20), "actions": (32, 7)}


def _install_records(monkeypatch, rows, *, collide=(), fill_of=None):
    """Point the canonical reader's decoder at in-memory records with DISTINCT actions per example.

    The seam is the module, never an argument (review pass 1, BLOCKER 1). ``collide`` makes a pair of
    examples carry byte-identical actions, which is the hazard the derangement exists to refuse.
    """
    from maxdiffusion import null_adapter_manifest_io, run_wan_null_inversion

    twins = dict(collide)
    by_shard: dict[str, list[dict]] = {}
    for row in rows:
        by_shard.setdefault(str(row["shard_path"]), []).append(dict(row))

    def _fill(name):
        source = twins.get(str(name), str(name))
        return float(int(hashlib.sha256(source.encode()).hexdigest()[:6], 16) % 9973) / 10.0 + 1.0

    def reader(shard_path, wanted):
        for row in by_shard[str(shard_path)]:
            if str(row["name"]) not in set(wanted):
                continue
            value = _fill(row["name"]) if fill_of is None else float(fill_of(row))
            yield (
                str(row["name"]),
                int(row["ordinal"]),
                np.full(_GEOMETRY["z_i0"], value, np.float32),
                np.full(_GEOMETRY["z_video"], value, np.float32),
                np.full(_GEOMETRY["actions"], value, np.float32),
            )

    def binder(shard_path):
        row = by_shard[str(shard_path)][0]
        return {"generation": str(row["shard_generation"]), "size": int(row["shard_size"])}

    monkeypatch.setattr(run_wan_null_inversion, "_tfrecord_reader", reader)
    monkeypatch.setattr(null_adapter_manifest_io, "shard_binding", binder)


class _StubBackend:
    """The DEVICE seam and nothing else: it never learns a name, a checkpoint or a horizon.

    ``template``/``bound`` exist because a phase restores through the backend's state template and
    then binds the restored parameters — but neither carries identity, which is the point: a device
    that returned any number it liked still could not misdescribe WHAT was measured.
    """

    def __init__(self, values, template=None, params=None):
        self.values = values
        self.template = template
        self.params = params
        self.calls = []

    def bound(self, params):
        return _StubBackend(self.values, template=self.template, params=params)

    def score(self, *, z_i0, z_video, actions, key, adapter_enabled=True):
        del z_i0
        self.calls.append((float(np.mean(np.asarray(actions, np.float32))), adapter_enabled))
        ssim, mse = self.values(z_video=z_video, actions=actions, adapter_enabled=adapter_enabled, params=self.params)
        execution = anchor.RolloutExecution(
            z_pred=np.zeros_like(np.asarray(z_video)),
            num_steps=anchor.DEPLOYED_SAMPLING_STEPS,
            grid_size=anchor.DEPLOYED_SAMPLING_STEPS + 1,
            guide_scale=5.0,
            draw_key_sha256=anchor.draw_key_digest(key),
        )
        return execution, {"ssim_avg": float(ssim), "latent_mse": float(mse)}


def _identity(run="rb", arm="rollout"):
    return anchor.CheckpointIdentity(
        run_name=run, step=10000, root=f"gs://b/{run}/checkpoints", source="selection", arm=arm, k_b=2, dev_metric=0.3
    )


def _values_for(high, low):
    """SSIM ``high`` when an example is fed its OWN actions, ``low`` otherwise.

    ``_install_records`` fills an example's z_video and its actions with the SAME value, so "these
    actions belong to this example" is a fact the device can see without being told a name — which is
    what makes this stand-in a fair simulation of an action-conditioned model.
    """

    def values(*, z_video, actions, adapter_enabled, params=None):
        del params
        if not adapter_enabled:
            return 0.10, 2.0
        matched = np.isclose(float(np.mean(np.asarray(actions))), float(np.mean(np.asarray(z_video))))
        return (high, 1.0) if matched else (low, 1.5)

    return values


def _produce(monkeypatch, *, high=0.36, low=0.30, arm="rollout"):
    """The whole §3e production path: cohort -> derangement -> plan -> the ONE table producer."""
    _install_records(monkeypatch, instrument.load_dev_cohort(_DEV).rows)
    cohort = instrument.load_dev_cohort(_DEV)
    derangement = gates.cohort_derangement(cohort)
    plan = gates.action_use_plan(cohort, derangement=derangement)
    backend = _StubBackend(_values_for(high, low))
    tables = {
        condition: gates.score_condition_table(
            plan,
            condition=condition,
            cohort=cohort,
            derangement=derangement,
            checkpoint=_identity(run="rb" if arm == "rollout" else "c0", arm=arm),
            arm=arm,
            backend=backend,
        )
        for condition in ("true", "wrong", "zero", "adapter_disabled")
    }
    return cohort, derangement, plan, tables, backend


def test_the_derangement_is_an_ARTIFACT_carrying_cohort_seed_permutation_and_action_digests(monkeypatch):
    """The reviewer: "byte-identical donors passed because nothing ever receives action bytes or
    digests". The artifact reads the cohort's OWN actions and records their digests."""
    _install_records(monkeypatch, instrument.load_dev_cohort(_DEV).rows)
    cohort = instrument.load_dev_cohort(_DEV)
    artifact = gates.cohort_derangement(cohort)
    assert isinstance(artifact, gates.DerangementArtifact)
    assert artifact.cohort == "dev64" and artifact.seed == gates.DERANGEMENT_SEED
    assert set(artifact.permutation) == set(cohort.names)
    assert set(artifact.action_sha256) == set(cohort.names)
    assert len(set(artifact.action_sha256.values())) == len(cohort.names), "distinct actions per example"
    assert artifact.fingerprint == gates.derangement_fingerprint(artifact)
    # No name is its own donor, and no donor's ACTION BYTES equal the receiver's.
    assert all(donor != name for name, donor in artifact.permutation.items())
    assert all(artifact.action_sha256[d] != artifact.action_sha256[n] for n, d in artifact.permutation.items())


def test_a_derangement_from_another_cohorts_seed_or_with_a_fixed_point_is_refused(monkeypatch):
    """A TEST-seeded mapping passed for DEV, and any permutation was accepted as "the derangement"."""
    cohort, derangement, _, tables, _ = _produce(monkeypatch)
    foreign = dataclasses.replace(derangement, cohort="test64")
    with pytest.raises(ValueError, match="cohort"):
        gates.action_use_gate(
            true_table=tables["true"], wrong_table=tables["wrong"], cohort=cohort, derangement=foreign
        )
    first = cohort.names[0]
    sneaky = dataclasses.replace(derangement, permutation={**derangement.permutation, first: first})
    with pytest.raises(ValueError, match="fixed point|fingerprint"):
        gates.action_use_gate(
            true_table=tables["true"], wrong_table=tables["wrong"], cohort=cohort, derangement=sneaky
        )


def test_a_cohort_with_byte_identical_actions_is_repaired_or_fails_closed(monkeypatch):
    rows = instrument.load_dev_cohort(_DEV).rows
    names = [str(row["name"]) for row in rows]
    _install_records(monkeypatch, rows, collide={names[1]: names[0]})
    artifact = gates.cohort_derangement(instrument.load_dev_cohort(_DEV))
    assert artifact.permutation[names[0]] != names[1] and artifact.permutation[names[1]] != names[0]

    _install_records(monkeypatch, rows, fill_of=lambda row: 1.0)  # every example identical
    with pytest.raises(ValueError, match="no legal"):
        gates.cohort_derangement(instrument.load_dev_cohort(_DEV))


def test_ONE_producer_consumes_the_plan_and_emits_receiver_donor_and_draw_key_identities(monkeypatch):
    """The reviewer: "nothing consumes ``action_use_plan``" — the identical-noise contract was a
    stamp with no binding. The producer is now the only way a table exists."""
    cohort, derangement, plan, tables, _ = _produce(monkeypatch)
    for name in cohort.names:
        true_row, wrong_row, zero_row = (tables[c].rows[name] for c in ("true", "wrong", "zero"))
        assert true_row["actions_from"] == name
        assert wrong_row["actions_from"] == derangement.permutation[name]
        assert zero_row["actions_from"] is None
        # ONE draw, keyed on the RECEIVER, in every condition.
        assert true_row["draw_key_sha256"] == wrong_row["draw_key_sha256"] == zero_row["draw_key_sha256"]
        # ...and the bytes actually fed are the donor's, verified against the artifact's digests.
        assert wrong_row["actions_sha256"] == derangement.action_sha256[derangement.permutation[name]]
        assert wrong_row["actions_sha256"] != true_row["actions_sha256"]
    for table in tables.values():
        assert table.num_steps == anchor.DEPLOYED_SAMPLING_STEPS
        assert table.cohort == "dev64" and table.manifest_sha256 == cohort.manifest_sha256
        assert table.checkpoint["run_name"] == "rb" and table.checkpoint["step"] == 10000
    assert tables["wrong"].derangement_sha256 == derangement.fingerprint


def test_a_producer_keying_the_wrong_row_on_the_DONOR_is_caught_by_the_gate(monkeypatch):
    """ "A future scorer can key wrong-action noise on the donor and still pass every scoped test."""
    cohort, derangement, _, tables, _ = _produce(monkeypatch)
    first = cohort.names[0]
    donor_keyed = dict(tables["wrong"].rows)
    donor_keyed[first] = {
        **donor_keyed[first],
        "draw_key_sha256": anchor.draw_key_digest(anchor.evaluation_draw_key(derangement.permutation[first])),
    }
    forged = anchor.build_score_table(
        rows=donor_keyed,
        cohort=cohort,
        condition="wrong",
        arm="rollout",
        checkpoint=_identity(),
        num_steps=anchor.DEPLOYED_SAMPLING_STEPS,
        derangement_sha256=derangement.fingerprint,
    )
    with pytest.raises(ValueError, match="draw"):
        gates.action_use_gate(true_table=tables["true"], wrong_table=forged, cohort=cohort, derangement=derangement)


def test_every_gate_accepts_only_loaded_artifacts_and_never_a_naked_table(monkeypatch):
    cohort, derangement, _, tables, _ = _produce(monkeypatch)
    naked = {name: {"ssim": 0.9, "mse": 0.1} for name in cohort.names}
    with pytest.raises(TypeError, match="ScoreTable"):
        gates.primary_gate(rollout=naked, control=tables["true"], cohort=cohort)
    with pytest.raises(TypeError, match="ScoreTable"):
        gates.action_use_gate(true_table=naked, wrong_table=tables["wrong"], cohort=cohort, derangement=derangement)
    with pytest.raises(TypeError, match="DerangementArtifact"):
        gates.action_use_gate(
            true_table=tables["true"],
            wrong_table=tables["wrong"],
            cohort=cohort,
            derangement=dict(derangement.permutation),
        )


def test_incomplete_coverage_keeps_its_provenance_and_reports_no_deltas(monkeypatch):
    """Reproduced by the reviewer: the early return omitted permutation/hash/cohort/manifest and
    ``action_use_report`` then raised ``KeyError``. Both paths are enriched from ONE helper."""
    cohort, derangement, _, tables, _ = _produce(monkeypatch)
    holed = anchor.build_score_table(
        rows={n: r for n, r in tables["true"].rows.items() if n != cohort.names[0]},
        cohort=cohort,
        condition="true",
        arm="rollout",
        checkpoint=_identity(),
        num_steps=anchor.DEPLOYED_SAMPLING_STEPS,
        allow_incomplete=True,
    )
    verdict = gates.action_use_gate(
        true_table=holed, wrong_table=tables["wrong"], cohort=cohort, derangement=derangement
    )
    assert not verdict.passed and verdict.reasons == ("coverage",)
    for key in ("derangement", "derangement_sha256", "cohort", "manifest_sha256", "margin"):
        assert key in verdict.numbers, key

    report = gates.action_use_report(
        cohort,
        derangement=derangement,
        tables={**tables, "true": holed},
        control_tables={c: tables[c] for c in ("true", "wrong", "zero")},
    )
    assert report["reported"]["coverage_ok"] is False
    assert "mean_delta_true_minus_wrong" not in report["reported"], "no deltas from an invalid table"
    assert "passed" not in report["reported"]


def test_matched_C0_receives_the_FULL_battery_and_both_of_its_deltas_are_reported(monkeypatch):
    cohort, derangement, _, tables, _ = _produce(monkeypatch)
    control = {c: tables[c] for c in ("true", "wrong", "zero")}
    with pytest.raises(ValueError, match="zero"):
        gates.action_use_report(
            cohort,
            derangement=derangement,
            tables=tables,
            control_tables={c: tables[c] for c in ("true", "wrong")},
        )
    report = gates.action_use_report(cohort, derangement=derangement, tables=tables, control_tables=control)
    assert "control_mean_delta_true_minus_wrong" in report["reported"]
    assert "control_mean_delta_true_minus_zero" in report["reported"]
    assert "passed" not in report["reported"]


def test_the_dev_certificate_computes_its_gate_INTERNALLY_and_the_marker_alone_forges_nothing(monkeypatch, fake_gs):
    """``{"certificate": GATE_CERTIFICATE, "passed": True}`` unlocked TEST, and
    ``dev_certificate(GateVerdict(True, (), {}), …)`` issued a pass with ``mean_delta=NaN``."""
    cohort, _, _, tables, _ = _produce(monkeypatch, high=0.40, low=0.30)
    _, _, _, control, _ = _produce(monkeypatch, high=0.30, low=0.30, arm="control")
    assert "verdict" not in inspect.signature(gates.dev_certificate).parameters
    path = "gs://bucket/run-x/dev_certificate.json"
    certificate = gates.dev_certificate(path, rollout=tables["true"], control=control["true"], cohort=cohort)
    assert certificate["passed"] and math.isfinite(certificate["mean_delta"]) and len(certificate["ci"]) == 2
    assert certificate["rollout_table_sha256"] == tables["true"].digest
    assert certificate["control_table_sha256"] == control["true"].digest

    forged = "gs://bucket/run-x/forged.json"
    anchor.publish_certificate(forged, {"certificate": gates.GATE_CERTIFICATE, "passed": True})
    with pytest.raises(ValueError, match="schema|protocol|certificate"):
        gates.load_dev_certificate(forged)


def test_TEST_confirmation_runs_BOTH_gates_behind_one_door(monkeypatch, fake_gs):
    """Plan §3e requires the action-use confirmation on TEST too, with an INDEPENDENTLY derived TEST
    derangement — ``confirm_on_test`` ran only the primary gate."""
    cohort, dev_derangement, _, dev_tables, _ = _produce(monkeypatch, high=0.40, low=0.30)
    _, _, _, dev_control, _ = _produce(monkeypatch, high=0.30, low=0.30, arm="control")
    path = "gs://bucket/run-x/dev_certificate.json"
    gates.dev_certificate(path, rollout=dev_tables["true"], control=dev_control["true"], cohort=cohort)

    test_cohort = gates.load_test_cohort(_TEST)
    _install_records(monkeypatch, test_cohort.rows)
    test_cohort = gates.load_test_cohort(_TEST)
    derangement = gates.cohort_derangement(test_cohort)
    assert derangement.cohort == "test64"
    assert derangement.seed == gates.DERANGEMENT_SEED
    plan = gates.action_use_plan(test_cohort, derangement=derangement)
    backend = _StubBackend(_values_for(0.40, 0.30))

    def _tables(arm, checkpoint):
        return {
            condition: gates.score_condition_table(
                plan,
                condition=condition,
                cohort=test_cohort,
                derangement=derangement,
                checkpoint=checkpoint,
                arm=arm,
                backend=backend,
            )
            for condition in ("true", "wrong", "zero", "adapter_disabled")
        }

    tables = _tables("rollout", _identity())
    control_backend = _StubBackend(_values_for(0.30, 0.28))
    control = {
        condition: gates.score_condition_table(
            plan,
            condition=condition,
            cohort=test_cohort,
            derangement=derangement,
            checkpoint=_identity(run="c0", arm="control"),
            arm="control",
            backend=control_backend,
        )
        for condition in ("true", "wrong", "zero")
    }

    confirmation = gates.confirm_on_test(
        path,
        test_cohort=test_cohort,
        derangement=derangement,
        tables=tables,
        control_tables=control,
    )
    assert set(confirmation) >= {"primary", "action_use", "dev_certificate_sha256", "cohort"}
    assert confirmation["cohort"] == "test64"
    assert confirmation["primary"].numbers["manifest_sha256"] == test_cohort.manifest_sha256
    assert "derangement_sha256" in confirmation["action_use"].numbers

    # The DEV derangement may not be reused for TEST: it is derived independently per cohort.
    with pytest.raises(ValueError, match="cohort"):
        gates.confirm_on_test(
            path, test_cohort=test_cohort, derangement=dev_derangement, tables=tables, control_tables=control
        )


# =============================================================================================
# 6. Holes THIS ROUND'S battery found in this round's own tests. An attacker who can edit a
#    permutation can also recompute its hash — so the fingerprint is a TAMPER check, never the
#    legality check. Each test below kills a mutant that survived the first battery run.
# =============================================================================================


def _resign(artifact, **changes):
    """Edit a derangement AND recompute its fingerprint — what an attacker would actually do."""
    edited = dataclasses.replace(artifact, **changes)
    return dataclasses.replace(edited, fingerprint=gates.derangement_fingerprint(edited))


def test_a_RESIGNED_derangement_with_a_fixed_point_is_still_refused(records):
    """Battery G19: the fixed-point check was unreachable because the fingerprint fired first. A
    consistent forgery reaches it, and it must hold on its own."""
    cohort = _cohort()
    names = list(cohort.names)
    artifact = _derangement(cohort)
    forged = _resign(artifact, permutation={**artifact.permutation, names[0]: names[0]})
    assert gates.derangement_fingerprint(forged) == forged.fingerprint, "the forgery is self-consistent"
    with pytest.raises(ValueError, match="fixed point"):
        gates.action_use_plan(cohort, derangement=forged)
    with pytest.raises(ValueError, match="not a permutation|fixed point"):
        gates.action_use_plan(cohort, derangement=_resign(artifact, permutation=dict.fromkeys(names, names[0])))


def test_a_RESIGNED_derangement_whose_donor_carries_identical_actions_is_still_refused(records):
    """Battery G20: same shape, on the digests. Two examples with byte-identical actions make "wrong
    actions" mean "the same actions" for that pair — the fixed point a name check cannot see."""
    cohort = _cohort()
    names = list(cohort.names)
    artifact = _derangement(cohort)
    victim = names[0]
    twinned = {**artifact.action_sha256, artifact.permutation[victim]: artifact.action_sha256[victim]}
    with pytest.raises(ValueError, match="byte-identical"):
        gates.action_use_plan(cohort, derangement=_resign(artifact, action_sha256=twinned))


def test_the_wrong_row_must_carry_the_DONORS_recorded_bytes(records):
    """Battery G24: nothing in the tests ever produced a wrong row whose action digest disagreed with
    the artifact — so the check that the right bytes were fed was never exercised."""
    cohort = _cohort()
    names = list(cohort.names)
    artifact = _derangement(cohort)
    wrong = _wrong(names, 0.30, artifact)
    rows = dict(wrong.rows)
    rows[names[0]] = {**rows[names[0]], "actions_sha256": "f" * 64}
    tampered = anchor.build_score_table(
        rows=rows,
        cohort=cohort,
        condition="wrong",
        arm="rollout",
        checkpoint=_checkpoint(),
        num_steps=25,
        derangement_sha256=artifact.fingerprint,
    )
    with pytest.raises(ValueError, match="not the ones the derangement recorded"):
        gates.action_use_gate(
            true_table=_table(names, 0.36, derangement=artifact),
            wrong_table=tampered,
            cohort=cohort,
            derangement=artifact,
        )
    # ...and a wrong row that names a donor other than the recorded one is refused by name, too.
    renamed = dict(wrong.rows)
    renamed[names[0]] = {**renamed[names[0]], "actions_from": names[1]}
    with pytest.raises(ValueError, match="not its donor's"):
        gates.action_use_gate(
            true_table=_table(names, 0.36, derangement=artifact),
            wrong_table=anchor.build_score_table(
                rows=renamed,
                cohort=cohort,
                condition="wrong",
                arm="rollout",
                checkpoint=_checkpoint(),
                num_steps=25,
                derangement_sha256=artifact.fingerprint,
            ),
            cohort=cohort,
            derangement=artifact,
        )


def test_the_shape_adapter_itself_refuses_a_naked_table():
    """Battery G25: ``_agree`` happened to catch a naked table first, so ``as_gate_table``'s own
    refusal was never exercised — and it is the public one a future caller would reach."""
    with pytest.raises(TypeError, match="built ScoreTable"):
        gates.as_gate_table({"a": {"ssim": 0.9, "mse": 1.0}})


def test_a_certificate_whose_pass_flag_disagrees_with_its_numbers_is_refused(records, fake_gs):
    """Battery G30: no test ever published a certificate that lied about its own decision — which is
    precisely the forgery a digest cannot catch, because the forger re-digests."""
    cohort = _cohort()
    names = list(cohort.names)
    path = "gs://b/run/decided.json"
    certificate = gates.dev_certificate(
        path, rollout=_table(names, 0.30), control=_control(names, 0.30), cohort=cohort
    )
    assert certificate["passed"] is False
    forged = {**{k: v for k, v in certificate.items() if k != "sha256"}, "passed": True, "reasons": []}
    payload = json.dumps(
        {"payload": forged, "sha256": hashlib.sha256(json.dumps(forged, sort_keys=True).encode()).hexdigest()}
    )
    fake_gs.blobs["gs://b/run/forged.json"] = payload.encode()
    with pytest.raises(ValueError, match="its own numbers decide"):
        gates.load_dev_certificate("gs://b/run/forged.json")
    # A non-finite mean is refused too: that was the shape of the reviewer's NaN certificate.
    nan_payload = {**forged, "mean_delta": float("nan"), "ci": [float("nan"), float("nan")]}
    body = json.dumps(
        {
            "payload": nan_payload,
            "sha256": hashlib.sha256(json.dumps(nan_payload, sort_keys=True).encode()).hexdigest(),
        }
    )
    fake_gs.blobs["gs://b/run/nan.json"] = body.encode()
    with pytest.raises(ValueError, match="must be finite"):
        gates.load_dev_certificate("gs://b/run/nan.json")
