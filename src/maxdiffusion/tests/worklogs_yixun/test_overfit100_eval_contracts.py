"""CPU-only tests for the cycle-D STRENGTHENING contracts (Codex review D1-D6).

The review's common theme: **the verdict must be derivable only from authenticated,
role-validated, fixed-cohort inputs -- nothing pass-derived, nothing operator-trusted.** This
file pins the machinery that makes that true:

  (D1) MANIFEST-DERIVED COHORTS. The canonical denominator is the 100 canonical windows of the
       authenticated manifest, derived independently of what any pass happened to select. Every
       artifact carries the full cohort plus explicit covered/missing sets, and the verdict CLI
       REFUSES an S3 verdict whose denominator is anything else. Flags are validated against
       that fixed cohort (scrutiny ruling 8's SPLIT), not against the current pass.
  (D2) PASS ROLES. Every artifact declares one of ``s2_gate`` / ``s3_intermediate`` /
       ``s3_segment_final`` / ``s3_full_set`` and is validated against that role's D11 contract
       (exact seeds, exact modes, exact cohort scope, 25 sampling steps, and a COMPLETE row grid).
       ``C3_100`` is only the role-validated ``s3_segment_final`` artifacts; the aggregator also
       refuses mixed run_name / manifest hash / dataset / commit and non-25-step artifacts.
  (D3) FULL-SET TIER. ``full_set_windows`` = all 1,629 manifest-derived keys, and the claim is
       evaluable ONLY from a role-validated ``s3_full_set`` artifact with COMPLETE seed-0
       correct-mode coverage at ``c*``. The CLI derives and feeds it; no operator lists.
  (D4) IMMUTABLE, ROLE-KEYED ARTIFACTS. Output paths carry the role, and a writer refuses to
       replace an existing artifact unless the bytes are identical.
  (D5) AUX COVERAGE. Summary-level requested/ok/failed/coverage_fraction + reason counts, with a
       loud WARNING when coverage is incomplete and an ERROR line when it is zero.
  (D6) WINDOW-NAME PARITY with the builder, including a six-digit start, and a parser that
       accepts the builder's "at least five digits" format while rejecting non-canonical padding.

Stdlib + numpy + the committed manifest; no weights, no GCS, no mesh.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import maxdiffusion.generate_wan_side_adapter as gen
import maxdiffusion.overfit100_success_statistic as stat
from maxdiffusion.data_preprocessing.build_overfit100_dataset import window_name as builder_window_name

_REPO = Path(gen.__file__).parents[2]
_MANIFEST = _REPO / "docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json"
_MANIFEST_DICT = json.loads(_MANIFEST.read_text())


# ======================================================================================
# (D6) Window-name parity with the builder + parser round-trip.
# ======================================================================================


def test_window_name_matches_the_builder_across_the_committed_cohort():
    for entry in _MANIFEST_DICT["episodes"]:
        episode_id = int(entry["episode_id"])
        for start in (0, 4, 4 * (int(entry["n_windows"]) - 1)):
            assert gen.overfit100_window_name(episode_id, start) == builder_window_name(episode_id, start)


def test_window_name_matches_the_builder_for_a_six_digit_start():
    # The builder zero-pads to AT LEAST five digits, so a >= 100000 start yields six.
    for start in (99996, 100000, 123456):
        name = builder_window_name(25189, start)
        assert gen.overfit100_window_name(25189, start) == name
        assert len(name.rsplit("_s", 1)[1]) == (5 if start < 100000 else 6)
        assert gen.parse_overfit100_window_name(name) == (25189, 0, start)


def test_parser_round_trips_every_committed_canonical_name():
    for entry in _MANIFEST_DICT["episodes"]:
        start = stat.canonical_window_start(int(entry["n_windows"]))
        name = builder_window_name(int(entry["episode_id"]), start)
        assert gen.parse_overfit100_window_name(name) == (int(entry["episode_id"]), 0, start)


def test_parser_rejects_non_canonical_zero_padding():
    # "s000004" is not the name the builder writes for start 4; accepting it would let two
    # spellings of one window into a cohort.
    with pytest.raises(ValueError) as ei:
        gen.parse_overfit100_window_name("ep100_v0_s000004")
    assert "ep100_v0_s000004" in str(ei.value)


# ======================================================================================
# (D1) Manifest-derived cohorts, and gen/statistic parity.
# ======================================================================================


def test_canonical_cohort_from_the_committed_manifest_is_100_keys():
    cohort = stat.canonical_cohort_from_manifest(_MANIFEST_DICT)
    assert len(cohort) == 100 == _MANIFEST_DICT["totals"]["episodes"]
    assert len(set(cohort)) == 100
    # Ordered by episode_index, keys are (episode_id, canonical_start).
    first = _MANIFEST_DICT["episodes"][0]
    assert cohort[0] == (int(first["episode_id"]), 4 * ((int(first["n_windows"]) - 1) // 2))


def test_all_window_keys_from_the_committed_manifest_is_1629_keys():
    keys = stat.all_window_keys_from_manifest(_MANIFEST_DICT)
    assert len(keys) == 1629 == _MANIFEST_DICT["totals"]["windows"]
    assert len(set(keys)) == 1629
    # Every canonical key is one of the built windows.
    assert set(stat.canonical_cohort_from_manifest(_MANIFEST_DICT)) <= set(keys)


def test_cohorts_can_be_restricted_to_a_sets_episodes():
    indices = list(range(10))  # train10
    cohort = stat.canonical_cohort_from_manifest(_MANIFEST_DICT, episode_indices=indices)
    keys = stat.all_window_keys_from_manifest(_MANIFEST_DICT, episode_indices=indices)
    assert len(cohort) == 10
    expected_windows = sum(int(e["n_windows"]) for e in _MANIFEST_DICT["episodes"][:10])
    assert len(keys) == expected_windows
    with pytest.raises(ValueError) as ei:
        stat.canonical_cohort_from_manifest(_MANIFEST_DICT, episode_indices=[0, 999])
    assert "999" in str(ei.value)


def test_gen_and_statistic_derive_the_same_cohorts_from_the_manifest():
    # gen reads the manifest file (through the trainer's strict reader); the statistic derives
    # from a loaded dict with stdlib only. They must agree exactly.
    assert gen.manifest_canonical_cohort(str(_MANIFEST)) == stat.canonical_cohort_from_manifest(_MANIFEST_DICT)
    assert gen.manifest_all_window_keys(str(_MANIFEST)) == stat.all_window_keys_from_manifest(_MANIFEST_DICT)


def test_canonical_cohort_is_independent_of_what_a_pass_selected():
    # THE D1 property: selecting ten windows cannot shrink the cohort.
    episodes = gen.manifest_episode_windows(str(_MANIFEST))
    sparse = {index: episodes[index] for index in range(10)}
    selected = gen.select_eval_windows(("canonical", ()), sparse)
    assert len(selected) == 10
    assert len(gen.manifest_canonical_cohort(str(_MANIFEST))) == 100


# ======================================================================================
# (D2) Pass roles.
# ======================================================================================


def test_pass_roles_are_the_four_d11_cells():
    assert stat.PASS_ROLES == ("s2_gate", "s3_intermediate", "s3_segment_final", "s3_full_set")
    assert stat.S3_ROLES == ("s3_intermediate", "s3_segment_final", "s3_full_set")
    assert stat.REQUIRED_SAMPLING_STEPS == 25


def test_parse_eval_pass_role_is_mandatory_and_closed():
    for role in stat.PASS_ROLES:
        assert gen.parse_eval_pass_role(SimpleNamespace(eval_pass_role=f" {role} ")) == role
    for bad in ("", "   ", "gate", "s3", "S3_SEGMENT_FINAL"):
        with pytest.raises(ValueError) as ei:
            gen.parse_eval_pass_role(SimpleNamespace(eval_pass_role=bad))
        assert "eval_pass_role" in str(ei.value)
    with pytest.raises(ValueError):
        gen.parse_eval_pass_role(SimpleNamespace())  # missing key entirely


def _cohort(n, *, offset=0):
    return tuple((100 + offset + i, 4 * i) for i in range(n))


def _plan_kwargs(**overrides):
    cohort = _cohort(3)
    base = {
        "seeds": (0, 1, 2),
        "modes": ("correct", "null", "shuffled"),
        "sampling_steps": 25,
        "covered_canonical": cohort,
        "covered_all": cohort,
        "cohort": cohort,
        "all_window_keys": cohort + ((100, 8), (101, 8)),
    }
    base.update(overrides)
    return base


def test_segment_final_role_requires_three_seeds_three_modes_full_cohort_25_steps():
    assert stat.pass_role_plan_reasons("s3_segment_final", **_plan_kwargs()) == []
    # each contract violated in turn
    assert stat.pass_role_plan_reasons("s3_segment_final", **_plan_kwargs(seeds=(0, 1))) != []
    assert stat.pass_role_plan_reasons("s3_segment_final", **_plan_kwargs(modes=("correct",))) != []
    assert stat.pass_role_plan_reasons("s3_segment_final", **_plan_kwargs(sampling_steps=20)) != []
    short = _cohort(3)[:2]
    reasons = stat.pass_role_plan_reasons("s3_segment_final", **_plan_kwargs(covered_canonical=short))
    assert reasons and any("cohort" in r or "missing" in r for r in reasons)


def test_intermediate_role_requires_one_seed_correct_only_full_cohort():
    kwargs = _plan_kwargs(seeds=(0,), modes=("correct",))
    assert stat.pass_role_plan_reasons("s3_intermediate", **kwargs) == []
    assert stat.pass_role_plan_reasons("s3_intermediate", **_plan_kwargs(seeds=(0, 1), modes=("correct",))) != []
    assert stat.pass_role_plan_reasons("s3_intermediate", **_plan_kwargs(seeds=(0,), modes=("null",))) != []


def test_full_set_role_requires_seed0_correct_over_every_built_window():
    all_keys = _cohort(3) + ((100, 8), (101, 8))
    kwargs = _plan_kwargs(seeds=(0,), modes=("correct",), covered_all=all_keys, all_window_keys=all_keys)
    assert stat.pass_role_plan_reasons("s3_full_set", **kwargs) == []
    partial = dict(kwargs)
    partial["covered_all"] = all_keys[:-1]
    reasons = stat.pass_role_plan_reasons("s3_full_set", **partial)
    assert reasons and any("built window" in r or "missing" in r for r in reasons)
    assert stat.pass_role_plan_reasons("s3_full_set", **{**kwargs, "seeds": (0, 1)}) != []


def test_s2_gate_role_requires_the_train10_cohort_and_three_seeds():
    cohort = _cohort(10)
    kwargs = _plan_kwargs(
        seeds=(0, 1, 2),
        modes=("correct",),
        covered_canonical=cohort,
        covered_all=cohort,
        cohort=cohort,
        all_window_keys=cohort,
    )
    assert stat.pass_role_plan_reasons("s2_gate", **kwargs) == []
    # null/shuffled are allowed at the gate's final checkpoint, correct is mandatory.
    assert stat.pass_role_plan_reasons("s2_gate", **{**kwargs, "modes": ("correct", "null", "shuffled")}) == []
    assert stat.pass_role_plan_reasons("s2_gate", **{**kwargs, "modes": ("null",)}) != []
    assert stat.pass_role_plan_reasons("s2_gate", **{**kwargs, "seeds": (0,)}) != []


def test_unknown_role_is_refused():
    with pytest.raises(ValueError):
        stat.pass_role_plan_reasons("bogus", **_plan_kwargs())


# ======================================================================================
# (D2) Artifact-level role validation + aggregator consistency.
# ======================================================================================


def _artifact_rows(cohort, *, checkpoint, seeds, modes, ssim=0.97):
    rows = []
    for episode_index, key in enumerate(cohort):
        for seed in seeds:
            for mode in modes:
                rows.append(
                    {
                        "name": gen.overfit100_window_name(key[0], key[1]),
                        "episode_id": key[0],
                        "episode_index": episode_index,
                        "window_start": key[1],
                        "canonical": True,
                        "checkpoint_step": checkpoint,
                        "seed": seed,
                        "context_mode": mode,
                        "ssim": ssim,
                        "latent_mse": 0.01,
                        "pixel_mse": 0.001,
                    }
                )
    return rows


def _artifact(
    *,
    role="s3_segment_final",
    checkpoint=2500,
    cohort=None,
    seeds=(0, 1, 2),
    modes=("correct", "null", "shuffled"),
    covered=None,
    sampling_steps=25,
    rows=None,
    **overrides,
):
    cohort = _cohort(3) if cohort is None else cohort
    covered = cohort if covered is None else covered
    artifact = {
        "schema": stat.AGGREGATION_SCHEMA,
        "eval_pass_role": role,
        "checkpoint_step": checkpoint,
        "run_name": "ovf-s3",
        "commit": "c" * 40,
        "manifest_sha256": "a" * 64,
        "eval_data_dir": "gs://v6_east1d/datasets/exp02_overfit100/train100",
        "sampling_steps": sampling_steps,
        "rollout_seeds": list(seeds),
        "context_modes": list(modes),
        "canonical_cohort": [list(k) for k in cohort],
        "covered_canonical_windows": [list(k) for k in covered],
        "covered_windows": [list(k) for k in covered],
        "missing_canonical_windows": [list(k) for k in cohort if k not in set(covered)],
        "rows": _artifact_rows(covered, checkpoint=checkpoint, seeds=seeds, modes=modes) if rows is None else rows,
    }
    artifact.update(overrides)
    return artifact


def test_validate_artifact_role_accepts_a_complete_segment_final_pass():
    cohort = _cohort(3)
    out = stat.validate_artifact_role(_artifact(), canonical_cohort=cohort, all_window_keys=cohort)
    assert out["ok"] is True and out["role"] == "s3_segment_final" and out["reasons"] == []


def test_validate_artifact_role_refuses_a_mislabeled_pass():
    cohort = _cohort(3)
    # Declares segment-final but rolled out correct mode only (the shipped default).
    bad = _artifact(modes=("correct",))
    out = stat.validate_artifact_role(bad, canonical_cohort=cohort, all_window_keys=cohort)
    assert out["ok"] is False
    assert any("mode" in reason for reason in out["reasons"])


def test_validate_artifact_role_refuses_an_incomplete_row_grid():
    cohort = _cohort(3)
    artifact = _artifact()
    artifact["rows"] = artifact["rows"][:-1]  # one (window, seed, mode) cell missing
    out = stat.validate_artifact_role(artifact, canonical_cohort=cohort, all_window_keys=cohort)
    assert out["ok"] is False
    assert any("row" in reason for reason in out["reasons"])


def test_validate_artifact_role_refuses_a_cohort_that_is_not_the_derived_one():
    artifact = _artifact()
    out = stat.validate_artifact_role(artifact, canonical_cohort=_cohort(4), all_window_keys=_cohort(4))
    assert out["ok"] is False
    assert any("cohort" in reason for reason in out["reasons"])


def test_validate_artifact_role_refuses_a_wrong_schema_or_missing_role():
    cohort = _cohort(3)
    out = stat.validate_artifact_role(
        _artifact(schema="overfit100_eval_aggregation_v1"), canonical_cohort=cohort, all_window_keys=cohort
    )
    assert out["ok"] is False and any("schema" in r for r in out["reasons"])
    artifact = _artifact()
    del artifact["eval_pass_role"]
    out = stat.validate_artifact_role(artifact, canonical_cohort=cohort, all_window_keys=cohort)
    assert out["ok"] is False and any("eval_pass_role" in r for r in out["reasons"])


def test_segment_final_checkpoints_come_only_from_validated_segment_final_artifacts():
    cohort = _cohort(3)
    good = _artifact(checkpoint=2500)
    later = _artifact(checkpoint=5000)
    s2 = _artifact(role="s2_gate", checkpoint=250, seeds=(0, 1, 2), modes=("correct",))
    intermediate = _artifact(role="s3_intermediate", checkpoint=1000, seeds=(0,), modes=("correct",))
    mislabeled = _artifact(checkpoint=7500, modes=("correct",))
    steps = stat.segment_final_checkpoints_from_artifacts(
        [s2, intermediate, later, good, mislabeled], canonical_cohort=cohort, all_window_keys=cohort
    )
    assert steps == [2500, 5000]  # sorted, S2/intermediate/mislabeled excluded


def test_segment_final_selection_refuses_when_nothing_validates():
    cohort = _cohort(3)
    with pytest.raises(ValueError) as ei:
        stat.segment_final_checkpoints_from_artifacts(
            [_artifact(modes=("correct",))], canonical_cohort=cohort, all_window_keys=cohort
        )
    msg = str(ei.value)
    assert "s3_segment_final" in msg and "mode" in msg  # names the role and why it failed


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_name", "other-run"),
        ("manifest_sha256", "b" * 64),
        ("eval_data_dir", "gs://v6_east1d/datasets/exp02_overfit100/train10"),
        ("commit", "d" * 40),
    ],
)
def test_aggregator_refuses_mixed_provenance(field, value):
    with pytest.raises(ValueError) as ei:
        stat.assert_artifacts_consistent([_artifact(checkpoint=2500), _artifact(checkpoint=5000, **{field: value})])
    assert field in str(ei.value)


def test_aggregator_refuses_a_non_25_step_artifact():
    with pytest.raises(ValueError) as ei:
        stat.assert_artifacts_consistent([_artifact(sampling_steps=20)])
    assert "25" in str(ei.value)


def test_aggregator_accepts_consistent_artifacts_and_reports_the_shared_provenance():
    shared = stat.assert_artifacts_consistent([_artifact(checkpoint=2500), _artifact(checkpoint=5000)])
    assert shared["run_name"] == "ovf-s3"
    assert shared["manifest_sha256"] == "a" * 64
    assert shared["sampling_steps"] == 25


# ======================================================================================
# (D1/D3) The verdict CLI: derived cohort, denominator refusal, derived full-set input.
# ======================================================================================


def _tiny_manifest(tmp_path, n_episodes=4, n_windows=3):
    payload = {
        "vae_fingerprint": {"hf_repo": "r", "revision": "a" * 40},
        "totals": {"episodes": n_episodes, "windows": n_episodes * n_windows},
        "episodes": [
            {
                "episode_index": i,
                "episode_id": 100 + i,
                "used_text": f"text {i}",
                "n_windows": n_windows,
            }
            for i in range(n_episodes)
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path, payload


def _write(tmp_path, name, artifact):
    path = tmp_path / name
    path.write_text(json.dumps(artifact, indent=2) + "\n")
    return str(path)


def test_cli_derives_the_cohort_from_the_manifest_and_verifies_its_hash(tmp_path):
    import hashlib

    manifest_path, payload = _tiny_manifest(tmp_path)
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    cohort = stat.canonical_cohort_from_manifest(payload)
    artifact = _artifact(cohort=cohort, covered=cohort, manifest_sha256=sha)
    path = _write(tmp_path, "agg.json", artifact)
    verdict = stat.verdict_from_artifact_files([path], manifest_path=str(manifest_path))
    assert verdict["denominator"] == len(cohort) == 4
    assert verdict["canonical_windows"] == [list(k) for k in cohort]
    assert verdict["segment_final_checkpoints"] == [2500]
    assert verdict["manifest_sha256"] == sha


def test_cli_refuses_an_artifact_whose_manifest_hash_does_not_match(tmp_path):
    manifest_path, payload = _tiny_manifest(tmp_path)
    cohort = stat.canonical_cohort_from_manifest(payload)
    path = _write(tmp_path, "agg.json", _artifact(cohort=cohort, covered=cohort, manifest_sha256="f" * 64))
    with pytest.raises(ValueError) as ei:
        stat.verdict_from_artifact_files([path], manifest_path=str(manifest_path))
    assert "manifest_sha256" in str(ei.value)


def test_cli_refuses_a_shrunken_denominator(tmp_path):
    import hashlib

    manifest_path, payload = _tiny_manifest(tmp_path)
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    cohort = stat.canonical_cohort_from_manifest(payload)
    sparse = cohort[:2]  # a two-window pass claiming to be a segment final
    path = _write(tmp_path, "agg.json", _artifact(cohort=sparse, covered=sparse, manifest_sha256=sha))
    with pytest.raises(ValueError) as ei:
        stat.verdict_from_artifact_files([path], manifest_path=str(manifest_path))
    msg = str(ei.value)
    assert "2" in msg and "4" in msg  # the shrunken and the derived cohort sizes


def test_evaluate_success_require_cohort_is_defence_in_depth():
    cohort = _cohort(4)
    rows = _artifact_rows(cohort, checkpoint=2500, seeds=(0, 1, 2), modes=("correct",))
    stat.evaluate_success(
        rows, canonical_windows=cohort, segment_final_checkpoints=[2500], require_cohort=cohort
    )  # matches -> fine
    with pytest.raises(ValueError) as ei:
        stat.evaluate_success(
            rows[: 3 * 2],
            canonical_windows=cohort[:2],
            segment_final_checkpoints=[2500],
            require_cohort=cohort,
        )
    assert "denominator" in str(ei.value).lower() or "cohort" in str(ei.value).lower()


def test_cli_derives_the_full_set_cohort_and_establishes_the_stronger_tier(tmp_path):
    import hashlib

    manifest_path, payload = _tiny_manifest(tmp_path, n_episodes=4, n_windows=3)
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    cohort = stat.canonical_cohort_from_manifest(payload)
    all_keys = stat.all_window_keys_from_manifest(payload)
    canonical = _artifact(cohort=cohort, covered=cohort, manifest_sha256=sha)
    full_rows = []
    for index, key in enumerate(all_keys):
        full_rows.append(
            {
                "name": gen.overfit100_window_name(key[0], key[1]),
                "episode_id": key[0],
                "episode_index": index,
                "window_start": key[1],
                "canonical": key in set(cohort),
                "checkpoint_step": 2500,
                "seed": 0,
                "context_mode": "correct",
                "ssim": 0.95,
                "latent_mse": 0.01,
                "pixel_mse": 0.001,
            }
        )
    full = _artifact(
        role="s3_full_set",
        cohort=cohort,
        covered=all_keys,
        seeds=(0,),
        modes=("correct",),
        manifest_sha256=sha,
        rows=full_rows,
    )
    paths = [_write(tmp_path, "canonical.json", canonical), _write(tmp_path, "full.json", full)]
    verdict = stat.verdict_from_artifact_files(
        paths, manifest_path=str(manifest_path), out_path=str(tmp_path / "v.json")
    )
    assert verdict["verdict"] == "established"
    assert verdict["full_set_claim"]["evaluable"] is True
    assert verdict["full_set_claim"]["n_windows"] == len(all_keys)
    assert verdict["full_set_claim"]["established"] is True
    assert json.loads((tmp_path / "v.json").read_text()) == verdict


def test_full_set_tier_is_not_evaluable_without_a_role_validated_full_set_pass(tmp_path):
    import hashlib

    manifest_path, payload = _tiny_manifest(tmp_path)
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    cohort = stat.canonical_cohort_from_manifest(payload)
    # Only the canonical pass is supplied: the stronger tier must stay unevaluable, and the
    # canonical claim must still be made.
    path = _write(tmp_path, "agg.json", _artifact(cohort=cohort, covered=cohort, manifest_sha256=sha))
    verdict = stat.verdict_from_artifact_files([path], manifest_path=str(manifest_path))
    assert verdict["verdict"] == "established"
    assert verdict["full_set_claim"]["evaluable"] is False
    assert "s3_full_set" in verdict["full_set_claim"]["reason"]


def test_full_set_tier_refuses_incomplete_seed0_coverage(tmp_path):
    import hashlib

    manifest_path, payload = _tiny_manifest(tmp_path)
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    cohort = stat.canonical_cohort_from_manifest(payload)
    all_keys = stat.all_window_keys_from_manifest(payload)
    canonical = _artifact(cohort=cohort, covered=cohort, manifest_sha256=sha)
    # A full-set artifact missing one window: role validation must reject it, so the tier stays
    # unevaluable rather than scoring a partial cohort.
    partial = all_keys[:-1]
    rows = [
        {
            "name": gen.overfit100_window_name(key[0], key[1]),
            "episode_id": key[0],
            "episode_index": i,
            "window_start": key[1],
            "canonical": key in set(cohort),
            "checkpoint_step": 2500,
            "seed": 0,
            "context_mode": "correct",
            "ssim": 0.99,
            "latent_mse": 0.01,
            "pixel_mse": 0.001,
        }
        for i, key in enumerate(partial)
    ]
    full = _artifact(
        role="s3_full_set",
        cohort=cohort,
        covered=partial,
        seeds=(0,),
        modes=("correct",),
        manifest_sha256=sha,
        rows=rows,
    )
    paths = [_write(tmp_path, "canonical.json", canonical), _write(tmp_path, "full.json", full)]
    verdict = stat.verdict_from_artifact_files(paths, manifest_path=str(manifest_path))
    assert verdict["full_set_claim"]["evaluable"] is False
    assert verdict["full_set_claim"]["established"] is False


def test_cli_usage_requires_the_manifest():
    with pytest.raises(SystemExit):
        stat.main(["prog", "out.json"])  # no manifest, no artifacts


# ======================================================================================
# (D1 / ruling 8) Flags are validated against the FIXED cohort.
# ======================================================================================


def test_flagged_windows_are_validated_against_the_fixed_cohort_not_the_pass():
    cohort = _cohort(4)
    selected = cohort[:2]  # this pass only covers two of them
    # A flag naming a cohort member the pass did not cover is FINE (ruling 8's SPLIT)...
    gen.assert_flagged_windows_in_cohort(("ep102_v0_s00008",), cohort)
    # ...while a flag outside the fixed cohort is still refused.
    with pytest.raises(ValueError) as ei:
        gen.assert_flagged_windows_in_cohort(("ep999_v0_s00000",), cohort)
    assert "ep999_v0_s00000" in str(ei.value)
    assert len(selected) == 2  # (the pass's own coverage is irrelevant to the check)


def test_statistic_still_refuses_a_flag_outside_the_denominator():
    cohort = _cohort(3)
    rows = _artifact_rows(cohort, checkpoint=2500, seeds=(0, 1, 2), modes=("correct",))
    with pytest.raises(ValueError):
        stat.evaluate_success(
            rows, canonical_windows=cohort, segment_final_checkpoints=[2500], flagged_windows=[(9999, 0)]
        )


# ======================================================================================
# (D4) Immutable, role-keyed artifacts.
# ======================================================================================


def test_step_root_includes_the_validated_pass_role():
    root = gen.overfit100_step_root("gs://bucket/run/validation", 2500, "s3_segment_final")
    assert root == "gs://bucket/run/validation/step_002500_s3_segment_final"
    assert gen.overfit100_step_root("x/", 0, "s3_full_set") == "x/step_000000_s3_full_set"
    # Two roles at the same checkpoint therefore cannot collide.
    assert gen.overfit100_step_root("x", 2500, "s3_full_set") != gen.overfit100_step_root(
        "x", 2500, "s3_segment_final"
    )


def test_immutable_json_write_is_a_noop_for_identical_bytes_and_refuses_a_change(tmp_path):
    path = str(tmp_path / "aggregation.json")
    payload = {"a": 1, "b": [1, 2]}
    gen._write_json_immutable(path, payload)
    gen._write_json_immutable(path, payload)  # byte-identical rewrite -> tolerated
    assert json.loads(Path(path).read_text()) == payload
    with pytest.raises(ValueError) as ei:
        gen._write_json_immutable(path, {"a": 2})
    msg = str(ei.value)
    assert "aggregation.json" in msg
    assert json.loads(Path(path).read_text()) == payload  # the original is intact


def test_immutable_text_write_refuses_a_changed_csv(tmp_path):
    path = str(tmp_path / "summary.csv")
    gen._write_text_immutable(path, "a,b\n1,2\n")
    gen._write_text_immutable(path, "a,b\n1,2\n")
    with pytest.raises(ValueError) as ei:
        gen._write_text_immutable(path, "a,b\n9,9\n")
    assert "summary.csv" in str(ei.value)


# ======================================================================================
# (D5) Auxiliary-RGB coverage visibility.
# ======================================================================================


def _aux_row(status, *, ceiling=None):
    return {
        "name": "ep100_v0_s00000",
        "episode_id": 100,
        "episode_index": 0,
        "window_start": 0,
        "canonical": True,
        "checkpoint_step": 2500,
        "seed": 0,
        "context_mode": "correct",
        "ssim": 0.9,
        "latent_mse": 0.01,
        "pixel_mse": 0.001,
        "vae_ceiling_ssim": ceiling,
        "aux_status": status,
    }


def test_aux_coverage_summary_counts_and_reasons():
    rows = [
        _aux_row("ok", ceiling=0.85),
        _aux_row("ok", ceiling=0.87),
        _aux_row("BuildError: pinned download failed"),
        _aux_row("BuildError: pinned download failed"),
        _aux_row("episode 7 not in the manifest"),
    ]
    summary = gen._overfit100_summary(rows)
    coverage = summary["aux_coverage"]
    assert coverage["requested"] == 5 and coverage["ok"] == 2 and coverage["failed"] == 3
    assert coverage["coverage_fraction"] == pytest.approx(0.4)
    assert coverage["failure_reason_counts"]["BuildError: pinned download failed"] == 2
    assert coverage["failure_reason_counts"]["episode 7 not in the manifest"] == 1


def test_aux_coverage_ignores_rows_that_never_requested_it():
    rows = [_aux_row("not_requested"), _aux_row("not_requested")]
    coverage = gen._overfit100_summary(rows)["aux_coverage"]
    assert coverage["requested"] == 0 and coverage["coverage_fraction"] is None


def test_aux_coverage_log_lines_are_loud_when_incomplete_or_zero():
    partial = gen.aux_coverage_log_lines(
        {"requested": 5, "ok": 2, "failed": 3, "coverage_fraction": 0.4, "failure_reason_counts": {"boom": 3}}
    )
    assert partial and any("WARNING" in line for line in partial)
    assert any("boom" in line for line in partial)
    zero = gen.aux_coverage_log_lines(
        {"requested": 5, "ok": 0, "failed": 5, "coverage_fraction": 0.0, "failure_reason_counts": {"boom": 5}}
    )
    assert any("ERROR" in line for line in zero)
    complete = gen.aux_coverage_log_lines(
        {"requested": 5, "ok": 5, "failed": 0, "coverage_fraction": 1.0, "failure_reason_counts": {}}
    )
    assert complete == []
    assert (
        gen.aux_coverage_log_lines(
            {"requested": 0, "ok": 0, "failed": 0, "coverage_fraction": None, "failure_reason_counts": {}}
        )
        == []
    )


# ======================================================================================
# (D1/D2) Gaps found by the mutation spot-checks: the artifact must carry the FULL cohort
# even when a pass covers a subset, and the CLI's C3_100 wiring must exclude other roles.
# ======================================================================================


def test_artifact_records_the_full_cohort_plus_covered_and_missing_when_a_pass_covers_a_subset():
    # D1 verbatim: "store all 100 keys plus separate covered/missing keys in every artifact".
    # Role validation normally forces full coverage, so this pins the ARTIFACT BUILDER itself --
    # the cohort it records must be the supplied (manifest-derived) one, never its own coverage.
    cohort = _cohort(4)
    windows = [
        {
            "name": gen.overfit100_window_name(key[0], key[1]),
            "episode_id": key[0],
            "episode_index": index,
            "window_start": key[1],
            "canonical": True,
            "used_text": f"text {index}",
        }
        for index, key in enumerate(cohort[:2])  # only two of the four cohort windows
    ]
    rows = _artifact_rows(cohort[:2], checkpoint=2500, seeds=(0,), modes=("correct",))
    artifact = gen.overfit100_aggregation_artifact(
        SimpleNamespace(
            run_name="drv",
            model_type="OVERFIT100_TI2V",
            eval_data_dir="gs://x/train100",
            train_data_dir="gs://x/train100",
            model_manifest_path="m.json",
            checkpoint_dir="gs://x/ck",
            eval_windows="canonical",
            context_shuffle_seed=0,
            side_adapter_sampling_steps=25,
            side_adapter_guide_scale=1.0,
            num_text_slots=4,
        ),
        checkpoint_step=2500,
        windows=windows,
        rows=rows,
        seeds=[0],
        modes=["correct"],
        derangement=None,
        flagged_windows=[],
        pass_role="s3_intermediate",
        canonical_cohort=cohort,
        all_window_keys=cohort,
        manifest_sha256="a" * 64,
        role_validation={"role": "s3_intermediate", "ok": True},
    )
    assert artifact["canonical_cohort"] == [list(key) for key in cohort]  # ALL four, not two
    assert artifact["cohort_size"] == 4
    assert artifact["covered_canonical_windows"] == [list(key) for key in cohort[:2]]
    assert artifact["missing_canonical_windows"] == [list(key) for key in cohort[2:]]
    # And a canonical window outside the cohort is a hard error (the two would disagree).
    stray = dict(windows[0], name="ep999_v0_s00000", episode_id=999, window_start=0)
    with pytest.raises(ValueError) as ei:
        gen.overfit100_aggregation_artifact(
            SimpleNamespace(
                run_name="drv",
                model_type="OVERFIT100_TI2V",
                eval_data_dir="gs://x/train100",
                train_data_dir="gs://x/train100",
                model_manifest_path="m.json",
                checkpoint_dir="gs://x/ck",
                eval_windows="canonical",
                context_shuffle_seed=0,
                side_adapter_sampling_steps=25,
                side_adapter_guide_scale=1.0,
                num_text_slots=4,
            ),
            checkpoint_step=2500,
            windows=[stray],
            rows=rows,
            seeds=[0],
            modes=["correct"],
            derangement=None,
            flagged_windows=[],
            pass_role="s3_intermediate",
            canonical_cohort=cohort,
            all_window_keys=cohort,
            manifest_sha256="a" * 64,
            role_validation={"role": "s3_intermediate", "ok": True},
        )
    assert "999" in str(ei.value)


def test_cli_excludes_non_segment_final_checkpoints_from_c3_100(tmp_path):
    # D2 wiring: an intermediate pass at step 1000 is supplied alongside the segment final at
    # 2500. C3_100 must stay [2500]; taking every artifact's checkpoint_step would admit 1000
    # (and score a 1-seed pass with the 3-seed median).
    import hashlib

    manifest_path, payload = _tiny_manifest(tmp_path)
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    cohort = stat.canonical_cohort_from_manifest(payload)
    segment_final = _artifact(cohort=cohort, covered=cohort, manifest_sha256=sha, checkpoint=2500)
    intermediate = _artifact(
        role="s3_intermediate",
        cohort=cohort,
        covered=cohort,
        seeds=(0,),
        modes=("correct",),
        manifest_sha256=sha,
        checkpoint=1000,
    )
    paths = [
        _write(tmp_path, "intermediate.json", intermediate),
        _write(tmp_path, "segment_final.json", segment_final),
    ]
    verdict = stat.verdict_from_artifact_files(paths, manifest_path=str(manifest_path))
    assert verdict["segment_final_checkpoints"] == [2500]
    assert [entry["checkpoint_step"] for entry in verdict["per_checkpoint"]] == [2500]
    # The intermediate pass is still RECORDED (with its role) for provenance.
    assert {entry["role"] for entry in verdict["artifact_roles"]} == {"s3_intermediate", "s3_segment_final"}


def test_cli_feeds_the_derived_cohort_as_require_cohort_and_filters_rows_by_role():
    # Deletion guards for two wirings the behavioural tests cannot isolate: the CLI must pass the
    # DERIVED cohort as require_cohort (defence in depth against a future caller change), and it
    # must build the canonical statistic from segment-final rows ONLY.
    import inspect

    src = inspect.getsource(stat.verdict_from_artifact_files)
    assert "require_cohort=cohort" in src
    assert "roles=(SEGMENT_FINAL_ROLE,)" in src
    assert "segment_final_checkpoints_from_artifacts" in src
    assert "full_set_input_from_artifacts" in src
