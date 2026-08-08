"""exp_06 `rollout_adapter` — T3b-3 `dev-instrument`: the DEV-64 selection estimand (plan §3d).

The contract: **deterministic, manifest-bound, and structurally unable to see TEST.**

The headline is the third clause. The S7-era failure was a config pointing at the whole validation
directory, so "we don't pass TEST" is exactly the assurance that already failed once. What is pinned
here is stronger: there is no API that turns a bare example name into a draw. Draws come from a
:class:`DevCohort`; a ``DevCohort`` comes only from :func:`load_dev_cohort`; and that refuses any
manifest whose cohort is not ``dev64``. A caller holding the TEST manifest cannot build the object
the instrument needs, and the tests below try every route in and are refused on each.

The other two clauses are what make a selection number mean anything: the same checkpoint must score
identically on every process and rerun (so two checkpoints differ only by their parameters), and
every emitted score must carry the digest of the cohort file it was measured on (so it can never be
quoted against a different cohort). Refusals are message-matched — the U08 lesson from T3b-1: a
guard whose message is untested is a guard that can be deleted without any test noticing.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion import pos_rollout_dev_instrument as instrument
from maxdiffusion import pos_rollout_support as support

_MODULE_PATH = Path(instrument.__file__).resolve()
_MANIFEST_DIR = _MODULE_PATH.parents[2] / "docs" / "worklogs_yixun" / "exp_04_null_adapter_claude" / "j0_manifests"
_DEV = str(_MANIFEST_DIR / "dev64.json")
_TEST = str(_MANIFEST_DIR / "test64.json")
_STEPS, _K, _SHAPE = 25, 2, (4, 3, 4, 6)


def _cohort():
    return instrument.load_dev_cohort(_DEV)


def _function_node(source: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _fresh_module():
    """Re-execute the module into a brand-new namespace -- a process restart, minus the process.

    Registered in ``sys.modules`` before execution because ``dataclasses`` resolves a field's
    annotation by looking its defining module up there; without it, class creation raises. (T1's
    version of this helper predates the module having a dataclass.)
    """
    name = "_restarted_pos_rollout_dev_instrument"
    spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _draw(cohort, name, module=instrument, replicate=0):
    del module  # the draw is a method on the issued capability now, not a module function
    return cohort.draw(name, num_steps=_STEPS, k_b=_K, example_shape=_SHAPE, replicate=replicate)


# =============================================================================================
# 1. TEST-64 is structurally unreachable -- the round's headline.
# =============================================================================================


def test_the_test_cohort_manifest_cannot_be_loaded_for_selection():
    # Pass the TEST file's own digest so the COHORT guard is what fires, not the digest guard --
    # both refusals matter and each is exercised on its own.
    digest = hashlib.sha256(Path(_TEST).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="TEST is confirmation only"):
        instrument.load_dev_cohort(_TEST, expected_sha256=digest)


@pytest.mark.parametrize("cohort_file", ["train2000.json", "trainfit16.json"])
def test_no_other_cohort_can_be_loaded_either(cohort_file):
    # Not a TEST blocklist -- an allowlist of one. A cohort added tomorrow is refused by default.
    path = str(_MANIFEST_DIR / cohort_file)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="scores 'dev64' only"):
        instrument.load_dev_cohort(path, expected_sha256=digest)


def test_a_test_example_name_cannot_be_drawn_for_even_with_a_valid_dev_cohort():
    cohort = _cohort()
    test_names = [str(row["name"]) for row in json.loads(Path(_TEST).read_text())["rows"]]
    assert test_names, "the TEST manifest is empty, so this proves nothing"
    for name in test_names[:3]:
        assert name not in cohort.names
        with pytest.raises(ValueError, match="is not in the dev64 cohort"):
            _draw(cohort, name)


def test_a_dev_labelled_cohort_wrapped_around_a_real_test_name_cannot_be_built():
    """The reviewer's exact attack (review BLOCKER 1), and it now has no entry point.

    The old guard checked that the caller had passed the string ``"dev64"`` — a claim about a LABEL.
    Wrapping that label around a genuine TEST name produced both a StepDraws and a bare key. The
    cohort is now a capability the loader ISSUES, so the forgery is unconstructible rather than
    validated: guarding a claim is not the same as making the wrong thing impossible.
    """
    test_name = str(json.loads(Path(_TEST).read_text())["rows"][0]["name"])
    assert test_name == "ep61399_v0_s00000", "the attack must use a genuine TEST name"
    for token in (object(), None, instrument.DevCohort):
        with pytest.raises(TypeError, match="issued by load_dev_cohort"):
            instrument.DevCohort(
                token,
                cohort="dev64",
                rows=[{"name": test_name, "split": "dev64"}],
                manifest_sha256="0" * 64,
                manifest_path=_DEV,
            )
    # ...and there is no bare-name key entry point left to reach around it.
    assert not hasattr(instrument, "dev_draw_key")
    assert not hasattr(instrument, "dev_example_draw")


def test_no_entry_point_accepts_a_bare_name_list():
    """The structural half: every draw route demands a DevCohort, so TEST has no way in."""
    for name in ("score_dev_cohort", "instrument_provenance"):
        parameters = list(inspect.signature(getattr(instrument, name)).parameters)
        assert parameters[0] == "cohort", f"{name} does not take a DevCohort first"
    # Batches are LOADED THROUGH the cohort, never supplied as a caller mapping (review BLOCKER 2):
    # TEST tensors filed under DEV keys used to be scored and stamped with DEV provenance.
    assert "batch_loader" in inspect.signature(instrument.score_dev_cohort).parameters
    assert "batches" not in inspect.signature(instrument.score_dev_cohort).parameters
    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "test64" in source, "the forbidden cohort must be named, so the refusal is greppable"
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = {argument.arg for argument in node.args.args + node.args.kwonlyargs}
            assert not names & {"eval_data_dir", "val_dir", "split_dir", "directory"}, node.name


# =============================================================================================
# 2. Manifest binding, fail-closed.
# =============================================================================================


def test_the_cohort_records_the_digest_of_the_file_it_read():
    cohort = _cohort()
    assert cohort.cohort == "dev64" and len(cohort.names) == 64
    assert cohort.manifest_sha256 == hashlib.sha256(Path(_DEV).read_bytes()).hexdigest()


def test_the_digest_is_required_and_defaults_to_the_approved_manifest():
    # It was optional, so any schema-valid file labelled dev64 was accepted (review BLOCKER 2). The
    # default IS the approved manifest's digest, so binding is the easy path and opting out is not a
    # path at all.
    assert instrument.J0_DEV64_SHA256 == hashlib.sha256(Path(_DEV).read_bytes()).hexdigest()
    assert inspect.signature(instrument.load_dev_cohort).parameters["expected_sha256"].default == (
        instrument.J0_DEV64_SHA256
    )
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        instrument.load_dev_cohort(_DEV, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        instrument.load_dev_cohort(_TEST)  # a different file cannot pass the DEV digest


def test_a_cohort_of_the_wrong_size_is_refused():
    with pytest.raises(ValueError, match="must carry exactly 63 examples"):
        instrument.load_dev_cohort(_DEV, expected_size=63)


def test_every_emitted_score_carries_its_cohort_provenance():
    cohort = _cohort()
    provenance = instrument.instrument_provenance(cohort, k_b=_K, eval_index=3, arm="rollout")
    assert provenance["manifest_sha256"] == cohort.manifest_sha256
    assert provenance["cohort"] == "dev64" and provenance["example_count"] == 64
    assert provenance["instrument_purpose"] == instrument.POS_ROLLOUT_DEV_PURPOSE
    assert provenance["k_b"] == _K
    # The training step is recorded as WHEN the score was measured, and is absent from the draw.
    assert provenance["measured_at_step"] == 3
    assert provenance["instrument_seed"] == instrument.INSTRUMENT_SEED


# =============================================================================================
# 3. Fixed draws: deterministic, order-free, process-free, arm-free.
# =============================================================================================


def test_the_draw_is_pinned_per_example_and_evaluation():
    cohort = _cohort()
    name = cohort.names[0]
    first, again = _draw(cohort, name), _draw(cohort, name)
    assert np.array_equal(np.asarray(first.epsilon), np.asarray(again.epsilon))
    assert int(first.support_start) == int(again.support_start) and int(first.t_idx[0]) == int(again.t_idx[0])
    # Different example -> different draw (it is an instrument, not a constant).
    other_example = _draw(cohort, cohort.names[1])
    assert not np.array_equal(np.asarray(first.epsilon), np.asarray(other_example.epsilon))


def test_the_same_checkpoint_scores_identically_whenever_it_is_measured():
    """BLOCKER 3: no training/eval step may reach the draw.

    Folding the evaluation index in meant a checkpoint measured at step 3,000 met different noise
    than one measured at step 4,000 — so the stop rule and the best-checkpoint choice compared
    scores differing by their DRAWS as well as their parameters, i.e. carried evaluation noise into
    a selection decision. The estimand is now a function of (example, replicate) alone.
    """
    cohort = _cohort()
    source = _MODULE_PATH.read_text(encoding="utf-8")
    key_fn = _function_node(source, "_dev_draw_key")
    forbidden = {"eval_index", "global_step", "step", "seed"}
    assert not {a.arg for a in key_fn.args.args + key_fn.args.kwonlyargs} & forbidden, ast.unparse(key_fn.args)
    # ...and the scores agree across two different "when"s, end to end.
    scores = [
        instrument.score_dev_cohort(
            cohort,
            _stub_loss,
            _loader,
            params=jnp.asarray(2.0),
            context=_Context(),
            example_shape=_SHAPE,
            eval_index=when,
        )
        for when in (3000, 4000)
    ]
    assert scores[0]["metric"] == scores[1]["metric"]
    assert scores[0]["per_example"] == scores[1]["per_example"]
    assert scores[0]["measured_at_step"] != scores[1]["measured_at_step"]


def test_the_draw_does_not_depend_on_the_order_examples_are_scored_in():
    # Keyed on the NAME, so batching, sharding and shuffling cannot move it.
    cohort = _cohort()
    forward = {name: _draw(cohort, name) for name in cohort.names}
    for name in list(cohort.names)[::-1][:5]:
        assert np.array_equal(np.asarray(forward[name].epsilon), np.asarray(_draw(cohort, name).epsilon))


@pytest.mark.parametrize("prefix", ["cold", "warmed"])
def test_the_draw_survives_a_process_restart(prefix):
    # The restart technique from T1: module-level state is stable in one process and silently wrong
    # across the restart a real evaluation job performs.
    cohort = _cohort()
    module = _fresh_module()
    fresh_cohort = module.load_dev_cohort(_DEV)
    if prefix == "warmed":
        for name in fresh_cohort.names[:20]:
            _draw(fresh_cohort, name, module=module)
    for name in (cohort.names[0], cohort.names[37]):
        got = _draw(fresh_cohort, name, module=module)
        want = _draw(cohort, name)
        assert np.array_equal(np.asarray(got.epsilon), np.asarray(want.epsilon))
        assert int(got.support_start) == int(want.support_start)
        assert int(got.t_idx[0]) == int(want.t_idx[0])


def test_the_instrument_uses_its_own_purpose_not_the_training_streams():
    """Selection randomness must not move when the training stream does."""
    assert instrument.POS_ROLLOUT_DEV_PURPOSE == "dev_instrument"
    assert instrument.POS_ROLLOUT_DEV_PURPOSE in support.EXP03_AUX_PURPOSES
    bits = lambda key: np.asarray(jax.random.key_data(key))  # noqa: E731
    dev_key = support.exp03_aux_key(seed=0, global_step=1, purpose=instrument.POS_ROLLOUT_DEV_PURPOSE)
    for training_purpose in ("rollout_epsilon", "index_support_rollout", "one_step_index"):
        other = support.exp03_aux_key(seed=0, global_step=1, purpose=training_purpose)
        assert not np.array_equal(bits(dev_key), bits(other)), training_purpose
    # Structural, on the AST rather than the raw source: the module docstring legitimately NAMES the
    # training purposes in order to say it does not use them (the same false positive T3a hit with
    # `stop_gradient`). What is pinned is that every key derivation asks for the instrument's purpose.
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    derivations = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "exp03_aux_key"
    ]
    assert derivations, "the instrument must derive its key through T1's primitive"
    for call in derivations:
        purposes = [kw.value for kw in call.keywords if kw.arg == "purpose"]
        assert len(purposes) == 1, ast.unparse(call)
        assert isinstance(purposes[0], ast.Name) and purposes[0].id == "POS_ROLLOUT_DEV_PURPOSE", ast.unparse(call)


def test_the_draw_is_arm_agnostic():
    # Both arms are scored by the same instrument on the same draws; only the loss differs.
    assert "arm" not in inspect.signature(instrument.DevCohort.draw).parameters
    assert "arm" not in inspect.signature(instrument._dev_draw_key).parameters
    cohort = _cohort()
    provenance = [instrument.instrument_provenance(cohort, k_b=_K, arm=arm) for arm in ("rollout", "one_step")]
    assert provenance[0]["arm"] != provenance[1]["arm"], "the arm is recorded"
    for key in ("cohort", "manifest_sha256", "instrument_purpose", "instrument_seed", "k_b"):
        assert provenance[0][key] == provenance[1][key], key


def test_the_support_stays_in_the_legal_range_and_never_reaches_the_terminal_sigma():
    # T1's characterized property, inherited: start in {0 .. N-1-k}, end <= N-1.
    cohort = _cohort()
    for name in cohort.names:
        drawn = _draw(cohort, name)
        start, end = int(drawn.support_start), int(drawn.support_end)
        assert end == start + _K
        assert 0 <= start <= _STEPS - 1 - _K
        assert end <= _STEPS - 1
        assert 0 <= int(drawn.t_idx[0]) < _STEPS


# =============================================================================================
# 4. Scoring: the estimand is the cohort, and it moves only with the parameters.
# =============================================================================================


class _Context:
    num_steps, k_b = _STEPS, _K


def _stub_loss(params, batch, context, *, draws):
    del context
    return jnp.sum(params) * jnp.mean(batch["z_video"]) + jnp.mean(draws.epsilon), {}


def _loader(row):
    """A stand-in reader driven BY the cohort's row -- what to read comes from the manifest."""
    return {"z_video": jnp.full((1, *_SHAPE), float(int(row["ordinal"]) % 7 + 1), jnp.float32)}


def test_the_score_is_the_cohort_mean_and_carries_its_provenance():
    cohort = _cohort()
    result = instrument.score_dev_cohort(
        cohort, _stub_loss, _loader, params=jnp.asarray(2.0), context=_Context(), example_shape=_SHAPE, arm="rollout"
    )
    assert set(result["per_example"]) == set(cohort.names)
    expected = sum(result["per_example"].values()) / len(cohort)
    assert np.allclose(result["metric"], expected)
    assert result["manifest_sha256"] == cohort.manifest_sha256 and result["cohort"] == "dev64"


def test_the_score_moves_with_the_parameters_and_with_nothing_else():
    cohort = _cohort()
    score = lambda p: instrument.score_dev_cohort(  # noqa: E731
        cohort, _stub_loss, _loader, params=p, context=_Context(), example_shape=_SHAPE
    )["metric"]
    base = score(jnp.asarray(2.0))
    assert score(jnp.asarray(2.0)) == base, "the same checkpoint must score identically"
    assert score(jnp.asarray(3.0)) != base, "a different checkpoint must score differently"


def test_a_batch_cannot_be_filed_under_the_wrong_name_because_the_cohort_drives_the_read():
    """The structural replacement for the old missing-example check (review BLOCKER 2).

    There is no caller mapping to omit an example from, or to file a TEST tensor into: the loop is
    over the cohort's own rows, and the loader is told which row to read.
    """
    cohort = _cohort()
    seen = []
    instrument.score_dev_cohort(
        cohort,
        _stub_loss,
        lambda row: seen.append(row) or _loader(row),
        params=jnp.asarray(1.0),
        context=_Context(),
        example_shape=_SHAPE,
    )
    assert [str(row["name"]) for row in seen] == list(cohort.names)
    assert all(str(row["split"]) == "dev64" for row in seen), "the loader is driven by DEV rows only"


def test_the_instrument_reads_no_config_and_draws_nothing_at_import_time():
    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert node.value.id not in {"config", "cfg", "scheduler"}, ast.unparse(node)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "getattr":
            assert len(node.args) < 3, f"three-argument getattr is forbidden (issue #11): {ast.unparse(node)}"
    for forbidden in ("print(", "max_logging", "jax.debug"):
        assert forbidden not in source, forbidden
