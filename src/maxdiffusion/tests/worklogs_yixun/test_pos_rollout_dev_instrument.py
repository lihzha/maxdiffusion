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
_GEOMETRY = {"z_i0": (48, 1, 12, 20), "z_video": (48, 9, 12, 20), "actions": (32, 7)}


def _cohort():
    return instrument.load_dev_cohort(_DEV)


def _forge(tmp_path, rows, *, cohort="dev64", name="forged.json") -> tuple[str, str]:
    """Write a schema-valid manifest of the caller's choosing; return ``(path, its own digest)``."""
    path = tmp_path / name
    path.write_text(json.dumps({"schema_version": 1, "cohort": cohort, "rows": rows}), encoding="utf-8")
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _test_row() -> dict:
    return dict(json.loads(Path(_TEST).read_text())["rows"][0])


def _record_source(rows, *, ordinal_of=None, name_of=None, generation_of=None, fill_of=None):
    """A stand-in for the source shards, yielding exactly what the TFRecord decoder yields.

    The three hooks let a test forge the *decoded* identity of a record -- a wrong ordinal, a wrong
    name, a shard that has been rewritten since J0 bound it -- which is what the canonical reader
    exists to refuse.
    """
    by_shard: dict[str, list[dict]] = {}
    for row in rows:
        by_shard.setdefault(str(row["shard_path"]), []).append(dict(row))
    opened: list[tuple[str, tuple[str, ...]]] = []

    def reader(shard_path, wanted):
        opened.append((str(shard_path), tuple(wanted)))
        for row in by_shard[str(shard_path)]:
            if str(row["name"]) not in set(wanted):
                continue
            fill = float(int(row["ordinal"]) % 7 + 1) if fill_of is None else float(fill_of(row))
            yield (
                str(row["name"]) if name_of is None else name_of(row),
                int(row["ordinal"]) if ordinal_of is None else ordinal_of(row),
                np.full(_GEOMETRY["z_i0"], fill, np.float32),
                np.full(_GEOMETRY["z_video"], fill, np.float32),
                np.full(_GEOMETRY["actions"], fill, np.float32),
            )

    def binder(shard_path):
        row = by_shard[str(shard_path)][0]
        generation = str(row["shard_generation"]) if generation_of is None else generation_of(row)
        return {"generation": generation, "size": int(row["shard_size"])}

    return reader, binder, opened


def _install_decoder(monkeypatch, rows, **hooks):
    """Point the instrument's OWN decoder and shard binding at in-memory records.

    The seam is the module, not an argument. Production has no parameter to substitute (review pass
    1, BLOCKER 1: an injectable `reader` was the previous unrestricted callback one layer lower), so
    a test that needs different bytes patches this process instead of the API.
    """
    from maxdiffusion import null_adapter_manifest_io, run_wan_null_inversion

    reader, binder, opened = _record_source(rows, **hooks)
    monkeypatch.setattr(run_wan_null_inversion, "_tfrecord_reader", reader)
    monkeypatch.setattr(null_adapter_manifest_io, "shard_binding", binder)
    return opened


def _score(cohort, **kwargs):
    values = {"params": jnp.asarray(1.0), "context": _Context(), "example_shape": _SHAPE}
    values.update(kwargs)
    return instrument.score_dev_cohort(cohort, _stub_loss, **values)


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


def test_the_test_cohort_manifest_cannot_be_loaded_for_selection(monkeypatch):
    # In production the digest gate fires first, and there is no argument that can get past it.
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        instrument.load_dev_cohort(_TEST)
    # With the pin itself relaxed -- a test-only lever; no caller has one (review BLOCKER A-B1) --
    # the cohort guard behind it still refuses. Both refusals matter and each is exercised on its own.
    monkeypatch.setattr(instrument, "J0_DEV64_SHA256", hashlib.sha256(Path(_TEST).read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="TEST is confirmation only"):
        instrument.load_dev_cohort(_TEST)


@pytest.mark.parametrize("cohort_file", ["train2000.json", "trainfit16.json"])
def test_no_other_cohort_can_be_loaded_either(cohort_file, monkeypatch):
    # Not a TEST blocklist -- an allowlist of one. A cohort added tomorrow is refused by default.
    path = str(_MANIFEST_DIR / cohort_file)
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        instrument.load_dev_cohort(path)
    monkeypatch.setattr(instrument, "J0_DEV64_SHA256", hashlib.sha256(Path(path).read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="scores 'dev64' only"):
        instrument.load_dev_cohort(path)


def test_a_test_example_name_cannot_be_drawn_for_even_with_a_valid_dev_cohort():
    cohort = _cohort()
    test_names = [str(row["name"]) for row in json.loads(Path(_TEST).read_text())["rows"]]
    assert test_names, "the TEST manifest is empty, so this proves nothing"
    for name in test_names[:3]:
        assert name not in cohort.names
        with pytest.raises(ValueError, match="is not in the dev64 cohort"):
            _draw(cohort, name)


def test_a_forged_dev_manifest_carrying_the_real_test_row_is_refused(tmp_path):
    """**The reviewer's second executed forgery** (review BLOCKER A-B1), which needed NO private
    access: a 64-row DEV-labelled manifest whose first row is the genuine TEST row, loaded by
    supplying its own computed digest through the public ``expected_sha256`` override.

    The override is gone. The digest is the pinned constant, so the only file that loads is the one
    whose bytes hash to it — the forgery's digest has nowhere to go, and passing it is a TypeError
    rather than a knob.
    """
    test_row = _test_row()
    assert test_row["name"] == "ep61399_v0_s00000", "the attack must use a genuine TEST row"
    rows = [{**test_row, "split": "dev64"}] + [{**row, "name": f"{row['name']}_x"} for row in _cohort().rows[1:]]
    assert len(rows) == 64 and rows[0]["name"] == "ep61399_v0_s00000"
    forged, digest = _forge(tmp_path, rows)

    with pytest.raises(ValueError, match="manifest hash mismatch"):
        instrument.load_dev_cohort(forged)
    with pytest.raises(TypeError):
        instrument.load_dev_cohort(forged, expected_sha256=digest)  # the override the attack used
    with pytest.raises(TypeError):
        instrument.DevCohort(forged, expected_sha256=digest)


def test_the_cohort_has_no_issue_secret_to_leak_and_no_constructor_takes_caller_rows():
    """**The reviewer's first executed forgery** (review BLOCKER A-B1): ``_ISSUE_TOKEN`` was reachable
    as ``instrument._ISSUE_TOKEN``, so the capability could simply be handed to itself.

    A secret held in a module attribute is not a capability, it is a password written on the door.
    There is no secret now because there is nothing to guard with one: constructing a ``DevCohort``
    IS loading the approved manifest, so no route accepts caller-supplied rows at all.
    """
    assert not [name for name in dir(instrument) if "TOKEN" in name.upper() or "SECRET" in name.upper()]
    forbidden = {"rows", "manifest_sha256", "expected_sha256", "expected_size", "token"}

    def _callables():
        """Every callable this module defines — module level AND class members, because a second
        constructor hidden on the class is exactly how the retired issue token would come back."""
        for name in dir(instrument):
            attribute = getattr(instrument, name)
            if getattr(attribute, "__module__", None) != instrument.__name__:
                continue
            if inspect.isfunction(attribute):
                yield name, attribute
            elif inspect.isclass(attribute):
                for member, value in vars(attribute).items():
                    if isinstance(value, (classmethod, staticmethod)):
                        value = value.__func__
                    if inspect.isfunction(value):
                        yield f"{name}.{member}", value

    checked = 0
    for name, function in _callables():
        try:
            parameters = set(inspect.signature(function).parameters)
        except (TypeError, ValueError):  # pragma: no cover - builtins carry no signature
            continue
        checked += 1
        assert not parameters & forbidden, f"{name}{sorted(parameters)}"
    assert checked >= 8, "the scan must actually reach this module's callables"
    # ...and there is no bare-name key entry point left to reach around it either.
    assert not hasattr(instrument, "dev_draw_key")
    assert not hasattr(instrument, "dev_example_draw")


def test_constructing_a_cohort_IS_loading_the_approved_manifest():
    """The structural claim, stated as a signature: a cohort takes a PATH and nothing else.

    The class is public and that is fine — public construction is safe exactly because construction
    performs the verification. (What no in-language guard can stop is ``object.__new__`` plus slot
    assignment; an attacker with that reach can equally rewrite the pinned digest, so it is a bound
    on Python, not a residual of this design.)
    """
    assert list(inspect.signature(instrument.DevCohort.__init__).parameters) == ["self", "path"]
    assert list(inspect.signature(instrument.load_dev_cohort).parameters) == ["path"]
    assert "expected_sha256" not in _MODULE_PATH.read_text(encoding="utf-8")
    assert instrument.load_dev_cohort(_DEV).manifest_sha256 == instrument.DevCohort(_DEV).manifest_sha256


def test_no_entry_point_accepts_a_bare_name_list():
    """The structural half: every draw route demands a DevCohort, so TEST has no way in."""
    for name in ("score_dev_cohort", "instrument_provenance"):
        parameters = list(inspect.signature(getattr(instrument, name)).parameters)
        assert parameters[0] == "cohort", f"{name} does not take a DevCohort first"
    # Batches are read by the INSTRUMENT from the row's own shard, never supplied by the caller as a
    # mapping (BLOCKER 2) and never produced by a caller's callback (BLOCKER A-B2).
    scoring = inspect.signature(instrument.score_dev_cohort).parameters
    assert not {"batches", "batch_loader", "loader", "batch_reader", "reader"} & set(scoring)
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


def test_the_digest_is_the_pinned_constant_and_nothing_else():
    # It was optional (BLOCKER 2), then defaulted-but-overridable (BLOCKER A-B1). It is now the
    # constant: binding is not the easy path, it is the only path.
    assert instrument.J0_DEV64_SHA256 == hashlib.sha256(Path(_DEV).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        instrument.load_dev_cohort(_TEST)  # a different file cannot pass the DEV digest


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda rows: rows[:63], "must carry exactly 64 examples"),
        (lambda rows: rows + [dict(rows[0])], "must carry exactly 64 examples"),
        (lambda rows: [dict(rows[0]), dict(rows[0])] + rows[2:], "duplicate example names"),
        (lambda rows: [{**rows[0], "split": "test64"}] + rows[1:], "declares split 'test64'"),
    ],
)
def test_the_cohort_shape_is_validated_behind_the_digest(tmp_path, monkeypatch, mutate, message):
    """Defence in depth, reachable only with the pin relaxed — which no caller can do.

    ``expected_size`` was a caller argument too; a knob that can only ever weaken a constant is not a
    feature, so the size is pinned like the digest and these refusals are exercised directly.
    """
    forged, digest = _forge(tmp_path, mutate([dict(row) for row in _cohort().rows]))
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        instrument.load_dev_cohort(forged)
    monkeypatch.setattr(instrument, "J0_DEV64_SHA256", digest)
    with pytest.raises(ValueError, match=message):
        instrument.load_dev_cohort(forged)


def test_the_dev_draw_key_signature_is_pinned_exactly():
    """The reviewer's correction (review, B4): a surviving mutant is not the structural proof — the
    step-free call graph and the SIGNATURES are, so the signature is what the test pins.

    A step can only reach the key by being added to this signature, and this assertion is what a
    widened signature has to get past.
    """
    signature = inspect.signature(instrument._dev_draw_key)
    assert list(signature.parameters) == ["name", "field", "replicate"]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in signature.parameters.values()
    ), "keyword-only: a positional call site could quietly acquire a fourth argument"
    assert signature.parameters["replicate"].default is inspect.Parameter.empty
    # ...and the only caller is the membership-checked draw, whose own signature carries no step.
    draw = inspect.signature(instrument.DevCohort.draw)
    assert list(draw.parameters) == ["self", "name", "num_steps", "k_b", "example_shape", "replicate", "dtype"]
    source = _MODULE_PATH.read_text(encoding="utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_dev_draw_key"
    ]
    assert calls, "the key derivation must be reached through the private helper"
    for call in calls:
        assert not call.args, ast.unparse(call)
        assert {keyword.arg for keyword in call.keywords} == {"name", "field", "replicate"}, ast.unparse(call)


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


def test_the_same_checkpoint_scores_identically_whenever_it_is_measured(monkeypatch):
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
    _install_decoder(monkeypatch, cohort.rows)
    scores = [_score(cohort, params=jnp.asarray(2.0), eval_index=when) for when in (3000, 4000)]
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


def test_the_score_is_the_cohort_mean_and_carries_its_provenance(monkeypatch):
    cohort = _cohort()
    _install_decoder(monkeypatch, cohort.rows)
    result = _score(cohort, params=jnp.asarray(2.0), arm="rollout")
    assert set(result["per_example"]) == set(cohort.names)
    expected = sum(result["per_example"].values()) / len(cohort)
    assert np.allclose(result["metric"], expected)
    assert result["manifest_sha256"] == cohort.manifest_sha256 and result["cohort"] == "dev64"


def test_the_score_moves_with_the_parameters_and_with_nothing_else(monkeypatch):
    cohort = _cohort()
    _install_decoder(monkeypatch, cohort.rows)
    score = lambda p: _score(cohort, params=p)["metric"]  # noqa: E731
    base = score(jnp.asarray(2.0))
    assert score(jnp.asarray(2.0)) == base, "the same checkpoint must score identically"
    assert score(jnp.asarray(3.0)) != base, "a different checkpoint must score differently"


# =============================================================================================
# 5. The canonical row reader: the instrument opens the row's own shard (review BLOCKER A-B2).
# =============================================================================================


def test_no_batch_SOURCE_of_any_kind_is_a_parameter():
    """Three designs, three layers, one hole — and this is where it stops.

    ``batch_loader(row)`` let ``lambda row: test_batch`` score TEST content as DEV (BLOCKER A-B2).
    Replacing it with an instrument-owned reader that still took ``reader``/``binder`` moved the same
    attack one layer down: the reviewer's decoder echoed every genuine DEV name and declared ordinal
    while returning tensors filled with 999, and its binder echoed the manifest's generation and
    size, producing ``metric 999.0`` stamped ``cohort dev64`` with the genuine digest over 64
    examples (review pass 1, BLOCKER 1). Nothing about a batch is a parameter now.
    """
    scoring = inspect.signature(instrument.score_dev_cohort).parameters
    assert list(scoring)[:2] == ["cohort", "loss_fn"]
    assert not {"batch_reader", "batch_loader", "batches", "loader", "reader", "binder"} & set(scoring)
    assert list(inspect.signature(instrument.DevBatchReader.__init__).parameters) == ["self", "cohort"]
    # ...and scoring builds the reader itself rather than accepting one.
    node = _function_node(_MODULE_PATH.read_text(encoding="utf-8"), "score_dev_cohort")
    built = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "DevBatchReader"
    ]
    assert len(built) == 1 and [ast.unparse(argument) for argument in built[0].args] == ["cohort"]


def test_the_reviewers_echoed_identity_decoder_is_no_longer_expressible(monkeypatch):
    """The executed attack, now a test. The forged decoder is installed the ONLY way it still can be
    — by patching this process — and what it proves is the residual's true shape: a caller of the
    scoring API has no way to reach it at all, because there is no argument to reach it through.
    """
    cohort = _cohort()
    with pytest.raises(TypeError):
        instrument.DevBatchReader(cohort, reader=lambda p, w: iter(()))
    with pytest.raises(TypeError):
        instrument.score_dev_cohort(cohort, _stub_loss, object(), params=1.0, context=_Context(), example_shape=_SHAPE)

    # Installed in-process, the forgery still scores -- and that is exactly why it may not be an
    # argument: patching a module is a property of a test run, not something a launch can express.
    _install_decoder(monkeypatch, cohort.rows, fill_of=lambda row: 999.0)
    poisoned = _score(cohort)
    assert poisoned["cohort"] == "dev64" and poisoned["metric"] != 0.0


def test_a_subclass_cannot_substitute_its_own_reading(monkeypatch):
    """The type check alone would move the forgery one class down; the read goes through the CLASS."""
    _install_decoder(monkeypatch, _cohort().rows)
    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "DevBatchReader.read(batch_reader, name)" in source, "the read must not dispatch on the instance"
    assert len(_score(_cohort())["per_example"]) == 64


def test_the_reader_opens_the_rows_own_shard_and_takes_what_the_manifest_bound(monkeypatch):
    cohort = _cohort()
    opened = _install_decoder(monkeypatch, cohort.rows)
    _score(cohort)
    assert opened, "nothing was read"
    wanted = [name for _, names in opened for name in names]
    assert wanted == list(cohort.names), "the reads are driven by the cohort's rows, in its order"
    for shard_path, names in opened:
        assert shard_path == cohort.row(names[0])["shard_path"]


@pytest.mark.parametrize(
    "hooks, message",
    [
        ({"ordinal_of": lambda row: int(row["ordinal"]) + 1}, "but the manifest binds it to ordinal"),
        ({"name_of": lambda row: f"{row['name']}_other"}, "did not yield"),
        ({"generation_of": lambda row: "0"}, "is not the object the manifest bound"),
    ],
)
def test_a_record_whose_decoded_identity_disagrees_with_its_row_is_refused(monkeypatch, hooks, message):
    """exp_04's standard: re-check the shard's generation and size, and refuse a record whose decoded
    ordinal disagrees with its row. These fire on the instrument's own decoder, which is the only one."""
    cohort = _cohort()
    _install_decoder(monkeypatch, cohort.rows, **hooks)
    with pytest.raises((ValueError, RuntimeError), match=message):
        _score(cohort)


def test_the_reader_uses_the_instruments_own_decoder_and_exp04s_shard_binding():
    from maxdiffusion import run_wan_null_inversion

    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "run_wan_null_inversion._tfrecord_reader" in source
    # On the AST, not the text: the docstring legitimately NAMES the seam it no longer has (the same
    # false positive T3a hit with `stop_gradient`). The only reader/binder keyword anywhere in this
    # module must be the module-resolved decoder, and no binder is passed at all -- so exp_04's
    # shard_binding default applies and the generation/size re-check is always on.
    seams = [
        (keyword.arg, ast.unparse(keyword.value))
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg in {"reader", "binder"}
    ]
    assert seams == [("reader", "run_wan_null_inversion._tfrecord_reader")], seams
    assert "shard_binding" in inspect.getsource(run_wan_null_inversion.build_read_batch)
    assert inspect.signature(run_wan_null_inversion.build_read_batch).parameters["binder"].default is None


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
