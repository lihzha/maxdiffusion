"""exp_06 `rollout_adapter` — T4 `dispatch-config`: a K3 run is launchable from a YAML (plan §5-4).

Three things are pinned here, to exp_05 S8's standard:

1. **The YAML is a SUPERSET of the side-adapter config, structurally.** It is generated from that
   file, so the property is a fact about how it was made rather than an aspiration — and the test
   regenerates it and compares, which is what makes the claim survive a later edit to either file.
2. **The declared defaults ARE the plan values**, each pinned individually: cadence 1,000 updates
   (§3d), GBS 256 and 10,000 steps (Yixun's §11 decisions), k=2 (§3), dropout 0.0 (which T3b-2's
   guard enforces at arm construction). A drifting default is the quietest way to run a different
   experiment than the one that was approved.
3. **The eval source is the DEV MANIFEST, never a directory.** The side-adapter YAML points
   ``eval_data_dir`` at the whole validation directory — precisely the S7-era hazard T3b-3 closed in
   code. Closing it in code and leaving it open in configuration would be closing one door of two, so
   this config empties that key and binds selection to the published manifest by path AND digest.

Dispatch is additive and every prior route is characterized: ``train_wan.py`` is a DUAL-TOUCH file
for future merges, so the diff is one ``elif`` and nothing else, and it is NOT reformatted (upstream
2-space style, S8's ruling).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from maxdiffusion import pos_rollout_arms, pos_rollout_dev_instrument, pos_rollout_loop

_CONFIGS = Path(pos_rollout_loop.__file__).resolve().parent / "configs"
_BASE = _CONFIGS / "base_wan_5b_side_adapter.yml"
_POS = _CONFIGS / "base_wan_5b_pos_rollout.yml"
_TRAIN_WAN = Path(pos_rollout_loop.__file__).resolve().parent / "train_wan.py"

# The exact addition, named so a later key cannot arrive unnoticed (S8's exact-addition standard).
_ADDED_KEYS = {
    "pos_rollout_arm",
    "pos_rollout_k",
    "pos_rollout_support_salt",
    "pos_logical_batch",
    "pos_microbatch",
    "pos_dev_manifest",
    "pos_dev_manifest_sha256",
    # T6: the evaluation launcher's keys. pyconfig coerces only keys the YAML already declares, so a
    # launcher override for an undeclared key is a launch-time failure, not a silent no-op.
    "pos_eval_phase",
    "pos_run_report",
    "pos_test_manifest",
    # T7: M1's fit authorization, which start_training refuses to run without.
    "pos_fit_authorization",
    # F5: where the fit probe LOOKS for cells a prior attempt of this same job already published, so
    # a ladder the zone kills at 90 minutes does not re-measure what it already banked. Empty means
    # adopt nothing, which is the pre-F5 behaviour; the launcher derives it in fit_probe mode. It is
    # a DESTINATION for fingerprint purposes, so the recipe fingerprint is unmoved by its arrival.
    "pos_fit_adoption_root",
    # F6: the cell-exclusion declaration. A cell that faults the chip deterministically (one_step at
    # scan chunk width >= 32 on v6e-8, 2/2 across two VMs) can be DECLARED unreachable so the table
    # publishes and still accounts for it; the reason is required and lands in the table. Both keys are
    # DESTINATION-class for the fingerprint -- they decide which cells the ladder visits, never what a
    # cell compiles to -- so declaring one does not invalidate another cell's banked artifact.
    "pos_fit_excluded_cells",
    "pos_fit_exclusion_reason",
    # Review pass 3 (T6-1/T6-2/T6-3): the derived paths the launchers transport. `pos_resume_parent`
    # is the immutable resume INPUT (distinct from the fresh attempt's OUTPUT tree); `pos_recipe_lock`
    # is the run-level artifact that makes a divergent second arm refuse to start; the three eval
    # inputs are the protocol's dependency edges (anchor -> benchmark -> gates -> confirm).
    "pos_resume_parent",
    "pos_recipe_lock",
    "pos_anchor_certificate",
    "pos_benchmark_row",
    "pos_dev_certificate",
}


#: The FOURTH intended substitution (review F1b, BLOCKER 1): the pilot trains the pre_context
#: adapter, and the generated copy inherited the side-adapter default. Kept verbatim so the config
#: stays a generated copy and the correction is visible in this file rather than only in the YAML.
_ADAPTER_SUBSTITUTION = "# exp_06 trains the UNCHANGED pre_context adapter (~128M) on the frozen 5B -- plan §3, \"the objective\n# is the only manipulated variable\". This value was inherited from the side-adapter YAML this config\n# is generated from, and it decides which architecture `NNXWanSideAdapterStack` builds: `side_adapter`\n# is the ~240M residual-injection variant, a DIFFERENT experiment. M1 consumes it directly, so the\n# probe would have measured the footprint of a model the pilot never trains.\naction_adapter_type: 'pre_context'"


def _load(path):
    return yaml.safe_load(path.read_text())


def test_the_config_is_a_superset_of_the_side_adapter_config():
    base, pos = _load(_BASE), _load(_POS)
    assert len(base) == 186, "the side-adapter baseline moved; regenerate and re-review"
    missing = sorted(set(base) - set(pos))
    assert not missing, f"the pos_rollout config dropped {missing}"
    assert sorted(set(pos) - set(base)) == sorted(_ADDED_KEYS)
    assert len(pos) == len(base) + len(_ADDED_KEYS) == 205


def test_only_the_four_intended_values_differ_from_the_side_adapter_config():
    """Review F1b, BLOCKER 1: `action_adapter_type` is the FOURTH intended substitution.

    The pilot trains the UNCHANGED **pre_context** adapter (~128M) on the frozen 5B. This config is
    generated from the side-adapter YAML and faithfully inherited its `side_adapter` default -- and
    `ProductionModelSource` consumes the value directly, so M1 would have measured the ~240M
    residual-injection architecture instead of the one the experiment trains. The generation rule is
    where a value like this has to be corrected; the drift test is where it has to be pinned.

    The launcher defaulted its storage parent to the rollout root while this config still declared
    the side-adapter root -- real drift, in the exact place the "every default equals the YAML" test
    was supposed to catch, and it survived because that test never looked at `output_dir`. Changing
    the generation rule rather than appending an override keeps the config a generated copy.
    """
    base, pos = _load(_BASE), _load(_POS)
    changed = {key: (base[key], pos[key]) for key in base if base[key] != pos[key]}
    assert set(changed) == {"model_type", "eval_data_dir", "output_dir", "action_adapter_type"}, changed
    assert changed["model_type"] == ("SIDE_ADAPTER_TI2V", "POS_ROLLOUT_TI2V")
    assert changed["output_dir"][1].endswith("wan-ti2v-pos-rollout")
    assert changed["action_adapter_type"] == ("side_adapter", "pre_context")


def test_the_config_is_generated_from_the_side_adapter_config():
    """The superset property as a fact about construction, not an aspiration (S8's technique)."""
    produced = []
    for line in _BASE.read_text().splitlines():
        if line.startswith("model_type:"):
            produced.append("model_type: 'POS_ROLLOUT_TI2V'")
        elif line.startswith("eval_data_dir:"):
            produced.append(
                'eval_data_dir: ""  # exp_06: selection binds to pos_dev_manifest, never a directory (plan §3d)'
            )
        elif line.startswith("output_dir:"):
            produced.append("output_dir: 'gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-pos-rollout'")
        elif line.startswith("action_adapter_type:"):
            produced.append(_ADAPTER_SUBSTITUTION)
        else:
            produced.append(line)
    generated = "\n".join(produced).rstrip("\n") + "\n"
    actual = _POS.read_text()
    assert actual.startswith(generated), "the pos_rollout config is no longer a generated copy plus an appended block"
    appended = actual[len(generated) :]
    assert set(yaml.safe_load(appended) or {}) == _ADDED_KEYS


@pytest.mark.parametrize(
    "key, value, why",
    [
        ("action_adapter_type", "pre_context", "plan §3: the UNCHANGED pre_context adapter (~128M) is the model"),
        ("eval_every", 1000, "plan §3d: evaluation and selection every 1,000 updates, both arms"),
        ("pos_logical_batch", 256, "Yixun §11-3: pilot GBS 256"),
        ("max_train_steps", 10000, "Yixun §11-3: pilot budget 10,000 steps"),
        ("pos_rollout_k", 2, "plan §3: k=2 is the predeclared primary horizon"),
        ("pos_rollout_arm", "rollout", "R-B is the primary arm; matched-C0 is selected by override"),
        ("dropout", 0.0, "T3b-2 enforces this at arm construction; nonzero makes the arms differ"),
        ("side_adapter_sampling_steps", 25, "the production sigma grid the support draws from"),
        ("side_adapter_guide_scale", 5.0, "the deployed CFG scale T3a proves parity against"),
        ("side_adapter_noise_mode", "fresh", "CLAUDE.md: `fixed` creates a train/validation mismatch"),
        ("pos_microbatch", 32, "accumulation preserves the logical batch; 256/32 = 8 microbatches"),
    ],
)
def test_each_pilot_critical_default_is_the_plan_value(key, value, why):
    assert _load(_POS)[key] == value, why


def test_selection_binds_to_the_dev_manifest_and_never_to_a_directory():
    """The S7-era hazard, closed in configuration as well as in code."""
    pos = _load(_POS)
    assert pos["eval_data_dir"] == "", "a directory eval source is what leaked TEST into selection"
    manifest = Path(pos["pos_dev_manifest"])
    assert manifest.name == "dev64.json" and manifest.suffix == ".json", "the eval source must be a manifest FILE"
    assert pos["pos_dev_manifest_sha256"] == pos_rollout_dev_instrument.J0_DEV64_SHA256
    # ...and the digest really is the published manifest's, so the binding is checkable end to end.
    repository = _CONFIGS.parents[2]
    assert (repository / manifest).exists(), f"{manifest} is not in the tree"
    cohort = pos_rollout_dev_instrument.load_dev_cohort(str(repository / manifest))
    assert cohort.cohort == "dev64" and len(cohort) == 64


def test_the_configs_digest_is_a_declaration_and_never_an_argument_to_the_loader():
    """T3b-4 review, BLOCKER A-B1: a caller-settable digest was the forgery that needed no private
    access. The YAML still DECLARES which manifest a run intends; it cannot widen what is accepted."""
    from maxdiffusion.trainers.wan_pos_rollout_trainer import WanPosRolloutTrainer

    trainer = WanPosRolloutTrainer(_config_from_yaml({"pos_dev_manifest_sha256": "0" * 64})[0])
    with pytest.raises(ValueError, match="cannot choose what the instrument will accept"):
        trainer.load_dev_cohort()
    import inspect as _inspect

    from maxdiffusion.trainers import wan_pos_rollout_trainer

    assert "expected_sha256" not in _inspect.getsource(wan_pos_rollout_trainer)
    assert list(_inspect.signature(pos_rollout_dev_instrument.load_dev_cohort).parameters) == ["path"]


def test_no_inversion_or_null_slot_key_leaks_into_the_training_config():
    # Nothing in the training path reads them; carrying them would invite an override that silently
    # does nothing, or worse, that something later starts honouring.
    pos = _load(_POS)
    leaked = [key for key in pos if any(token in key for token in ("null_", "inversion", "invert", "_slot"))]
    assert not leaked, leaked


# =============================================================================================
# Dispatch: additive, and every prior route characterized.
# =============================================================================================


def _dispatch_routes() -> dict[str, str]:
    """The (model_type -> trainer class) table, read off the AST rather than by importing."""
    train = next(
        node
        for node in ast.walk(ast.parse(_TRAIN_WAN.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    routes: dict[str, str] = {}
    node = train.body[0]
    while isinstance(node, ast.If):
        model_type = ast.literal_eval(node.test.comparators[0])
        routes[model_type] = node.body[-1].value.func.id
        node = node.orelse[0] if node.orelse and isinstance(node.orelse[0], ast.If) else None
    return routes


def test_every_pre_existing_dispatch_route_is_byte_preserved():
    """``train_wan.py`` is dual-touch for future merges: the diff must be one ``elif``."""
    routes = _dispatch_routes()
    assert routes == {
        "I2V": "WanI2VTrainer",
        "TI2V": "WanTI2VTrainer",
        "AC_TI2V": "WanCtrlWorldTrainer",
        "SIDE_ADAPTER_TI2V": "WanTI2VSideAdapterTrainer",
        "POS_CONTEXT_TI2V": "WanPosContextRegressionTrainer",
        "POS_ROLLOUT_TI2V": "WanPosRolloutTrainer",
    }
    # The new arm is LAST, so no existing comparison order changed.
    assert list(routes)[-1] == "POS_ROLLOUT_TI2V"


def test_the_dispatch_file_keeps_its_upstream_two_space_style():
    """S8's ruling: do not reformat an upstream file, or a small diff becomes a merge hazard.

    The first version of this test ended its predicate with ``and False``, so it could never fire
    (review, MINOR). Indentation is now asserted on the AST's own column offsets, which is what
    "two-space style" actually means and which no amount of line-prefix cleverness can fake.
    """
    train = next(
        node
        for node in ast.walk(ast.parse(_TRAIN_WAN.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    assert train.col_offset == 0
    assert [statement.col_offset for statement in train.body] == [2] * len(train.body), "the body is at column 2"
    dispatch = [node for node in train.body if isinstance(node, ast.If)]
    assert dispatch, "no dispatch chain found"
    branches = 0
    node = dispatch[0]
    while isinstance(node, ast.If):
        assert [statement.col_offset for statement in node.body] == [4] * len(node.body), ast.unparse(node.test)
        branches += 1
        node = node.orelse[0] if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If) else None
        if node is None:
            break
    assert branches >= 6, f"only {branches} dispatch branches were checked"
    assert "  if config.model_type ==" in _TRAIN_WAN.read_text(), "the two-space dispatch body was reformatted"


def test_the_fallback_route_still_catches_an_unknown_model_type():
    source = _TRAIN_WAN.read_text()
    assert "  else:\n    from maxdiffusion.trainers.wan_trainer import WanTrainer" in source
    assert "  trainer.start_training()" in source


# =============================================================================================
# Config resolution: every read resolves against this YAML, on the REAL HyperParameters class.
# =============================================================================================


def _config_from_yaml(overrides=None):
    """The DEPLOYED ``HyperParameters``, extracted by AST and bound to this YAML (S4/S8's technique).

    Importing ``pyconfig`` drags in ``max_utils`` -> ``transformers``, which a test venv need not
    have; and the point is the ACCESSOR, not the loader. So the real class is exec'd on its own
    against a key store built from this config — including its ``ValueError`` on an undeclared key,
    which is exactly why three-argument ``getattr`` is forbidden (issue #11).
    """
    source = (Path(pos_rollout_loop.__file__).resolve().parent / "pyconfig.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "HyperParameters")
    keys = dict(_load(_POS))
    keys.update(overrides or {})

    class _Store:
        pass

    store = _Store()
    store.keys = keys
    namespace = {"_config": store}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<pyconfig:HyperParameters>", "exec"), namespace)
    return namespace["HyperParameters"](), keys


def test_every_key_the_loop_and_trainer_read_is_declared_by_this_yaml():
    """The config-drift gate, on the real accessor: an undeclared read raises, never defaults."""
    from maxdiffusion.trainers.wan_pos_rollout_trainer import WanPosRolloutTrainer

    config, _ = _config_from_yaml()
    schedule = pos_rollout_loop.LoopSchedule.from_config(config)
    assert schedule.eval_every == 1000 and schedule.logical_batch == 256
    assert schedule.max_train_steps == 10000 and schedule.k_b == 2 and schedule.arm == "rollout"
    assert schedule.num_steps == 25 and schedule.microbatch == 32

    trainer = WanPosRolloutTrainer(config)
    assert trainer.dropout_rate == pos_rollout_arms.PRODUCTION_DROPOUT_RATE
    assert trainer.dev_manifest.endswith("dev64.json")
    assert trainer.dev_manifest_sha256 == pos_rollout_dev_instrument.J0_DEV64_SHA256
    # The real accessor refuses an undeclared key rather than returning a default (issue #11).
    with pytest.raises(ValueError, match="not in config"):
        _ = config.a_key_this_yaml_does_not_declare


def test_no_three_argument_getattr_anywhere_in_the_exp06_modules():
    """Issue #11: `HyperParameters.__getattr__` raises, so the three-arg form never falls back."""
    package = Path(pos_rollout_loop.__file__).resolve().parent
    targets = sorted(package.glob("pos_rollout_*.py")) + [package / "trainers" / "wan_pos_rollout_trainer.py"]
    assert len(targets) >= 6
    for path in targets:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "getattr" and len(node.args) >= 3:
                raise AssertionError(f"{path.name}: three-argument getattr is forbidden: {ast.unparse(node)}")


def test_the_trainer_is_constructible_from_the_config_alone_and_refuses_a_bad_arm():
    from maxdiffusion.trainers.wan_pos_rollout_trainer import WanPosRolloutTrainer

    with pytest.raises(ValueError, match="unknown arm"):
        WanPosRolloutTrainer(_config_from_yaml({"pos_rollout_arm": "corrective_ss"})[0])
    with pytest.raises(ValueError, match="dropout must be 0.0"):
        WanPosRolloutTrainer(_config_from_yaml({"dropout": 0.1})[0])


def test_the_trainer_validates_its_dev_binding_before_anything_expensive():
    """The DEV binding is still decided from configuration alone, and still before the pipeline load.

    W3 moved M1's authorization gate to be LITERALLY first (final review, LOW 5), so
    ``start_training`` on this YAML now stops at the missing authorization rather than at the DEV
    manifest. The binding's own refusal is therefore asserted on the unit, and its POSITION — before
    anything expensive — on the order of ``start_training``'s statements.
    """
    import inspect as _inspect

    from maxdiffusion.trainers.wan_pos_rollout_trainer import WanPosRolloutTrainer

    trainer = WanPosRolloutTrainer(_config_from_yaml({"pos_dev_manifest": ""})[0])
    with pytest.raises(ValueError, match="never a directory"):
        trainer.load_dev_cohort()

    import textwrap as _textwrap

    statements = [
        ast.unparse(statement)
        for statement in ast.parse(_textwrap.dedent(_inspect.getsource(WanPosRolloutTrainer.start_training)))
        .body[0]
        .body
    ]
    position = {
        name: next(index for index, text in enumerate(statements) if name in text)
        for name in ("authorized_context", "load_dev_cohort", "load_backbone")
    }
    assert position["authorized_context"] == 0, "M1's gate is the first executable statement"
    assert position["authorized_context"] < position["load_dev_cohort"] < position["load_backbone"]
