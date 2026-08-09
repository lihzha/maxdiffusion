"""exp_05 S8 — dispatch and config: a K3 run is launchable from a YAML plus overrides.

Plan §5 item 4's other half. Three things carry the round:

- **The dispatch table is characterized, not just extended.** ``train_wan.train`` is a shared file that
  exp_04 merges across, so the round's protection is a test that pins *every* arm -- the four that
  existed and the new ``POS_CONTEXT_TI2V`` -- by executing the real function against stubbed trainer
  modules. A merge that reroutes ``SIDE_ADAPTER_TI2V`` fails here rather than on a TPU.
- **The training config declares what the trainer reads, at the plan's values.** The keys are found in
  the trainer's source by AST rather than listed by hand, and resolved through the **real**
  ``HyperParameters`` class, because that class is where three-argument ``getattr`` lies.
- **The training surface carries no inversion surface.** K3 consumes K2's cache; it does not invert.
  ``embedding_slot``, the ``null_*`` recipe and the ``pos_*`` artifact roots stay out of this YAML.

Both extractions -- ``train`` here, ``HyperParameters``/pyconfig's override contract -- are the S4 and
S10a technique: importing these modules drags in the full TPU stack, so the code under test is lifted
out of its file and executed, byte-for-byte, against fakes.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import types

import pytest
import yaml

from test_pos_context_launcher import _pyconfig_override_parser

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
TRAIN_WAN = REPO_ROOT / "src" / "maxdiffusion" / "train_wan.py"
TRAINER_SOURCE = REPO_ROOT / "src" / "maxdiffusion" / "trainers" / "wan_pos_context_regression_trainer.py"
PYCONFIG = REPO_ROOT / "src" / "maxdiffusion" / "pyconfig.py"
TRAIN_CONFIG = REPO_ROOT / "src" / "maxdiffusion" / "configs" / "base_wan_5b_pos_context_train.yml"
SIDE_ADAPTER_CONFIG = REPO_ROOT / "src" / "maxdiffusion" / "configs" / "base_wan_5b_side_adapter.yml"
POS_INVERSION_CONFIG = REPO_ROOT / "src" / "maxdiffusion" / "configs" / "base_wan_5b_pos_inversion.yml"

# model_type -> (trainer module, trainer class). The four that existed plus this round's arm.
DISPATCH_TABLE = {
    "I2V": ("maxdiffusion.trainers.wan_i2v_trainer", "WanI2VTrainer"),
    "TI2V": ("maxdiffusion.trainers.wan_ti2v_trainer", "WanTI2VTrainer"),
    "AC_TI2V": ("maxdiffusion.trainers.wan_ctrl_world_trainer", "WanCtrlWorldTrainer"),
    "SIDE_ADAPTER_TI2V": ("maxdiffusion.trainers.wan_ti2v_side_adapter_trainer", "WanTI2VSideAdapterTrainer"),
    "POS_CONTEXT_TI2V": (
        "maxdiffusion.trainers.wan_pos_context_regression_trainer",
        "WanPosContextRegressionTrainer",
    ),
}
FALLBACK = ("maxdiffusion.trainers.wan_trainer", "WanTrainer")

# The inversion surface K3 must not require: it consumes K2's cache, it does not invert.
INVERSION_ONLY = ("embedding_slot", "pos_artifact_dir", "pos_staging_dir", "pos_selection_uri", "pos_adequacy_uri")


# --------------------------------------------------------------------------------------------------
# Extraction helpers -- the repo's technique for code whose module cannot be imported here.
# --------------------------------------------------------------------------------------------------


def _extracted_train():
    """``train_wan.train``, lifted out of its module and compiled on its own."""
    tree = ast.parse(TRAIN_WAN.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAIN_WAN), "exec"), namespace)  # noqa: S102
    return namespace["train"]


def _real_hyperparameters(keys: dict):
    """The deployed ``HyperParameters``, bound to a fake key store (S4's technique)."""
    tree = ast.parse(PYCONFIG.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HyperParameters")
    namespace = {"_config": types.SimpleNamespace(keys=dict(keys))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(PYCONFIG), "exec"), namespace)  # noqa: S102
    return namespace["HyperParameters"]()


class _Recorder:
    """A stand-in trainer module set: every arm's class records construction and start_training."""

    def __init__(self):
        self.built: list[tuple[str, object]] = []
        self.started: list[str] = []

    def install(self, monkeypatch):
        for module_name, class_name in (*DISPATCH_TABLE.values(), FALLBACK):
            monkeypatch.setitem(sys.modules, module_name, self._module(module_name, class_name))

    def _module(self, module_name: str, class_name: str):
        recorder = self
        module = types.ModuleType(module_name)

        class _Trainer:
            def __init__(self, config):
                self.config = config
                recorder.built.append((class_name, config))

            def start_training(self):
                recorder.started.append(class_name)

        _Trainer.__name__ = class_name
        setattr(module, class_name, _Trainer)
        return module


@pytest.fixture
def yaml_config() -> dict:
    return yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------------
# 1. The dispatch table -- every arm, executed.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", sorted(DISPATCH_TABLE))
def test_every_dispatch_arm_builds_its_own_trainer(model_type, monkeypatch):
    """The characterization: the four pre-existing arms are byte-preserved and the new one is added.

    ``train_wan.py`` is shared with exp_04, so a merge that reroutes an arm has to fail here.
    """
    recorder = _Recorder()
    recorder.install(monkeypatch)
    config = types.SimpleNamespace(model_type=model_type)

    _extracted_train()(config)

    assert recorder.built == [(DISPATCH_TABLE[model_type][1], config)]
    assert recorder.started == [DISPATCH_TABLE[model_type][1]]


@pytest.mark.parametrize("model_type", ["T2V", "", "SIDE_ADAPTER", "pos_context_ti2v"])
def test_an_unknown_model_type_still_falls_back_to_the_plain_wan_trainer(model_type, monkeypatch):
    """The fallback is part of the contract, and the new arm is case-sensitive like every other."""
    recorder = _Recorder()
    recorder.install(monkeypatch)

    _extracted_train()(types.SimpleNamespace(model_type=model_type))

    assert recorder.built == [(FALLBACK[1], recorder.built[0][1])]
    assert recorder.started == [FALLBACK[1]]


def test_the_positive_arm_names_this_experiments_trainer():
    """Pinned on the source too: the arm must import exp_05's module, not re-use exp_04's trainer."""
    source = TRAIN_WAN.read_text(encoding="utf-8")

    assert 'config.model_type == "POS_CONTEXT_TI2V"' in source
    assert "from maxdiffusion.trainers.wan_pos_context_regression_trainer import" in source


def test_the_dispatch_edit_is_additive():
    """The S4 discipline on a shared file: exp_05 adds one arm and touches nothing else."""
    source = TRAIN_WAN.read_text(encoding="utf-8")
    tree = ast.parse(source)
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")

    tested = [
        node.test.comparators[0].value
        for node in ast.walk(train)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
    ]
    # A PREFIX, not an exact list. exp_06's T4 appends `POS_ROLLOUT_TI2V` to this same shared file --
    # exactly the dual-touch this test exists to police -- and an exact-equality assertion fires on
    # every future additive arm while catching nothing it is meant to catch. The property that
    # matters is unchanged and still enforced: exp_05's arms are present, in order, with nothing
    # rerouted or reordered ahead of them. (Relaxed by exp_06 T4; exp_05's contract is preserved.)
    assert tested[:5] == ["I2V", "TI2V", "AC_TI2V", "SIDE_ADAPTER_TI2V", "POS_CONTEXT_TI2V"]
    assert tested.index("POS_CONTEXT_TI2V") == 4, "exp_05's arm must keep its position in the chain"
    # One statement per arm plus the call: nothing else crept into the shared entrypoint.
    assert [type(node).__name__ for node in train.body] == ["If", "Expr"]


# --------------------------------------------------------------------------------------------------
# 2. The training config: what the trainer reads, at the plan's values.
# --------------------------------------------------------------------------------------------------


def _config_keys_the_trainer_reads() -> set[str]:
    """Every ``optional_config_value(config, "<key>", ...)`` and ``config.<key>`` in the trainer."""
    tree = ast.parse(TRAINER_SOURCE.read_text(encoding="utf-8"))
    keys = {
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "optional_config_value"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
    }
    keys |= {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "config"
    }
    return keys


def test_the_training_config_declares_every_key_the_trainer_reads(yaml_config):
    """pyconfig rejects an override for a key the YAML does not declare, and ``HyperParameters``
    raises for a read of one -- so the trainer's own source is the checklist."""
    missing = sorted(key for key in _config_keys_the_trainer_reads() if key not in yaml_config)

    assert missing == [], f"the trainer reads keys this config does not declare: {missing}"


def test_the_declared_defaults_are_the_plan_values(yaml_config):
    """Plan §4-P3': logical GBS 256 (F3), DEV every 1k, the fixed 30k budget (F2)."""
    assert yaml_config["model_type"] == "POS_CONTEXT_TI2V"
    assert yaml_config["pos_logical_batch"] == 256
    assert yaml_config["pos_microbatch"] == 256  # == logical: no accumulation until a fit probe asks
    assert yaml_config["eval_every"] == 1000
    assert yaml_config["max_train_steps"] == 30000
    assert yaml_config["seed"] == 0


def test_the_model_surface_is_the_deployed_pre_context_configuration(yaml_config):
    """No representation shim (plan §3 F1): the head this trains is the head that ships."""
    from maxdiffusion.models.wan.pos_context_inversion_wan import POS_L

    assert yaml_config["action_adapter_type"] == "pre_context"
    assert yaml_config["pre_context_tokens"] == POS_L == 8
    assert yaml_config["text_dim"] == 4096
    assert (yaml_config["latent_frames"], yaml_config["latent_height"], yaml_config["latent_width"]) == (9, 12, 20)
    assert yaml_config["action_dim"] == 7 and yaml_config["action_len"] == 32


def test_the_sharding_contract_is_the_side_adapter_trainers(yaml_config):
    """F3: adapter replicated, data batch-sharded, frozen params FSDP-sharded -- i.e. the mesh the
    side-adapter trainer already runs, unchanged."""
    side_adapter = yaml.safe_load(SIDE_ADAPTER_CONFIG.read_text(encoding="utf-8"))

    for key in ("mesh_axes", "logical_axis_rules", "data_sharding", "ici_fsdp_parallelism", "ici_data_parallelism"):
        assert yaml_config[key] == side_adapter[key], key


def test_the_training_config_is_the_side_adapter_config_plus_the_training_keys(yaml_config):
    """Same invariant the inversion config keeps: a copy of the settled training YAML plus this
    round's keys, so no standard key a trainer launch needs can quietly go missing."""
    side_adapter = yaml.safe_load(SIDE_ADAPTER_CONFIG.read_text(encoding="utf-8"))

    assert set(side_adapter) <= set(yaml_config)
    added = sorted(set(yaml_config) - set(side_adapter))
    assert added == sorted(POS_TRAINING_KEYS), added


POS_TRAINING_KEYS = (
    "pos_logical_batch",
    "pos_microbatch",
    "pos_train_cache_dir",
    "pos_dev_cache_dir",
    "pos_variance_table_uri",
)


def test_the_k2_cache_roots_the_run_consumes_are_declared(yaml_config):
    """K3 reads K2's published targets; the roots are config, not launcher folklore."""
    for key in ("pos_train_cache_dir", "pos_dev_cache_dir", "pos_variance_table_uri"):
        assert key in yaml_config and isinstance(yaml_config[key], str)


def test_the_training_config_carries_no_inversion_surface(yaml_config):
    """Guard: training consumes caches, it does not invert. The slot switch and the artifact roots
    belong to the inversion config, and a training YAML that declared them would let a K3 launch
    silently accept inversion overrides that nothing reads."""
    leaked = sorted(key for key in INVERSION_ONLY if key in yaml_config)
    assert leaked == [], f"inversion-only keys in the training config: {leaked}"

    null_keys = sorted(key for key in yaml_config if key.startswith("null_"))
    assert null_keys == [], f"null-slot recipe keys in the training config: {null_keys}"
    # ... and the inversion config is untouched by this round: it still declares them.
    inversion = yaml.safe_load(POS_INVERSION_CONFIG.read_text(encoding="utf-8"))
    assert inversion["embedding_slot"] == "positive" and "null_mode" in inversion


# --------------------------------------------------------------------------------------------------
# 3. Wiring: YAML + overrides -> real config object -> dispatch.
# --------------------------------------------------------------------------------------------------


def _resolved(**overrides):
    """The real pyconfig override contract over the real YAML, then the real config class."""
    argv = ["prog", str(TRAIN_CONFIG), *[f"{key}={value}" for key, value in overrides.items()]]
    return _real_hyperparameters(_pyconfig_override_parser()(argv))


def test_a_launch_from_the_yaml_dispatches_to_the_regression_trainer(monkeypatch):
    recorder = _Recorder()
    recorder.install(monkeypatch)
    config = _resolved()

    _extracted_train()(config)

    assert recorder.built[0][0] == "WanPosContextRegressionTrainer"
    assert recorder.started == ["WanPosContextRegressionTrainer"]


def test_overrides_reach_the_trainer_through_the_real_config_object(monkeypatch):
    """The K3 launch shape: a YAML plus ``key=value`` overrides, typed by pyconfig against the YAML."""
    recorder = _Recorder()
    recorder.install(monkeypatch)
    config = _resolved(max_train_steps=2000, pos_microbatch=64, run_name="k3-fit-probe")

    _extracted_train()(config)

    built = recorder.built[0][1]
    assert built.max_train_steps == 2000 and isinstance(built.max_train_steps, int)
    assert built.pos_microbatch == 64 and built.run_name == "k3-fit-probe"


def test_the_trainer_resolves_its_whole_schedule_from_the_real_config_object():
    """The S7 obligation, closed: every ``optional_config_value`` read lands on a declared key with
    the plan's value, on the class that raises ``ValueError`` for undeclared ones."""
    from maxdiffusion.trainers.wan_pos_context_regression_trainer import TrainingSchedule

    schedule = TrainingSchedule.from_config(_resolved())

    assert schedule.max_train_steps == 30000 and schedule.eval_every == 1000
    assert schedule.logical_batch == 256 and schedule.microbatch == 256 and schedule.accumulation_steps == 1
    assert schedule.seed == 0


def test_an_override_can_ask_for_gradient_accumulation():
    """F3's fallback, reachable from the command line: microbatch 64 x 4 == the logical 256."""
    from maxdiffusion.trainers.wan_pos_context_regression_trainer import TrainingSchedule

    schedule = TrainingSchedule.from_config(_resolved(pos_microbatch=64))

    assert (schedule.microbatch, schedule.accumulation_steps, schedule.logical_batch) == (64, 4, 256)


def test_the_trainer_is_constructible_from_the_config_alone(tmp_path):
    """What ``train_wan`` actually does: ``Trainer(config)``. Nothing else is available at that point,
    so a trainer that needed a model to be constructed could not be dispatched to at all."""
    from maxdiffusion.trainers.wan_pos_context_regression_trainer import WanPosContextRegressionTrainer

    trainer = WanPosContextRegressionTrainer(_resolved(checkpoint_dir=str(tmp_path / "ckpt")))

    assert trainer.schedule.max_train_steps == 30000
    assert trainer.state is None  # no parameters yet: the model is S9's seam
    assert trainer.manager is not None and trainer.selection_manager is not None


def test_a_dispatched_run_says_which_round_wires_the_rest(tmp_path):
    """Honest boundary: the loop is complete, its two external seams are not. A K3 launch today must
    be told that, precisely, rather than crash on an attribute or train on nothing."""
    from maxdiffusion.trainers.wan_pos_context_regression_trainer import WanPosContextRegressionTrainer

    trainer = WanPosContextRegressionTrainer(_resolved(checkpoint_dir=str(tmp_path / "ckpt")))

    with pytest.raises(NotImplementedError, match="S9"):
        trainer.start_training()


def test_a_config_without_a_checkpoint_directory_builds_no_managers():
    """The dirs are optional: a smoke run that never checkpoints must not create a bucket path."""
    from maxdiffusion.trainers.wan_pos_context_regression_trainer import WanPosContextRegressionTrainer

    trainer = WanPosContextRegressionTrainer(_resolved(checkpoint_dir=""))

    assert trainer.manager is None and trainer.selection_manager is None
