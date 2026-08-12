"""exp_06 T6: the two launch scripts, pinned by EXECUTING them under bash.

The contract: **a K3/M-job launch is reproducible from the shell, and the two arms of the pilot
differ by the objective and nothing else.**

S10a set the technique and this reuses it: a substring assertion cannot tell
``pos_rollout_arm="${POS_ROLLOUT_ARM}"`` from a default that silently drifted, cannot prove that a
key is ABSENT from the command line, and cannot prove that two arms' command lines differ in exactly
one word. So each launcher is copied into a sandbox whose ``PATH`` holds shims — a ``python`` that
records its argv, a stub HF prefetch — and the real script runs end to end under the real
``/bin/bash``. **The recorded argv IS the assertion surface.**

**Review pass 3 rejected the first version of this file on three counts, and the layers below are
the answer to each:**

1. **The comparison test proved nothing (T6-1).** It showed that two IDENTICAL supplied environments
   expand identically — while the two arms shared a checkpoint root, so running R-B and then
   matched-C0 could make C0 restore R-B's parameters, optimizer, step and history. Section 2 now
   pins that the arms are emitted from ONE common recipe array, that every destination carries the
   arm, and — because no shell can police two submissions a day apart — that the run-level RECIPE
   LOCK refuses a second arm whose recipe differs anywhere else.
2. **The "exhaustive" default pin was not exhaustive and already missed real drift (T6 MAJOR).**
   ``OUTPUT_DIR`` defaulted to the rollout root while the checked-in YAML declared the side-adapter
   root, and the test never looked. Section 1 carries an INTERFACE TABLE covering every emitted
   override, with the only permitted exemptions — run identity and derived storage — named in the
   table with their reason, and a test that no emitted key is missing from it.
3. **The sandbox was faithful for argv, not for "working launcher" (T6 MAJOR).** The ``python`` shim
   never parsed the preflight heredoc and returned success although ``src/`` and the config were
   absent. Section 5 adds a REAL-PYTHON integration layer: the sandbox carries the actual package,
   the heredocs are executed by a real interpreter against controlled module stubs, and the emitted
   overrides are parsed against the real YAML with pyconfig's own coercion rules.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
TRAIN_LAUNCHER = REPO_ROOT / "bash_scripts" / "train_wan_pos_rollout.sh"
EVAL_LAUNCHER = REPO_ROOT / "bash_scripts" / "eval_wan_pos_rollout.sh"
NULL_LAUNCHER = REPO_ROOT / "bash_scripts" / "run_wan_null_inversion.sh"
POS_CONFIG = REPO_ROOT / "src" / "maxdiffusion" / "configs" / "base_wan_5b_pos_rollout.yml"

TRAIN_ENTRY = "src/maxdiffusion/train_wan.py"
PROBE_ENTRY = "src/maxdiffusion/pos_rollout_fit_probe.py"
EVAL_ENTRY = "src/maxdiffusion/eval_wan_pos_rollout.py"
CONFIG_ARG = "src/maxdiffusion/configs/base_wan_5b_pos_rollout.yml"

UNIT = "\x1f"  # no path, flag or override contains it
COMMIT = "a1b2c3d4" * 5
EXTERNALS = ("bash", "date", "mkdir", "tee", "grep", "cat", "git", "sed", "rm", "chmod", "printf", "env")

#: A valid M1 authorization is a *prerequisite* of train mode, refused before the prefetch. The argv
#: layer's ``python`` shim never reads it, so any non-empty path exercises the mapping; section 5
#: builds a real one.
FAKE_AUTHORIZATION = "gs://bucket/m1/authorization.json"

#: The M2 topology (v6e-8). Every sandbox declares a topology because W2b made it required; the
#: number itself is only load-bearing in section 6, which parametrizes over both real topologies.
DEFAULT_DEVICE_COUNT = 8

_PY_SHIM = """#!/bin/sh
{ printf 'PYTHON'; for arg in "$@"; do printf '\\037%s' "$arg"; done; printf '\\n'; } >> "$SHIM_RECORD"
if [ "$1" = "-" ]; then
  if [ -n "${SHIM_REAL_PYTHON:-}" ]; then
    count=0
    [ -f "$SHIM_RECORD.n" ] && count=$(cat "$SHIM_RECORD.n")
    count=$((count + 1))
    echo "$count" > "$SHIM_RECORD.n"
    if [ "$count" -le "${SHIM_REAL_LIMIT:-99}" ]; then exec "$SHIM_REAL_PYTHON" "$@"; fi
  fi
  exit "${SHIM_PREFLIGHT_EXIT:-0}"
fi
exit "${SHIM_ENTRYPOINT_EXIT:-0}"
"""

_PREFETCH_STUB = """#!/bin/sh
{ printf 'PREFETCH'; for arg in "$@"; do printf '\\037%s' "$arg"; done; printf '\\n'; } >> "$SHIM_RECORD"
echo "[prefetch stub] $1"
exit "${SHIM_PREFETCH_EXIT:-0}"
"""

# Controlled module stubs for the real-Python layer: the launcher's preflight imports these purely to
# prove they are installed, and the storage layer reaches gfile for local paths.
_TENSORFLOW_STUB = '''"""A controlled stand-in for tensorflow: gfile over the local filesystem."""
from . import io  # noqa: F401

__version__ = "stub"
'''
_TENSORFLOW_IO_STUB = """import pathlib
import shutil


class _GFile:
    def __init__(self, path, mode):
        self._handle = open(path, mode)

    def read(self):
        return self._handle.read()

    def write(self, payload):
        return self._handle.write(payload)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._handle.close()
        return False


class _Gfile:
    GFile = _GFile

    @staticmethod
    def exists(path):
        return pathlib.Path(path).exists()

    @staticmethod
    def makedirs(path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def listdir(path):
        root = pathlib.Path(path)
        return sorted(child.name for child in root.iterdir()) if root.is_dir() else []

    @staticmethod
    def rmtree(path):
        shutil.rmtree(path, ignore_errors=True)


gfile = _Gfile()
"""
_SKIMAGE_STUB = '__version__ = "stub"\n'


def _sandbox(tmp_path: Path, launcher: Path) -> tuple[Path, dict]:
    """A curated-PATH sandbox carrying the REAL package tree.

    Curated rather than inherited, so a missing binary is a property of this test's PATH and not of
    whichever machine runs the suite -- but ``src/`` is the real one, symlinked, because the pass-3
    review found the old sandbox "returns success even though src/ and the config are absent". Both
    launchers now refuse a working tree without their entrypoint and config, and that refusal has to
    be exercised against the real files.
    """
    root = tmp_path / "sandbox"
    (root / "bash_scripts").mkdir(parents=True)
    (root / "bin").mkdir()
    (root / "home").mkdir()
    shutil.copy2(launcher, root / "bash_scripts" / launcher.name)
    os.symlink(REPO_ROOT / "src", root / "src")
    for name, body in (("python", _PY_SHIM), ("prefetch_hf_snapshot.sh", _PREFETCH_STUB)):
        target = (root / "bin" / name) if name == "python" else (root / "bash_scripts" / name)
        target.write_text(body)
        target.chmod(target.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    for tool in EXTERNALS:
        found = shutil.which(tool)
        if found:
            os.symlink(found, root / "bin" / tool)
    env = {
        "PATH": str(root / "bin"),
        "HOME": str(root / "home"),
        "SHIM_RECORD": str(root / "record.txt"),
        "COMMIT": COMMIT,
        # W2b: the topology is a REQUIRED declaration of the training launcher (it is what the
        # per-device batch is derived FROM), so every sandbox declares one and the tests that are
        # about the declaration itself override or remove it.
        "POS_DEVICE_COUNT": str(DEFAULT_DEVICE_COUNT),
    }
    return root, env


def _stub_modules(root: Path) -> str:
    """tensorflow / skimage stand-ins, ahead of ``src`` on the path the heredocs import through."""
    stubs = root / "stubs"
    (stubs / "tensorflow").mkdir(parents=True, exist_ok=True)
    (stubs / "tensorflow" / "__init__.py").write_text(_TENSORFLOW_STUB)
    (stubs / "tensorflow" / "io.py").write_text(_TENSORFLOW_IO_STUB)
    (stubs / "skimage.py").write_text(_SKIMAGE_STUB)
    return f"{stubs}:src"


def _run(tmp_path: Path, launcher: Path, **overrides) -> tuple[subprocess.CompletedProcess, list[list[str]]]:
    root, env = _sandbox(tmp_path, launcher)
    if launcher is TRAIN_LAUNCHER and "POS_FIT_AUTHORIZATION" not in overrides:
        env["POS_FIT_AUTHORIZATION"] = FAKE_AUTHORIZATION
    env.update({key: str(value) for key, value in overrides.items()})
    proc = subprocess.run(
        ["/bin/bash", f"bash_scripts/{launcher.name}"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    calls = []
    record = root / "record.txt"
    if record.exists():
        for line in record.read_text().splitlines():
            if line.startswith("PYTHON") or line.startswith("PREFETCH"):
                calls.append(line.split(UNIT))
    return proc, calls


def _entrypoint_argv(calls) -> list[str]:
    """The recorded entrypoint invocation: the python call that is not a heredoc."""
    for call in calls:
        if call[0] == "PYTHON" and len(call) > 1 and call[1] != "-":
            return call[1:]
    raise AssertionError(f"no entrypoint invocation was recorded: {calls}")


def _overrides(argv) -> dict:
    return dict(word.split("=", 1) for word in argv if "=" in word and not word.startswith("--"))


# =============================================================================================
# 1. THE LAUNCHER INTERFACE TABLE — every emitted override, its env variable and its default.
#
# `default` is either "yaml" (the launcher's default must equal the checked-in YAML's value) or an
# EXEMPTION with its written reason. The review permits exemptions only for derived run identity and
# storage, and the test below refuses an exemption that is not one of those two.
# =============================================================================================

_RUN_IDENTITY = "run identity: the launcher names the run, the YAML cannot"
_DERIVED_STORAGE = "derived storage: built from the parent + run + arm + phase + attempt, never an operator input"
_PREREQUISITE = "prior-phase artifact: an operator input in one mode and derived in the other"
#: W2b. The THIRD exemption category, and the only one added since the table was written. The YAML
#: cannot carry a correct `per_device_batch_size`, because the correct value is a function of the
#: topology the job is submitted to (32.0 on v6e-8, 4.0 on v6e-64 at the pilot's GBS 256) and one
#: file cannot be right for both. So the launcher DERIVES it from the declared topology and the
#: logical batch, and the YAML's 1.0 is deliberately not passed through.
_DERIVED_TOPOLOGY = "derived topology arithmetic: pos_logical_batch / POS_DEVICE_COUNT, never an operator input"

TRAIN_INTERFACE = (
    ("RUN_NAME", "run_name", "m3-rb-k2", _RUN_IDENTITY),
    ("MODEL_DIR", "pretrained_model_name_or_path", "Wan-AI/Wan2.2-TI2V-5B-Diffusers-pinned", "yaml"),
    ("TRAIN_DATA_DIR", "train_data_dir", "gs://bucket/datasets/train-x", "yaml"),
    ("OUTPUT_DIR", "output_dir", "gs://bucket/parent-x", "yaml"),
    ("POS_DEV_MANIFEST", "pos_dev_manifest", "docs/other/dev64.json", "yaml"),
    ("POS_DEV_MANIFEST_SHA256", "pos_dev_manifest_sha256", "0" * 64, "yaml"),
    ("POS_ROLLOUT_ARM", "pos_rollout_arm", "one_step", "yaml"),
    ("POS_ROLLOUT_K", "pos_rollout_k", "4", "yaml"),
    ("POS_LOGICAL_BATCH", "pos_logical_batch", "512", "yaml"),
    ("POS_MICROBATCH", "pos_microbatch", "64", "yaml"),
    ("POS_SUPPORT_SALT", "pos_rollout_support_salt", "7", "yaml"),
    ("MAX_TRAIN_STEPS", "max_train_steps", "30000", "yaml"),
    ("EVAL_EVERY", "eval_every", "500", "yaml"),
    ("CHECKPOINT_EVERY", "checkpoint_every", "250", "yaml"),
    ("SAMPLING_STEPS", "side_adapter_sampling_steps", "40", "yaml"),
    ("GUIDE_SCALE", "side_adapter_guide_scale", "7.5", "yaml"),
    ("NOISE_MODE", "side_adapter_noise_mode", "fixed", "yaml"),
    ("DROPOUT", "dropout", "0.1", "yaml"),
    ("LEARNING_RATE", "learning_rate", "1.e-4", "yaml"),
    ("SEED", "seed", "77", "yaml"),
    ("WANDB_PROJECT", "wandb_project", "exp06-rollout", "yaml"),
    ("HARDWARE", "hardware", "gpu", "yaml"),
    ("POS_FIT_AUTHORIZATION", "pos_fit_authorization", "gs://bucket/m1/other.json", _PREREQUISITE),
    ("POS_FIT_ADOPTION_ROOT", "pos_fit_adoption_root", "gs://bucket/m1-prior-attempts", "yaml"),
)
#: Emitted, but with no env variable at all: derived inside the launcher, which is the point.
TRAIN_DERIVED = {
    "checkpoint_dir": _DERIVED_STORAGE,
    "base_output_directory": _DERIVED_STORAGE,
    "pos_resume_parent": _DERIVED_STORAGE,
    "pos_recipe_lock": _DERIVED_STORAGE,
    # W2b: `PER_DEVICE_BATCH_SIZE` was an env variable defaulting to the YAML's 1.0, which on any
    # real topology makes the input pipeline load fewer examples than `pos_logical_batch` -- so an
    # M2/M3 submission as written was refused at startup by the trainer's width check. The knob is
    # gone; the value is computed here.
    "per_device_batch_size": _DERIVED_TOPOLOGY,
}

EVAL_INTERFACE = (
    ("RUN_NAME", "run_name", "m4", _RUN_IDENTITY),
    ("MODEL_DIR", "pretrained_model_name_or_path", "Wan-AI/Wan2.2-TI2V-5B-Diffusers-pinned", "yaml"),
    ("OUTPUT_DIR", "output_dir", "gs://bucket/parent-y", "yaml"),
    ("POS_EVAL_PHASE", "pos_eval_phase", "benchmark", "yaml"),
    ("POS_ROLLOUT_ARM", "pos_rollout_arm", "one_step", "yaml"),
    ("POS_DEV_MANIFEST", "pos_dev_manifest", "docs/other/dev64.json", "yaml"),
    ("POS_TEST_MANIFEST", "pos_test_manifest", "docs/other/test64.json", "yaml"),
    ("SAMPLING_STEPS", "side_adapter_sampling_steps", "40", "yaml"),
    ("GUIDE_SCALE", "side_adapter_guide_scale", "7.5", "yaml"),
    ("POS_ROLLOUT_K", "pos_rollout_k", "4", "yaml"),
    ("SEED", "seed", "77", "yaml"),
    ("PER_DEVICE_BATCH_SIZE", "per_device_batch_size", "2.0", "yaml"),
    ("HARDWARE", "hardware", "gpu", "yaml"),
    ("RUN_REPORT", "pos_run_report", "gs://bucket/run/report.json", _DERIVED_STORAGE),
    ("POS_ANCHOR_CERTIFICATE", "pos_anchor_certificate", "gs://bucket/anchor.json", "yaml"),
    ("POS_BENCHMARK_ROW", "pos_benchmark_row", "gs://bucket/bench.json", "yaml"),
    ("POS_DEV_CERTIFICATE", "pos_dev_certificate", "gs://bucket/dev.json", "yaml"),
)
EVAL_DERIVED = {
    "checkpoint_dir": _DERIVED_STORAGE,
    "base_output_directory": _DERIVED_STORAGE,
    "pos_resume_parent": _DERIVED_STORAGE,
}

_PERMITTED_EXEMPTIONS = {_RUN_IDENTITY, _DERIVED_STORAGE, _PREREQUISITE, _DERIVED_TOPOLOGY}


def _yaml_config():
    return yaml.safe_load(POS_CONFIG.read_text())


def test_the_training_launcher_hands_the_entrypoint_the_exp06_config_as_argv1(tmp_path):
    proc, calls = _run(tmp_path, TRAIN_LAUNCHER)
    assert proc.returncode == 0, proc.stdout[-3000:]
    argv = _entrypoint_argv(calls)
    assert argv[0] == TRAIN_ENTRY
    assert argv[1] == CONFIG_ARG, "argv[1] decides the config; every override is coerced against it"
    assert "base_wan_5b_side_adapter.yml" not in " ".join(argv)
    assert "base_wan_5b_null_inversion.yml" not in " ".join(argv)


@pytest.mark.parametrize(
    "launcher, table, derived",
    [(TRAIN_LAUNCHER, TRAIN_INTERFACE, TRAIN_DERIVED), (EVAL_LAUNCHER, EVAL_INTERFACE, EVAL_DERIVED)],
    ids=["train", "eval"],
)
def test_the_interface_table_covers_every_emitted_override(tmp_path, launcher, table, derived):
    """T6 MAJOR: "the env-mapping table omits several training mappings and nearly all evaluation
    mappings" — so the table is now checked for completeness against what the launcher emits."""
    _, calls = _run(tmp_path / launcher.stem, launcher)
    emitted = set(_overrides(_entrypoint_argv(calls)))
    described = {key for _, key, _, _ in table} | set(derived)
    assert (
        emitted - described == set()
    ), f"{launcher.name} emits keys no interface row describes: {sorted(emitted - described)}"
    assert (
        described - emitted == set()
    ), f"the table describes keys {launcher.name} does not emit: {sorted(described - emitted)}"


@pytest.mark.parametrize(
    "launcher, variable, key, value",
    [(TRAIN_LAUNCHER, v, k, s) for v, k, s, _ in TRAIN_INTERFACE]
    + [(EVAL_LAUNCHER, v, k, s) for v, k, s, _ in EVAL_INTERFACE],
    ids=[f"train-{v}" for v, _, _, _ in TRAIN_INTERFACE] + [f"eval-{v}" for v, _, _, _ in EVAL_INTERFACE],
)
def test_every_env_variable_lands_on_its_own_config_key(tmp_path, launcher, variable, key, value):
    """One variable at a time, each with a value distinguishable from the default and from every
    other key's — so a crossed mapping shows up as the wrong key carrying this value."""
    extra = (
        {
            "POS_EVAL_PHASE": "confirm",
            "POS_ANCHOR_CERTIFICATE": "a",
            "POS_BENCHMARK_ROW": "b",
            "POS_DEV_CERTIFICATE": "c",
        }
        if launcher is EVAL_LAUNCHER
        else {}
    )
    extra.pop(variable, None)
    proc, calls = _run(tmp_path, launcher, **{variable: value}, **extra)
    assert proc.returncode == 0, proc.stdout[-3000:]
    overrides = _overrides(_entrypoint_argv(calls))
    assert overrides[key] == value, f"{variable} -> {key}"
    crossed = [name for name, carried in overrides.items() if carried == value and name != key]
    assert not crossed, f"{variable}'s value also reached {crossed}"


@pytest.mark.parametrize(
    "launcher, variable, key, default",
    [(TRAIN_LAUNCHER, v, k, d) for v, k, _, d in TRAIN_INTERFACE]
    + [(EVAL_LAUNCHER, v, k, d) for v, k, _, d in EVAL_INTERFACE],
    ids=[f"train-{k}" for _, k, _, _ in TRAIN_INTERFACE] + [f"eval-{k}" for _, k, _, _ in EVAL_INTERFACE],
)
def test_every_launcher_default_equals_the_yaml_unless_it_is_a_named_exemption(
    tmp_path, launcher, variable, key, default
):
    """A launcher that recipes differently from the config everyone reads is exactly the drift the
    SOP's parity audit exists for — and it HAD drifted: ``OUTPUT_DIR`` defaulted to the rollout root
    while the YAML still declared the side-adapter root, in a launcher whose test claimed to check
    "every default". The YAML is read here, so an edit to either side fails this."""
    if default != "yaml":
        assert default in _PERMITTED_EXEMPTIONS, f"{key}: {default!r} is not a permitted exemption"
        return
    config = _yaml_config()
    _, calls = _run(tmp_path, launcher)
    overrides = _overrides(_entrypoint_argv(calls))
    assert str(config[key]) == overrides[key], f"{key}: YAML {config[key]!r} vs launcher {overrides[key]!r}"


def test_every_override_the_launchers_pass_is_declared_by_the_yaml(tmp_path):
    """pyconfig coerces only keys the YAML already declares, so an undeclared override is a
    launch-time failure. This is the cheapest possible catch for it."""
    declared = set(_yaml_config())
    for launcher in (TRAIN_LAUNCHER, EVAL_LAUNCHER):
        _, calls = _run(tmp_path / f"decl_{launcher.stem}", launcher)
        undeclared = sorted(set(_overrides(_entrypoint_argv(calls))) - declared)
        assert not undeclared, f"{launcher.name} passes keys the YAML does not declare: {undeclared}"


# =============================================================================================
# 2. THE COMPARABILITY CONTRACT (T6-1): the two arms differ by the objective and its destinations,
#    and a second arm whose recipe differs anywhere else cannot start at all.
# =============================================================================================

_PERMITTED_ARM_DIFFERENCES = {"pos_rollout_arm", "checkpoint_dir", "base_output_directory", "pos_resume_parent"}


def _arm_argv(tmp_path, arm, **extra):
    fixed = {"RUN_NAME": "fixed", "ATTEMPT": "att-FIXED", "POS_ROLLOUT_ARM": arm}
    fixed.update(extra)
    _, calls = _run(tmp_path / arm, TRAIN_LAUNCHER, **fixed)
    return _entrypoint_argv(calls)


def test_the_two_arms_are_emitted_from_one_recipe_and_differ_only_in_the_arm_and_its_destinations(tmp_path):
    """Yixun's decision 1 at the shell. The previous version of this test proved only that two
    IDENTICAL environments expand identically; what it had to prove is that the normalized RECIPE is
    the same object for both arms and that only the arm and the paths derived from it may differ."""
    rollout, control = _arm_argv(tmp_path, "rollout"), _arm_argv(tmp_path, "one_step")
    assert len(rollout) == len(control)
    differences = {left.split("=", 1)[0] for left, right in zip(rollout, control) if left != right}
    assert differences == _PERMITTED_ARM_DIFFERENCES, differences
    left, right = _overrides(rollout), _overrides(control)
    for key in _PERMITTED_ARM_DIFFERENCES - {"pos_rollout_arm"}:
        assert (
            left[key].replace("/rollout/", "/one_step/") == right[key]
        ), f"{key} differs in more than the arm segment: {left[key]} vs {right[key]}"
        assert "/rollout/" in left[key] and "/one_step/" in right[key]


def test_the_two_arms_cannot_share_checkpoint_state(tmp_path):
    """The scientifically critical finding: both arms got the same run name and a stable checkpoint
    root WITHOUT the arm, so running R-B and then matched-C0 could make C0 restore R-B's parameters,
    optimizer state, step and history -- and two concurrent arms collided outright."""
    left = _overrides(_arm_argv(tmp_path, "rollout"))
    right = _overrides(_arm_argv(tmp_path, "one_step"))
    assert left["checkpoint_dir"] != right["checkpoint_dir"]
    assert left["pos_resume_parent"] != right["pos_resume_parent"]
    assert left["base_output_directory"] != right["base_output_directory"]
    assert "/rollout/" in left["checkpoint_dir"] and "/one_step/" in right["checkpoint_dir"]
    # ...and the run-level recipe lock is the one thing they DO share, by design.
    assert left["pos_recipe_lock"] == right["pos_recipe_lock"]
    assert left["run_name"] == right["run_name"], "the pair is one run; only the destinations split"


def test_a_second_arm_with_a_different_recipe_is_refused_by_the_run_level_lock(tmp_path):
    """No shell can police two queue submissions a day apart, so the launcher's one-array emission is
    only half of the fix. The lock is the other half: the first arm publishes the normalized recipe,
    and a second arm that differs in a seed, a learning rate or a data directory does not start."""
    from maxdiffusion.trainers.wan_pos_rollout_trainer import normalized_recipe, publish_recipe_lock

    class _C:
        def __init__(self, mapping):
            self.__dict__.update(mapping)

        def get_keys(self):
            return dict(self.__dict__)

    base = _yaml_config()
    path = str(tmp_path / "recipe_lock.json")
    first = publish_recipe_lock(path, _C({**base, "pos_rollout_arm": "rollout"}), arm="rollout")
    assert first["adopted"] is False

    paired = publish_recipe_lock(
        path,
        _C(
            {
                **base,
                "pos_rollout_arm": "one_step",
                "checkpoint_dir": "gs://b/one_step/ckpt",
                "base_output_directory": "gs://b/one_step/art",
                "pos_resume_parent": "gs://b/one_step/attempts",
            }
        ),
        arm="one_step",
    )
    assert paired["adopted"] is True, "only the arm and its destinations changed: that IS the control"

    for key, value in (
        ("seed", 7),
        ("learning_rate", 1e-4),
        ("train_data_dir", "gs://other/train"),
        ("eval_every", 500),
        ("pos_logical_batch", 128),
    ):
        with pytest.raises(ValueError, match="DIFFERENT recipe") as excinfo:
            publish_recipe_lock(path, _C({**base, "pos_rollout_arm": "one_step", key: value}), arm="one_step")
        assert key in str(excinfo.value), "the refusal names the key, so it says what to fix"

    recipe = normalized_recipe(_C(base))
    for excluded in _PERMITTED_ARM_DIFFERENCES | {"pos_recipe_lock"}:
        assert excluded not in recipe, f"{excluded} is a permitted difference and must not be locked"
    assert recipe["seed"] == 0 and recipe["max_train_steps"] == 10000


def test_an_unknown_arm_is_refused_before_anything_starts(tmp_path):
    proc, calls = _run(tmp_path, TRAIN_LAUNCHER, POS_ROLLOUT_ARM="corrective_ss")
    assert proc.returncode != 0
    assert "is not an arm exp_06 declares" in proc.stdout
    assert not [call for call in calls if call[0] == "PREFETCH"], "it must refuse before the prefetch"


def test_an_explicitly_empty_arm_or_mode_is_a_wrapper_bug_and_is_not_defaulted(tmp_path):
    """``${VAR-default}`` and not ``${VAR:-default}``: an empty arm means a wrapper interpolated an
    unset variable, and quietly running R-B there is the one substitution that voids the pilot."""
    proc, _ = _run(tmp_path / "arm", TRAIN_LAUNCHER, POS_ROLLOUT_ARM="")
    assert proc.returncode != 0 and "is not an arm exp_06 declares" in proc.stdout
    proc, _ = _run(tmp_path / "mode", TRAIN_LAUNCHER, POS_JOB_MODE="")
    assert proc.returncode != 0 and "is not a job this launcher runs" in proc.stdout


# =============================================================================================
# 3. THE RUNBOOK RULES (plan §4; issues #10-#13), executed.
# =============================================================================================


def test_the_output_roots_are_derived_and_a_caller_cannot_flatten_them(tmp_path):
    """T6-3: a caller-supplied ARTIFACT_ROOT or CHECKPOINT_DIR used to remove phase/attempt/arm
    scoping entirely, in BOTH launchers. The customizable input is now the storage PARENT."""
    for launcher, extra in ((TRAIN_LAUNCHER, {}), (EVAL_LAUNCHER, {"POS_EVAL_PHASE": "anchor"})):
        _, calls = _run(
            tmp_path / launcher.stem,
            launcher,
            ARTIFACT_ROOT="gs://bucket/flat",
            CHECKPOINT_DIR="gs://bucket/flat",
            OUTPUT_DIR="gs://bucket/parent",
            RUN_NAME="m3",
            ATTEMPT="att-X",
            **extra,
        )
        overrides = _overrides(_entrypoint_argv(calls))
        assert overrides["base_output_directory"] != "gs://bucket/flat", launcher.name
        assert overrides["checkpoint_dir"] != "gs://bucket/flat", launcher.name
        assert overrides["base_output_directory"].startswith("gs://bucket/parent/m3/")
        assert "att-X" in overrides["base_output_directory"] or launcher is EVAL_LAUNCHER


def test_the_resume_input_is_distinct_from_the_fresh_attempt_output(tmp_path):
    """T6-3: "distinguish immutable resume INPUT from fresh attempt OUTPUT". The launcher used to
    reuse one mutable checkpoint tree for both, so a retry inherited whatever was in it."""
    first = _overrides(_arm_argv(tmp_path / "a", "rollout", ATTEMPT="att-FIRST"))
    second = _overrides(_arm_argv(tmp_path / "b", "rollout", ATTEMPT="att-SECOND"))
    assert first["checkpoint_dir"].endswith("/att-FIRST/checkpoints")
    assert second["checkpoint_dir"].endswith("/att-SECOND/checkpoints")
    assert first["checkpoint_dir"] != second["checkpoint_dir"], "a retry writes into its OWN root"
    assert first["base_output_directory"] != second["base_output_directory"], "a retry would collide"
    assert first["pos_resume_parent"] == second["pos_resume_parent"], "both search the same attempts root"
    assert "att-" not in first["pos_resume_parent"], "the search root is not attempt-scoped; the output is"
    assert first["checkpoint_dir"].startswith(first["pos_resume_parent"])


def test_only_a_complete_publication_of_this_arm_at_this_sha_is_adopted(tmp_path):
    """The other half of T6-3, where it can actually be executed: "select only the latest COMPLETE
    checkpoint whose recorded SHA matches the derived running code"."""
    from maxdiffusion.trainers import wan_pos_rollout_trainer as trainer

    parent = str(tmp_path / "attempts")
    mine, theirs = "a" * 40, "b" * 40
    trainer.publish_attempt(
        parent,
        attempt="att-1",
        arm="rollout",
        code_sha=mine,
        context_digest="d" * 64,
        step=1000,
        checkpoint_dir=f"{parent}/att-1/checkpoints",
    )
    trainer.publish_attempt(
        parent,
        attempt="att-2",
        arm="rollout",
        code_sha=mine,
        context_digest="d" * 64,
        step=3000,
        checkpoint_dir=f"{parent}/att-2/checkpoints",
    )
    trainer.publish_attempt(
        parent,
        attempt="att-3",
        arm="rollout",
        code_sha=theirs,
        context_digest="d" * 64,
        step=9000,
        checkpoint_dir=f"{parent}/att-3/checkpoints",
    )
    trainer.publish_attempt(
        parent,
        attempt="att-4",
        arm="one_step",
        code_sha=mine,
        context_digest="d" * 64,
        step=9000,
        checkpoint_dir=f"{parent}/att-4/checkpoints",
    )
    # An attempt that crashed before publishing its marker: a directory, but not COMPLETE.
    (Path(parent) / "att-5" / "checkpoints").mkdir(parents=True)

    selected = trainer.select_resume_publication(parent, code_sha=mine, arm="rollout", context_digest="d" * 64)
    assert selected["attempt"] == "att-2" and selected["step"] == 3000, "the newest COMPLETE one, not att-3/4/5"
    assert (
        trainer.select_resume_publication(parent, code_sha=mine, arm="one_step", context_digest="d" * 64)["attempt"]
        == "att-4"
    )
    assert trainer.select_resume_publication(parent, code_sha="c" * 40, arm="rollout", context_digest="d" * 64) is None
    assert (
        trainer.select_resume_publication(
            str(tmp_path / "nothing-here"), code_sha=mine, arm="rollout", context_digest="d" * 64
        )
        is None
    )

    # ...and an edited marker is not adopted at all, rather than adopted with better numbers.
    marker = Path(parent) / "att-2" / trainer.PUBLICATION_NAME
    stored = json.loads(marker.read_text())
    stored["payload"]["step"] = 99999
    marker.write_text(json.dumps(stored))
    assert (
        trainer.select_resume_publication(parent, code_sha=mine, arm="rollout", context_digest="d" * 64)["attempt"]
        == "att-1"
    )
    with pytest.raises(ValueError, match="does not describe its payload"):
        trainer.load_publication(str(marker))


def test_an_incomplete_publication_is_never_adopted(tmp_path):
    from maxdiffusion.trainers import wan_pos_rollout_trainer as trainer

    parent = tmp_path / "attempts"
    (parent / "att-1").mkdir(parents=True)
    payload = {
        "protocol": trainer.PUBLICATION_PROTOCOL,
        "attempt": "att-1",
        "arm": "rollout",
        "code_sha": "a" * 40,
        "context_digest": "d" * 64,
        "step": 5000,
        "checkpoint_dir": str(parent / "att-1" / "checkpoints"),
        "complete": False,
    }
    import hashlib

    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    (parent / "att-1" / trainer.PUBLICATION_NAME).write_text(json.dumps({"payload": payload, "sha256": digest}))
    assert (
        trainer.select_resume_publication(str(parent), code_sha="a" * 40, arm="rollout", context_digest="d" * 64)
        is None
    )
    with pytest.raises(ValueError, match="not marked complete"):
        trainer.load_publication(str(parent / "att-1" / trainer.PUBLICATION_NAME))


def test_the_attempt_stamp_is_generated_when_the_queue_does_not_supply_one(tmp_path):
    _, calls = _run(tmp_path, TRAIN_LAUNCHER, RUN_NAME="m3")
    root = _overrides(_entrypoint_argv(calls))["base_output_directory"]
    assert re.search(r"/att-\d{8}T\d{6}Z/artifacts$", root), root


def test_an_attempt_id_that_is_a_path_is_refused(tmp_path):
    """A caller-shaped segment would let a job escape its own namespace even though the root is derived."""
    for launcher in (TRAIN_LAUNCHER, EVAL_LAUNCHER):
        proc, calls = _run(tmp_path / launcher.stem, launcher, ATTEMPT="../../elsewhere")
        assert proc.returncode != 0 and "is not an attempt id" in proc.stdout
        assert not [call for call in calls if call[0] == "PREFETCH"]


def test_the_caller_s_xtrace_state_is_preserved_around_the_secrets(tmp_path):
    """Issue #12. exp_04's launcher force-enables ``set -x`` after sourcing secrets, spraying every
    later expansion into a teed log. Both properties are pinned: tracing is off inside the source,
    and the CALLER's state — on or off — is what comes back out."""
    for launcher in (TRAIN_LAUNCHER, EVAL_LAUNCHER):
        source = launcher.read_text()
        assert '__shell_flags="$-"' in source and "set +x" in source
        assert 'case "${__shell_flags}" in *x*) set -x ;; esac' in source
        # ...and no unconditional re-enable, which is the exact defect.
        assert not re.search(r"^\s*set -x\s*$", source, re.MULTILINE), launcher.name
    # The settled null launcher IS the counterexample, so the assertion above is not vacuous.
    assert re.search(r"^\s*set -x\s*$", NULL_LAUNCHER.read_text(), re.MULTILINE)


def test_the_log_is_teed_before_anything_that_can_fail(tmp_path):
    """exp_04's R10 finding 11: a run that died in prefetch left a log that never mentioned prefetch."""
    for launcher in (TRAIN_LAUNCHER, EVAL_LAUNCHER):
        lines = launcher.read_text().splitlines()
        tee = next(index for index, line in enumerate(lines) if line.startswith("exec > >(tee"))
        for risky in ("prefetch_hf_snapshot.sh", "git rev-parse", "python -", "source .venv"):
            first = next((index for index, line in enumerate(lines) if risky in line), None)
            if first is not None:
                assert first > tee, f"{launcher.name}: {risky!r} runs before the log is teed"
    proc, _ = _run(tmp_path, TRAIN_LAUNCHER, SHIM_PREFETCH_EXIT="1")
    assert proc.returncode != 0
    assert "[prefetch stub]" in proc.stdout, "a run that died in prefetch must still have logged it"


def test_a_failed_preflight_stops_the_launch_before_the_entrypoint(tmp_path):
    """A shim that always succeeds cannot tell a launcher that stops from one that carries on to the
    5B model (S10a's LOW finding), so the shim is made to fail on request."""
    for launcher in (TRAIN_LAUNCHER, EVAL_LAUNCHER):
        proc, calls = _run(tmp_path / launcher.stem, launcher, SHIM_PREFLIGHT_EXIT="1")
        assert proc.returncode != 0
        assert not [call for call in calls if call[0] == "PYTHON" and len(call) > 1 and call[1] != "-"]


def test_the_preflight_is_present_and_checks_the_exp06_dispatch(tmp_path):
    for launcher, expected in (
        (
            TRAIN_LAUNCHER,
            ("wan_pos_rollout_trainer", "pos_rollout_arms", "orbax.checkpoint", "describe_resume_candidates"),
        ),
        (EVAL_LAUNCHER, ("eval_wan_pos_rollout", "pos_rollout_gates", "skimage")),
    ):
        source = launcher.read_text()
        assert "PREFLIGHT" in source, launcher.name
        for token in expected:
            assert token in source, f"{launcher.name}: preflight does not check {token}"


def test_a_missing_commit_stops_the_launch(tmp_path):
    """SHA-bound adoption starts here: a run whose provenance is not real cannot be adopted later."""
    proc, calls = _run(tmp_path, TRAIN_LAUNCHER, COMMIT="not-a-sha")
    assert proc.returncode != 0 and "40-hex code_sha" in proc.stdout
    assert not calls, "nothing may run before provenance is established"


def test_a_working_tree_without_the_entrypoint_or_the_config_is_refused(tmp_path):
    """T6 MAJOR: the old sandbox returned success although ``src/`` and the config were absent, so
    "the launcher works" was never actually asserted. Both launchers now check."""
    for launcher in (TRAIN_LAUNCHER, EVAL_LAUNCHER):
        root, env = _sandbox(tmp_path / launcher.stem, launcher)
        env["POS_FIT_AUTHORIZATION"] = FAKE_AUTHORIZATION
        (root / "src").unlink()
        proc = subprocess.run(
            ["/bin/bash", f"bash_scripts/{launcher.name}"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode != 0
        assert "is not in this working tree" in proc.stdout, proc.stdout[-2000:]


def test_the_fail_closed_reconciliation_note_is_surfaced_to_the_operator(tmp_path):
    """The blocker round's operational consequence: a root whose selection sibling cannot be
    reconciled will not start, and a worker log must say that rather than look like a crash."""
    for launcher in (TRAIN_LAUNCHER, EVAL_LAUNCHER):
        proc, _ = _run(tmp_path / launcher.stem, launcher)
        assert "selection artifact" in proc.stdout
        assert "DESIGNED refusal" in proc.stdout or "not that this job crashed" in proc.stdout
    proc, _ = _run(tmp_path / "lock", TRAIN_LAUNCHER)
    assert "DIFFERENT recipe" in proc.stdout, "the pair-lock refusal is named too"


def test_no_three_argument_getattr_and_no_settled_file_is_touched():
    for launcher in (TRAIN_LAUNCHER, EVAL_LAUNCHER):
        source = launcher.read_text()
        assert "getattr(" not in source, launcher.name
        # Comment lines legitimately NAME the file this launcher deliberately is not, so the scan
        # is on executable lines (the same false positive T3a hit with `stop_gradient`).
        executable = "\n".join(
            line for line in source.splitlines() if line.strip() and not line.lstrip().startswith("#")
        )
        assert "generate_wan_null_adapter" not in executable, "exp_05's S9 tripwire"
        assert source.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")


# =============================================================================================
# 4. THE M1 PROBE MODE (T7-4's launcher half) and the evaluation protocol's dependencies (T6-2).
# =============================================================================================


def test_the_launcher_has_an_m1_probe_mode_that_runs_the_probe_entrypoint(tmp_path):
    """ "The launcher has no M1 probe mode at all, though the plan requires one" — plan §4-P1 and
    §5-7. M1 PUBLISHES the authorization, so its path is derived and attempt-scoped."""
    proc, calls = _run(
        tmp_path,
        TRAIN_LAUNCHER,
        POS_JOB_MODE="fit_probe",
        RUN_NAME="m1",
        ATTEMPT="att-X",
        OUTPUT_DIR="gs://bucket/parent",
        POS_FIT_AUTHORIZATION="",
    )
    assert proc.returncode == 0, proc.stdout[-3000:]
    argv = _entrypoint_argv(calls)
    assert argv[0] == PROBE_ENTRY and argv[1] == CONFIG_ARG
    overrides = _overrides(argv)
    assert (
        overrides["pos_fit_authorization"] == "gs://bucket/parent/m1/fit_probe/attempts/att-X/fit_authorization.json"
    )
    assert "/rollout/" not in overrides["pos_fit_authorization"], "M1 measures BOTH arms in one job"


def test_probe_mode_derives_an_adoption_root_that_spans_this_jobs_attempts(tmp_path):
    """F5. The ladder is ~3.5 h and the zone kills the VM inside it, so a restart must be able to
    adopt the cells a prior attempt already published. The root it looks in is the ATTEMPTS root —
    the same immutable resume INPUT `pos_resume_parent` names — not this attempt's own output tree,
    which is by construction empty when the attempt starts (issue #13)."""
    overrides = _overrides(
        _entrypoint_argv(
            _run(
                tmp_path,
                TRAIN_LAUNCHER,
                POS_JOB_MODE="fit_probe",
                RUN_NAME="m1",
                ATTEMPT="att-X",
                OUTPUT_DIR="gs://bucket/parent",
                POS_FIT_AUTHORIZATION="",
            )[1]
        )
    )
    assert overrides["pos_fit_adoption_root"] == "gs://bucket/parent/m1/fit_probe/attempts"
    assert "att-X" not in overrides["pos_fit_adoption_root"], "an attempt cannot adopt only from itself"
    assert overrides["pos_fit_authorization"].startswith(overrides["pos_fit_adoption_root"] + "/att-X/")


def test_probe_mode_lets_a_wrapper_widen_the_adoption_root(tmp_path):
    """The submit wrapper scopes OUTPUT_DIR per attempt (`$M1ROOT/$ATT`), which makes the derived
    attempts root a fresh empty tree on every attempt. Such a wrapper passes its M1 root explicitly,
    and the launcher must carry that through rather than overriding it with the derived default."""
    overrides = _overrides(
        _entrypoint_argv(
            _run(
                tmp_path,
                TRAIN_LAUNCHER,
                POS_JOB_MODE="fit_probe",
                RUN_NAME="m1",
                ATTEMPT="att-X",
                OUTPUT_DIR="gs://bucket/parent/att-OUTER",
                POS_FIT_ADOPTION_ROOT="gs://bucket/parent",
                POS_FIT_AUTHORIZATION="",
            )[1]
        )
    )
    assert overrides["pos_fit_adoption_root"] == "gs://bucket/parent"


def test_train_mode_does_not_ask_a_training_job_to_adopt_cells(tmp_path):
    """Nothing in M2/M3 publishes or adopts a fit-probe cell; the key stays at its YAML default."""
    overrides = _overrides(_entrypoint_argv(_run(tmp_path, TRAIN_LAUNCHER)[1]))
    assert overrides["pos_fit_adoption_root"] == ""


def test_train_mode_without_an_m1_authorization_is_refused_before_the_prefetch(tmp_path):
    proc, calls = _run(tmp_path, TRAIN_LAUNCHER, POS_FIT_AUTHORIZATION="")
    assert proc.returncode != 0
    assert "POS_FIT_AUTHORIZATION is empty" in proc.stdout
    assert "fit_probe first" in proc.stdout, "the refusal says what to run"
    assert not [call for call in calls if call[0] == "PREFETCH"]


def test_an_unknown_job_mode_is_refused(tmp_path):
    proc, calls = _run(tmp_path, TRAIN_LAUNCHER, POS_JOB_MODE="benchmark")
    assert proc.returncode != 0 and "is not a job this launcher runs" in proc.stdout
    assert not [call for call in calls if call[0] == "PREFETCH"]


def test_the_eval_launcher_hands_the_evaluator_the_exp06_config(tmp_path):
    proc, calls = _run(tmp_path, EVAL_LAUNCHER)
    assert proc.returncode == 0, proc.stdout[-3000:]
    argv = _entrypoint_argv(calls)
    assert argv[0] == EVAL_ENTRY and argv[1] == CONFIG_ARG
    assert _overrides(argv)["pos_eval_phase"] == "anchor", "the anchor is the default phase"


@pytest.mark.parametrize(
    "phase, required",
    [
        ("anchor", ()),
        ("benchmark", ("POS_ANCHOR_CERTIFICATE",)),
        ("gates", ("POS_ANCHOR_CERTIFICATE", "POS_BENCHMARK_ROW")),
        ("confirm", ("POS_ANCHOR_CERTIFICATE", "POS_BENCHMARK_ROW", "POS_DEV_CERTIFICATE")),
    ],
)
def test_each_phase_declares_and_transports_its_prior_phase_artifacts(tmp_path, phase, required):
    """T6-2: ``confirm`` was accepted directly, and the evaluator was handed no anchor certificate,
    no frozen benchmark row and no DEV certificate -- so with separate attempt roots a later phase
    could not even LOCATE its prerequisites. Refused BEFORE the prefetch when absent."""
    supplied = {name: f"gs://bucket/{name.lower()}.json" for name in required}
    proc, calls = _run(tmp_path / f"ok-{phase}", EVAL_LAUNCHER, POS_EVAL_PHASE=phase, **supplied)
    assert proc.returncode == 0, proc.stdout[-3000:]
    overrides = _overrides(_entrypoint_argv(calls))
    assert overrides["pos_eval_phase"] == phase
    for name in required:
        key = name.lower().replace("pos_", "pos_", 1)
        assert overrides[key.lower()] == supplied[name], f"{phase} does not transport {name}"

    for missing in required:
        partial = {name: value for name, value in supplied.items() if name != missing}
        proc, calls = _run(tmp_path / f"missing-{phase}-{missing}", EVAL_LAUNCHER, POS_EVAL_PHASE=phase, **partial)
        assert proc.returncode != 0, f"{phase} ran without {missing}"
        assert f"requires {missing}" in proc.stdout
        assert not [call for call in calls if call[0] == "PREFETCH"], "refused before the prefetch"


def test_each_protocol_phase_gets_its_own_phase_and_attempt_scoped_root(tmp_path):
    for phase in ("anchor", "benchmark", "gates", "confirm"):
        supplied = {"POS_ANCHOR_CERTIFICATE": "a", "POS_BENCHMARK_ROW": "b", "POS_DEV_CERTIFICATE": "c"}
        _, calls = _run(
            tmp_path / phase,
            EVAL_LAUNCHER,
            POS_EVAL_PHASE=phase,
            RUN_NAME="m4",
            ATTEMPT="att-X",
            OUTPUT_DIR="gs://bucket/p",
            **supplied,
        )
        root = _overrides(_entrypoint_argv(calls))["base_output_directory"]
        assert root == f"gs://bucket/p/m4/rollout/eval/{phase}/attempts/att-X"


def test_an_unknown_or_empty_eval_phase_is_refused(tmp_path):
    for value in ("scoring", ""):
        proc, calls = _run(tmp_path / f"p{value or 'empty'}", EVAL_LAUNCHER, POS_EVAL_PHASE=value)
        assert proc.returncode != 0
        assert "is not a phase this evaluator wires" in proc.stdout
        assert not [call for call in calls if call[0] == "PREFETCH"]


def test_the_evaluator_is_pointed_at_the_named_attempt_or_at_the_selection_root(tmp_path):
    _, calls = _run(
        tmp_path / "named", EVAL_LAUNCHER, RUN_NAME="m4", OUTPUT_DIR="gs://b/p", POS_CHECKPOINT_ATTEMPT="att-TRAINED"
    )
    named = _overrides(_entrypoint_argv(calls))
    assert named["checkpoint_dir"] == "gs://b/p/m4/rollout/train/attempts/att-TRAINED/checkpoints"
    assert named["pos_resume_parent"] == "gs://b/p/m4/rollout/train/attempts"

    _, calls = _run(tmp_path / "unnamed", EVAL_LAUNCHER, RUN_NAME="m4", OUTPUT_DIR="gs://b/p")
    unnamed = _overrides(_entrypoint_argv(calls))
    assert unnamed["checkpoint_dir"] == unnamed["pos_resume_parent"], "unset means 'resolve the latest COMPLETE'"

    proc, _ = _run(tmp_path / "bad", EVAL_LAUNCHER, POS_CHECKPOINT_ATTEMPT="../../elsewhere")
    assert proc.returncode != 0 and "is not an attempt id" in proc.stdout


# =============================================================================================
# 5. THE REAL-PYTHON INTEGRATION LAYER (T6 MAJOR).
#
# The layer above records argv and is faithful for shell expansion; it cannot tell a working
# launcher from one whose preflight heredoc is a syntax error, because the shim never parses it.
# Here a REAL interpreter executes the heredocs against controlled module stubs, and the emitted
# overrides are parsed against the real YAML using pyconfig's own coercion rules.
# =============================================================================================

_YAML_PARSERS = {str: str, int: int, float: float, bool: lambda s: s.lower() in ("true", "false"), list: str}


def _run_real(tmp_path: Path, launcher: Path, *, limit: int = 99, **overrides):
    root, env = _sandbox(tmp_path, launcher)
    env.update(
        {
            "SHIM_REAL_PYTHON": sys.executable,
            "SHIM_REAL_LIMIT": str(limit),
            "PYTHONPATH": _stub_modules(root),
            "JAX_PLATFORMS": "cpu",
        }
    )
    env.update({key: str(value) for key, value in overrides.items()})
    proc = subprocess.run(
        ["/bin/bash", f"bash_scripts/{launcher.name}"], cwd=root, env=env, capture_output=True, text=True, timeout=900
    )
    calls = []
    record = root / "record.txt"
    if record.exists():
        calls = [
            line.split(UNIT) for line in record.read_text().splitlines() if line.startswith(("PYTHON", "PREFETCH"))
        ]
    return proc, calls


def _write_authorization(path: Path, *, code_sha: str, cells=((("rollout", 32, 2)),)):
    """A REAL M1 artifact, built through the module the launcher's prerequisite check loads.

    The context is derived and then re-stamped with ``code_sha`` because this test process runs
    inside the checkout, where the derivation correctly refuses to call two SHAs one program. The
    artifact this produces is what a probe on a WORKER (a tarball, no git objects, ``COMMIT``
    exported) would have published.
    """
    import dataclasses

    from maxdiffusion import pos_rollout_fit_probe as probe

    class _Device:
        device_kind = "v6e"

    import yaml as _yaml

    values = _yaml.safe_load(POS_CONFIG.read_text())
    # Provenance is CONTENT-BOUND since round F1 (LS-10): `derive_model_revision` fails closed on a
    # name it cannot resolve to an immutable revision, so this fixture names a real directory rather
    # than depending on whether this machine happens to have the Wan snapshot cached.
    model = path.parent / "model_snapshot"
    model.mkdir(parents=True, exist_ok=True)
    (model / "weights.safetensors").write_bytes(b"w" * 64)
    values["pretrained_model_name_or_path"] = str(model)

    class _C:
        def __init__(self, mapping):
            self.__dict__.update(mapping)

    context = dataclasses.replace(
        probe.derive_probe_context(_C(values), devices=[_Device()], environ={}), code_sha=code_sha
    )
    measurements = [
        probe.CellMeasurement(
            cell=probe.FitCell(arm, microbatch, k_b),
            context_digest=context.digest(),
            compile_seconds=480.0,
            step_seconds=3.5,
            eval_seconds=600.0,
            checkpoint_seconds=90.0,
            peak_bytes=20 * 1024**3,
            capacity_bytes=32 * 1024**3,
            reservation_failures=0,
            # Review W1 A3: provenance is a REQUIRED field, and only a runtime-attributable peak
            # authorizes. This fixture stands in for a probe that ran on a resettable backend.
            peak_source=probe.PEAK_SOURCE_RUNTIME_RESET,
        )
        for arm, microbatch, k_b in cells
    ]
    evidence = probe.build_evidence(
        context, measurements, max_train_steps=10_000, eval_every=1_000, checkpoint_every=1_000
    )
    probe.publish_authorization(str(path), evidence)


def test_the_real_preflight_executes_and_the_launcher_reaches_the_entrypoint(tmp_path):
    """The whole launcher, with real Python parsing and executing both heredocs."""
    authorization = tmp_path / "m1.json"
    _write_authorization(authorization, code_sha=COMMIT)
    proc, calls = _run_real(
        tmp_path,
        TRAIN_LAUNCHER,
        POS_FIT_AUTHORIZATION=str(authorization),
        OUTPUT_DIR=str(tmp_path / "store"),
        RUN_NAME="m3",
        ATTEMPT="att-X",
    )
    assert proc.returncode == 0, proc.stdout[-4000:]
    assert "[prereq] M1 authorization" in proc.stdout
    assert "[preflight] ok: imports, exp_06 dispatch" in proc.stdout
    assert "[preflight] resume: no COMPLETE publication" in proc.stdout
    argv = _entrypoint_argv(calls)
    assert argv[0] == TRAIN_ENTRY and Path(REPO_ROOT / argv[0]).exists()
    assert Path(REPO_ROOT / argv[1]).exists(), "argv[1] must be a config that is really there"


def test_the_real_preflight_refuses_an_authorization_measured_on_another_commit(tmp_path):
    """The prerequisite gate is not decoration: it loads the artifact through the real module."""
    authorization = tmp_path / "m1.json"
    _write_authorization(authorization, code_sha="0" * 40)
    proc, calls = _run_real(
        tmp_path, TRAIN_LAUNCHER, limit=1, POS_FIT_AUTHORIZATION=str(authorization), OUTPUT_DIR=str(tmp_path / "store")
    )
    assert proc.returncode != 0
    assert "was measured on 0000" in proc.stdout and "re-run M1 at this SHA" in proc.stdout
    assert not [call for call in calls if call[0] == "PREFETCH"], "refused before the prefetch"


def test_the_real_preflight_refuses_an_absent_or_unreadable_authorization(tmp_path):
    proc, _ = _run_real(
        tmp_path / "absent",
        TRAIN_LAUNCHER,
        limit=1,
        POS_FIT_AUTHORIZATION=str(tmp_path / "nope.json"),
        OUTPUT_DIR=str(tmp_path / "s"),
    )
    assert proc.returncode != 0 and "does not exist" in proc.stdout

    junk = tmp_path / "junk.json"
    junk.write_text('{"payload": {"protocol": "exp06.fit_authorization.v2"}, "sha256": "0"}')
    proc, _ = _run_real(
        tmp_path / "junk", TRAIN_LAUNCHER, limit=1, POS_FIT_AUTHORIZATION=str(junk), OUTPUT_DIR=str(tmp_path / "s")
    )
    assert proc.returncode != 0 and "not a usable fit authorization" in proc.stdout


def test_the_real_preflight_reports_the_publication_it_would_adopt(tmp_path):
    """T6-3 end to end: a COMPLETE publication at this SHA is named in the worker log before the run."""
    from maxdiffusion.trainers import wan_pos_rollout_trainer as trainer

    store = tmp_path / "store"
    parent = store / "m3" / "rollout" / "train" / "attempts"
    parent.mkdir(parents=True)
    trainer.publish_attempt(
        str(parent),
        attempt="att-OLD",
        arm="rollout",
        code_sha=COMMIT,
        context_digest="d" * 64,
        step=4000,
        checkpoint_dir=str(parent / "att-OLD" / "checkpoints"),
    )
    trainer.publish_attempt(
        str(parent),
        attempt="att-OTHER",
        arm="one_step",
        code_sha=COMMIT,
        context_digest="d" * 64,
        step=9000,
        checkpoint_dir=str(parent / "att-OTHER" / "checkpoints"),
    )
    authorization = tmp_path / "m1.json"
    _write_authorization(authorization, code_sha=COMMIT)
    proc, _ = _run_real(
        tmp_path,
        TRAIN_LAUNCHER,
        POS_FIT_AUTHORIZATION=str(authorization),
        OUTPUT_DIR=str(store),
        RUN_NAME="m3",
        ATTEMPT="att-NEW",
    )
    assert proc.returncode == 0, proc.stdout[-4000:]
    # F5c: the preflight REPORTS candidates and never decides. It cannot derive the context that
    # decides adoption -- it runs before the HF prefetch and before the distributed system is up, so
    # it can identify neither the model snapshot nor the real device count -- and a preflight that
    # predicted adoption from a commit label would be the very defect the selector was just fixed for.
    assert "att-OLD step 4000" in proc.stdout, "the candidate for this arm at this SHA is reported"
    assert "ADOPTION IS NOT DECIDED HERE" in proc.stdout, "and it says so, rather than implying a decision"
    assert "att-OTHER" not in proc.stdout, "the other arm's publication is not this arm's to adopt"


def test_the_real_eval_prerequisite_check_refuses_a_failing_dev_certificate(tmp_path):
    """T6-2's sharpest clause: 'the issued PASSING DEV certificate for confirm'."""
    certificate = tmp_path / "dev.json"
    certificate.write_text(json.dumps({"certificate": "exp06.dev_primary_gate.v1", "passed": False}))
    others = {"POS_ANCHOR_CERTIFICATE": str(tmp_path / "a.json"), "POS_BENCHMARK_ROW": str(tmp_path / "b.json")}
    for name in others.values():
        Path(name).write_text(json.dumps({"protocol": "exp06.anchor.v1", "code_sha": COMMIT}))
    proc, calls = _run_real(
        tmp_path,
        EVAL_LAUNCHER,
        limit=1,
        POS_EVAL_PHASE="confirm",
        POS_DEV_CERTIFICATE=str(certificate),
        OUTPUT_DIR=str(tmp_path / "s"),
        **others,
    )
    assert proc.returncode != 0
    assert "did NOT pass" in proc.stdout
    assert not [call for call in calls if call[0] == "PREFETCH"]


def test_the_real_eval_prerequisite_check_refuses_an_unissued_or_foreign_sha_artifact(tmp_path):
    loose, foreign = tmp_path / "loose.json", tmp_path / "foreign.json"
    loose.write_text(json.dumps({"mean_ssim": 0.2946}))
    foreign.write_text(json.dumps({"protocol": "exp06.anchor.v1", "code_sha": "0" * 40}))
    for artifact, message in ((loose, "carries no issuing marker"), (foreign, "issued at 0000")):
        proc, _ = _run_real(
            tmp_path / artifact.stem,
            EVAL_LAUNCHER,
            limit=1,
            POS_EVAL_PHASE="benchmark",
            POS_ANCHOR_CERTIFICATE=str(artifact),
            OUTPUT_DIR=str(tmp_path / "s"),
        )
        assert proc.returncode != 0 and message in proc.stdout, proc.stdout[-2000:]


def test_every_emitted_override_parses_against_the_real_config(tmp_path):
    """pyconfig types each override like the YAML value it overrides, so a value that cannot be
    coerced is a launch-time crash. The argv layer records the strings; this parses them."""
    config = _yaml_config()
    for launcher in (TRAIN_LAUNCHER, EVAL_LAUNCHER):
        _, calls = _run(tmp_path / f"parse_{launcher.stem}", launcher)
        for key, value in _overrides(_entrypoint_argv(calls)).items():
            assert key in config, f"{launcher.name}: {key} is not declared by the YAML"
            parser = _YAML_PARSERS.get(type(config[key]))
            assert parser is not None, f"{key}: pyconfig cannot coerce a {type(config[key]).__name__} from the shell"
            if type(config[key]) is bool:
                assert value.lower() in ("true", "false"), f"{key}={value!r} is not a bool pyconfig accepts"
            else:
                parser(value)  # raises if the launcher emitted something the config cannot type


def test_both_secret_states_are_exercised_and_xtrace_never_leaks_a_secret(tmp_path):
    """The old sandbox bypassed secrets entirely. Here the secrets file EXISTS, and the launcher is
    run once with the caller's xtrace off and once with it on."""
    for traced in (False, True):
        root, env = _sandbox(tmp_path / f"secrets_{traced}", TRAIN_LAUNCHER)
        env["POS_FIT_AUTHORIZATION"] = FAKE_AUTHORIZATION
        secrets = Path(env["HOME"]) / ".config" / "irom-tpu"
        secrets.mkdir(parents=True)
        (secrets / "secrets.env").write_text('export IROM_TPU_TOKEN="s3cr3t-do-not-log"\n')
        command = (
            ["/bin/bash", "-x", f"bash_scripts/{TRAIN_LAUNCHER.name}"]
            if traced
            else ["/bin/bash", f"bash_scripts/{TRAIN_LAUNCHER.name}"]
        )
        proc = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stdout[-3000:]
        combined = proc.stdout + proc.stderr
        assert "s3cr3t-do-not-log" not in combined, "the secret reached the log"
        if traced:
            assert "+ " in combined, "the caller asked for xtrace and it must come back on"
        else:
            assert "+ python" not in combined, "xtrace was off and must have stayed off"


# =============================================================================================
# 6. W2b — THE TOPOLOGY, declared once and derived from.
#
# The wiring round (W2) added `assert_loader_yields_the_logical_batch`: the input pipeline must hand
# the loop exactly `pos_logical_batch` examples per step, because accumulation preserves the logical
# batch and never adapts to what the loader produced. `pyconfig` derives that width as
# `device_count x per_device_batch_size` -- so an M2 submission with this launcher's old
# `PER_DEVICE_BATCH_SIZE=1.0` default loaded 8 examples where the run declares 256, and was refused
# at startup. The knob is now gone: the operator declares the TOPOLOGY (which they necessarily know,
# having chosen it) and the launcher does the arithmetic.
#
# Declared -> derived -> VERIFIED: the declaration is checked where the authoritative number exists,
# which is inside the real process. A preflight cannot do it -- on a multi-host job
# `jax.device_count()` is the LOCAL count until the distributed system is initialized, so a preflight
# deriving from it would compute 256/8 = 32 on every host of a v6e-64 job and train at a global batch
# of 2048. The check that is right on one topology and silently wrong on the other is exactly the
# "agrees today by coincidence" failure this campaign keeps finding, so the verification is left
# where it is exact.
# =============================================================================================


def _config_from(overrides, *, device_count):
    """The rollout YAML, updated by what the launcher actually emitted, plus pyconfig's own
    ``global_batch_size_to_load = device_count x per_device_batch_size``."""
    from maxdiffusion.trainers.wan_pos_rollout_trainer import WanPosRolloutTrainer

    values = _yaml_config()
    for key, raw in overrides.items():
        if key in values and type(values[key]) in _YAML_PARSERS:
            values[key] = _YAML_PARSERS[type(values[key])](raw)
    values["global_batch_size_to_load"] = int(device_count * float(values["per_device_batch_size"]))

    class _Config(dict):
        def __getattr__(self, key):
            if key not in self:
                raise ValueError(f"Key {key} not in config")
            return self[key]

        def get_keys(self):
            return dict(self)

    return WanPosRolloutTrainer(_Config(values))


@pytest.mark.parametrize(
    "device_count, expected",
    [(8, "32.0"), (64, "4.0"), (256, "1.0")],
    ids=["v6e-8", "v6e-64", "one-per-chip"],
)
def test_the_per_device_batch_is_DERIVED_from_the_declared_topology(tmp_path, device_count, expected):
    """The arithmetic the operator no longer performs. This is the round's red: with the old
    `PER_DEVICE_BATCH_SIZE=1.0` default the emitted value was 1.0 on every topology."""
    proc, calls = _run(tmp_path / f"t{device_count}", TRAIN_LAUNCHER, POS_DEVICE_COUNT=device_count)
    assert proc.returncode == 0, proc.stdout[-3000:]
    overrides = _overrides(_entrypoint_argv(calls))
    assert overrides["per_device_batch_size"] == expected
    assert float(overrides["per_device_batch_size"]) * device_count == float(overrides["pos_logical_batch"])


def test_the_emitted_recipe_PASSES_the_trainers_width_check_and_the_old_default_did_not(tmp_path):
    """The loop closed against W2's own refusal: the launcher's emitted recipe is fed to the trainer
    check that an M2 submission would hit, on the real class. Both directions are asserted, so this
    stays a reproduction of the defect and not merely a test of the fix."""
    _, calls = _run(tmp_path, TRAIN_LAUNCHER, POS_DEVICE_COUNT=8)
    emitted = _overrides(_entrypoint_argv(calls))
    assert _config_from(emitted, device_count=8).assert_loader_yields_the_logical_batch() == 256

    # ...and the value this launcher used to emit is refused, by name, with the knob to turn.
    with pytest.raises(ValueError, match="per_device_batch_size") as refusal:
        _config_from(
            {**emitted, "per_device_batch_size": "1.0"}, device_count=8
        ).assert_loader_yields_the_logical_batch()
    assert "loads 8 examples per step but pos_logical_batch is 256" in str(refusal.value)


def test_no_operator_can_supply_a_per_device_batch_at_all(tmp_path):
    """The knob is GONE, not defaulted better: an env variable by that name reaches nothing. A
    documented value is one an operator can still get wrong; a derived one is not."""
    proc, calls = _run(tmp_path, TRAIN_LAUNCHER, POS_DEVICE_COUNT=8, PER_DEVICE_BATCH_SIZE="2.0")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert _overrides(_entrypoint_argv(calls))["per_device_batch_size"] == "32.0", "an operator value was honoured"
    # The source-side twin of the executed assertion above, and it pins the EXPANSION rather than the
    # word: the launcher's comment names the historical default on purpose, and a test that could not
    # tell a comment from a read would have forced the record out of the file to stay green.
    assert "${PER_DEVICE_BATCH_SIZE" not in TRAIN_LAUNCHER.read_text(), "the launcher still reads the variable"


def test_an_absent_or_empty_topology_is_refused_and_never_defaulted(tmp_path):
    """S10a's no-colon lesson: `${VAR-default}`, not `${VAR:-default}`. A wrapper that computes an
    empty topology has a bug, and substituting a default would run the wrong global batch silently."""
    proc, _ = _run(tmp_path / "empty", TRAIN_LAUNCHER, POS_DEVICE_COUNT="")
    assert proc.returncode != 0 and "POS_DEVICE_COUNT" in proc.stdout

    root, env = _sandbox(tmp_path / "absent", TRAIN_LAUNCHER)
    env["POS_FIT_AUTHORIZATION"] = FAKE_AUTHORIZATION
    env.pop("POS_DEVICE_COUNT")
    absent = subprocess.run(
        ["/bin/bash", f"bash_scripts/{TRAIN_LAUNCHER.name}"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert absent.returncode != 0 and "POS_DEVICE_COUNT" in absent.stdout


@pytest.mark.parametrize(
    "declared, why",
    [
        ("0", "a topology of zero chips is not a topology"),
        ("-8", "a negative chip count"),
        ("v6e-8", "the accelerator NAME is not its chip count"),
        ("8.0", "a float chip count would silently truncate"),
        ("7", "7 does not divide the logical batch 256, so no per-device batch is correct"),
    ],
)
def test_a_topology_that_cannot_produce_the_logical_batch_is_refused(tmp_path, declared, why):
    proc, _ = _run(tmp_path / f"bad{abs(hash(declared)) % 997}", TRAIN_LAUNCHER, POS_DEVICE_COUNT=declared)
    assert proc.returncode != 0, why
    assert "POS_DEVICE_COUNT" in proc.stdout


@pytest.mark.parametrize("declared", ["0", "-256", "abc", "25.6"], ids=["zero", "negative", "word", "float"])
def test_a_logical_batch_that_is_not_a_count_of_examples_is_refused(tmp_path, declared):
    """Battery Y09 — a survivor, and a genuine hole in this file rather than in the launcher.

    The derivation divides by the topology, so BOTH operands have to be counts. Shell arithmetic
    evaluates a non-numeric operand as 0 rather than failing, which would make the divisibility rule
    pass (`0 % 8 == 0`), emit a per-device batch of `0.0`, and push the failure all the way into
    pyconfig's coercion — after the prefetch. The guard existed; nothing exercised it.
    """
    proc, _ = _run(tmp_path / f"lb{abs(hash(declared)) % 997}", TRAIN_LAUNCHER, POS_LOGICAL_BATCH=declared)
    assert proc.returncode != 0, "a logical batch that is not a positive whole number must not launch"
    assert "POS_LOGICAL_BATCH" in proc.stdout


def test_the_probe_and_the_training_run_on_one_topology_derive_the_SAME_per_device_batch(tmp_path):
    """`per_device_batch_size` is inside M1's recipe fingerprint, so a probe measured at one value
    cannot authorize a run at another. Deriving it from the topology makes M1-authorizes-M2 (same
    chips) and M1'-authorizes-M3 (different chips) fall out of the arithmetic instead of being
    remembered -- which is plan v2.7's M1' rule, enforced rather than documented."""
    _, probe_calls = _run(tmp_path / "probe", TRAIN_LAUNCHER, POS_JOB_MODE="fit_probe", POS_DEVICE_COUNT=8)
    _, train_calls = _run(tmp_path / "train", TRAIN_LAUNCHER, POS_DEVICE_COUNT=8)
    probed = _overrides(_entrypoint_argv(probe_calls))["per_device_batch_size"]
    trained = _overrides(_entrypoint_argv(train_calls))["per_device_batch_size"]
    assert probed == trained == "32.0"

    _, other = _run(tmp_path / "other", TRAIN_LAUNCHER, POS_DEVICE_COUNT=64)
    assert (
        _overrides(_entrypoint_argv(other))["per_device_batch_size"] != trained
    ), "a different topology, a different value"


def test_the_derived_value_is_echoed_with_the_topology_it_came_from(tmp_path):
    """A worker log has to say what global batch the job is actually running, or the first thing an
    operator does after a refusal is guess."""
    proc, _ = _run(tmp_path, TRAIN_LAUNCHER, POS_DEVICE_COUNT=8)
    assert "POS_DEVICE_COUNT=8" in proc.stdout
    assert "PER_DEVICE_BATCH=32.0" in proc.stdout
    assert "global batch 256" in proc.stdout, "the log must say what the job actually runs"


# =============================================================================================
# Round F5c — the F5b re-review's BLOCKER 2: resume selection dropped the manifest-bearing
# identity. Publications have recorded `context_digest` since T6-3, but nothing REQUIRED or
# MATCHED it: `load_publication` did not demand the field, `select_resume_publication` filtered
# on `(code_sha, arm)`, and `resume_source` derived the whole running context and then kept only
# `code_sha`. The reviewer executed the consequence — two same-SHA publications with different
# context digests, and the selector took the higher-step FOREIGN one.
#
# These run the REAL selector rather than reading its source (the F3c liveness lesson: a guard
# that inspects source proves a spelling, not a behaviour).
# =============================================================================================


def _publish(parent, attempt, *, arm="rollout", code_sha="a" * 40, context_digest="d" * 64, step=1000):
    from maxdiffusion.trainers import wan_pos_rollout_trainer as trainer

    return trainer.publish_attempt(
        parent,
        attempt=attempt,
        arm=arm,
        code_sha=code_sha,
        context_digest=context_digest,
        step=step,
        checkpoint_dir=f"{parent}/{attempt}/checkpoints",
    )


def test_two_same_sha_publications_with_different_contexts_do_not_resume_each_other(tmp_path):
    """The reviewer's executed construction. Two git-less deployments can carry the same `COMMIT`
    label and different running bytes; before F5c the higher-step foreign one won, and one arm's
    optimizer state resumed under another program's identity."""
    from maxdiffusion.trainers import wan_pos_rollout_trainer as trainer

    parent = str(tmp_path / "attempts")
    mine, foreign = "d" * 64, "e" * 64
    _publish(parent, "att-1", context_digest=mine, step=1000)
    _publish(parent, "att-2", context_digest=foreign, step=9000)

    selected = trainer.select_resume_publication(parent, code_sha="a" * 40, arm="rollout", context_digest=mine)
    assert selected is not None and selected["attempt"] == "att-1", "the FOREIGN higher-step attempt must lose"
    assert selected["context_digest"] == mine
    assert (
        trainer.select_resume_publication(parent, code_sha="a" * 40, arm="rollout", context_digest="f" * 64) is None
    ), "a context nobody published must adopt nothing at all"


def test_the_selector_cannot_be_called_without_a_context_to_match(tmp_path):
    """Fail-closed by signature: the identity cannot be dropped by forgetting an argument, which is
    exactly how it came to be dropped."""
    from maxdiffusion.trainers import wan_pos_rollout_trainer as trainer

    parent = str(tmp_path / "attempts")
    _publish(parent, "att-1")
    with pytest.raises(TypeError):
        trainer.select_resume_publication(parent, code_sha="a" * 40, arm="rollout")


def test_a_publication_without_a_context_digest_is_not_adoptable(tmp_path):
    """`load_publication` records the field but never demanded it, so a marker written by any older
    or hand-rolled publisher read as adoptable. It fails closed now."""
    import hashlib

    from maxdiffusion.trainers import wan_pos_rollout_trainer as trainer

    parent = tmp_path / "attempts"
    (parent / "att-1").mkdir(parents=True)
    payload = {
        "protocol": trainer.PUBLICATION_PROTOCOL,
        "attempt": "att-1",
        "arm": "rollout",
        "code_sha": "a" * 40,
        "step": 5000,
        "checkpoint_dir": str(parent / "att-1" / "checkpoints"),
        "complete": True,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    (parent / "att-1" / trainer.PUBLICATION_NAME).write_text(json.dumps({"payload": payload, "sha256": digest}))

    with pytest.raises(ValueError, match="context_digest"):
        trainer.load_publication(str(parent / "att-1" / trainer.PUBLICATION_NAME))
    assert (
        trainer.select_resume_publication(str(parent), code_sha="a" * 40, arm="rollout", context_digest="d" * 64)
        is None
    ), "unreadable markers are skipped, not fatal -- one damaged attempt must not stop a good resume"


def test_resume_source_matches_the_whole_derived_context_not_just_its_sha(tmp_path):
    """`resume_source` derived the complete context and then threw all but `code_sha` away. It now
    hands the selector the same digest adoption uses, and this runs the real method."""
    from maxdiffusion import pos_rollout_fit_probe as probe
    from maxdiffusion.trainers.wan_pos_rollout_trainer import WanPosRolloutTrainer

    parent = str(tmp_path / "attempts")
    derived = probe.ProbeContext(
        code_sha="a" * 40,
        manifest_digest="1" * 64,
        model_revision="m@" + "0" * 40,
        device_kind="v6e",
        device_count=8,
        geometry=(("height", 192),),
        recipe_fingerprint="9" * 64,
    )
    other = dataclasses.replace(derived, manifest_digest="2" * 64)
    _publish(parent, "att-1", context_digest=derived.digest(), step=1000)
    _publish(parent, "att-2", context_digest=other.digest(), step=9000)

    trainer = WanPosRolloutTrainer.__new__(WanPosRolloutTrainer)
    trainer.resume_parent = parent
    trainer.checkpoint_dir = f"{parent}/att-9/checkpoints"
    trainer.schedule = types.SimpleNamespace(arm="rollout")

    adopted = trainer.resume_source(derived)
    assert adopted is not None and adopted["attempt"] == "att-1", "the manifest-bearing identity must decide"
    assert trainer.resume_source(other)["attempt"] == "att-2", "and each context adopts its own"
