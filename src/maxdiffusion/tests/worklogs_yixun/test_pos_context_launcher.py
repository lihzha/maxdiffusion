"""exp_05 S10a: the positive-slot launch script, pinned by EXECUTING it under bash.

``bash_scripts/run_wan_pos_inversion.sh`` is K1's only launch path: the exp_04 launcher hard-codes the
null YAML, and ``embedding_slot`` cannot ride the null config's command line by design (pyconfig only
coerces keys the YAML already declares). exp_04's settled launcher is read-only for exp_05 (plan
§6/F6), so this is a new sibling file and this is its test.

Two techniques, deliberately combined:

* **source pins** (exp_04's technique in ``test_null_adapter_entrypoint.py``) for properties that are
  about *ordering inside the file* -- tee before everything, COMMIT exported before the run, the XLA
  flags identical to the side-adapter launcher.
* **executed-under-bash pins** (new here) for everything about the env -> config-key interface. A
  substring assertion cannot tell ``pos_artifact_dir="${POS_ARTIFACT_DIR}"`` from a default that
  silently drifted, and it cannot prove that ``null_artifact_dir`` never reaches the command line.
  So the launcher is copied into a sandbox whose ``PATH`` holds shims -- a ``python`` that records its
  argv and exits, a stub HF prefetch, an optional ``timeout`` -- and the real thing runs end to end.
  The recorded argv IS the assertion surface: what the entrypoint would actually have been handed.

The shims are on a curated PATH (symlinks to the handful of real binaries the script needs), so a
missing ``timeout`` is a property of the test's PATH rather than of the machine running the suite.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import stat
import subprocess
import time
from collections import OrderedDict
from pathlib import Path

import pytest
import yaml

# --------------------------------------------------------------------------- paths and constants

REPO_ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = REPO_ROOT / "bash_scripts" / "run_wan_pos_inversion.sh"
NULL_LAUNCHER = REPO_ROOT / "bash_scripts" / "run_wan_null_inversion.sh"
SIDE_ADAPTER_LAUNCHER = REPO_ROOT / "bash_scripts" / "train_wan_side_adapter.sh"
POS_CONFIG = REPO_ROOT / "src" / "maxdiffusion" / "configs" / "base_wan_5b_pos_inversion.yml"

ENTRYPOINT = "src/maxdiffusion/run_wan_null_inversion.py"
CONFIG_ARG = "src/maxdiffusion/configs/base_wan_5b_pos_inversion.yml"
NULL_CONFIG_ARG = "src/maxdiffusion/configs/base_wan_5b_null_inversion.yml"

# The record separator between recorded argv words: no path, flag or override contains it.
UNIT = "\x1f"
COMMIT = "a1b2c3d4" * 5  # 40 hex characters, which is all the launcher's provenance check wants
EXTERNALS = ("bash", "date", "mkdir", "tee", "grep")

# The shims can FAIL on request (S10a review, the LOW finding): a shim that always succeeds cannot
# tell a launcher that stops on a failed preflight from one that carries on to the 5B model.
_PY_SHIM = """#!/bin/sh
{ printf 'PYTHON'; for arg in "$@"; do printf '\\037%s' "$arg"; done; printf '\\n'; } >> "$SHIM_RECORD"
if [ "$1" = "-" ]; then exit "${SHIM_PREFLIGHT_EXIT:-0}"; fi
exit "${SHIM_ENTRYPOINT_EXIT:-0}"
"""

_PREFETCH_STUB = """#!/bin/sh
{ printf 'PREFETCH'; for arg in "$@"; do printf '\\037%s' "$arg"; done; printf '\\n'; } >> "$SHIM_RECORD"
echo "[prefetch_hf_snapshot stub] $1"
exit "${SHIM_PREFETCH_EXIT:-0}"
"""

# Records, drops `--signal=... --kill-after=... <seconds>` and runs what it was wrapping, so an armed
# watchdog still has to deliver the entrypoint.
_TIMEOUT_SHIM = """#!/bin/sh
{ printf 'TIMEOUT'; for arg in "$@"; do printf '\\037%s' "$arg"; done; printf '\\n'; } >> "$SHIM_RECORD"
shift 3
exec "$@"
"""

# Every env the launcher exposes -> the config key it must land on, with a value distinguishable from
# both the default and every other key's value.
ENV_TO_KEY = {
    "RUN_NAME": ("run_name", "k1-b-arms-probe"),
    "MODEL_DIR": ("pretrained_model_name_or_path", "Wan-AI/Wan2.2-TI2V-5B-Diffusers-pinned"),
    "POS_MODE": ("null_mode", "adequacy_probe"),
    "POS_COHORT": ("null_cohort", "trainfit16"),
    "POS_DATA_DIR": ("null_data_dir", "gs://bucket/datasets/val"),
    "POS_MANIFEST_DIR": ("null_manifest_dir", "gs://bucket/manifests/j0/"),
    "POS_ARTIFACT_DIR": ("pos_artifact_dir", "gs://bucket/artifacts/exp05/k1"),
    "POS_STAGING_DIR": ("pos_staging_dir", "gs://bucket/artifacts/exp05/k1/_staging"),
    "POS_SELECTION_URI": ("pos_selection_uri", "gs://bucket/artifacts/exp05/k1/selection.json"),
    "POS_ADEQUACY_URI": ("pos_adequacy_uri", "gs://bucket/artifacts/exp05/probe/adequacy.json"),
    "POS_BATCH_SIZE": ("null_batch_size", "4"),
    "POS_DECODE_BATCH_SIZE": ("null_decode_batch_size", "2"),
    "POS_SMOKE_EXAMPLES": ("null_smoke_examples", "3"),
    "POS_INNER_ITERS": ("null_inner_iters", "25"),
    "POS_LR": ("null_lr", "0.03"),
    "POS_GUIDE_SCALE": ("null_guide_scale", "7.5"),
    "POS_L": ("pos_L", "1"),
    "POS_ABLATION_L": ("pos_ablation_L", "1,4,8"),
    "POS_NOISE_CONVENTION": ("null_noise_convention", "global"),
    "POS_LATENT_DTYPE": ("null_latent_dtype", "fp32"),
    "POS_PIXEL_CONVENTION": ("null_pixel_convention", "signed"),
    "PER_DEVICE_BATCH_SIZE": ("per_device_batch_size", "2.0"),
}

# Keys whose launcher default must be the checked-in YAML's value: a launcher that quietly recipes
# differently from the config everyone reads is exactly the drift the SOP's parity audit is for.
YAML_PINNED_DEFAULTS = (
    "null_mode",
    "null_cohort",
    "null_batch_size",
    "null_decode_batch_size",
    "null_smoke_examples",
    "null_inner_iters",
    "null_lr",
    "null_guide_scale",
    "null_noise_convention",
    "null_latent_dtype",
    "null_pixel_convention",
    "pos_L",
    "pos_ablation_L",
    "pos_selection_uri",
    "pos_adequacy_uri",
    "embedding_slot",
    "per_device_batch_size",
)

# The null slot's roots. The entrypoint enforces slot isolation (``positive_roots``); the launcher's
# job is to never hand it the question in the first place.
NULL_ROOT_KEYS = ("null_artifact_dir", "null_staging_dir")


# --------------------------------------------------------------------------- the sandbox


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _sandbox(tmp_path: Path, *, with_timeout: bool = False, with_git: bool = False) -> Path:
    """A working tree holding the launcher, a stub prefetch, and a PATH of shims -- nothing else."""
    root = tmp_path / "work"
    (root / "bash_scripts").mkdir(parents=True)
    (root / "home").mkdir()
    shutil.copy2(LAUNCHER, root / "bash_scripts" / LAUNCHER.name)
    _executable(root / "bash_scripts" / "prefetch_hf_snapshot.sh", _PREFETCH_STUB)

    binroot = root / "bin"
    binroot.mkdir()
    names = list(EXTERNALS) + (["git"] if with_git else [])
    for name in names:
        real = shutil.which(name)
        assert real, f"the test host has no {name!r}, which the launcher needs"
        os.symlink(real, binroot / name)
    _executable(binroot / "python", _PY_SHIM)
    if with_timeout:
        _executable(binroot / "timeout", _TIMEOUT_SHIM)
    return root


class _Result:
    """One end-to-end execution of the launcher, with everything the shims recorded."""

    def __init__(self, root: Path, completed: subprocess.CompletedProcess):
        self.root = root
        self.returncode = completed.returncode
        self.stdout = completed.stdout + completed.stderr

    @property
    def trace(self) -> list[list[str]]:
        record = self.root / "record.txt"
        if not record.exists():
            return []
        return [line.split(UNIT) for line in record.read_text(encoding="utf-8").splitlines() if line]

    @property
    def tags(self) -> list[str]:
        return [entry[0] for entry in self.trace]

    @property
    def command(self) -> list[str]:
        """The argv the entrypoint would have been launched with, or [] if it never ran."""
        for entry in self.trace:
            if entry[0] == "PYTHON" and len(entry) > 1 and entry[1].endswith("run_wan_null_inversion.py"):
                return entry[1:]
        return []

    @property
    def overrides(self) -> dict[str, str]:
        return dict(arg.split("=", 1) for arg in self.command[2:] if "=" in arg)

    def log(self, needle: str = "", timeout: float = 4.0) -> str:
        """The teed log file. Polled: ``tee`` is a separate process and may still be flushing."""
        logs = self.root / "logs"
        deadline = time.time() + timeout
        text = ""
        while True:
            files = sorted(logs.glob("*.log")) if logs.exists() else []
            text = "".join(handle.read_text(encoding="utf-8", errors="replace") for handle in files)
            if not needle or needle in text or time.time() > deadline:
                return text
            time.sleep(0.05)


def _run(root: Path, env: dict | None = None, *, commit: str | None = COMMIT) -> _Result:
    environment = {
        "PATH": str(root / "bin"),
        "HOME": str(root / "home"),
        "SHIM_RECORD": str(root / "record.txt"),
        "LOG_DIR": str(root / "logs"),
    }
    if commit is not None:
        environment["COMMIT"] = commit
    environment.update(env or {})
    completed = subprocess.run(
        [str(root / "bin" / "bash"), f"bash_scripts/{LAUNCHER.name}"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return _Result(root, completed)


@pytest.fixture(scope="module")
def default_run(tmp_path_factory) -> _Result:
    """One clean launch with nothing but COMMIT set: the defaults, as K1 would get them."""
    result = _run(_sandbox(tmp_path_factory.mktemp("defaults")))
    assert result.returncode == 0, result.stdout
    return result


@pytest.fixture(scope="module")
def configured_run(tmp_path_factory) -> _Result:
    """One launch with every exposed env set to a distinguishable value."""
    env = {name: value for name, (_, value) in ENV_TO_KEY.items()}
    result = _run(_sandbox(tmp_path_factory.mktemp("configured")), env)
    assert result.returncode == 0, result.stdout
    return result


@pytest.fixture(scope="module")
def yaml_config() -> dict:
    return yaml.safe_load(_source(POS_CONFIG))


# --------------------------------------------------------------------------- the config it launches


def test_the_launcher_hands_the_entrypoint_the_positive_config(default_run):
    """argv[1] decides the slot: the pos YAML is the only config that declares ``embedding_slot``."""
    assert default_run.command[0] == ENTRYPOINT
    assert default_run.command[1] == CONFIG_ARG
    assert NULL_CONFIG_ARG not in default_run.command


def test_the_slot_is_declared_on_the_command_line_not_merely_in_the_yaml(default_run):
    """A one-character edit to the YAML default would otherwise silently route K1 to the null slot."""
    assert default_run.overrides["embedding_slot"] == "positive"


def test_the_run_is_a_tpu_run(default_run):
    assert default_run.overrides["hardware"] == "tpu"


def test_every_override_is_a_key_the_config_declares(default_run, configured_run, yaml_config):
    """pyconfig coerces overrides to the YAML key's type and rejects undeclared keys -- an override the
    config never declared kills the job at argument parsing, after the TPU is already allocated."""
    for result in (default_run, configured_run):
        undeclared = [key for key in result.overrides if key not in yaml_config]
        assert not undeclared, f"the launcher passes keys the pos YAML does not declare: {undeclared}"


# --------------------------------------------------------------------------- env -> key mapping


@pytest.mark.parametrize("env_name", sorted(ENV_TO_KEY))
def test_every_exposed_env_lands_on_its_config_key(configured_run, env_name):
    key, value = ENV_TO_KEY[env_name]
    assert configured_run.overrides.get(key) == value, f"{env_name} did not reach {key}"


def test_the_defaults_are_the_checked_in_config_values(default_run, yaml_config):
    """The config-drift pin: read the YAML here, so a recipe change in either file has to be a change
    in both. J=10, lr=1e-2, w=5, L_pos=8 are the audited K1 recipe (worklog, parity audit)."""
    for key in YAML_PINNED_DEFAULTS:
        expected, actual = yaml_config[key], default_run.overrides[key]
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            assert float(actual) == float(expected), key
        else:
            assert actual == str(expected), key


def test_the_audited_recipe_defaults_are_the_ones_k1_was_approved_with(default_run):
    """Belt and braces on the four numbers the parity audit ratified: a YAML edit that moved both
    files together would pass the drift pin above and still change the experiment."""
    overrides = default_run.overrides
    assert float(overrides["null_inner_iters"]) == 10.0
    assert float(overrides["null_lr"]) == 0.01
    assert float(overrides["null_guide_scale"]) == 5.0
    assert float(overrides["pos_L"]) == 8.0
    assert overrides["pos_ablation_L"] == "1,8"
    assert overrides["null_mode"] == "capacity"
    assert overrides["null_cohort"] == "dev64"


def test_the_cohort_data_and_manifest_defaults_are_exp_04s_published_j0_set(default_run):
    """exp_05 consumes exp_04's J0 manifests verbatim (plan §4), so these two defaults are not free:
    they are read out of the null launcher, which is the file that ratified them."""
    null_source = _source(NULL_LAUNCHER)

    def null_default(name: str) -> str:
        match = re.search(rf'^{name}="\$\{{{name}:-(.*?)\}}"$', null_source, re.MULTILINE)
        assert match, f"{name} has no simple default in the null launcher"
        return match.group(1)

    assert default_run.overrides["null_manifest_dir"] == null_default("NULL_MANIFEST_DIR")
    assert default_run.overrides["null_data_dir"] == null_default("NULL_DATA_DIR")


def test_the_default_artifact_roots_are_the_positive_slots_own_run_scoped_tree(default_run):
    overrides = default_run.overrides
    run_name = overrides["run_name"]
    assert overrides["pos_artifact_dir"] == f"gs://v6_east1d/artifacts/exp05/{run_name}"
    assert overrides["pos_staging_dir"] == f"gs://v6_east1d/artifacts/exp05/{run_name}/_staging"
    assert "exp04" not in overrides["pos_artifact_dir"] and "exp04" not in overrides["pos_staging_dir"]


def test_the_run_name_scopes_the_default_roots(tmp_path):
    result = _run(_sandbox(tmp_path), {"RUN_NAME": "k1-dev64"})

    assert result.returncode == 0, result.stdout
    assert result.overrides["pos_artifact_dir"] == "gs://v6_east1d/artifacts/exp05/k1-dev64"
    assert result.overrides["pos_staging_dir"] == "gs://v6_east1d/artifacts/exp05/k1-dev64/_staging"


# --------------------------------------------------------------------------- slot safety


def test_no_null_slot_root_ever_reaches_the_command_line(default_run, configured_run):
    """``positive_roots`` refuses a positive run whose roots are the null slot's; the launcher must not
    hand it those keys at all -- with both set, two experiments' selection.json share a directory."""
    for result in (default_run, configured_run):
        for key in NULL_ROOT_KEYS:
            assert key not in result.overrides, f"{key} reached the entrypoint"


def test_the_launcher_source_never_mentions_the_null_slots_roots():
    source = _source(LAUNCHER)

    for key in NULL_ROOT_KEYS:
        assert key not in source, f"{key} appears in the positive launcher"
    for name in ("NULL_ARTIFACT_DIR", "NULL_STAGING_DIR"):
        assert name not in source, f"{name} appears in the positive launcher"


def test_a_null_slot_root_in_the_environment_is_ignored(tmp_path):
    """The env of a host that just ran exp_04's launcher still carries these."""
    env = {
        "NULL_ARTIFACT_DIR": "gs://v6_east1d/artifacts/exp04/j1",
        "NULL_STAGING_DIR": "gs://v6_east1d/artifacts/exp04/j1/_staging",
    }
    result = _run(_sandbox(tmp_path), env)

    assert result.returncode == 0, result.stdout
    assert not [arg for arg in result.command if "exp04" in arg]
    for key in NULL_ROOT_KEYS:
        assert key not in result.overrides


@pytest.mark.parametrize("env_name", ["POS_ARTIFACT_DIR", "POS_STAGING_DIR"])
def test_an_empty_artifact_root_is_refused_before_python_starts(tmp_path, env_name):
    """``positive_roots`` refuses an empty root too, but only after the model is on device."""
    result = _run(_sandbox(tmp_path), {env_name: ""})

    assert result.returncode == 1
    assert env_name in result.log(env_name)
    assert result.trace == []


@pytest.mark.parametrize("env_name", ["POS_ARTIFACT_DIR", "POS_STAGING_DIR"])
def test_a_root_under_exp_04s_tree_is_refused_before_python_starts(tmp_path, env_name):
    result = _run(_sandbox(tmp_path), {env_name: "gs://v6_east1d/artifacts/exp04/j1"})

    assert result.returncode == 1
    assert "exp04" in result.log("exp04")
    assert result.trace == []


# --------------------------------------------------------------------------- the mode gate


@pytest.mark.parametrize("mode", ["cache", "verify_replay", "direct_opt", "", "Capacity"])
def test_a_mode_the_positive_slot_does_not_wire_fails_before_python_starts(tmp_path, mode):
    """``pos_execute`` raises on these -- after the manifests are read and the 5B model is loaded. The
    gate is here because it costs nothing here."""
    result = _run(_sandbox(tmp_path), {"POS_MODE": mode})

    assert result.returncode == 1
    text = result.log("capacity|adequacy_probe")
    assert "POS_MODE" in text and "capacity|adequacy_probe" in text
    assert result.trace == [], "the gate let the run reach prefetch or python"


@pytest.mark.parametrize("mode", ["capacity", "adequacy_probe"])
def test_both_wired_modes_reach_the_entrypoint(tmp_path, mode):
    result = _run(_sandbox(tmp_path), {"POS_MODE": mode})

    assert result.returncode == 0, result.stdout
    assert result.overrides["null_mode"] == mode


# --------------------------------------------------------------------------- preflight and ordering


def test_the_log_is_all_terminal_output(default_run):
    """R10's lesson, inherited: a run that dies in prefetch must leave a log that mentions prefetch."""
    source = _source(LAUNCHER)
    tee = source.find('exec > >(tee -a "${LOG_FILE}") 2>&1')
    assert tee >= 0
    for later in ("prefetch_hf_snapshot.sh", "PREFLIGHT", "COMMIT=", "git status", ENTRYPOINT):
        assert tee < source.find(later), f"{later} is emitted before tee opens the log"

    text = default_run.log("[prefetch_hf_snapshot stub]")
    assert "LOG_FILE=" in text and "[prefetch_hf_snapshot stub]" in text
    assert "RUN_NAME=" in text and "POS_MODE=" in text


def test_the_prefetch_and_both_preflights_run_before_the_entrypoint(default_run):
    """Order is the whole point: an import failure or a drifted noise draw must be found before the
    5B model is on device, not after the arms have run."""
    tags = default_run.tags
    entry = next(index for index, entry in enumerate(default_run.trace) if entry[1:2] == [ENTRYPOINT])

    assert tags.index("PREFETCH") < entry
    assert tags.count("PYTHON") >= 3, "the two preflight heredocs and the entrypoint"
    assert all(tag in ("PREFETCH", "PYTHON") for tag in tags[:entry])


def test_the_prefetch_is_handed_the_model_directory(configured_run):
    prefetch = [entry for entry in configured_run.trace if entry[0] == "PREFETCH"]

    assert prefetch and prefetch[0][1:] == [ENV_TO_KEY["MODEL_DIR"][1]]


def test_the_preflight_imports_the_pipeline_class_the_driver_loads_and_the_positive_modes():
    source = _source(LAUNCHER)

    assert "from maxdiffusion.pipelines.wan.wan_pipeline_ti2v_2p2 import WanPipelineTI2V_2_2" in source
    assert 'getattr(WanPipelineTI2V_2_2, "from_pretrained", None)' in source
    assert "maxdiffusion.pos_context_modes" in source, "the positive slot's own module is never imported"


def test_the_ffmpeg_check_requires_an_executable_not_merely_a_file():
    source = _source(LAUNCHER)

    assert "os.access(ffmpeg, os.X_OK)" in source
    assert "shutil.which" not in source


def test_the_r1_noise_golden_is_asserted_on_device_before_any_arm_runs():
    """Every positive record is keyed to this draw by import from exp_04's module; if the device
    disagrees, K1's artifacts are keyed to noise nothing else can reproduce."""
    source = _source(LAUNCHER)

    assert source.find("keyed_noise(name, k)") < source.find(ENTRYPOINT)
    assert "1.392072319984436" in source and "0.18953724205493927" in source


def test_the_commit_is_exported_and_validated_before_anything_runs():
    source = _source(LAUNCHER)
    statements = [line.strip() for line in source.splitlines() if not line.strip().startswith("#")]

    assert "export COMMIT" in statements
    assert "grep -Eq '^[0-9a-f]{40}$'" in source
    assert source.find("export COMMIT") < source.find(ENTRYPOINT)


def test_a_run_without_a_real_commit_stops_before_it_prefetches(tmp_path):
    """Every published K1 record is stamped with COMMIT; 'unknown' provenance is not publishable."""
    result = _run(_sandbox(tmp_path), commit=None)

    assert result.returncode == 1
    assert "COMMIT" in result.log("COMMIT")
    assert result.trace == []


def test_a_real_commit_is_taken_from_git_when_the_environment_does_not_carry_one(tmp_path):
    root = _sandbox(tmp_path, with_git=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        check=True,
        capture_output=True,
    )
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()

    result = _run(root, commit=None)

    assert result.returncode == 0, result.stdout
    assert head in result.log(head)


# --------------------------------------------------------------------------- the watchdog


def test_the_watchdog_is_off_by_default_and_says_so(default_run):
    """The positive slot wires no direct-optimization mode, so no ceiling follows from the recipe: the
    hard stop is opt-in and job-level protection is the queue's own timeout. Silence would be worse
    than either -- an operator cannot tell 'no watchdog' from 'watchdog failed to arm'."""
    assert default_run.command and default_run.tags.count("TIMEOUT") == 0
    assert "watchdog" in default_run.log("watchdog").lower()


def test_an_armed_watchdog_wraps_the_entrypoint_without_swallowing_it(tmp_path):
    result = _run(_sandbox(tmp_path, with_timeout=True), {"POS_WATCHDOG_SECONDS": "900"})

    assert result.returncode == 0, result.stdout
    wrapper = [entry for entry in result.trace if entry[0] == "TIMEOUT"]
    assert wrapper, "POS_WATCHDOG_SECONDS was set but nothing wrapped the run"
    assert wrapper[0][1:4] == ["--signal=TERM", "--kill-after=60", "900"]
    assert wrapper[0][4] == "python" and wrapper[0][5] == ENTRYPOINT
    assert result.overrides["null_mode"] == "capacity"


def test_a_watchdog_that_cannot_arm_warns_and_still_runs(tmp_path):
    """``timeout`` is coreutils; a host without it must not silently lose the hard stop."""
    result = _run(_sandbox(tmp_path, with_timeout=False), {"POS_WATCHDOG_SECONDS": "900"})

    assert result.returncode == 0, result.stdout
    assert "WARNING" in result.log("WARNING")
    assert result.command, "the run was dropped because the watchdog could not arm"


# --------------------------------------------------------------------------- what the shim cannot see
#
# S10a review, the LOW finding: the ``python`` shim makes every invocation succeed, so a heredoc that
# is not valid Python, or an override pyconfig would reject, could pass every test above. Three
# answers: parse the embedded blocks for real, run pyconfig's REAL override contract over the recorded
# argv, and make the shims fail on request so the stop-on-failure behaviour is observed rather than
# assumed.


def _heredoc(marker: str) -> str:
    """The body of one embedded python block, exactly as bash would feed it to the interpreter."""
    source = _source(LAUNCHER)
    opener = f"<<'{marker}'\n"
    start = source.index(opener) + len(opener)
    end = source.index(f"\n{marker}\n", start)
    return source[start:end]


@pytest.mark.parametrize("marker", ["PREFLIGHT", "GOLDEN"])
def test_the_embedded_python_blocks_are_real_python_that_can_fail_the_run(marker):
    """A SyntaxError here is a launcher that dies on the host after the queue allocated the TPU -- and
    a block with no ``sys.exit(1)`` is a preflight that reports problems instead of stopping."""
    tree = ast.parse(_heredoc(marker), filename=f"<{marker}>")  # raises SyntaxError if it is not python
    compile(tree, f"<{marker}>", "exec")

    exits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "exit"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "sys"
        and [arg for arg in node.args if isinstance(arg, ast.Constant) and arg.value == 1]
    ]
    assert exits, f"the {marker} block cannot fail the run"


def _imported_names(marker: str) -> dict[str, set[str]]:
    tree = ast.parse(_heredoc(marker))
    imported: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.setdefault(node.module, set()).update(alias.name for alias in node.names)
    return imported


def test_the_preflight_block_imports_the_two_things_this_slot_needs_at_runtime():
    """AST, not substring: a commented-out import is exactly the regression a substring pin misses."""
    imported = _imported_names("PREFLIGHT")

    assert "WanPipelineTI2V_2_2" in imported.get("maxdiffusion.pipelines.wan.wan_pipeline_ti2v_2p2", set())
    assert {"pos_execute", "positive_plan", "positive_roots"} <= imported.get("maxdiffusion.pos_context_modes", set())


def test_the_golden_block_asserts_exp_04s_own_noise_primitives():
    """K1's B2/probe noise is exp_04's ``keyed_noise`` by import, so the golden must be that module's."""
    imported = _imported_names("GOLDEN")

    assert {"LATENT_SHAPE", "NOISE_DOMAIN", "keyed_noise"} <= imported.get(
        "maxdiffusion.models.wan.null_inversion_wan", set()
    )


def _pyconfig_override_parser():
    """pyconfig's REAL command-line contract, extracted from source and executed here.

    ``import maxdiffusion.pyconfig`` pulls the whole TPU/transformers stack, which is why S4 already
    extracts ``HyperParameters`` from the file rather than importing it (test_pos_context_runner.py).
    The same technique carries more weight here: this runs pyconfig's actual "key was passed at the
    command line but isn't in config" check and its actual per-key type coercion over the actual YAML,
    so an override the launcher passes has to survive the parser that will really see it.
    """
    tree = ast.parse((REPO_ROOT / "src" / "maxdiffusion" / "pyconfig.py").read_text(encoding="utf-8"))
    helpers = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in {"string_to_bool", "string_to_list"})
        or (
            isinstance(node, ast.Assign)
            and any(getattr(target, "id", "") == "_yaml_types_to_parser" for target in node.targets)
        )
    ]
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_HyperParameters")
    init = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    load = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "_load_kwargs")
    # Everything up to the first runtime step (`is_unittest = ...` guards the jax/distributed work) is
    # the override contract; the rest needs a TPU host.
    cut = next(
        index
        for index, stmt in enumerate(init.body)
        if isinstance(stmt, ast.Assign) and any(getattr(t, "id", "") == "is_unittest" for t in stmt.targets)
    )
    init.name, init.body = "parse", init.body[:cut] + [ast.Return(value=ast.Name(id="raw_keys", ctx=ast.Load()))]
    klass.body, tree.body = [init, load], helpers + [klass]
    ast.fix_missing_locations(tree)

    namespace: dict = {"yaml": yaml, "json": json, "ast": ast, "OrderedDict": OrderedDict}
    exec(compile(tree, "pyconfig-override-contract", "exec"), namespace)  # noqa: S102 -- the repo's own source
    return lambda argv: namespace["_HyperParameters"]().parse(argv)


def _parsed(result: _Result) -> dict:
    """The launcher's recorded argv, through pyconfig's own parser, against the real YAML."""
    return _pyconfig_override_parser()([result.command[0], str(POS_CONFIG), *result.command[2:]])


def test_the_real_config_parser_accepts_every_override_the_launcher_passes(default_run):
    """Undeclared key or uncoercible value => pyconfig raises at startup, after the queue has already
    allocated the v6e. The shim cannot see that; this does."""
    keys = _parsed(default_run)

    assert keys["embedding_slot"] == "positive"
    assert keys["pos_L"] == 8 and isinstance(keys["pos_L"], int)  # coerced to the YAML's type
    assert keys["null_lr"] == 0.01 and isinstance(keys["null_lr"], float)
    assert keys["null_smoke_examples"] == 0 and isinstance(keys["null_smoke_examples"], int)
    assert keys["pos_ablation_L"] == "1,8" and keys["hardware"] == "tpu"
    assert keys["per_device_batch_size"] == 1.0
    assert keys["pos_artifact_dir"].startswith("gs://") and keys["pos_staging_dir"].endswith("/_staging")
    assert keys["null_artifact_dir"] == "" and keys["null_staging_dir"] == ""  # untouched YAML defaults


def test_the_real_config_parser_accepts_every_configured_override_too(configured_run):
    keys = _parsed(configured_run)

    assert keys["pos_L"] == 1 and keys["null_inner_iters"] == 25 and keys["null_lr"] == 0.03
    assert keys["null_mode"] == "adequacy_probe" and keys["null_cohort"] == "trainfit16"
    assert keys["per_device_batch_size"] == 2.0 and keys["null_guide_scale"] == 7.5
    assert keys["pos_selection_uri"].endswith("selection.json")


def test_the_extracted_parser_is_the_real_one_and_still_refuses_what_pyconfig_refuses():
    """Liveness: if the extraction degraded into a no-op, the tests above would prove nothing."""
    parse = _pyconfig_override_parser()

    with pytest.raises(ValueError, match="isn't in config"):
        parse(["prog", str(POS_CONFIG), "pos_artifact_dirs=gs://typo"])
    with pytest.raises(ValueError, match="Couldn't parse value"):
        parse(["prog", str(POS_CONFIG), "pos_L=eight"])


def test_a_failing_prefetch_stops_the_run_before_any_python(tmp_path):
    """The HF download is the failure exp_04 hit repeatedly; it must not be followed by a model load."""
    result = _run(_sandbox(tmp_path), {"SHIM_PREFETCH_EXIT": "1"})

    assert result.returncode != 0
    assert result.tags == ["PREFETCH"], result.tags


def test_a_failing_preflight_stops_the_run_before_the_entrypoint(tmp_path):
    result = _run(_sandbox(tmp_path), {"SHIM_PREFLIGHT_EXIT": "1"})

    assert result.returncode != 0
    assert result.command == [], "the entrypoint ran after the preflight failed"
    assert result.tags == ["PREFETCH", "PYTHON"], result.tags  # died on the first block, not the second


def test_the_entrypoints_exit_code_is_the_launchers_exit_code(tmp_path):
    """``main`` returns the mode's exit code; a queue that sees 0 for a failed K1 run learns nothing."""
    result = _run(_sandbox(tmp_path), {"SHIM_ENTRYPOINT_EXIT": "3"})

    assert result.returncode == 3
    assert result.command, "the entrypoint never ran"


# --------------------------------------------------------------------------- parity with exp_04


def test_the_xla_flags_stay_identical_to_the_side_adapter_launcher():
    """Same 5B transformer, same mesh: a divergence here is a performance mystery, not a feature."""

    def flags(path: Path) -> set[str]:
        block = _source(path).split("LIBTPU_INIT_ARGS=", 1)[1]
        return {line.strip().rstrip('\\}"').strip() for line in block.split("}")[0].splitlines() if "--xla" in line}

    assert flags(LAUNCHER) == flags(SIDE_ADAPTER_LAUNCHER)
    assert flags(LAUNCHER) == flags(NULL_LAUNCHER)


def test_the_settled_null_launcher_still_launches_the_null_slot():
    """exp_05 never edits exp_04's settled files (plan §6/F6). The pair must stay slot-disjoint."""
    null_source = _source(NULL_LAUNCHER)

    assert NULL_CONFIG_ARG in null_source and CONFIG_ARG not in null_source
    assert "POS_" not in null_source and "pos_artifact_dir" not in null_source


def test_the_launcher_is_executable_and_parses():
    assert os.access(LAUNCHER, os.X_OK), "the launcher is not executable"
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True, capture_output=True)
