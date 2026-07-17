"""CPU-only config + launcher-wiring tests for the exp_01 full-FT overfit probe (round 5).

Round 5 ("configs-launchers") ships four artifacts and NO Python production code:

  * ``configs/base_wan_5b_full_ft.yml``   -- a standalone copy of the side-adapter base
    with exactly the review's deltas (pyconfig loads ONE file; no includes).
  * ``bash_scripts/train_wan_full_ft.sh`` -- the training wrapper (fresh-noise default,
    train-split eval default, full-FT env knobs).
  * ``bash_scripts/launch_wan_train.sh``  -- gains a ``full_ft`` queue arm WITHOUT the
    common ``CHECKPOINT_EVERY=100`` / ``CHECKPOINT_KEEP_PERIOD=1000`` / val-split
    ``EVAL_DATA_DIR`` / hard-coded side-adapter train script clobbering it.
  * ``bash_scripts/validate_wan_full_ft.sh`` -- the cohort validation wrapper.

The tests exercise these five concerns:

  (A) YAML DELTAS -- ``yaml.safe_load`` the new yml; every review-listed delta is present
      with its exact value; the seven genuine deltas actually differ from the base.
  (B) RETAINED PARITY -- load BOTH ymls; the review's retained-key list is byte-equal to
      the side-adapter base's values (dtypes, sharding rules, latent/action geometry,
      flow/sampling params, Adam coefficients, clipping, warmup, checkpoint/eval keys).
  (C) WRAPPER STATIC CHECKS -- textual greps (rung-1 style) pin the fresh-noise default,
      the train-split eval default, and the full-FT env knobs + python overrides.
  (D) LAUNCHER SEMANTICS -- the launcher is EXECUTED with ``tpu`` stubbed (and ``HOME``
      redirected so the real ``~/.local/bin/tpu`` cannot shadow the stub), capturing the
      exact env it would submit. This proves the ACTUAL variable-resolution semantics
      (arm value vs. common-default ordering), not just text presence -- and that the
      pre_context / side_adapter arms stay byte-identical.

  (E) PLAIN-COMMAND SELF-CONSISTENCY (round-5 strengthen, Codex F1) -- the standalone yml
      must satisfy plan §6 with NO overrides: ``per_device_batch_size: 8.0`` (-> GBS 512
      on the 64-chip primary target) and a live ``wandb_project``. pyconfig's
      ``user_init`` RECOMPUTES ``global_batch_size_to_{load,train_on}`` unconditionally
      as ``int(num_devices * per_device_batch_size)`` after the yaml+CLI merge (the two
      global keys are inert inputs; per-device is the only authoritative knob), so the
      REAL ``_HyperParameters.calculate_global_batch_sizes`` is executed against the
      yml's per-device value under a patched 64-device view and must reproduce the yml's
      stated 512s. The launcher full_ft arm's W&B project and the yml default are pinned
      to each other (single source of truth), and the train wrapper's batch defaults stay
      SMOKE-SCALED but are always passed as explicit CLI overrides (bare wrapper = dev
      smoke; the launcher exports 8/512/512 for real runs) -- that contract is pinned too.

CPU-only: no pipeline loads, no 5B weights. Concern (E) imports ``maxdiffusion.pyconfig``
(hence jax, on CPU) inside the test solely to execute its pure batch-derivation formula.
Shell files are validated with ``bash -n`` via subprocess. The darwin grain import stub in
``conftest.py`` covers the pyconfig import chain under pytest.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------------------
# Paths (repo root is five parents up: worklogs_yixun/tests/maxdiffusion/src/<root>).
# ---------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIGS = REPO_ROOT / "src" / "maxdiffusion" / "configs"
BASH = REPO_ROOT / "bash_scripts"

SIDE_YML = CONFIGS / "base_wan_5b_side_adapter.yml"
FULL_YML = CONFIGS / "base_wan_5b_full_ft.yml"

TRAIN_WRAPPER = BASH / "train_wan_full_ft.sh"
VALIDATE_WRAPPER = BASH / "validate_wan_full_ft.sh"
LAUNCHER = BASH / "launch_wan_train.sh"
SIDE_VALIDATE = BASH / "validate_wan_side_adapter.sh"

# Canonical DROID / output locations (plan §2.2/§2.3, §3).
TRAIN_PATH = "gs://v6_east1d/datasets/droid_wan_side_adapter/train"
VAL_PATH = "gs://v6_east1d/datasets/droid_wan_side_adapter/val"
FULL_OUTPUT = "gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-full-ft"


def _load(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _text(path: Path) -> str:
    with open(path, "r") as f:
        return f.read()


# =======================================================================================
# (A) YAML deltas -- the review's required-delta list, exact values.
# =======================================================================================


def test_full_yml_exists_and_parses():
    assert FULL_YML.exists(), f"missing {FULL_YML}"
    cfg = _load(FULL_YML)
    assert isinstance(cfg, dict) and cfg, "full-FT yml must parse to a non-empty mapping"


def test_full_yml_required_deltas():
    cfg = _load(FULL_YML)
    # model dispatch + CFG bypass + noise safety (plan §2.1, F1).
    assert cfg["model_type"] == "FULL_FT_TI2V"
    assert cfg["side_adapter_guide_scale"] == 1.0
    assert cfg["side_adapter_noise_mode"] == "fresh"
    # probe recipe (plan §2.2). learning_rate MUST be authored as ``1.e-5`` so
    # yaml.safe_load yields a float (``1e-5`` without the dot parses as a STRING).
    assert cfg["learning_rate"] == pytest.approx(1e-5)
    assert isinstance(cfg["learning_rate"], float)
    assert cfg["max_train_steps"] == 10000
    # checkpoint cadence + retention (plan §2.2, finding G1: keep_period == every).
    assert cfg["checkpoint_every"] == 2500
    assert cfg["checkpoint_keep_period"] == 2500
    # in-training eval on the TRAIN split (plan §2.2, F7): eval == train == DROID train.
    assert cfg["eval_data_dir"] == TRAIN_PATH
    assert cfg["train_data_dir"] == TRAIN_PATH
    assert cfg["eval_data_dir"] == cfg["train_data_dir"]
    # full-FT checkpoint root.
    assert cfg["output_dir"] == FULL_OUTPUT
    # new CLI-overridable cohort key MUST exist in the yml (pyconfig requires the key to
    # pre-exist for a `validation_ordinals=...` override), defaulting to contiguous.
    assert "validation_ordinals" in cfg
    assert cfg["validation_ordinals"] == ""
    # F1 (round-5 strengthen): the PLAIN yml command must satisfy plan §6 with no
    # overrides -- primary-run batch recipe (per_device 8.0 -> GBS 512 on 64 devices;
    # authored `8.0` so yaml yields a float, matching the base key's type for pyconfig
    # CLI coercion) and W&B live by default.
    assert cfg["per_device_batch_size"] == 8.0
    assert isinstance(cfg["per_device_batch_size"], float)
    assert cfg["global_batch_size_to_train_on"] == 512
    assert cfg["global_batch_size_to_load"] == 512
    assert cfg["wandb_project"] == "maxdiffusion-wan-full-ft"


def test_full_yml_genuine_deltas_differ_from_base():
    # Guard against a copy that silently dropped a delta: the genuine deltas must NOT
    # equal the side-adapter base. (noise_mode/max_train_steps equal the base by design,
    # so they are excluded here and asserted absolutely above.) The last four are the
    # round-5 F1 additions (plain-command §6 self-consistency).
    side = _load(SIDE_YML)
    full = _load(FULL_YML)
    for key in (
        "model_type",
        "side_adapter_guide_scale",
        "learning_rate",
        "checkpoint_every",
        "checkpoint_keep_period",
        "eval_data_dir",
        "output_dir",
        "per_device_batch_size",
        "global_batch_size_to_train_on",
        "global_batch_size_to_load",
        "wandb_project",
    ):
        assert full[key] != side[key], f"delta {key!r} was not changed from the base"


def test_full_yml_batch_trio_coheres_with_pyconfig_derivation(monkeypatch):
    # F1 (round-5 strengthen): execute the REAL pyconfig derivation against the yml.
    # pyconfig.user_init unconditionally overwrites global_batch_size_to_{load,train_on}
    # via _HyperParameters.calculate_global_batch_sizes(per_device_batch_size) AFTER the
    # yaml+CLI merge -- per_device is the only authoritative input. Under the 64-device
    # primary target, the yml's per-device value must resolve to exactly the GBS-512
    # recipe plan §6 requires, and the yml's own (inert, documentary) global values must
    # state that same result so the standalone file never contradicts itself.
    import jax  # deferred: only this test needs jax

    from maxdiffusion import pyconfig

    cfg = _load(FULL_YML)
    monkeypatch.setattr(jax, "devices", lambda: [object()] * 64)
    to_load, to_train_on = pyconfig._HyperParameters.calculate_global_batch_sizes(cfg["per_device_batch_size"])
    assert (to_load, to_train_on) == (512, 512)  # plan §6: GBS 512 on v6e-64
    # The yml's documentary globals equal what pyconfig actually derives from it.
    assert cfg["global_batch_size_to_load"] == to_load
    assert cfg["global_batch_size_to_train_on"] == to_train_on


def test_full_yml_header_documents_rationale():
    # The header comment block must state purpose+plan ref, the §2.1 guide-scale
    # rationale, the inert-adapter-keys note, and the G1 keep-period rationale.
    head = _text(FULL_YML).lower()
    assert "overfit" in head and "full" in head  # purpose: full-FT overfit probe
    assert "plan_full_ft_overfit" in head  # plan reference
    assert "guide" in head and "1.0" in head  # §2.1 guide-scale / CFG bypass rationale
    assert "inert" in head  # inert adapter-keys note
    assert "keep_period" in head or "checkpoint_keep_period" in head  # G1 retention
    assert "2500" in head and "g1" in head  # G1 keep-period rationale one-liner


# =======================================================================================
# (B) Retained parity -- every retained key equals the side-adapter base's value.
# =======================================================================================

# The review's retained-key list, plus the task's additions (Adam coefficients, clipping,
# warmup, and the checkpoint/eval keys). Each must be byte-equal to the side-adapter base.
_RETAINED_KEYS = [
    # dtypes / precision
    "weights_dtype",
    "activations_dtype",
    "scan_layers",
    # sampling / flow schedule
    "side_adapter_sampling_steps",
    "flow_shift",
    "flow_sigma_min",
    "flow_sigma_max",
    "side_adapter_t_sampling",
    # sharding rules (logical + data + mesh)
    "mesh_axes",
    "logical_axis_rules",
    "vae_logical_axis_rules",
    "data_sharding",
    # latent geometry 48 x 9 x 12 x 20
    "latent_channels",
    "latent_frames",
    "latent_height",
    "latent_width",
    # action geometry 32 x 7
    "action_len",
    "action_dim",
    # Adam coefficients + weight decay
    "adam_b1",
    "adam_b2",
    "adam_eps",
    "adam_weight_decay",
    # gradient clipping
    "opt_enable_grad_clipping",
    "max_grad_value",
    "opt_enable_grad_global_norm_clipping",
    "max_grad_norm",
    # warmup
    "warmup_steps_fraction",
    # checkpoint / eval compatibility keys
    "run_name",
    "checkpoint_dir",
    "checkpoint_step",
    "num_eval_videos",
    "validation_start_index",
    "validation_seed",
    "validation_output_dir",
    "seed",
    "fps",
]


def test_retained_keys_equal_side_adapter_base():
    side = _load(SIDE_YML)
    full = _load(FULL_YML)
    mismatched = {k: (side.get(k), full.get(k)) for k in _RETAINED_KEYS if side.get(k) != full.get(k)}
    assert mismatched == {}, f"retained keys drifted from the side-adapter base: {mismatched}"


def test_retained_geometry_absolute_values():
    # Belt-and-suspenders on the load-bearing geometry the review calls out explicitly.
    full = _load(FULL_YML)
    assert (full["latent_channels"], full["latent_frames"], full["latent_height"], full["latent_width"]) == (
        48,
        9,
        12,
        20,
    )
    assert (full["action_len"], full["action_dim"]) == (32, 7)
    assert full["weights_dtype"] == "bfloat16" and full["activations_dtype"] == "bfloat16"
    assert full["scan_layers"] is False
    assert full["side_adapter_sampling_steps"] == 25
    assert full["side_adapter_t_sampling"] == "uniform"
    assert (full["flow_shift"], full["flow_sigma_min"], full["flow_sigma_max"]) == (5.0, 0.0, 1.0)


# =======================================================================================
# (C) Training-wrapper static checks (train_wan_full_ft.sh).
# =======================================================================================


def test_train_wrapper_exists():
    assert TRAIN_WRAPPER.exists(), f"missing {TRAIN_WRAPPER}"


def test_train_wrapper_defaults_and_overrides():
    txt = _text(TRAIN_WRAPPER)
    # F1 foot-gun fix: the wrapper MUST default fresh (the side-adapter template defaults
    # `fixed`, which silently breaks generation).
    assert 'SIDE_ADAPTER_NOISE_MODE="${SIDE_ADAPTER_NOISE_MODE:-fresh}"' in txt
    assert ":-fixed}" not in txt  # no lingering fixed default anywhere
    # F7: in-training eval defaults to the TRAIN split.
    assert 'EVAL_DATA_DIR="${EVAL_DATA_DIR:-$TRAIN_DATA_DIR}"' in txt
    # probe recipe defaults (review's default list).
    assert 'LEARNING_RATE="${LEARNING_RATE:-1e-5}"' in txt
    assert 'MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"' in txt
    assert 'CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2500}"' in txt
    assert 'CHECKPOINT_KEEP_PERIOD="${CHECKPOINT_KEEP_PERIOD:-2500}"' in txt
    # exposed batch knobs -- SMOKE-SCALED defaults by contract (F1): a bare wrapper run is
    # a dev smoke that passes these as EXPLICIT CLI overrides (so it never silently
    # inherits the yml's 512-recipe); the launcher full_ft arm exports 8/512/512 for real
    # runs. The contract must be documented in the wrapper itself.
    assert 'PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1.0}"' in txt
    assert 'GLOBAL_BATCH_SIZE_TO_TRAIN_ON="${GLOBAL_BATCH_SIZE_TO_TRAIN_ON:-1}"' in txt
    assert 'GLOBAL_BATCH_SIZE_TO_LOAD="${GLOBAL_BATCH_SIZE_TO_LOAD:-$GLOBAL_BATCH_SIZE_TO_TRAIN_ON}"' in txt
    assert "SMOKE-SCALED" in txt  # the batch-contract doc comment


def test_train_wrapper_targets_full_ft_yml_and_passes_overrides():
    txt = _text(TRAIN_WRAPPER)
    assert "src/maxdiffusion/configs/base_wan_5b_full_ft.yml" in txt
    assert "base_wan_5b_side_adapter.yml" not in txt  # never the adapter config
    assert "src/maxdiffusion/train_wan.py" in txt
    # LR + noise + cadence overrides reach the python entrypoint.
    assert 'learning_rate="${LEARNING_RATE}"' in txt
    assert 'side_adapter_noise_mode="${SIDE_ADAPTER_NOISE_MODE}"' in txt
    assert 'max_train_steps="${MAX_TRAIN_STEPS}"' in txt
    assert 'checkpoint_every="${CHECKPOINT_EVERY}"' in txt
    assert 'checkpoint_keep_period="${CHECKPOINT_KEEP_PERIOD}"' in txt
    # ...and the batch trio is ALWAYS an explicit CLI override (F1 contract: no silent
    # yml inheritance on the wrapper path).
    assert 'per_device_batch_size="${PER_DEVICE_BATCH_SIZE}"' in txt
    assert 'global_batch_size_to_train_on="${GLOBAL_BATCH_SIZE_TO_TRAIN_ON}"' in txt
    assert 'global_batch_size_to_load="${GLOBAL_BATCH_SIZE_TO_LOAD}"' in txt


def test_train_wrapper_keeps_preamble():
    txt = _text(TRAIN_WRAPPER)
    assert txt.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in txt
    assert "ulimit -n" in txt  # open-file limit preamble kept verbatim
    assert "LIBTPU_INIT_ARGS" in txt  # XLA flags preamble kept


# =======================================================================================
# (C') Validation-wrapper static checks (validate_wan_full_ft.sh).
# =======================================================================================


def test_validate_wrapper_exists():
    assert VALIDATE_WRAPPER.exists(), f"missing {VALIDATE_WRAPPER}"


def test_validate_wrapper_targets_full_ft_yml_and_passes_cohort_knobs():
    txt = _text(VALIDATE_WRAPPER)
    assert "src/maxdiffusion/configs/base_wan_5b_full_ft.yml" in txt
    assert "base_wan_5b_side_adapter.yml" not in txt
    assert "src/maxdiffusion/generate_wan_side_adapter.py" in txt
    # The cohort passthroughs the review requires (A5).
    assert 'checkpoint_step="${CHECKPOINT_STEP}"' in txt
    assert 'validation_ordinals="${VALIDATION_ORDINALS}"' in txt
    assert 'validation_seed="${VALIDATION_SEED}"' in txt
    assert 'validation_output_dir="${VALIDATION_OUTPUT_DIR}"' in txt
    assert 'eval_data_dir="${EVAL_DATA_DIR}"' in txt
    # The VALIDATION_ORDINALS env knob exists and defaults to empty (contiguous fallback).
    assert 'VALIDATION_ORDINALS="${VALIDATION_ORDINALS:-}"' in txt


def test_adapter_validate_script_untouched():
    # HARD RULE: the adapter validation wrapper must stay byte-identical in behavior --
    # it still points at the adapter yml and never learns the full-FT config or ordinals.
    txt = _text(SIDE_VALIDATE)
    assert "src/maxdiffusion/configs/base_wan_5b_side_adapter.yml" in txt
    assert "base_wan_5b_full_ft.yml" not in txt
    assert "validation_ordinals" not in txt


# =======================================================================================
# (D) Launcher semantics -- execute launch_wan_train.sh with `tpu` stubbed and capture
#     the submitted env. This validates the ACTUAL variable-resolution ordering.
# =======================================================================================

# Launcher-controlled vars we strip from the inherited env so the launcher's own logic
# (case arm + common defaults + full-FT override) determines every value deterministically.
_LAUNCHER_CONTROLLED = [
    "SMOKE",
    "NAME",
    "TPU_CHIPS",
    "WAN_EXPERIMENT",
    "RUN_NAME",
    "OUTPUT_DIR",
    "WANDB_PROJECT",
    "MAX_TRAIN_STEPS",
    "CHECKPOINT_EVERY",
    "CHECKPOINT_KEEP_PERIOD",
    "EVAL_EVERY",
    "EVAL_MAX_BATCHES",
    "LOG_PERIOD",
    "SAVE_FINAL_CHECKPOINT",
    "PER_DEVICE_BATCH_SIZE",
    "GLOBAL_BATCH_SIZE_TO_TRAIN_ON",
    "GLOBAL_BATCH_SIZE_TO_LOAD",
    "TFRECORD_SHUFFLE_BUFFER_SIZE",
    "EVAL_DATA_DIR",
    "TRAIN_DATA_DIR",
    "MODEL_DIR",
    "ACTION_ADAPTER_TYPE",
    "TRAIN_SCRIPT",
    "COMMIT",
]


def _run_launcher(experiment: str, tmp_path: Path) -> list[str]:
    """Run the launcher with a stub ``tpu`` and return the argv lines it would submit.

    ``HOME`` is redirected to an empty temp dir so the launcher's own
    ``export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"`` cannot resolve the
    real ``~/.local/bin/tpu`` ahead of our stub; the stub dir is prepended to PATH so it
    wins over every real ``tpu`` on the inherited PATH.
    """
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    tpu_stub = shim_dir / "tpu"
    tpu_stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    tpu_stub.chmod(tpu_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    fake_home = tmp_path / "home"
    fake_home.mkdir()

    env = {k: v for k, v in os.environ.items() if k not in _LAUNCHER_CONTROLLED}
    env["PATH"] = f"{shim_dir}:{env.get('PATH', '')}"
    env["HOME"] = str(fake_home)
    env["WAN_EXPERIMENT"] = experiment
    env["WANDB_API_KEY"] = "test-key"  # keep WANDB_PROJECT populated (not blanked out)

    bash_exe = shutil.which("bash") or "/bin/bash"
    proc = subprocess.run(
        [bash_exe, str(LAUNCHER)],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, f"launcher exited {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return proc.stdout.splitlines()


def _env_value(lines: list[str], key: str) -> str:
    """Return the VALUE of the last ``KEY=VALUE`` token the launcher passed to `tpu`."""
    prefix = f"{key}="
    hits = [ln[len(prefix) :] for ln in lines if ln.startswith(prefix)]
    assert hits, f"{key} not among submitted env: {lines}"
    return hits[-1]


def test_launcher_full_ft_arm_semantics(tmp_path):
    lines = _run_launcher("full_ft", tmp_path)
    # The full-FT probe's values survive the common defaults (the crux of the restructure).
    assert _env_value(lines, "CHECKPOINT_EVERY") == "2500"  # NOT the common 100
    assert _env_value(lines, "CHECKPOINT_KEEP_PERIOD") == "2500"  # NOT the common 1000
    assert _env_value(lines, "EVAL_DATA_DIR") == TRAIN_PATH  # NOT the val split
    assert _env_value(lines, "OUTPUT_DIR") == FULL_OUTPUT
    assert _env_value(lines, "WANDB_PROJECT") == "maxdiffusion-wan-full-ft"
    # One source of truth (F1): the launcher submits EXACTLY the yml's default project.
    assert _env_value(lines, "WANDB_PROJECT") == _load(FULL_YML)["wandb_project"]
    assert _env_value(lines, "MAX_TRAIN_STEPS") == "10000"
    assert _env_value(lines, "SIDE_ADAPTER_NOISE_MODE") == "fresh"
    assert _env_value(lines, "RUN_NAME").startswith("wan-full-ft-")
    # Real queue runs get the primary-run batch recipe explicitly (F1 contract: the
    # launcher, not silent yml inheritance, sets 8/512/512 through the wrapper).
    assert _env_value(lines, "PER_DEVICE_BATCH_SIZE") == "8"
    assert _env_value(lines, "GLOBAL_BATCH_SIZE_TO_TRAIN_ON") == "512"
    assert _env_value(lines, "GLOBAL_BATCH_SIZE_TO_LOAD") == "512"
    # ...and it dispatches the full-FT train wrapper, NOT the hard-coded adapter script.
    assert "bash_scripts/train_wan_full_ft.sh" in lines
    assert "bash_scripts/train_wan_side_adapter.sh" not in lines


def test_launcher_side_adapter_arm_byte_identical(tmp_path):
    # Behavior preservation: the side_adapter arm submits exactly the historical values.
    lines = _run_launcher("side_adapter", tmp_path)
    assert _env_value(lines, "CHECKPOINT_EVERY") == "100"
    assert _env_value(lines, "CHECKPOINT_KEEP_PERIOD") == "1000"
    assert _env_value(lines, "EVAL_DATA_DIR") == VAL_PATH
    assert _env_value(lines, "OUTPUT_DIR") == "gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter"
    assert _env_value(lines, "WANDB_PROJECT") == "maxdiffusion-wan-side-adapter"
    assert _env_value(lines, "MAX_TRAIN_STEPS") == "10000"
    assert _env_value(lines, "ACTION_ADAPTER_TYPE") == "side_adapter"
    assert _env_value(lines, "RUN_NAME").startswith("wan-side_adapter-")
    assert "bash_scripts/train_wan_side_adapter.sh" in lines
    assert "bash_scripts/train_wan_full_ft.sh" not in lines


def test_launcher_pre_context_arm_byte_identical(tmp_path):
    lines = _run_launcher("pre_context", tmp_path)
    assert _env_value(lines, "CHECKPOINT_EVERY") == "100"
    assert _env_value(lines, "CHECKPOINT_KEEP_PERIOD") == "1000"
    assert _env_value(lines, "EVAL_DATA_DIR") == VAL_PATH
    assert _env_value(lines, "OUTPUT_DIR") == "gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-pre-context-adapter"
    assert _env_value(lines, "WANDB_PROJECT") == "maxdiffusion-wan-pre-context-adapter"
    assert _env_value(lines, "MAX_TRAIN_STEPS") == "30000"
    assert _env_value(lines, "ACTION_ADAPTER_TYPE") == "pre_context"
    assert _env_value(lines, "RUN_NAME").startswith("wan-pre_context-")
    assert "bash_scripts/train_wan_side_adapter.sh" in lines
    assert "bash_scripts/train_wan_full_ft.sh" not in lines


def test_launcher_full_ft_smoke_disables_checkpoints(tmp_path):
    # SMOKE must still win over the full-FT cadence (its override runs AFTER the arm's).
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    tpu_stub = shim_dir / "tpu"
    tpu_stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    tpu_stub.chmod(tpu_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in _LAUNCHER_CONTROLLED}
    env["PATH"] = f"{shim_dir}:{env.get('PATH', '')}"
    env["HOME"] = str(fake_home)
    env["WAN_EXPERIMENT"] = "full_ft"
    env["SMOKE"] = "1"
    env["WANDB_API_KEY"] = "test-key"
    bash_exe = shutil.which("bash") or "/bin/bash"
    proc = subprocess.run([bash_exe, str(LAUNCHER)], env=env, capture_output=True, text=True, timeout=90)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert _env_value(lines, "CHECKPOINT_EVERY") == "0"  # smoke disables checkpointing
    assert _env_value(lines, "MAX_TRAIN_STEPS") == "1"
    # Storage-light: no periodic checkpoints AND no final checkpoint.
    assert _env_value(lines, "SAVE_FINAL_CHECKPOINT") == "False"
    # Smoke sets its batch EXPLICITLY (the historical 1-step-at-GBS-512 queue smoke,
    # identical to the adapter arms) -- never silently inheriting the yml default (F1).
    assert _env_value(lines, "PER_DEVICE_BATCH_SIZE") == "8"
    assert "bash_scripts/train_wan_full_ft.sh" in lines  # still the full-FT wrapper


def test_launcher_rejects_unknown_experiment(tmp_path):
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    (shim_dir / "tpu").write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    (shim_dir / "tpu").chmod(0o755)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in _LAUNCHER_CONTROLLED}
    env["PATH"] = f"{shim_dir}:{env.get('PATH', '')}"
    env["HOME"] = str(fake_home)
    env["WAN_EXPERIMENT"] = "bogus"
    bash_exe = shutil.which("bash") or "/bin/bash"
    proc = subprocess.run([bash_exe, str(LAUNCHER)], env=env, capture_output=True, text=True, timeout=90)
    assert proc.returncode != 0  # invalid experiment -> non-zero exit
    assert "full_ft" in proc.stderr  # the error message now lists full_ft as a choice


# =======================================================================================
# bash -n syntax check on every touched/created shell file.
# =======================================================================================


@pytest.mark.parametrize("script", [TRAIN_WRAPPER, VALIDATE_WRAPPER, LAUNCHER], ids=lambda p: p.name)
def test_shell_scripts_pass_bash_n(script):
    assert script.exists(), f"missing {script}"
    bash_exe = shutil.which("bash") or "/bin/bash"
    proc = subprocess.run([bash_exe, "-n", str(script)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"bash -n failed for {script.name}:\n{proc.stderr}"
