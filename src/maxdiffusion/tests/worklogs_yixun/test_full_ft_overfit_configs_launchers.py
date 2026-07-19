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

  (E) PLAIN-COMMAND SELF-CONSISTENCY (round-5 strengthen, Codex F1; recipe amended by
      mini-cycle 8 / plan §2.2 v3.1 / Query 7) -- the standalone yml must satisfy plan §6
      with NO overrides: ``per_device_batch_size: 4.0`` (-> GBS 256 on the 64-chip
      primary target; per-device 8 OOMs v6e-64 for full-FT, fit probe #4) and a live
      ``wandb_project``. pyconfig's ``user_init`` RECOMPUTES
      ``global_batch_size_to_{load,train_on}`` unconditionally as
      ``int(num_devices * per_device_batch_size)`` after the yaml+CLI merge (the two
      global keys are inert inputs; per-device is the only authoritative knob), so the
      REAL ``_HyperParameters.calculate_global_batch_sizes`` is executed against the
      yml's per-device value under a patched 64-device view and must reproduce the yml's
      stated 256s. The launcher full_ft arm's W&B project and the yml default are pinned
      to each other (single source of truth); the launcher batch defaults are now
      ARM-DEPENDENT (adapters 8/512/512, full_ft 4/256/256; explicit env wins for every
      arm), and the train wrapper's batch defaults stay SMOKE-SCALED but are always
      passed as explicit CLI overrides (bare wrapper = dev smoke).

  (F) SETUP.SH APT HARDENING (mini-cycle 7, strengthened x2 per Codex reviews) -- static
      checks pinning the fix for the 2026-07-18 v6e-64 fit-probe failures (workers stuck
      forever on the dpkg lock held by Ubuntu's post-boot auto-update; healthy hosts then
      died at the ~10-min JAX distributed-init deadline). Contract, with EVERY structural
      assertion evaluated on COMMAND text (comment lines stripped -- round-2 F3): one
      GLOBAL 420s wall-clock budget for the whole apt-critical section with apt/curl
      EXECUTION bounded via ``timeout <remaining>`` (round-2 F1), timeout-30-bounded
      systemctl calls (both apt-daily timers stopped synchronously first, then the
      service units), a jammy-safe escalation on the actual ``unattended-upgrade``
      process (KillMode=process means unit stops cannot kill it, Launchpad #1690980)
      using pgrep-captured exact PIDs -- never pattern-kill (round-2 MINOR) -- with
      60s per-apt lock bounds after verified release, the SIGKILL path DISCARDING the
      worker via loud exit instead of installing on unverifiable dpkg state (round-2
      MAJOR), the persistent ``systemctl disable`` gated behind ``EPHEMERAL_WORKER=1``
      (queue SETUP_CMD; the general TPU/GPU/dev installer stays current-boot-only), and
      the failure-swallowing ``(sudo bash || bash)`` wrapper replaced by
      ``$SUDO env ... bash`` so errors propagate through the outer ``set -e``. On
      bash < 4.2 the ``bash -n`` check runs against a copy with only the PRE-EXISTING
      ``[[ ! -v MODE ]]`` neutralized (real syntax coverage on darwin, not a skip).

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
SETUP_SH = REPO_ROOT / "setup.sh"  # root setup; the queue's --setup-cmd runs it on every worker

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
    # amended recipe (plan §2.2 v3.1, Query 7): 20000 steps @ GBS 256 keeps the 3.55-pass
    # sample budget after fit probe #4 proved per-device 8 OOMs v6e-64.
    assert cfg["max_train_steps"] == 20000
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
    # F1 (round-5 strengthen; amended §2.2 v3.1): the PLAIN yml command must satisfy
    # plan §6 with no overrides -- amended batch recipe (per_device 4.0 -> GBS 256 on 64
    # devices; per-device 8 OOMs v6e-64 for full-FT, fit probe #4; authored `4.0` so
    # yaml yields a float, matching the base key's type for pyconfig CLI coercion) and
    # W&B live by default.
    assert cfg["per_device_batch_size"] == 4.0
    assert isinstance(cfg["per_device_batch_size"], float)
    assert cfg["global_batch_size_to_train_on"] == 256
    assert cfg["global_batch_size_to_load"] == 256
    assert cfg["wandb_project"] == "maxdiffusion-wan-full-ft"


def test_full_yml_genuine_deltas_differ_from_base():
    # Guard against a copy that silently dropped a delta: the genuine deltas must NOT
    # equal the side-adapter base. (noise_mode equals the base by design, so it is
    # excluded here and asserted absolutely above; max_train_steps joined this list with
    # the §2.2 v3.1 amendment -- 20000 vs the base's 10000.)
    side = _load(SIDE_YML)
    full = _load(FULL_YML)
    for key in (
        "model_type",
        "side_adapter_guide_scale",
        "learning_rate",
        "max_train_steps",
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
    # primary target, the yml's per-device value must resolve to exactly the amended
    # GBS-256 recipe (plan §2.2 v3.1 / §6), and the yml's own (inert, documentary) global
    # values must state that same result so the standalone file never contradicts itself.
    import jax  # deferred: only this test needs jax

    from maxdiffusion import pyconfig

    cfg = _load(FULL_YML)
    monkeypatch.setattr(jax, "devices", lambda: [object()] * 64)
    to_load, to_train_on = pyconfig._HyperParameters.calculate_global_batch_sizes(cfg["per_device_batch_size"])
    assert (to_load, to_train_on) == (256, 256)  # plan §2.2 v3.1: GBS 256 on v6e-64
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


def _run_launcher(experiment: str, tmp_path: Path, extra_env: dict | None = None) -> list[str]:
    """Run the launcher with a stub ``tpu`` and return the argv lines it would submit.

    ``HOME`` is redirected to an empty temp dir so the launcher's own
    ``export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"`` cannot resolve the
    real ``~/.local/bin/tpu`` ahead of our stub; the stub dir is prepended to PATH so it
    wins over every real ``tpu`` on the inherited PATH. ``extra_env`` is applied AFTER
    the ``_LAUNCHER_CONTROLLED`` strip -- it simulates a caller deliberately exporting
    launcher knobs (the mini-cycle-6 small-topology smoke story).
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
    if extra_env:
        env.update(extra_env)

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
    # amended recipe (plan §2.2 v3.1, Query 7): 20000 steps @ GBS 256 (3.55 passes kept).
    assert _env_value(lines, "MAX_TRAIN_STEPS") == "20000"
    assert _env_value(lines, "SIDE_ADAPTER_NOISE_MODE") == "fresh"
    assert _env_value(lines, "RUN_NAME").startswith("wan-full-ft-")
    # cycle-8 change order: the run name interpolates the RESOLVED global batch, so a
    # full_ft run is named gbs256 -- never the old hard-coded gbs512 lie.
    assert "-gbs256-" in _env_value(lines, "RUN_NAME")
    assert "gbs512" not in _env_value(lines, "RUN_NAME")
    # Real queue runs get the AMENDED full-FT batch recipe explicitly (F1 contract +
    # mini-cycle 8: env unset => the full_ft arm-default 4/256/256 -- per-device 8 OOMs
    # v6e-64 for full-FT, fit probe #4 -- never silent yml inheritance).
    assert _env_value(lines, "PER_DEVICE_BATCH_SIZE") == "4"
    assert _env_value(lines, "GLOBAL_BATCH_SIZE_TO_TRAIN_ON") == "256"
    assert _env_value(lines, "GLOBAL_BATCH_SIZE_TO_LOAD") == "256"
    # ...and it dispatches the full-FT train wrapper, NOT the hard-coded adapter script.
    assert "bash_scripts/train_wan_full_ft.sh" in lines
    assert "bash_scripts/train_wan_side_adapter.sh" not in lines
    # F2: the queue SETUP_CMD marks workers ephemeral so root setup.sh may persistently
    # disable auto-updates there (and ONLY there).
    assert any(ln.startswith("EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh") for ln in lines)


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
    # RUN_NAME relocation (cycle-8 change order): adapters still resolve gbs512, so the
    # historical name shape is byte-identical.
    assert "-gbs512-" in _env_value(lines, "RUN_NAME")
    # Mini-cycle 6 regression: env-unset defaults still submit the historical 8/512/512.
    assert _env_value(lines, "PER_DEVICE_BATCH_SIZE") == "8"
    assert _env_value(lines, "GLOBAL_BATCH_SIZE_TO_TRAIN_ON") == "512"
    assert _env_value(lines, "GLOBAL_BATCH_SIZE_TO_LOAD") == "512"
    # F2: SETUP_CMD is arm-common -- adapter queue workers are ephemeral too.
    assert any(ln.startswith("EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh") for ln in lines)
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
    # RUN_NAME relocation (cycle-8 change order): adapters still resolve gbs512.
    assert "-gbs512-" in _env_value(lines, "RUN_NAME")
    # Mini-cycle 6 regression: env-unset defaults still submit the historical 8/512/512.
    assert _env_value(lines, "PER_DEVICE_BATCH_SIZE") == "8"
    assert _env_value(lines, "GLOBAL_BATCH_SIZE_TO_TRAIN_ON") == "512"
    assert _env_value(lines, "GLOBAL_BATCH_SIZE_TO_LOAD") == "512"
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
    # Smoke sets its batch EXPLICITLY -- env unset means the full_ft ARM default
    # (4/256/256 after the §2.2 v3.1 amendment), never silent yml inheritance (F1).
    assert _env_value(lines, "PER_DEVICE_BATCH_SIZE") == "4"
    # RUN_NAME relocation kept the ordering: the SMOKE block runs AFTER the (relocated)
    # non-smoke construction and still overwrites the whole name with its own template.
    assert _env_value(lines, "RUN_NAME").startswith("smoke-full-ft-")
    assert "gbs" not in _env_value(lines, "RUN_NAME")
    assert "bash_scripts/train_wan_full_ft.sh" in lines  # still the full-FT wrapper


def test_launcher_batch_env_overrides_reach_submission(tmp_path):
    # Mini-cycle 6 (v6e-8 smoke OOM at per-device 8): the batch trio must be
    # env-overridable so a small-topology smoke can submit the plan-§5.5 GBS-8 recipe
    # (per-device 1) WITHOUT touching full-run defaults. The globals ride along so the
    # worker-log echo stays honest (they are inert to training: pyconfig recomputes both
    # from per-device, proven by the derivation test above).
    lines = _run_launcher(
        "full_ft",
        tmp_path,
        extra_env={
            "PER_DEVICE_BATCH_SIZE": "1",
            "GLOBAL_BATCH_SIZE_TO_TRAIN_ON": "8",
            "GLOBAL_BATCH_SIZE_TO_LOAD": "8",
            "TPU_CHIPS": "8",
        },
    )
    assert _env_value(lines, "PER_DEVICE_BATCH_SIZE") == "1"
    assert _env_value(lines, "GLOBAL_BATCH_SIZE_TO_TRAIN_ON") == "8"
    assert _env_value(lines, "GLOBAL_BATCH_SIZE_TO_LOAD") == "8"
    # The batch override must not disturb the rest of the full-FT arm.
    assert _env_value(lines, "CHECKPOINT_EVERY") == "2500"
    assert _env_value(lines, "EVAL_DATA_DIR") == TRAIN_PATH
    assert "bash_scripts/train_wan_full_ft.sh" in lines


def test_launcher_full_ft_smoke_attempt2_recipe(tmp_path):
    # The EXACT smoke-attempt-2 launch (worklog 2026-07-18T21:10): SMOKE=1 on v6e-8 with
    # per-device 1 -> 1 step at GBS 8, checkpoints fully off, full-FT wrapper dispatched.
    lines = _run_launcher(
        "full_ft",
        tmp_path,
        extra_env={
            "SMOKE": "1",
            "TPU_CHIPS": "8",
            "PER_DEVICE_BATCH_SIZE": "1",
            "GLOBAL_BATCH_SIZE_TO_TRAIN_ON": "8",
            "GLOBAL_BATCH_SIZE_TO_LOAD": "8",
        },
    )
    assert _env_value(lines, "PER_DEVICE_BATCH_SIZE") == "1"  # the OOM fix
    assert _env_value(lines, "GLOBAL_BATCH_SIZE_TO_TRAIN_ON") == "8"
    assert _env_value(lines, "GLOBAL_BATCH_SIZE_TO_LOAD") == "8"
    assert _env_value(lines, "MAX_TRAIN_STEPS") == "1"
    assert _env_value(lines, "CHECKPOINT_EVERY") == "0"
    assert _env_value(lines, "SAVE_FINAL_CHECKPOINT") == "False"
    assert _env_value(lines, "RUN_NAME").startswith("smoke-full-ft-")
    assert "bash_scripts/train_wan_full_ft.sh" in lines


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
# (F) setup.sh apt hardening (mini-cycle 7) -- static text contract.
# =======================================================================================


def _command_text(txt: str) -> str:
    """The file with comment lines removed. F3 (round 2): EVERY structural assertion in
    the setup tests must hold on COMMAND text, so neither a comment nor a commented-out
    line can ever satisfy one."""
    return "\n".join(ln for ln in txt.splitlines() if not ln.strip().startswith("#"))


def _systemctl_commands(cmd_txt: str) -> list[str]:
    """systemctl invocations from COMMAND text, normalized to start at ``systemctl``
    (each is wrapped in ``timeout 30`` for the global-deadline proof)."""
    out = []
    for ln in cmd_txt.splitlines():
        s = ln.strip()
        if "systemctl " in s:
            out.append(s[s.index("systemctl") :])
    return out


def test_setup_sh_stops_apt_daily_machinery_before_first_apt():
    # Unit set, ordering, and the F2 gate -- all asserted on COMMAND text only.
    cmd = _command_text(_text(SETUP_SH))
    cmds = _systemctl_commands(cmd)
    # (1) BOTH timers stopped SYNCHRONOUSLY (no --no-block): prevents new triggers.
    timer_stops = [c for c in cmds if c.startswith("systemctl stop ") and "--no-block" not in c]
    assert len(timer_stops) == 1, f"expected exactly one synchronous timer stop, got {timer_stops}"
    assert "apt-daily.timer" in timer_stops[0]
    assert "apt-daily-upgrade.timer" in timer_stops[0]
    # (2) the service units stopped too (async acceptable: jammy KillMode=process means
    # the in-flight unattended-upgrade child is handled by the escalation instead).
    svc_stops = [c for c in cmds if c.startswith("systemctl stop --no-block")]
    assert len(svc_stops) == 1, f"expected exactly one --no-block service stop, got {svc_stops}"
    for unit in ("unattended-upgrades", "apt-daily.service", "apt-daily-upgrade.service"):
        assert unit in svc_stops[0], f"{unit} missing from the service stop command"
    # (3) F2 (closed): exactly one PERSISTENT disable covering the full unit set...
    disables = [c for c in cmds if c.startswith("systemctl disable")]
    assert len(disables) == 1, f"expected exactly one persistent disable, got {disables}"
    for unit in ("unattended-upgrades", "apt-daily.timer", "apt-daily-upgrade.timer"):
        assert unit in disables[0], f"{unit} missing from the disable command"
    # ...sitting INSIDE the EPHEMERAL_WORKER if-block (persistent GPU/dev hosts keep
    # their security-update posture) -- located in command text.
    gate_idx = cmd.index('if [ "${EPHEMERAL_WORKER:-0}" = "1" ]')
    disable_idx = cmd.index(disables[0])
    gate_fi_idx = cmd.index("\nfi", gate_idx)
    assert gate_idx < disable_idx < gate_fi_idx, "persistent disable is not inside the EPHEMERAL_WORKER gate"
    # Ordering IN COMMANDS: timers -> services -> gated disable -> escalation -> apt.
    esc_idx = cmd.index("apt_locked()")
    first_apt = cmd.index("apt_deadline_run apt-get")
    assert cmd.index(timer_stops[0]) < cmd.index(svc_stops[0]) < disable_idx < esc_idx < first_apt
    # Every systemctl call is self-bounded (timeout 30; part of the F1 global-deadline
    # proof) and a guarded no-op on systemd-less machines.
    assert cmd.count("timeout 30 systemctl") == 3
    for c in cmds:
        assert c.endswith("|| true") and "2>/dev/null" in c, f"unguarded systemctl command: {c}"


def test_setup_sh_global_deadline_escalation_and_loud_failure():
    txt = _text(SETUP_SH)
    cmd = _command_text(txt)
    # F1 (round 2): ONE global wall-clock budget bounds the whole section, and apt/curl
    # EXECUTION runs under `timeout <remaining>` -- not merely a dpkg lock wait.
    assert "APT_BUDGET=420" in cmd
    assert "APT_SECTION_START=$SECONDS" in cmd
    assert "rem=$((APT_BUDGET - (SECONDS - APT_SECTION_START)))" in cmd
    assert 'timeout "$rem" "$@"' in cmd  # the gate really uses timeout(1) on execution
    assert cmd.count("apt_deadline_run apt-get") == 4  # every apt call budget-gated
    assert "apt_deadline_run curl" in cmd  # an ungated network fetch would break the bound
    # Legacy unbounded / near-window lock waits are gone FROM COMMANDS; per-call lock
    # wait is 60s now that contention is resolved AND verified by the escalation.
    assert "DPkg::Lock::Timeout=-1" not in cmd
    assert "DPkg::Lock::Timeout=600" not in cmd
    assert "DPkg::Lock::Timeout=180" not in cmd
    assert cmd.count("DPkg::Lock::Timeout=60 ") == 4
    # Jammy escalation (KillMode=process): lock/process detection + PID-targeted signals
    # captured via pgrep first -- and NO pattern-kill anywhere (new MINOR).
    assert "apt_locked()" in cmd
    assert "fuser /var/lib/dpkg/lock-frontend" in cmd
    assert 'pids="$(pgrep -f unattended-upgrade 2>/dev/null || true)"' in cmd
    assert "kill -TERM $pids" in cmd
    assert "kill -KILL $pids" in cmd
    assert "pkill" not in cmd
    assert '"$waited" -lt 120' in cmd  # grace window for a clean finish
    assert '"$waited" -lt 30' in cmd  # post-TERM re-check window
    # NEW MAJOR: reaching SIGKILL DISCARDS the worker -- the KILL if-block must contain
    # the loud exit and no apt call (dpkg state is unverifiable after SIGKILL).
    kill_idx = cmd.index("kill -KILL $pids")
    kill_fi = cmd.index("\nfi", kill_idx)  # the inner `; fi` is same-line; this is the block end
    kill_block = cmd[kill_idx:kill_fi]
    assert "exit 1" in kill_block, "KILL path must exit loudly, never fall through to apt"
    assert "unverifiable" in kill_block  # the discard rationale is in the message itself
    assert "apt_deadline_run" not in kill_block
    # LOUD failure everywhere: budget-exhausted + KILL-discard + two apt-chain handlers.
    assert cmd.count("[setup.sh] ERROR") == 4
    # Wrapper propagation + F2 flag forwarding, asserted on command text.
    assert "(sudo bash || bash) <<" not in cmd
    assert 'env EPHEMERAL_WORKER="${EPHEMERAL_WORKER:-0}" bash <<' in cmd
    # Failure signature + the jammy citation are DOCUMENTATION requirements, so the full
    # text including comments is the right target for these three.
    assert "unattended-upgr" in txt
    assert "2026-07-18" in txt
    assert "1690980" in txt  # Launchpad: KillMode=process leaves the child running


# =======================================================================================
# bash -n syntax check on every touched/created shell file.
# =======================================================================================


def _bash_version() -> tuple[int, int]:
    bash_exe = shutil.which("bash") or "/bin/bash"
    out = subprocess.run(
        [bash_exe, "-c", 'echo "${BASH_VERSINFO[0]} ${BASH_VERSINFO[1]}"'], capture_output=True, text=True, timeout=10
    )
    try:
        major, minor = out.stdout.split()[:2]
        return (int(major), int(minor))
    except (ValueError, IndexError):
        return (0, 0)


@pytest.mark.parametrize("script", [TRAIN_WRAPPER, VALIDATE_WRAPPER, LAUNCHER, SETUP_SH], ids=lambda p: p.name)
def test_shell_scripts_pass_bash_n(script, tmp_path):
    assert script.exists(), f"missing {script}"
    target = script
    if script == SETUP_SH and _bash_version() < (4, 2):
        # Upstream setup.sh uses `[[ ! -v MODE ]]` (bash >= 4.2; PRE-EXISTING -- HEAD's
        # setup.sh fails bash-3.2 -n at that exact line). macOS system bash is 3.2 and
        # the queue workers (Ubuntu, bash >= 5) check the real file, so on old bash we
        # syntax-check a COPY with only that one pre-existing expression neutralized
        # (F3: the changed hardening block gets real `bash -n` coverage, no skip).
        txt = _text(SETUP_SH)
        assert txt.count("! -v MODE") == 1, "pre-existing [[ -v ]] drifted; revisit this 3.2 shim"
        target = tmp_path / "setup_v_neutralized.sh"
        target.write_text(txt.replace("! -v MODE", '-z "${MODE:-}"'))
    bash_exe = shutil.which("bash") or "/bin/bash"
    proc = subprocess.run([bash_exe, "-n", str(target)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"bash -n failed for {script.name}:\n{proc.stderr}"
