#!/bin/bash

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Description:
# bash setup.sh MODE={stable,nightly} DEVICE={tpu,gpu}

# You need to specify a MODE, default value stable.
# For MODE=stable you may additionally specify JAX_VERSION, e.g. JAX_VERSION=0.4.33
# Enable "exit immediately if any command fails" option
set -e
export DEBIAN_FRONTEND=noninteractive

echo "Checking Python version..."
# This command will fail if the Python version is less than 3.12
if ! python3 -c 'import sys; assert sys.version_info >= (3, 12)' 2>/dev/null; then
    # If the command fails, print an error
    CURRENT_VERSION=$(python3 --version 2>&1) # Get the full version string
    echo -e "\n\e[31mERROR: Outdated Python Version! You are currently using $CURRENT_VERSION, but MaxDiffusion requires Python version 3.12 or higher.\e[0m"
    # Ask the user if they want to create a virtual environment with uv
    read -p "Would you like to create a Python 3.12 virtual environment using uv? (y/n) " -n 1 -r
    echo # Move to a new line after input
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Check if uv is installed first; if not, install uv
        if ! command -v uv &> /dev/null; then
            # echo -e "\n'uv' command not found. Installing it now via the official installer..."
            # curl -LsSf https://astral.sh/uv/install.sh | sh

            # echo -e "\n\e[33m'uv' has been installed.\e[0m"
            # echo "The installer likely printed instructions to update your shell's PATH."
            # echo "Please open a NEW terminal session (or 'source ~/.bashrc') and re-run this script."
            # exit 1
            pip install uv
        fi
        maxdiffusion_dir=$(pwd)
        cd
        # Ask for the venv name
        read -p "Please enter a name for your new virtual environment (default: .venv): " venv_name
        # Use a default name if the user provides no input
        if [ -z "$venv_name" ]; then
            venv_name=".venv"
            echo "No name provided. Using default name: '$venv_name'"
        fi
        echo "Creating virtual environment '$venv_name' with Python 3.12..."
        uv venv --python 3.12 "$venv_name" --seed
        printf '%s\n' "$(realpath -- "$venv_name")" >> /tmp/venv_created
        echo -e "\n\e[32mVirtual environment '$venv_name' created successfully!\e[0m"
        echo "To activate it, run the following command:"
        echo -e "\e[33m  source ~/$venv_name/bin/activate\e[0m"
        echo "After activating the environment, please re-run this script."
        cd $maxdiffusion_dir
    else
        echo "Exiting. Please upgrade your Python environment to continue."
    fi
    # Exit the script since the initial Python check failed
    exit 1
fi
echo "Python version check passed. Continuing with script."
echo "--------------------------------------------------"

# --- Ephemeral-worker apt hardening (fork change; exp_01 mini-cycle 7, strengthened x2) -
# 2026-07-18 v6e-64 fit-probe attempts 1+2: freshly-provisioned workers sat forever in
# "Waiting for cache lock: /var/lib/dpkg/lock-frontend ... held by unattended-upgr"
# (Ubuntu post-boot auto-update; the dpkg-lock timeout was -1 = wait FOREVER) while every
# healthy host died at the ~10-min JAX distributed-init deadline. Hardening, jammy-verified:
#  (1) GLOBAL DEADLINE: the whole apt-critical section runs under one 420s wall-clock
#      budget (APT_BUDGET). Every command in the section is either trivially O(1),
#      self-bounded (timeout-30 systemctl calls; escalation loops capped at 120s/30s),
#      or gated by apt_deadline_run, which executes the command under
#      `timeout <remaining budget>` -- bounding apt/curl EXECUTION time, not merely the
#      lock wait. Worst case: <=90s systemctl + <=150s escalation + budget-gated
#      remainder, all inside 420s + a seconds-scale ungated tail (tee/rm/var reads)
#      ~= 7 min, provably within the ~10-min JAX window. Per-apt dpkg-lock waits drop
#      to 60s: contention was already resolved AND verified by the escalation, so a
#      long per-call lock allowance is no longer needed;
#  (2) stop BOTH apt-daily timers synchronously first (no new triggers this boot), then
#      the service units; on jammy an in-flight unattended-upgrade CANNOT be stopped via
#      its unit (apt-daily-upgrade.service is KillMode=process: `systemctl stop` kills
#      only the controlling shell while the child keeps the dpkg lock -- Launchpad
#      #1690980), so the PROCESS is handled directly: 120s grace to finish cleanly,
#      then SIGTERM to the exact PIDs captured via pgrep (its handler completes the
#      current dpkg transaction and exits; 30s re-check). If SIGKILL is ever needed,
#      dpkg state is UNVERIFIABLE: kill the captured PIDs to unwedge teardown, then
#      exit 1 immediately -- an ephemeral worker is discarded, NEVER continues
#      installing on top of possibly-corrupt dpkg state. "|| true" guards keep all of
#      this a no-op without systemd / the units / the processes (e.g. containers);
#  (3) every failure path is LOUD (budget exhausted, apt failed/timed out, KILL path):
#      a visible setup failure beats silently starving the whole multi-host job;
#  (4) PERSISTENT `systemctl disable` only under EPHEMERAL_WORKER=1 (set by the queue
#      launcher's SETUP_CMD): this file doubles as the general TPU/GPU/dev installer,
#      and persistent hosts must keep their security-update posture -- they get the
#      current-boot stops only. The flag survives sudo's env_reset via `env` on the
#      invocation line;
#  (5) heredoc runs via `$SUDO env ... bash` instead of the old `(sudo bash || bash)`,
#      which swallowed every failure (a failed `sudo bash` had already consumed the
#      heredoc from stdin, so the fallback `bash` read EOF and exited 0). With the
#      outer `set -e`, budget/apt/KILL-path errors abort setup.sh visibly.
SUDO=""
if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi
$SUDO env EPHEMERAL_WORKER="${EPHEMERAL_WORKER:-0}" bash <<'EOF'
APT_BUDGET=420
APT_SECTION_START=$SECONDS
apt_deadline_run() {
  rem=$((APT_BUDGET - (SECONDS - APT_SECTION_START)))
  if [ "$rem" -le 0 ]; then
    echo "[setup.sh] ERROR: apt section exceeded its ${APT_BUDGET}s global budget -- failing setup loudly" >&2
    exit 1
  fi
  timeout "$rem" "$@"
}
mkdir -p /etc/needrestart/conf.d
echo '$nrconf{restart} = "a";' > /etc/needrestart/conf.d/99-noninteractive.conf
timeout 30 systemctl stop apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true
timeout 30 systemctl stop --no-block unattended-upgrades apt-daily.service apt-daily-upgrade.service 2>/dev/null || true
if [ "${EPHEMERAL_WORKER:-0}" = "1" ]; then
  timeout 30 systemctl disable unattended-upgrades apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true
fi
apt_locked() {
  fuser /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock >/dev/null 2>&1 && return 0
  pgrep -f unattended-upgrade >/dev/null 2>&1
}
waited=0
while [ "$waited" -lt 120 ] && apt_locked; do sleep 5; waited=$((waited + 5)); done
if apt_locked; then
  echo "[setup.sh] dpkg lock still held after ${waited}s grace; SIGTERM to the in-flight updater PIDs" >&2
  pids="$(pgrep -f unattended-upgrade 2>/dev/null || true)"
  if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi
  waited=0
  while [ "$waited" -lt 30 ] && apt_locked; do sleep 5; waited=$((waited + 5)); done
fi
if apt_locked; then
  pids="$(pgrep -f unattended-upgrade 2>/dev/null || true)"
  if [ -n "$pids" ]; then kill -KILL $pids 2>/dev/null || true; fi
  echo "[setup.sh] ERROR: dpkg state unverifiable after SIGKILL -- discarding this ephemeral worker (not proceeding to apt)" >&2
  exit 1
fi
apt_deadline_run apt-get -o DPkg::Lock::Timeout=60 update && \
apt_deadline_run apt-get -o DPkg::Lock::Timeout=60 install -y numactl lsb-release gnupg curl net-tools iproute2 procps lsof git ethtool || \
  { echo "[setup.sh] ERROR: apt update/install failed or timed out (global ${APT_BUDGET}s budget, 60s lock bound)" >&2; exit 1; }
DISTRO=$(lsb_release -cs 2>/dev/null)
[ -z "$DISTRO" ] && DISTRO=$(. /etc/os-release && echo "$VERSION_CODENAME")
export GCSFUSE_REPO="gcsfuse-${DISTRO}"
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
apt_deadline_run curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt_deadline_run apt-get -o DPkg::Lock::Timeout=60 update -y && apt_deadline_run apt-get -o DPkg::Lock::Timeout=60 install -y gcsfuse || \
  { echo "[setup.sh] ERROR: gcsfuse install failed or timed out (global ${APT_BUDGET}s budget, 60s lock bound)" >&2; exit 1; }
rm -rf /var/lib/apt/lists/*
EOF

python3 -m pip install -U setuptools wheel uv hatchling hatch-requirements-txt

# Set environment variables from command line arguments
for ARGUMENT in "$@"; do
  IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
  export "$KEY"="$VALUE"
done

# Default device is TPU
if [[ -z "$DEVICE" ]]; then
  export DEVICE="tpu"
fi

# Unset JAX_VERSION if set to "NONE"
if [[ $JAX_VERSION == NONE ]]; then
  unset JAX_VERSION
fi

# Validate JAX_VERSION is only used with stable mode
if [[ -n $JAX_VERSION && ! ($MODE == "stable" || -z $MODE) ]]; then
  echo -e "\n\nError: You can only specify a JAX_VERSION with stable mode.\n\n"
  exit 1
fi

# Respect an activated virtualenv (e.g. maxdiffusion_venv) instead of forcing system Python.
# Forcing UV_SYSTEM_PYTHON=1 caused uv to resolve against /usr/bin/python3.10 even when a
# 3.12 venv was active, breaking cp312-only wheels like array-record.
export UV_SYSTEM_PYTHON=0

# Install core dependencies
uv pip install -U --resolution=lowest-direct \
        -r dependencies/requirements/generated_requirements/requirements.txt

# Install GitHub-hosted extras (torch CPU, qwix). Best-effort: qwix requires a
# live GitHub connection and is only needed when use_qwix_quantization=True.
uv pip install -U --resolution=lowest-direct \
        -r src/install_maxdiffusion_extra_deps/extra_deps_from_github.txt || \
    echo "Warning: some GitHub-hosted extras failed to install (e.g. qwix). Safe to ignore if use_qwix_quantization=False."

# Install JAX and JAXlib based on the specified mode
if [[ "$MODE" == "stable" || ! -v MODE ]]; then
  # Stable mode
  if [[ $DEVICE == "tpu" ]]; then
    echo "Installing stable jax, jaxlib for tpu"
    if [[ -n "$JAX_VERSION" ]]; then
      echo "Installing stable jax, jaxlib, libtpu version ${JAX_VERSION}"
      uv pip install "jax[tpu]==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    else
      echo "Installing stable jax, jaxlib, libtpu for tpu"
      uv pip install 'jax[tpu]>0.4' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    fi
    # jax[tpu] from the Google releases index bundles an older libtpu; Pallas
    # requires libtpu built within the last month, so pin to the latest release.
    uv pip install 'libtpu==0.0.41'
  elif [[ $DEVICE == "gpu" ]]; then
      echo "Installing stable jax, jaxlib for NVIDIA gpu"
    if [[ -n "$JAX_VERSION" ]]; then
        echo "Installing stable jax, jaxlib ${JAX_VERSION}"
        uv pip install -U "jax[cuda12]==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    else
        echo "Installing stable jax, jaxlib, libtpu for NVIDIA gpu"
        uv pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    fi
    export NVTE_FRAMEWORK=jax
    uv pip install transformer_engine[jax]==2.1.0
  fi

elif [[ $MODE == "nightly" ]]; then
  # Nightly mode
  if [[ $DEVICE == "gpu" ]]; then
      echo "Installing jax-nightly, jaxlib-nightly"
      # Install jax-nightly
      uv pip install -U --pre jax jaxlib jax-cuda12-plugin[with_cuda] jax-cuda12-pjrt -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
      # Install Transformer Engine
      export NVTE_FRAMEWORK=jax
      uv pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
  elif [[ $DEVICE == "tpu" ]]; then
    echo "Installing jax-nightly,jaxlib-nightly"
    # Install jax-nightly
    uv pip install --pre -U jax -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
    # Install jaxlib-nightly
    uv pip install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
    # Install libtpu-nightly
    uv pip install --pre -U libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  fi
  echo "Installing nightly tensorboard plugin profile"
  uv pip install tbp-nightly --upgrade
else
  echo -e "\n\nError: You can only set MODE to [stable,nightly].\n\n"
  exit 1
fi

# Install maxdiffusion
uv pip install --no-deps -e .
uv pip install wandb