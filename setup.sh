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

(sudo bash || bash) <<'EOF'
mkdir -p /etc/needrestart/conf.d
echo '$nrconf{restart} = "a";' > /etc/needrestart/conf.d/99-noninteractive.conf
apt-get -o DPkg::Lock::Timeout=-1 update && \
apt-get -o DPkg::Lock::Timeout=-1 install -y numactl lsb-release gnupg curl net-tools iproute2 procps lsof git ethtool && \
DISTRO=$(lsb_release -cs 2>/dev/null)
[ -z "$DISTRO" ] && DISTRO=$(. /etc/os-release && echo "$VERSION_CODENAME")
export GCSFUSE_REPO="gcsfuse-${DISTRO}"
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt-get -o DPkg::Lock::Timeout=-1 update -y && apt-get -o DPkg::Lock::Timeout=-1 install -y gcsfuse
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
# This is the ONLY place qwix may be installed from — it must never go into the
# core requirements above, which run without failure tolerance under `set -e`.
# On a 32-host TPU slice all workers fetch concurrently and GitHub refuses some
# HTTP/2 streams, so a hard dependency on it aborts training at random.
uv pip install -U --resolution=lowest-direct \
        -r src/install_maxdiffusion_extra_deps/extra_deps_from_github.txt || \
    echo "Warning: some GitHub-hosted extras failed to install (e.g. qwix). Safe to ignore if use_qwix_quantization=False."

# Install JAX and JAXlib based on the specified mode
if [[ "$MODE" == "stable" || ! -v MODE ]]; then
  # Stable mode
  if [[ $DEVICE == "tpu" ]]; then
    echo "Installing stable jax, jaxlib for tpu"
    # NEVER pin libtpu independently of jax here. jax's `tpu` extra already
    # hard-pins the exact libtpu it was built against (jax 0.10.1 -> 0.0.41.*,
    # 0.11.0 -> 0.0.44.*, 0.11.1 -> 0.0.46.*) and Pallas refuses to lower
    # against anything older than that pin. A standalone `uv pip install
    # 'libtpu==0.0.41'` used to follow these lines (added 2026-06-08, correct
    # for the then-current jax 0.10.1); by 2026-08-25 it was silently
    # DOWNGRADING a jax 0.11.1 install onto a 3-month-old libtpu and every
    # flash-attention train step died with "Pallas TPU requires a recent libtpu
    # version (at least 0.0.44)". To move libtpu, bump JAX_VERSION.
    #
    # --upgrade-package libtpu/jax/jaxlib is required because the core
    # requirements install above runs with --resolution=lowest-direct, which
    # satisfies `jax>=0.9.0` / `libtpu>=0.0.34` at their floors; without these
    # flags uv considers `jax[tpu]>0.4` already satisfied by that stale jax and
    # pulls the libtpu pin that matches IT, not the current release.
    UPGRADE_PKGS=(--upgrade-package libtpu --upgrade-package jax --upgrade-package jaxlib)
    if [[ -n "$JAX_VERSION" ]]; then
      echo "Installing stable jax, jaxlib, libtpu version ${JAX_VERSION}"
      uv pip install "${UPGRADE_PKGS[@]}" "jax[tpu]==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    else
      echo "Installing stable jax, jaxlib, libtpu for tpu"
      uv pip install "${UPGRADE_PKGS[@]}" 'jax[tpu]>0.4' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    fi
    # Print the resolved pair so a mismatch is visible in the setup log rather
    # than 20 minutes into a run, when the first flash-attention step tries to
    # lower a Pallas kernel. Deliberately does NOT touch the backend: on a
    # 32-host slice, initialising the TPU here would take the device lock
    # before training starts.
    python3 -c "import importlib.metadata as m; print('resolved: jax', m.version('jax'), '/ jaxlib', m.version('jaxlib'), '/ libtpu', m.version('libtpu'))"
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
# moviepy/imageio(-ffmpeg) are needed by wandb.Video to encode numpy frames (W&B video logging)
uv pip install wandb moviepy imageio imageio-ffmpeg